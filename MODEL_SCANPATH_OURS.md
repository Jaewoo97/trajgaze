# 63.01% model #2 — Scanpath channel (ours)

**egtea 2-way, n=1011 (StreamGaze 526 + EgoGazeVQA 485).** Reaches 63.01% —
but **the number is a tie, not a gain.** This model adds a learned gaze-trajectory
side-channel on top of M1's frozen selection, and during training the channel's
gate **collapses toward zero**, so at the best checkpoint the channel is nearly
inert and the model has effectively reproduced M1. Read the caveat below before
treating this as a second winning method.

| | |
|---|---|
| **Name** | `ours_scanpath` |
| **Accuracy** | **63.01%** — egtea 2-way, n=1011 (epoch 2) — **ties M1, does not beat it** |
| **Budget** | M1's 10% visual tokens **+ K=8 extra "intent" tokens** (side-channel, outside the 10%) |
| **Base VLM** | Qwen2.5-VL-7B (frozen) + LoRA |
| **Extra models** | M1's TAS Stage-1 encoder (~37M, frozen) **+** trainable `ScanpathEncoder` |
| **Checkpoint** | `/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_scanpath/best.pth` (epoch 2) |
| **Trainer** | `TrajGazeMerge/training/train_visionzip_scanpath_lora.py` |
| **Module** | `TrajGazeMerge/models/scanpath_encoder.py` (`ScanpathEncoder`, `build_inputs_with_gaze`) |

---

## Core idea — escape the zero-sum with a side-channel

M1 spends a fixed 10% token budget; within it, adding gaze is zero-sum (every
gaze token displaces a content token). The scanpath idea sidesteps that: keep M1's
selection **exactly**, then **append** a small block of gaze tokens *outside* the
budget, so gaze information is added rather than traded.

`ScanpathEncoder` reads the ordered gaze + hand trajectory (T-scan = 32 steps)
and Perceiver-pools it into **K = 8 "intent" tokens**. `build_inputs_with_gaze`
inserts those 8 tokens immediately after the last video token in the Qwen
sequence, with position-id surgery so the text that follows stays contiguous. A
**Flamingo-style gate** (init 0.1) multiplies the channel so it can fade in — or,
as it turned out, fade out.

The hypothesis: a sequence-level "where the eyes/hands are going" summary gives
the LLM intent context the per-token selection can't. The result falsified it.

---

## The caveat — why 63.01 here is a tie, not a win

During training the gate **annealed from 0.1 → ~0.026** by the epoch-2 (best)
checkpoint, and kept drifting down to ~0.02 by the end of epoch 3. A gate near
zero means the 8 intent tokens contribute almost nothing — the model learned to
**ignore its own side-channel** and fall back on M1's token set.

So the 63.01 at epoch 2 is **M1's accuracy reproduced with an inert appendage**,
not an improvement from gaze intent. The clean tell is the comparison with the
sister method (gaze-tag), whose gate instead *grew* to ~0.52 (channel used) and
which **underperformed** M1 at 61.62 — overfit. One channel collapsed → recovered
M1; the other channel was used → hurt generalization. Both point the same way:
the additive gaze channel adds no usable signal at the 10% regime.

---

## Per-epoch results

| epoch | avg_loss | egtea 2-way | gate |
|---|---|---|---|
| 1 | 0.946 | 61.92% | ~0.05 |
| **2** | 0.607 | **63.01%** ← best (ties M1) | ~0.026 |
| 3 | 0.371 | 59.64% (overfit) | ~0.02 |

`best.pth` = epoch 2. Note the gate is already near-collapsed at the best epoch.

---

## Components

- **Qwen2.5-VL-7B** + VisionZip + **M1's frozen complementary-union selection**
  (unchanged: 7% content ∪ 3% trajectory complement = 10%). See
  `MODEL_M1_VZ_COMPLEMENT.md`.
- **TAS Stage-1 encoder** — frozen, same `stage1_tas_3way_overlay/best.pth` M1
  uses, to score the 3% complement.
- **`ScanpathEncoder`** — the only *new* trainable module: trajectory → K=8 intent
  tokens via Perceiver pooling, with a learnable fade-in gate.
- **LoRA adapter** — adapts the LLM to the (M1 tokens + 8 intent tokens) sequence.

---

## How TAS is used (and where it is NOT)

This is the key structural point of the scanpath approach: **TAS and the scanpath
channel are two separate trajectory readers that never touch each other.**

1. **TAS — used only for the inherited M1 selection (frozen).** The trainer loads
   the same frozen `TrajGazeV2Temporal` TAS encoder M1 uses
   (`load_traj_encoder("full", stage1_ckpt)`, `train_visionzip_scanpath_lora.py:222`,
   commented *"frozen TAS encoder for the (unchanged) M1 complement selection"*).
   Every step calls `select_complementary(..., complement_mode="topk")` — byte-for-byte
   M1: TAS salience ranks the 3% complement among the tokens VisionZip discarded,
   giving the same 7% ∪ 3% = 10% token set. TAS's role here is identical to
   `MODEL_M1_VZ_COMPLEMENT.md`; nothing about it changed.

2. **The scanpath channel — does NOT use TAS.** `ScanpathEncoder` reads the **raw**
   gaze/hand trajectory directly: `_traj_features` resamples the ordered `(x,y)`
   gaze fixations + velocity + both hands to a fixed length (`t_scan=32`, 11
   features/step via `_pool_to_T`), then a small Transformer + Perceiver pooling
   produces K=8 tokens. It never calls the TAS encoder, never sees TAS's `(T,196)`
   patch salience, and never sees Qwen's token layout. It is a parallel,
   *trainable* trajectory encoder, deliberately orthogonal to TAS's *per-patch
   selection* role — the whole premise was to add a behavioural/intent signal that
   patch selection (TAS or otherwise) can't express.

So in this approach TAS does exactly one thing — drive M1's frozen token
selection — and the novel part is TAS-independent. Because the channel's gate then
collapses toward zero, what survives at the best checkpoint is essentially **M1
running on its TAS-selected tokens with an inert scanpath appendage** — which is
why the 63.01 is a tie with M1, not a gain.

---

## Training protocol

Identical to M1 except for the added channel and its separate LR:

- **Data (2-way, `--no-hdepic`):** StreamGaze train 5799 + EgoGazeVQA train 1265,
  gaze-overlay (`GAZE_OVERLAY=1`).
- **Eval:** egtea 2-way, n=1011.
- **Optimization:** 3 epochs, LoRA lr 1e-4, scanpath lr 1e-3, grad-clip 1.0.
- **Batching:** 4-GPU DDP, grad-accum 2 → **eff-batch 8** (matches M1).
- **Early stop:** stop after epoch 2 if epoch-2 val ≤ epoch-1 val.

Reproduce:

```bash
cd /workspace/trajgaze_st
export GAZE_OVERLAY=1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29662 \
  -m TrajGazeMerge.training.train_visionzip_scanpath_lora \
  --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth \
  --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_scanpath \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --gaze-tokens 8 --t-scan 32 \
  --epochs 3 --lr 1e-4 --scan-lr 1e-3 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval
```

(Or `scripts/launch_ours_scanpath.sh`.)

---

## Verdict

At the 10% budget, the scanpath side-channel **does not beat M1** — it ties M1's
63.01 only by gating itself off. Combined with the gaze-tag result (61.62) and the
re-selection / soft-fusion failures, this closes the "add a gaze channel" family:
to actually exceed M1 you must change the regime (larger token budget, a different
base selector, or an eval where gaze reasoning is the real bottleneck), not append
or re-weight a gaze channel. See `MODEL_M1_VZ_COMPLEMENT.md` for the method that
genuinely earns 63.01.
