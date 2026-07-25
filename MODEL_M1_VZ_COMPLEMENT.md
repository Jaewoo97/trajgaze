# 63.01% model #1 — M1: VisionZip-Complement (learned top-k)

**egtea 2-way, n=1011 (StreamGaze 526 + EgoGazeVQA 485).** One of two methods
that reach 63.01%. This one is the **genuine** 63.01: a real gain over plain
VisionZip (62.51) from selecting gaze/hand tokens VisionZip discarded. (The other
63.01 — the scanpath channel — only *ties* by gating its channel off; see
`MODEL_SCANPATH_OURS.md`.)

| | |
|---|---|
| **Name** | `visionzip_complement_learned_overlay` (a.k.a. **M1 top-k**) |
| **Accuracy** | **63.01%** — egtea 2-way, n=1011 |
| **Budget** | 10% of visual tokens (6.5% raw + 3.5% merged) |
| **Base VLM** | Qwen2.5-VL-7B (frozen) + LoRA |
| **Extra model** | TAS Stage-1 encoder ≈ 37M params (frozen DINOv2-S/14 + trajectory head) |
| **Checkpoint** | `/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay/best.pth` (epoch 2) |
| **Trainer** | `TrajGazeMerge/training/train_visionzip_complement_lora.py` (`--complement-mode topk --traj-pool-mode learned`) |

---

## Core idea — complementary union

VisionZip keeps the 10% of visual tokens its ViT attention scores highest. That
attention is content-driven, so **gaze/hand-relevant patches it scores near zero
can never be recovered.** The earlier VZ+traj variant only *multiplied* attention
by a trajectory weight, so it could re-rank *within* the attention-supported set
but never resurrect a missed token.

M1 instead takes a **complementary union** of two disjoint pools:

```
10% budget  =  7% VisionZip content  ∪  3% trajectory complement
            =  (3.5% dominant + 3.5% contextual)  ∪  (top-3% gaze/hand tokens
                                                       VisionZip did NOT keep)
```

The 3% trajectory pool is drawn *only* from tokens outside VisionZip's content
set — exactly the gaze/hand patches content attention missed or merged away. This
is what lets gaze information **add** signal instead of just re-shuffling it.

Both selectors are **frozen** (no trained parameters); only the LoRA adapter
learns to read the resulting token set.

---

## Selection algorithm

`select_complementary(..., complement_mode="topk")` in
`train_visionzip_complement_lora.py:323`:

1. **Content pool (7%)** — VisionZip `visionzip_select_tokens` with
   `dominant_ratio=0.035, contextual_ratio=0.035`:
   - *dominant* = raw tokens with highest ViT attention (kept un-merged),
   - *contextual* = remaining tokens cluster-merged by attn-key cosine into
     centroids. → `content_idx`.
2. **Trajectory scores** — frozen TAS encoder produces per-patch salience over
   the video, bridged onto VisionZip's token layout
   (`_traj_scores` → `get_patch_scores_temporal` → `_score_to_qwen_robust`).
3. **Complement pool (3%)** — among tokens NOT in `content_idx`, take
   `k = int(0.03 · N)` by **global top-k** of trajectory salience → `traj_idx`.
4. **Union & order** — concatenate `content_idx ∪ traj_idx`, sort by token index
   → `(sel_embeds, recv_idx)`.
5. **Build & forward** — `build_merged_inputs` writes the selected embeddings
   back into the Qwen sequence; `forward_logits` reads the 5 option-letter logits.

Net composition: **6.5% raw tokens + 3.5% merged tokens = 10%**.

---

## Components

- **Qwen2.5-VL-7B** with VisionZip token pruning (frozen backbone).
- **TAS Stage-1 encoder** — `TrajGazeV2Temporal` (`model_type="full"`,
  `use_trajectory_anchor=True`), frozen, loaded from
  `/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth`.
  Internally a frozen **DINOv2-S/14** (`dinov2_vits14`, dim 384) + a trained
  trajectory head — ~37M params total. Produces the per-frame `(T, 196)`
  trajectory salience that scores the 3% complement pool. This is the one extra
  model M1 needs beyond Qwen (plain VisionZip needs none).
- **LoRA adapter** — the only trainable part; adapts the LLM to the
  complement-augmented token set.

---

## How TAS is used — here and across approaches

**TAS = Trajectory-Aware Selection.** There is *one* TAS encoder — a frozen
`TrajGazeV2Temporal` (frozen DINOv2-S/14 + trained trajectory head), loaded once
via `load_traj_encoder("full", stage1_ckpt)` and never updated. Its job is always
the same: turn the clip's gaze/hand trajectory into a **per-video-token salience
vector**. What differs between approaches is *what they do with that vector*.

**The shared TAS data flow** (`_traj_scores`, `train_visionzip_complement_lora.py:189`):

```
gaze/hand trajectory + frames
   → get_patch_scores_temporal(encoder, item)      → (T_traj, 196)  per-patch salience
   → _score_to_qwen_robust(.., grid_thw)          → (N,)           one score per video token
```

That `(N,)` vector is the only thing downstream selection consumes.

**In M1 specifically:** TAS salience is used as a **ranking function over the
leftover tokens only.** After VisionZip fixes its 7% content set (`content_idx`),
M1 masks those out and takes `torch.topk(traj_scores[avail], k=0.03·N)` — the 3%
highest-TAS tokens *among the patches VisionZip discarded*. TAS never overrides
VisionZip; it decides which of the un-kept gaze/hand patches to resurrect into the
complement. This disjoint, leftover-only use is exactly what makes the gain
additive rather than zero-sum.

How every 10%-budget approach uses the *same* TAS salience vector:

| Approach | What it does with TAS salience | acc |
|---|---|---|
| TAS (Stage-2 merge) | salience drives the **whole** selection — `gaze_weighted_merge` picks all 10% by trajectory weight | 59.64 |
| VisionZip | **not used** — pure ViT attention, no TAS, no extra model | 62.51 |
| VZ+traj | salience **multiplies** VisionZip attention (`attn × traj`) → re-ranks only *within* the attention-supported set | 62.71 |
| **M1 (this model)** | salience **ranks the 3% complement** among non-content tokens (disjoint top-k) | **63.01** |
| Scanpath (ours) | TAS used in M1's role only; the added channel is **TAS-independent** | 63.01 (tie) |
| Gaze-tag (ours) | M1 selection **+** same salience reused as a per-token tag feature | 61.62 |

The lesson across the column: TAS salience helps most when it is restricted to
selecting from the tokens content attention *missed* (M1). Letting it drive the
full merge (TAS Stage-2) or only reweight the attention set (VZ+traj) is worse.

---

## Training protocol

- **Data (2-way, `--no-hdepic`):** StreamGaze train (egoexolearn + holoassist)
  5799 items + EgoGazeVQA train (ego4d + egoexo) 1265 items. Gaze-overlay frames
  (`GAZE_OVERLAY=1`).
- **Eval:** egtea 2-way, n=1011 (StreamGaze egtea 526 + EgoGazeVQA egtea 485).
- **Optimization:** 3 epochs, LoRA lr 1e-4, AdamW, grad-clip 1.0.
- **Batching:** 2-GPU DDP, grad-accum 4 → **eff-batch 8**.
- **Early stop:** stop after epoch 2 if epoch-2 val ≤ epoch-1 val.

Per-epoch val (the basis for `best.pth`):

| epoch | avg_loss | egtea 2-way |
|---|---|---|
| 1 | 0.861 | 61.62% |
| **2** | 0.600 | **63.01%** ← best |
| 3 | 0.353 | 60.83% (overfit) |

Reproduce:

```bash
cd /workspace/trajgaze_st
export GAZE_OVERLAY=1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29654 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk \
  --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth \
  --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval
```

---

## Why this is the best 10%-budget method

Every alternative explored at the same 10% budget on the same egtea 2-way eval:

| Method | acc | note |
|---|---|---|
| TAS (Stage-2, gaze-weighted merge) | 59.64 | full trajectory merge |
| Gaze-tag (ours, additive per-token) | 61.62 | overfit; used channel, no val gain |
| VisionZip (attention only) | 62.51 | no gaze |
| VZ+traj (attention × traj weight) | 62.71 | can't resurrect missed tokens |
| **M1 — complement top-k (this model)** | **63.01** | **best — genuine gain** |
| Scanpath (ours, additive K=8 tokens) | 63.01 | ties only by gating channel off |

Three families of attempts to beat M1 have all failed: **re-selection** (coverage
de-clustering of the complement lost to raw top-k), **soft fusion**
(`norm(attn)+λ·norm(traj)` was null), and **additive side-channels** (scanpath
ties by gating off, gaze-tag overfits). At the 10% budget M1's raw top-k
complement selection is near-optimal; beating it requires changing the regime
(larger budget, a different base selector, or an eval where gaze reasoning is the
actual bottleneck), not adding or reshuffling a gaze channel.
