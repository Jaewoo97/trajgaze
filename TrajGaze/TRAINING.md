# TrajGaze Training Guide

TrajGaze trains in two sequential stages. Stage 1 uses direct supervision from
preprocessed gaze and hand trajectories to teach the selector *what matters*.
Stage 2 uses reinforcement learning with the frozen VLM as a reward oracle to
teach the selector *what helps the downstream task*.

The VLM is **never trained** — its weights are frozen throughout both stages.

---

## Prerequisites

Run these once before any training.

```bash
# 1. Merge gaze + hand JSONs, normalize coordinates to 224×224 space
python -m TrajGaze.data.adapter \
    --gaze-dir   datasets/StreamGaze_v2/gaze \
    --hand-dir   datasets/StreamGaze_v2/hand \
    --frames-dir datasets/StreamGaze_v2/frames \
    --output-dir datasets/StreamGaze_v2/adapted

# 2. Precompute interaction scores (I(p,t), attend labels, interaction features)
python -m TrajGaze.data.interaction \
    --adapted-dir datasets/StreamGaze_v2/adapted \
    --output-dir  datasets/StreamGaze_v2/interaction \
    --workers     32
```

Output locations:
- `datasets/StreamGaze_v2/adapted/{dataset}/viz/{video}.json` — merged per-frame coords
- `datasets/StreamGaze_v2/interaction/{dataset}/viz/{video}.npz` — precomputed labels

---

## Dataset Split

StreamGaze_v2 comprises three datasets. Training uses a fixed split:

| Role | Datasets |
|------|----------|
| Train | EgoExoLearn, HoloAssist (246 clips) |
| Val   | EGTEA (35 clips) |

---

## Stage 1 — NTP Pretraining (150 epochs)

### What it learns

Stage 1 teaches the model to recognize frames and patches where gaze-hand
interaction is happening, using direct supervision from preprocessed trajectories.
No VLM is involved.

### Token budget design

The selector targets **10% of all T×196 visual tokens** via a two-level split:
- **Frame selector**: attends top-50% of frames by score
- **Patch decoder**: selects top-40 patches per attended frame (20% of 196)
- Combined: 50% × 20% = 10% total

The budget is enforced at inference via top-k selection — not during training.
BCE naturally drives the frame selector toward the label attend rate; top-k
gives exact budget control without L_ratio penalty (which caused attend collapse).

### Supervision signals

**Step 1 — Compute the interaction importance score `I(p, t)`**

```
I(p, t) = G(p, t) · H(p, t) · φ(τ*) · ψ(dD/dt)
```

| Factor | What it captures |
|--------|-----------------|
| `G(p, t)` | Gaze Gaussian heatmap: high where the person is looking |
| `H(p, t)` | Hand proximity heatmap, weighted by hand speed |
| `φ(τ*)` | Lead-lag modulator: `1.0 + 0.2·max(0, xcorr_peak)` — boosts when gaze leads hand |
| `ψ(dD/dt)` | Convergence modulator: boosts when gaze-to-hand distance is decreasing |

**Step 2 — Frame-level attend labels**

```
attend(t) = 1   if  max_p I(p, t) > ε
            0   otherwise
```

**Step 3 — Patch-level NTP targets**

For each frame, argsort patches by I(p,t) descending → top-40 indices.
The model learns to output the most important patch first, then the next, etc.

### Loss function

```
L = L_frame_BCE + L_NTP + 0.1 · L_traj
```

| Loss | Computed over | Purpose |
|------|--------------|---------|
| `L_frame_BCE` | **All T frames** | Supervises frame selector for every frame |
| `L_NTP` | **Attended frames only** | Supervises patch decoder to rank patches by importance |
| `L_traj` | **Attended frames only**, masked for null steps | Rewards patches predictive of future gaze/hand |

`L_traj`: from mean-pooled visual memory of attended frames, predict gaze and
hand positions Δ=8 frames (0.8s) ahead. The trajectory predictor is auxiliary
and discarded after Stage 1.

No L_ratio loss. The token budget is fixed by design (top-k inference), not
penalized during training.

### What each component learns

```
TrajectoryTokenizer        — embed gaze/hand positions + interaction features,
                             MISSING embeddings for null detections
FrameSelector              — 2-layer causal transformer → s(t) ∈ (0,1)
                             learns: which frames have meaningful interaction?
TwoLevelTrajectoryEncoder  — intra-frame (Level 1) + inter-frame (Level 2) attention
                             Level 2 uses only causal mask, not key_padding_mask
                             learns: how do gaze and hand interact across time?
VideoEncoder               — 4-channel (RGB + heatmap) → patch embeddings
                             learns: what does a high-importance frame look like?
ARPatchDecoder             — dual cross-attention → ranked patch selection
                             single shared lm_head (196 patch classes)
                             learns: which spatial regions matter most?
TrajectoryPredictor        — auxiliary head, discarded after Stage 1
```

### Forward pass (one training step, one clip)

```
1. All T frames:
   gaze/hand data → TrajectoryTokenizer → 4 tokens per frame (B=1, T, 128)
                 → FrameSelector → s(t) per frame
                 → L_frame_BCE against attend(t) labels
                 → attended_idx = frames where s(t) ≥ 0.5  [training threshold]

2. Attended frames only (n_att = |attended_idx|):
   trajectory tokens → TwoLevelTrajectoryEncoder → (n_att, 64, 256)
   raw frames (4-ch) → VideoEncoder → visual memory (1, ≤16×196, 192)
                     → expand to (n_att, ≤3136, 192)
   NTP targets (n_att, 40) → ARPatchDecoder → logits (n_att, 40, 196) → L_NTP
   visual memory mean      → TrajectoryPredictor → (n_att, 8, 6) → L_traj

3. Backward: L = L_frame_BCE + L_NTP + 0.1·L_traj
```

At inference, hard frame selection uses `select_frames(ratio=0.50)` (top-k),
and patch selection uses `decode_greedy(n_patches=40)`.

### Launch

```bash
# 4-GPU training on H200s (recommended)
bash TrajGaze/scripts/run_stage1.sh full

# Single-GPU (for development / debugging)
python -m TrajGaze.training.stage1 \
    --adapted-dir  datasets/StreamGaze_v2/adapted \
    --interact-dir datasets/StreamGaze_v2/interaction \
    --frames-dir   datasets/StreamGaze_v2/frames \
    --output-dir   TrajGaze/checkpoints/stage1/full \
    --mode full \
    --epochs 150 \
    --max-frames 200
```

Key hyperparameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 150 | |
| Learning rate | 3e-4 | AdamW, cosine decay to 1e-6 |
| Max frames per clip | 800 | Random window if clip is longer |
| Gradient accumulation | 4 steps | Effective batch = 4 clips per GPU |
| Attend threshold (train) | 0.5 | Hard threshold during training forward pass |
| Attend ratio (inference) | 0.50 | Top-k, exact 50% of frames |
| Patches per frame | 40 | Top-40 of 196 (20%) |
| Trajectory horizon Δ | 8 frames | 0.8 seconds at 10 FPS |
| λ_traj | 0.1 | Weight of auxiliary trajectory loss |
| MAX_VIS_FRAMES | 16 | Cap on visual memory (prevents cross-attn OOM) |
| GPUs | 4 × H200 | torchrun, manual all_reduce (no DDP wrapper) |
| dtype | bfloat16 | `torch.amp.autocast`; BCE computed in float32 |

Checkpoints saved every 10 epochs to `TrajGaze/checkpoints/stage1/full/`.
Logs written to `TrajGaze/logs/stage1_full.log`.

---

## Stage 2 — GRPO RL Fine-tuning (3 epochs)

### What it learns

Stage 2 refines the frame selector and patch decoder using the frozen VLM's
accuracy as a reward signal. Stage 1 teaches the selector to recognize
gaze-hand interaction; Stage 2 teaches it to favor selections that actually
help the VLM answer correctly.

### The GRPO algorithm

```
for each clip:
  ┌──────────────────────────────────────────────────────┐
  │  ROLLOUT (no gradients)                               │
  │  for k = 1..12:                                       │
  │    temp = anneal(1.0 → 0.01 across k)                │
  │    attended_idx = top-k sample from s(t) at temp k   │
  │    patches_k    = ARPatchDecoder(attended frames)     │
  │    pred = frozen NVILA(all attended frames +          │
  │             multi-scale patches, question)            │
  │    reward_k = correct(pred == gt) − 0.1·token_ratio  │
  └──────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────┐
  │  POLICY GRADIENT (with gradients)                    │
  │  Recompute s(t) with grad_fn                         │
  │  For each rollout k:                                  │
  │    log_prob_k = Σ_t [a_k(t)·log s(t)                │
  │                     + (1-a_k(t))·log(1-s(t))]        │
  │  advantage_k  = (reward_k − mean_reward) / std_reward│
  │  loss = −mean_k(advantage_k · log_prob_k)            │
  │  loss.backward() → update FrameSelector + Tokenizer  │
  └──────────────────────────────────────────────────────┘
```

Key GRPO design choices:
- **Temperature annealing**: early rollouts explore, later exploit
- **Full 10% token budget per rollout**: all attended frames with multi-scale patches passed to NVILA;
  reward signal reflects the actual inference-time token budget
- **Multi-scale injection**: 40 scale-196 patches propagated to all 4 NVILA scales (56, 112, 196, 392)
  → ~120 patches/frame ≈ 11% of 1060; same injection path as AutoGaze mask_with_gazing
- **Token ratio penalty**: `−0.1·token_ratio` prevents attending everything
- **Dr. GRPO (no KL)**: no frozen reference policy; Stage 1 init is stable enough
- **VideoEncoder, TrajEncoder, ARPatchDecoder frozen**: GRPO signal too noisy for these

### Launch

```bash
# Stage 2 with NVILA oracle (requires Stage 1 checkpoint)
bash TrajGaze/scripts/run_stage2.sh nvila

# Specify Stage 1 checkpoint explicitly
bash TrajGaze/scripts/run_stage2.sh nvila \
    TrajGaze/checkpoints/stage1/full/epoch_0150.pt
```

Key hyperparameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 3 | Short RL phase |
| Learning rate | 5e-4 | AdamW, constant |
| Group size K | 12 | Rollouts per clip |
| δ (token penalty) | 0.1 | |
| Temperature range | 1.0 → 0.01 | Across group members |
| VLM frames per eval | all attended | Full 10% token budget; multi-scale injection |

---

## Full Pipeline

```bash
# Prerequisites (once)
bash TrajGaze/scripts/run_preprocess.sh

# Stage 1 (4 GPUs, ~X hours)
bash TrajGaze/scripts/run_stage1.sh full

# Stage 2 (after Stage 1 completes)
bash TrajGaze/scripts/run_stage2.sh nvila
```

---

## Evaluation

```bash
bash TrajGaze/scripts/run_eval.sh nvila
```

Evaluates on EGTEA (val set). Output:
`TrajGaze/results/trajgaze_{checkpoint}_results.json`
with per-dataset accuracy and mean token ratio.

---

## Ablation Variants

All ablations use the same training procedure. Toggle via `--mode`:

| Mode | What changes |
|------|-------------|
| `full` | Proposed method: all losses, 50%/20% budget |
| `no_frame_selector` | All frames attended (AutoGaze-style) |
| `no_traj_loss` | λ_traj=0; NTP only |
| `no_ntp` | L_NTP=0; trajectory prediction loss only |
| `no_gaze` | Gaze signal zeroed; hand-only |
| `no_hand` | Hand signal zeroed; gaze-only |
| `stage1_only` | No Stage 2 GRPO |

```bash
bash TrajGaze/scripts/run_stage1.sh no_frame_selector
bash TrajGaze/scripts/run_stage1.sh no_traj_loss
# etc.
```

---

## Data Flow Summary

```
datasets/StreamGaze_v2/
├── gaze/{dataset}/viz/{video}.json       # green-dot gaze coordinates
├── hand/{dataset}/viz/{video}.json       # left/right hand centers
├── frames/{dataset}/viz/{video}/         # raw JPEG frames at 10 FPS
├── adapted/{dataset}/viz/{video}.json    # merged + normalized to [0, 224)
└── interaction/{dataset}/viz/{video}.npz # I(p,t), attend(t), interaction features

TrajGaze/
├── checkpoints/stage1/full/epoch_*.pt    # Stage 1 checkpoints (every 10 epochs)
├── checkpoints/stage2/nvila/epoch_*.pt   # Stage 2 checkpoints
├── logs/stage1_full.log                  # training metrics per epoch
├── logs/stage2_nvila.log
└── results/trajgaze_*_results.json       # evaluation output
```

---

## Expected Training Time

On 4 × H200 GPUs:

| Stage | Duration |
|-------|----------|
| Prerequisite data prep | ~5–10 min |
| Stage 1 (150 epochs, 246 train clips, max 800 frames) | ~TBD hours |
| Stage 2 (3 epochs, EGTEA val, group 12) | ~TBD hours |
| Evaluation on EGTEA (35 clips) | ~15–30 min |

Stage 1 is the bottleneck. Using `--max-frames 200` during development
significantly speeds up iteration.

## Numerical Stability Notes

- **bfloat16**: all forward passes under `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
  BCE loss computed in float32 via local `autocast(enabled=False)`.
- **No key_padding_mask in L2 transformer**: combining causal mask + key_padding
  at padded boundary positions makes all keys masked → softmax(-inf,...) = NaN.
  Fixed by using only the causal mask; missing frames carry MISSING embeddings.
- **Visual memory cap**: MAX_VIS_FRAMES=16 in VideoEncoder prevents bfloat16
  cross-attention overflow with large n_att.
- **DDP via manual all_reduce**: training_step uses submodule calls (not a single
  forward()), so DDP wrapper cannot be used. Manual `_ddp_avg_grads` fills None
  grads with zeros before all_reduce to prevent distributed deadlock when a clip
  is skipped.
