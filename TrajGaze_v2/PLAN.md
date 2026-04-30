# TrajGaze_v2 — Design & Implementation Plan

## Overview

TrajGaze_v2 is a lightweight visual token selector that uses **gaze and hand trajectories**
to select the most informative 10% of visual tokens for a downstream VLM, without requiring
the VLM to process all tokens. The model is trained via a two-stage pipeline:

- **Stage 1**: Supervised trajectory + interaction-score prediction (no VLM)
- **Stage 2**: GRPO reinforcement learning with frozen Qwen2.5-VL-7B accuracy as reward

---

## Dataset

| Split | Source | Clips |
|-------|--------|-------|
| Train | ego4d + egoexo | 421 |
| Val   | egtea           | 82  |

**Per-clip data:**
- Frames: `no_gaze/{video_id}/{start}_{end}_{frame_num}.jpg` (10 FPS, 350–430 frames)
- Gaze: `gaze_mapping/{video_id}/{start}_{end}_mapping.csv` — columns: `frame_idx, gaze_x, gaze_y` (normalized [0,1])
- Hands: `hand_locations/{video_id}.json` — keyed by `{start}_{end}_{frame_num}.jpg`, values: `{left: [x,y]|null, right: [x,y]|null}` (native resolution → normalize by image dims)

---

## Model Architecture

### Input
| Signal | Shape | Notes |
|--------|-------|-------|
| gaze_pos | (B, T, 2) | normalized [0,1], zero if missing |
| gaze_mask | (B, T) | bool, True = present |
| left_pos | (B, T, 2) | normalized [0,1] |
| left_mask | (B, T) | bool |
| right_pos | (B, T, 2) | normalized [0,1] |
| right_mask | (B, T) | bool |
| query_text | list[str] or None | QA question (null in Stage 1) |

T_sample = 32 frames (uniformly sampled from clip)

### A. Trajectory Tokenizer (D_TRAJ=128)

Reused/adapted from TrajGaze v1. Produces 4 tokens per frame:
- `tok_gaze` (B, T, 128): gaze position + speed
- `tok_left` (B, T, 128): left hand position + velocity + present bit
- `tok_right` (B, T, 128): right hand position + velocity + present bit
- `tok_interact` (B, T, 128): interaction features (d_left, d_right, v_rel, convergence, lead_lag)

Missing signals replaced by learned MISSING embeddings.

### B. SpatiotemporalEncoder

```
IntraFrameBlock (2L, 4H) — per-frame self-attn across 4 tokens
    (B, T, 4, 128) → (B, T, 4, 128)

Linear 128→256 + reshape
    (B, T, 4, 128) → (B, T*4, 256)

Sinusoidal temporal PE (groups of 4)

InterFrameTransformer (6L, 8H, D=256)
    (B, T*4, 256) → (B, T*4, 256)

FiLM conditioning on query_emb (B, 128→256 scale+shift)

Reshape → (B, T, 4, 256)

Patch cross-attention (non-causal):
    Q = learnable patch_pos_embed (196, 256)   [repeated B times]
    K, V = trajectory context at each frame (B, 4, 256)
    → attention weights (B, T, 196, 4) → mean over 4 heads → (B, T, 196)
    → patch_scores (B, T, 196) ∈ [0,1]
```

### C. Token Selection (10%)

Aggregate: `scores = mean(patch_scores, dim=1)` → (B, 196) spatial priority score

Top-K: K = ceil(0.10 × 196) = 20 patches selected per frame, same mask applied to all frames.

For Stage 2 GRPO stochastic selection: add Gumbel noise before top-K.

### D. TrajectoryDecoder (Decoder 1)

Predicts future gaze + hand positions from past context.

```
Input: context (B, T_past, 4, 256)
Aggregate: mean-pool over tokens → (B, T_past, 256)
Cross-attn decoder (3L): learnable future queries (T_future_max, 256) × context
Output: (B, T_future, 6) = [gaze_x, gaze_y, left_x, left_y, right_x, right_y]
```

Loss: masked MSE (only valid positions, i.e., where mask=True in future)

### E. ScoreDecoder (Decoder 2)

Predicts future per-patch interaction scores from past context.

```
Input: context (B, T_past, 4, 256)
Aggregate: mean-pool → (B, T_past, 256)
Cross-attn decoder (3L): learnable future queries (T_future_max, 256) × context
Output: (B, T_future, 196) → sigmoid → interaction score prediction
```

Loss: MSE against computed GT interaction scores (from TrajGaze.data.interaction)

---

## Stage 1 Training

**Setup:** DDP on 4 GPUs, ego4d + egoexo (421 clips)

**Per-sample procedure:**
1. Load clip → sample T=32 frames uniformly
2. Load gaze (T, 2) + hand (T, 2 each) + compute interaction scores GT
3. Random split: T_past = round(T × r), r ~ U[0.4, 0.6]
4. Input: past frames [0:T_past]; Target: future frames [T_past:T]
5. Forward pass → decoder predictions
6. Compute losses

**Loss:**
```
L = L_traj + λ_score · L_score
L_traj  = MSE(pred_gaze, gt_gaze) + MSE(pred_left, gt_left) + MSE(pred_right, gt_right)  [masked]
L_score = MSE(pred_scores, gt_scores)
λ_score = 1.0
```

**Hyperparams:**
- Epochs: 100
- LR: 3e-4 (AdamW), cosine decay
- Batch size: 4 per GPU = 16 total
- Weight decay: 1e-4

---

## Stage 2 GRPO

**Setup:** 4 GPUs, egtea + ego4d + egoexo metadata.csv QA pairs

**Policy:** TrajGaze_v2 encoder (trainable) + Gumbel-top-K selection

**Reward model:** Frozen Qwen2.5-VL-7B

**Per-step:**
1. For each QA sample, run TrajGaze_v2 → (B, T, 196) scores
2. Stochastic top-K (Gumbel noise): select 20 patches per frame
3. Filter Qwen visual tokens using mask (same pipeline as autogaze eval)
4. Forward through frozen Qwen → get answer
5. Reward: 1.0 if correct else 0.0

**GRPO (group_size=8):**
- Sample 8 different masks per QA via Gumbel noise
- Baseline: mean reward within group
- Policy gradient: ∇L = E[(R - baseline) · ∇log π(mask|trajectory)]

**Hyperparams:**
- LR: 1e-5, constant schedule
- Batch size: 1 QA × 8 group = 8 per GPU
- Epochs: 3
- group_size: 8
- gumbel_temperature: 0.5 (annealed to 0.1)

---

## Evaluation

Dataset: EGTEA validation set (~82 clips from metadata.csv)

Metrics per qa_type (causal, spatial, temporal):
- Accuracy (%)
- Avg inference latency (seconds)

Baseline comparison:
- Qwen2.5-VL-7B full tokens (from fold_c eval)
- Qwen2.5-VL-7B + AutoGaze 10% tokens
- Qwen2.5-VL-7B + TrajGaze_v2 10% tokens (ours)

---

## Key Design Differences from TrajGaze v1

| Aspect | v1 | v2 |
|--------|----|----|
| Visual input | 4-channel RGB+heatmap | Pure trajectory (no visual) |
| Selection granularity | Frame-level then patch | Patch-level directly |
| Decoder | Autoregressive | Parallel (non-AR) |
| Stage 1 loss | BCE + NTP + traj | MSE traj + MSE score |
| Stage 2 | Not implemented | GRPO with Qwen |
| Query conditioning | Throughout | FiLM in encoder |
