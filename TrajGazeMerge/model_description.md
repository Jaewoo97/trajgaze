# TrajGazeVQA — Model Description

## Overview

A two-stage system for egocentric video question answering. Stage 1 trains a trajectory-driven patch scoring model from gaze/hand motion data alone. Stage 3 uses those scores to selectively compress visual tokens before feeding them to a VLM.

---

## Stage 1 — TrajGazeV2Temporal

### Goal

Learn which image patches are spatially and temporally important for understanding an ongoing egocentric action, by observing where people look and reach across video clips.

The key principle: **a patch is important if attending to it helps predict where gaze and hands will move next.** This is stronger than simply marking where gaze currently is — it captures patches that are causally relevant to the action.

### Inputs

- Gaze and hand trajectory for T=128 past frames per clip:
  - Gaze position (x, y) + visibility mask
  - Left hand position (x, y) + visibility mask
  - Right hand position (x, y) + visibility mask
- Video frames (used only for visual patch features via frozen DINOv2)
- No text labels — training is self-supervised from trajectory data only

### Architecture

```
Video frames ──► DINOv2-S/14 (frozen)
                 K=16 keyframe sample + temporal interpolation
                 → (B, T, 196, 256)  per-frame patch features
                        │
Trajectory ──► Tokenizer ──► 4 tokens/frame: gaze, left, right, interaction
               IntraFrameBlock: self-attention across 4 tokens within each frame
               InterFrameTransformer: global temporal context across all T×4 tokens
               FiLM: query conditioning (zero vector in Stage 1)
                        │
               TemporalVisualTrajFusion
               per-frame cross-attention: trajectory tokens (Q) × visual patches (K,V)
               → enriched_context (B, T, 4, D)
               → enc_attn         (B, T, 4, 196)  per-token visual attention weights
               → past_scores      (B, T, 196)      raw cross-attn readout
                        │
               ┌────────┴──────────────────┐
               │                           │
         TrajScoreHead              TrajectoryDecoder + ScoreDecoder
         (inference path)           (training signal only)
               │                           │
       (B, T, 196)               traj_pred (B, T_future, 6)
     per-frame patch scores       dec_attn  (B, T_future, T_past×4)
                                  score_pred (B, T_future, 196)
```

### Three-Part Architecture

**Encoder** — *"What has happened, and what does it mean visually?"*

Processes past trajectory frames and builds a rich per-frame representation by cross-attending trajectory tokens to DINOv2 visual patch features. Each frame's gaze/hand tokens learn to point at the visually relevant patches for that moment. The inter-frame transformer gives the encoder a global temporal context — it knows not just what happened in one frame but how the action has evolved across all past frames.

**Decoder** — *"What will happen next?"* (training only, discarded at inference)

Takes the encoder's context and predicts future gaze and hand positions. This is the training signal that defines what "important" means — if the decoder correctly predicts future movement, it must have relied on the right past tokens. The decoder's cross-attention weights (`dec_attn`) record which past tokens it depended on most.

**TrajScoreHead** — *"Distill the decoder's judgment into a score map"* (inference path)

A lightweight MLP head trained to reproduce the decoder's derived importance signal directly from the encoder context. At inference, only this head runs — no decoder needed, no GT labels needed, adaptive to any T.

### Training Signal Chain

```
GT future trajectory positions
        ↓  L_traj (MSE)
Decoder learns which past tokens predict future movement
        ↓  dec_attn: (B, T_future, T_past×4)
"These past tokens mattered most for future prediction"
        ↓  × enc_attn: (B, T_past, 4, 196)
"These patches, attended by those tokens, drove the prediction"
        ↓  = traj_driven (B, T_past, 196)  — detached target
        ↓  L_score_traj (MSE)
TrajScoreHead learns to reproduce this signal
```

### Four Training Losses

| Loss | Supervises | Purpose |
|------|-----------|---------|
| **L_traj** | Decoder trajectory prediction vs GT future positions | Primary task; shapes what encoder attends to |
| **L_score_traj** | TrajScoreHead output vs `traj_driven` | Primary score signal; teaches head to reproduce trajectory-prediction-driven importance |
| **L_score_past** | Encoder raw cross-attn scores vs GT gaze-hand heatmaps | Auxiliary grounding; keeps visual cross-attn near GT-relevant patches |
| **L_score_future** | Score decoder output vs GT future gaze-hand heatmaps | Auxiliary; trains decoder to anticipate future spatial attention |

`L_total = L_traj + L_score_traj + L_score_past + L_score_future`

### Dataset

StreamGaze_v2 clips from EgoExoLearn + HoloAssist. Each clip is sampled to T=128 frames. Per iteration, a random 40–60% past/future split is applied — the encoder sees the past, the decoder predicts the future. No MCQ labels used.

### Output at Inference

`get_patch_scores(traj_batch, frame_paths)` → `(B, T, 196)` per-frame patch importance scores, where each value reflects how causally relevant that patch was for ongoing trajectory prediction. Decoder is not called.

---

## Stage 3 — TrajGazeMerge Temporal

### Goal

Fine-tune Qwen2.5-VL on MCQ tasks while compressing visual tokens to 10% of their original count, guided by the Stage 1 score maps. The model must answer questions as well as an uncompressed teacher despite seeing 10× fewer visual tokens.

### Inputs

- Video clip (128 frames, Qwen processor)
- MCQ question + answer choices
- Gaze/hand trajectory (T=128 frames) — for score map generation
- Teacher model (full-token Qwen LoRA, frozen) — for KD target

### Architecture

```
Video ──► Qwen2.5-VL visual encoder ──► visual tokens (T_merged × n_spatial)
                                                │
Trajectory + Frames ──► TrajGazeV2Temporal      │
                        (Stage 1 weights,        │
                         fine-tuned)             │
                        → (T_traj, 196)          │
                               │                 │
                    score_to_qwen_spatiotemporal  │
                    spatial:  14×14 → 8×8        │
                    temporal: T_traj → T_merged   │
                    flatten → (n_video,) scores   │
                               │                 │
                               └────────────────►│
                                    gaze_weighted_merge
                                    receivers = top 10% by score (kept)
                                    sources   = bottom 90% (merged into nearest receiver
                                                by weighted cosine similarity)
                                    → merged tokens (10% of original count)
                                               │
                              Qwen2.5-VL LLM (LoRA fine-tuned)
                                               │
                                        answer logits
```

### Score Alignment Pipeline

Stage 1 produces scores at TrajGaze resolution; Qwen produces visual tokens at its own resolution. Alignment:

```
(T_traj=128, 196)   14×14 spatial, T_traj temporal
      ↓ spatial rescale
      14×14 → 16×16 (nearest) → 8×8 (avg_pool) when n_spatial=64
(T_traj, 64)
      ↓ temporal linear interpolation T_traj → T_merged
(T_merged, 64)
      ↓ flatten
(n_video,)          one score per Qwen visual token
```

This is fully differentiable — gradients flow back through the score alignment into the TrajScoreHead and encoder.

### Token Merging — gaze_weighted_merge

Visual tokens are sorted by score into:
- **Receivers** (top 10%) — kept unchanged, form the output token sequence
- **Sources** (bottom 90%) — each source is merged into its most similar receiver by weighted cosine similarity, where similarity weights are modulated by the source's score

The merge is differentiable with respect to scores, enabling end-to-end training.

### Training

**Two loss terms:**

| Loss | Formula | Purpose |
|------|---------|---------|
| **L_CE** | Cross-entropy vs GT answer label | Direct task supervision |
| **L_KL** | KL divergence vs teacher logits (full tokens) | Knowledge distillation — match uncompressed model's output distribution |

`L_total = 0.5 × L_KL + 0.5 × L_CE`

**What is trained:**
- Qwen LoRA adapters — adapt LLM to work with merged tokens
- TrajGazeV2Temporal encoder + TrajScoreHead — fine-tune score maps to be useful for the MCQ task

**What is frozen:**
- Qwen base weights
- DINOv2 visual encoder (inside TrajGaze)
- Teacher model

**Knowledge distillation teacher:** The same Qwen2.5-VL with LoRA weights fine-tuned on full (uncompressed) visual tokens. It sees all visual tokens and its logits serve as soft targets — the student learns to replicate the teacher's answer distribution while running on 10% of the tokens.

### Dataset

StreamGaze_v2 MCQ items from EgoExoLearn + HoloAssist (train split), ~5799 items across 8 tasks. Eval on full EGTEA test set (526 items) every 400 steps.

### Evaluation

Two accuracy numbers are reported at each eval:
- **merge_acc**: student model (10% tokens) accuracy on EGTEA test
- **full_acc**: teacher model (100% tokens) accuracy on same items — reference ceiling

---

## Key Design Decisions

**Why trajectory-prediction-driven scores (Option C)?**
A patch scored by raw cross-attention (where gaze currently is) answers "where is attention now?" Option C answers "which patches, when attended to, caused this trajectory to unfold?" — a stronger causal signal for predicting task-relevant content.

**Why per-frame scores (temporal model)?**
The original model produced one `(196,)` clip-level map tiled uniformly across all frames — a rough approximation. The temporal model produces `(T, 196)`, giving each frame its own spatial importance map. A hand reaching for an object at t=50 gets high scores at t=50, not at every frame.

**Why KD instead of CE only?**
With 90% of visual tokens removed, CE loss alone may not converge — the task is very hard. The teacher provides a smooth, rich training signal (full probability distribution over answers) that guides the student even when it makes wrong predictions.

**Why freeze DINOv2?**
DINOv2 features are strong general-purpose patch representations. Fine-tuning it end-to-end would be expensive and risk overfitting to the small trajectory dataset. The Stage 1 model learns to *read* the fixed features, not to produce them.
