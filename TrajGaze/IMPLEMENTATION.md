# TrajGaze — Implementation Spec

## What This System Does

A lightweight visual token selector for egocentric action understanding.
Given an egocentric video clip with preprocessed gaze fixation trajectories and
hand position trajectories, the selector jointly selects which frames and which
spatial patches within those frames are causally important for action understanding.

The key thesis: patches where a person's gaze and moving hand simultaneously
converge are where the action is happening. Frames with no meaningful gaze-hand
interaction are skipped entirely. Only the spatiotemporal regions that matter
are passed to the frozen VLM.

**Token budget**: exactly 10% of all T×196 visual tokens, split two ways:
- **Frame selector** attends 50% of frames (top-k by score, not threshold)
- **Patch decoder** selects 20% of 196 patches = 40 patches per attended frame
- Combined: 50% × 20% = 10% total — better temporal coverage than attending
  10% of frames with all patches, better spatial precision than all frames with
  2% of patches

### Full inference flow

```
Gaze/hand JSON + raw frames  (variable length, at 10 FPS)
        ↓
[Interaction tokens computed for ALL T frames — cheap, CPU, no GPU]
        ↓
  Frame Selector (causal, lightweight)
  scores all T frames → top-50% by score selected
        ↓
  [Only attended frames proceed]
        ↓
CausalVideoEncoder              Two-Level TrajectoryEncoder
(B, T_att, 4, H, W)                  (n_att, 4·T_window, 256)
→ (B, T_att·196, 192)                        ↓
        ↓                      interaction-gated trajectory context
                  AR Patch Decoder
      (cross-attn: visual + interaction-gated trajectory)
                       ↓
       top-40 patch indices per attended frame
       e.g. [(t=12, p=142), (t=12, p=23), (t=31, p=69), ...]
                       ↓
       Pixel crops extracted from selected (frame, patch) coordinates
                       ↓
       Frozen SigLIP2 ViT → frozen VLM → action prediction
```

---

## Inputs (Already Preprocessed)

Per video clip, cached JSON with per-frame entries. Coordinates in [0, 224) space.
Missing detections are null.

Raw data locations:
- Gaze:  `datasets/StreamGaze_v2/gaze/{dataset}/viz/{video}.json`
- Hand:  `datasets/StreamGaze_v2/hand/{dataset}/viz/{video}.json`
- Frames: `datasets/StreamGaze_v2/frames/{dataset}/viz/{video}/frame_*.jpg`

Preprocessed locations:
- `datasets/StreamGaze_v2/adapted/{dataset}/viz/{video}.json` — merged + normalized
- `datasets/StreamGaze_v2/interaction/{dataset}/viz/{video}.npz` — I(p,t), attend, features

---

## Core Philosophy

### The selector is a standalone plug-in, not part of the VLM

The VLM is frozen throughout all training. Only the selector is trained.
- No LoRA, no adapter layers, no VLM weight updates ever
- The selector outputs sparse (frame, patch) index pairs and pixel data
- Those patches pass through a frozen SigLIP2 ViT into the frozen VLM

### Fixed 10% token budget via two-level selection

**Level 1 — Frame selector**: top-k selection with k = round(0.50 × T).
Uses `select_frames(ratio=0.50)` at inference — exact budget, no threshold tuning.

**Level 2 — Patch selector**: 40 patches per attended frame (top-40 by NTP score).
20% of 196 single-scale patches.

Combined: 0.50 × 40/196 ≈ 10% of all T×196 tokens. Both levels trained jointly
in Stage 1. The ratio (50/20 split) is a design choice, not a trained parameter.

### Gaze-hand interaction is an explicit, structured signal

Rather than treating gaze, hand_left, and hand_right as independent tokens,
the interaction is made architecturally explicit at four levels:

1. A dedicated **interaction token** per timestep captures geometric and dynamic
   relationships (distances, relative velocities, convergence rate, lead-lag).
2. A **frame selector** scores each frame directly from the interaction token.
3. A **two-level trajectory encoder** first resolves intra-frame gaze-hand
   relations before modeling temporal dynamics.
4. An **interaction gate** in the AR patch decoder modulates trajectory
   cross-attention per frame — high-convergence frames get stronger trajectory signal.

The visual stream and trajectory stream are coupled: the gaze-hand interaction
heatmap (4th visual channel) is the spatial rendering of I(p,t), so both streams
are grounded in the same interaction signal.

### Missing trajectory data is a first-class input type

Gaze and hand detections are frequently null (blinks, occlusion, off-screen).
Represent missing data with learned embeddings — not zeroed out, not interpolated.
Three distinct embeddings: `MISSING_GAZE`, `MISSING_LEFT`, `MISSING_RIGHT`.

The interaction token degrades gracefully: when a hand is missing, relative
velocity terms are zeroed but the gaze-velocity term remains. The Level 2
trajectory encoder uses only the causal mask (not key_padding_mask) — missing
frames carry their MISSING embeddings from Level 1 and the model learns to
interpret these appropriately. Using key_padding_mask at boundary-padded
positions causes all-masked attention rows (softmax of all -inf = NaN).

### Two-stage training: imitation then optimization

**Stage 1 — NTP pretraining (150 epochs)**

Teach the selector what matters via direct supervision before any RL.

Compute the interaction importance score `I(p,t)` per patch per frame:

```
I(p,t) = G(p,t) · H(p,t) · φ(τ*) · ψ(dD/dt)
```

- `G(p,t)` = gaze Gaussian heatmap accumulated over temporal window W=8 frames
- `H(p,t)` = hand proximity heatmap weighted by hand velocity over same window
- `φ(τ*)` = lead-lag modulator: `1.0 + 0.2 · max(0, local_xcorr_peak)`.
  τ*>0 means gaze leads hand (anticipatory) → boost importance
- `ψ(dD/dt)` = convergence modulator from rate of change of gaze-to-hand
  distance. Negative dD/dt (converging) → pre-contact phase → boost importance

**Frame-level target** derived from I(p,t):
```
attend(t) = 1  if  max_p I(p,t) > ε  else 0
```

**Patch-level target** (only for attended frames):
Sort patches by `I(p,t)` descending → top-40 indices = NTP target sequence.

**Auxiliary trajectory prediction loss** `L_traj`:
From mean-pooled visual memory of attended frames, predict future gaze and hand
positions with horizon Δ=8 (0.8s). Only on attended frames with valid future steps.

Total Stage 1 loss:
```
L = L_frame_BCE + L_NTP + 0.1 · L_traj

L_frame_BCE  — over ALL T frames      (frame selector always supervised)
L_NTP        — over attended frames only (teacher-forced, 40 targets per frame)
L_traj       — over attended frames only, masked for null future steps
```

No L_ratio penalty. The 50%/20% token budget is enforced at inference via top-k
selection, not during training. BCE naturally drives attend rates toward the label
distribution; top-k inference gives exact budget control.

The trajectory predictor head is auxiliary and discarded after Stage 1.

**Stage 2 — GRPO RL finetuning (3 epochs)**

Refine toward task performance using the frozen VLM as reward oracle.
See TRAINING.md for details.

---

## Architecture

### Model components (~7M trainable parameters)

**Interaction token computation (per timestep, all T frames)**

```
interaction(t):
  d_left(t)        — gaze→hand_left vector: (dx, dy, |d|)
  d_right(t)       — gaze→hand_right vector: (dx, dy, |d|)
  v_rel_left(t)    — hand_left velocity − gaze velocity  (egomotion-corrected)
  v_rel_right(t)   — hand_right velocity − gaze velocity
  convergence(t)   — dD/dt of dominant gaze-to-hand distance
  lead_lag(t)      — sign of local cross-correlation peak (gaze leading?)
```

**TrajectoryTokenizer**
- Projects each of {gaze, hand_left, hand_right, interaction} features to D=128
- LayerNorm after each projection
- MISSING embeddings for null detections (3 learned parameters per missing type)

**FrameSelector** (`d_model=128, 2 layers, 4 heads`)
- 2-layer causal transformer over all T interaction token embeddings
- Per-frame score: `s(t) = sigmoid(MLP(h_t)) ∈ (0, 1)`
- Training: soft scores → `L_frame_BCE` against attend(t) labels
- Inference: `select_frames(ratio=0.50)` → top-k=round(0.5·T) frames by score

**TwoLevelTrajectoryEncoder**

*Level 1 — IntraFrameBlock (per frame, 2-layer full attention):*
- Stacks 4 tokens per frame: (B·T, 4, 128) → 2-layer TransformerEncoderLayer
- Resolves intra-frame gaze-hand relationships
- Output: (B, T, 4, 128) enriched tokens

*Level 2 — InterFrameTransformer (per attended frame, 6 layers, 8 heads, d=256):*
- Extracts T_window=16 frame context window around each attended frame
- Boundary clips padded with zeros (assigned MISSING embeddings from L1)
- Only causal mask — no key_padding_mask (avoids all-masked-key NaN at boundaries)
- Projects 128→256, adds sinusoidal PE, 6-layer causal TransformerEncoderLayer
- Output: (n_att, 4·16, 256) trajectory context per attended frame

**VideoEncoder**
- Input: (B, T_att, 4, 224, 224) — RGB + interaction heatmap as 4th channel
- Conv2d(4→192, kernel=16, stride=16): spatial patch embedding → 14×14=196 patches
- Causal 3D conv: sees [t-2, t-1, t] via left-padding (no future leakage)
- LayerNorm + learnable 2D positional embedding
- Output: (B, T_att·196, 192)
- Visual memory capped at MAX_VIS_FRAMES=16 during training (prevents cross-attn OOM)

**ARPatchDecoder** (`d_model=192, 4 layers, 6 heads`)
- Token vocabulary: 196 single-scale patches (14×14 at stride 16) + pad token 196
  → VOCAB_SIZE = 197
- Shared `lm_head = Linear(192, 196)` — single head, position-awareness via causal
  self-attention + patch positional embedding (not per-position heads)
- Frame positional embedding: distinguishes which attended frame is being processed
- Per-layer architecture:
  1. Causal self-attention (patch selection history within current frame)
  2. Cross-attention to visual memory (B_att, T_vis·196, 192)
  3. Interaction-gated cross-attention to trajectory context (B_att, 64, 192):
     `gate(t) = sigmoid(MLP(interact_mean_t)) ∈ (0,1)`
  4. FFN (d_ff=384)
- Training: teacher-forced on top-40 I(p,t) patch indices, cross-entropy loss
- Inference: `decode_greedy(n_patches=40)` — already-selected patches masked to -inf

**TrajectoryPredictor** (auxiliary, discarded after Stage 1)
- Input: mean-pooled visual memory (n_att, 192)
- Output: (n_att, 8, 6) — [gaze_x, gaze_y, lh_x, lh_y, rh_x, rh_y] for Δ=8 steps

### Token budget

```
Token budget: 50% frames × 20% patches = 10% of all T×196 visual tokens

For a 100-frame clip at 10 FPS:
  Attended frames:     50  (top-50% of 100)
  Patches per frame:   40  (top-20% of 196)
  Total tokens:     2,000  (vs 19,600 for all frames all patches)
  Reduction:          ~10% of full token count

AutoGaze-style (fixed 10% of all frames, all patches):
  Attended frames:     10
  Patches per frame:  196
  Total tokens:     1,960  (similar budget, but ~5× worse temporal coverage)
```

---

## What to Implement, In Priority Order

All Stage 1 components are implemented. Remaining:

**1. Stage 2 GRPO training loop** (`TrajGaze/training/stage2.py`)
Joint update of frame selector + patch decoder using frozen VLM reward.
Frame attend/skip sampling, patch index sampling with temperature annealing,
VLM reward computation, policy gradient update.

**2. Inference server** (`TrajGaze/inference/server.py`)
FastAPI endpoint accepting preprocessed gaze/hand JSON + raw frames,
returning selected (frame_idx, patch_idx) pairs and pixel data.
Use `select_frames(ratio=0.50)` + `decode_greedy(n_patches=40)`.

**3. Evaluation harness** (`TrajGaze/eval/`)
Run inference on val set (EGTEA), score VLM predictions against ground truth.

---

## Ablations (Runnable via `--mode` flag)

| Flag                | What it tests                                              |
|---------------------|------------------------------------------------------------|
| `full`              | Proposed method (all losses, 50%/20% budget)               |
| `no_frame_selector` | All frames attended (AutoGaze-style temporal selection)    |
| `no_traj_loss`      | NTP only, λ_traj=0                                         |
| `no_ntp`            | Trajectory prediction loss only, no patch NTP              |
| `no_gaze`           | Hand signal only                                           |
| `no_hand`           | Gaze signal only                                           |
| `stage1_only`       | No Stage 2 GRPO                                            |

---

## Known Limitations

**Fixed 50/20 split**: The frame/patch budget split is a hyperparameter, not
learned. The 50% frame ratio drives a 50% BCE target from the interact labels.
Alternative splits (e.g., 30% frames × 33% patches) could be explored.

**Rule-based I(p,t)**: The formula is a useful but imperfect approximation.
Stage 2 RL corrects systematic formula errors.

**Egomotion**: Partially mitigated by computing hand velocity relative to gaze
(`v_rel = v_hand - v_gaze`), which cancels much shared ego-motion.

**Visual memory cap**: Training caps visual memory at MAX_VIS_FRAMES=16 frames
to prevent cross-attention memory explosion. At inference with fewer attended
frames, this cap is rarely hit.

**No contact state**: Our hand detector outputs bounding box centers only.
Convergence and relative velocity serve as a proxy for contact state.
