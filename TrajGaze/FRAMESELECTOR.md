# Query-Aware Frame Selector

## Overview

The TrajGaze Frame Selector is a lightweight module that determines **which frames
are important for answering a given question** in egocentric video. It combines
two signals for frame selection:

1. **Gaze-hand interaction** — frames where gaze and hand converge (where action happens)
2. **Query-frame similarity** — frames visually related to the question text

Previously, only signal (1) was used, making the selector question-agnostic.
By adding Talk2DINO-based query similarity, the selector can now **choose different
frames for different questions even within the same clip**.

---

## Benchmark and Dataset

### EgoGazeVQA

The benchmark used for training and evaluation. Composed of egocentric videos
with gaze tracking, hand detection, and QA pairs from three datasets
(ego4d, egoexo, egtea).

```
/home/yujin/dataset/EgoGazeVQA/all_gaze_v1/
│
├── metadata.csv                    ← 1,750 QA pairs
│   columns: file_name, video_id, dataset, qa_type,
│            question, answer_options, correct_answer
│
├── {dataset}/                      ← egtea, ego4d, egoexo
│   ├── no_gaze/                    ← original frames (no gaze overlay)
│   │   └── {video_id}/
│   │       ├── {clip_name}_{frame_num}.jpg
│   │       └── ...
│   │
│   ├── gaze_mapping/               ← per-frame gaze coordinates
│   │   └── {video_id}/
│   │       └── {clip_name}_mapping.csv
│   │       columns: frame_idx, gaze_frame_num, matched_gaze_frame,
│   │                gaze_x, gaze_y, is_exact_match
│   │       (gaze_x, gaze_y: normalized [0, 1])
│   │
│   └── hand_locations/             ← per-frame hand coordinates
│       └── {video_id}.json
│       format: {"frame_name.jpg": {"left": [x,y]|null, "right": [x,y]|null}}
│       (coordinates in pixel space, detected by Faster-RCNN handobj_100K)
│
├── query_sim/                      ← preprocessing output (Talk2DINO similarity)
│   └── {dataset}/{video_id}/
│       └── {qa_hash}.npz
│       fields: query_sim (T,), uniform_sim (T,), question (str)
│
└── talk2dino_similarity_results.json  ← full results (for analysis)
```

### Dataset Scale

| Item | Count |
|------|-------|
| QA pairs | 1,750 |
| Unique clips | 827 |
| Total frames | ~391,700 |
| Datasets | egtea (485 QA), ego4d (577 QA), egoexo (688 QA) |
| Gaze coverage | ~100% |
| Hand coverage | egtea ~11%, ego4d ~30%, egoexo ~45% |

---

## Preprocessing (Run once before Stage 1 training)

### Purpose

Compute **question-frame similarity** for all frames of each QA pair and save
to disk. These values are used as additional input to the FrameSelector
(query_sim) and for attend label modulation during training.

### Pipeline

```
no_gaze frame (frame_t.jpg)
        │
        ▼
DINOv2 ViT-L/14 (frozen, resize=518)
        │
        ▼
patch features (1369, 1024)       ← 37×37 grid
        │
        ├── Uniform pooling:
        │     feat = mean(patches)              → (1024,)
        │
        └── Gaze+Hand weighted pooling:
              w(p) = gauss(p, gaze, σ_g=0.14)
                   + gauss(p, hand_L, σ_h=0.18)
                   + gauss(p, hand_R, σ_h=0.18)
                   + ε_bg(0.01)
              w = normalize(w)
              feat = Σ w(p) · patch(p)          → (1024,)

Talk2DINO text encoder (CLIP → DINOv2 space):
  question text → text_emb                      → (1024,)

similarity:
  query_sim(t) = cosine(weighted_feat(t), text_emb)   → scalar
  uniform_sim(t) = cosine(uniform_feat(t), text_emb)  → scalar
```

### Gaze+Hand Weighted Pooling

Emphasizes DINOv2 patch features near gaze and hand positions when pooling.
This amplifies visual information from action-relevant regions (hand movement,
gaze fixation) for more accurate similarity measurement with the question.

```
37×37 patch grid (DINOv2 ViT-L/14 @ resize=518)

σ_g = 32/224 = 0.1429    gaze Gaussian std (in normalized [0,1] space)
σ_h = 40/224 = 0.1786    hand Gaussian std (wider than gaze)
ε_bg = 0.01              background preservation constant

Per-frame weight computation:
  w(p) = 0
  if gaze present:    w(p) += exp(-||center(p) - gaze||² / (2·σ_g²))
  if hand_L present:  w(p) += exp(-||center(p) - hand_L||² / (2·σ_h²))
  if hand_R present:  w(p) += exp(-||center(p) - hand_R||² / (2·σ_h²))
  w(p) += ε_bg
  w = w / sum(w)    ← L1 normalization

  If gaze and hand are all null → fallback to uniform pooling (w = 1/1369)
```

### How to Run

```bash
conda activate trajgaze

python /home/yujin/gaze/trajgaze/scripts/compute_talk2dino_similarity.py \
    --data_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
    --batch_size 64 \
    --device cuda:0 \
    --output /home/yujin/dataset/EgoGazeVQA/all_gaze_v1/talk2dino_similarity_results.json \
    --save_npz \
    --npz_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1/query_sim
```

### Output

| File | Purpose |
|------|---------|
| `talk2dino_similarity_results.json` | Full results (analysis, visualization) |
| `query_sim/{dataset}/{video_id}/{qa_hash}.npz` | Per-QA files loaded during training |

npz fields:
- `query_sim`: (T,) float32 — gaze+hand weighted similarity
- `uniform_sim`: (T,) float32 — uniform pooling similarity
- `question`: str — original question text

### Environment Setup

```bash
# trajgaze environment (cloned from gaze)
conda create --clone gaze -n trajgaze
conda activate trajgaze
pip install git+https://github.com/openai/CLIP.git
pip install timm omegaconf webdataset scikit-image
pip install transformers==4.48.3    # Talk2DINO HF model compatibility
```

The Talk2DINO-ViTL checkpoint is automatically downloaded from HuggingFace
on first run (`lorebianchi98/Talk2DINO-ViTL`, ~1.6 GB).

---

## Frame Selector Architecture

### Module Location

`TrajGaze/models/frame_selector.py` → `FrameSelector` class

### Input and Output

```
Input:
  tok_interact  (B, T, 128)   ← gaze-hand interaction token (from TrajectoryTokenizer)
  query_sim     (B, T, 1)     ← precomputed Talk2DINO similarity (optional)

Output:
  scores        (B, T)        ← per-frame attend probability ∈ (0, 1)
```

### Forward Pass

```
tok_interact (B, T, 128)     query_sim (B, T, 1)
        │                         │
        │                    [when use_query=True]
        │                    query_proj:
        │                      Linear(1 → 128) + LayerNorm(128)
        │                         │
        │                         ▼
        │                    (B, T, 128)
        │                         │
        └────── + (additive) ─────┘
                │
                ▼
          (B, T, 128)
                │
          SinusoidalPE(128, max_len=4096)
                │
                ▼
          CausalTransformerLayer × 2
          ┌──────────────────────────────┐
          │ Pre-LayerNorm                │
          │ CausalSelfAttention(4 heads) │  ← upper-triangular mask
          │ Residual + Dropout(0.1)      │     s(t) depends only on t'≤t
          │                              │
          │ Pre-LayerNorm                │
          │ FFN: 128 → 256 → 128 (GELU) │
          │ Residual + Dropout(0.1)      │
          └──────────────────────────────┘
                │
          LayerNorm(128)
                │
          MLP Head:
            Linear(128 → 64) → GELU → Linear(64 → 1)
                │
          Sigmoid
                │
                ▼
          s(t) ∈ (0, 1)   per-frame attend probability
```

### How Query-Aware Selection Works

`query_sim(t)` is a scalar indicating "how visually relevant the gaze+hand
region of this frame is to the question."

`query_proj` expands this scalar to a 128-dim embedding and adds it to the
interaction token. This shifts the CausalTransformer input depending on the
question, causing s(t) to vary accordingly.

```
Example (kitchen knife-cutting video, same clip):

  Q1: "Why did I pick up the knife?"
  frame 42 (gaze → knife):   query_sim=0.52 → query_proj positive → s(42) ↑
  frame 100 (gaze → pot):    query_sim=0.18 → query_proj weak     → s(100) ↓

  Q2: "What did I put into the pot?"
  frame 42 (gaze → knife):   query_sim=0.21 → s(42) ↓
  frame 100 (gaze → pot):    query_sim=0.48 → s(100) ↑

→ Different frames are selected for different questions on the same clip
```

### Backward Compatibility

When `use_query=False` (default), `query_proj` is not created and
`forward(tok_interact)` behaves identically to the original implementation.
Existing checkpoints can be loaded as-is (query_proj weights absent,
loaded with strict=False).

---

## Training Pipeline

### Stage 1 — NTP Pretraining

```bash
python -m TrajGaze.training.stage1 \
    --adapted-dir   datasets/StreamGaze_v2/adapted \
    --interact-dir  datasets/StreamGaze_v2/interaction \
    --frames-dir    datasets/StreamGaze_v2/frames \
    --output-dir    TrajGaze/checkpoints/stage1/full \
    --use-query \
    --epochs 150 --lr 3e-4 --max-frames 800
```

#### Loss Function

```
L = L_frame_BCE + L_NTP + 0.1 · L_traj
```

| Loss | Scope | Description |
|------|-------|-------------|
| L_frame_BCE | All T frames | FrameSelector output s(t) vs attend label |
| L_NTP | Attended frames only | AR Patch Decoder patch ordering prediction |
| L_traj | Attended frames only | Future gaze/hand position prediction (auxiliary) |

#### Query-Aware Attend Label Modulation

When `--use-query` is enabled, attend labels are soft-modulated with query_sim:

```python
# query_sim: (1, T, 1) — precomputed Talk2DINO similarity
q_norm = normalize(query_sim, to=[0, 1])
attend_labels = 0.7 * attend_labels + 0.3 * q_norm
```

| Frame State | Original attend | High query_sim | Low query_sim |
|-------------|----------------|----------------|---------------|
| High interaction (attend=1) | 1.0 | **1.0** | **0.7** (suppressed) |
| Low interaction (attend=0) | 0.0 | **0.3** (boosted) | **0.0** |

BCE loss naturally handles soft targets in [0, 1].
This enables the FrameSelector to learn both interaction importance and
query relevance **simultaneously**.

#### Training Forward Pass Flow

```
1. TrajectoryTokenizer
     gaze/hand/interaction features → tok_interact (1, T, 128)

2. FrameSelector
     tok_interact + query_proj(query_sim) → s(t) per frame
     L_frame_BCE: s(t) vs query-modulated attend labels
     Hard threshold s(t) ≥ 0.5 → attended_idx

3. TwoLevelTrajectoryEncoder (attended frames only)
     trajectory context (n_att, 64, 256)

4. VideoEncoder (attended frames only, max 16 frames)
     RGB + heatmap → visual memory (1, K×196, 192)

5. ARPatchDecoder (teacher-forced, 40 patches per frame)
     cross-attn(visual + trajectory) → logits → L_NTP

6. TrajectoryPredictor (auxiliary)
     mean-pooled visual → future positions → L_traj
```

### Stage 2 — GRPO RL Fine-tuning

```bash
python -m TrajGaze.training.stage2 \
    --stage1-ckpt TrajGaze/checkpoints/stage1/full/epoch_0150.pt \
    --use-query \
    --query-sim-dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1/query_sim \
    ...
```

Stage 2 trains with VLM reward. query_sim is used only as input to the
FrameSelector; attend label modulation is not applied (GRPO does not
use BCE loss).

### Inference

```python
# Top-k frame selection (exactly 50%)
scores, attended_idx = model.selector.select_frames(
    tok_interact, ratio=0.50, query_sim=query_sim
)
```

---

## Related Code Files

| File | Role |
|------|------|
| `TrajGaze/models/frame_selector.py` | FrameSelector module (includes query_proj) |
| `TrajGaze/models/tokenizer.py` | TrajectoryTokenizer (gaze/hand → 128d tokens) |
| `TrajGaze/training/stage1.py` | Stage 1 training (NTP + query-aware attend label) |
| `TrajGaze/training/stage2.py` | Stage 2 training (GRPO + query_sim input) |
| `TrajGaze/training/evaluate.py` | Evaluation (select_frames with query_sim) |
| `scripts/compute_talk2dino_similarity.py` | Preprocessing: Talk2DINO similarity computation |
| `scripts/visualize_similarity_gif.py` | Visualization: per-QA similarity GIF generation |

## References

| Paper | Role |
|-------|------|
| Talk2DINO (ICCV 2025) | CLIP text → DINOv2 space mapping for text-patch similarity |
| A.I.R. (ICLR 2026) | Adaptive frame selection via CLIP similarity |
| Gaze-VLM (NeurIPS 2025) | VLM attention regularization with gaze heatmaps |
| Voila-A (NeurIPS 2024) | Injecting gaze as spatial prior into VLMs |
