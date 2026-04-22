# TrajGazeMerge: Gaze-Guided Visual Token Merging for Egocentric VLMs

## Motivation

TrajGaze_v2 uses egocentric gaze trajectories to **select** the top-K most task-relevant visual patches, discarding the rest. Our StreamGaze_v2 evaluations showed:

| Task | n | No Visual | Random 10% | Baseline 100% | Visual Gap |
|---|---|---|---|---|---|
| Future Action Prediction | 921 | 25.6% | 30.8% | 31.9% | +6.3pp |
| Past Non-Fixated Object ID | 650 | 40.2% | — | 52.5% | +12.3pp |
| Past Scene Recall | 211 | 37.4% | — | 41.7% | +4.3pp |
| Past Gaze Sequence Matching | 186 | 33.3% | 40.9% | 46.8% | +13.4pp |
| Past Object Transition Prediction | 494 | 30.0% | 34.0% | 34.6% | +4.7pp |
| Present Object Attr. Recognition | 1419 | 42.0% | 45.4% | 44.2% | +2.2pp |
| Present Object ID (Easy) | 1487 | 28.9% | 44.0% | 45.5% | +16.6pp |
| Present Object ID (Hard) | 1005 | 37.4% | 44.0% | 46.4% | +8.9pp |

Two problems with token selection (TrajGaze_v2):
1. Dropped tokens are permanently lost — the VLM sees OOD sparse inputs it was never trained for
2. Selector-only training ceiling is bounded by the frozen VLM's 100% accuracy — no room to exceed it

**Token merging** offers a complementary approach: instead of dropping low-importance tokens, merge similar ones into their neighbors, preserving all information in compressed form. Combined with VLM LoRA adaptation, the model can co-adapt to merged representations and potentially exceed the frozen-VLM ceiling.

**The core idea**: use gaze trajectories to guide *which* tokens get merged and *which* are preserved intact, then jointly train the gaze encoder and VLM LoRA layers to operate on this compressed representation.

---

## Background

### Standard Token Merging (ToMe, Bolya et al. 2022)

Inside each attention block, after computing Q, K, V:
1. Split tokens into two sets A (sources) and B (receivers)
2. For each source, find its most similar receiver by cosine similarity of K vectors
3. Average matched pairs: `merged = (source + receiver) / 2`
4. Sequence length drops by r tokens per layer

**Problem**: globally similar tokens are merged regardless of semantic importance. A repeating background texture might be more similar than two semantically distinct foreground objects.

---

## Proposed Method: TrajGazeMerge

### Core Insight

Gaze trajectory tells us **where task-relevant information is concentrated**:

- **High gaze score** → patch carries unique, attended information → **protect from merging** (receiver)
- **Low gaze score** → patch is peripheral, likely redundant → **prefer as merge source**

Rather than splitting tokens randomly (ToMe), gaze scores make this assignment semantically meaningful. The VLM is then jointly adapted via LoRA to work with merged representations.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Past Gaze Trajectory               │
│            (fixation sequence + frame paths)         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   TrajGaze Encoder      │   ← trainable (Stage 1 pretrained init)
         │  (Stage 1 pretrained)   │
         └────────────┬────────────┘
                      │
                      ▼
           patch_scores ∈ ℝ^{N}          ← one score per spatial patch
                      │
          ────────────┼──────────────────────────────────
                      │                                  │
                      ▼                                  ▼
         ┌────────────────────────┐     ┌───────────────────────────────┐
         │   Video Frames         │     │   Gaze-Weighted Bipartite      │
         │   → Vision Encoder     │────▶│   Matching & Merge             │
         │   (frozen ViT)         │     │   (no ViT modification)        │
         │   → visual tokens      │     └──────────────┬────────────────┘
         │     (T×N_spatial, d)   │                    │
         └────────────────────────┘                    ▼
                                         Merged tokens (T×K, d), K << N_spatial
                                                        │
                                                        ▼
                                          ┌─────────────────────────┐
                                          │   VLM LLM Layers        │   ← LoRA trainable
                                          │   (Qwen2.5-VL-7B)       │
                                          └─────────────┬───────────┘
                                                        │
                                                        ▼
                                                   answer logits
```

**Frozen**: Vision encoder (ViT)
**Trainable**: TrajGaze encoder + VLM LoRA (LLM layers only)

### Gaze-Weighted Bipartite Matching

```python
# patch_scores: (N,) — from TrajGaze encoder, interpolated to match VLM token grid
# tokens:       (N, d) — visual tokens after vision encoder

# 1. Score-guided receiver/source assignment
threshold = patch_scores.topk(k=N - r).values.min()
is_receiver = patch_scores >= threshold      # (N,) — top (N-r) protected
receivers = tokens[is_receiver]              # (N-r, d)
sources   = tokens[~is_receiver]             # (r, d) — to be merged

# 2. Find best receiver for each source (cosine similarity of K vectors, as in ToMe)
sim = F.normalize(sources, dim=-1) @ F.normalize(receivers, dim=-1).T   # (r, N-r)
best_match = sim.argmax(dim=-1)             # (r,)

# 3. Gaze-weighted merge: high-score receivers dominate
w_r = patch_scores[is_receiver][best_match]         # (r,)
w_s = patch_scores[~is_receiver]                    # (r,)
receivers = receivers.clone()
receivers.index_add_(0, best_match,
    (sources * w_s.unsqueeze(-1) - receivers[best_match] * w_r.unsqueeze(-1))
    / (w_r + w_s).unsqueeze(-1))

merged_tokens = receivers                           # (N-r, d)
```

Gaze-attended patches keep embeddings nearly intact (`w_r >> w_s`). Peripheral patches flow signal into their nearest attended neighbor.

---

## Training Strategy

### Stage 1: Unchanged from TrajGaze_v2 (already complete)

Train TrajGaze encoder to predict future patch importance from past gaze:
- `L_traj`: trajectory prediction loss
- `L_score`: interaction score MSE
- `L_attn`: MSE between patch_scores and future attended regions

Stage 1 checkpoint is reused as initialization for Stage 2.

### Stage 2: Joint Encoder + LoRA Training

Both the TrajGaze encoder and the VLM LoRA layers are trained end-to-end.

**Two forward passes per step:**

```
Teacher pass (frozen VLM, full tokens, no grad):
    frames → frozen ViT → all N tokens → frozen VLM → logits_teacher

Student pass (trainable):
    frames → frozen ViT → all N tokens
           → TrajGaze encoder → patch_scores
           → gaze-weighted merge → K merged tokens
           → VLM + LoRA → logits_student
```

**Loss:**
```
L_distill = KL(logits_student || logits_teacher)   # match full-token distribution
L_task    = CrossEntropy(logits_student, label)    # task supervision
L_total   = α * L_distill + (1 - α) * L_task
```

**Why two losses:**
- `L_task` alone: student may learn shortcuts that satisfy labels but diverge from full-token behavior
- `L_distill` alone: student targets an unattainable ceiling if merged tokens structurally can't match full tokens
- Together: `L_distill` provides dense token-level signal; `L_task` grounds it in task performance

**Gradient flow:**
```
L_total → logits_student → VLM LoRA (weight update)
                         → merged_tokens → merge op (differentiable weighted avg)
                                        → patch_scores → TrajGaze encoder (weight update)
```

**What stays frozen:** Vision encoder (ViT), VLM non-LoRA weights

**LoRA config (Qwen2.5-VL-7B):**
- Apply to: Q, K, V, O projections in LLM transformer layers
- Rank: r=16, alpha=32
- Target: LLM layers only (not ViT, not projector/connector)

---

## Baseline Comparison

Since VLM LoRA is part of TrajGazeMerge, **all baselines must also use LoRA** for a fair comparison. The independent variable is the compression method; LoRA adaptation is held constant.

| Method | Guidance | Compression | LoRA | Notes |
|---|---|---|---|---|
| Full tokens + LoRA | — | None | ✓ | Upper bound; also shows LoRA's own benefit |
| No visual + LoRA | — | — | ✓ | Lower bound |
| Random merge + LoRA | Random | Merge r tokens | ✓ | Ablates gaze guidance |
| Standard ToMe + LoRA | Similarity | Merge r tokens | ✓ | Ablates gaze guidance, uses K-similarity |
| TrajGazeMerge + LoRA | Gaze | Merge r tokens | ✓ | **Proposed method** |

**Key ablation**: TrajGazeMerge vs Standard ToMe (both with LoRA, same r) isolates the value of gaze-guided assignment over similarity-guided assignment.

**Additional zero-shot reference** (no LoRA, no training):
| Method | Notes |
|---|---|
| Full tokens, frozen VLM | Current evaluated baseline |
| Random 10%, frozen VLM | Already evaluated |
| TrajGazeMerge zero-shot | Stage 1 scores only, no distillation, no LoRA |

---

## Implementation Plan

### File Structure

```
TrajGazeMerge/
├── PROPOSAL.md
├── models/
│   ├── merge.py          # gaze_weighted_merge() — standalone, differentiable
│   └── model.py          # TrajGazeMergeModel: encoder + merge hook + LoRA VLM
├── training/
│   └── train.py          # Stage 2 joint training loop
└── eval/
    └── evaluate.py       # evaluation on StreamGaze tasks
```

### Step 1: Merge Module (`models/merge.py`)

Implement `gaze_weighted_merge(tokens, patch_scores, r)` as a standalone differentiable function:
- Input: `tokens (N, d)`, `patch_scores (N,)`, merge count `r`
- Output: `merged (N-r, d)`
- No dependencies on VLM internals — works on any token sequence

### Step 2: Model Wrapper (`models/model.py`)

`TrajGazeMergeModel` wraps Qwen2.5-VL-7B:
- Loads TrajGaze encoder (Stage 1 checkpoint)
- Loads Qwen2.5-VL-7B, applies LoRA to LLM layers via `peft`
- Hooks into Qwen's forward: intercept visual tokens after vision encoder (`visual_features`), apply gaze-weighted merge, then pass merged tokens to LLM
- Score resolution: TrajGaze produces 14×14 = 196 scores; Qwen token grid may differ → bilinear interpolation (same as TrajGaze_v2)
- Frozen: ViT, Qwen non-LoRA weights

### Step 3: Training Loop (`training/train.py`)

```
for batch in dataloader:
    # Teacher pass (no_grad)
    with torch.no_grad():
        logits_teacher = frozen_vlm(full_tokens)

    # Student pass
    patch_scores = traj_encoder(gaze_trajectory, visual_features)
    merged = gaze_weighted_merge(visual_features, patch_scores, r=merge_ratio*N)
    logits_student = vlm_lora(merged)

    loss = alpha * KL(logits_student, logits_teacher) \
         + (1 - alpha) * CE(logits_student, labels)
    loss.backward()
    optimizer.step()   # updates: encoder params + LoRA params
```

- Dataset: StreamGaze_v2 train split (same items used for evaluation)
- Optimizer: AdamW, separate LR for encoder vs LoRA (LoRA typically 1e-4, encoder 5e-5)
- Merge ratio r: fixed at 50% initially, then sweep (25%, 50%, 75%)
- α: 0.5 as default, treat as hyperparameter

### Step 4: Evaluation (`eval/evaluate.py`)

Same protocol as `TrajGaze_v2/eval/evaluate_streamgaze.py`:
- All 8 StreamGaze MCQ tasks
- Report per-dataset (egtea, egoexolearn, holoassist) and overall accuracy
- Compare against all baselines in the table above

---

## Key Design Decisions

### Where to apply merging

**After vision encoder, before LLM** (not layer-by-layer inside ViT).

Rationale:
- No ViT modification needed — cleaner implementation
- LoRA on LLM layers compensates for the fact that merging happens at the boundary
- Layer-by-layer inside ViT is stronger but requires patching Qwen's ViT forward pass and is harder to maintain

### Score resolution handling

TrajGaze produces 14×14 = 196 scores per frame. Qwen's vision encoder produces a different token grid per frame depending on input resolution. Use bilinear interpolation to upsample/downsample patch_scores to match the actual token grid size at runtime.

### Merge ratio

Start with a fixed ratio (50% token reduction). After initial results:
- If accuracy is close to baseline → try higher compression (75%)
- If accuracy drops significantly → try lower compression (25%)
- Adaptive ratio based on gaze entropy is a future extension

### Temporal merging

Apply merge independently per frame (spatial-only merge) in the first implementation. Cross-frame temporal merging (using `patch_scores_temporal.mean(dim=0)`) is a follow-up extension once spatial merge is validated.

---

## Expected Results

With joint LoRA adaptation:
- **Full tokens + LoRA** may slightly exceed frozen baseline (LoRA itself helps)
- **TrajGazeMerge + LoRA** should approach Full tokens + LoRA at 50% token budget
- **TrajGazeMerge + LoRA** should outperform **Standard ToMe + LoRA** on high-gap tasks (Non-Fixated Object ID, Gaze Sequence Matching) where gaze guidance is most informative

If TrajGazeMerge + LoRA ≈ Full tokens + LoRA at 50% tokens: strong result — same accuracy at half the LLM compute.
