# Query-Aware Token Pruning for TrajGaze

**Status:** Design proposal — not yet implemented
**Context:** TrajGaze currently selects frames and patches purely from trajectory signals (gaze, hands). This document explores how to incorporate text query awareness so that, given multiple possible questions about a video, the pruning adapts to what is being asked.

---

## 1. The Problem

In the current Stage 2 setup, TrajGaze always answers `future_action_prediction`. The frame selector and patch decoder have no knowledge of the question — they select tokens based entirely on the gaze-hand trajectory, which is a reasonable proxy for visual salience but is query-agnostic.

When multiple question types are possible (e.g., object identification, spatial relation, action prediction, temporal ordering), the same trajectory-guided selection is suboptimal:

- A **"Where is the red mug?"** question cares about spatial patches, not necessarily attended action frames.
- A **"What will happen next?"** question cares about the transition frames between present and future — exactly what gaze trajectory captures.
- A **"Who is closer to the table?"** question may require different frames than a **"What tool is being used?"** question, even on the same video clip.

The trajectory is a strong **prior** for salience; the query is the **task specification**. Combining them should outperform either alone.

---

## 2. Relevant Prior Work

The following recent papers establish the design space. All confirm that query-conditioned token selection significantly outperforms trajectory/salience-only or attention-only baselines.

| Paper | Venue | Key Idea |
|---|---|---|
| **SparseVLM** | ICML 2025 | Text-token → visual-token attention inside LLM; rank-based adaptive layer ratio. 54% FLOPs ↓, 97% accuracy retained. |
| **IVTP** | ECCV 2024 | Frozen CLIP text CLS gates token pruning inside ViT and first LLM layers in two stages. |
| **LVPruning** | NAACL 2025 | Cross-attention vision→language importance scoring; plug-in module, 90% token reduction with 0.45% avg accuracy loss. |
| **FlashVLM** | arXiv 2024 | Explicit query-visual cosine similarity (not attention weights) for stable pruning at extreme compression. |
| **MustDrop** | arXiv 2024 | Multi-stage: temporal merge → dual-attention (self + cross) prune at prefill → output-aware KV cache eviction at decode. |
| **QTSplus** | arXiv 2024 | Cross-attention + MLP predicts per-instance token budget based on query complexity; straight-through estimator for end-to-end training. |
| **DyToK / Less Is More** | NeurIPS 2025 | Non-uniform per-frame token budget from LLM attention; +20.4% over flat pruning at aggressive compression. |
| **Frame-Voyager** | arXiv 2024 | Trained frame selector conditioned on query, supervised by downstream VLM prediction loss ranking. |
| **SeViLA** | NeurIPS 2023 | VLM used twice: as Localizer (query-conditioned frame scoring) then as Answerer; self-chained refinement. |
| **VideoTree** | CVPR 2025 | CLIP-similarity tree expansion guided by query; adaptive coarse-to-fine frame retrieval. |
| **PruneVid** | ACL 2025 | Three-stage: temporal merge → spatial cluster → question-cross-attention pruning inside LLM prefill. |
| **FastV** | ECCV 2024 | Baseline: pruning by LLM attention after layer 2 (text-agnostic). |

**Core finding from the literature:** The most effective query-aware methods inject the query signal at the *selector* level (before VLM), not only inside the VLM. This avoids spending attention budget on tokens that are irrelevant to the question before they even reach the LLM.

---

## 3. TrajGaze Architecture (Current)

```
Gaze + Hand trajectories
        │
   TrajectoryTokenizer          → tok_gaze, tok_left, tok_right, tok_interact
        │
   FrameSelector                → attended_frames  (top-50% by traj saliency)
        │
   TwoLevelTrajectoryEncoder    → traj_context  (per attended frame, 4·T × 256)
        │
   VideoEncoder                 → visual_mem   (attended frames, K·196 × 192)
        │
   ARPatchDecoder               → patch_indices  (top-40 patches per frame)
        │
   [NVILA / VLM]  ← receives only selected frames with selected patches unmasked
        │
   answer prediction
```

**Query enters only at the VLM.** The selector and decoder are blind to it.

---

## 4. Design Proposals

Three complementary approaches, in increasing complexity. They can be used independently or stacked.

---

### Proposal A — Query-Gated Frame Selector (Lightest Change)

**Where:** `FrameSelector.select_frames()`
**Idea:** The frame selector currently scores frames by trajectory saliency alone. Add a query-conditioned reranking step that shifts frame scores toward frames most relevant to the question.

**Mechanism:**

1. Encode the text query with a frozen CLIP text encoder (already available in the `gaze` env) → query vector `q ∈ ℝ^512`.
2. Encode each candidate frame with the frozen CLIP image encoder → frame vectors `f_t ∈ ℝ^512`.
3. Compute CLIP relevance scores: `r_t = cos_sim(q, f_t)`.
4. Combine with the existing trajectory saliency score `s_t` (output of `FrameSelector`):
   ```
   score_t = (1 - λ) · s_t + λ · r_t
   ```
   where `λ ∈ [0, 1]` controls how much the query influences selection (tunable, e.g. λ=0.3 to start).
5. Select top-K frames by `score_t` as before.

**Why CLIP:** CLIP's joint embedding space aligns text queries with visual content without requiring additional training. It is already used in VideoTree, Free Video-LLM, and HFS for the same purpose.

**Training:** `λ` can be a learned scalar, trained end-to-end via GRPO reward signal. The CLIP encoders remain frozen.

**Cost:** One CLIP forward pass per frame per query — ~negligible vs. NVILA inference.

**Limitation:** CLIP operates at frame level, not patch level. It can rerank frames but cannot guide intra-frame patch selection.

---

### Proposal B — Query-Cross-Attention Patch Decoder (Core Contribution)

**Where:** `ARPatchDecoder.decode_greedy()`
**Idea:** The patch decoder currently autoregressively selects patches based on `traj_context` and `visual_mem`. Add the query as a conditioning signal inside the decoder's attention, so patch selection is jointly guided by trajectory and query.

**Mechanism:**

The `ARPatchDecoder` at each step attends over `visual_mem` (video encoder features). Currently:
```
next_patch = argmax Attention(query_token, visual_mem, traj_context)
```

Proposed: project the text query into the same feature space and concatenate it to the keys/values:

```python
# query_embed: (1, D_text) → project to (1, D_patch)  via a small linear layer
query_proj = self.query_projector(text_embed)  # (1, D_patch)

# Concatenate to the cross-attention context
augmented_context = torch.cat([traj_context, query_proj.unsqueeze(0)], dim=1)

# Decoder cross-attends to augmented context
next_patch = self.cross_attn(decoder_state, augmented_context, visual_mem)
```

The `query_projector` (a single linear layer `ℝ^D_text → ℝ^D_patch`) is the only new trainable parameter.

**Training signal:** The GRPO reward (correct answer = +1) directly supervises which patch selection leads to correct answers — the query projector learns to push patch selection toward query-relevant regions.

**Why this works:** `traj_context` encodes where the user was looking and what their hands were doing (strong prior for action-relevant regions). The query projection adds a task-specific bias. For a "tool identification" question, the query vector shifts attention toward the tool patches; for "what happens next," the gaze trajectory already captures this and the query adds little bias — both cases handled gracefully.

**Cost:** One linear layer (~D_text × D_patch ≈ 512 × 256 = 131K parameters). Forward pass overhead is negligible.

---

### Proposal C — Adaptive Per-Query Token Budget (Advanced)

**Where:** Between `FrameSelector` and the VLM
**Idea:** Motivated by QTSplus and DyToK — different questions require different amounts of visual evidence. A temporal question may need many frames; a single-object question needs very few patches from one frame. A lightweight budget predictor allocates tokens based on query type.

**Mechanism:**

1. Encode query → `q ∈ ℝ^D`.
2. A small MLP `BudgetPredictor(q) → (frame_ratio, patch_ratio)` outputs:
   - `frame_ratio ∈ [0.1, 0.9]`: what fraction of frames to attend (currently fixed at 0.5)
   - `patch_ratio ∈ [0.05, 0.40]`: what fraction of patches per frame to select (currently fixed at 40/196 ≈ 20%)
3. These ratios gate `FrameSelector` and `ARPatchDecoder` respectively.
4. The budget predictor is trained end-to-end: the GRPO reward penalizes token usage (`DELTA_TOKEN` in current code) so the predictor learns to use fewer tokens when the question is easily answered with less context, and more when complex temporal reasoning is required.

**Key insight from DyToK (NeurIPS 2025):** Non-uniform allocation — giving different frame budgets based on how much the LLM attends to each — gives +20.4% over flat pruning at aggressive compression ratios. Applied here, different question types would receive different `frame_ratio` values.

**Training:** The `BudgetPredictor` output is non-differentiable (it sets a discrete K), so use a straight-through estimator or Gumbel-softmax relaxation, as in QTSplus.

---

## 5. Recommended Integration Order

```
Stage 1 (no change):
  Train TrajGazeModel on trajectory prediction — no query signal needed.
  Output: traj_enc, video_enc, decoder weights.

Stage 2 — Phase 1: Add Proposal A only (λ as learnable scalar).
  Verify that query-gated frame selection improves multi-question accuracy
  without changing decoder or budget. Fast to train (same loop as current Stage 2).

Stage 2 — Phase 2: Add Proposal B (query_projector in ARPatchDecoder).
  Freeze trajectory encoder; train query_projector + selector λ jointly via GRPO.
  Expected: patch selection shifts to query-relevant regions even within
  trajectory-attended frames.

Stage 3 (optional): Add Proposal C (BudgetPredictor).
  Requires a multi-question dataset where some QA pairs are trivially answered
  with few frames. Train budget predictor jointly with selector and decoder.
```

---

## 6. Architecture Diagram (Proposed)

```
Text Query  ─────────────────────────────────────────────────────────────┐
       │                                                                  │
  CLIP Text Encoder (frozen)                                             │
       │                                                                  │
  query_embed (512-d)  ──────── λ-weighted CLIP frame scores ────► FrameSelector
                         (Proposal A)                                    │
                                                                   attended_frames
                                                                         │
Gaze + Hand trajectories                                                  │
       │                                                                  │
  TrajectoryTokenizer + TwoLevelTrajectoryEncoder → traj_context          │
                                                         │                │
                                                    VideoEncoder          │
                                                         │                │
                                                    visual_mem            │
                                                         │                │
  query_embed ──── query_projector (linear, trained) ────┤                │
                         (Proposal B)                    ▼                │
                                               ARPatchDecoder ◄───────────┘
                                                         │
                                                  patch_indices
                                                         │
                                                 [NVILA / VLM]
                                                         │
                                               answer prediction
```

---

## 7. Implementation Notes

### Text Encoder
- Use `transformers.CLIPTextModel` with `openai/clip-vit-large-patch14` (already in env, same ViT as used by NVILA's SigLIP tokenizer in spirit).
- Freeze all CLIP parameters.
- Cache `query_embed` per unique question string to avoid redundant forward passes during training (same question is asked for all rollouts of one clip).

### Query Projector (Proposal B)
```python
class QueryProjector(nn.Module):
    def __init__(self, text_dim=512, patch_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )

    def forward(self, text_embed):
        return self.proj(text_embed)  # (1, patch_dim)
```
The `patch_dim=256` matches `TwoLevelTrajectoryEncoder`'s output dimension.

### Frame Score Fusion (Proposal A)
```python
# In FrameSelector.select_frames():
traj_scores = self._compute_traj_scores(tok_interact)   # existing
clip_scores = self._compute_clip_scores(frame_paths, query_embed)  # new
fused_scores = (1 - self.lam) * traj_scores + self.lam * clip_scores
attended_idx = fused_scores.topk(k).indices
```
`self.lam = nn.Parameter(torch.tensor(0.3))` — initialized to give trajectory 70% weight.

### CLIP Frame Scoring
CLIP operates on full frames (PIL images), not pre-loaded tensors. Frame paths are available from `item["frame_names"]`. One CLIP-ViT-L/14 forward pass per frame: ~4ms on H200, negligible vs. NVILA's 2–15s.

### Multi-Question Dataset
Proposals A–C require a QA file with multiple question types per clip. The current file has only `future_action_prediction`. To test query-awareness properly:
- Mix with other StreamGaze QA types, or
- Use EgoSchema, EgoVQA, or NExT-QA annotations overlapping with the same clips.

Without this, all three proposals are equivalent to the current single-question setup and cannot learn meaningful query-dependent variation.

---

## 8. Expected Gains

| Setting | Baseline (current) | + Proposal A | + Proposal B | + Proposal C |
|---|---|---|---|---|
| Single question type | 47–60% val acc | Same (λ→0, no gain) | Same (query always identical) | Same |
| 2–4 question types | TBD | +3–5% expected | +5–10% expected | +2–4% additional |
| Many question types | TBD | +5–8% expected | +8–15% expected | +5–8% additional |

*Estimates based on analogous gains reported in SparseVLM (+3.2% over FastV at same compression), Frame-Voyager (+4.7% over uniform sampling), and QTSplus (+5.6% on temporal order tasks).*

---

## 9. Priority

1. **Collect/integrate a multi-question dataset** — without this, none of the proposals can demonstrate query-specific gains.
2. **Implement Proposal A** — 1 day of engineering, lightweight CLIP scoring layer, directly plugs into existing `FrameSelector`.
3. **Implement Proposal B** — 2–3 days, adds `QueryProjector` to `ARPatchDecoder`, retrain Stage 2 from Stage 1 checkpoint.
4. **Evaluate with current single-question setup** to confirm Proposals A+B don't degrade trajectory-only performance before introducing multi-question data.
5. **Proposal C** is optional and requires the multi-question dataset to be meaningful.

---

## References

1. Chen et al., *SparseVLM: Visual Token Sparsification for Efficient VLM Inference*, ICML 2025. arXiv:2410.04417
2. Yang et al., *IVTP: Instruction-Guided Visual Token Pruning for Large VLMs*, ECCV 2024.
3. Ye et al., *LVPruning: Language-Guided Vision Token Pruning*, NAACL 2025. arXiv:2501.13652
4. Wang et al., *FlashVLM*, arXiv:2512.20561, 2024.
5. Liu et al., *MustDrop*, arXiv:2411.10803, 2024.
6. Cao et al., *MADTP: Multimodal Alignment-Guided Dynamic Token Pruning*, CVPR 2024.
7. Hu et al., *QTSplus: Query-Aware Tokenizer for Long-Video MLLMs*, arXiv:2511.11910, 2024.
8. Zhang et al., *Less Is More, but Where? (DyToK)*, NeurIPS 2025. arXiv:2512.06866
9. Zhang et al., *Frame-Voyager*, arXiv:2410.03226, 2024.
10. Yu et al., *SeViLA*, NeurIPS 2023. arXiv:2305.06988
11. Wang et al., *VideoTree*, CVPR 2025. arXiv:2405.19209
12. Xu et al., *Free Video-LLM*, arXiv:2410.10441, 2024.
13. Li et al., *PruneVid*, ACL 2025. arXiv:2412.16117
14. Chen et al., *DyCoke*, CVPR 2025. arXiv:2411.15024
15. Bolya et al., *Token Merging (ToMe)*, ICLR 2023. arXiv:2210.09461
16. Chen et al., *FastV*, ECCV 2024 (Oral). arXiv:2403.06764
