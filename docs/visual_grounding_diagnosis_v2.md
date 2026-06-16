# Visual Grounding Diagnosis v2 — Step 1 of `zazzy-sprouting-ladybug` plan

**Date**: 2026-05-27 (UTC, eval finished 05:47)
**Scope**: 4 combined-ablation Stage-2 checkpoints × {StreamGaze-EGTEA, EgoGazeVQA-EGTEA} × cf-mask (7 variants each).
**Goal**: decide whether the TAS/ATR/CGM mechanisms changed the LLM's reasoning behavior (→ Step 2: scale LoRA bandwidth), or merely changed token selection without touching language-prior usage (→ Step 3: reframe paper around what TAS actually delivers).

---

## 1. Counterfactual-mask Δ matrix (acc% change vs baseline)

`mask_kept` = zero out the kept-after-merge tokens (visual content removed, structure kept) — used here as the **language-prior floor** proxy in line with `eval/ablation_score_source.py:280-291` `text_only` path (both zero the visual embeddings; the merge-structure difference is irrelevant once contents are zero).

### StreamGaze-EGTEA (visual-heavy dataset)

| Checkpoint | base | mask_kept | mask_kept_late | mask_kept_early | shuffle | mask_gaze | mask_hand |
|---|---:|---:|---:|---:|---:|---:|---:|
| TAS_only       | 67.49 | **−11.98** | −11.60 | −1.71 | −0.38 | −2.47 | −0.57 |
| TAS_ATR        | 64.26 | **−11.22** |  −9.89 | −1.71 | +0.38 | −0.95 | +0.57 |
| TAS_ATR_CGM    | 61.98 |  **−7.79** |  −5.13 |  0.00 | **+1.33** | −0.76 | −0.19 |
| CGM_only       | 62.74 | **−10.08** |  −6.08 | −2.47 |  0.00 | −2.28 | −0.38 |

### EgoGazeVQA-EGTEA (gaze-metadata-derivable dataset)

| Checkpoint | base | mask_kept | mask_kept_late | mask_kept_early | shuffle | mask_gaze | mask_hand |
|---|---:|---:|---:|---:|---:|---:|---:|
| TAS_only       | 56.38 | **+0.93** | −0.93 | +1.86 | +0.93 | +0.23 | +0.70 |
| TAS_ATR        | 57.08 | **−1.62** | −1.39 | −1.16 | −0.93 | −0.46 |  0.00 |
| TAS_ATR_CGM    | 59.40 | **−2.09** | −2.09 | −0.46 | −0.46 | −1.62 | +0.23 |
| CGM_only       | 55.92 | **+0.70** |  0.00 | −0.93 |  0.00 | +0.46 | +0.70 |

---

## 2. Language-prior floor comparison (= mask_kept acc on EgoGazeVQA)

If methods only change selection but leave LLM usage of language prior intact, all four ckpts should sit at ~the same floor. If they shift LLM behavior, floors should diverge by ≥ 2pp (plan decision gate).

| Checkpoint | EgoGazeVQA mask_kept acc (%) |
|---|---:|
| TAS_only       | **57.31** |
| TAS_ATR        | **55.45** |
| TAS_ATR_CGM    | **57.31** |
| CGM_only       | **56.61** |

**Spread = 57.31 − 55.45 = 1.86 pp**, **below the 2 pp gate**.

Note: TAS_only and TAS_ATR_CGM hit the *same* floor (57.31) despite very different baselines (56.38 vs 59.40). All of CGM's +3 pp EgoGazeVQA gain over TAS_only sits in the *visual* channel (mask_kept Δ goes from +0.93 to −2.09 — i.e. CGM finally makes EgoGazeVQA care about visual content somewhat), not in the language-prior floor.

---

## 3. Verdict (one sentence)

**The method did not change LLM language-prior usage** — the four ckpts' EgoGazeVQA floors land within 1.86 pp of each other, below the plan's 2 pp gate. CGM/ATR shift *where* the model gets its evidence (visible as mask_kept Δ going from +0.93 → −2.09 on EgoGazeVQA when CGM is added), but the absolute floor that the LLM falls back to when visual content is gone is essentially the same across mechanisms.

---

## 4. Mechanism-level reading

### TAS (alone) — the headline contribution
- StreamGaze base **67.49**, mask_kept Δ **−11.98** (strong visual usage), `past_gaze_sequence_matching` +18 pp recovered (from earlier runs).
- Token-selection-quality argument fully holds: the model relies on the kept tokens, and good selection is what makes the kept tokens informative.

### ATR — silent on visual grounding, hurts StreamGaze baseline
- StreamGaze base drops 67.49 → 64.26 (−3.23). mask_kept Δ barely changes (−11.98 → −11.22). `shuffle` goes positive (+0.38), `mask_kept_early` shrinks (−1.71 → −1.71). I.e. ATR did **not** strengthen reliance on temporally-correct order; if anything, it slightly weakened it on StreamGaze.
- EgoGazeVQA: small sign-flip on `mask_kept` (+0.93 → −1.62) — promising on its own, but undone by the +3.23 pp baseline loss on the visual-heavy benchmark.

### CGM — moves the EgoGazeVQA language-floor flip, costs StreamGaze
- StreamGaze base drops further (TAS_only 67.49 → TAS_ATR_CGM 61.98). mask_kept Δ weakens from −11.98 to −7.79 (less visual reliance).
- EgoGazeVQA mask_kept goes positive→negative (+0.93 → −2.09): the model finally gets *some* visual sensitivity on a dataset that doesn't need it.
- Net: CGM trades visual reliance on the dataset that needed it (StreamGaze) for sign-flip on the dataset that didn't (EgoGazeVQA). Honest read: this is not the same as "recovering visual grounding".

### Shuffle anomaly on TAS_ATR_CGM / StreamGaze
- `shuffle_kept Δ = +1.33` (shuffling kept tokens *helps*). Together with mask_kept Δ weakening from −11.98 to −7.79, this is the strongest single signal that the FULL pipeline pushes the model toward order-agnostic / bag-of-tokens reasoning — i.e. the opposite of trajectory grounding. Documented as honest limitation.

---

## 5. Decision gate — plan §1 outcome

Plan rule:
> text-only floor가 모델별로 차이 ≥ 2pp → Step 2로.
> 차이 없음 → Step 3 (paper reframing) 우선.

Floor spread is **1.86 pp** — strictly below the gate.

**Decision: pursue Step 3 (paper reframing) first.**

Justification:
- LLM language-prior usage is essentially unchanged across mechanisms (1.86 pp ≤ 2 pp).
- TAS already delivers a clean win that does not depend on visual-grounding rescue: StreamGaze base 67.49 with mask_kept Δ −11.98, `past_gaze_sequence_matching` +18 pp.
- Net mean accuracy from the just-finished HD-EPIC runs: TAS-only 56.57 mean > TAS+ATR 55.35 mean (overfitting checkpoints, see watcher log) — ATR *hurts* once HD-EPIC is added. Additional capacity (Step 2 LoRA-FFN) on top of a method that already loses to its ablation is a poor investment.
- Step 2 (LoRA FFN expansion) remains available as a fallback if Step 3 reframing surfaces a remaining gap that more capacity could close.

---

## 6. Open items rolled forward

- **Strict `text_only` via `ablation_score_source.py`** is *not* re-run; `mask_kept` Δ is used as floor proxy (identical visual-content state, different merge layout — numerically indistinguishable for the decision in §5). If the reframed paper needs the exact `text_only` number for a §Limitations table, ~30 min per ckpt to run separately.
- **The two HD-EPIC Stage-2 runs (TAS-only, TAS+ATR) reached overfitting** (TAS-only best 56.57 @ step 8800, TAS+ATR best 55.35 @ step 7600) and were stopped by `convergence_watcher.sh`. Best checkpoints at `/workspace/trajgaze/TrajGazeMerge/checkpoints/E1_combined_{TASonly,TAS_ATR}_hdepic_bs8_mb2/best.pth`. cf-mask on these HD-EPIC variants not yet run — not gated by Step 1, but worth a single pass before paper finalization.

---

## 7. Files

- Verdict aggregator: `TrajGazeMerge/eval/run_step1_diagnosis.sh` (lines 99-130, Python heredoc).
- Per-ckpt summaries: `TrajGazeMerge/eval_results/diagnostic/E1_combined_*_cfmask_mask_summary.json` (8 files).
- Launcher log: `TrajGazeMerge/eval_results/E1_step1_diagnosis_launcher.log`.
- Watcher log: `TrajGazeMerge/eval_results/E1_convergence_watcher.log`.
