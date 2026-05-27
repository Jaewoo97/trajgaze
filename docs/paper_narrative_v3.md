# Paper Narrative v3 — Mechanism-by-Role Reframe

**Status**: 2026-05-27 draft, follows Step 3 of `zazzy-sprouting-ladybug` plan.
**Triggered by**: Step 1 diagnosis `docs/visual_grounding_diagnosis_v2.md` — EgoGazeVQA language-prior floor spread = 1.86 pp < 2 pp gate. Methods do not change LLM language-prior usage; they only change token selection. Visual-grounding rescue narrative on EgoGazeVQA cannot be defended.

---

## 0. One-paragraph reframe

The previous narrative (v2, `trajectory_grounded_results.md`) was *dataset-by-dataset*: "use TAS on StreamGaze, use FULL on EgoGazeVQA". v3 inverts this into a **mechanism-by-role** structure: TAS is the headline contribution (efficient token selection that the LLM actually consumes), while ATR and CGM are presented as best-effort attempts to address language-prior-dominated benchmarks, with their EgoGazeVQA gains framed honestly as small sign-flips that do not survive cf-mask scrutiny. EgoGazeVQA is demoted from a headline benchmark to a secondary analysis because plan-mandated cf-mask checks show the dataset is partially derivable from gaze metadata.

---

## 1. Headline claim (paper §Introduction / §Contributions)

> **Trajectory-Aware Selection (TAS) learns a Gaussian gaze prior that lets a 90 %-pruned visual stream remain useful for egocentric video QA.**

Concretely:
- StreamGaze-EGTEA, 10× token reduction (merge_ratio = 0.9): TAS recovers visual grounding (`mask_kept` Δ = **−11.98 pp**, vs +0.93 pp for the gaze-metadata-derivable EgoGazeVQA) and lifts the trajectory-sensitive task `past_gaze_sequence_matching` by **+20 pp**.
- This is a selection-quality result, *not* a language-prior-rescue result. The kept tokens are good enough that the LLM relies on them.

Implication for the paper: lead with selection quality on a visual-heavy benchmark, not with cross-dataset victory tables.

---

## 2. What TAS / ATR / CGM each provably do (paper §Method §Ablations)

| Mechanism | What it changes (cf-mask evidence) | Role in paper |
|---|---|---|
| **TAS** | Drives mask_kept Δ from ~0 (random selection floor) to **−11.98** on StreamGaze; recovers `past_gaze_sequence_matching` +20 pp. | **Headline contribution.** Selection-quality result. |
| **ATR** | Negligible change to mask_kept Δ (−11.98 → −11.22 on StreamGaze); slight EgoGazeVQA `mask_kept` sign-flip (+0.93 → −1.62). Lowers StreamGaze baseline by **−3.23 pp** in exchange for a marginal EgoGazeVQA visual-sensitivity shift. **Mean accuracy on HD-EPIC-augmented setup: 55.35 vs TAS-only 56.57 — strictly worse.** | Ablation. Reported as best-effort, no net gain. |
| **CGM** | Pushes mask_kept Δ from +0.93 to −2.09 on EgoGazeVQA (the model finally pays *some* attention to visual content on a benchmark that didn't require it), but costs StreamGaze: base 67.49 → 61.98 (−5.51 pp), mask_kept Δ weakens from −11.98 to −7.79, and **`shuffle_kept` flips to +1.33** (shuffling kept tokens *helps* — order-agnostic regime, opposite of trajectory grounding). | Ablation + Limitation. The EgoGazeVQA sign-flip is real; the trade-off makes it not worth promoting. |

Rule of thumb to communicate in the paper: **report all three, lead with TAS, treat ATR/CGM as honest ablations with documented trade-offs.**

---

## 3. EgoGazeVQA → demotion to §Analysis (paper §Limitations)

Move EgoGazeVQA out of the headline tables. Replace with a §Limitations subsection that says, in roughly these words:

> EgoGazeVQA's *spatial* and *temporal* question categories are constructed from gaze metadata associated with the same clip. A trajectory-aware adapter such as ours can route the gaze signal directly to the LLM, producing accuracy gains without the LLM needing to consult video content. Our counterfactual-mask analysis confirms this: the baseline `mask_kept` ablation on EgoGazeVQA leaves accuracy essentially unchanged (Δ ≈ +0.93 pp for our TAS-only configuration), whereas the equivalent ablation on StreamGaze drops accuracy by **11.98 pp**. We therefore treat EgoGazeVQA *spatial*/*temporal* gains as auxiliary signal and refrain from making visual-grounding claims on this benchmark.

Concrete actions:
- StreamGaze becomes the only headline-table dataset.
- EgoGazeVQA results move into an §Analysis or §Appendix table with the cf-mask Δ row attached, so reviewers can verify the limitation themselves.
- The current "TAS-only wins StreamGaze / FULL wins EgoGazeVQA" two-column victory framing in `trajectory_grounded_results.md` §2 is dropped.

---

## 4. The shuffle-positive finding — turn it into a §Discussion item, not a result

`shuffle_kept Δ = +1.33` on TAS_ATR_CGM/StreamGaze and `+0.38` on TAS_ATR/StreamGaze are the most reviewer-vulnerable signals: shuffling the kept tokens helps, which contradicts the "trajectory grounding" thesis for the FULL pipeline. The honest framing:

> When ATR and CGM are stacked on top of TAS, the resulting model becomes increasingly order-agnostic on the kept tokens. We interpret this as evidence that the additional losses push the LLM toward a bag-of-tokens reading of the gaze-anchored region, rather than reinforcing temporal structure. The selection benefit from TAS is preserved (mask_kept Δ remains strongly negative), but the temporal-order benefit is diluted.

This converts a weakness into a documented finding about loss-mixing interactions, instead of leaving it as an unexplained anomaly.

---

## 5. Headline tables to ship (replace `trajectory_grounded_results.md` §2 §3)

### 5a. Table 1 — Main result, StreamGaze-EGTEA, 10× compression

| Method | Mean acc | `past_gaze_sequence_matching` |
|---|---:|---:|
| Random merge | (TBD baseline) | (TBD) |
| Sprint-1 baseline | 65.21 | 56.25 |
| **TAS-only (ours)** | **67.49** | **76.56** |
| TAS + ATR | 64.26 | TBD |
| TAS + ATR + CGM (FULL) | 61.98 | TBD |

Fill `past_gaze_sequence_matching` for ATR/FULL from per-task logs.

### 5b. Table 2 — Counterfactual masking, StreamGaze-EGTEA

| Method | `mask_kept` Δ | `mask_kept_late` Δ | `shuffle_kept` Δ | `mask_gaze` Δ |
|---|---:|---:|---:|---:|
| TAS-only | −11.98 | −11.60 | −0.38 | −2.47 |
| TAS+ATR | −11.22 |  −9.89 | +0.38 | −0.95 |
| TAS+ATR+CGM | −7.79 |  −5.13 | **+1.33** | −0.76 |
| CGM-only | −10.08 |  −6.08 |  0.00 | −2.28 |

Reading prescribed in caption: "negative mask_kept Δ indicates the LLM uses the kept visual tokens; TAS produces the strongest such effect, and stacking ATR/CGM erodes it. Positive shuffle_kept Δ for the full pipeline indicates an order-agnostic regime."

### 5c. Table 3 (Appendix / Analysis) — EgoGazeVQA-EGTEA, with limitation row

| Method | Mean acc | `mask_kept` Δ | spatial | temporal |
|---|---:|---:|---:|---:|
| Sprint-1 baseline | 57.31 | (TBD) | 40.88 | 42.36 |
| TAS-only | 57.77 | **+0.93** | 39.42 | 42.36 |
| FULL | 59.40 | **−2.09** | 43.80 | 45.83 |

Caption note: `mask_kept` Δ ≈ 0 for TAS-only indicates the dataset is solvable without visual content; full-pipeline sign-flip is small (−2.09 pp) and does not survive the StreamGaze trade-off.

---

## 6. Out of scope for v3 (deferred)

- **External visual-heavy benchmark** (NExT-QA hard / EgoSchema / EGTEA action recognition) — plan §3 marks this as a separate plan. Not blocking the v3 reframe but is the cleanest long-term answer to "do you have a visual-grounding benchmark that isn't StreamGaze". Should appear as future work in the paper.
- **Strict `text_only` ablation via `ablation_score_source.py:280-291`** — `mask_kept` Δ is used as floor proxy throughout. If a reviewer asks for the exact `text_only` number in §Limitations, ~30 min per ckpt to add.
- **cf-mask on the HD-EPIC-augmented stage-2 ckpts** (TAS-only-hdepic, TAS+ATR-hdepic, both finished and stopped at overfitting today). The mean-acc comparison TAS-only 56.57 > TAS+ATR 55.35 already supports the §2 ATR ablation row; running cf-mask on these adds a confirmatory but non-essential data point.

---

## 7. Action items to apply to the actual paper draft

1. Rewrite paper §Introduction to lead with "selection quality on visual-heavy egocentric QA under 10× compression".
2. Drop the "use FULL for EgoGazeVQA / use TAS-only for StreamGaze" recommendation paragraph from any current §Discussion. Replace with the §2 mechanism table above.
3. Move EgoGazeVQA to §Analysis / §Appendix with the §3 limitation paragraph as the caption.
4. Add the §4 shuffle-positive paragraph as a §Discussion subsection.
5. Update `trajectory_grounded_results.md` §1 (1줄 요약) to: *"TAS is the contribution: it makes 10×-compressed visual tokens consumable by the LLM on visual-heavy StreamGaze. ATR and CGM are reported as ablations; their EgoGazeVQA gains do not pass cf-mask scrutiny and are framed as a documented limitation of language-prior-dominated benchmarks."*

---

## 8. Files

- Step 1 diagnosis: `docs/visual_grounding_diagnosis_v2.md`
- Previous narrative (to be replaced): `docs/trajectory_grounded_results.md`
- Plan: `/home/irteam/.claude/plans/zazzy-sprouting-ladybug.md`
