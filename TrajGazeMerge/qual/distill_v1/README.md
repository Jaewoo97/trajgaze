# Qualitative token-selection figures — specialist KD students

What each figure shows: one benchmark item, one frame strip, three rows. Every row spends the
**same 10% visual-token budget** (7% content + 3% complement) on the **same frozen
Qwen2.5-VL-7B backbone**, and each row loads **its own LoRA adapter**. Only the 3% complement's
*selector* — and, for row 3, the pixels it selects from — differ.

| row | adapter | who picks the 3% | frames drawn |
|---|---|---|---|
| **M1 teacher** | `visionzip_complement_learned_{SG,EG}only_overlay` | gaze/hand → frozen TAS encoder | `viz` / `gaze` |
| **KD (gaze-overlay)** | `visionzip_kd_selection_{SG,EG}only_overlay` | `TrajSaliencePredictor`, RGB only | `viz` / `gaze` |
| **KD (raw video)** | `visionzip_kd_selection_{SG,EG}only_nooverlay` | same module, own weights | `original` / `no_gaze` |

Colours: **green outline** = content-based 7% (VisionZip 3.5% dominant + 3.5% contextual).
**magenta fill** = that row's 3% complement. **yellow ring** = ground-truth gaze, drawn on
rows 1–2 only — row 3 has neither the pixel marker nor any gaze input, so a ring there would
assert an input it does not have.

Row naming follows `kd_handoff_v2.md` §12.5's recommendation for the paper
(`KD (gaze-overlay)` vs `KD (raw video)`).

```
qual/
  StreamGaze/figures/*.png  EgoGazeVQA/figures/*.png     the figures
  StreamGaze/layout/*.json  EgoGazeVQA/layout/*.json     geometry of each figure — re-render without a GPU
  scan/{sg,eg}.jsonl                          every scanned item: verdicts, margins, overlaps,
                                              and the selected token indices
  scan/{sg,eg}_r2.jsonl                       the independent second pass (stability check)
  manifest.csv                                figure -> item, checkpoints, metrics, frame strip
```

The `--source sg|eg` flag still spells the sources `sg` / `eg`; only these output directories
carry the full benchmark names.

Filenames encode the verdict triple: `sg_T1O1N0_idx372_<task>.png` = teacher correct, overlay
student correct, raw-video student wrong.

## Fidelity — this pass reproduces the handoff's numbers item-for-item

The full eval split was scored through this script's own forward path (SG 526/526, EG 485/485).
Both students land **exactly** on their published totals *and on every per-task column*, which
is the evidence that these figures draw the selection that produced those numbers rather than a
re-derivation of it.

| system | this pass | published | source |
|---|---|---|---|
| SG · KD (gaze-overlay) | **369** / 70.15% | 369 / 70.15% | §2.2a |
| SG · KD (raw video) | **360** / 68.44% | 360 / 68.44% | §7.7 |
| EG · KD (gaze-overlay) | **272** / 56.08% | 272 / 56.08% | §10.3 |
| EG · KD (raw video) | **268** / 55.26% | 268 / 55.26% | §7.7 |

Per-task, every column matches: SG overlay `49 48 1 21 88 70 42 50` (§2.2a) and SG raw-video
`45 46 1 19 86 72 43 48` (§7.7), in GSM / NFI / OTP / SR / OAR / OI-E / OI-H / FAP order;
EG overlay `66 68 138` and EG raw-video `70 59 139` in spatial / temporal / causal order
(§10.3, §7.7). §7.7's finding that removing the marker drops EG temporal 68 → 59, back to the
teacher's own level, reproduces here.

The **teacher** row is a single run and lands at SG 373 / EG 262. EG matches §10's 3-run mean
(262) exactly; SG's 373 sits inside §8's measured 372–377 spread. Per-column the teacher moves
by ±1–2 items against §8's mean, which is what §8 says a single run does — do not read those as
disagreements.

Two further invariants held over both full splits: `pct_kept` = 9.98% everywhere, and
content-pool IoU between the teacher and the overlay student = **1.000** on every item.

## Regenerating / re-rendering

```bash
cd $REPO && source env.sh

# full pass (~50 min per source; SG and EG fit one GPU each). Launch under tmux.
scripts/run_qual_kd.sh sg 0 --n-figures 12 --per-task-cap 2 &
scripts/run_qual_kd.sh eg 1 --n-figures 12 --per-task-cap 4 &
wait

# render specific items instead of ranking
scripts/run_qual_kd.sh sg 0 --render-idxs 31,372,470

# redraw every figure from the saved scan — no model, no GPU, seconds
python -m TrajGazeMerge.viz.qual.qual_kd_render --source sg --from-scan --n-figures 12
```

**This node evicts jobs.** Both of this figure set's first full runs were SIGKILLed two minutes
in, when another tenant's job landed on the machine — the same class of interruption §13.4
records three times. So `scan/{sg,eg}.jsonl` stores each item's *selected token indices*, not
just its metrics, and the launcher passes `--resume` and retries up to `$QUAL_RETRIES` (4)
times. A restart re-scans only the item that was in flight. `--from-scan` then rebuilds every
figure from that file on CPU, so re-ranking or restyling never needs the GPU again.

`GAZE_OVERLAY=1` with `VLM_GAZE_OVERLAY` **unset** is required — the launcher enforces it, and
the script asserts at the first item that the three frame streams resolve to different
directories (`rows1-2 VLM='viz' / row3 VLM='original' / teacher TAS='viz'`). This mirrors
§7.3's assertion: training or drawing both streams on one variant changes no shape, raises no
error, and does not show up in the numbers, so it has to be caught structurally.

Any figure can be re-rendered as a **vector PDF** from its layout JSON, on CPU, in seconds:

```bash
python3 TrajGazeMerge/viz/qual/render_qual_vector.py \
  TrajGazeMerge/qual/StreamGaze/layout/<name>.json \
  --font TrajGazeMerge/viz/qual/Inter.ttf --out out/<name>.pdf --type-scale 1.6
```

`--type-scale 1.6` is the value to use for a paper: the figure is ~28 in wide, so at
`\linewidth` everything shrinks 4× and the question lands at 9.2 pt against 10 pt body text.
That is a layout property, not a resolution one — raising the supersample does not fix it.

## Metrics in `manifest.csv` / `scan/*.jsonl`

- `iou_overlay_vs_teacher` / `iou_nooverlay_vs_teacher` — Jaccard overlap of the **3%
  complement** against the teacher's, in cell space `(t, row, col)`, i.e. the boxes the reader
  actually sees. Cell space rather than raw token indices because row 3's `avail` set is
  computed on different pixels, so index intersection would not be comparable between the two
  students.
- `iou_content_*_vs_teacher` — the same for the **content 7%**. This is **1.000 for the overlay
  student by construction**: content selection depends only on the frozen ViT's `attn_scores`,
  so it is adapter-independent, and both rows read the same pixels. It is a useful invariant —
  if it ever prints below 1.0 something is wrong with the preprocessing path.
- `follow = min(iou_overlay, iou_nooverlay)` — "both students track the teacher", the ranking
  signal. Split means over the full eval set: SG complement IoU **0.286** (overlay) / **0.232**
  (raw video), EG **0.170** / **0.098**. The 12 figures per source sit well above their split
  mean (SG 0.32–0.36, EG 0.16–0.28), which is what selecting on `follow` is for. **EG's
  students track the teacher about half as closely as SG's** — worth noting alongside §7.7,
  where EG *loses less accuracy* to marker removal than SG (4 items vs 9) while its selection
  diverges more.
- `agree_overlay_exact` — the *training-time* metric, exact token-index intersection over k
  (`selection_kd_loss`, `train_visionzip_kd_lora.py:236-239`). Measured ≈ **0.42 on SG**, which
  matches the ~0.41 the KD training logs report — the fidelity check that these figures draw
  the same selection that produced the handoff's numbers.
- `margin_*` — softmax gap between the top-2 option letters for that row.
- `pct_kept` — measured **9.98%**, matching the `pct_kept` in the training logs
  (9.977 / 9.988 / 9.991); the budget arithmetic is floor-based, so it never reaches exactly 10.

## Read these before using a figure as evidence

1. **Row 3's complement divergence is not purely the predictor's.** Removing the marker moves
   VisionZip's *own* content pick too: content-pool IoU vs the teacher is only ~0.17 (SG) /
   ~0.20 (EG). So part of the complement difference is inherited from a different `avail` set.
   Both numbers are printed on the row label for exactly this reason — quoting one alone reads
   as if the predictor diverged.

2. **EgoGazeVQA items are weak qualitative evidence.** The upstream figure toolkit audited 8 EG
   items against their own frames and **7 failed** — object on the wrong side of the fixation,
   the referenced event absent from the clip, one item with two character-identical options. It
   does not invalidate the EG accuracies (all rows face the same labels), but an EG figure
   illustrates *token selection*, not correct answering. Do not present an EG row's ✓/✗ as
   evidence the model understood the scene.

3. **StreamGaze's evidence often starts after the input ends.** 401 of 526 SG items (76%,
   median 2.1 s) have the fixation episode their answer was generated from beginning *after*
   the last of the 128 frames the model receives. The strip shows the run-up, not the grounding
   moment. `past_gaze_sequence_matching` and `past_scene_recall` are 100% affected.

4. **Eval is not deterministic.** §8 measures a 3–5 item spread when re-scoring identical
   weights, and option logits quantise to 1/8 in bf16 so ties are common. A row's ✓/✗ on one
   item is not a model property; low-`margin_*` items are the ones that flip. Every figure
   here was scanned twice in independent processes and checked with:

   ```bash
   scripts/run_qual_kd.sh sg 0 --render-idxs <the chosen idxs> --scan-only --tag r2
   python3 TrajGazeMerge/viz/qual/check_stability.py \
       TrajGazeMerge/qual/scan/sg.jsonl TrajGazeMerge/qual/scan/sg_r2.jsonl
   ```

   Run that before adding any new item — a margin threshold alone is not a stability test.
   All 24 figures here reproduced **exactly** (same three predictions, same `follow`) in a
   second independent process.

   Selection also passes a `--min-margin 0.05` gate, which dropped **66 of 485 EG** items and
   **14 of 526 SG** — EG is far more tie-prone, as 5 options rather than 4 would suggest. One
   EG item (idx 456) had both students at margin *exactly* 0.0000, a true two-way tie; it
   reproduced only because argmax breaks ties deterministically, so it was replaced rather
   than kept. Pass `--min-margin 0` to disable the gate.

5. **Contextual tokens are centroids.** Half the green boxes (3.5% of 7%) mark VisionZip
   *cluster centers* whose embedding averages ~90% of all tokens. The box location does not
   mean that region was kept verbatim.

6. **The rows differ in training input, not only in selector.** The teacher and the overlay
   student were trained on overlay frames; the raw-video student was not. And the teacher's
   own visual branch (Stage-1 TAS) stays overlay-trained in every configuration — §9 lists
   retraining it as out of budget.

7. **The blue `final fixation` chip is StreamGaze-only** and only on `present_*` / `proactive_*`
   tasks. StreamGaze cuts the frame list at the question timestamp so its last frame *is* the
   query moment; EgoGazeVQA has no cutoff, so the chip is suppressed there unconditionally.

## Provenance

The visual language, the layout-JSON schema and both offline renderers come from the
`qual_viz_share` bundle. Ported verbatim into `TrajGazeMerge/viz/qual/`:
`render_qual_vector.py`, `render_qual_pptx.py`, `layout_common.py`, `Inter.ttf`,
`pick_fixation_frames.py` (two path constants repointed). `BUNDLE_README.md` and
`BUNDLE_HANDOFF.md` are the bundle's own docs, kept for reference.

Local changes to the ported code, all noted in-place:

- **per-row `strip`** — the schema had one shared frame strip per figure; row 3 needs different
  image files (same moments, marker-free pixels). `layout_common.resolve_strip` and both
  renderers now honour an optional per-row `strip` and fall back to the shared one, so a layout
  written by the original pipeline still renders unchanged.
- **`render_qual_vector.wrap`** now breaks an over-wide single word. StreamGaze's options are
  set-notation strings with almost no spaces
  (`{lettuce,bag,hands}>{cucumber,knife,plate,lettuce}>{onion,mesh bag}`), so a space-only wrap
  left one unbreakable token running past the prediction column — visibly so at the
  `--type-scale 1.6` the bundle itself recommends.
- **footer wrapping** in both renderers; three rows made it longer than one line.
- **row-name auto-shrink** in both renderers; the bundle's names ("Ours", "VisionZip") were
  short enough that this never came up.
- `∪` is not in Inter.ttf and rendered as tofu, so the footer spells the budget with `+`.

The bundle's `pipeline/` is **not** ported: it is bound to the other machine's absolute paths
and to VisionZip / PruneVid / FastVID adapters that do not exist here, which is why the rows
are these three systems rather than the bundle's four baselines. `example/frames/` was not
copied either — those are EGTEA Gaze+ frames under that dataset's redistribution terms; these
figures read from `$SG_ROOT` directly.
