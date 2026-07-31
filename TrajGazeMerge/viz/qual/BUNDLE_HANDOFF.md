# Qualitative figures — handoff (2026-07-27, rev. 2)

Ours vs VisionZip vs PruneVid vs FastVID at the same 10% visual-token budget, one figure per
item: four rows, same frame strip, each row's kept tokens drawn as boxes plus that row's MCQ
prediction.

**What changed in rev. 2** (the rest of the file is updated to match):
1. The hand-picked strips were audited against the fixation episodes StreamGaze generates its
   answers from: they showed **7 of 25**, the misses landing a median 7 s (worst 52 s) away
   (§3). Strips are now anchored on those episodes by `pick_fixation_frames.py`, which still
   uses no answer information: 25 of 25, median 1 s.
2. The blue `final fixation` chip is now drawn **only on tasks that ask about the query
   moment** (`present_*` / `proactive_*`); on a past-tense question it pointed the reader at a
   frame the answer does not use (§4).
3. A benchmark-level property surfaced while auditing: the fixation episode an answer refers
   to starts **after** the causal cutoff in 401 of 526 sg test items (§5). It is a small gap
   (median 2.1 s) but it decides which items may be used as qualitative evidence.

## 1. Canonical set

**`scripts/viz_qual/pretty_sg_v2/` — 18 figures, StreamGaze only. This is the set to use.**

| verdict | items |
|---|---|
| `O1V0P0F0` only Ours correct | 11, 17, 39, 46, 195, 264, 401 |
| `O1V1P0F0` Ours + VisionZip correct | 9, 22, 30, 134, 172, 244 |
| `O1V0P1F0` Ours + PruneVid correct | 10, 18, 40, 41, 83 |

Tasks covered: `past_gaze_sequence_matching` (10), `present_future_action_prediction` (4),
`present_object_identification_easy` (1), `past_non_fixated_object_identification` (1),
`past_scene_recall` (1).

`pretty_sg_manual/` (18 figures) is the rev. 1 set and is **superseded**: hand-picked strips,
and a `final fixation` chip on every item. Keep it only for comparison.

**idx 9 is unstable.** Rev. 2 was rendered twice from identical inputs; the first run returned
`O0V1P0F0` (Ours wrong) and the second `O1V1P0F0`. This is the bf16 tie-flipping of §7, not a
rendering problem: predictions do not depend on which frames the strip shows, since all 128
frames go to the model either way. Treat idx 9 as unusable until the margin sweep clears it.

Before using any item in the paper, check it in §6: three more of the 18 are figure-unsafe.

**EgoGazeVQA is excluded** — see §5. The `pretty_eg*` directories are kept only as a record;
do not put them in the paper. `pretty_eg_manual/eg_O1V0P0F0_idx79_spatial.png` is additionally
stale (idx 79 no longer reproduces that verdict).

Earlier sg variants (`pretty_sg/` uniform, `pretty_sg_rel/` automatic relevance) predate the
gaze fix in §4 and the legend changes, so they are **outdated**; regenerate them if wanted.

## 2. How to regenerate

```bash
cd /workspace/trajgaze_st
# 1. strips (CPU, seconds) — regenerate only if the item list changes
/opt/conda/envs/trajgaze/bin/python scripts/viz_qual/pick_fixation_frames.py \
  --idxs 9,10,11,17,18,22,30,39,40,41,46,83,134,172,195,244,264,401 \
  --out scripts/viz_qual/frames_sg.json

# 2. figures (~15 min for 18 items on one GPU)
CK=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/trajgaze/bin/python -m scripts.viz_qual_pretty \
  --source sg --idxs 9,10,11,17,18,22,30,39,40,41,46,83,134,172,195,244,264,401 \
  --flags O1V0P0F0,O1V1P0F0,O1V0P1F0 --limit 99 \
  --frames-json scripts/viz_qual/frames_sg.json \
  --frames-note "{n} frames anchored on the clip's annotated fixation episodes; the answer was not used" \
  --font scripts/viz_qual/Inter.ttf --gpu 0 \
  --ours $CK/visionzip_complement_learned_overlay/best.pth \
  --vz   $CK/visionzip_lora_sgeg_overlay/best.pth \
  --pv   $CK/prunevid_sgeg_overlay/best.pth \
  --fv   $CK/fastvid_sgeg_overlay/best.pth \
  --out-dir scripts/viz_qual/pretty_sg_v2
```

`--frames-note` matters: without it the caption still claims the strips were chosen by hand.
Checkpoints above are the ones that reproduce the verdicts (17/18 on the rev. 2 run; the eg
counterparts are the `*_EGonly_overlay` variants). `--flags` accepts comma-separated patterns
and `?` wildcards; `--idxs` accepts `3,7`, `0-300` or `all`.

## 3. Frame selection — audited, then rebuilt

### What was wrong with the hand-picked strips

StreamGaze does not generate its questions from the whole clip. Every item comes from a few
**fixation episodes** listed in `metadata/egtea.csv` (3-6 s each, `start/end_time_seconds`,
the object gazed at, the objects near it), and the options are permutations of those same
objects. `audit/audit_evidence.py` matches each item's answer back to its episodes and asks
whether the strip shows them. Against the rev. 1 hand-picked `frames_sg.json`:

| over the 18 items, 25 visible evidence episodes | rev. 1 hand-picked | rev. 2 anchored |
|---|---|---|
| episodes the strip actually shows | 7 | **25** |
| offset from the episode, median / worst | 7 s / 52 s | **1 s / 4 s** |

("shows" = a strip frame within half a display step or half the episode of it; reproduce with
`audit_evidence.py --frames-json scripts/viz_qual/frames_sg_handpicked_v1.json`.)

The old per-task rules ("the moments where gaze lands on each object named in the options")
describe what was intended, not what the numbers say happened: e.g. idx 46 seg1 `{box,kettle}`
is at 7.7-10.8 s and the nearest strip frame was t16 = 61 s; idx 83's only visible gazed
distractor (`poster`, 8.6-14.9 s) was missing entirely from a strip that starts at 48 s.

### The rule now

`scripts/viz_qual/pick_fixation_frames.py` builds the strips:

- **anchors** = the displayable frame nearest each fixation episode inside the causal window,
  plus the window end;
- **filler** = a uniform grid over the rest, each filler slid up to ±2 groups to the sharpest
  neighbour (variance of Laplacian) so it does not land mid-saccade on a smear of the ceiling.
  Anchors never move. Compare idx 264: two blurred filler frames became a pan and a pot;
- more episodes than cells → the most recent win, since a question is built from the episodes
  that end at its timestamp.

**This uses no answer information.** The episode list is the same for every option — the GT
picks none of the frames, and a reader with only the question and options could place the same
strip. Caption: *"N frames anchored on the clip's annotated fixation episodes; the answer was
not used."*

The reason it beats picking by hand is not care, it is that the moments are 3-6 s long inside
clips of up to 10 min, and the display grid is one frame per ~2-14 s, so eyeballing a contact
sheet lands next to them far more often than on them.

### 3b. Choosing a strip by hand

`contact_sheet_ours.py` draws **every one of the 128 input frames** with Ours' kept tokens on
it, labelled `t28 / t28b` plus wall-clock time, so a strip can be revised by reading indices
off the sheet. Marks: blue = already in the strip, yellow = an annotated fixation episode,
red = the last frame. The header states where the clip is cut and, when the episode the
answer refers to starts after the cut, how many seconds late it is.

Two traps this sheet was fixed for, both worth knowing:

- **A group is two frames.** Qwen merges (2t, 2t+1) into group t, and the renderer used to
  show only the first. Both carry the same token boxes, so both are legal strip frames, and
  over these 25 items **6 of 32 evidence episodes fall on a second frame only** (worst case:
  2.4 s away from any first frame, 0.2 s once both are allowed). Strip entries therefore
  accept `28` or `28b` (`parse_group` in the renderer).
- **Episode labels can leak the answer.** For `present_object_identification_*` and
  `present_object_attribute_recognition` the annotated object at the query moment *is* the
  answer, so those sheets show episode positions without names. For the other tasks the names
  are already spread across all four options, so printing them gives nothing away.

Contact sheet for a new item (question + options, never the answer):

```bash
GAZE_OVERLAY=1 python scripts/make_contact_sheet.py sg 401 /tmp/sheets --cols 8 --cell 220
```

Note its docstring is stale: it still maps `t → round(t/(T-1)*(L-1))` while the renderer now
uses `vi = 2t`. The two differ by at most one extracted frame (0.1 s), so sheets stay usable.

Automatic alternatives exist in the script (`--frame-select uniform | relevance | relevance-seg
| probe | probe-seg`) but were **not** used for the final set:
- `relevance` (cosine of visual tokens against question **content-word** embeddings, stopwords
  and benchmark boilerplate dropped) separates frames only weakly, ~0.035–0.062 per item.
- `probe` (single-frame answer probe of the frozen base) has a good dynamic range but is
  *negatively* correlated with which frames actually carry the answer (Spearman −0.15…−0.81);
  only the GT-conditioned variant (`--probe-score gt`, an oracle) finds them, so it must not
  be used for a figure.
- Both were measured, not assumed — see `audit/` and the numbers above.

## 4. Rendering decisions

1. **Gaze marker bug (important).** The ring was drawn from `_pool_to_T(gaze, T)[t]`
   (128→64 interpolation) while the displayed image was the raw frame, so on saccades the
   ring landed *between* two fixations: **47% of shown frames were off by >5% of frame width,
   worst 0.52**. Now the frame is the first of the merged pair (`vi = 2t`) and the ring uses
   that frame's raw gaze, which coincides exactly with the green gaze dot baked into the
   frames the model sees (verified: median |traj − dot| = 0.0001). **Every figure produced
   before this fix is affected**, including the pre-session ones.
2. Legend: gaze swatch is a **ring** (15px, same size as the cells, 2.5px stroke), not a
   filled square; labels are 17px Medium in ink black; swatches and labels share one mid-line.
3. **The blue `t… · final fixation` chip is now conditional** (rev. 2). The window is cut at
   the question timestamp (`dataset.py:71`, `cutoff = ts_sec * 10fps`), so the last frame is
   always the query moment — but only `present_*` / `proactive_*` questions ask about it.
   On the 13 past-tense items the chip was telling the reader to look at a frame the answer
   does not use, so those now get a plain `t63` chip like any other frame.
   `asks_about_query_moment(item, source)` in `viz_qual_pretty.py` decides: **source first**
   (only sg has a cutoff at all, see §5), then task name, then question wording
   (`"currently"`, `"do next"`, …) for sources whose task field is only a coarse qa_type.

## 5. Benchmark audit — why EgoGazeVQA was dropped

All 8 eg items in the figure set were checked against their frames. **7 of 8 fail.**

| idx | GT claims | frames show | verdict |
|---|---|---|---|
| 9 | countertop, in front + slightly right | matches | **usable** |
| 79 | tomato **on top of** cheese, right of fixation | tomato **beside** cheese, **left** of fixation | contradicted |
| 120 | milk **on the countertop**, right of fixation | milk **inside the fridge**, **left** of fixation | contradicted |
| 481 | left side of **the drawer** | items come out of a **paper grocery bag**, no drawer | contradicted |
| 210 | tomato put in the fridge | no fridge and no tomato in any of 64 sampled frames | event absent |
| 405 | bowl in right sink basin, beside sponge | no bowl / sponge / sink close-up | event absent |
| 419 | after taking bread container from fridge | that event does not occur in the clip | event absent |
| 317 | option C | **options B and C are character-identical** (verified in code) | item defect |

This does **not** invalidate the eg numbers: all four methods face the same labels, so the
relative comparison stands. It does mean eg absolute accuracy measures agreement with
auto-generated labels rather than scene understanding, and that eg items must not be used as
qualitative evidence. Consider one limitation sentence in the paper.

### eg rendering: what was actually wrong (checked in rev. 2)

| check | result |
|---|---|
| `final fixation` chip | **wrong on all 8 figures.** Verified by pixel-inspecting `pretty_eg_manual/` |
| gaze marker vs `traj.gaze_pos` | fine: median offset 0.0014-0.0016 over idx 9, 79, 120, 317, 481 |
| strips vs per-item T | fine: all 8 strips stay inside their own T |

The chip is the real defect and it is a benchmark difference, not a bug in one figure.
StreamGaze cuts the frame list at the question timestamp, so its last frame is the query
moment. **EgoGazeVQA has no cutoff**: `egogaze_dataset.py` takes every frame of the subclip
(`_frame_paths` globs `{subclip}_*.jpg`), so the last frame is only where the subclip ends and
no option is anchored to it. `asks_about_query_moment(item, source)` now returns False for any
non-sg source, so eg can no longer get the chip regardless of question wording.

Two eg properties worth knowing before rendering eg again:
- clips are **short and variable**, 73-628 raw frames, so `_sample_paths` often returns fewer
  than 128 and T lands anywhere in 37-64. A strip copied from an sg item would be silently
  truncated (`tsel = [t for t in manual[idx] if t < T]`).
- gaze is valid on **100%** of frames (sg has gaps), and the marker is baked in by the
  benchmark itself rather than by our pipeline.

**sg is qualitatively different**: every object named in an sg GT actually appears in the clip
and gaze sits on it (EGTEA gaze+ human annotation). No sg item was falsified. But sg has a
structural problem of its own, found in rev. 2 and quantified below.

### The answer's fixation episode starts after the causal cutoff

`audit/audit_evidence.py --all` matched every sg egtea test item back to the fixation episodes
its answer was generated from, and compared them with the last of the 128 frames the model
actually receives:

| task | items | evidence starts after the input ends |
|---|---|---|
| `past_gaze_sequence_matching` | 64 | **64 (100%)** |
| `past_scene_recall` | 37 | **37 (100%)** |
| `present_object_identification_easy` | 101 | 85 (84%) |
| `present_object_attribute_recognition` | 96 | 81 (84%) |
| `past_non_fixated_object_identification` | 68 | 57 (84%) |
| `present_object_identification_hard` | 64 | 49 (77%) |
| `present_future_action_prediction` | 94 | 28 (30%) |
| all | 526 | **401 (76%)** |

Two causes compound. StreamGaze puts the question timestamp at the *onset* of the episode the
answer refers to, and `_sample_paths` (`dataset.py:85`) steps by `int(i*L/128)`, so the last
sampled frame sits ~0.8% short of the cutoff. The gap is small — median **2.1 s**, p90 4.9 s,
max 7.0 s — so the model sees the run-up to the fixation but never the fixation itself.

What this does and does not mean:
- It does **not** invalidate the numbers. All four methods get the same frames, and the visible
  part still determines the answer on most items (checked per item: on all 11 sequence items
  the first two groups already single out the GT option).
- `present_object_identification_*` is therefore not "read the current gaze target off the last
  frame" but "predict the next fixation target" — worth one sentence if the paper describes
  what the task measures.
- For a **figure** it is the binding constraint: the strip cannot show the moment that grounds
  the answer, only what came 1-7 s earlier. Whether that still supports the answer is item by
  item — see §6.

## 6. Which items are safe to put in the paper

Run `audit/audit_evidence.py --idxs …` for any candidate item, then look at the frames.
For the current 17:

| items | status |
|---|---|
| 10, 11, 17, 18, 22, 30, 39, 40, 41, 46 (`past_gaze_sequence_matching`) | **usable.** Groups 1-2 are shown on the annotated episode (0-4 s off) and already identify the GT option. Group 3 always falls after the cutoff — do not claim the figure shows the full sequence. |
| 172, 264 (`present_future_action_prediction`) | **usable.** The recent-fixation episode is inside the input and is the last frame. |
| 195, 244 (`present_future_action_prediction`) | **weak.** The named fixation pattern starts 4.1 s / 3.3 s after the input ends; the strip shows the activity leading up to it, not the pattern. Not contradicted, not verifiable. |
| 83 (`past_non_fixated`) | **do not use as evidence.** Two of the three gazed distractors (`spatula`, `patty`) are outside the input, so 3 of 4 options remain consistent with what the clip shows. `glass` cannot be ruled in from the strip. |
| 134 (`past_scene_recall`) | **do not use.** The `{tomato,knife,plate}` fixation is at 142.8-155.2 s and the input ends at 140.9 s, so the referenced moment is absent — and the strip shows a **stove** in most frames while the GT says the stove was not visible. The figure reads as contradicting its own answer. |
| 401 (`present_object_identification_easy`) | **do not use.** The `paper` fixation starts at 413.9 s, the input ends at 410.8 s and the last displayed frame at 407.6 s shows gaze on a **pot**. The only frame with gaze on the paper is t13 (84 s), five minutes earlier. A reader sees a pot and an answer that says paper. |

Dropping 134 and 401 costs one `O1V1P0F0` and one `O1V0P0F0`; 83 costs one `O1V0P1F0`. If
replacements are wanted, the cheap route is `audit_evidence.py --all` (CPU, ~3 min): 125 of the
526 sg items have all their evidence inside the input, and the GPU verdict sweep then only has
to run on those.

## 6b. The full candidate landscape (all 526 sg items scanned)

`scan_candidates.py` swept every sg test item (verdict + top-2 margin per method, no
rendering, ~25 min split over 3 GPUs; raw results in the session scratchpad):

| | items |
|---|---|
| Ours correct | 362 of 526 |
| ... and at least one baseline wrong = **figure candidates** | **68** |
| ... passing stability (min margin ≥ 0.05 and Ours ≥ 0.25) | 44 |
| ... of those, not already in the set | 26 |
| ... of those, evidence-clean (tier A) | **6** |

New candidates worth rendering, tiered the way §6 judges the current set (all 7 tier-A/B
leaders are rendered in `pretty_sg_candidates/`):

| tier | idx | verdict | Ours margin | min margin | task |
|---|---|---|---|---|---|
| **A** clean | 205 | `O1V1P0F1` | 0.841 | 0.647 | future_action |
| **A** | 215 | `O1V1P0F1` | 0.620 | 0.319 | future_action |
| **A** | 198 | `O1V0P1F0` | 0.605 | 0.226 | future_action — best of the set, the last frame shows the egg in the bowl and the GT is "mix the egg" |
| **A** | 262 | `O1V0P1F1` | 0.713 | 0.117 | future_action |
| **A** | 241 | `O1V0P1F0` | 0.783 | 0.103 | future_action |
| **A** | 70 | `O1V1P0F1` | 0.800 | 0.085 | non_fixated |
| **B** seq, 2 of 3 groups visible (same limitation as the 10 in use) | 58, 59, 36, 7 | `O1V1P0F1` ×3, `O1V0P1F1` | 0.46-0.87 | 0.14-0.59 | sequence_matching |
| C | 16 more | — | — | — | key evidence outside the input; do not use |

Two things this rules out:

- **There is no untapped "only Ours correct" item.** `O1V0P0F0` exists 7 times in the whole
  test split; the set already uses 6 of them, and the 7th (idx 56) sits at margin 0.051.
  Every new candidate is a weaker pattern (`O1V1P0F1` = only PruneVid wrong, etc.).
- **Nothing can replace a sequence-matching figure with a cleaner one.** All 22 sequence
  contrast items carry the seg-3 limitation; tier B is not better than what is in use, only
  different.

Also rendered but **not** recommended: 297 (`present_object_attribute_recognition`, "what
colour is the object" — the whole scene is dark and the answer `black` cannot be read off the
strip) and 390 (`O1V0P1F1`, weak contrast). Margin measures verdict stability, not whether a
reader can see the evidence; both gates have to pass.

## 7. Verdict instability — check before adding any new item

Option logits are quantised to 1/8 in bf16, so exact ties are common, and the Ours path adds
its own run-to-run variance: repeating idx 9 in five separate processes gave Ours margins of
0.040 / 0.083 / 0.129 / 0.129 / 0.174 while VisionZip, PruneVid and FastVID returned bit-identical
margins every time. The variance is in the selection, not the logits — the traj encoder's
scores decide a topk, and near-ties there swap tokens in the 3% complement.

**Consequence: a margin threshold alone is not a stability test.** idx 9 flipped at margin
0.129 and idx 401 at 0.124-0.244. Gate a new item on *both*: min margin over the four methods
≥ 0.05, Ours margin ≥ 0.25, **and** the same verdict from two independent `scan_candidates.py`
processes.

Items found unstable or false so far:

| item | issue |
|---|---|
| sg 121 | Ours p(A)=p(C)=0.451, exact tie; **deleted** |
| eg 327 | Ours A/D tie; **deleted** |
| sg 430 | all four methods wrong (GT cable, all answered bread) — filename claim false |
| sg 13 | flips O1↔O0 between runs |
| sg 9 | `O0V1P0F0` in 1 of 6 observed runs (Ours margin 0.04-0.17 across processes) |
| sg 401 | `O0V0P0F0` in 1 of 4 observed runs (Ours margin 0.124 / 0.244); already unusable per §6 |
| eg 79 | flipped to `O0V0P0F0` after the gaze fix |

Measured over the current 18 (three independent scans, plus the two renders): **16 of 18
reproduced every time**; 9 and 401 did not. The four with Ours margin below 0.25 are
9 (0.13), 39 (0.04-0.07), 264 (0.11) and 401 (0.12-0.24) — the same four, so the gate above
separates them correctly, but only 9 and 401 have actually been caught flipping.

Backups of the deleted files were in the session scratchpad (`removed_tied/`), which is
session-scoped and by now gone. Re-render from the script if they are ever needed.

**Recommended next step: a margin sweep.** For each item, print Ours' top-2 option margin and
drop anything below ~0.5. `audit/probe_two_items.py` already does this for a single item (it
prints the full option distribution plus alternative checkpoints); generalise it over the
list. ~15 min. Worth combining with the §6 replacement search in one GPU pass.

## 8. Files

| path | what |
|---|---|
| `scripts/viz_qual_pretty.py` | the renderer (all options documented in its docstring) |
| `scripts/viz_qual/pick_fixation_frames.py` | builds the strips from the fixation episodes (rev. 2, answer-free) |
| `scripts/viz_qual/scan_candidates.py` | verdict + per-method top-2 margin for any item list, no rendering (§6b, §7) |
| `scripts/viz_qual/contact_sheet_ours.py` | every input frame of a clip with Ours' selection drawn, for choosing a strip by hand (§3b) |
| `scripts/viz_qual/sheets_ours/` | those sheets, sg: the 18 + the 7 candidates (answers shown) |
| `scripts/viz_qual/sheets_ours_eg/` | the same for the 8 eg items |
| `scripts/viz_qual/render_qual_vector.py` | redraws a figure as a **vector PDF** from a layout dump (§10) |
| `scripts/viz_qual/render_qual_pptx.py` | redraws the same layout dump as an **editable .pptx** (§10b) |
| `scripts/viz_qual/layout_common.py` | shared by both offline renderers: where the frames are (`--frames-root`), and the eyebrow / legend / footer defaults a layout can override |
| `scripts/viz_qual/layout/` | those layout dumps (`--dump-layout`) |
| `qual_viz_share/`, `qual_viz_share.tar.gz` | the bundle handed to another machine (§12) |
| `scripts/viz_qual/pretty_sg_custom{,_hires}/` | idx 83 with a hand-picked 6-frame strip, 2020px / 6060px |
| `scripts/viz_qual/frames_sg_custom.json` | that strip (`["11b", 29, 42, "52b", "54b", 62]`) |
| `scripts/viz_qual/QUAL_FIGURE_sg83.md` | its caption, subsection text, and the measurements behind them |
| `scripts/make_contact_sheet.py` | contact sheet of every frame, question + options, **no answer** |
| `scripts/viz_qual/frames_sg.json` | the strips in use (generated) |
| `scripts/viz_qual/frames_sg_handpicked_v1.json` | the rev. 1 hand-picked strips, kept for comparison |
| `scripts/viz_qual/frames_eg.json` | eg strips, kept for the record only |
| `scripts/viz_qual/pretty_sg_v2/` | **the 18 current figures** |
| `scripts/viz_qual/pretty_sg_candidates/` | 7 new candidates from §6b, same renderer and rule |
| `scripts/viz_qual/frames_sg_candidates.json` | their strips |
| `scripts/viz_qual/pretty_sg_manual/` | rev. 1 figures, superseded |
| `scripts/viz_qual/audit/` | audit + probe scripts (see below) |
| `scripts/viz_qual/Inter.ttf`, `design.md` | type and visual language |

`audit/`:
- `audit_evidence.py --idxs … | --all` — matches each answer back to the fixation episodes it
  was generated from; flags evidence outside the model's input, strips that miss visible
  evidence, and options left under-determined. CPU only. **The gate to run on any new item.**
- `audit_sg.py <idxs>` / `audit_eg.py` — print Q + options + **GT** and dump a 12-frame strip
  with the gaze crosshair, for checking whether a GT is verifiable.
- `check_gaze_align.py` — traj gaze vs the gaze dot baked into the frames (should be ~0.0001).
- `check_pool_offset.py` — how far the drawn ring is from the true gaze of the shown frame
  (was 47% >0.05 before the fix; should now be 0).
- `probe_two_items.py` — per-method option distribution and top-2 margin for one item, plus
  alternative Ours checkpoints.
- `probe_stability.py` — repeat one item N times in-process to test verdict stability.
- `fix_gaze_legend.py` — repaints the legend row of already-rendered PNGs (no GPU). Only
  needed for old figures; the renderer now draws it correctly.

## 9. Open items

1. **Decide the final item list.** Evidence (§6) and stability (§7) both argue for dropping
   401 (fails both), 134 and 83 (evidence), and 9 (stability); 39 and 264 sit under the Ours
   margin gate without having been caught flipping. Replacements exist only as the tier-A/B
   items in §6b, all rendered in `pretty_sg_candidates/` — 198 is the one to look at first.
   Note that dropping 401 and 9 leaves the strongest pattern `O1V0P0F0` with 11, 17, 39, 46,
   195, 264, and 195/264 have their own caveats, so the "only Ours correct" story rests on
   11, 17, 46.
2. ~~Margin sweep~~ — done (§7); `scan_candidates.py` is the reusable gate.
3. Time labels instead of `t0…t63`. StreamGaze windows grow with the question timestamp
   (a 30 s history and a 10 min history both become 128 frames), so the index axis hides the
   streaming structure. Frame files are `frame_{N}.jpg` at 10 fps, so labels like
   `−4:12 … −0:08 … final fixation` are exactly computable. Would make the causal-prefix
   setting visible in the figure. §5 makes this more valuable: the gap between the last frame
   and the queried moment is exactly the thing a time axis would show.
4. Decide whether the paper says anything about §5. One sentence covers it ("the annotated
   moment a question refers to begins at the causal cutoff, so the model predicts rather than
   observes it"), and it also explains why absolute sg accuracy is what it is.
5. Decide what to do with `pretty_eg*` and the outdated `pretty_sg`, `pretty_sg_rel`,
   `pretty_sg_manual` directories (delete, or regenerate with the current renderer).
6. `viz_qual_compare.py` shares the old gaze-pooling code path; if it is still used anywhere,
   apply the same fix (`vi = 2t` + raw gaze).

## 10. Output format: raster vs vector (added 2026-07-28)

The PNG path bakes text and shapes into pixels. Two knobs and one alternative:

- `--supersample 3 --out-scale 3` keeps every drawn pixel instead of supersampling and
  throwing half away. A frame cell lands at 630 px, the native resolution of the 640 px EGTEA
  source, so this is the raster ceiling: 6060x3390 for a 6-frame figure.
- `--dump-layout DIR` writes the figure's geometry, and `render_qual_vector.py` redraws it as
  a **PDF whose text, boxes, rings, cards and chips are vector** (Inter embedded as FontFile2,
  verified). Only the photographs stay raster. No GPU needed, seconds per figure.
- `--type-scale` on the vector renderer multiplies font sizes only; the layout re-flows.

**The type is too small for a paper at the default scale, and this is a layout problem, not a
resolution one.** The figure is 28.1 in wide, so at `\linewidth` = 7 in everything shrinks 4x:

| element | design | as placed, ts=1.0 | ts=1.6 | ts=2.0 |
|---|---|---|---|---|
| question | 23 | **5.7 pt** | 9.2 pt | 11.5 pt |
| options | 16 | **4.0 pt** | 6.4 pt | 8.0 pt |
| prediction | 15 | 3.7 pt | 6.0 pt | 7.5 pt |
| footer, frame chip | 12 / 11 | 3.0 / 2.7 pt | 4.8 / 4.4 pt | 6.0 / 5.5 pt |

`--type-scale 1.6` is the recommendation: question 9.2 pt against 10 pt body text. At 2.0 the
question outgrows the body text. Rendering at a higher supersample does **not** fix this.

## 10b. Editing the figure by hand: the .pptx path (added 2026-07-28)

`render_qual_pptx.py` reads the same layout dump and writes a PowerPoint slide in which every
element except the photographs is a native object: the cards, the 609 token boxes, the gaze
rings, the chips and every string. Text is real text, so the question can be reworded, a row
moved or a colour changed without going back to the GPU.

```bash
/opt/conda/bin/python scripts/viz_qual/render_qual_pptx.py \
  scripts/viz_qual/layout/sg_O1V0P1F0_idx83_past_non_fixated_object_identification.json \
  --font scripts/viz_qual/Inter.ttf \
  --out scripts/viz_qual/pretty_sg_custom/fig_qual_sg83.pptx
```

Use the base env (`/opt/conda/bin/python`); `trajgaze` has no python-pptx. Geometry was
checked against `pretty_sg_custom/*.png` by drawing the saved shape tree back out with PIL:
frames, token boxes and gaze rings land on the same pixels.

- Slide = 2020 x 1129 pt = 28.1 x 15.7 in, one design unit per point, the same page as the PDF.
  `--type-scale 1.6` applies the §10 recommendation and re-flows the layout with it.
- The token boxes of one frame are grouped with that frame, its gaze ring and its chip, giving
  24 groups; double-click to get inside one, or pass `--no-group` for 700+ loose shapes.
- Shapes carry names (`option-B`, `card-Ours`, `pred-VisionZip`, `token`, `chip`), which the
  PowerPoint selection pane lists, so a specific element can be found without hunting.
- Type is Inter. A machine without Inter installed substitutes a default sans and the line
  breaks move; `--font-name` picks another family. Line breaking inside a box is PowerPoint's,
  not PIL's, so a reworded question rewraps on its own.
- Two things the .pptx cannot do: it does not re-run the model, so an edited prediction is
  just text, and it does not regenerate the layout. Anything structural still goes through
  `viz_qual_pretty.py --dump-layout`.

Also corrected this session: the footer used to read "only the token-selection rule differs",
which is false. The backbone is frozen and shared, but every row loads its own LoRA adapter
(224 tensors, `best.pth` chosen on validation accuracy) trained under its own selector, and
Ours additionally uses a trained trajectory encoder (`stage1_tas_3way_overlay`). Any text
describing the comparison has to say the same.

## 11. Where this session stopped

Done: fixation-anchored strips (§3), chip rule (§4, source-aware after the eg finding in §5),
evidence audit (§5-6), full 526-item candidate sweep and margin gate (§6b-7), contact sheets
for 25 sg + 8 eg items, white page background, the vector PDF path, and one hand-picked figure
for idx 83 with its caption and text in `QUAL_FIGURE_sg83.md`.

Waiting on a decision:

1. **Final item list.** Drop 401 and 9 (fail evidence and stability), 134 and 83 (evidence)?
   That leaves the "only Ours correct" story resting on 11, 17, 46. Replacements are the
   tier-A items in §6b, rendered in `pretty_sg_candidates/`; **198** is the strongest.
2. **`--type-scale`** for the final figures: 1.6 unless the figure is placed narrower.
3. Whether `pretty_sg_v2/` gets re-rendered with the white background, the vector path and the
   chosen type scale, or the paper takes only the hand-picked customs.
4. The user was picking frames off the contact sheets when the session ended; only idx 83 has
   a hand-picked strip so far.

## 12. Handing the code to another machine (added 2026-07-28)

`qual_viz_share/` (and `qual_viz_share.tar.gz`, 5.2 MB) is the bundle, with `README.md` in
Korean as its entry point. It splits the code the way the dependencies do:

- **`render/` runs anywhere.** A layout dump plus pillow, and matplotlib or python-pptx. No
  GPU, no checkpoints, no `TrajGazeMerge`. This is the part another project can actually
  reuse: emit the §3 JSON from their own model and they get this figure.
- **`pipeline/` is ours.** It carries `sys.path.insert("/workspace/...")` and checkpoint,
  dataset and font constants; the README lists every line that has to be repointed.

Three changes were needed to make the offline renderers portable, all backwards compatible
(both outputs verified pixel-identical on idx 83 afterwards):

1. `--frames-root`, plus resolution of a dump's frame paths relative to the JSON itself, so a
   layout copied to another machine finds its frames (`layout_common.py`).
2. Optional `title` / `legend` / `footer` keys. Without them the figure prints *our* chrome
   ("StreamGaze", "content-based selection 7%", "frozen Qwen2.5-VL-7B backbone") on someone
   else's data, which is the first thing that goes wrong on a port.
3. Both renderers create the output directory instead of failing on it.

`example/minimal_layout.json` is the portability test: a hand-written 2-row, 2-frame layout on
a 6x8 grid with no model involved. If it renders, the port is done.

Note the bundle ships 6 EGTEA Gaze+ frames as example data. Check the EGTEA license before
sending it outside the group, or drop `example/frames/` (only the example stops working).
