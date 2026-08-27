# ViT-KD contact sheets — SG teacher vs ViT-distilled (raw video)

Qualitative material for `docs/kd_handoff_v3.md` setting 1. One sheet per **method** per item:
every temporal group of the clip, with that method's kept visual tokens drawn on each frame,
labelled with the group index and its wall-clock time.

Sibling set: `../distill_v1/` holds the v2 KD-student figures (M1 teacher / KD gaze-overlay /
KD raw video, 3-row comparison figures rather than contact sheets).

## What is here

**50 items, 100 sheets**, selected from **275 of the 526** SG eval items — a partial scan, stopped
once enough candidates had passed the gates (see *How items are chosen*). Measured over the 50:

| | min | mean | max |
|---|---|---|---|
| `recall_traj` | 0.336 | **0.439** | 0.568 |
| `conc_teacher` | 0.106 | 0.189 | 0.242 |
| `conc_student` | 0.093 | 0.148 | 0.212 |
| `iou_kept` | 0.290 | 0.311 | 0.340 |
| min option margin | 0.283 | 0.698 | — |

The selected `recall_traj` mean of 0.439 sits above the adapter's whole-split 0.383, which is what
selecting on agreement is for. Task spread: NFI 13, OAR 11, GSM 8, OI-E/OI-H 8, FAP 8, SR 2, over
50 distinct input windows drawn from 14 clips.

**Stability: all 50 reproduced exactly** — same predictions and same `recall_traj` — when
re-scanned in an independent process (`scan/sg_r2.jsonl`).

One item to know about: **idx497's clip is only 2 seconds long** (its question timestamp is at
2.0 s, so T=10 rather than the usual 64). The budget arithmetic is still correct (215 of 2160
tokens, 9.95%) and the sheet is valid, but it is a thin exhibit — one short row of cells instead of
sixteen.

```
distill_v2/
  M1_teacher/sheets/sg_idx<N>_<task>.png         the SG specialist teacher, gaze at test
  ViT-KD_raw_video/sheets/sg_idx<N>_<task>.png   the distilled ViT, 0 extra params, epoch 1
  scan/sg.jsonl        every scanned item: verdicts, margins, recalls, concentration, selections
  manifest.csv         item -> both sheets, checkpoints, metrics
```

Both sheets of an item share a filename stem, so they sort next to each other.

## What the two methods are

| | M1 teacher | ViT-KD (raw video) |
|---|---|---|
| selection | 7% VisionZip content ∪ 3% gaze/hand complement (frozen TAS encoder) | pure VisionZip, 6.5% dominant + 3.5% contextual |
| ViT | frozen | rank-8 LoRA on `visual.blocks[31].attn.{qkv,proj}`, 61,440 params |
| gaze at test | **yes** | no |
| extra params at inference | 36.85 M | **0** |
| frames | `viz` — the gaze marker is in the pixels | `original` — raw video, no marker |
| checkpoint | `visionzip_complement_learned_SGonly_overlay/best.pth` | `vitkd_p1_sg_raw/best.pth` (ViT) + `vitkd_p2_sg_raw/epoch_01.pth` (LLM) |

**Epoch 1, not best-of-2.** v3 §5.4: *"Report epoch 1 (366), or report both. Do not report only
the best-of-2."* Epoch 2 gains 1 item on Avg — inside the ±4 noise floor — while collapsing GSM
from 45 to 39 items. The sheets use epoch 1.

## Reading the colours

| | teacher sheet | ViT-KD sheet |
|---|---|---|
| green outline | content 7% | the kept 10%; **thick = dominant 6.5%, thin = contextual 3.5%** |
| magenta fill | the gaze/hand complement 3% | the dominant tokens that **coincide with the teacher's 3%** |
| yellow ring | the annotated gaze point | — (see below) |
| yellow cell border | the cell sits on an annotated fixation episode | same |
| red cell border | the last frame of the clip | same |

The student's magenta is **`recall_traj` drawn in place** — v3 §5.2's metric, which distillation
moved from 0.042 to 0.383. So the magenta on the ViT-KD sheet answers, per frame, *where* the
recovered 38% of the teacher's gaze complement actually landed.

The ViT-KD sheet has **no gaze ring**: that model sees neither the pixel marker nor any gaze
input, so a ring would assert an input it does not have. Its numeric gaze concentration is in the
subtitle instead, and `--student-gaze-ring` turns the ring on for eyeballing.

## Metrics

- **`recall_traj`** — the repo's own `selection_metrics` (`train_vit_selection_kd.py:214`):
  the student's **dominant top-6.5%** ∩ the teacher's gaze complement, over `|traj|`. Measured
  against the dominant set only, because the contextual centres are index-spaced and crediting
  them would be noise. `recall_P` / `recall_S` are recorded alongside.
- **`iou_kept`** — cell-space Jaccard of the two kept-10% sets.
- **`conc_teacher` / `conc_student`** — the fraction of that method's kept cells whose centre
  lies within `--gaze-radius` (default 0.15 of frame width) of the annotated gaze point, averaged
  over frames with a valid gaze sample. This is the "tokens on the key object" proxy: EGTEA's gaze
  is a human annotation of what the person was actually looking at.

**One subtlety in `recall_traj`.** The teacher appears twice, for two different purposes:
- the **teacher sheet** draws M1 *as deployed* — frozen ViT on `viz` frames, which is the system
  in v3 §5.5's table row 1;
- `recall_traj` is measured against the **distillation target**, which Phase 1 computed with the
  frozen ViT on the student's own `original` frames (`vitkd_sg_raw_p1.log`:
  `student VLM='original' teacher TAS='viz'`).

Those are different token sets, because VisionZip's content pick depends on the pixels. The script
computes both — three encoder passes per item — so the sheet shows the real teacher while the
metric keeps the doc's definition. Do not compare `recall_traj` against the teacher sheet's boxes
directly.

## How items are chosen

Three gates, then a score:

1. **both rows correct** — the teacher and the ViT-KD student must each answer the item.
2. **min top-2 option margin ≥ 0.25.** The floor for *reproducibility* is 0.05 (bf16 logits
   quantise to 1/8, so exact ties happen), but a "correct" at margin 0.06 is a coin flip that
   landed — the first sample surfaced exactly such an item. Raising it to 0.25 cost 4 of 34
   candidates.
3. **evidence gate, 3 s.** On `present_*` / `proactive_*` tasks the question asks about the query
   moment, and SG cuts the clip there — but the answer's fixation episode often starts *after* the
   last input frame, so the moment is not in the clip at all (v2 §6; 76% of SG items overall).
   Items whose next annotated fixation begins within 3 s of the window end are dropped. At the
   stricter 8 s these tasks were nearly eliminated (FAP 1, OAR 1, OI-E 0, OI-H 0 of 58); 3 s keeps
   the task balance while removing the worst cases, and every sheet still prints its own gap in red.

Survivors are scored by `0.5 · pct(recall_traj) + 0.5 · pct(conc_student)` — percentiles within the
scanned set, because agreement and concentration are on different scales. Then:

- **at most 1 item per (clip, input window)**, not per clip. The split has **35 clips but 251
  distinct windows**: items sharing a clip *and* a question timestamp read the same 128 frames and
  produce byte-identical sheets (idx 31/32/33 have identical `recall_traj`/`iou`/`conc`), while the
  same clip at another timestamp is a genuinely different sheet. A clip-level cap would have hard-
  limited the output to 35.
- at most `ceil(n_items/5)` items per task.

## Regenerating

```bash
cd $REPO && source env.sh
scripts/run_vitkd_sheets.sh --n-items 50 --target-items 65     # top 50, partial scan (~30 min)
scripts/run_vitkd_sheets.sh --render-idxs 31,205,372           # specific items
python -m TrajGazeMerge.viz.qual.vitkd_contact_sheet --from-scan --n-items 50   # CPU re-rank
```

`--target-items N` stops the scan once N items have passed the gates, so a top-N set does not need
the whole split. The scan runs **round-robin across the eight task blocks**, because SG item
indices are grouped by task — scanning in index order would make any partial scan cover only the
first few tasks and the "top N" silently task-biased.

Re-ranking is free: every item's selections are persisted, so `--from-scan` redraws on CPU in
seconds with different weights, caps or gates. `--resume` extends a scan rather than repeating it.

`GAZE_OVERLAY=1` with `VLM_GAZE_OVERLAY` unset is required; the launcher enforces it and the
script asserts `teacher VLM='viz' / student VLM='original' / teacher TAS='viz'` at the first item.
Per-item selections are persisted to `scan/sg.jsonl`, so a kill costs only the item in flight and
re-ranking never needs the GPU again (`--resume`, `--from-scan`).

## Read before using a sheet as evidence

1. **§5.4's finding is what these sheets exist to show, not a fault in them.** `recall_traj` = 0.383
   with **zero** GSM movement is the documented result: the ViT genuinely recovers ~38% of the
   teacher's gaze complement, and that recovery bought nothing on the tasks the complement exists
   to serve. Magenta landing away from the gazed object is evidence *for* that reading.
2. **§5.6: GSM is governed by whether the marker is in the pixels, not by which tokens are
   selected.** Overlay 49 items → raw video 45, and replacing the whole selection mechanism leaves
   it at 45. Do not read a GSM sheet as showing a selection failure.
3. **Single runs, ±4 item noise floor** (v2 §8). A method's ✓/✗ on one item is not a property of
   the method. Items are gated at min top-2 margin ≥ 0.05 because bf16 option logits quantise to
   1/8 and exact ties occur.
4. **Contextual tokens are centroids, and they are not independent.** The thin-outlined 3.5% mark
   `arange(0, n_non_dom, step)` over the *non-dominant* indices — so they move whenever the
   dominant set moves, and their embedding averages ~90% of all tokens. Their position carries far
   less meaning than the dominant boxes.
5. **The distilled score is a cross-frame, unmasked quantity** (v3 §4.2): VisionZip's column-sum is
   taken over the full T×T softmax while the actual attention output is block-diagonal per frame.
   What these sheets draw is a *global* saliency over the whole clip, not a per-frame one.
6. **SG's evidence often starts after the input ends.** 401 of 526 SG items have the fixation
   episode their answer was generated from beginning *after* the last input frame (median 2.1 s).
   Each sheet reports this in its own query-moment note, in red.
7. **The teacher uses gaze at test time and was trained on overlay frames**; the student has
   neither. The two differ in input, not only in selector. And Stage-1 TAS stays overlay-trained in
   every configuration (v2 §9) — an assumption, not a verified equivalence.
