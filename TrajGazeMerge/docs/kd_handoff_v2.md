# KD Handoff v2 — gaze-free distillation, one specialist per dataset

Written 2026-07-27, revised the same day (§1 rescoped, §7.4 / §10 / §11 added). Supersedes
`kd_handoff.md` as the working task definition; v1 remains the record of how we got here.

**One teacher and one student per benchmark.** What is dropped is *joint* training, not
EgoGazeVQA. Each of StreamGaze and EgoGazeVQA gets its own M1 teacher and its own gaze-free
student, selected with `--source {sg,eg}`.

> An earlier revision of this document declared the task "StreamGaze only". That was too strong:
> the measurement behind it (§1) rules out the *joint* setting, not EG. EG is back in scope as a
> separate specialist pair. Current EG state: teacher measured (§10), student **not yet run**
> (§11) — deferred by the user in favour of finishing the SG side first.

> **Read §7 first.** "gaze-free" in §1–§6 means *no trajectory-coordinate stream*. It does
> **not** mean the frames are free of gaze: the SG `viz` frames have the gaze marker drawn into
> the pixels, and that channel turns out to matter about twice as much as the one this project
> removes. §7 documents it and the setup that fixes it.

---

## 1. Why one specialist per dataset, and no joint model

v1 §7.3 measured, on the joint 1011-item benchmark, what having gaze/hand at inference is
actually worth:

| joint setting | gaze at test | items /1011 |
|---|---|---|
| M1 joint teacher | yes | 635 |
| content-only VisionZip | no | 632 |
| KD student | no | 631 |

**3 items.** The numeric noise floor — the same weights re-scored on different hardware — is
3–4 items. The entire privileged-information advantage the project exists to distil is inside
the noise, so **no method can be demonstrated in the joint setting**, however well tuned.

Note what that does and does not rule out. It is a statement about the *joint* model, because
that is the only setting it was measured in. On StreamGaze alone the same quantity is **16
items** (§2) — 4× the noise floor. So the joint model is what has to go, not a benchmark.

Three independent results all point at per-dataset specialists:

| evidence | finding |
|---|---|
| v1 §1 / §6.1 | SG-only teacher beats joint-on-SG (+2.28); joint-on-EG beats EG-only (+1.86). The best teacher **differs by dataset**. |
| v1 §6.3 | Both specialist *students* beat their own teacher, and the per-source pair (63.40) beats every gaze-using teacher measured. |
| v1 §7.5 | SG falls monotonically as complement replaces content while EG rises — SG's optimum is at or beyond 8/2, EG's near 6/4. **One global split cannot satisfy both.** |

Hence: one M1 teacher and one gaze-free student per benchmark, both selected with
`--source {sg,eg}`, and no joint row in any results table.

---

## 2. Reference numbers (this machine)

§2–§2.2 are StreamGaze (egtea n=526); **§2.3 is EgoGazeVQA** (egtea n=485). Never pool them into
one accuracy — they are different benchmarks with different option counts and task sets.

All gaze-free eval unless the column says otherwise, `GAZE_OVERLAY=1`, 10% budget at 7/3.
**Item counts are given alongside percentages: 1 item = 0.19%, and every effect here is
smaller than 2 points.**

| system | trained on | gaze at test | SG | items |
|---|---|---|---|---|
| M1 SG-only teacher | SG | **yes** | 71.67 | **377** |
| **SG specialist KD student** | SG | no | 70.15 | **369** |
| M1 joint teacher | SG∪EG | yes | 69.39 | 365 |
| joint KD student | SG∪EG | no | 68.82 | 362 |
| VisionZip content-only | SG∪EG | no | 68.63 | 361 |
| **VisionZip content-only, SG-only** | **SG** | no | *pending* | *pending* |

**Gaze is worth 16 items on SG (377 − 361); the gaze-free student recovers 8 of them.** That is
4× the noise floor, with 8 items still available — a real target.

### 2.1 The bar is not yet established — do not quote "+8 over VisionZip"

The 361 above comes from `visionzip_lora_sgeg_overlay`, a **jointly trained** LoRA scored on the
SG slice. The 369 student is **SG-specialized**. Those are different training regimes and the
regime is worth more than the effect:

| regime | teacher (gaze) | student (gaze-free) | VisionZip |
|---|---|---|---|
| SG-only | 377 | 369 | **pending** |
| joint | 365 | 362 | 361 |

Specialization alone moves the M1 teacher **365 → 377 = +12 items**. If VisionZip gains
comparably, an SG-only VisionZip lands near 373 and **beats** the student. In the one regime
where both are measured (joint), the student's margin over VisionZip is **+1 item — noise**.

Until the pending row is filled, **nothing supports the claim that the KD student beats
VisionZip.** That run is the first item in §5.

### 2.2 Per-task, SG specialist student (best epoch)

| task | specialist | joint student |
|---|---|---|
| past_gaze_sequence_matching (GSM) | **76.56** | 75.00 |
| past_non_fixated_object_identification | **70.59** | 66.18 |
| past_object_transition_prediction | 50.00 | 50.00 |
| past_scene_recall | 56.76 | — |
| present_future_action_prediction | 53.19 | — |
| present_object_attribute_recognition | **91.67** | 90.62 |
| present_object_identification_easy | 69.31 | — |
| present_object_identification_hard | 65.62 | 68.75 |

The specialist's gains concentrate in the **gaze-driven** tasks (GSM, non-fixated object) —
which is where the privileged signal should help, and where any new method must show up.

### 2.3 Reference numbers, EG (this machine, EG egtea n=485)

All from v1 §6.1 / §6.3 / §7.2, all machine-2, all `GAZE_OVERLAY=1` (marker in the pixels),
10% budget at 7/3. **Every one is a single run** — none meets §8's mean-of-≥3 rule.
1 item = 0.21%.

| system | trained on | gaze at test | EG | items |
|---|---|---|---|---|
| **EG specialist KD student** | EG | no | **56.08** | **272** |
| M1 joint teacher | SG∪EG | yes | 55.67 | 270 |
| joint KD student | SG∪EG | no | 55.46 | 269 |
| **M1 EG-only teacher** | EG | **yes** | 53.81 | **261** |

Two things to carry, both from v1:

- **The EG specialist student beats its own teacher by 11 items** while using *less* information.
  v1 §6.3 reads this as the complement barely helping on EG, so the RGB student loses nothing from
  dropping it and gains outright from single-source CE training. v1 §7.5 supports it structurally:
  EG *rises* as complement replaces content, the opposite of SG.
- **The shipped `EGonly/best.pth` is probably the ep2 snapshot, not the best epoch.** It scores
  53.81 here, which equals v1 §1's *ep2* value to two decimals (§1 records ep1 54.85 / ep2 53.81,
  best-of-2 = 54.85). Matching to within 1/485 by coincidence is unlikely. So the EG teacher row —
  including §10's per-task breakdown — measures the shipped file, which may be ~5 items below the
  teacher the specialist grid actually selected.

---

## 3. Fixed configuration — do not vary

| | value |
|---|---|
| dataset | **one specialist per source**, never joint — SG: train egoexolearn+holoassist (5799), eval egtea (526) · EG: train ego4d+egoexo (1265), eval egtea (485) |
| visual-token budget | **10%**, split **7% content ∪ 3% complement** |
| selection rule | hard top-k over the tokens VisionZip discarded |
| student inputs | **RGB-only**: `video_embeds`, `attn_scores`, `grid_thw`. No gaze, no query embedding. |
| gaze/hand | TRAIN time only (teacher labels); eval is gaze-free |
| frames | `GAZE_OVERLAY=1` (marker burned into pixels, as in every baseline) |
| batch | eff-batch 8 (2 GPUs × grad-accum 4) |
| protocol | `--epochs 2 --early-stop`, best-of-2 |
| MCQ | SG: 4 options (A–D), 8 task types · EG: 5 options (A–E), 3 task types (spatial/temporal/causal) |

Every trainer takes `--source {sg,eg}`. `train_visionzip_lora.py` gained it in v2; the M1 and KD
trainers already had it. **Run one trainer per source** — a `--source both` run is the joint model
§1 rules out.

---

## 4. What carries over from v1, and what does not

> Every `§7.x` in this section means **v1's** `kd_handoff.md` §7, not this document's §7. The two
> now collide (v2 has its own §7.3 / §7.4), so section numbers here are written as **v1 §7.x**.

**Carries over:**
- **v1 §7.6 port fix.** `train_autogaze_lora.py` hardcoded machine-1 SG paths, so
  `StreamGazeSimpleDataset` silently returned 0 items and `CombinedSimpleDataset` scored EG-only
  while reporting the full set. Fixed. **Any VisionZip-side eval from before that fix is
  invalid.**
- **Numeric noise floor ±3–4 items per source.** Re-scoring identical weights here moved the
  joint student SG +4 / EG −3. Treat sub-4-item differences as unmeasured.
- **Reproduction.** Both machine-1 checkpoints re-score here to within 1 item, so the port is
  faithful.

**Does NOT carry over — re-measure per source before relying on it:**
- **v1 §7.4 (agreement ↑ → −8 SG items).** Measured on the *joint* student with a joint LoRA. On
  SG, where gaze is worth 16 items rather than 3, it may not hold. **This is the pivotal open
  question** (§5, item 3).
- **v1 §7.5 split sweep.** Joint student, and the split is locked at 7/3 here anyway. Its
  *direction* — SG and EG wanting opposite allocations — is what §1 relies on, not its values.
- **v1 §7.3's 3-item ceiling.** A joint-setting fact. On SG the ceiling is 16.

---

## 5. Open items, in order

1. **SG student retrained overlay-free** — running (§9). The one number that decides whether a
   genuinely gaze-free student is viable at all; 354 is the bar it must beat.
2. **SG-only VisionZip bar** — **not running.** §9 records that it was started and stopped at step
   2580/2900 with no checkpoint. Until it exists, §2.1 stands and "+8 over VisionZip" must not be
   quoted. (An earlier revision of this list said "running"; that was stale.)
3. **Does agreement help on SG?** `--freeze-lora --source sg` from the existing SG specialist,
   one epoch. The LoRA is held fixed, so any change is selection alone. Up ⇒ selection-KD is
   alive on SG and soft-field distillation is next; down ⇒ v1 §7.4 generalises and the
   remaining 8 items need a different mechanism.
4. **Training noise floor** — two seed repeats of the SG specialist run. Needs a `--seed` flag
   (not yet present). Every training number in this document is a single run, and the target
   margins are 8 items.
5. **EG specialist student** (§11) — deferred, not blocked.

---

## 6. Reproduce

```bash
cd /NHNHOME/VILAB/vilab_yj/trajgaze && source env.sh    # sets SG_ROOT, GAZE_OVERLAY=1, $TORCHRUN

# content-only VisionZip bar, SG-only
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29771 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --source sg --dominant-ratio 0.05 --contextual-ratio 0.05 \
  --output-dir $REPO/TrajGazeMerge/checkpoints/visionzip_content10_SGonly_overlay \
  --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop

# gaze-free KD student, SG-only (overlay still in the pixels — the §2 row)
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29661 \
  -m TrajGazeMerge.training.train_visionzip_kd_lora \
  --source sg --warmstart-ckpt "$M1_SGONLY" --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir $REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_SGonly_overlay \
  --epochs 2 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 --no-hdepic --early-stop

# TRULY gaze-free student — student pixels de-gazed, teacher stream keeps the marker (§7.3)
scripts/run_kd_sg_nooverlay.sh          # VLM_GAZE_OVERLAY=0 GAZE_OVERLAY=1, --source sg
# EG equivalent: same script with --source eg, --warmstart-ckpt "$M1_EGONLY"

# M1 teacher, per-task eval (per-source; --source eg → n=485, --source sg → n=526)
CUDA_VISIBLE_DEVICES=1 $TORCHRUN --nproc_per_node=1 --master_port=29821 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.07 --traj-ratio 0.03 --source eg --eval-ckpt "$M1_EGONLY" \
  --output-dir $REPO/TrajGazeMerge/checkpoints/_eval_egteacher_pertask \
  --no-hdepic --eval-progress-every 100
```

Every `--eval-ckpt` run must report **n=526** for `--source sg` and **n=485** for `--source eg`.
If it reports 1011 the source filter did not apply.

### Checkpoints

- M1 SG-only teacher: `$M1_SGONLY` (→ `aaai/visionzip_complement_learned_SGonly_overlay`)
- M1 EG-only teacher: `$M1_EGONLY` (→ `aaai/visionzip_complement_learned_EGonly_overlay`)
- SG specialist KD student: `TrajGazeMerge/checkpoints/visionzip_kd_selection_SGonly_overlay/best.pth`
- SG student, overlay-free: `…/visionzip_kd_selection_SGonly_nooverlay/best.pth` (§9, in progress)
- machine-1 references: `datasets/trajgazemerge/hf_m1/aaai/…` (separate root — the KD student
  collides by name with a local run directory)

### Flags and scripts added in v2

`train_visionzip_lora.py`: `--source`, `--eval-ckpt`, per-source eval reporting.
`train_visionzip_kd_lora.py`: `--freeze-lora`, `--balance-sources`; `--warmstart-ckpt` now
carries `pred_state` when present, so a *student* checkpoint resumes at its own score instead of
being paired with a fresh random head; and the §7.3 stream-differ assertion.
`data/egogaze_dataset.py`: `VLM_GAZE_OVERLAY` support (`_EG_VLM_FRAME_SUB`), mirroring `dataset.py`.
`scripts/`: `run_kd_sg_nooverlay.sh`, `fix_sg_original_fps.sh`; `extract_sg_original_frames.sh`
now rate-corrects and verifies against `viz` instead of skipping any non-empty directory (§7.4).

---

## 7. The gaze overlay — "gaze-free" was never gaze-free

### 7.1 What was found

SG frames under `frames/{ds}/viz/` have the gaze point drawn into the pixels as a **red circle
with a green dot** (verified visually). `viz` is the only variant that was extracted, and the
dataset README labels those tars "with gaze visualization". So every result in §1–§6 — teacher,
student, and VisionZip alike — was produced on frames that *show the model where the person was
looking*.

What the KD student removes is only the **trajectory-coordinate stream** (gaze x/y, hand
pos/vel, interaction features → TAS encoder). v1 §2 states this correctly, but v1 §1's claim
that M1 "needs an eye-tracker at test time — that is exactly what the KD student removes" does
**not** hold: drawing the marker requires knowing the gaze, so the student still needs an
eye-tracker.

### 7.2 The pixel channel is worth ~2× the stream

`videos_*_original.tar` (same footage, no overlay) exists for all three datasets. Extracting
egtea and re-scoring the SG specialist student, changing nothing else:

| SG KD student | frames | SG | items |
|---|---|---|---|
| as reported in §2 | `viz` (marker present) | 70.15 | 369 |
| **true gaze-free** | `original` (no marker) | **67.30** | **354** |

| gaze channel removed | cost |
|---|---|
| trajectory coordinate stream (teacher → student, both on `viz`) | **7 items** |
| **pixel overlay** (student, `viz` → `original`) | **15 items** |

**The project removed the smaller of the two gaze channels and kept the larger one.**

Caveat: this student was *trained* on `viz`, so 15 items conflates the overlay's information
content with distribution shift. It is an **upper bound** on overlay dependence; retraining on
`original` (§9) is the clean measurement.

### 7.3 Setup: teacher keeps the overlay, student does not

Decided with the user. The teacher exists only at train time, so it may use everything; the
student is the deployed artifact and must not see the marker.

This also sidesteps a blocker. `get_patch_scores_temporal` feeds `item["traj_frame_paths"]` to
the frozen TAS encoder, whose checkpoint (`stage1_tas_3way_overlay`) was trained on overlay
frames — and retraining it is **100 epochs × 4 GPUs** (`launch_stage1_tas_3way.sh`), out of
reach on 2 GPUs. Keeping the teacher on `viz` leaves it exactly in-distribution.

It also widens the gap distillation has to close: teacher **375** (overlay + coords, §8's 4-run
mean) vs student 354 (neither) = **21 items**, versus 7 in the all-overlay setting.

`GAZE_OVERLAY` used to move both streams at once, so this configuration was not expressible.
Both dataset modules now separate them —
[data/dataset.py](VILAB/vilab_yj/trajgaze/TrajGazeMerge/data/dataset.py) for SG and
[data/egogaze_dataset.py](VILAB/vilab_yj/trajgaze/TrajGazeMerge/data/egogaze_dataset.py) for EG:

| env var | controls | set to | SG dir | EG dir |
|---|---|---|---|---|
| `GAZE_OVERLAY` | `traj_frame_paths` → TAS teacher | **1** | `viz` | `gaze` |
| `VLM_GAZE_OVERLAY` | `vlm_frame_paths` → student's VLM input | **0** | `original` | `no_gaze` |

`VLM_GAZE_OVERLAY` defaults to `GAZE_OVERLAY`, so unset behaviour is unchanged. `_find_dataset`
stays on the teacher variant, which is always present.

Verified on both sources:

```
SG   teacher_sub=viz   vlm_sub=original
       student VLM : original/OP01-R01-PastaSalad/frame_000001.jpg
       teacher TAS : viz/OP01-R01-PastaSalad/frame_000001.jpg          n=526
EG   teacher_sub=gaze  vlm_sub=no_gaze
       student VLM : no_gaze/OP01-R01-PastaSalad/4017_4713_4018.jpg
       teacher TAS : gaze/OP01-R01-PastaSalad/4017_4713_4018.jpg       n=485
```

The KD trainer now also **asserts** this at startup: whenever `VLM_GAZE_OVERLAY` differs from
`GAZE_OVERLAY`, rank 0 prints `[KD] frame streams: student VLM='…' teacher TAS='…'` and raises if
the two resolve to the same directory. Training both streams on one variant changes no shape,
raises no error, and does not show up in the accuracy — it has to be caught structurally.

### 7.4 Index alignment is a precondition, and it was violated (holoassist)

An earlier revision of this section asserted "both variants hold identical per-video frame counts,
so the timestamp cutoff and sampling indices stay aligned." **That was verified on egtea only, and
it was false for holoassist.** Both loaders address frames *by index* and cut at
`int(ts_sec × EXTRACTED_FPS)`, so the two variants must be frame-for-frame parallel or the student
and teacher silently look at different moments.

What was wrong: the two encodes of the same footage declare **different frame rates**.

| `R005-7July-GoPro` | `viz` | `original` |
|---|---|---|
| nb_frames | 8817 | **8817** — identical, nothing is missing |
| frame rate | 24.46 fps | **29.83 fps** |
| duration | 360.5 s | **295.6 s** |
| jpgs after `-vf fps=10` | 3605 | **2956** |

`3605/2956 = 0.820 = 24.46/29.83` exactly. `fps=10` samples by **time**, so the same cadence walks
the same frames at a different rate. Measured as normalised cross-correlation against `viz` at the
same index, the drift accumulates until the "same" frame is unrelated footage:

| frame index | before fix | after fix |
|---|---|---|
| 300 | 0.79 | **1.000** |
| 1500 | 0.31 | **0.999** |
| 2700 | **−0.34** | **0.998** |

**32 of 66 holoassist videos** were affected — all in the SG *training* split, so the damage would
never have surfaced in an egtea eval.

Fix: both encodes hold the same frame *sequence*, so retiming the original to the viz rate
restores index alignment — `fps_viz = nb_frames × 10 / n_viz_jpgs`, then
`ffmpeg -r $fps_viz -i org.mp4 -vf fps=10`. Applied by
`scripts/fix_sg_original_fps.sh` (stages each video, swaps in only on an exact count match).
`scripts/extract_sg_original_frames.sh` was carrying the root cause: its skip guard accepted **any
non-empty directory**, so re-running could never repair this. It now skips only on an exact count
match with `viz`, derives the rate per video, and logs `MISMATCH` if the result still disagrees.

**This is an SG-only defect.** holoassist belongs to StreamGaze and does not exist in EgoGazeVQA —
`DATASETS = ["egtea","egoexolearn","holoassist"]` in `data/dataset.py` versus
`TRAIN_DS = ["ego4d","egoexo"] / VAL_DS = ["egtea"]` in `data/egogaze_dataset.py`, and EG's
`metadata.csv` holds only egoexo 688 / ego4d 577 / egtea 485. So the repair touches the SG
*training* split and nothing on the EG side.

Post-fix parity, re-verified independently:

| source | dataset | videos | count mismatch |
|---|---|---|---|
| SG | egtea (eval) | 35 | **0** |
| SG | holoassist (train) | 66 | **0** |
| SG | egoexolearn (train) | 180 | 4, each ±1 frame out of 4.5k–21k (NCC ≥ 0.91 — benign, left alone) |
| EG | egtea + ego4d + egoexo, `gaze` ↔ `no_gaze` | 263 | **0**, filenames identical — no extraction was ever needed |

> **`egtea` is in both benchmarks and is not the same data.** SG's `egtea/viz` holds 35 videos,
> EG's `egtea/gaze` holds 82; different footage, different QA, different option counts (SG 4-way
> A–D, EG 5-way A–E). The video totals never overlap — SG 35+180+66 = 281, EG 82+27+154 = 263 —
> so a count that matches neither is a sign the wrong loader is in play.

**egtea was clean throughout, so every SG number already published stands.** What this
invalidates is nothing measured; it is what it would have silently broken in the retrain.

---

## 8. Measurement protocol — eval is NOT deterministic

Re-scoring the **same checkpoint** on the **same machine** with the **same flags**:

| M1 SG-only teacher | Avg | items |
|---|---|---|
| run 1 (no per-task) | 71.67 | 377 |
| run 2 | 71.29 | 375 |
| run 3 | 71.48 | 376 |
| run 4 | 70.72 | **372** |
| **mean of 4** | **71.29** | **375**, spread **5 items** |

All four report n=526 with no items skipped, so this is bf16/flash-attn nondeterminism, not
data loss. Note run 4 is 5 items below run 1 — **three samples had suggested ±1; four show ±3.**
Do not trust a spread estimated from fewer than ~4 runs.

**Per-task columns swing harder still**, because their n is small (3 runs with per-task):

| | GSM (64) | NFI (68) | SR (37) | OAR (96) | OI-E (101) | OI-H (64) | FAP (94) |
|---|---|---|---|---|---|---|---|
| run 2 | 70.31 | 64.71 | 59.46 | 93.75 | 73.27 | 73.44 | 56.38 |
| run 3 | 71.88 | 63.24 | 56.76 | 93.75 | 73.27 | 75.00 | 57.45 |
| run 4 | 71.88 | 61.76 | 56.76 | 92.71 | 73.27 | 75.00 | 55.32 |
| **mean** | **71.36** | **63.24** | **57.66** | **93.40** | **73.27** | **74.48** | **56.38** |
| range | 1.57 | **2.95** | **2.70** | 1.04 | 0.00 | 1.56 | 2.13 |

Mean over those 3 complete runs: **71.17% (374.3 items)**.

**Consequence for the paper table:** the teacher does *not* beat the Full-token (100 %) row.
Full-token is 71.29; run 3 alone reads 71.48 and appears to win, but the 3-run mean is 71.17 and
the 4-run Avg mean is 71.29 — a tie at best. Any "beats full-token" claim here is a
single-run artifact.

Rules that follow:
- Report the **mean of ≥3 evals**, not a single run, and say so in table captions.
- Do not select the best of N runs — that is an upward-biased estimator.
- **Column-level differences under ~2 points are not evidence.** SR's 37 items make one
  question worth 2.70%.
- Cross-machine is worse still: the same VisionZip checkpoint gives GSM 65.62 on machine 1 and
  70.31 here. Do not mix machine-1 and machine-2 rows in one table.

---

## 9. Status — done and remaining

### Done

- [x] `--source` / `--eval-ckpt` / per-source eval added to `train_visionzip_lora.py`
- [x] Per-task printing added to the eval-only branch of **both** the M1 and KD trainers
      (`evaluate()` computed `per_task` and discarded it)
- [x] **Port fix**: `train_autogaze_lora.py` hardcoded machine-1 SG paths →
      `StreamGazeSimpleDataset` returned 0 items and `CombinedSimpleDataset` silently scored
      EG-only (485) as if it were the full set. *Any VisionZip-side eval predating this is invalid.*
- [x] Gaze overlay identified and quantified (§7.2)
- [x] Frame streams decoupled via `VLM_GAZE_OVERLAY` (§7.3) for **both** SG
      (`viz`/`original`) and EG (`gaze`/`no_gaze`), verified both ways on both sources
- [x] Stream-differ **assertion** added to `train_visionzip_kd_lora.py` (§7.3)
- [x] SG `original` frames extracted for all three datasets **and verified per video** —
      egtea 35/35, holoassist 66/66, egoexolearn 176/180 exact + 4 off by one frame (§7.4)
- [x] **holoassist fps misalignment found and repaired** — 32/66 videos were sampling different
      moments than `viz`; `scripts/fix_sg_original_fps.sh`, plus the root-cause guard in
      `scripts/extract_sg_original_frames.sh` (§7.4)
- [x] EG `gaze`/`no_gaze` parity verified — 263 videos, 0 mismatches, no extraction needed
- [x] Teacher measured 4× → **375 items, spread 5**; 3 of them with per-task (§8)
- [x] SG student measured on `original` → **354 items** (single run)
- [x] **M1 EG-only teacher per-task** — Spat./Temp./Caus./Avg (§10)

### In progress

*Nothing running.* All extraction finished and was verified; no GPU job is active.

### Halted by user instruction (2026-07-27)

- [ ] **SG student retrain** with `VLM_GAZE_OVERLAY=0 GAZE_OVERLAY=1 --source sg`, warm-started
      from the overlay-trained M1 SG-only teacher (~4.3 h) — `scripts/run_kd_sg_nooverlay.sh`,
      log `kd_train_sgonly_nooverlay.log`. **Not started.** This remains the highest-probability
      lever on the 354 number: that figure is an overlay-trained model run off-distribution, so
      removing the shift should recover much of the 15-item overlay loss. It would also be the
      first run to use the repaired holoassist frames.

Prerequisites are all in place — extraction complete and verified, streams decoupled, assertion
added — so this is a single launch whenever it is wanted.

### To do

- [ ] Repeat the student eval ≥3× before publishing any student number (§8); every student
      figure in this document is a single run.
- [ ] Per-task breakdown of the no-overlay student — GSM and NFI are *defined* in terms of
      gaze and are expected to carry most of the 15-item drop.
- [ ] **EG specialist student** (§11) — deferred by the user, not blocked. Everything it needs is
      in place: `no_gaze` frames verified, decoupling implemented, `--source eg` supported.

### Explicitly dropped

- No-overlay **VisionZip** bar — user decision.
- SG-only VisionZip bar (§2.1) — training was started then stopped at step 2580/2900 of
  epoch 1, no checkpoint written. **§2's "+8 over VisionZip" therefore remains unverified**
  and must not be quoted.
- Stage-1 TAS retraining on `original` — 100 epochs × 4 GPUs, out of budget. The teacher's
  visual branch stays overlay-trained; state this as an assumption, not a verified equivalence.

---

## 10. EgoGazeVQA specialist teacher — per-task

`M1_EGONLY` (`visionzip_complement_learned_EGonly_overlay/best.pth`) scored with
`--source eg --eval-ckpt`, `GAZE_OVERLAY=1`, 10% budget at 7/3, this machine.
Log: `eval_egteacher_pertask.log`. EG task columns are `qa_type`; **1 item = 0.61–0.63%** per
column and 0.21% on Avg.

| M1 EG-only teacher | Spat. (163) | Temp. (160) | Caus. (162) | **Avg (485)** |
|---|---|---|---|---|
| run 1 (2026-07-26, no per-task) | — | — | — | 53.81 (261) |
| run 2 | 39.88 (65) | 36.25 (58) | 86.42 (140) | 54.23 (263) |
| run 3 | 39.88 (65) | 36.25 (58) | 85.80 (139) | 54.02 (262) |
| **mean** | **39.88 (65)** | **36.25 (58)** | **86.11 (139.5)** | **54.02 (262)** |

**Three Avg samples, spread 2 items (261/262/263) — this meets §8's mean-of-≥3 rule.** The two
per-task samples are *identical* on spatial and temporal and differ by one item on causal, so this
checkpoint's per-task profile is unusually stable; unlike SG's 37-item columns (§8), EG's ~160-item
columns are steady.

Logs: `eval_m1_egonly.log`, `eval_egteacher_pertask.log`, `eval_egonlyteacher_r2_eg_pertask.log`.

### 10.1 Reference rows (this machine, all `gaze` frames)

| system | gaze at test | Spat. | Temp. | Caus. | Avg |
|---|---|---|---|---|---|
| **M1 joint teacher, on EG** | **yes** | 40.49 (66) | **43.12 (69)** | 83.95 (136) | **55.88 (271)** |
| EG specialist KD student (v1 §6.3) | no | 40.49 (66) | 42.50 (68) | 85.19 (138) | 56.08 (272) |
| **M1 EG-only teacher** (mean of 3) | **yes** | 39.88 (65) | 36.25 (58) | **86.11 (139.5)** | 54.02 (262) |
| joint KD student (v1 §6.3) | no | 38.04 (62) | 36.25 (58) | 84.57 (137) | 52.99 (257) |

The joint-student row comes from the §6.2 run that v1 §7.2 later identified as a **bad training
run**, so treat it as a lower bound rather than a fair joint baseline. The joint-teacher row is new
(log `eval_jointteacher_eg_pertask.log`) and is a single run.

### 10.1a The `42.50 / 84.57 / 56.29` row is the joint teacher — and it does not reproduce here

That row could not be matched to any system whose per-task had been measured, which raised the
possibility it was a composite of three (42.50 = EG specialist student's temporal, 84.57 = joint
student's causal, 56.29 = joint teacher's Avg). **Measuring the joint teacher's per-task settles it
the other way**: every column lands within 2 items of the row.

| column | joint teacher, here | target row (machine 1) | Δ items |
|---|---|---|---|
| Spat. | 40.49 (66) | 41.72 (68) | −2 |
| Temp. | 43.12 (69) | 42.50 (68) | +1 |
| Caus. | 83.95 (136) | 84.57 (137) | −1 |
| **Avg** | **55.88 (271)** | **56.29 (273)** | **−2** |

So the row is one system — the **joint teacher** — and the whole discrepancy is 2 items of
cross-machine variance. Both columns sum correctly (66+69+136 = 271; 68+68+137 = 273), so neither
is a transcription error.

**Consequence for any "≥56.29" target.** This checkpoint has now been scored on this machine twice
— 55.67 (270 items, v1 §6.1) and 55.88 (271) — against 56.29 (273) on machine 1. **56.29 is not
reproducible here**; the same weights give 270–271. A target of 56.29 is therefore 2–3 items above
what the machine returns for the model that defined it, i.e. inside the noise floor (§8). State the
bar as **271 items / 55.88% on this machine**, and treat any margin under ~4 items as unmeasured.

### 10.1b Joint vs EG-only, same machine: 9 items, all temporal

With both teachers now measured here, the "joint training helps EG" finding (v1 §1) can be split by
column instead of taken as a single number:

| column | EG-only | joint | Δ items |
|---|---|---|---|
| Spat. | 65 | 66 | +1 |
| Caus. | **139** | 136 | **−3** |
| **Temp.** | **58** | **69** | **+11** |
| **Avg** | **262** | **271** | **+9** |

The EG-only teacher is *better* on causal and level on spatial; its entire 9-item deficit — and
more — is **temporal**. Joint training does not lift EG broadly; it buys temporal specifically.

This is the same axis as §10.2: the EG specialist *student* also beat its teacher almost entirely on
temporal (+10 items, to 68). Three systems now cluster at **68–69 temporal items** (joint teacher
69, EG specialist student 68, target row 68) while the EG-only teacher sits alone at **58**.

That pattern favours the undertraining explanation over cross-dataset transfer. EG train is 1265
items = **158 optimizer steps/epoch** at eff-batch 8, so the 2-epoch specialist recipe saw **316
steps**, against ~2649 for the joint model — **8×** fewer. Temporal questions plausibly need the
most optimisation to fit, and they are exactly what the short EG-only schedule fails to learn. This
is testable: train EG-only for more epochs and watch whether the temporal column climbs toward 68.

### 10.2 The student's whole margin is temporal

Teacher → EG specialist student, in items:

| column | Δ items |
|---|---|
| Caus. | −2 |
| Spat. | +1 |
| **Temp.** | **+10** |
| Avg | **+9** |

The gaze-free student's entire 9-item win over its own gaze-using teacher sits in **temporal**;
causal and spatial are flat within noise. At 10 items — 2.5× the 3–4 item floor and 6.25 points on
a 160-item column — this is the one EG effect large enough to be a measurement rather than a
coincidence.

Read with v1 §7.5 (EG *rises* as complement replaces content, SG falls), the reading is that the
trajectory complement actively hurts EG's temporal questions, and dropping it is a gain rather
than a loss. That is the opposite of v1 §3's expectation. Two caveats before leaning on it:

1. The two rows come from **different eval paths** — the teacher via `--eval-ckpt`, the student
   from the training loop — and the student is a single run (the teacher is now a mean of 3).
2. The teacher checkpoint is probably the ep2 snapshot rather than the best epoch (§2.3), which
   would depress every teacher column by an unknown amount.
3. §10.1b now shows the joint *teacher* reaches the same temporal level (69 items) with gaze
   fully available, so "dropping the complement helps temporal" is not the only reading — "more
   optimisation steps help temporal" fits all four systems at once and is the simpler one.

Also worth noting for the paper: with 5 options, chance is 20%. The teacher clears it by ~4× on
causal (86.42) but only ~2× on spatial and temporal (39.88 / 36.25). EG's difficulty is
concentrated in exactly the two columns where the gaze complement does not help.

---

## 11. EgoGazeVQA specialist student — not yet run

Deferred by the user in favour of finishing the SG side; **not blocked**. Recorded here so the
next session does not re-derive the setup.

What exists today is the *overlay-trained* EG student, 56.08 / 272 items (§2.3) — trained and
evaluated on `gaze` frames, i.e. with the marker in the pixels. It is the EG analogue of the SG
student's 369, and the same criticism applies: it is not gaze-free in the sense §7 means.

Everything the overlay-free run needs is in place:

- `no_gaze` frames — present for all three EG splits, **263 videos with byte-identical filenames
  to `gaze`** and no count mismatches (§7.4). No extraction step at all, unlike SG.
- `VLM_GAZE_OVERLAY` support in `data/egogaze_dataset.py` — implemented and verified both ways
  (§7.3).
- `--source eg` in the KD trainer, and `$M1_EGONLY` for the warm-start.

Command (~1.5–2 h at eff-batch 8, from v1 §6.3's timing for the overlay run):

```bash
GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=0 \
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29662 \
  -m TrajGazeMerge.training.train_visionzip_kd_lora \
  --source eg --warmstart-ckpt "$M1_EGONLY" --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir $REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_EGonly_nooverlay \
  --epochs 2 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 --no-hdepic --early-stop
```

Expect the §7.3 assertion line `[KD] frame streams: student VLM='no_gaze' teacher TAS='gaze'` in
the log; its absence means the run is not the experiment it claims to be.

Two predictions worth recording before the fact, so the result can falsify them:

1. **The EG overlay should cost less than SG's 15 items.** v1 §7.5 shows the gaze complement helps
   EG and hurts SG, so EG leans less on gaze overall. If EG drops as hard as SG, that reading is
   wrong.
2. **The warm-start is overlay-trained**, so as on SG some of any drop is distribution shift
   rather than lost information. Only this retrain separates the two.
