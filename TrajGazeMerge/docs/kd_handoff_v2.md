# KD Handoff v2 — gaze-free distillation, one specialist per dataset

Written 2026-07-27, revised the same day (§1 rescoped, §7.4 / §10 / §11 added). Supersedes
`kd_handoff.md` as the working task definition; v1 remains the record of how we got here.

**Revised 2026-07-28.** The overlay-free students both exist now — **§7.7 is the section to read**.

New: §2.2a / §10.3 (per-task for both *overlay* students, the baselines), **§7.7** (both
overlay-free students, per task), §7.4a / §7.4b (the frame-extraction defects), §10.4 (the
warm-start confound), §12.3a (measured module sizes), §13 (environment migration).

Four results changed what this document says:

1. **The overlay is worth 9 items, not 15** (§7.2). Retraining on `original` recovered 6 of the
   15, so the two gaze channels are 7 vs 9 — comparable, not 2×.
2. **Frame extraction was broken on two datasets** — 13/66 holoassist and **54/180 egoexolearn**
   stems sampled the wrong moments while holding the right frame *count*, because the count was
   derived from the target (§7.4a, §7.4b). 12.7% of SG training items were affected. egtea, the
   eval split, was clean, so no published number moves.
3. **§10.2 is retracted.** It read v1 §7.5 backwards; that sweep says the complement *helps* EG.
   The EG student's temporal margin was the pixel marker, and it disappears without it (§7.7).
4. **"The student beats its teacher" is a training-budget artefact** (§10.4) — the student is
   warm-started from the teacher and then trained for twice the optimizer steps.

Corrections marked inline: §5-1 vs §9 disagreed about whether the SG overlay-free retrain had
started (**it started and was SIGTERM'd at step 480/2900**); §10.1 mis-attributed the EG student
row to machine 1 (**it is a machine-2 measurement**); and §5-5 understated the seeding gap
(**no trainer seeds any RNG at all**).

**One teacher and one student per benchmark.** What is dropped is *joint* training, not
EgoGazeVQA. Each of StreamGaze and EgoGazeVQA gets its own M1 teacher and its own gaze-free
student, selected with `--source {sg,eg}`.

> An earlier revision of this document declared the task "StreamGaze only". That was too strong:
> the measurement behind it (§1) rules out the *joint* setting, not EG. EG is back in scope as a
> separate specialist pair. **Both EG models now exist**: teacher (§10), overlay student (§10.3),
> overlay-free student (§7.7, §11).

> **Read §7 first.** "gaze-free" in §1–§6 means *no trajectory-coordinate stream*. It does
> **not** mean the frames are free of gaze: the SG `viz` frames have the gaze marker drawn into
> the pixels, and that channel is worth **more** than the one this project removes — 9 items vs
> 7, measured in §7.2 after the overlay-free retrain. §7.7 has the genuinely marker-free numbers;
> §1–§6 do not.

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

### 2.2a SG overlay student vs its teacher, per task — the baseline the no-overlay run must be read against

Both overlay students were trained **on this machine on 2026-07-27**, best = epoch 2, scored by the
training loop's own `evaluate()`. Recorded here in full so the no-overlay retrains (§9) have a
like-for-like row to sit next to. Item counts are derived from §8's per-task `n` and sum **exactly**
to the reported total (49+48+1+21+50+88+70+42 = 369).

`visionzip_kd_selection_SGonly_overlay/best.pth` — `70.15% (n=526)`, log `kd_train_sgonly.log:396-406`.
Teacher column is the 3-run mean from §8.

| task | n | student % | items | teacher (mean of 3) | Δ items |
|---|---|---|---|---|---|
| GSM  `past_gaze_sequence_matching` | 64 | **76.56** | 49 | 71.36 | **+3** |
| NFI  `past_non_fixated_object_identification` | 68 | **70.59** | 48 | 63.24 | **+5** |
| OTP  `past_object_transition_prediction` | 2 | 50.00 | 1 | — | — |
| SR   `past_scene_recall` | 37 | 56.76 | 21 | 57.66 | 0 |
| OAR  `present_object_attribute_recognition` | 96 | 91.67 | 88 | 93.40 | −2 |
| OI-E `present_object_identification_easy` | 101 | 69.31 | 70 | 73.27 | −4 |
| OI-H `present_object_identification_hard` | 64 | 65.62 | 42 | 74.48 | **−6** |
| FAP  `present_future_action_prediction` | 94 | 53.19 | 50 | 56.38 | −3 |
| **Avg** | **526** | **70.15** | **369** | 71.17 | **−6** |

The trade is legible: the gaze-free student **gains on the two gaze-driven columns** (GSM +3, NFI +5)
and **loses on object identification** (OI-H −6, OI-E −4). It does not beat its teacher overall; it
beats it exactly where the privileged signal was supposed to matter.

**This is the column profile the no-overlay student has to be compared against**, not just the 369
total — the overlay is a *pixel* cue, so if it is carrying the GSM/NFI gains those two columns should
fall hardest when it is removed.

Caveat: single run, and §8 measures a 5-item spread on re-evaluation of identical weights. Columns
with n < 40 (SR, OTP) move ≥2.7% per question — do not read those Δ.

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

Revised 2026-07-28. The two remaining *training* jobs are the overlay-free students, SG and EG.
Everything else on this list is either closed or not scheduled.

1. **SG student retrained overlay-free** — **queued, behind the Table 6/7 ablation** (§9).
   The one number that decides whether a genuinely gaze-free student is viable at all;
   **354 is the bar it must beat**, and §2.2a is the per-task profile to compare against.
   Prerequisite: the holoassist `original` frame re-download (§9).
2. **EG student retrained overlay-free** — **queued, behind item 1** (§11). Bar: **272 items /
   56.08%**, per-task in §10.3. Needs a launcher; none exists yet (`run_kd_sg_nooverlay.sh` is the
   template). No frame download needed — EG `no_gaze` is already parity-verified.
3. **SG-only VisionZip bar** — **closed, will not be re-run.** The user holds this measurement
   outside this repo. **§2.1's pending cell is still literally empty**, so until the held value is
   pasted into that table "+8 over VisionZip" remains unquotable *in this document*. Filling that
   cell is a transcription task, not a GPU task.
4. **Does agreement help on SG?** — **still unanswered; the run that exists does not answer it.**
   `kd_train_frozenlora.log` was launched with **`--source both`, not `--source sg`**, warm-started
   from the machine-1 *joint* student, and died during epoch 2 of 3 with only one eval recorded
   (61.62%, n=1011, agree 0.455 → 0.476). So it speaks to the joint setting, partially. The SG
   question needs `--freeze-lora --source sg` from the SG specialist. Not scheduled.
5. **Training noise floor** — not scheduled, and **worse than previously recorded.** §5 used to say
   a `--seed` flag was missing; in fact **none of the three trainers seeds any RNG at all** — no
   `torch.manual_seed` / `random.seed` / `np.random.seed`, and `DistributedSampler` is constructed
   without `seed=`. Data order is reproducible by PyTorch's default; LoRA init and dropout are not.
   `--balance-seed` exists but only reseeds source balancing. Consequence: every repeat run in this
   document differs by unrecorded nondeterminism, and two "seeds" would not be a seed comparison.
   The 4-line idiom to copy is `train_merge_lora_temporal_no_kd.py:293-296`.
6. **Repeat every student eval ≥3×** — not scheduled. Every student number here is a single run
   while teacher rows are 3-run means (§8). `repeat_student_r1.log` died before scoring, which is
   why the student side never got past one sample.

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

### 7.2 The pixel channel is worth more than the stream — 9 items vs 7, not 15 vs 7

`videos_*_original.tar` (same footage, no overlay) exists for all three datasets. Extracting
egtea and re-scoring the SG specialist student, changing nothing else:

| SG KD student | frames | SG | items |
|---|---|---|---|
| as reported in §2 | `viz` (marker present) | 70.15 | 369 |
| overlay-trained, scored off-distribution | `original` (no marker) | 67.30 | 354 |
| **retrained on `original`** (§7.7) | `original` (no marker) | **68.44** | **360** |

The 15-item figure this section originally reported was flagged as an **upper bound**, because
that student was *trained* on `viz` and so paid distribution shift on top of any lost
information. Retraining on `original` settles it: **6 of the 15 items come back**.

| gaze channel removed | cost |
|---|---|
| trajectory coordinate stream (teacher → student, both on `viz`) | **7 items** |
| **pixel overlay** (student retrained on `original`) | **9 items** |
| ~~pixel overlay, upper bound from the off-distribution score~~ | ~~15 items~~ |

**The project still removed the smaller of the two gaze channels** — but the gap is 1.3×, not
the 2× this section previously claimed. Both channels are worth roughly the same.

Per-task, the loss lands where it should: GSM, the gaze-driven task, is the single largest drop
(§7.7).

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

### 7.4a The local repair did NOT actually fix it — 13/66 stems were still misaligned

Measured 2026-07-28, and it is the reason the frames were re-downloaded rather than reused.

`fix_sg_original_fps.sh` was verified by **frame count** — and by that test it passed, 66/66. The
table above says "count mismatch: 0" and that is true. **Count parity does not imply index
alignment**, which is the property that actually matters, and this section said so two paragraphs
earlier without then testing for it.

Pixel comparison against `viz` at the same index (mean absolute difference, middle frame of each
stem; the published marker-free build is the reference for what "correct" looks like):

| | stems | mean MAD vs `viz` |
|---|---|---|
| locally repaired `original`, stems that changed | 14 | **13.70** |
| published build, same stems | 14 | **3.69** |

3.69 is JPEG-noise level — same moment, marker gone. **13.70 is not.** 13 of the 14 changed stems
sat further from `viz` than the published build, the worst by a wide margin:

| stem | published | local repair |
|---|---|---|
| `z045-june-24-22-gopro` | 3.18 | **26.34** |
| `z176-sep-05-22-rashult_disassemble` | 3.76 | **26.31** |
| `z168-sep-01-22-espresso` | 3.66 | **24.75** |
| `R206-11Nov-ATV` | 4.48 | **15.94** |
| `R196-25Oct-RAM` | 3.69 | **14.10** |

A MAD of 26 against a 3-ish noise floor is unrelated footage. The other 52 stems are **byte-identical**
between the two builds — exactly the P-prefixed CFR recordings that never needed retiming. Every
divergent stem is an `R###`/`z###` recording, which is the same population the fps defect hit.

**Consequence.** Had the overlay-free retrain launched on the locally repaired frames, 13 holoassist
training videos would have fed the student frames from the wrong moments while the teacher read the
right ones — the precise silent corruption §7.4 exists to prevent, undetectable in an egtea eval
because holoassist is training-only.

**Resolution.** `frames/holoassist/original` now holds the published build
(`Peanuttoad/gaze_dataset_full` → `StreamGaze_v2/frames_shards_holoassist_original`, 8 tars / 29.09 GB),
verified before swapping: 66 stems, **642,515 jpgs**, per-stem count parity with `viz` 66/66, zero
stems missing. The old tree is retained as `frames/holoassist/original.OLD_local_repair` — **it should
not be used**, and any result produced from it is suspect.

**Rule this generalises to:** when two frame trees must be index-aligned, verifying counts is not
verifying alignment. Compare pixels at a shared index on at least the stems that were transformed.

### 7.4b egoexolearn was worse — 54/180 stems, and the verifier could never have caught it

Found 2026-07-28 by applying §7.4a's pixel test to the datasets it had not been run on.
**egtea is clean (35/35, MAD ≤ 3.5), so no published SG number is affected** — but egoexolearn is
StreamGaze *training* data, and 54 of its 180 stems held frames from the wrong moments, MAD up to
68 against a ~3 noise floor. Visually they are unrelated footage: at the same index one shows
writing in a notebook, the other handling flasks at a bench.

**Why the check was vacuous, not merely weak.** `extract_sg_original_frames.sh` computes its
retiming rate as `nb_frames × 10 / n_viz_jpgs` — derived *from the viz count*. So the output is
forced to have exactly as many frames as viz **whatever the timing does**. The script then
verified... the frame count. It constructs the property it checks, so the check cannot fail. 52 of
the 54 broken stems had counts matching `viz` exactly.

**Root cause.** The retiming was applied as `ffmpeg -r RATE -i in.mp4`, an *input* option that
reinterprets container timestamps. That is a no-op on constant-rate input, which is why holoassist
mostly survived it. egoexolearn's originals declare a bogus `r_frame_rate` (`235/12` = 19.58 fps)
while their true average matches viz (22.30 fps) — effectively VFR. Forcing an input rate on a VFR
stream re-spaces every frame.

Measured on `beeabf86-…`, mean MAD vs `viz` across the clip:

| method | MAD | |
|---|---|---|
| `-r RATE -i` (what shipped) | **46.98** | different footage |
| `setpts=N/RATE/TB,fps=10` | **2.80** | same moment, marker gone |

`setpts` rebuilds timestamps from the frame **index**, so it is immune to whatever the container
claims. It is what the published holoassist build used, which is why that build was clean.

**Resolution.** All 54 stems re-extracted with `setpts` and swapped in; the old tree is kept as
`frames/egoexolearn/original.OLD_broken` and must not be used. Whole-dataset pixel verification now
passes everywhere:

| dataset | stems | aligned | median MAD |
|---|---|---|---|
| egtea (eval) | 35 | **35** | 3.26 |
| holoassist (train) | 66 | **66** | 2.98 |
| egoexolearn (train) | 180 | **180** | 2.89 |

**Impact had this not been caught.** 739 of 5799 SG training items — **12.7%**, and 18.8% of the
egoexolearn portion — would have shown the student one moment while the teacher scored another.
Worst hit was `past_object_transition_prediction` at 34% of its items. It is training-only data, so
an egtea eval would have reported nothing wrong. A run was in fact started on the bad frames at
11:18 and killed at step 2140/2900 once this was found.

**Tooling added.** `scripts/verify_frame_alignment.py` (per-stem pixel check),
`scripts/refix_egoexolearn_original.sh` (the repair), `scripts/make_sanity_check_figures.py` plus
`docs/sanity_check/` (side-by-side figures for all six dataset variants).
`extract_sg_original_frames.sh` now uses `setpts`, refuses to skip on a count match, and runs the
pixel verifier before declaring success.

**A note on thresholds.** A bare MAD cut is not enough: fast-moving clips read 8–12 while correctly
aligned. The discriminating signal is *shape* — for an aligned stem MAD(k) is a sharp V centred on
k=0 (11.7 at k=0 vs 34.3 one frame away); a misaligned stem is flat and high across the window
(53.4 / 49.3 / 52.2 / 53.1 / 53.2). The verifier tests for that V.

---

### 7.7 Both overlay-free students, measured 2026-07-28

The runs §7.3 was set up for. Trained with `VLM_GAZE_OVERLAY=0 GAZE_OVERLAY=1` — student pixels
have no marker, teacher TAS stream keeps it — on the **repaired frame trees** (§7.4a, §7.4b);
these are the first runs whose `original` frames are pixel-verified against `viz`.
Logs `kd_train_sgonly_nooverlay.log`, `kd_train_egonly_nooverlay.log`.

| student | best ep | Avg | items | overlay counterpart | Δ |
|---|---|---|---|---|---|
| SG overlay-free | 1 of 2 | 68.44 | **360** | 369 (§2.2a) | **−9** |
| EG overlay-free | 2 of 2 | 55.26 | **268** | 272 (§10.3) | **−4** |

**SG, per task** — teacher column is §8's 3-run mean.

| task | n | overlay (§2.2a) | overlay-free | Δ items |
|---|---|---|---|---|
| **GSM** | 64 | 76.56 (49) | 70.31 (45) | **−4** |
| NFI | 68 | 70.59 (48) | 67.65 (46) | −2 |
| OTP | 2 | 50.00 (1) | 50.00 (1) | 0 |
| SR | 37 | 56.76 (21) | 51.35 (19) | −2 |
| OAR | 96 | 91.67 (88) | 89.58 (86) | −2 |
| **OI-E** | 101 | 69.31 (70) | 71.29 (72) | **+2** |
| **OI-H** | 64 | 65.62 (42) | 67.19 (43) | **+1** |
| FAP | 94 | 53.19 (50) | 51.06 (48) | −2 |
| **Avg** | 526 | 70.15 (369) | **68.44 (360)** | **−9** |

§2.2a predicted that if the overlay were carrying the GSM/NFI gains, those columns should fall
hardest. **GSM is the single largest drop.** The reverse is also informative: object
identification *improves* without the marker (OI-E +2, OI-H +1) — for those tasks the marker is
an occluder, not a cue.

**EG, per qa_type.**

| qa_type | n | overlay (§10.3) | overlay-free | Δ items |
|---|---|---|---|---|
| Spat. | 163 | 40.49 (66) | 42.94 (70) | **+4** |
| **Temp.** | 160 | 42.50 (68) | 36.88 (59) | **−9** |
| Caus. | 162 | 85.19 (138) | 85.80 (139) | +1 |
| **Avg** | 485 | 56.08 (272) | **55.26 (268)** | **−4** |

**§11's prediction 1 holds**: EG leans on the overlay less than SG (4 items vs 9).

**And it overturns §10.2.** That section read the EG student's whole +9 over its teacher as
temporal, and attributed it to dropping the trajectory complement. Removing the overlay drops
temporal to **59 — the teacher's own level (58)**. The temporal advantage was overlay-driven,
not complement-driven. See §10.2's correction note and §10.4.

**Caveats.**
- **Best-of-2 is inside the noise on SG**: ep1 360, ep2 358. §8's floor is 4–5 items, so the
  choice between them is arbitrary and 359 is the honest figure. §8 also warns that best-of-N
  is an upward-biased estimator.
- Single runs. Of the deltas above only SG's GSM (−4) and EG's temporal (−9) clear the noise
  floor; the ±1–2 entries are not measurements.
- Row for the paper table (Overall = (360+268)/1011 = 62.12%):

```latex
KD (raw video) & 70.31 & 67.65 & 51.35 & 89.58 & 71.29 & 67.19 & 51.06 & 68.44 & 42.94 & 36.88 & 85.80 & 55.26 & 62.12 \\
```

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

### In progress (2026-07-28 02:07) — Table 6/7 ablation, `scripts/run_ablation_tab6_tab7.sh`

A workstream that post-dates this document's first revision. §9 previously read "*Nothing running.*"
— that was true at 22:45 and false 17 minutes later. GPUs are 2, so rows run **serially**.

| row | state | ETA |
|---|---|---|
| `tab7_nospatial` | trained 2900/2900 (`kept=9.4%`), **in eval** | ~02:25 |
| `tab7_notemporal` | queued | ~04:20 |
| `tab6_scoreonly` | queued | ~06:15 |
| `tab6_nopretrain` (2nd retry) | queued | ~08:10 |

- **Stage-1 `score-only` encoder: done.** 100/100 epochs, `exit=0`, best loss 0.0168 at epoch 13,
  `TrajGaze_v2/checkpoints/stage1_scoreonly_overlay/best.pth` (147 MB). This unblocks
  `tab6_scoreonly`.
- **Stage-2: 0 of 4 rows have produced a checkpoint.** `ablation_table6_7.md` is a spec with every
  result cell still empty.
- `kept=9.4%` held to the last step, matching the **9.38%** realized budget `ablation_table6_7.md`
  §0a predicts for `no_spatial` — evidence the geometry branch actually engaged.
- **`tab6_nopretrain` has now failed twice.** Run 1 (22:33–23:59) reached step 2280/2900 and was
  killed by the container re-provisioning (§13). Retry 1 died **22 seconds in** at 00:22:43,
  `exit=1` (rank 1 exitcode 1, rank 0 SIGTERM, no traceback captured). If the 2nd retry dies the
  same way, suspect the `--random-encoder` path itself rather than the environment.

### Completed 2026-07-28 (was "queued behind the ablation")

- [x] **Table 6/7 ablation — all 4 rows.** `tab7_nospatial` 62.93, `tab7_notemporal` 67.30,
      `tab6_scoreonly` 66.92, `tab6_nopretrain` 65.02 (3rd attempt; the first two died to the
      reprovision and to an `exit=1` 22 s in). Both tables assemble cleanly via
      `scripts/collect_ablation_tab6_tab7.py` and are monotone in the expected direction —
      No pretrain 64.55 < Only score loss 65.83 < All losses 69.97, and No spatial 61.88 <
      No temporal 66.19 < Spatio-temporal 69.97. Written up separately in `ablation_table6_7.md`.
- [x] **holoassist `original` replaced** with the published build and pixel-verified (§7.4a) —
      the local repair had left 13/66 stems on the wrong moments.
- [x] **egoexolearn `original` repaired** — 54/180 stems re-extracted with `setpts` (§7.4b).
      All three SG trees now pass per-stem pixel verification.
- [x] **SG overlay-free student** — 68.44 / **360 items**, beats the 354 bar (§7.7).
- [x] **EG overlay-free student** — 55.26 / **268 items** (§7.7, §11), launched automatically
      by `scripts/chain_kd_sg_then_eg.sh`.
- [x] **§7.2 resolved**: the overlay is worth **9 items**, not the 15-item upper bound.
- [x] **§10.2 retracted and replaced** by §10.4 — the EG student's margin over its teacher is a
      training-budget artefact, not a method effect.

### Nothing is queued or running

All GPU work in this document is finished. The historical note worth keeping: the SG overlay-free
retrain was attempted three times before it produced a checkpoint —

| attempt | reached | ended by |
|---|---|---|
| 2026-07-27 18:05 | step 480/2900 | SIGTERM at 18:24 |
| 2026-07-28 11:18 | step 2140/2900 | killed once §7.4b was found — it was on the bad egoexolearn frames |
| 2026-07-28 12:58 | complete | `exit=0` at 17:09, **360 items** |

Only the third ran on pixel-verified frames. Attempts 1 and 2 produced no weights.

### To do (not scheduled)

- [ ] **Retrain the EG-only teacher at equal budget** (4 epochs / 632 optimizer steps, ~1 h) and
      re-compare. Until this exists, §10.1's "student beats teacher" is confounded by the
      warm-start (§10.4), and the defensible claim is *matches* the teacher without gaze.
- [ ] Repeat every student eval ≥3× before publishing any student number (§8); every student
      figure in this document is a single run, and SG's best-of-2 (360 vs 358) is inside the
      noise floor.
- [ ] Paste the user-held SG-only VisionZip value into §2.1's empty cell (§5 item 3).
- [ ] `--seed` and global RNG seeding (§5 item 5) — still absent from all three trainers.
- [ ] Clean up the retained backups once nothing needs them:
      `frames/holoassist/original.OLD_local_repair`, `frames/egoexolearn/original.OLD_broken`,
      and the `_dl_holoassist_original` / `_refix_egoexolearn` / `_probe` staging dirs (~85 GB).

### Explicitly dropped

- No-overlay **VisionZip** bar — user decision.
- SG-only VisionZip bar (§2.1) — **will not be re-run.** The local attempt stopped at step
  2580/2900 of epoch 1 with no checkpoint, but the user holds this measurement outside the repo.
  §2.1's cell is nonetheless still empty, so **"+8 over VisionZip" stays unquotable from this
  document** until that value is transcribed in.
- **EG overlay specialist student — reuse, do not retrain.** `visionzip_kd_selection_EGonly_overlay/best.pth`
  (56.08% / 272 items, §10.3) stands as the EG overlay row. User decision, 2026-07-28.
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
| EG specialist KD student (**this machine**, §10.3) | no | 40.49 (66) | 42.50 (68) | 85.19 (138) | 56.08 (272) |
| **M1 EG-only teacher** (mean of 3) | **yes** | 39.88 (65) | 36.25 (58) | **86.11 (139.5)** | 54.02 (262) |
| joint KD student (v1 §6.3) | no | 38.04 (62) | 36.25 (58) | 84.57 (137) | 52.99 (257) |

> **Provenance correction.** The EG specialist student row was previously labelled "(v1 §6.3)", i.e.
> attributed to machine 1. It is not: `kd_train_egonly.log` (this machine, 2026-07-27, epoch 2)
> reproduces all four values **to two decimals**. The row is a machine-2 measurement — see §10.3.
> The joint-student row genuinely is v1's and stays labelled as such.

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

> **RETRACTED 2026-07-28.** This section used to conclude, from v1 §7.5, that "the trajectory
> complement actively hurts EG's temporal questions, and dropping it is a gain rather than a
> loss." **That inverts what §7.5 measured**, and two further results contradict it. The
> paragraph is kept below, struck through, because §1 and §12 cite this reading.
>
> ~~Read with v1 §7.5 (EG rises as complement replaces content, SG falls), the reading is that
> the trajectory complement actively hurts EG's temporal questions, and dropping it is a gain
> rather than a loss.~~
>
> **1. v1 §7.5 says the opposite.** Its sweep holds the 10% budget fixed and varies the
> content∶complement split. On EG, *more* complement is *better*:
>
> | split | EG items |
> |---|---|
> | 8/2 (least complement) | 258 |
> | 7/3 (default) | 269 |
> | **6/4** | **271** |
> | 5/5 | 269 |
>
> and its own summary line reads **"The complement helps EG and hurts SG."** So dropping the
> complement should cost EG, not gain it.
>
> **2. The student never drops the complement anyway.** The budget stays 7/3. What changes is
> *who chooses* the 3% — the gaze/hand field or the RGB predictor (§12.3).
>
> **3. The temporal margin was the overlay.** §7.7 retrained the EG student without the pixel
> marker: temporal falls 68 → **59**, i.e. back to the teacher's 58. The +10 was not the
> complement being dropped; it was the marker being visible.
>
> The surviving explanation is §10.4's — the student is the teacher plus 2 more epochs.

Caveats that still apply to the +9 as measured:

1. The two rows come from **different eval paths** — the teacher via `--eval-ckpt`, the student
   from the training loop — and the student is a single run (the teacher is now a mean of 3).
2. The teacher checkpoint is probably the ep2 snapshot rather than the best epoch (§2.3), which
   would depress every teacher column by an unknown amount.
3. §10.1b shows the joint *teacher* reaches the same temporal level (69 items) with gaze fully
   available, so "more optimisation steps help temporal" fits all four systems at once and is
   the simpler reading. §10.4 makes that quantitative.

Also worth noting for the paper: with 5 options, chance is 20%. The teacher clears it by ~4× on
causal (86.42) but only ~2× on spatial and temporal (39.88 / 36.25). EG's difficulty is
concentrated in exactly the two columns where the gaze complement does not help.

### 10.3 EG overlay student, per task — measured here, not inherited from v1

`visionzip_kd_selection_EGonly_overlay/best.pth` — `56.08% (n=485)`, best = epoch 2, trained **on
this machine 2026-07-27**, log `kd_train_egonly.log:163-168`. Item counts sum exactly to the total
(66+68+138 = 272). Teacher column is §10's mean of 3.

| qa_type | n | student % | items | teacher (mean of 3) | Δ items |
|---|---|---|---|---|---|
| Spatial | 163 | 40.49 | 66 | 39.88 | +1 |
| **Temporal** | 160 | **42.50** | **68** | 36.25 | **+10** |
| Causal | 162 | 85.19 | 138 | 86.11 | −2 |
| **Avg** | **485** | **56.08** | **272** | 54.02 | **+9** |

This is the same profile §10.2 derived, now confirmed from the local training log rather than from
v1: the whole +9 is **temporal**, spatial and causal are flat within noise.

**It also removes §10.2's caveat 1.** That caveat warned the teacher and student rows came from
different eval paths and machines. The student row is machine-2, from the training loop; the teacher
row is machine-2, via `--eval-ckpt`. The eval-path difference stands; the machine difference does
not. Caveats 2 (teacher may be the ep2 snapshot) and 3 (the "more optimisation steps" reading) are
unaffected — and §10.4 now makes 3 quantitative.

### 10.4 "The EG student beats its teacher" is mostly a training-budget artefact

Do not put this claim in the paper as written. The student is **warm-started from the teacher and
then trained further**, so the comparison is not like-for-like.

`kd_train_egonly.log:10` and `kd_train_egonly_nooverlay.log:11`, both runs:

```
[KD] warm-started LoRA from .../visionzip_complement_learned_EGonly_overlay/best.pth
     (missing=0 unexpected=0)
```

That is `$M1_EGONLY` — the exact checkpoint whose score is the "teacher" row. `missing=0
unexpected=0` confirms the whole teacher LoRA was inherited, so this is not a partial load.

EG train is 1265 items; at eff-batch 8 that is **158 optimizer steps per epoch** (the log's
`step 620/633` is per-rank micro-steps ÷ grad-accum 4).

| | optimizer steps on the LoRA |
|---|---|
| EG-only teacher | 2 epochs = **316** |
| EG student | inherits 316, adds 2 epochs = **632 cumulative** |

So "student 272 > teacher 262" compares **632 steps against 316**. Three things make the
optimisation reading the strong one:

1. **The overlay-free student also wins (+6) with strictly less information than the teacher** —
   no gaze coordinates *and* no marker in the pixels, while the teacher has both (§7.7). An
   information-based explanation cannot produce that; a training-budget one can.
2. **The teacher checkpoint is probably ep2, not its best epoch** (§2.3). Against a best-epoch
   teacher (54.85 ≈ 266 items) the margins shrink to +6 (overlay) and +2 (overlay-free).
3. **SG shows the opposite sign.** With the same warm-start advantage the SG student lands
   *below* its teacher (369 vs 375). SG gets 725 steps/epoch, so its teacher was already well
   optimised and the extra epochs buy little. The effect appears only on the benchmark that is
   most undertrained — which is what §10.1b predicted.

The repo already flagged this pattern for a different experiment —
`scripts/run_kd_experiments.sh:23-24`: *"this stacks epochs on top of an already-trained student,
so any gain is NOT attributable to the flag alone."* The same caveat belongs on §10.1's teacher
comparison and was missing.

**Note this is not a KD violation.** There are two distinct "teachers": the *distilled* one is the
frozen TAS encoder (`--stage1-ckpt`), which produces the BCE labels; the M1 checkpoint only
supplies the LoRA initialisation and is never distilled (v1 §3 states this). Warm-starting from
the privileged model is standard in LUPI settings, and v1 §5-4 lists it as an improvement. What is
not defensible is *comparing against that same checkpoint* and calling the difference a method
effect.

**The clean experiment**, and it is cheap: retrain the EG-only teacher to 4 epochs (632 steps) and
re-compare at equal budget. ~1 h at 29 min/epoch. §10.1b already framed the prediction — watch
whether the temporal column climbs toward 68. Until then the defensible claim is
**"matches the teacher without gaze at test time"**, not "beats it".

---

## 11. EgoGazeVQA specialist student — DONE 2026-07-28

Both EG students now exist. The overlay one (§10.3) is reused rather than retrained (user
decision); the overlay-free one was run 2026-07-28 and is reported in **§7.7**.

| EG student | frames | Avg | items |
|---|---|---|---|
| overlay (`gaze`), §10.3 | marker present | 56.08 | 272 |
| **overlay-free (`no_gaze`)** | marker removed | **55.26** | **268** |

Log `kd_train_egonly_nooverlay.log`, `exit=0` at 18:35:27, best = epoch 2. The §7.3 assertion
fired as required: `[KD] frame streams: student VLM='no_gaze'  teacher TAS='gaze'`.

**Both predictions this section recorded in advance were testable, and both resolved:**

1. *"The EG overlay should cost less than SG's 15 items."* — **Held.** EG pays 4 items, SG 9
   (§7.7). EG leans on the marker less, as v1 §7.5's opposite budget preferences suggested.
2. *"The warm-start is overlay-trained, so some of any drop is distribution shift."* — **Confirmed
   on SG**, where retraining recovered 6 of 15 items (§7.2). EG's drop is too small (4 items,
   at the noise floor) to decompose the same way.

The run was launched automatically after the SG one by `scripts/chain_kd_sg_then_eg.sh`, which
gates on the SG *process* exiting rather than on GPU memory — the SG job briefly frees memory
between its training loop and its end-of-epoch eval, and the launcher's own `wait_gpu` would
start EG on top of it.

The setup notes below are kept for reference.

What exists today is the *overlay-trained* EG student, 56.08 / 272 items (§2.3, **per-task in
§10.3**) — trained and evaluated on `gaze` frames, i.e. with the marker in the pixels. It is the EG
analogue of the SG student's 369, and the same criticism applies: it is not gaze-free in the sense
§7 means. **It is being reused, not retrained** (user decision, §9 "Explicitly dropped"), so §10.3
is the fixed baseline this run is measured against.

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

---

## 12. Checkpoint anatomy and facts for the paper

Established by opening the shipped checkpoints directly (`torch.load(..., mmap=True)`), not by
reading the code. Recorded because three of these contradict what is currently written down.

### 12.1 What a "student checkpoint" actually contains

`visionzip_kd_selection_SGonly_overlay/best.pth` — keys `{acc, epoch, lora_state, pred_state}`,
`epoch=2`, `acc=70.15`:

| key | tensors | params | what it is |
|---|---|---|---|
| `pred_state` | 16 | **3.95 M** | the distilled student head |
| `lora_state` → LoRA | 224 | 10.09 M | rank-16 adapters on `q/k/v/o_proj` |
| `lora_state` → base | 729 | **8.29 B** | **the entire frozen backbone** |

**`lora_state` is a misnomer: it holds the whole 7B model**, because `state_dict()` on a
`PeftModel` returns base weights too. That is the 16.6 GB, not optimizer state — v1 §1's
"~16.6 GB (LoRA state + optimizer)" is wrong on both counts; there is no optimizer state in the
file. Only **14 M** parameters are ever trained, so a weights-only export is **~30 MB**. Every
epoch checkpoint currently re-saves the frozen backbone.

### 12.2 LoRA configuration — the `r{=}64` in the draft is from a different model line

The checkpoint tensors settle it: `layers.0.self_attn.q_proj.lora_A.default.weight` is
`(16, 3584)`, i.e. **$r{=}16$**, and `models/model.py` sets `LORA_RANK=16, LORA_ALPHA=32`.
`r=64, α=128` appears only in the **PLLaVA** line (`train_pllava_trajgaze_kd_r64_v3.py`,
`eval_pllava_*`). If Qwen2.5-VL numbers are being reported, the Stage-2 paragraph must say
$r{=}16,\alpha{=}32$.

Also `target_modules=["q_proj","k_proj","v_proj","o_proj"]` — the **feed-forward sub-layers are
not adapted**. "all attention projection layers" is true but reads as if everything is covered;
state the MLP exclusion explicitly.

### 12.3 The student, precisely

`TrajSaliencePredictor` is a **standalone module**, not a head inside the LLM. Its `in_dim` is
read from the backbone only to match width.

```
tok_proj : LN(3584) → Linear(3584→512) → GELU
attn_proj: Linear(1→512)          ← per-clip standardized ViT importance
ctx_proj : LN(3584) → Linear(3584→512) → GELU   ← per-frame mean embedding, O(N)
head     : LN(512) → Linear(512→512) → GELU → Linear(512→1)
```

`attn_proj` having shape `(512, 1)` is a **verifiable claim for reviewers**: the module has no
input path wide enough for a 2-D gaze coordinate or the 15-channel trajectory tensor. It cannot
consume gaze even in principle.

Its output selects *which* tokens reach the VLM; it never contributes features to the forward
pass. The frozen TAS encoder (35.8 M) is called only in the training loop; `evaluate()` never
touches it — though it is still *loaded* before the `--eval-ckpt` branch, so `--stage1-ckpt`
is required even for a gaze-free eval. Harmless, but confusing when demonstrating the claim.

### 12.3a What the deployed model actually costs — measured, not estimated

Counted from the checkpoints directly (`torch.load(..., map_location="cpu")`), 2026-07-28.

| component | params | trained | **at inference** |
|---|---|---|---|
| Qwen2.5-VL-7B backbone | 8.29 B | frozen | ✅ |
| LoRA (r=16, q/k/v/o) | 10.09 M | ✅ | ✅ |
| **`TrajSaliencePredictor`** | **3.95 M** | ✅ | ✅ |
| TAS Stage-1 encoder (the teacher) | **36.85 M** | frozen | ❌ **train-only** |

Trainable total is **14.04 M — 0.17% of the system.**

Inside the teacher, most of the mass is a generic vision backbone, not trajectory machinery:

| submodule | params | share |
|---|---|---|
| `visual_encoder.dino` | 22.06 M | 59.9% |
| `encoder.inter_frame` | 4.74 M | 12.9% |
| `score_decoder.decoder` | 3.26 M | 8.8% |
| `traj_decoder.decoder` | 3.20 M | 8.7% |
| `query_encoder.embedding` | 1.05 M | 2.8% |
| `encoder.pe` (buffer) | 1.05 M | 2.8% |

§12.3's "35.8 M" and the 36.85 M above are the same number: 36,852,576 − 1,048,576
(`encoder.pe`, a buffer rather than a parameter) = **35,804,000**. Quote whichever, but say which.

**The point is the inference column.** `evaluate()` never calls `_traj_scores` or the TAS encoder,
so what the method removes at deployment is the eye-tracker *and* a 36.85 M encoder — dominated by
running DINOv2 over every frame. What it adds is one **3.95 M** head (15.8 MB fp32) that reuses
`video_embeds` and `attn_scores` already computed for VisionZip, in O(N). No frame is re-encoded.

`topk_in_avail` is a function, not a module — **zero parameters** (`torch.topk` plus indexing).

Inside the student, the two 3584→512 projections are essentially all of it:

| | params | share |
|---|---|---|
| `tok_proj` | 1,842,688 | 46.6% |
| `ctx_proj` | 1,842,688 | 46.6% |
| `head` | 264,193 | 6.7% |
| `attn_proj` | **1,024** | 0.0% |

`attn_proj` being `Linear(1→512)` is the reviewer-checkable claim from §12.3 — 1,024 parameters,
scalar input, no path wide enough for a 2-D gaze coordinate.

### 12.4 Rows prepared for the results table

Teacher = **mean of the 3 per-task runs** (§8), not a single run and not the best of N:

```latex
\sys-T (SG-only teacher) & 71.36 & 63.24 & 57.66 & 93.40 & 73.27 & 74.48 & 56.38 & 71.17 \\
\sys + KD (gaze-overlay) & 76.56 & 70.59 & 56.76 & 91.67 & 69.31 & 65.62 & 53.19 & 70.15 \\
```

Column order GSM / NFI / SR / OAR / OI-E / OI-H / FAP / Avg. `past_object_transition_prediction`
has **2 items** and no column, but is inside every Avg — state that in the caption.

Against the draft's visible rows the KD student is best on **GSM (+1.56 over \sys)** and **NFI
(+2.94 over PruneVid)** — the two gaze-driven tasks, which is the story. It is *not* best on
OI-E/OI-H (Full-token leads by ~10) or Avg.

**Both margins are inside the measured noise.** §8 shows NFI swinging 2.95 and SR 2.70 across
re-evaluations of identical weights. The student row is a **single run**; the teacher row is a
3-run mean. Before bolding anything, evaluate the student ≥3× as well.

### 12.5 Open discrepancies to resolve before submission

1. Student rows are single runs (§8 requires ≥3). SG's best-of-2 spread is 2 items — inside the
   noise floor — so which epoch is "best" is arbitrary there.
2. Machine-1 vs machine-2 mixing — the draft's baselines were measured elsewhere; the same
   VisionZip checkpoint reads GSM 65.62 there and 70.31 here.
3. §2.1 still unresolved: the SG-only VisionZip cell is empty, so "+8 over VisionZip" is
   unquotable from this document until the user's held value is transcribed in (§5 item 3).
4. **Do not write "the gaze-free student beats its gaze-using teacher."** It is true only on EG
   and only against a teacher trained for half the optimizer steps (§10.4). Either retrain the
   teacher at equal budget or phrase it as *matching* the teacher without gaze at test time.
5. **"gaze-free" needs qualifying wherever §2/§10's numbers appear.** Those rows are `viz`/`gaze`
   frames with the marker in the pixels; only the trajectory-coordinate stream is removed (§7).
   The genuinely marker-free rows are in §7.7. Label them distinctly in the paper — e.g.
   `KD (gaze-overlay)` vs `KD (raw video)`.

---

## 13. Environment migration, 2026-07-28 — paths moved, nothing was lost

Recorded because it killed a run mid-epoch and because the current fix is one symlink deep.

### 13.1 What happened

At **00:01** the container was re-provisioned and the storage root changed:

```
/NHNHOME/VILAB/vilab_yj/…   →   /NHNHOME/WORKSPACE/26msit001_A/vilab_yj/…
```

`env.sh` hardcodes the old prefix in four places (`REPO`, `DATA`, `PATH`, `HF_HOME`), the three
`visionzip_complement_learned_*` checkpoint symlinks and `stage1_tas_3way_overlay` all resolve
through it, and ~13 launcher scripts open with a literal `cd /NHNHOME/VILAB/vilab_yj/trajgaze`.
For ~20 minutes nothing in the repo could launch.

At **00:21** a compatibility symlink was created:

```
/NHNHOME/VILAB -> /NHNHOME/WORKSPACE/26msit001_A
```

That restored every path at once. Verified 02:07: `$STAGE1_CKPT` 147,568,266 B, `$M1_SGONLY` and
`$M1_EGONLY` 16,625,163,681 B each, `$SG_ROOT`/`$EG_ROOT` present, `which python` → the venv,
torch 2.11.0+cu128 / peft 0.15.1, 2 GPUs visible. **No source edits were needed and none were made.**

### 13.2 No data was lost

| asset | check | result |
|---|---|---|
| SG frames, `viz` ↔ `original` | stem counts per dataset | egtea 35/35, holoassist 66/66, egoexolearn 180/180 |
| holoassist fps repair (§7.4) | per-stem `viz` vs `original` jpg counts | match on spot-check |
| EG frames, `gaze` ↔ `no_gaze` | stem counts | egtea 82, ego4d 27, egoexo 154 — parity |
| teacher / stage-1 checkpoints | size + resolve | all 4 intact |
| venv, HF cache (Qwen2.5-VL-7B) | import + list | intact |

### 13.3 The fix is fragile — state it as a known risk

Everything currently works **because of a single symlink**. If it is removed, the same total outage
returns. The durable fix is four lines in `env.sh`, deriving the roots from the file's own location
instead of hardcoding them:

```bash
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VJ="$(dirname "$REPO")"
export DATA="$_VJ/datasets/trajgazemerge"
export PATH="$_VJ/envs/trajgaze/bin:$PATH"
export HF_HOME="$_VJ/.cache/huggingface"
```

The launcher scripts would still need `cd "$(dirname "$0")/.."`, which
`scripts/run_ablation_tab6_tab7.sh:16` already uses. Not done — deliberately out of scope while runs
are in flight.

### 13.4 Three long runs have been lost to interruption

All three died with **no checkpoint**, because the trainers only save at epoch end. A 1-epoch
ablation row that dies at 79% leaves nothing at all.

| run | reached | cause |
|---|---|---|
| SG-only VisionZip bar | 2580/2900, epoch 1/2 | hard kill, no signal recorded |
| SG KD student, no-overlay | 480/2900, epoch 1/2 | SIGTERM 18:24:41 |
| `tab6_nopretrain` | 2280/2900, epoch 1/1 | container re-provisioning, 00:01 |

Mitigations for anything long-running from here: launch **detached under tmux** (`/usr/bin/tmux` is
available) so a terminal or SSH loss cannot take the job with it; use the KD trainer's `--resume`,
which restores from the newest `epoch_*.pth` and therefore saves epoch 1 if epoch 2 dies. Neither
helps a death inside epoch 1 — only periodic mid-epoch checkpointing would, and that is not
implemented.
