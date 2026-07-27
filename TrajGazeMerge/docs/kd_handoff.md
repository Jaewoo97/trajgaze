# KD Handoff — gaze-free distillation of the best per-dataset M1

Goal of this doc: give another machine everything needed to (a) reproduce/serve the
best **separately-trained** M1 teachers on StreamGaze and EgoGazeVQA, and (b) develop
the gaze-free **knowledge-distillation (KD)** student further. Written 2026-07-25.

All numbers below are verified from run logs (`/tmp/spec_*.log`, `/tmp/kd_selection.log`),
best-of-2 epochs, EGTEA eval, multiple-choice accuracy (%). GAZE_OVERLAY=1 for every run.

---

## 1. The teachers — best M1, trained on ONE dataset (specialists)

M1 = `\sys` = complementary token selection at a **10% visual-token budget**:
7% VisionZip content ∪ 3% trajectory complement, where the complement is the top-3% of
the tokens VisionZip *discarded*, ranked by a frozen gaze/hand salience field (TAS Stage-1
encoder). Backbone Qwen2.5-VL-7B is frozen; only a LoRA adapter trains. **M1 needs the
gaze/hand streams at inference** (an eye-tracker at test time) — that is exactly what the
KD student removes.

| Teacher            | Train src | Eval src | best-of-2 | (ep1 / ep2)     | checkpoint (`best.pth`) |
|--------------------|-----------|----------|-----------|-----------------|-------------------------|
| **M1 SG-only**     | SG        | SG (526) | **69.96** | 65.59 / 69.96   | `…/checkpoints/visionzip_complement_learned_SGonly_overlay/best.pth` |
| **M1 EG-only**     | EG        | EG (485) | **54.85** | 54.85 / 53.81   | `…/checkpoints/visionzip_complement_learned_EGonly_overlay/best.pth` |
| M1 joint (ref)     | SG∪EG     | 1011     | 63.01     | SG 69.20/EG 56.29 | `…/checkpoints/visionzip_complement_learned_overlay/best.pth` |

`…` = `/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints`. Each M1 `best.pth` is ~16.6 GB
(LoRA state + optimizer). The joint M1 is the teacher the **current** KD student distills;
the two specialists are the new teachers you want to distill.

> Re-measured on a second machine — see **§6.1**. The finding reproduces (more strongly),
> but the shipped EG-only `best.pth` scores 53.81, not 54.85; use 53.81 as its baseline.

**Reproduce a specialist teacher** (2-GPU, eff-batch 8):
```bash
export GAZE_OVERLAY=1
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"   # NOT the `gaze` env (broken transformers)
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29701 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 --source sg \   # or: --source eg
  --output-dir "$CKPT/visionzip_complement_learned_SGonly_overlay" \
  --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop
```
Key finding from the specialist grid: **joint training helps EG** (joint EG 56.29 >
EG-only 54.85) — cross-dataset signal from SG transfers to EG — while **SG-only helps SG**
(SG-only 69.96 > joint SG 69.20). So the "best per dataset" teacher differs by dataset.

---

## 2. How the CURRENT KD works (baseline to improve)

Trainer: `TrajGazeMerge/training/train_visionzip_kd_lora.py`. It is **privileged-information
selection distillation** — the gaze/hand streams are available at TRAIN time (teacher) and
absent at TEST time (student picks the complement from RGB alone).

**Pieces**
- **Student** = `TrajSaliencePredictor` (RGB-only), a small head over content-side features:
  token embeddings + ViT importance (`attn_scores`) + frame position (`grid_thw`). ~few M params.
- **Teacher field** = frozen TAS Stage-1 encoder → per-token gaze/hand salience via
  `_traj_scores(...)` (imported verbatim from the M1 trainer so the student distills exactly
  M1's selection signal). TRAIN-only.
- **Shared content set**: both teacher-label and student-pick operate over the *discarded*
  (available) tokens = everything VisionZip's 7% content set did not keep.

**Two decoupled objectives** (top-k selection is non-differentiable, so they do NOT share
gradients):
1. `predictor ← selection-KD`: `BCE(student salience, teacher top-k membership)` over the
   available tokens, `pos_weight` balanced (capped 50). This teaches the RGB head *which*
   discarded tokens the gaze/hand field would pick.
2. `LoRA ← task CE`: cross-entropy on the answer, computed on the **student-selected** 10%
   tokens (student's own top-3% complement ∪ 7% content). Top-k is detached.

**Warm-start**: LoRA is initialized from the **joint** M1 `best.pth` so the readout starts
at teacher quality and only has to absorb the student/teacher selection gap.

**Inference is gaze-free**: student salience → own top-3% complement → NO gaze/hand read.
(The gaze *overlay* is still burned into the input frames, same as every baseline; what's
removed is the trajectory-coordinate stream.)

**Command** (4-GPU DDP, eff-batch 8 = 4 GPU × grad-accum 2):
```bash
export GAZE_OVERLAY=1
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29661 \
  -m TrajGazeMerge.training.train_visionzip_kd_lora \
  --warmstart-ckpt …/visionzip_complement_learned_overlay/best.pth \
  --stage1-ckpt    …/stage1_tas_3way_overlay/best.pth \
  --output-dir     …/visionzip_kd_selection_overlay \
  --epochs 3 --lr 1e-4 --pred-lr 1e-3 --grad-accum 2 --no-hdepic --early-stop
```

> Now run to completion on the second machine — see **§6.2**. Honest best-of-3 is **61.33**
> (ep1), i.e. below the joint teacher; ep2 trades SG away for EG.

**Current result** (only ep1 captured — training was stopped early to free GPUs):
- **Overall 62.31%** (n=1011) — **SG 68.06** / **EG 56.08**.
- vs teachers: joint M1 63.01 (SG 69.20/EG 56.29), content-only VisionZip 62.51.
- **Selection agreement ≈ 0.41** = the RGB student recovers ~41% of the gaze/hand top-k
  complement. This is the ceiling lever: the student can only be as good as the selection it
  reconstructs. EG is nearly recovered (56.08 vs 56.29, −0.21); SG loses more (68.06 vs
  69.20, −1.14) because the gaze complement pays off most on SG's gaze-driven task (GSM).

---

## 3. To KD the SEPARATELY-TRAINED specialists (what to change)

> **Done — results in §6.3.** Both specialist students beat their own teachers, and the
> per-source pair (63.40 composite, gaze-free) beats every gaze-using teacher measured.

The teacher salience field is the *same* TAS encoder regardless of teacher — the M1 ckpt
only supplies the LoRA warm-start. So to distill a specialist:
1. `--warmstart-ckpt` → the specialist `best.pth` (SG-only or EG-only) instead of joint M1.
2. `--stage1-ckpt` → unchanged (`stage1_tas_3way_overlay/best.pth`).
3. **Add a `--source {sg,eg}` filter to the KD trainer.** `train_visionzip_kd_lora.py`
   currently builds `CombinedMergeDataset(...)` with **no source filter** (unlike the M1 /
   baseline trainers, which already take `--source`). Port that flag: pass `source=` into the
   `CombinedMergeDataset` calls in both `evaluate()` (line ~164) and `main()`'s train build
   (line ~270). Then train + eval on the matching single source.
4. Match the specialist protocol: `--epochs 2 --early-stop` (best-of-2), eff-batch 8.

Expected story: the gaze complement helps SG (GSM +9–11 in the teacher), so a **SG student**
is where KD is worth it; on EG the complement only ties content-only pruning, so an EG
student may barely need it (an RGB VisionZip-only student may already match).

---

## 4. Port to another machine (deps / data / env)

**Env**: conda env `trajgaze` at `/opt/conda/envs/trajgaze` (the `gaze` env has broken
transformers — do not use). `export GAZE_OVERLAY=1` always.

**`sys.path` roots** the code inserts (see top of the trainer): `/workspace/trajgaze_st`,
`/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL`, `/workspace/EgoGazeVQA/AutoGaze`,
`/workspace/EgoGazeVQA`.

**Checkpoints to copy**
- TAS teacher: `…/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth` (147 MB) — required.
- Specialist teachers: `visionzip_complement_learned_{SGonly,EGonly}_overlay/best.pth` (16.6 GB each).
- Joint M1 (current-KD warm-start / reference): `visionzip_complement_learned_overlay/best.pth` (16.6 GB).
- Backbone: Qwen2.5-VL-7B (the VisionZip fork under `/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL`).

**Data**: `CombinedMergeDataset` (StreamGaze + EgoGazeVQA, EGTEA split). Frames are the
**gaze-overlay** mirror (`frames_gaze`), plus per-frame trajectory streams (gaze + left/right
hand pos/vel + interaction features) for the teacher field. Use `--no-hdepic` for the 2-way
(SG+EG) setup. Eval sizes: SG n=526, EG n=485, combined 1011.

**Key modules** (all under `TrajGazeMerge/`):
`models/traj_salience_predictor.py::TrajSaliencePredictor` (student),
`training/train_visionzip_complement_lora.py::_traj_scores` (teacher field),
`training/train_visionzip_lora.py::{load_visionzip_lora, preprocess_visionzip_item,
visionzip_select_tokens}` (content selection),
`training/train_merge_lora_temporal_no_kd.py::load_traj_encoder` (TAS encoder loader),
`models/model.py::{build_merged_inputs, forward_logits, get_option_ids}` (VLM forward).

**Sanity check without GPU**: `/tmp/smoke_kd.py` (a few-item smoke of the KD trainer).

---

## 5. Ideas to improve the KD (ranked by expected payoff)

1. **Raise selection agreement (currently ~0.41).** This is the hard ceiling. Give the RGB
   student more signal: cross-frame/temporal attention in `TrajSaliencePredictor`, optical
   flow or motion-energy features (a hands/gaze proxy), larger `--pred-hidden`. Higher agree →
   directly higher student accuracy, especially on SG/GSM.
2. **Soft-field distillation, not just top-k membership.** Current KD is BCE on top-k
   membership (hard). Regress the *continuous* teacher salience field (MSE / soft-KL) so the
   student learns the ordering, not just the boundary — usually transfers more.
3. **Add response KD on top of selection KD.** Distill the teacher M1's answer logits (KL) in
   addition to task CE, so the student matches the teacher's decision, not only its token pick.
4. ~~**Warm-start from the matching specialist**~~ — **DONE (§6.3)**, together with 5.
5. ~~**Per-source students.**~~ — **DONE (§6.3), biggest win so far:** +2.07 composite over
   the joint student. 4 and 5 were changed together (specialist warm-start *and* `--source`
   filter), so this experiment cannot say how the +2.07 splits between them — an ablation
   (joint warm-start + `--source sg`) would separate them. Note EG did not need to fall back
   to content-only as predicted: it beat its own teacher outright.
6. ~~**Finish training.**~~ — **DONE (§6.2):** honest joint-student best-of-3 = **61.33**.
7. Tune `--lambda-sel` (selection vs task weight) and `--pred-lr`; the two losses are decoupled,
   so they can be scheduled independently.

---

## 6. Results on the second machine (2026-07-27)

Everything below was produced on the §4 port (2× B200, venv at
`/NHNHOME/VILAB/vilab_yj/envs/trajgaze`, all paths from `env.sh`; checkpoints served from
`datasets/trajgazemerge/aaai/` via symlinks). Same protocol as §1–§3: EGTEA eval,
multiple-choice accuracy, `GAZE_OVERLAY=1`, eff-batch 8. Split sizes verified identical to
§4 — eval SG 526 / EG 485 / 1011, train SG 5799 / EG 1265 / 7064.

### 6.1 Teachers re-evaluated here (§1)

| Teacher (`--eval-ckpt`) | Eval | §1 | here | Δ |
|---|---|---|---|---|
| M1 SG-only  | SG 526  | 69.96 | **71.67** | +1.71 |
| M1 EG-only  | EG 485  | 54.85 | **53.81** | −1.04 |
| M1 joint    | 1011    | 63.01 | **62.81** | −0.20 |
| M1 joint    | SG / EG | 69.20 / 56.29 | 69.39 / 55.67 | +0.19 / −0.62 |

§1's key finding reproduces, and more strongly: SG-only beats joint-on-SG by **+2.28** here
(§1: +0.76), and joint-on-EG beats EG-only by **+1.86** (§1: +1.44). "Best teacher differs by
dataset" holds in both directions.

**Environment deviation is one-sided: EG is systematically lower here, SG is equal or higher.**
Across every comparable run — joint teacher EG −0.62, EG-only teacher −1.04, joint KD student
EG −3.09 (§6.2) — while SG is +0.19 / +1.71 / +0.95. Treat cross-machine EG deltas under
~1 point as environment, and compare EG numbers only within one machine.

**Caveat on the shipped EG-only `best.pth`:** it scores **53.81**, which equals §1's *ep2*
value to two decimals (§1 records ep1 54.85 / ep2 53.81, best-of-2 = 54.85). Matching to
within 1/485 by coincidence is unlikely, so the file in `aaai/` is probably the ep2 snapshot
rather than the best epoch. The SG-only +1.71 is *not* explained this way (there ep2 *is* the
best epoch) and is left as numeric variance (bf16 / flash-attn on B200).

### 6.2 Baseline joint KD, run to completion (§2)

§2 captured only ep1. Full run, `--epochs 3 --early-stop`, 2 GPUs × grad-accum 4:

| epoch | overall (1011) | SG | EG | agree | epoch time |
|---|---|---|---|---|---|
| **1 (best)** | **61.33** | 69.01 | 52.99 | 0.409 | 7727 s |
| 2 | 60.63 | 66.54 | 54.23 | 0.474 | 8195 s |
| 3 | skipped — early stop (60.63 ≤ 61.33) | | | | |

The honest joint-student number is **61.33**, below the joint teacher (62.81 here). Note SG
*drops* 69.01 → 66.54 in ep2 while EG rises 52.99 → 54.23: the single joint student trades SG
away for EG as training continues — exactly the per-source tension §5.5 anticipated.

vs §2's ep1 (62.31 = SG 68.06 / EG 56.08): SG +0.95, EG −3.09 — the one-sided EG deviation of
§6.1, amplified in the student.

### 6.3 Specialist KD (§3) — implemented and run

`--source {sg,eg,both}` ported into `train_visionzip_kd_lora.py` per §3.3. **Mechanism note:**
`CombinedMergeDataset` takes no `source` kwarg — the M1 trainer filters its flat `.items` list
after building, so the KD trainer now does the same, inside `evaluate()` and before
`DistributedSampler` is constructed. Launcher: `scripts/run_kd_specialists.sh` (SG then EG,
crash-retry + `--resume`, one log per specialist so the completion grep can't cross-talk).

| student (gaze-free) | warm-start | train | ep1 | ep2 | best | agree ep1→ep2 | wall |
|---|---|---|---|---|---|---|---|
| **SG specialist** | M1 SG-only | SG 5799 | 66.54 | **70.15** | **70.15** | 0.399 → 0.460 | 4.3 h |
| **EG specialist** | M1 EG-only | EG 1265 | 55.05 | **56.08** | **56.08** | 0.413 → 0.495 | 1.5 h |

**Both students beat their own teacher, and the pair beats every gaze-using teacher measured
here:**

| system | gaze at test | SG 526 | EG 485 | composite 1011 |
|---|---|---|---|---|
| **specialist KD students** (per-source) | **no** | **70.15** | **56.08** | **63.40** |
| M1 specialist teachers (per-source)     | yes | 71.67 | 53.81 | 63.11 |
| M1 joint teacher                        | yes | 69.39 | 55.67 | 62.81 |
| joint KD student (§6.2 best)             | no  | 69.01 | 52.99 | 61.33 |

"composite" = pooled accuracy over all 1011 items using each source's own model. For the
per-source rows that means **two checkpoints, routed by source** — not a single model, so it is
not directly comparable to a joint model's 1011-item number; it is the right comparison only
because the routing input (which benchmark an item came from) is available at test time.

- **SG**: 70.15 is −1.52 vs its own teacher, but **+0.76 above the joint teacher** — gaze-free
  selection recovers enough of the complement to pass a gaze-using model.
- **EG**: 56.08 is **+2.27 above its own teacher** (53.81) and +0.41 above the joint teacher.
  §3's prediction that the complement barely helps on EG holds — the RGB student loses nothing,
  and single-source CE training gains outright.
- **Both students improve ep1 → ep2**, unlike the joint student. Reading ep1 alone *inverts*
  the SG conclusion (66.54 would look like specialist KD hurts, −2.47 vs the joint student), so
  2 epochs is the minimum honest protocol here.
- `agree` still tracks accuracy across runs: EG's 0.495 is the highest of the three runs and EG
  is the only student to clear its teacher by >2 points. §5.1 remains the lever.

Per-task, best epoch, specialist vs joint student: GSM **76.56** vs 75.00, non-fixated-object
**70.59** vs 66.18, attribute-recognition **91.67** vs 90.62, object-id-hard 65.62 vs 68.75;
EG side causal **85.19** vs 84.57, temporal **42.50** vs 36.25, spatial **40.49** vs 38.04.

### 6.4 Artifacts

Under `/NHNHOME/VILAB/vilab_yj/trajgaze`:

- students: `TrajGazeMerge/checkpoints/visionzip_kd_selection_{SGonly,EGonly}_overlay/best.pth`
- logs: `eval_m1_{sgonly,egonly}.log`, `kd_train.log`, `kd_train_{sgonly,egonly}.log`,
  `kd_train_specialists.log`
- launchers: `scripts/run_m1_specialist_evals.sh` (§1, 2 GPUs in parallel, ~25 min),
  `scripts/run_kd_specialists.sh` (§3, SG→EG chained), `scripts/run_kd_train.sh` (§2)
- env: `env.sh` (data roots, `GAZE_OVERLAY=1`, `$TORCHRUN`, ckpt vars);
  smoke: `TrajGazeMerge/scripts/smoke_kd.py`

### 6.5 Next

§5.1 / §5.2 are now the top levers — agreement is only 0.46–0.50 and soft-field distillation is
untested. §5.3 (response KD) is the other untried transfer. Both should be measured **per
source**, against §6.3's 70.15 / 56.08, since §6.3 settles that per-source students dominate.

> **Superseded by §7.** §5.1 is refuted and §5.2/§5.3 are shown to be capped inside the noise
> floor. Do not start from §6.5.

---

## 7. Machine-2 re-measurement and the selection-KD refutation (2026-07-27)

Same port as §6 (2× B200). Everything here is the joint protocol: EGTEA 1011, gaze-free eval,
`GAZE_OVERLAY=1`, 10% budget at 7/3 unless stated. Accuracies are also given as **item counts
out of 1011**, because every effect in this section is smaller than one percentage point and
percentages hide that.

### 7.1 Both machine-1 checkpoints reproduce here

Pulled from HF (`Peanuttoad/gaze_dataset_full`) and re-scored on this machine:

| checkpoint | §2 value | here | Δ items |
|---|---|---|---|
| `visionzip_lora_sgeg_overlay` (content-only VisionZip) | 62.51 | **62.51** (632) | 0 |
| `visionzip_kd_selection_overlay` (KD student, ep1) | 62.31 | **62.41** (631) | +1 |

The bar reproduces exactly; the student to within one item. The port is faithful.

Per source the same student moved **SG +4 / EG −3** (358→362, 272→269) and cancelled out. That
is the numeric noise floor from re-scoring *identical weights* — bf16/flash-attn on different
hardware. **Any per-source difference under ~4 items is not a measurement.**

This also corrects §6.1: the one-sided EG deviation is real but only ~0.6 points (3 items),
matching §6.1's own joint-teacher figure. It is far too small to explain §6.2's −2.47 on EG.

### 7.2 §6.2's 61.33 was a bad training run, not the environment

| student | SG 526 | EG 485 | 1011 |
|---|---|---|---|
| machine-1, re-scored here | 68.82 (362) | 55.46 (269) | **62.41** (631) |
| machine-2 (§6.2 ep1) | 69.01 (363) | 52.99 (257) | 61.33 (620) |

Identical eval, identical protocol: **−11 items, essentially all EG** (SG +1). The log shows
one clean attempt, no crash, no resume. §6.2's number should not be used as the joint-student
baseline; 62.41 is the sound reference.

Note this also removes §6.2's "inverted pattern": on the sound checkpoint the gaps versus the
joint teacher are SG −0.57 / EG −0.21, i.e. §2's original pattern.

### 7.3 The ceiling: privileged information is worth 3 items

| system | gaze at test | items /1011 |
|---|---|---|
| M1 joint teacher | **yes** | 635 |
| content-only VisionZip | no | **632** |
| KD student @ 7/3 | no | 631 |

**Having the gaze/hand streams at inference is worth 3 items over content-only pruning**, and
the gaze-free student already sits 1 item under the bar and 4 under the teacher.

So any distillation whose teacher is the M1 gaze model is capped at **+4 items**, against a
noise floor of 3–4 items (§7.1) and ~11 items across training runs (§7.2). §5.2 (soft-field)
and §5.3 (response KD from M1) are inside the noise *by construction* — not because they are
untuned, but because there is almost nothing at the source to transfer. This is independent of
the mechanism in §7.4.

### 7.4 §5.1 is refuted: higher agreement makes the student worse

`--freeze-lora` run (new flag): LoRA held at the §7.1 student's weights, **only** the
`TrajSaliencePredictor` trained on the selection-KD BCE. Because the readout cannot move, any
accuracy change is attributable to selection alone.

| | agree | SG | EG | 1011 |
|---|---|---|---|---|
| start | ~0.41 | 362 | 269 | 631 |
| after 1 epoch | **0.455** | **354** | 269 | **623** |

Agreement rose and accuracy fell by **8 items, all on SG, with EG exactly unchanged**. §5.1
predicted the opposite and specifically predicted the gain would appear "especially on
SG/GSM". §6.2 hinted at this (agree 0.409→0.474 while accuracy fell 61.33→60.63) but could not
separate a worse selection from a damaged readout; frozen, there is no ambiguity.

### 7.5 Split sweep: SG and EG want opposite budget allocations

Eval-only on the §7.1 student, total budget fixed at 10% (all points verified at 1380 tokens),
varying only the content∶complement division:

| split | SG items | EG items | total |
|---|---|---|---|
| 8/2 | **364** | 258 | 622 |
| 7/3 (M1 default) | 362 | 269 | 631 |
| 6/4 | 361 | **271** | **632** |
| 5/5 | 357 | 269 | 626 |

**SG falls monotonically as complement replaces content; EG rises then saturates.** The
complement helps EG and hurts SG — inverting §3's expected story ("the gaze complement helps
SG… on EG the complement only ties content-only pruning").

6/4 is an interior optimum but beats the 7/3 default by 1 item, i.e. nothing. Its
632/361/271 is identical to the VisionZip baseline on all three figures; the compositions
genuinely differ, so this is presumed coincidence but was not verified per-item.

This gives a *structural* reason the single joint student underperforms, separate from §6.2's
training-schedule story: the two benchmarks disagree about how to spend the budget, and one
global split cannot satisfy both. SG's optimum is at or beyond 8/2, EG's is near 6/4.

### 7.6 Port gap in §4 (affects any VisionZip-side eval)

`TrajGazeMerge/training/train_autogaze_lora.py` hardcoded `/workspace/datasets/StreamGaze_v2/…`
for `FRAMES_BASE`/`QA_BASE`, while `data/dataset.py` had been made env-driven via `SG_ROOT`.
On any other machine every `qa_path` missed and was silently `continue`d, so
`StreamGazeSimpleDataset` returned **0 items** and `CombinedSimpleDataset` scored **EG-only
(485)** while reporting it as the full set. Fixed to use `SG_ROOT`; now 1011 (sg=526, eg=485),
matching `CombinedMergeDataset`. **Any VisionZip-side eval run on a ported machine before this
fix is invalid.**

Also: §6.4 lists the smoke test at `TrajGazeMerge/scripts/smoke_kd.py`; it is at
`scripts/smoke_kd.py`.

### 7.7 Artifacts

Under `/NHNHOME/VILAB/vilab_yj/trajgaze`:

- machine-1 checkpoints: `datasets/trajgazemerge/hf_m1/aaai/{visionzip_lora_sgeg_overlay,
  visionzip_kd_selection_overlay}/best.pth` — kept in a separate `hf_m1/` root because the
  latter collides by name with the local §6.2 run directory.
- logs: `eval_m1bar_{visionzip,kdstudent}.log`, `eval_split_{8_2,6_4,5_5}.log`,
  `kd_train_frozenlora.log`
- new flags in `train_visionzip_kd_lora.py`: `--freeze-lora`, `--balance-sources`
  (epoch-size-preserving, resampled per epoch), and `--warmstart-ckpt` now carries
  `pred_state` when the checkpoint has one, so a *student* checkpoint resumes at its own score
  instead of being paired with a fresh random head.
- `train_visionzip_lora.py` gained `--eval-ckpt` and per-source reporting (it had neither).

### 7.8 Next

Selection distillation from the gaze teacher is closed: refuted mechanically (§7.4) and capped
below noise (§7.3). To improve the gaze-free student at a fixed 10%/7:3 budget, a distillation
target is needed whose teacher is meaningfully better than 635 items — the M1 gaze model is
not. Untested candidates: a teacher at a larger token budget (student unchanged; teacher exists
only at train time), or an ensemble-of-seeds teacher distilled back into one student.

Before any of it, pin the noise floor with seed repeats. Every number in §7 is a single run,
and the effects being chased are 1–4 items.
