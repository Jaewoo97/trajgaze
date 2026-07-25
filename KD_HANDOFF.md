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

**Current result** (only ep1 captured — training was stopped early to free GPUs):
- **Overall 62.31%** (n=1011) — **SG 68.06** / **EG 56.08**.
- vs teachers: joint M1 63.01 (SG 69.20/EG 56.29), content-only VisionZip 62.51.
- **Selection agreement ≈ 0.41** = the RGB student recovers ~41% of the gaze/hand top-k
  complement. This is the ceiling lever: the student can only be as good as the selection it
  reconstructs. EG is nearly recovered (56.08 vs 56.29, −0.21); SG loses more (68.06 vs
  69.20, −1.14) because the gaze complement pays off most on SG's gaze-driven task (GSM).

---

## 3. To KD the SEPARATELY-TRAINED specialists (what to change)

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
4. **Warm-start from the matching specialist** (SG student ← SG-only M1) rather than joint M1 —
   removes the train/eval domain gap in the readout.
5. **Per-source students.** Because the complement only helps SG, train an SG student hard and
   let EG fall back to content-only — likely a better SG/EG trade than one joint student.
6. **Finish training.** Only ep1 (62.31) was captured; run full best-of-3 with `--early-stop`
   for the honest student number before iterating.
7. Tune `--lambda-sel` (selection vs task weight) and `--pred-lr`; the two losses are decoupled,
   so they can be scheduled independently.
