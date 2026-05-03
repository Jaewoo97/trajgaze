# Ablation Runbook — E1_patch_temporal

Ablations for the **E1_patch_temporal** model  
(`SpatiotemporalEncoderTemporal` + `use_patch_temporal_branch=True`, gate=0 frozen).

All commands run from `/workspace/EgoGazeVQA/TrajGazeMerge_v2` with  
`PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2`.

---

## Baseline (full E1 model)

Already trained via the main pipeline:

```bash
setsid nohup bash TrajGazeMerge/training/run_e1_patch_temporal.sh &
```

Checkpoints:
- Stage-1: `TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth`
- Stage-2: `TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/`

---

## Ablation overview

| ID  | Name             | What changes                                  | Stage-1 losses           |
|-----|------------------|-----------------------------------------------|--------------------------|
| 1.a | No pretraining   | Skip stage-1 entirely (random encoder)        | —                        |
| 1.b | Score only       | Drop `l_traj + l_score_traj`                  | `l_sp + l_sf`            |
| 2.a | Hand only        | 3 hand tokens, no gaze; patch branch active   | All 4                    |
| 2.b | Gaze only        | 1 gaze token, no hand; patch branch active    | All 4                    |
| 3.a | No spatial       | No `TemporalVisualTrajFusion` (traj-only scores) | All 4 (`l_st`=0)      |
| 3.b | No temporal      | No `patch_temporal_branch`, gate=0 frozen     | All 4                    |

---

## 1.a — No pretraining

No stage-1 script needed. Run stage-2 directly with an absent checkpoint;
the encoder starts from random init.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2 \
python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type  full \
    --stage1-ckpt /nonexistent \
    --output-dir  TrajGazeMerge/checkpoints/e1_no_pretrain \
    --epochs      3 \
    --merge-ratio 0.9 \
    --grad-accum  4
```

---

## 1.b — Without trajectory prediction loss

E1 encoder, trains with `l_score_past + l_score_future` only  
(`l_traj` and `l_score_traj` dropped — decoder is skipped entirely).

**Stage-1:**
```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2 \
python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation    score_only \
    --output-dir  TrajGaze_v2/checkpoints/e1_score_only \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_score_only \
    --epochs      100 \
    --lr          3e-4 \
    --batch-size  2
```

Stage-2 launches automatically after stage-1 completes  
(`train_merge_lora_temporal_no_kd --model-type full`).

---

## 2.a — Hand only

3 hand tokens (left, right, bimanual) + patch_temporal_branch.  
Gaze inputs are ignored by the encoder.

**Stage-1:**
```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2 \
python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation    hand_only \
    --output-dir  TrajGaze_v2/checkpoints/e1_hand_only \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_hand_only \
    --epochs      100 \
    --lr          3e-4 \
    --batch-size  2
```

Stage-2 uses `--model-type hand_only` automatically.

---

## 2.b — Gaze only

1 gaze token + patch_temporal_branch.  
Hand and IMU inputs are ignored by the encoder.

**Stage-1:**
```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2 \
python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation    gaze_only \
    --output-dir  TrajGaze_v2/checkpoints/e1_gaze_only \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_gaze_only \
    --epochs      100 \
    --lr          3e-4 \
    --batch-size  2
```

Stage-2 uses `--model-type gaze_only` automatically.

---

## 3.a — No spatial dimension

E1 encoder, `visual_feat=None` throughout. The encoder falls back to  
`_trajectory_only_scores()` (cross-attends learned patch embeddings to  
trajectory context). `TemporalVisualTrajFusion` is never called.  
`patch_temporal_branch` still applies its temporal modulation to those scores.  
`l_score_traj = 0` because `enc_attn` is unavailable without visual fusion.

**Stage-1:**
```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2 \
python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation    no_spatial \
    --output-dir  TrajGaze_v2/checkpoints/e1_no_spatial \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_no_spatial \
    --epochs      100 \
    --lr          3e-4 \
    --batch-size  2
```

---

## 3.b — No temporal dimension

Full encoder with `TemporalVisualTrajFusion` active (spatial is present),  
but `patch_temporal_branch` is disabled and `inter_frame_gate` is frozen  
at 0 (InterFrameTransformer completely bypassed in all paths).  
Zero temporal signal flows anywhere.

**Stage-1:**
```bash
CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2 \
python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation    no_temporal \
    --output-dir  TrajGaze_v2/checkpoints/e1_no_temporal \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_no_temporal \
    --epochs      100 \
    --lr          3e-4 \
    --batch-size  2
```

---

## Running all ablations in parallel

Assign one GPU per ablation. Example with 6 GPUs:

```bash
PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2

# 1.a — no stage-1, launch stage-2 directly
CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type full --stage1-ckpt /nonexistent \
    --output-dir TrajGazeMerge/checkpoints/e1_no_pretrain \
    --epochs 3 --merge-ratio 0.9 --grad-accum 4 &

# 1.b — score only
CUDA_VISIBLE_DEVICES=1 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation score_only \
    --output-dir TrajGaze_v2/checkpoints/e1_score_only \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_score_only &

# 2.a — hand only
CUDA_VISIBLE_DEVICES=2 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation hand_only \
    --output-dir TrajGaze_v2/checkpoints/e1_hand_only \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_hand_only &

# 2.b — gaze only
CUDA_VISIBLE_DEVICES=3 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation gaze_only \
    --output-dir TrajGaze_v2/checkpoints/e1_gaze_only \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_gaze_only &

# 3.a — no spatial
CUDA_VISIBLE_DEVICES=4 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation no_spatial \
    --output-dir TrajGaze_v2/checkpoints/e1_no_spatial \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_no_spatial &

# 3.b — no temporal
CUDA_VISIBLE_DEVICES=5 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \
    --ablation no_temporal \
    --output-dir TrajGaze_v2/checkpoints/e1_no_temporal \
    --stage2-output-dir TrajGazeMerge/checkpoints/e1_no_temporal &

wait
```

---

## Key files

| File | Purpose |
|------|---------|
| `TrajGaze_v2/training/stage1_temporal_e1_ablation.py` | Unified stage-1 script for all E1 ablations |
| `TrajGaze_v2/training/stage1_temporal.py` | Full E1 baseline stage-1 (used by run script) |
| `TrajGazeMerge/training/train_merge_lora_temporal_no_kd.py` | Stage-2 (CE only, no KD) |
| `TrajGazeMerge/training/run_e1_patch_temporal.sh` | Full baseline pipeline |
| `TrajGaze_v2/models/encoder_temporal.py` | `PatchTemporalBranch` + `SpatiotemporalEncoderTemporal` |
| `TrajGaze_v2/models/encoder_temporal_gaze_only.py` | Gaze-only encoder with E1 branch |
| `TrajGaze_v2/models/encoder_temporal_hand_only.py` | Hand-only encoder with E1 branch |

## Expected checkpoint paths after all runs

```
TrajGaze_v2/checkpoints/
  E1_patch_temporal/best.pth          ← baseline (already trained)
  e1_score_only/best.pth
  e1_gaze_only/best.pth
  e1_hand_only/best.pth
  e1_no_spatial/best.pth
  e1_no_temporal/best.pth

TrajGazeMerge/checkpoints/
  E1_patch_temporal_keep10_bs4/       ← baseline (already trained)
  e1_no_pretrain/
  e1_score_only/
  e1_gaze_only/
  e1_hand_only/
  e1_no_spatial/
  e1_no_temporal/
```
