#!/usr/bin/env bash
# VisionZip+traj 2-way (StreamGaze + EgoGazeVQA only) LoRA — gaze-overlay input,
# egtea val (SG egtea 526 + EG egtea 485 = 1011), 3 epochs with epoch1->epoch2
# early-stop. ALL 4 GPUs (0,1,2,3), CO-LOCATED alongside the running VZ (0,1) +
# TAS (2,3) 2-GPU jobs — each card is ~140GB with ~110GB free, so the 7B+LoRA
# footprint (~30GB/GPU) fits; the three jobs share SM compute (slower, expected).
# eff-batch 1*2*4 = 8 (matched to the current VZ/TAS runs).
# Trajectory-weighted VisionZip selection: attn_scores *= (1+spatial)*temporal
# around gaze/hand (parameter-free; NO Stage-1 encoder / no stage1-ckpt needed).
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
LOG=/tmp/vztraj_sgeg_overlay.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_traj_lora_sgeg_overlay
mkdir -p "$OUT"
echo "[vztraj-sgeg] $(date) launch (4 GPU co-located, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=4 --master_port=29653 \
  -m TrajGazeMerge.training.train_visionzip_traj_lora_3way \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[vztraj-sgeg] $(date) exit=$? :: VZTRAJ_SGEG_DONE" >> "$LOG"
