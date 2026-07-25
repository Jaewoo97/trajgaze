#!/usr/bin/env bash
# VisionZip 2-way (StreamGaze + EgoGazeVQA only) LoRA — gaze-overlay input,
# egtea val (SG egtea 526 + EG egtea 485 = 1011), 3 epochs with epoch1->epoch2
# early-stop. GPUs 0,1 (eff-batch 1*4*2 = 8). 10% token budget (5% dom + 5% ctx).
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1
LOG=/tmp/vz_sgeg_overlay.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_lora_sgeg_overlay
mkdir -p "$OUT"
echo "[vz-sgeg] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29651 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop >> "$LOG" 2>&1
echo "[vz-sgeg] $(date) exit=$? :: VZ_SGEG_DONE" >> "$LOG"
