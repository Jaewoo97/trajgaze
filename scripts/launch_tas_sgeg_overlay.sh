#!/usr/bin/env bash
# TAS Stage-2 2-way (StreamGaze + EgoGazeVQA only) LoRA — gaze-overlay input,
# egtea val (1011), 3 epochs with epoch1->epoch2 early-stop. GPUs 2,3
# (eff-batch 1*4*2 = 8, matched with VisionZip). merge-ratio 0.9 = keep 10%.
# Stage-1 encoder REUSED from the 3-way overlay pretrain (encoder saw HD-EPIC
# gaze trajectories at pretrain; Stage-2 VQA training here is strictly 2-way).
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=2,3
LOG=/tmp/tas_sgeg_overlay.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tas_lora_sgeg_overlay
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[tas-sgeg] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29652 \
  -m TrajGazeMerge.training.train_merge_lora_tas_3way \
  --model-type full --stage1-ckpt "$S1" \
  --output-dir "$OUT" \
  --epochs 3 --merge-ratio 0.9 \
  --micro-batch 1 --grad-accum 4 \
  --no-hdepic --early-stop --eval-every 0 \
  --dataloader-num-workers 8 >> "$LOG" 2>&1
echo "[tas-sgeg] $(date) exit=$? :: TAS_SGEG_DONE" >> "$LOG"
