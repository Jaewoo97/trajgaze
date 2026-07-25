#!/usr/bin/env bash
# OUR METHOD — gaze-scanpath token channel, additive on top of M1's frozen
# selection. Keeps M1's exact 6.5% raw + 3.5% merged token selection (topk
# complement, learned TAS) and ADDS K=8 scanpath tokens after the video block.
# Escapes the fixed-budget zero-sum that flattened coverage/fusion/anchored.
#
# 4 GPUs, eff-batch = grad_accum 2 * 4 = 8 (matches M1's eff-batch), 3 epochs,
# epoch1->epoch2 early-stop, gaze-overlay, egtea 2-way val. LoRA lr 1e-4;
# fresh scanpath encoder lr 1e-3.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}

LOG=/tmp/ours_scanpath.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_scanpath
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[scanpath] $(date) launch (4 GPU, gaze-overlay, egtea 2-way val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=4 --master_port=29662 \
  -m TrajGazeMerge.training.train_visionzip_scanpath_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --gaze-tokens 8 --t-scan 32 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --scan-lr 1e-3 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[scanpath] $(date) exit=$? :: OURS_SCANPATH_DONE" >> "$LOG"
