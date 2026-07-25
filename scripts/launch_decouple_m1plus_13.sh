#!/usr/bin/env bash
# Decoupling grid — M1+13% (CANDIDATE).
# Budget-decoupling test: keep M1's complementary selection but STOP trading
# content for gaze. 13% budget = full 10% VisionZip content (5% dominant + 5%
# contextual) UNION 3% top-k TAS complement (gaze ADDED on top of the 10%, not
# within it). => 8% raw + 5% merged. Same frozen TAS Stage-1 encoder + LoRA-only
# as M1; identical 2-way protocol (gaze-overlay, eff-batch 1*4*2=8, early-stop).
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1
LOG=/tmp/m1plus_13pct.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/m1plus_13pct
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[m1plus-13] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29671 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$S1" \
  --content-ratio 0.10 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[m1plus-13] $(date) exit=$? :: M1PLUS_13_DONE" >> "$LOG"
