#!/usr/bin/env bash
# Method 1 — VisionZip + COMPLEMENTARY learned-trajectory selection.
# 10% budget = 7% VisionZip content ∪ 3% top-k from a FROZEN TAS Stage-1 encoder
# (the gaze/hand-relevant tokens content attention missed). LoRA-only training,
# identical protocol to VZ / VZ+traj: 2 GPUs (0,1), eff-batch 1*4*2 = 8,
# 3 epochs with epoch1->epoch2 early-stop, gaze-overlay input, egtea 1011 val.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1
LOG=/tmp/vzcomp_learned_overlay.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[vzcomp-learned] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29654 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[vzcomp-learned] $(date) exit=$? :: VZCOMP_LEARNED_DONE" >> "$LOG"
