#!/usr/bin/env bash
# Decoupling grid — VZ-(8/5) 13% gaze-matched CONTROL (no gaze).
# Plain VisionZip at 13% budget (8% dominant + 5% contextual = 8% raw + 5% merged),
# composition-matched to M1+13% but with NO trajectory complement. Isolates
# "does the +gaze gain scale at 13%" from the raw/merged balance.
# WIN test: (M1+13 - VZ-8/5) > (M1-10 - VZ-6.5/3.5)  =>  gaze scales past 10% cap.
# Co-located with VZ-(6.5/3.5) on GPUs 2,3. Same 2-way protocol (gaze-overlay,
# eff-batch 1*4*2=8, early-stop).
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=2,3
LOG=/tmp/vz_8_5.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/vz_8_5
mkdir -p "$OUT"
echo "[vz-8/5] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29672 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --dominant-ratio 0.08 --contextual-ratio 0.05 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop >> "$LOG" 2>&1
echo "[vz-8/5] $(date) exit=$? :: VZ_8_5_DONE" >> "$LOG"
