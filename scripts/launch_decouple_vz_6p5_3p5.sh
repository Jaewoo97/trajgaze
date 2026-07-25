#!/usr/bin/env bash
# Decoupling grid — VZ-(6.5/3.5) 10% DIAGNOSTIC (no gaze).
# Plain VisionZip at 10% budget but composition-matched to M1-10% (6.5% dominant
# + 3.5% contextual = 6.5% raw + 3.5% merged) instead of plain VZ's 5/5. Isolates
# the gaze-vs-fidelity confound inside M1's original +0.5 over plain VZ: how much
# of it is the gaze complement vs. simply having more raw (less-merged) tokens.
# Co-located with VZ-(8/5) on GPUs 2,3. Same 2-way protocol (gaze-overlay,
# eff-batch 1*4*2=8, early-stop).
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=2,3
LOG=/tmp/vz_6p5_3p5.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/vz_6p5_3p5
mkdir -p "$OUT"
echo "[vz-6.5/3.5] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29673 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --dominant-ratio 0.065 --contextual-ratio 0.035 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop >> "$LOG" 2>&1
echo "[vz-6.5/3.5] $(date) exit=$? :: VZ_6P5_3P5_DONE" >> "$LOG"
