#!/usr/bin/env bash
# OUR METHOD (variant 2) — per-token gaze/hand salience TAG, additive on top of
# M1's frozen selection. Keeps M1's exact 6.5% raw + 3.5% merged token selection
# (topk complement, learned TAS) and ADDS a learned per-token salience embedding
# to EACH retained visual token. Different axis from the scanpath channel
# (sequence-level intent tokens); here it's per-token spatial relevance.
# Both escape the fixed-budget zero-sum.
#
# Runs in PARALLEL with launch_ours_scanpath.sh, co-located on the same 4 GPUs.
# 4 GPUs, eff-batch = grad_accum 2 * 4 = 8 (matches M1's eff-batch), 3 epochs,
# epoch1->epoch2 early-stop, gaze-overlay, egtea 2-way val. LoRA lr 1e-4;
# fresh tag embedding lr 1e-3.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}

LOG=/tmp/ours_gazetag.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_gazetag
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[gazetag] $(date) launch (4 GPU, gaze-overlay, egtea 2-way val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=4 --master_port=29663 \
  -m TrajGazeMerge.training.train_visionzip_gazetag_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --tag-bins 16 --tag-norm minmax \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --tag-lr 1e-3 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[gazetag] $(date) exit=$? :: OURS_GAZETAG_DONE" >> "$LOG"
