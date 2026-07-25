#!/usr/bin/env bash
# OUR METHOD — Arm B. Same coverage complement as Arm A (temporal quota + spatial
# NMS), but a LARGER complement budget: 6% content ∪ 4% complement = 10%.
# Hypothesis: M1's analysis shows top-k wastes ~71% of the 3% on spatially
# redundant patches; once the complement is de-clustered by coverage, each token
# is a DISTINCT task-relevant region, so it may be worth trading 1% of content
# budget for 1% more (now-informative) complement. Arm A (7/3) is the safe split;
# Arm B (6/4) tests leaning harder on the diverse complement.
# Identical otherwise: frozen TAS Stage-1 encoder, LoRA-only, 2 GPUs (2,3),
# eff-batch 1*4*2 = 8, 3 epochs, epoch1->epoch2 early-stop, gaze-overlay, egtea 1011 val.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=3,0   # GPU3 free; rank1 co-locates on GPU0 with train_p3.py (1.9GB, 20%) — accuracy unaffected
LOG=/tmp/ours_coverage_b.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_coverage_b
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[ours-B] $(date) launch coverage nms=1 6/4 (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29657 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --complement-mode coverage --nms-radius 1 \
  --content-ratio 0.06 --traj-ratio 0.04 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[ours-B] $(date) exit=$? :: OURS_B_DONE" >> "$LOG"
