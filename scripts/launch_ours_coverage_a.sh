#!/usr/bin/env bash
# OUR METHOD — Arm A. VisionZip + COMPLEMENTARY learned-trajectory selection with
# TEMPORAL+SPATIAL COVERAGE (vs M1's raw global top-k).
# 10% budget = 7% VisionZip content ∪ 3% complement, where the 3% is chosen by:
#   (1) per-frame quota: floor >=1 token to every gaze-active frame, remainder
#       split by per-frame trajectory mass  (temporal coverage), then
#   (2) intra-frame greedy NMS (radius 1) so picks land on distinct regions
#       instead of the contiguous blob around the single gaze peak (spatial
#       coverage). Fixes M1's top-k collapse → more DISTINCT task-relevant
#       gaze/hand regions resurrected for the same budget.
# Same philosophy as M1 (union with the tokens content-attention missed), same
# frozen TAS Stage-1 encoder, same LoRA-only protocol: 2 GPUs (0,1),
# eff-batch 1*4*2 = 8, 3 epochs, epoch1->epoch2 early-stop, gaze-overlay, egtea 1011 val.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=1,2   # fully-free GPUs (GPU0 held by train_p3.py)
LOG=/tmp/ours_coverage_a.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_coverage_a
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[ours-A] $(date) launch coverage nms=1 7/3 (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29656 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --complement-mode coverage --nms-radius 1 \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[ours-A] $(date) exit=$? :: OURS_A_DONE" >> "$LOG"
