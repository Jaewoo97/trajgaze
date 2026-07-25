#!/usr/bin/env bash
# OUR METHOD — soft fusion. Drops M1's disjoint two-pool union: instead of
# 7% VZ-content ∪ 3% trajectory-complement, a single fused score
#   fused = norm(attn) + LAMBDA * norm(traj)
# picks the dominant (raw) tokens, while VisionZip's contextual merge is
# unchanged. Same 10% composition as M1 topk (6.5% raw + 3.5% merged at 7/3),
# but the raw budget is allocated softly between attention and gaze rather than
# hard-split. λ=0 -> pure VisionZip; large λ -> gaze-dominated dominant pool.
#
# SET LAM / NORM from the winning pre-check config before launching.
# Identical protocol to the M1 / coverage arms otherwise: frozen TAS Stage-1,
# LoRA-only, 2 GPUs, eff-batch 1*4*2 = 8, 3 epochs, epoch1->epoch2 early-stop,
# gaze-overlay, egtea 1011 2-way val.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}

LAM=${LAM:-1.0}
NORM=${NORM:-minmax}
LOG=/tmp/ours_fusion.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_fusion
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
mkdir -p "$OUT"
echo "[ours-fusion] $(date) launch fusion λ=$LAM norm=$NORM (2 GPU, gaze-overlay, egtea 2-way val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29658 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --complement-mode fusion --fusion-lambda "$LAM" --fusion-norm "$NORM" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[ours-fusion] $(date) exit=$? :: OURS_FUSION_DONE" >> "$LOG"
