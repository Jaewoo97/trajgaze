#!/usr/bin/env bash
# Control TRAINING runs for the query_gaze ablation.
#
# Each control is byte-for-byte the cosine run (content 7% ∪ query 1% ∪ gaze 2%
# = 10%, single GPU, grad-accum 8 → eff-batch 8, 3 epochs, early-stop) EXCEPT
# --query-mode, so the comparison isolates the query signal:
#
#   cosine   the method            (already running → visionzip_query_gaze_overlay)
#   random   +1% ARBITRARY tokens  (CONTROL: gain from query relevance vs just +1% tokens)
#   shuffle  +1% by a DIFFERENT    (CONTROL: does the CORRECT question matter, or any)
#            item's question
#
# The M1 reference (7% ∪ 3% gaze) already exists at
#   .../checkpoints/visionzip_complement_learned_overlay/best.pth
#
# NOTE: controls must be TRAINED (LoRA adapts to whatever token set it is fed) —
# you cannot get them by re-evaluating the cosine checkpoint. Each takes ~10-12h
# on a shared GPU. Launch when a GPU frees up.
#
# Usage:  bash run_query_controls.sh <GPU> <random|shuffle>
set -euo pipefail

GPU="${1:?usage: run_query_controls.sh <GPU> <random|shuffle>}"
MODE="${2:?usage: run_query_controls.sh <GPU> <random|shuffle>}"
case "$MODE" in random|shuffle) ;; *) echo "MODE must be random|shuffle"; exit 1;; esac

PORT=$(( 29700 + RANDOM % 200 ))
ROOT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
OUT="$ROOT/visionzip_query_gaze_${MODE}_overlay"
STAGE1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
PYBIN=/opt/conda/envs/trajgaze/bin/torchrun

mkdir -p "$OUT"
cd /workspace/trajgaze_st
export GAZE_OVERLAY=1

CUDA_VISIBLE_DEVICES="$GPU" setsid nohup "$PYBIN" \
  --nproc_per_node=1 --master_port="$PORT" \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode query_gaze \
  --stage1-ckpt "$STAGE1" --output-dir "$OUT" \
  --content-ratio 0.07 --query-ratio 0.01 --traj-ratio 0.02 --query-mode "$MODE" \
  --epochs 3 --lr 1e-4 --grad-accum 8 \
  --no-hdepic --early-stop --no-mid-eval \
  > "$OUT/run.log" 2>&1 &

echo "launched query_gaze[$MODE] on GPU $GPU (port $PORT, pid $!) → $OUT/run.log"
