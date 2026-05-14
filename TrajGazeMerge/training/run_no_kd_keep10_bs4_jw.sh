#!/usr/bin/env bash
# /workspace/trajgaze (main) — train_merge_lora_temporal_no_kd, 10% tokens, bs4 (grad-accum), GPU 0.
set -euo pipefail

REPO=/workspace/trajgaze
OUT=$REPO/TrajGazeMerge/checkpoints/no_kd_keep10_bs4_jw
mkdir -p "$OUT"

echo "[$(date)] no_kd_keep10_bs4_jw launched (GPU 0, single python, detached)" > "$OUT/launcher.log"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  /workspace/trajgaze_msk/temporal_best.pth \
    --output-dir   "$OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    > "$OUT/stdout.log" 2>&1

echo "[$(date)] Training exited with code $?" >> "$OUT/launcher.log"
