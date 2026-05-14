#!/usr/bin/env bash
# /workspace/trajgaze (main) — train_merge_lora_temporal_mrcons, 10% primary + 50% aux self-distill,
# bs4 (grad-accum), GPU 1. Uses torchrun (script is DDP-based).
set -euo pipefail

REPO=/workspace/trajgaze
OUT=$REPO/TrajGazeMerge/checkpoints/mr_cons_keep50_bs4_jw
mkdir -p "$OUT"

echo "[$(date)] mr_cons_keep50_bs4_jw launched (GPU 1, single torchrun, detached)" > "$OUT/launcher.log"

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=1 --master_port=29543 \
    -m TrajGazeMerge.training.train_merge_lora_temporal_mrcons \
    --stage1-ckpt    /workspace/trajgaze_msk/temporal_best.pth \
    --output-dir     "$OUT" \
    --epochs         3 \
    --alpha          0.0 \
    --merge-ratio    0.9 \
    --grad-accum     4 \
    --mr-cons-weight 0.5 \
    --mr-cons-keep   0.5 \
    --mr-cons-mode   kl_to_anchor \
    --log-every      20 \
    --eval-every     400 \
    > "$OUT/stdout.log" 2>&1

echo "[$(date)] Training exited with code $?" >> "$OUT/launcher.log"
