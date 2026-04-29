#!/usr/bin/env bash
# No-KD CE-only on TrajGazeV2 temporal — keep 10% tokens, single GPU, batch=4.
#
# Setup vs run_no_kd_v2_temporal.sh:
#   - 1 GPU (not 2-GPU DDP)  → effective batch = grad_accum × 1 GPU = 4
#   - Same --alpha 0.0  (teacher load + per-step forward auto-skipped)
#   - --merge-ratio 0.9  (keep 10%)
#
# Designed to run in parallel with run_no_kd_keep05_bs4.sh on the other GPU.

set -euo pipefail

REPO=/workspace/trajgaze_v2
OUT=$REPO/TrajGazeMerge/checkpoints/no_kd_keep10_bs4
mkdir -p "$OUT"

TRAIN_LOG="$OUT/stdout.log"
EVAL_LOG="$OUT/per_task_eval_best.log"

echo "[$(date)] No-KD keep10 bs4 launched (detached)" > "$OUT/launcher.log"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=1 --master_port=29512 \
    -m TrajGazeMerge.training.train_merge_lora_temporal \
    --stage1-ckpt  /workspace/trajgaze_msk/temporal_best.pth \
    --output-dir   "$OUT" \
    --epochs       3 \
    --lr-lora      1e-4 \
    --lr-enc       1e-5 \
    --alpha        0.0 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --log-every    20 \
    --eval-every   400 \
    > "$TRAIN_LOG" 2>&1

TRAIN_EXIT=$?
echo "[$(date)] Training exited with code $TRAIN_EXIT" >> "$OUT/launcher.log"
[ "$TRAIN_EXIT" -ne 0 ] && exit "$TRAIN_EXIT"
[ ! -f "$OUT/best.pth" ] && { echo "[$(date)] best.pth missing" >> "$OUT/launcher.log"; exit 1; }
echo "[$(date)] Training done. Starting n=526 per-task eval..." >> "$OUT/launcher.log"

sleep 30

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.eval.eval_per_task_temporal \
    --ckpt         "$OUT/best.pth" \
    --teacher-ckpt /workspace/trajgaze_msk/king_ms.pth \
    --stage1-ckpt  /workspace/trajgaze_msk/temporal_best.pth \
    --merge-ratio  0.9 \
    > "$EVAL_LOG" 2>&1

echo "[$(date)] Eval exited with code $?" >> "$OUT/launcher.log"
echo "[$(date)] All done." >> "$OUT/launcher.log"
