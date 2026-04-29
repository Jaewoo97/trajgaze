#!/usr/bin/env bash
# mr-cons (KD-free, axis 6) on TrajGazeV2 temporal.
#
# --alpha 0.0                  : logit KL OFF (no external teacher in loss)
# --mr-cons-weight 0.5         : multi-ratio consistency loss
# --mr-cons-keep   0.15        : aux forward keeps 15%% (vs primary 10%%)
# --mr-cons-mode   kl_to_anchor: primary imitates detached aux
#
# With α=0 the trainer skips loading + per-step forwarding the teacher
# entirely (~25%% wall-clock and ~16GB GPU saved), so we omit --teacher-ckpt
# below.  The per-task eval below still uses the teacher to report a
# full-token baseline number alongside the merged-token student accuracy.

set -euo pipefail

REPO=/workspace/trajgaze_v2
OUT=$REPO/TrajGazeMerge/checkpoints/mr_cons_v2_temporal
mkdir -p "$OUT"

TRAIN_LOG="$OUT/stdout.log"
EVAL_LOG="$OUT/per_task_eval_best.log"

echo "[$(date)] mr-cons v2-temporal launched (detached)" > "$OUT/launcher.log"

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=2 --master_port=29511 \
    -m TrajGazeMerge.training.train_merge_lora_temporal \
    --stage1-ckpt  /workspace/trajgaze_msk/temporal_best.pth \
    --output-dir   "$OUT" \
    --epochs       3 \
    --lr-lora      1e-4 \
    --lr-enc       1e-5 \
    --alpha        0.0 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --mr-cons-weight 0.5 \
    --mr-cons-keep   0.15 \
    --mr-cons-mode   kl_to_anchor \
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
