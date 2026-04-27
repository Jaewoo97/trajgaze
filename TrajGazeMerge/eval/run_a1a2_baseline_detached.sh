#!/usr/bin/env bash
# A1 + A2 (baseline + feat-KD + KD region answer_full) — survives parent shell termination.
#
# Identical to A1 (baseline + feat-KD) PLUS:
#   --kd-seq answer_full  : KD over 8-token answer span instead of last token only
#
# Hypothesis: extending KD supervision from last-1-token to answer-full-8-tokens
# provides 8x denser logit-KD signal, stacking on top of feature-level KD (A1).
# All within baseline params budget — no ViT LoRA, no aux trajectory hint.
#
# Comparison targets:
#   baseline (no axes)        : 64.45% (n=526)
#   axis 3 only               : 63.50% (regression)
#   A1 (baseline + feat-KD)   : 66.92% ⭐ current production
#   axes23 (axis 2+3)         : 66.54% (uses extra ViT LoRA params)
#
# Chains: training → n=526 per-task eval. All logs to stdout.log/eval.log.

set -euo pipefail

OUT=/workspace/trajgaze_msk/TrajGazeMerge/checkpoints/a1a2_baseline_feat_seq
mkdir -p "$OUT"

TRAIN_LOG="$OUT/stdout.log"
EVAL_LOG="$OUT/per_task_eval_best.log"
DONE_FLAG="$OUT/.training_done"

echo "[$(date)] A1+A2 baseline+feat-KD+answer_full launched (detached)" > "$OUT/launcher.log"

# Step 1: training (2-GPU torchrun)
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/workspace/trajgaze_msk \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=2 --master_port=29508 \
    -m TrajGazeMerge.training.train_merge_lora \
    --stage1-ckpt  /workspace/trajgaze_msk/TrajGaze_v2/checkpoints/stage1_v3/best.pth \
    --teacher-ckpt /workspace/trajgaze_msk/king_ms.pth \
    --output-dir   "$OUT" \
    --epochs       3 \
    --lr-lora      1e-4 \
    --lr-enc       1e-5 \
    --alpha        0.5 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --kd-feat-layers=-1,-2 \
    --kd-feat-weight 0.3 \
    --kd-seq       answer_full \
    --log-every    20 \
    --eval-every   200 \
    > "$TRAIN_LOG" 2>&1

TRAIN_EXIT=$?
echo "[$(date)] Training exited with code $TRAIN_EXIT" >> "$OUT/launcher.log"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "[$(date)] Training failed — skipping eval" >> "$OUT/launcher.log"
    exit "$TRAIN_EXIT"
fi

if [ ! -f "$OUT/best.pth" ]; then
    echo "[$(date)] best.pth not found — skipping eval" >> "$OUT/launcher.log"
    exit 1
fi

touch "$DONE_FLAG"
echo "[$(date)] Training done. Starting n=526 per-task eval..." >> "$OUT/launcher.log"

sleep 30

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/workspace/trajgaze_msk \
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.eval.eval_per_task \
    --ckpt         "$OUT/best.pth" \
    --teacher-ckpt /workspace/trajgaze_msk/king_ms.pth \
    --stage1-ckpt  /workspace/trajgaze_msk/TrajGaze_v2/checkpoints/stage1_v3/best.pth \
    --merge-ratio  0.9 \
    --drop-ratio   0.0 \
    --score-transform sigmoid \
    --match-score-penalty 0.0 \
    > "$EVAL_LOG" 2>&1

EVAL_EXIT=$?
echo "[$(date)] Eval exited with code $EVAL_EXIT" >> "$OUT/launcher.log"
echo "[$(date)] All done." >> "$OUT/launcher.log"
