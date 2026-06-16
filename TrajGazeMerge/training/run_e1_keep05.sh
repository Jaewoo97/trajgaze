#!/usr/bin/env bash
# E1 patch-temporal stage-2 with keep=5% (merge_ratio=0.95).
# Reuses the existing E1_patch_temporal stage-1 encoder ckpt — no Stage 1 retrain.
# Stage-2 only: token-merge + Qwen LoRA at 5% retention.
#
# Paper main table: token retention {10%, 5%, 3%}.
#   10% (merge_ratio=0.9):   E1_patch_temporal_keep10_bs4 → best 68.44% (Ep2 step 4400)
#   5%  (merge_ratio=0.95):  THIS RUN → E1_patch_temporal_keep05_bs4
#   3%  (merge_ratio=0.97):  TBD
#
# GPU 0 single-GPU. Detached (setsid nohup) so survives parent shell termination.

set -euo pipefail

REPO=/workspace/trajgaze
ENC_CKPT=$REPO/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth
S2_OUT=$REPO/TrajGazeMerge/checkpoints/E1_patch_temporal_keep05_bs4
mkdir -p "$S2_OUT"

LAUNCHER=$S2_OUT/launcher.log
echo "[$(date)] E1 patch-temporal keep=5% pipeline launched (GPU 0, detached)" > "$LAUNCHER"

if [ ! -f "$ENC_CKPT" ]; then
    echo "[$(date)] Stage-1 ckpt not found at $ENC_CKPT — aborting." >> "$LAUNCHER"
    exit 1
fi
echo "[$(date)] Reusing existing Stage-1 ckpt: $ENC_CKPT" >> "$LAUNCHER"

# ── Stage 2: token-merge + Qwen LoRA (merge_ratio=0.95 → keep 5%) ─────────────
echo "[$(date)] Stage-2: training token-merge + Qwen LoRA (keep=5%) ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$ENC_CKPT" \
    --output-dir   "$S2_OUT" \
    --epochs       3 \
    --merge-ratio  0.95 \
    --grad-accum   4 \
    > "$S2_OUT/stdout.log" 2>&1

S2_EXIT=$?
echo "[$(date)] Stage-2 exited with code $S2_EXIT" >> "$LAUNCHER"
echo "[$(date)] All done." >> "$LAUNCHER"
