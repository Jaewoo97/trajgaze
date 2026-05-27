#!/usr/bin/env bash
# Sprint 1 (A+B) — Stage 2 launcher.
#
# Waits for Stage 1 (E1_sprint1_AB) to finish, then trains the LoRA + merge
# Stage 2 model that consumes the new Stage 1 encoder.
#
# Run AFTER Stage 1 has been kicked off (or is finished).

set -euo pipefail

REPO=/workspace/trajgaze
S1_OUT=$REPO/TrajGaze_v2/checkpoints/E1_sprint1_AB
S1_PID_HINT=${S1_PID_HINT:-}
S2_OUT=$REPO/TrajGazeMerge/checkpoints/E1_sprint1_AB_keep10
mkdir -p "$S2_OUT"

LAUNCHER=$S2_OUT/launcher.log
echo "[$(date -u)] Sprint1 Stage-2 launcher waiting on Stage-1 best.pth" > "$LAUNCHER"

# Wait until Stage 1 best.pth appears AND its training process is gone.
# best.pth is rewritten throughout training; we need to wait for the trainer
# to terminate (final epoch completes).
while true; do
    if [ -n "$S1_PID_HINT" ] && kill -0 "$S1_PID_HINT" 2>/dev/null; then
        sleep 60
        continue
    fi
    # No hint or hint died: also confirm via pgrep just in case
    if pgrep -f "stage1_temporal.*E1_sprint1_AB" >/dev/null 2>&1; then
        sleep 60
        continue
    fi
    if [ -f "$S1_OUT/best.pth" ]; then
        echo "[$(date -u)] Stage-1 done. best.pth detected." >> "$LAUNCHER"
        break
    fi
    echo "[$(date -u)] Stage-1 process gone but best.pth missing; retrying in 60s." >> "$LAUNCHER"
    sleep 60
done

echo "[$(date -u)] Stage-2: training token-merge + Qwen LoRA ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_OUT/best.pth" \
    --output-dir   "$S2_OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    > "$S2_OUT/stdout.log" 2>&1

S2_EXIT=$?
echo "[$(date -u)] Stage-2 exited with code $S2_EXIT" >> "$LAUNCHER"
