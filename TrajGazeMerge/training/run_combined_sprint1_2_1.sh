#!/usr/bin/env bash
# Combined training chain: waits for Stage 1 → runs Sprint 1 Stage 2 → Sprint 2.1 Stage 2.
#
# Stage 1 must be launched separately (PID 2116249 or similar). This watcher waits for
# /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_combined_AB/best.pth, then chains
# the two Stage 2 runs (uses GPUs 0 and 1 respectively).

set -euo pipefail

REPO=/workspace/trajgaze
S1_OUT=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB
SP1_OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_sprint1
SP21_OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_sprint2_1
LAUNCHER=$REPO/TrajGazeMerge/checkpoints/E1_combined_chain.log
mkdir -p "$SP1_OUT" "$SP21_OUT" "$(dirname "$LAUNCHER")"

echo "[$(date -u)] Combined chain — waiting on Stage 1" > "$LAUNCHER"

# Wait for Stage 1 to finish: best.pth exists AND no stage1 process active
while true; do
    if pgrep -f "stage1_temporal.*E1_combined_AB" >/dev/null 2>&1; then
        sleep 120
        continue
    fi
    if [ -f "$S1_OUT/best.pth" ]; then
        echo "[$(date -u)] Stage 1 done. Found $S1_OUT/best.pth" >> "$LAUNCHER"
        break
    fi
    echo "[$(date -u)] Stage 1 process gone but best.pth missing; sleeping 120s" >> "$LAUNCHER"
    sleep 120
done

# ── Sprint 1 on GPU 0 (no shuffle aug) ────────────────────────────────────────
echo "[$(date -u)] Launching Sprint 1 Stage 2 (GPU 0)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_OUT/best.pth" \
    --output-dir   "$SP1_OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --use-egovqa \
    --eval-egovqa-egtea \
    < /dev/null > "$SP1_OUT/stdout.log" 2>&1 &
SP1_PID=$!
echo "[$(date -u)] Sprint 1 PID=$SP1_PID" >> "$LAUNCHER"

# Give Sprint 1 a head start before launching Sprint 2.1 on GPU 1
sleep 60

# ── Sprint 2.1 on GPU 1 (shuffle aug λ=0.2) ──────────────────────────────────
echo "[$(date -u)] Launching Sprint 2.1 Stage 2 (GPU 1)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_OUT/best.pth" \
    --output-dir   "$SP21_OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --shuffle-aug --shuffle-prob 0.3 --shuffle-margin 0.5 --shuffle-lambda 0.2 \
    --shuffle-warmup-steps 600 \
    --use-egovqa \
    --eval-egovqa-egtea \
    < /dev/null > "$SP21_OUT/stdout.log" 2>&1 &
SP21_PID=$!
echo "[$(date -u)] Sprint 2.1 PID=$SP21_PID" >> "$LAUNCHER"

# Wait for both to finish
wait $SP1_PID 2>/dev/null || true
echo "[$(date -u)] Sprint 1 finished" >> "$LAUNCHER"
wait $SP21_PID 2>/dev/null || true
echo "[$(date -u)] Sprint 2.1 finished" >> "$LAUNCHER"
echo "[$(date -u)] Chain complete." >> "$LAUNCHER"
