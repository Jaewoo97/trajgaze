#!/usr/bin/env bash
# TAS-only + (TAS+ATR) with HD-EPIC included in training.
# GPU 0: TAS-only (CE only, no ATR/CGM)
# GPU 1: TAS+ATR  (CE + ATR, no CGM)
#
# Both use:
#   - TAS Stage 1 ckpt (E1_combined_AB_TAS)
#   - StreamGaze + EgoGazeVQA + HD-EPIC P01-P08 train
#   - Val: StreamGaze EGTEA + EgoGazeVQA EGTEA + HD-EPIC P09
#   - Batched training (--micro-batch 4 --grad-accum 4 → effective batch 16)
#
# Note: First epoch slow due to HD-EPIC frame extraction cache miss.
# DataLoader 8-worker parallelism should hide most of it after warmup.

set -uo pipefail

REPO=/workspace/trajgaze
S1_TAS=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth

OUT_TAS=$REPO/TrajGazeMerge/checkpoints/E1_combined_TASonly_hdepic
OUT_TAS_ATR=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR_hdepic
LAUNCHER=$REPO/TrajGazeMerge/checkpoints/E1_combined_tas_hdepic.log
mkdir -p "$OUT_TAS" "$OUT_TAS_ATR" "$(dirname "$LAUNCHER")"

echo "[$(date -u)] TAS+HD-EPIC chain start" > "$LAUNCHER"

COMMON_ARGS=(
    --model-type   full
    --stage1-ckpt  "$S1_TAS"
    --epochs       3
    --merge-ratio  0.9
    --micro-batch  8
    --grad-accum   2
    --use-egovqa
    --use-hd-epic
    --eval-egovqa-egtea
    --eval-hd-epic
    --dataloader-num-workers 8
    --eval-every 400
)

# ── GPU 0 — TAS-only (no ATR, no CGM) ───────────────────────────────────────
echo "[$(date -u)] Launching TAS-only + HD-EPIC (GPU 0)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_batched \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUT_TAS" \
    < /dev/null > "$OUT_TAS/stdout.log" 2>&1 &
T_PID=$!
echo "[$(date -u)] TAS-only PID=$T_PID" >> "$LAUNCHER"

sleep 60

# ── GPU 1 — TAS+ATR (no CGM) ────────────────────────────────────────────────
echo "[$(date -u)] Launching TAS+ATR + HD-EPIC (GPU 1)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_batched \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUT_TAS_ATR" \
    --use-atr --atr-lambda 0.5 \
    < /dev/null > "$OUT_TAS_ATR/stdout.log" 2>&1 &
A_PID=$!
echo "[$(date -u)] TAS+ATR PID=$A_PID" >> "$LAUNCHER"

wait $T_PID 2>/dev/null
echo "[$(date -u)] TAS-only finished" >> "$LAUNCHER"
wait $A_PID 2>/dev/null
echo "[$(date -u)] TAS+ATR finished" >> "$LAUNCHER"
echo "[$(date -u)] Chain complete." >> "$LAUNCHER"
