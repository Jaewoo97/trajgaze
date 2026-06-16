#!/usr/bin/env bash
# TAS-only + (TAS+ATR) with HD-EPIC, RETRY with effective batch 8 (was 16, OOM).
# GPU 0: TAS-only (CE only, no ATR/CGM)
# GPU 1: TAS+ATR  (CE + ATR, no CGM)
#
# Changes from previous failed run (run_tas_hdepic.sh):
#   - micro_batch 8 → 4   (per-step memory halved, was OOM @ step 540 epoch 1)
#   - grad_accum  2 → 2   (effective batch 8, was 16)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation
#
# Datasets unchanged:
#   - Train: StreamGaze 5799 + EgoGazeVQA 647 + HD-EPIC P01-P08 22551 = 28997
#   - Val: StreamGaze EGTEA 526 + EgoGazeVQA EGTEA 431 + HD-EPIC P09 3893 = 4850

set -uo pipefail

REPO=/workspace/trajgaze
S1_TAS=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth

OUT_TAS=$REPO/TrajGazeMerge/checkpoints/E1_combined_TASonly_hdepic_bs8
OUT_TAS_ATR=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR_hdepic_bs8
LAUNCHER=$REPO/TrajGazeMerge/checkpoints/E1_combined_tas_hdepic_bs8.log
mkdir -p "$OUT_TAS" "$OUT_TAS_ATR" "$(dirname "$LAUNCHER")"

echo "[$(date -u)] TAS+HD-EPIC bs8 chain start" > "$LAUNCHER"

COMMON_ARGS=(
    --model-type   full
    --stage1-ckpt  "$S1_TAS"
    --epochs       3
    --merge-ratio  0.9
    --micro-batch  4
    --grad-accum   2
    --use-egovqa
    --use-hd-epic
    --eval-egovqa-egtea
    --eval-hd-epic
    --dataloader-num-workers 8
    --eval-every 400
)

# ── GPU 0 — TAS-only ────────────────────────────────────────────────────────
echo "[$(date -u)] Launching TAS-only + HD-EPIC bs8 (GPU 0)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_batched \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUT_TAS" \
    < /dev/null > "$OUT_TAS/stdout.log" 2>&1 &
T_PID=$!
echo "[$(date -u)] TAS-only PID=$T_PID" >> "$LAUNCHER"

sleep 60

# ── GPU 1 — TAS+ATR ─────────────────────────────────────────────────────────
echo "[$(date -u)] Launching TAS+ATR + HD-EPIC bs8 (GPU 1)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup /opt/conda/envs/gaze/bin/python -m \
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
