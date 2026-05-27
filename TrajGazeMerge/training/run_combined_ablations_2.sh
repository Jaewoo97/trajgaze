#!/usr/bin/env bash
# Missing ablations from plan: +TAS+ATR (no CGM) and +CGM only (no TAS).
#
# GPU 0 — +TAS+ATR  : new TAS encoder + ATR loss (no CGM)        → isolates ATR effect
# GPU 1 — +CGM only : OLD non-TAS encoder + CGM loss (no ATR)    → isolates CGM effect
#
# Both use --grad-accum 16 for more stable gradients vs prior runs (was 4).
# Datasets: StreamGaze + EgoGazeVQA combined, same as before (no HD-EPIC).

set -uo pipefail

REPO=/workspace/trajgaze
S1_TAS=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth      # new (with TAS module)
S1_OLD=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB/best.pth          # original (no TAS)

OUT_TAS_ATR=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR
OUT_CGM_ONLY=$REPO/TrajGazeMerge/checkpoints/E1_combined_CGM_only

LAUNCHER=$REPO/TrajGazeMerge/checkpoints/E1_combined_ablations2.log
mkdir -p "$OUT_TAS_ATR" "$OUT_CGM_ONLY" "$(dirname "$LAUNCHER")"

echo "[$(date -u)] Ablations-2 chain start" > "$LAUNCHER"

# ── GPU 0 — +TAS+ATR (no CGM) ───────────────────────────────────────────────
echo "[$(date -u)] Launching +TAS+ATR (GPU 0)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_TAS" \
    --output-dir   "$OUT_TAS_ATR" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   16 \
    --use-egovqa \
    --eval-egovqa-egtea \
    --use-atr        --atr-lambda 0.5 \
    --dataloader-num-workers 8 \
    --eval-every 400 \
    < /dev/null > "$OUT_TAS_ATR/stdout.log" 2>&1 &
A_PID=$!
echo "[$(date -u)] +TAS+ATR PID=$A_PID" >> "$LAUNCHER"

# Stagger by 60s before grabbing GPU 1
sleep 60

# ── GPU 1 — +CGM only (no TAS, no ATR) ──────────────────────────────────────
# Uses OLD encoder (no TAS module). use_trajectory_anchor inferred False from ckpt.
echo "[$(date -u)] Launching +CGM only (GPU 1)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_OLD" \
    --output-dir   "$OUT_CGM_ONLY" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   16 \
    --use-egovqa \
    --eval-egovqa-egtea \
    --cgm-aug        --cgm-lambda 0.3 --cgm-prob 0.3 \
    --cgm-warmup-steps 600 --cgm-radius 0.2 --cgm-margin 0.5 \
    --dataloader-num-workers 8 \
    --eval-every 400 \
    < /dev/null > "$OUT_CGM_ONLY/stdout.log" 2>&1 &
C_PID=$!
echo "[$(date -u)] +CGM only PID=$C_PID" >> "$LAUNCHER"

# Wait for both
wait $A_PID 2>/dev/null
echo "[$(date -u)] +TAS+ATR finished" >> "$LAUNCHER"
wait $C_PID 2>/dev/null
echo "[$(date -u)] +CGM only finished" >> "$LAUNCHER"

echo "[$(date -u)] Ablations-2 complete." >> "$LAUNCHER"
echo "  +TAS+ATR:  $OUT_TAS_ATR/best.pth"  >> "$LAUNCHER"
echo "  +CGM only: $OUT_CGM_ONLY/best.pth" >> "$LAUNCHER"
