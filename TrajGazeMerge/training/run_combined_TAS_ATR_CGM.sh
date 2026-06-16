#!/usr/bin/env bash
# Trajectory-grounded chain: Stage 1 TAS → (Stage 2 full ★ + Stage 2 TAS-only ablation)
#
# Stage 1 retrains the A+B encoder with --use-trajectory-anchor on 2 GPUs (DDP).
# Once `E1_combined_AB_TAS/best.pth` exists, two Stage 2 runs launch in parallel:
#   GPU 0 — ★ +TAS +ATR +CGM   (full grounded method)
#   GPU 1 — +TAS only          (ablation: encoder change alone)
#
# Plan reference: /home/irteam/.claude/plans/streamgaze-egogazevqa-swirling-koala.md

set -uo pipefail

REPO=/workspace/trajgaze
S1_OUT=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS
S2_FULL_OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR_CGM
S2_TAS_OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_only
LAUNCHER=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_chain.log
mkdir -p "$S1_OUT" "$S2_FULL_OUT" "$S2_TAS_OUT" "$(dirname "$LAUNCHER")"

echo "[$(date -u)] Chain start" > "$LAUNCHER"

# ── Stage 1 ─────────────────────────────────────────────────────────────────
# A+B (patch_temporal_branch) + TAS (trajectory_anchor) on combined dataset.
# Matches previous E1_combined_AB recipe: --freeze-gate --gate-init 0.0
# (inter-frame transformer bypassed, encoder converges via spatial-only path).
# 2 GPU via torchrun DDP; previous run took ~2h on 1 GPU so ≈ 1h here.
echo "[$(date -u)] Launching Stage 1 (DDP nproc=2)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$REPO \
    /opt/conda/envs/gaze/bin/torchrun --nproc_per_node=2 --master_port=29502 \
    -m TrajGaze_v2.training.stage1_temporal \
    --output-dir "$S1_OUT" \
    --use-patch-temporal-branch \
    --use-trajectory-anchor \
    --freeze-gate --gate-init 0.0 \
    --use-egovqa \
    --epochs       100 \
    --batch-size   2 \
    --n-frames     128 \
    --workers      4 \
    --log-every    50 \
    --save-every   20 \
    < /dev/null > "$S1_OUT/stdout.log" 2>&1
S1_RC=$?
echo "[$(date -u)] Stage 1 exit=$S1_RC" >> "$LAUNCHER"

if [ ! -f "$S1_OUT/best.pth" ]; then
    echo "[$(date -u)] Stage 1 failed (no best.pth). Aborting." >> "$LAUNCHER"
    exit 1
fi

# ── Stage 2 — ★ full method (GPU 0) ─────────────────────────────────────────
# TAS in encoder (via Stage 1 ckpt) + ATR + CGM
echo "[$(date -u)] Launching Stage 2 FULL (GPU 0): TAS+ATR+CGM" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_OUT/best.pth" \
    --output-dir   "$S2_FULL_OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --use-egovqa \
    --eval-egovqa-egtea \
    --use-atr        --atr-lambda 0.5 \
    --cgm-aug        --cgm-lambda 0.3 --cgm-prob 0.3 \
    --cgm-warmup-steps 600 --cgm-radius 0.2 --cgm-margin 0.5 \
    --dataloader-num-workers 8 \
    --eval-every 400 \
    < /dev/null > "$S2_FULL_OUT/stdout.log" 2>&1 &
S2F_PID=$!
echo "[$(date -u)] Stage 2 FULL PID=$S2F_PID" >> "$LAUNCHER"

# Give the full run a head start before the ablation grabs GPU 1
sleep 60

# ── Stage 2 — TAS-only ablation (GPU 1) ─────────────────────────────────────
echo "[$(date -u)] Launching Stage 2 TAS-only (GPU 1)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$S1_OUT/best.pth" \
    --output-dir   "$S2_TAS_OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    --use-egovqa \
    --eval-egovqa-egtea \
    --dataloader-num-workers 8 \
    --eval-every 400 \
    < /dev/null > "$S2_TAS_OUT/stdout.log" 2>&1 &
S2T_PID=$!
echo "[$(date -u)] Stage 2 TAS-only PID=$S2T_PID" >> "$LAUNCHER"

# Wait for both
wait $S2F_PID 2>/dev/null
echo "[$(date -u)] Stage 2 FULL finished" >> "$LAUNCHER"
wait $S2T_PID 2>/dev/null
echo "[$(date -u)] Stage 2 TAS-only finished" >> "$LAUNCHER"

echo "[$(date -u)] Chain complete." >> "$LAUNCHER"
echo "  Stage 1 ckpt: $S1_OUT/best.pth"        >> "$LAUNCHER"
echo "  ★ Stage 2 full: $S2_FULL_OUT/best.pth"  >> "$LAUNCHER"
echo "  Stage 2 TAS-only: $S2_TAS_OUT/best.pth" >> "$LAUNCHER"
