#!/usr/bin/env bash
# E1+B: patch-temporal modulation + iframe-conditioned patch query.
# Builds on E1 (patch-level temporal branch) by conditioning the static (196, D)
# patch_temporal_query on per-frame x_iframe context (mean-pooled across 4 tokens
# → MLP → additive offset broadcast over 196 patches). Same x_iframe used as KV
# (per-token detail) and query conditioning (frame-level summary).
#
# Chains stage-1 encoder pretraining (~25 min) → stage-2 token-merge + Qwen LoRA (~12-15h).
#
# GPU 1 single-GPU. Detached (setsid nohup) so survives parent shell termination.
#
# Stage-1 acceptance gates: best loss ≤ 0.026 (E1 baseline 0.0185).
# Stage-2 hard target: > 68.44% (current EA1 / E1 ceiling).

set -euo pipefail

REPO=/workspace/trajgaze
ENC_OUT=$REPO/TrajGaze_v2/checkpoints/E1B_iframe_cond
S2_OUT=$REPO/TrajGazeMerge/checkpoints/E1B_iframe_cond_keep10_bs4
mkdir -p "$ENC_OUT" "$S2_OUT"

LAUNCHER=$S2_OUT/launcher.log
echo "[$(date)] E1+B iframe-cond pipeline launched (GPU 1, detached)" > "$LAUNCHER"

# ── Stage 1: encoder pretraining ──────────────────────────────────────────────
echo "[$(date)] Stage-1: training E1+B encoder ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=1 --master_port=29842 \
    -m TrajGaze_v2.training.stage1_temporal \
    --output-dir "$ENC_OUT" \
    --use-patch-temporal-branch \
    --use-iframe-query-conditioning \
    --freeze-gate \
    > "$ENC_OUT/stdout.log" 2>&1

S1_EXIT=$?
echo "[$(date)] Stage-1 exited with code $S1_EXIT" >> "$LAUNCHER"

if [ "$S1_EXIT" -ne 0 ]; then
    echo "[$(date)] Stage-1 FAILED — aborting." >> "$LAUNCHER"
    exit "$S1_EXIT"
fi

if [ ! -f "$ENC_OUT/best.pth" ]; then
    echo "[$(date)] Stage-1 best.pth not found — aborting." >> "$LAUNCHER"
    exit 1
fi

# ── Stage 2: token-merge + Qwen LoRA (no-KD baseline-style) ───────────────────
echo "[$(date)] Stage-2: training token-merge + Qwen LoRA ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type   full \
    --stage1-ckpt  "$ENC_OUT/best.pth" \
    --output-dir   "$S2_OUT" \
    --epochs       3 \
    --merge-ratio  0.9 \
    --grad-accum   4 \
    > "$S2_OUT/stdout.log" 2>&1

S2_EXIT=$?
echo "[$(date)] Stage-2 exited with code $S2_EXIT" >> "$LAUNCHER"
echo "[$(date)] All done." >> "$LAUNCHER"
