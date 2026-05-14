#!/usr/bin/env bash
# EA1 FIX encoder + mr-cons (multi-ratio consistency, KD-free self-distill).
#
# Encoder: /workspace/trajgaze/TrajGaze_v2/checkpoints/EA1_parallel_branch/best.pth
# Self-distill: primary forward keep=10% (merge=0.9) imitates stop-grad aux forward keep=50%.
# Aux gap is 5x primary (vs prior msk run's 1.5x) — stronger distill signal hypothesis.
#
# Past data points (NOT directly comparable — different encoders):
#   - msk mr-cons keep=0.15 on temporal_best lineage : 66.16% (vs no-KD 64.45%, +1.71pt)
#   - trajgaze mr-cons keep=0.50 on temporal_best.pth: best 66.35%
#     (vs trajgaze no-KD baseline 67.49%, REGRESSION -1.14pt)
#   - EA1 FIX (no-KD on EA1 encoder): 68.44%
#
# Uncertainty: mr-cons gain may not survive on stronger encoder. Worth testing.

set -euo pipefail

REPO=/workspace/trajgaze
OUT=$REPO/TrajGazeMerge/checkpoints/EA1fix_mrcons_keep50_bs4
mkdir -p "$OUT"

LAUNCHER=$OUT/launcher.log
echo "[$(date)] EA1 FIX + mr-cons keep=0.50 launched (GPU 1, detached)" > "$LAUNCHER"

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=1 --master_port=29543 \
    -m TrajGazeMerge.training.train_merge_lora_temporal_mrcons \
    --stage1-ckpt    "$REPO/TrajGaze_v2/checkpoints/EA1_parallel_branch/best.pth" \
    --output-dir     "$OUT" \
    --epochs         3 \
    --alpha          0.0 \
    --merge-ratio    0.9 \
    --grad-accum     4 \
    --mr-cons-weight 0.5 \
    --mr-cons-keep   0.5 \
    --mr-cons-mode   kl_to_anchor \
    --log-every      20 \
    --eval-every     400 \
    > "$OUT/stdout.log" 2>&1

EXIT=$?
echo "[$(date)] Training exited with code $EXIT" >> "$LAUNCHER"
echo "[$(date)] All done." >> "$LAUNCHER"
