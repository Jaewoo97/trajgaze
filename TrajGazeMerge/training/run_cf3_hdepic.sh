#!/usr/bin/env bash
# CF-3 (CE + L_mask + L_shuf) — Direction A from docs/current_state.md.
#
# Stage-2 LoRA finetune of TAS Stage-1 ckpt on combined StreamGaze + EgoGazeVQA + HD-EPIC,
# with two counterfactual margin losses:
#   - shuffle_aug  → V_shuf component (kept tokens permuted)
#   - use-cf-mask  → V_mask component (kept tokens all zeroed)
#
# Hypothesis: forcing baseline gt_logit to exceed both counterfactuals by `margin`
# will (a) break EgoGazeVQA language-prior dominance (mask_kept Δ from +0.93 → <-2.0)
# and (b) fix the FULL-pipeline shuffle anomaly (shuffle_kept Δ from +1.33 → <0).
#
# Decision gate is in /home/irteam/.claude/plans/cf-mask-augmented-training.md §4.2.

set -euo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
STAGE1=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth
OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf3_hdepic_bs8_mb2

# Single-GPU; mirror existing TASonly/TAS_ATR runs (micro_batch=2, grad_accum=4, eff_bs=8).
# Convergence watcher should be started separately after launch.
CUDA_VISIBLE_DEVICES=${1:-0} \
PYTHONPATH=$REPO \
$PY -m TrajGazeMerge.training.train_merge_lora_batched \
    --model-type full \
    --stage1-ckpt "$STAGE1" \
    --epochs 3 \
    --merge-ratio 0.9 \
    --micro-batch 2 \
    --grad-accum 4 \
    --use-egovqa --use-hd-epic \
    --eval-egovqa-egtea --eval-hd-epic \
    --dataloader-num-workers 8 \
    --eval-every 400 \
    --output-dir "$OUT" \
    --shuffle-aug --shuffle-prob 0.3 --shuffle-margin 1.0 --shuffle-lambda 0.3 --shuffle-warmup-steps 600 \
    --use-cf-mask --cf-mask-prob 0.3 --cf-mask-margin 1.0 --cf-mask-lambda 0.3 --cf-mask-warmup-steps 600
