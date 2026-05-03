#!/bin/bash
# E1_patch_temporal full pipeline: stage-1 pretraining + stage-2 LoRA merge.
#
# Architecture: SpatiotemporalEncoderTemporal with --use-patch-temporal-branch
#   - InterFrameTransformer gate frozen at 0 (bypass main path)
#   - 196 learned patch queries cross-attend to x_iframe per frame
#   - Produces (B, T, 196) per-patch temporal modulation applied to VisualTrajFusion scores
#   - vs EA1: frame_score_branch gives only (B, T, 1) scalar — same weight for all patches
#
# Usage:
#   setsid nohup bash /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/training/run_e1_patch_temporal.sh &

set -euo pipefail

PYTHONPATH=/workspace/EgoGazeVQA/TrajGazeMerge_v2
STAGE1_CKPT_DIR=/workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/E1_patch_temporal
STAGE2_OUT_DIR=/workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4

echo "======================================================"
echo " E1_patch_temporal pipeline"
echo " Stage 1 ckpt : $STAGE1_CKPT_DIR"
echo " Stage 2 out  : $STAGE2_OUT_DIR"
echo "======================================================"

# ── Stage 1: encoder pretraining (~25 min) ─────────────────────────────────────
echo ""
echo "[Stage 1] Starting encoder pretraining..."
mkdir -p "$STAGE1_CKPT_DIR"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$PYTHONPATH \
torchrun --nproc_per_node=1 --master_port=29841 \
    -m TrajGaze_v2.training.stage1_temporal \
    --output-dir "$STAGE1_CKPT_DIR" \
    --use-patch-temporal-branch \
    --freeze-gate \
    --epochs 100 \
    --lr 3e-4 \
    --batch-size 2 \
    --n-frames 128

echo "[Stage 1] Done. Best checkpoint: $STAGE1_CKPT_DIR/best.pth"

# ── Stage 2: token-merge + Qwen LoRA (~24h) ───────────────────────────────────
echo ""
echo "[Stage 2] Starting token-merge + Qwen LoRA training..."
mkdir -p "$STAGE2_OUT_DIR"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$PYTHONPATH \
python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
    --model-type  full \
    --stage1-ckpt "$STAGE1_CKPT_DIR/best.pth" \
    --output-dir  "$STAGE2_OUT_DIR" \
    --epochs      3 \
    --merge-ratio 0.9 \
    --grad-accum  4

echo ""
echo "[Done] E1_patch_temporal pipeline complete."
echo "  Stage-1 best: $STAGE1_CKPT_DIR/best.pth"
echo "  Stage-2 out:  $STAGE2_OUT_DIR"
