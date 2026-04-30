#!/bin/bash
# TrajGaze_v2 Stage 2 GRPO Training — 4 GPUs
# Policy: TrajGazeV2 token selector
# Reward: frozen Qwen2.5-VL-7B accuracy

set -e

STAGE1_CKPT="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1/best.pth"
OUTPUT_DIR="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage2"
mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "TrajGaze_v2 Stage 2 GRPO Training"
echo "  Stage 1 checkpoint: $STAGE1_CKPT"
echo "  Output: $OUTPUT_DIR"
echo "======================================================"

PYTHONPATH="/workspace/EgoGazeVQA:$PYTHONPATH" \
conda run -n gaze torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    /workspace/EgoGazeVQA/TrajGaze_v2/training/stage2.py \
    --stage1-ckpt "$STAGE1_CKPT" \
    --output-dir  "$OUTPUT_DIR" \
    --epochs 3 \
    --lr 1e-5 \
    --group-size 8 \
    --gumbel-temp 0.5 \
    --gumbel-temp-final 0.1 \
    --n-frames 32 \
    --n-qwen-frames 64 \
    --log-every 5 \
    --save-every 1 \
    --datasets ego4d egoexo egtea \
    2>&1 | tee "$OUTPUT_DIR/stage2_train.log"

echo "Stage 2 GRPO training complete!"
echo "Best checkpoint: $OUTPUT_DIR/best.pth"
