#!/bin/bash
# TrajGaze_v2 Stage 1 Training — 4 GPUs
# Trajectory + interaction score prediction

set -e

OUTPUT_DIR="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1"
mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "TrajGaze_v2 Stage 1 Training"
echo "  GPUs: 4"
echo "  Output: $OUTPUT_DIR"
echo "======================================================"

PYTHONPATH="/workspace/EgoGazeVQA:$PYTHONPATH" \
conda run -n gaze torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    /workspace/EgoGazeVQA/TrajGaze_v2/training/stage1.py \
    --output-dir "$OUTPUT_DIR" \
    --epochs 100 \
    --lr 3e-4 \
    --batch-size 4 \
    --weight-decay 1e-4 \
    --n-frames 32 \
    --workers 4 \
    --log-every 10 \
    --save-every 20 \
    2>&1 | tee "$OUTPUT_DIR/stage1_train.log"

echo "Stage 1 training complete!"
echo "Best checkpoint: $OUTPUT_DIR/best.pth"
