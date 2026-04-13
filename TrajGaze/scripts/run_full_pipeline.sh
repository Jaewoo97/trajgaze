#!/bin/bash
# Full TrajGaze training pipeline — both folds, Stage 1 then Stage 2.
#
# Runs sequentially on a single node. For parallel fold training, run
# run_stage1.sh and run_stage2.sh manually on separate GPUs.
#
# Usage: bash TrajGaze/scripts/run_full_pipeline.sh [vlm]
#   vlm: nvila (default), qwen, or none

set -e
cd /workspace/EgoGazeVQA

VLM=${1:-nvila}

echo "=========================================="
echo "TrajGaze Full Pipeline"
echo "VLM oracle: $VLM"
echo "=========================================="

# Precompute interaction scores if not done
ADAPTED=/workspace/datasets/StreamGaze_v2/adapted
INTERACT=/workspace/datasets/StreamGaze_v2/interaction
if [ ! -d "$INTERACT" ] || [ -z "$(ls -A $INTERACT 2>/dev/null)" ]; then
    echo "[0] Computing interaction scores ..."
    /opt/conda/envs/gaze/bin/python -m TrajGaze.data.interaction \
        --adapted-dir "$ADAPTED" \
        --output-dir  "$INTERACT" \
        --workers     32
fi

# Stage 1 — both folds
for FOLD in fold_a fold_b; do
    echo ""
    echo "[Stage 1] $FOLD ..."
    bash TrajGaze/scripts/run_stage1.sh "$FOLD" full
done

# Stage 2 — both folds
for FOLD in fold_a fold_b; do
    echo ""
    echo "[Stage 2] $FOLD | VLM=$VLM ..."
    bash TrajGaze/scripts/run_stage2.sh "$FOLD" "$VLM"
done

echo ""
echo "Full pipeline complete."
echo "Checkpoints: TrajGaze/checkpoints/"
