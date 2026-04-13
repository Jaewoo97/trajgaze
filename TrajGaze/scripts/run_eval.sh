#!/bin/bash
# Evaluate TrajGaze on StreamGaze VQA benchmark.
#
# Usage:
#   bash TrajGaze/scripts/run_eval.sh [fold_a|fold_b|all] [vlm] [ckpt_path]
#
#   fold:     fold_a — evaluate on EGTEA+HoloAssist val split
#             fold_b — evaluate on EgoExoLearn val split
#             all    — evaluate on all datasets (default)
#   vlm:      nvila (default), qwen, or none (proxy)
#   ckpt:     path to .pt checkpoint (default: auto-detect latest stage2)
#
# Examples:
#   bash TrajGaze/scripts/run_eval.sh fold_a nvila
#   bash TrajGaze/scripts/run_eval.sh all none   # quick proxy eval

set -e
cd /workspace/EgoGazeVQA

FOLD=${1:-all}
VLM=${2:-nvila}
CKPT=${3:-""}

PYTHON=/opt/conda/envs/gaze/bin/python

ADAPTED_DIR=/workspace/datasets/StreamGaze_v2/adapted
INTERACT_DIR=/workspace/datasets/StreamGaze_v2/interaction
FRAMES_DIR=/workspace/datasets/StreamGaze_v2/frames
QA_FILE=/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json
RESULTS_DIR=/workspace/EgoGazeVQA/TrajGaze/results

mkdir -p "$RESULTS_DIR"

# Auto-detect checkpoint
if [ -z "$CKPT" ]; then
    if [ "$FOLD" = "all" ]; then
        SEARCH_FOLD=fold_a
    else
        SEARCH_FOLD=$FOLD
    fi
    CKPT=$(ls -t \
        /workspace/EgoGazeVQA/TrajGaze/checkpoints/stage2/${SEARCH_FOLD}_${VLM}/epoch_*.pt \
        /workspace/EgoGazeVQA/TrajGaze/checkpoints/stage1/${SEARCH_FOLD}_full/epoch_*.pt \
        2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found. Run training first, or specify path as 3rd argument."
        exit 1
    fi
fi

echo "=============================================="
echo "TrajGaze Evaluation"
echo "Fold/split: $FOLD"
echo "VLM:        $VLM"
echo "Checkpoint: $CKPT"
echo "=============================================="

FOLD_ARG=""
if [ "$FOLD" != "all" ]; then
    FOLD_ARG="--fold $FOLD"
fi

$PYTHON -m TrajGaze.training.evaluate \
    --stage2-ckpt  "$CKPT" \
    --adapted-dir  "$ADAPTED_DIR" \
    --interact-dir "$INTERACT_DIR" \
    --frames-dir   "$FRAMES_DIR" \
    --qa-file      "$QA_FILE" \
    --vlm          "$VLM" \
    --output       "$RESULTS_DIR" \
    $FOLD_ARG \
    --device       cuda \
    2>&1 | tee /workspace/EgoGazeVQA/TrajGaze/logs/eval_${FOLD}_${VLM}.log

echo "Evaluation complete. Results in $RESULTS_DIR"
