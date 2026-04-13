#!/bin/bash
# Stage 2 — GRPO fine-tuning for TrajGaze
#
# Usage:
#   bash TrajGaze/scripts/run_stage2.sh [vlm] [stage1_ckpt]
#
#   vlm:          nvila          — NVILA-8B-HD-Video oracle (default)
#                 qwen           — Qwen2.5-VL-7B oracle
#                 none           — proxy reward (attend-label accuracy, no VLM)
#
#   stage1_ckpt:  path to Stage 1 checkpoint
#                 (default: auto-detect latest in checkpoints/stage1/full/)
#
# Train: EgoExoLearn + HoloAssist
# Val:   EGTEA

set -e
cd /workspace/EgoGazeVQA

VLM=${1:-nvila}
STAGE1_CKPT=${2:-""}

# vila_eval_env has the compatible transformers/torch version for NVILA-8B-HD-Video
PYTHON=/workspace/vila_eval_env/bin/python
TORCHRUN="$PYTHON -m torch.distributed.run"

ADAPTED_DIR=/workspace/datasets/StreamGaze_v2/adapted
INTERACT_DIR=/workspace/datasets/StreamGaze_v2/interaction
FRAMES_DIR=/workspace/datasets/StreamGaze_v2/frames
QA_FILE=/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json
OUTPUT_DIR=/workspace/EgoGazeVQA/TrajGaze/checkpoints/stage2/${VLM}
LOG_DIR=/workspace/EgoGazeVQA/TrajGaze/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Auto-select latest Stage 1 checkpoint if not specified
if [ -z "$STAGE1_CKPT" ]; then
    STAGE1_DIR=/workspace/EgoGazeVQA/TrajGaze/checkpoints/stage1/full
    STAGE1_CKPT=$(ls -1 "$STAGE1_DIR"/epoch_*.pt 2>/dev/null | sort | tail -1)
    if [ -z "$STAGE1_CKPT" ]; then
        echo "ERROR: No Stage 1 checkpoint found in $STAGE1_DIR"
        echo "       Run run_stage1.sh first, or specify path as 2nd argument."
        exit 1
    fi
fi

echo "=============================================="
echo "TrajGaze Stage 2 — GRPO Fine-tuning"
echo "VLM oracle:  $VLM"
echo "Stage1 ckpt: $STAGE1_CKPT"
echo "Train:       egoexolearn + holoassist"
echo "Val:         egtea"
echo "Output:      $OUTPUT_DIR"
echo "=============================================="

$TORCHRUN \
    --nproc-per-node 4 \
    --master-port    29502 \
    -m TrajGaze.training.stage2 \
    --adapted-dir    "$ADAPTED_DIR" \
    --interact-dir   "$INTERACT_DIR" \
    --frames-dir     "$FRAMES_DIR" \
    --qa-file        "$QA_FILE" \
    --stage1-ckpt    "$STAGE1_CKPT" \
    --output-dir     "$OUTPUT_DIR" \
    --vlm            "$VLM" \
    --epochs         3 \
    --lr             5e-4 \
    --max-frames     300 \
    --group-size     12 \
    --delta-token    0.1 \
    --train-datasets egoexolearn holoassist \
    --val-datasets   egtea \
    2>&1 | tee "$LOG_DIR/stage2_${VLM}.log"

echo "Stage 2 done. Checkpoint at: $OUTPUT_DIR"
