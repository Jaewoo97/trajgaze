#!/bin/bash
# Stage 1 — NTP pretraining for TrajGaze (4-GPU, bfloat16)
#
# Usage:
#   bash TrajGaze/scripts/run_stage1.sh [mode]
#
#   mode:  full               — full proposed method (default)
#          no_frame_selector  — all frames attended (AutoGaze-style)
#          no_traj_loss       — NTP only, λ_traj=0
#          no_ntp             — trajectory prediction loss only
#          no_gaze            — hand signal only
#          no_hand            — gaze signal only
#
# Train: EgoExoLearn + HoloAssist
# Val:   EGTEA

set -e
cd /workspace/EgoGazeVQA

MODE=${1:-full}

PYTHON=/opt/conda/envs/gaze/bin/python
TORCHRUN=/opt/conda/envs/gaze/bin/torchrun

ADAPTED_DIR=/workspace/datasets/StreamGaze_v2/adapted
INTERACT_DIR=/workspace/datasets/StreamGaze_v2/interaction
FRAMES_DIR=/workspace/datasets/StreamGaze_v2/frames
OUTPUT_DIR=/workspace/EgoGazeVQA/TrajGaze/checkpoints/stage1/${MODE}
LOG_DIR=/workspace/EgoGazeVQA/TrajGaze/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================================="
echo "TrajGaze Stage 1 — NTP Pretraining"
echo "Mode:        $MODE"
echo "GPUs:        4 (H200)"
echo "Train:       egoexolearn + holoassist"
echo "Val:         egtea"
echo "max_frames:  800"
echo "Token budget: 50% frames x 20% patches = 10% total"
echo "Output:      $OUTPUT_DIR"
echo "=============================================="

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m TrajGaze.training.stage1 \
    --adapted-dir    "$ADAPTED_DIR" \
    --interact-dir   "$INTERACT_DIR" \
    --frames-dir     "$FRAMES_DIR" \
    --output-dir     "$OUTPUT_DIR" \
    --mode           "$MODE" \
    --epochs         150 \
    --lr             3e-4 \
    --max-frames     800 \
    --accum-steps    4 \
    --num-workers    8 \
    --save-every     10 \
    --val-every      10 \
    --train-datasets egoexolearn holoassist \
    --val-datasets   egtea \
    2>&1 | tee "$LOG_DIR/stage1_${MODE}.log"

echo "Stage 1 done. Checkpoint at: $OUTPUT_DIR"
