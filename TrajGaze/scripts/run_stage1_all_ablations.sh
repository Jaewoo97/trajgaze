#!/bin/bash
# Run Stage 1 for all ablation variants in parallel across GPUs.
#
# GPU assignment:
#   GPU 0 — full (fold_a)
#   GPU 1 — no_frame_selector (fold_a)
#   GPU 2 — no_traj_loss (fold_a)
#   GPU 3 — no_ntp (fold_a)
#
# Usage: bash TrajGaze/scripts/run_stage1_all_ablations.sh [fold_a|fold_b]

set -e
cd /workspace/EgoGazeVQA

FOLD=${1:-fold_a}
PYTHON=/opt/conda/envs/gaze/bin/python
LOG_DIR=/workspace/EgoGazeVQA/TrajGaze/logs
mkdir -p "$LOG_DIR"

ADAPTED_DIR=/workspace/datasets/StreamGaze_v2/adapted
INTERACT_DIR=/workspace/datasets/StreamGaze_v2/interaction
FRAMES_DIR=/workspace/datasets/StreamGaze_v2/frames

MODES=(full no_frame_selector no_traj_loss no_ntp)
GPUS=(0 1 2 3)

echo "Launching Stage 1 for fold=$FOLD across ${#MODES[@]} GPUs ..."

for i in "${!MODES[@]}"; do
    MODE=${MODES[$i]}
    GPU=${GPUS[$i]}
    OUT=/workspace/EgoGazeVQA/TrajGaze/checkpoints/stage1/${FOLD}_${MODE}
    mkdir -p "$OUT"

    echo "  GPU $GPU — mode=$MODE → $OUT"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m TrajGaze.training.stage1 \
        --adapted-dir  "$ADAPTED_DIR" \
        --interact-dir "$INTERACT_DIR" \
        --frames-dir   "$FRAMES_DIR" \
        --output-dir   "$OUT" \
        --mode         "$MODE" \
        --epochs       150 \
        --lr           3e-4 \
        --max-frames   200 \
        --save-every   10 \
        --device       cuda \
        2>&1 | tee "$LOG_DIR/stage1_${FOLD}_${MODE}.log" &
done

echo "All ablations launched. Waiting ..."
wait
echo "All Stage 1 ablations complete."
