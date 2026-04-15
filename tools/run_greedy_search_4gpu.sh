#!/bin/bash
# Run paper-faithful greedy search GT generation on 4 GPUs
# Full loss: L1(1.0) + DINOv2-Large(0.3) + SigLIP2-Large(0.3)
# Estimated: ~25 hours

FOLD_DIR=/workspace/datasets/EgoGazeVQA/autogaze_fold
LOG_DIR=/workspace/EgoGazeVQA/AutoGaze/exps/greedy_search_logs
mkdir -p "$LOG_DIR"

echo "=== Launching greedy search GT generation on 4 GPUs ==="
echo "Fold dir: $FOLD_DIR"
echo "Logs: $LOG_DIR"

for GPU_IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_IDX PYTHONUNBUFFERED=1 conda run -n gaze --no-capture-output \
        python -u /workspace/EgoGazeVQA/tools/greedy_search_gt.py \
            --fold-dir "$FOLD_DIR" \
            --gpu $GPU_IDX \
            --total_gpus 4 \
            --cuda_device 0 \
            --seed 42 \
        > "$LOG_DIR/gpu${GPU_IDX}.log" 2>&1 &
    echo "  GPU $GPU_IDX launched (PID $!)"
done

echo "All 4 GPUs launched. Monitor with: tail -f $LOG_DIR/gpu0.log"
echo "After completion, run:"
echo "  conda run -n gaze python /workspace/EgoGazeVQA/tools/greedy_search_gt.py --fold-dir $FOLD_DIR --merge"
