#!/bin/bash
# Monitor greedy search completion (by clip count) then auto-start NTP training.
# Only triggers when BOTH train and val clips reach expected totals.
FOLD_DIR=/workspace/datasets/EgoGazeVQA/autogaze_fold
LOG_DIR=/workspace/EgoGazeVQA/AutoGaze/exps/greedy_search_logs
TRAIN_SCRIPT=/workspace/EgoGazeVQA/AutoGaze/train_egogaze_ntp.sh

TOTAL_TRAIN=18637
TOTAL_VAL=5674

echo "[wait_and_train] Monitoring greedy search. Expected: train=$TOTAL_TRAIN  val=$TOTAL_VAL"

count_clips() {
    local pattern="$1"
    local total=0
    for f in $FOLD_DIR/$pattern; do
        [ -f "$f" ] || continue
        n=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo 0)
        total=$((total + n))
    done
    echo $total
}

while true; do
    sleep 600

    DONE_TRAIN=$(count_clips "gt_train_gpu*.json")
    DONE_VAL=$(count_clips "gt_val_gpu*.json")
    GREEDY_PIDS=$(pgrep -f "greedy_search_gt.py" 2>/dev/null | wc -l)

    echo "[$(date '+%H:%M')] Train: $DONE_TRAIN/$TOTAL_TRAIN  Val: $DONE_VAL/$TOTAL_VAL  PIDs: $GREEDY_PIDS"

    if [ "$DONE_TRAIN" -ge "$TOTAL_TRAIN" ] && [ "$DONE_VAL" -ge "$TOTAL_VAL" ]; then
        echo "[$(date '+%H:%M')] All clips done! Merging..."
        conda run -n gaze python /workspace/EgoGazeVQA/tools/greedy_search_gt.py \
            --fold-dir "$FOLD_DIR" --merge
        echo "[$(date '+%H:%M')] Launching NTP training on 4 GPUs..."
        cd /workspace/EgoGazeVQA/AutoGaze
        bash "$TRAIN_SCRIPT" 4 >> "$LOG_DIR/training_autorestart.log" 2>&1 &
        echo "Training launched (PID $!)."
        break
    fi

    if [ "$GREEDY_PIDS" -eq 0 ] && ([ "$DONE_TRAIN" -lt "$TOTAL_TRAIN" ] || [ "$DONE_VAL" -lt "$TOTAL_VAL" ]); then
        echo "[$(date '+%H:%M')] WARNING: processes died early. Train: $DONE_TRAIN/$TOTAL_TRAIN  Val: $DONE_VAL/$TOTAL_VAL"
    fi
done
