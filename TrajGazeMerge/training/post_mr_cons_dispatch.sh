#!/usr/bin/env bash
# Post-mr-cons dispatcher.
#
# Steps:
#   1. Wait for current mr-cons v2 run to mark "All done" in launcher.log
#   2. Print mr-cons final per-task report into a clearly visible file
#   3. Launch the two No-KD parallel runs on GPU 0 / GPU 1:
#        - run_no_kd_keep10_bs4.sh  (merge_ratio 0.9, keep 10%)
#        - run_no_kd_keep05_bs4.sh  (merge_ratio 0.95, keep 5%)
#
# Both new runs use a single GPU each + grad_accum=4 → effective batch = 4.

set -uo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=/workspace/trajgaze_v2/TrajGazeMerge/checkpoints
MR=$ROOT/mr_cons_v2_temporal
REPORT=$ROOT/mr_cons_v2_temporal_FINAL_REPORT.txt
LOG=$ROOT/post_mr_cons_dispatch.log

echo "[$(date)] [post-mr-cons] waiting for mr-cons All done ..." > "$LOG"

while ! grep -q "All done" "$MR/launcher.log" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] [post-mr-cons] mr-cons finished. Writing FINAL_REPORT." >> "$LOG"

{
    echo "================================================================================"
    echo "mr-cons v2 temporal — FINAL n=526 per-task results"
    echo "Generated $(date)"
    echo "================================================================================"
    echo ""
    if [ -f "$MR/per_task_eval_best.log" ]; then
        tail -25 "$MR/per_task_eval_best.log"
    else
        echo "(per_task_eval_best.log missing!)"
    fi
} > "$REPORT"

echo "[$(date)] [post-mr-cons] launching No-KD keep10 (GPU 0) and keep05 (GPU 1) in parallel" >> "$LOG"
echo "[$(date)] === START No-KD keep10 / keep05 (parallel) ===" >> "$LOG"

setsid nohup bash "$DIR/run_no_kd_keep10_bs4.sh" \
    > "$ROOT/no_kd_keep10_bs4_launcher.out" 2>&1 < /dev/null &
PID10=$!

setsid nohup bash "$DIR/run_no_kd_keep05_bs4.sh" \
    > "$ROOT/no_kd_keep05_bs4_launcher.out" 2>&1 < /dev/null &
PID05=$!

echo "[$(date)] keep10 PID=$PID10  keep05 PID=$PID05" >> "$LOG"

wait $PID10
echo "[$(date)] keep10 exited" >> "$LOG"

wait $PID05
echo "[$(date)] keep05 exited" >> "$LOG"

echo "[$(date)] === Both No-KD runs done ===" >> "$LOG"
