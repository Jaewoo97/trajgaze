#!/usr/bin/env bash
# Conditional mr-cons launcher.
#
# Waits for No-KD v2-temporal training+eval to finish, parses the OVERALL
# score from per_task_eval_best.log, and launches mr-cons v2-temporal ONLY
# IF the OVERALL beats the msk No-KD baseline (64.45 on n=526).
#
# Otherwise: log + exit (saves ~13h of compute on a regressed run).
#
# Started detached via:
#   setsid nohup ./conditional_mr_cons_launcher.sh > .../conditional_launcher.out 2>&1 < /dev/null &

set -uo pipefail

OUT=/workspace/trajgaze_v2/TrajGazeMerge/checkpoints/no_kd_v2_temporal
LOG=/workspace/trajgaze_v2/TrajGazeMerge/checkpoints/chain_v2_temporal.log
THRESHOLD=64.45   # msk no_kd_ce_only OVERALL on n=526
DIR=$(cd "$(dirname "$0")" && pwd)

echo "[$(date)] [conditional] Waiting for No-KD launcher to mark 'All done' ..." >> "$LOG"

# Poll launcher.log every 60s for "All done"
while true; do
    if [ -f "$OUT/launcher.log" ] && grep -q "All done" "$OUT/launcher.log"; then
        break
    fi
    sleep 60
done

echo "[$(date)] [conditional] No-KD All done detected." >> "$LOG"

# Parse OVERALL from per_task_eval_best.log
EVAL_LOG="$OUT/per_task_eval_best.log"
if [ ! -f "$EVAL_LOG" ]; then
    echo "[$(date)] [conditional] eval log missing: $EVAL_LOG" >> "$LOG"
    exit 1
fi

OVERALL=$(grep "^OVERALL" "$EVAL_LOG" | awk '{print $2}' | tr -d '%')
if [ -z "$OVERALL" ]; then
    echo "[$(date)] [conditional] failed to parse OVERALL from $EVAL_LOG" >> "$LOG"
    exit 1
fi

echo "[$(date)] [conditional] No-KD v2 OVERALL=${OVERALL}%, msk No-KD threshold=${THRESHOLD}%" >> "$LOG"

# Compare (bc -l for float comparison)
if [ "$(echo "$OVERALL >= $THRESHOLD" | bc -l)" -eq 1 ]; then
    echo "[$(date)] [conditional] PASS — launching mr-cons v2-temporal" >> "$LOG"
    echo "[$(date)] === START mr-cons v2-temporal (conditional) ===" >> "$LOG"
    bash "$DIR/run_mr_cons_v2_temporal.sh"
    echo "[$(date)] === END mr-cons v2-temporal ===" >> "$LOG"
    echo "[$(date)] === Chain complete ===" >> "$LOG"
else
    echo "[$(date)] [conditional] FAIL — No-KD ($OVERALL) < threshold ($THRESHOLD). Skipping mr-cons." >> "$LOG"
    echo "[$(date)] === Chain complete (mr-cons skipped) ===" >> "$LOG"
fi
