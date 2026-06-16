#!/usr/bin/env bash
# Monitors CF-1 and CF-3 train_log.jsonl, kills both training processes when both
# converge (same criteria as the original convergence_watcher.sh used for the
# TAS-only / TAS+ATR HD-EPIC runs on 2026-05-26).

set -uo pipefail

REPO=/workspace/trajgaze
LOG_CF1=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf1_hdepic_bs8_mb2/train_log.jsonl
LOG_CF3=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf3_hdepic_bs8_mb2/train_log.jsonl
PID_CF1=1994525
PID_CF3=1995481
WATCH_LOG=$REPO/TrajGazeMerge/eval_results/E1_cf_convergence_watcher.log
POLL=1800
THRESHOLD=0.3

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

check_converged() {
    local LOG=$1
    python3 - "$LOG" "$THRESHOLD" <<'PY'
import json, sys
path, thr = sys.argv[1], float(sys.argv[2])
evals = []
try:
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') == 'eval':
                evals.append((d['step'], d['mean_acc']))
except FileNotFoundError:
    print('NOT_FOUND')
    sys.exit(0)
if len(evals) < 8:
    print(f'TOO_FEW n_evals={len(evals)}')
    sys.exit(0)
last4 = max(e[1] for e in evals[-4:])
prev4 = max(e[1] for e in evals[-8:-4])
delta = last4 - prev4
status = 'CONVERGED' if delta < thr else 'IMPROVING'
print(f'{status} n_evals={len(evals)} last_step={evals[-1][0]} prev4_best={prev4:.2f} last4_best={last4:.2f} delta={delta:+.3f}')
PY
}

echo "[$(ts)] watcher start (pid=$$) — threshold=${THRESHOLD}pp / 4-eval window, poll=${POLL}s" >> "$WATCH_LOG"
echo "[$(ts)] CF-1 PID=$PID_CF1, CF-3 PID=$PID_CF3" >> "$WATCH_LOG"

while true; do
    CF1_ALIVE=0; CF3_ALIVE=0
    kill -0 $PID_CF1 2>/dev/null && CF1_ALIVE=1
    kill -0 $PID_CF3 2>/dev/null && CF3_ALIVE=1

    if [ $CF1_ALIVE -eq 0 ] && [ $CF3_ALIVE -eq 0 ]; then
        echo "[$(ts)] both training PIDs already dead — watcher exit" >> "$WATCH_LOG"
        break
    fi

    CF1_RES=$(check_converged "$LOG_CF1")
    CF3_RES=$(check_converged "$LOG_CF3")
    echo "[$(ts)] CF-1 alive=$CF1_ALIVE :: $CF1_RES" >> "$WATCH_LOG"
    echo "[$(ts)] CF-3 alive=$CF3_ALIVE :: $CF3_RES" >> "$WATCH_LOG"

    CF1_OK=0; CF3_OK=0
    if [ $CF1_ALIVE -eq 0 ] || [[ "$CF1_RES" == CONVERGED* ]]; then CF1_OK=1; fi
    if [ $CF3_ALIVE -eq 0 ] || [[ "$CF3_RES" == CONVERGED* ]]; then CF3_OK=1; fi

    if [ $CF1_OK -eq 1 ] && [ $CF3_OK -eq 1 ]; then
        echo "[$(ts)] === BOTH CONVERGED — issuing SIGTERM to training PIDs ===" >> "$WATCH_LOG"
        [ $CF1_ALIVE -eq 1 ] && kill $PID_CF1 2>/dev/null && echo "[$(ts)] SIGTERM -> $PID_CF1 (CF-1)" >> "$WATCH_LOG"
        [ $CF3_ALIVE -eq 1 ] && kill $PID_CF3 2>/dev/null && echo "[$(ts)] SIGTERM -> $PID_CF3 (CF-3)" >> "$WATCH_LOG"
        sleep 30
        # Escalate to SIGKILL on main PIDs + orphan workers
        for P in $PID_CF1 $PID_CF3; do
            if kill -0 $P 2>/dev/null; then
                kill -9 $P 2>/dev/null
                echo "[$(ts)] SIGKILL -> $P" >> "$WATCH_LOG"
            fi
        done
        sleep 10
        ORPH=$(pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" | tr '\n' ' ')
        if [ -n "$ORPH" ]; then
            echo "[$(ts)] SIGTERM orphan workers: $ORPH" >> "$WATCH_LOG"
            echo "$ORPH" | xargs -r kill 2>/dev/null
            sleep 15
            ORPH2=$(pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" | tr '\n' ' ')
            if [ -n "$ORPH2" ]; then
                echo "[$(ts)] SIGKILL orphan workers: $ORPH2" >> "$WATCH_LOG"
                echo "$ORPH2" | xargs -r kill -9 2>/dev/null
            fi
        fi
        sleep 30
        echo "[$(ts)] cf-mask launcher should detect free GPUs within 5min and proceed." >> "$WATCH_LOG"
        break
    fi

    sleep "$POLL"
done

echo "[$(ts)] watcher done" >> "$WATCH_LOG"
