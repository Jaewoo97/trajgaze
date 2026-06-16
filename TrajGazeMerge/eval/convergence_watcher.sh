#!/usr/bin/env bash
# Background watcher: monitors TAS-only and TAS+ATR train_log.jsonl, decides
# when both runs have converged, then kills both training processes so the
# Step 1 cf-mask launcher (already polling) can take over.
#
# Convergence rule (per run): best(mean_acc, last 4 evals) - best(prev 4 evals) < 0.3pp.
# Requires >= 8 evals before any decision. Both runs must converge to trigger kill.
#
# Idempotent: if both PIDs are already dead, exits cleanly. Re-launching is safe.

set -uo pipefail

REPO=/workspace/trajgaze
LOG_TAS=$REPO/TrajGazeMerge/checkpoints/E1_combined_TASonly_hdepic_bs8_mb2/train_log.jsonl
LOG_ATR=$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR_hdepic_bs8_mb2/train_log.jsonl
PID_TAS=3436296
PID_ATR=3437781
WATCH_LOG=$REPO/TrajGazeMerge/eval_results/E1_convergence_watcher.log
POLL=1800        # 30min between checks
THRESHOLD=0.3    # pp improvement floor

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

while true; do
    TAS_ALIVE=0; ATR_ALIVE=0
    kill -0 $PID_TAS 2>/dev/null && TAS_ALIVE=1
    kill -0 $PID_ATR 2>/dev/null && ATR_ALIVE=1

    if [ $TAS_ALIVE -eq 0 ] && [ $ATR_ALIVE -eq 0 ]; then
        echo "[$(ts)] both training PIDs already dead — watcher exit" >> "$WATCH_LOG"
        break
    fi

    TAS_RES=$(check_converged "$LOG_TAS")
    ATR_RES=$(check_converged "$LOG_ATR")
    echo "[$(ts)] TAS-only  alive=$TAS_ALIVE :: $TAS_RES" >> "$WATCH_LOG"
    echo "[$(ts)] TAS+ATR   alive=$ATR_ALIVE :: $ATR_RES" >> "$WATCH_LOG"

    # Treat a dead training as "converged" (whatever it produced is final).
    TAS_OK=0; ATR_OK=0
    if [ $TAS_ALIVE -eq 0 ] || [[ "$TAS_RES" == CONVERGED* ]]; then TAS_OK=1; fi
    if [ $ATR_ALIVE -eq 0 ] || [[ "$ATR_RES" == CONVERGED* ]]; then ATR_OK=1; fi

    if [ $TAS_OK -eq 1 ] && [ $ATR_OK -eq 1 ]; then
        echo "[$(ts)] === BOTH CONVERGED — issuing SIGTERM to training PIDs ===" >> "$WATCH_LOG"
        [ $TAS_ALIVE -eq 1 ] && kill $PID_TAS 2>/dev/null && echo "[$(ts)] SIGTERM -> $PID_TAS (TAS-only)" >> "$WATCH_LOG"
        [ $ATR_ALIVE -eq 1 ] && kill $PID_ATR 2>/dev/null && echo "[$(ts)] SIGTERM -> $PID_ATR (TAS+ATR)" >> "$WATCH_LOG"
        sleep 30
        # Escalate if needed
        for P in $PID_TAS $PID_ATR; do
            if kill -0 $P 2>/dev/null; then
                kill -9 $P 2>/dev/null
                echo "[$(ts)] SIGKILL -> $P (didn't exit on SIGTERM)" >> "$WATCH_LOG"
            fi
        done
        sleep 30
        echo "[$(ts)] cf-mask launcher should detect free GPUs within 5min and proceed." >> "$WATCH_LOG"
        break
    fi

    sleep "$POLL"
done

echo "[$(ts)] watcher done" >> "$WATCH_LOG"
