#!/usr/bin/env bash
# Qualitative token-selection figures for the specialist KD students.
#
#   scripts/run_qual_kd.sh sg 0 --scan-limit 60 --n-figures 3      # sample
#   scripts/run_qual_kd.sh sg 0 --n-figures 12                      # full run
#
# Arg 1 = source (sg|eg), arg 2 = GPU index; everything after is passed through to
# qual_kd_render.py. Both sources fit one GPU each, so a full run is:
#   scripts/run_qual_kd.sh sg 0 & scripts/run_qual_kd.sh eg 1 & wait
#
# GAZE_OVERLAY=1 with VLM_GAZE_OVERLAY unset is REQUIRED: rows 1-2 and the teacher's TAS
# stream read the overlay tree, and the script derives row 3's marker-free frames itself.
# The script asserts this at the first item and exits if the streams collapse (§7.3).
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source env.sh

SRC="${1:?usage: run_qual_kd.sh <sg|eg> <gpu> [extra args]}"
GPU="${2:?usage: run_qual_kd.sh <sg|eg> <gpu> [extra args]}"
shift 2

unset VLM_GAZE_OVERLAY || true
export GAZE_OVERLAY=1

LOG="qual_${SRC}.log"
echo "[run_qual_kd] source=$SRC gpu=$GPU -> $LOG"
# This node is shared and evicts jobs (kd_handoff_v2.md §13.4 records three such losses; both
# of this figure set's first full runs were SIGKILLed two minutes in when another job landed).
# --resume makes a restart cost only the item in flight, so retry rather than lose the pass.
# Append, so a retry does not erase the attempt that got furthest.
RETRIES="${QUAL_RETRIES:-4}"
for attempt in $(seq 1 "$RETRIES"); do
  echo "=== attempt $attempt/$RETRIES $(date -Is) ===" >> "$LOG"
  if CUDA_VISIBLE_DEVICES="$GPU" python -m TrajGazeMerge.viz.qual.qual_kd_render \
       --source "$SRC" --gpu 0 --resume "$@" >> "$LOG" 2>&1; then
    echo "[run_qual_kd] $SRC done on attempt $attempt" | tee -a "$LOG"
    exit 0
  fi
  echo "[run_qual_kd] $SRC attempt $attempt died; resuming in 30s" | tee -a "$LOG"
  sleep 30
done
echo "[run_qual_kd] $SRC gave up after $RETRIES attempts" | tee -a "$LOG"
exit 1
