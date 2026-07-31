#!/usr/bin/env bash
# Contact sheets for the ViT-KD ablation (docs/kd_handoff_v3.md, setting 1 = SG raw video).
#
#   scripts/run_vitkd_sheets.sh --scan-limit 60 --n-items 3     # sample
#   scripts/run_vitkd_sheets.sh --n-items 10                    # full
#
# GAZE_OVERLAY=1 with VLM_GAZE_OVERLAY unset is required: the teacher reads `viz`, the student
# reads `original` (derived in-script), and the TAS stream reads `viz`. The script asserts all
# three at the first item and exits if they collapse.
#
# GPU: pass --gpu N. The ViT-KD chain may be occupying both cards; memory is not the constraint
# (~22 GB of 183 GB per B200), so co-existing is fine — it is just slower. Do not kill that chain.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source env.sh

unset VLM_GAZE_OVERLAY || true
export GAZE_OVERLAY=1

GPU="${QUAL_GPU:-0}"
LOG="qual_vitkd_sheets.log"
RETRIES="${QUAL_RETRIES:-4}"

echo "[run_vitkd_sheets] gpu=$GPU -> $LOG"
for attempt in $(seq 1 "$RETRIES"); do
  echo "=== attempt $attempt/$RETRIES $(date -Is) ===" >> "$LOG"
  if CUDA_VISIBLE_DEVICES="$GPU" python -m TrajGazeMerge.viz.qual.vitkd_contact_sheet \
       --gpu 0 --resume "$@" >> "$LOG" 2>&1; then
    echo "[run_vitkd_sheets] done on attempt $attempt" | tee -a "$LOG"
    exit 0
  fi
  echo "[run_vitkd_sheets] attempt $attempt died; resuming in 30s" | tee -a "$LOG"
  sleep 30
done
echo "[run_vitkd_sheets] gave up after $RETRIES attempts" | tee -a "$LOG"
exit 1
