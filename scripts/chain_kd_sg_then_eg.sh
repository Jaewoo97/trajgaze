#!/usr/bin/env bash
# Wait for the SG overlay-free KD run to finish, then start the EG one.
#
# Why not just rely on run_kd_eg_nooverlay.sh's own wait_gpu loop: that polls for
# <2000 MiB free on both GPUs, and the SG run briefly dips there between the training
# loop and its end-of-epoch eval. The EG job would then start on top of it and both
# would contend for memory. Gating on the SG *process* is unambiguous.
#
# EG does not consume anything SG produces, so it runs regardless of how SG ended;
# the SG outcome is recorded here so a failure is not silently buried.
#
#   nohup setsid scripts/chain_kd_sg_then_eg.sh &
set -u

cd "$(dirname "$0")/.." || exit 1
source env.sh

SG_LOG=$REPO/kd_train_sgonly_nooverlay.log
SG_OUT=$REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_SGonly_nooverlay
CHAIN=$REPO/kd_chain.log

echo "=== chain start $(date -Is) ===" >>"$CHAIN"

# 1) Wait for the SG training processes to exit.
while pgrep -f "train_visionzip_kd_lora.*--source sg" >/dev/null 2>&1; do
    sleep 60
done
echo "[chain] SG process exited $(date -Is)" >>"$CHAIN"

# 2) Record how it ended.
if grep -q "SG KD student NO-OVERLAY DONE" "$SG_LOG" 2>/dev/null; then
    echo "[chain] SG launcher reported DONE" >>"$CHAIN"
else
    echo "[chain] WARNING: no DONE marker in $SG_LOG — SG may have died early" >>"$CHAIN"
fi
if [ -f "$SG_OUT/best.pth" ]; then
    echo "[chain] SG best.pth present ($(stat -c%s "$SG_OUT/best.pth") bytes)" >>"$CHAIN"
    grep -E "Overall:|\[src\] sg:" "$SG_LOG" 2>/dev/null | tail -4 >>"$CHAIN"
else
    echo "[chain] WARNING: no SG best.pth — SG produced no checkpoint" >>"$CHAIN"
fi

# 3) Let the GPUs drain before handing over.
for i in 0 1; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)" -gt 2000 ]; do
        sleep 30
    done
done
echo "[chain] GPUs free, launching EG $(date -Is)" >>"$CHAIN"

bash "$REPO/scripts/run_kd_eg_nooverlay.sh"

echo "[chain] EG finished $(date -Is)" >>"$CHAIN"
grep -E "Overall:|\[src\] eg:" "$REPO/kd_train_egonly_nooverlay.log" 2>/dev/null | tail -4 >>"$CHAIN"
echo "=== chain done $(date -Is) ===" >>"$CHAIN"
