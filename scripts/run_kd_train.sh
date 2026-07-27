#!/bin/bash
# Supervised KD training: relaunch on crash, resuming from the last epoch checkpoint.
#
# The trainer prints "[KD] TRAINING COMPLETE" when it finishes (or early-stops);
# anything else ending the process is treated as a crash and retried. --resume is
# safe on the first attempt too: with no epoch_*.pth it just starts fresh.
#
#   nohup setsid scripts/run_kd_train.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

OUT=$REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_overlay
LOG=$REPO/kd_train.log
MAX_ATTEMPTS=20
mkdir -p "$OUT"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== attempt $attempt  $(date -Is) ===" >>"$LOG"

    CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29661 \
      -m TrajGazeMerge.training.train_visionzip_kd_lora \
      --warmstart-ckpt "$M1_JOINT" \
      --stage1-ckpt    "$STAGE1_CKPT" \
      --output-dir     "$OUT" \
      --epochs 3 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 \
      --no-hdepic --early-stop --resume >>"$LOG" 2>&1

    if grep -q "TRAINING COMPLETE" "$LOG"; then
        echo "=== KD DONE $(date -Is) ===" >>"$LOG"
        exit 0
    fi

    echo "--- attempt $attempt ended without completion; retrying in 120s ---" >>"$LOG"
    # Free any stale DDP shared memory / zombie ranks before relaunching.
    pkill -9 -f "train_visionzip_kd_lora" 2>/dev/null
    sleep 120
done

echo "=== KD GAVE UP after $MAX_ATTEMPTS attempts $(date -Is) ===" >>"$LOG"
exit 1
