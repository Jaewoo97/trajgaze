#!/bin/bash
# §3 of docs/kd_handoff.md: gaze-free KD of the SEPARATELY-TRAINED M1 specialists.
# SG-only runs first (~4h), then EG-only is chained automatically (~1.2h).
#
# Per specialist: warm-start from that specialist's M1 best.pth, unchanged TAS Stage-1
# teacher field, matching --source filter, specialist protocol (--epochs 2 --early-stop,
# eff-batch 8 = 2 GPUs x grad-accum 4).
#
# Each specialist gets its OWN log, because the crash-retry loop decides "finished vs
# crashed" by grepping for the trainer's terminal marker — a shared log would let SG's
# marker satisfy EG's check.
#
#   nohup setsid scripts/run_kd_specialists.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

DRIVER_LOG=$REPO/kd_train_specialists.log
MAX_ATTEMPTS=20

run_spec () {          # $1=source  $2=warm-start ckpt  $3=master_port
    local SRC=$1 WARM=$2 PORT=$3
    local TAG=$(echo "$SRC" | tr '[:lower:]' '[:upper:]')only
    local OUT=$REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_${TAG}_overlay
    local LOG=$REPO/kd_train_${SRC}only.log
    mkdir -p "$OUT"

    echo "=== [$SRC] KD start $(date -Is) → $OUT ===" >>"$DRIVER_LOG"

    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        echo "=== attempt $attempt  $(date -Is) ===" >>"$LOG"

        CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=$PORT \
          -m TrajGazeMerge.training.train_visionzip_kd_lora \
          --warmstart-ckpt "$WARM" \
          --stage1-ckpt    "$STAGE1_CKPT" \
          --source         "$SRC" \
          --output-dir     "$OUT" \
          --epochs 2 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 \
          --no-hdepic --early-stop --resume >>"$LOG" 2>&1

        if grep -q "TRAINING COMPLETE" "$LOG"; then
            echo "=== [$SRC] KD DONE $(date -Is) ===" >>"$LOG"
            echo "=== [$SRC] KD DONE $(date -Is) ===" >>"$DRIVER_LOG"
            return 0
        fi

        echo "--- [$SRC] attempt $attempt ended without completion; retrying in 120s ---" >>"$LOG"
        # Free any stale DDP shared memory / zombie ranks before relaunching.
        pkill -9 -f "train_visionzip_kd_lora" 2>/dev/null
        sleep 120
    done

    echo "=== [$SRC] KD GAVE UP after $MAX_ATTEMPTS attempts $(date -Is) ===" >>"$DRIVER_LOG"
    return 1
}

# SG first: the handoff predicts the gaze complement pays off most on SG's gaze-driven
# tasks, so this is where KD is worth it. EG is small (1265 train items) and cheap.
run_spec sg "$M1_SGONLY" 29671
run_spec eg "$M1_EGONLY" 29672

echo "=== ALL SPECIALIST KD RUNS FINISHED $(date -Is) ===" >>"$DRIVER_LOG"
