#!/bin/bash
# Runs 2 and 3 of docs/kd_handoff plan: two single-flag variants of the §6.2 joint KD
# baseline (61.33). Protocol is byte-identical to scripts/run_kd_train.sh — 2 GPUs,
# grad-accum 4 (eff-batch 8), --epochs 3 --early-stop, same warm-start, same stage1 —
# so any delta is attributable to the one flag that changed.
#
#   run2  --balance-sources : SG/EG contribute equally many steps per epoch. Epoch size
#                             is UNCHANGED (3532 each vs 5799/1265), so this is a mix
#                             change, not a longer run.
#   run3  --freeze-lora     : predictor-only distillation, LoRA held at teacher quality.
#
# Sequential: each needs both GPUs to keep eff-batch 8 comparable.
#
#   nohup setsid scripts/run_kd_experiments.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

# Warm-start from the MACHINE-1 KD student (62.41 re-evaluated here), not M1 joint.
# Local training of this student lands at 61.33 — 11 items worse, essentially all EG —
# so starting from the sound checkpoint avoids inheriting that regression. Both
# lora_state and pred_state are carried over, so the run begins at 62.41, not below it.
# Caveat for the writeup: this stacks epochs on top of an already-trained student, so
# any gain is NOT attributable to the flag alone (the §6.3 confound).
M1_STUDENT=/NHNHOME/VILAB/vilab_yj/datasets/trajgazemerge/hf_m1/aaai/visionzip_kd_selection_overlay/best.pth

MAX_ATTEMPTS=20

run_variant () {          # $1=name  $2=extra flags
    local OUT="$REPO/TrajGazeMerge/checkpoints/visionzip_kd_$1"
    local LOG="$REPO/kd_train_$1.log"
    mkdir -p "$OUT"
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        echo "=== [$1] attempt $attempt  $(date -Is) ===" >>"$LOG"
        CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29661 \
          -m TrajGazeMerge.training.train_visionzip_kd_lora \
          --warmstart-ckpt "$M1_STUDENT" \
          --stage1-ckpt    "$STAGE1_CKPT" \
          --output-dir     "$OUT" \
          --epochs 3 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 \
          --no-hdepic --early-stop --resume $2 >>"$LOG" 2>&1
        if grep -q "TRAINING COMPLETE" "$LOG"; then
            echo "=== [$1] DONE $(date -Is) ===" >>"$LOG"
            return 0
        fi
        echo "--- [$1] attempt $attempt ended without completion; retrying in 120s ---" >>"$LOG"
        pkill -9 -f "train_visionzip_kd_lora" 2>/dev/null
        sleep 120
    done
    echo "=== [$1] GAVE UP after $MAX_ATTEMPTS attempts $(date -Is) ===" >>"$LOG"
    return 1
}

run_variant balanced_overlay  "--balance-sources"
run_variant frozenlora_overlay "--freeze-lora"
echo "=== ALL KD EXPERIMENTS DONE $(date -Is) ===" >>"$REPO/kd_train_balanced_overlay.log"
