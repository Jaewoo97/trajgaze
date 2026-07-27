#!/bin/bash
# Repeat the SG-only evals to average out eval-time nondeterminism.
#
# The same checkpoint, same machine, same flags produced 71.67 then 71.29 on the
# M1 SG-only teacher (377 vs 375 items, both n=526, no items skipped). So a single
# eval carries ~±2 items of noise from bf16/flash-attn nondeterminism alone,
# before any training variance. Table values must be averaged over repeats.
#
# Every repeat goes through the SAME --eval-ckpt code path, so the samples are
# comparable; the student's 70.15 came from the in-training eval loop and is kept
# out of the average for that reason.
#
#   nohup setsid scripts/run_repeat_evals.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

SGSTU=$REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_SGonly_overlay/best.pth

# $1=tag $2=ckpt $3=rep $4=gpu $5=port
one () {
    local LOG="$REPO/repeat_$1_r$3.log"
    echo "=== $1 repeat $3 start $(date -Is) ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=$4 $TORCHRUN --nproc_per_node=1 --master_port=$5 \
      -m TrajGazeMerge.training.train_visionzip_complement_lora \
      --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio 0.07 --traj-ratio 0.03 --source sg \
      --eval-ckpt "$2" \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_rep_$1_$3" \
      --no-hdepic >>"$LOG" 2>&1
    echo "=== $1 repeat $3 DONE $(date -Is) ===" >>"$LOG"
}

# The KD student is a student checkpoint (lora_state + pred_state) and must be
# scored by the KD trainer, which drives selection from the predictor.
onestu () {
    local LOG="$REPO/repeat_student_r$1.log"
    echo "=== student repeat $1 start $(date -Is) ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=$2 $TORCHRUN --nproc_per_node=1 --master_port=$3 \
      -m TrajGazeMerge.training.train_visionzip_kd_lora \
      --eval-ckpt "$SGSTU" --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio 0.07 --traj-ratio 0.03 --source sg \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_rep_student_$1" \
      --no-hdepic >>"$LOG" 2>&1
    echo "=== student repeat $1 DONE $(date -Is) ===" >>"$LOG"
}

# Round 1 — teacher rep3 (we already have 2 samples) + student rep1
one teacher "$M1_SGONLY" 3 0 29801 & P1=$!
onestu 1 1 29802 & P2=$!
wait $P1 $P2

# Round 2 — student rep2 and rep3
onestu 2 0 29803 & P3=$!
onestu 3 1 29804 & P4=$!
wait $P3 $P4

echo "=== ALL REPEAT EVALS DONE $(date -Is) ===" >>"$REPO/repeat_teacher_r3.log"
