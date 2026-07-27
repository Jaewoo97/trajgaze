#!/bin/bash
# §1 of docs/kd_handoff.md: eval the two SEPARATELY-TRAINED M1 teachers, each on its
# own benchmark (SG-only → SG n=526, EG-only → EG n=485). These are the baselines the
# §3 specialist KD students are measured against.
#
# The eval-only path (--eval-ckpt) scores on rank 0 only, so 1 GPU per job is correct;
# the two jobs run concurrently on GPU 0 and GPU 1.
#
#   nohup setsid scripts/run_m1_specialist_evals.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

run_eval () {          # $1=source  $2=teacher ckpt  $3=gpu  $4=master_port  $5=log
    echo "=== M1 $1-only teacher eval start $(date -Is) ===" >>"$5"
    CUDA_VISIBLE_DEVICES=$3 $TORCHRUN --nproc_per_node=1 --master_port=$4 \
      -m TrajGazeMerge.training.train_visionzip_complement_lora \
      --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio 0.07 --traj-ratio 0.03 --source "$1" \
      --eval-ckpt "$2" \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_eval_m1_$1only" \
      --no-hdepic --eval-progress-every 50 >>"$5" 2>&1
    echo "=== M1 $1-only EVAL DONE $(date -Is) ===" >>"$5"
}

run_eval sg "$M1_SGONLY" 0 29711 "$REPO/eval_m1_sgonly.log" &
PID_SG=$!
run_eval eg "$M1_EGONLY" 1 29712 "$REPO/eval_m1_egonly.log" &
PID_EG=$!

wait $PID_SG $PID_EG
echo "=== BOTH SPECIALIST EVALS DONE $(date -Is) ===" >>"$REPO/eval_m1_specialists.log"
