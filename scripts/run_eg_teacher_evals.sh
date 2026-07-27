#!/bin/bash
# Phase 1: two per-task evals on EgoGazeVQA (egtea, n=485), one GPU each, in parallel.
#
# 1a  JOINT teacher on EG  — never measured per-task before. The paper row
#     "Temp 42.50 / Caus 84.57 / Avg 56.29" cannot be read off any system on record:
#     42.50 is the EG SPECIALIST STUDENT's temporal, 84.57 is the JOINT STUDENT's
#     causal, 56.29 is the JOINT TEACHER's Avg (machine 1). If that row really is
#     the joint teacher, this eval must return
#         Spat 41.72 (68) / Temp 42.50 (68) / Caus 84.57 (137) / Avg 56.29 (273),
#     the only per-task split consistent with all four printed values.
#     Note v1 §6.1 already records this same checkpoint as 56.29 (m1) -> 55.67 (270)
#     here, a 3-item gap = the documented noise floor.
#
# 1b  EG-ONLY specialist teacher — second per-task sample. v2 §10 has one; §8 wants >=3.
#     Prior Avg samples on this machine: 53.81 (2026-07-26), 54.23 (today).
#
# Teachers keep the gaze overlay (v2 §7.3), so GAZE_OVERLAY=1 and VLM_GAZE_OVERLAY unset.
#
#   nohup setsid scripts/run_eg_teacher_evals.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze || exit 1
source env.sh
unset VLM_GAZE_OVERLAY

one () {           # $1=tag  $2=ckpt  $3=gpu  $4=port
    local LOG="$REPO/eval_${1}_eg_pertask.log"
    echo "=== $1 on EG, per-task $(date -Is) ===" >>"$LOG"
    echo "=== ckpt: $2" >>"$LOG"
    CUDA_VISIBLE_DEVICES=$3 $TORCHRUN --nproc_per_node=1 --master_port=$4 \
      -m TrajGazeMerge.training.train_visionzip_complement_lora \
      --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio 0.07 --traj-ratio 0.03 --source eg \
      --eval-ckpt "$2" \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_eval_${1}_eg_pertask" \
      --no-hdepic --eval-progress-every 100 >>"$LOG" 2>&1
    echo "=== $1 DONE $(date -Is) ===" >>"$LOG"
}

one jointteacher    "$M1_JOINT"   0 29831 &  P1=$!
one egonlyteacher_r2 "$M1_EGONLY" 1 29832 &  P2=$!
wait $P1 $P2
echo "=== BOTH EG TEACHER EVALS DONE $(date -Is) ===" >>"$REPO/eval_jointteacher_eg_pertask.log"
