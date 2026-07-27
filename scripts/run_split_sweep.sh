#!/bin/bash
# Content/complement split sweep on the MACHINE-1 KD student (62.41 here at 7/3).
# Eval-only: nothing is trained, so the checkpoint cannot be damaged.
#
# Total visual-token budget stays 10%, selection stays hard top-k, student inputs stay
# RGB-only, one joint model — only how the 10% is divided between VisionZip content and
# the distilled complement changes. The student's complement agrees with the gaze teacher
# only ~41% of the time, so those tokens may be worth less than simply spending the
# budget on content; this measures that directly.
#
#   10/0 is the ablation the original doc never ran: this exact readout with NO
#   complement at all. (k = max(1, ...) forces one token, so it is 10% + 1 token.)
#
#   nohup setsid scripts/run_split_sweep.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

M1_STUDENT=/NHNHOME/VILAB/vilab_yj/datasets/trajgazemerge/hf_m1/aaai/visionzip_kd_selection_overlay/best.pth

run_split () {          # $1=content  $2=traj  $3=gpu  $4=port  $5=tag
    local LOG="$REPO/eval_split_$5.log"
    echo "=== split $5 (content=$1 traj=$2) start $(date -Is) ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=$3 $TORCHRUN --nproc_per_node=1 --master_port=$4 \
      -m TrajGazeMerge.training.train_visionzip_kd_lora \
      --eval-ckpt "$M1_STUDENT" \
      --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio "$1" --traj-ratio "$2" --source both \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_eval_split_$5" \
      --no-hdepic >>"$LOG" 2>&1
    echo "=== split $5 DONE $(date -Is) ===" >>"$LOG"
}

# Two GPUs → two points concurrently.
run_split 0.08 0.02 0 29731 "8_2" & P1=$!
run_split 0.09 0.01 1 29732 "9_1" & P2=$!
wait $P1 $P2

run_split 0.10 0.00 0 29733 "10_0" & P3=$!
# 6/4 probes the other direction: does MORE complement help, or is 7/3 already past peak?
run_split 0.06 0.04 1 29734 "6_4" & P4=$!
wait $P3 $P4

echo "=== SPLIT SWEEP DONE $(date -Is) ===" >>"$REPO/eval_split_8_2.log"
