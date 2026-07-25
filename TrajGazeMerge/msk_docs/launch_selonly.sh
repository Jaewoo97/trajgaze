#!/bin/bash
# Selection-only single-LoRA test: M1 LoRA + per-task selection switching.
set -u
ROOT=/workspace/trajgaze_st
CKPTS=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
M1=$CKPTS/visionzip_complement_learned_overlay/best.pth
DUMPS=$CKPTS/dumps
cd $ROOT
run () { # gpu arm dumpname
  CUDA_VISIBLE_DEVICES=$1 nohup conda run -n trajgaze python -u -m TrajGazeMerge.eval.eval_dump_gazeconf \
    --ckpt $M1 --arm $2 --dump $DUMPS/$3.jsonl --gpu 0 \
    > $DUMPS/$3.log 2>&1 &
  echo "launched arm=$2 gpu=$1 pid=$! -> $DUMPS/$3.log"
}
run 0 none          selonly_m1_none
run 1 task_adaptive selonly_m1_adaptive
