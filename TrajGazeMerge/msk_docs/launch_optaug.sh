#!/bin/bash
# Train-time option-order augmentation on M1 (target option-position bias).
# Same M1 config (7%C∪3%G topk, learned encoder, 128f, 3ep) + --option-aug.
# Eval stays original-order. GPU 1, parallel to signrouted(GPU2).
set -u
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
out=$R/m1_optaug
mkdir -p $out
PY="conda run --no-capture-output -n trajgaze torchrun --nproc_per_node=1 --master_port=29699"
MOD=TrajGazeMerge.training.train_visionzip_complement_lora
cd /workspace/trajgaze_st
GAZE_OVERLAY=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 $PY -m $MOD \
  --output-dir $out --option-aug \
  --content-ratio 0.07 --traj-ratio 0.03 --complement-mode topk \
  --traj-pool-mode learned --epochs 3 --lr 1e-4 --grad-accum 4 --no-hdepic \
  > $out/run.log 2>&1
echo "rc=$? done optaug" >> $out/run.log
