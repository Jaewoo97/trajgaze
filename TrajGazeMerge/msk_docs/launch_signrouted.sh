#!/bin/bash
# Direction C: sign-routed gaze lean (object→confidence, spatial/temporal→inverse,
# else→none). Single LoRA, 1 pass, budget unchanged. Full 3ep (no early-stop) for
# fair comparison vs task_adaptive_3ep (63.20). GPU 2.
set -u
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
out=$R/gazeconf_signrouted
mkdir -p $out
PY="conda run -n trajgaze torchrun --nproc_per_node=1 --master_port=29691"
MOD=TrajGazeMerge.training.train_visionzip_gazeconf_lora
cd /workspace/trajgaze_st
GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=2 $PY -m $MOD \
  --arm signrouted --output-dir $out \
  --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic \
  > $out/run.log 2>&1
echo "rc=$? done signrouted" >> $out/run.log
