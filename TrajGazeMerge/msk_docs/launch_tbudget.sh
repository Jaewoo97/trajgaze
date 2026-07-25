#!/usr/bin/env bash
set -u
cd /workspace/trajgaze_st
PY=/opt/conda/envs/trajgaze/bin
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
HB=$R/foveal_launch.log
run () {  # tag w gpu port
  local tag=$1 w=$2 gpu=$3 port=$4 out=$R/tbudget_w${tag}_2way
  mkdir -p $out
  echo "[tbudget] $(date) START w=$w GPU=$gpu -> $out/run.log" >> $HB
  CUDA_VISIBLE_DEVICES=$gpu GAZE_OVERLAY=1 $PY/torchrun --nproc_per_node=1 --master_port=$port \
    -m TrajGazeMerge.training.train_visionzip_tbudget_lora_3way \
    --traj-weight $w --no-hdepic --epochs 2 --lr 1e-4 --grad-accum 8 \
    --output-dir $out > $out/run.log 2>&1
  echo "[tbudget] $(date) EXIT w=$w rc=$? (Overall: $(grep -c Overall: $out/run.log 2>/dev/null))" >> $HB
}
run 10 1.0 0 29730 &
run 00 0.0 2 29732 &
run 05 0.5 3 29734 &
wait
echo "[tbudget] $(date) ALL 3 weights DONE" >> $HB
