#!/bin/bash
# Fixation-confidence-weighted gaze complement — 4 arms over 2 GPUs (0, 2).
# Wave A: confidence (GPU0) ∥ inverse (GPU2)   ← decisive pair (beat-M1 + kill-test)
# Wave B: none (GPU0) ∥ random (GPU2)          ← in-protocol baseline + placebo
# gazesampled keeps GPUs 1,3 — do not touch.
set -u
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
HB=$R/gazeconf_launch.log
PY="conda run -n trajgaze torchrun --nproc_per_node=1"
MOD=TrajGazeMerge.training.train_visionzip_gazeconf_lora
COMMON="--epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop"

run_arm () {  # gpu, port, arm
  local gpu=$1 port=$2 arm=$3
  local out=$R/gazeconf_$arm
  mkdir -p $out
  echo "$(date -u) launch arm=$arm gpu=$gpu" >> $HB
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=$gpu $PY --master_port=$port \
    -m $MOD --arm $arm --output-dir $out $COMMON > $out/run.log 2>&1
  echo "$(date -u) done   arm=$arm gpu=$gpu rc=$?" >> $HB
}

echo "$(date -u) === gazeconf launcher start ===" > $HB
# Wave A
run_arm 0 29681 confidence &
PA=$!
run_arm 2 29682 inverse &
PB=$!
wait $PA $PB
echo "$(date -u) === Wave A complete ===" >> $HB
# Wave B
run_arm 0 29683 none &
PC=$!
run_arm 2 29684 random &
PD=$!
wait $PC $PD
echo "$(date -u) === Wave B complete — ALL DONE ===" >> $HB
