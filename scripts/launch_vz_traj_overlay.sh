#!/usr/bin/env bash
# VZ-traj (Recipe 1+3, gaze/hand-anchored selection) — 4-GPU overlay training on
# the consistent all-overlay 3-way data, current matched protocol:
#   eff. batch 8 = 4 GPUs x grad-accum 2 x bs1, lr 1e-4, 10% tokens, GAZE_OVERLAY=1.
# 3 epochs (MATCHED to plain-VZ/tbudget/TAS overlay rows, all epochs=3). Per-epoch
# eval on the full 3-way val n=4936; saves best.pth + epoch_0N.pth. Mirrors the
# tbudget/VZ overlay launches (eff-batch 8, lr 1e-4, grad-accum 2).
set -u
cd /workspace/trajgaze_st
PYBIN=/opt/conda/envs/trajgaze/bin
export PATH="$PYBIN:$PATH"
CKDIR=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_traj_lora_3way_overlay
TRLOG=/tmp/vz_traj_overlay_train.log

mkdir -p "$CKDIR"

echo "[vz-traj] $(date) launching 4-GPU VZ-traj overlay training (3 epochs, per-epoch eval)" >> "$TRLOG"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYBIN/torchrun" --nproc_per_node=4 --master_port=29843 \
  -m TrajGazeMerge.training.train_visionzip_traj_lora_3way \
  --output-dir "$CKDIR" \
  --epochs 3 --lr 1e-4 --grad-accum 2 >> "$TRLOG" 2>&1
rc=$?
echo "[vz-traj] $(date) training finished, exit=$rc" >> "$TRLOG"
echo "VZ_TRAJ_OVERLAY: COMPLETE (exit=$rc)" >> "$TRLOG"
