#!/usr/bin/env bash
# Chained launcher: wait for the running QC-Gate overlay 4-GPU run to finish,
# then launch the plain VisionZip baseline on the SAME consistent-overlay 3-way
# data, matched protocol (eff. batch 8 = 4 GPUs x grad-accum 2, 3 ep, lr 1e-4,
# 10% tokens). Only difference vs the original VZ-3way run: overlay frames.
set -u
cd /workspace/trajgaze_st
LOG=/tmp/vz_overlay_train.log
PYBIN=/opt/conda/envs/trajgaze/bin
export PATH="$PYBIN:$PATH"

echo "[chain] $(date) waiting for QC-Gate overlay run to release GPUs..." >> "$LOG"
while pgrep -f "train_visionzip_qcgate_lora_3way" >/dev/null 2>&1; do
  sleep 60
done
echo "[chain] $(date) QC-Gate done; launching VZ-overlay baseline." >> "$LOG"

export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYBIN/torchrun" --nproc_per_node=4 --master_port=29833 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_lora_3way_overlay \
  --epochs 3 --lr 1e-4 --grad-accum 2 >> "$LOG" 2>&1
rc=$?
echo "[chain] $(date) VZ-overlay finished, exit=$rc" >> "$LOG"
echo "VZ_OVERLAY: COMPLETE" >> "$LOG"
