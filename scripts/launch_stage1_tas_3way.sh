#!/usr/bin/env bash
# 3-way Stage-1 TAS pretrain (TrajGazeV2Temporal encoder, trajectory-anchored)
# on the gaze-COMPLETE combined data:
#   StreamGaze(egoexolearn+holoassist) + EgoGazeVQA(ego4d+egoexo) + HD-EPIC(P01-P08).
# HD-EPIC P09 is held out (VAL_PARTICIPANT). HD-EPIC now has full projected gaze
# (156/156 JSONs in /workspace/HD-EPIC/gaze), so the degenerate-trajectory issue
# that killed the May-27 stage1_tas_trajanchor_3way run at epoch 13 is resolved.
# Matched protocol: 4-GPU DDP, 100 epochs, lr 3e-4, batch/GPU 2, 128 frames.
set -u
cd /workspace/trajgaze_st
PYBIN=/opt/conda/envs/trajgaze/bin
export PATH="$PYBIN:$PATH"
CKDIR=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay
TRLOG=/tmp/stage1_tas_3way_train.log

mkdir -p "$CKDIR"

echo "[stage1-tas] $(date) launching 4-GPU 3-way Stage-1 TAS pretrain (100 ep, trajanchor, gaze-complete)" >> "$TRLOG"
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYBIN/torchrun" --nproc_per_node=4 --master_port=29827 \
  -m TrajGaze_v2.training.stage1_temporal \
  --output-dir "$CKDIR" \
  --epochs 100 --lr 3e-4 --batch-size 2 \
  --use-egovqa --use-hd-epic --use-trajectory-anchor >> "$TRLOG" 2>&1
rc=$?
echo "[stage1-tas] $(date) Stage-1 finished, exit=$rc" >> "$TRLOG"
echo "STAGE1_TAS_3WAY: COMPLETE (exit=$rc)" >> "$TRLOG"
