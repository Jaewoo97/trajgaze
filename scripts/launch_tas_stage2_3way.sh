#!/usr/bin/env bash
# TAS Stage-2 LoRA — 4-GPU DDP on the 3-way combined data (StreamGaze +
# EgoGazeVQA + HD-EPIC, P09 held out as test), MATCHED overlay protocol:
#   eff. batch 8 = 4 GPUs x grad-accum 2 x micro-batch 1, lr 1e-4, 10% tokens,
#   GAZE_OVERLAY=1 — identical optimization to the tbudget/VisionZip rows, so
#   the only variable vs those rows is the token-selection method (TAS).
# Selection rule: TrajGazeV2Temporal encoder -> (B,T,196) scores ->
#   score_to_qwen_spatiotemporal -> gaze_weighted_merge (10% receivers).
# Stage-1 encoder: stage1_tas_3way_overlay/best.pth (gaze-complete 3-way).
# 3 epochs, no-KD (CE on answer letter only). In-script per-source eval every
# 400 microsteps; saves best.pth (mean-acc) + epoch_0N.pth.
set -u
cd /workspace/trajgaze_st
PYBIN=/opt/conda/envs/trajgaze/bin
export PATH="$PYBIN:$PATH"
STAGE1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
CKDIR=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tas_lora_3way_overlay
TRLOG=/tmp/tas_stage2_3way_overlay_train.log

mkdir -p "$CKDIR"

if [[ ! -f "$STAGE1" ]]; then
  echo "[tas-s2] $(date) MISSING Stage-1 ckpt: $STAGE1 — aborting" >> "$TRLOG"
  echo "TAS_STAGE2_3WAY: ABORTED (no stage1 ckpt)" >> "$TRLOG"
  exit 2
fi

echo "[tas-s2] $(date) launching 4-GPU TAS Stage-2 (eff-batch 8, 3 ep, overlay, matched)" >> "$TRLOG"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYBIN/torchrun" --nproc_per_node=4 --master_port=29829 \
  -m TrajGazeMerge.training.train_merge_lora_tas_3way \
  --model-type full \
  --stage1-ckpt "$STAGE1" \
  --output-dir "$CKDIR" \
  --epochs 3 --merge-ratio 0.9 \
  --micro-batch 1 --grad-accum 2 \
  --lr-lora 1e-4 \
  --dataloader-num-workers 8 --eval-every 3700 >> "$TRLOG" 2>&1
rc=$?
echo "[tas-s2] $(date) Stage-2 finished, exit=$rc" >> "$TRLOG"
echo "TAS_STAGE2_3WAY: COMPLETE (exit=$rc)" >> "$TRLOG"
