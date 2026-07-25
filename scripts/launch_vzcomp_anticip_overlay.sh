#!/usr/bin/env bash
# Method 2 — VisionZip + COMPLEMENTARY anticipatory-tube selection.
# 10% budget = 7% VisionZip content ∪ 3% top-k by a PARAMETER-FREE anticipatory
# score: gaze/hand positions extrapolated forward by velocity, Gaussian "tube"
# over current ∪ predicted-future anchors (gaze leads action). LoRA-only,
# identical protocol: 2 GPUs (2,3), eff-batch 1*4*2 = 8, 3 epochs with
# epoch1->epoch2 early-stop, gaze-overlay input, egtea 1011 val.
set -u
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=2,3
LOG=/tmp/vzcomp_anticip_overlay.log
OUT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_anticip_overlay
mkdir -p "$OUT"
echo "[vzcomp-anticip] $(date) launch (2 GPU, gaze-overlay, egtea val, early-stop)" >> "$LOG"
torchrun --nproc_per_node=2 --master_port=29655 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode anticipatory --horizon 2.0 \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval >> "$LOG" 2>&1
echo "[vzcomp-anticip] $(date) exit=$? :: VZCOMP_ANTICIP_DONE" >> "$LOG"
