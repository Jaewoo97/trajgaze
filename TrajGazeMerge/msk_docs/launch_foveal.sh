#!/usr/bin/env bash
# ④ Foveated-ROI launcher (v2 — use free GPUs immediately).
# GPU 2,3 are free now (VZ curve done); M1 curve still holds GPU 0,1.
#   NOW:        gaze (GPU 2)  ∥  attn (GPU 3)   ← core contribution test, single-GPU
#   WHEN FREE:  random (GPU 0)                  ← placebo, after M1 curve frees GPU 0,1
# All arms: SINGLE-GPU + grad-accum 8  → eff-batch 8 and same #optimizer-steps as M1's
# 2-GPU DDP grad-accum 4 (gradient-equivalent), so gaze/attn/random share one protocol.
# Selection = M1 10% (content 0.07 ∪ traj 0.03, topk). Crop: crop_frac 0.35 margin 0.08.
# Detached (setsid) so it survives session close. Heartbeat: checkpoints/foveal_launch.log
set -u
cd /workspace/trajgaze_st

PY=/opt/conda/envs/trajgaze/bin
CKPT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
STAGE1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
HB=$CKPT/foveal_launch.log
mkdir -p "$CKPT"
echo "[launch v2] armed $(date)" >> "$HB"

run_arm () {  # $1=arm  $2=GPUS  $3=PORT  $4=nproc
  local arm=$1 gpus=$2 port=$3 np=$4
  local out=$CKPT/foveal_$arm
  mkdir -p "$out"
  echo "[launch] $(date) START arm=$arm GPU=$gpus np=$np -> $out/run.log" >> "$HB"
  CUDA_VISIBLE_DEVICES=$gpus GAZE_OVERLAY=1 \
    $PY/torchrun --nproc_per_node=$np --master_port=$port \
    -m TrajGazeMerge.training.train_visionzip_foveal_lora \
    --traj-pool-mode learned --complement-mode topk \
    --stage1-ckpt "$STAGE1" \
    --roi-arm "$arm" --roi-crop_frac 0.35 --roi-margin_frac 0.08 \
    --content-ratio 0.07 --traj-ratio 0.03 \
    --output-dir "$out" \
    --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop --no-mid-eval \
    > "$out/run.log" 2>&1
  echo "[launch] $(date) EXIT arm=$arm rc=$? (Overall: $(grep -c Overall: $out/run.log 2>/dev/null))" >> "$HB"
}

# --- Wave A: gaze ∥ attn on the free GPUs 2,3 (single-GPU each), NOW ---
run_arm gaze 2 29720 1 &
PA=$!
run_arm attn 3 29722 1 &
PB=$!

# --- random: wait for the M1 curve to free GPU 0,1, then launch on GPU 0 ---
for i in $(seq 1 360); do          # up to 360 min (covers M1 epoch 3 + eval if it doesn't early-stop)
  if ! pgrep -f "curve_m1_05pct" >/dev/null 2>&1; then
    echo "[launch] $(date) M1 curve gone after ${i}min -> launching random on GPU 0" >> "$HB"
    run_arm random 0 29724 1
    break
  fi
  sleep 60
done
if pgrep -f "curve_m1_05pct" >/dev/null 2>&1; then
  echo "[launch] $(date) WARN: M1 curve still running after 360min — random NOT launched" >> "$HB"
fi

wait $PA; wait $PB
echo "[launch] $(date) ALL ARMS DONE" >> "$HB"
