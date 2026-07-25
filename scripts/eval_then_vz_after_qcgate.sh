#!/usr/bin/env bash
# Chained: wait for the QC-Gate overlay run to release the GPUs, then
#   (1) run the per-source full eval (SG/EG/HD/Combined) on each epoch ckpt
#       in parallel (one GPU each, GAZE_OVERLAY=1 to match training), THEN
#   (2) launch the plain VisionZip overlay baseline on the freed 4 GPUs,
#       matched protocol (eff batch 8 = 4 GPUs x grad-accum 2, 3 ep, lr 1e-4).
set -u
cd /workspace/trajgaze_st
PYBIN=/opt/conda/envs/trajgaze/bin
export PATH="$PYBIN:$PATH"
CKDIR=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/qcgate_lora_3way_overlay
EVLOG=/tmp/qcgate_persrc_eval.log
VZLOG=/tmp/vz_overlay_train.log

echo "[chain2] $(date) waiting for QC-Gate overlay run to release GPUs..." >> "$EVLOG"
while pgrep -f "train_visionzip_qcgate_lora_3way" >/dev/null 2>&1; do
  sleep 60
done
echo "[chain2] $(date) QC-Gate done; starting per-source eval." >> "$EVLOG"

# ---- (1) per-source full eval, one ckpt per GPU, in parallel ----
declare -A GPU=( [epoch_01]=0 [epoch_02]=1 [epoch_03]=2 )
pids=()
for ep in epoch_01 epoch_02 epoch_03; do
  ck="$CKDIR/$ep.pth"
  if [[ ! -f "$ck" ]]; then
    echo "[chain2] $(date) MISSING $ck — skipping" >> "$EVLOG"
    continue
  fi
  echo "[chain2] $(date) eval $ep on GPU ${GPU[$ep]}" >> "$EVLOG"
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=${GPU[$ep]} \
    "$PYBIN/python" scripts/eval_qcgate_persource.py \
      --ckpt "$ck" --gpu 0 \
      --out "$CKDIR/$ep.full_eval.json" >> "$EVLOG" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[chain2] $(date) per-source eval COMPLETE" >> "$EVLOG"
echo "QCGATE_PERSRC_EVAL: COMPLETE" >> "$EVLOG"

# ---- (2) VZ overlay baseline on freed 4 GPUs (unchanged matched protocol) ----
echo "[chain2] $(date) launching VZ-overlay baseline." >> "$VZLOG"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYBIN/torchrun" --nproc_per_node=4 --master_port=29833 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_lora_3way_overlay \
  --epochs 3 --lr 1e-4 --grad-accum 2 >> "$VZLOG" 2>&1
rc=$?
echo "[chain2] $(date) VZ-overlay finished, exit=$rc" >> "$VZLOG"
echo "VZ_OVERLAY: COMPLETE" >> "$VZLOG"
