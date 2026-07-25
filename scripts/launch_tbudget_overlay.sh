#!/usr/bin/env bash
# Temporal-Budget VisionZip — 4-GPU overlay training on the 3-way combined data,
# matched protocol (eff. batch 8 = 4 GPUs x grad-accum 2, 3 ep, lr 1e-4, 10%
# tokens, GAZE_OVERLAY=1). Refined trajectory utilizer: VZ within-frame selection
# unchanged; per-frame token budget reallocated by gaze/hand temporal interaction.
# After training, runs per-source full eval (SG/EG/HD/Combined) on each epoch
# ckpt in parallel (one GPU each), matching the QC-Gate / VZ comparison rows.
set -u
cd /workspace/trajgaze_st
PYBIN=/opt/conda/envs/trajgaze/bin
export PATH="$PYBIN:$PATH"
CKDIR=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tbudget_lora_3way_overlay
TRLOG=/tmp/tbudget_overlay_train.log
EVLOG=/tmp/tbudget_persrc_eval.log

mkdir -p "$CKDIR"

# ---- (1) train ----
echo "[tbudget] $(date) launching 4-GPU temporal-budget overlay training" >> "$TRLOG"
export GAZE_OVERLAY=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
"$PYBIN/torchrun" --nproc_per_node=4 --master_port=29841 \
  -m TrajGazeMerge.training.train_visionzip_tbudget_lora_3way \
  --output-dir "$CKDIR" \
  --epochs 3 --lr 1e-4 --grad-accum 2 \
  --tau 1.0 --traj-weight 0.5 >> "$TRLOG" 2>&1
rc=$?
echo "[tbudget] $(date) training finished, exit=$rc" >> "$TRLOG"
echo "TBUDGET_TRAIN: COMPLETE (exit=$rc)" >> "$TRLOG"

# ---- (2) per-source full eval, one ckpt per GPU, in parallel ----
echo "[tbudget] $(date) starting per-source eval" >> "$EVLOG"
declare -A GPU=( [epoch_01]=0 [epoch_02]=1 [epoch_03]=2 )
pids=()
for ep in epoch_01 epoch_02 epoch_03; do
  ck="$CKDIR/$ep.pth"
  if [[ ! -f "$ck" ]]; then
    echo "[tbudget] $(date) MISSING $ck — skipping" >> "$EVLOG"
    continue
  fi
  echo "[tbudget] $(date) eval $ep on GPU ${GPU[$ep]}" >> "$EVLOG"
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=${GPU[$ep]} \
    "$PYBIN/python" scripts/eval_tbudget_persource.py \
      --ckpt "$ck" --gpu 0 \
      --out "$CKDIR/$ep.full_eval.json" >> "$EVLOG" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[tbudget] $(date) per-source eval COMPLETE" >> "$EVLOG"
echo "TBUDGET_PERSRC_EVAL: COMPLETE" >> "$EVLOG"
