#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID=3969430
REPO=/workspace/EgoGazeVQA
OUT=$REPO/TrajGazeMerge/checkpoints/eval_results
LORA=$REPO/TrajGazeMerge/checkpoints/pllava_baseline_lora/best_delta.pth
STAGE1=$REPO/TrajGaze_v2/checkpoints/stage1_temporal/best.pth

CUDA=3
PY=/opt/conda/envs/gaze/bin/python
LOG_PREFIX="$OUT/eval"

echo "[$(date)] Waiting for training PID $TRAIN_PID to finish ..."
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 30; done
echo "[$(date)] Training done. Starting zero-shot evals on GPU $CUDA."

# ── 1. PruneVid ────────────────────────────────────────────────────────────
echo "[$(date)] === Eval 1/3: PruneVid ===" | tee "$OUT/eval_prunevid.log"
CUDA_VISIBLE_DEVICES=$CUDA PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
$PY -m TrajGazeMerge.training.eval_pllava_prunevid \
    --lora-ckpt "$LORA" \
    --out "$OUT/pllava_prunevid.json" \
    >> "$OUT/eval_prunevid.log" 2>&1
echo "[$(date)] PruneVid done (exit $?)." | tee -a "$OUT/eval_prunevid.log"

# ── 2. PruMerge ────────────────────────────────────────────────────────────
echo "[$(date)] === Eval 2/3: PruMerge ===" | tee "$OUT/eval_prumerge.log"
CUDA_VISIBLE_DEVICES=$CUDA PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
$PY -m TrajGazeMerge.training.eval_pllava_prumerge \
    --lora-ckpt "$LORA" \
    --out "$OUT/pllava_prumerge.json" \
    >> "$OUT/eval_prumerge.log" 2>&1
echo "[$(date)] PruMerge done (exit $?)." | tee -a "$OUT/eval_prumerge.log"

# ── 3. TrajGaze Temporal ───────────────────────────────────────────────────
echo "[$(date)] === Eval 3/3: TrajGaze Temporal ===" | tee "$OUT/eval_trajgaze_temporal.log"
CUDA_VISIBLE_DEVICES=$CUDA PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
$PY -m TrajGazeMerge.training.eval_pllava_trajgaze_temporal \
    --lora-ckpt "$LORA" \
    --stage1-ckpt "$STAGE1" \
    --out "$OUT/pllava_trajgaze_temporal.json" \
    >> "$OUT/eval_trajgaze_temporal.log" 2>&1
echo "[$(date)] TrajGaze Temporal done (exit $?)." | tee -a "$OUT/eval_trajgaze_temporal.log"

echo "[$(date)] All 3 evals complete. Results in $OUT/"
