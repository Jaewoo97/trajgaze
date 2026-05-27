#!/usr/bin/env bash
# Direction A pipeline: sanity → CF-1 (GPU 0) + CF-3 (GPU 1) in parallel.
#
# Phase 1 — sanity: short CF-3 run on GPU 0 with low warmup, until mstep>=100.
#   Validates: CE doesn't explode, loss_cf_mask and loss_shuf are logged and non-zero.
# Phase 2 — real runs: CF-1 (mask only) on GPU 0, CF-3 (mask+shuf) on GPU 1, 3 epochs.
#
# Outputs:
#   $REPO/TrajGazeMerge/eval_results/E1_cf_pipeline.log    — orchestrator status
#   $REPO/TrajGazeMerge/checkpoints/E1_cf3_sanity/         — sanity run (deleted on PASS)
#   $REPO/TrajGazeMerge/checkpoints/E1_combined_cf1_hdepic_bs8_mb2/best.pth
#   $REPO/TrajGazeMerge/checkpoints/E1_combined_cf3_hdepic_bs8_mb2/best.pth

set -uo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
STAGE1=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth
PIPE_LOG=$REPO/TrajGazeMerge/eval_results/E1_cf_pipeline.log

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$PIPE_LOG"; }

mkdir -p $REPO/TrajGazeMerge/eval_results
echo "" > $PIPE_LOG
log "=== Direction A pipeline start (pid=$$) ==="

# ── Phase 1: sanity ────────────────────────────────────────────────────────
SANITY_OUT=$REPO/TrajGazeMerge/checkpoints/E1_cf3_sanity
SANITY_LOG=$SANITY_OUT/train_log.jsonl
mkdir -p $SANITY_OUT
rm -f $SANITY_LOG  # fresh

log "Phase 1: CF-3 sanity on GPU 0 (warmup=30, target mstep=100)"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
nohup $PY -m TrajGazeMerge.training.train_merge_lora_batched \
    --model-type full --stage1-ckpt "$STAGE1" \
    --output-dir "$SANITY_OUT" \
    --epochs 1 --merge-ratio 0.9 --micro-batch 2 --grad-accum 4 \
    --use-egovqa --use-hd-epic --eval-egovqa-egtea --eval-hd-epic \
    --dataloader-num-workers 8 --eval-every 999999 --log-every 10 \
    --shuffle-aug --shuffle-prob 0.3 --shuffle-margin 1.0 --shuffle-lambda 0.3 --shuffle-warmup-steps 30 \
    --use-cf-mask --cf-mask-prob 0.3 --cf-mask-margin 1.0 --cf-mask-lambda 0.3 --cf-mask-warmup-steps 30 \
    > $SANITY_OUT/stdout.log 2>&1 &
SANITY_PID=$!
log "sanity launched PID=$SANITY_PID"

# Wait until mstep >= 100, or 45 min timeout
TIMEOUT=2700
START=$SECONDS
while true; do
    if [ -f "$SANITY_LOG" ]; then
        MAX_STEP=$($PY -c "
import json
m=0
try:
    with open('$SANITY_LOG') as f:
        for line in f:
            d=json.loads(line)
            if d.get('type') != 'eval' and 'step' in d:
                m = max(m, d['step'])
except: pass
print(m)
" 2>/dev/null || echo 0)
        if [ "$MAX_STEP" -ge 100 ]; then
            log "sanity reached mstep=$MAX_STEP"
            break
        fi
    fi
    if ! kill -0 $SANITY_PID 2>/dev/null; then
        log "sanity DIED (PID gone before mstep 100)"
        log "stdout tail:"
        tail -20 $SANITY_OUT/stdout.log | tee -a $PIPE_LOG
        exit 2
    fi
    if [ $((SECONDS - START)) -ge $TIMEOUT ]; then
        log "sanity TIMEOUT after ${TIMEOUT}s (max_step=$MAX_STEP)"
        kill $SANITY_PID 2>/dev/null
        exit 3
    fi
    sleep 30
done

# Kill sanity + orphan workers
log "stopping sanity"
kill $SANITY_PID 2>/dev/null || true
sleep 5
pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" | xargs -r kill 2>/dev/null
sleep 15
ORPH=$(pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" | tr '\n' ' ')
if [ -n "$ORPH" ]; then
    log "SIGKILL orphans: $ORPH"
    echo "$ORPH" | xargs -r kill -9 2>/dev/null
fi
sleep 5

# Health check
HEALTH=$($PY -c "
import json
ces=[]; cfs=[]; shufs=[]
with open('$SANITY_LOG') as f:
    for line in f:
        d=json.loads(line)
        if d.get('type') == 'eval': continue
        if 'loss' in d and d['loss'] is not None: ces.append(d['loss'])
        if d.get('loss_cf_mask') is not None: cfs.append(d['loss_cf_mask'])
        if d.get('loss_shuf') is not None: shufs.append(d['loss_shuf'])
import math
if not ces:
    print('FAIL no_ce'); exit()
if math.isnan(ces[-1]) or math.isinf(ces[-1]):
    print(f'FAIL ce_nan_or_inf {ces[-1]}'); exit()
if ces[-1] > 5.0:
    print(f'FAIL ce_too_high {ces[-1]:.3f}'); exit()
if len(cfs) < 2:
    print(f'FAIL cf_mask_not_logged n={len(cfs)}'); exit()
if len(shufs) < 2:
    print(f'FAIL shuf_not_logged n={len(shufs)}'); exit()
print(f'OK n_ce={len(ces)} ce_first={ces[0]:.3f} ce_last={ces[-1]:.3f} n_cf={len(cfs)} cf_last={cfs[-1]:.4f} n_shuf={len(shufs)} shuf_last={shufs[-1]:.4f}')
")
log "sanity health: $HEALTH"

if [[ "$HEALTH" == FAIL* ]]; then
    log "=== ABORT pipeline due to sanity failure ==="
    exit 4
fi

log "Phase 1 PASS"

# ── Phase 2: launch CF-1 + CF-3 in parallel ─────────────────────────────────
CF1_OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf1_hdepic_bs8_mb2
CF3_OUT=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf3_hdepic_bs8_mb2
mkdir -p $CF1_OUT $CF3_OUT

log "Phase 2: launching CF-1 (GPU 0, mask only)"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
nohup $PY -m TrajGazeMerge.training.train_merge_lora_batched \
    --model-type full --stage1-ckpt "$STAGE1" \
    --output-dir "$CF1_OUT" \
    --epochs 3 --merge-ratio 0.9 --micro-batch 2 --grad-accum 4 \
    --use-egovqa --use-hd-epic --eval-egovqa-egtea --eval-hd-epic \
    --dataloader-num-workers 8 --eval-every 400 \
    --use-cf-mask --cf-mask-prob 0.3 --cf-mask-margin 1.0 --cf-mask-lambda 0.3 --cf-mask-warmup-steps 600 \
    > $CF1_OUT/stdout.log 2>&1 &
CF1_PID=$!
log "CF-1 launched PID=$CF1_PID"

sleep 30   # stagger so they don't both load processor concurrently

log "Phase 2: launching CF-3 (GPU 1, mask + shuf)"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO \
nohup $PY -m TrajGazeMerge.training.train_merge_lora_batched \
    --model-type full --stage1-ckpt "$STAGE1" \
    --output-dir "$CF3_OUT" \
    --epochs 3 --merge-ratio 0.9 --micro-batch 2 --grad-accum 4 \
    --use-egovqa --use-hd-epic --eval-egovqa-egtea --eval-hd-epic \
    --dataloader-num-workers 8 --eval-every 400 \
    --shuffle-aug --shuffle-prob 0.3 --shuffle-margin 1.0 --shuffle-lambda 0.3 --shuffle-warmup-steps 600 \
    --use-cf-mask --cf-mask-prob 0.3 --cf-mask-margin 1.0 --cf-mask-lambda 0.3 --cf-mask-warmup-steps 600 \
    > $CF3_OUT/stdout.log 2>&1 &
CF3_PID=$!
log "CF-3 launched PID=$CF3_PID"

log "=== pipeline orchestrator complete. Watch:"
log "  tail -f $PIPE_LOG"
log "  tail -f $CF1_OUT/stdout.log"
log "  tail -f $CF3_OUT/stdout.log"
log "CF-1 PID=$CF1_PID  CF-3 PID=$CF3_PID"

# Write PIDs for later watcher
echo "CF1_PID=$CF1_PID" > $REPO/TrajGazeMerge/eval_results/E1_cf_pids.env
echo "CF3_PID=$CF3_PID" >> $REPO/TrajGazeMerge/eval_results/E1_cf_pids.env
