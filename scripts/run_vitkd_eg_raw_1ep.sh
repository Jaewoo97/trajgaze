#!/usr/bin/env bash
# eg_raw (setting 3) at ONE epoch per phase, run outside the main chain.
#
# Why this exists: run_vitkd_all.sh is strictly serial and puts eg_raw third, behind
# sg_ovl — ~10 h of queueing against a 4 h budget on 2026-07-29. sg_ovl was paused
# (its step_latest.pth is intact; relaunching run_vitkd_all.sh resumes it) and this
# script runs eg_raw alone.
#
# One epoch per phase, not two. docs/kd_handoff_v3.md §5.4's own numbers say both
# phases are past the point of return on SG: P1 ep1->ep2 moved recall_P +0.003 /
# recall_traj +0.020, and P2 ep2 gained +1 item on Avg while losing 6 on GSM, which
# is why §5.4 instructs "Report epoch 1 (366) ... Do not report only the best-of-2".
# The resulting protocol asymmetry (SG's P2 consumed P1's epoch-2 adapter, EG's
# consumes epoch-1) is real and belongs in the caption — see the plan's §4.
#
# Phase 1 is ALREADY RUNNING when this is launched; the script waits it out rather
# than starting a second copy. Everything after is the same command the chain would
# have issued, with --epochs 1.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

CKPT=$REPO/TrajGazeMerge/checkpoints
P1_OUT=$CKPT/vitkd_p1_eg_raw
P2_OUT=$CKPT/vitkd_p2_eg_raw
P1_BEST=$P1_OUT/best.pth
CHAIN=$REPO/vitkd_eg_raw_chain.log

export GAZE_OVERLAY=1
export VLM_GAZE_OVERLAY=0

log() { echo "[$(date -Is)] $*" | tee -a "$CHAIN"; }

log "########## eg_raw 1-epoch chain start ##########"

# ── wait for the in-flight Phase 1 ────────────────────────────────────────────
# Bracketed first character so the pattern cannot match this script's own line.
if pgrep -f "[t]rain_vit_selection_kd" >/dev/null 2>&1; then
    log "Phase 1 already running; waiting for it to exit"
    while pgrep -f "[t]rain_vit_selection_kd" >/dev/null 2>&1; do sleep 30; done
    log "Phase 1 process exited"
fi

# The trainer writes best.pth only after the epoch-end eval, so its presence is the
# real completion signal — a process that died mid-epoch leaves step_latest.pth only.
if [ ! -f "$P1_BEST" ]; then
    log "ABORT: $P1_BEST missing — Phase 1 did not finish its epoch-end eval."
    log "       Inspect $REPO/vitkd_eg_raw_p1.log; gate step_latest.pth by hand if needed (§3)."
    exit 1
fi
log "Phase 1 COMPLETE: $P1_BEST present"

# ── integrity gate ───────────────────────────────────────────────────────────
# Exits 2 on |Δ| > 4 items. Must not be skipped: it is what stops Phase 2 from
# training a readout on a damaged encoder (§3).
log "START gate -> $REPO/vitkd_eg_raw_gate.log"
CUDA_VISIBLE_DEVICES=0 python scripts/vitkd_integrity_gate.py \
    --source eg --lora-ckpt "$M1_EGONLY" --vit-lora-ckpt "$P1_BEST" \
    --dominant-ratio 0.065 --contextual-ratio 0.035 \
    >>"$REPO/vitkd_eg_raw_gate.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "ABORT: gate exit=$rc (2 = |Δ| > 4 items = FAIL). Phase 2 NOT started."
    exit 1
fi
log "gate PASS"

# ── Phase 2 ──────────────────────────────────────────────────────────────────
# --early-stop omitted: it only fires at (epoch+1)==2 and is a no-op at 1 epoch.
log "START p2 -> $REPO/vitkd_eg_raw_p2.log"
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29723 \
    -m TrajGazeMerge.training.train_visionzip_lora \
    --source eg --vit-lora-ckpt "$P1_BEST" \
    --dominant-ratio 0.065 --contextual-ratio 0.035 \
    --epochs 1 --lr 1e-4 --grad-accum 4 --no-hdepic --seed 0 \
    --ckpt-every-steps 200 --resume \
    --output-dir "$P2_OUT" \
    >>"$REPO/vitkd_eg_raw_p2.log" 2>&1
rc=$?
[ $rc -ne 0 ] && { log "ABORT: p2 exit=$rc"; exit 1; }
log "p2 COMPLETE"

# ── eval repeats 2 and 3 ─────────────────────────────────────────────────────
# P2's own epoch-end eval is run 1. v2 §8 needs >=3 and v3 §9 lists "every number
# here is currently a single run" as an open gap. evaluate() is rank-0-only, so two
# single-process jobs on separate GPUs cost the same wall time as one.
#
# --vit-lora-ckpt is mandatory, not decorative: train_visionzip_lora.py raises
# without it, because a P2 checkpoint scored on a stock ViT is plain VisionZip.
for i in 1 2; do
    log "START eval repeat $i on GPU $((i-1))"
    CUDA_VISIBLE_DEVICES=$((i-1)) $TORCHRUN --nproc_per_node=1 \
        --master_port=$((29730+i)) \
        -m TrajGazeMerge.training.train_visionzip_lora \
        --source eg --eval-ckpt "$P2_OUT/best.pth" --vit-lora-ckpt "$P1_BEST" \
        --dominant-ratio 0.065 --contextual-ratio 0.035 --no-hdepic \
        --output-dir "$P2_OUT" \
        >>"$REPO/vitkd_eg_raw_eval_r$i.log" 2>&1 &
done
wait
log "eval repeats COMPLETE"

log "########## eg_raw COMPLETE ##########"
