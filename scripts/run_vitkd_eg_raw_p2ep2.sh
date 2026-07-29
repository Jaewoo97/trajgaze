#!/usr/bin/env bash
# eg_raw Phase 2, epoch 2 — budget-match the EG row to v2's KD-student bar.
#
# WHY. The 1-epoch eg_raw run (2026-07-29) scored 259 items vs the KD student's 268,
# but gave the LLM readout HALF the optimizer budget the student got (1 epoch vs 2).
# On SG, docs/kd_handoff_v3.md §5.5 could call its +6 budget-matched precisely because
# both sides had M1 warm-start + 2 LoRA epochs; the EG -9 has no such defence until
# this epoch runs. Note SG's own ep1->ep2 moved only 366->367, so the expected outcome
# is that -9 survives roughly intact and simply becomes interpretable.
#
# WHAT IS PRESERVED. --resume finds epoch_01.pth (key (1, 1<<30) beats step_latest's
# (1, ~600) in find_resume_ckpt) and trains epoch 2 ONLY. epoch_01.pth stays on disk,
# and the three identical 53.40% / 259-item re-scores already collected refer to it.
# best.pth is overwritten only if epoch 2 beats 53.40 (best_acc is restored from
# epoch_accs on resume), so the epoch-1 row remains recoverable either way.
#
# AFTER. Marks eg_raw done and relaunches the main chain, which resumes sg_ovl from
# its step_latest.pth (~step 1600, paused 14:06 to free the GPUs for this setting)
# and then runs eg_ovl.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

CKPT=$REPO/TrajGazeMerge/checkpoints
P1_BEST=$CKPT/vitkd_p1_eg_raw/best.pth
P2_OUT=$CKPT/vitkd_p2_eg_raw
CHAIN=$REPO/vitkd_eg_raw_chain.log

export GAZE_OVERLAY=1
export VLM_GAZE_OVERLAY=0

log() { echo "[$(date -Is)] $*" | tee -a "$CHAIN"; }

# Never start on top of a live job: one Phase-2 rank peaks well under the 183 GB card,
# but overlapping with ourselves is what actually costs wall time.
while pgrep -f "[t]rain_vit_selection_kd|[t]rain_visionzip_lora|[v]itkd_integrity_gate" \
        >/dev/null 2>&1; do
    sleep 20
done

log "START p2 epoch 2 (budget-match to the 268 bar)"
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29726 \
    -m TrajGazeMerge.training.train_visionzip_lora \
    --source eg --vit-lora-ckpt "$P1_BEST" \
    --dominant-ratio 0.065 --contextual-ratio 0.035 \
    --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --seed 0 \
    --ckpt-every-steps 200 --resume \
    --output-dir "$P2_OUT" \
    >>"$REPO/vitkd_eg_raw_p2.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    log "ABORT: p2 epoch 2 exit=$rc — chain NOT relaunched, epoch-1 results intact."
    exit 1
fi
log "p2 epoch 2 COMPLETE"

# Hand the GPUs back. The markers make run_vitkd_all.sh skip eg_raw; it resumes
# sg_ovl first (that setting comes earlier in the script) and then runs eg_ovl.
touch "$REPO/vitkd_state/eg_raw_p1.done" \
      "$REPO/vitkd_state/eg_raw_gate.done" \
      "$REPO/vitkd_state/eg_raw_p2.done"
log "eg_raw markers written; relaunching main chain (resumes sg_ovl, then eg_ovl)"
nohup setsid bash "$REPO/scripts/run_vitkd_all.sh" >/dev/null 2>&1 &
log "########## eg_raw fully COMPLETE ##########"
