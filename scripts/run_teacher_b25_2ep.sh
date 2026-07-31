#!/usr/bin/env bash
# M1 SG-only teacher at the 25% budget, TWO epochs — best-of-2.
#
# WHY a second run rather than a second epoch. The one-epoch teacher from
# run_vitkd25_sg_raw.sh scored 68.44% (360/526), below the 10% teacher's paper number
# (71.29%, 375 items, best-of-2). The 10% teacher gained +4.4 points in its second epoch
# (65.59 -> 69.96), so the comparison the row has to win is a best-of-2 one.
#
# train_visionzip_complement_lora has no --resume, no --warmstart-ckpt and no LR
# schedule (constant 1e-4 AdamW, line 701) -- there is no way to append epoch 2 to the
# finished epoch 1, so this re-runs both. Costs the 2h05m epoch 1 again, and buys a run
# whose recipe matches TRAINING_RUNS.md's specialist protocol exactly, which is the
# protocol the 69.96 / 71.29 it must beat was produced under.
#
# SEPARATE OUTPUT DIR, and that is not cosmetic: the 25% chain's integrity gate reads
# $CKPT/visionzip_complement_learned_SGonly_overlay_b25/best.pth while this runs.
# Writing into that directory would swap the gate's readout underneath it mid-eval.
# It also keeps the 1-epoch checkpoint for the epoch-1-vs-epoch-2 comparison.
#
#   cd /NHNHOME/VILAB/vilab_yj/trajgaze && source env.sh
#   nohup setsid bash scripts/run_teacher_b25_2ep.sh >/dev/null 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

OUT=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_SGonly_overlay_b25_2ep
LOG=$REPO/vitkd25_teacher_b25_2ep.log
CHAIN=$REPO/vitkd25_chain.log

log() { echo "[$(date -Is)] [teacher-2ep] $*" | tee -a "$CHAIN"; }

# Queue behind the 25% ViT-KD chain (p1 -> gate -> p2) instead of sharing the GPUs with
# it. Both fit in memory (p1 ~26 GB/GPU, this ~47 GB/GPU, of 183 GB) but they would each
# run at roughly half speed, so overlapping finishes nothing sooner and delays both.
# Same four-trainer pattern the 25% chain uses; bracketed first character so it cannot
# match this script's own command line.
log "waiting for the GPUs (the 25% chain is running p1 -> gate -> p2)"
waited=0
while pgrep -f "[t]rain_vit_selection_kd|[t]rain_visionzip_lora|[v]itkd_integrity_gate|[t]rain_visionzip_complement_lora" >/dev/null 2>&1; do
    sleep 60; waited=$((waited + 60))
    [ $((waited % 1800)) -eq 0 ] && log "still waiting (${waited}s)"
    [ $waited -ge 57600 ] && { log "WARN GPUs still busy after ${waited}s, starting anyway"; break; }
done
log "GPUs free after ${waited}s — starting"

# Teachers keep the gaze overlay (v2 7.3): GAZE_OVERLAY=1, VLM_GAZE_OVERLAY unset.
# The 25% chain exported VLM_GAZE_OVERLAY=0 for its raw-video stages, so unset it
# explicitly rather than trusting scope.
#
# --early-stop is a no-op at exactly 2 epochs (it only skips epoch 3); it is passed
# anyway so the command matches TRAINING_RUNS.md's specialist launch verbatim.
echo "=== teacher_b25_2ep start $(date -Is) ===" >>"$LOG"
env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0,1 \
    $TORCHRUN --nproc_per_node=2 --master_port=29880 \
    -m TrajGazeMerge.training.train_visionzip_complement_lora \
    --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
    --content-ratio 0.15 --traj-ratio 0.10 --source sg \
    --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --no-mid-eval \
    --output-dir "$OUT" >>"$LOG" 2>&1
rc=$?
echo "=== teacher_b25_2ep exit=$rc $(date -Is) ===" >>"$LOG"

if [ $rc -eq 0 ]; then
    touch "$REPO/vitkd25_state/teacher_b25_2ep.done"
    log "DONE (best-of-2 in $OUT/best.pth)"
else
    log "FAILED exit=$rc — inspect $LOG"
fi
exit $rc
