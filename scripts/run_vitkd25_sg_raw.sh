#!/usr/bin/env bash
# Budget ablation: the SAME ViT selection-distillation experiment at 25% instead of 10%.
#
#   teacher -> p1 -> gate -> p2      (SG raw video only, ONE epoch per stage)
#
# WHY. Every row in docs/kd_handoff_v3.md sits on a single token budget: M1's 10% =
# VisionZip content 7% union TAS trajectory complement 3%. This chain re-runs setting 1
# (SG raw video) at content 15% union traj 10% = 25%, changing nothing else, so the
# question "does the ViT absorb the gaze signal" can be asked at a second budget.
#
#   dom_ratio = content/2 + traj = 0.175      (train_visionzip_complement_lora.py:450)
#   ctx_ratio = content/2        = 0.075
#   kept tokens = 3,456 of N=13,824 (10% kept 1,382)
#
# ONE EPOCH PER STAGE, by decision, for turnaround. Consequences that must travel with
# every number this produces:
#   * The ViT-KD row is comparable to v3 5.4's P2 **ep1 = 366 items**, not ep2's 367 --
#     a P2 epoch-k checkpoint has had exactly k epochs of readout training (v3 8a).
#   * Setting 1's P1 ran TWO epochs, so 25%-vs-10% mixes the budget change with less
#     Phase-1 training. Direction is known: 5.4 measured recall_traj 0.3636 (ep1) ->
#     0.3833 (ep2), so 25% starts at a selection-fidelity disadvantage. State it.
#   * The 10% M1 SG-only teacher went 65.59 (ep1) -> 69.96 (ep2), i.e. +4.4 in its
#     second epoch. A one-epoch teacher here is a FLOOR and cannot be compared to that
#     teacher's best-of-2 (375 items). Label the row "teacher @25%, 1 epoch".
#   Its other two jobs are unaffected: the gate is a difference between two arms that
#   share this LoRA, and Phase 1 never forwards the LLM at all.
#
# SEPARATE STATE FROM THE 10% CHAIN, deliberately. run_vitkd_all.sh is mid-flight with
# sg_ovl_p2 / eg_ovl outstanding; its vitkd_state/*.done markers and vitkd_*.log names
# must stay exactly as they are so it can still be resumed. Hence vitkd25_state/ and
# vitkd25_*.log here, and no edit to that script.
#
#   cd /NHNHOME/VILAB/vilab_yj/trajgaze && source env.sh
#   nohup setsid bash scripts/run_vitkd25_sg_raw.sh >/dev/null 2>&1 &
#   bash scripts/vitkd_status.sh          # (reads the 10% chain; see vitkd25_chain.log for this one)

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

CKPT=$REPO/TrajGazeMerge/checkpoints
STATE=$REPO/vitkd25_state
CHAIN=$REPO/vitkd25_chain.log
mkdir -p "$STATE"

# The BEST-OF-2 teacher, not the one-epoch teacher this chain originally trained.
#
# That first run scored 68.44% (360/526) against the 10% teacher's 71.29% (375 items,
# 4-eval mean, v2 §8), so it was retrained for two epochs by
# scripts/run_teacher_b25_2ep.sh into the _2ep directory. The teacher_b25.done marker
# below is left in place deliberately: the teacher job is SKIPped and this path is what
# the integrity gate loads as its readout LoRA.
#
# Only the gate is affected by the swap. Phase 1 loads the warm-start but never forwards
# the LLM (train_vit_selection_kd.py:548-583 calls base_qwen.visual() and nothing else),
# and Phase 2 trains a fresh LoRA without consulting the teacher at all.
TEACHER_OUT=$CKPT/visionzip_complement_learned_SGonly_overlay_b25_2ep
TEACHER_BEST=$TEACHER_OUT/best.pth

log() { echo "[$(date -Is)] $*" | tee -a "$CHAIN"; }

# Serialise against BOTH chains' trainers.
#
# run_vitkd_all.sh's version omits train_visionzip_complement_lora -- it never launched
# that module. This chain does (the teacher), and the 10% chain is running right now, so
# the pattern has to cover all four trainers or the two chains land on the same GPUs.
#
# The cap is 12h, not the 15 minutes the 10% chain uses. That cap exists there to stop a
# single hung job from stalling twelve queued ones; here the thing being waited on is
# another chain with hours of work left, and starting anyway would halve both chains'
# throughput. Bracketed first character so the pattern cannot match this script.
wait_gpu () {
    local waited=0 announced=0
    while pgrep -f "[t]rain_vit_selection_kd|[t]rain_visionzip_lora|[v]itkd_integrity_gate|[t]rain_visionzip_complement_lora" >/dev/null 2>&1; do
        sleep 30; waited=$((waited + 30))
        if [ $((waited % 600)) -eq 0 ] && [ $waited -ne $announced ]; then
            log "WAIT another trainer holds the GPUs (${waited}s)"; announced=$waited
        fi
        [ $waited -ge 43200 ] && { log "WARN still busy after ${waited}s, continuing anyway"; break; }
    done
}

# run_job <name> <max_retries> <command...>
run_job () {
    local name=$1 retries=$2; shift 2
    local marker=$STATE/$name.done
    local log_file=$REPO/vitkd25_$name.log

    if [ -f "$marker" ]; then log "SKIP $name (already done)"; return 0; fi

    local attempt=1
    while [ $attempt -le "$retries" ]; do
        wait_gpu
        log "START $name (attempt $attempt/$retries) -> $log_file"
        echo "=== $name attempt $attempt start $(date -Is) ===" >>"$log_file"
        "$@" >>"$log_file" 2>&1
        local rc=$?
        echo "=== $name attempt $attempt exit=$rc $(date -Is) ===" >>"$log_file"
        if [ $rc -eq 0 ]; then
            touch "$marker"; log "DONE  $name"; return 0
        fi
        log "FAIL  $name attempt $attempt exit=$rc"
        attempt=$((attempt + 1))
        sleep 60
    done
    log "ABORT chain: $name failed $retries times. Inspect $log_file."
    return 1
}

log "########## vitkd 25% chain start (SG raw, 1 epoch/stage) ##########"

# ── 0. teacher — M1 SG-only at 25% ────────────────────────────────────────────
# VLM_GAZE_OVERLAY UNSET on purpose: teachers keep the gaze overlay (v2 7.3), and the
# 10% teacher this is being compared against was trained that way. `env -u` rather than
# relying on scope, because the p1/gate/p2 exports below set it to 0.
#
# --early-stop is omitted: it only compares epoch 2 against epoch 1 and there is no
# epoch 2. best.pth still tracks the best (here: only) epoch on its own.
#
# NOTE this trainer has neither --resume nor --ckpt-every-steps (it writes epoch_NN.pth
# and best.pth only), so a death inside the epoch loses the whole epoch and run_job's
# retry restarts from zero rather than resuming.
run_job teacher_b25 2 \
    env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0,1 \
        $TORCHRUN --nproc_per_node=2 --master_port=29870 \
        -m TrajGazeMerge.training.train_visionzip_complement_lora \
        --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.15 --traj-ratio 0.10 --source sg \
        --epochs 1 --lr 1e-4 --grad-accum 4 --no-hdepic --no-mid-eval \
        --output-dir "$TEACHER_OUT" || exit 1

[ -f "$TEACHER_BEST" ] || { log "ABORT: $TEACHER_BEST missing after the teacher job"; exit 1; }

# ── the ViT-KD setting: SG raw video at 25% ───────────────────────────────────
TAG=sg_raw_b25
P1_OUT=$CKPT/vitkd25_p1_$TAG
P2_OUT=$CKPT/vitkd25_p2_$TAG
P1_BEST=$P1_OUT/best.pth

export GAZE_OVERLAY=1
export VLM_GAZE_OVERLAY=0          # raw video: SG vlm='original'
log "=== SETTING $TAG (source=sg VLM_GAZE_OVERLAY=0 m1=$(basename "$TEACHER_OUT")) ==="

# Phase 1 — distil the 25% selection into the ViT's own attention. Everything except
# the four ratios is setting 1's configuration verbatim: r=8/alpha=16 on block 31,
# lr 2e-3 (5.2's probe: 2e-5 and 1e-4 were indistinguishable from doing nothing),
# query_frac 1.0 so the trained score is the one eval uses.
#
# lr 2e-3 was tuned against the 10% target. At 25% the positives are 2.5x as many
# (pos_weight 9 -> 3), so READ THE FIRST 200 STEPS: if the windowed recall_traj is not
# climbing, kill and raise lr (5e-3, then 1e-2) rather than spending the epoch on it.
run_job "${TAG}_p1" 3 \
    env CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29871 \
        -m TrajGazeMerge.training.train_vit_selection_kd \
        --source sg --warmstart-ckpt "$TEACHER_BEST" --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.15 --traj-ratio 0.10 \
        --dom-primary 0.175 --ctx-primary 0.075 \
        --vit-lora-r 8 --vit-lora-alpha 16 --lr 2e-3 \
        --lambda-sel 1.0 --lambda-anchor 1.0 --score-query-frac 1.0 \
        --epochs 1 --grad-accum 4 --no-hdepic --seed 0 \
        --ckpt-every-steps 200 --resume \
        --output-dir "$P1_OUT" || exit 1

# Integrity gate — selection held FIXED at the frozen ViT's choice, features swapped
# frozen -> tuned, so any delta is representation drift in block 31 and nothing else.
# Exits 2 on |delta| > 4 items, stopping the chain before Phase 2 trains a readout on a
# damaged encoder. The frozen baseline for recall_traj is NOT 5.2's 0.042 here: dominant
# widened 6.5% -> 17.5%, so more of the complement lands there by chance. Read the delta.
#
# OMP_NUM_THREADS=1: this is a bare `python`, not a torchrun launch, so nothing caps
# torch's intra-op threads and it fans every CPU op out over all 72 cores. Measured here
# on the eval path, an uncapped 1-rank job spent 13 cores spinning in an OpenMP barrier
# and ran ~4x slower than the same work at OMP=1. The 10% gate (3.42 s/item, 30 min) paid
# that too. Same reason the p1/p2 torchrun jobs do not need it: nproc_per_node > 1 makes
# torch.distributed.run set it to 1 for them.
run_job "${TAG}_gate" 2 \
    env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python scripts/vitkd_integrity_gate.py \
        --source sg --lora-ckpt "$TEACHER_BEST" --vit-lora-ckpt "$P1_BEST" \
        --dominant-ratio 0.175 --contextual-ratio 0.075 || exit 1

# Phase 2 — re-adapt the LLM readout to the distilled 25% selection. ViT frozen.
# --early-stop omitted for the same reason as the teacher: there is no epoch 2.
run_job "${TAG}_p2" 3 \
    env CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29874 \
        -m TrajGazeMerge.training.train_visionzip_lora \
        --source sg --vit-lora-ckpt "$P1_BEST" \
        --dominant-ratio 0.175 --contextual-ratio 0.075 \
        --epochs 1 --lr 1e-4 --grad-accum 4 --no-hdepic --seed 0 \
        --ckpt-every-steps 200 --resume \
        --output-dir "$P2_OUT" || exit 1

log "=== SETTING $TAG COMPLETE ==="
log "########## vitkd 25% chain COMPLETE ##########"
