#!/usr/bin/env bash
# ViT selection-distillation ablation — all four settings, serially.
#
#   SG raw video  ->  SG overlay  ->  EG raw video  ->  EG overlay
#
# Each setting runs four jobs:
#   p1          train_vit_selection_kd      distil M1's 10% into the ViT's attention
#   gate_base   100%-token eval, FROZEN ViT   ) the integrity gate: these two must
#   gate_tuned  100%-token eval, TUNED  ViT   ) agree within +-4 items (§8)
#   p2          train_visionzip_lora        re-adapt the LLM readout to the new
#                                           selection, at dominant .065 + contextual .035
#
# Bars to beat, from docs/kd_handoff_v2.md:
#   SG raw 360 (§7.7) | SG overlay 369 (§2.2a) | EG raw 268 (§7.7) | EG overlay 272 (§10.3)
#
# RESUMPTION, two independent layers:
#   * chain level  — a .done marker per job; re-running skips finished jobs
#   * trainer level— every job passes --resume, and both trainers now checkpoint
#                    mid-epoch (adapter tensors only). Before this, a death inside
#                    epoch 1 left nothing at all and three runs were lost that way
#                    (§13.4). Relaunching this script is always the correct recovery.
#
#   nohup setsid scripts/run_vitkd_all.sh >/dev/null 2>&1 &
#
# Only ONE setting is in flight at a time: this box has 2 GPUs and each job uses both.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

CKPT=$REPO/TrajGazeMerge/checkpoints
STATE=$REPO/vitkd_state
CHAIN=$REPO/vitkd_chain.log
mkdir -p "$STATE"

log() { echo "[$(date -Is)] $*" | tee -a "$CHAIN"; }

# Serialise THIS chain's jobs against each other — not against everything on the box.
#
# Gating on process exit rather than on free memory: a training job briefly frees
# memory between its loop and its end-of-epoch eval, and a memory-threshold wait
# starts the next job on top of it.
#
# Deliberately scoped to our own trainer processes. A previous version waited for
# *any* CUDA process, which meant an unrelated job someone else is running (a
# rendering pass, an eval) would stall every one of the 16 jobs for the full 15-min
# cap — hours of dead time to no purpose. The GPUs hold 183 GB and one Phase-1 rank
# peaks at ~22 GB, so coexisting with someone else's work is fine; overlapping with
# OURSELVES is what has to be prevented.
#
# Bracketed first character so the pattern cannot match this script's own line.
wait_gpu () {
    local waited=0
    while pgrep -f "[t]rain_vit_selection_kd|[t]rain_visionzip_lora|[v]itkd_integrity_gate" >/dev/null 2>&1; do
        sleep 30; waited=$((waited + 30))
        [ $waited -ge 900 ] && { log "WARN previous job still alive after ${waited}s, continuing"; break; }
    done
}

# run_job <name> <max_retries> <command...>
# Retries are for transient deaths (SIGTERM, node reprovision). A job that fails
# repeatedly stops the whole chain rather than burning hours on a broken config —
# tab6_nopretrain died the same way twice and blind retries bought nothing (§9).
run_job () {
    local name=$1 retries=$2; shift 2
    local marker=$STATE/$name.done
    local log_file=$REPO/vitkd_$name.log

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

# ── one setting = p1 -> gate_base -> gate_tuned -> p2 ─────────────────────────
# $1 tag  $2 source(sg|eg)  $3 VLM_GAZE_OVERLAY  $4 M1 checkpoint  $5 base port
run_setting () {
    local tag=$1 src=$2 vlm_ovl=$3 m1=$4 port=$5
    local p1_out=$CKPT/vitkd_p1_$tag
    local p2_out=$CKPT/vitkd_p2_$tag
    local p1_best=$p1_out/best.pth

    export GAZE_OVERLAY=1
    export VLM_GAZE_OVERLAY=$vlm_ovl
    log "=== SETTING $tag (source=$src VLM_GAZE_OVERLAY=$vlm_ovl m1=$(basename "$(dirname "$m1")")) ==="

    # Phase 1 — distil the selection into the ViT. query_frac 1.0: step 0 measured
    # 2.19s/step for the full-query gradient, which fits the budget, and subsampling
    # would make the trained score differ from the one eval uses.
    #
    # lr 2e-3, not the 2e-5 originally planned. An LR probe (2e-5 / 1e-4 / 5e-4 / 2e-3,
    # 200 steps each) found 2e-5 and 1e-4 indistinguishable from doing nothing —
    # recall_traj stuck at the frozen baseline of ~0.042 — which would have produced a
    # false negative after six hours rather than a measurement. At 2e-3, 50 optimizer
    # steps take recall_traj 0.042 -> 0.269 while recall_P RISES 0.398 -> 0.430 (so
    # gaze tokens are not being promoted at the content tokens' expense) and the
    # anchor's growth rate decays ~4x, i.e. lambda_anchor is finding an equilibrium.
    # Drift is bounded by the gate below, not by a small LR.
    run_job "${tag}_p1" 3 \
        env CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=$((port+0)) \
            -m TrajGazeMerge.training.train_vit_selection_kd \
            --source "$src" --warmstart-ckpt "$m1" --stage1-ckpt "$STAGE1_CKPT" \
            --content-ratio 0.07 --traj-ratio 0.03 \
            --dom-primary 0.065 --ctx-primary 0.035 \
            --vit-lora-r 8 --vit-lora-alpha 16 --lr 2e-3 \
            --lambda-sel 1.0 --lambda-anchor 1.0 --score-query-frac 1.0 \
            --epochs 2 --grad-accum 4 --no-hdepic --seed 0 \
            --ckpt-every-steps 200 --resume \
            --output-dir "$p1_out" || return 1

    # Integrity gate: selection held FIXED at the frozen ViT's choice, features swapped
    # frozen -> tuned. Any delta is representation drift in block 31 and nothing else.
    # Exits 2 on |Δ| > 4 items (§8 noise floor), which stops the chain before Phase 2
    # spends four hours training a readout on a damaged encoder.
    run_job "${tag}_gate" 2 \
        env CUDA_VISIBLE_DEVICES=0 python scripts/vitkd_integrity_gate.py \
            --source "$src" --lora-ckpt "$m1" --vit-lora-ckpt "$p1_best" \
            --dominant-ratio 0.065 --contextual-ratio 0.035 || return 1

    # Phase 2 — readout re-adaptation against the distilled selection.
    run_job "${tag}_p2" 3 \
        env CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=$((port+3)) \
            -m TrajGazeMerge.training.train_visionzip_lora \
            --source "$src" --vit-lora-ckpt "$p1_best" \
            --dominant-ratio 0.065 --contextual-ratio 0.035 \
            --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --seed 0 \
            --ckpt-every-steps 200 --resume \
            --output-dir "$p2_out" || return 1

    log "=== SETTING $tag COMPLETE ==="
}

log "########## vitkd chain start ##########"

run_setting sg_raw sg 0 "$M1_SGONLY" 29700 || exit 1
run_setting sg_ovl sg 1 "$M1_SGONLY" 29710 || exit 1
run_setting eg_raw eg 0 "$M1_EGONLY" 29720 || exit 1
run_setting eg_ovl eg 1 "$M1_EGONLY" 29730 || exit 1

log "########## vitkd chain COMPLETE — all four settings ##########"
