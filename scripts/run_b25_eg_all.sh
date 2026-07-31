#!/usr/bin/env bash
# EgoGazeVQA at the 25% token budget (content 15% ∪ traj 10%), end to end, on 4 GPUs.
#
#   teacher (2 ep) -> re-score x3 -> p1 -> integrity gate -> p2
#
# The EG counterpart of the SG run recorded in docs/kd_handoff_v3.md; the operating
# manual for this script is docs/kd_handoff_v4_egogazevqa_25.md. Ratios, epoch counts
# and lr are identical to that SG run so the two are comparable -- only --source and
# the GPU count differ.
#
# FOUR GPUS. The protocol is eff-batch 8, so --nproc_per_node=4 pairs with
# --grad-accum 2 (4 x 2 == 2 x 4, the same effective batch every number on record was
# produced at). EG train is 1,265 items, so each rank sees ~317 micro-steps per epoch.
#
# What extra GPUs do NOT speed up: evaluate() runs on rank 0 only
# (train_visionzip_complement_lora.py:805 `if is_main:`). On EG that is the dominant
# term -- roughly 920 s of eval against 824 s of training per teacher epoch. The three
# re-scores are therefore run as three single-rank jobs side by side instead.
#
# RESUMPTION: a .done marker per job under vitkd25eg_state/. Re-running skips finished
# jobs, and p1/p2 additionally resume mid-epoch from their own step_latest.pth. The
# teacher has neither --resume nor --ckpt-every-steps, so a death inside its epoch loses
# that epoch and run_job restarts it from zero.
#
#   cd <repo> && source env.sh
#   tmux new -s b25eg
#   bash scripts/run_b25_eg_all.sh
#
# tmux is not optional: an ssh drop kills the whole chain otherwise.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

CKPT=$REPO/TrajGazeMerge/checkpoints
STATE=$REPO/vitkd25eg_state
CHAIN=$REPO/vitkd25eg_chain.log
mkdir -p "$STATE"

TEACHER_OUT=$CKPT/visionzip_complement_learned_EGonly_overlay_b25_2ep
TEACHER_BEST=$TEACHER_OUT/best.pth
P1_OUT=$CKPT/vitkd25eg_p1_eg_raw_b25
P2_OUT=$CKPT/vitkd25eg_p2_eg_raw_b25
P1_BEST=$P1_OUT/best.pth
PREFIX=vitkd25eg_teacher_b25_2ep
N_EVALS=${N_EVALS:-3}
NGPU=${NGPU:-4}

log() { echo "[$(date -Is)] $*" | tee -a "$CHAIN"; }

# Serialise this chain against itself and against anything else using these trainers.
# Bracketed first character so the pattern cannot match this script's own command line.
wait_gpu () {
    local waited=0
    while pgrep -f "[t]rain_vit_selection_kd|[t]rain_visionzip_lora|[v]itkd_integrity_gate|[t]rain_visionzip_complement_lora" >/dev/null 2>&1; do
        sleep 30; waited=$((waited + 30))
        [ $((waited % 1800)) -eq 0 ] && log "WAIT another trainer holds the GPUs (${waited}s)"
        [ $waited -ge 43200 ] && { log "WARN still busy after ${waited}s, continuing"; break; }
    done
}

# run_job <name> <max_retries> <command...>
run_job () {
    local name=$1 retries=$2; shift 2
    local marker=$STATE/$name.done
    local log_file=$REPO/vitkd25eg_$name.log

    if [ -f "$marker" ]; then log "SKIP $name (already done)"; return 0; fi

    local attempt=1
    while [ $attempt -le "$retries" ]; do
        wait_gpu
        log "START $name (attempt $attempt/$retries) -> $log_file"
        echo "=== $name attempt $attempt start $(date -Is) ===" >>"$log_file"
        "$@" >>"$log_file" 2>&1
        local rc=$?
        echo "=== $name attempt $attempt exit=$rc $(date -Is) ===" >>"$log_file"
        if [ $rc -eq 0 ]; then touch "$marker"; log "DONE  $name"; return 0; fi
        log "FAIL  $name attempt $attempt exit=$rc"
        attempt=$((attempt + 1)); sleep 60
    done
    log "ABORT chain: $name failed $retries times. Inspect $log_file."
    return 1
}

ALL_GPUS=$(seq -s, 0 $((NGPU - 1)))

log "########## b25 EG chain start (4 GPU, teacher 2ep / p1 1ep / p2 1ep) ##########"

# ── 1. teacher, 25%, 2 epochs (best-of-2) ─────────────────────────────────────
# Teachers keep the gaze overlay (v2 §7.3): GAZE_OVERLAY=1 with VLM_GAZE_OVERLAY unset
# resolves EG to the `gaze` frames. `env -u` rather than trusting scope, because the
# student stages below export VLM_GAZE_OVERLAY=0.
#
# --early-stop is a no-op at exactly 2 epochs (it only skips epoch 3); it is passed so
# the command matches TRAINING_RUNS.md's specialist protocol verbatim.
run_job teacher 2 \
    env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=$ALL_GPUS \
        $TORCHRUN --nproc_per_node=$NGPU --master_port=29890 \
        -m TrajGazeMerge.training.train_visionzip_complement_lora \
        --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.15 --traj-ratio 0.10 --source eg \
        --epochs 2 --lr 1e-4 --grad-accum 2 --no-hdepic --early-stop --no-mid-eval \
        --output-dir "$TEACHER_OUT" || exit 1

[ -f "$TEACHER_BEST" ] || { log "ABORT: $TEACHER_BEST missing after the teacher job"; exit 1; }

# ── 2. re-score best.pth N times ──────────────────────────────────────────────
# There is no seed to vary: the eval path holds no RNG (data/*.py and models/model.py
# contain no random/np.random/shuffle; the option permutation in the complement trainer
# only fires under --option-aug, in the training loop) and this trainer has no --seed.
# The spread v2 §8 measured across re-scores of identical weights is bf16/flash-attn
# kernel nondeterminism, and it appears by rerunning the same command.
#
# OMP_NUM_THREADS=1 is load-bearing here: torch.distributed.run only sets it when
# nproc_per_node > 1, and these are single-rank. Left unset, torch fans every CPU op
# across all cores and the job spends its time in an OpenMP spin barrier with the GPU
# idle -- measured at 6.6 s/item against 1.2 s/item capped, and far worse with several
# uncapped jobs at once.
eval_one () {
    local i=$1 gpu=$2
    local tag; tag=$(printf "%02d" "$i")
    local out=$REPO/${PREFIX}_eval$tag.log
    if grep -aq "\[eval-only\] Overall:" "$out" 2>/dev/null; then
        log "SKIP eval $tag (already scored)"; return 0
    fi
    echo "=== eval $tag start $(date -Is) (GPU $gpu) ===" >"$out"
    env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=$gpu \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $TORCHRUN --nproc_per_node=1 --master_port=$((29900 + i)) \
        -m TrajGazeMerge.training.train_visionzip_complement_lora \
        --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.15 --traj-ratio 0.10 --source eg --no-hdepic \
        --eval-ckpt "$TEACHER_BEST" --eval-progress-every 50 \
        --output-dir /tmp/eval_b25eg_$tag >>"$out" 2>&1
    echo "=== eval $tag exit=$? $(date -Is) ===" >>"$out"
}

if [ -f "$STATE/rescore.done" ]; then
    log "SKIP rescore (already done)"
else
    wait_gpu
    log "START rescore x$N_EVALS (one per GPU, in parallel)"
    pids=""
    for i in $(seq 1 "$N_EVALS"); do
        eval_one "$i" $(( (i - 1) % NGPU )) & pids="$pids $!"
    done
    for p in $pids; do wait "$p"; done
    python "$REPO/scripts/collect_b25_pertask.py" --source eg 2>&1 | tee -a "$CHAIN"
    touch "$STATE/rescore.done"; log "DONE  rescore"
fi

# ── 3. ViT-KD student, EG raw video ───────────────────────────────────────────
export GAZE_OVERLAY=1
export VLM_GAZE_OVERLAY=0          # raw video: EG vlm='no_gaze'
log "=== SETTING eg_raw_b25 (source=eg VLM_GAZE_OVERLAY=0) ==="

# Phase 1 — distil the 25% selection into the ViT's own attention. Everything except the
# four ratios is setting 1's configuration: r=8/alpha=16 on block 31, lr 2e-3 (v3 §5.2's
# probe found 2e-5 and 1e-4 indistinguishable from doing nothing), query_frac 1.0 so the
# trained score is the one eval uses.
#
# lr 2e-3 was tuned against a 10% target. At 25% the positives are 2.5x as many
# (pos_weight 9 -> 3), so WATCH THE FIRST 200 STEPS: if the windowed recall_traj is not
# climbing, kill and raise lr (5e-3, then 1e-2) rather than spending the epoch.
run_job p1 3 \
    env CUDA_VISIBLE_DEVICES=$ALL_GPUS $TORCHRUN --nproc_per_node=$NGPU --master_port=29920 \
        -m TrajGazeMerge.training.train_vit_selection_kd \
        --source eg --warmstart-ckpt "$TEACHER_BEST" --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.15 --traj-ratio 0.10 \
        --dom-primary 0.175 --ctx-primary 0.075 \
        --vit-lora-r 8 --vit-lora-alpha 16 --lr 2e-3 \
        --lambda-sel 1.0 --lambda-anchor 1.0 --score-query-frac 1.0 \
        --epochs 1 --grad-accum 2 --no-hdepic --seed 0 \
        --ckpt-every-steps 200 --resume \
        --output-dir "$P1_OUT" || exit 1

# Integrity gate — selection held FIXED at the frozen ViT's choice, features swapped
# frozen -> tuned, so any difference is representation drift in block 31 and nothing
# else. Exits 2 on |delta| > 4 items, stopping the chain before Phase 2 trains a readout
# on a damaged encoder. Single-process, hence the thread cap again.
#
# The frozen recall_traj baseline is NOT v3 §5.2's 0.042 here: dominant widened from
# 6.5% to 17.5%, so more of the complement lands in it by chance. Read the delta.
run_job gate 2 \
    env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python scripts/vitkd_integrity_gate.py \
        --source eg --lora-ckpt "$TEACHER_BEST" --vit-lora-ckpt "$P1_BEST" \
        --dominant-ratio 0.175 --contextual-ratio 0.075 || exit 1

# Phase 2 — re-adapt the LLM readout to the distilled 25% selection. ViT frozen.
# One epoch, so --early-stop (which only compares epoch 2 to epoch 1) is omitted.
run_job p2 3 \
    env CUDA_VISIBLE_DEVICES=$ALL_GPUS $TORCHRUN --nproc_per_node=$NGPU --master_port=29924 \
        -m TrajGazeMerge.training.train_visionzip_lora \
        --source eg --vit-lora-ckpt "$P1_BEST" \
        --dominant-ratio 0.175 --contextual-ratio 0.075 \
        --epochs 1 --lr 1e-4 --grad-accum 2 --no-hdepic --seed 0 \
        --ckpt-every-steps 200 --resume \
        --output-dir "$P2_OUT" || exit 1

log "=== SETTING eg_raw_b25 COMPLETE ==="
log "teacher table: $REPO/${PREFIX}_pertask.txt"
log "student result: tail -40 $REPO/vitkd25eg_p2.log"
log "########## b25 EG chain COMPLETE ##########"
