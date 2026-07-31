#!/usr/bin/env bash
# Training-side efficiency benchmark for the supplementary KD table.
# Companion to scripts/measure_kd_inference_cost.py (deployment side).
# Results land in docs/kd_efficiency.md via scripts/collect_kd_efficiency.py.
#
# Four jobs, all StreamGaze, matching the three systems of kd_handoff_v3.md §5.5:
#
#   teacher    M1 SG specialist        train_visionzip_complement_lora
#   student    KD student, raw video   train_visionzip_kd_lora
#   vitkd_p1   ViT-KD Phase 1          train_vit_selection_kd
#   vitkd_p2   ViT-KD Phase 2          train_visionzip_lora
#
# ViT-KD is a two-phase protocol (§2.5), so its cost is p1 + p2 and quoting p2 alone
# is the misreading §10-7 warns about. Hence two rows, not one.
#
# ONE GPU per job, and the jobs run SERIALLY even though the box has two.
# §5.8 item 2 measured this loop as dataloader-bound -- the GPU idles ~28% of wall
# time because preprocess_video runs in the training loop with num_workers=2 of 16
# cores. Two concurrent jobs would contend for those cores and corrupt s/step. Memory
# would survive; the speed column would not.
#
# grad-accum is 8, not the 4 the real runs used, because those ran on 2 ranks:
# 1 x 8 == 2 x 4 keeps the effective batch at 8. That is this repo's own convention
# for single-GPU reruns -- see scripts/run_eg_teacher_retrain.sh.
#
#   tmux new -s bench
#   bash scripts/bench_kd_efficiency.sh
#
# Idempotent: a job with a .done marker is skipped, so an interrupted sweep resumes.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

GPU=${GPU:-0}
STEPS=${STEPS:-300}          # micro-steps per job
WARMUP=${WARMUP:-50}         # excluded from the summary; peak-mem counter resets here
PORT=${PORT:-29850}

# torch.distributed.run pins OMP_NUM_THREADS=1 only when nproc_per_node > 1, so every
# recorded 2-GPU run on this box got it implicitly and a --nproc_per_node=1 run does
# not. Left unset, torch takes all 72 cores per process: the first attempt at this
# benchmark ran the teacher at 8.2 s/step with 99 threads spinning and a load average
# of 291, against the 2.14 s/step the same trainer with the same arguments is on
# record for in tab6_nopretrain_overlay.log. That 3.8x is the thread pool, not the
# method. Setting it here reproduces the historic condition exactly, which is what
# makes the single-GPU numbers comparable to the recorded per-rank ones.
export OMP_NUM_THREADS=1

OUT=$REPO/bench_kd_efficiency
CKPT=$REPO/TrajGazeMerge/checkpoints
mkdir -p "$OUT"
DRIVER=$OUT/driver.log

log() { echo "[$(date -Is)] $*" | tee -a "$DRIVER"; }

# Nothing else may be on the GPU: the chain's jobs and ours would each report the
# other's contention as its own cost. run_vitkd_all.sh must stay down until this ends.
if pgrep -f "[t]rain_vit_selection_kd|[t]rain_visionzip|[v]itkd_integrity_gate" >/dev/null; then
    log "ABORT: a trainer is already running. Stop it before benchmarking."
    exit 1
fi

# run_bench <tag> <env-assignments> -- <torchrun args...>
run_bench () {
    local tag=$1; shift
    local marker=$OUT/$tag.done
    local job_log=$OUT/$tag.log
    local job_out=$OUT/ckpt_$tag

    if [ -f "$marker" ]; then log "SKIP $tag (already done)"; return 0; fi

    mkdir -p "$job_out"
    log "START $tag -> $job_log"
    echo "=== $tag start $(date -Is) | GPU=$GPU steps=$STEPS warmup=$WARMUP ===" >>"$job_log"

    # nvidia-smi in parallel: the allocator's max_memory_allocated excludes the CUDA
    # context and the caching allocator's slack, so it reads several GB below what a
    # user watching nvidia-smi sees. Report both rather than picking one.
    ( while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"
        sleep 1
      done ) >"$OUT/$tag.smi" 2>/dev/null &
    local smi_pid=$!

    ( set -x; "$@" --output-dir "$job_out" \
                   --max-steps "$STEPS" --bench-warmup "$WARMUP" ) >>"$job_log" 2>&1
    local rc=$?

    kill "$smi_pid" 2>/dev/null; wait "$smi_pid" 2>/dev/null

    echo "=== $tag exit=$rc $(date -Is) ===" >>"$job_log"
    if [ $rc -eq 0 ] && grep -q "\[BENCH\]" "$job_log"; then
        touch "$marker"; log "DONE  $tag  $(grep -h '\[BENCH\]' "$job_log" | tail -1)"
    else
        log "FAIL  $tag exit=$rc (no [BENCH] summary). Inspect $job_log."
        return 1
    fi
}

TR="python -m torch.distributed.run --nproc_per_node=1"

log "########## kd efficiency bench start (GPU $GPU, $STEPS steps, warmup $WARMUP) ##########"
# nproc honours OMP_NUM_THREADS, so it reports 1 here; --all is the machine's count.
log "OMP_NUM_THREADS=$OMP_NUM_THREADS  cores=$(nproc --all)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"

# ── 1. M1 SG specialist teacher ───────────────────────────────────────────────
# Teachers keep the gaze overlay on BOTH streams (v2 §7.3), so VLM_GAZE_OVERLAY is
# unset here and set to 0 for the two student rows below. Same compute either way --
# it selects a JPEG directory -- but the frame stream has to be stated (§10-3).
export GAZE_OVERLAY=1
unset VLM_GAZE_OVERLAY
run_bench teacher \
    env CUDA_VISIBLE_DEVICES=$GPU $TR --master_port=$((PORT+0)) \
        -m TrajGazeMerge.training.train_visionzip_complement_lora \
        --source sg --traj-pool-mode learned --complement-mode topk \
        --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.07 --traj-ratio 0.03 \
        --epochs 1 --lr 1e-4 --grad-accum 8 --no-hdepic || exit 1

# ── 2. KD student, raw video ──────────────────────────────────────────────────
export VLM_GAZE_OVERLAY=0          # student pixels: no marker
run_bench student \
    env CUDA_VISIBLE_DEVICES=$GPU $TR --master_port=$((PORT+1)) \
        -m TrajGazeMerge.training.train_visionzip_kd_lora \
        --source sg --warmstart-ckpt "$M1_SGONLY" --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.07 --traj-ratio 0.03 \
        --epochs 1 --lr 1e-4 --pred-lr 1e-3 --grad-accum 8 --no-hdepic || exit 1

# ── 3. ViT-KD Phase 1 ─────────────────────────────────────────────────────────
# --ckpt-every-steps 0: the real run writes a 250 KB adapter every 200 steps, which
# would land one filesystem blip inside a 300-step window. Immaterial to the total,
# but there is no reason to measure it.
run_bench vitkd_p1 \
    env CUDA_VISIBLE_DEVICES=$GPU $TR --master_port=$((PORT+2)) \
        -m TrajGazeMerge.training.train_vit_selection_kd \
        --source sg --warmstart-ckpt "$M1_SGONLY" --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.07 --traj-ratio 0.03 \
        --dom-primary 0.065 --ctx-primary 0.035 \
        --vit-lora-r 8 --vit-lora-alpha 16 --lr 2e-3 \
        --lambda-sel 1.0 --lambda-anchor 1.0 --score-query-frac 1.0 \
        --epochs 1 --grad-accum 8 --no-hdepic --seed 0 \
        --ckpt-every-steps 0 || exit 1

# ── 4. ViT-KD Phase 2 ─────────────────────────────────────────────────────────
# Consumes setting 1's finished Phase-1 adapter, exactly as run_vitkd_all.sh does.
run_bench vitkd_p2 \
    env CUDA_VISIBLE_DEVICES=$GPU $TR --master_port=$((PORT+3)) \
        -m TrajGazeMerge.training.train_visionzip_lora \
        --source sg --vit-lora-ckpt "$CKPT/vitkd_p1_sg_raw/best.pth" \
        --dominant-ratio 0.065 --contextual-ratio 0.035 \
        --epochs 1 --lr 1e-4 --grad-accum 8 --no-hdepic --seed 0 \
        --ckpt-every-steps 0 || exit 1

log "########## kd efficiency bench COMPLETE ##########"
log "next: python scripts/collect_kd_efficiency.py"
