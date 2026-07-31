#!/usr/bin/env bash
# Re-score the best-of-2 25% teacher N times (default 3), then hand the GPUs to the student chain.
#
# WHY REPEATED RUNS OF AN IDENTICAL COMMAND. There is no seed to vary: the eval path holds no
# RNG at all (data/combined_dataset.py, data/dataset.py and models/model.py contain no
# random/np.random/shuffle; the option permutation at train_visionzip_complement_lora.py:68
# only fires under --option-aug, in the training loop), and this trainer has no --seed
# flag. The spread v2 §8 measured on this very checkpoint family -- 71.67 / 71.29 / 71.48
# / 70.72, five items across four re-scores of identical weights -- is bf16/flash-attn
# kernel nondeterminism, and it shows up by simply running the same command again.
#
# WHAT IS REPORTED. The user asked for the maximum of the runs. v2 §8 forbids exactly that
# ("do not select the best of N runs -- that is an upward-biased estimator") and requires
# the mean of >=3, so the summary prints max, mean AND spread and every individual value.
# When this goes in a table the max must be labelled best-of-N; the 10% teacher's 71.29
# it gets compared against is a 4-run MEAN, and max-vs-mean is not a comparison.
#
# OMP_NUM_THREADS=1 IS LOAD-BEARING. torch.distributed.run only sets it when
# nproc_per_node > 1, and evaluate() runs on rank 0 only
# (train_visionzip_complement_lora.py:805 `if is_main:`), so these are 1-rank launches
# and inherit an UNSET variable -- torch then parallelises every CPU op across all 72
# cores. Measured on this box, 2026-07-31, with it unset:
#
#   one eval,  72-way  -> 6.6 s/item, 13 cores burnt, GPU mostly idle. /proc showed 218
#                         python threads with ~70 of them at an identical 129 s of CPU:
#                         an OpenMP spin barrier, not work.
#   two evals, 72-way  -> 144 spinning threads on 72 cores: 97 KB/s and 0 B/s of disk
#                         reads, 0% GPU across 40 samples, ~1 item/minute.
#   training, OMP=1    -> 2.1 s/step INCLUDING the backward pass.
#
# So the parallelism was never the problem and two at a time is safe once the threads are
# capped: a 2-GPU training run is exactly this configuration -- two OMP=1 processes
# reading frames off Lustre -- and it sustains 2.1 s/step.
#
# --eval-progress-every 50 is passed so the run is never silent again; the default 0
# prints nothing until the final line, which is what made the stall look like a hang.
#
#   nohup setsid bash scripts/eval_teacher_b25_x10.sh >/dev/null 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

CKPT=$REPO/TrajGazeMerge/checkpoints
STATE=$REPO/vitkd25_state
CHAIN=$REPO/vitkd25_chain.log
BEST=$CKPT/visionzip_complement_learned_SGonly_overlay_b25_2ep/best.pth
T2EP_LOG=$REPO/vitkd25_teacher_b25_2ep.log
SUMMARY=$REPO/vitkd25_teacher_b25_2ep_eval_summary.txt
N_EVALS=${N_EVALS:-3}

log() { echo "[$(date -Is)] [eval-x10] $*" | tee -a "$CHAIN"; }

# ── wait for the two-epoch teacher ────────────────────────────────────────────
# Gate on the trainer's own exit line rather than only on the .done marker: if the
# teacher dies, the marker never appears and a marker-only wait would hang forever.
log "waiting for the 2-epoch teacher to finish"
while ! grep -aq "teacher_b25_2ep exit=" "$T2EP_LOG" 2>/dev/null; do sleep 60; done
rc=$(grep -a "teacher_b25_2ep exit=" "$T2EP_LOG" | tail -1 | sed 's/.*exit=\([0-9-]*\).*/\1/')
if [ "$rc" != "0" ]; then
    log "ABORT: the teacher exited $rc — inspect $T2EP_LOG"; exit 1
fi
[ -f "$BEST" ] || { log "ABORT: $BEST missing"; exit 1; }

# The trainer's ranks can still be tearing down after the exit line is written.
while pgrep -f "[t]rain_visionzip_complement_lora" >/dev/null 2>&1; do sleep 20; done
log "teacher done — starting $N_EVALS re-scores of $BEST"

# ── the re-scores ─────────────────────────────────────────────────────────
# Teachers read the overlay frames (v2 §7.3): GAZE_OVERLAY=1, VLM_GAZE_OVERLAY unset.
# The eval-only branch returns at :685, before output_dir is used for anything but the
# argparse default, so a scratch directory is enough.
# Skip an eval whose log already carries a finished result, so this script can be
# relaunched after an interruption without redoing the re-scores that completed --
# the same idempotence vitkd_state/*.done gives the training chain.
eval_done () {
    grep -aq "\[eval-only\] Overall:" "$REPO/vitkd25_teacher_b25_2ep_eval$(printf "%02d" "$1").log" 2>/dev/null
}

run_eval () {
    local i=$1 gpu=$2
    local tag; tag=$(printf "%02d" "$i")
    local out=$REPO/vitkd25_teacher_b25_2ep_eval$tag.log
    if eval_done "$i"; then log "SKIP eval $tag (already scored)"; return 0; fi
    echo "=== eval $tag start $(date -Is) (GPU $gpu) ===" >"$out"
    env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=$gpu \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $TORCHRUN --nproc_per_node=1 --master_port=$((29900 + i)) \
        -m TrajGazeMerge.training.train_visionzip_complement_lora \
        --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
        --content-ratio 0.15 --traj-ratio 0.10 --source sg --no-hdepic \
        --eval-ckpt "$BEST" --eval-progress-every 50 \
        --output-dir /tmp/eval_b25_$tag >>"$out" 2>&1
    echo "=== eval $tag exit=$? $(date -Is) ===" >>"$out"
}

i=1
while [ $i -le $N_EVALS ]; do
    t0=$(date +%s)
    run_eval "$i" 0 & pa=$!
    j=$((i + 1)); pb=""
    [ $j -le $N_EVALS ] && { run_eval "$j" 1 & pb=$!; }
    wait $pa; [ -n "$pb" ] && wait $pb
    log "eval $i${pb:+ and $j} done in $(( $(date +%s) - t0 ))s"
    i=$((i + 2))
done

# ── summary ───────────────────────────────────────────────────────────────────
# Delegated to scripts/collect_b25_pertask.py rather than reimplemented here. That
# script already parses these logs, and it also reconstructs the per-task item counts
# and checks they sum to each run's Overall -- the consistency check v3 §5.4/§5.7's
# tables depend on. An inline copy of the same logic drifted from it once already.
python "$REPO/scripts/collect_b25_pertask.py" --out "$SUMMARY" 2>&1 | tee -a "$CHAIN"

log "summary written to $SUMMARY — handing the GPUs to the student chain"

# ── student chain ─────────────────────────────────────────────────────────────
# Runs regardless of the number (user decision). teacher_b25.done already exists, so the
# chain SKIPs its teacher job and picks up the _2ep checkpoint for the gate; Phase 1
# resumes mid-epoch from step_latest.pth.
exec bash "$REPO/scripts/run_vitkd25_sg_raw.sh"
