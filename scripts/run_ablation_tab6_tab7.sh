#!/usr/bin/env bash
# Table 6 / Table 7 ablation rows — Stage-2 LoRA, 1 epoch, StreamGaze only.
#
# Protocol (identical for every row; see TrajGazeMerge/docs/ablation_table6_7.md):
#   Qwen2.5-VL-7B frozen + LoRA r=16/a=32 | content 0.07 + traj 0.03 (10% budget)
#   --source sg --no-hdepic | --epochs 1 | lr 1e-4 | grad-accum 4 | 2-GPU DDP (eff-batch 8)
# Only the encoder source (Table 6) or the selection geometry (Table 7) changes.
#
# The \sys row of both tables is NOT run here: it reuses the already-measured
# SG-only specialist ($M1_SGONLY, per-task logged in eval_m1_sgonly_pertask.log
# and repeat_teacher_r{3,4}.log).
#
# Rows run serially — this machine has 2 GPUs, so co-resident runs are impossible.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

unset VLM_GAZE_OVERLAY      # env.sh does not clear it; a stale 0 silently switches
                            # every VLM frame to non-overlay and voids the run
export GAZE_OVERLAY=1

CKPT=$REPO/TrajGazeMerge/checkpoints
S1ROOT=$REPO/TrajGaze_v2/checkpoints
S1_SCOREONLY=$S1ROOT/stage1_scoreonly_overlay/best.pth

# Let stragglers from the previous row release their GPU memory. Bounded so an
# unrelated process on the box can never wedge the script.
wait_gpu () {
    local waited=0
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . ; do
        sleep 30; waited=$((waited + 30))
        [ $waited -ge 300 ] && { echo "[warn] GPU still busy after ${waited}s, continuing" >&2; break; }
    done
}

run_row () {   # $1=outdir-name  $2=master-port  $3..=row-specific flags
    local name=$1 port=$2; shift 2
    local out=$CKPT/$name log=$REPO/$name.log
    mkdir -p "$out"
    echo "=== $name start $(date -Is) ===" >>"$log"
    CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port="$port" \
        -m TrajGazeMerge.training.train_visionzip_complement_lora \
        --traj-pool-mode learned --complement-mode topk \
        --content-ratio 0.07 --traj-ratio 0.03 \
        --source sg --no-hdepic \
        --epochs 1 --lr 1e-4 --grad-accum 4 \
        --output-dir "$out" "$@" >>"$log" 2>&1
    local rc=$?
    echo "=== $name exit=$rc $(date -Is) ===" >>"$log"
    # 1 epoch => epoch_01.pth and best.pth are byte-identical 16.6GB copies
    [ $rc -eq 0 ] && rm -f "$out/epoch_01.pth"
    return $rc
}

# --- Table 6: Stage-1 pretraining objectives -------------------------------
run_row tab6_nopretrain_overlay 29911 \
    --stage1-ckpt "$STAGE1_CKPT" --random-encoder
wait_gpu

# --- Table 7: selection geometry -------------------------------------------
run_row tab7_nospatial_overlay 29912 \
    --stage1-ckpt "$STAGE1_CKPT" --select-geom no_spatial
wait_gpu

run_row tab7_notemporal_overlay 29913 \
    --stage1-ckpt "$STAGE1_CKPT" --select-geom no_temporal
wait_gpu

# --- Table 6: score-regression-only encoder (needs Phase 1 to have finished) -
if [ -f "$S1_SCOREONLY" ]; then
    run_row tab6_scoreonly_overlay 29914 --stage1-ckpt "$S1_SCOREONLY"
else
    echo "[SKIP] tab6_scoreonly: $S1_SCOREONLY not found — run Stage-1 first" >&2
fi

echo "=== all ablation rows done $(date -Is) ==="
