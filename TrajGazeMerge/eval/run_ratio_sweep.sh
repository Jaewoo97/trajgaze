#!/usr/bin/env bash
# Ratio sweep: extreme merge_ratio × {baseline, +drop, +gaze-match, +dyn-α, all}.
# Each cell runs 1 short training epoch then reports the final egtea eval
# (merge_acc, full_acc, keep_ratio) which is parsed from the training log.
#
# Expected usage (edit paths below for your environment):
#   bash TrajGazeMerge/eval/run_ratio_sweep.sh /tmp/sweep_out
#
# Outputs:
#   <out_root>/sweep_results.csv — one row per (merge_ratio, cell).
#   <out_root>/<cell>/train_log_rank0.jsonl — raw per-cell log.

set -euo pipefail

OUT_ROOT="${1:-/tmp/trajgazemerge_sweep}"
STAGE1_CKPT="${STAGE1_CKPT:-/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth}"
TEACHER_CKPT="${TEACHER_CKPT:-/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth}"
NPROC="${NPROC:-2}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

mkdir -p "${OUT_ROOT}"
CSV="${OUT_ROOT}/sweep_results.csv"
echo "merge_ratio,cell,drop_ratio,score_transform,penalty,alpha_mode" > "${CSV}"

run_cell () {
    local merge_ratio="$1"
    local cell="$2"
    local drop_ratio="$3"
    local transform="$4"
    local penalty="$5"
    local alpha_mode="$6"

    local cell_dir="${OUT_ROOT}/r${merge_ratio}_${cell}"
    echo "=== ${cell_dir} ==="
    torchrun --nproc_per_node="${NPROC}" \
        -m TrajGazeMerge.training.train_merge_lora \
        --stage1-ckpt   "${STAGE1_CKPT}" \
        --teacher-ckpt  "${TEACHER_CKPT}" \
        --output-dir    "${cell_dir}" \
        --epochs        "${EPOCHS}" \
        --grad-accum    "${GRAD_ACCUM}" \
        --merge-ratio   "${merge_ratio}" \
        --drop-ratio    "${drop_ratio}" \
        --score-transform "${transform}" \
        --match-score-penalty "${penalty}" \
        --alpha-mode    "${alpha_mode}"

    echo "${merge_ratio},${cell},${drop_ratio},${transform},${penalty},${alpha_mode}" >> "${CSV}"
}

for merge_ratio in 0.90 0.95 0.98; do
    run_cell "${merge_ratio}" "baseline"   "0.0" "none"    "0.0" "static"
    run_cell "${merge_ratio}" "drop"       "0.1" "sigmoid" "0.0" "static"
    run_cell "${merge_ratio}" "gaze_match" "0.0" "sigmoid" "2.0" "static"
    run_cell "${merge_ratio}" "dyn_alpha"  "0.0" "sigmoid" "0.0" "p_gt"
    run_cell "${merge_ratio}" "all"        "0.1" "sigmoid" "2.0" "p_gt"
done

echo "Sweep done. Per-cell logs under ${OUT_ROOT}/."
echo "Each cell_dir's train_log_rank0.jsonl holds loss/alpha/kept_ratio curves;"
echo "and the final evaluation line is printed to stdout (grep for '[Final] egtea')."
