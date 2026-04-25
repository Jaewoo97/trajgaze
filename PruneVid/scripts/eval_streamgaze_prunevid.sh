#!/usr/bin/env bash
# PruneVid (pruning ON) zero-shot on StreamGaze_v2 EGTEA subset (526 QA, 8 tasks).
# Hyperparameters follow scripts/eval.sh defaults for the PLLaVA-7B backbone:
#   selected_layer=10, alpha=0.4, tau=0.8, temporal_segment_ratio=0.25, cluster_ratio=0.5
set -e

cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-MODELS/pllava-7b}
LORA_ALPHA=${LORA_ALPHA:-14}
POOLING_SHAPE=${POOLING_SHAPE:-16-12-12}
NUM_FRAMES=${NUM_FRAMES:-16}
SAVE_ROOT=${SAVE_ROOT:-test_results/streamgaze_egtea}
PYBIN=${PYBIN:-/opt/conda/envs/prunevid/bin/python}

SELECTED_LAYER=${SELECTED_LAYER:-10}
ALPHA=${ALPHA:-0.4}
TAU=${TAU:-0.8}
TEMPORAL_SEGMENT_RATIO=${TEMPORAL_SEGMENT_RATIO:-0.25}
CLUSTER_RATIO=${CLUSTER_RATIO:-0.5}

SAVE_DIR=${SAVE_ROOT}/prunevid_${NUM_FRAMES}f_tau${TAU}_seg${TEMPORAL_SEGMENT_RATIO}_clu${CLUSTER_RATIO}
mkdir -p "${SAVE_DIR}"

echo "==============================================================="
echo "PruneVid (pruning ON) | num_frames=${NUM_FRAMES} | save=${SAVE_DIR}"
echo "  selected_layer=${SELECTED_LAYER} alpha=${ALPHA}"
echo "  tau=${TAU} temporal_segment_ratio=${TEMPORAL_SEGMENT_RATIO} cluster_ratio=${CLUSTER_RATIO}"
echo "==============================================================="
"${PYBIN}" -m tasks.eval.streamgaze.pllava_eval_streamgaze \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --save_path "${SAVE_DIR}" \
  --num_frames "${NUM_FRAMES}" \
  --use_lora \
  --lora_alpha "${LORA_ALPHA}" \
  --weight_dir "${MODEL_DIR}" \
  --pooling_shape "${POOLING_SHAPE}" \
  --conv_mode eval_mvbench \
  --selected_layer "${SELECTED_LAYER}" \
  --alpha "${ALPHA}" \
  --tau "${TAU}" \
  --temporal_segment_ratio "${TEMPORAL_SEGMENT_RATIO}" \
  --cluster_ratio "${CLUSTER_RATIO}" \
  --top_p 1.0 \
  --temperature 1.0 \
  "$@"
