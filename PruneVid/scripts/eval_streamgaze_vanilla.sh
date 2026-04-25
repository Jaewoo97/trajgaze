#!/usr/bin/env bash
# Vanilla PLLaVA-7B zero-shot on StreamGaze_v2 EGTEA subset (526 QA, 8 tasks).
# Runs three frame settings: 16 / 32 / 64.
# Pruning is fully disabled (tau=1, temporal_segment_ratio=1, cluster_ratio=1).
set -e

cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-MODELS/pllava-7b}
LORA_ALPHA=${LORA_ALPHA:-4}   # PLLaVA-official default (PruneVid eval.sh uses 14)
POOLING_SHAPE=${POOLING_SHAPE:-16-12-12}
FRAMES=${FRAMES:-"16 32 64"}
SAVE_ROOT=${SAVE_ROOT:-test_results/streamgaze_egtea}
PYBIN=${PYBIN:-/opt/conda/envs/prunevid/bin/python}

for NF in $FRAMES; do
  SAVE_DIR=${SAVE_ROOT}/vanilla_${NF}f
  mkdir -p "${SAVE_DIR}"
  echo "==============================================================="
  echo "Vanilla PLLaVA-7B | num_frames=${NF} | save=${SAVE_DIR}"
  echo "==============================================================="
  "${PYBIN}" -m tasks.eval.streamgaze.pllava_eval_streamgaze \
    --pretrained_model_name_or_path "${MODEL_DIR}" \
    --save_path "${SAVE_DIR}" \
    --num_frames "${NF}" \
    --use_lora \
    --lora_alpha "${LORA_ALPHA}" \
    --weight_dir "${MODEL_DIR}" \
    --pooling_shape "${POOLING_SHAPE}" \
    --conv_mode eval_mvbench \
    --tau 1.0 \
    --temporal_segment_ratio 1.0 \
    --cluster_ratio 1.0 \
    --top_p 1.0 \
    --temperature 1.0 \
    "$@"
done

"${PYBIN}" -m tasks.eval.streamgaze.aggregate_results \
  --run 16f=${SAVE_ROOT}/vanilla_16f \
  --run 32f=${SAVE_ROOT}/vanilla_32f \
  --run 64f=${SAVE_ROOT}/vanilla_64f
