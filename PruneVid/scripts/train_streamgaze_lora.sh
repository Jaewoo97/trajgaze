#!/usr/bin/env bash
# StreamGaze_v2 LoRA fine-tune launcher.
#   FRAMES   : 64 (default; PHASE A) or 16 (PHASE B)
#   OUTDIR   : checkpoint output dir
#   NPROC    : number of GPUs (default 2 for 2×H200)
#   ACCEL    : accelerate config (default singlegpu — use multigpu if NPROC>1)
set -e
cd /home/yujin/gaze/PruneVid

FRAMES=${FRAMES:-64}
OUTDIR=${OUTDIR:-test_results/streamgaze_lora/pllava7b_r16_${FRAMES}f}
NPROC=${NPROC:-2}
if [ "${NPROC}" -gt 1 ]; then
    ACCEL=${ACCEL:-scripts/accel_config_multigpu.yaml}
else
    ACCEL=${ACCEL:-scripts/accel_config_singlegpu.yaml}
fi
ACCELERATE_BIN=${ACCELERATE_BIN:-/opt/conda/envs/prunevid/bin/accelerate}

echo "FRAMES=${FRAMES}  OUTDIR=${OUTDIR}  NPROC=${NPROC}  ACCEL=${ACCEL}  BIN=${ACCELERATE_BIN}"

${ACCELERATE_BIN} launch \
    --config_file ${ACCEL} \
    --num_processes ${NPROC} \
    -m tasks.train.train_pllava_nframe_accel \
    tasks/train/config_pllava_streamgaze.py \
    num_frames ${FRAMES} \
    output_dir ${OUTDIR}
