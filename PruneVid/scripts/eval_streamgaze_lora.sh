#!/usr/bin/env bash
# Evaluate a fine-tuned StreamGaze LoRA adapter on the 526 EGTEA QA held-out set.
#   CKPT     : adapter weight dir (e.g. test_results/streamgaze_lora/.../pretrained_epoch01)
#   FRAMES   : 64 (PHASE A) or 16 (PHASE B)
#   SAVE     : output dir for all_results.json + upload_leaderboard.json
set -e
cd /home/yujin/gaze/PruneVid

FRAMES=${FRAMES:-64}
CKPT=${CKPT:-test_results/streamgaze_lora/pllava7b_r16_${FRAMES}f/pretrained_epoch01}
SAVE=${SAVE:-test_results/streamgaze_egtea/lora_r16_${FRAMES}f}

PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/prunevid/bin/python}
echo "FRAMES=${FRAMES}  CKPT=${CKPT}  SAVE=${SAVE}  PYTHON=${PYTHON_BIN}"

${PYTHON_BIN} -m tasks.eval.streamgaze.pllava_eval_streamgaze \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --weight_dir ${CKPT} \
    --save_path ${SAVE} \
    --num_frames ${FRAMES} \
    --use_lora \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --pooling_shape 16-12-12 \
    --conv_mode eval_mvbench_streamgaze \
    --tau 1.0 --temporal_segment_ratio 1.0 --cluster_ratio 1.0 \
    --top_p 1.0 --temperature 1.0 \
    --answer_prompt none --return_prompt none
