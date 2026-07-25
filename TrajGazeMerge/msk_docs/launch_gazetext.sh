#!/bin/bash
# Launch gazetext experiments (gaze + random arms) on GPUs 0,2 once tbudget frees them.
# Also triggers tbudget McNemar (w=0 vs w=1) on GPU 3.
# Run: nohup bash msk_docs/launch_gazetext.sh > checkpoints/gazetext_launch.log 2>&1 &

set -euo pipefail
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
LOG=$R/gazetext_launch.log
PYTHON=/opt/conda/envs/trajgaze/bin/python3

echo "[$(date)] launch_gazetext.sh started" | tee -a "$LOG"

# ── wait for tbudget to finish ──────────────────────────────────────────────
echo "[$(date)] Waiting for tbudget (w=1.0, w=0.0, w=0.5) to complete..." | tee -a "$LOG"
while true; do
    alive=$(pgrep -f "tbudget_lora_3way" | wc -l)
    if [ "$alive" -eq 0 ]; then
        echo "[$(date)] tbudget processes done." | tee -a "$LOG"
        break
    fi
    echo "[$(date)] tbudget still alive (${alive} procs). Sleeping 300s..." | tee -a "$LOG"
    sleep 300
done

# ── tbudget McNemar (w=1.0 vs w=0.0) on GPU 3 ──────────────────────────────
echo "[$(date)] Launching tbudget McNemar (w=1.0 vs w=0.0) on GPU 3 ..." | tee -a "$LOG"
DUMP=$R/dumps
mkdir -p "$DUMP"

# dump w=1.0 checkpoint
CKPT_W10=$R/tbudget_w10_2way/best.pth
CKPT_W00=$R/tbudget_w00_2way/best.pth
CKPT_W05=$R/tbudget_w05_2way/best.pth

setsid bash -c "
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=3 $PYTHON -m TrajGazeMerge.eval.eval_dump \
    --ckpt $CKPT_W10 --dump $DUMP/tbudget_w10.jsonl \
    --complement-mode topk --content-ratio 0.07 --traj-ratio 0.03 --gpu 0 \
    && GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=3 $PYTHON -m TrajGazeMerge.eval.eval_dump \
    --ckpt $CKPT_W00 --dump $DUMP/tbudget_w00.jsonl \
    --complement-mode topk --content-ratio 0.07 --traj-ratio 0.03 --gpu 0 \
    && GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=3 $PYTHON -m TrajGazeMerge.eval.eval_dump \
    --ckpt $CKPT_W05 --dump $DUMP/tbudget_w05.jsonl \
    --complement-mode topk --content-ratio 0.07 --traj-ratio 0.03 --gpu 0 \
    && $PYTHON -m TrajGazeMerge.eval.mcnemar \
    --a $DUMP/tbudget_w10.jsonl --b $DUMP/tbudget_w00.jsonl \
    > $DUMP/tbudget_mcnemar_w10_vs_w00.txt 2>&1 \
    && $PYTHON -m TrajGazeMerge.eval.mcnemar \
    --a $DUMP/tbudget_w05.jsonl --b $DUMP/tbudget_w00.jsonl \
    >> $DUMP/tbudget_mcnemar_w10_vs_w00.txt 2>&1 \
    && echo '[tbudget McNemar done]'
" >> "$R/tbudget_mcnemar.log" 2>&1 &
echo "[$(date)] tbudget McNemar runner armed (GPU3)" | tee -a "$LOG"

# ── gazetext_gaze on GPU 0 ──────────────────────────────────────────────────
echo "[$(date)] Launching gazetext_gaze on GPU 0 ..." | tee -a "$LOG"
mkdir -p "$R/gazetext_gaze"
setsid bash -c "
  cd /workspace/trajgaze_st
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29750 \
    -m TrajGazeMerge.training.train_visionzip_gazetext_lora \
    --arm gaze \
    --output-dir $R/gazetext_gaze \
    --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop
" >> "$R/gazetext_gaze/run.log" 2>&1 &
echo "[$(date)] gazetext_gaze launched (GPU0)" | tee -a "$LOG"

# ── gazetext_random on GPU 2 ────────────────────────────────────────────────
echo "[$(date)] Launching gazetext_random on GPU 2 ..." | tee -a "$LOG"
mkdir -p "$R/gazetext_random"
setsid bash -c "
  cd /workspace/trajgaze_st
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=2 $PYTHON -m torch.distributed.run \
    --nproc_per_node=1 --master_port=29752 \
    -m TrajGazeMerge.training.train_visionzip_gazetext_lora \
    --arm random \
    --output-dir $R/gazetext_random \
    --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop
" >> "$R/gazetext_random/run.log" 2>&1 &
echo "[$(date)] gazetext_random launched (GPU2)" | tee -a "$LOG"

echo "[$(date)] All gazetext arms armed. Monitoring..." | tee -a "$LOG"

# ── wait for gazetext to complete, then McNemar vs M1 ──────────────────────
while true; do
    gaze_done=0
    random_done=0
    [ -f "$R/gazetext_gaze/best.pth" ] && gaze_done=1
    [ -f "$R/gazetext_random/best.pth" ] && random_done=1
    if [ "$gaze_done" -eq 1 ] && [ "$random_done" -eq 1 ]; then
        echo "[$(date)] Both gazetext arms done. Running McNemar vs M1..." | tee -a "$LOG"
        break
    fi
    echo "[$(date)] gazetext: gaze_done=$gaze_done random_done=$random_done. Sleep 300s..." | tee -a "$LOG"
    sleep 300
done

# Dump + McNemar for gazetext arms
CKPT_GAZE=$R/gazetext_gaze/best.pth
CKPT_RAND=$R/gazetext_random/best.pth
M1_DUMP=$R/dumps/m1.jsonl

setsid bash -c "
  GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0 $PYTHON -m TrajGazeMerge.eval.eval_dump_gazetext \
    --ckpt $CKPT_GAZE --arm gaze --dump $DUMP/gazetext_gaze.jsonl --gpu 0 \
    && GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0 $PYTHON -m TrajGazeMerge.eval.eval_dump_gazetext \
    --ckpt $CKPT_RAND --arm random --dump $DUMP/gazetext_random.jsonl --gpu 0 \
    && cd /workspace/trajgaze_st \
    && $PYTHON -m TrajGazeMerge.eval.mcnemar --a $DUMP/gazetext_gaze.jsonl --b $M1_DUMP \
    > $DUMP/gazetext_mcnemar.txt 2>&1 \
    && $PYTHON -m TrajGazeMerge.eval.mcnemar --a $DUMP/gazetext_random.jsonl --b $M1_DUMP \
    >> $DUMP/gazetext_mcnemar.txt 2>&1 \
    && $PYTHON -m TrajGazeMerge.eval.mcnemar --a $DUMP/gazetext_gaze.jsonl --b $DUMP/gazetext_random.jsonl \
    >> $DUMP/gazetext_mcnemar.txt 2>&1 \
    && echo '[gazetext McNemar done]'
" >> "$R/gazetext_mcnemar.log" 2>&1
echo "[$(date)] gazetext McNemar complete. Results: $DUMP/gazetext_mcnemar.txt" | tee -a "$LOG"
