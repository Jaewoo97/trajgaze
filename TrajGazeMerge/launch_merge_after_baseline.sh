#!/bin/bash
# Wait for baseline LoRA (PIDs 204471/204611/204612) to finish,
# then launch TrajGazeMerge with the trained LoRA as teacher on GPU 0,1.

BASELINE_PID=204471
BASELINE_CKPT="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth"
LOG="/workspace/EgoGazeVQA/TrajGazeMerge/logs/merge_lora.log"

echo "[$(date)] Waiting for baseline LoRA (PID $BASELINE_PID) to finish..."
while kill -0 $BASELINE_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] Baseline LoRA finished. Launching TrajGazeMerge..."

mkdir -p "$(dirname $LOG)"

CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/gaze/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    -m TrajGazeMerge.training.train_merge_lora \
    --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \
    --teacher-ckpt "$BASELINE_CKPT" \
    --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora \
    --epochs 3 --lr-lora 1e-4 --lr-enc 1e-5 --alpha 0.5 \
    --merge-ratio 0.9 --grad-accum 4 --log-every 20 --eval-every 200 \
    > "$LOG" 2>&1

echo "[$(date)] TrajGazeMerge training complete."
