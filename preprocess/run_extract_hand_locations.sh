#!/bin/bash
# Run hand location extraction across 4 GPUs in parallel (sharded by video).
# Each GPU handles ~1/4 of the 281 videos.
#
# Usage: bash preprocess/run_extract_hand_locations.sh

PYTHON=/opt/conda/envs/gaze/bin/python
SCRIPT=preprocess/extract_hand_locations.py
FRAMES_DIR=/workspace/datasets/StreamGaze_v2/frames
OUTPUT=/workspace/EgoGazeVQA/preprocess/hand_locations
CKPT=/workspace/EgoGazeVQA/hand_object_detector/models/res101_handobj_100k/pascal_voc/faster_rcnn_1_8_132028.pth
LOG_DIR=/workspace/EgoGazeVQA/preprocess/logs
N_SHARDS=4

mkdir -p "$LOG_DIR"

cd /workspace/EgoGazeVQA

for SHARD in 0 1 2 3; do
  GPU=$SHARD
  echo "Launching shard $SHARD on GPU $GPU ..."
  CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda-11.8 \
    $PYTHON $SCRIPT \
      --frames-dir "$FRAMES_DIR" \
      --output     "$OUTPUT" \
      --ckpt       "$CKPT" \
      --shard      $SHARD \
      --n-shards   $N_SHARDS \
      --skip-existing \
    2>&1 | tee "$LOG_DIR/shard_${SHARD}.log" &
done

echo "All 4 shards launched. PIDs:"
jobs -l
wait
echo "All shards done."
