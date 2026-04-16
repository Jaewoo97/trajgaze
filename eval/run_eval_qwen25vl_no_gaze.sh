cd /home/yujin/gaze/trajgaze
CUDA_VISIBLE_DEVICES=2,3 python eval/eval_qwen25vl_no_gaze.py \
    --frames-dir /workspace/datasets/StreamGaze_v2/frames \
    --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction \
    2>&1 | tee /workspace/EgoGazeVQA/AutoGaze/results/qwen25vl_no_gaze_64f.log
