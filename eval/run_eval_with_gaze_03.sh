cd /workspace/EgoGazeVQA
CUDA_VISIBLE_DEVICES=2,3 /workspace/vila_eval_env/bin/python eval/eval_nvila_with_gaze_03.py \
    --frames-dir /workspace/datasets/StreamGaze_v2/frames \
    --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction
