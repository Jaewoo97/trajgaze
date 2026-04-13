cd /workspace/EgoGazeVQA
CUDA_VISIBLE_DEVICES=0,1 /workspace/vila_eval_env/bin/python eval/eval_nvila_no_gaze.py \
    --frames-dir /workspace/datasets/StreamGaze_v2/frames \
    --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction
