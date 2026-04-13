cd /workspace/EgoGazeVQA
/workspace/vila_eval_env/bin/python eval/eval_nvila_with_autogaze.py \
    --autogaze ntp \
    --frames-dir /workspace/datasets/StreamGaze_v2/frames \
    --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction
