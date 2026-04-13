cd /workspace/EgoGazeVQA
CUDA_VISIBLE_DEVICES=0,1 /workspace/vila_eval_env/bin/python eval/eval_qwen25vl_with_autogaze.py \
    --frames-dir /workspace/datasets/StreamGaze_v2/frames \
    --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction \
    2>&1 | tee /workspace/EgoGazeVQA/AutoGaze/results/qwen25vl_with_autogaze_01_64f.log
