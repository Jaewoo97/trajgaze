cd /home/yujin/gaze/trajgaze
CUDA_VISIBLE_DEVICES=0,1 python eval/eval_qwen25vl_no_gaze_egogazevqa.py \
    --data-root /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
    --output /home/yujin/gaze/trajgaze/results/egogazevqa \
    2>&1 | tee /home/yujin/gaze/trajgaze/results/qwen25vl_no_gaze_egogazevqa_64f.log
