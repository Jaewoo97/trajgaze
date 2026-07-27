#!/bin/bash
# Re-evaluate the two MACHINE-1 checkpoints on THIS machine, so the bar and the
# reference student are measured in the same environment as everything else here.
# §6.1 established a one-sided EG deviation across machines, so machine-1's
# 62.51 / 62.31 cannot be compared to local numbers directly — they must be re-run.
#
#   visionzip_lora_sgeg_overlay   → the 62.51 content-only VisionZip baseline = THE BAR
#   visionzip_kd_selection_overlay → the 62.31 KD student (ep1); tells us whether the
#                                    local 61.33 is an environment shift or a worse run.
#
# Eval-only scores on rank 0 only, so 1 GPU per job; the two run concurrently.
#
#   nohup setsid scripts/run_m1_bar_evals.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze
source env.sh

M1DIR=/NHNHOME/VILAB/vilab_yj/datasets/trajgazemerge/hf_m1/aaai

# ── the bar: content-only VisionZip @ 10% (dominant 5% + contextual 5%) ──
run_baseline () {
    local LOG="$REPO/eval_m1bar_visionzip.log"
    echo "=== VisionZip baseline (machine-1 ckpt) eval start $(date -Is) ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=0 $TORCHRUN --nproc_per_node=1 --master_port=29721 \
      -m TrajGazeMerge.training.train_visionzip_lora \
      --eval-ckpt "$M1DIR/visionzip_lora_sgeg_overlay/best.pth" \
      --dominant-ratio 0.05 --contextual-ratio 0.05 \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_eval_m1bar_visionzip" \
      --no-hdepic >>"$LOG" 2>&1
    echo "=== VisionZip baseline EVAL DONE $(date -Is) ===" >>"$LOG"
}

# ── the machine-1 KD student (62.31, ep1) under the identical gaze-free protocol ──
run_student () {
    local LOG="$REPO/eval_m1bar_kdstudent.log"
    echo "=== machine-1 KD student eval start $(date -Is) ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=1 $TORCHRUN --nproc_per_node=1 --master_port=29722 \
      -m TrajGazeMerge.training.train_visionzip_kd_lora \
      --eval-ckpt "$M1DIR/visionzip_kd_selection_overlay/best.pth" \
      --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio 0.07 --traj-ratio 0.03 --source both \
      --output-dir "$REPO/TrajGazeMerge/checkpoints/_eval_m1bar_kdstudent" \
      --no-hdepic >>"$LOG" 2>&1
    echo "=== machine-1 KD student EVAL DONE $(date -Is) ===" >>"$LOG"
}

run_baseline & PID_B=$!
run_student  & PID_S=$!
wait $PID_B $PID_S
echo "=== BOTH BAR EVALS DONE $(date -Is) ===" >>"$REPO/eval_m1bar_visionzip.log"
