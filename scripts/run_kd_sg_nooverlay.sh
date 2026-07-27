#!/bin/bash
# SG specialist KD student, TRULY gaze-free: the student's VLM input has no gaze
# marker in the pixels, while the teacher's TAS stream keeps it.
#
#   GAZE_OVERLAY=1      -> traj_frame_paths = frames/{ds}/viz/...       (teacher)
#   VLM_GAZE_OVERLAY=0  -> vlm_frame_paths  = frames/{ds}/original/...  (student)
#
# The teacher stays on viz because stage1_tas_3way_overlay was trained on overlay
# frames and retraining it is 100 epochs x 4 GPUs. The student is the deployed
# artifact and must not see the marker.
#
# PREREQUISITE, now satisfied: frames/{ds}/original had to match frames/{ds}/viz
# per video. holoassist did not — the two encodes declare different frame rates
# (same nb_frames, 24.46 vs 29.83 fps), so `-vf fps=10` sampled different moments
# and 32/66 videos were silently desynchronised. Repaired by
# scripts/fix_sg_original_fps.sh; all three datasets now verified parity-clean
# (egtea 35/35, holoassist 66/66, egoexolearn 176/180 exact + 4 off by one frame).
#
# Baseline to beat: 354 items (67.30%), which is the OVERLAY-trained student run
# off-distribution on original frames. This run removes that distribution shift.
#
#   nohup setsid scripts/run_kd_sg_nooverlay.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze || exit 1
source env.sh

export VLM_GAZE_OVERLAY=0          # student pixels: no marker
export GAZE_OVERLAY=1              # teacher TAS stream: marker kept

LOG=$REPO/kd_train_sgonly_nooverlay.log
OUT=$REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_SGonly_nooverlay

# Wait for both GPUs to be free — an eval may still be finishing on one of them.
wait_gpu () {
    local i=$1
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i")" -gt 2000 ]; do
        sleep 60
    done
}
echo "=== SG KD student, VLM_GAZE_OVERLAY=0 — waiting for GPUs $(date -Is) ===" >>"$LOG"
wait_gpu 0; wait_gpu 1
echo "=== GPUs free, starting $(date -Is) ===" >>"$LOG"

CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29663 \
  -m TrajGazeMerge.training.train_visionzip_kd_lora \
  --source sg --warmstart-ckpt "$M1_SGONLY" --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 2 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 --no-hdepic --early-stop \
  >>"$LOG" 2>&1

echo "=== SG KD student NO-OVERLAY DONE $(date -Is) ===" >>"$LOG"
