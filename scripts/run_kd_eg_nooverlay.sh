#!/bin/bash
# EG specialist KD student, TRULY gaze-free: the student's VLM input has no gaze
# marker in the pixels, while the teacher's TAS stream keeps it.
# EG analogue of run_kd_sg_nooverlay.sh.
#
#   GAZE_OVERLAY=1      -> traj_frame_paths = {ds}/gaze/...      (teacher)
#   VLM_GAZE_OVERLAY=0  -> vlm_frame_paths  = {ds}/no_gaze/...   (student)
#
# The teacher stays on the overlay frames because stage1_tas_3way_overlay was
# trained on them and retraining it is 100 epochs x 4 GPUs. The student is the
# deployed artifact and must not see the marker.
#
# NO frame preparation is needed on the EG side, unlike SG: `no_gaze` already
# exists for all three splits with byte-identical filenames to `gaze` and zero
# count mismatches (263 videos: egtea 82 / ego4d 27 / egoexo 154). There was no
# extraction step and therefore no fps-desynchronisation risk — the holoassist
# defect (kd_handoff_v2.md §7.4) is SG-only, holoassist does not exist in EG.
#
# Baseline to beat: 272 items (56.08%), the OVERLAY-trained EG student
# (visionzip_kd_selection_EGonly_overlay). Per-task baseline in §10.3 — its whole
# +9 over the teacher is temporal, so watch that column.
#
# Expect this line in the log; its absence means the run is not the experiment
# it claims to be:
#   [KD] frame streams: student VLM='no_gaze'  teacher TAS='gaze'
#
#   nohup setsid scripts/run_kd_eg_nooverlay.sh &

cd "$(dirname "$0")/.." || exit 1
source env.sh

export VLM_GAZE_OVERLAY=0          # student pixels: no marker
export GAZE_OVERLAY=1              # teacher TAS stream: marker kept

LOG=$REPO/kd_train_egonly_nooverlay.log
OUT=$REPO/TrajGazeMerge/checkpoints/visionzip_kd_selection_EGonly_nooverlay

# Wait for both GPUs to be free — an eval may still be finishing on one of them.
wait_gpu () {
    local i=$1
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i")" -gt 2000 ]; do
        sleep 60
    done
}
echo "=== EG KD student, VLM_GAZE_OVERLAY=0 — waiting for GPUs $(date -Is) ===" >>"$LOG"
wait_gpu 0; wait_gpu 1
echo "=== GPUs free, starting $(date -Is) ===" >>"$LOG"

CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29662 \
  -m TrajGazeMerge.training.train_visionzip_kd_lora \
  --source eg --warmstart-ckpt "$M1_EGONLY" --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir "$OUT" \
  --epochs 2 --lr 1e-4 --pred-lr 1e-3 --grad-accum 4 --no-hdepic --early-stop \
  >>"$LOG" 2>&1

echo "=== EG KD student NO-OVERLAY DONE $(date -Is) ===" >>"$LOG"
