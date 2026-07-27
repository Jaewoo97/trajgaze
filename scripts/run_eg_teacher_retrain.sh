#!/bin/bash
# PHASE 2 — retrain the M1 EG-only teacher. NOT to be launched until the Phase 1
# evals (scripts/run_eg_teacher_evals.sh) have been read; they can change the target.
#
# WHY. The shipped visionzip_complement_learned_EGonly_overlay/best.pth scores
# 53.81 / 54.23 here, and 53.81 equals machine-1's *ep2* value to two decimals while
# its ep1 was 54.85 — i.e. the shipped file is probably the ep2 snapshot, not the
# best epoch. Retraining and keeping the genuine best recovers ~5 items for free.
#
# THREE HYPOTHESES, only the third is speculative:
#   1. best-epoch selection  -> run A recovers it by construction.
#   2. budget split          -> run B tests v1 §7.5's measured EG optimum (6/4),
#                               where the complement HELPS EG (opposite of SG).
#   3. undertraining         -> EG train is 1265 items = 158 optimizer steps/epoch at
#                               eff-batch 8, so the 2-epoch recipe saw 316 steps. The
#                               joint model saw ~2649 (7064 x 3 / 8) — 8x more. The
#                               "joint training helps EG" story may just be step count.
#                               Hence 6 epochs here, and read the per-epoch curve: if it
#                               is still climbing at 6, the answer is more epochs.
#
# --early-stop is deliberately NOT passed: it only compares epoch 2 against epoch 1,
# which would abort exactly the hypothesis being tested. best.pth already tracks the
# best epoch on its own, so letting all 6 run costs nothing but time.
#
# One run per GPU with --grad-accum 8 keeps eff-batch at 8 (1 x 8 == 2 x 4), so both
# configs stay comparable to every number on record.
#
# Teachers keep the gaze overlay (v2 §7.3): GAZE_OVERLAY=1, VLM_GAZE_OVERLAY unset.
#
#   nohup setsid scripts/run_eg_teacher_retrain.sh &

cd /NHNHOME/VILAB/vilab_yj/trajgaze || exit 1
source env.sh
unset VLM_GAZE_OVERLAY

EPOCHS=${EPOCHS:-6}

one () {           # $1=tag  $2=content_ratio  $3=traj_ratio  $4=gpu  $5=port
    local LOG="$REPO/train_egteacher_$1.log"
    local OUT="$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_EGonly_$1"
    echo "=== EG teacher retrain [$1] content=$2 traj=$3 epochs=$EPOCHS $(date -Is) ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=$4 $TORCHRUN --nproc_per_node=1 --master_port=$5 \
      -m TrajGazeMerge.training.train_visionzip_complement_lora \
      --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
      --content-ratio "$2" --traj-ratio "$3" --source eg \
      --output-dir "$OUT" \
      --epochs "$EPOCHS" --lr 1e-4 --grad-accum 8 --no-hdepic \
      --eval-progress-every 100 >>"$LOG" 2>&1
    echo "=== [$1] DONE $(date -Is) ===" >>"$LOG"
}

# A: locked 7/3 default — the honest EG-only ceiling with the true best epoch kept.
# B: 6/4 — v1 §7.5's measured EG optimum.
one A_7_3 0.07 0.03 0 29841 &  PA=$!
one B_6_4 0.06 0.04 1 29842 &  PB=$!
wait $PA $PB
echo "=== BOTH EG TEACHER RETRAINS DONE $(date -Is) ===" >>"$REPO/train_egteacher_A_7_3.log"
