#!/bin/bash
# Cross-keep-ratio diagnostic: run M1 (diagnostic_eval) + M2 (analyze_diagnostics)
# on E1_patch_temporal ckpts trained at different keep ratios (03/05/10).
#
# Compares how temporal bias, gt_gaze_recall, and feature effect sizes change
# as the token budget shrinks — tells us whether the "anti-gaze" / "late-frame"
# patterns are fundamental properties or budget-dependent.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash TrajGazeMerge/eval/run_cross_keep_diagnostic.sh

set -e
ROOT=/workspace/trajgaze
S1=$ROOT/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth
PY=/opt/conda/envs/gaze/bin/python
LOG_DIR=$ROOT/TrajGazeMerge/eval_results/diagnostic/logs

mkdir -p $LOG_DIR
cd $ROOT

declare -A RATIOS=(
  ["keep03"]="0.97"
  ["keep05"]="0.95"
  ["keep10"]="0.90"
)

for kr in keep03 keep05 keep10; do
  CKPT=$ROOT/TrajGazeMerge/checkpoints/E1_patch_temporal_${kr}_bs4/best.pth
  TAG=E1_${kr}_diag
  MR=${RATIOS[$kr]}

  echo "[$(date +%T)] starting $TAG (merge_ratio=$MR, ckpt=$CKPT)"
  $PY -m TrajGazeMerge.eval.diagnostic_eval \
    --stage1-ckpt $S1 --lora-ckpt $CKPT \
    --merge-ratio $MR --tag $TAG \
    > $LOG_DIR/cross_${kr}_diag.log 2>&1
  echo "[$(date +%T)]   $TAG eval done"

  $PY -m TrajGazeMerge.eval.analyze_diagnostics --tag $TAG \
    > $LOG_DIR/cross_${kr}_analyze.log 2>&1
  echo "[$(date +%T)]   $TAG analyze done"
done

echo "[$(date +%T)] cross-keep-ratio diagnostic ALL DONE"
echo ""
echo "Compare across ratios:"
for kr in keep03 keep05 keep10; do
  echo "  $kr summary:"
  $PY -c "
import json
d = json.load(open('$ROOT/TrajGazeMerge/eval_results/diagnostic/E1_${kr}_diag_summary.json'))
g = d['global_means']
print(f'    overall_acc={d[\"overall_accuracy\"]:.2f}%  '
      f'late_half={g[\"late_half_ratio\"]:.3f}  '
      f'gt_gaze_recall={g[\"gt_gaze_recall\"]:.3f}  '
      f'temporal_CoM={g[\"temporal_center_of_mass\"]:.3f}')
"
done
