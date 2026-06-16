#!/usr/bin/env bash
# Sprint 1 (A+B) — Diagnostic re-run on the new ckpt.
#
# Waits for Stage 2 (E1_sprint1_AB_keep10) to finish, then runs the 4-metric
# verdict pipeline:
#   - diagnostic_eval  (per-sample M1 + summary)
#   - analyze_diagnostics  (aggregate)
#   - counterfactual_mask_eval  (shuffle_kept, mask_kept_*)
#   - prints 4-metric verdict to stdout

set -euo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
S1=$REPO/TrajGaze_v2/checkpoints/E1_sprint1_AB/best.pth
LORA=$REPO/TrajGazeMerge/checkpoints/E1_sprint1_AB_keep10/best.pth
TAG=E1_sprint1_AB
LAUNCHER=$REPO/TrajGazeMerge/eval_results/E1_sprint1_AB_diagnostics_launcher.log
mkdir -p "$(dirname "$LAUNCHER")"

echo "[$(date -u)] Sprint1 diagnostics launcher waiting on Stage-2 ckpt" > "$LAUNCHER"

# Wait until Stage 2 best.pth + Stage 2 process gone
while true; do
    if pgrep -f "train_merge_lora_temporal_no_kd.*E1_sprint1_AB_keep10" >/dev/null 2>&1; then
        sleep 120
        continue
    fi
    if [ -f "$LORA" ]; then
        echo "[$(date -u)] Stage-2 done. best.pth detected at $LORA" >> "$LAUNCHER"
        break
    fi
    echo "[$(date -u)] Stage-2 process gone but best.pth missing; retrying in 120s." >> "$LAUNCHER"
    sleep 120
done

cd "$REPO"

echo "[$(date -u)] Running diagnostic_eval ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.diagnostic_eval \
    --stage1-ckpt "$S1" --lora-ckpt "$LORA" --tag ${TAG}_diag \
    >> "$LAUNCHER" 2>&1

echo "[$(date -u)] Running analyze_diagnostics ..." >> "$LAUNCHER"
PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.analyze_diagnostics --tag ${TAG}_diag \
    >> "$LAUNCHER" 2>&1

echo "[$(date -u)] Running counterfactual_mask_eval ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.counterfactual_mask_eval \
    --stage1-ckpt "$S1" --lora-ckpt "$LORA" --tag ${TAG}_mask \
    >> "$LAUNCHER" 2>&1

echo "[$(date -u)] Verdict:" >> "$LAUNCHER"
$PY <<EOF >> "$LAUNCHER" 2>&1
import json
diag = json.load(open("$REPO/TrajGazeMerge/eval_results/diagnostic/${TAG}_diag_summary.json"))
mask = json.load(open("$REPO/TrajGazeMerge/eval_results/diagnostic/${TAG}_mask_mask_summary.json"))
g = diag.get("global_means", {})
acc = diag.get("overall_accuracy", float("nan"))
recall = g.get("gt_gaze_recall", float("nan"))
late = g.get("late_half_ratio", float("nan"))
b = mask["variants"]["baseline"]["overall_accuracy"]
sh = mask["variants"]["shuffle_kept"]["overall_accuracy"]
print(f"acc            = {acc:.2f}%  (target >= 67)")
print(f"gt_gaze_recall = {recall:.3f}   (target > 0.20)")
print(f"late_half      = {late:.3f}   (target ~ 0.50)")
print(f"shuffle_kept   Delta = {sh-b:+.2f}pp   (target < -3)")
EOF

echo "[$(date -u)] Diagnostics complete." >> "$LAUNCHER"
