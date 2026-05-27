#!/usr/bin/env bash
# E1 A+B encoder + keep 5% (option B) — Diagnostic re-run.
# Tests intervention E (tighter budget) on top of Sprint 1's A+B encoder.

set -euo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
S1=$REPO/TrajGaze_v2/checkpoints/E1_sprint1_AB/best.pth
LORA=$REPO/TrajGazeMerge/checkpoints/E1_AB_keep05/best.pth
TAG=E1_AB_keep05
LAUNCHER=$REPO/TrajGazeMerge/eval_results/E1_AB_keep05_diagnostics_launcher.log
mkdir -p "$(dirname "$LAUNCHER")"

echo "[$(date -u)] E1_AB_keep05 diagnostics launcher waiting on Stage-2 ckpt" > "$LAUNCHER"

while true; do
    if pgrep -f "train_merge_lora_temporal_no_kd.*E1_AB_keep05" >/dev/null 2>&1; then
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

# Use GPU 1 since GPU 0 may still be running Sprint 2.1 diagnostics.
# If GPU 0 is free by then, this still works (--gpu arg already on script).
GPU_USE=1

echo "[$(date -u)] Running diagnostic_eval ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=$GPU_USE PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.diagnostic_eval \
    --stage1-ckpt "$S1" --lora-ckpt "$LORA" --tag ${TAG}_diag --merge-ratio 0.95 \
    >> "$LAUNCHER" 2>&1

echo "[$(date -u)] Running analyze_diagnostics ..." >> "$LAUNCHER"
PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.analyze_diagnostics --tag ${TAG}_diag \
    >> "$LAUNCHER" 2>&1

echo "[$(date -u)] Running counterfactual_mask_eval ..." >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=$GPU_USE PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.counterfactual_mask_eval \
    --stage1-ckpt "$S1" --lora-ckpt "$LORA" --tag ${TAG}_mask --merge-ratio 0.95 \
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
me = mask["variants"]["mask_kept_early"]["overall_accuracy"]
print(f"=== Option B (E1 A+B encoder + keep 5%) verdict ===")
print(f"acc            = {acc:.2f}%   (baseline E1 keep05 64.83; Sprint1 keep10 65.59)")
print(f"gt_gaze_recall = {recall:.3f}   (random for keep05 = 0.05)")
print(f"late_half      = {late:.3f}")
print(f"shuffle_kept   Delta = {sh-b:+.2f}pp   (target < -1.5)")
print(f"mask_kept_early Delta = {me-b:+.2f}pp")
EOF

echo "[$(date -u)] Diagnostics complete." >> "$LAUNCHER"
