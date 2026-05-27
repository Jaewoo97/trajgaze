#!/usr/bin/env bash
# Combined diagnostics: waits for Sprint 1 + Sprint 2.1 (combined) Stage 2 to finish,
# then runs diagnostic_eval + counterfactual_mask_eval on BOTH val sets
# (streamgaze_egtea + egovqa_egtea) for each ckpt.
#
# 4 ckpt-val combinations total:
#   - E1_combined_sprint1   × {streamgaze, egovqa}
#   - E1_combined_sprint2_1 × {streamgaze, egovqa}

set -euo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
S1=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB/best.pth
SP1=$REPO/TrajGazeMerge/checkpoints/E1_combined_sprint1/best.pth
SP21=$REPO/TrajGazeMerge/checkpoints/E1_combined_sprint2_1/best.pth
LAUNCHER=$REPO/TrajGazeMerge/eval_results/E1_combined_diagnostics_launcher.log
mkdir -p "$(dirname "$LAUNCHER")"

echo "[$(date -u)] Combined diagnostics launcher waiting on Sprint 2.1 ckpt" > "$LAUNCHER"

# Wait until both Stage 2 trainings finish
while true; do
    if pgrep -f "train_merge_lora_temporal_no_kd.*E1_combined_sprint" >/dev/null 2>&1; then
        sleep 180
        continue
    fi
    if [ -f "$SP1" ] && [ -f "$SP21" ]; then
        echo "[$(date -u)] Both Stage 2 ckpts present, proceeding" >> "$LAUNCHER"
        break
    fi
    echo "[$(date -u)] Trainers gone but ckpts missing; sleep 180s" >> "$LAUNCHER"
    sleep 180
done

cd "$REPO"

run_diag () {
    local CKPT="$1"; local TAG="$2"; local VALDS="$3"; local GPU="$4"
    echo "[$(date -u)] $TAG diagnostic_eval on $VALDS (GPU $GPU)" >> "$LAUNCHER"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.diagnostic_eval \
        --stage1-ckpt "$S1" --lora-ckpt "$CKPT" \
        --val-dataset "$VALDS" \
        --tag "${TAG}_${VALDS}_diag" \
        >> "$LAUNCHER" 2>&1

    echo "[$(date -u)] $TAG analyze_diagnostics on $VALDS" >> "$LAUNCHER"
    PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.analyze_diagnostics \
        --tag "${TAG}_${VALDS}_diag" >> "$LAUNCHER" 2>&1

    echo "[$(date -u)] $TAG counterfactual_mask_eval on $VALDS (GPU $GPU)" >> "$LAUNCHER"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO $PY -m TrajGazeMerge.eval.counterfactual_mask_eval \
        --stage1-ckpt "$S1" --lora-ckpt "$CKPT" \
        --val-dataset "$VALDS" \
        --tag "${TAG}_${VALDS}_mask" \
        >> "$LAUNCHER" 2>&1
}

# 4 combinations
run_diag "$SP1"  "E1_combined_sprint1"   streamgaze 0
run_diag "$SP1"  "E1_combined_sprint1"   egovqa     0
run_diag "$SP21" "E1_combined_sprint2_1" streamgaze 1
run_diag "$SP21" "E1_combined_sprint2_1" egovqa     1

echo "[$(date -u)] Verdict summary:" >> "$LAUNCHER"
$PY <<EOF >> "$LAUNCHER" 2>&1
import json
def load(tag):
    base = "/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic"
    try:
        d = json.load(open(f"{base}/{tag}_diag_summary.json"))
        m = json.load(open(f"{base}/{tag}_mask_mask_summary.json"))
        return d, m
    except FileNotFoundError:
        return None, None

print(f"{'ckpt':30s} {'val':15s} {'acc':>6s} {'recall':>8s} {'late':>6s} {'shuf_Δ':>8s} {'early_Δ':>8s}")
for ckpt in ("E1_combined_sprint1", "E1_combined_sprint2_1"):
    for v in ("streamgaze", "egovqa"):
        d, m = load(f"{ckpt}_{v}")
        if d is None:
            print(f"  {ckpt}/{v}: MISSING")
            continue
        g = d.get("global_means", {})
        acc = d.get("overall_accuracy", float("nan"))
        recall = g.get("gt_gaze_recall", float("nan"))
        late = g.get("late_half_ratio", float("nan"))
        b = m["variants"]["baseline"]["overall_accuracy"]
        sh = m["variants"]["shuffle_kept"]["overall_accuracy"]
        me = m["variants"]["mask_kept_early"]["overall_accuracy"]
        print(f"  {ckpt:28s} {v:13s} {acc:6.2f} {recall:8.3f} {late:6.3f} {sh-b:+8.2f} {me-b:+8.2f}")
EOF

echo "[$(date -u)] Combined diagnostics complete." >> "$LAUNCHER"
