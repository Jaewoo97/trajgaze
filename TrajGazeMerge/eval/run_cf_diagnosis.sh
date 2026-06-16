#!/usr/bin/env bash
# Step S3 — cf-mask diagnostic on CF-1 and CF-3 best.pth after training ends.
# Waits for training procs to exit (same pgrep pattern as run_step1_diagnosis.sh),
# then runs cf-mask 7 variants × {streamgaze, egovqa} × {CF-1, CF-3} = 4 slots.
#
# (HD-EPIC val not supported by counterfactual_mask_eval.py; documented limitation.)

set -uo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
S1_TAS=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth
DIAG_DIR=$REPO/TrajGazeMerge/eval_results/diagnostic
LAUNCHER=$REPO/TrajGazeMerge/eval_results/E1_cf_diagnosis_launcher.log

CF1_CKPT=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf1_hdepic_bs8_mb2/best.pth
CF3_CKPT=$REPO/TrajGazeMerge/checkpoints/E1_combined_cf3_hdepic_bs8_mb2/best.pth

mkdir -p "$DIAG_DIR"
echo "[$(date -u)] CF diagnosis launcher start" > "$LAUNCHER"

# Wait for training to exit
while pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" >/dev/null 2>&1; do
    echo "[$(date -u)] Waiting on train_merge_lora_batched..." >> "$LAUNCHER"
    sleep 300
done
echo "[$(date -u)] Training exited; sleeping 60s for CUDA release..." >> "$LAUNCHER"
sleep 60

GPU0_USED=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
GPU1_USED=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
echo "[$(date -u)] GPU mem after wait: 0=${GPU0_USED}MiB 1=${GPU1_USED}MiB" >> "$LAUNCHER"
if [ "$GPU0_USED" -gt 5000 ] || [ "$GPU1_USED" -gt 5000 ]; then
    echo "[$(date -u)] WARN — GPU mem still high; attempting orphan cleanup" >> "$LAUNCHER"
    pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" | xargs -r kill -9 2>/dev/null
    sleep 30
    GPU0_USED=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU1_USED=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "[$(date -u)] GPU mem after cleanup: 0=${GPU0_USED}MiB 1=${GPU1_USED}MiB" >> "$LAUNCHER"
    if [ "$GPU0_USED" -gt 5000 ] || [ "$GPU1_USED" -gt 5000 ]; then
        echo "[$(date -u)] ABORT — GPU mem still high" >> "$LAUNCHER"
        exit 1
    fi
fi
echo "[$(date -u)] GPUs free; proceeding." >> "$LAUNCHER"

cd "$REPO"

run_cfmask () {
    local CKPT="$1"; local TAG_PREFIX="$2"; local VAL="$3"; local GPU="$4"
    local TAG="${TAG_PREFIX}_${VAL}_cfmask"
    local SUMMARY="$DIAG_DIR/${TAG}_mask_summary.json"

    if [ -f "$SUMMARY" ]; then
        echo "[$(date -u)] SKIP $TAG (summary exists)" >> "$LAUNCHER"
        return 0
    fi
    if [ ! -f "$CKPT" ]; then
        echo "[$(date -u)] MISSING ckpt $CKPT — skipping $TAG" >> "$LAUNCHER"
        return 1
    fi
    echo "[$(date -u)] RUN  $TAG (GPU $GPU)" >> "$LAUNCHER"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO $PY -m \
        TrajGazeMerge.eval.counterfactual_mask_eval \
        --stage1-ckpt "$S1_TAS" --lora-ckpt "$CKPT" \
        --val-dataset "$VAL" --tag "$TAG" \
        >> "$LAUNCHER" 2>&1
    local RC=$?
    echo "[$(date -u)] DONE $TAG (rc=$RC)" >> "$LAUNCHER"
    return $RC
}

# Two parallel GPU streams; 2 jobs each
(
    run_cfmask "$CF1_CKPT" "E1_combined_cf1" streamgaze 0
    run_cfmask "$CF1_CKPT" "E1_combined_cf1" egovqa     0
) &
G0=$!

(
    run_cfmask "$CF3_CKPT" "E1_combined_cf3" streamgaze 1
    run_cfmask "$CF3_CKPT" "E1_combined_cf3" egovqa     1
) &
G1=$!

wait $G0; echo "[$(date -u)] GPU 0 stream done" >> "$LAUNCHER"
wait $G1; echo "[$(date -u)] GPU 1 stream done" >> "$LAUNCHER"

echo "[$(date -u)] === Verdict summary ===" >> "$LAUNCHER"
$PY <<EOF >> "$LAUNCHER" 2>&1
import json, os
base = "$DIAG_DIR"
ckpts_old = [  # for comparison with prior Step 1 runs
    ("E1_combined_TAS_only",   "TAS-only (baseline)"),
]
ckpts_new = [
    ("E1_combined_cf1",        "CF-1 (mask only)"),
    ("E1_combined_cf3",        "CF-3 (mask+shuf)"),
]
vals = ["streamgaze", "egovqa"]
def load(ck, v):
    p = os.path.join(base, f"{ck}_{v}_cfmask_mask_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None

print(f"{'ckpt':32s} {'val':12s} {'base':>6s} {'mask_kept':>10s} {'mask_gaze':>10s} {'shuf':>8s}")
for ck, label in ckpts_old + ckpts_new:
    for v in vals:
        s = load(ck, v)
        if s is None:
            print(f"  {label:30s} {v:10s} MISSING")
            continue
        var = s["variants"]
        b = var["baseline"]["overall_accuracy"]
        mk = var["mask_kept"]["overall_accuracy"]
        mg = var["mask_gaze"]["overall_accuracy"]
        sh = var["shuffle_kept"]["overall_accuracy"]
        print(f"  {label:30s} {v:10s} {b:6.2f} {mk-b:+10.2f} {mg-b:+10.2f} {sh-b:+8.2f}")

# Decision gate per plan §4.2
print()
print("=== Decision gate ===")
for ck, label in ckpts_new:
    s_eg = load(ck, "egovqa")
    s_sg = load(ck, "streamgaze")
    if s_eg is None or s_sg is None:
        print(f"  {label}: MISSING summaries"); continue
    eg = s_eg["variants"]
    sg = s_sg["variants"]
    mk_eg = eg["mask_kept"]["overall_accuracy"] - eg["baseline"]["overall_accuracy"]
    sg_drop = 63.69 - sg["baseline"]["overall_accuracy"]  # vs TAS-only-hdepic StreamGaze
    gate_a = mk_eg <= -2.0
    gate_b = sg_drop <= 2.0
    print(f"  {label}: EgoGazeVQA mask_kept Δ = {mk_eg:+.2f}  (gate: <= -2.0  {'PASS' if gate_a else 'FAIL'})")
    print(f"  {label}: StreamGaze drop vs TAS-only-hdepic = {sg_drop:+.2f} pp  (gate: <= 2.0  {'PASS' if gate_b else 'FAIL'})")
    print(f"  {label}: OVERALL {'PASS' if gate_a and gate_b else 'FAIL'}")
EOF

echo "[$(date -u)] CF diagnosis launcher complete." >> "$LAUNCHER"
