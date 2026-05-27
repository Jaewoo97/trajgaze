#!/usr/bin/env bash
# Step 1 diagnostics — cf-mask on the 4 combined ablation Stage-2 ckpts.
# Waits for in-flight HD-EPIC training (train_merge_lora_batched) to free
# GPU 0/1 before launching anything. Uses `mask_kept` variant as the
# language-prior floor proxy (text_only equivalent).
#
# Comparisons produced (per ckpt × val × 7 variants):
#   - E1_combined_TAS_only          (Stage1 = AB_TAS,  TAS encoder, CE only)
#   - E1_combined_TAS_ATR_CGM       (Stage1 = AB_TAS,  + ATR + CGM)         ← FULL
#   - E1_combined_TAS_ATR           (Stage1 = AB_TAS,  + ATR only)
#   - E1_combined_CGM_only          (Stage1 = AB,      + CGM only, no TAS)
#
# NOTE: TAS_only and FULL already have cf-mask results from a prior run
# (see eval_results/diagnostic/E1_combined_TAS_*_*_cfmask_mask_summary.json).
# They are RE-RUN here only if their summary file is missing, so this script
# is idempotent — safe to re-launch.

set -uo pipefail

REPO=/workspace/trajgaze
PY=/opt/conda/envs/gaze/bin/python
S1_TAS=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth
S1_OLD=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB/best.pth
DIAG_DIR=$REPO/TrajGazeMerge/eval_results/diagnostic
LAUNCHER=$REPO/TrajGazeMerge/eval_results/E1_step1_diagnosis_launcher.log

mkdir -p "$DIAG_DIR"
echo "[$(date -u)] Step 1 cf-mask launcher start" > "$LAUNCHER"

# ── Wait for HD-EPIC training to release GPUs ─────────────────────────────
# Match the in-flight processes (train_merge_lora_batched); exclude this script.
while pgrep -f "TrajGazeMerge.training.train_merge_lora_batched" >/dev/null 2>&1; do
    echo "[$(date -u)] Waiting on train_merge_lora_batched (HD-EPIC)..." >> "$LAUNCHER"
    sleep 300
done
# Give CUDA a moment to fully release memory after training exits.
echo "[$(date -u)] Training exited; sleeping 60s for CUDA release..." >> "$LAUNCHER"
sleep 60
# Verify GPUs are actually free (< 5GB used each); abort if not.
GPU0_USED=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
GPU1_USED=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
echo "[$(date -u)] GPU mem after wait: 0=${GPU0_USED}MiB 1=${GPU1_USED}MiB" >> "$LAUNCHER"
if [ "$GPU0_USED" -gt 5000 ] || [ "$GPU1_USED" -gt 5000 ]; then
    echo "[$(date -u)] ABORT — GPU mem still high, something else is using it." >> "$LAUNCHER"
    exit 1
fi
echo "[$(date -u)] GPUs free; proceeding." >> "$LAUNCHER"

cd "$REPO"

# Run one cf-mask job; skip if summary already exists (idempotent).
run_cfmask () {
    local STAGE1="$1"; local CKPT="$2"; local TAG_PREFIX="$3"
    local VAL="$4"; local GPU="$5"
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
        --stage1-ckpt "$STAGE1" --lora-ckpt "$CKPT" \
        --val-dataset "$VAL" --tag "$TAG" \
        >> "$LAUNCHER" 2>&1
    local RC=$?
    echo "[$(date -u)] DONE $TAG (rc=$RC)" >> "$LAUNCHER"
    return $RC
}

# Two parallel GPU streams; each does 4 jobs sequentially (one ckpt × 2 vals).
(
    run_cfmask "$S1_TAS" "$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_only/best.pth"    "E1_combined_TAS_only"     streamgaze 0
    run_cfmask "$S1_TAS" "$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_only/best.pth"    "E1_combined_TAS_only"     egovqa     0
    run_cfmask "$S1_TAS" "$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR/best.pth"     "E1_combined_TAS_ATR"      streamgaze 0
    run_cfmask "$S1_TAS" "$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR/best.pth"     "E1_combined_TAS_ATR"      egovqa     0
) &
G0_PID=$!

(
    run_cfmask "$S1_TAS" "$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR_CGM/best.pth" "E1_combined_TAS_ATR_CGM"  streamgaze 1
    run_cfmask "$S1_TAS" "$REPO/TrajGazeMerge/checkpoints/E1_combined_TAS_ATR_CGM/best.pth" "E1_combined_TAS_ATR_CGM"  egovqa     1
    run_cfmask "$S1_OLD" "$REPO/TrajGazeMerge/checkpoints/E1_combined_CGM_only/best.pth"    "E1_combined_CGM_only"     streamgaze 1
    run_cfmask "$S1_OLD" "$REPO/TrajGazeMerge/checkpoints/E1_combined_CGM_only/best.pth"    "E1_combined_CGM_only"     egovqa     1
) &
G1_PID=$!

wait $G0_PID
echo "[$(date -u)] GPU 0 stream done" >> "$LAUNCHER"
wait $G1_PID
echo "[$(date -u)] GPU 1 stream done" >> "$LAUNCHER"

# ── Aggregate verdict table ───────────────────────────────────────────────
echo "[$(date -u)] === Verdict summary ===" >> "$LAUNCHER"
$PY <<EOF >> "$LAUNCHER" 2>&1
import json, os
base = "$DIAG_DIR"
ckpts = [
    "E1_combined_TAS_only",
    "E1_combined_TAS_ATR",
    "E1_combined_TAS_ATR_CGM",
    "E1_combined_CGM_only",
]
vals = ["streamgaze", "egovqa"]
def load(ck, v):
    p = os.path.join(base, f"{ck}_{v}_cfmask_mask_summary.json")
    if not os.path.exists(p): return None
    return json.load(open(p))

print(f"{'ckpt':28s} {'val':12s} {'base':>6s} {'mask_kept':>10s} {'mask_gaze':>10s} {'mask_hand':>10s} {'shuf':>8s}")
for ck in ckpts:
    for v in vals:
        s = load(ck, v)
        if s is None:
            print(f"  {ck:26s} {v:10s} MISSING")
            continue
        var = s["variants"]
        b = var["baseline"]["overall_accuracy"]
        mk = var["mask_kept"]["overall_accuracy"]
        mg = var["mask_gaze"]["overall_accuracy"]
        mh = var["mask_hand"]["overall_accuracy"]
        sh = var["shuffle_kept"]["overall_accuracy"]
        print(f"  {ck:26s} {v:10s} {b:6.2f} {mk-b:+10.2f} {mg-b:+10.2f} {mh-b:+10.2f} {sh-b:+8.2f}")
EOF

echo "[$(date -u)] Step 1 cf-mask launcher complete." >> "$LAUNCHER"
