#!/bin/bash
# Parallel evaluation on EgoGazeVQA fold-c val (EGTEA) across 4 GPUs.
#
#  GPU 0: NVILA without AutoGaze
#  GPU 1: NVILA with AutoGaze (GRPO, 10% tokens)
#  GPU 2: Qwen2.5-VL-7B without AutoGaze
#  GPU 3: Qwen2.5-VL-7B with AutoGaze (GRPO, 10% tokens)
#
# Logs: /workspace/EgoGazeVQA/eval/results/egogaze_fold_c/logs/

set -e
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$EVAL_DIR/results/egogaze_fold_c/logs"
mkdir -p "$LOG_DIR"

VILA_PY=/workspace/vila_eval_env/bin/python
GAZE_PY="conda run -n gaze python"

echo "======================================================"
echo "EgoGazeVQA fold-c evaluation — 4 GPUs in parallel"
echo "======================================================"

# GPU 0: NVILA no-gaze
echo "[GPU 0] NVILA no-gaze ..."
CUDA_VISIBLE_DEVICES=0 $VILA_PY "$EVAL_DIR/eval_egogaze_nvila_no_gaze.py" \
  > "$LOG_DIR/gpu0_nvila_no_gaze.log" 2>&1 &
PID0=$!

# GPU 1: NVILA + AutoGaze
echo "[GPU 1] NVILA + AutoGaze ..."
CUDA_VISIBLE_DEVICES=1 $VILA_PY "$EVAL_DIR/eval_egogaze_nvila_autogaze.py" \
  > "$LOG_DIR/gpu1_nvila_autogaze.log" 2>&1 &
PID1=$!

# GPU 2: Qwen no-gaze
echo "[GPU 2] Qwen no-gaze ..."
CUDA_VISIBLE_DEVICES=2 conda run -n gaze python "$EVAL_DIR/eval_egogaze_qwen_no_gaze.py" \
  > "$LOG_DIR/gpu2_qwen_no_gaze.log" 2>&1 &
PID2=$!

# GPU 3: Qwen + AutoGaze
echo "[GPU 3] Qwen + AutoGaze ..."
CUDA_VISIBLE_DEVICES=3 conda run -n gaze python "$EVAL_DIR/eval_egogaze_qwen_autogaze.py" \
  > "$LOG_DIR/gpu3_qwen_autogaze.log" 2>&1 &
PID3=$!

echo ""
echo "All 4 jobs launched. PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Logs: $LOG_DIR"
echo ""

# Wait for all
wait $PID0 && echo "[GPU 0] NVILA no-gaze: DONE" || echo "[GPU 0] NVILA no-gaze: FAILED"
wait $PID1 && echo "[GPU 1] NVILA+AutoGaze: DONE" || echo "[GPU 1] NVILA+AutoGaze: FAILED"
wait $PID2 && echo "[GPU 2] Qwen no-gaze: DONE"   || echo "[GPU 2] Qwen no-gaze: FAILED"
wait $PID3 && echo "[GPU 3] Qwen+AutoGaze: DONE"  || echo "[GPU 3] Qwen+AutoGaze: FAILED"

echo ""
echo "======================================================"
echo "Results in: $EVAL_DIR/results/egogaze_fold_c/"
echo "======================================================"

# Print summary if all succeeded
conda run -n gaze python - <<'EOF'
import json, os, glob

results_dir = "/workspace/EgoGazeVQA/eval/results/egogaze_fold_c"
for f in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
    stem = os.path.basename(f).replace("_metrics.json", "")
    m = json.load(open(f))
    ov = m["overall"]
    print(f"\n{'='*50}")
    print(f"  {stem}")
    print(f"{'='*50}")
    print(f"  Overall : {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"  Latency : {ov.get('avg_inference_time_sec', 'N/A')}s")
    for qt, qm in sorted(m.get("per_qa_type", {}).items()):
        print(f"  {qt:10s}: {qm['accuracy']:.2f}%  (n={qm['n_samples']})")
EOF
