#!/usr/bin/env bash
# Probe 3 — MLP target LoRA expansion via batched training.
# GPU 0: Probe 3 (q,k,v,o + gate,up,down) — ~30M trainable vs original 10M
# GPU 1: control run (default attention-only LoRA) for direct A/B comparison
#
# Both use FULL recipe (TAS + ATR + CGM) + new TAS Stage 1 ckpt.
# Both use --micro-batch 4 for ~2-3× speedup.
#
# Probe 4 (ViT LoRA) is deferred — requires extending load_qwen_lora to add
# a second LoraConfig over visual module (q,k,v,proj). Separate sprint.

set -uo pipefail

REPO=/workspace/trajgaze
S1_TAS=$REPO/TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth

OUT_P3=$REPO/TrajGazeMerge/checkpoints/E1_combined_probe3_mlp
OUT_CTRL=$REPO/TrajGazeMerge/checkpoints/E1_combined_probe3_control

LAUNCHER=$REPO/TrajGazeMerge/checkpoints/E1_combined_probe3.log
mkdir -p "$OUT_P3" "$OUT_CTRL" "$(dirname "$LAUNCHER")"

echo "[$(date -u)] Probe 3 chain start" > "$LAUNCHER"

COMMON_ARGS=(
    --model-type   full
    --stage1-ckpt  "$S1_TAS"
    --epochs       3
    --merge-ratio  0.9
    --micro-batch  4
    --grad-accum   4
    --use-egovqa
    --eval-egovqa-egtea
    --use-atr        --atr-lambda 0.5
    --cgm-aug        --cgm-lambda 0.3 --cgm-prob 0.3
    --cgm-warmup-steps 600 --cgm-radius 0.2 --cgm-margin 0.5
    --dataloader-num-workers 8
    --eval-every 400
)

# ── GPU 0 — MLP target LoRA (Probe 3) ───────────────────────────────────────
echo "[$(date -u)] Launching Probe 3 MLP-target LoRA (GPU 0)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
    LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_batched \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUT_P3" \
    < /dev/null > "$OUT_P3/stdout.log" 2>&1 &
P3_PID=$!
echo "[$(date -u)] Probe 3 PID=$P3_PID" >> "$LAUNCHER"

sleep 60

# ── GPU 1 — control (same recipe, default attention-only LoRA) ──────────────
# Apples-to-apples baseline for Probe 3: same micro-batch path, same FULL recipe,
# original LoRA target list. Gives clean A/B for "does MLP LoRA help?".
echo "[$(date -u)] Launching control (default LoRA, GPU 1)" >> "$LAUNCHER"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$REPO \
    setsid nohup /opt/conda/envs/gaze/bin/python -m \
    TrajGazeMerge.training.train_merge_lora_batched \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OUT_CTRL" \
    < /dev/null > "$OUT_CTRL/stdout.log" 2>&1 &
C_PID=$!
echo "[$(date -u)] Control PID=$C_PID" >> "$LAUNCHER"

wait $P3_PID 2>/dev/null
echo "[$(date -u)] Probe 3 finished" >> "$LAUNCHER"
wait $C_PID 2>/dev/null
echo "[$(date -u)] Control finished" >> "$LAUNCHER"
echo "[$(date -u)] Probe 3 chain complete." >> "$LAUNCHER"
