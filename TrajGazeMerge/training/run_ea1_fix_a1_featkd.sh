#!/usr/bin/env bash
# EA1 FIX encoder + A1 feat-KD stage-2 training.
#
# Encoder: /workspace/trajgaze/TrajGaze_v2/checkpoints/EA1_parallel_branch/best.pth
#   (the validated EA1 stage-1 ckpt, which when loaded with the silent-drop fix
#    produced 68.44% best on stage-2 — beat baseline 67.49% by +0.95pt)
# Recipe: A1 = baseline (alpha=0.5 logit KL) + feat-KD on last 2 LLM hidden states.
#   memory says A1 added +2.47pt over no-KD baseline → expect ~70%+ if additive.
#
# GPU 1 single-GPU torchrun (KD trainer uses DDP).
# Argparse pitfall: use `--kd-feat-layers=-1,-2` (with `=`) to avoid -1 being parsed as flag.

set -euo pipefail

REPO=/workspace/trajgaze
OUT=$REPO/TrajGazeMerge/checkpoints/EA1fix_a1_featkd_keep10_bs4
mkdir -p "$OUT"

LAUNCHER=$OUT/launcher.log
echo "[$(date)] EA1 FIX + A1 feat-KD launched (GPU 1, detached)" > "$LAUNCHER"

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$REPO \
/opt/conda/envs/gaze/bin/torchrun --nproc_per_node=1 --master_port=29842 \
    -m TrajGazeMerge.training.train_merge_lora_temporal \
    --stage1-ckpt   "$REPO/TrajGaze_v2/checkpoints/EA1_parallel_branch/best.pth" \
    --teacher-ckpt  /workspace/trajgaze_msk/king_ms.pth \
    --output-dir    "$OUT" \
    --epochs        3 \
    --alpha         0.5 \
    --kd-feat-layers=-1,-2 \
    --kd-feat-weight 0.3 \
    --merge-ratio   0.9 \
    --grad-accum    4 \
    --eval-every    400 \
    > "$OUT/stdout.log" 2>&1

EXIT=$?
echo "[$(date)] Training exited with code $EXIT" >> "$LAUNCHER"
echo "[$(date)] All done." >> "$LAUNCHER"
