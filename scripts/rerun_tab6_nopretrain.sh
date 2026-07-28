#!/usr/bin/env bash
# Re-run Table 6's "No pretrain" row, which died 33 s in on 2026-07-28 00:22.
#
# Cause: the node reprovision wiped $HOME/.cache, so both DDP ranks re-downloaded
# DINOv2 at once and one lost the torch.hub extract race
# ("OSError: [Errno 39] Directory not empty: 'dinov2'"). The cache is warm again and
# env.sh now pins TORCH_HOME to lustre, so this is a one-time repair, not a rerun policy.
#
# Waits for run_ablation_tab6_tab7.sh to finish its remaining rows — this machine has
# 2 GPUs and rows cannot be co-resident. Protocol is byte-identical to run_row() there.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh

unset VLM_GAZE_OVERLAY
export GAZE_OVERLAY=1

while pgrep -f "run_ablation_tab6_tab7.sh" >/dev/null; do sleep 60; done

waited=0
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . ; do
    sleep 30; waited=$((waited + 30))
    [ $waited -ge 600 ] && { echo "[warn] GPU still busy after ${waited}s, continuing" >&2; break; }
done

name=tab6_nopretrain_overlay
out=$REPO/TrajGazeMerge/checkpoints/$name
log=$REPO/$name.log
mkdir -p "$out"

echo "=== $name (retry) start $(date -Is) ===" >>"$log"
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29915 \
    -m TrajGazeMerge.training.train_visionzip_complement_lora \
    --traj-pool-mode learned --complement-mode topk \
    --content-ratio 0.07 --traj-ratio 0.03 \
    --source sg --no-hdepic \
    --epochs 1 --lr 1e-4 --grad-accum 4 \
    --output-dir "$out" \
    --stage1-ckpt "$STAGE1_CKPT" --random-encoder >>"$log" 2>&1
rc=$?
echo "=== $name exit=$rc $(date -Is) ===" >>"$log"
# 1 epoch => epoch_01.pth and best.pth are byte-identical 16.6GB copies
[ $rc -eq 0 ] && rm -f "$out/epoch_01.pth"

echo "=== tab6_nopretrain retry done rc=$rc $(date -Is) ==="
