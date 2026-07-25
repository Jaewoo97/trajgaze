#!/usr/bin/env bash
# ① anticipatory: wait for training to finish → per-item dump → McNemar vs attn(control) & gaze.
# Detached (setsid). Result → checkpoints/dumps/antic_mcnemar_result.txt
set -u
cd /workspace/trajgaze_st
PY=/opt/conda/envs/trajgaze/bin/python
R=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints
D=$R/dumps
HB=$R/foveal_launch.log
echo "[antic-mcnemar] $(date) armed, waiting for anticipatory training to finish" >> $HB

for i in $(seq 1 600); do        # up to 600 min
  pgrep -f "roi-arm anticipatory" >/dev/null 2>&1 || { echo "[antic-mcnemar] $(date) training done after ${i}min" >> $HB; break; }
  sleep 60
done

# dump anticipatory best.pth on a free GPU (2)
CUDA_VISIBLE_DEVICES=2 GAZE_OVERLAY=1 $PY -m TrajGazeMerge.eval.eval_dump_foveal \
  --gpu 0 --roi-arm anticipatory --ckpt $R/foveal_anticipatory/best.pth \
  --roi-crop_frac 0.35 --roi-margin_frac 0.08 --roi-antic_horizon 12 \
  --dump $D/foveal_anticipatory.jsonl > $D/dump_anticipatory.log 2>&1

{
  echo "=== anticipatory dump $(date) ==="
  grep -E "Overall:" $D/dump_anticipatory.log
  echo ""
  echo "################## McNemar: attn(control) vs anticipatory(①) ##################"
  $PY -m TrajGazeMerge.eval.mcnemar --a $D/foveal_attn.jsonl --label-a attn \
     --b $D/foveal_anticipatory.jsonl --label-b antic
  echo ""
  echo "################## McNemar: gaze(④) vs anticipatory(①) ##################"
  $PY -m TrajGazeMerge.eval.mcnemar --a $D/foveal_gaze.jsonl --label-a gaze \
     --b $D/foveal_anticipatory.jsonl --label-b antic
} > $D/antic_mcnemar_result.txt 2>&1
echo "[antic-mcnemar] $(date) DONE -> $D/antic_mcnemar_result.txt" >> $HB
