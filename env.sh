# Environment for the TrajGazeMerge KD pipeline on this machine.
# Usage:  cd /NHNHOME/VILAB/vilab_yj/trajgaze && source env.sh

# /NHNHOME is node-local and is wiped on reprovision, taking this alias with it —
# that killed a Table 6 ablation row mid-epoch on 2026-07-27. Everything below, plus
# the checkpoint symlinks under */checkpoints, resolves through it, so restore it first.
[ -e /NHNHOME/VILAB ] || ln -sfn /NHNHOME/WORKSPACE/26msit001_A /NHNHOME/VILAB

export REPO=/NHNHOME/VILAB/vilab_yj/trajgaze
export DATA=/NHNHOME/VILAB/vilab_yj/datasets/trajgazemerge

# Data roots (consumed by TrajGazeMerge/data/*.py via os.environ.get)
export SG_ROOT=$DATA/StreamGaze_v2
export EG_ROOT=$DATA/EgoGazeVQA
export HD_ROOT=$DATA/HD-EPIC

# Gaze-overlay frames — the handoff uses GAZE_OVERLAY=1 for every run.
export GAZE_OVERLAY=1

# venv layered on the `vgent` conda env (torch 2.11+cu128, flash_attn 2.8.3 on B200)
export PATH=/NHNHOME/VILAB/vilab_yj/envs/trajgaze/bin:$PATH

# The venv has no torchrun shim, and bare `torchrun` resolves to /usr/local/bin
# (system python, no peft). Always launch DDP through the venv interpreter.
export TORCHRUN="python -m torch.distributed.run"

# Qwen2.5-VL-7B backbone is already cached here
export HF_HOME=/NHNHOME/VILAB/vilab_yj/.cache/huggingface

# DINOv2 (visual_encoder_temporal.py calls torch.hub.load) — keep it off node-local
# $HOME, which the reprovision wipes. A cold cache makes the two DDP ranks race in
# torch.hub._get_cache_or_reload and one dies on "Directory not empty: 'dinov2'".
export TORCH_HOME=/NHNHOME/VILAB/vilab_yj/.cache/torch

# Checkpoints (symlinks into $DATA/aaai)
export STAGE1_CKPT=$REPO/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
export M1_JOINT=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay/best.pth
export M1_SGONLY=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_SGonly_overlay/best.pth
export M1_EGONLY=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_EGonly_overlay/best.pth

# 25%-budget teacher (content 15% ∪ traj 10%), SG-only, 1 epoch — scripts/run_vitkd25_sg_raw.sh
export M1_SGONLY_B25=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_SGonly_overlay_b25/best.pth
