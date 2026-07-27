# Environment for the TrajGazeMerge KD pipeline on this machine.
# Usage:  cd /NHNHOME/VILAB/vilab_yj/trajgaze && source env.sh

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

# Checkpoints (symlinks into $DATA/aaai)
export STAGE1_CKPT=$REPO/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
export M1_JOINT=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay/best.pth
export M1_SGONLY=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_SGonly_overlay/best.pth
export M1_EGONLY=$REPO/TrajGazeMerge/checkpoints/visionzip_complement_learned_EGonly_overlay/best.pth
