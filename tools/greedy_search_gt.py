"""
Patch scoring preprocessing for AutoGaze NTP training.

Selects N_GAZES patches per frame using **independent scoring**:
  - One forward pass per frame evaluates all 265 patches simultaneously
  - Patches are ranked by reconstruction loss (highest = most informative)
  - Top N_GAZES patches are selected in descending loss order

This approximates greedy search (which conditions each step on prior selections)
at 10x lower cost: T=16 forward passes per clip-batch instead of T×N_GAZES=160.
Reconstruction loss correlates strongly with visual salience, making the resulting
patch sets comparable to greedy in quality for gaze prediction training.

Architecture:
  - VideoMAE_AutoGaze weights (facebook/vit-mae-large, videomae.pt)
  - Multi-scale 32+64+112+224 → 265 candidate patches per frame
  - GT stored as ABSOLUTE patch indices (t * 265 + frame_relative)
  - Each GPU writes to gt_{split}_gpu{N}.json; run --merge when all done

Usage:
    for i in 0 1 2 3; do
      PYTHONUNBUFFERED=1 conda run -n gaze --no-capture-output \\
        python -u greedy_search_gt.py --fold fold_a --gpu $i --total_gpus 4 &
    done
    wait
    conda run -n gaze python -u greedy_search_gt.py --fold fold_a --merge
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

from transformers import VivitImageProcessor
from autogaze.tasks.video_mae_reconstruction.modeling_video_mae import ViTMAEForPreTraining
from autogaze.datasets.video_utils import read_video_pyav, sample_frame_indices, process_video_frames
import av

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "/workspace/datasets/StreamGaze/autogaze_data"
RECON_MODEL = "facebook/vit-mae-large"
VIDEOMAE_PT = "/workspace/EgoGazeVQA/AutoGaze/VideoMAE_AutoGaze/videomae.pt"
SCALES      = "32+64+112+224"
CLIP_LEN    = 16
N_GAZES     = 10      # top-N patches to select per frame by reconstruction loss
CLIP_BATCH  = 16      # clips scored simultaneously — ~70 GB on H200 per GPU
SAVE_EVERY  = 64      # flush per-GPU JSON every N clips processed

# Derived
_SCALES_LIST = sorted(int(s) for s in SCALES.split("+"))
_PATCH_SIZE  = 16
_N_PATCHES   = sum((s // _PATCH_SIZE) ** 2 for s in _SCALES_LIST)  # 265


# ── Model loading ──────────────────────────────────────────────────────────────

def load_mae(device):
    from transformers import ViTMAEConfig

    print(f"Loading ViTMAEConfig from {RECON_MODEL}...", flush=True)
    config = ViTMAEConfig.from_pretrained(RECON_MODEL)
    config.scales               = SCALES
    config.scale_embed          = True
    config.max_num_frames       = 256
    config.time_embed           = True
    config.causal               = True
    config.loss_type            = "l1"
    config.loss_weights         = "1"
    config._attn_implementation = "sdpa"

    print(f"Instantiating ViTMAEForPreTraining ({_N_PATCHES} patches/frame)...", flush=True)
    mae = ViTMAEForPreTraining(config)

    print(f"Loading weights from {VIDEOMAE_PT}...", flush=True)
    sd_full = torch.load(VIDEOMAE_PT, map_location="cpu", weights_only=False)
    prefix  = "module.mae."
    sd      = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, _ = mae.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}", flush=True)

    mae.eval().to(device)
    transform = VivitImageProcessor.from_pretrained(RECON_MODEL, size=_SCALES_LIST[-1])
    return mae, transform


# ── Video loading ──────────────────────────────────────────────────────────────

def load_clip(mp4_path, transform, device):
    container = av.open(mp4_path)
    indices   = sample_frame_indices(
        clip_len=CLIP_LEN,
        frame_sample_rate=1,
        seg_len=container.streams.video[0].frames,
        random_sample_frame=False,
    )
    frames = read_video_pyav(container, indices)
    container.close()
    frames = process_video_frames(frames, CLIP_LEN)

    imgs = transform(list(frames)).pixel_values
    if isinstance(imgs[0], list):
        imgs = imgs[0]
    imgs = np.array(imgs)
    if imgs.ndim == 5:
        imgs = imgs[0]
    video = torch.tensor(imgs, dtype=torch.float32).to(device)
    return video.unsqueeze(0)   # (1, T, C, H, W)


# ── Independent patch scoring (one forward pass per frame) ────────────────────

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def score_patches_frame(mae, frames_k, device):
    """
    Score all _N_PATCHES candidates for K clips in ONE forward pass.

    frames_k : list[K] of (1, 1, C, H, W)
    Returns  : (K, _N_PATCHES) float32 loss array — loss per patch per clip.
    """
    K        = len(frames_k)
    B_total  = K * _N_PATCHES   # e.g. 16 × 265 = 4240

    # Build gazing_info on CPU with numpy (avoids 4240 CUDA kernel launches)
    gp_np = np.full((B_total, 2), -1, dtype=np.int64)
    ip_np = np.zeros((B_total, 2), dtype=bool)
    row = 0
    for k in range(K):
        for cand in range(_N_PATCHES):
            gp_np[row, 0] = cand   # single candidate patch; EOS at col 1 stays -1
            ip_np[row, 1] = True   # col 1 is EOS (padded)
            row += 1

    gp = torch.from_numpy(gp_np).to(device, non_blocking=True)
    ip = torch.from_numpy(ip_np).to(device, non_blocking=True)

    # Pixel values: each candidate gets its clip's frame
    parts = [frames_k[k].expand(_N_PATCHES, 1, -1, -1, -1) for k in range(K)]
    frame_batch = torch.cat(parts, dim=0)   # (B_total, 1, C, H, W)

    gazing_info = {
        "gazing_pos":            gp,
        "if_padded_gazing":      ip,
        "num_gazing_each_frame": torch.tensor([2], dtype=torch.long, device=device),
    }

    out = mae(
        pixel_values=frame_batch,
        gazing_info=gazing_info,
        frame_idx_to_reconstruct=torch.tensor([0], device=device),
        interpolate_pos_encoding=True,
    )
    losses_flat = out.loss_each_reconstruction_frame[:, 0].float().cpu().numpy()
    return losses_flat.reshape(K, _N_PATCHES)   # (K, 265)


def process_clips_batch(mae, transform, mp4_paths, device):
    """
    Score patches for a batch of clips and build GT entries.
    Returns {mp4_path: {"gazing_pos": [...], "task_losses": [...]}}
    """
    videos = {}
    for path in mp4_paths:
        try:
            videos[path] = load_clip(path, transform, device)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}", flush=True)

    valid_paths = list(videos.keys())
    K = len(valid_paths)
    if K == 0:
        return {}

    T = videos[valid_paths[0]].shape[1]
    gazing_pos_all  = [[[] for _ in range(T)] for _ in range(K)]
    task_losses_all = [[[] for _ in range(T)] for _ in range(K)]

    for t in range(T):
        frames_k = [videos[p][:, t:t+1, :, :, :] for p in valid_paths]
        # losses shape: (K, _N_PATCHES)
        losses_km = score_patches_frame(mae, frames_k, device)

        for k in range(K):
            # Select top N_GAZES patches by highest reconstruction loss (most informative)
            top_rel = np.argsort(losses_km[k])[-N_GAZES:][::-1]   # descending
            gazing_pos_all[k][t]  = [int(t * _N_PATCHES + r) for r in top_rel]
            task_losses_all[k][t] = [float(losses_km[k][r]) for r in top_rel]

    return {
        p: {"gazing_pos": gazing_pos_all[k], "task_losses": task_losses_all[k]}
        for k, p in enumerate(valid_paths)
    }


# ── Key matching ───────────────────────────────────────────────────────────────

def get_gt_key(mp4_path):
    parts = mp4_path.replace("\\", "/").split("/")
    return "/".join(parts[-3:])


# ── Merge mode ─────────────────────────────────────────────────────────────────

def merge_gpu_files(fold_dir):
    import glob
    for split in ("train", "val"):
        pattern  = os.path.join(fold_dir, f"gt_{split}_gpu*.json")
        gpu_files = sorted(glob.glob(pattern))
        if not gpu_files:
            print(f"[{split}] No per-GPU files found, skipping.")
            continue
        merged = {}
        for gf in gpu_files:
            with open(gf) as f:
                merged.update(json.load(f))
        out_json = os.path.join(fold_dir, f"gt_{split}.json")
        with open(out_json, "w") as f:
            json.dump(merged, f)
        print(f"[{split}] Merged {len(gpu_files)} GPU files → {len(merged)} entries → {out_json}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--total_gpus",  type=int, default=1)
    parser.add_argument("--cuda_device", type=int, default=None,
                        help="CUDA device index to use (defaults to --gpu)")
    parser.add_argument("--fold",        type=str, default="fold_a")
    parser.add_argument("--merge",       action="store_true")
    args = parser.parse_args()

    fold_dir = os.path.join(DATA_DIR, args.fold)

    if args.merge:
        merge_gpu_files(fold_dir)
        return

    cuda_device = args.cuda_device if args.cuda_device is not None else args.gpu
    device = torch.device(f"cuda:{cuda_device}")
    print(f"=== Patch scoring GT: {args.fold}, GPU {args.gpu}/{args.total_gpus} ===", flush=True)
    print(f"    N_GAZES={N_GAZES}, CLIP_BATCH={CLIP_BATCH}, SCALES={SCALES} ({_N_PATCHES} patches/frame)", flush=True)
    print(f"    Method: independent scoring (1 forward pass per frame, top-{N_GAZES} by loss)", flush=True)
    mae, transform = load_mae(device)
    print("Model ready.\n", flush=True)

    for split in ("train", "val"):
        split_dir = os.path.join(fold_dir, split)
        out_json  = os.path.join(fold_dir, f"gt_{split}_gpu{args.gpu}.json")

        gt_dict = json.load(open(out_json)) if os.path.exists(out_json) else {}

        if not os.path.isdir(split_dir):
            print(f"[{split}] Directory not found, skipping.", flush=True)
            continue

        mp4_files = sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".mp4")
        )
        if not mp4_files:
            print(f"[{split}] No clips found, skipping.", flush=True)
            continue

        mp4_files = [f for i, f in enumerate(mp4_files) if i % args.total_gpus == args.gpu]
        todo      = [f for f in mp4_files if get_gt_key(f) not in gt_dict]
        n_done    = len(mp4_files) - len(todo)
        print(f"[{split}] {len(mp4_files)} clips on GPU {args.gpu} ({n_done} already done, {len(todo)} to go)", flush=True)

        done = 0
        for batch_start in range(0, len(todo), CLIP_BATCH):
            batch   = todo[batch_start: batch_start + CLIP_BATCH]
            results = process_clips_batch(mae, transform, batch, device)
            for path, entry in results.items():
                gt_dict[get_gt_key(path)] = entry
            done += len(batch)

            if done // SAVE_EVERY > (done - len(batch)) // SAVE_EVERY or done >= len(todo):
                with open(out_json, "w") as f:
                    json.dump(gt_dict, f)
                print(f"  [{split}] {done}/{len(todo)} clips done ({len(gt_dict)} entries in gpu{args.gpu} file)", flush=True)

        with open(out_json, "w") as f:
            json.dump(gt_dict, f)
        print(f"[{split}] Done. {len(gt_dict)} entries → {out_json}\n", flush=True)


if __name__ == "__main__":
    main()
