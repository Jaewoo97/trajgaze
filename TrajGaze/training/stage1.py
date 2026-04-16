"""
Step 5 — Stage 1 training loop.

Total loss:
  L = L_frame_BCE + L_NTP + 0.1 · L_traj

  L_frame_BCE  — over ALL T frames (frame selector always supervised)
  L_NTP        — over attended frames only (teacher-forced)
  L_traj       — over attended frames only, masked for null future steps
                 horizon Δ=8, predicted from mean-pooled selected patch embeddings

Components trained: TrajectoryTokenizer, FrameSelector, TwoLevelTrajectoryEncoder,
                    VideoEncoder, ARPatchDecoder, TrajectoryPredictor.

The VLM is not used in Stage 1.

Usage:
  python -m TrajGaze.training.stage1 \
      --adapted-dir   /workspace/datasets/StreamGaze_v2/adapted \
      --interact-dir  /workspace/datasets/StreamGaze_v2/interaction \
      --frames-dir    /workspace/datasets/StreamGaze_v2/frames \
      --output-dir    /workspace/EgoGazeVQA/TrajGaze/checkpoints/stage1 \
      --epochs 150 \
      --lr 3e-4 \
      --batch-size 1 \
      --max-frames 200
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from TrajGaze.data.dataset import TrajGazeDataset, collate_variable_length, TRAJ_HORIZON
from TrajGaze.models.tokenizer import TrajectoryTokenizer
from TrajGaze.models.frame_selector import FrameSelector, frame_selector_loss
from TrajGaze.models.trajectory_encoder import TwoLevelTrajectoryEncoder
from TrajGaze.models.video_encoder import VideoEncoder
from TrajGaze.models.patch_decoder import (
    ARPatchDecoder, TrajectoryPredictor,
    patch_ntp_loss, traj_prediction_loss,
    MAX_PATCHES_PER_FRAME,
)
from TrajGaze.data.interaction import compute_heatmap
from TrajGaze.data.adapter import load_adapted, get_frame_sequence

LAMBDA_TRAJ   = 0.1
ATTEND_THRESH = 0.5
MAX_VIS_FRAMES = 16   # cap visual memory bank (fixes cross-attn OOM + bfloat16 NaN)


# ── Frame loading ──────────────────────────────────────────────────────────────

def load_frame_4ch(frame_path: str, heatmap: np.ndarray) -> torch.Tensor:
    """
    Load a single frame as RGB + heatmap 4-channel tensor.
    Heatmap is a (224, 224) float32 array from compute_heatmap().

    Returns: (4, 224, 224) float32 tensor, pixels in [0, 1]
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(frame_path).convert("RGB").resize((224, 224))
    rgb = TF.to_tensor(img)                                     # (3, 224, 224) in [0, 1]
    heat = torch.from_numpy(heatmap).unsqueeze(0)               # (1, 224, 224)
    return torch.cat([rgb, heat], dim=0)                        # (4, 224, 224)


def load_attended_frames_4ch(
    batch_item: dict,
    attended_idx: list[int],
    frames: list[dict],
) -> Optional[torch.Tensor]:
    """
    Load visual frames (4-channel) for the attended frames only.
    Returns (T_att, 4, 224, 224) or None if frames_dir doesn't exist.
    """
    frames_dir  = batch_item["frames_dir"]
    frame_names = batch_item["frame_names"]
    T = batch_item["T"]

    if not os.path.isdir(frames_dir) or not attended_idx:
        return None

    tensors = []
    for t in attended_idx:
        if t >= len(frame_names):
            continue
        fname = frame_names[t]
        fpath = os.path.join(frames_dir, fname)
        # Compute heatmap for this frame
        heat  = compute_heatmap(frames, t)
        try:
            frame_t = load_frame_4ch(fpath, heat)
        except Exception:
            frame_t = torch.zeros(4, 224, 224)
        tensors.append(frame_t)

    if not tensors:
        return None
    return torch.stack(tensors, dim=0)   # (T_att, 4, 224, 224)


# ── Selector model wrapper ─────────────────────────────────────────────────────

class TrajGazeModel(nn.Module):
    """All trainable components of the Stage 1 system."""

    def __init__(self, cfg: argparse.Namespace):
        super().__init__()
        use_query = getattr(cfg, 'use_query', False)
        self.tokenizer     = TrajectoryTokenizer()
        self.selector      = FrameSelector(threshold=ATTEND_THRESH,
                                           use_query=use_query)
        self.traj_enc      = TwoLevelTrajectoryEncoder()
        self.video_enc     = VideoEncoder()
        self.decoder       = ARPatchDecoder()
        self.traj_pred     = TrajectoryPredictor(horizon=TRAJ_HORIZON)

    @property
    def device(self):
        return next(self.parameters()).device


# ── Training step ──────────────────────────────────────────────────────────────

def training_step(
    model:  TrajGazeModel,
    item:   dict,
    device: torch.device,
    mode:   str = "full",
) -> dict[str, torch.Tensor]:
    """
    Run one clip through the full Stage 1 forward pass.

    item is a single (unbatched) sample from TrajGazeDataset.__getitem__.
    (We process one clip at a time due to variable T.)

    Returns dict with loss scalars for logging.
    """
    # ── Move trajectory tensors to device ─────────────────────────────────────
    def to(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    tokenizer_keys = [
        "gaze_pos", "gaze_speed", "gaze_mask",
        "left_pos", "left_vel", "left_mask",
        "right_pos", "right_vel", "right_mask",
        "d_left", "d_right", "v_rel_left", "v_rel_right",
        "convergence", "lead_lag",
    ]
    tok_in = {k: to(item[k]) for k in tokenizer_keys}

    # ── Step 1: Tokenize ──────────────────────────────────────────────────────
    tok_gaze, tok_left, tok_right, tok_interact = model.tokenizer(
        gaze_pos=tok_in["gaze_pos"],
        gaze_speed=tok_in["gaze_speed"],
        gaze_mask=tok_in["gaze_mask"],
        left_pos=tok_in["left_pos"],
        left_vel=tok_in["left_vel"],
        left_mask=tok_in["left_mask"],
        right_pos=tok_in["right_pos"],
        right_vel=tok_in["right_vel"],
        right_mask=tok_in["right_mask"],
        d_left=tok_in["d_left"],
        d_right=tok_in["d_right"],
        v_rel_left=tok_in["v_rel_left"],
        v_rel_right=tok_in["v_rel_right"],
        convergence=tok_in["convergence"],
        lead_lag=tok_in["lead_lag"],
    )  # each (1, T, 128)

    T = tok_interact.shape[1]

    # ── Step 2: Frame selector ────────────────────────────────────────────────
    query_sim = to(item.get("query_sim", None))  # (1, T, 1) or None
    scores = model.selector(tok_interact, query_sim=query_sim)  # (1, T) ∈ (0, 1)
    attend_labels = to(item["attend"]).float().unsqueeze(0)   # (1, T)
    pad_mask_1d   = torch.ones(1, T, device=device, dtype=torch.bool)

    # Query-aware attend label modulation:
    # Blend original interaction label with query relevance so that
    # frames relevant to the question get boosted attend targets.
    #   attend' = (1-α)·attend + α·norm(query_sim)
    # BCE naturally handles soft [0,1] targets.
    if query_sim is not None:
        q = query_sim.squeeze(-1)       # (1, T)
        q_min = q.min()
        q_range = q.max() - q_min + 1e-8
        q_norm = (q - q_min) / q_range  # normalize to [0, 1]
        alpha = 0.3
        attend_labels = (1 - alpha) * attend_labels + alpha * q_norm

    if mode == "no_frame_selector":
        # All frames attended
        attended_idx = list(range(T))
    else:
        # Hard threshold for downstream; soft scores used for loss
        with torch.no_grad():
            attended_idx = torch.nonzero(
                scores[0] >= ATTEND_THRESH, as_tuple=False
            ).squeeze(-1).tolist()

    L_bce = frame_selector_loss(scores, attend_labels, mask=pad_mask_1d)

    if not attended_idx:
        zero = torch.tensor(0.0, device=device)
        return {"total": L_bce, "L_bce": L_bce, "L_ntp": zero,
                "L_traj": zero, "n_att": 0, "T": T}

    # ── Step 3: Trajectory encoder ────────────────────────────────────────────
    present = tok_in["gaze_mask"] | tok_in["left_mask"] | tok_in["right_mask"]  # (1, T)
    traj_context = model.traj_enc(
        tok_gaze, tok_left, tok_right, tok_interact,
        present, attended_idx,
    )  # (n_att, 4·T_window, 256)

    # ── Step 4a: Load visual frames (4-channel) for attended frames ───────────
    frames = get_frame_sequence(
        {"frames": {k: v for k, v in zip(item["frame_names"], _rebuild_frames(item))}},
        item["frame_names"]
    )
    visual_frames = load_attended_frames_4ch(item, attended_idx, frames)

    if visual_frames is not None:
        visual_frames = visual_frames.to(device).unsqueeze(0)  # (1, T_att, 4, 224, 224)
        T_att_full = visual_frames.shape[1]
        # Cap visual memory to MAX_VIS_FRAMES to avoid quadratic cross-attn memory
        if T_att_full > MAX_VIS_FRAMES:
            step = T_att_full / MAX_VIS_FRAMES
            vis_idx = [int(i * step) for i in range(MAX_VIS_FRAMES)]
            vis_frames_sub = visual_frames[:, vis_idx, :, :, :]   # (1, MAX_VIS_FRAMES, 4, 224, 224)
        else:
            vis_frames_sub = visual_frames
        visual_mem = model.video_enc(vis_frames_sub)              # (1, K*196, 192)
        visual_mem_att = visual_mem.expand(len(attended_idx), -1, -1).contiguous()
    else:
        # Fallback: dummy visual memory
        n_att = len(attended_idx)
        visual_mem_att = torch.zeros(n_att, MAX_VIS_FRAMES * 196, 192, device=device)

    # ── Step 4b: AR patch decoder (teacher-forced) ───────────────────────────
    n_att = len(attended_idx)
    patch_targets_att = to(item["patch_targets"])[attended_idx]  # (n_att, max_K)
    patch_mask_att    = to(item["patch_mask"])[attended_idx]      # (n_att, max_K)
    max_frame_embed   = model.decoder.frame_pos_embed.num_embeddings
    frame_idx_t       = torch.arange(n_att, device=device).clamp(0, max_frame_embed - 1)

    # Interaction mean for gate: mean of tok_interact at each attended frame
    interact_mean = tok_interact[0][attended_idx]  # (n_att, 128)

    dec_out = model.decoder(
        target_patches = patch_targets_att,
        frame_idx      = frame_idx_t,
        visual_mem     = visual_mem_att,
        traj_context   = traj_context,
        interact_mean  = interact_mean,
    )

    if mode == "no_ntp":
        L_ntp = torch.tensor(0.0, device=device)
    else:
        L_ntp = patch_ntp_loss(dec_out["logits"], patch_targets_att, patch_mask_att)

    # ── Step 5: Trajectory prediction (L_traj, attended frames only) ─────────
    # Mean-pool the selected patch embeddings (approximate: use visual_mem mean)
    if mode == "no_traj_loss":
        L_traj = torch.tensor(0.0, device=device)
    else:
        patch_emb_mean = visual_mem_att.mean(dim=1)   # (n_att, 192)
        pred_traj = model.traj_pred(patch_emb_mean)   # (n_att, Δ, 6)

        traj_future_att   = to(item["traj_future"])[attended_idx]    # (n_att, Δ, 6)
        traj_mask_att     = to(item["traj_step_mask"])[attended_idx] # (n_att, Δ)

        if mode == "no_gaze":
            traj_future_att[..., :2] = 0.0
            traj_mask_att = traj_mask_att & to(item["traj_step_mask"])[attended_idx]
        if mode == "no_hand":
            traj_future_att[..., 2:] = 0.0

        L_traj = traj_prediction_loss(pred_traj, traj_future_att, traj_mask_att)

    total = L_bce + L_ntp + LAMBDA_TRAJ * L_traj
    return {
        "total":  total,
        "L_bce":  L_bce,
        "L_ntp":  L_ntp,
        "L_traj": L_traj,
        "n_att":  len(attended_idx),
        "T":      T,
    }


def _rebuild_frames(item: dict) -> list[dict]:
    """Reconstruct minimal frame dicts from item tensors for heatmap computation."""
    # This is a lightweight reconstruction — gaze/hand positions only
    gaze_pos   = item["gaze_pos"][0].numpy()   # (T, 2) scaled to [0,1]
    gaze_mask  = item["gaze_mask"][0].numpy()
    left_pos   = item["left_pos"][0].numpy()
    left_mask  = item["left_mask"][0].numpy()
    right_pos  = item["right_pos"][0].numpy()
    right_mask = item["right_mask"][0].numpy()

    SCALE = 224.0
    T = len(gaze_mask)
    frames = []
    for t in range(T):
        frames.append({
            "gaze":       (gaze_pos[t] * SCALE).tolist() if gaze_mask[t] else None,
            "hand_left":  (left_pos[t] * SCALE).tolist() if left_mask[t] else None,
            "hand_right": (right_pos[t] * SCALE).tolist() if right_mask[t] else None,
        })
    return frames


# ── Distributed helpers ────────────────────────────────────────────────────────

TRAIN_DATASETS = ["egoexolearn", "holoassist"]
VAL_DATASETS   = ["egtea"]
ACCUM_STEPS    = 4   # gradient accumulation — effective batch = ACCUM_STEPS clips


def _ddp_avg_grads(model: nn.Module, world_size: int) -> None:
    """All-reduce gradients across all ranks.

    All ranks MUST call this at the same step (it's a collective op).
    Ranks with no gradient (NaN/OOM skips) contribute zeros.
    """
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def _val_epoch(model: nn.Module, val_loader: DataLoader, device: torch.device,
               mode: str) -> dict:
    """Run one pass over val set, return mean losses (no grad)."""
    model.eval()
    metrics = {"total": 0.0, "L_bce": 0.0, "L_ntp": 0.0, "L_traj": 0.0, "n_att": 0}
    n = 0
    with torch.no_grad():
        for item in val_loader:
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    losses = training_step(model, item, device, mode=mode)
            except Exception:
                continue
            for k in metrics:
                v = losses.get(k, 0)
                metrics[k] += float(v) if isinstance(v, torch.Tensor) else v
            n += 1
    model.train()
    if n == 0:
        return metrics
    return {k: v / n for k, v in metrics.items()}


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-dir",    required=True)
    parser.add_argument("--interact-dir",   required=True)
    parser.add_argument("--frames-dir",     required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--max-frames",     type=int,   default=800,
                        help="Truncate clips longer than N frames")
    parser.add_argument("--checkpoint",     default=None, help="Resume from checkpoint")
    parser.add_argument("--mode",           default="full",
                        choices=["full", "no_frame_selector", "stage1_only",
                                 "no_traj_loss", "no_ntp", "no_gaze", "no_hand",
                                 "baseline=autogaze"],
                        help="Ablation mode")
    parser.add_argument("--use-query",      action="store_true",
                        help="Enable query-aware frame selection (Talk2DINO sim)")
    parser.add_argument("--save-every",     type=int, default=10)
    parser.add_argument("--val-every",      type=int, default=10,
                        help="Run validation every N epochs (0 = skip)")
    parser.add_argument("--train-datasets", nargs="+", default=TRAIN_DATASETS)
    parser.add_argument("--val-datasets",   nargs="+", default=VAL_DATASETS)
    parser.add_argument("--accum-steps",    type=int, default=ACCUM_STEPS)
    parser.add_argument("--num-workers",    type=int, default=8)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Distributed init ───────────────────────────────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    is_dist     = world_size > 1
    is_main     = (not is_dist) or local_rank == 0

    if is_dist:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"Mode: {args.mode} | GPUs: {world_size} | "
              f"max_frames: {args.max_frames} | accum_steps: {args.accum_steps}")
        print(f"Train: {args.train_datasets}   Val: {args.val_datasets}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    common = dict(
        adapted_dir  = args.adapted_dir,
        interact_dir = args.interact_dir,
        frames_dir   = args.frames_dir,
        max_frames   = args.max_frames,
    )
    train_ds = TrajGazeDataset(**common, datasets=args.train_datasets)
    val_ds   = TrajGazeDataset(**common, datasets=args.val_datasets)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    train_loader  = DataLoader(
        train_ds,
        batch_size  = 1,
        shuffle     = (train_sampler is None),
        sampler     = train_sampler,
        num_workers = args.num_workers,
        collate_fn  = lambda x: x[0],
        pin_memory  = False,
        persistent_workers = (args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = lambda x: x[0],
        pin_memory  = False,
        persistent_workers = (args.num_workers > 0),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TrajGazeModel(args).to(device)
    if is_main:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer + scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.output_dir, "train_log.txt")

    # ── Training ───────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if is_dist:
            train_sampler.set_epoch(epoch)

        model.train()
        metrics  = {"total": 0.0, "L_bce": 0.0, "L_ntp": 0.0, "L_traj": 0.0, "n_att": 0}
        n_clips  = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=not is_main)

        for step_i, item in enumerate(pbar):
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    losses = training_step(model, item, device, mode=args.mode)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if is_main:
                    print(f"\n  [OOM] {item.get('stem', '?')} T={item.get('T', '?')}")
                continue
            except Exception as e:
                if is_main:
                    print(f"\n  [skip] {item.get('stem', '?')}: {e}")
                continue

            total_val = losses["total"]
            if not torch.isfinite(total_val):
                if is_main:
                    print(f"\n  [nan] {item.get('stem', '?')} "
                          f"ntp={float(losses.get('L_ntp', float('nan'))):.3f}")
                continue

            # Scale loss for accumulation
            scaled = total_val / args.accum_steps
            scaled.backward()

            is_update_step = ((step_i + 1) % args.accum_steps == 0 or
                              (step_i + 1) == len(train_loader))
            if is_update_step:
                if is_dist:
                    _ddp_avg_grads(model, world_size)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            for k in metrics:
                v = losses.get(k, 0)
                metrics[k] += float(v) if isinstance(v, torch.Tensor) else v
            n_clips += 1

            if is_main and n_clips > 0:
                pbar.set_postfix({
                    "total": f"{metrics['total']/n_clips:.3f}",
                    "bce":   f"{metrics['L_bce']/n_clips:.3f}",
                    "ntp":   f"{metrics['L_ntp']/n_clips:.3f}",
                    "traj":  f"{metrics['L_traj']/n_clips:.3f}",
                    "att%":  f"{100*metrics['n_att']/(n_clips*max(1,args.max_frames)):.0f}",
                })

        scheduler.step()

        # Validation
        val_metrics = {}
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0 and is_main:
            val_metrics = _val_epoch(model, val_loader, device, args.mode)
            print(f"\n  [val] epoch {epoch+1}: "
                  f"total={val_metrics['total']:.3f}  "
                  f"bce={val_metrics['L_bce']:.3f}  "
                  f"ntp={val_metrics['L_ntp']:.3f}  "
                  f"traj={val_metrics['L_traj']:.3f}")

        # Logging
        if is_main and n_clips > 0:
            train_summary = {k: v / n_clips for k, v in metrics.items()}
            log_line = (
                f"epoch={epoch+1}  "
                f"train_total={train_summary['total']:.4f}  "
                f"bce={train_summary['L_bce']:.4f}  "
                f"ntp={train_summary['L_ntp']:.4f}  "
                f"traj={train_summary['L_traj']:.4f}  "
                f"att%={100*train_summary['n_att']/max(1,args.max_frames):.1f}"
            )
            if val_metrics:
                log_line += (f"  val_total={val_metrics['total']:.4f}  "
                             f"val_bce={val_metrics['L_bce']:.4f}")
            with open(log_path, "a") as f:
                f.write(log_line + "\n")

        # Checkpoint (rank 0 only)
        if is_main and ((epoch + 1) % args.save_every == 0 or
                        epoch == args.epochs - 1):
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "mode":      args.mode,
            }, ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    if is_dist:
        dist.destroy_process_group()

    if is_main:
        print("Stage 1 training complete.")


if __name__ == "__main__":
    main()
