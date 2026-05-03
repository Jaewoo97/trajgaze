"""
TrajGazeV2Temporal Stage 1 Training.

Trains TrajGazeV2Temporal on StreamGaze_v2 clips (egoexolearn + holoassist).
The model outputs per-frame (T, 196) patch score maps instead of a single
clip-level (196,) map.

Training setup:
  - Random 40–60% past/future split per iteration
  - Encoder observes T_past frames, supervised on GT past-frame I_scores
  - Decoder predicts T_future trajectory positions + T_future per-frame I_scores
  - Loss: L_traj + L_score_future + L_score_past
  - n_frames=128 to match Qwen2.5-VL temporal resolution

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29820 \
        -m TrajGaze_v2.training.stage1_temporal \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal \
        --epochs 100 --lr 3e-4 --batch-size 2

    # Resume
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29821 \
        -m TrajGaze_v2.training.stage1_temporal \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal \
        --epochs 100 --resume /workspace/.../stage1_temporal/epoch_0050.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal
from TrajGaze_v2.data.dataset_temporal import (
    StreamGazeStage1DatasetTemporal, collate_stage1_temporal,
)
from TrajGaze_v2.training.loss_schedule import (
    LossWeights, curriculum_weights, total_from_weighted,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",     default="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal")
    p.add_argument("--n-frames",       type=int,   default=128,
                   help="Number of trajectory frames per clip (should match VLM input)")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--batch-size",     type=int,   default=2,
                   help="Per-GPU batch size (T=128 is memory-heavy; use 1-2)")
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--workers",        type=int,   default=2)
    p.add_argument("--log-every",      type=int,   default=10)
    p.add_argument("--save-every",     type=int,   default=10)
    p.add_argument("--n-vis-keyframes",type=int,   default=16,
                   help="DINOv2 keyframes per clip for visual encoder")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--init-from",      type=str,   default=None,
                   help="Path to a stage-1 checkpoint to load model weights from "
                        "as a warm start. Unlike --resume, this only loads weights "
                        "(strict=False, missing keys keep fresh init), and starts "
                        "epoch=0 / global_step=0 with a fresh optimizer/scheduler.")
    p.add_argument("--use-curriculum", action="store_true",
                   help="Apply step-aware loss curriculum (ramps in score_past, "
                        "score_future, score_traj). Default off = legacy "
                        "uniform-weight 4-loss sum.")
    p.add_argument("--total-steps",    type=int, default=None,
                   help="Total optimiser steps used for curriculum progress. "
                        "Defaults to epochs * len(loader).")
    p.add_argument("--gate-init",      type=float, default=0.0,
                   help="Initial value of encoder.inter_frame_gate (pre-tanh). "
                        "0.0 -> tanh=0 (InterFrameTransformer fully bypassed). "
                        "2.5 -> tanh~0.987 (full pass-through, equivalent to "
                        "legacy joint architecture). Default 0.0.")
    p.add_argument("--freeze-gate",    action="store_true",
                   help="Freeze encoder.inter_frame_gate at its init value. "
                        "Use with --gate-init 2.5 to reproduce the legacy "
                        "joint architecture for E0 baseline.")
    p.add_argument("--drop-loss-score-traj", action="store_true",
                   help="Permanently set l_score_traj weight to 0 (the noisiest "
                        "of the 4 losses, chains decoder x encoder attention). "
                        "Yields 3-loss training: l_traj + l_score_past + l_score_fut.")
    p.add_argument("--no-visual", action="store_true",
                   help="Skip DINOv2 visual encoding entirely. The encoder falls "
                        "back to its trajectory-only score path (no visual "
                        "cross-attention). Approximates the doc's no_spatial "
                        "ablation when combined with --freeze-gate --gate-init 2.5.")
    p.add_argument("--use-frame-score-branch", action="store_true",
                   help="Enable parallel frame-level score head sourced from the "
                        "InterFrameTransformer output. Output broadcast-multiplied "
                        "into per-patch scores. (a)+(b1) architecture variant.")
    p.add_argument("--use-post-fusion-iframe", action="store_true",
                   help="Add a second InterFrameTransformer AFTER per-frame visual "
                        "fusion, gated residual, applied to enriched_context. "
                        "(b2) architecture variant.")
    p.add_argument("--use-patch-temporal-branch", action="store_true",
                   help="E1: patch-level temporal modulation. 196 learned patch queries "
                        "cross-attend to x_iframe (B*T, 4, D), producing (B, T, 196) "
                        "modulation map elementwise-multiplied into per_frame_scores. "
                        "Mutually exclusive with --use-frame-score-branch.")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def main():
    args                          = parse_args()
    rank, local_rank, world_size  = setup_ddp()
    is_main                       = rank == 0
    device                        = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[stage1_temporal] n_frames={args.n_frames}  world_size={world_size}  "
              f"batch/GPU={args.batch_size}")
        print(f"[stage1_temporal] output: {args.output_dir}")

    dataset = StreamGazeStage1DatasetTemporal(n_frames=args.n_frames)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        sampler     = sampler,
        collate_fn  = collate_stage1_temporal,
        num_workers = args.workers,
        pin_memory  = True,
        drop_last   = True,
    )

    model = TrajGazeV2Temporal(
        n_vis_keyframes=args.n_vis_keyframes,
        use_frame_score_branch=args.use_frame_score_branch,
        use_post_fusion_iframe=args.use_post_fusion_iframe,
        use_patch_temporal_branch=args.use_patch_temporal_branch,
    ).to(device)

    # Apply gate init / freeze before DDP wrap so DDP picks up the right param state.
    with torch.no_grad():
        model.encoder.inter_frame_gate.fill_(args.gate_init)
    if args.freeze_gate:
        model.encoder.inter_frame_gate.requires_grad_(False)
    if is_main:
        gate_after = torch.tanh(model.encoder.inter_frame_gate.detach()).item()
        print(f"[stage1_temporal] gate_init={args.gate_init} -> tanh={gate_after:+.3f} "
              f"frozen={args.freeze_gate}")

    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        # Allow extending training beyond the original cosine T_max:
        # if --epochs is larger than the loaded scheduler's T_max, override
        # so cosine continues to the new endpoint instead of staying at eta_min.
        if args.epochs > scheduler.T_max:
            if is_main:
                print(f"[stage1_temporal] Extending cosine T_max: "
                      f"{scheduler.T_max} -> {args.epochs}")
            scheduler.T_max = args.epochs
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", start_epoch * len(loader))
        if is_main:
            print(f"[stage1_temporal] Resumed from epoch {start_epoch} "
                  f"(global_step={global_step})")
    elif args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd   = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        result = model.module.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[stage1_temporal] Warm start from {args.init_from}")
            if result.missing_keys:
                print(f"  missing keys (kept fresh init): {result.missing_keys}")
            if result.unexpected_keys:
                print(f"  unexpected keys (ignored): {result.unexpected_keys}")

    total_steps = args.total_steps if args.total_steps else args.epochs * len(loader)
    if is_main and args.use_curriculum:
        print(f"[stage1_temporal] Curriculum ON, total_steps={total_steps}")

    log_path  = os.path.join(args.output_dir, "train_log.jsonl")
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        total_loss = total_traj = total_sf = total_sp = total_st = 0.0
        n_batches  = 0
        t_start    = time.time()

        for step, batch in enumerate(loader):
            if batch is None:
                continue

            def to_dev(d):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in d.items()}

            batch["past"]            = to_dev(batch["past"])
            batch["future"]          = to_dev(batch["future"])
            batch["I_scores_past"]   = batch["I_scores_past"].to(device)
            batch["I_scores_future"] = batch["I_scores_future"].to(device)
            batch["T_past"]          = batch["T_past"].to(device)
            batch["T_future"]        = batch["T_future"].to(device)

            if args.no_visual:
                batch["frame_paths"] = None

            optimizer.zero_grad()
            loss_dict = model.module.stage1_forward(batch)

            if args.use_curriculum:
                weights = curriculum_weights(global_step, total_steps)
                if args.drop_loss_score_traj:
                    weights = LossWeights(
                        traj         = weights.traj,
                        score_past   = weights.score_past,
                        score_future = weights.score_future,
                        score_traj   = 0.0,
                    )
                loss = total_from_weighted(loss_dict, weights)
            elif args.drop_loss_score_traj:
                loss = (loss_dict["loss_traj"]
                        + loss_dict["loss_score_past"]
                        + loss_dict["loss_score_fut"])
            else:
                loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                if is_main:
                    print(f"[WARNING] Non-finite loss at step {step}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_traj += loss_dict["loss_traj"].item()
            total_sf   += loss_dict["loss_score_fut"].item()
            total_sp   += loss_dict["loss_score_past"].item()
            total_st   += loss_dict["loss_score_traj"].item()
            n_batches  += 1
            global_step += 1

            if is_main and (step + 1) % args.log_every == 0:
                avg     = lambda x: x / n_batches
                elapsed = time.time() - t_start
                gate_val = torch.tanh(
                    model.module.encoder.inter_frame_gate.detach()
                ).item()
                extra = ""
                if args.use_curriculum:
                    w = curriculum_weights(global_step, total_steps)
                    st = 0.0 if args.drop_loss_score_traj else w.score_traj
                    extra = (f" | w(p,f,t)=({w.score_past:.2f},"
                            f"{w.score_future:.2f},{st:.2f})")
                elif args.drop_loss_score_traj:
                    extra = " | (3-loss: l_score_traj dropped)"
                print(f"Epoch {epoch+1:3d} | step {step+1:4d} (g={global_step}) | "
                      f"loss={avg(total_loss):.4f} traj={avg(total_traj):.4f} "
                      f"score_fut={avg(total_sf):.4f} score_past={avg(total_sp):.4f} "
                      f"score_traj={avg(total_st):.4f} | "
                      f"gate={gate_val:+.3f}{extra} | t={elapsed:.1f}s")

        scheduler.step()

        loss_tensor = torch.tensor(total_loss / max(1, n_batches), device=device)
        reduce_mean(loss_tensor, world_size)
        epoch_loss = loss_tensor.item()

        if is_main:
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={epoch_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"time={elapsed:.1f}s ===\n")

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch":            epoch + 1,
                    "loss":             epoch_loss,
                    "loss_traj":        total_traj / max(1, n_batches),
                    "loss_score_fut":   total_sf   / max(1, n_batches),
                    "loss_score_past":  total_sp   / max(1, n_batches),
                    "loss_score_traj":  total_st   / max(1, n_batches),
                    "lr":               scheduler.get_last_lr()[0],
                }) + "\n")

            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth")
                torch.save({
                    "epoch":       epoch,
                    "global_step": global_step,
                    "model":       model.module.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "loss":        epoch_loss,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    {"model": model.module.state_dict(), "loss": best_loss},
                    os.path.join(args.output_dir, "best.pth"),
                )

    if is_main:
        torch.save({"model": model.module.state_dict()},
                   os.path.join(args.output_dir, "final.pth"))
        print(f"\n[stage1_temporal] Done. Best loss: {best_loss:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
