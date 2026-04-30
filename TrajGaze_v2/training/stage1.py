"""
TrajGaze_v2 Stage 1 Training.

Trajectory + interaction-score prediction from past → future.
DDP on N GPUs (default: 4).

Usage:
    # Original: ego4d + egoexo, full model
    torchrun --nproc_per_node=4 -m TrajGaze_v2.training.stage1 \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1 \
        --epochs 100 --lr 3e-4 --batch-size 4

    # StreamGaze domain (egoexolearn + holoassist), full model
    CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29805 \
        -m TrajGaze_v2.training.stage1 \
        --dataset streamgaze \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_streamgaze \
        --epochs 100 --lr 3e-4 --batch-size 4

    # Gaze-only ablation (ego4d + egoexo, gaze-only model)
    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29803 \
        -m TrajGaze_v2.training.stage1 \
        --dataset egogazevqa --model gaze-only \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_gaze_only \
        --epochs 100 --lr 3e-4 --batch-size 4

Flags:
    --dataset   egogazevqa   ego4d + egoexo clips (default)
                streamgaze   egoexolearn + holoassist from StreamGaze_v2
    --model     full         TrajGazeV2 with all 4 tokens (default)
                gaze-only    TrajGazeV2GazeOnly with 1 gaze token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",   default="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1")
    p.add_argument("--dataset",      default="egogazevqa",
                   choices=["egogazevqa", "streamgaze"],
                   help="egogazevqa = ego4d+egoexo | streamgaze = egoexolearn+holoassist")
    p.add_argument("--model",        default="full",
                   choices=["full", "gaze-only", "hand-only"],
                   help="full = TrajGazeV2 (4 tokens) | gaze-only = TrajGazeV2GazeOnly (1 token) | hand-only = TrajGazeV2HandOnly (3 tokens)")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-frames",     type=int,   default=32)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--save-every",   type=int,   default=10)
    p.add_argument("--resume",       type=str,   default=None)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def build_dataset(args):
    if args.dataset == "streamgaze":
        if args.model == "gaze-only":
            from TrajGaze_v2.data.dataset_gaze_only import StreamGazeStage1DatasetGazeOnly, collate_stage1
            return StreamGazeStage1DatasetGazeOnly(n_frames=args.n_frames), collate_stage1
        elif args.model == "hand-only":
            from TrajGaze_v2.data.dataset_hand_only import StreamGazeStage1DatasetHandOnly
            from TrajGaze_v2.data.dataset import collate_stage1
            return StreamGazeStage1DatasetHandOnly(n_frames=args.n_frames), collate_stage1
        else:
            from TrajGaze_v2.data.dataset_streamgaze_stage1 import StreamGazeStage1Dataset, collate_stage1
            return StreamGazeStage1Dataset(n_frames=args.n_frames), collate_stage1
    else:  # egogazevqa
        if args.model == "gaze-only":
            from TrajGaze_v2.data.dataset_gaze_only import TrajGazeV2Stage1DatasetGazeOnly, collate_stage1
            return TrajGazeV2Stage1DatasetGazeOnly(datasets=["ego4d", "egoexo"], n_frames=args.n_frames), collate_stage1
        else:
            from TrajGaze_v2.data.dataset import TrajGazeV2Stage1Dataset, collate_stage1
            return TrajGazeV2Stage1Dataset(datasets=["ego4d", "egoexo"], n_frames=args.n_frames), collate_stage1


def build_model(args, device):
    if args.model == "gaze-only":
        from TrajGaze_v2.models.model_gaze_only import TrajGazeV2GazeOnly
        return TrajGazeV2GazeOnly().to(device)
    elif args.model == "hand-only":
        from TrajGaze_v2.models.model_hand_only import TrajGazeV2HandOnly
        return TrajGazeV2HandOnly().to(device)
    else:
        from TrajGaze_v2.models.model import TrajGazeV2
        return TrajGazeV2().to(device)


def main():
    args     = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main  = rank == 0
    device   = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Stage 1] dataset={args.dataset}  model={args.model}")
        print(f"[Stage 1] output: {args.output_dir}")
        print(f"[Stage 1] world_size={world_size}  batch/GPU={args.batch_size}")

    dataset, collate_fn = build_dataset(args)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        sampler     = sampler,
        collate_fn  = collate_fn,
        num_workers = args.workers,
        pin_memory  = True,
        drop_last   = True,
    )

    model = build_model(args, device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            print(f"[Stage 1] Resumed from epoch {start_epoch}")

    log_path  = os.path.join(args.output_dir, "train_log.jsonl")
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        total_loss = total_traj = total_score = total_attn = 0.0
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
            batch["I_scores_future"] = batch["I_scores_future"].to(device)
            batch["T_past"]          = batch["T_past"].to(device)
            batch["T_future"]        = batch["T_future"].to(device)

            optimizer.zero_grad()
            loss_dict = model.module.stage1_forward(batch)
            loss      = loss_dict["loss"]

            if not torch.isfinite(loss):
                if is_main:
                    print(f"[WARNING] Non-finite loss at step {step}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_traj  += loss_dict["loss_traj"].item()
            total_score += loss_dict["loss_score"].item()
            total_attn  += loss_dict.get("loss_attn", torch.tensor(0.0)).item()
            n_batches   += 1

            if is_main and (step + 1) % args.log_every == 0:
                avg_loss  = total_loss  / n_batches
                avg_traj  = total_traj  / n_batches
                avg_score = total_score / n_batches
                avg_attn  = total_attn  / n_batches
                elapsed   = time.time() - t_start
                print(f"Epoch {epoch+1:3d} | step {step+1:4d} | "
                      f"loss={avg_loss:.4f} traj={avg_traj:.4f} "
                      f"score={avg_score:.4f} attn={avg_attn:.4f} | "
                      f"t={elapsed:.1f}s")

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
                    "epoch":      epoch + 1,
                    "loss":       epoch_loss,
                    "loss_traj":  total_traj  / max(1, n_batches),
                    "loss_score": total_score / max(1, n_batches),
                    "loss_attn":  total_attn  / max(1, n_batches),
                    "lr":         scheduler.get_last_lr()[0],
                }) + "\n")

            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth")
                torch.save({
                    "epoch":     epoch,
                    "model":     model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss":      epoch_loss,
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
        print(f"\n[Stage 1] Done. Best loss: {best_loss:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
