"""
TrajGazeV2Temporal Stage 1 — single-GPU ablation training.

Supports gaze-only and hand-only model variants via --model-type.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m TrajGaze_v2.training.stage1_temporal_ablation \
        --model-type gaze_only \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal_gaze_only \
        --epochs 100 --lr 3e-4 --batch-size 4

    CUDA_VISIBLE_DEVICES=3 python -m TrajGaze_v2.training.stage1_temporal_ablation \
        --model-type hand_only \
        --output-dir /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal_hand_only \
        --epochs 100 --lr 3e-4 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGaze_v2.data.dataset_temporal import StreamGazeStage1DatasetTemporal, collate_stage1_temporal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",      choices=["gaze_only", "hand_only"], required=True)
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--n-frames",        type=int,   default=128)
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--workers",         type=int,   default=2)
    p.add_argument("--log-every",       type=int,   default=10)
    p.add_argument("--save-every",      type=int,   default=10)
    p.add_argument("--n-vis-keyframes", type=int,   default=16)
    p.add_argument("--resume",          type=str,   default=None)
    return p.parse_args()


def build_model(model_type: str, n_vis_keyframes: int):
    if model_type == "gaze_only":
        from TrajGaze_v2.models.model_temporal_gaze_only import TrajGazeV2TemporalGazeOnly
        return TrajGazeV2TemporalGazeOnly(n_vis_keyframes=n_vis_keyframes)
    elif model_type == "hand_only":
        from TrajGaze_v2.models.model_temporal_hand_only import TrajGazeV2TemporalHandOnly
        return TrajGazeV2TemporalHandOnly(n_vis_keyframes=n_vis_keyframes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args   = parse_args()
    device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[stage1_ablation] model={args.model_type}  n_frames={args.n_frames}")
    print(f"[stage1_ablation] output: {args.output_dir}")

    dataset = StreamGazeStage1DatasetTemporal(n_frames=args.n_frames)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collate_stage1_temporal,
        num_workers = args.workers,
        pin_memory  = True,
        drop_last   = True,
    )

    model = build_model(args.model_type, args.n_vis_keyframes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[stage1_ablation] Resumed from epoch {start_epoch}")

    log_path  = os.path.join(args.output_dir, "train_log.jsonl")
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
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

            optimizer.zero_grad()
            loss_dict = model.stage1_forward(batch)
            loss      = loss_dict["loss"]

            if not torch.isfinite(loss):
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

            if (step + 1) % args.log_every == 0:
                avg = lambda x: x / n_batches
                print(f"Epoch {epoch+1:3d} | step {step+1:4d} | "
                      f"loss={avg(total_loss):.4f} traj={avg(total_traj):.4f} "
                      f"sf={avg(total_sf):.4f} sp={avg(total_sp):.4f} "
                      f"st={avg(total_st):.4f} | t={time.time()-t_start:.1f}s")

        scheduler.step()
        epoch_loss = total_loss / max(1, n_batches)

        print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={epoch_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | time={time.time()-t_start:.1f}s ===\n")

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1, "loss": epoch_loss,
                "loss_traj": total_traj / max(1, n_batches),
                "loss_score_fut": total_sf / max(1, n_batches),
                "loss_score_past": total_sp / max(1, n_batches),
                "loss_score_traj": total_st / max(1, n_batches),
                "lr": scheduler.get_last_lr()[0],
            }) + "\n")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth")
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": epoch_loss,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "loss": best_loss},
                       os.path.join(args.output_dir, "best.pth"))

    torch.save({"model": model.state_dict()},
               os.path.join(args.output_dir, "final.pth"))
    print(f"\n[stage1_ablation] Done. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
