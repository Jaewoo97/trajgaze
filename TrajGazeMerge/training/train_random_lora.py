"""
Random 10%: fine-tune Qwen2.5-VL-7B with LoRA on StreamGaze_v2 MCQ tasks.

Condition : random 10% visual tokens kept (128 frames → ~10% spatial tokens randomly selected)
Train      : egoexolearn + holoassist split
Test       : egtea split (evaluated periodically)
GPU usage  : 1 GPU

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m TrajGazeMerge.training.train_random_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/random_lora \
        --epochs 3 --lr 1e-4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)

OUTPUT_ROOT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/random_lora"
KEEP_RATIO  = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",  default=OUTPUT_ROOT)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad-accum",  type=int,   default=4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=20)
    p.add_argument("--eval-every",  type=int,   default=200)
    p.add_argument("--n-frames",    type=int,   default=128)
    p.add_argument("--keep-ratio",  type=float, default=KEEP_RATIO)
    return p.parse_args()


def build_random_inputs(base_qwen, cached: dict, keep_ratio: float, device) -> dict:
    """Randomly select keep_ratio fraction of video tokens, discard the rest."""
    from TrajGazeMerge.models.merge import gaze_weighted_merge
    video_embeds = cached["video_embeds"]  # (N_video, d)
    N_video = video_embeds.shape[0]
    n_keep  = max(1, int(keep_ratio * N_video))

    # Random scores → top-n_keep become receivers
    scores = torch.rand(N_video, device=device)
    merged_video, receiver_idx = gaze_weighted_merge(video_embeds, scores, N_video - n_keep)
    return build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)


def evaluate(processor, model, base_qwen, option_ids, device, keep_ratio, max_items=200):
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=128)
    test_ds.items = test_ds.items[:max_items]
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue
                inputs = build_random_inputs(base_qwen, cached, keep_ratio, device)
                logits = forward_logits(model, inputs)
                option_logits = logits[option_ids]
                pred_idx = option_logits.argmax().item()
                gt_idx   = ["A", "B", "C", "D"].index(item["answer"])
                correct += int(pred_idx == gt_idx)
                total   += 1
            except Exception:
                pass
    model.train()
    return 100.0 * correct / max(1, total), total


def main():
    args   = parse_args()
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Random LoRA] output: {args.output_dir}")
    print(f"[Random LoRA] keep_ratio={args.keep_ratio}, epochs={args.epochs}, "
          f"lr={args.lr}, grad_accum={args.grad_accum}")

    processor, model = load_qwen_lora(device)
    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor)
    print("Model loaded.")

    train_ds = StreamGazeMergeDataset(split="train", n_vlm_frames=args.n_frames)
    loader   = DataLoader(train_ds, batch_size=1, shuffle=True,
                          collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_steps    = 0
        t_start    = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                inputs = build_random_inputs(base_qwen, cached, args.keep_ratio, device)
                logits = forward_logits(model, inputs)
                option_logits = logits[option_ids]
                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                loss   = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
                loss = loss / args.grad_accum
                loss.backward()

                epoch_loss += loss.item() * args.grad_accum
                n_steps    += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed  = time.time() - t_start
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "elapsed": elapsed,
                        }) + "\n")

                if n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate(
                        processor, model, base_qwen, option_ids, device, args.keep_ratio
                    )
                    print(f"  → eval egtea: {acc:.2f}% (n={n_eval})")
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({
                            "epoch": epoch, "step": n_steps,
                            "lora_state": model.state_dict(), "acc": acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)")

            except Exception:
                traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed  = time.time() - t_start
        print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")
        torch.save({
            "epoch": epoch, "lora_state": model.state_dict(), "loss": avg_loss,
        }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    acc, n_eval = evaluate(
        processor, model, base_qwen, option_ids, device, args.keep_ratio, max_items=500
    )
    print(f"\n[Final] egtea: {acc:.2f}% (n={n_eval})")


if __name__ == "__main__":
    main()
