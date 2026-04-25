"""
Baseline: fine-tune Qwen2.5-VL-7B with LoRA on StreamGaze_v2 MCQ tasks.

Condition : 100% visual tokens (128 frames, 224×224)
Train      : egoexolearn split
Test       : egtea split (evaluated periodically)
GPU usage  : 2 GPUs via torchrun

Usage:
    torchrun --nproc_per_node=2 -m TrajGazeMerge.training.train_baseline_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora \
        --epochs 3 --lr 1e-4 --grad-accum 4 --log-every 20 --eval-every 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item, build_full_inputs, forward_logits
)

OUTPUT_ROOT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora"


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
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def evaluate(processor, model, base_qwen, option_ids, device, max_items=200):
    """Evaluate on egtea test split; returns accuracy."""
    from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
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
                full_inputs = build_full_inputs(base_qwen, cached)
                logits = forward_logits(model, full_inputs)
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
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Baseline LoRA] output: {args.output_dir}")
        print(f"[Baseline LoRA] GPUs={world_size}, epochs={args.epochs}, "
              f"lr={args.lr}, grad_accum={args.grad_accum}")

    # ── Load model ────────────────────────────────────────────────────────────
    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)

    # base_qwen: underlying model without PEFT wrapper (for preprocess_item)
    base_qwen = model.get_base_model()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)

    if is_main:
        print("Model loaded.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = StreamGazeMergeDataset(split="train", n_vlm_frames=args.n_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size,
                                  rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    # ── Optimizer: only LoRA parameters ──────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path    = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc    = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        epoch_loss  = 0.0
        n_steps     = 0
        t_start     = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                # Preprocess: tokenize + extract ViT features (no grad needed for ViT)
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                # Build full-token input embeddings
                full_inputs = build_full_inputs(base_qwen, cached)

                # Forward through LoRA model (all visual tokens)
                logits = forward_logits(model, full_inputs)  # (vocab_size,)

                # MCQ loss: CE over 4 option logits
                option_logits = logits[option_ids]           # (4,)
                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                loss   = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
                loss = loss / args.grad_accum
                loss.backward()

                epoch_loss += loss.item() * args.grad_accum
                n_steps    += 1
                global_step += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed  = time.time() - t_start
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "elapsed": elapsed,
                        }) + "\n")

                if is_main and n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate(
                        processor, model.module, base_qwen, option_ids, device
                    )
                    print(f"  → eval egtea: {acc:.2f}% (n={n_eval})")
                    if acc > best_acc:
                        best_acc = acc
                        ckpt = {
                            "epoch": epoch, "step": n_steps,
                            "lora_state": model.module.state_dict(),
                            "acc": acc,
                        }
                        torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # End of epoch: flush remaining gradients
        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_loss = epoch_loss / max(1, n_steps)
            elapsed  = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")

            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth")
            torch.save({
                "epoch": epoch,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

    # Final evaluation
    if is_main:
        acc, n_eval = evaluate(
            processor, model.module, base_qwen, option_ids, device, max_items=500
        )
        print(f"\n[Final] egtea accuracy: {acc:.2f}% (n={n_eval})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
