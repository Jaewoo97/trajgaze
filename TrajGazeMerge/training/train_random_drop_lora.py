"""
Random Drop 10%: fine-tune Qwen2.5-VL-7B LoRA keeping a random 10% of visual tokens,
hard-dropping the other 90% (no merging/averaging of dropped tokens).

Contrast with train_random_lora.py which uses gaze_weighted_merge (the 90% sources
are score-weighted-averaged into the 10% receivers). Here the 90% are simply discarded.

Train : egoexolearn + holoassist (all MCQ)
Val   : egtea (full 526-item eval after every epoch)
GPUs  : 4 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29603 \
        -m TrajGazeMerge.training.train_random_drop_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/random_drop_lora \
        --epochs 3 --lr 1e-4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import datetime
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

from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)

KEEP_RATIO = 0.10


def build_random_drop_inputs(base_qwen, cached: dict, device) -> dict:
    """
    Randomly select 10% of video tokens and hard-drop the rest.
    Unlike gaze_weighted_merge, dropped tokens are not averaged into kept ones.
    """
    video_embeds = cached["video_embeds"]   # (N_video, d)
    N_video = video_embeds.shape[0]
    n_keep  = max(1, int(KEEP_RATIO * N_video))

    # Random permutation → pick first n_keep as receivers
    perm = torch.randperm(N_video, device=video_embeds.device)
    receiver_idx = perm[:n_keep]                          # (n_keep,)
    receiver_idx, _ = receiver_idx.sort()                 # preserve spatial order

    # Pass raw (un-merged) embeddings for the kept tokens
    kept_embeds = video_embeds[receiver_idx]              # (n_keep, d)

    return build_merged_inputs(base_qwen, cached, kept_embeds, receiver_idx)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",  default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/random_drop_lora")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad-accum",  type=int,   default=4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=20)
    p.add_argument("--n-frames",    type=int,   default=128)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def evaluate(processor, model, base_qwen, option_ids, device):
    from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    model.eval()
    correct = 0
    total   = 0
    by_task: dict[str, list] = {}
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
                inputs        = build_random_drop_inputs(base_qwen, cached, device)
                logits        = forward_logits(model, inputs)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)
            except Exception:
                pass
    model.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Random Drop LoRA] output: {args.output_dir}")
        print(f"[Random Drop LoRA] GPUs={world_size}, keep={KEEP_RATIO*100:.0f}% (hard drop), "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")

    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen  = model.get_base_model()
    model      = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main:
        print("Model loaded.")

    train_ds = StreamGazeSimpleDataset(split="train", n_vlm_frames=args.n_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_steps    = 0
        t_start    = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                with torch.no_grad():
                    cached = preprocess_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device
                    )
                if cached is None:
                    continue

                with torch.no_grad():
                    inputs_dict = build_random_drop_inputs(base_qwen, cached, device)

                logits        = forward_logits(model, inputs_dict)
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

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed  = time.time() - t_start
                    n_video  = cached["video_embeds"].shape[0]
                    n_kept   = max(1, int(KEEP_RATIO * n_video))
                    pct_kept = 100.0 * n_kept / n_video
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed  = time.time() - t_start
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")
            torch.save({
                "epoch": epoch,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on full EGTEA val set ...")
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})")
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
