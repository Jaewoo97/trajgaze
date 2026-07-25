"""
Random-10% control: Qwen2.5-VL-7B + LoRA trained on combined 3-way data with
**uniform-random 10% visual token selection at every step** (no learned
encoder, no gaze/hand anchor).

Provides the apples-to-apples control vs the learned-selection methods on
this 3-way val. Same protocol as train_merge_lora_no_kd.py minus the
encoder. Mirrors the existing trainers (combined dataset, N-option MCQ,
DDP, AdamW, CE on answer letter).

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29713 \\
      -m TrajGazeMerge.training.train_random_lora_no_kd \\
      --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/random_lora_no_kd_3way \\
      --epochs 3 --lr 1e-4 --merge-ratio 0.9 --grad-accum 4 --log-every 10 --eval-every 600
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

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
)

OUTPUT_ROOT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/random_lora_no_kd_3way"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",  default=OUTPUT_ROOT)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--merge-ratio", type=float, default=0.9,
                   help="Fraction of tokens to merge away (0.9 = keep 10%%)")
    p.add_argument("--grad-accum",  type=int,   default=4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=20)
    p.add_argument("--eval-every",  type=int,   default=200)
    p.add_argument("--n-frames",    type=int,   default=128)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _random_merge(cached, device, merge_ratio, eval_seed=None):
    """Apply uniform-random 10% selection and return (merged_inputs, base_inputs_dict_unused)."""
    n_video = cached["video_embeds"].shape[0]
    r = max(1, int(merge_ratio * n_video))
    if eval_seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(eval_seed))
        scores = torch.rand(n_video, generator=gen, device=device, dtype=torch.float32)
    else:
        scores = torch.rand(n_video, device=device, dtype=torch.float32)
    video_det = cached["video_embeds"].detach()
    merged_video, receiver_idx = gaze_weighted_merge(video_det, scores, r)
    # caller will build merged inputs; we only return the pair
    return merged_video, receiver_idx


def evaluate(processor, model, base_qwen, option_ids, device, merge_ratio, max_items=200):
    """Eval on the combined egtea+P09 test split with deterministic per-item random merge."""
    test_ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=8)
    test_ds.items = test_ds.items[:max_items]
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for idx, item in enumerate(test_ds):
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                oids = option_ids[:n_opt]

                merged_video, receiver_idx = _random_merge(cached, device, merge_ratio, eval_seed=idx)
                merged_inputs = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                logits = forward_logits(model, merged_inputs)
                pred_idx = logits[oids].argmax().item()
                gt_idx = letters.index(item["answer"])
                correct += int(pred_idx == gt_idx); total += 1
            except Exception:
                pass
    model.train()
    return 100.0 * correct / max(1, total), total


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[RandomLoRA] output={args.output_dir}  merge_ratio={args.merge_ratio}")
        print(f"  GPUs={world_size}  epochs={args.epochs}  lr={args.lr}  grad_accum={args.grad_accum}")

    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)
    if is_main:
        print("Model loaded.")

    train_ds = CombinedMergeDataset(split="train", n_vlm_frames=args.n_frames, n_traj_frames=8)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_steps = 0
        t_start = time.time()
        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                oids = option_ids[:n_opt]

                merged_video, receiver_idx = _random_merge(cached, device, args.merge_ratio)
                merged_inputs = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                logits = forward_logits(model, merged_inputs)[oids]
                gt_tensor = torch.tensor([letters.index(item["answer"])], device=device)
                loss = F.cross_entropy(logits.unsqueeze(0), gt_tensor) / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg:.4f} | t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"epoch": epoch+1, "step": n_steps,
                                             "loss": avg, "elapsed": elapsed}) + "\n")

                if is_main and args.eval_every > 0 and n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate(
                        processor, model.module, base_qwen, option_ids, device, args.merge_ratio
                    )
                    print(f"  → eval egtea (random merge): {acc:.2f}% (n={n_eval})", flush=True)
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({"epoch": epoch, "step": n_steps,
                                    "lora_state": model.module.state_dict(),
                                    "acc": acc},
                                   os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)", flush=True)
            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg = epoch_loss / max(1, n_steps)
        if is_main:
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg:.4f} | time={elapsed:.0f}s ===",
                  flush=True)
            torch.save({"epoch": epoch, "lora_state": model.module.state_dict(), "loss": avg},
                       os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    if is_main:
        acc, n_eval = evaluate(
            processor, model.module, base_qwen, option_ids, device, args.merge_ratio, max_items=500
        )
        print(f"\n[Final] egtea (random merge): {acc:.2f}% (n={n_eval})", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
