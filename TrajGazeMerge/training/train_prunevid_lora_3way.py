"""
PruneVid + LoRA training on the 3-way combined data (SG + EG + HD-EPIC).

Same selection rule as the original PruneVid trainer (question-guided cosine
similarity, hard top-10% kept), with the dataset swapped to
CombinedSimpleDataset so it sees HD-EPIC + EgoGazeVQA + StreamGaze and
evaluates on the full 4936-item per-source set after each epoch.

LoRA specs: r=16, alpha=32, dropout=0.05, targets=[q,k,v,o]_proj — identical
to VisionZip's 3-way trainer so the rows are apples-to-apples.

Usage (matches VisionZip's 2-GPU setup):
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29724 \\
        -m TrajGazeMerge.training.train_prunevid_lora_3way \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/prunevid_lora_3way_v3upright \\
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

from TrajGazeMerge.data.combined_simple_dataset import CombinedSimpleDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)

KEEP_RATIO = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/prunevid_lora_3way_v3upright")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--keep-ratio", type=float, default=KEEP_RATIO)
    p.add_argument("--resume-ckpt", type=str,  default=None)
    p.add_argument("--start-epoch", type=int,  default=0)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def compute_prunevid_scores(base_qwen, cached: dict, device: torch.device) -> torch.Tensor:
    input_ids    = cached["input_ids"]
    video_embeds = cached["video_embeds"]

    video_token_id = base_qwen.config.video_token_id
    is_text = (input_ids[0] != video_token_id)

    text_ids = input_ids[:, is_text]
    with torch.no_grad():
        text_embeds = base_qwen.get_input_embeddings()(
            text_ids.to(device)
        ).squeeze(0).float()

    query  = F.normalize(text_embeds.mean(dim=0), dim=-1)
    vf     = F.normalize(video_embeds.float(), dim=-1)
    scores = vf @ query
    return scores


def select_prunevid_tokens(base_qwen, cached: dict, keep_ratio: float, device: torch.device):
    video_embeds = cached["video_embeds"]
    N            = video_embeds.shape[0]
    n_keep       = max(1, int(keep_ratio * N))

    scores       = compute_prunevid_scores(base_qwen, cached, device)
    top_idx      = torch.topk(scores, n_keep, largest=True).indices
    receiver_idx, _ = top_idx.sort()

    return video_embeds[receiver_idx], receiver_idx


def evaluate(processor, model, base_qwen, option_ids, device, keep_ratio):
    """Eval on full 3-way test set (526 SG + 485 EG + 3925 HD = 4936 items)."""
    test_ds = CombinedSimpleDataset(split="test", n_vlm_frames=128)
    model.eval()
    correct  = 0
    total    = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue
                n_opt   = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue

                kept_embeds, receiver_idx = select_prunevid_tokens(
                    base_qwen, cached, keep_ratio, device
                )
                inputs_dict   = build_merged_inputs(base_qwen, cached, kept_embeds, receiver_idx)
                logits        = forward_logits(model, inputs_dict)
                pred_idx      = logits[option_ids[:n_opt]].argmax().item()
                gt_idx        = letters.index(item["answer"])
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
        print(f"[PruneVid 3way LoRA] output: {args.output_dir}")
        print(f"[PruneVid 3way LoRA] GPUs={world_size}, keep={args.keep_ratio*100:.0f}%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")

    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor, 5)   # A-E pool; sliced per item

    resume_path = args.resume_ckpt
    if resume_path is None and args.start_epoch > 0:
        resume_path = os.path.join(args.output_dir, f"epoch_{args.start_epoch:02d}.pth")
    if resume_path and os.path.isfile(resume_path):
        ckpt  = torch.load(resume_path, map_location="cpu", weights_only=False)
        state = ckpt.get("lora_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            print(f"Resumed from {resume_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        print("Qwen loaded.")

    train_ds = CombinedSimpleDataset(split="train", n_vlm_frames=args.n_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
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
                        item["vlm_frame_paths"], item["question"],
                        item["options"], device,
                    )
                if cached is None:
                    continue
                n_opt   = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue

                with torch.no_grad():
                    kept_embeds, receiver_idx = select_prunevid_tokens(
                        base_qwen, cached, args.keep_ratio, device
                    )
                    inputs_dict = build_merged_inputs(
                        base_qwen, cached, kept_embeds, receiver_idx
                    )

                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx        = letters.index(item["answer"])
                loss = F.cross_entropy(
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
                    n_kept   = max(1, int(args.keep_ratio * n_video))
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s",
                          flush=True)
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
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch + 1,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on full 3-way val set ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device, args.keep_ratio,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch + 1,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
