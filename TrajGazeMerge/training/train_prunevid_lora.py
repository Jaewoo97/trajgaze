"""
PruneVid + LoRA training on StreamGaze.

Token selection: question-guided cosine similarity between mean-pooled text embeddings
and video token embeddings → hard top-10% kept, bottom 90% discarded.

LoRA specs: r=16, alpha=32, dropout=0.05, targets=[q,k,v,o]_proj.
Train: egoexolearn + holoassist (~5,799 MCQ items).
Eval:  EGTEA (~526 MCQ items) after each epoch.
GPUs:  2 via torchrun.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29609 \\
        -m TrajGazeMerge.training.train_prunevid_lora \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/prunevid_lora \\
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/prunevid_lora")
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
    """
    Score visual tokens by cosine similarity with mean-pooled question embeddings.
    Returns (N_video,) scores on device.
    """
    input_ids    = cached["input_ids"]     # (1, L)
    video_embeds = cached["video_embeds"]  # (N_video, d)

    video_token_id = base_qwen.config.video_token_id
    is_text = (input_ids[0] != video_token_id)  # (L,) bool

    text_ids = input_ids[:, is_text]  # (1, n_text)
    with torch.no_grad():
        text_embeds = base_qwen.get_input_embeddings()(
            text_ids.to(device)
        ).squeeze(0).float()  # (n_text, d)

    query  = F.normalize(text_embeds.mean(dim=0), dim=-1)  # (d,)
    vf     = F.normalize(video_embeds.float(), dim=-1)      # (N_video, d)
    scores = vf @ query                                      # (N_video,)
    return scores


def select_prunevid_tokens(base_qwen, cached: dict, keep_ratio: float, device: torch.device):
    """Top-k hard selection; returns (kept_embeds, receiver_idx)."""
    video_embeds = cached["video_embeds"]
    N            = video_embeds.shape[0]
    n_keep       = max(1, int(keep_ratio * N))

    scores       = compute_prunevid_scores(base_qwen, cached, device)
    top_idx      = torch.topk(scores, n_keep, largest=True).indices  # (n_keep,)
    receiver_idx, _ = top_idx.sort()

    return video_embeds[receiver_idx], receiver_idx


def evaluate(processor, model, base_qwen, option_ids, device, keep_ratio):
    """Eval on full EGTEA test set; returns (acc, n, per_task)."""
    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
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

                kept_embeds, receiver_idx = select_prunevid_tokens(
                    base_qwen, cached, keep_ratio, device
                )
                inputs_dict   = build_merged_inputs(base_qwen, cached, kept_embeds, receiver_idx)
                logits        = forward_logits(model, inputs_dict)
                pred_idx      = logits[option_ids].argmax().item()
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
        print(f"[PruneVid LoRA] output: {args.output_dir}")
        print(f"[PruneVid LoRA] GPUs={world_size}, keep={args.keep_ratio*100:.0f}%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")

    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor)

    # Resume from checkpoint if specified
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

    train_ds = StreamGazeSimpleDataset(split="train", n_vlm_frames=args.n_frames)
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

                # PruneVid-style question-guided token selection (no grad through indices)
                with torch.no_grad():
                    kept_embeds, receiver_idx = select_prunevid_tokens(
                        base_qwen, cached, args.keep_ratio, device
                    )
                    inputs_dict = build_merged_inputs(
                        base_qwen, cached, kept_embeds, receiver_idx
                    )

                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
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

        # Flush remaining gradients at end of epoch
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
            print(f"Evaluating epoch {epoch+1} on EGTEA test set ...", flush=True)
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
