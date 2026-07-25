"""VisionZip-Complement + Gaze-Text prefix — LoRA training.

M1 token selection (7%C ∪ 3%G topk, learned TAS encoder) is unchanged.
The ONLY addition: a compact fixation-sequence text is prepended to each
question so the LLM sees gaze temporal dynamics as language:

    [Gaze] center(5f) → upper-left(3f+hand) → center(6f+hand) → lower-right(2f)

This is orthogonal to visual token selection — attention cannot encode
temporal gaze order or fixation duration. Arms:

  gaze   — actual gaze trajectory text        (the method)
  random — random region sequence, same format (placebo: format without content)
  none   — no prefix                          (baseline ≈ M1; sanity check)

Training protocol: single-GPU + grad-accum 8 = eff-batch 8
(matches tbudget / anticipatory / foveal-ROI novelty experiments).

Usage:
    GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=<N> torchrun --nproc_per_node=1 \\
        --master_port=<PORT> \\
        -m TrajGazeMerge.training.train_visionzip_gazetext_lora \\
        --arm gaze \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/gazetext_gaze \\
        --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.gaze_text import build_question_prefix
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import (
    select_complementary,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"
CONTENT_RATIO  = 0.07
TRAJ_RATIO     = 0.03


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["gaze", "random", "none"], default="gaze",
                   help="gaze=gaze-text prefix (method); random=placebo; none=baseline")
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/gazetext_gaze")
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=8)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if epoch-2 val <= epoch-1 val.")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _item_seed(item) -> int:
    """Deterministic per-item seed for the random placebo arm."""
    s = "|".join([str(item.get("task", "")), str(item.get("question", ""))])
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _make_question(arm: str, item: dict, n_frames: int) -> str:
    """Prepend gaze text (if any) to the item's question."""
    prefix = build_question_prefix(arm, item["traj"], n_frames, _item_seed(item))
    return f"{prefix}\n{item['question']}" if prefix else item["question"]


def _hp_default():
    return dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0, alpha_hand=0.7,
                sigma_v=0.05, sigma_gh=0.10)


def evaluate(processor, model, base_qwen, option_ids, device,
             arm, encoder, n_frames, include_hdepic):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=n_frames, n_traj_frames=n_frames,
        include_hdepic=include_hdepic,
    )
    hp = _hp_default()
    model.eval()
    correct = 0; total = 0
    by_task: dict[str, list] = {}
    with torch.no_grad():
        for item in test_ds:
            if item is None: continue
            try:
                question = _make_question(arm, item, n_frames)
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], question, item["options"], device,
                )
                if cached is None: continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                sel_embeds, recv_idx = select_complementary(
                    cached, item, device, "learned", encoder, hp,
                    CONTENT_RATIO, TRAJ_RATIO, complement_mode="topk")
                inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                logits = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok; total += 1
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
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[GazeText] arm={args.arm}  output: {args.output_dir}")
        print(f"[GazeText] M1 selection: {CONTENT_RATIO*100:.0f}%C ∪ {TRAJ_RATIO*100:.0f}%G "
              f"(topk, learned encoder) + gaze-text prefix on question")
        print(f"[GazeText] GPUs={world_size} epochs={args.epochs} lr={args.lr} "
              f"grad_accum={args.grad_accum}", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    if is_main: print("Model + encoder loaded.", flush=True)

    hp = _hp_default()

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0
    epoch_accs: list[float] = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None: continue
            try:
                question = _make_question(args.arm, item, args.n_frames)
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], question, item["options"], device,
                    )
                if cached is None: continue
                n_video = cached["video_embeds"].shape[0]

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                with torch.no_grad():
                    sel_embeds, recv_idx = select_complementary(
                        cached, item, device, "learned", encoder, hp,
                        CONTENT_RATIO, TRAJ_RATIO, complement_mode="topk")
                    inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)

                logits = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                loss = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
                loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    pct_kept = 100.0 * recv_idx.shape[0] / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s",
                          flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")
            except Exception:
                if is_main: traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed = time.time() - t_start
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} "
                  f"| time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
                "arm": args.arm,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            print(f"Evaluating epoch {epoch+1} (egtea 2-way) ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device,
                args.arm, encoder, args.n_frames, args.include_hdepic,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            epoch_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "acc": acc, "arm": args.arm,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)

            if args.early_stop and epoch >= 1 and acc <= epoch_accs[-2]:
                print(f"  Early stop: epoch{epoch+1} {acc:.2f}% <= "
                      f"epoch{epoch} {epoch_accs[-2]:.2f}% → stopping.", flush=True)
                stop.fill_(1.0)

        dist.broadcast(stop, src=0)
        dist.barrier()
        if stop.item() > 0:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
