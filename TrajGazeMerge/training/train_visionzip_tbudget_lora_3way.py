"""Temporal-Budget VisionZip — 3-way LoRA training (4-GPU DDP).

Refined trajectory utilizer (see TrajGazeMerge/models/temporal_budget.py). Same
VisionZip backbone, same 10% token budget (5% dominant + 5% contextual), same
DDP / dataset / LoRA protocol as plain VisionZip / VZ-traj / QC-Gate. The ONLY
change vs plain VisionZip is the token-SELECTION rule: instead of VisionZip's
global top-K (which on overlay data redundantly concentrates on the already-
visible gaze marker), the 10% budget is reallocated ACROSS FRAMES by the
gaze/hand trajectory's per-frame interaction signal, while each frame's WITHIN-
frame selection stays exactly VisionZip's (top-K-by-attention + cluster
contextual). Zero learnable selection params — only the LoRA adapter trains,
exactly as for plain VisionZip (no controller -> no dead-gradient failure mode).

Usage (4-GPU DDP, matching the VZ/VZ-traj/QC-Gate protocol; overlay frames):
    GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
      --master_port=29841 \\
      -m TrajGazeMerge.training.train_visionzip_tbudget_lora_3way \\
      --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tbudget_lora_3way_overlay \\
      --epochs 3 --lr 1e-4 --grad-accum 2
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

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.temporal_budget import temporal_budget_select_tokens
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, CONTEXTUAL_RATIO,
    load_visionzip_lora, preprocess_visionzip_item,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tbudget_lora_3way_overlay")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad-accum",  type=int,   default=2)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=20)
    p.add_argument("--n-frames",    type=int,   default=128)
    # temporal-budget knobs (zero learnable params)
    p.add_argument("--tau",         type=float, default=1.0,
                   help="softmax temperature on per-frame interaction (lower = more peaked)")
    p.add_argument("--traj-weight", type=float, default=0.5,
                   help="blend: (1-w)*VZ attention share + w*trajectory softmax")
    p.add_argument("--sigma-v",     type=float, default=0.05)
    p.add_argument("--sigma-gh",    type=float, default=0.10)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="2-way: StreamGaze + EgoGazeVQA only (egtea n=1011) — matches the "
                        "other clean experiments.")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def select(cached, item, hp):
    """Temporal-budget selection (no grad; zero-param). Returns
    (sel_embeds, recv_idx, budget_stats)."""
    sel_embeds, recv_idx, K = temporal_budget_select_tokens(
        cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
        cached["grid_thw"], item["traj"],
        tau=hp["tau"], traj_weight=hp["traj_weight"],
        sigma_v=hp["sigma_v"], sigma_gh=hp["sigma_gh"],
    )
    if K is None:
        stats = (0, 0.0)
    else:
        tot = int(K.sum().item())
        active = int((K > 0).sum().item())
        max_frac = float(K.max().item()) / max(1, tot)
        stats = (active, max_frac)
    return sel_embeds, recv_idx, stats


def evaluate(processor, model, base_qwen, option_ids, device, hp):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128,
        include_hdepic=hp.get("include_hdepic", True),
    )
    model.eval()
    correct = 0; total = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for item in test_ds:
            if item is None: continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None: continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                sel_embeds, recv_idx, _ = select(cached, item, hp)
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

    overlay = os.environ.get("GAZE_OVERLAY", "0")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[T-Budget 3way] output: {args.output_dir}")
        print(f"[T-Budget 3way] GPUs={world_size}, dominant={DOMINANT_RATIO*100:.0f}% + "
              f"contextual={CONTEXTUAL_RATIO*100:.0f}% = 10%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")
        print(f"[T-Budget 3way] GAZE_OVERLAY={overlay}  tau={args.tau} "
              f"traj_weight={args.traj_weight} sigma_v={args.sigma_v} sigma_gh={args.sigma_gh}")
        print("[T-Budget 3way] selection: VZ within-frame, per-frame budget by "
              "(1-w)*attn_share + w*softmax(interaction/tau). Zero learnable selection params.")

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)
    if is_main: print("Model loaded.")

    hp = dict(tau=args.tau, traj_weight=args.traj_weight,
              sigma_v=args.sigma_v, sigma_gh=args.sigma_gh,
              include_hdepic=args.include_hdepic)

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

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_steps = 0
        act_sum = 0.0; mf_sum = 0.0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None: continue
            try:
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device,
                    )
                if cached is None: continue
                n_video = cached["video_embeds"].shape[0]

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                with torch.no_grad():
                    sel_embeds, recv_idx, stats = select(cached, item, hp)
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
                act_sum += stats[0]; mf_sum += stats[1]

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    n_kept = recv_idx.shape[0]
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    act = act_sum / n_steps; mf = mf_sum / n_steps
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | "
                          f"active_frames={act:.1f} max_frame_share={mf:.2f} | "
                          f"t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept,
                            "active_frames": act, "max_frame_share": mf,
                            "elapsed": elapsed,
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
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | "
                  f"time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
                "tau": args.tau, "traj_weight": args.traj_weight,
                "sigma_v": args.sigma_v, "sigma_gh": args.sigma_gh,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on full 3-way val set ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device, hp,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                    "tau": args.tau, "traj_weight": args.traj_weight,
                    "sigma_v": args.sigma_v, "sigma_gh": args.sigma_gh,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
