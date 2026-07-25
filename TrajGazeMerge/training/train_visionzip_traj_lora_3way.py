"""VisionZip + trajectory-weighted selection — 3-way LoRA training.

Same as train_visionzip_lora.py except:
  1. Dataset: CombinedMergeDataset (provides item["traj"] = gaze + L/R hand pos)
     instead of CombinedSimpleDataset.
  2. Selection: attn_scores are multiplied by (1 + spatial_w) * temporal_w
     (Recipe 1 + Recipe 3) before VisionZip's top-K dominant pool selection.
     Zero added params.

Token budget unchanged: 5% dominant + 5% contextual = 10%.
LoRA config + protocol unchanged: r=16, α=32, dropout 0.05, lr 1e-4,
grad_accum 4, 3 epochs.

Usage (2-GPU DDP, matching VZ protocol):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29823 \\
      -m TrajGazeMerge.training.train_visionzip_traj_lora_3way \\
      --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_traj_lora_3way_v3upright \\
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

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.traj_weights import traj_weighted_attn_scores
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, CONTEXTUAL_RATIO,
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_traj_lora_3way_v3upright")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    # Trajectory weighting hyperparameters (defaults are sensible for normalized coords)
    p.add_argument("--sigma-g",    type=float, default=2.0)
    p.add_argument("--sigma-h",    type=float, default=3.0)
    p.add_argument("--alpha-hand", type=float, default=0.7)
    p.add_argument("--sigma-v",    type=float, default=0.05)
    p.add_argument("--sigma-gh",   type=float, default=0.10)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="2-way: StreamGaze + EgoGazeVQA only (exclude HD-EPIC).")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if epoch-2 val <= epoch-1 val.")
    p.add_argument("--no-mid-eval", action="store_true",
                   help="Disable the mid-epoch eval/checkpoint (end-of-epoch-only protocol).")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def select_with_traj(cached, item, device, hp):
    """Run VisionZip selection with trajectory-aware attn_score weighting."""
    weighted_scores = traj_weighted_attn_scores(
        cached["attn_scores"], cached["grid_thw"], item["traj"], device,
        sigma_g=hp["sigma_g"], sigma_h=hp["sigma_h"], alpha_hand=hp["alpha_hand"],
        sigma_v=hp["sigma_v"], sigma_gh=hp["sigma_gh"],
    )
    return visionzip_select_tokens(
        cached["video_embeds"], weighted_scores, cached["attn_key"],
    )


def evaluate(processor, model, base_qwen, option_ids, device, hp, include_hdepic=True):
    """Full val eval (3-way, or 2-way egtea when include_hdepic=False)."""
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic,
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

                sel_embeds, recv_idx = select_with_traj(cached, item, device, hp)
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
        print(f"[VZ-traj 3way] output: {args.output_dir}")
        print(f"[VZ-traj 3way] GPUs={world_size}, dominant={DOMINANT_RATIO*100:.0f}% + "
              f"contextual={CONTEXTUAL_RATIO*100:.0f}% = 10%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")
        print(f"[VZ-traj 3way] hp: σ_g={args.sigma_g} σ_h={args.sigma_h} α_hand={args.alpha_hand} "
              f"σ_v={args.sigma_v} σ_gh={args.sigma_gh}")

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)
    if is_main: print("Model loaded.")

    hp = dict(
        sigma_g=args.sigma_g, sigma_h=args.sigma_h, alpha_hand=args.alpha_hand,
        sigma_v=args.sigma_v, sigma_gh=args.sigma_gh,
    )

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
        mid_step = max(1, len(loader) // 2)
        mid_done = False

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

                with torch.no_grad():
                    sel_embeds, recv_idx = select_with_traj(cached, item, device, hp)
                    inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue
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
                    n_kept = recv_idx.shape[0]
                    pct_kept = 100.0 * n_kept / max(1, n_video)
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

            # ---- mid-epoch eval at 50% of the epoch ----
            if (not args.no_mid_eval) and n_steps == mid_step and not mid_done:
                mid_done = True
                dist.barrier()
                if is_main:
                    print(f"\n=== Epoch {epoch+1} MID ({n_steps}/{len(loader)}) — saving + evaluating ===",
                          flush=True)
                    torch.save({
                        "epoch": epoch+1, "mid_epoch": True, "step": n_steps,
                        "lora_state": model.module.state_dict(),
                    }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}_mid.pth"))
                    acc, n_eval, per_task = evaluate(
                        processor, model.module, base_qwen, option_ids, device, hp,
                        include_hdepic=args.include_hdepic,
                    )
                    print(f"  [MID 50%] Overall: {acc:.2f}%  (n={n_eval})", flush=True)
                    for task, task_acc in per_task.items():
                        print(f"    {task}: {task_acc:.2f}%", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "phase": "mid", "step": n_steps,
                            "eval_acc": acc, "n_eval": n_eval, "per_task": per_task,
                        }) + "\n")
                dist.barrier()
                model.train()

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
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} on full {label} val set ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device, hp,
                include_hdepic=args.include_hdepic,
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
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
            if args.early_stop and (epoch + 1) == 2 and len(epoch_accs) >= 2 \
                    and epoch_accs[1] <= epoch_accs[0]:
                stop[0] = 1.0
                print(f"  Early stop: epoch2 {epoch_accs[1]:.2f}% <= epoch1 "
                      f"{epoch_accs[0]:.2f}% → skipping epoch 3.", flush=True)
        dist.barrier()
        dist.broadcast(stop, src=0)
        if stop.item() > 0:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
