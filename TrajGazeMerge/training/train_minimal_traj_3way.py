"""Minimal-trajectory-scorer 3-way trainer — 4-GPU DDP.

Philosophy (same as TAS): predict gaze/hand trajectories from visual
features; use predictor's attention map over patches as a per-token
importance score for gaze_weighted_merge selection.

Difference vs TAS: no separate ViT/DINOv2. The scorer takes Qwen's already-
computed video_embeds (zero extra ViT forward) and uses 3 learnable queries
(gaze, left_hand, right_hand) with one cross-attention layer + a coord head.
Total ~493K params (vs TAS's 35.8M).

Losses:
    L = L_vqa + lambda_traj * L_traj
        L_vqa  = cross-entropy on answer letter (same as TAS)
        L_traj = Huber on predicted (x,y) for gaze/L/R, masked by validity

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29822 \\
      -m TrajGazeMerge.training.train_minimal_traj_3way \\
      --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/minimal_traj_3way_v3upright \\
      --epochs 3 --merge-ratio 0.9 --lambda-traj 0.5 \\
      --micro-batch 2 --grad-accum 4 \\
      --dataloader-num-workers 8
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
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

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.minimal_traj_scorer import MinimalTrajScorer
from TrajGazeMerge.models.model import load_qwen_lora, get_option_ids
from TrajGazeMerge.models.model_batched import (
    preprocess_batch, slice_sample_video,
    build_merged_inputs_batch, forward_logits_batch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/minimal_traj_3way_v3upright")
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--lr-lora",         type=float, default=1e-4)
    p.add_argument("--lr-scorer",       type=float, default=1e-3)
    p.add_argument("--lambda-traj",     type=float, default=0.5)
    p.add_argument("--merge-ratio",     type=float, default=0.9)
    p.add_argument("--d-hidden",        type=int,   default=128)
    p.add_argument("--micro-batch",     type=int,   default=2)
    p.add_argument("--grad-accum",      type=int,   default=4)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--log-every",       type=int,   default=20)
    p.add_argument("--eval-every",      type=int,   default=400)
    p.add_argument("--n-frames",        type=int,   default=128)
    p.add_argument("--n-traj-frames",   type=int,   default=128)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--dataloader-num-workers", type=int, default=8)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _filter_valid(items):
    return [x for x in items if x is not None]


def _pool_to_T(pos: torch.Tensor, mask: torch.Tensor, T_target: int):
    """Map (T_src, 2) positions + (T_src,) mask to (T_target, 2) + (T_target,).

    Positions: linear interpolation. Mask: nearest down/upsample.
    """
    T_src = pos.shape[0]
    if T_src == T_target:
        return pos.float(), mask.bool()
    p = pos.t().unsqueeze(0).float()                                     # (1, 2, T_src)
    p_t = F.interpolate(p, size=T_target, mode="linear", align_corners=False)
    p_t = p_t.squeeze(0).t().contiguous()                                # (T_target, 2)
    m = mask.float().view(1, 1, T_src)
    m_t = F.interpolate(m, size=T_target, mode="nearest").view(T_target) > 0.5
    return p_t, m_t


def _build_gt_for_item(item: dict, T_merged: int, device: torch.device):
    """Return gt_coords (T_merged, 3, 2) and gt_mask (T_merged, 3) for one item."""
    traj = item["traj"]
    gp, gm = _pool_to_T(traj["gaze_pos"],  traj["gaze_mask"],  T_merged)
    lp, lm = _pool_to_T(traj["left_pos"],  traj["left_mask"],  T_merged)
    rp, rm = _pool_to_T(traj["right_pos"], traj["right_mask"], T_merged)
    coords = torch.stack([gp, lp, rp], dim=1).to(device)   # (T, 3, 2)
    mask   = torch.stack([gm, lm, rm], dim=1).to(device)   # (T, 3)
    return coords, mask


def _traj_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Huber loss on (T, 3, 2) pred vs gt, masked by (T, 3) mask."""
    if not mask.any():
        return pred.sum() * 0.0
    m = mask.float().unsqueeze(-1)                          # (T, 3, 1)
    diff = (pred - gt) * m
    n = m.sum().clamp(min=1)
    return F.huber_loss(diff, torch.zeros_like(diff), reduction="sum") / n


@torch.no_grad()
def evaluate_per_source(qwen_model, base_qwen, scorer, processor,
                         option_ids_full, device, merge_ratio, max_items=200):
    """Per-source SG/EG/HD eval on a small subset of CombinedMergeDataset(test)."""
    qwen_model.eval(); scorer.eval()
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True,
    )
    per_source = {"sg": {"ok": 0, "n": 0}, "eg": {"ok": 0, "n": 0}, "hd": {"ok": 0, "n": 0}}

    # Iterate balanced subset: first max_items from each source
    SG_END, EG_END = 526, 526 + 485
    indices = (list(range(min(max_items, SG_END)))
             + list(range(SG_END, SG_END + min(max_items, EG_END - SG_END)))
             + list(range(EG_END, EG_END + min(max_items, len(test_ds) - EG_END))))

    for idx in indices:
        try:
            item = test_ds[idx]
        except Exception:
            continue
        if item is None: continue
        src = "sg" if idx < SG_END else ("eg" if idx < EG_END else "hd")
        try:
            batched = preprocess_batch(processor, base_qwen, [item], device)
            if batched is None or not batched["valid_idx"]: continue
            video_b = slice_sample_video(batched, 0)
            N_b = video_b.shape[0]
            T_b = int(batched["grid_thw"][0, 0].item())
            S_b = N_b // T_b
            r_b = max(1, int(merge_ratio * N_b))
            _, scores = scorer(video_b, T_b, S_b)
            merged_video, recv = gaze_weighted_merge(video_b, scores, r_b)
            inputs = build_merged_inputs_batch(base_qwen, batched, [merged_video], [recv])
            logits = forward_logits_batch(qwen_model, inputs)
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters: continue
            pred_idx = logits[0, option_ids_full[:n_opt]].argmax().item()
            gt_idx = letters.index(item["answer"])
            per_source[src]["ok"] += int(pred_idx == gt_idx)
            per_source[src]["n"]  += 1
        except Exception:
            continue

    res = {}
    for s in ("sg", "eg", "hd"):
        n = max(1, per_source[s]["n"])
        res[s] = (100.0 * per_source[s]["ok"] / n, per_source[s]["n"])
    mean_acc = sum(res[s][0] for s in ("sg", "eg", "hd")) / 3
    res["mean"] = (mean_acc, sum(per_source[s]["n"] for s in ("sg", "eg", "hd")))
    qwen_model.train(); scorer.train()
    return res


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[MinTraj-3way] world_size={world_size} micro_batch={args.micro_batch} "
              f"grad_accum={args.grad_accum} effective_batch={args.micro_batch * args.grad_accum * world_size}",
              flush=True)
        print(f"  output_dir: {args.output_dir}", flush=True)
        print(f"  lambda_traj={args.lambda_traj}, d_hidden={args.d_hidden}", flush=True)

    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    option_ids_full = get_option_ids(processor, 5)

    # Infer d_in from a probe (Qwen visual hidden size = LLM hidden size = 3584)
    d_in = base_qwen.config.hidden_size
    scorer = MinimalTrajScorer(d_in=d_in, d_hidden=args.d_hidden).to(device)
    if is_main:
        print(f"[MinTraj] scorer d_in={d_in} d_hidden={args.d_hidden} "
              f"params={scorer.param_count()/1e3:.1f}K", flush=True)

    qwen_model = DDP(qwen_model, device_ids=[local_rank], find_unused_parameters=True)
    scorer     = DDP(scorer,     device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        print("Models loaded + wrapped in DDP.", flush=True)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
        include_hdepic=True,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=args.micro_batch, sampler=sampler,
                         collate_fn=lambda b: b, num_workers=args.dataloader_num_workers)

    lora_params   = [p for p in qwen_model.parameters() if p.requires_grad]
    scorer_params = list(scorer.parameters())
    optimizer = AdamW([
        {"params": lora_params,   "lr": args.lr_lora},
        {"params": scorer_params, "lr": args.lr_scorer},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        qwen_model.train(); scorer.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; epoch_ce = 0.0; epoch_traj = 0.0
        micro_step = 0; accum_step = 0
        t_start = time.time()

        for raw_batch in loader:
            items = _filter_valid(raw_batch)
            if not items:
                continue
            try:
                batched = preprocess_batch(processor, base_qwen, items, device)
                if batched is None:
                    continue
                items = [items[i] for i in batched["valid_idx"]]
                B = len(items)
                if B == 0:
                    continue

                merged_videos = []; receiver_idxs = []
                gt_coords_list = []; gt_masks_list = []; pred_coords_list = []
                gt_indices = []; opt_logit_idx = []

                for b, item in enumerate(items):
                    video_b = slice_sample_video(batched, b)
                    N_b = video_b.shape[0]
                    T_b = int(batched["grid_thw"][b, 0].item())
                    S_b = N_b // max(1, T_b)
                    r_b = max(1, int(args.merge_ratio * N_b))

                    coords_b, scores_b = scorer.module(video_b, T_b, S_b)
                    merged_video, recv = gaze_weighted_merge(video_b, scores_b, r_b)
                    merged_videos.append(merged_video)
                    receiver_idxs.append(recv)
                    pred_coords_list.append(coords_b)

                    gt_c, gt_m = _build_gt_for_item(item, T_b, device)
                    gt_coords_list.append(gt_c)
                    gt_masks_list.append(gt_m)

                    n_opt = len(item["options"])
                    letters = [chr(65 + i) for i in range(n_opt)]
                    if item["answer"] not in letters:
                        gt_indices.append(-1); opt_logit_idx.append(None)
                    else:
                        gt_indices.append(letters.index(item["answer"]))
                        opt_logit_idx.append(option_ids_full[:n_opt])

                valid = [i for i, g in enumerate(gt_indices) if g >= 0]
                if not valid:
                    continue
                merged_videos = [merged_videos[i] for i in valid]
                receiver_idxs = [receiver_idxs[i] for i in valid]
                gt_keep    = [gt_indices[i]      for i in valid]
                opts_keep  = [opt_logit_idx[i]   for i in valid]
                pred_keep  = [pred_coords_list[i] for i in valid]
                gtc_keep   = [gt_coords_list[i]   for i in valid]
                gtm_keep   = [gt_masks_list[i]    for i in valid]

                if len(valid) != B:
                    items = [items[i] for i in valid]
                    batched = preprocess_batch(processor, base_qwen, items, device)
                    if batched is None:
                        continue
                    items = [items[i] for i in batched["valid_idx"]]
                    if len(items) == 0:
                        continue
                B = len(items)

                batched_inputs = build_merged_inputs_batch(
                    base_qwen, batched, merged_videos, receiver_idxs,
                )
                logits = forward_logits_batch(qwen_model, batched_inputs)  # (B, vocab)

                ce_losses = []
                for b in range(B):
                    oids = opts_keep[b]
                    logit_row = logits[b, oids]
                    gt = torch.tensor([gt_keep[b]], device=device, dtype=torch.long)
                    ce_losses.append(F.cross_entropy(logit_row.unsqueeze(0), gt))
                loss_ce = torch.stack(ce_losses).mean()

                traj_losses = []
                for pred, gtc, gtm in zip(pred_keep, gtc_keep, gtm_keep):
                    traj_losses.append(_traj_loss(pred, gtc, gtm))
                loss_traj = torch.stack(traj_losses).mean() if traj_losses else loss_ce.detach() * 0

                loss = loss_ce + args.lambda_traj * loss_traj
                loss_full = loss / args.grad_accum
                loss_full.backward()

                epoch_loss += loss.item()
                epoch_ce   += loss_ce.item()
                epoch_traj += loss_traj.item()
                micro_step += 1; accum_step += 1

                if accum_step >= args.grad_accum:
                    torch.nn.utils.clip_grad_norm_(lora_params + scorer_params, args.grad_clip)
                    optimizer.step(); optimizer.zero_grad()
                    accum_step = 0

                if is_main and micro_step % args.log_every == 0:
                    avg = epoch_loss / micro_step
                    avg_ce   = epoch_ce / micro_step
                    avg_traj = epoch_traj / micro_step
                    print(f"Epoch {epoch+1} | mstep {micro_step}/{len(loader)} | "
                          f"loss={avg:.4f} (ce={avg_ce:.4f} traj={avg_traj:.4f}) | "
                          f"t={time.time()-t_start:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"epoch": epoch+1, "step": micro_step,
                                             "loss": avg, "loss_ce": avg_ce, "loss_traj": avg_traj,
                                             "elapsed": time.time()-t_start}) + "\n")

                if is_main and args.eval_every > 0 and micro_step % args.eval_every == 0:
                    res = evaluate_per_source(
                        qwen_model.module, base_qwen, scorer.module, processor,
                        option_ids_full, device, args.merge_ratio,
                    )
                    line = "  → eval " + " | ".join(
                        f"{s}={res[s][0]:.2f}%(n={res[s][1]})" for s in ("sg","eg","hd","mean"))
                    print(line, flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"type": "eval", "epoch": epoch+1,
                                             "step": micro_step, "per_source": res}) + "\n")
                    mean_acc = res["mean"][0]
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        torch.save({
                            "epoch": epoch, "step": micro_step,
                            "lora_state": qwen_model.module.state_dict(),
                            "scorer_state": scorer.module.state_dict(),
                            "per_source_acc": res,
                            "mean_acc": mean_acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (mean={mean_acc:.2f}%)", flush=True)

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if accum_step > 0:
            torch.nn.utils.clip_grad_norm_(lora_params + scorer_params, args.grad_clip)
            optimizer.step(); optimizer.zero_grad()
            accum_step = 0

        if is_main:
            avg = epoch_loss / max(1, micro_step)
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg:.4f} "
                  f"| time={time.time()-t_start:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": qwen_model.module.state_dict(),
                "scorer_state": scorer.module.state_dict(),
                "loss": avg,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    if is_main:
        print(f"\nTraining complete. Best mean acc: {best_acc:.2f}%", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
