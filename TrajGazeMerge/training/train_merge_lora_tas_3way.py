"""
TAS Stage-2 trainer — 4-GPU DDP variant on the 3-way combined dataset.

Port of cf-mask branch's `train_merge_lora_batched.py` with:
  - DDP wrapping (mirrors train_merge_lora_no_kd_anchored.py setup)
  - CombinedMergeDataset (StreamGaze + EgoGazeVQA + HD-EPIC) instead of the
    cf-mask branch's missing `dataset_combined.build_combined_train_dataset()`
  - TAS-only: dropped --shuffle-aug / --use-atr / --cgm-aug / --use-cf-mask
    flags and their auxiliary losses (loss = CE on answer letter only)
  - N-option (A–E) MCQ support, like the other trainers in this repo
  - Per-source eval (sg / eg / hd) using CombinedMergeDataset(split="test")

Selection rule (TAS):
  encoder(TrajGazeV2Temporal) → (B,T,196) per-frame scores
  → score_to_qwen_spatiotemporal → (N_video,)
  → gaze_weighted_merge → 10% receivers + soft-merged sources

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29721 \\
      -m TrajGazeMerge.training.train_merge_lora_tas_3way \\
      --model-type full \\
      --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_trajanchor/best.pth \\
      --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tas_lora_3way \\
      --epochs 3 --merge-ratio 0.9 \\
      --micro-batch 2 --grad-accum 4 \\
      --dataloader-num-workers 8 --eval-every 400
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
from TrajGazeMerge.models.model import load_qwen_lora, get_option_ids
from TrajGazeMerge.models.model_batched import (
    preprocess_batch, slice_sample_video,
    build_merged_inputs_batch, forward_logits_batch,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal, score_to_qwen_spatiotemporal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",      choices=["full", "gaze_only", "hand_only"], required=True)
    p.add_argument("--stage1-ckpt",     required=True)
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--lr-lora",         type=float, default=1e-4)
    p.add_argument("--lr-enc",          type=float, default=1e-5)
    p.add_argument("--merge-ratio",     type=float, default=0.9)
    p.add_argument("--micro-batch",     type=int,   default=2)
    p.add_argument("--grad-accum",      type=int,   default=4)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--log-every",       type=int,   default=20)
    p.add_argument("--eval-every",      type=int,   default=400)
    p.add_argument("--n-frames",        type=int,   default=128)
    p.add_argument("--n-traj-frames",   type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int,   default=16)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--dataloader-num-workers", type=int, default=8)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="Train/eval on StreamGaze+EgoGazeVQA only (drop HD-EPIC); "
                        "val then = SG egtea + EG egtea.")
    p.set_defaults(include_hdepic=True)
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if egtea val did not improve over epoch 1.")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _filter_valid(items):
    return [x for x in items if x is not None]


@torch.no_grad()
def evaluate_per_source(qwen_model, base_qwen, traj_encoder, processor,
                         option_ids_full, device, merge_ratio, max_items=200,
                         include_hdepic=True):
    """Per-source SG/EG(/HD) eval on CombinedMergeDataset(test). Small max_items =
    cheap mid-train progress signal; max_items=inf = full end-of-epoch egtea val.
    Returns per-source acc, source-mean, and pooled 'overall'."""
    qwen_model.eval(); traj_encoder.eval()
    test_ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                                     include_hdepic=include_hdepic)
    sources = ["sg", "eg"] + (["hd"] if include_hdepic else [])
    # Sample max_items per source for balanced subset
    by_src = {s: [] for s in sources}
    for i, (s, _) in enumerate(test_ds.items):
        if s in by_src:
            by_src[s].append(i)
    pick = []
    for s in sources:
        pick.extend(by_src[s][:max_items])
    stats = {s: {"c": 0, "n": 0} for s in sources}
    for idx in pick:
        src, _ = test_ds.items[idx]
        try:
            item = test_ds[idx]
            if item is None:
                continue
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters:
                continue
            oids = option_ids_full[:n_opt]
            batched = preprocess_batch(processor, base_qwen, [item], device)
            if batched is None or len(batched.get("valid_idx", [])) == 0:
                continue
            video_b = slice_sample_video(batched, 0)
            N_b = video_b.shape[0]
            T_merged_b = int(batched["grid_thw"][0, 0].item())
            n_spatial_b = N_b // max(1, T_merged_b)
            r_b = max(1, int(merge_ratio * N_b))
            scores = get_patch_scores_temporal(traj_encoder, item, device)
            scores_all = score_to_qwen_spatiotemporal(scores, n_spatial_b, T_merged_b)
            if scores_all.shape[0] != N_b:
                scores_all = (scores_all[:N_b] if scores_all.shape[0] > N_b
                              else scores_all.repeat((N_b + scores_all.shape[0] - 1) // scores_all.shape[0])[:N_b])
            merged_video, recv = gaze_weighted_merge(video_b, scores_all, r_b)
            batched_inputs = build_merged_inputs_batch(base_qwen, batched, [merged_video], [recv])
            logits = forward_logits_batch(qwen_model, batched_inputs)[0]   # (vocab,)
            pred = logits[oids].argmax().item()
            gt = letters.index(item["answer"])
            stats[src]["c"] += int(pred == gt); stats[src]["n"] += 1
        except Exception:
            continue
    qwen_model.train(); traj_encoder.train()
    out = {}
    for s in sources:
        n = max(1, stats[s]["n"])
        out[s] = (100.0 * stats[s]["c"] / n, stats[s]["n"])
    accs = [out[s][0] for s in sources]
    out["mean"] = (sum(accs) / len(accs), sum(stats[s]["n"] for s in sources))
    tot_c = sum(stats[s]["c"] for s in sources)
    tot_n = sum(stats[s]["n"] for s in sources)
    out["overall"] = (100.0 * tot_c / max(1, tot_n), tot_n)
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[TAS-3way] world_size={world_size} micro_batch={args.micro_batch} "
              f"grad_accum={args.grad_accum} effective_batch={args.micro_batch * args.grad_accum * world_size}")
        print(f"  stage1_ckpt: {args.stage1_ckpt}")
        print(f"  output_dir:  {args.output_dir}", flush=True)

    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    option_ids_full = get_option_ids(processor, 5)   # A–E pool

    traj_encoder = load_traj_encoder(args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes)

    qwen_model   = DDP(qwen_model,   device_ids=[local_rank], find_unused_parameters=True)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        print("Models loaded + wrapped in DDP.", flush=True)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
        include_hdepic=args.include_hdepic,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=args.micro_batch, sampler=sampler,
                         collate_fn=lambda b: b, num_workers=args.dataloader_num_workers)

    lora_params = [p for p in qwen_model.parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0            # mid-epoch subset-mean best (only when --eval-every > 0)
    best_egtea_acc = 0.0      # end-of-epoch full-egtea best → best.pth
    epoch_accs: list[float] = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        qwen_model.train(); traj_encoder.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; micro_step = 0; accum_step = 0
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

                merged_videos = []; receiver_idxs = []; gt_indices = []; opt_logit_idx = []
                for b, item in enumerate(items):
                    video_b = slice_sample_video(batched, b)
                    N_b = video_b.shape[0]
                    T_merged_b = int(batched["grid_thw"][b, 0].item())
                    n_spatial_b = N_b // max(1, T_merged_b)
                    r_b = max(1, int(args.merge_ratio * N_b))

                    scores = get_patch_scores_temporal(traj_encoder.module, item, device)
                    scores_all = score_to_qwen_spatiotemporal(scores, n_spatial_b, T_merged_b)
                    if scores_all.shape[0] != N_b:
                        scores_all = (scores_all[:N_b] if scores_all.shape[0] > N_b
                                      else scores_all.repeat((N_b + scores_all.shape[0] - 1) // scores_all.shape[0])[:N_b])

                    merged_video, receiver_idx = gaze_weighted_merge(video_b, scores_all, r_b)
                    merged_videos.append(merged_video)
                    receiver_idxs.append(receiver_idx)

                    n_opt = len(item["options"])
                    letters = [chr(65 + i) for i in range(n_opt)]
                    if item["answer"] not in letters:
                        # mark for skip below
                        gt_indices.append(-1); opt_logit_idx.append(None)
                    else:
                        gt_indices.append(letters.index(item["answer"]))
                        opt_logit_idx.append(option_ids_full[:n_opt])

                # Filter out items with bad answers
                valid = [i for i, g in enumerate(gt_indices) if g >= 0]
                if not valid:
                    continue
                merged_videos = [merged_videos[i] for i in valid]
                receiver_idxs = [receiver_idxs[i] for i in valid]
                gt_keep = [gt_indices[i] for i in valid]
                opts_keep = [opt_logit_idx[i] for i in valid]

                # Rebuild batched dict to keep only valid items
                # Easiest: re-run preprocess_batch with the filtered item list
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
                logits = forward_logits_batch(qwen_model, batched_inputs)   # (B, vocab)

                # Per-sample CE with per-sample n_opt
                losses = []
                for b in range(B):
                    oids = opts_keep[b]
                    logit_row = logits[b, oids]
                    gt = torch.tensor([gt_keep[b]], device=device, dtype=torch.long)
                    losses.append(F.cross_entropy(logit_row.unsqueeze(0), gt))
                loss_ce = torch.stack(losses).mean()
                loss = loss_ce / args.grad_accum
                loss.backward()
                epoch_loss += loss_ce.item()
                micro_step += 1; accum_step += 1

                if accum_step >= args.grad_accum:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_step = 0

                if is_main and micro_step % args.log_every == 0:
                    avg = epoch_loss / micro_step
                    print(f"Epoch {epoch+1} | mstep {micro_step}/{len(loader)} | "
                          f"loss={avg:.4f} | t={time.time()-t_start:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"epoch": epoch+1, "step": micro_step,
                                             "loss": avg,
                                             "elapsed": time.time()-t_start}) + "\n")

                if is_main and args.eval_every > 0 and micro_step % args.eval_every == 0:
                    res = evaluate_per_source(
                        qwen_model.module, base_qwen, traj_encoder.module, processor,
                        option_ids_full, device, args.merge_ratio,
                        include_hdepic=args.include_hdepic,
                    )
                    line = "  → eval " + " | ".join(
                        f"{s}={res[s][0]:.2f}%(n={res[s][1]})"
                        for s in ("sg", "eg", "hd", "mean", "overall") if s in res)
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
                            "encoder_state": traj_encoder.module.state_dict(),
                            "per_source_acc": res,
                            "mean_acc": mean_acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (mean={mean_acc:.2f}%)", flush=True)

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # End of epoch
        if accum_step > 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            accum_step = 0
        if is_main:
            avg = epoch_loss / max(1, micro_step)
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg:.4f} "
                  f"| time={time.time()-t_start:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": qwen_model.module.state_dict(),
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        # End-of-epoch full egtea val + early-stop (mirrors VisionZip trainer).
        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            res = evaluate_per_source(
                qwen_model.module, base_qwen, traj_encoder.module, processor,
                option_ids_full, device, args.merge_ratio,
                max_items=10**9, include_hdepic=args.include_hdepic,
            )
            ev_acc = res["overall"][0]
            print(f"=== Epoch {epoch+1} egtea val: overall={ev_acc:.2f}% "
                  f"(n={res['overall'][1]}) | " +
                  " | ".join(f"{s}={res[s][0]:.2f}%(n={res[s][1]})"
                             for s in res if s not in ("mean", "overall")) +
                  " ===", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"type": "epoch_eval", "epoch": epoch + 1,
                                     "overall": ev_acc, "per_source": res}) + "\n")
            epoch_accs.append(ev_acc)
            if ev_acc > best_egtea_acc:
                best_egtea_acc = ev_acc
                torch.save({
                    "epoch": epoch + 1,
                    "lora_state": qwen_model.module.state_dict(),
                    "encoder_state": traj_encoder.module.state_dict(),
                    "egtea_acc": ev_acc, "per_source_acc": res,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (egtea={ev_acc:.2f}%)", flush=True)
            if args.early_stop and (epoch + 1) == 2 and len(epoch_accs) >= 2 \
                    and epoch_accs[1] <= epoch_accs[0]:
                stop[0] = 1.0
                print(f"  Early stop: epoch2 {epoch_accs[1]:.2f}% <= epoch1 "
                      f"{epoch_accs[0]:.2f}% → skipping epoch 3.", flush=True)
        dist.broadcast(stop, src=0)
        if stop.item() > 0:
            break

    if is_main:
        print(f"\nTraining complete. Best egtea acc: {best_egtea_acc:.2f}%")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
