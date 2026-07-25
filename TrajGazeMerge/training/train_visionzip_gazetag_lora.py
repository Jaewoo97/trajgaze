"""OUR METHOD (variant 2) — per-token gaze/hand salience TAG on M1's selection.

Companion to the scanpath channel. Same philosophy (use gaze/hand trajectory),
different mechanism: instead of adding sequence-level intent tokens, keep M1's
exact selection and ADD a learned salience embedding to each retained visual
token (GazeTagEmbedding) — annotating which patches the person looked at /
manipulated. Additive (no token added/removed) -> escapes the zero-sum.

Selection is M1's top-k complement, replicated inline here from a single TAS
trajectory-score pass (so we don't run the encoder twice): 7% VisionZip content
(half dominant / half contextual) ∪ 3% top-k trajectory complement among the
non-content tokens. The same trajectory score, min-max normalized per video,
also drives the per-token tag. Trained jointly with LoRA on the M1 protocol.

Question: does M1-selection + per-token gaze tag beat M1-selection alone (63.01)?

Usage (4-GPU DDP):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29663 \\
      -m TrajGazeMerge.training.train_visionzip_gazetag_lora \\
      --stage1-ckpt .../stage1_tas_3way_overlay/best.pth \\
      --output-dir  .../checkpoints/ours_gazetag \\
      --epochs 3 --lr 1e-4 --tag-lr 1e-3 --grad-accum 2 \\
      --no-hdepic --early-stop --no-mid-eval
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
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits
from TrajGazeMerge.models.gaze_tag import GazeTagEmbedding
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import (
    _traj_scores, _norm_scores, setup_ddp, STAGE1_DEFAULT,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_gazetag")
    p.add_argument("--traj-pool-mode", choices=["learned", "anticipatory"], default="learned")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio",    type=float, default=0.03)
    p.add_argument("--stage1-ckpt",   default=STAGE1_DEFAULT)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--tag-bins",  type=int, default=16, help="salience quantization bins.")
    p.add_argument("--tag-norm",  choices=["minmax", "rank"], default="minmax")
    p.add_argument("--tag-lr",    type=float, default=1e-3, help="LR for the fresh tag embedding.")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=2)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--no-mid-eval", action="store_true")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def select_m1_with_salience(cached, item, device, mode, encoder, hp,
                            content_ratio, traj_ratio, tag_norm):
    """Replicate M1's top-k complement selection from a SINGLE trajectory-score
    pass, and return (sel_embeds, recv_idx, sal_kept01) where sal_kept01 is the
    per-token normalized salience for the kept tokens (for the tag). Mirrors
    train_visionzip_complement_lora.select_complementary(complement_mode='topk').
    All under no_grad (frozen ViT + frozen TAS encoder)."""
    video_embeds = cached["video_embeds"]
    attn_scores  = cached["attn_scores"]
    attn_key     = cached["attn_key"]
    N = video_embeds.shape[0]

    traj_scores = _traj_scores(cached, item, device, mode, encoder, hp).to(video_embeds.device)

    half = content_ratio / 2.0
    content_embeds, content_idx = visionzip_select_tokens(
        video_embeds, attn_scores, attn_key, dominant_ratio=half, contextual_ratio=half)

    avail_mask = torch.ones(N, dtype=torch.bool, device=video_embeds.device)
    avail_mask[content_idx] = False
    avail = avail_mask.nonzero(as_tuple=True)[0]
    k_traj = min(max(1, int(traj_ratio * N)), avail.numel())

    if k_traj > 0 and avail.numel() > 0:
        top = torch.topk(traj_scores[avail], k_traj).indices
        traj_idx = avail[top]
        all_embeds = torch.cat([content_embeds, video_embeds[traj_idx]], dim=0)
        all_idx = torch.cat([content_idx, traj_idx])
    else:
        all_embeds, all_idx = content_embeds, content_idx

    order = all_idx.argsort()
    sel_embeds, recv_idx = all_embeds[order], all_idx[order]

    sal01 = _norm_scores(traj_scores, tag_norm)
    sal_kept = sal01[recv_idx]
    return sel_embeds, recv_idx, sal_kept


def evaluate(processor, model, base_qwen, gaze_tag, option_ids, device,
             mode, encoder, hp, content_ratio, traj_ratio, tag_norm, include_hdepic=True):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic)
    model.eval(); gaze_tag.eval()
    correct = 0; total = 0
    by_task: dict[str, list] = {}
    with torch.no_grad():
        for item in test_ds:
            if item is None: continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device)
                if cached is None: continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                sel, idx, sal_kept = select_m1_with_salience(
                    cached, item, device, mode, encoder, hp,
                    content_ratio, traj_ratio, tag_norm)
                sel = sel + gaze_tag(sal_kept).to(sel.dtype)
                inputs = build_merged_inputs(base_qwen, cached, sel, idx)
                logits = forward_logits(model, inputs)
                pred = logits[option_ids[:n_opt]].argmax().item()
                ok = int(pred == letters.index(item["answer"]))
                correct += ok; total += 1
                by_task.setdefault(item["task"], []).append(ok)
            except Exception:
                pass
    model.train(); gaze_tag.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0,
              alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[gazetag] output: {args.output_dir}", flush=True)
        print(f"[gazetag] M1 selection (topk complement, mode={args.traj_pool_mode}, "
              f"content={args.content_ratio*100:.1f}% ∪ traj={args.traj_ratio*100:.1f}%) "
              f"+ per-token tag bins={args.tag_bins} norm={args.tag_norm}", flush=True)
        print(f"[gazetag] GPUs={world_size}, epochs={args.epochs}, lora_lr={args.lr}, "
              f"tag_lr={args.tag_lr}, grad_accum={args.grad_accum}, "
              f"eff_batch={args.grad_accum*world_size}", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    hidden = base_qwen.config.hidden_size
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    gaze_tag = GazeTagEmbedding(out_dim=hidden, n_bins=args.tag_bins).to(device)
    gaze_tag = DDP(gaze_tag, device_ids=[local_rank])
    if is_main:
        n_tag = sum(p.numel() for p in gaze_tag.parameters())
        print(f"Model loaded. GazeTagEmbedding params: {n_tag/1e3:.1f}K", flush=True)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    tag_params = list(gaze_tag.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": tag_params, "lr": args.tag_lr},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0
    epoch_accs: list[float] = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train(); gaze_tag.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None: continue
            try:
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device)
                if cached is None: continue
                n_video = cached["video_embeds"].shape[0]
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                with torch.no_grad():
                    sel, idx, sal_kept = select_m1_with_salience(
                        cached, item, device, args.traj_pool_mode, encoder, hp,
                        args.content_ratio, args.traj_ratio, args.tag_norm)

                # tag + input build OUTSIDE no_grad (trainable path)
                sel_tagged = sel + gaze_tag(sal_kept).to(sel.dtype)
                inputs = build_merged_inputs(base_qwen, cached, sel_tagged, idx)
                logits = forward_logits(model, inputs)
                gt = letters.index(item["answer"])
                loss = F.cross_entropy(logits[option_ids[:n_opt]].unsqueeze(0),
                                       torch.tensor([gt], device=device))
                loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + tag_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    pct_kept = 100.0 * idx.shape[0] / max(1, n_video)
                    gate = float(gaze_tag.module.gate.detach().item())
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | gate={gate:.3f} | "
                          f"t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps, "loss": avg_loss,
                            "pct_kept": pct_kept, "gate": gate, "elapsed": elapsed}) + "\n")
            except Exception:
                if is_main: traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + tag_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed = time.time() - t_start
        tag_cfg = dict(out_dim=hidden, n_bins=args.tag_bins)
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} "
                  f"| time={elapsed:.0f}s ===", flush=True)
            torch.save({"epoch": epoch+1, "lora_state": model.module.state_dict(),
                        "tag_state": gaze_tag.module.state_dict(), "tag_cfg": tag_cfg,
                        "loss": avg_loss},
                       os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} on full {label} val set ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, gaze_tag.module, option_ids, device,
                args.traj_pool_mode, encoder, hp, args.content_ratio, args.traj_ratio,
                args.tag_norm, include_hdepic=args.include_hdepic)
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch+1, "eval_acc": acc,
                                    "n_eval": n_eval, "per_task": per_task}) + "\n")
            epoch_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save({"epoch": epoch+1, "lora_state": model.module.state_dict(),
                            "tag_state": gaze_tag.module.state_dict(), "tag_cfg": tag_cfg,
                            "acc": acc}, os.path.join(args.output_dir, "best.pth"))
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
