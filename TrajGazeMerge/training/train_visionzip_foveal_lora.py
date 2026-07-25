"""Foveated ROI (④) — gaze-guided high-resolution tokens on top of M1 selection.

Forked from train_visionzip_complement_lora.py. Token SELECTION is unchanged (M1 =
7% content ∪ 3% gaze top-k); this trainer additionally crops the gaze-fixated region
of a few frames, re-encodes the crop at HIGH resolution through the SAME Qwen ViT, and
injects K foveal tokens after the video block (models/foveal_roi.py). Unlike every
selection variant (all falsified at the 10% budget), this adds *new pixels* — detail on
small interaction targets that no token reshuffle can supply.

The contribution is proven against an ATTENTION-TWIN control, not against VZ alone
(see GAZE_EXCLUSIVE_PERFORMANCE_STRATEGY.md §3). Three (+1) arms, switched by --roi-arm:

    gaze    crop centered on gaze fixations            (the method)
    attn    crop centered on VisionZip attention peaks (content-only twin = control)
    random  crop at random frames/points               (placebo)
    none    no foveal tokens                            (== M1 baseline exactly)

Contribution = gaze-ROI > attn-ROI on the object-ID/attribute subset (gaze is
load-bearing in resolution allocation). V1 = 'added' regime: 10% tokens kept intact +
K foveal appended. All ROI knobs are --roi-* flags → FovealROIConfig (sweepable later).

Usage (2-GPU DDP); run all arms with the SAME selection so only the ROI source differs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29720 \\
      -m TrajGazeMerge.training.train_visionzip_foveal_lora \\
      --traj-pool-mode learned --complement-mode topk \\
      --stage1-ckpt .../stage1_tas_3way_overlay/best.pth \\
      --roi-arm gaze \\
      --output-dir .../foveal_gaze --epochs 3 --lr 1e-4 --grad-accum 4 \\
      --no-hdepic --early-stop --no-mid-eval
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
import zlib

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import get_option_ids, forward_logits
from TrajGazeMerge.models.traj_weights import _solve_spatial_dims
from TrajGazeMerge.models.foveal_roi import (
    FovealROIConfig, roi_config_from_args,
    detect_fixation_frames, detect_attn_frames, detect_random_frames,
    detect_anticipatory_frames,
    crops_from_fixations, encode_foveal_tokens, build_inputs_with_foveal,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder
# Selection is identical to M1 — reuse it verbatim so the only change vs the
# complement trainer is the foveal injection (controls share the selection too).
from TrajGazeMerge.training.train_visionzip_complement_lora import (
    setup_ddp, select_complementary, STAGE1_DEFAULT,
)

ROI_ARMS = ("gaze", "attn", "random", "none", "anticipatory")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/foveal_gaze")
    # ── ROI arm + knobs (FovealROIConfig fields; roi_config_from_args reads --roi-*) ──
    p.add_argument("--roi-arm", choices=ROI_ARMS, default="gaze",
                   help="ROI crop source: 'gaze' (current fixation) | 'attn' (attention-twin "
                        "control) | 'random' (placebo) | 'none' (== M1 baseline) | "
                        "'anticipatory' (① crop the current frame at the future fixation "
                        "gaze[t+Δ] — attention can't see where gaze goes next).")
    p.add_argument("--roi-antic_horizon", type=int, default=12,
                   help="① anticipatory: lookahead Δ (gaze-trajectory steps) for gaze[t+Δ].")
    p.add_argument("--roi-n_fix_frames",   type=int,   default=4)
    p.add_argument("--roi-crop_frac",      type=float, default=0.25)
    p.add_argument("--roi-margin_frac",    type=float, default=0.05)
    p.add_argument("--roi-max_crop_frac",  type=float, default=0.50)
    p.add_argument("--roi-foveal_k",       type=int,   default=32)
    p.add_argument("--roi-roi_max_pixels", type=int,   default=256 * 28 * 28)
    p.add_argument("--roi-min_frame_gap",  type=float, default=0.08)
    p.add_argument("--roi-include_hand",   dest="roi_include_hand", action="store_true")
    p.add_argument("--no-roi-include_hand", dest="roi_include_hand", action="store_false")
    p.set_defaults(roi_include_hand=True)
    p.add_argument("--roi-pool", choices=["uniform", "topk_attn", "all"], default="uniform")
    p.add_argument("--roi-mode", choices=["added", "replace"], default="added",
                   help="'added' (V1): keep 10%% selection + K foveal tokens. "
                        "'replace' (V2): reserved (reduce gaze complement instead).")
    # ── Selection (identical to M1 / complement trainer) ──
    p.add_argument("--traj-pool-mode", choices=["learned", "anticipatory"], default="learned")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio",    type=float, default=0.03)
    p.add_argument("--query-ratio",   type=float, default=0.0)
    p.add_argument("--query-mode",    choices=["cosine", "random", "shuffle"], default="cosine")
    p.add_argument("--complement-mode", choices=["topk", "coverage", "fusion", "query_gaze"],
                   default="topk")
    p.add_argument("--nms-radius", type=int, default=1)
    p.add_argument("--fusion-lambda", type=float, default=1.0)
    p.add_argument("--fusion-norm", choices=["minmax", "rank"], default="minmax")
    p.add_argument("--stage1-ckpt",   default=STAGE1_DEFAULT)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--horizon",       type=float, default=2.0)
    # ── Training protocol (identical to M1) ──
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--sigma-g",    type=float, default=2.0)
    p.add_argument("--sigma-h",    type=float, default=3.0)
    p.add_argument("--alpha-hand", type=float, default=0.7)
    p.add_argument("--sigma-v",    type=float, default=0.05)
    p.add_argument("--sigma-gh",   type=float, default=0.10)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="2-way: StreamGaze + EgoGazeVQA only (exclude HD-EPIC).")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--no-mid-eval", action="store_true")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def compute_foveal_embeds(processor, base_qwen, cached, item, cfg: FovealROIConfig,
                          arm: str, device):
    """Build K foveal ROI tokens for one item under the given control arm. Returns a
    (K, d) tensor in LLM embedding space, or None (→ build_inputs_with_foveal falls back
    to the plain M1 merged build). All crop/encode work runs under the frozen ViT
    (no_grad inside encode_foveal_tokens); LoRA adapts via the LLM forward only."""
    if arm == "none":
        return None
    frame_paths = item.get("vlm_frame_paths") or []
    if not frame_paths:
        return None
    n_frames = len(frame_paths)

    if arm == "gaze":
        fix = detect_fixation_frames(item.get("traj"), n_frames, cfg)
    elif arm == "anticipatory":
        fix = detect_anticipatory_frames(item.get("traj"), n_frames, cfg)
    elif arm == "random":
        # deterministic per-item seed (stable across processes, unlike hash())
        seed = zlib.crc32(frame_paths[0].encode()) & 0x7fffffff
        fix = detect_random_frames(n_frames, cfg, seed=seed)
    elif arm == "attn":
        grid = cached["grid_thw"]
        t_merged = int(grid[0, 0].item())
        N = cached["video_embeds"].shape[0]
        n_spatial = N // max(1, t_merged)
        if t_merged * n_spatial != N:
            return None
        s_h, s_w = _solve_spatial_dims(n_spatial, int(grid[0, 1].item()), int(grid[0, 2].item()))
        if s_h * s_w != n_spatial:
            return None
        fix = detect_attn_frames(cached["attn_scores"], t_merged, s_h, s_w, n_frames, cfg)
    else:
        return None

    if not fix:
        return None
    crops = crops_from_fixations(frame_paths, fix, cfg)
    if not crops:
        return None
    return encode_foveal_tokens(processor, base_qwen, crops, cfg, device)


def evaluate(processor, model, base_qwen, option_ids, device, mode, encoder, hp,
             content_ratio, traj_ratio, cfg, roi_arm, include_hdepic=True,
             complement_mode="topk", nms_radius=1,
             fusion_lambda=1.0, fusion_norm="minmax",
             query_ratio=0.0, query_mode="cosine"):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic,
    )
    model.eval()
    correct = 0; total = 0; n_foveal = 0
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

                sel_embeds, recv_idx = select_complementary(
                    cached, item, device, mode, encoder, hp, content_ratio, traj_ratio,
                    complement_mode=complement_mode, nms_radius=nms_radius,
                    fusion_lambda=fusion_lambda, fusion_norm=fusion_norm,
                    query_ratio=query_ratio, query_mode=query_mode)
                foveal = compute_foveal_embeds(processor, base_qwen, cached, item, cfg, roi_arm, device)
                if foveal is not None:
                    n_foveal += 1
                inputs_dict = build_inputs_with_foveal(base_qwen, cached, sel_embeds, recv_idx, foveal)
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
    return 100.0 * correct / max(1, total), total, per_task, n_foveal


def main():
    args = parse_args()
    cfg = roi_config_from_args(args)
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    hp = dict(
        horizon=args.horizon, sigma_g=args.sigma_g, sigma_h=args.sigma_h,
        alpha_hand=args.alpha_hand, sigma_v=args.sigma_v, sigma_gh=args.sigma_gh,
    )

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[foveal-ROI] output: {args.output_dir}")
        print(f"[foveal-ROI] arm={args.roi_arm}  {cfg.describe()}", flush=True)
        print(f"[foveal-ROI] selection mode={args.traj_pool_mode}, GPUs={world_size}, "
              f"content={args.content_ratio*100:.1f}% ∪ traj={args.traj_ratio*100:.1f}% = "
              f"{(args.content_ratio+args.traj_ratio)*100:.0f}% (complement={args.complement_mode}), "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}", flush=True)
        if args.traj_pool_mode == "learned":
            print(f"[foveal-ROI] learned encoder: {args.stage1_ckpt}", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    encoder = None
    if args.traj_pool_mode == "learned":
        encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    if is_main: print("Model loaded.", flush=True)

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
        epoch_loss = 0.0; n_steps = 0; n_foveal = 0
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

                with torch.no_grad():
                    sel_embeds, recv_idx = select_complementary(
                        cached, item, device, args.traj_pool_mode, encoder, hp,
                        args.content_ratio, args.traj_ratio,
                        complement_mode=args.complement_mode, nms_radius=args.nms_radius,
                        fusion_lambda=args.fusion_lambda, fusion_norm=args.fusion_norm,
                        query_ratio=args.query_ratio, query_mode=args.query_mode)
                    foveal = compute_foveal_embeds(
                        processor, base_qwen, cached, item, cfg, args.roi_arm, device)
                    inputs_dict = build_inputs_with_foveal(
                        base_qwen, cached, sel_embeds, recv_idx, foveal)
                if foveal is not None:
                    n_foveal += 1

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue
                logits = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                loss = F.cross_entropy(option_logits.unsqueeze(0),
                                       torch.tensor([gt_idx], device=device))
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
                    fk = foveal.shape[0] if foveal is not None else 0
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}%+{fk}fov | "
                          f"fov_hit={n_foveal}/{n_steps} | t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps, "loss": avg_loss,
                            "pct_kept": pct_kept, "foveal_k": fk,
                            "foveal_hit": n_foveal, "elapsed": elapsed,
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
                  f"| foveal_hit={n_foveal}/{n_steps} | time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
                "roi_arm": args.roi_arm,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} on full {label} val set (arm={args.roi_arm}) ...",
                  flush=True)
            acc, n_eval, per_task, ev_fov = evaluate(
                processor, model.module, base_qwen, option_ids, device,
                args.traj_pool_mode, encoder, hp, args.content_ratio, args.traj_ratio,
                cfg, args.roi_arm, include_hdepic=args.include_hdepic,
                complement_mode=args.complement_mode, nms_radius=args.nms_radius,
                fusion_lambda=args.fusion_lambda, fusion_norm=args.fusion_norm,
                query_ratio=args.query_ratio, query_mode=args.query_mode,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval}, foveal_hit={ev_fov}/{n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc, "n_eval": n_eval,
                    "foveal_hit": ev_fov, "roi_arm": args.roi_arm, "per_task": per_task,
                }) + "\n")
            epoch_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                    "roi_arm": args.roi_arm,
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
