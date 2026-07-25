"""
TrajGazeMerge Stage 3 — Temporal + gaze/hand force.

Same as `train_merge_lora_temporal.py` (TrajGazeV2Temporal scores + soft
`gaze_weighted_merge` + KD against fine-tuned baseline LoRA), with one
additional rule: visual tokens that cover a valid gaze or hand position
(from the trajectory annotations) are *always* selected as receivers.
Implementation is a +1e6 score boost at those flat token positions
before the existing top-K sort inside `gaze_weighted_merge`, so the 10%
token budget is unchanged — the remaining slots fill from TrajGazeV2.

Score alignment pipeline:
  TrajGaze output : (T_traj, 196)  — 14×14 spatial, T_traj temporal
  Spatial rescale : (T_traj, 196) → (T_traj, n_spatial)
                    14×14 → 16×16 (nearest) → 8×8 (avg_pool) when n_spatial=64
  Temporal align  : (T_traj, n_spatial) → (T_merged, n_spatial)
                    linear interpolation along time axis (differentiable)
  Flatten         : (T_merged × n_spatial,) = (n_video,)

Gradient flow:
  loss → merged_tokens → gaze_weighted_merge → scores_all
       → score_to_qwen_spatiotemporal (differentiable)
       → TrajScoreHead → enriched_context → encoder cross-attn → tokenizer
       → LoRA (separate grad path via qwen_model)

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29811 \\
        -m TrajGazeMerge.training.train_merge_lora_temporal \\
        --stage1-ckpt  /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth \\
        --teacher-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \\
        --output-dir   /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_temporal \\
        --epochs 3 --merge-ratio 0.9 --alpha 0.5 --grad-accum 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.dataset  import StreamGazeMergeDataset
from TrajGazeMerge.models.merge  import gaze_weighted_merge
from TrajGazeMerge.models.gaze_hand_force import gaze_hand_force_mask
from TrajGazeMerge.models.model  import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
)
from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal

STAGE1_CKPT = "/workspace/checkpoints/stage1_temporal/best.pth"
OUTPUT_ROOT = "/workspace/checkpoints/merge_lora_temporal_gh_force"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",    default=STAGE1_CKPT)
    p.add_argument("--teacher-ckpt",   default=None)
    p.add_argument("--output-dir",     default=OUTPUT_ROOT)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr-lora",        type=float, default=1e-4)
    p.add_argument("--lr-enc",         type=float, default=1e-5)
    p.add_argument("--alpha",          type=float, default=0.5)
    p.add_argument("--merge-ratio",    type=float, default=0.9)
    p.add_argument("--grad-accum",     type=int,   default=4)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--log-every",      type=int,   default=20)
    p.add_argument("--eval-every",     type=int,   default=400)
    p.add_argument("--n-frames",       type=int,   default=128)
    p.add_argument("--n-traj-frames",  type=int,   default=128,
                   help="Trajectory frames — must match temporal model training")
    p.add_argument("--n-vis-keyframes",type=int,   default=16,
                   help="DINOv2 keyframes for visual encoder")
    p.add_argument("--resume-ckpt",    default=None)
    p.add_argument("--start-epoch",    type=int,   default=0)
    p.add_argument("--resume-step",    type=int,   default=0,
                   help="Skip this many steps at the start of start-epoch (already completed)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--gh-boost-scale", type=float, default=1e6,
                   help="Score boost added to gaze/hand-positioned tokens before "
                        "the gaze_weighted_merge sort, guaranteeing they end up "
                        "in the receiver set. Set 0 to disable.")
    p.add_argument("--gh-disable",     action="store_true",
                   help="Disable the gaze/hand force-include (A/B against baseline temporal).")
    p.add_argument("--single-gpu",     action="store_true",
                   help="Bypass DDP for smoke tests.")
    p.add_argument("--max-train-steps", type=int, default=0,
                   help=">0: stop training after this many micro-steps per rank.")
    return p.parse_args()


# ── DDP ───────────────────────────────────────────────────────────────────────

def setup_ddp(single_gpu: bool = False):
    if single_gpu:
        torch.cuda.set_device(0)
        return 0, 0, 1
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_teacher(teacher_ckpt: str, device):
    from TrajGazeMerge.models.model import load_qwen_lora, load_qwen_frozen
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        print(f"[Teacher] Loading fine-tuned LoRA from: {teacher_ckpt}")
        _, teacher = load_qwen_lora(device)
        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["lora_state"], strict=False)
        print("[Teacher] LoRA weights loaded.")
    else:
        print("[Teacher] No teacher ckpt, falling back to base pretrained Qwen.")
        _, teacher = load_qwen_frozen(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_traj_encoder(ckpt_path: str, device, n_vis_keyframes: int = 16) -> TrajGazeV2Temporal:
    model = TrajGazeV2Temporal(n_vis_keyframes=n_vis_keyframes).to(device)
    if os.path.exists(ckpt_path):
        ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
        state   = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[TrajEnc] Loaded {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[TrajEnc] WARNING: ckpt not found: {ckpt_path}, using random init")
    return model


# ── Score computation ─────────────────────────────────────────────────────────

def get_patch_scores_temporal(traj_encoder, item: dict, device) -> torch.Tensor:
    """
    Run TrajGazeV2Temporal to get (T_traj, 196) per-frame patch scores.
    Returns tensor with grad attached for backprop through merge.
    """
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    scores = traj_encoder.get_patch_scores(
        traj_batch,
        queries     = [item["question"]],
        frame_paths = [item["traj_frame_paths"]],
    )                           # (1, T_traj, 196)
    return scores.squeeze(0)   # (T_traj, 196) with grad


def apply_gh_force(
    scores_all: torch.Tensor,    # (n_video,) with grad
    traj:       dict,
    grid_thw:   torch.Tensor,
    n_video:    int,
    boost:      float,
    disabled:   bool,
) -> tuple[torch.Tensor, int]:
    """Add `boost` to scores at gaze/hand-positioned tokens; return (new_scores, n_forced)."""
    if disabled or boost <= 0.0:
        return scores_all, 0
    force = gaze_hand_force_mask(traj, grid_thw.cpu(), n_video).to(scores_all.device)
    # Operate in float so the boost dominates any per-frame score scale.
    scores_all = scores_all.float() + boost * force.to(scores_all.dtype).float()
    return scores_all, int(force.sum().item())


def score_to_qwen_spatiotemporal(
    scores:    torch.Tensor,   # (T_traj, 196)
    n_spatial: int,            # Qwen spatial tokens per frame
    T_merged:  int,            # Qwen temporal frames
) -> torch.Tensor:
    """
    Align per-frame TrajGaze scores to Qwen's video token layout.

    Steps:
      1. Spatial: 14×14 → Qwen spatial grid per frame
         Standard path (n_spatial=64): 14→16 nearest, 16→8 avg_pool
         Generic path: bilinear interpolation
      2. Temporal: T_traj → T_merged via linear interpolation
      3. Flatten: (T_merged, n_spatial) → (n_video,)

    Returns (T_merged * n_spatial,) differentiable scores.
    """
    T_traj = scores.shape[0]
    side   = int(n_spatial ** 0.5)

    # ── Step 1: spatial rescaling per frame ───────────────────────────────────
    s2d = scores.float().reshape(T_traj, 1, 14, 14)   # (T, 1, 14, 14)

    if side == 8:
        s16 = F.interpolate(s2d, size=(16, 16), mode="nearest")
        s8  = F.avg_pool2d(s16, kernel_size=2, stride=2)   # differentiable
        scores_spatial = s8.reshape(T_traj, n_spatial)      # (T_traj, 64)
    else:
        out = F.interpolate(s2d, size=(side, side),
                            mode="bilinear", align_corners=False)
        scores_spatial = out.reshape(T_traj, n_spatial)

    # ── Step 2: temporal alignment ────────────────────────────────────────────
    if T_traj != T_merged:
        # (T_traj, n_spatial) → (1, n_spatial, T_traj) → (1, n_spatial, T_merged)
        scores_spatial = F.interpolate(
            scores_spatial.T.unsqueeze(0).float(),
            size=T_merged, mode="linear", align_corners=False,
        ).squeeze(0).T                                      # (T_merged, n_spatial)

    return scores_spatial.reshape(-1)                       # (T_merged * n_spatial,)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(processor, qwen_model, base_qwen, traj_encoder,
             option_ids, device, merge_ratio, max_items=None,
             teacher_model=None, gh_boost: float = 0.0, gh_disable: bool = True):
    """Evaluate on egtea test split. Returns (merge_acc, full_acc, n, per_task_merge, per_task_full)."""
    test_ds = StreamGazeMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128,
    )
    if max_items is not None:
        test_ds.items = test_ds.items[:max_items]

    qwen_model.eval()
    traj_encoder.eval()
    correct_merge = correct_full = total = 0
    by_task_merge: dict[str, list] = {}
    by_task_full:  dict[str, list] = {}

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

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r         = max(1, int(merge_ratio * n_video))
                task      = item.get("task", "unknown")

                _teacher = teacher_model if teacher_model is not None else qwen_model
                logits_full = forward_logits(_teacher, build_full_inputs(base_qwen, cached))
                pred_full   = logits_full[option_ids].argmax().item()

                # Per-frame scores → flat video scores
                scores     = get_patch_scores_temporal(traj_encoder, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)

                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                        else scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]

                scores_all, _ = apply_gh_force(
                    scores_all, item["traj"], cached["grid_thw"], n_video,
                    boost=gh_boost, disabled=gh_disable,
                )

                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r,
                )
                logits_merge = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                )
                pred_merge = logits_merge[option_ids].argmax().item()

                gt_idx         = ["A", "B", "C", "D"].index(item["answer"])
                ok_merge = int(pred_merge == gt_idx)
                ok_full  = int(pred_full  == gt_idx)
                correct_full  += ok_full
                correct_merge += ok_merge
                total         += 1
                by_task_merge.setdefault(task, []).append(ok_merge)
                by_task_full.setdefault(task,  []).append(ok_full)
            except Exception:
                pass

    qwen_model.train()
    traj_encoder.train()

    per_task_merge = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task_merge.items())}
    per_task_full  = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task_full.items())}

    return (
        100.0 * correct_merge / max(1, total),
        100.0 * correct_full  / max(1, total),
        total,
        per_task_merge,
        per_task_full,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rank, local_rank, world_size = setup_ddp(args.single_gpu)
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[TrajGazeMergeTemporal+GH] output: {args.output_dir}")
        print(f"  GPUs={world_size}  epochs={args.epochs}  merge_ratio={args.merge_ratio}")
        print(f"  n_traj_frames={args.n_traj_frames}  n_vis_keyframes={args.n_vis_keyframes}")
        print(f"  teacher_ckpt={args.teacher_ckpt}")
        print(f"  gh_boost_scale={args.gh_boost_scale}  gh_disable={args.gh_disable}")

    # Models
    if is_main: print("Loading teacher ...")
    teacher_model = load_teacher(args.teacher_ckpt, device)
    if is_main: print("Loading TrajGaze temporal encoder ...")
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device, args.n_vis_keyframes)
    if not args.single_gpu:
        traj_encoder = DDP(traj_encoder, device_ids=[local_rank], find_unused_parameters=True)
    if is_main: print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    if not args.single_gpu:
        qwen_model = DDP(qwen_model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main: print("All models loaded.")

    # Module aliases (DDP-aware)
    traj_encoder_module = traj_encoder.module if hasattr(traj_encoder, "module") else traj_encoder

    # Resume
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        if is_main: print(f"[Resume] {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        (qwen_model.module if hasattr(qwen_model, "module") else qwen_model
         ).load_state_dict(ckpt["lora_state"], strict=False)
        traj_encoder_module.load_state_dict(ckpt["encoder_state"], strict=False)

    # Dataset
    train_ds = StreamGazeMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
    )
    if args.single_gpu:
        sampler = None
        loader  = DataLoader(train_ds, batch_size=1, shuffle=True,
                             collate_fn=lambda b: b[0], num_workers=2)
    else:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader  = DataLoader(train_ds, batch_size=1, sampler=sampler,
                             collate_fn=lambda b: b[0], num_workers=2)

    # Optimizer
    lora_params = [p for n, p in qwen_model.named_parameters()  if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer   = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        qwen_model.train()
        traj_encoder.train()
        optimizer.zero_grad()

        epoch_loss = epoch_ce = epoch_kl = 0.0
        epoch_forced = 0
        steps = 0
        t_start = time.time()

        skip_steps = args.resume_step if epoch == args.start_epoch else 0

        for step, item in enumerate(loader):
            if step < skip_steps:
                continue
            if item is None:
                continue
            if args.max_train_steps > 0 and steps >= args.max_train_steps:
                break
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r         = max(1, int(args.merge_ratio * n_video))
                gt_tensor = torch.tensor(
                    [["A","B","C","D"].index(item["answer"])], device=device
                )

                # Teacher: full tokens, frozen. Skip when alpha=0 (CE-only).
                if args.alpha > 0.0:
                    with torch.no_grad():
                        logits_teacher = forward_logits(
                            teacher_model, build_full_inputs(base_qwen, cached)
                        )[option_ids].detach()
                else:
                    logits_teacher = None

                # TrajGaze temporal scores: (T_traj, 196) → (n_video,)
                scores     = get_patch_scores_temporal(traj_encoder_module, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)

                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                        else scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]

                scores_all, n_forced = apply_gh_force(
                    scores_all, item["traj"], cached["grid_thw"], n_video,
                    boost=args.gh_boost_scale, disabled=args.gh_disable,
                )

                # Merge + student forward
                video_embeds_detached = cached["video_embeds"].detach()
                merged_video, receiver_idx = gaze_weighted_merge(
                    video_embeds_detached, scores_all, r,
                )
                logits_student = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                )[option_ids]

                # Loss
                loss_ce = F.cross_entropy(logits_student.unsqueeze(0), gt_tensor)
                if logits_teacher is not None:
                    loss_kl = F.kl_div(
                        F.log_softmax(logits_student, dim=-1),
                        F.softmax(logits_teacher,     dim=-1),
                        reduction="batchmean",
                    )
                    loss = (args.alpha * loss_kl + (1.0 - args.alpha) * loss_ce) / args.grad_accum
                else:
                    loss_kl = torch.tensor(0.0, device=device)
                    loss    = loss_ce / args.grad_accum
                loss.backward()

                epoch_loss += loss.item() * args.grad_accum
                epoch_ce   += loss_ce.item()
                epoch_kl   += loss_kl.item()
                epoch_forced += n_forced
                steps      += 1

                if steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and steps % args.log_every == 0:
                    avg_l, avg_ce, avg_kl = epoch_loss/steps, epoch_ce/steps, epoch_kl/steps
                    avg_force = epoch_forced / max(1, steps)
                    print(f"Epoch {epoch+1} | step {steps}/{len(loader)} | "
                          f"loss={avg_l:.4f} ce={avg_ce:.4f} kl={avg_kl:.4f} | "
                          f"gh_forced={avg_force:.1f} | t={time.time()-t_start:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": steps,
                            "loss": avg_l, "ce": avg_ce, "kl": avg_kl,
                            "gh_forced_avg": avg_force,
                        }) + "\n")

                if args.eval_every > 0 and steps % args.eval_every == 0:
                    if not args.single_gpu:
                        dist.barrier()
                    if is_main:
                        acc_m, acc_f, n_eval, pt_merge, pt_full = evaluate(
                            processor,
                            qwen_model.module if hasattr(qwen_model, "module") else qwen_model,
                            base_qwen, traj_encoder_module,
                            option_ids, device, args.merge_ratio,
                            teacher_model=teacher_model,
                            gh_boost=args.gh_boost_scale, gh_disable=args.gh_disable,
                        )
                        print(f"  → eval egtea: merge={acc_m:.2f}% full={acc_f:.2f}% (n={n_eval})")
                        for t in sorted(pt_merge):
                            print(f"     {t}: merge={pt_merge[t]:.2f}% full={pt_full[t]:.2f}%")
                        with open(log_path, "a") as f:
                            f.write(json.dumps({
                                "type": "eval",
                                "epoch": epoch+1, "step": steps,
                                "acc_merge": acc_m, "acc_full": acc_f, "n_eval": n_eval,
                                "per_task_merge": pt_merge, "per_task_full": pt_full,
                            }) + "\n")
                        if acc_m > best_acc:
                            best_acc = acc_m
                            qm_state = (qwen_model.module if hasattr(qwen_model, "module") else qwen_model).state_dict()
                            torch.save({
                                "epoch": epoch, "step": steps,
                                "lora_state":    qm_state,
                                "encoder_state": traj_encoder_module.state_dict(),
                                "acc_merge": acc_m, "acc_full": acc_f,
                            }, os.path.join(args.output_dir, "best.pth"))
                            print(f"  → saved best (merge={acc_m:.2f}%)")
                    if not args.single_gpu:
                        dist.barrier()

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Flush remaining gradients
        if steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_l = epoch_loss / max(1, steps)
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_l:.4f} | "
                  f"time={time.time()-t_start:.0f}s ===")
            qm_state = (qwen_model.module if hasattr(qwen_model, "module") else qwen_model).state_dict()
            torch.save({
                "epoch": epoch,
                "lora_state":    qm_state,
                "encoder_state": traj_encoder_module.state_dict(),
                "loss": avg_l,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    # Final eval
    if not args.single_gpu:
        dist.barrier()
    if is_main:
        acc_m, acc_f, n_eval, pt_merge, pt_full = evaluate(
            processor,
            qwen_model.module if hasattr(qwen_model, "module") else qwen_model,
            base_qwen, traj_encoder_module,
            option_ids, device, args.merge_ratio,
            teacher_model=teacher_model,
            gh_boost=args.gh_boost_scale, gh_disable=args.gh_disable,
        )
        print(f"\n[Final] egtea: merge={acc_m:.2f}%  full={acc_f:.2f}%  (n={n_eval})")
        for t in sorted(pt_merge):
            print(f"   {t}: merge={pt_merge[t]:.2f}% full={pt_full[t]:.2f}%")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "type": "eval_final",
                "acc_merge": acc_m, "acc_full": acc_f, "n_eval": n_eval,
                "per_task_merge": pt_merge, "per_task_full": pt_full,
            }) + "\n")
    if not args.single_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
