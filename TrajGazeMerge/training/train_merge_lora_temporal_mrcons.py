"""
TrajGazeMerge Stage 3 — Temporal + Multi-Ratio Consistency (mr-cons).

Self-distillation variant: no external teacher needed.
Two student forwards per step at different keep ratios:
  Primary  : merge-ratio=0.9  → keep 10% tokens  (trained)
  Auxiliary: mr-cons-keep=0.15 → keep 15% tokens  (detached anchor)

Total loss:
  loss = (1-alpha)*CE + alpha*KL(student||teacher)   [alpha=0 → CE only]
       + mr-cons-weight * KL(primary || stop_grad(auxiliary))

Setting --alpha 0.0 --mr-cons-weight 0.5 --mr-cons-keep 0.15 gives
pure self-distillation with no external teacher loaded.

Usage:
    # 2-GPU, self-distill only (no external teacher):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29815 \\
        -m TrajGazeMerge.training.train_merge_lora_temporal_mrcons \\
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth \\
        --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_mrcons \\
        --epochs 3 --alpha 0.0 --merge-ratio 0.9 --grad-accum 4 \\
        --mr-cons-weight 0.5 --mr-cons-keep 0.15 --mr-cons-mode kl_to_anchor

    # 2-GPU, self-distill + external KD:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29815 \\
        -m TrajGazeMerge.training.train_merge_lora_temporal_mrcons \\
        --stage1-ckpt  /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth \\
        --teacher-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \\
        --output-dir   /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_mrcons_kd \\
        --epochs 3 --alpha 0.5 --merge-ratio 0.9 --grad-accum 4 \\
        --mr-cons-weight 0.5 --mr-cons-keep 0.15 --mr-cons-mode kl_to_anchor
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
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset  import StreamGazeMergeDataset
from TrajGazeMerge.models.merge  import gaze_weighted_merge
from TrajGazeMerge.models.model  import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
)
from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal

STAGE1_CKPT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth"
OUTPUT_ROOT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_mrcons"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",     default=STAGE1_CKPT)
    p.add_argument("--teacher-ckpt",    default=None)
    p.add_argument("--output-dir",      default=OUTPUT_ROOT)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--lr-lora",         type=float, default=1e-4)
    p.add_argument("--lr-enc",          type=float, default=1e-5)
    p.add_argument("--alpha",           type=float, default=0.0,
                   help="Weight of external KD loss (0=off, skips teacher load).")
    p.add_argument("--merge-ratio",     type=float, default=0.9)
    p.add_argument("--grad-accum",      type=int,   default=4)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--log-every",       type=int,   default=20)
    p.add_argument("--eval-every",      type=int,   default=400)
    p.add_argument("--n-frames",        type=int,   default=128)
    p.add_argument("--n-traj-frames",   type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int,   default=16)
    p.add_argument("--resume-ckpt",     default=None)
    p.add_argument("--start-epoch",     type=int,   default=0)
    p.add_argument("--resume-step",     type=int,   default=0)
    p.add_argument("--seed",            type=int,   default=42)
    # Multi-ratio consistency (self-distillation)
    p.add_argument("--mr-cons-weight",  type=float, default=0.5,
                   help="Weight of mr-cons loss (0=off).")
    p.add_argument("--mr-cons-keep",    type=float, default=0.15,
                   help="Keep ratio for auxiliary forward (e.g. 0.15 = 15%% tokens).")
    p.add_argument("--mr-cons-mode",    type=str,   default="kl_to_anchor",
                   choices=["kl_to_anchor", "js_symmetric"],
                   help="kl_to_anchor: primary KL→stop_grad(aux). "
                        "js_symmetric: symmetric Jensen-Shannon.")
    return p.parse_args()


# ── DDP ───────────────────────────────────────────────────────────────────────

def setup_ddp():
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
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[TrajEnc] Loaded {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[TrajEnc] WARNING: ckpt not found: {ckpt_path}, using random init")
    return model


# ── Score computation ─────────────────────────────────────────────────────────

def get_patch_scores_temporal(traj_encoder, item: dict, device) -> torch.Tensor:
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    scores = traj_encoder.get_patch_scores(
        traj_batch,
        queries     = [item["question"]],
        frame_paths = [item["traj_frame_paths"]],
    )
    return scores.squeeze(0)   # (T_traj, 196)


def score_to_qwen_spatiotemporal(
    scores:    torch.Tensor,
    n_spatial: int,
    T_merged:  int,
) -> torch.Tensor:
    T_traj = scores.shape[0]
    side   = int(n_spatial ** 0.5)
    s2d    = scores.float().reshape(T_traj, 1, 14, 14)
    if side == 8:
        s16 = F.interpolate(s2d, size=(16, 16), mode="nearest")
        s8  = F.avg_pool2d(s16, kernel_size=2, stride=2)
        scores_spatial = s8.reshape(T_traj, n_spatial)
    else:
        out = F.interpolate(s2d, size=(side, side), mode="bilinear", align_corners=False)
        scores_spatial = out.reshape(T_traj, n_spatial)
    if T_traj != T_merged:
        scores_spatial = F.interpolate(
            scores_spatial.T.unsqueeze(0).float(),
            size=T_merged, mode="linear", align_corners=False,
        ).squeeze(0).T
    return scores_spatial.reshape(-1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(processor, qwen_model, base_qwen, traj_encoder,
             option_ids, device, merge_ratio, teacher_model=None):
    """Returns (merge_acc, full_acc, n, per_task_merge, per_task_full)."""
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128)
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

                _teacher    = teacher_model if teacher_model is not None else qwen_model
                logits_full = forward_logits(_teacher, build_full_inputs(base_qwen, cached))
                pred_full   = logits_full[option_ids].argmax().item()

                scores     = get_patch_scores_temporal(traj_encoder, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                        else scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]

                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r,
                )
                logits_merge = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                )
                pred_merge = logits_merge[option_ids].argmax().item()

                gt_idx   = ["A", "B", "C", "D"].index(item["answer"])
                ok_merge = int(pred_merge == gt_idx)
                ok_full  = int(pred_full  == gt_idx)
                correct_merge += ok_merge
                correct_full  += ok_full
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

    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    need_teacher = args.alpha > 0.0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[TrajGazeMerge-mrcons] output: {args.output_dir}")
        print(f"  GPUs={world_size}  epochs={args.epochs}  merge_ratio={args.merge_ratio}")
        print(f"  alpha={args.alpha}  need_teacher={need_teacher}")
        print(f"  mr_cons_weight={args.mr_cons_weight}  mr_cons_keep={args.mr_cons_keep}  "
              f"mr_cons_mode={args.mr_cons_mode}")
        print(f"  lr_lora={args.lr_lora}  lr_enc={args.lr_enc}  seed={args.seed}")

    # Models
    if need_teacher:
        if is_main: print("Loading teacher ...")
        teacher_model = load_teacher(args.teacher_ckpt, device)
    else:
        if is_main: print("Skipping teacher load (alpha=0).")
        teacher_model = None

    if is_main: print("Loading TrajGaze temporal encoder ...")
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device, args.n_vis_keyframes)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank], find_unused_parameters=True)

    if is_main: print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, qwen_model = load_qwen_lora(device)
    base_qwen  = qwen_model.get_base_model()
    qwen_model = DDP(qwen_model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main: print("All models loaded.")

    # Resume
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        if is_main: print(f"[Resume] {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        qwen_model.module.load_state_dict(ckpt["lora_state"],    strict=False)
        traj_encoder.module.load_state_dict(ckpt["encoder_state"], strict=False)

    # Dataset
    train_ds = StreamGazeMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(train_ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b[0], num_workers=2)

    # Optimizer
    lora_params = [p for n, p in qwen_model.named_parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer   = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        qwen_model.train()
        traj_encoder.train()
        optimizer.zero_grad()

        epoch_loss = epoch_ce = epoch_kl = epoch_cons = 0.0
        steps   = 0
        t_start = time.time()
        skip_steps = args.resume_step if epoch == args.start_epoch else 0

        for step, item in enumerate(loader):
            if step < skip_steps:
                continue
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
                r         = max(1, int(args.merge_ratio * n_video))
                gt_tensor = torch.tensor(
                    [["A","B","C","D"].index(item["answer"])], device=device
                )

                # External teacher KD (only when alpha > 0)
                if teacher_model is not None:
                    with torch.no_grad():
                        logits_teacher = forward_logits(
                            teacher_model, build_full_inputs(base_qwen, cached)
                        )[option_ids].detach()
                else:
                    logits_teacher = None

                # TrajGaze scores
                scores     = get_patch_scores_temporal(traj_encoder.module, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                        else scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]

                video_embeds_detached = cached["video_embeds"].detach()

                # Primary forward: keep merge-ratio% tokens (e.g. 10%)
                merged_video, receiver_idx = gaze_weighted_merge(
                    video_embeds_detached, scores_all, r,
                )
                logits_student = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                )[option_ids]

                # Auxiliary forward: keep mr-cons-keep% tokens (e.g. 15%) — detached anchor
                logits_student_aux = None
                if args.mr_cons_weight > 0.0:
                    r_aux = max(1, int((1.0 - args.mr_cons_keep) * n_video))
                    merged_video2, receiver_idx2 = gaze_weighted_merge(
                        video_embeds_detached, scores_all, r_aux,
                    )
                    logits_student_aux = forward_logits(
                        qwen_model,
                        build_merged_inputs(base_qwen, cached, merged_video2, receiver_idx2),
                    )[option_ids]

                # CE loss
                loss_ce = F.cross_entropy(logits_student.unsqueeze(0), gt_tensor)

                # External KD loss (alpha > 0)
                if logits_teacher is not None:
                    loss_kl = F.kl_div(
                        F.log_softmax(logits_student, dim=-1),
                        F.softmax(logits_teacher,     dim=-1),
                        reduction="batchmean",
                    )
                else:
                    loss_kl = torch.tensor(0.0, device=device)

                loss_unscaled = args.alpha * loss_kl + (1.0 - args.alpha) * loss_ce

                # Multi-ratio consistency loss (self-distillation)
                loss_cons_val = 0.0
                if args.mr_cons_weight > 0.0 and logits_student_aux is not None:
                    if args.mr_cons_mode == "kl_to_anchor":
                        anchor    = logits_student_aux.detach()
                        loss_cons = F.kl_div(
                            F.log_softmax(logits_student, dim=-1),
                            F.softmax(anchor, dim=-1),
                            reduction="batchmean",
                        )
                    else:  # js_symmetric
                        p_dist = F.softmax(logits_student,     dim=-1)
                        q_dist = F.softmax(logits_student_aux, dim=-1)
                        m_dist = (0.5 * (p_dist + q_dist)).clamp(min=1e-12)
                        log_m  = m_dist.log()
                        loss_cons = 0.5 * (
                            F.kl_div(log_m, p_dist, reduction="batchmean") +
                            F.kl_div(log_m, q_dist, reduction="batchmean")
                        )
                    loss_unscaled = loss_unscaled + args.mr_cons_weight * loss_cons
                    loss_cons_val = float(loss_cons.detach())

                loss = loss_unscaled / args.grad_accum
                loss.backward()

                epoch_loss += loss_unscaled.item()
                epoch_ce   += loss_ce.item()
                epoch_kl   += loss_kl.item()
                epoch_cons += loss_cons_val
                steps      += 1

                if steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and steps % args.log_every == 0:
                    avg_l    = epoch_loss / steps
                    avg_ce   = epoch_ce   / steps
                    avg_kl   = epoch_kl   / steps
                    avg_cons = epoch_cons / steps
                    print(f"Epoch {epoch+1} | step {steps}/{len(loader)} | "
                          f"loss={avg_l:.4f} ce={avg_ce:.4f} kl={avg_kl:.4f} "
                          f"cons={avg_cons:.4f} | t={time.time()-t_start:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": steps,
                            "loss": avg_l, "ce": avg_ce, "kl": avg_kl, "cons": avg_cons,
                        }) + "\n")

                if steps % args.eval_every == 0:
                    dist.barrier()
                    if is_main:
                        acc_m, acc_f, n_eval, pt_merge, pt_full = evaluate(
                            processor, qwen_model.module, base_qwen, traj_encoder.module,
                            option_ids, device, args.merge_ratio,
                            teacher_model=teacher_model,
                        )
                        print(f"  → eval: merge={acc_m:.2f}% full={acc_f:.2f}% (n={n_eval})")
                        for t in sorted(pt_merge):
                            print(f"     {t}: merge={pt_merge[t]:.2f}% full={pt_full[t]:.2f}%")
                        with open(log_path, "a") as f:
                            f.write(json.dumps({
                                "type": "eval", "epoch": epoch+1, "step": steps,
                                "acc_merge": acc_m, "acc_full": acc_f, "n_eval": n_eval,
                                "per_task_merge": pt_merge, "per_task_full": pt_full,
                            }) + "\n")
                        if acc_m > best_acc:
                            best_acc = acc_m
                            torch.save({
                                "epoch": epoch, "step": steps,
                                "lora_state":    qwen_model.module.state_dict(),
                                "encoder_state": traj_encoder.module.state_dict(),
                                "acc_merge": acc_m, "acc_full": acc_f,
                            }, os.path.join(args.output_dir, "best.pth"))
                            print(f"  → saved best (merge={acc_m:.2f}%)")
                    dist.barrier()

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_l = epoch_loss / max(1, steps)
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_l:.4f} | "
                  f"time={time.time()-t_start:.0f}s ===")
            torch.save({
                "epoch": epoch,
                "lora_state":    qwen_model.module.state_dict(),
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg_l,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    # Final eval
    dist.barrier()
    if is_main:
        acc_m, acc_f, n_eval, pt_merge, pt_full = evaluate(
            processor, qwen_model.module, base_qwen, traj_encoder.module,
            option_ids, device, args.merge_ratio,
            teacher_model=teacher_model,
        )
        print(f"\n[Final] merge={acc_m:.2f}%  full={acc_f:.2f}%  (n={n_eval})")
        for t in sorted(pt_merge):
            print(f"   {t}: merge={pt_merge[t]:.2f}% full={pt_full[t]:.2f}%")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "type": "eval_final",
                "acc_merge": acc_m, "acc_full": acc_f, "n_eval": n_eval,
                "per_task_merge": pt_merge, "per_task_full": pt_full,
            }) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
