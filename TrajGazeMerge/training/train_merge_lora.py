"""
TrajGazeMerge Stage 2: joint training of TrajGaze encoder + Qwen LoRA.

Mechanism:
  - Teacher pass : fine-tuned baseline LoRA Qwen + full visual tokens → logits_teacher
  - Student pass : Qwen + LoRA + gaze-merged tokens (10% budget)      → logits_student
  - Loss         : α * KL(student || teacher) + (1-α) * CE(student, label)
  - Trainable    : TrajGaze encoder + LoRA adapters (ViT frozen)

Gradient flow:
  loss → student logits → LoRA weights
                        → merged_tokens → merge op (diff. weighted avg)
                                       → patch_scores → TrajGaze encoder

Train : egoexolearn + holoassist
Test  : egtea (periodic eval)
GPUs  : 2 via torchrun

Usage:
    torchrun --nproc_per_node=2 -m TrajGazeMerge.training.train_merge_lora \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \
        --teacher-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora \
        --epochs 3 --lr-lora 1e-4 --lr-enc 1e-5 --alpha 0.5 \
        --merge-ratio 0.9 --grad-accum 4
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

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge, score_to_qwen_spatial
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
)
from TrajGaze_v2.models.model import TrajGazeV2

STAGE1_CKPT  = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"
OUTPUT_ROOT  = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",  default=STAGE1_CKPT)
    p.add_argument("--teacher-ckpt", default=None,
                   help="Path to trained baseline LoRA checkpoint for teacher")
    p.add_argument("--output-dir",   default=OUTPUT_ROOT)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr-lora",      type=float, default=1e-4)
    p.add_argument("--lr-enc",       type=float, default=1e-5)
    p.add_argument("--alpha",        type=float, default=0.5,
                   help="KL loss weight; (1-alpha) for CE loss")
    p.add_argument("--merge-ratio",  type=float, default=0.9,
                   help="Fraction of visual tokens to merge away (0.9 = keep 10%%)")
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--log-every",    type=int,   default=20)
    p.add_argument("--eval-every",   type=int,   default=200)
    p.add_argument("--n-frames",     type=int,   default=128)
    p.add_argument("--n-traj-frames", type=int,  default=32)
    p.add_argument("--resume-ckpt",  default=None,
                   help="Path to checkpoint to resume from (loads lora_state + encoder_state)")
    p.add_argument("--start-epoch",  type=int,   default=0,
                   help="Epoch to start from when resuming")
    return p.parse_args()


def setup_ddp():
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def load_teacher(teacher_ckpt: str, device: torch.device):
    """
    Load a frozen copy of the fine-tuned baseline LoRA Qwen as the teacher model.
    If teacher_ckpt is None or not found, falls back to base pretrained Qwen (frozen).
    Returns the teacher model (PeftModel or base model) with all params frozen.
    """
    from TrajGazeMerge.models.model import load_qwen_lora, load_qwen_frozen
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        print(f"[Teacher] Loading fine-tuned LoRA from: {teacher_ckpt}")
        _, teacher = load_qwen_lora(device)
        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["lora_state"], strict=False)
        print("[Teacher] LoRA weights loaded.")
    else:
        print("[Teacher] No teacher ckpt found, falling back to base pretrained Qwen.")
        _, teacher = load_qwen_frozen(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_traj_encoder(ckpt_path: str, device: torch.device) -> TrajGazeV2:
    model = TrajGazeV2().to(device)
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[TrajEnc] Loaded {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[TrajEnc] WARNING: ckpt not found: {ckpt_path}, using random init")
    return model


def get_patch_scores(
    traj_encoder,
    item:   dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Run TrajGaze encoder forward to get (196,) patch scores.
    Returns tensor with grad attached (for backprop through merge).
    """
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb   = traj_encoder.query_encoder([item["question"]], device)           # (1, D_Q)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)  # (1, 196, D)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)      # (1, 196)
    return scores_raw.squeeze(0)   # (196,) with grad


def evaluate(processor, qwen_model, base_qwen, traj_encoder,
             option_ids, device, merge_ratio, max_items=None,
             teacher_model=None):
    """Evaluate TrajGazeMerge on egtea test split; returns (merge_acc, full_acc, n)."""
    from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=32)
    if max_items is not None:
        test_ds.items = test_ds.items[:max_items]

    qwen_model.eval()
    traj_encoder.eval()
    correct_merge = 0
    correct_full  = 0
    total         = 0

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r         = max(1, int(merge_ratio * n_video))

                # Full tokens with teacher (or fall back to base model)
                _teacher = teacher_model if teacher_model is not None else qwen_model
                full_inputs  = build_full_inputs(base_qwen, cached)
                logits_full  = forward_logits(_teacher, full_inputs)
                pred_full = logits_full[option_ids].argmax().item()

                # Merged tokens (with LoRA)
                scores     = get_patch_scores(traj_encoder, item, device)
                scores_q   = score_to_qwen_spatial(scores, n_spatial)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)

                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                                 else scores_all.repeat_interleave(
                                     (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                                 )[:n_video]

                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r
                )
                merged_inputs = build_merged_inputs(
                    base_qwen, cached, merged_video, receiver_idx
                )
                logits_merge  = forward_logits(qwen_model, merged_inputs)
                pred_merge    = logits_merge[option_ids].argmax().item()

                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                correct_full  += int(pred_full  == gt_idx)
                correct_merge += int(pred_merge == gt_idx)
                total         += 1
            except Exception:
                pass

    qwen_model.train()
    traj_encoder.train()
    acc_merge = 100.0 * correct_merge / max(1, total)
    acc_full  = 100.0 * correct_full  / max(1, total)
    return acc_merge, acc_full, total


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[TrajGazeMerge] output: {args.output_dir}")
        print(f"[TrajGazeMerge] GPUs={world_size}, epochs={args.epochs}, "
              f"lr_lora={args.lr_lora}, lr_enc={args.lr_enc}, "
              f"alpha={args.alpha}, merge_ratio={args.merge_ratio}")
        print(f"[TrajGazeMerge] teacher_ckpt={args.teacher_ckpt}")

    # ── Load teacher model (frozen, fine-tuned baseline LoRA) ─────────────────
    if is_main:
        print("Loading teacher model ...")
    teacher_model = load_teacher(args.teacher_ckpt, device)
    if is_main:
        print("Teacher loaded.")

    # ── Load TrajGaze encoder ─────────────────────────────────────────────────
    if is_main:
        print("Loading TrajGaze encoder ...")
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank],
                       find_unused_parameters=True)
    if is_main:
        print("TrajGaze encoder loaded.")

    # ── Load Qwen + LoRA ──────────────────────────────────────────────────────
    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()

    qwen_model = DDP(qwen_model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main:
        print("Qwen loaded.")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        if is_main:
            print(f"[Resume] Loading weights from {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        qwen_model.module.load_state_dict(ckpt["lora_state"], strict=False)
        traj_encoder.module.load_state_dict(ckpt["encoder_state"], strict=False)
        if is_main:
            print(f"[Resume] Weights loaded. Starting from epoch {args.start_epoch}.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = StreamGazeMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                 rank=rank, shuffle=True)
    loader  = DataLoader(train_ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b[0], num_workers=2)

    # ── Optimizer: separate LR for encoder vs LoRA ───────────────────────────
    lora_params = [p for n, p in qwen_model.named_parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer   = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    log_path  = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc  = 0.0
    n_steps   = 0

    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        qwen_model.train()
        traj_encoder.train()
        optimizer.zero_grad()

        epoch_loss    = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_kl = 0.0
        steps_this_epoch = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                # ── Preprocess (no grad for ViT) ──────────────────────────────
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r         = max(1, int(args.merge_ratio * n_video))

                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                gt_tensor = torch.tensor([gt_idx], device=device)

                # ── Teacher forward: full tokens, fine-tuned baseline LoRA, no grad ──
                with torch.no_grad():
                    full_inputs = build_full_inputs(base_qwen, cached)
                    logits_teacher = forward_logits(
                        teacher_model, full_inputs
                    )[option_ids].detach()   # (4,)

                # ── Patch scores from TrajGaze encoder (with grad) ────────────
                scores = get_patch_scores(traj_encoder.module, item, device)  # (196,)

                # Interpolate 14×14 → Qwen spatial grid, expand across T_merged
                scores_q   = score_to_qwen_spatial(scores, n_spatial)   # (n_spatial,)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                # Handle size mismatch (safety)
                if scores_all.shape[0] != n_video:
                    if scores_all.shape[0] > n_video:
                        scores_all = scores_all[:n_video]
                    else:
                        reps = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        scores_all = scores_all.repeat(reps)[:n_video]

                # ── Gaze-weighted merge ────────────────────────────────────────
                # video_embeds is detached (ViT frozen) — grad only through scores_all
                video_embeds_detached = cached["video_embeds"].detach()
                merged_video, receiver_idx = gaze_weighted_merge(
                    video_embeds_detached, scores_all, r
                )
                # merged_video: (n_video - r, d) with grad through scores_all

                # ── Student forward: merged tokens, LoRA active ───────────────
                merged_inputs = build_merged_inputs(
                    base_qwen, cached, merged_video, receiver_idx
                )
                logits_student = forward_logits(
                    qwen_model, merged_inputs
                )[option_ids]   # (4,) with grad

                # ── Loss ──────────────────────────────────────────────────────
                loss_ce = F.cross_entropy(
                    logits_student.unsqueeze(0), gt_tensor
                )
                loss_kl = F.kl_div(
                    F.log_softmax(logits_student, dim=-1),
                    F.softmax(logits_teacher,     dim=-1),
                    reduction="batchmean",
                )
                loss = (args.alpha * loss_kl + (1.0 - args.alpha) * loss_ce) / args.grad_accum
                loss.backward()

                epoch_loss    += loss.item() * args.grad_accum
                epoch_loss_ce += loss_ce.item()
                epoch_loss_kl += loss_kl.item()
                steps_this_epoch += 1
                n_steps          += 1

                if steps_this_epoch % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_params + enc_params, args.grad_clip
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and steps_this_epoch % args.log_every == 0:
                    avg_l  = epoch_loss    / steps_this_epoch
                    avg_ce = epoch_loss_ce / steps_this_epoch
                    avg_kl = epoch_loss_kl / steps_this_epoch
                    elapsed = time.time() - t_start
                    print(
                        f"Epoch {epoch+1} | step {steps_this_epoch}/{len(loader)} | "
                        f"loss={avg_l:.4f} ce={avg_ce:.4f} kl={avg_kl:.4f} | "
                        f"t={elapsed:.0f}s"
                    )
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": steps_this_epoch,
                            "loss": avg_l, "ce": avg_ce, "kl": avg_kl,
                        }) + "\n")

                if steps_this_epoch % args.eval_every == 0:
                    dist.barrier()
                    if is_main:
                        acc_m, acc_f, n_eval = evaluate(
                            processor, qwen_model.module, base_qwen, traj_encoder.module,
                            option_ids, device, args.merge_ratio,
                            teacher_model=teacher_model,
                        )
                        print(f"  → eval egtea: merge={acc_m:.2f}% full={acc_f:.2f}% (n={n_eval})")
                        if acc_m > best_acc:
                            best_acc = acc_m
                            torch.save({
                                "epoch": epoch, "step": steps_this_epoch,
                                "lora_state": qwen_model.module.state_dict(),
                                "encoder_state": traj_encoder.module.state_dict(),
                                "acc_merge": acc_m, "acc_full": acc_f,
                            }, os.path.join(args.output_dir, "best.pth"))
                            print(f"  → saved best (merge={acc_m:.2f}%)")
                    dist.barrier()

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Flush remaining gradients
        if steps_this_epoch % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_l   = epoch_loss / max(1, steps_this_epoch)
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_l:.4f} | time={elapsed:.0f}s ===")
            torch.save({
                "epoch": epoch,
                "lora_state":    qwen_model.module.state_dict(),
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg_l,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    # Final evaluation
    dist.barrier()
    if is_main:
        acc_m, acc_f, n_eval = evaluate(
            processor, qwen_model.module, base_qwen, traj_encoder.module,
            option_ids, device, args.merge_ratio, max_items=500,
            teacher_model=teacher_model,
        )
        print(f"\n[Final] egtea: merge={acc_m:.2f}%  full={acc_f:.2f}%  (n={n_eval})")
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
