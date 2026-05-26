"""
Anchored TrajGazeMerge variant: gaze + left/right-hand spatial cells are
mandatory receivers; remaining slots in the 10% token budget are filled by
the encoder's score top-k.

Default A (when anchor count > target receiver count): the highest-score
anchors stay, lower-score anchors fall back to source pool. Implemented by
adding a large constant (ANCHOR_BOOST) to scores_all at anchor positions —
argsort naturally promotes anchors first, and the top-(N-r) cut keeps the
highest-score anchors first.

Identical to train_merge_lora_no_kd.py except for the anchor-boost lines.
Loss: CE(student_logits, label) only (no KL divergence, no teacher model).

Trainable: TrajGaze encoder + Qwen LoRA adapters (ViT frozen).
Train : egoexolearn + holoassist
Test  : egtea (periodic eval)
GPUs  : 4 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \\
        -m TrajGazeMerge.training.train_merge_lora_no_kd \\
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_no_kd \\
        --epochs 3 --lr-lora 1e-4 --lr-enc 1e-5 \\
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

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.merge import (
    gaze_weighted_merge, score_to_qwen_spatial, compute_anchor_mask_for_qwen,
)
ANCHOR_BOOST = 1.0e6   # ensures anchored cells outrank any non-anchor scores
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
)
from TrajGaze_v2.models.model import TrajGazeV2

STAGE1_CKPT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"
OUTPUT_ROOT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_no_kd_anchored"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",  default=STAGE1_CKPT)
    p.add_argument("--output-dir",   default=OUTPUT_ROOT)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr-lora",      type=float, default=1e-4)
    p.add_argument("--lr-enc",       type=float, default=1e-5)
    p.add_argument("--merge-ratio",  type=float, default=0.9,
                   help="Fraction of visual tokens to merge away (0.9 = keep 10%%)")
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--log-every",    type=int,   default=20)
    p.add_argument("--eval-every",   type=int,   default=200)
    p.add_argument("--n-frames",     type=int,   default=128)
    p.add_argument("--n-traj-frames", type=int,  default=32)
    p.add_argument("--resume-ckpt",  default=None)
    p.add_argument("--start-epoch",  type=int,   default=0)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def load_traj_encoder(ckpt_path: str, device: torch.device) -> TrajGazeV2:
    model = TrajGazeV2().to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[TrajEnc] {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[TrajEnc] WARNING: {ckpt_path} not found — random init")
    return model


def get_patch_scores(traj_encoder, item: dict, device: torch.device) -> torch.Tensor:
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb   = traj_encoder.query_encoder([item["question"]], device)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)
    return scores_raw.squeeze(0)   # (196,) with grad


def evaluate(processor, qwen_model, base_qwen, traj_encoder,
             option_ids, device, merge_ratio, max_items=200):
    test_ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=32)
    test_ds.items = test_ds.items[:max_items]
    qwen_model.eval(); traj_encoder.eval()
    correct_merge, correct_full, total = 0, 0, 0

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
                n_video = cached["video_embeds"].shape[0]
                T_merged = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r = max(1, int(merge_ratio * n_video))

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                oids = option_ids[:n_opt]

                full_inputs  = build_full_inputs(base_qwen, cached)
                logits_full  = forward_logits(qwen_model, full_inputs)
                pred_full    = logits_full[oids].argmax().item()

                scores     = get_patch_scores(traj_encoder, item, device)
                scores_q   = score_to_qwen_spatial(scores, n_spatial)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                if scores_all.shape[0] != n_video:
                    if scores_all.shape[0] > n_video:
                        scores_all = scores_all[:n_video]
                    else:
                        scores_all = scores_all.repeat((n_video + scores_all.shape[0] - 1) // scores_all.shape[0])[:n_video]

                # Anchor boost (gaze + left/right hand cells → mandatory receivers).
                anchor_spat = compute_anchor_mask_for_qwen(item["traj"], n_spatial).to(scores_all.device)
                anchor_all  = anchor_spat.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                if anchor_all.shape[0] != n_video:
                    anchor_all = anchor_all[:n_video] if anchor_all.shape[0] > n_video \
                        else anchor_all.repeat((n_video + anchor_all.shape[0] - 1) // anchor_all.shape[0])[:n_video]
                scores_all = scores_all + ANCHOR_BOOST * anchor_all

                video_det = cached["video_embeds"].detach()
                merged_video, receiver_idx = gaze_weighted_merge(video_det, scores_all, r)
                merged_inputs = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                logits_merge  = forward_logits(qwen_model, merged_inputs)
                pred_merge    = logits_merge[oids].argmax().item()

                gt_idx = letters.index(item["answer"])
                correct_full  += int(pred_full  == gt_idx)
                correct_merge += int(pred_merge == gt_idx)
                total += 1
            except Exception:
                pass

    qwen_model.train(); traj_encoder.train()
    return (100.0 * correct_merge / max(1, total),
            100.0 * correct_full  / max(1, total), total)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Exp2-no-KD] output={args.output_dir}  merge_ratio={args.merge_ratio}")
        print(f"  GPUs={world_size} epochs={args.epochs} lr_lora={args.lr_lora} lr_enc={args.lr_enc}")

    traj_encoder = load_traj_encoder(args.stage1_ckpt, device)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank], find_unused_parameters=True)

    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    qwen_model = DDP(qwen_model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)   # A–E pool; sliced per item
    if is_main:
        print("Models loaded.", flush=True)

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        qwen_model.module.load_state_dict(ckpt["lora_state"], strict=False)
        traj_encoder.module.load_state_dict(ckpt["encoder_state"], strict=False)
        if is_main:
            print(f"Resumed from {args.resume_ckpt}", flush=True)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(train_ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b[0], num_workers=2)

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
        qwen_model.train(); traj_encoder.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                oids = option_ids[:n_opt]

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r         = max(1, int(args.merge_ratio * n_video))
                gt_tensor = torch.tensor(
                    [letters.index(item["answer"])], device=device
                )

                scores     = get_patch_scores(traj_encoder.module, item, device)
                scores_q   = score_to_qwen_spatial(scores, n_spatial)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                if scores_all.shape[0] != n_video:
                    if scores_all.shape[0] > n_video:
                        scores_all = scores_all[:n_video]
                    else:
                        scores_all = scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]

                # Anchor boost (gaze + left/right hand cells → mandatory receivers).
                anchor_spat = compute_anchor_mask_for_qwen(item["traj"], n_spatial).to(scores_all.device)
                anchor_all  = anchor_spat.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                if anchor_all.shape[0] != n_video:
                    anchor_all = anchor_all[:n_video] if anchor_all.shape[0] > n_video \
                        else anchor_all.repeat((n_video + anchor_all.shape[0] - 1) // anchor_all.shape[0])[:n_video]
                scores_all = scores_all + ANCHOR_BOOST * anchor_all

                video_det = cached["video_embeds"].detach()
                merged_video, receiver_idx = gaze_weighted_merge(video_det, scores_all, r)
                merged_inputs = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                logits = forward_logits(qwen_model, merged_inputs)[oids]

                loss = F.cross_entropy(logits.unsqueeze(0), gt_tensor) / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                steps += 1

                if steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and steps % args.log_every == 0:
                    avg = epoch_loss / steps
                    elapsed = time.time() - t_start
                    print(f"Epoch {epoch+1} | step {steps}/{len(loader)} | "
                          f"loss={avg:.4f} | t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": steps,
                            "loss": avg, "elapsed": elapsed,
                        }) + "\n")

                if is_main and steps % args.eval_every == 0:
                    acc_m, acc_f, n_eval = evaluate(
                        processor, qwen_model.module, base_qwen, traj_encoder.module,
                        option_ids, device, args.merge_ratio,
                    )
                    print(f"  → eval: merge={acc_m:.2f}% full={acc_f:.2f}% (n={n_eval})", flush=True)
                    if acc_m > best_acc:
                        best_acc = acc_m
                        torch.save({
                            "epoch": epoch, "step": steps,
                            "lora_state": qwen_model.module.state_dict(),
                            "encoder_state": traj_encoder.module.state_dict(),
                            "acc_merge": acc_m, "acc_full": acc_f,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (merge={acc_m:.2f}%)", flush=True)

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg = epoch_loss / max(1, steps)
        if is_main:
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} avg_loss={avg:.4f} t={elapsed:.0f}s ===",
                  flush=True)
            torch.save({
                "epoch": epoch,
                "lora_state": qwen_model.module.state_dict(),
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "avg_loss": avg, "elapsed": elapsed,
                }) + "\n")

    if is_main:
        acc_m, acc_f, n_eval = evaluate(
            processor, qwen_model.module, base_qwen, traj_encoder.module,
            option_ids, device, args.merge_ratio, max_items=500,
        )
        print(f"\n[Final] egtea: merge={acc_m:.2f}%  full={acc_f:.2f}%  (n={n_eval})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
