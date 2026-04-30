"""
TrajGazeMerge Stage 3 — single-GPU, no KD, ablation training.

Supports three model variants via --model-type:
  full       : TrajGazeV2Temporal (gaze + hand, 4 tokens/frame)
  gaze_only  : TrajGazeV2TemporalGazeOnly (gaze only, 1 token/frame)
  hand_only  : TrajGazeV2TemporalHandOnly (hand only, 3 tokens/frame)

Loss: CE only (no KD teacher). Trains LoRA + TrajGaze encoder end-to-end.

Usage:
    # GPU 0: full model, no KD
    CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
        --model-type full \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth \
        --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_temporal_no_kd \
        --epochs 3 --merge-ratio 0.9 --grad-accum 4

    # GPU 1: gaze-only, no KD (run after stage1_temporal_ablation --model-type gaze_only)
    CUDA_VISIBLE_DEVICES=1 python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
        --model-type gaze_only \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal_gaze_only/best.pth \
        --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_temporal_gaze_only \
        --epochs 3 --merge-ratio 0.9 --grad-accum 4

    # GPU 3: hand-only, no KD (run after stage1_temporal_ablation --model-type hand_only)
    CUDA_VISIBLE_DEVICES=3 python -m TrajGazeMerge.training.train_merge_lora_temporal_no_kd \
        --model-type hand_only \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal_hand_only/best.pth \
        --output-dir  /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_temporal_hand_only \
        --epochs 3 --merge-ratio 0.9 --grad-accum 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
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
    p.add_argument("--grad-accum",      type=int,   default=4)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--log-every",       type=int,   default=20)
    p.add_argument("--eval-every",      type=int,   default=400)
    p.add_argument("--n-frames",        type=int,   default=128)
    p.add_argument("--n-traj-frames",   type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int,   default=16)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def load_traj_encoder(model_type, stage1_ckpt, device, n_vis_keyframes):
    if model_type == "full":
        from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal
        model = TrajGazeV2Temporal(n_vis_keyframes=n_vis_keyframes).to(device)
    elif model_type == "gaze_only":
        from TrajGaze_v2.models.model_temporal_gaze_only import TrajGazeV2TemporalGazeOnly
        model = TrajGazeV2TemporalGazeOnly(n_vis_keyframes=n_vis_keyframes).to(device)
    elif model_type == "hand_only":
        from TrajGaze_v2.models.model_temporal_hand_only import TrajGazeV2TemporalHandOnly
        model = TrajGazeV2TemporalHandOnly(n_vis_keyframes=n_vis_keyframes).to(device)

    ckpt  = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
    model.load_state_dict(state, strict=False)
    print(f"[TrajEncoder] loaded {model_type} from {stage1_ckpt}")
    return model


def get_patch_scores_temporal(traj_encoder, item, device):
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    scores = traj_encoder.get_patch_scores(
        traj_batch,
        queries     = [item["question"]],
        frame_paths = [item["traj_frame_paths"]],
    )
    return scores.squeeze(0)   # (T_traj, 196)


def score_to_qwen_spatiotemporal(scores, n_spatial, T_merged):
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


def evaluate(processor, qwen_model, base_qwen, traj_encoder, option_ids, device, merge_ratio):
    from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128)
    qwen_model.eval()
    traj_encoder.eval()
    correct = total = 0
    by_task: dict[str, list] = {}

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
                scores    = get_patch_scores_temporal(traj_encoder, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                        else scores_all.repeat((n_video + scores_all.shape[0] - 1) // scores_all.shape[0])[:n_video]
                merged_video, receiver_idx = gaze_weighted_merge(cached["video_embeds"], scores_all, r)
                logits = forward_logits(qwen_model, build_merged_inputs(base_qwen, cached, merged_video, receiver_idx))
                pred   = logits[option_ids].argmax().item()
                gt     = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred == gt)
                correct += ok
                total   += 1
                by_task.setdefault(item.get("task", "unknown"), []).append(ok)
            except Exception:
                pass

    qwen_model.train()
    traj_encoder.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


def main():
    args   = parse_args()

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    print(f"[Stage3-NoKD] model={args.model_type}  merge_ratio={args.merge_ratio}")
    print(f"  stage1_ckpt: {args.stage1_ckpt}")
    print(f"  output_dir:  {args.output_dir}", flush=True)

    print("Loading Qwen2.5-VL + LoRA ...", flush=True)
    processor, qwen_model = load_qwen_lora(device)
    base_qwen  = qwen_model.get_base_model()
    option_ids = get_option_ids(processor)

    print(f"Loading TrajGaze encoder ({args.model_type}) ...", flush=True)
    traj_encoder = load_traj_encoder(args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes)

    print("All models loaded.", flush=True)

    train_ds = StreamGazeMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
    )
    loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in qwen_model.parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer   = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(args.epochs):
        qwen_model.train()
        traj_encoder.train()
        optimizer.zero_grad()
        epoch_loss = epoch_ce = 0.0
        steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
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

                scores    = get_patch_scores_temporal(traj_encoder, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                        else scores_all.repeat((n_video + scores_all.shape[0] - 1) // scores_all.shape[0])[:n_video]

                merged_video, receiver_idx = gaze_weighted_merge(cached["video_embeds"], scores_all, r)
                logits_merge = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                )
                loss_ce = torch.nn.functional.cross_entropy(
                    logits_merge[option_ids].unsqueeze(0), gt_tensor
                )
                loss = loss_ce / args.grad_accum

                loss.backward()
                epoch_loss += loss_ce.item()
                epoch_ce   += loss_ce.item()
                steps      += 1

                if steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if steps % args.log_every == 0:
                    avg_l = epoch_loss / steps
                    print(f"Epoch {epoch+1} | step {steps}/{len(loader)} | "
                          f"loss={avg_l:.4f} | t={time.time()-t_start:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": steps,
                            "loss": avg_l,
                        }) + "\n")

                if steps % args.eval_every == 0:
                    acc, n_eval, per_task = evaluate(
                        processor, qwen_model, base_qwen, traj_encoder,
                        option_ids, device, args.merge_ratio,
                    )
                    print(f"  → eval: acc={acc:.2f}% (n={n_eval})")
                    for t, a in per_task.items():
                        print(f"     {t}: {a:.2f}%")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "type": "eval", "epoch": epoch+1, "step": steps,
                            "acc": acc, "n_eval": n_eval, "per_task": per_task,
                        }) + "\n")
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({
                            "epoch": epoch, "step": steps,
                            "lora_state":    qwen_model.state_dict(),
                            "encoder_state": traj_encoder.state_dict(),
                            "acc": acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best ({acc:.2f}%)")

            except Exception:
                traceback.print_exc()
                continue

        if steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_l = epoch_loss / max(1, steps)
        print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_l:.4f} | "
              f"time={time.time()-t_start:.0f}s ===")
        torch.save({
            "epoch": epoch, "lora_state": qwen_model.state_dict(),
            "encoder_state": traj_encoder.state_dict(), "loss": avg_l,
        }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    # Final eval
    acc, n_eval, per_task = evaluate(
        processor, qwen_model, base_qwen, traj_encoder,
        option_ids, device, args.merge_ratio,
    )
    print(f"\n[Final] acc={acc:.2f}%  (n={n_eval})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%")
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "type": "eval_final", "acc": acc, "n_eval": n_eval, "per_task": per_task,
        }) + "\n")


if __name__ == "__main__":
    main()
