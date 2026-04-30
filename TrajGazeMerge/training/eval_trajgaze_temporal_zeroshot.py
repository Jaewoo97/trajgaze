"""
Zero-shot TrajGazeMerge eval — temporal version.

Uses TrajGazeV2Temporal (stage1_temporal/best.pth) to produce per-frame
(T, 196) patch scores, aligned to Qwen's video token layout via
score_to_qwen_spatiotemporal, then gaze_weighted_merge keeps 10% of tokens.

VLM: baseline LoRA-finetuned Qwen2.5-VL (no stage 3 joint training).
No gradient computation — pure inference.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=2 python -m TrajGazeMerge.training.eval_trajgaze_temporal_zeroshot \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth \
        --lora-ckpt   /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \
        --out         /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/trajgaze_temporal_zeroshot_metrics.json \
        --merge-ratio 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    get_option_ids, preprocess_item, build_merged_inputs, forward_logits,
    QWEN_PATH, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
)
from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", default="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth")
    p.add_argument("--lora-ckpt",   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth")
    p.add_argument("--out",         default="/workspace/EgoGazeVQA/TrajGazeMerge/eval_results/trajgaze_temporal_zeroshot_metrics.json")
    p.add_argument("--merge-ratio", type=float, default=0.9)
    p.add_argument("--n-vlm-frames",   type=int, default=128)
    p.add_argument("--n-traj-frames",  type=int, default=128)
    p.add_argument("--n-vis-keyframes",type=int, default=16)
    p.add_argument("--split",       default="test")
    return p.parse_args()


def load_lora_model(device):
    from transformers import Qwen2_5_VLForConditionalGeneration
    processor  = AutoProcessor.from_pretrained(QWEN_PATH)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    return processor, get_peft_model(base_model, lora_cfg)


def load_traj_encoder(ckpt_path, device, n_vis_keyframes):
    enc = TrajGazeV2Temporal(n_vis_keyframes=n_vis_keyframes).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Stage 1 ckpt uses "model" / "model_state_dict"; Stage 3 ckpt uses "encoder_state"
    state = ckpt.get("encoder_state", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
    enc.load_state_dict(state, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def get_patch_scores_temporal(traj_encoder, item, device):
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    scores = traj_encoder.get_patch_scores(
        traj_batch,
        queries     = [item["question"]],
        frame_paths = [item["traj_frame_paths"]],
    )                           # (1, T_traj, 196)
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

    return scores_spatial.reshape(-1)   # (T_merged * n_spatial,)


def main():
    args   = parse_args()
    device = torch.device("cuda:0")   # mapped from CUDA_VISIBLE_DEVICES=2

    print(f"[ZeroShot-Temporal] merge_ratio={args.merge_ratio}  keep={100*(1-args.merge_ratio):.0f}%")
    print(f"  stage1: {args.stage1_ckpt}")
    print(f"  VLM:    {args.lora_ckpt}")
    print(f"  split:  {args.split}", flush=True)

    print("Loading baseline LoRA VLM ...", flush=True)
    processor, model = load_lora_model(device)
    ckpt  = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("lora_state", ckpt), strict=False)
    model.eval()
    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor)

    print("Loading TrajGazeV2Temporal stage1 encoder ...", flush=True)
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device, args.n_vis_keyframes)

    print("Models loaded.", flush=True)

    ds = StreamGazeMergeDataset(
        split=args.split, n_vlm_frames=args.n_vlm_frames, n_traj_frames=args.n_traj_frames,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=lambda b: b[0], num_workers=2)
    print(f"Evaluating {len(ds)} items ...", flush=True)

    correct = 0
    total   = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for i, item in enumerate(loader):
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

                scores     = get_patch_scores_temporal(traj_encoder, item, device)  # (T_traj, 196)
                scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)  # (n_video,)

                if scores_all.shape[0] != n_video:
                    if scores_all.shape[0] > n_video:
                        scores_all = scores_all[:n_video]
                    else:
                        reps = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        scores_all = scores_all.repeat(reps)[:n_video]

                r = max(1, int(args.merge_ratio * n_video))
                selected_embeds, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r,
                )

                inputs_dict = build_merged_inputs(base_qwen, cached, selected_embeds, receiver_idx)
                logits      = forward_logits(model, inputs_dict)
                pred_idx    = logits[option_ids].argmax().item()
                gt_idx      = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(loader)}] acc={100.*correct/max(1,total):.2f}%", flush=True)

            except Exception:
                traceback.print_exc()
                continue

    acc      = 100.0 * correct / max(1, total)
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}

    print(f"\n=== TrajGaze Temporal Zero-shot (merge={args.merge_ratio}) ===")
    print(f"Overall: {acc:.2f}%  (n={total})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%  (n={len(by_task[t])})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "acc": acc, "n": total, "per_task": per_task,
            "stage1_ckpt": args.stage1_ckpt,
            "lora_ckpt":   args.lora_ckpt,
            "merge_ratio": args.merge_ratio,
            "method": "trajgaze_temporal_zeroshot",
        }, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
