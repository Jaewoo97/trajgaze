"""
Per-task evaluation of a TrajGazeV2 temporal trained TrajGazeMerge checkpoint.

Adapted from /workspace/trajgaze_msk/TrajGazeMerge/eval/eval_per_task.py for
the v2 temporal encoder (TrajGazeV2Temporal + score_to_qwen_spatiotemporal).

Single-GPU eval. Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/trajgaze_v2 \
        python -m TrajGazeMerge.eval.eval_per_task_temporal \
        --ckpt         /workspace/trajgaze_v2/TrajGazeMerge/checkpoints/no_kd_v2/best.pth \
        --teacher-ckpt /workspace/trajgaze_msk/king_ms.pth \
        --stage1-ckpt  /workspace/trajgaze_msk/temporal_best.pth \
        --merge-ratio  0.9
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch

if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True, help="Path to Stage 2 best.pth")
    p.add_argument("--teacher-ckpt", required=True, help="Path to baseline LoRA teacher")
    p.add_argument("--stage1-ckpt",  required=True, help="Path to TrajGaze v2 temporal Stage 1 ckpt")
    p.add_argument("--merge-ratio",  type=float, default=0.9)
    p.add_argument("--n-frames",     type=int, default=128)
    p.add_argument("--n-traj-frames",type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--max-items",    type=int, default=-1,
                   help="Limit #test items (-1 = full split)")
    p.add_argument("--device",       default="cuda:0")
    return p.parse_args()


def score_to_qwen_spatiotemporal(scores, n_spatial, T_merged):
    """
    Inline copy from train_merge_lora_temporal.py for eval-time use.
    scores: (T_traj, 196) → flattened (T_merged * n_spatial,)
    """
    import torch.nn.functional as F
    T_traj, _ = scores.shape
    side = int(n_spatial ** 0.5)
    # Spatial: 14×14 → side×side
    s = scores.view(T_traj, 14, 14)
    if side == 8:
        # 14 → 16 nearest, 16 → 8 avg_pool
        s = F.interpolate(s.unsqueeze(1), size=(16, 16), mode="nearest")
        s = F.avg_pool2d(s, 2).squeeze(1)
    else:
        s = F.interpolate(s.unsqueeze(1), size=(side, side), mode="bilinear",
                          align_corners=False).squeeze(1)
    # Temporal: T_traj → T_merged via linear interp
    s = s.permute(1, 2, 0).reshape(side * side, T_traj).unsqueeze(0)  # (1, n_spatial, T_traj)
    s = F.interpolate(s, size=T_merged, mode="linear", align_corners=False).squeeze(0)
    # Now shape (n_spatial, T_merged) → (T_merged, n_spatial) → flat
    return s.t().contiguous().view(-1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
    from TrajGazeMerge.models.merge import gaze_weighted_merge
    from TrajGazeMerge.models.model import (
        load_qwen_lora, get_option_ids, preprocess_item,
        build_merged_inputs, build_full_inputs, forward_logits,
    )
    from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal

    # Teacher (frozen, fine-tuned baseline LoRA)
    print(f"[Teacher] Loading {args.teacher_ckpt}")
    _, teacher = load_qwen_lora(device)
    t_ck = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(t_ck["lora_state"], strict=False)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)

    # Student (Qwen LoRA loaded from Stage 2 best.pth)
    print(f"[Student] Loading LoRA from {args.ckpt}")
    processor, student = load_qwen_lora(device)
    s_ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(s_ck["lora_state"], strict=False)
    student.eval()
    for pr in student.parameters():
        pr.requires_grad_(False)
    base_qwen  = student.get_base_model()
    option_ids = get_option_ids(processor)

    # TrajGaze v2 temporal encoder: load stage1, then overlay stage2 encoder_state
    print(f"[TrajEnc] Loading {args.stage1_ckpt}, overlaying {args.ckpt}")
    traj = TrajGazeV2Temporal(n_vis_keyframes=args.n_vis_keyframes).to(device)
    t_stage1 = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    state    = t_stage1.get("model", t_stage1.get("model_state_dict", t_stage1))
    traj.load_state_dict(state, strict=False)
    traj.load_state_dict(s_ck["encoder_state"], strict=False)
    traj.eval()
    for pr in traj.parameters():
        pr.requires_grad_(False)

    ds = StreamGazeMergeDataset(split="test",
                                n_vlm_frames=args.n_frames,
                                n_traj_frames=args.n_traj_frames)
    if args.max_items > 0:
        ds.items = ds.items[:args.max_items]
    print(f"[Eval] {len(ds.items)} items")

    per_task = defaultdict(lambda: {"merge_ok": 0, "full_ok": 0, "n": 0})

    for i, item in enumerate(ds):
        if item is None:
            continue
        try:
            task = item["task"]
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

            # Teacher: full tokens
            full_inputs = build_full_inputs(base_qwen, cached)
            logits_full = forward_logits(teacher, full_inputs)
            pred_full = logits_full[option_ids].argmax().item()

            # Student: merged tokens via temporal scores
            traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
            scores = traj.get_patch_scores(
                traj_batch,
                queries     = [item["question"]],
                frame_paths = [item["traj_frame_paths"]],
            ).squeeze(0)                            # (T_traj, 196)
            scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
            if scores_all.shape[0] != n_video:
                if scores_all.shape[0] > n_video:
                    scores_all = scores_all[:n_video]
                else:
                    reps = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                    scores_all = scores_all.repeat(reps)[:n_video]

            merged_video, receiver_idx = gaze_weighted_merge(
                cached["video_embeds"], scores_all, r,
            )
            merged_inputs = build_merged_inputs(
                base_qwen, cached, merged_video, receiver_idx,
            )
            logits_merge = forward_logits(student, merged_inputs)
            pred_merge = logits_merge[option_ids].argmax().item()

            gt = ["A", "B", "C", "D"].index(item["answer"])
            per_task[task]["full_ok"]  += int(pred_full  == gt)
            per_task[task]["merge_ok"] += int(pred_merge == gt)
            per_task[task]["n"]        += 1
        except Exception as e:
            if i < 5:
                print(f"  [skip item {i}] {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(ds.items)} items")

    # Aggregate
    print("\n" + "=" * 82)
    print(f"{'Task':<50} {'Merge':>8} {'Full':>8} {'Gap':>7} {'N':>4}")
    print("-" * 82)
    tot = {"merge_ok": 0, "full_ok": 0, "n": 0}
    for task in sorted(per_task.keys()):
        s = per_task[task]
        if s["n"] == 0:
            continue
        acc_m = 100.0 * s["merge_ok"] / s["n"]
        acc_f = 100.0 * s["full_ok"]  / s["n"]
        gap   = acc_f - acc_m
        print(f"{task:<50} {acc_m:7.2f}% {acc_f:7.2f}% {gap:+6.2f} {s['n']:4d}")
        for k in tot:
            tot[k] += s[k]
    if tot["n"] > 0:
        amm = 100.0 * tot["merge_ok"] / tot["n"]
        aff = 100.0 * tot["full_ok"]  / tot["n"]
        print("-" * 82)
        print(f"{'OVERALL':<50} {amm:7.2f}% {aff:7.2f}% {aff-amm:+6.2f} {tot['n']:4d}")
    print("=" * 82)


if __name__ == "__main__":
    main()
