"""
Counterfactual masking evaluation.

For each test sample, compute the learned merge as in production, then run
multiple variants where parts of the merged video embedding are zeroed out:

  - baseline       : use merged_video unchanged (matches diagnostic_eval)
  - mask_kept      : zero out the merged_video tensor entirely
                     (= visual signal removed at receiver positions)
  - mask_kept_late : zero only receivers whose frame_idx >= T_merged/2
  - mask_kept_early: zero only receivers whose frame_idx <  T_merged/2
  - shuffle_kept   : randomly permute merged_video along the token axis
                     (preserve content distribution, destroy spatial-temporal alignment)
  - mask_gaze      : zero only receivers whose patch location is within
                     `--mask-radius` (normalized) of the GT gaze point of their frame
                     (only on frames with valid gaze_mask)
  - mask_hand      : same as mask_gaze but for L/R hand positions

Interpretation:
  - If mask_kept ≈ baseline → receivers carry no information; method is fake.
  - If mask_kept << baseline → receivers carry information; method works.
  - Comparing mask_kept_late vs mask_kept_early → which half drives accuracy.
  - shuffle_kept tests whether the *identity* of the merged tokens matters
    or only their aggregate statistics.
  - mask_gaze / mask_hand : directly probe whether selected receivers near the
    trajectory carry task-relevant info. Strongly negative Δ = method uses
    trajectory-anchored visual evidence (the goal of the trajectory-grounded plan).

Outputs:
  <tag>_mask_<variant>_per_sample.parquet  (one per variant)
  <tag>_mask_summary.json                  (aggregated)

Usage:
  python -m TrajGazeMerge.eval.counterfactual_mask_eval \
      --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth \
      --lora-ckpt   /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
      --tag E1_keep10_mask
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/trajgaze")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal, score_to_qwen_spatiotemporal,
)

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results"
ALL_VARIANTS = [
    "baseline", "mask_kept", "mask_kept_late", "mask_kept_early", "shuffle_kept",
    "mask_gaze", "mask_hand",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",    choices=["full", "gaze_only", "hand_only"], default="full")
    p.add_argument("--stage1-ckpt",   required=True)
    p.add_argument("--lora-ckpt",     required=True)
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--tag",           default="E1_keep10_mask")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--split",         default="test", choices=["test", "train"])
    p.add_argument("--val-dataset",   default="streamgaze",
                   choices=["streamgaze", "egovqa"])
    p.add_argument("--limit",         type=int, default=0)
    p.add_argument("--variants",      nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    p.add_argument("--mask-radius",   type=float, default=0.2,
                   help="Normalized radius for mask_gaze / mask_hand variants.")
    p.add_argument("--seed",          type=int, default=0)
    return p.parse_args()


def _trajectory_region_mask(
    receiver_idx: torch.Tensor,        # (N_keep,)
    T_merged: int, n_spatial: int,
    targets: np.ndarray,               # (T_traj, K, 2) — K anchor points per traj frame, NaN if invalid
    target_mask: np.ndarray,           # (T_traj, K) bool
    radius: float,
) -> np.ndarray:
    """
    Mark receivers whose patch (x, y) is within `radius` (normalized) of ANY
    valid target anchor at the same temporal location.

    Returns boolean array (N_keep,), True where receiver should be zeroed.
    """
    side = int(round(n_spatial ** 0.5))
    recv = receiver_idx.detach().cpu().numpy()
    recv_frame = recv // n_spatial          # 0 .. T_merged-1
    recv_patch = recv %  n_spatial
    recv_row   = recv_patch // side
    recv_col   = recv_patch %  side
    recv_x = (recv_col + 0.5) / side
    recv_y = (recv_row + 0.5) / side

    T_traj = targets.shape[0]
    # Proportional mapping T_merged → T_traj
    traj_frame = np.minimum(
        (recv_frame.astype(np.float32) * T_traj / max(1, T_merged)).astype(np.int64),
        T_traj - 1,
    )

    tgt   = targets[traj_frame]                 # (N_keep, K, 2)
    tmask = target_mask[traj_frame]             # (N_keep, K)
    dx    = tgt[..., 0] - recv_x[:, None]       # (N_keep, K)
    dy    = tgt[..., 1] - recv_y[:, None]
    dist  = np.sqrt(dx * dx + dy * dy)          # (N_keep, K)
    dist  = np.where(tmask, dist, np.inf)
    return dist.min(axis=1) < radius


def apply_mask_variant(
    variant: str,
    merged_video: torch.Tensor,        # (N_keep, d)
    receiver_idx: torch.Tensor,        # (N_keep,) indices into original n_video
    T_merged: int, n_spatial: int,
    item: dict,
    radius: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Return a modified merged_video for the given counterfactual variant."""
    if variant == "baseline":
        return merged_video
    if variant == "mask_kept":
        return torch.zeros_like(merged_video)
    if variant == "shuffle_kept":
        perm = torch.randperm(merged_video.shape[0], generator=rng, device=merged_video.device)
        return merged_video[perm]

    if variant in ("mask_kept_late", "mask_kept_early"):
        recv = receiver_idx.detach().cpu().numpy()
        frame_idx = recv // n_spatial
        half = T_merged // 2
        if variant == "mask_kept_late":
            keep_mask = frame_idx < half          # keep early, zero late
        else:
            keep_mask = frame_idx >= half         # keep late, zero early
        out = merged_video.clone()
        zero_rows = torch.from_numpy(~keep_mask).to(out.device)
        out[zero_rows] = 0.0
        return out

    if variant in ("mask_gaze", "mask_hand"):
        traj = item["traj"]
        if variant == "mask_gaze":
            pos  = traj["gaze_pos"].detach().cpu().numpy()        # (T_traj, 2)
            mask = traj["gaze_mask"].detach().cpu().numpy()       # (T_traj,)
            targets     = pos[:, None, :]                          # (T_traj, 1, 2)
            target_mask = mask[:, None]                            # (T_traj, 1)
        else:
            l_pos  = traj["left_pos"].detach().cpu().numpy()      # (T_traj, 2)
            r_pos  = traj["right_pos"].detach().cpu().numpy()
            l_mask = traj["left_mask"].detach().cpu().numpy()
            r_mask = traj["right_mask"].detach().cpu().numpy()
            targets     = np.stack([l_pos, r_pos], axis=1)         # (T_traj, 2, 2)
            target_mask = np.stack([l_mask, r_mask], axis=1)       # (T_traj, 2)

        zero_np = _trajectory_region_mask(
            receiver_idx, T_merged, n_spatial,
            targets.astype(np.float32),
            target_mask.astype(bool),
            radius,
        )
        out = merged_video.clone()
        zero_rows = torch.from_numpy(zero_np).to(out.device)
        out[zero_rows] = 0.0
        return out

    raise ValueError(variant)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = os.path.join(RESULTS_DIR, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[CFMask] tag={args.tag}  variants={args.variants}  merge_ratio={args.merge_ratio}")

    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    if os.path.exists(args.lora_ckpt):
        ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "lora_state" in ckpt:
            qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
    qwen_model.eval()

    traj_encoder = load_traj_encoder(
        args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes
    )
    if os.path.exists(args.lora_ckpt):
        merge_ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "encoder_state" in merge_ckpt:
            traj_encoder.load_state_dict(merge_ckpt["encoder_state"], strict=False)
    traj_encoder.eval()

    option_ids = get_option_ids(processor)

    if args.val_dataset == "egovqa":
        from TrajGazeMerge.data.dataset_egovqa import EgoGazeVQAMergeDataset
        ds = EgoGazeVQAMergeDataset(
            split=args.split, n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
            datasets=("egtea",) if args.split == "test" else ("ego4d", "egoexo"),
        )
    else:
        ds = StreamGazeMergeDataset(
            split=args.split, n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
        )
    print(f"  {args.split} items (val={args.val_dataset}): {len(ds)}")

    rng = torch.Generator(device=device).manual_seed(args.seed)
    rows_by_variant: dict[str, list[dict]] = {v: [] for v in args.variants}

    with torch.no_grad():
        for idx in range(len(ds)):
            if args.limit > 0 and idx >= args.limit:
                break
            item = ds[idx]
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

                scores_2d  = get_patch_scores_temporal(traj_encoder, item, device)
                scores_all = score_to_qwen_spatiotemporal(scores_2d, n_spatial, T_merged)
                if scores_all.shape[0] != n_video:
                    scores_all = (
                        scores_all[:n_video] if scores_all.shape[0] > n_video
                        else scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]
                    )

                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r
                )
                gt_letter = item["answer"]

                for variant in args.variants:
                    mv = apply_mask_variant(
                        variant, merged_video, receiver_idx, T_merged, n_spatial,
                        item, args.mask_radius, rng,
                    )
                    logits = forward_logits(
                        qwen_model,
                        build_merged_inputs(base_qwen, cached, mv, receiver_idx),
                    )
                    opt_logits = logits[option_ids].float().cpu()
                    opt_probs  = F.softmax(opt_logits, dim=0)
                    pred_idx   = int(opt_logits.argmax().item())
                    pred_letter = "ABCD"[pred_idx]
                    correct = (pred_letter == gt_letter)

                    rows_by_variant[variant].append({
                        "idx": idx,
                        "task": item.get("task", "unknown"),
                        "answer": gt_letter,
                        "prediction": pred_letter,
                        "correct": bool(correct),
                        "top1_prob": float(opt_probs.max().item()),
                        "logit_margin": float(
                            (opt_logits.sort(descending=True).values[0]
                             - opt_logits.sort(descending=True).values[1]).item()
                        ),
                    })

                if (idx + 1) % 25 == 0:
                    parts = []
                    for v in args.variants:
                        rs = rows_by_variant[v]
                        a = 100.0 * sum(r["correct"] for r in rs) / max(1, len(rs))
                        parts.append(f"{v}={a:.1f}%")
                    print(f"  [{idx+1}/{len(ds)}] " + " | ".join(parts))

            except Exception:
                traceback.print_exc()
                continue

    summary: dict = {
        "tag": args.tag,
        "merge_ratio": args.merge_ratio,
        "mask_radius": args.mask_radius,
        "variants": {},
    }
    for variant, rows in rows_by_variant.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"{args.tag}_mask_{variant}_per_sample.parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".jsonl")
            df.to_json(path, orient="records", lines=True)
        acc = 100.0 * df["correct"].mean()
        per_task = (
            df.groupby("task")["correct"].agg(["mean", "count"]).reset_index()
              .rename(columns={"mean": "accuracy", "count": "n"})
        )
        per_task["accuracy"] = per_task["accuracy"] * 100
        summary["variants"][variant] = {
            "n": int(len(df)),
            "overall_accuracy": float(acc),
            "mean_top1_prob":   float(df["top1_prob"].mean()),
            "mean_logit_margin": float(df["logit_margin"].mean()),
            "per_task": per_task.to_dict(orient="records"),
            "path": path,
        }

    # Print comparison table
    summary_path = os.path.join(out_dir, f"{args.tag}_mask_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Counterfactual masking [{args.tag}]")
    print(f"{'='*70}")
    print(f"  {'variant':<18s} {'n':>5s} {'acc%':>8s} {'top1':>8s} {'lmrg':>8s}")
    for variant in args.variants:
        if variant not in summary["variants"]:
            continue
        s = summary["variants"][variant]
        print(f"  {variant:<18s} {s['n']:>5d} {s['overall_accuracy']:>8.2f} "
              f"{s['mean_top1_prob']:>8.3f} {s['mean_logit_margin']:>8.3f}")
    print(f"\n  summary -> {summary_path}")


if __name__ == "__main__":
    main()
