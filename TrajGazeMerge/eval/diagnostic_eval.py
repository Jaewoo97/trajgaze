"""
Per-example diagnostic logger for TrajGazeMerge.

Wraps the same eval path used in train_merge_lora_temporal_no_kd.evaluate()
but logs rich per-sample metadata to a parquet file for downstream analysis:

  - prediction, ground truth, correctness
  - full 4-option logits + softmax probs + logit margin
  - score distribution stats (mean/std/max/entropy) at TrajGaze 14x14 resolution
  - temporal distribution of kept tokens (per-frame count, center-of-mass,
    entropy, late-half ratio) at Qwen T_merged resolution
  - spatial distribution of kept tokens (center-of-mass, entropy)
  - merge cluster statistics (sources/receiver count, source->receiver cosine)
  - GT-gaze recall@keep_ratio: fraction of valid gaze frames whose 8x8 patch
    falls inside the kept set for that Qwen frame

Output:
  RESULTS_DIR/diagnostic/<tag>_per_sample.parquet   (one row per sample)
  RESULTS_DIR/diagnostic/<tag>_summary.json         (overall + per-task acc)

Usage:
  python -m TrajGazeMerge.eval.diagnostic_eval \
      --model-type full \
      --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/stage1_temporal/best.pth \
      --lora-ckpt   /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
      --merge-ratio 0.9 \
      --tag E1_keep10_diag
"""

from __future__ import annotations

import argparse
import json
import math
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",    choices=["full", "gaze_only", "hand_only"], default="full")
    p.add_argument("--stage1-ckpt",   required=True)
    p.add_argument("--lora-ckpt",     required=True)
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--tag",           default="E1_keep10_diag")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--split",         default="test", choices=["test", "train"])
    p.add_argument("--limit",         type=int, default=0, help="if >0, only run first N items (debug)")
    return p.parse_args()


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.float().flatten()
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    return float(-(p * (p + eps).log()).sum().item())


def _score_stats(scores_2d: torch.Tensor) -> dict:
    """scores_2d: (T_traj, 196). Per-sample summary statistics."""
    s = scores_2d.float().flatten()
    # Normalize to probability for entropy
    s_pos = (s - s.min()).clamp(min=0)
    return {
        "score_mean": float(s.mean().item()),
        "score_std":  float(s.std().item()),
        "score_max":  float(s.max().item()),
        "score_min":  float(s.min().item()),
        "score_entropy": _entropy(s_pos),
    }


def _temporal_stats(kept_per_frame: np.ndarray) -> dict:
    """kept_per_frame: (T_merged,) ints — number of kept tokens per Qwen frame."""
    T = len(kept_per_frame)
    total = int(kept_per_frame.sum())
    if total == 0:
        return {
            "temporal_center_of_mass": float("nan"),
            "temporal_entropy": 0.0,
            "late_half_ratio": float("nan"),
            "first_half_ratio": float("nan"),
            "kept_total": 0,
        }
    p = kept_per_frame.astype(np.float64) / total
    idx = np.arange(T, dtype=np.float64) / max(1, T - 1)        # 0..1
    com = float((p * idx).sum())
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    late = float(kept_per_frame[T // 2:].sum() / total)
    early = float(kept_per_frame[: T // 2].sum() / total)
    return {
        "temporal_center_of_mass": com,
        "temporal_entropy": ent,
        "late_half_ratio": late,
        "first_half_ratio": early,
        "kept_total": total,
    }


def _spatial_stats(kept_per_spatial: np.ndarray, side: int) -> dict:
    """kept_per_spatial: (n_spatial,) ints — kept-token count summed across frames per spatial position."""
    total = int(kept_per_spatial.sum())
    if total == 0:
        return {"spatial_com_x": float("nan"), "spatial_com_y": float("nan"), "spatial_entropy": 0.0}
    grid = kept_per_spatial.reshape(side, side).astype(np.float64)
    p = grid / total
    ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    com_x = float((p * xs).sum() / max(1, side - 1))
    com_y = float((p * ys).sum() / max(1, side - 1))
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {"spatial_com_x": com_x, "spatial_com_y": com_y, "spatial_entropy": ent}


def _gt_pos_recall(pos: np.ndarray, mask: np.ndarray, T_merged: int, side: int,
                   kept_mask: np.ndarray, prefix: str) -> dict:
    """Generic position-recall: fraction of valid frames whose position's
    `side x side` patch falls inside the kept (receiver) mask."""
    T_traj = pos.shape[0]
    if T_traj == 0 or not mask.any():
        return {f"{prefix}_recall": float("nan"), f"{prefix}_n_valid": 0,
                f"{prefix}_mean_dist": float("nan")}
    t_qwen = np.clip((np.arange(T_traj) * T_merged / max(1, T_traj)).astype(int), 0, T_merged - 1)
    x_idx = np.clip(np.floor(pos[:, 0] * side).astype(int), 0, side - 1)
    y_idx = np.clip(np.floor(pos[:, 1] * side).astype(int), 0, side - 1)
    spatial_idx = y_idx * side + x_idx
    valid = mask
    hit = kept_mask[t_qwen[valid], spatial_idx[valid]]

    # Mean distance (in [0,1] normalized coords) from gt position to nearest kept patch
    # Build per-Qwen-frame kept (x,y) sets once
    dists: list[float] = []
    keep_x = np.where(kept_mask)[1] % side
    keep_y = np.where(kept_mask)[1] // side
    keep_t = np.where(kept_mask)[0]
    for t_q, gx, gy, v in zip(t_qwen[valid], pos[valid, 0], pos[valid, 1], valid[valid]):
        idx_t = keep_t == t_q
        if not idx_t.any():
            continue
        xs = keep_x[idx_t] / max(1, side - 1)
        ys = keep_y[idx_t] / max(1, side - 1)
        d2 = (xs - gx) ** 2 + (ys - gy) ** 2
        dists.append(float(np.sqrt(d2.min())))
    return {
        f"{prefix}_recall": float(hit.mean()) if hit.size > 0 else float("nan"),
        f"{prefix}_n_valid": int(valid.sum()),
        f"{prefix}_mean_dist": float(np.mean(dists)) if dists else float("nan"),
    }


def _trajectory_recall_stats(traj, T_merged: int, side: int, kept_mask: np.ndarray) -> dict:
    """Computes recall + mean-distance metrics for gaze, left_hand, right_hand,
    and hand-midpoint positions w.r.t. the kept-token mask."""
    out = {}
    out.update(_gt_pos_recall(
        traj["gaze_pos"].cpu().numpy(), traj["gaze_mask"].cpu().numpy().astype(bool),
        T_merged, side, kept_mask, "gt_gaze",
    ))
    out.update(_gt_pos_recall(
        traj["left_pos"].cpu().numpy(), traj["left_mask"].cpu().numpy().astype(bool),
        T_merged, side, kept_mask, "gt_hand_left",
    ))
    out.update(_gt_pos_recall(
        traj["right_pos"].cpu().numpy(), traj["right_mask"].cpu().numpy().astype(bool),
        T_merged, side, kept_mask, "gt_hand_right",
    ))
    # Hand midpoint
    lp = traj["left_pos"].cpu().numpy(); rp = traj["right_pos"].cpu().numpy()
    lm = traj["left_mask"].cpu().numpy().astype(bool)
    rm = traj["right_mask"].cpu().numpy().astype(bool)
    both = lm & rm
    mid = np.zeros_like(lp)
    mid[both] = (lp[both] + rp[both]) / 2
    out.update(_gt_pos_recall(mid, both, T_merged, side, kept_mask, "gt_hand_mid"))
    # Frame center reference
    T_traj = lp.shape[0]
    ctr = np.full_like(lp, 0.5)
    out.update(_gt_pos_recall(ctr, np.ones(T_traj, dtype=bool), T_merged, side, kept_mask, "frame_center"))
    # Either hand (left OR right)
    either_pos = np.where(lm[:, None], lp, rp)
    either_mask = lm | rm
    out.update(_gt_pos_recall(either_pos, either_mask, T_merged, side, kept_mask, "gt_hand_either"))
    return out


def _gt_gaze_recall(traj, T_merged: int, side: int, kept_per_frame_spatial: np.ndarray) -> dict:
    """
    Fraction of valid gaze frames whose Qwen spatial patch (side x side) is in the
    receiver (kept) set for the corresponding Qwen frame.

    kept_per_frame_spatial: (T_merged, n_spatial) bool — receiver mask.
    """
    gaze_pos = traj["gaze_pos"].cpu().numpy()    # (T_traj, 2) in [0,1] (x, y)
    gaze_mask = traj["gaze_mask"].cpu().numpy().astype(bool)
    T_traj = gaze_pos.shape[0]
    if T_traj == 0 or not gaze_mask.any():
        return {"gt_gaze_recall": float("nan"), "gt_gaze_n_valid": 0}

    # Map each valid gaze frame -> Qwen frame index via linear scaling
    t_qwen = np.clip((np.arange(T_traj) * T_merged / max(1, T_traj)).astype(int), 0, T_merged - 1)
    x_idx = np.clip(np.floor(gaze_pos[:, 0] * side).astype(int), 0, side - 1)
    y_idx = np.clip(np.floor(gaze_pos[:, 1] * side).astype(int), 0, side - 1)
    spatial_idx = y_idx * side + x_idx

    valid = gaze_mask
    hit = kept_per_frame_spatial[t_qwen[valid], spatial_idx[valid]]
    return {
        "gt_gaze_recall": float(hit.mean()) if hit.size > 0 else float("nan"),
        "gt_gaze_n_valid": int(valid.sum()),
    }


def _merge_stats(stats: dict, n_video: int) -> dict:
    """Cluster size distribution + similarity stats from merge stats dict."""
    best_match = stats["best_match"].cpu().numpy()
    n_recv = n_video - len(best_match)
    cluster_sizes = np.bincount(best_match, minlength=n_recv) + 1   # +1 for the receiver itself
    sim_max = stats["sim_max"].cpu().float().numpy()
    return {
        "cluster_size_mean": float(cluster_sizes.mean()),
        "cluster_size_max":  int(cluster_sizes.max()),
        "cluster_size_std":  float(cluster_sizes.std()),
        "src_recv_cos_mean": float(sim_max.mean()) if sim_max.size > 0 else float("nan"),
        "src_recv_cos_min":  float(sim_max.min())  if sim_max.size > 0 else float("nan"),
    }


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = os.path.join(RESULTS_DIR, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Diagnostic] tag={args.tag}  split={args.split}  merge_ratio={args.merge_ratio}")
    print(f"  stage1_ckpt: {args.stage1_ckpt}")
    print(f"  lora_ckpt:   {args.lora_ckpt}")

    # ── Load Qwen + LoRA ──────────────────────────────────────────────────────
    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    if os.path.exists(args.lora_ckpt):
        ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "lora_state" in ckpt:
            qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
            print(f"  Loaded LoRA state from: {args.lora_ckpt}")
    qwen_model.eval()

    # ── Load TrajGaze encoder (E1 patch_temporal inferred from ckpt keys) ─────
    traj_encoder = load_traj_encoder(
        args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes
    )
    # If lora ckpt also has encoder_state, use it (matches training behavior)
    if os.path.exists(args.lora_ckpt):
        merge_ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "encoder_state" in merge_ckpt:
            traj_encoder.load_state_dict(merge_ckpt["encoder_state"], strict=False)
            print(f"  Loaded encoder_state from merge ckpt")
    traj_encoder.eval()

    option_ids = get_option_ids(processor)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = StreamGazeMergeDataset(
        split=args.split, n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    print(f"  {args.split} items: {len(ds)}")

    rows: list[dict] = []
    n_correct = 0
    n_total = 0

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
                side      = int(round(math.sqrt(n_spatial)))
                r         = max(1, int(args.merge_ratio * n_video))

                # ── TrajGaze scores ───────────────────────────────────────────
                scores_2d  = get_patch_scores_temporal(traj_encoder, item, device)  # (T_traj, 196)
                scores_all = score_to_qwen_spatiotemporal(scores_2d, n_spatial, T_merged)
                if scores_all.shape[0] != n_video:
                    scores_all = (
                        scores_all[:n_video] if scores_all.shape[0] > n_video
                        else scores_all.repeat(
                            (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        )[:n_video]
                    )

                merged_video, receiver_idx, merge_st = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r, return_stats=True
                )

                # ── Forward pass ──────────────────────────────────────────────
                logits = forward_logits(
                    qwen_model,
                    build_merged_inputs(base_qwen, cached, merged_video, receiver_idx),
                )
                opt_logits = logits[option_ids].float().cpu()              # (4,)
                opt_probs  = F.softmax(opt_logits, dim=0)
                pred_idx   = int(opt_logits.argmax().item())
                pred_char  = "ABCD"[pred_idx]
                gt_char    = item["answer"]
                correct    = (pred_char == gt_char)

                sorted_l = opt_logits.sort(descending=True).values
                logit_margin = float((sorted_l[0] - sorted_l[1]).item())
                top1_prob = float(opt_probs.max().item())

                # ── Receiver mask (T_merged, n_spatial) ───────────────────────
                recv = receiver_idx.detach().cpu().numpy()
                recv_frame = recv // n_spatial
                recv_pos   = recv % n_spatial
                kept_per_frame = np.bincount(recv_frame, minlength=T_merged)         # (T_merged,)
                kept_per_pos   = np.bincount(recv_pos,   minlength=n_spatial)        # (n_spatial,)
                kept_mask = np.zeros((T_merged, n_spatial), dtype=bool)
                kept_mask[recv_frame, recv_pos] = True

                # ── Stats ─────────────────────────────────────────────────────
                sst = _score_stats(scores_2d)
                tst = _temporal_stats(kept_per_frame)
                spt = _spatial_stats(kept_per_pos, side)
                mst = _merge_stats(merge_st, n_video)
                gst = _gt_gaze_recall(item["traj"], T_merged, side, kept_mask)
                # M1.1 + M1.3: hand & midpoint recall + distance to gaze/hand/center
                rst = _trajectory_recall_stats(item["traj"], T_merged, side, kept_mask)

                row = {
                    "idx": idx,
                    "task": item.get("task", "unknown"),
                    "dataset": item.get("dataset", "unknown"),
                    "question": item["question"][:300],
                    "answer": gt_char,
                    "prediction": pred_char,
                    "correct": bool(correct),
                    "n_video": int(n_video),
                    "T_merged": int(T_merged),
                    "n_spatial": int(n_spatial),
                    "merge_r": int(r),
                    "keep_count": int(n_video - r),
                    # logits
                    "logit_A": float(opt_logits[0]),
                    "logit_B": float(opt_logits[1]),
                    "logit_C": float(opt_logits[2]),
                    "logit_D": float(opt_logits[3]),
                    "prob_A": float(opt_probs[0]),
                    "prob_B": float(opt_probs[1]),
                    "prob_C": float(opt_probs[2]),
                    "prob_D": float(opt_probs[3]),
                    "logit_margin": logit_margin,
                    "top1_prob": top1_prob,
                    # arrays as lists (parquet-friendly)
                    "kept_per_frame": kept_per_frame.tolist(),
                    # stats
                    **sst, **tst, **spt, **mst, **gst, **rst,
                }
                rows.append(row)
                n_correct += int(correct)
                n_total += 1

                if (idx + 1) % 50 == 0:
                    print(f"  [{idx+1}/{len(ds)}] acc={100*n_correct/max(1,n_total):.2f}%")

            except Exception:
                traceback.print_exc()
                continue

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    parquet_path = os.path.join(out_dir, f"{args.tag}_per_sample.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"  parquet write failed ({e}); falling back to jsonl")
        parquet_path = os.path.join(out_dir, f"{args.tag}_per_sample.jsonl")
        df.to_json(parquet_path, orient="records", lines=True)
    print(f"  saved {len(df)} rows -> {parquet_path}")

    # Summary
    overall = 100.0 * n_correct / max(1, n_total)
    per_task = (
        df.groupby("task")["correct"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_samples"})
    )
    per_task["accuracy"] = per_task["accuracy"] * 100
    summary = {
        "tag": args.tag,
        "split": args.split,
        "merge_ratio": args.merge_ratio,
        "n_total": n_total,
        "n_correct": n_correct,
        "overall_accuracy": overall,
        "per_task": per_task.to_dict(orient="records"),
        "global_means": {
            "logit_margin": float(df["logit_margin"].mean()) if len(df) else float("nan"),
            "top1_prob":    float(df["top1_prob"].mean())    if len(df) else float("nan"),
            "temporal_center_of_mass": float(df["temporal_center_of_mass"].mean()) if len(df) else float("nan"),
            "late_half_ratio":         float(df["late_half_ratio"].mean())         if len(df) else float("nan"),
            "gt_gaze_recall":          float(df["gt_gaze_recall"].mean(skipna=True)) if len(df) else float("nan"),
            "cluster_size_mean":       float(df["cluster_size_mean"].mean())       if len(df) else float("nan"),
        },
    }
    summary_path = os.path.join(out_dir, f"{args.tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Diagnostic Eval [{args.tag}]")
    print(f"{'='*60}")
    print(f"  Overall: {overall:.2f}%  (n={n_total})")
    print(f"  Mean logit margin: {summary['global_means']['logit_margin']:.3f}")
    print(f"  Mean top1 prob:    {summary['global_means']['top1_prob']:.3f}")
    print(f"  Mean temporal CoM: {summary['global_means']['temporal_center_of_mass']:.3f} (0.5 = uniform)")
    print(f"  Mean late_half:    {summary['global_means']['late_half_ratio']:.3f} (0.5 = uniform)")
    print(f"  Mean gt_gaze_recall: {summary['global_means']['gt_gaze_recall']:.3f}")
    print(f"  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
