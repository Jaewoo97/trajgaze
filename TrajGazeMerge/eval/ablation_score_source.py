"""
Counterfactual ablation: swap the source of the per-token score while keeping
the rest of the TrajGazeMerge pipeline identical, and report per-task accuracy.

Score sources:
  learned   : TrajGaze encoder output (baseline)
  uniform   : all tokens equal weight (cosine matching still active)
  random    : torch.rand
  inverted  : -learned  (worst tokens become receivers)
  center    : 2D Gaussian centered on 8x8 spatial grid (egocentric prior)
  oracle    : 1 at the patch containing GT gaze (per Qwen frame), 0 elsewhere
              — falls back to uniform when no gaze frames are valid
  text_only : zero out merged video embeddings (visual ablation, not score)

If learned > {uniform, random, center}, the architecture truly contributes.
If learned ≈ inverted, the score signal is unused.
If learned ≈ text_only, visual tokens carry no information.

Output:
  <tag>_ablation_<source>_per_sample.parquet
  <tag>_ablation_summary.json   (single file, all sources aggregated)

Usage:
  python -m TrajGazeMerge.eval.ablation_score_source \
      --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth \
      --lora-ckpt   /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
      --tag E1_keep10_abl \
      --sources learned uniform random inverted center oracle text_only
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
ALL_SOURCES = ["learned", "uniform", "random", "inverted", "center", "oracle", "text_only"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",    choices=["full", "gaze_only", "hand_only"], default="full")
    p.add_argument("--stage1-ckpt",   required=True)
    p.add_argument("--lora-ckpt",     required=True)
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--tag",           default="E1_keep10_abl")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--split",         default="test", choices=["test", "train"])
    p.add_argument("--limit",         type=int, default=0)
    p.add_argument("--sources",       nargs="+", default=ALL_SOURCES,
                   choices=ALL_SOURCES)
    p.add_argument("--seed",          type=int, default=0)
    return p.parse_args()


def _center_gaussian_scores(T_merged: int, side: int, sigma_frac: float = 0.25,
                            device: torch.device = "cpu") -> torch.Tensor:
    """Per-frame 2D Gaussian centered at (0.5, 0.5) of the spatial grid.
    Returns flat (T_merged * n_spatial,) on device."""
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    cx = cy = (side - 1) / 2
    sigma = sigma_frac * side
    g = torch.exp(-((x.float() - cx) ** 2 + (y.float() - cy) ** 2) / (2 * sigma ** 2))
    g = g.flatten().to(device)
    return g.unsqueeze(0).expand(T_merged, -1).reshape(-1)


def _oracle_scores(traj, T_merged: int, side: int, device: torch.device) -> torch.Tensor:
    """1.0 at the 8x8 patch containing GT gaze, 0 elsewhere per Qwen frame.
    Per-frame fallback to uniform when no valid gaze in that frame's gaze window."""
    gaze_pos = traj["gaze_pos"].cpu().numpy()
    gaze_mask = traj["gaze_mask"].cpu().numpy().astype(bool)
    T_traj = gaze_pos.shape[0]
    n_spatial = side * side
    scores = np.zeros((T_merged, n_spatial), dtype=np.float32)

    if T_traj == 0:
        scores += 1.0 / n_spatial
        return torch.from_numpy(scores).to(device).reshape(-1)

    # Assign each traj frame to a Qwen frame
    t_qwen = np.clip((np.arange(T_traj) * T_merged / max(1, T_traj)).astype(int), 0, T_merged - 1)
    x_idx = np.clip(np.floor(gaze_pos[:, 0] * side).astype(int), 0, side - 1)
    y_idx = np.clip(np.floor(gaze_pos[:, 1] * side).astype(int), 0, side - 1)
    spatial = y_idx * side + x_idx

    # Vote: accumulate per Qwen frame
    for tq, sp, valid in zip(t_qwen, spatial, gaze_mask):
        if valid:
            scores[tq, sp] += 1.0

    # Fallback for any frame with no valid gaze
    no_signal = scores.sum(axis=1) == 0
    if no_signal.any():
        scores[no_signal] = 1.0 / n_spatial
    return torch.from_numpy(scores).to(device).reshape(-1)


def _build_scores_for_source(
    source: str,
    learned_2d: torch.Tensor,         # (T_traj, 196)
    n_video: int, T_merged: int, n_spatial: int,
    item: dict, device: torch.device, rng: torch.Generator,
) -> torch.Tensor:
    """Return a (n_video,) score tensor for the requested source.
    For 'text_only' returns None — caller handles separately."""
    side = int(round(math.sqrt(n_spatial)))

    if source == "learned":
        scores_all = score_to_qwen_spatiotemporal(learned_2d, n_spatial, T_merged)
    elif source == "uniform":
        scores_all = torch.ones(T_merged * n_spatial, device=device)
    elif source == "random":
        scores_all = torch.rand(T_merged * n_spatial, device=device, generator=rng)
    elif source == "inverted":
        # invert the learned score; shift to non-negative so weighted merge still works
        s = score_to_qwen_spatiotemporal(learned_2d, n_spatial, T_merged)
        s = s.max() - s + 1e-3
        scores_all = s
    elif source == "center":
        scores_all = _center_gaussian_scores(T_merged, side, device=device)
    elif source == "oracle":
        scores_all = _oracle_scores(item["traj"], T_merged, side, device)
    elif source == "text_only":
        # Caller handles: zero out merged_video and ignore score path
        return None
    else:
        raise ValueError(source)

    if scores_all.shape[0] != n_video:
        scores_all = (scores_all[:n_video] if scores_all.shape[0] > n_video
                      else scores_all.repeat((n_video + scores_all.shape[0] - 1) // scores_all.shape[0])[:n_video])
    return scores_all


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = os.path.join(RESULTS_DIR, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Ablation] tag={args.tag}  sources={args.sources}  merge_ratio={args.merge_ratio}")

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

    ds = StreamGazeMergeDataset(
        split=args.split, n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    print(f"  {args.split} items: {len(ds)}")

    rng = torch.Generator(device=device).manual_seed(args.seed)

    # Per-source row buffers
    rows_by_source: dict[str, list[dict]] = {s: [] for s in args.sources}

    with torch.no_grad():
        for idx in range(len(ds)):
            if args.limit > 0 and idx >= args.limit:
                break
            item = ds[idx]
            if item is None:
                continue
            try:
                # Preprocess once: visual features + tokenization are shared across sources
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

                # Learned scores (computed once if any source needs them)
                need_learned = any(s in ("learned", "inverted") for s in args.sources)
                learned_2d = (
                    get_patch_scores_temporal(traj_encoder, item, device)
                    if need_learned else None
                )

                # GT for shared bookkeeping
                gt_letter = item["answer"]

                for src in args.sources:
                    if src == "text_only":
                        # Build the same merge structure as learned but zero the embeddings
                        # to preserve seq length / RoPE structure with no visual content.
                        scores_all = (
                            torch.ones(n_video, device=device)
                            if learned_2d is None else
                            _build_scores_for_source("uniform", None, n_video, T_merged, n_spatial, item, device, rng)
                        )
                        merged_video, receiver_idx = gaze_weighted_merge(
                            cached["video_embeds"], scores_all, r
                        )
                        merged_video = torch.zeros_like(merged_video)
                    else:
                        scores_all = _build_scores_for_source(
                            src, learned_2d, n_video, T_merged, n_spatial, item, device, rng
                        )
                        merged_video, receiver_idx = gaze_weighted_merge(
                            cached["video_embeds"], scores_all, r
                        )

                    logits = forward_logits(
                        qwen_model,
                        build_merged_inputs(base_qwen, cached, merged_video, receiver_idx),
                    )
                    opt_logits = logits[option_ids].float().cpu()
                    opt_probs  = F.softmax(opt_logits, dim=0)
                    pred_idx   = int(opt_logits.argmax().item())
                    pred_letter = "ABCD"[pred_idx]
                    correct = (pred_letter == gt_letter)

                    rows_by_source[src].append({
                        "idx": idx,
                        "task": item.get("task", "unknown"),
                        "dataset": item.get("dataset", "unknown"),
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
                    for src in args.sources:
                        rs = rows_by_source[src]
                        a = 100.0 * sum(r["correct"] for r in rs) / max(1, len(rs))
                        parts.append(f"{src}={a:.1f}%")
                    print(f"  [{idx+1}/{len(ds)}] " + " | ".join(parts))

            except Exception:
                traceback.print_exc()
                continue

    # ── Save per-source parquets ──────────────────────────────────────────────
    summary: dict = {"tag": args.tag, "merge_ratio": args.merge_ratio, "sources": {}}
    for src, rows in rows_by_source.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"{args.tag}_ablation_{src}_per_sample.parquet")
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
        summary["sources"][src] = {
            "n": int(len(df)),
            "overall_accuracy": float(acc),
            "mean_top1_prob":   float(df["top1_prob"].mean()),
            "mean_logit_margin": float(df["logit_margin"].mean()),
            "per_task": per_task.to_dict(orient="records"),
            "path": path,
        }

    summary_path = os.path.join(out_dir, f"{args.tag}_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Counterfactual ablation [{args.tag}]")
    print(f"{'='*70}")
    print(f"  {'source':<12s} {'n':>5s} {'acc%':>8s} {'mean_top1':>10s} {'logit_mrg':>10s}")
    for src in args.sources:
        if src not in summary["sources"]:
            continue
        s = summary["sources"][src]
        print(f"  {src:<12s} {s['n']:>5d} {s['overall_accuracy']:>8.2f} "
              f"{s['mean_top1_prob']:>10.3f} {s['mean_logit_margin']:>10.3f}")
    print(f"\n  summary -> {summary_path}")


if __name__ == "__main__":
    main()
