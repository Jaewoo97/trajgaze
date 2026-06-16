"""
Phase 0b-1: Frozen Qwen + method eval.

Compares three conditions on the same 526 EGTEA test:
  - baseline_frozen      : frozen Qwen, full tokens (no method, zero-shot)
  - merge_frozen_learned : frozen Qwen + learned-score merge (method without LoRA)
  - (reference) lora_learned : already measured = 67.68% baseline from M1

Question answered: does the method work without LoRA fine-tuning, or does
LoRA absorb everything?
  - If merge_frozen_learned > baseline_frozen → method has value even on frozen LLM
  - If merge_frozen_learned ≈ baseline_frozen → method only works *with* LoRA

This is the cheapest decisive cross-architecture sanity check.

Usage:
  python -m TrajGazeMerge.eval.frozen_method_eval \
      --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth \
      --tag frozen_method
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/trajgaze")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_frozen, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal, score_to_qwen_spatiotemporal,
)

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",   required=True)
    p.add_argument("--lora-ckpt",     default=None,
                   help="Optional: load encoder_state from this ckpt (matches "
                        "the Stage 2 LoRA run's fine-tuned encoder).")
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--tag",           default="frozen_method")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--split",         default="test")
    p.add_argument("--limit",         type=int, default=0)
    p.add_argument("--conditions",    nargs="+",
                   default=["baseline_frozen", "merge_frozen"],
                   choices=["baseline_frozen", "merge_frozen"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[FrozenMethod] tag={args.tag}  conditions={args.conditions}  mr={args.merge_ratio}")

    processor, qwen_model = load_qwen_frozen(device)
    base_qwen = qwen_model           # frozen base, no PEFT wrapper

    need_encoder = "merge_frozen" in args.conditions
    traj_encoder = None
    if need_encoder:
        traj_encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
        # Optionally load Stage 2 fine-tuned encoder state (matches production setting)
        if args.lora_ckpt and os.path.exists(args.lora_ckpt):
            merge_ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
            if "encoder_state" in merge_ckpt:
                traj_encoder.load_state_dict(merge_ckpt["encoder_state"], strict=False)
                print(f"  Loaded encoder_state from {args.lora_ckpt}")
        traj_encoder.eval()

    option_ids = get_option_ids(processor)

    ds = StreamGazeMergeDataset(
        split=args.split, n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    print(f"  {args.split} items: {len(ds)}")

    rows_by_cond: dict[str, list[dict]] = {c: [] for c in args.conditions}

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
                gt_letter = item["answer"]

                for cond in args.conditions:
                    if cond == "baseline_frozen":
                        inputs_dict = build_full_inputs(base_qwen, cached)
                    else:  # merge_frozen
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
                        inputs_dict = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)

                    logits = forward_logits(qwen_model, inputs_dict)
                    opt_logits = logits[option_ids].float().cpu()
                    opt_probs  = F.softmax(opt_logits, dim=0)
                    pred_letter = "ABCD"[int(opt_logits.argmax().item())]
                    correct = (pred_letter == gt_letter)

                    rows_by_cond[cond].append({
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
                    for c in args.conditions:
                        rs = rows_by_cond[c]
                        a = 100.0 * sum(r["correct"] for r in rs) / max(1, len(rs))
                        parts.append(f"{c}={a:.1f}%")
                    print(f"  [{idx+1}/{len(ds)}] " + " | ".join(parts))
            except Exception:
                traceback.print_exc()
                continue

    # Save
    summary = {"tag": args.tag, "merge_ratio": args.merge_ratio, "conditions": {}}
    for c, rows in rows_by_cond.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = os.path.join(RESULTS_DIR, f"{args.tag}_{c}_per_sample.parquet")
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
        summary["conditions"][c] = {
            "n": int(len(df)),
            "overall_accuracy": float(acc),
            "mean_top1_prob":   float(df["top1_prob"].mean()),
            "per_task": per_task.to_dict(orient="records"),
            "path": path,
        }

    summary_path = os.path.join(RESULTS_DIR, f"{args.tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Frozen-method eval [{args.tag}]")
    print(f"{'='*70}")
    print(f"  {'condition':<22s} {'n':>5s} {'acc%':>8s} {'top1':>8s}")
    for c in args.conditions:
        if c not in summary["conditions"]:
            continue
        s = summary["conditions"][c]
        print(f"  {c:<22s} {s['n']:>5d} {s['overall_accuracy']:>8.2f} {s['mean_top1_prob']:>8.3f}")
    if "baseline_frozen" in summary["conditions"] and "merge_frozen" in summary["conditions"]:
        d = summary["conditions"]["merge_frozen"]["overall_accuracy"] - summary["conditions"]["baseline_frozen"]["overall_accuracy"]
        print(f"\n  Δ (merge_frozen − baseline_frozen): {d:+.2f}pp")
        print("    > 0 ⇒ method has value even without LoRA")
        print("    ≈ 0 ⇒ LoRA absorbs the method contribution")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
