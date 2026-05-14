"""
Option-permutation evaluation: cycle the A/B/C/D positions and re-evaluate
to separate "true understanding" from option-letter bias / position bias.

For each test sample we run 4 evaluations with cyclic shifts k ∈ {0, 1, 2, 3}:
  shifted_options[i] = original[(i + k) % 4]   (re-prefixed with "ABCD"[i].")
  new_answer_letter  = "ABCD"[(orig_answer_idx - k) % 4]

A model that truly understands should answer the semantically same option across
all 4 shifts (`agree4 = 1`); a guesser converges to 25%.

Outputs (under eval_results/diagnostic/):
  <tag>_permutation.parquet  — one row per (sample, shift) with prediction/correct
  <tag>_permutation_summary.json — agree4/agree2 rates, per-task accuracy by shift,
                                   pick-frequency table (ABCD), confidence vs agree

Usage:
  python -m TrajGazeMerge.eval.option_permutation_eval \
      --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth \
      --lora-ckpt   /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
      --tag E1_keep10_perm

Notes:
- Reuses preprocess_item / get_patch_scores_temporal / score_to_qwen_spatiotemporal
  exactly as in diagnostic_eval.py, but re-tokenizes the prompt for each shift.
- Video features are computed by the frozen Qwen ViT inside preprocess_item, so
  we DON'T re-extract them per shift — caching ve from shift 0 would require
  changes to preprocess_item. The current approach re-runs the ViT 4× but is
  simple and correct. Total wallclock ≈ 4× of diagnostic_eval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

_PREFIX_RE = re.compile(r"^\s*([A-D])\s*[\.\):]\s*")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",    choices=["full", "gaze_only", "hand_only"], default="full")
    p.add_argument("--stage1-ckpt",   required=True)
    p.add_argument("--lora-ckpt",     required=True)
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--tag",           default="E1_keep10_perm")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--split",         default="test", choices=["test", "train"])
    p.add_argument("--limit",         type=int, default=0)
    return p.parse_args()


def strip_prefix(opt: str) -> str:
    return _PREFIX_RE.sub("", opt).strip()


def shift_options(options: list[str], k: int) -> list[str]:
    """Cyclic shift: new position i gets the content originally at position (i+k)%4."""
    texts = [strip_prefix(o) for o in options]
    return [f"{'ABCD'[i]}. {texts[(i + k) % 4]}" for i in range(4)]


def shifted_answer(orig_letter: str, k: int) -> str:
    c = "ABCD".index(orig_letter)
    return "ABCD"[(c - k) % 4]


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = os.path.join(RESULTS_DIR, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[OptionPermutation] tag={args.tag}  split={args.split}  merge_ratio={args.merge_ratio}")

    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    if os.path.exists(args.lora_ckpt):
        ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "lora_state" in ckpt:
            qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
            print(f"  Loaded LoRA from: {args.lora_ckpt}")
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

    rows: list[dict] = []
    with torch.no_grad():
        for idx in range(len(ds)):
            if args.limit > 0 and idx >= args.limit:
                break
            item = ds[idx]
            if item is None:
                continue
            try:
                # TrajGaze scores: identical across shifts (no dependence on options),
                # but preprocess_item re-tokenizes for each shift so we re-extract video
                # features per shift (simpler; cost ~4x). Score extraction once is OK.
                scores_2d = get_patch_scores_temporal(traj_encoder, item, device)

                per_shift = []
                for k in range(4):
                    options_k = shift_options(item["options"], k)
                    ans_k     = shifted_answer(item["answer"], k)

                    cached = preprocess_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], options_k, device,
                    )
                    if cached is None:
                        per_shift.append(None)
                        continue

                    n_video   = cached["video_embeds"].shape[0]
                    T_merged  = int(cached["grid_thw"][0, 0].item())
                    n_spatial = n_video // max(1, T_merged)
                    r         = max(1, int(args.merge_ratio * n_video))

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
                    logits = forward_logits(
                        qwen_model,
                        build_merged_inputs(base_qwen, cached, merged_video, receiver_idx),
                    )
                    opt_logits = logits[option_ids].float().cpu()
                    opt_probs  = F.softmax(opt_logits, dim=0)
                    pred_idx   = int(opt_logits.argmax().item())
                    pred_letter = "ABCD"[pred_idx]
                    correct    = (pred_letter == ans_k)

                    # Resolve which ORIGINAL option content the model picked under shift k:
                    # at position i (= pred_idx), content came from original index (i+k)%4
                    orig_content_idx = (pred_idx + k) % 4
                    orig_content_letter = "ABCD"[orig_content_idx]

                    per_shift.append({
                        "shift": k,
                        "shifted_answer": ans_k,
                        "pred_letter": pred_letter,
                        "correct": bool(correct),
                        "picked_original_content": orig_content_letter,
                        "top1_prob": float(opt_probs.max().item()),
                        "logit_margin": float((opt_logits.sort(descending=True).values[0]
                                              - opt_logits.sort(descending=True).values[1]).item()),
                        "prob_A": float(opt_probs[0]),
                        "prob_B": float(opt_probs[1]),
                        "prob_C": float(opt_probs[2]),
                        "prob_D": float(opt_probs[3]),
                    })

                if any(s is None for s in per_shift):
                    continue
                # Aggregate: did all 4 shifts pick the same original content?
                picked_orig = [s["picked_original_content"] for s in per_shift]
                agree4 = len(set(picked_orig)) == 1
                # Agree2: at least the 2 most common shifts agree
                from collections import Counter
                most_common, mc_count = Counter(picked_orig).most_common(1)[0]
                agree2 = (mc_count >= 2)
                consistent_correct = agree4 and (most_common == item["answer"])
                per_shift_correct = [s["correct"] for s in per_shift]

                base = {
                    "idx": idx,
                    "task": item.get("task", "unknown"),
                    "dataset": item.get("dataset", "unknown"),
                    "question": item["question"][:200],
                    "orig_answer": item["answer"],
                    "agree4": bool(agree4),
                    "agree2": bool(agree2),
                    "majority_content": most_common,
                    "majority_count": int(mc_count),
                    "consistent_correct": bool(consistent_correct),
                    "n_correct_shifts": int(sum(per_shift_correct)),
                    "picked_original_content_seq": picked_orig,
                    "per_shift_correct": per_shift_correct,
                    "mean_top1_prob": float(np.mean([s["top1_prob"] for s in per_shift])),
                    "mean_logit_margin": float(np.mean([s["logit_margin"] for s in per_shift])),
                }
                # Add per-shift columns
                for s in per_shift:
                    k = s["shift"]
                    base[f"correct_k{k}"] = s["correct"]
                    base[f"pred_letter_k{k}"] = s["pred_letter"]
                rows.append(base)

                if (idx + 1) % 25 == 0:
                    n = len(rows)
                    acc = 100.0 * sum(r["n_correct_shifts"] for r in rows) / max(1, 4 * n)
                    a4 = 100.0 * sum(r["agree4"] for r in rows) / max(1, n)
                    print(f"  [{idx+1}/{len(ds)}] avg_acc(4x)={acc:.2f}%  agree4={a4:.2f}%  (n={n})")

            except Exception:
                traceback.print_exc()
                continue

    if not rows:
        print("No rows produced.")
        return

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(out_dir, f"{args.tag}_permutation.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = parquet_path.replace(".parquet", ".jsonl")
        df.to_json(parquet_path, orient="records", lines=True)
    print(f"  saved {len(df)} samples -> {parquet_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(df)
    per_shift_acc = {f"k{k}": 100.0 * df[f"correct_k{k}"].mean() for k in range(4)}
    avg_acc_all_shifts = sum(per_shift_acc.values()) / 4

    # Pick frequency over positions (which letter the model tends to pick)
    pick_freq = {l: 0 for l in "ABCD"}
    for k in range(4):
        for l in df[f"pred_letter_k{k}"]:
            pick_freq[l] += 1
    total = 4 * n
    pick_freq_pct = {l: 100.0 * v / max(1, total) for l, v in pick_freq.items()}

    by_task = (
        df.groupby("task")
          .agg(n=("agree4", "size"),
               agree4=("agree4", "mean"),
               agree2=("agree2", "mean"),
               consistent_correct=("consistent_correct", "mean"),
               n_correct_shifts_mean=("n_correct_shifts", "mean"))
          .reset_index()
    )
    by_task["agree4"] = by_task["agree4"] * 100
    by_task["agree2"] = by_task["agree2"] * 100
    by_task["consistent_correct"] = by_task["consistent_correct"] * 100

    summary = {
        "tag": args.tag,
        "n_samples": n,
        "avg_acc_across_4_shifts": avg_acc_all_shifts,
        "per_shift_acc": per_shift_acc,
        "agree4_rate": float(100.0 * df["agree4"].mean()),
        "agree2_rate": float(100.0 * df["agree2"].mean()),
        "consistent_correct_rate": float(100.0 * df["consistent_correct"].mean()),
        "pick_freq_pct": pick_freq_pct,
        "mean_top1_prob_when_agree4": float(df.loc[df["agree4"], "mean_top1_prob"].mean()) if df["agree4"].any() else float("nan"),
        "mean_top1_prob_when_disagree": float(df.loc[~df["agree4"], "mean_top1_prob"].mean()) if (~df["agree4"]).any() else float("nan"),
        "per_task": by_task.to_dict(orient="records"),
        "interpretation": {
            "if_avg_acc_near_25": "pure guessing",
            "if_avg_acc_high_but_agree4_low": "uses superficial cues per position, not content",
            "if_avg_acc_high_and_agree4_high": "model genuinely understands content",
            "if_pick_freq_uneven": "position bias (e.g. always favors C)",
        },
    }
    summary_path = os.path.join(out_dir, f"{args.tag}_permutation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Option Permutation Eval [{args.tag}]")
    print(f"{'='*60}")
    print(f"  n_samples: {n}")
    print(f"  avg accuracy across 4 shifts: {avg_acc_all_shifts:.2f}%  (random = 25.0%)")
    print(f"  per-shift acc: " + "  ".join(f"{k}={a:.2f}%" for k, a in per_shift_acc.items()))
    print(f"  agree4 rate:  {summary['agree4_rate']:.2f}%  (random = {100*(1/4**3):.2f}%)")
    print(f"  agree2 rate:  {summary['agree2_rate']:.2f}%")
    print(f"  consistent-correct rate: {summary['consistent_correct_rate']:.2f}%")
    print(f"  pick frequency: " + "  ".join(f"{l}={v:.1f}%" for l, v in pick_freq_pct.items()) + "  (uniform=25%)")
    print(f"  conf when agree4 vs disagree: "
          f"{summary['mean_top1_prob_when_agree4']:.3f} vs {summary['mean_top1_prob_when_disagree']:.3f}")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
