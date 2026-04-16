"""
Evaluation script for TrajGaze.

Evaluates a trained TrajGaze model on the StreamGaze VQA benchmark
(present_future_action_prediction) across all datasets.

Computes:
  - VLM accuracy (verb-noun / action letter match)
  - Mean token ratio (frames attended / total)
  - Per-dataset breakdown

Also supports the ablation flag to evaluate any mode variant.

Usage:
  python -m TrajGaze.training.evaluate \
      --stage2-ckpt  TrajGaze/checkpoints/stage2/fold_a_nvila/epoch_0003.pt \
      --adapted-dir  /workspace/datasets/StreamGaze_v2/adapted \
      --interact-dir /workspace/datasets/StreamGaze_v2/interaction \
      --frames-dir   /workspace/datasets/StreamGaze_v2/frames \
      --qa-file      /workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json \
      --vlm          nvila \
      --output       TrajGaze/results/
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm

from TrajGaze.training.stage1 import TrajGazeModel
from TrajGaze.training.stage2 import (
    Stage2Dataset, VLMOracle, FOLD_VAL, FRAMES_PER_STEP, ATTEND_THRESH
)


def evaluate(
    model:   TrajGazeModel,
    ds:      Stage2Dataset,
    oracle:  VLMOracle,
    device:  torch.device,
    max_samples: int | None = None,
) -> dict:
    """
    Run greedy frame selection and VLM scoring over the dataset.

    Returns metrics dict with per-dataset and overall results.
    """
    model.eval()

    tokenizer_keys = [
        "gaze_pos", "gaze_speed", "gaze_mask",
        "left_pos", "left_vel", "left_mask",
        "right_pos", "right_vel", "right_mask",
        "d_left", "d_right", "v_rel_left", "v_rel_right",
        "convergence", "lead_lag",
    ]

    per_ds: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0,
                                                     "token_ratios": []})
    predictions = []

    indices = list(range(len(ds)))
    if max_samples:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Evaluating"):
        try:
            item = ds[idx]
        except Exception as e:
            print(f"  [skip] sample {idx}: {e}")
            continue

        ds_name = item["dataset"]
        T = item["T"]

        with torch.no_grad():
            tok_in = {k: item[k].to(device) for k in tokenizer_keys}
            _, _, _, tok_interact = model.tokenizer(**tok_in)
            query_sim = item.get("query_sim")
            if query_sim is not None:
                query_sim = query_sim.to(device)
            _, attended_idx_list  = model.selector.select_frames(
                tok_interact, query_sim=query_sim
            )
            attended_idx = attended_idx_list[0]

        token_ratio = len(attended_idx) / max(1, T)
        per_ds[ds_name]["token_ratios"].append(token_ratio)

        # VLM scoring
        if oracle.model is not None and attended_idx:
            eval_paths = []
            # Use up to 8 frames, evenly sampled from attended
            sample_idx = _evenly_sample(attended_idx, n=8)
            for t in sorted(sample_idx):
                if t < len(item["frame_names"]):
                    p = os.path.join(item["frames_dir"], item["frame_names"][t])
                    if os.path.exists(p):
                        eval_paths.append(p)

            if eval_paths:
                correct = oracle.score(
                    frame_paths = eval_paths,
                    question    = item["question"],
                    options     = item["options"],
                    gt_answer   = item["gt_answer"],
                )
            else:
                correct = 0.0
        else:
            # Proxy: attend label accuracy
            attend = item["attend"].float().to(device)
            scores = model.selector(tok_interact, query_sim=query_sim)[0]
            att = (scores >= ATTEND_THRESH).float()
            correct = float((att == attend).float().mean().item())

        per_ds[ds_name]["correct"] += correct
        per_ds[ds_name]["total"]   += 1

        predictions.append({
            "dataset":    ds_name,
            "stem":       item["stem"],
            "question":   item["question"][:60],
            "gt":         item["gt_answer"],
            "correct":    correct,
            "n_attended": len(attended_idx),
            "T":          T,
            "token_ratio": token_ratio,
        })

    # Aggregate
    overall_correct = sum(v["correct"] for v in per_ds.values())
    overall_total   = sum(v["total"]   for v in per_ds.values())
    overall_acc     = overall_correct / max(1, overall_total)
    overall_tr      = sum(
        sum(v["token_ratios"]) for v in per_ds.values()
    ) / max(1, overall_total)

    results = {
        "overall": {
            "accuracy":     overall_acc,
            "n_correct":    overall_correct,
            "n_total":      overall_total,
            "mean_token_ratio": overall_tr,
        },
        "per_dataset": {
            ds_name: {
                "accuracy": v["correct"] / max(1, v["total"]),
                "n_total":  v["total"],
                "mean_token_ratio": sum(v["token_ratios"]) / max(1, len(v["token_ratios"])),
            }
            for ds_name, v in per_ds.items()
        },
        "predictions": predictions,
    }
    return results


def _evenly_sample(idx_list: list[int], n: int) -> list[int]:
    if len(idx_list) <= n:
        return idx_list
    step = len(idx_list) / n
    return [idx_list[int(i * step)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-ckpt",  default=None,
                        help="Stage 2 checkpoint (if None, uses stage1-ckpt)")
    parser.add_argument("--stage1-ckpt",  default=None,
                        help="Stage 1 checkpoint (used if stage2-ckpt not given)")
    parser.add_argument("--adapted-dir",  required=True)
    parser.add_argument("--interact-dir", required=True)
    parser.add_argument("--frames-dir",   required=True)
    parser.add_argument("--qa-file",      required=True)
    parser.add_argument("--vlm",          default="nvila",
                        choices=["nvila", "qwen", "none"])
    parser.add_argument("--fold",         default=None,
                        help="Evaluate only val split of this fold. "
                             "If None, evaluates all datasets.")
    parser.add_argument("--output",       required=True,
                        help="Directory to save results JSON")
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # Load model
    model = TrajGazeModel(args).to(device)
    ckpt_path = args.stage2_ckpt or args.stage1_ckpt
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint specified — using random weights.")

    # Load VLM
    oracle = VLMOracle(backend=args.vlm if args.vlm != "none" else "none")
    if args.vlm != "none":
        oracle.load()

    # Determine which datasets to evaluate
    if args.fold:
        eval_datasets = FOLD_VAL[args.fold]
    else:
        eval_datasets = ["egoexolearn", "egtea", "holoassist"]

    ds = Stage2Dataset(
        qa_file      = args.qa_file,
        adapted_dir  = args.adapted_dir,
        interact_dir = args.interact_dir,
        frames_dir   = args.frames_dir,
        datasets     = eval_datasets,
        max_frames   = 200,
    )

    results = evaluate(model, ds, oracle, device, max_samples=args.max_samples)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Overall accuracy: {100*results['overall']['accuracy']:.1f}%  "
          f"({results['overall']['n_correct']:.0f}/{results['overall']['n_total']})")
    print(f"Mean token ratio: {100*results['overall']['mean_token_ratio']:.1f}%")
    for ds_name, m in results["per_dataset"].items():
        print(f"  {ds_name}: {100*m['accuracy']:.1f}%  "
              f"(token_ratio={100*m['mean_token_ratio']:.1f}%)")
    print("=" * 50)

    # Save
    ckpt_name = os.path.basename(ckpt_path).replace(".pt", "") if ckpt_path else "no_ckpt"
    out_file = os.path.join(args.output, f"trajgaze_{ckpt_name}_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
