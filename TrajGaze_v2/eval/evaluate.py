"""
TrajGaze_v2 Evaluation on EGTEA.

Compares VQA accuracy and latency of:
  - Qwen2.5-VL-7B + TrajGaze_v2 token selection (10%)

Saves results to /workspace/EgoGazeVQA/TrajGaze_v2/eval_results/

Usage:
    conda run -n gaze python -m TrajGaze_v2.eval.evaluate \
        --stage2-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage2/best.pth \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

import torch
from tqdm import tqdm

from TrajGaze_v2.data.dataset import TrajGazeV2QADataset, _load_clip_traj, _traj_to_tensors
from TrajGaze_v2.models.model import TrajGazeV2
from TrajGaze_v2.training.stage2 import (
    load_qwen, load_frames_pil, qwen_inference_with_mask, extract_answer,
    QWEN_MODEL_PATH, FRAME_SIZE,
)

RESULTS_DIR = "/workspace/EgoGazeVQA/TrajGaze_v2/eval_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-ckpt", required=True)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--n-frames",    type=int, default=32)
    p.add_argument("--n-qwen-frames", type=int, default=64)
    p.add_argument("--output-tag",  default="trajgaze_v2_10pct")
    return p.parse_args()


def compute_metrics(results: list[dict]) -> dict:
    by_type: dict[str, list] = {}
    correct_all = 0
    for r in results:
        qt  = r.get("qa_type", "unknown")
        ok  = r.get("correct", False)
        correct_all += int(ok)
        by_type.setdefault(qt, []).append(ok)

    overall_acc = 100.0 * correct_all / max(1, len(results))
    avg_latency = sum(r.get("generate_time", r.get("inference_time", 0)) for r in results) / max(1, len(results))
    avg_tokens  = sum(r.get("n_input_tokens", 0) for r in results) / max(1, len(results))

    per_type = {}
    for qt, flags in sorted(by_type.items()):
        per_type[qt] = {
            "accuracy":  100.0 * sum(flags) / max(1, len(flags)),
            "n_samples": len(flags),
        }

    return {
        "overall":     {"accuracy": overall_acc, "n_samples": len(results),
                        "avg_generate_time_sec": round(avg_latency, 4),
                        "avg_input_tokens": round(avg_tokens, 1)},
        "per_qa_type": per_type,
    }


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load TrajGaze_v2
    print(f"Loading TrajGaze_v2 from {args.stage2_ckpt} ...")
    model = TrajGazeV2().to(device)
    ckpt  = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("TrajGaze_v2 loaded.")

    # Load Qwen
    print("Loading Qwen2.5-VL-7B ...")
    qwen_processor, qwen_model = load_qwen(device)
    print("Qwen loaded.")

    # EGTEA QA dataset
    dataset = TrajGazeV2QADataset(
        datasets  = ["egtea"],
        n_frames  = args.n_frames,
    )

    results = []
    for item in tqdm(dataset, desc="TrajGazeV2+Qwen"):
        if item is None:
            continue
        try:
            traj_tensors = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in item["traj"].items()
            }
            question  = item["question"]
            options   = item["options"]
            answer_gt = item["answer"]
            qa_type   = item["qa_type"]

            # Selector forward
            with torch.no_grad():
                patch_mask = model.select_tokens(
                    traj_tensors,
                    queries      = [question],
                    frame_paths  = [item["frame_paths"]],
                    gumbel_noise = False,
                )   # (1, 196)

            # Qwen inference — generate_time is ONLY the generate() call
            response, generate_time, n_input_tokens = qwen_inference_with_mask(
                qwen_processor, qwen_model,
                item["frame_paths"], question, options,
                patch_mask[0],
                n_qwen_frames = args.n_qwen_frames,
                device        = device,
            )

            pred    = extract_answer(response)
            correct = (pred == answer_gt)

            results.append({
                "question":        question,
                "answer":          answer_gt,
                "prediction":      pred,
                "correct":         correct,
                "qa_type":         qa_type,
                "generate_time":   generate_time,
                "n_input_tokens":  n_input_tokens,
            })

        except Exception:
            traceback.print_exc()
            results.append({
                "question":      item.get("question", ""),
                "answer":        item.get("answer", ""),
                "prediction":    "",
                "correct":       False,
                "qa_type":       item.get("qa_type", ""),
                "inference_time": 0.0,
            })

    # Save results
    tag = args.output_tag
    results_path  = os.path.join(RESULTS_DIR, f"{tag}_results.json")
    metrics_path  = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics = compute_metrics(results)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    ov = metrics["overall"]
    print(f"\n{'='*50}")
    print(f"  {tag}")
    print(f"{'='*50}")
    print(f"  Overall       : {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"  Generate time : {ov['avg_generate_time_sec']:.4f}s  (VLM generate() only)")
    print(f"  Avg tokens    : {ov['avg_input_tokens']:.0f}")
    for qt, qm in sorted(metrics["per_qa_type"].items()):
        print(f"  {qt:10s}: {qm['accuracy']:.2f}%  (n={qm['n_samples']})")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
