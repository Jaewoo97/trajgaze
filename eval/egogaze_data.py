"""
Shared data-loading utilities for EgoGazeVQA fold-c (EGTEA) evaluation.

Dataset layout:
  /workspace/datasets/EgoGazeVQA/egtea/no_gaze/{recipe}/{start}_{end}_{frame}.jpg
  metadata.csv: file_name, dataset, qa_type, question, answer_options, correct_answer
"""

import os
import re

import numpy as np
from PIL import Image

EGOGAZE_ROOT  = "/workspace/datasets/EgoGazeVQA"
METADATA_CSV  = os.path.join(EGOGAZE_ROOT, "metadata.csv")
RESULTS_DIR   = "/workspace/EgoGazeVQA/eval/results/egogaze_fold_c"


def load_egtea_samples() -> list[dict]:
    """Return list of EGTEA sample dicts from metadata.csv."""
    import csv
    samples = []
    with open(METADATA_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] != "egtea":
                continue
            # file_name: egtea/OP01-R01-PastaSalad/4017_4713.mp4
            parts     = row["file_name"].split("/")       # ['egtea', 'recipe', 'clip.mp4']
            recipe    = parts[1]
            clip_stem = parts[2].replace(".mp4", "")     # '4017_4713'
            frame_dir = os.path.join(EGOGAZE_ROOT, "egtea", "no_gaze", recipe)

            # Parse answer options: "A: text|B: text|..."
            raw_opts = row["answer_options"].split("|")
            options  = [o.strip() for o in raw_opts]

            samples.append({
                "file_name":    row["file_name"],
                "recipe":       recipe,
                "clip_stem":    clip_stem,
                "frame_dir":    frame_dir,
                "qa_type":      row["qa_type"],
                "question":     row["question"],
                "options":      options,
                "answer":       row["correct_answer"].strip().upper(),
            })
    return samples


def load_clip_frames(frame_dir: str, clip_stem: str, n: int, size: int) -> list[Image.Image]:
    """Load n evenly-sampled frames for clip_stem from frame_dir, resized to size×size."""
    all_files = sorted(
        f for f in os.listdir(frame_dir)
        if f.startswith(clip_stem + "_") and f.endswith(".jpg")
    )
    if not all_files:
        raise FileNotFoundError(f"No frames for {clip_stem} in {frame_dir}")
    indices = np.round(np.linspace(0, len(all_files) - 1, n)).astype(int)
    frames  = []
    for i in indices:
        img = Image.open(os.path.join(frame_dir, all_files[i])).convert("RGB")
        if size:
            img = img.resize((size, size), Image.BILINEAR)
        frames.append(img)
    return frames


def make_clip_dir(frame_dir: str, clip_stem: str, tmp_dir: str) -> str:
    """Symlink all frames for clip_stem into a sequential-named temp directory."""
    all_files = sorted(
        f for f in os.listdir(frame_dir)
        if f.startswith(clip_stem + "_") and f.endswith(".jpg")
    )
    seg_dir = os.path.join(tmp_dir, f"{clip_stem}")
    os.makedirs(seg_dir, exist_ok=True)
    for i, fname in enumerate(all_files):
        dst = os.path.join(seg_dir, f"frame_{i:06d}.jpg")
        if not os.path.lexists(dst):
            os.symlink(os.path.join(frame_dir, fname), dst)
    return seg_dir


def extract_choice(text: str) -> str:
    """Extract answer letter A-E from model output."""
    text = text.strip()
    for pat in [r"\b([ABCDE])\b", r"([ABCDE])[.)]", r"\(([ABCDE])\)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def compute_metrics_by_type(results: list[dict]) -> dict:
    """Compute overall + per-qa_type accuracy and avg latency."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        buckets[r["qa_type"]].append(r)

    def _acc(items):
        if not items:
            return {"accuracy": 0.0, "n_samples": 0, "empty_preds": 0}
        correct = sum(1 for r in items if extract_choice(r["prediction"]) == r["answer"])
        empty   = sum(1 for r in items if not extract_choice(r["prediction"]))
        return {
            "accuracy":    round(correct / len(items) * 100, 2),
            "n_samples":   len(items),
            "empty_preds": empty,
        }

    per_type = {qt: _acc(items) for qt, items in sorted(buckets.items())}
    overall  = _acc(results)

    times = [r["inference_time"] for r in results if r.get("inference_time", 0) > 0]
    overall["avg_inference_time_sec"] = round(sum(times) / len(times), 4) if times else 0.0

    return {"overall": overall, "per_qa_type": per_type}


def save_eval_results(results: list[dict], stem: str) -> None:
    import json
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pred_path    = os.path.join(RESULTS_DIR, f"{stem}_predictions.json")
    metrics_path = os.path.join(RESULTS_DIR, f"{stem}_metrics.json")

    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions → {pred_path}")

    metrics = compute_metrics_by_type(results)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    ov = metrics["overall"]
    print(f"\n=== {stem} ===")
    print(f"Overall accuracy : {ov['accuracy']:.2f}%  (n={ov['n_samples']}, empty={ov['empty_preds']})")
    print(f"Avg latency      : {ov['avg_inference_time_sec']:.3f}s")
    print("Per question type:")
    for qt, m in metrics["per_qa_type"].items():
        print(f"  {qt:10s}: {m['accuracy']:.2f}%  (n={m['n_samples']})")
    print(f"Saved metrics → {metrics_path}")
