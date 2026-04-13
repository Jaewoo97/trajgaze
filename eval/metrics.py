"""Shared evaluation metrics and frame loading utilities."""

import json
import os
import re
from PIL import Image


def build_frames_map(frames_root: str) -> dict[str, str]:
    """Build a mapping {video_filename -> frame_dir} by walking frames_root.

    frames_root mirrors the videos/ structure:
      frames_root/{dataset}/{split}/{video_stem}/frame_*.jpg

    Returns dict keyed by video filename (with .mp4 extension).
    """
    mapping = {}
    for dataset in os.listdir(frames_root):
        dataset_dir = os.path.join(frames_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for split in os.listdir(dataset_dir):
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for stem in os.listdir(split_dir):
                frame_dir = os.path.join(split_dir, stem)
                if os.path.isdir(frame_dir):
                    mapping[stem + ".mp4"] = frame_dir
    return mapping


def load_frames(frame_dir: str, start: float, end: float) -> list[Image.Image] | None:
    """Load frames for a response_time window from a pre-extracted frame directory.

    Args:
        frame_dir:  Directory containing frame_%06d.jpg files for one video
                    (obtained from build_frames_map()).
        start:      Window start in seconds.
        end:        Window end in seconds.

    Returns:
        List of PIL Images, or None if no frames fall in the window.
    """
    # Frames are named frame_%06d.jpg (1-indexed) at 10 FPS.
    # Frame N corresponds to time (N-1) / 10 seconds.
    start_idx = max(0, int(start * 10))  # inclusive, 0-indexed
    end_idx   = int(end * 10)            # inclusive, 0-indexed

    all_frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    selected = [
        os.path.join(frame_dir, f)
        for f in all_frames
        if start_idx <= (int(f.split("_")[1].split(".")[0]) - 1) <= end_idx
    ]
    if not selected:
        return None

    return [Image.open(p).convert("RGB") for p in selected]


def extract_choice(text: str) -> str:
    """Extract the answer letter (A/B/C/D) from model output."""
    text = text.strip()
    # Match patterns like: "A", "A.", "A)", "(A)", "Answer: A", "The answer is A"
    patterns = [
        r"\b([ABCD])\b",
        r"([ABCD])[.)]",
        r"\(([ABCD])\)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy over a list of prediction dicts (multiple-choice)."""
    correct = 0
    empty_preds = 0
    for r in results:
        pred_letter = extract_choice(r["prediction"])
        gold_letter = r["answer"].strip().upper()
        if not pred_letter:
            empty_preds += 1
        elif pred_letter == gold_letter:
            correct += 1
    n = len(results)
    return {
        "n_samples":   n,
        "empty_preds": empty_preds,
        "accuracy":    round(correct / n * 100, 2) if n else 0.0,
    }


def dataset_from_frames_map(video_path: str, frames_map: dict[str, str]) -> str:
    """Return the dataset name for a video by inspecting its frame directory path.

    frames_map values look like:  .../frames/{dataset}/{split}/{stem}/
    Known dataset directory names: egtea, egoexolearn, holoassist.
    Returns 'unknown' if the path cannot be parsed.
    """
    frame_dir = frames_map.get(video_path, "")
    for part in frame_dir.replace("\\", "/").split("/"):
        if part.lower() in ("egtea", "egoexolearn", "holoassist"):
            return part.lower()
    return "unknown"


def compute_per_dataset_metrics(
    results: list[dict],
    frames_map: dict[str, str],
) -> dict[str, dict]:
    """Split results by dataset and compute accuracy per dataset."""
    buckets: dict[str, list[dict]] = {}
    for r in results:
        ds = dataset_from_frames_map(r["video_path"], frames_map)
        buckets.setdefault(ds, []).append(r)
    return {ds: compute_metrics(items) for ds, items in sorted(buckets.items())}


def save_results(
    results: list[dict],
    out_dir: str,
    stem: str,
    extra: dict | None = None,
    frames_map: dict[str, str] | None = None,
) -> None:
    """Save per-sample predictions and aggregate metrics into out_dir.

    Files written:
      <out_dir>/<stem>_predictions.json   — full per-sample results
      <out_dir>/<stem>_metrics.json       — aggregate accuracy + extra fields

    If frames_map is provided, per-dataset accuracy is included under
    the key "per_dataset" in the metrics JSON.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    pred_path    = os.path.join(out_dir, f"{stem}_predictions.json")
    metrics_path = os.path.join(out_dir, f"{stem}_metrics.json")

    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions → {pred_path}")

    metrics = compute_metrics(results)
    if extra:
        metrics.update(extra)
    if frames_map:
        metrics["per_dataset"] = compute_per_dataset_metrics(results, frames_map)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: accuracy={metrics['accuracy']:.2f}%  "
          f"(n={metrics['n_samples']}, empty={metrics['empty_preds']})")
    if "per_dataset" in metrics:
        for ds, m in metrics["per_dataset"].items():
            print(f"  {ds}: {m['accuracy']:.2f}% (n={m['n_samples']})")
    print(f"Saved metrics    → {metrics_path}")
