"""
Extract gaze coordinates (green dot inside red circle) from StreamGaze_v2 viz frames.

The gaze is rendered as a fixed green dot (~200 px, G≈240, R≈20, B≈20).
Frames without a gaze annotation return null.

Output per video:
  <output_dir>/{dataset}/viz/{video_stem}.json
  {
    "frame_000001.jpg": [cx, cy],   # gaze present
    "frame_000002.jpg": null,       # no gaze annotation
    ...
  }

cx/cy are pixel coordinates in the original image.

Usage:
  python preprocess/extract_gaze_locations.py \
      --frames-dir /workspace/datasets/StreamGaze_v2/frames \
      --output     /workspace/datasets/StreamGaze_v2/gaze
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Gaze detection thresholds (calibrated from viz frames) ────────────────────
# The green dot has: G > 150, G > R*1.5, G > B*1.5
# Require at least 50 qualifying pixels to count as a valid gaze annotation
# (noise / foliage rarely produces this many pure-green pixels in a tight cluster)
GREEN_G_MIN    = 150
GREEN_RATIO    = 1.5    # G must be > R * ratio and > B * ratio
MIN_PX         = 50     # minimum green pixel count to accept as gaze dot


def detect_gaze(img_path: str) -> list | None:
    """
    Return [cx, cy] if a green gaze dot is found, else None.
    cx/cy are in the original image coordinate space.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    R, G, B = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)

    mask = (G > GREEN_G_MIN) & (G > R * GREEN_RATIO) & (G > B * GREEN_RATIO)
    count = int(mask.sum())
    if count < MIN_PX:
        return None

    ys, xs = np.where(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    return [round(cx, 1), round(cy, 1)]


def collect_videos(frames_dir: str):
    """Return list of (dataset, split, stem, frame_dir) for all viz splits."""
    videos = []
    for dataset in sorted(os.listdir(frames_dir)):
        dataset_dir = os.path.join(frames_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for split in sorted(os.listdir(dataset_dir)):
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for stem in sorted(os.listdir(split_dir)):
                frame_dir = os.path.join(split_dir, stem)
                if os.path.isdir(frame_dir):
                    videos.append((dataset, split, stem, frame_dir))
    return videos


def process_video(args):
    dataset, split, stem, frame_dir, out_path = args
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    results = {fname: detect_gaze(os.path.join(frame_dir, fname)) for fname in frames}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)

    n_gaze = sum(1 for v in results.values() if v is not None)
    return len(results), n_gaze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True,
                        help="Root of StreamGaze_v2/frames")
    parser.add_argument("--output", required=True,
                        help="Output directory (StreamGaze_v2/gaze)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--split", default=None,
                        help="Only process this split (e.g. 'viz')")
    args = parser.parse_args()

    videos = collect_videos(args.frames_dir)
    if args.split:
        videos = [(d, s, st, fd) for d, s, st, fd in videos if s == args.split]
    print(f"Found {len(videos)} videos to process")

    todo = []
    for dataset, split, stem, frame_dir in videos:
        out_path = os.path.join(args.output, dataset, split, f"{stem}.json")
        if args.skip_existing and os.path.exists(out_path):
            continue
        todo.append((dataset, split, stem, frame_dir, out_path))
    print(f"Skipping {len(videos)-len(todo)} already done. Processing {len(todo)} videos.")

    n_workers = min(64, len(todo)) if todo else 1
    print(f"Using {n_workers} parallel workers")
    total_frames = total_gaze = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for n_frames, n_gaze in tqdm(pool.map(process_video, todo), total=len(todo), desc="Extracting gaze"):
            total_frames += n_frames
            total_gaze   += n_gaze

    print(f"\nDone. {total_frames} frames processed, "
          f"{total_gaze} with gaze ({100*total_gaze/total_frames:.1f}%)" if total_frames else "\nDone.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
