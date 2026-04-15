"""
Prepare EgoGazeVQA dataset for AutoGaze NTP training.

For each sub-clip in ego4d/egoexo/egtea:
  1. Chunk frames into non-overlapping 16-frame windows
  2. Write each window as an .mp4 file into autogaze_fold/{train|val}/

GT gaze patches are generated SEPARATELY by greedy_search_gt.py using
VideoMAE reconstruction loss (following the actual AutoGaze training protocol).
Do NOT use actual gaze coordinates here.

Multi-scale patch system (32+64+112+224, patch_size=16):
  Scale  32 → 2×2  =  4 patches, offset   0
  Scale  64 → 4×4  = 16 patches, offset   4
  Scale 112 → 7×7  = 49 patches, offset  20
  Scale 224 → 14×14=196 patches, offset  69
  Total: 265 patches per frame

Split:
  train = ego4d + egoexo
  val   = egtea

Usage:
    conda run -n gaze python tools/prepare_egogaze_autogaze.py
    conda run -n gaze python tools/prepare_egogaze_autogaze.py --dry-run
"""

import argparse
import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = "/workspace/datasets/EgoGazeVQA"
FOLD_DIR   = "/workspace/datasets/EgoGazeVQA/autogaze_fold"
CLIP_LEN   = 16      # frames per AutoGaze clip
FPS        = 10      # mp4 output fps (AutoGaze samples by count, not time)
FOURCC     = cv2.VideoWriter_fourcc(*"mp4v")

TRAIN_DATASETS = ["ego4d", "egoexo"]
VAL_DATASETS   = ["egtea"]


# ── Sub-clip enumeration ───────────────────────────────────────────────────────

def collect_subclips():
    """
    Returns list of:
      (dataset, split, recording_id, subclip_stem, frame_dir, rows)

    rows: list of dicts with keys frame_idx, gaze_frame_num, gaze_x, gaze_y
    Frames in frame_dir are named: {subclip_stem}_{gaze_frame_num}.jpg
    """
    subclips = []
    for ds in TRAIN_DATASETS + VAL_DATASETS:
        split = "train" if ds in TRAIN_DATASETS else "val"
        mapping_root = os.path.join(DATA_ROOT, ds, "gaze_mapping")
        no_gaze_root = os.path.join(DATA_ROOT, ds, "no_gaze")

        if not os.path.isdir(mapping_root):
            print(f"WARNING: {mapping_root} not found, skipping {ds}.")
            continue

        for rec_id in sorted(os.listdir(mapping_root)):
            rec_map_dir   = os.path.join(mapping_root, rec_id)
            rec_frame_dir = os.path.join(no_gaze_root, rec_id)
            if not os.path.isdir(rec_map_dir) or not os.path.isdir(rec_frame_dir):
                continue

            for csv_file in sorted(os.listdir(rec_map_dir)):
                if not csv_file.endswith("_mapping.csv"):
                    continue
                subclip_stem = csv_file.replace("_mapping.csv", "")
                csv_path = os.path.join(rec_map_dir, csv_file)

                with open(csv_path) as f:
                    all_rows = [r for r in csv.DictReader(f)
                                if r.get("gaze_frame_num")]

                # Only keep rows where the jpg file actually exists on disk
                # (frames are sparse — not every gaze_frame_num has a jpg)
                rows = [r for r in all_rows
                        if os.path.exists(os.path.join(
                            rec_frame_dir,
                            f"{subclip_stem}_{r['gaze_frame_num']}.jpg"
                        ))]

                if len(rows) < CLIP_LEN:
                    continue  # too short

                subclips.append((ds, split, rec_id, subclip_stem, rec_frame_dir, rows))

    return subclips


# ── MP4 writer ─────────────────────────────────────────────────────────────────

def write_mp4(out_path: str, frame_paths: list) -> bool:
    """
    Write a list of frame image paths to an mp4. Returns True on success.
    """
    frames = []
    for p in frame_paths:
        im = cv2.imread(p)
        if im is None:
            return False
        frames.append(im)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, FOURCC, FPS, (w, h))
    for im in frames:
        writer.write(im)
    writer.release()
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Count clips without writing files")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    # Output directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(FOLD_DIR, split), exist_ok=True)

    subclips = collect_subclips()
    print(f"Found {len(subclips)} sub-clips across all datasets")

    n_written = n_skipped = n_error = 0

    for ds, split, rec_id, subclip_stem, frame_dir, rows in tqdm(subclips, desc="Sub-clips"):
        out_split_dir = os.path.join(FOLD_DIR, split)

        # Chunk rows into non-overlapping 16-frame windows
        n_windows = len(rows) // CLIP_LEN
        for win_idx in range(n_windows):
            window_rows = rows[win_idx * CLIP_LEN: (win_idx + 1) * CLIP_LEN]

            # Build frame paths using gaze_frame_num to locate the image file
            frame_paths = []
            for r in window_rows:
                gfn = r["gaze_frame_num"]
                fname = f"{subclip_stem}_{gfn}.jpg"
                frame_paths.append(os.path.join(frame_dir, fname))

            # Output mp4 name
            mp4_name = f"{ds}_{rec_id}_{subclip_stem}_w{win_idx:04d}.mp4"
            mp4_path = os.path.join(out_split_dir, mp4_name)

            if args.skip_existing and os.path.exists(mp4_path):
                n_skipped += 1
                continue

            if args.dry_run:
                n_written += 1
                continue

            # Verify frames exist
            if not all(os.path.exists(p) for p in frame_paths):
                n_error += 1
                continue

            # Write mp4
            if not write_mp4(mp4_path, frame_paths):
                n_error += 1
                continue

            n_written += 1

    print(f"\nDone: {n_written} clips written, {n_skipped} skipped, {n_error} errors")

    if args.dry_run:
        print(f"[dry-run] Would create ~{n_written} clips")
        train_count = sum(1 for sc in subclips if sc[1] == "train")
        val_count   = sum(1 for sc in subclips if sc[1] == "val")
        print(f"  train (ego4d+egoexo): {train_count} sub-clips")
        print(f"  val   (egtea):        {val_count} sub-clips")
    else:
        print(f"\nNext: run greedy_search_gt.py to generate MAE-based GT")
        print(f"  bash AutoGaze/preprocess_egogaze.sh")


if __name__ == "__main__":
    main()
