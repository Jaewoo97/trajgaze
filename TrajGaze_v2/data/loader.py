"""
Data loading utilities for TrajGaze_v2.

- load_gaze_csv(csv_path)        -> dict: gaze_frame_num (int) -> (gaze_x, gaze_y)
- load_hand_json(json_path)      -> dict: frame_key -> {'left': (x,y)|None, 'right': (x,y)|None}
- normalize_hand_coords(coords, img_w, img_h) -> (x, y) in [0,1]
- list_clips(dataset_root, dataset_name) -> list of clip dicts
- get_frame_num_from_path(path)  -> int frame number embedded in filename
"""

from __future__ import annotations

import csv
import json
import os
from typing import Optional


# ── Gaze loading ──────────────────────────────────────────────────────────────

def load_gaze_csv(csv_path: str) -> dict[int, tuple[float, float]]:
    """
    Load gaze mapping CSV for one clip.

    CSV columns: frame_idx, gaze_frame_num, matched_gaze_frame, gaze_x, gaze_y, is_exact_match

    The CSV has one row per ORIGINAL video frame (e.g., 30 FPS), while actual
    image files are downsampled (e.g., every 3rd frame = 10 FPS).

    Returns:
        Dict: gaze_frame_num (int) → (gaze_x, gaze_y) in [0,1]
        Key is the original video frame number as in the image filename.
    """
    result: dict[int, tuple[float, float]] = {}
    if not os.path.exists(csv_path):
        return result
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Use gaze_frame_num to match against image filenames
                fnum = int(row["gaze_frame_num"])
                gx   = float(row["gaze_x"])
                gy   = float(row["gaze_y"])
                result[fnum] = (gx, gy)
            except (KeyError, ValueError):
                continue
    return result


def get_frame_num_from_path(frame_path: str) -> int:
    """
    Extract original video frame number from image filename.

    Filename format: "{clip_stem}_{frame_num}.jpg"
    e.g., "12668_13163_12671.jpg" → 12671
    """
    stem = os.path.splitext(os.path.basename(frame_path))[0]  # "12668_13163_12671"
    return int(stem.rsplit("_", 1)[1])


# ── Hand loading ──────────────────────────────────────────────────────────────

def load_hand_json(json_path: str) -> dict[str, dict]:
    """
    Load hand locations JSON for one video.

    Returns:
        Dict keyed by "{start}_{end}_{frame}.jpg".
        Value: {'left': [x, y] or None, 'right': [x, y] or None}
        Coordinates are in native image resolution.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        return json.load(f)


def normalize_hand_coords(
    coord: Optional[list],
    img_w: float,
    img_h: float,
) -> Optional[tuple[float, float]]:
    """Normalize hand coordinate from native resolution to [0, 1]."""
    if coord is None:
        return None
    x = max(0.0, min(1.0, coord[0] / img_w))
    y = max(0.0, min(1.0, coord[1] / img_h))
    return (x, y)


# ── Clip enumeration ───────────────────────────────────────────────────────────

def list_clips(dataset_root: str, dataset_name: str) -> list[dict]:
    """
    Enumerate all clips in a dataset split.

    For ego4d and egoexo:
        gaze_mapping/{video_id}/{start}_{end}_mapping.csv
        video_id = directory name

    For egtea:
        gaze_mapping/{recipe}/{start}_{end}_mapping.csv
        video_id = recipe name (e.g., "OP01-R01-PastaSalad")

    Returns list of dicts with keys:
        dataset, video_id, start, end,
        gaze_csv, hand_json, frame_dir
    """
    gaze_map_dir = os.path.join(dataset_root, dataset_name, "gaze_mapping")
    hand_loc_dir = os.path.join(dataset_root, dataset_name, "hand_locations")
    frame_dir_base = os.path.join(dataset_root, dataset_name, "no_gaze")

    clips = []
    if not os.path.isdir(gaze_map_dir):
        return clips

    for video_id in sorted(os.listdir(gaze_map_dir)):
        vid_dir = os.path.join(gaze_map_dir, video_id)
        if not os.path.isdir(vid_dir):
            continue
        hand_json = os.path.join(hand_loc_dir, f"{video_id}.json")
        frame_dir = os.path.join(frame_dir_base, video_id)
        for fname in sorted(os.listdir(vid_dir)):
            if not fname.endswith("_mapping.csv"):
                continue
            stem = fname[: -len("_mapping.csv")]  # e.g., "123_1205"
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                start = int(parts[0])
                end   = int(parts[1])
            except ValueError:
                continue
            gaze_csv = os.path.join(vid_dir, fname)
            clips.append({
                "dataset":   dataset_name,
                "video_id":  video_id,
                "start":     start,
                "end":       end,
                "clip_stem": stem,
                "gaze_csv":  gaze_csv,
                "hand_json": hand_json,
                "frame_dir": frame_dir,
            })
    return clips


def get_clip_frame_paths(frame_dir: str, clip_stem: str) -> list[str]:
    """
    Return sorted list of frame paths for a given clip.

    Frame files follow the pattern: {clip_stem}_{frame_num}.jpg
    e.g., "12314_13371_12314.jpg", "12314_13371_12317.jpg", ...
    """
    if not os.path.isdir(frame_dir):
        return []
    prefix = clip_stem + "_"
    paths = []
    for fname in sorted(os.listdir(frame_dir)):
        if fname.startswith(prefix) and fname.endswith(".jpg"):
            paths.append(os.path.join(frame_dir, fname))
    return paths


def get_image_size(frame_path: str) -> tuple[int, int]:
    """Return (width, height) of image without loading full pixels."""
    from PIL import Image
    with Image.open(frame_path) as img:
        return img.size  # (W, H)
