"""
Prepare StreamGaze dataset for AutoGaze NTP training (2-fold cross-evaluation).

Datasets: EGTEA (69 videos) and EgoExoLearn (180 videos).
  Fold A: train=EgoExoLearn, val=EGTEA
  Fold B: train=EGTEA,       val=EgoExoLearn

Steps:
  1. Extract clips from all videos into a shared pool:
       autogaze_data/clips/{egtea,egoexolearn}/*.mp4
  2. Create fold directories with symlinks into the pool:
       autogaze_data/fold_a/train/ -> egoexolearn clips
       autogaze_data/fold_a/val/   -> egtea clips
       autogaze_data/fold_b/train/ -> egtea clips
       autogaze_data/fold_b/val/   -> egoexolearn clips

Usage:
    conda run -n gaze python prepare_streamgaze_autogaze.py
"""

import os
import csv
import json
import subprocess
from multiprocessing import Pool

# ── Paths ─────────────────────────────────────────────────────────────────────
GAZE_DIR  = "/workspace/datasets/StreamGaze/dataset_parsed_gaze"
VIDEO_DIR = "/workspace/datasets/StreamGaze/videos/original_video"
META_DIR  = "/workspace/datasets/StreamGaze/metadata"
OUT_DIR   = "/workspace/datasets/StreamGaze/autogaze_data"

# ── Clip parameters ───────────────────────────────────────────────────────────
TARGET_FPS        = 10
CLIP_LEN          = 16      # frames per clip  (= max_num_frames in AutoGaze)
CLIP_STRIDE       = 8       # stride in frames (50% overlap)
MIN_GAZE_FRAMES   = 8       # min frames with valid gaze to keep a clip
GAZE_SEARCH_RANGE = 6       # ±N original-FPS frames to search for nearest gaze


# ── Dataset membership ────────────────────────────────────────────────────────

def build_video_to_dataset():
    mapping = {}
    for ds in ("egtea", "egoexolearn"):
        with open(os.path.join(META_DIR, f"{ds}.csv")) as f:
            for row in csv.DictReader(f):
                mapping[row["video_source"]] = ds
    return mapping


# ── Gaze helpers ──────────────────────────────────────────────────────────────

def build_gaze_lookup(gaze_frames):
    return {f["frame"]: (f["x"], f["y"]) for f in gaze_frames if f["x"] is not None}


def get_gaze_for_frame(frame_idx, gaze_lookup):
    if frame_idx in gaze_lookup:
        return gaze_lookup[frame_idx]
    for delta in range(1, GAZE_SEARCH_RANGE + 1):
        if frame_idx + delta in gaze_lookup:
            return gaze_lookup[frame_idx + delta]
        if frame_idx - delta in gaze_lookup:
            return gaze_lookup[frame_idx - delta]
    return None


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clip_ffmpeg(src_video, start_sec, duration_sec, out_path):
    if os.path.exists(out_path):
        return True   # already extracted
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", src_video,
        "-t", str(duration_sec),
        "-r", str(TARGET_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(out_path)


def process_video(video_stem, dataset):
    """
    Extract clips for one video into autogaze_data/clips/{dataset}/.
    Returns list of absolute clip paths that were successfully written.
    """
    gaze_path  = os.path.join(GAZE_DIR, video_stem + ".json")
    video_path = os.path.join(VIDEO_DIR, video_stem + ".mp4")

    if not os.path.exists(gaze_path) or not os.path.exists(video_path):
        return []

    with open(gaze_path) as f:
        gaze_data = json.load(f)

    orig_fps       = gaze_data["fps"]
    n_frames       = gaze_data["num_frames"]
    gaze_lookup    = build_gaze_lookup(gaze_data["frames"])

    if not gaze_lookup:
        return []

    clip_dur_sec     = CLIP_LEN / TARGET_FPS
    clip_frames_orig = int(clip_dur_sec * orig_fps)
    stride_orig      = int(CLIP_STRIDE / TARGET_FPS * orig_fps)

    clip_dir = os.path.join(OUT_DIR, "clips", dataset)
    os.makedirs(clip_dir, exist_ok=True)

    results  = []
    clip_idx = 0
    start    = 0

    while start + clip_frames_orig <= n_frames:
        # Check gaze coverage
        sampled = [start + int(i * clip_frames_orig / CLIP_LEN) for i in range(CLIP_LEN)]
        n_valid = sum(1 for fr in sampled if get_gaze_for_frame(fr, gaze_lookup) is not None)

        if n_valid >= MIN_GAZE_FRAMES:
            clip_name = f"{video_stem}_clip{clip_idx:04d}.mp4"
            clip_path = os.path.join(clip_dir, clip_name)
            ok = extract_clip_ffmpeg(video_path, start / orig_fps, clip_dur_sec, clip_path)
            if ok:
                results.append(clip_path)

        start    += stride_orig
        clip_idx += 1

    return results


# ── Fold directory creation ───────────────────────────────────────────────────

def make_fold_symlinks(fold_name, train_dataset, val_dataset):
    """
    Create fold_name/train/ and fold_name/val/ with symlinks to shared clips.
    """
    for split, ds in (("train", train_dataset), ("val", val_dataset)):
        link_dir = os.path.join(OUT_DIR, fold_name, split)
        os.makedirs(link_dir, exist_ok=True)
        src_dir  = os.path.join(OUT_DIR, "clips", ds)

        if not os.path.exists(src_dir):
            continue

        created = 0
        for fname in os.listdir(src_dir):
            if not fname.endswith(".mp4"):
                continue
            src  = os.path.join(src_dir, fname)
            dest = os.path.join(link_dir, fname)
            if not os.path.exists(dest):
                os.symlink(src, dest)
                created += 1

        print(f"  {fold_name}/{split}: {created} symlinks → clips/{ds}/")


N_WORKERS = 64  # parallel ffmpeg workers


def _process_video_worker(args):
    stem, ds = args
    clips = process_video(stem, ds)
    return stem, ds, len(clips)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    video_to_dataset = build_video_to_dataset()

    gaze_stems = sorted(
        f.replace(".json", "")
        for f in os.listdir(GAZE_DIR)
        if f.endswith(".json")
    )

    tasks = [(stem, video_to_dataset[stem]) for stem in gaze_stems if stem in video_to_dataset]

    # Step 1: Extract all clips into shared pool (parallel)
    print(f"=== Step 1: Extracting clips ({N_WORKERS} workers) ===")
    counts = {"egtea": 0, "egoexolearn": 0}

    with Pool(N_WORKERS) as pool:
        for i, (stem, ds, n) in enumerate(pool.imap_unordered(_process_video_worker, tasks)):
            counts[ds] += n
            print(f"[{i+1}/{len(tasks)}] {stem} ({ds}): {n} clips", flush=True)

    print(f"\nClips extracted: egtea={counts['egtea']}, egoexolearn={counts['egoexolearn']}")

    # Step 2: Create fold directories with symlinks
    print("\n=== Step 2: Creating fold symlinks ===")
    #   Fold A: train=egoexolearn, val=egtea
    make_fold_symlinks("fold_a", train_dataset="egoexolearn", val_dataset="egtea")
    #   Fold B: train=egtea, val=egoexolearn
    make_fold_symlinks("fold_b", train_dataset="egtea", val_dataset="egoexolearn")

    print("\nDone. Directory structure:")
    print("  autogaze_data/clips/egtea/        ← all EGTEA clips")
    print("  autogaze_data/clips/egoexolearn/  ← all EgoExoLearn clips")
    print("  autogaze_data/fold_a/train/       ← EgoExoLearn (symlinks)")
    print("  autogaze_data/fold_a/val/         ← EGTEA       (symlinks)")
    print("  autogaze_data/fold_b/train/       ← EGTEA       (symlinks)")
    print("  autogaze_data/fold_b/val/         ← EgoExoLearn (symlinks)")
    print("\nNext: run greedy_search_gt.py --fold fold_a (and fold_b)")


if __name__ == "__main__":
    main()
