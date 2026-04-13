"""
Pre-extract full videos as JPEG frames at 10 FPS, mirroring the source
directory structure under the output root.

Source layout:
  StreamGaze_v2/videos/egoexolearn/original/<video>.mp4
  StreamGaze_v2/videos/egtea/original/<video>.mp4
  StreamGaze_v2/videos/holoassist/original/<video>.mp4

Output layout (mirrors source):
  StreamGaze_v2/frames/egoexolearn/original/<video_stem>/frame_%06d.jpg
  StreamGaze_v2/frames/egtea/original/<video_stem>/frame_%06d.jpg
  StreamGaze_v2/frames/holoassist/original/<video_stem>/frame_%06d.jpg

At eval time, frames for a specific response_time window [start, end] are
loaded by index (10 FPS → frame N = second N/10):
  start_frame = int(start_sec * 10)   (0-indexed)
  end_frame   = int(end_sec   * 10)   (0-indexed)

Usage:
  python eval/preprocess_frames.py \
      --video-root /workspace/datasets/StreamGaze_v2/videos \
      --output     /workspace/datasets/StreamGaze_v2/frames \
      [--workers 8]
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

QA_DIR     = "/workspace/datasets/StreamGaze_v2/qa"
VIDEO_ROOT = "/workspace/datasets/StreamGaze_v2/videos"

# Map video filename → (full src path, relative subdir e.g. "egoexolearn/original")
VIDEO_MAP: dict[str, tuple[str, str]] = {}
for _dataset in os.listdir(VIDEO_ROOT):
    for _split in ("original", "viz"):
        _split_dir = os.path.join(VIDEO_ROOT, _dataset, _split)
        if not os.path.isdir(_split_dir):
            continue
        for _f in os.listdir(_split_dir):
            if _f.endswith(".mp4"):
                VIDEO_MAP[_f] = (
                    os.path.join(_split_dir, _f),
                    os.path.join(_dataset, _split),
                )


def collect_videos() -> list[str]:
    """Collect all unique video filenames referenced across all QA files."""
    videos = set()
    for fname in os.listdir(QA_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(QA_DIR, fname)) as f:
            data = json.load(f)
        for s in data:
            videos.add(s["video_path"])
    return sorted(videos)


def extract_video(video_path: str, out_root: str) -> str | None:
    """Extract all frames of a video at 10 FPS, mirroring the source subdir."""
    entry = VIDEO_MAP.get(video_path)
    if entry is None:
        return None

    src, subdir = entry
    stem      = video_path.replace(".mp4", "")
    frame_dir = os.path.join(out_root, subdir, stem)

    if os.path.isdir(frame_dir) and len(os.listdir(frame_dir)) > 0:
        return frame_dir  # already extracted

    os.makedirs(frame_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-vf", "fps=10", "-q:v", "2",
         os.path.join(frame_dir, "frame_%06d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return frame_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root",
                        default="/workspace/datasets/StreamGaze_v2/videos",
                        help="Root directory containing dataset/split/video.mp4 structure.")
    parser.add_argument("--output",
                        default="/workspace/datasets/StreamGaze_v2/frames",
                        help="Root directory to save frames (mirrors video-root structure).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel ffmpeg workers.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    videos = collect_videos()
    print(f"Unique videos to extract: {len(videos)}")

    missing = [v for v in videos if v not in VIDEO_MAP]
    if missing:
        print(f"WARNING: {len(missing)} videos not found — will be skipped:")
        for v in missing:
            print(f"  {v}")

    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_video, v, args.output): v for v in videos}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            if future.result() is None:
                skipped += 1

    print(f"Done. {len(videos) - skipped} videos extracted, {skipped} skipped.")
    print(f"Frames saved → {args.output}")


if __name__ == "__main__":
    main()
