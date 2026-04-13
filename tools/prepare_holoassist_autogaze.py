"""
Prepare HoloAssist clips from StreamGaze_v2 JPEG frames for AutoGaze NTP training.

Source:
  /workspace/datasets/StreamGaze_v2/frames/holoassist/viz/{video}/frame_XXXXXX.jpg  (10 FPS)
  /workspace/datasets/StreamGaze_v2/gaze/holoassist/viz/{video}.json  (per-frame gaze)

Output:
  /workspace/datasets/StreamGaze/autogaze_data/clips/holoassist/{video}_clip{N:04d}.mp4

Clip parameters match EgoExoLearn/EGTEA:
  CLIP_LEN    = 16 frames
  CLIP_STRIDE = 8  frames (50% overlap)
  MIN_GAZE_FRAMES = 8 (min frames with valid gaze per clip)

Usage:
    conda run -n gaze python tools/prepare_holoassist_autogaze.py
"""

import os
import json
import subprocess
from multiprocessing import Pool

# ── Paths ─────────────────────────────────────────────────────────────────────
FRAMES_BASE = "/workspace/datasets/StreamGaze_v2/frames/holoassist/viz"
GAZE_BASE   = "/workspace/datasets/StreamGaze_v2/gaze/holoassist/viz"
OUT_DIR     = "/workspace/datasets/StreamGaze/autogaze_data/clips/holoassist"

# ── Clip parameters ───────────────────────────────────────────────────────────
TARGET_FPS      = 10
CLIP_LEN        = 16   # frames per clip
CLIP_STRIDE     = 8    # stride in frames (50% overlap)
MIN_GAZE_FRAMES = 8    # min frames with valid gaze to keep a clip
N_WORKERS       = 16


def make_clip_ffmpeg(frame_dir: str, frame_names: list[str], out_path: str) -> bool:
    """Create MP4 from a list of JPEG filenames using ffmpeg concat demuxer."""
    if os.path.exists(out_path):
        return True

    concat_list = out_path + ".txt"
    with open(concat_list, "w") as f:
        for name in frame_names:
            f.write(f"file '{os.path.join(frame_dir, name)}'\n")
            f.write(f"duration {1.0 / TARGET_FPS}\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-r", str(TARGET_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    os.remove(concat_list)
    return result.returncode == 0 and os.path.exists(out_path)


def process_video(video_name: str) -> tuple[str, int]:
    frame_dir = os.path.join(FRAMES_BASE, video_name)
    gaze_path = os.path.join(GAZE_BASE, video_name + ".json")

    if not os.path.isdir(frame_dir) or not os.path.exists(gaze_path):
        return video_name, 0

    with open(gaze_path) as f:
        gaze = json.load(f)  # {frame_XXXXXX.jpg: [x,y] or None}

    all_frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not all_frames:
        return video_name, 0

    os.makedirs(OUT_DIR, exist_ok=True)

    n_clips = 0
    clip_idx = 0
    start = 0

    while start + CLIP_LEN <= len(all_frames):
        clip_frames = all_frames[start: start + CLIP_LEN]

        # Count frames with valid gaze
        n_valid = sum(1 for fn in clip_frames if gaze.get(fn) is not None)

        if n_valid >= MIN_GAZE_FRAMES:
            clip_name = f"{video_name}_clip{clip_idx:04d}.mp4"
            out_path  = os.path.join(OUT_DIR, clip_name)
            ok = make_clip_ffmpeg(frame_dir, clip_frames, out_path)
            if ok:
                n_clips += 1

        start    += CLIP_STRIDE
        clip_idx += 1

    return video_name, n_clips


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    videos = sorted(
        d for d in os.listdir(FRAMES_BASE)
        if os.path.isdir(os.path.join(FRAMES_BASE, d))
    )
    print(f"HoloAssist videos found: {len(videos)}")

    existing = len([f for f in os.listdir(OUT_DIR) if f.endswith(".mp4")])
    print(f"Existing clips in output: {existing}")

    with Pool(N_WORKERS) as pool:
        results = pool.starmap(process_video, [(v,) for v in videos])

    total = 0
    for name, n in results:
        print(f"  {name}: {n} clips")
        total += n

    print(f"\nTotal HoloAssist clips created: {total}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
