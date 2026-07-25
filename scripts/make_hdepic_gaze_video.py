"""Build a video of HD-EPIC frames with the projected gaze overlay.

Picks a recording, takes a contiguous chunk of frames, draws the gaze marker on
each, writes frames to a tmp dir, encodes to MP4 with ffmpeg.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

ROOT = "/workspace/HD-EPIC"
DEFAULT_STEM = "P03-20240217-210958"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--start", type=int, default=8800,
                    help="First frame index to include")
    ap.add_argument("--count", type=int, default=300,
                    help="Number of consecutive frames to include")
    ap.add_argument("--fps", type=int, default=10,
                    help="Output video framerate (HD-EPIC frames extracted at 10fps)")
    ap.add_argument("--scale", type=int, default=3,
                    help="Upscale factor (nearest) for the 224x224 frames")
    ap.add_argument("--out", default="/tmp/hdepic_gaze_overlay.mp4")
    args = ap.parse_args()

    participant = args.stem.split("-", 1)[0]
    fdir = os.path.join(ROOT, "frames_extracted", participant, args.stem)
    gpath = os.path.join(ROOT, "gaze", f"{args.stem}.json")
    with open(gpath) as f:
        gaze = json.load(f)

    avail = sorted(os.listdir(fdir))
    # Filter to consecutive range starting at --start
    picked = []
    for fname in avail:
        try:
            idx = int(fname[len("frame_"):-len(".jpg")])
        except ValueError:
            continue
        if args.start <= idx < args.start + args.count:
            picked.append(fname)
    if not picked:
        # Fall back to first count frames
        picked = avail[:args.count]
    picked.sort()

    W = H = 224 * args.scale
    tmp = tempfile.mkdtemp()
    try:
        for i, fname in enumerate(picked):
            img = Image.open(os.path.join(fdir, fname)).convert("RGB")
            img = img.resize((W, H), Image.NEAREST)
            dr = ImageDraw.Draw(img)
            g = gaze.get(fname)
            if g and g[0] is not None:
                nx, ny = g
                px, py = int(nx * W), int(ny * H)
                # Outer ring (high contrast)
                dr.ellipse([(px - 20, py - 20), (px + 20, py + 20)],
                           outline=(255, 255, 0), width=3)
                # Crosshair
                dr.line([(px - 30, py), (px + 30, py)], fill=(255, 0, 0), width=3)
                dr.line([(px, py - 30), (px, py + 30)], fill=(255, 0, 0), width=3)
                # Center dot
                dr.ellipse([(px - 5, py - 5), (px + 5, py + 5)], fill=(255, 0, 0))
                gaze_text = f"gaze=({nx:.3f}, {ny:.3f})"
            else:
                gaze_text = "gaze=missing"
            # Header bar
            dr.rectangle([(0, 0), (W, 28)], fill=(0, 0, 0))
            dr.text((8, 7),
                    f"{args.stem}  {fname}  {gaze_text}",
                    fill=(255, 255, 255))
            img.save(os.path.join(tmp, f"f_{i:06d}.png"))

        # ffmpeg encode
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(tmp, "f_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            "-movflags", "+faststart",
            args.out,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"saved → {args.out}  ({len(picked)} frames @ {args.fps}fps = "
              f"{len(picked)/args.fps:.1f}s, {W}x{H})")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
