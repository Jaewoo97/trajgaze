"""
Sanity-check visualization for projected HD-EPIC gaze.

Loads a recording's gaze JSON, picks N evenly-spaced frames, overlays the
gaze marker (red dot + crosshair) at the normalized coords, saves a 1xN
horizontal montage to /tmp/.
"""
import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="P01-20240202-110250")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default="/tmp/hdepic_gaze_overlay.png")
    args = ap.parse_args()

    participant = args.stem.split("-", 1)[0]
    fdir = f"/workspace/HD-EPIC/frames_extracted/{participant}/{args.stem}"
    gpath = f"/workspace/HD-EPIC/gaze/{args.stem}.json"

    with open(gpath) as f:
        gaze = json.load(f)

    frames = sorted(os.listdir(fdir))
    keep_keys = [k for k in frames if k in gaze and gaze[k] is not None]
    if not keep_keys:
        raise SystemExit(f"No frames with gaze for {args.stem}")
    step = max(1, len(keep_keys) // args.n_frames)
    picked = keep_keys[::step][:args.n_frames]

    tiles = []
    tile_size = 224
    for fname in picked:
        img = Image.open(os.path.join(fdir, fname)).convert("RGB")
        img = img.resize((tile_size, tile_size))
        nx, ny = gaze[fname]
        px, py = int(nx * tile_size), int(ny * tile_size)
        d = ImageDraw.Draw(img)
        # crosshair
        d.line([(px-10, py), (px+10, py)], fill=(255, 0, 0), width=2)
        d.line([(px, py-10), (px, py+10)], fill=(255, 0, 0), width=2)
        d.ellipse([(px-4, py-4), (px+4, py+4)], outline=(255, 0, 0), width=2)
        # label
        d.text((4, 4), fname.replace("frame_", "").replace(".jpg", ""),
               fill=(255, 255, 0))
        tiles.append(img)

    W = tile_size * len(tiles)
    montage = Image.new("RGB", (W, tile_size + 24), (32, 32, 32))
    for i, t in enumerate(tiles):
        montage.paste(t, (i * tile_size, 0))
    d = ImageDraw.Draw(montage)
    d.text((4, tile_size + 4),
           f"{args.stem}: red dot = projected MPS gaze (normalized → pixel)",
           fill=(255, 255, 255))
    montage.save(args.out)
    print(f"saved → {args.out}  ({len(tiles)} frames at {tile_size}px)")


if __name__ == "__main__":
    main()
