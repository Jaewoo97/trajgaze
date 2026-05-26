"""
Parse HD-EPIC dense predicted hand masks (contours_preds/*.json) into a
StreamGaze-style hand_locations JSON per recording:

  {"frame_000123.jpg": {"left": [cx, cy], "right": [cx, cy]}, ...}

  - Decodes per-frame COCO RLE polygons (size [H, W]=[1408,1408] by default)
    using the helper in HD-EPIC/Hands-Masks/decode_json_to_masks.py.
  - Centroid (cx, cy) per hand = mean (x, y) of mask=1 pixels.
  - Coords scaled from the mask resolution (1408×1408) to the extracted-frame
    resolution (224×224) so they line up with /workspace/HD-EPIC/frames_extracted.
  - Mask file frame index → frame_{N+1:06d}.jpg name (ffmpeg's 1-indexed output).
  - Empty entries (`[]`) → both sides null.

NOTE: HD-EPIC gaze is in Aria MPS (radians, CPF frame) — projecting to image
pixels requires the projectaria_tools SDK. For this pipeline we leave gaze
absent for HD-EPIC; the StreamGaze trajectory code gracefully degrades
(gaze_mask = False, gaze-dependent terms zero out).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from pycocotools import mask as cocomask


MASK_H = 1408
MASK_W = 1408
OUT_H  = 224
OUT_W  = 224


def _decode_rle(rle: dict) -> np.ndarray:
    return cocomask.decode({"size": rle["size"],
                             "counts": rle["counts"].encode("utf-8")}).astype(np.uint8)


def _centroid_xy(bin_mask: np.ndarray):
    ys, xs = np.nonzero(bin_mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def parse_one(json_path: str, out_path: str) -> tuple[int, int, int]:
    """Returns (n_frames, n_left, n_right)."""
    data = json.load(open(json_path))
    sx = OUT_W / MASK_W
    sy = OUT_H / MASK_H
    out: dict[str, dict] = {}
    n_left = n_right = 0
    n_frames = 0
    for k in data:
        frame_idx = int(k)
        # ffmpeg writes 1-indexed names; mask frame 0 → frame_000001.jpg
        fname = f"frame_{frame_idx + 1:06d}.jpg"
        entry = data[k]
        left = right = None
        if isinstance(entry, dict):
            if "left" in entry and entry["left"]:
                m = _decode_rle(entry["left"])
                xy = _centroid_xy(m)
                if xy is not None:
                    left = [round(xy[0] * sx, 1), round(xy[1] * sy, 1)]
                    n_left += 1
            if "right" in entry and entry["right"]:
                m = _decode_rle(entry["right"])
                xy = _centroid_xy(m)
                if xy is not None:
                    right = [round(xy[0] * sx, 1), round(xy[1] * sy, 1)]
                    n_right += 1
        # only record when at least one side present (StreamGaze schema; loader
        # tolerates missing keys for absent frames)
        if left is not None or right is not None:
            out[fname] = {"left": left, "right": right}
        n_frames += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return n_frames, n_left, n_right


def _process_task(args_tuple):
    f, out, overwrite = args_tuple
    stem = os.path.basename(f)[:-len(".json")]
    if os.path.exists(out) and not overwrite:
        return (stem, "skip", 0, 0, 0, None)
    try:
        n, l, r = parse_one(f, out)
        return (stem, "ok", n, l, r, None)
    except Exception as e:
        return (stem, "err", 0, 0, 0, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/HD-EPIC/Hands-Masks/contours_preds")
    ap.add_argument("--dst", default="/workspace/HD-EPIC/hand_locations")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.json")))
    print(f"found {len(files)} mask JSONs · workers={args.workers}", flush=True)
    os.makedirs(args.dst, exist_ok=True)
    tasks = [(f, os.path.join(args.dst, f"{os.path.basename(f)[:-len('.json')]}.json"),
              args.overwrite) for f in files]

    tot_f = tot_l = tot_r = 0
    n_ok = n_skip = n_err = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_task, tasks), 1):
            stem, status, n, l, r, err = res
            if status == "ok":
                n_ok += 1; tot_f += n; tot_l += l; tot_r += r
                print(f"  [{i}/{len(files)}] {stem}: frames={n} left={l} right={r}  "
                      f"(elapsed {time.time()-t0:.0f}s)", flush=True)
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(f"  [{i}/{len(files)}] ERR {stem}: {err}", flush=True)

    print(f"\nDONE. ok={n_ok} skip={n_skip} err={n_err} → {args.dst}")
    print(f"      total frames={tot_f}  left detections={tot_l}  right detections={tot_r}")


if __name__ == "__main__":
    main()
