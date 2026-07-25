"""Render a gaze-overlay mirror of HD-EPIC extracted frames.

For every frame under FRAMES_EXTRACTED/P##/{stem}/frame_*.jpg, draw a gaze
marker (red ring + green center dot, matching StreamGaze's viz style) at the
normalized gaze coordinate from GAZE/{stem}.json, and write the result to the
mirror tree FRAMES_GAZE/P##/{stem}/frame_*.jpg.

Frames whose gaze entry is missing/None are copied through WITHOUT a marker
(gaze-absent for that frame) so the mirror stays a complete drop-in.

Parallel over recordings, resumable (skips frames already rendered).
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

from PIL import Image, ImageDraw

ROOT            = "/workspace/HD-EPIC"
FRAMES_EXTRACTED = os.path.join(ROOT, "frames_extracted")
FRAMES_GAZE     = os.path.join(ROOT, "frames_gaze")
GAZE            = os.path.join(ROOT, "gaze")

RING_R = 14   # red ring radius (px on 224)
DOT_R  = 4    # green center dot radius


def _recordings() -> list[str]:
    stems = []
    for part in sorted(os.listdir(FRAMES_EXTRACTED)):
        pdir = os.path.join(FRAMES_EXTRACTED, part)
        if not os.path.isdir(pdir):
            continue
        for stem in sorted(os.listdir(pdir)):
            if os.path.isdir(os.path.join(pdir, stem)):
                stems.append(stem)
    # largest recordings first → better load balance across workers
    stems.sort(key=lambda s: -len(os.listdir(
        os.path.join(FRAMES_EXTRACTED, s.split("-", 1)[0], s))))
    return stems


def render_recording(stem: str) -> tuple[str, int, int]:
    part = stem.split("-", 1)[0]
    src_dir = os.path.join(FRAMES_EXTRACTED, part, stem)
    dst_dir = os.path.join(FRAMES_GAZE, part, stem)
    os.makedirs(dst_dir, exist_ok=True)

    gpath = os.path.join(GAZE, f"{stem}.json")
    gaze = {}
    if os.path.exists(gpath):
        try:
            gaze = json.load(open(gpath))
        except Exception:
            gaze = {}

    done = skipped = 0
    for fname in os.listdir(src_dir):
        if not fname.endswith(".jpg"):
            continue
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            gv = gaze.get(fname)
            if gv and gv[0] is not None:
                w, h = img.size
                px = min(max(int(float(gv[0]) * w), 0), w - 1)
                py = min(max(int(float(gv[1]) * h), 0), h - 1)
                d = ImageDraw.Draw(img)
                d.ellipse([px - RING_R, py - RING_R, px + RING_R, py + RING_R],
                          outline=(255, 0, 0), width=2)
                d.ellipse([px - DOT_R, py - DOT_R, px + DOT_R, py + DOT_R],
                          fill=(0, 255, 0))
            img.save(dst, quality=90)
            done += 1
        except Exception as e:
            print(f"  [warn] {stem}/{fname}: {e}", flush=True)
    return stem, done, skipped


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    stems = _recordings()
    print(f"[render] {len(stems)} recordings, {n_workers} workers → {FRAMES_GAZE}",
          flush=True)
    t0 = time.time()
    tot_done = tot_skip = 0
    with Pool(n_workers) as pool:
        for i, (stem, done, skip) in enumerate(
                pool.imap_unordered(render_recording, stems), 1):
            tot_done += done
            tot_skip += skip
            print(f"[render] {i}/{len(stems)} {stem}: +{done} (skip {skip}) "
                  f"| total rendered={tot_done} skipped={tot_skip} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    print(f"[render] DONE: rendered={tot_done} skipped={tot_skip} "
          f"in {time.time()-t0:.0f}s", flush=True)
    print("HDEPIC_OVERLAY: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
