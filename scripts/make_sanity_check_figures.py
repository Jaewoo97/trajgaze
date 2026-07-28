#!/usr/bin/env python
"""Side-by-side viz vs no-overlay frames for every dataset, into docs/sanity_check/.

Purpose: make the index-alignment property *visible*. The overlay-free tree must show the
same moment as its viz counterpart with only the gaze marker removed. When extraction goes
wrong the two panels show different scenes entirely — which is exactly what happened to
54/180 egoexolearn stems and 13/66 holoassist stems, and what frame-count checks could not
detect (see kd_handoff_v2.md §7.4a).

Each figure samples several points across one clip so drift, which grows with frame index,
cannot hide at the start. MAD (mean absolute difference) is printed per pair: ~3-5 means the
same moment, 10+ means misaligned.

  python scripts/make_sanity_check_figures.py [dataset ...]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

SG = os.environ["SG_ROOT"]
EG = os.environ["EG_ROOT"]
OUT = os.environ.get("SANITY_OUT",
                     f"{os.environ['REPO']}/TrajGazeMerge/docs/sanity_check")

# (name, viz_root, nooverlay_root, viz_label, nooverlay_label)
SOURCES = [
    ("SG-egtea",       f"{SG}/frames/egtea/viz",       f"{SG}/frames/egtea/original",       "viz", "original"),
    ("SG-egoexolearn", f"{SG}/frames/egoexolearn/viz", f"{SG}/frames/egoexolearn/original", "viz", "original"),
    ("SG-holoassist",  f"{SG}/frames/holoassist/viz",  f"{SG}/frames/holoassist/original",  "viz", "original"),
    ("EG-egtea",       f"{EG}/egtea/gaze",             f"{EG}/egtea/no_gaze",               "gaze", "no_gaze"),
    ("EG-ego4d",       f"{EG}/ego4d/gaze",             f"{EG}/ego4d/no_gaze",               "gaze", "no_gaze"),
    ("EG-egoexo",      f"{EG}/egoexo/gaze",            f"{EG}/egoexo/no_gaze",              "gaze", "no_gaze"),
]

FRACS = (0.05, 0.30, 0.55, 0.80)     # spread across the clip so drift cannot hide early
PANEL = 340                          # px per panel edge
PAD, HDR, LBL = 8, 46, 26


def load(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def pick_stem(viz_root: str, nov_root: str) -> str | None:
    """First stem present in both trees with frames on each side."""
    if not (os.path.isdir(viz_root) and os.path.isdir(nov_root)):
        return None
    for s in sorted(os.listdir(nov_root)):
        a, b = f"{viz_root}/{s}", f"{nov_root}/{s}"
        if os.path.isdir(a) and os.path.isdir(b) and os.listdir(a) and os.listdir(b):
            return s
    return None


def build(name: str, viz_root: str, nov_root: str, lv: str, ln: str) -> str | None:
    stem = pick_stem(viz_root, nov_root)
    if stem is None:
        print(f"[sanity] {name}: no usable stem — SKIP")
        return None

    files = sorted(os.listdir(f"{nov_root}/{stem}"))
    vfiles = set(os.listdir(f"{viz_root}/{stem}"))
    shared = [f for f in files if f in vfiles]
    if not shared:
        print(f"[sanity] {name}: no shared filenames between the two trees — SKIP")
        return None
    picks = [shared[min(int(len(shared) * f), len(shared) - 1)] for f in FRACS]

    cols = len(picks)
    W = cols * PANEL + (cols + 1) * PAD
    H = HDR + 2 * (PANEL + LBL) + 3 * PAD
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 8), f"{name}   stem={stem}   {len(shared)} shared frames",
           fill=(15, 15, 15))
    d.text((PAD, 26), f"top: {lv} (marker present)    bottom: {ln} (marker removed)"
                      "    MAD 3-5 = same moment | 10+ = MISALIGNED", fill=(90, 90, 90))

    mads = []
    for c, fn in enumerate(picks):
        x = PAD + c * (PANEL + PAD)
        va, na = load(f"{viz_root}/{stem}/{fn}"), load(f"{nov_root}/{stem}/{fn}")
        mad = float(np.abs(va - na).mean()) if va.shape == na.shape else float("nan")
        mads.append(mad)

        for r, (im, lab) in enumerate([(va, lv), (na, ln)]):
            y = HDR + PAD + r * (PANEL + LBL + PAD)
            pil = Image.fromarray(im.astype(np.uint8))
            pil.thumbnail((PANEL, PANEL))
            canvas.paste(pil, (x + (PANEL - pil.width) // 2, y))
            ok = mad < 6
            txt = (f"{lab}  {fn}" if r == 0
                   else f"{lab}  MAD={mad:.2f}  {'OK' if ok else 'MISALIGNED'}")
            d.text((x, y + PANEL + 5), txt,
                   fill=(15, 15, 15) if (r == 0 or ok) else (185, 25, 25))

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    canvas.save(path)
    m = np.nanmean(mads)
    print(f"[sanity] {name}: stem={stem}  MAD mean={m:.2f}  "
          f"{'OK' if m < 6 else 'MISALIGNED'}  -> {path}")
    return path


def main() -> int:
    want = set(sys.argv[1:])
    todo = [s for s in SOURCES if not want or s[0] in want]
    made = [p for p in (build(*s) for s in todo) if p]
    print(f"\n[sanity] {len(made)} figure(s) written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
