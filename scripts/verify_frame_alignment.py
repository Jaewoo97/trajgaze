#!/usr/bin/env python
"""Verify that a `original`/no-overlay frame tree is INDEX-ALIGNED with its `viz` counterpart.

Why this exists: both loaders address frames by index, so a tree that holds the right
NUMBER of frames but samples different moments desynchronises student and teacher and
raises no error — the epoch just gets quieter, not louder. Counting frames cannot detect
it, and worse, extract_sg_original_frames.sh derives its output count from the viz count,
so count parity is constructed rather than observed (see refix_egoexolearn_original.sh).

The only reliable signal is pixels: two encodes of the same moment differ by JPEG noise
plus the gaze marker, which lands around MAD 3-5. Misaligned pairs measure 10-70.

A bare MAD threshold is NOT sufficient on its own. High-motion clips differ more between
adjacent frames, so a correctly aligned stem can still read MAD 8-12 — four such stems were
initially misjudged as broken. The discriminating test is whether MAD is *minimised at zero
offset*: for an aligned stem, shifting viz by one frame makes it sharply worse (11.7 -> 34.3
on beeab6b2), while a misaligned stem has no such minimum. So a stem passes if it is either
comfortably below the absolute threshold, or clearly minimised at k=0.

  python scripts/verify_frame_alignment.py egoexolearn [tree]

`tree` defaults to $SG_ROOT/frames/{ds}/original. Exit code is non-zero if any stem fails.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
from PIL import Image

FRAME_RE = re.compile(r"frame_\d+\.jpg$")

# MAD below this = unambiguously the same moment. Clean low-motion stems measure 2.6-5.5.
THRESH = 6.0
# Above THRESH, accept only if MAD(k) forms a V centred on k=0: zero offset must be the
# argmin AND sit this far below the neighbour mean. Aligned high-motion stems measure
# ~0.5x; genuinely misaligned ones show no dip at all.
DIP = 0.85
SHIFTS = (-2, -1, 0, 1, 2)
SHIFT_FRACS = (0.25, 0.5, 0.75)
FRACS = (0.02, 0.25, 0.5, 0.75, 0.98)   # sample across the clip; drift grows with index


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def main() -> int:
    ds = sys.argv[1]
    sg = os.environ["SG_ROOT"]
    viz = f"{sg}/frames/{ds}/viz"
    tree = sys.argv[2] if len(sys.argv) > 2 else f"{sg}/frames/{ds}/original"

    stems = sorted(os.listdir(tree))
    ok, bad, skipped = [], [], []

    for s in stems:
        d = f"{tree}/{s}"
        if not os.path.isdir(d):
            continue
        # Filter to frame files: a stray dotfile would shift every index by one.
        fs = sorted(f for f in os.listdir(d) if FRAME_RE.match(f))
        nv = (sorted(f for f in os.listdir(f"{viz}/{s}") if FRAME_RE.match(f))
              if os.path.isdir(f"{viz}/{s}") else [])
        if not fs or not nv:
            skipped.append((s, "empty or no viz counterpart"))
            continue
        n = min(len(fs), len(nv))
        mads = []
        for f in FRACS:
            fn = fs[int(n * f)]
            pv = f"{viz}/{s}/{fn}"
            if not os.path.exists(pv):
                continue
            mads.append(float(np.abs(load(pv) - load(f"{d}/{fn}")).mean()))
        if not mads:
            skipped.append((s, "no shared filenames"))
            continue
        m = float(np.mean(mads))

        if m < THRESH:
            ok.append((s, m, "below threshold"))
            continue

        # Above threshold, decide by the SHAPE of MAD(k) rather than its level. An aligned
        # stem shows a V centred on k=0: correct frame, then one frame of motion either
        # side. A stem sampling a different moment is uniformly bad across the window with
        # no dip. Level alone cannot separate them, because fast motion inflates MAD(0) to
        # 8-11 on correctly aligned clips.
        by_k = {}
        for k in SHIFTS:
            vals = []
            for f in SHIFT_FRACS:
                i = int(n * f)
                j = i + k
                if 0 <= j < n and os.path.exists(f"{viz}/{s}/{fs[j]}"):
                    vals.append(float(np.abs(load(f"{viz}/{s}/{fs[j]}")
                                             - load(f"{d}/{fs[i]}")).mean()))
            if vals:
                by_k[k] = float(np.mean(vals))
        m0 = by_k.get(0)
        neigh = [v for k, v in by_k.items() if k != 0]
        dip = m0 < DIP * float(np.mean(neigh)) if (m0 is not None and neigh) else False
        argmin0 = m0 is not None and all(m0 <= v for v in neigh)
        if dip and argmin0:
            ok.append((s, m, f"V at k=0 ({m0:.1f} vs neighbour mean "
                             f"{float(np.mean(neigh)):.1f})"))
        else:
            nb = (",".join(f"{k:+d}:{by_k[k]:.1f}" for k in SHIFTS if k in by_k)
                  if by_k else "n/a")
            bad.append((s, m, len(fs), len(nv), nb))

    print(f"[verify] {ds}: {len(stems)} stems  ->  {len(ok)} aligned, "
          f"{len(bad)} MISALIGNED, {len(skipped)} skipped   (tree={tree})")
    if ok:
        arr = np.array([m for _, m, _ in ok])
        print(f"[verify]   aligned MAD: median {np.median(arr):.2f}  max {arr.max():.2f}")
        for s, m, why in sorted(ok, key=lambda r: -r[1])[:5]:
            if m >= THRESH:
                print(f"[verify]   (high-motion, accepted) {s}  MAD={m:.1f}  {why}")
    for s, why in skipped:
        print(f"[verify]   SKIP {s}: {why}")
    for s, m, a, b, nb in sorted(bad, key=lambda r: -r[1]):
        print(f"[verify]   MISALIGNED {s}  MAD={m:.1f}  neighbour={nb}  frames={a} viz={b}")

    if bad:
        print(f"[verify] FAIL — {len(bad)} stem(s) sample different moments than viz")
        return 1
    print("[verify] PASS — every stem is index-aligned with viz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
