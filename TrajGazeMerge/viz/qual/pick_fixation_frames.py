"""Build the figure's frame strips from StreamGaze's own fixation episodes.

Why not by hand: an audit of the hand-picked `frames_sg.json` (see audit/audit_evidence.py)
found that 16 of 18 strips missed at least one of the fixation episodes the item's answer was
generated from, by 5-52 s. The episodes are in the dataset — `metadata/egtea.csv` lists every
one with its time span and the object gazed at — so the strip can be anchored on them exactly.

This uses NO answer information. StreamGaze generates every option of an item by permuting the
objects of the same episodes, so "show the episodes inside the causal window" is available to
the reader from the options alone; the ground truth picks none of the frames. (The renderer's
own --frame-select relevance/probe modes were measured and rejected, see the handoff.)

Rule, per item:
  anchors = the frame nearest each fixation episode inside the window, plus the window end
  (the query moment); the remaining cells are filled from a uniform grid, skipping anything
  within min-gap of an anchor. If there are more episodes than cells, the LAST ones win:
  StreamGaze builds each question from the episodes that end at the query timestamp.

  python scripts/viz_qual/pick_fixation_frames.py --idxs 9,10,... \
      --out scripts/viz_qual/frames_sg.json [--n-frames 8] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys

# repo root = .../trajgaze, three levels up from TrajGazeMerge/viz/qual/
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

from TrajGazeMerge.data.dataset import (MCQ_TASKS, QA_BASE, _find_dataset, _get_frame_paths,
                                        _parse_ts, _sample_paths)

# $SG_ROOT is set by env.sh; the fallback mirrors it for a bare invocation.
META = os.path.join(
    os.environ.get("SG_ROOT",
                   os.path.join(os.path.dirname(_REPO), "datasets/trajgazemerge/StreamGaze_v2")),
    "metadata", "egtea.csv")


def build_items(source_ds="egtea"):
    """The test-split items in the exact order StreamGazeMergeDataset builds them."""
    items = []
    for task in MCQ_TASKS:
        path = os.path.join(QA_BASE, f"{task}.json")
        if not os.path.exists(path):
            continue
        for entry in json.load(open(path)):
            stem = os.path.splitext(entry["video_path"])[0]
            if _find_dataset(stem) != source_ds:
                continue
            for q in entry.get("questions", []):
                if q.get("options"):
                    items.append(dict(stem=stem, task=task, question=q["question"],
                                      options=q["options"], answer=q["answer"].strip().upper(),
                                      time_stamp=q.get("time_stamp", ""), q=q))
    return items


def load_episodes(path=META):
    """video -> [{t0, t1, obj, near}], the fixation episodes every QA item is generated from."""
    eps: dict[str, list[dict]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rep = ast.literal_eval(r["representative_object"]).get("object_identity", "?")
            except Exception:
                rep = "?"
            try:
                near = [str(o.get("object_identity", "")) for o in
                        ast.literal_eval(r["other_objects_in_cropped_area"] or "[]")]
            except Exception:
                near = []
            eps.setdefault(r["video_source"], []).append(
                dict(t0=float(r["episode_start_time"]), t1=float(r["episode_end_time"]),
                     obj=rep, near=[n for n in near if n]))
    for v in eps.values():
        v.sort(key=lambda e: e["t0"])
    return eps


def frame_times(item):
    """(T, [sec per group t], last_input_sec).

    tsec[t] is the frame the renderer shows for group t (vi = 2t). last_input_sec is the last
    of the 128 frames the MODEL gets, which is later than tsec[-1] and ~0.8% short of the
    causal cutoff, because _sample_paths steps by int(i*L/128) and never returns the tail.
    """
    vlm = _sample_paths(_get_frame_paths(item["stem"], "egtea",
                                         _parse_ts(item["time_stamp"])), 128)
    L = len(vlm)
    if not L:
        return 0, [], 0.0
    num = [int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)) for p in vlm]
    T = (L + 1) // 2
    return (T,                                                        # frames are 10 fps
            [num[min(2 * t, L - 1)] / 10.0 for t in range(T)],
            num[L - 1] / 10.0)


def sharpness(path, width=160):
    """Variance of the Laplacian — low on the motion blur an egocentric clip is full of.

    Only ever used to nudge a FILLER frame to a legible neighbour; anchors stay pinned to
    their episode, and the score sees no question, options or answer.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("L")
    im = im.resize((width, max(1, int(width * im.height / im.width))), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def pick(item, eps, nf=8, sharp_window=2):
    """(strip, anchor_labels) for one item."""
    T, tsec, last_in = frame_times(item)
    if T < nf:
        return list(range(T)), []
    anchors, labels = [], []
    for e in eps.get(item["stem"], []):
        if e["t1"] < tsec[0] or e["t0"] > last_in:                    # not in the window
            continue
        mid = (e["t0"] + e["t1"]) / 2
        t = min(range(T), key=lambda t: abs(tsec[t] - mid))
        if t in anchors:
            continue
        anchors.append(t)
        labels.append(f"t{t}@{tsec[t]:.0f}s {e['obj']}"
                      + (f"+{','.join(e['near'])}" if e["near"] else "")
                      + (f" [episode {e['t0']:.0f}-{e['t1']:.0f}s, "
                         f"off {abs(tsec[t] - mid):.0f}s]"))
    anchors = anchors[-(nf - 1):]                     # most recent episodes win a tie for space
    keep = sorted(set(anchors + [T - 1]))
    min_gap = max(2, T // 24)
    for f in [int(round(i * (T - 1) / (nf - 1))) for i in range(nf)]:
        if len(keep) >= nf:
            break
        if all(abs(f - k) >= min_gap for k in keep):
            keep = sorted(keep + [f])
    while len(keep) > nf:                             # trim the tightest filler, never an anchor
        cand = [(min(keep[i] - keep[i - 1], keep[i + 1] - keep[i]), i)
                for i in range(1, len(keep) - 1) if keep[i] not in anchors]
        if not cand:
            break
        keep.pop(min(cand)[1])

    if sharp_window:
        # a filler that lands mid-saccade shows a smear of the ceiling; slide it to the
        # sharpest frame within +-sharp_window groups, keeping order and the min gap
        vlm = _sample_paths(_get_frame_paths(item["stem"], "egtea",
                                             _parse_ts(item["time_stamp"])), 128)
        L = len(vlm)
        for i, t in enumerate(keep):
            if t in anchors or t == T - 1 or t == 0:
                continue
            lo = max(keep[i - 1] + min_gap, t - sharp_window) if i else t - sharp_window
            hi = (min(keep[i + 1] - min_gap, t + sharp_window) if i + 1 < len(keep)
                  else t + sharp_window)
            cands = [c for c in range(max(0, lo), min(T - 1, hi) + 1) if c not in keep or c == t]
            if len(cands) > 1:
                keep[i] = max(cands, key=lambda c: sharpness(vlm[min(2 * c, L - 1)]))
        keep = sorted(set(keep))
    return keep, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idxs", required=True, help="comma-separated item indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--sharp-window", type=int, default=2,
                    help="slide each filler frame up to this many groups to the sharpest "
                         "neighbour (0 = keep the uniform grid). Anchors never move.")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    items, eps = build_items(), load_episodes()
    strips = {}
    for idx in [int(x) for x in args.idxs.split(",")]:
        it = items[idx]
        strip, labels = pick(it, eps, args.n_frames, args.sharp_window)
        strips[str(idx)] = strip
        print(f"idx={idx:<5} {it['task']:<40} {len(labels)} episodes in window")
        for lb in labels:
            print(f"        {lb}")
        print(f"        strip = {strip}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return
    with open(args.out, "w") as f:
        json.dump(strips, f, indent=1)
        f.write("\n")
    print(f"\n-> {args.out}  ({len(strips)} items)")


if __name__ == "__main__":
    main()
