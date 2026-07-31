"""Compare two scan passes of the same items and report what did not reproduce.

Eval here is not deterministic. `kd_handoff_v2.md` §8 measures a 3-5 item spread when the SAME
checkpoint is re-scored with the SAME flags, and the upstream figure toolkit found the option
logits quantise to 1/8 in bf16, so exact ties between two option letters are common and a
near-tie in the top-k that picks the 3% complement swaps tokens between runs. A margin
threshold alone is therefore not a stability test — an item has to give the same answer twice,
in two independent processes.

    # pass 2 of the items a figure was made from, in a fresh process
    scripts/run_qual_kd.sh sg 0 --render-idxs 31,372,470 --scan-only --tag r2

    python TrajGazeMerge/viz/qual/check_stability.py \
        TrajGazeMerge/qual/scan/sg.jsonl TrajGazeMerge/qual/scan/sg_r2.jsonl

Exit code is 1 if any compared item flipped, so this can gate a release of the figure set.
"""
from __future__ import annotations

import argparse
import json
import sys

def row_keys(rec):
    """The per-model keys of a scan record, discovered rather than hardcoded.

    Two scan schemas share this checker: the v2 figure set writes
    `teacher` / `student_overlay` / `student_nooverlay`, the ViT-KD sheets write
    `teacher` / `student`. Any key whose value is a dict carrying a `pred` is a model row.
    """
    return tuple(k for k, v in rec.items() if isinstance(v, dict) and "pred" in v)


def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run1")
    ap.add_argument("run2")
    ap.add_argument("--idxs", default=None,
                    help="only these item indices (default: everything in both files)")
    ap.add_argument("--follow-tol", type=float, default=0.05,
                    help="flag an item whose teacher-tracking IoU moves more than this")
    args = ap.parse_args()

    a, b = load(args.run1), load(args.run2)
    shared = sorted(set(a) & set(b))
    if args.idxs:
        want = {int(x) for x in args.idxs.replace(" ", "").split(",") if x}
        shared = [i for i in shared if i in want]
    if not shared:
        print("no items in common — nothing to compare")
        return 0

    ROWS = row_keys(a[shared[0]])
    if not ROWS:
        print("no model rows found in the records"); return 0
    # the agreement scalar is named differently by the two producers; compare whichever is there
    AGREE = next((k for k in ("follow", "recall_traj") if k in a[shared[0]]), None)
    print(f"rows: {'/'.join(ROWS)}" + (f"   agreement metric: {AGREE}" if AGREE else ""))

    flipped, drifted = [], []
    for i in shared:
        p1 = tuple(a[i][r]["pred"] for r in ROWS)
        p2 = tuple(b[i][r]["pred"] for r in ROWS)
        d = abs(a[i][AGREE] - b[i][AGREE]) if AGREE else 0.0
        if p1 != p2:
            flipped.append((i, a[i]["task"], p1, p2,
                            min(a[i][r]["margin"] for r in ROWS)))
        elif d > args.follow_tol:
            drifted.append((i, a[i]["task"], a[i][AGREE], b[i][AGREE]))

    print(f"compared {len(shared)} items")
    if flipped:
        print(f"\nVERDICT FLIPPED — {len(flipped)}; do not use these as evidence:")
        print(f"  {'idx':<6}{'task':<42}{'run1':<14}{'run2':<14}min margin")
        for i, t, p1, p2, m in flipped:
            print(f"  {i:<6}{t[:40]:<42}{'/'.join(p1):<14}{'/'.join(p2):<14}{m:.3f}")
    if drifted:
        print(f"\nfollow moved > {args.follow_tol} — {len(drifted)} "
              f"(selection near-ties, verdict held):")
        for i, t, f1, f2 in drifted:
            print(f"  {i:<6}{t[:40]:<42}{f1:.3f} -> {f2:.3f}")
    if not flipped and not drifted:
        print("all compared items reproduced exactly")
    return 1 if flipped else 0


if __name__ == "__main__":
    sys.exit(main())
