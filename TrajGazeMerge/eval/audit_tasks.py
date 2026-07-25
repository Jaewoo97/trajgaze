"""Data audit — gate for the gaze-exclusive levers ①(future/intent) ②(referential).

No GPU, no model. Iterates the egtea test set and reports, per task:
  - count, mean gaze-validity (fraction of valid gaze frames),
  - keyword hits for FUTURE/INTENT (①) and REFERENTIAL/DEICTIC (②).
Writes a candidate `gaze-necessary` subset (item keys) for later targeted eval.

Purpose: decide whether ①/② have enough items to be measurable BEFORE building
anything. If future/intent items are too few, ④(ROI) is the readier bet.

Usage:
    python -m TrajGazeMerge.eval.audit_tasks [--split test] [--out /tmp/audit.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset

FUTURE = re.compile(r"\b(will|going to|about to|next|then|after (this|that)|"
                    r"intend|plan to|predict|future|upcoming|likely to|"
                    r"what.*do next|going to do)\b", re.I)
DEICTIC = re.compile(r"\b(this|that|these|those|currently|right now|"
                     r"looking at|being (held|used)|in (his|her|their) hand)\b", re.I)


def item_key(item) -> str:
    s = "|".join([str(item.get("task", "")), str(item.get("question", "")),
                  "||".join(item.get("options", [])), str(item.get("answer", ""))])
    return hashlib.md5(s.encode()).hexdigest()


def gaze_validity(item) -> float:
    try:
        import torch
        m = torch.as_tensor(item["traj"]["gaze_mask"], dtype=torch.float32)
        return float(m.mean().item()) if m.numel() else 0.0
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--include-hdepic", action="store_true")
    ap.add_argument("--out", default="/tmp/audit_gaze_necessary.json")
    args = ap.parse_args()

    ds = CombinedMergeDataset(split=args.split, n_vlm_frames=8, n_traj_frames=128,
                              include_hdepic=args.include_hdepic)
    n = len(ds)
    print(f"[audit] {args.split}: {n} items", flush=True)

    by_task = defaultdict(lambda: {"n": 0, "future": 0, "deictic": 0,
                                   "gv_sum": 0.0, "gv_n": 0, "multiopt": 0})
    necessary = []   # candidate gaze-necessary item keys
    n_future = n_deictic = 0

    for idx in range(n):
        try:
            it = ds[idx]
            if it is None:
                continue
        except Exception:
            continue
        t = it.get("task", "?")
        q = it.get("question", "") or ""
        rec = by_task[t]
        rec["n"] += 1
        is_fut = bool(FUTURE.search(q))
        is_dei = bool(DEICTIC.search(q))
        rec["future"] += int(is_fut)
        rec["deictic"] += int(is_dei)
        n_future += int(is_fut); n_deictic += int(is_dei)
        gv = gaze_validity(it)
        if gv >= 0:
            rec["gv_sum"] += gv; rec["gv_n"] += 1
        if len(it.get("options", [])) > 2:
            rec["multiopt"] += 1
        # gaze-necessary candidate: deictic question OR gaze-named task, with usable gaze
        gaze_task = ("gaze" in t) or ("non_fixated" in t) or ("future" in t)
        if (is_dei or gaze_task) and gv > 0.3:
            necessary.append({"key": item_key(it), "task": t,
                              "future": is_fut, "deictic": is_dei, "gaze_valid": round(gv, 2)})
        if (idx + 1) % 200 == 0:
            print(f"  ... {idx+1}/{n}", flush=True)

    print(f"\n[audit] {'task':42s} {'n':>4} {'fut':>4} {'deic':>4} {'gaze_val':>8} {'>2opt':>5}", flush=True)
    print("  " + "-" * 74, flush=True)
    tot = defaultdict(int)
    for t in sorted(by_task):
        r = by_task[t]
        gv = r["gv_sum"] / max(1, r["gv_n"])
        print(f"  {t:42s} {r['n']:4d} {r['future']:4d} {r['deictic']:4d} {gv:8.2f} {r['multiopt']:5d}", flush=True)
        for k in ("n", "future", "deictic"):
            tot[k] += r[k]
    print("  " + "-" * 74, flush=True)
    print(f"  {'TOTAL':42s} {tot['n']:4d} {tot['future']:4d} {tot['deictic']:4d}", flush=True)
    print(f"\n[audit] ① future/intent (kw): {n_future}   ② referential/deictic (kw): {n_deictic}", flush=True)
    print(f"[audit] gaze-necessary candidates: {len(necessary)}  → {args.out}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "n_total": tot["n"],
                   "n_future_kw": n_future, "n_deictic_kw": n_deictic,
                   "necessary": necessary}, f, indent=2)


if __name__ == "__main__":
    main()
