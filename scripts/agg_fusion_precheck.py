#!/usr/bin/env python
"""Aggregate the per-shard JSON tallies from eval_fusion_precheck.py into one
full-set table (Overall + per-task), summing correct/total across shards."""
import glob
import json
import sys

paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                         else "/tmp/fusion_pc_shard*.json"))
if not paths:
    sys.exit("no shard JSONs found")

agg = {}   # config -> {"c","n","task":{t:[c,n]}}
for p in paths:
    with open(p) as f:
        d = json.load(f)
    for name, s in d.items():
        a = agg.setdefault(name, {"c": 0, "n": 0, "task": {}})
        a["c"] += s["c"]
        a["n"] += s["n"]
        for t, (tc, tn) in s["task"].items():
            at = a["task"].setdefault(t, [0, 0])
            at[0] += tc
            at[1] += tn

print(f"shards={len(paths)}  ({', '.join(paths)})\n")
rows = []
for name, a in agg.items():
    acc = 100.0 * a["c"] / max(1, a["n"])
    rows.append((acc, name, a))
for acc, name, a in sorted(rows, reverse=True):
    print(f"{name:18s} Overall {acc:6.2f}%   (n={a['n']})")
print("\n--- per-task (rows = configs) ---")
all_tasks = sorted({t for _, _, a in rows for t in a["task"]})
hdr = "config".ljust(18) + "".join(t[:14].rjust(16) for t in all_tasks)
print(hdr)
for acc, name, a in sorted(rows, reverse=True):
    line = name.ljust(18)
    for t in all_tasks:
        if t in a["task"]:
            tc, tn = a["task"][t]
            line += f"{100.0*tc/max(1,tn):6.1f}".rjust(16)
        else:
            line += "—".rjust(16)
    print(line)
