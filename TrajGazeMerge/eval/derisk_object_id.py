"""De-risk for lever ② — are M1's referential object-ID errors gaze-ADDRESSABLE?

Joins the M1 per-item dump (eval_dump.py) with gaze features re-derived from the
dataset (NO GPU, joined by the same md5 item key), and asks, on the referential
object-ID tasks:

  1. WHERE does M1 fail?            per-task acc / error counts.
  2. Are errors object CONFUSIONS?  (object-ID options ARE objects → every error is;
                                     report pred/gt lexical similarity = confusable?).
  3. Are errors gaze-ADDRESSABLE?   among ERRORS, what fraction have a CLEAR gaze
                                     fixation (dominant region holds > conc-thresh of
                                     valid frames)? Clear fixation on an error = the
                                     referent was available in gaze but the model
                                     didn't use it → ③ grounding has headroom.
  4. Granularity hint:              of the clear-fixation errors, how spread are the
                                     dominant regions (separable → 3×3 ok).
  5. Qualitative:                   write N error samples (question/options[pred|gt]/
                                     the ③ sentence we WOULD inject) for eyeballing.

LIMIT: no object boxes, so "gaze points at the correct object" isn't directly
verifiable — this is a PRELIMINARY de-risk; the definitive test is the ③ run.

Usage:
    python -m TrajGazeMerge.eval.derisk_object_id \
        --dump .../checkpoints/dumps/m1.jsonl --out /tmp/derisk_samples.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/workspace/trajgaze_st")

import torch

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.gaze_grounding import (
    GazeGroundingConfig, region_word, dominant_gaze_point, grounding_sentence, _as2d,
)

OBJID_TASKS = ["present_object_identification_hard",
               "present_object_identification_easy",
               "present_object_attribute_recognition"]
_PREFIX = re.compile(r"^[A-E][\.\)]\s*")


def item_key(item) -> str:   # MUST match eval_dump.item_key
    s = "|".join([str(item.get("task", "")), str(item.get("question", "")),
                  "||".join(item.get("options", [])), str(item.get("answer", ""))])
    return hashlib.md5(s.encode()).hexdigest()


def gaze_concentration(traj, cfg):
    """(dominant_region, concentration, n_valid). concentration = fraction of valid
    gaze frames whose region == the dominant region (1.0 = perfectly fixed)."""
    g = _as2d(traj.get("gaze_pos")) if isinstance(traj, dict) else None
    if g is None or g.shape[0] == 0:
        return None, 0.0, 0
    L = g.shape[0]
    m = torch.as_tensor(traj.get("gaze_mask", torch.ones(L)), dtype=torch.float32).view(-1)
    if m.shape[0] != L:
        m = torch.ones(L)
    gv = g[m > 0]
    if gv.shape[0] == 0:
        return None, 0.0, 0
    keys = [region_word(float(p[0]), float(p[1]), cfg.grid) for p in gv]
    region, cnt = Counter(keys).most_common(1)[0]
    return region, cnt / len(keys), gv.shape[0]


def jaccard(a: str, b: str) -> float:
    sa = set(_PREFIX.sub("", a).lower().split())
    sb = set(_PREFIX.sub("", b).lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tasks", nargs="*", default=OBJID_TASKS)
    ap.add_argument("--conc-thresh", type=float, default=0.5,
                    help="dominant region must hold > this fraction of valid gaze to count as 'clear fixation'")
    ap.add_argument("--n-samples", type=int, default=25)
    ap.add_argument("--out", default="/tmp/derisk_samples.txt")
    ap.add_argument("--include-hdepic", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = GazeGroundingConfig()

    dump = {}
    with open(args.dump) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                dump[r["key"]] = r
    print(f"[derisk] dump rows: {len(dump)}", flush=True)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=8, n_traj_frames=128,
                              include_hdepic=args.include_hdepic)
    feats = {}   # key -> gaze features
    for idx in range(len(ds)):
        try:
            it = ds[idx]
            if it is None:
                continue
            k = item_key(it)
            if k not in dump:
                continue
            region, conc, nval = gaze_concentration(it.get("traj"), cfg)
            pt = dominant_gaze_point(it.get("traj"), cfg)
            feats[k] = {"region": region, "conc": conc, "nval": nval,
                        "sentence": grounding_sentence(pt, it.get("traj"), cfg)}
        except Exception:
            continue
    print(f"[derisk] joined gaze features: {len(feats)}", flush=True)

    tasks = set(args.tasks)
    rows = [r for r in dump.values() if r["task"] in tasks and r["key"] in feats]
    print(f"\n[derisk] referential object-ID items: {len(rows)}  (tasks={sorted(tasks)})", flush=True)

    # 1. per-task acc
    pt_stat = defaultdict(lambda: [0, 0])
    for r in rows:
        s = pt_stat[r["task"]]; s[0] += 1; s[1] += r["ok"]
    print("  per-task  (n | acc):", flush=True)
    for t in sorted(pt_stat):
        n, c = pt_stat[t]; print(f"    {t:42s} n={n:4d}  {100.0*c/max(1,n):6.2f}%", flush=True)

    errors = [r for r in rows if not r["ok"]]
    corrects = [r for r in rows if r["ok"]]
    if not errors:
        print("  no errors — nothing to de-risk."); return

    # 2. object confusion: pred/gt lexical similarity
    sim = sum(jaccard(r["pred_text"], r["gt_text"]) for r in errors) / len(errors)

    # 3. gaze-addressable: clear fixation on errors vs corrects
    def clear_frac(group):
        if not group:
            return 0.0, 0.0
        concs = [feats[r["key"]]["conc"] for r in group]
        clear = sum(c > args.conc_thresh for c in concs)
        return clear / len(group), sum(concs) / len(concs)
    err_clear, err_meanconc = clear_frac(errors)
    cor_clear, cor_meanconc = clear_frac(corrects)

    # 4. granularity: spread of dominant regions among clear-fixation errors
    clear_err_regions = Counter(feats[r["key"]]["region"] for r in errors
                                if feats[r["key"]]["conc"] > args.conc_thresh)

    print(f"\n[derisk] errors={len(errors)}  corrects={len(corrects)}", flush=True)
    print(f"  (2) pred↔gt lexical sim (errors): {sim:.2f}   "
          f"(high → confusing SIMILAR objects = referential)", flush=True)
    print(f"  (3) CLEAR-fixation fraction  errors={err_clear:.2f} (mean conc {err_meanconc:.2f})  "
          f"vs corrects={cor_clear:.2f} ({cor_meanconc:.2f})", flush=True)
    print(f"      → {err_clear*len(errors):.0f}/{len(errors)} errors have a clear gaze fixation "
          f"the model didn't exploit = ③-ADDRESSABLE headroom", flush=True)
    print(f"  (4) clear-error dominant regions (spread → 3×3 grounding separates them):", flush=True)
    for reg, c in clear_err_regions.most_common():
        print(f"        {reg:16s} {c}", flush=True)

    # verdict heuristic
    verdict = ("STRONG: errors keep a clear gaze fixation → explicit ③ grounding has room"
               if err_clear >= 0.5 else
               "WEAK: errors have diffuse gaze → ③ grounding may not help; reconsider")
    print(f"\n[derisk] VERDICT: {verdict}", flush=True)

    # 5. qualitative samples
    with open(args.out, "w") as fo:
        for r in errors[:args.n_samples]:
            ft = feats[r["key"]]
            fo.write(f"TASK {r['task']}  conc={ft['conc']:.2f}  region={ft['region']}\n")
            fo.write(f"Q: {r['question']}\n")
            for i, o in enumerate(r["options"]):
                mark = "✗pred" if i == r["pred"] else ("✓gt" if i == r["gt"] else "")
                fo.write(f"   {o}  {mark}\n")
            fo.write(f"③ would inject: {ft['sentence']}\n\n")
    print(f"[derisk] {min(len(errors), args.n_samples)} error samples → {args.out}", flush=True)


if __name__ == "__main__":
    main()
