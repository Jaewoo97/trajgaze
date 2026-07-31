#!/usr/bin/env python
"""Per-task table for the 25% budget runs, in kd_handoff_v3's reporting format.

The trainers print per-task accuracy as a percentage only. v3 §5.4/§5.7 report item
counts as well, and require that they sum exactly to the total -- that sum is the check
that catches a silently mis-filtered split, so it is reproduced here rather than trusted.

Task sizes are the SG egtea test split's, from v3 §5.4 (64+68+2+37+96+101+64+94 = 526).
They are not taken on faith: every percentage in every log must land on an integer item
count within 0.01, and those counts must sum to the run's own Overall count, or the row
is flagged. On the first eval log this reconstructs 49+43+1+20+88+76+47+56 = 380 against
an Overall of 72.24% x 526 = 380.

Usage:  python scripts/collect_b25_pertask.py [--out FILE]
"""
import argparse
import os
import re
import statistics
import sys

REPO = os.environ.get("REPO", "/NHNHOME/VILAB/vilab_yj/trajgaze")

# (key in the logs, short label used by the paper tables, n)
TASKS = [
    ("past_gaze_sequence_matching",             "GSM",  64),
    ("past_non_fixated_object_identification",  "NFI",  68),
    ("past_object_transition_prediction",       "OTP",   2),
    ("past_scene_recall",                       "SR",   37),
    ("present_object_attribute_recognition",    "OAR",  96),
    ("present_object_identification_easy",      "OI-E", 101),
    ("present_object_identification_hard",      "OI-H", 64),
    ("present_future_action_prediction",        "FAP",  94),
]
N_TOTAL = sum(n for _, _, n in TASKS)

# v2 §8, 3-run mean over re-scores of the identical 10% M1 SG-only teacher. Kept as the
# reference column: it is the row the 25% teacher has to beat, and it is a MEAN, so only
# the mean column below is a like-for-like comparison to it.
REF_10PCT = {"GSM": 71.36, "NFI": 63.24, "OTP": None, "SR": 57.66, "OAR": 93.40,
             "OI-E": 73.27, "OI-H": 74.48, "FAP": 56.38}
REF_10PCT_OVERALL = 71.29


def parse_eval_only(path):
    """One --eval-ckpt run: '[eval-only] Overall: X%  (n=N)' + '[eval-only]  task: X%'."""
    txt = open(path, errors="replace").read()
    m = re.search(r"\[eval-only\] Overall: ([\d.]+)%\s+\(n=(\d+)\)", txt)
    if not m:
        return None
    per = {k: float(v) for k, v in
           re.findall(r"\[eval-only\]\s+(\w+): ([\d.]+)%", txt)}
    return {"overall": float(m.group(1)), "n": int(m.group(2)), "per_task": per}


def parse_training_epochs(path):
    """The in-process epoch-end evals of a training run, in order."""
    if not os.path.exists(path):
        return []
    txt = open(path, errors="replace").read()
    out = []
    # Each block: "  Overall: X%  (n=N)" followed by indented "    task: X%" lines.
    for m in re.finditer(r"^  Overall: ([\d.]+)%\s+\(n=(\d+)\)$", txt, re.M):
        tail = txt[m.end():m.end() + 2000]
        per = {}
        for line in tail.splitlines()[1:]:
            t = re.match(r"^    (\w+): ([\d.]+)%$", line)
            if not t:
                break
            per[t.group(1)] = float(t.group(2))
        out.append({"overall": float(m.group(1)), "n": int(m.group(2)), "per_task": per})
    return out


def items(pct, n):
    """Item count behind a percentage, with the integrality check.

    The tolerance is derived from the printing precision, not picked: the trainers print
    '%.2f', so a true count k can come back as anything within 0.005 percentage points of
    k/n, i.e. 0.005/100*n items. For n=526 that is 0.026 -- 381/526 prints as 72.43%,
    which reconstructs to 380.98. A tighter constant flags every such row as broken.
    """
    raw = pct / 100.0 * n
    k = round(raw)
    return k, abs(raw - k) <= 0.005 / 100.0 * n + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "vitkd25_teacher_b25_2ep_pertask.txt"))
    args = ap.parse_args()

    runs = []          # (label, result)
    for r in parse_training_epochs(os.path.join(REPO, "vitkd25_teacher_b25_2ep.log")):
        runs.append((f"train ep{len(runs) + 1}", r))
    rescores = []
    for i in range(1, 11):
        p = os.path.join(REPO, "vitkd25_teacher_b25_2ep_eval%02d.log" % i)
        if not os.path.exists(p):
            continue
        r = parse_eval_only(p)
        if r:
            rescores.append((f"re-score {i}", r))
    runs += rescores

    if not runs:
        print("no results found", file=sys.stderr)
        return 1

    L = []
    L.append("25% teacher (M1 SG-only, content 15% ∪ traj 10%), best-of-2 epochs")
    L.append("per-task accuracy, SG egtea test, n=%d" % N_TOTAL)
    L.append("")

    head = f"{'task':6s} {'n':>4s}" + "".join(f" {lab:>14s}" for lab, _ in runs)
    if rescores:
        head += f" {'re-score mean':>14s}"
    head += f" {'10% ref':>9s}"
    L.append(head)
    L.append("-" * len(head))

    warnings = []
    sums = {lab: 0 for lab, _ in runs}
    for key, short, n in TASKS:
        row = f"{short:6s} {n:>4d}"
        vals = []
        for lab, r in runs:
            pct = r["per_task"].get(key)
            if pct is None:
                row += f" {'--':>14s}"
                continue
            k, exact = items(pct, n)
            sums[lab] += k
            vals.append(pct)
            row += f" {pct:8.2f}%({k:>3d})"
            if not exact:
                warnings.append(f"{lab} {short}: {pct}% of {n} is not an integer item count")
        if rescores:
            rv = [r["per_task"][key] for _, r in rescores if key in r["per_task"]]
            row += f" {statistics.fmean(rv):13.2f}%" if rv else f" {'--':>14s}"
        ref = REF_10PCT.get(short)
        row += f" {ref:8.2f}%" if ref is not None else f" {'--':>9s}"
        L.append(row)

    L.append("-" * len(head))
    row = f"{'Avg':6s} {N_TOTAL:>4d}"
    for lab, r in runs:
        k, exact = items(r["overall"], r["n"])
        row += f" {r['overall']:8.2f}%({k:>3d})"
        if sums[lab] != k:
            warnings.append(f"{lab}: per-task items sum to {sums[lab]} but Overall is {k}")
        if not exact:
            warnings.append(f"{lab}: Overall {r['overall']}% of {r['n']} is not an integer")
    if rescores:
        ov = [r["overall"] for _, r in rescores]
        row += f" {statistics.fmean(ov):13.2f}%"
    row += f" {REF_10PCT_OVERALL:8.2f}%"
    L.append(row)

    if rescores:
        ov = [r["overall"] for _, r in rescores]
        mx, mn = max(ov), min(ov)
        L += ["",
              f"re-scores of the SAME checkpoint: n={len(ov)}, "
              f"max {mx:.2f}% ({round(mx / 100 * N_TOTAL)}), "
              f"mean {statistics.fmean(ov):.2f}% ({round(statistics.fmean(ov) / 100 * N_TOTAL)}), "
              f"min {mn:.2f}% ({round(mn / 100 * N_TOTAL)}), "
              f"spread {mx - mn:.2f} pt",
              "",
              "The 10% ref column is v2 §8's 3-run MEAN over re-scores of identical weights",
              "(Overall 71.29% = 375 items, a 4-run mean). Compare it to the re-score mean",
              "column, not to max: max is best-of-N and v2 §8 calls that upward-biased.",
              "OTP has no 10% reference and its n=2 makes one question worth 50 points."]

    L += ["", "item-count check: every per-task percentage lands on an integer and the",
          "per-task counts sum to each run's own Overall count."] if not warnings else \
         ["", "!! CHECK FAILED:"] + [f"  - {w}" for w in warnings]

    text = "\n".join(L)
    open(args.out, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
