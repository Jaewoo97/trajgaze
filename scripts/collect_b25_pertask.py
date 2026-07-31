#!/usr/bin/env python
"""Per-task table for the 25% budget runs, in kd_handoff_v3's reporting format.

The trainers print per-task accuracy as a percentage only. v3 §5.4/§5.7 report item
counts as well, and require that they sum exactly to the total -- that sum is the check
that catches a silently mis-filtered split, so it is reproduced here rather than trusted.

Task sizes are the egtea test split's, per source, from v3 §5.4 (SG) and §5.7 (EG).
They are not taken on faith: every percentage in every log must land on an integer item
count, and those counts must sum to the run's own Overall count, or the row is flagged.
On the first SG eval log this reconstructs 49+43+1+20+88+76+47+56 = 380 against an
Overall of 72.24% x 526 = 380.

Usage:  python scripts/collect_b25_pertask.py [--source {sg,eg}] [--prefix P] [--out FILE]
"""
import argparse
import os
import re
import statistics
import sys

REPO = os.environ.get("REPO", "/NHNHOME/VILAB/vilab_yj/trajgaze")

# Per source: the log key, the short label the paper tables use, and n.
#
# The reference column is the 10%-budget M1 teacher on the same split -- the row the 25%
# teacher has to beat. Both references are MEANS over re-scores of identical weights, so
# only the re-score-mean column below compares to them like for like; max is best-of-N,
# which v2 §8 calls an upward-biased estimator.
SOURCES = {
    "sg": {
        # v3 §5.4. Reference = v2 §8's 3-run mean, Overall its 4-run mean (375 items).
        "tasks": [
            ("past_gaze_sequence_matching",            "GSM",  64, 71.36),
            ("past_non_fixated_object_identification", "NFI",  68, 63.24),
            ("past_object_transition_prediction",      "OTP",   2, None),
            ("past_scene_recall",                      "SR",   37, 57.66),
            ("present_object_attribute_recognition",   "OAR",  96, 93.40),
            ("present_object_identification_easy",     "OI-E", 101, 73.27),
            ("present_object_identification_hard",     "OI-H", 64, 74.48),
            ("present_future_action_prediction",       "FAP",  94, 56.38),
        ],
        "ref_overall": 71.29,
        "ref_label": "10% M1 SG-only teacher, 71.29% (375 items), 4-run mean, v2 §8",
        "prefix": "vitkd25_teacher_b25_2ep",
        "splits": {"train": 5799, "test": 526},
    },
    "eg": {
        # v3 §5.7's "M1 EG teacher (r2)" column: 139 + 65 + 58 = 262 items, 54.02%.
        # A single run there, not a mean -- v3 §9 leaves ">=3 evals" open for every EG row.
        "tasks": [
            ("causal",   "Caus.", 162, 85.80),
            ("spatial",  "Spat.", 163, 39.88),
            ("temporal", "Temp.", 160, 36.25),
        ],
        "ref_overall": 54.02,
        "ref_label": "10% M1 EG-only teacher, 54.02% (262 items), SINGLE run, v3 §5.7",
        "prefix": "vitkd25eg_teacher_b25_2ep",
        "splits": {"train": 1265, "test": 485},
    },
}


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
    ap.add_argument("--source", choices=sorted(SOURCES), default="sg",
                    help="which benchmark's task list and 10%% reference to use")
    ap.add_argument("--prefix", default=None,
                    help="log basename stem; defaults to the source's own teacher run. "
                         "Point it at a p2 log to tabulate the student instead.")
    ap.add_argument("--label", default=None,
                    help="title line. Defaults to the teacher wording, which is wrong "
                         "for any other --prefix — set it when tabulating a student.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    tasks = cfg["tasks"]
    n_total = sum(t[2] for t in tasks)
    prefix = args.prefix or cfg["prefix"]
    out_path = args.out or os.path.join(REPO, f"{prefix}_pertask.txt")

    runs = []          # (label, result)
    for r in parse_training_epochs(os.path.join(REPO, f"{prefix}.log")):
        runs.append((f"train ep{len(runs) + 1}", r))
    rescores = []
    for i in range(1, 11):
        p = os.path.join(REPO, "%s_eval%02d.log" % (prefix, i))
        if not os.path.exists(p):
            continue
        r = parse_eval_only(p)
        if r:
            rescores.append((f"re-score {i}", r))
    runs += rescores

    if not runs:
        print(f"no results found for prefix {prefix!r} under {REPO}", file=sys.stderr)
        return 1

    L = []
    L.append(args.label or
             "25%% teacher (M1 %s-only, content 15%% ∪ traj 10%%), best-of-2 epochs"
             % args.source.upper())
    L.append("per-task accuracy, %s egtea test, n=%d" % (args.source.upper(), n_total))
    L.append("")

    head = f"{'task':6s} {'n':>4s}" + "".join(f" {lab:>14s}" for lab, _ in runs)
    if rescores:
        head += f" {'re-score mean':>14s}"
    head += f" {'10% ref':>9s}"
    L.append(head)
    L.append("-" * len(head))

    warnings = []
    sums = {lab: 0 for lab, _ in runs}
    for key, short, n, ref in tasks:
        row = f"{short:6s} {n:>4d}"
        for lab, r in runs:
            pct = r["per_task"].get(key)
            if pct is None:
                row += f" {'--':>14s}"
                continue
            k, exact = items(pct, n)
            sums[lab] += k
            row += f" {pct:8.2f}%({k:>3d})"
            if not exact:
                warnings.append(f"{lab} {short}: {pct}% of {n} is not an integer item count")
        if rescores:
            rv = [r["per_task"][key] for _, r in rescores if key in r["per_task"]]
            row += f" {statistics.fmean(rv):13.2f}%" if rv else f" {'--':>14s}"
        row += f" {ref:8.2f}%" if ref is not None else f" {'--':>9s}"
        L.append(row)

    L.append("-" * len(head))
    row = f"{'Avg':6s} {n_total:>4d}"
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
    row += f" {cfg['ref_overall']:8.2f}%"
    L.append(row)

    if rescores:
        ov = [r["overall"] for _, r in rescores]
        mx, mn = max(ov), min(ov)
        L += ["",
              f"re-scores of the SAME checkpoint: n={len(ov)}, "
              f"max {mx:.2f}% ({round(mx / 100 * n_total)}), "
              f"mean {statistics.fmean(ov):.2f}% ({round(statistics.fmean(ov) / 100 * n_total)}), "
              f"min {mn:.2f}% ({round(mn / 100 * n_total)}), "
              f"spread {mx - mn:.2f} pt",
              "",
              f"10% ref column: {cfg['ref_label']}.",
              "Compare it to the re-score mean column, not to max: max is best-of-N and",
              "v2 §8 calls that an upward-biased estimator."]
        if any(t[3] is None for t in tasks):
            L.append("Tasks shown as -- have no 10% reference on record.")
        small = [t[1] for t in tasks if t[2] <= 5]
        if small:
            L.append("n<=5 columns (%s): one question moves them by >=20 points."
                     % ", ".join(small))

    L += ["", "item-count check: every per-task percentage lands on an integer and the",
          "per-task counts sum to each run's own Overall count."] if not warnings else \
         ["", "!! CHECK FAILED:"] + [f"  - {w}" for w in warnings]

    text = "\n".join(L)
    open(out_path, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
