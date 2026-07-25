"""Per-task comparison of complement runs from their TRAINING logs (no retraining).

Cheap Tier-1 read: reads each run's train_log_rank0.jsonl, takes the BEST epoch
(max eval_acc), and tabulates per-task accuracy with:
  - Δ vs a reference run (default: first --run),
  - per-task n RECOVERED from the accuracy fraction k/n via continued fractions
    (Fraction.limit_denominator) — flags tiny-n tasks whose deltas are noise,
  - the REFERENCE run's OWN epoch-to-epoch jitter (min..max across its epochs) as
    the per-task noise floor. A Δ is flagged '!' only if |Δ| exceeds that jitter
    AND the task is not tiny-n.

It shows WHERE a method moves accuracy. For SIGNIFICANCE (is the move real), use
eval_dump.py + mcnemar.py — recovered-n and jitter are heuristics, not a test.

Usage:
    python -m TrajGazeMerge.eval.pertask_compare --ref M1 \
        --run "M1=/.../visionzip_complement_learned_overlay" \
        --run "qg_cosine=/.../visionzip_query_gaze_overlay" \
        --run "qg_random=/.../visionzip_query_gaze_random_overlay"
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from fractions import Fraction


def read_log(run_dir: str) -> list:
    """All {epoch, eval_acc, per_task} records from a run dir's train_log_rank0.jsonl."""
    path = os.path.join(run_dir, "train_log_rank0.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "eval_acc" in r and "per_task" in r:
                out.append(r)
    return out


def best_epoch(evals: list):
    return max(evals, key=lambda r: r["eval_acc"]) if evals else None


def jitter_band(evals: list) -> dict:
    """Per-task (min, max) across all epochs of a run → its own noise floor."""
    band = {}
    for r in evals:
        for t, v in r["per_task"].items():
            lo, hi = band.get(t, (v, v))
            band[t] = (min(lo, v), max(hi, v))
    return band


def recover_n(values, cap: int = 1011):
    """Recover task n as the MAX denominator of k/n across ALL observed accuracy
    values (every run × every epoch). A single 'clean' value (0/50/100%) gives a
    too-small denominator; aggregating across epochs/runs reveals the true n
    (e.g. 46.8%→22/47 and 52.1%→49/94 ⇒ n=94). Still a heuristic — eval_dump.py
    reports exact n."""
    best = None
    for pct in values:
        if pct is None:
            continue
        den = Fraction(pct / 100.0).limit_denominator(cap).denominator
        best = den if best is None else max(best, den)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=dir (repeatable)")
    ap.add_argument("--ref", default=None, help="reference label (default: first --run)")
    ap.add_argument("--small-n", type=int, default=20, help="tasks with recovered n < this are flagged small-n")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, run_dir = spec.split("=", 1)
        runs.append((label, run_dir, read_log(run_dir)))
    ref_label = args.ref or runs[0][0]
    ref = next((r for r in runs if r[0] == ref_label), runs[0])
    ref_best, ref_band = best_epoch(ref[2]), jitter_band(ref[2])

    # per-task n = max k/n denominator across EVERY run × EVERY epoch (best signal)
    all_vals = defaultdict(list)
    for _, _, ev in runs:
        for r in ev:
            for t, v in r["per_task"].items():
                all_vals[t].append(v)
    task_n = {t: recover_n(vs) for t, vs in all_vals.items()}

    print(f"# reference = {ref_label}   (jitter = ref's own epoch min..max spread per task)")
    for label, _, ev in runs:
        be = best_epoch(ev)
        print(f"#   {label:14s} " + (f"ep{be['epoch']}  {be['eval_acc']:.2f}%  "
              f"({len(ev)} epochs logged)" if be else "[no eval log yet]"))
    if ref_best is None:
        print("# no reference eval found — run has not produced an epoch eval yet.")
        return

    labels = [r[0] for r in runs]
    print()
    print(f"{'task':42s} {'n':>4} {'jitr':>5}  " + " ".join(f"{l:>10}" for l in labels)
          + "   Δ vs " + ref_label)
    print("-" * (42 + 6 + 7 + 11 * len(labels) + 12))
    for t in sorted(ref_best["per_task"].keys()):
        refv = ref_best["per_task"].get(t)
        n = task_n.get(t)
        lo, hi = ref_band.get(t, (refv, refv))
        jit = hi - lo
        small = n is not None and n < args.small_n
        cells, deltas = [], []
        for label, _, ev in runs:
            be = best_epoch(ev)
            v = be["per_task"].get(t) if be else None
            cells.append(f"{v:10.1f}" if v is not None else f"{'—':>10}")
            if label != ref_label and v is not None and refv is not None:
                flag = "!" if (abs(v - refv) > jit and not small) else " "
                deltas.append(f"{label}:{v - refv:+.1f}{flag}")
        tag = f"{t}" + (" «small-n»" if small else "")
        print(f"{tag:42s} {str(n):>4} {jit:5.1f}  " + " ".join(cells) + "   " + "  ".join(deltas))

    print()
    for label, _, ev in runs:
        be = best_epoch(ev)
        if be:
            print(f"  {label:14s} overall best = {be['eval_acc']:.2f}%  (epoch {be['epoch']})")
    print("\n  '!' = |Δ| exceeds the reference's own epoch jitter on a non-tiny task "
          "(candidate real effect; confirm with mcnemar.py).")


if __name__ == "__main__":
    main()
