"""HandOracleDeRisk verdict — paired McNemar(gt_hand vs placebo) on target/control.

Greenlight rule (from the Seed):
  greenlight IFF  gt_hand beats placebo on the action/how/why TARGET at McNemar p<0.05
             AND  gt_hand does NOT significantly beat placebo on the 3d/object CONTROL
             AND  leakage audit clean (no injected text leaked verb/order/option).
Otherwise KILL (hand-kinematics text is redundant with RGB, or the gain is generic
information-quantity rather than hand-specific).
"""
from __future__ import annotations
import json, sys
from math import comb
from collections import defaultdict

DUMP = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/dumps/handoracle.jsonl"


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def cmp(rows, a, ref):
    """b = a-wins (a right, ref wrong), c = ref-wins. net=b-c is a's advantage."""
    b = sum(1 for r in rows if r[a] and not r[ref])
    c = sum(1 for r in rows if r[ref] and not r[a])
    accs = 100 * sum(r[a] for r in rows) / max(1, len(rows))
    accr = 100 * sum(r[ref] for r in rows) / max(1, len(rows))
    return accs, accr, b, c, mcnemar_p(b, c)


rows = [json.loads(l) for l in open(DUMP)]
leak_fail = sum(1 for r in rows if not r.get("leak_ok", 1))
print(f"n={len(rows)}  leak_fail={leak_fail} (must be 0)\n")

for grp in ("target", "control"):
    g = [r for r in rows if r["group"] == grp]
    if not g:
        continue
    print(f"===== {grp.upper()} (n={len(g)}) =====")
    for a, ref, lab in [("ok_gt", "ok_pb", "gt_hand vs placebo"),
                        ("ok_gt", "ok_base", "gt_hand vs baseline"),
                        ("ok_pb", "ok_base", "placebo vs baseline")]:
        accs, accr, b, c, p = cmp(g, a, ref)
        print(f"  {lab:22s} {accs:6.2f} vs {accr:6.2f}  Δ={accs-accr:+.2f} "
              f"net={b-c:+d} (win={b} lose={c}) p={p:.4f}")
    print("  per-task (gt vs pb):")
    for t in sorted(set(r["task"] for r in g)):
        tk = [r for r in g if r["task"] == t]
        accs, accr, b, c, p = cmp(tk, "ok_gt", "ok_pb")
        print(f"    {t:44s} n={len(tk):4d}  gt={accs:5.1f} pb={accr:5.1f} Δ={accs-accr:+5.1f} p={p:.3f}")
    print()

# ---- verdict ----
tgt = [r for r in rows if r["group"] == "target"]
ctl = [r for r in rows if r["group"] == "control"]
_, _, tb, tc, tp = cmp(tgt, "ok_gt", "ok_pb") if tgt else (0, 0, 0, 0, 1.0)
_, _, cb, cc, cp = cmp(ctl, "ok_gt", "ok_pb") if ctl else (0, 0, 0, 0, 1.0)
target_win = tgt and tp < 0.05 and (tb - tc) > 0
control_null = (not ctl) or not (cp < 0.05 and (cb - cc) > 0)
greenlight = bool(target_win and control_null and leak_fail == 0)
print("===== VERDICT =====")
print(f"  target gt>pb significant : {bool(target_win)} (p={tp:.4f}, net={tb-tc:+d})")
print(f"  control null (gt≯pb)     : {bool(control_null)} (p={cp:.4f}, net={cb-cc:+d})")
print(f"  leakage clean            : {leak_fail == 0}")
print(f"  >>> {'GREENLIGHT — build hand-fusion mechanism' if greenlight else 'KILL — hand-kinematics text not complementary (or generic info effect)'}")
