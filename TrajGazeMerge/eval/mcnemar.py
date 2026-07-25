"""McNemar paired significance test from two per-item eval dumps (eval_dump.py).

Pairs items by 'key' (same eval item under two models), builds the 2x2 table
    a = both correct,  b = A right / B wrong,  c = A wrong / B right,  d = both wrong
and reports, OVERALL and PER TASK:
    n_paired, accA, accB, Δ=accB-accA, net=c-b (B's gain in #items),
    b, c (discordant counts), and the exact two-sided McNemar p-value
    (binomial — no scipy dependency).

Why: at the <1%p effect sizes in this project, the SIGN of a per-task delta is
meaningless on its own. McNemar conditions on the paired discordant items (b, c),
which is the only honest "is B different from A" test here. Net = c - b is the
number of items B flips to correct minus those it flips to wrong.

Usage:
    python -m TrajGazeMerge.eval.mcnemar \
        --a dumps/m1.jsonl --label-a M1 \
        --b dumps/qg_cosine.jsonl --label-b qg_cosine
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load(path: str) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["key"]] = r
    return d


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p-value for discordant counts b, c.

    Under H0 the c discordant 'B-wins' ~ Binomial(b+c, 0.5); p = 2·P(X ≤ min(b,c))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * cdf)


def contingency(A: dict, B: dict, keys):
    a = b = c = d = 0
    for k in keys:
        oa, ob = A[k]["ok"], B[k]["ok"]
        if oa and ob:
            a += 1
        elif oa and not ob:
            b += 1
        elif not oa and ob:
            c += 1
        else:
            d += 1
    return a, b, c, d


def fmt_row(name, n, accA, accB, b, c, p):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return (f"{name:42s} n={n:4d}  A={accA:5.1f}  B={accB:5.1f}  "
            f"Δ={accB - accA:+5.1f}  net={c - b:+4d}  b={b:3d} c={c:3d}  p={p:6.3f} {star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) & set(B))
    print(f"# McNemar  A={args.label_a} (n={len(A)})  B={args.label_b} (n={len(B)})  "
          f"paired={len(keys)}")
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    if only_a or only_b:
        print(f"# WARNING unpaired (excluded): only-A={only_a}  only-B={only_b}")
    print(f"#   * p<.05   ** p<.01   *** p<.001     (net = c - b = B's net flips to correct)")
    print()

    by_task = defaultdict(list)
    for k in keys:
        by_task[A[k]["task"]].append(k)
    for t in sorted(by_task):
        ks = by_task[t]
        a, b, c, d = contingency(A, B, ks)
        n = len(ks)
        print("  " + fmt_row(t, n, 100.0 * (a + b) / n, 100.0 * (a + c) / n,
                             b, c, exact_mcnemar_p(b, c)))
    a, b, c, d = contingency(A, B, keys)
    n = max(1, len(keys))
    print()
    print("  " + fmt_row("OVERALL", n, 100.0 * (a + b) / n, 100.0 * (a + c) / n,
                         b, c, exact_mcnemar_p(b, c)))


if __name__ == "__main__":
    main()
