"""Selection-only single-LoRA test analysis.

Both dumps come from the SAME M1 LoRA, ONE forward pass each:
  - none      : raw M1 selection (control, must reproduce M1)
  - adaptive  : object-grounding tasks reweighted by fixation confidence,
                dynamic tasks identical to M1 (per-task selection switch)

Since dynamic tasks use the same selection in both arms, any McNemar
discordance is the PURE effect of selection-only confidence on object tasks
-- with no second model / no extra forward pass (efficiency thesis intact).
"""
from __future__ import annotations
import json, sys
from math import comb

D = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/dumps"
NONE     = f"{D}/selonly_m1_none.jsonl"
ADAPT    = f"{D}/selonly_m1_adaptive.jsonl"
M1REF    = f"{D}/m1.jsonl"
OBJECT_TASKS = {
    "present_object_identification_easy",
    "present_object_identification_hard",
    "present_object_attribute_recognition",
    "past_non_fixated_object_identification",
}


def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            d[r["key"]] = r
    return d


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def acc(keys, dd):
    return 100.0 * sum(dd[k]["ok"] for k in keys) / max(1, len(keys))


def mcnemar(keys, a, bdump):
    """Returns (b, c, p): b = a-wins (a right, b wrong), c = bdump-wins.
    net = b - c is a's advantage."""
    b = c = 0
    for k in keys:
        if a[k]["ok"] and not bdump[k]["ok"]: b += 1
        elif bdump[k]["ok"] and not a[k]["ok"]: c += 1
    return b, c, mcnemar_p(b, c)


N, A = load(NONE), load(ADAPT)
keys = [k for k in N if k in A]
print(f"paired (none∩adaptive): {len(keys)}")

# --- validity: does the harness's 'none' reproduce M1 ref? ---
try:
    M = load(M1REF)
    kk = [k for k in keys if k in M]
    b, c, p = mcnemar(kk, N, M)
    print(f"\n[validity] none vs m1.jsonl  none={acc(kk,N):.2f}  m1={acc(kk,M):.2f}  "
          f"net={c-b:+d} (b={b} c={c}) p={p:.4f}  (want ~tie / identical)")
    dyn = [k for k in kk if M[k]["task"] not in OBJECT_TASKS]
    diff = sum(1 for k in dyn if N[k]["pred"] != M[k]["pred"])
    print(f"           dynamic-task pred mismatches none vs m1: {diff}/{len(dyn)} (want 0)")
except FileNotFoundError:
    print("\n[validity] m1.jsonl not found — skipping")

# --- experiment: adaptive vs none (same LoRA, selection-only) ---
print("\n=== selection-only: adaptive vs none (SAME M1 LoRA, 1 pass) ===")
b, c, p = mcnemar(keys, A, N)
print(f"  overall   adaptive={acc(keys,A):.2f}  none={acc(keys,N):.2f}  "
      f"Δ={acc(keys,A)-acc(keys,N):+.2f}  net={c-b:+d} (b={b} c={c}) p={p:.4f}")

obj = [k for k in keys if N[k]["task"] in OBJECT_TASKS]
b, c, p = mcnemar(obj, A, N)
print(f"  object    adaptive={acc(obj,A):.2f}  none={acc(obj,N):.2f}  "
      f"Δ={acc(obj,A)-acc(obj,N):+.2f}  net={c-b:+d} (b={b} c={c}) p={p:.4f}  (n={len(obj)})")

dyn = [k for k in keys if N[k]["task"] not in OBJECT_TASKS]
diff = sum(1 for k in dyn if A[k]["pred"] != N[k]["pred"])
print(f"  dynamic   pred mismatches adaptive vs none: {diff}/{len(dyn)} (MUST be 0)")

print("\n  per-task (adaptive − none):")
tasks = sorted({N[k]["task"] for k in keys})
for t in tasks:
    tk = [k for k in keys if N[k]["task"] == t]
    mark = " *obj" if t in OBJECT_TASKS else ""
    print(f"    {t:42s} n={len(tk):4d}  none={acc(tk,N):6.2f}  "
          f"adapt={acc(tk,A):6.2f}  Δ={acc(tk,A)-acc(tk,N):+6.2f}{mark}")
