"""Direction C (sign-routed) analysis vs M1 and the hard task_adaptive switch.

signrouted = ONE LoRA, ONE pass: object→confidence(suppress saccade),
spatial/temporal→inverse(keep saccade), else→none. Tests whether routing each
task GROUP to its empirically-better gaze lean beats (a) M1 raw and (b) the
2-way hard switch (task_adaptive=63.20) that this is meant to improve on.
"""
from __future__ import annotations
import json, sys
from math import comb

D = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/dumps"
SIGN = f"{D}/signrouted.jsonl"
NONE = f"{D}/selonly_m1_none.jsonl"   # M1-LoRA raw (same harness control)
M1   = f"{D}/m1.jsonl"
OBJECT_TASKS  = {"present_object_identification_easy", "present_object_identification_hard",
                 "present_object_attribute_recognition", "past_non_fixated_object_identification"}
DYNAMIC_TASKS = {"spatial", "temporal"}


def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line); d[r["key"]] = r
    return d


def mcnemar_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def acc(keys, d): return 100.0 * sum(d[k]["ok"] for k in keys) / max(1, len(keys))


def cmp(keys, a, ref, label):
    b = c = 0   # b = a wins, c = ref wins
    for k in keys:
        if a[k]["ok"] and not ref[k]["ok"]: b += 1
        elif ref[k]["ok"] and not a[k]["ok"]: c += 1
    print(f"  {label:22s} sign={acc(keys,a):.2f}  ref={acc(keys,ref):.2f}  "
          f"Δ={acc(keys,a)-acc(keys,ref):+.2f}  net={b-c:+d} (sign_w={b} ref_w={c}) "
          f"p={mcnemar_p(b,c):.4f}  (n={len(keys)})")


S = load(SIGN)
refs = {}
for name, path in (("none", NONE), ("m1", M1)):
    try: refs[name] = load(path)
    except FileNotFoundError: print(f"[warn] {path} missing")

for rn, R in refs.items():
    keys = [k for k in S if k in R]
    print(f"\n=== signrouted vs {rn}  (paired {len(keys)}) ===")
    cmp(keys, S, R, "overall")
    cmp([k for k in keys if S[k]["task"] in OBJECT_TASKS],  S, R, "object (→conf)")
    cmp([k for k in keys if S[k]["task"] in DYNAMIC_TASKS], S, R, "spatial/temporal(→inv)")
    cmp([k for k in keys if S[k]["task"] not in OBJECT_TASKS | DYNAMIC_TASKS], S, R, "other (→none)")

print("\n=== signrouted per-task ===")
ref = refs.get("none") or refs.get("m1")
keys = [k for k in S if k in ref] if ref else list(S)
for t in sorted({S[k]["task"] for k in keys}):
    tk = [k for k in keys if S[k]["task"] == t]
    grp = "obj→conf" if t in OBJECT_TASKS else ("dyn→inv" if t in DYNAMIC_TASKS else "→none")
    d = f"  Δ={acc(tk,S)-acc(tk,ref):+6.2f}" if ref else ""
    print(f"    {t:42s} n={len(tk):4d}  sign={acc(tk,S):6.2f}{d}  [{grp}]")
