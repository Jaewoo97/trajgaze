"""Direction A — task-adaptive confidence: feasibility from existing dumps.

Routes each item to either the confidence-weighted model (object/static grounding)
or the raw M1 model (dynamic/relational), then measures vs M1. Two policies:

  1. SEMANTIC  — object set fixed a-priori by question semantics (test-informed,
                 optimistic upper bound).
  2. CV        — for each test half, learn the winning-task set (net>0) from the
                 OTHER half, apply to this half, pool held-out predictions
                 (removes the choose-then-evaluate circularity).

Both are post-hoc mixtures of two separately-trained LoRAs — a feasibility
estimate, NOT a deployable model. If CV beats M1, train a single task-adaptive LoRA.
"""
from __future__ import annotations
import hashlib, json, sys
from math import comb

M1   = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/dumps/m1.jsonl"
CONF = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/dumps/gazeconf_confidence.jsonl"

# a-priori semantic split: object identity/attribute grounding → confidence
OBJECT_TASKS = {
    "present_object_identification_easy",
    "present_object_identification_hard",
    "present_object_attribute_recognition",
    "past_non_fixated_object_identification",
}


def load(path):
    d = {}
    with open(path) as f:
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


def evaluate(keys, route):  # route: key -> 'conf'|'m1'
    """Return (acc_adaptive, acc_m1, b, c) over keys."""
    corr_a = corr_m = 0
    b = c = 0
    for k in keys:
        m = M1D[k]; cf = CONFD[k]
        ok_a = cf["ok"] if route[k] == "conf" else m["ok"]
        corr_a += ok_a; corr_m += m["ok"]
        if ok_a and not m["ok"]: c += 1
        elif m["ok"] and not ok_a: b += 1
    n = len(keys)
    return 100.0 * corr_a / n, 100.0 * corr_m / n, b, c


M1D, CONFD = load(M1), load(CONF)
keys = [k for k in M1D if k in CONFD]
print(f"paired items: {len(keys)}")

# ---------- 1. SEMANTIC policy ----------
route_sem = {k: ("conf" if M1D[k]["task"] in OBJECT_TASKS else "m1") for k in keys}
a, m, b, c = evaluate(keys, route_sem)
print("\n=== 1. SEMANTIC policy (object→conf, dynamic→M1) — test-informed, optimistic ===")
print(f"  adaptive={a:.2f}%  M1={m:.2f}%  Δ={a-m:+.2f}  net={c-b:+d} (b={b} c={c})  p={mcnemar_p(b,c):.4f}")
n_obj = sum(1 for k in keys if M1D[k]['task'] in OBJECT_TASKS)
print(f"  (routed to conf: {n_obj}/{len(keys)} items in {len(OBJECT_TASKS)} object tasks)")

# ---------- 2. CROSS-VALIDATED data-driven policy ----------
def half(k):  # stable 50/50 split by key hash
    return int(hashlib.md5(k.encode()).hexdigest(), 16) & 1

H = {0: [k for k in keys if half(k) == 0], 1: [k for k in keys if half(k) == 1]}

def task_net(train_keys):
    """net(conf-m1) per task on the training half → winning task set."""
    net = {}
    for k in train_keys:
        t = M1D[k]["task"]
        net.setdefault(t, 0)
        if CONFD[k]["ok"] and not M1D[k]["ok"]: net[t] += 1
        elif M1D[k]["ok"] and not CONFD[k]["ok"]: net[t] -= 1
    return {t for t, v in net.items() if v > 0}

route_cv = {}
win_sets = {}
for h in (0, 1):
    train_h = 1 - h
    win = task_net(H[train_h])               # learn on the OTHER half
    win_sets[h] = win
    for k in H[h]:                            # apply to THIS (held-out) half
        route_cv[k] = "conf" if M1D[k]["task"] in win else "m1"

a, m, b, c = evaluate(keys, route_cv)
print("\n=== 2. CROSS-VALIDATED policy (learn on other half, eval held-out) — honest ===")
print(f"  adaptive={a:.2f}%  M1={m:.2f}%  Δ={a-m:+.2f}  net={c-b:+d} (b={b} c={c})  p={mcnemar_p(b,c):.4f}")
print(f"  winning tasks (half0 learned on half1): {sorted(win_sets[0])}")
print(f"  winning tasks (half1 learned on half0): {sorted(win_sets[1])}")
