#!/usr/bin/env python
"""Assemble Tables 6 and 7 from the ablation runs.

Reads each Stage-2 row's end-of-epoch eval record from train_log_rank0.jsonl and the
\\sys row's per-task numbers from the already-measured SG-only specialist eval logs.

Avg = unweighted macro-average over the 7 reported tasks. OTP is excluded: it has
n=2 out of 526, so it only ever reads 0/50/100%.
"""
from __future__ import annotations

import json
import os
import re
import sys

REPO = os.environ.get("REPO", "/NHNHOME/VILAB/vilab_yj/trajgaze")
CKPT = f"{REPO}/TrajGazeMerge/checkpoints"

# paper column -> internal task key (TrajGazeMerge/data/dataset.py MCQ_TASKS)
COLS = [
    ("GSM",  "past_gaze_sequence_matching"),
    ("NFI",  "past_non_fixated_object_identification"),
    ("SR",   "past_scene_recall"),
    ("OAR",  "present_object_attribute_recognition"),
    ("OI-E", "present_object_identification_easy"),
    ("OI-H", "present_object_identification_hard"),
    ("FAP",  "present_future_action_prediction"),
]
OTP = "past_object_transition_prediction"

TRAINED_ROWS = [                       # (table, row label, run dir)
    ("T6", "No pretrain",     "tab6_nopretrain_overlay"),
    ("T6", "Only score loss", "tab6_scoreonly_overlay"),
    ("T7", "No spatial",      "tab7_nospatial_overlay"),
    ("T7", "No temporal",     "tab7_notemporal_overlay"),
]
# \sys row: already measured, 3 independent evals of $M1_SGONLY
SYS_LOGS = ["eval_m1_sgonly_pertask.log", "repeat_teacher_r3.log", "repeat_teacher_r4.log"]


def macro7(per_task: dict[str, float]) -> float:
    return sum(per_task[k] for _, k in COLS) / len(COLS)


def from_jsonl(run_dir: str):
    """Last end-of-epoch eval record of a training run."""
    path = f"{CKPT}/{run_dir}/train_log_rank0.jsonl"
    if not os.path.exists(path):
        return None, f"missing {path}"
    rec = None
    with open(path) as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_acc" in obj:      # step lines have no eval_acc
                rec = obj
    if rec is None:
        return None, f"no eval record in {path}"
    per_task = {k: v for k, v in rec.get("per_task", {}).items()}
    missing = [k for _, k in COLS if k not in per_task]
    if missing:
        return None, f"missing task keys {missing}"
    n = rec.get("per_src", {}).get("sg", [None, None])[1]
    return {"per_task": per_task, "n": n, "overall": rec.get("eval_acc"),
            "otp": per_task.get(OTP)}, None


def from_eval_log(path: str):
    """Per-task block of an eval-only stdout log."""
    if not os.path.exists(path):
        return None, f"missing {path}"
    per_task, n = {}, None
    pat = re.compile(r"^\[eval-only\]\s+(past_\w+|present_\w+):\s+([\d.]+)%")
    npat = re.compile(r"^\[eval-only\]\s+Overall:\s+([\d.]+)%\s+\(n=(\d+)\)")
    with open(path) as fh:
        for line in fh:
            if (m := pat.match(line)):
                per_task[m.group(1)] = float(m.group(2))
            elif (m := npat.match(line)):
                n = int(m.group(2))
    missing = [k for _, k in COLS if k not in per_task]
    if missing:
        return None, f"{path}: missing {missing}"
    return {"per_task": per_task, "n": n, "otp": per_task.get(OTP)}, None


def fmt_row(label: str, per_task: dict[str, float], note: str = "") -> str:
    cells = " | ".join(f"{per_task[k]:5.2f}" for _, k in COLS)
    return f"| {label:<24} | {cells} | **{macro7(per_task):5.2f}** |{note}"


def main() -> int:
    header = "| " + " " * 24 + " | " + " | ".join(f"{c:>5}" for c, _ in COLS) + " | **Avg** |"
    sep = "|" + "---|" * (len(COLS) + 2)

    rows, problems = {}, []

    for table, label, run_dir in TRAINED_ROWS:
        res, err = from_jsonl(run_dir)
        if err:
            problems.append(f"{label}: {err}")
            continue
        if res["n"] != 526:
            problems.append(f"{label}: sg n={res['n']}, expected 526 "
                            "(items were silently dropped — row not comparable)")
        rows.setdefault(table, []).append((label, res["per_task"], ""))

    # \sys row — mean over the 3 existing evals
    sys_runs = []
    for name in SYS_LOGS:
        res, err = from_eval_log(f"{REPO}/{name}")
        if err:
            problems.append(err)
        else:
            sys_runs.append(res["per_task"])
    if sys_runs:
        mean = {k: sum(r[k] for r in sys_runs) / len(sys_runs)
                for _, k in COLS}
        note = f"  <- mean of {len(sys_runs)} evals of the SG-only specialist"
        rows.setdefault("T6", []).append(("All losses (ours)", mean, note))
        rows.setdefault("T7", []).append(("Spatio-temporal (ours)", mean, note))

    for table, title in [("T6", "Table 6 — Stage-1 pretraining objectives"),
                         ("T7", "Table 7 — spatial vs. temporal selection")]:
        print(f"\n### {title}  (StreamGaze EGTEA, n=526, macro-7, OTP excluded)\n")
        print(header)
        print(sep)
        for label, per_task, note in rows.get(table, []):
            print(fmt_row(label, per_task, note))

    if problems:
        print("\n### Problems\n")
        for p in problems:
            print(f"- {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
