"""
Pick 50 stratified sample indices for Phase M1.4 visualization.

Strategy: cover the diversity space by stratifying on:
  - task (≤ 10 / task)
  - correctness (correct + incorrect roughly balanced)
  - gt_gaze_recall extremes (high recall + low recall)
  - temporal_CoM extremes (early-biased + late-biased)

Writes a JSON list of dataset indices that viz_token_selection.py can consume
via --indices-file.
"""

from __future__ import annotations

import json
import os
import random

import pandas as pd

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic"

NUM_TARGET = 50


def pick_indices(parquet_path: str, num: int = 50, seed: int = 13) -> list[int]:
    df = pd.read_parquet(parquet_path)
    rng = random.Random(seed)

    selected: set[int] = set()
    per_task = max(1, num // df["task"].nunique() // 2)

    # 1. Per-task: 1 correct + 1 incorrect (low priority extras)
    for task, g in df.groupby("task"):
        c = g[g["correct"] == True]
        w = g[g["correct"] == False]
        for sub in (c, w):
            picks = sub.sample(min(per_task, len(sub)), random_state=seed) if len(sub) > 0 else []
            for _, r in (picks.iterrows() if hasattr(picks, "iterrows") else []):
                selected.add(int(r["idx"]))

    # 2. gt_gaze_recall extremes
    if "gt_gaze_recall" in df.columns:
        top = df.nlargest(5, "gt_gaze_recall")
        bot = df.nsmallest(5, "gt_gaze_recall")
        for _, r in pd.concat([top, bot]).iterrows():
            selected.add(int(r["idx"]))

    # 3. hand_recall extremes if present
    if "gt_hand_either_recall" in df.columns:
        top = df.nlargest(5, "gt_hand_either_recall")
        for _, r in top.iterrows():
            selected.add(int(r["idx"]))

    # 4. temporal_CoM extremes
    if "temporal_center_of_mass" in df.columns:
        late = df.nlargest(5, "temporal_center_of_mass")
        early = df.nsmallest(5, "temporal_center_of_mass")
        for _, r in pd.concat([late, early]).iterrows():
            selected.add(int(r["idx"]))

    # Trim or pad to num
    sel_list = sorted(selected)
    if len(sel_list) > num:
        sel_list = sorted(rng.sample(sel_list, num))
    elif len(sel_list) < num:
        remaining = [i for i in df["idx"].astype(int).tolist() if i not in selected]
        sel_list += rng.sample(remaining, num - len(sel_list))
        sel_list = sorted(sel_list)

    return sel_list


def main():
    parquet_path = f"{RESULTS_DIR}/E1_keep10_diag_v2_per_sample.parquet"
    if not os.path.exists(parquet_path):
        parquet_path = f"{RESULTS_DIR}/E1_keep10_diag_per_sample.parquet"
    indices = pick_indices(parquet_path, NUM_TARGET)
    out_path = f"{RESULTS_DIR}/M1_viz_indices.json"
    with open(out_path, "w") as f:
        json.dump(indices, f, indent=2)
    print(f"selected {len(indices)} indices → {out_path}")
    print(indices[:20], "...")


if __name__ == "__main__":
    main()
