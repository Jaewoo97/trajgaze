"""
Phase M1.2 — Score-trajectory feature correlation analysis.

For each test sample, correlates the per-frame number of kept tokens with
trajectory-derived features:
  - gaze_speed (per-frame scalar)
  - left_velocity_magnitude
  - right_velocity_magnitude
  - convergence (gaze-hand distance)
  - hand_presence (left + right mask sum)
  - frame_index (temporal position, normalized 0-1)

Output: per-feature mean Pearson correlation across 526 samples + plots.

If `kept_per_frame` correlates strongly with hand_velocity → "hand-tracking"
hypothesis (H1). If with frame_index → "temporal late-bias" (H2). If with
convergence → "hand-object interaction" (H3).

Usage:
  python -m TrajGazeMerge.eval.analyze_M1_correlations
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/trajgaze")
from TrajGazeMerge.data.dataset import StreamGazeMergeDataset

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic"


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(
        ["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]) + "|")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt) if v == v else "—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def frame_feature_from_traj(traj: dict, T_traj: int) -> dict[str, np.ndarray]:
    """Per-trajectory-frame scalar features, length T_traj."""
    f: dict[str, np.ndarray] = {}
    f["gaze_speed"] = traj["gaze_speed"].cpu().numpy().squeeze()      # (T,)
    lv = traj["left_vel"].cpu().numpy()                                # (T, 2)
    rv = traj["right_vel"].cpu().numpy()
    f["left_velocity"]  = np.linalg.norm(lv, axis=1)
    f["right_velocity"] = np.linalg.norm(rv, axis=1)
    f["hand_velocity"]  = (f["left_velocity"] + f["right_velocity"]) / 2
    conv = traj["convergence"].cpu().numpy()                           # (T,) — distance between gaze and hand midpoint
    f["convergence"]    = conv.squeeze() if conv.ndim > 1 else conv
    f["hand_presence"]  = (
        traj["left_mask"].cpu().numpy().astype(float) +
        traj["right_mask"].cpu().numpy().astype(float)
    )
    f["gaze_presence"]  = traj["gaze_mask"].cpu().numpy().astype(float)
    f["frame_index"]    = np.linspace(0, 1, T_traj)
    # Distance from gaze position to frame center (= "gaze peripherality")
    gp = traj["gaze_pos"].cpu().numpy()                                # (T, 2)
    f["gaze_to_center"] = np.linalg.norm(gp - 0.5, axis=1)
    return f


def resample_to_T_merged(x: np.ndarray, T_merged: int) -> np.ndarray:
    """Linear-interp x of length T_traj to length T_merged."""
    T_traj = len(x)
    if T_traj == T_merged:
        return x
    return np.interp(np.linspace(0, T_traj - 1, T_merged), np.arange(T_traj), x)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="If set, use {RESULTS_DIR}/{tag}_per_sample.parquet "
                         "(e.g. --tag E1_sprint2_1_C_diag). Default: legacy E1_keep10 path.")
    args = ap.parse_args()

    if args.tag:
        parquet_path = f"{RESULTS_DIR}/{args.tag}_per_sample.parquet"
        out_suffix = args.tag
    else:
        parquet_path = f"{RESULTS_DIR}/E1_keep10_diag_v2_per_sample.parquet"
        if not os.path.exists(parquet_path):
            parquet_path = f"{RESULTS_DIR}/E1_keep10_diag_per_sample.parquet"
            print(f"[warn] v2 parquet not found, falling back to {parquet_path}")
        out_suffix = "E1_keep10"
    df = pd.read_parquet(parquet_path)
    print(f"loaded {len(df)} rows from {parquet_path}")

    # Need to load dataset to access trajectory data
    ds = StreamGazeMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128)

    # For each row in parquet, fetch trajectory via dataset[idx]
    features = [
        "gaze_speed", "left_velocity", "right_velocity", "hand_velocity",
        "convergence", "hand_presence", "gaze_presence",
        "frame_index", "gaze_to_center",
    ]
    correlations: dict[str, list[float]] = {f: [] for f in features}
    n_processed = 0
    n_skipped = 0

    for i, row in df.iterrows():
        idx = int(row["idx"])
        item = ds[idx]
        if item is None:
            n_skipped += 1
            continue
        kept = np.asarray(row["kept_per_frame"], dtype=float)
        T_merged = len(kept)
        feats = frame_feature_from_traj(item["traj"], item["traj"]["gaze_pos"].shape[0])
        for f_name in features:
            f_resampled = resample_to_T_merged(feats[f_name], T_merged)
            correlations[f_name].append(pearson(kept, f_resampled))
        n_processed += 1
        if (n_processed + 1) % 50 == 0:
            print(f"  [{n_processed+1}/{len(df)}]")

    print(f"processed {n_processed}, skipped {n_skipped}")

    # Summary stats
    summary_rows = []
    for f_name in features:
        vals = np.array(correlations[f_name])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary_rows.append({
            "feature": f_name,
            "mean_corr": float(vals.mean()),
            "median_corr": float(np.median(vals)),
            "std_corr": float(vals.std()),
            "pct_positive": float(100 * (vals > 0).mean()),
            "pct_strong_pos": float(100 * (vals > 0.3).mean()),
            "pct_strong_neg": float(100 * (vals < -0.3).mean()),
            "n_samples": int(len(vals)),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_corr",
                                                        ascending=False, key=abs)

    # By correctness
    df["correct_bool"] = df["correct"].astype(bool)
    correct_rows = []
    for f_name in features:
        vals = np.array(correlations[f_name])
        mask_correct = df["correct_bool"].iloc[:len(vals)].values
        c_v = vals[mask_correct]; w_v = vals[~mask_correct]
        c_v = c_v[~np.isnan(c_v)]; w_v = w_v[~np.isnan(w_v)]
        correct_rows.append({
            "feature": f_name,
            "mean_corr_correct": float(c_v.mean()) if len(c_v) else float("nan"),
            "mean_corr_incorrect": float(w_v.mean()) if len(w_v) else float("nan"),
            "diff": float(c_v.mean() - w_v.mean()) if len(c_v) and len(w_v) else float("nan"),
        })
    correct_df = pd.DataFrame(correct_rows).sort_values("diff", ascending=False, key=abs)

    # Save markdown
    lines: list[str] = []
    lines.append("# Phase M1.2 — Score–Trajectory Feature Correlation\n")
    lines.append(f"For each of {n_processed} samples, Pearson correlation between "
                 "`kept_per_frame` (count of receiver tokens per Qwen frame) and "
                 "a trajectory-derived per-frame feature, after linear-interp resampling "
                 "to T_merged.\n")
    lines.append("## Summary (sorted by |mean_corr|)\n")
    lines.append(df_to_md(summary_df, ".3f"))
    lines.append("")
    lines.append("**Interpretation guide:**")
    lines.append("- `frame_index` mean_corr > 0.3 ⇒ kept-token count strongly increases with frame number → temporal late-bias (H2)")
    lines.append("- `hand_velocity` mean_corr > 0.2 ⇒ encoder concentrates tokens when hands move → hand-tracking (H1)")
    lines.append("- `hand_presence` mean_corr > 0.2 ⇒ encoder follows when hand is visible (H1)")
    lines.append("- `convergence` mean_corr ≈ 0 if no relationship; significant ⇒ hand-object interaction (H3)")
    lines.append("- `gaze_speed` ≈ 0 expected (encoder doesn't follow raw gaze speed)\n")

    lines.append("## Correlation diff (correct − incorrect samples)\n")
    lines.append(df_to_md(correct_df, ".3f"))
    lines.append("\nLarge |diff| ⇒ that feature's encoder-alignment differs between correct and incorrect answers.\n")

    # Plot distribution of correlations per feature
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, f_name in zip(axes.flat, features):
        vals = np.array(correlations[f_name])
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=40, color="tab:blue", alpha=0.7)
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.axvline(vals.mean(), color="red", ls="-", lw=1.5, label=f"mean={vals.mean():.3f}")
        ax.set_title(f_name)
        ax.set_xlabel("Pearson r")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Per-sample correlation of kept_per_frame with trajectory features")
    fig.tight_layout()
    fig_path = f"{RESULTS_DIR}/M1_correlations_{out_suffix}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"  plot -> {fig_path}")

    out_path = f"{RESULTS_DIR}/PHASE_M1_2_correlations_{out_suffix}.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"summary -> {out_path}")
    print()
    print("\n".join(lines[:60]))

    # Save raw correlations for further analysis
    np.savez(f"{RESULTS_DIR}/M1_correlations_raw_{out_suffix}.npz",
             **{f_name: np.array(correlations[f_name]) for f_name in features})


if __name__ == "__main__":
    main()
