"""
Aggregate analysis of TrajGazeMerge diagnostic eval output.

Reads <tag>_per_sample.parquet (from diagnostic_eval.py) and produces:

  1. score_histogram.png            — score_mean/std/max/entropy distributions,
                                       split by correct/incorrect and by task
  2. temporal_concentration.png     — average kept_per_frame curve over 64 frames,
                                       correct vs incorrect overlay  [issue 2]
  3. temporal_features_scatter.png  — temporal_CoM / late_half_ratio vs correct
  4. token_quality_vs_acc.png       — gt_gaze_recall, temporal_entropy vs correct
                                       per-task scatter                [issue 1]
  5. cluster_size_hist.png          — cluster size + cosine sim distributions [issue 3]
  6. feature_effect_size.csv        — Cohen's d, AUC of each feature for correct
                                       prediction                      [issue 1,3]
  7. confidence_calibration.png     — logit_margin and top1_prob distributions
                                       + reliability diagram (ECE)     [issue 4b]
  8. summary.md                     — markdown summary of key findings

Usage:
  python -m TrajGazeMerge.eval.analyze_diagnostics --tag E1_keep10_diag
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    return p.parse_args()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    sx, sy = x.var(ddof=1), y.var(ddof=1)
    sp = math_sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / sp)


def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(max(0.0, x))


def feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of feature `values` predicting binary `labels` (True=positive)."""
    v = np.asarray(values, dtype=float); l = np.asarray(labels, dtype=bool)
    m = np.isfinite(v)
    v, l = v[m], l[m]
    if len(v) < 4 or l.sum() == 0 or (~l).sum() == 0:
        return float("nan")
    order = np.argsort(v)
    l_sorted = l[order]
    n_pos = l.sum(); n_neg = (~l).sum()
    cum_neg = np.cumsum(~l_sorted)
    auc = float((cum_neg[l_sorted].sum()) / (n_pos * n_neg))
    return auc


def ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bins[-1] = 1.0 + 1e-9
    rows = []
    err = 0.0
    n = len(probs)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]), "n": 0,
                         "mean_conf": float("nan"), "acc": float("nan")})
            continue
        mean_conf = float(probs[m].mean())
        acc       = float(correct[m].mean())
        rows.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]), "n": cnt,
                     "mean_conf": mean_conf, "acc": acc})
        err += (cnt / n) * abs(mean_conf - acc)
    return float(err), rows


def plot_score_histogram(df: pd.DataFrame, out_path: str):
    cols = ["score_mean", "score_std", "score_max", "score_entropy"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, cols):
        c = df.loc[df["correct"], col].dropna()
        w = df.loc[~df["correct"], col].dropna()
        ax.hist(c, bins=30, alpha=0.55, label=f"correct (n={len(c)})", color="tab:green")
        ax.hist(w, bins=30, alpha=0.55, label=f"incorrect (n={len(w)})", color="tab:red")
        ax.set_title(col); ax.legend(fontsize=8)
    fig.suptitle("TrajGaze score distribution statistics (per sample)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_temporal_concentration(df: pd.DataFrame, out_path: str):
    # Samples may have different T_merged. Filter to the modal length to compare
    # frame-by-frame; report how many samples we kept.
    lengths = df["kept_per_frame"].apply(len)
    modal_T = int(lengths.mode().iloc[0])
    mask = (lengths == modal_T).values
    df_use = df[mask]
    kpf = np.stack(df_use["kept_per_frame"].apply(np.asarray).to_list())   # (N, T)
    T = kpf.shape[1]
    avg_all = kpf.mean(axis=0)
    correct = df_use["correct"].values.astype(bool)
    avg_corr = kpf[correct].mean(axis=0) if correct.any() else np.zeros(T)
    avg_wrong = kpf[~correct].mean(axis=0) if (~correct).any() else np.zeros(T)
    uniform = kpf.sum() / (kpf.shape[0] * T)
    drop_pct = 100.0 * (1 - mask.mean())
    title_suffix = f"  |  T_merged={T} (kept {mask.sum()}/{len(df)} samples; dropped {drop_pct:.1f}% with other lengths)"

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    x = np.arange(T)
    ax.plot(x, avg_all, label=f"all (n={len(df_use)})", color="black", lw=2)
    ax.plot(x, avg_corr, label=f"correct (n={int(correct.sum())})", color="tab:green")
    ax.plot(x, avg_wrong, label=f"incorrect (n={int((~correct).sum())})", color="tab:red")
    ax.axhline(uniform, color="gray", ls="--", lw=1, label=f"uniform = {uniform:.2f}")
    ax.set_xlabel("Qwen frame index (0..T_merged-1)")
    ax.set_ylabel("avg kept tokens / frame")
    ax.set_title("Temporal concentration of kept tokens" + title_suffix)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_temporal_features_scatter(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["temporal_center_of_mass", "late_half_ratio"]):
        for label, color in [(True, "tab:green"), (False, "tab:red")]:
            sub = df[df["correct"] == label]
            ax.hist(sub[col].dropna(), bins=25, alpha=0.55,
                    label=f"{'correct' if label else 'incorrect'} (n={len(sub)})",
                    color=color)
        ax.axvline(0.5, color="gray", ls="--", lw=1, label="uniform")
        ax.set_xlabel(col); ax.set_ylabel("count"); ax.legend(fontsize=8)
        ax.set_title(col)
    fig.suptitle("Temporal bias features vs correctness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_token_quality_vs_acc(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, xlabel in zip(
        axes,
        ["gt_gaze_recall", "temporal_entropy"],
        ["gt_gaze_recall @ keep_ratio", "temporal_entropy of kept frames"],
    ):
        for task, g in df.groupby("task"):
            ax.scatter(g[col], g["correct"].astype(int) + np.random.uniform(-0.05, 0.05, size=len(g)),
                       alpha=0.45, s=14, label=task[:30])
        ax.set_xlabel(xlabel); ax.set_ylabel("correct (jittered)")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["wrong", "correct"])
        if col == "gt_gaze_recall":
            # Show uniform baseline = keep_ratio
            if "keep_count" in df.columns and "n_video" in df.columns:
                kr = float((df["keep_count"] / df["n_video"]).mean())
                ax.axvline(kr, color="gray", ls="--", lw=1, label=f"random = {kr:.2f}")
        ax.legend(fontsize=7, ncol=2, loc="center right")
    fig.suptitle("Token-selection quality vs correctness (per task)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_cluster_size(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].hist(df["cluster_size_mean"].dropna(), bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("cluster_size_mean"); axes[0].set_title("mean sources per receiver")
    axes[1].hist(df["cluster_size_max"].dropna(),  bins=30, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("cluster_size_max");  axes[1].set_title("max sources per receiver")
    axes[2].hist(df["src_recv_cos_mean"].dropna(), bins=30, color="tab:green", alpha=0.8)
    axes[2].set_xlabel("src->recv cosine (mean)"); axes[2].set_title("source-receiver similarity")
    fig.suptitle("Merge structural statistics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_confidence_calibration(df: pd.DataFrame, out_path: str, n_bins: int = 10):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # (a) logit_margin distribution
    for label, color in [(True, "tab:green"), (False, "tab:red")]:
        sub = df[df["correct"] == label]
        axes[0].hist(sub["logit_margin"].dropna(), bins=40, alpha=0.55,
                     color=color, label=f"{'correct' if label else 'incorrect'}")
    axes[0].set_xlabel("logit_margin (top1 - top2)"); axes[0].set_title("Logit margin"); axes[0].legend()
    # (b) top1_prob distribution
    for label, color in [(True, "tab:green"), (False, "tab:red")]:
        sub = df[df["correct"] == label]
        axes[1].hist(sub["top1_prob"].dropna(), bins=40, alpha=0.55,
                     color=color, label=f"{'correct' if label else 'incorrect'}")
    axes[1].set_xlabel("top1 softmax prob"); axes[1].set_title("Top1 confidence"); axes[1].legend()
    # (c) reliability diagram
    p = df["top1_prob"].to_numpy()
    c = df["correct"].astype(bool).to_numpy()
    ece_val, rows = ece(p, c, n_bins=n_bins)
    confs = [r["mean_conf"] for r in rows]
    accs  = [r["acc"]       for r in rows]
    counts = [r["n"]        for r in rows]
    axes[2].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[2].scatter(confs, accs, s=[5 + 0.5*n for n in counts], alpha=0.7)
    axes[2].set_xlabel("confidence (top1 prob)"); axes[2].set_ylabel("accuracy")
    axes[2].set_title(f"Reliability  |  ECE = {ece_val:.3f}")
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    fig.suptitle("Confidence calibration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return ece_val, rows


def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "score_mean", "score_std", "score_max", "score_entropy",
        "temporal_center_of_mass", "temporal_entropy",
        "late_half_ratio", "first_half_ratio",
        "spatial_entropy", "spatial_com_x", "spatial_com_y",
        "cluster_size_mean", "cluster_size_max", "cluster_size_std",
        "src_recv_cos_mean", "src_recv_cos_min",
        "gt_gaze_recall",
        "logit_margin", "top1_prob",
    ]
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        c = df.loc[df["correct"], f].to_numpy()
        w = df.loc[~df["correct"], f].to_numpy()
        rows.append({
            "feature": f,
            "mean_correct":   float(np.nanmean(c)) if len(c) else float("nan"),
            "mean_incorrect": float(np.nanmean(w)) if len(w) else float("nan"),
            "cohens_d": cohens_d(c, w),
            "auc_correct": feature_auc(df[f].to_numpy(), df["correct"].astype(bool).to_numpy()),
        })
    out = pd.DataFrame(rows)
    out["abs_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_d", ascending=False)


def write_summary_md(df: pd.DataFrame, eff: pd.DataFrame, ece_val: float, out_path: str, tag: str):
    overall = 100.0 * df["correct"].mean()
    keep_ratio = float((df["keep_count"] / df["n_video"]).mean())
    by_task = (
        df.groupby("task")["correct"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    by_task["accuracy"] = (by_task["accuracy"] * 100).round(2)

    lines: list[str] = []
    lines.append(f"# Diagnostic Summary — {tag}\n")
    lines.append(f"- n_samples: **{len(df)}**, overall accuracy: **{overall:.2f}%**")
    lines.append(f"- keep ratio (kept / n_video): **{keep_ratio:.3f}**")
    lines.append(f"- ECE (top1 prob, 10 bins): **{ece_val:.4f}**\n")

    lines.append("## Per-task accuracy")
    lines.append("| task | n | acc (%) |")
    lines.append("|---|---:|---:|")
    for _, r in by_task.iterrows():
        lines.append(f"| {r['task']} | {int(r['n'])} | {r['accuracy']:.2f} |")
    lines.append("")

    lines.append("## Key global means")
    gm = {
        "temporal_center_of_mass (0.5=uniform)": df["temporal_center_of_mass"].mean(),
        "late_half_ratio (0.5=uniform)":          df["late_half_ratio"].mean(),
        "first_half_ratio (0.5=uniform)":         df["first_half_ratio"].mean(),
        f"gt_gaze_recall (random={keep_ratio:.3f})": df["gt_gaze_recall"].mean(skipna=True),
        "mean cluster size (uniform=N/keep)":     df["cluster_size_mean"].mean(),
        "max cluster size":                       df["cluster_size_max"].mean(),
        "src→recv cosine":                        df["src_recv_cos_mean"].mean(),
        "top1 softmax prob":                      df["top1_prob"].mean(),
        "logit margin (top1 - top2)":             df["logit_margin"].mean(),
    }
    lines.append("| metric | mean |\n|---|---:|")
    for k, v in gm.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    lines.append("## Feature effect sizes (sorted by |Cohen's d|)")
    lines.append("Cohen's d > 0 ⇒ higher in correct samples. AUC = ROC-AUC of feature predicting correctness.\n")
    lines.append("| feature | mean_correct | mean_incorrect | Cohen's d | AUC |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in eff.iterrows():
        lines.append(
            f"| {r['feature']} | {r['mean_correct']:.4f} | {r['mean_incorrect']:.4f} | "
            f"{r['cohens_d']:.3f} | {r['auc_correct']:.3f} |"
        )
    lines.append("")

    lines.append("## Reading guide")
    lines.append("- **Issue 2 (temporal bias)**: `temporal_center_of_mass`/`late_half_ratio` "
                 "far from 0.5 ⇒ kept tokens skewed in time. Compare correct vs incorrect to see "
                 "if bias hurts accuracy.")
    lines.append("- **Issue 1 (usefulness)**: large positive Cohen's d on `gt_gaze_recall` would "
                 "indicate selecting tokens that overlap GT gaze helps. If `gt_gaze_recall` ≤ keep ratio, "
                 "the encoder is not selecting gaze-aligned tokens.")
    lines.append("- **Issue 3 (merge structure)**: `cluster_size_max` near `n_video - keep_count` "
                 "means a single receiver absorbed most sources. `src_recv_cos_mean` near 1 ⇒ sources "
                 "and receivers are redundant; near 0 ⇒ merge mixes dissimilar content.")
    lines.append("- **Issue 4b (guessing)**: ECE near 0 ⇒ well-calibrated. High top1 prob on incorrect "
                 "samples ⇒ overconfident errors. Bimodal logit_margin ⇒ \"sure\" vs \"guess\" regimes.")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    parquet_path = os.path.join(args.results_dir, "diagnostic", f"{args.tag}_per_sample.parquet")
    if not os.path.exists(parquet_path):
        jsonl = parquet_path.replace(".parquet", ".jsonl")
        if os.path.exists(jsonl):
            df = pd.read_json(jsonl, lines=True)
        else:
            raise FileNotFoundError(f"No per-sample file at {parquet_path}")
    else:
        df = pd.read_parquet(parquet_path)

    out_dir = os.path.join(args.results_dir, "diagnostic", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Analyze] tag={args.tag}  n_samples={len(df)}  acc={100*df['correct'].mean():.2f}%")

    plot_score_histogram(df,           os.path.join(out_dir, "score_histogram.png"))
    plot_temporal_concentration(df,    os.path.join(out_dir, "temporal_concentration.png"))
    plot_temporal_features_scatter(df, os.path.join(out_dir, "temporal_features_scatter.png"))
    plot_token_quality_vs_acc(df,      os.path.join(out_dir, "token_quality_vs_acc.png"))
    plot_cluster_size(df,              os.path.join(out_dir, "cluster_size_hist.png"))
    ece_val, _ = plot_confidence_calibration(df, os.path.join(out_dir, "confidence_calibration.png"))

    eff = compute_effect_sizes(df)
    eff.to_csv(os.path.join(out_dir, "feature_effect_size.csv"), index=False)

    summary_path = os.path.join(out_dir, "summary.md")
    write_summary_md(df, eff, ece_val, summary_path, args.tag)
    print(f"  outputs -> {out_dir}")
    print(f"  summary  -> {summary_path}")
    print(f"  ECE: {ece_val:.4f}")
    print("\nTop 5 features by |Cohen's d|:")
    print(eff.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
