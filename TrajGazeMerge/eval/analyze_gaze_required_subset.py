"""
Phase 0a-1: Gaze-required subset analysis.

Filters the existing diagnostic / ablation / permutation parquets to the subset
of tasks where gaze is intrinsically required to answer, then compares:
  - learned vs all ablation sources on the subset
  - per-shift / agree4 on the subset
  - vs full 526 to see if method margin grows when gaze actually matters

Rationale: on the full 526, text_only=53.6% and Open-ended=34.5% suggest
much of the headline 68.4% comes from non-gaze cues. If our method is "real",
it should show a *larger* margin on tasks that actually need gaze.

Two subset definitions:
  conservative : past_gaze_sequence_matching + past_non_fixated_object_identification
  liberal      : conservative + past_scene_recall + present_object_identification_hard

Usage:
  python -m TrajGazeMerge.eval.analyze_gaze_required_subset
"""

from __future__ import annotations

import json
import os

import pandas as pd

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results/diagnostic"


def df_to_md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Minimal markdown table renderer (avoids tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]) + "|")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt) if v == v else "—")  # NaN handling
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

CONSERVATIVE_TASKS = [
    "past_gaze_sequence_matching",
    "past_non_fixated_object_identification",
]
LIBERAL_TASKS = CONSERVATIVE_TASKS + [
    "past_scene_recall",
    "present_object_identification_hard",
]
GAZE_INTRINSIC_TASKS = [
    "past_gaze_sequence_matching",   # gaze pattern IS the question
]
NON_GAZE_TASKS = [
    "present_object_attribute_recognition",
    "present_object_identification_easy",
    "present_future_action_prediction",
]


def acc_table(df: pd.DataFrame, subsets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, tasks in subsets.items():
        if not tasks:                                   # full set
            sub = df
        else:
            sub = df[df["task"].isin(tasks)]
        if len(sub) == 0:
            rows.append({"subset": name, "n": 0, "accuracy": float("nan")})
            continue
        rows.append({
            "subset": name,
            "n": int(len(sub)),
            "accuracy": float(100 * sub["correct"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    subsets = {
        "full":         [],
        "conservative": CONSERVATIVE_TASKS,
        "liberal":      LIBERAL_TASKS,
        "gaze_intrinsic": GAZE_INTRINSIC_TASKS,
        "non_gaze":     NON_GAZE_TASKS,
    }

    out_lines: list[str] = []
    out_lines.append("# Phase 0a-1 — Gaze-Required Subset Analysis\n")
    out_lines.append("Subsets:")
    out_lines.append("- **full**: all 526 EGTEA test items")
    out_lines.append("- **conservative**: past_gaze_sequence_matching + past_non_fixated_object_identification (n≈132)")
    out_lines.append("- **liberal**: conservative + past_scene_recall + present_object_identification_hard (n≈233)")
    out_lines.append("- **gaze_intrinsic**: past_gaze_sequence_matching only (gaze pattern IS the question)")
    out_lines.append("- **non_gaze**: object attribute / easy obj ID / future action (gaze less central)\n")

    # ── Diagnostic baseline (learned, full 526) ───────────────────────────────
    df_diag = pd.read_parquet(f"{RESULTS_DIR}/E1_keep10_diag_per_sample.parquet")
    out_lines.append("## 1. Learned method accuracy by subset\n")
    diag_tab = acc_table(df_diag, subsets)
    out_lines.append(diag_tab.pipe(df_to_md, ".2f"))
    out_lines.append("")

    # Extract baselines
    baselines: dict[str, dict] = {}
    for source in ["learned", "uniform", "random", "inverted", "center",
                   "oracle", "soft_oracle", "text_only"]:
        p = f"{RESULTS_DIR}/E1_keep10_abl_ablation_{source}_per_sample.parquet"
        if not os.path.exists(p) and source == "soft_oracle":
            p = f"{RESULTS_DIR}/E1_keep10_soft_oracle_ablation_soft_oracle_per_sample.parquet"
        if not os.path.exists(p):
            print(f"missing: {p}")
            continue
        df_s = pd.read_parquet(p)
        baselines[source] = {
            name: float(100 * df_s[df_s["task"].isin(tasks) if tasks else df_s["task"].notna()]["correct"].mean())
            if len(df_s[df_s["task"].isin(tasks) if tasks else df_s["task"].notna()]) > 0 else float("nan")
            for name, tasks in subsets.items()
        }

    out_lines.append("## 2. All ablation sources by subset (acc %)\n")
    abl_df = pd.DataFrame(baselines).T.reset_index().rename(columns={"index": "source"})
    out_lines.append(abl_df.pipe(df_to_md, ".2f"))
    out_lines.append("")

    # ── Margin analysis ───────────────────────────────────────────────────────
    out_lines.append("## 3. Margins on gaze-required vs non-gaze subsets\n")
    out_lines.append("If method is *genuinely* leveraging gaze, learned − soft_oracle "
                     "and learned − text_only should be LARGER on gaze-required subsets.\n")
    if "learned" in baselines:
        margin_rows = []
        for sub_name in subsets:
            row = {"subset": sub_name}
            row["learned"] = baselines["learned"][sub_name]
            for ref in ["soft_oracle", "uniform", "random", "text_only"]:
                if ref in baselines:
                    row[f"learned − {ref}"] = baselines["learned"][sub_name] - baselines[ref][sub_name]
            margin_rows.append(row)
        out_lines.append(pd.DataFrame(margin_rows).pipe(df_to_md, ".2f"))
        out_lines.append("")

    # ── Permutation consistency on gaze-required ──────────────────────────────
    perm_path = f"{RESULTS_DIR}/E1_keep10_perm_permutation.parquet"
    if os.path.exists(perm_path):
        df_perm = pd.read_parquet(perm_path)
        df_perm["correct"] = df_perm["majority_content"] == df_perm["orig_answer"]
        out_lines.append("## 4. Permutation consistency by subset\n")
        out_lines.append("`agree4` = same option content chosen across all 4 letter-shifts. "
                         "`consistent_correct` = agree4 AND chose the correct content.\n")
        rows = []
        for name, tasks in subsets.items():
            sub = df_perm if not tasks else df_perm[df_perm["task"].isin(tasks)]
            if len(sub) == 0:
                rows.append({"subset": name, "n": 0,
                             "agree4_pct": float("nan"),
                             "consistent_correct_pct": float("nan"),
                             "avg_4shift_acc_pct": float("nan")})
                continue
            agree4 = float(100 * sub["agree4"].mean())
            consistent = float(100 * sub["consistent_correct"].mean())
            shift_cols = [f"correct_k{k}" for k in range(4)]
            avg4 = float(100 * sub[shift_cols].values.mean())
            rows.append({"subset": name, "n": int(len(sub)),
                         "agree4_pct": agree4,
                         "consistent_correct_pct": consistent,
                         "avg_4shift_acc_pct": avg4})
        out_lines.append(pd.DataFrame(rows).pipe(df_to_md, ".2f"))
        out_lines.append("")

    # ── Counterfactual masking on gaze-required ───────────────────────────────
    cf_baseline_path = f"{RESULTS_DIR}/E1_keep10_mask_mask_baseline_per_sample.parquet"
    if os.path.exists(cf_baseline_path):
        out_lines.append("## 5. Counterfactual masking by subset\n")
        out_lines.append("`Δ vs baseline` should be more negative on gaze-required if "
                         "method's chosen receivers carry gaze-relevant information.\n")
        variants = ["baseline", "mask_kept", "mask_kept_late", "mask_kept_early", "shuffle_kept"]
        var_results: dict[str, dict] = {}
        for v in variants:
            p = f"{RESULTS_DIR}/E1_keep10_mask_mask_{v}_per_sample.parquet"
            if not os.path.exists(p):
                continue
            df_v = pd.read_parquet(p)
            var_results[v] = {
                name: float(100 * df_v[df_v["task"].isin(tasks) if tasks else df_v["task"].notna()]["correct"].mean())
                for name, tasks in subsets.items()
            }
        if var_results:
            cf_df = pd.DataFrame(var_results).T.reset_index().rename(columns={"index": "variant"})
            out_lines.append(cf_df.pipe(df_to_md, ".2f"))
            out_lines.append("")
            # Δ vs baseline
            delta_rows = []
            for v in variants:
                if v == "baseline" or v not in var_results:
                    continue
                row = {"variant": v}
                for sub_name in subsets:
                    row[f"Δ {sub_name}"] = var_results[v][sub_name] - var_results["baseline"][sub_name]
                delta_rows.append(row)
            out_lines.append("Δ vs baseline (acc % drop when applying counterfactual):\n")
            out_lines.append(pd.DataFrame(delta_rows).pipe(df_to_md, ".2f"))
            out_lines.append("")

    # ── gt_gaze_recall by subset ──────────────────────────────────────────────
    out_lines.append("## 6. gt_gaze_recall by subset\n")
    out_lines.append("Mean recall of GT-gaze patch in kept-token set. Random baseline = keep ratio (0.10).\n")
    rows = []
    for name, tasks in subsets.items():
        sub = df_diag if not tasks else df_diag[df_diag["task"].isin(tasks)]
        if len(sub) == 0:
            continue
        rows.append({
            "subset": name,
            "n": int(len(sub)),
            "mean_gt_gaze_recall": float(sub["gt_gaze_recall"].mean(skipna=True)),
            "mean_late_half": float(sub["late_half_ratio"].mean()),
            "mean_temporal_CoM": float(sub["temporal_center_of_mass"].mean()),
        })
    out_lines.append(pd.DataFrame(rows).pipe(df_to_md, ".3f"))
    out_lines.append("")

    # ── Conclusion checklist ──────────────────────────────────────────────────
    out_lines.append("## 7. Phase 0a-1 verdict checklist\n")
    out_lines.append("- If learned − soft_oracle margin is **LARGER on gaze_intrinsic** → method genuinely uses gaze ✓")
    out_lines.append("- If learned − text_only margin is **LARGER on gaze_intrinsic** → visual contribution is real ✓")
    out_lines.append("- If mask_kept Δ is **more negative on gaze-required** → receivers carry task-relevant info ✓")
    out_lines.append("- If gt_gaze_recall is **higher on gaze-intrinsic** → anti-gaze pattern is task-driven, not architectural")
    out_lines.append("- If all margins **shrink** on gaze-required → StreamGaze does NOT validate gaze-attention claim; consider other datasets")

    out_path = f"{RESULTS_DIR}/PHASE0_gaze_required_subset.md"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"Phase 0a-1 analysis saved -> {out_path}")
    # Echo to stdout
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
