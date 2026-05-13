"""
Stage 1 held-out evaluation across epoch checkpoints.

Splits StreamGazeStage1DatasetTemporal clips into deterministic 90/10
train/val via index hash, evaluates per-epoch checkpoints on the val subset
under a fixed seed (so the past/future split ratio and frame sampling are
identical across epochs), and writes:

  RESULTS_DIR/holdout_<ckpt_name>_val_loss.json   (raw per-epoch losses)
  RESULTS_DIR/holdout_<ckpt_name>_loss_curve.png  (plot)

A val curve that plateaus or rises while train loss keeps falling is direct
evidence of Stage 1 overfitting on the small training set.

Usage:
  python -m TrajGaze_v2.training.eval_stage1_holdout \
      --ckpt-dir /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_patch_temporal \
      --epochs 10 20 30 40 50 60 70 80 90 100 \
      --n-val 60 \
      --tag E1_patch_temporal_holdout
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/workspace/trajgaze")

from TrajGaze_v2.data.dataset_temporal import (
    StreamGazeStage1DatasetTemporal, collate_stage1_temporal,
)
from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal

RESULTS_DIR = "/workspace/trajgaze/TrajGaze_v2/eval_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir",   required=True, help="dir containing epoch_NNNN.pth")
    p.add_argument("--epochs",     type=int, nargs="+",
                   default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument("--also-best",  action="store_true",
                   help="also evaluate best.pth and final.pth if present")
    p.add_argument("--n-val",      type=int, default=60,
                   help="cap on val subset size for speed (random subset)")
    p.add_argument("--val-frac",   type=float, default=0.10,
                   help="held-out fraction of clips")
    p.add_argument("--n-frames",   type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--seed",       type=int, default=2026)
    p.add_argument("--no-visual",  action="store_true",
                   help="set frame_paths=None during forward (faster)")
    p.add_argument("--tag",        required=True)
    return p.parse_args()


def deterministic_val_indices(n_total: int, val_frac: float, seed: int = 2026) -> list[int]:
    """Pick val indices deterministically by hash, independent of order."""
    val: list[int] = []
    for i in range(n_total):
        h = hashlib.sha1(f"{seed}-{i}".encode()).hexdigest()
        if (int(h[:8], 16) / 0xFFFFFFFF) < val_frac:
            val.append(i)
    return val


def parse_epoch_from_filename(path: str) -> int:
    m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _set_all_seeds(s: int):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    random.seed(s); np.random.seed(s)


def to_dev(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def infer_arch_flags(state: dict) -> dict:
    return {
        "use_frame_score_branch": any(
            k.startswith("encoder.frame_attn_pool") or k.startswith("encoder.frame_score_head")
            for k in state
        ),
        "use_post_fusion_iframe": any(k.startswith("encoder.inter_frame_post") for k in state),
        "use_patch_temporal_branch": any(
            k.startswith("encoder.patch_temporal_query")
            or k.startswith("encoder.patch_temporal_attn")
            or k.startswith("encoder.patch_temporal_head")
            for k in state
        ),
        "use_iframe_query_conditioning": any(
            k.startswith("encoder.iframe_to_query") for k in state
        ),
    }


@torch.no_grad()
def evaluate_one(model, val_loader, device, seed: int, no_visual: bool) -> dict:
    """Return mean losses over val_loader. Seed is reset for deterministic sampling."""
    _set_all_seeds(seed)
    model.eval()
    sums = {"loss": 0.0, "loss_traj": 0.0, "loss_score_fut": 0.0,
            "loss_score_past": 0.0, "loss_score_traj": 0.0}
    n = 0
    for batch in val_loader:
        if batch is None:
            continue
        batch["past"]            = to_dev(batch["past"], device)
        batch["future"]          = to_dev(batch["future"], device)
        batch["I_scores_past"]   = batch["I_scores_past"].to(device)
        batch["I_scores_future"] = batch["I_scores_future"].to(device)
        batch["T_past"]          = batch["T_past"].to(device)
        batch["T_future"]        = batch["T_future"].to(device)
        if no_visual:
            batch["frame_paths"] = None
        out = model.stage1_forward(batch)
        for k in sums:
            sums[k] += float(out[k].item())
        n += 1
    return {k: (v / max(1, n)) for k, v in sums.items()} | {"n_batches": n}


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build full dataset, then deterministically pick held-out subset.
    full_ds = StreamGazeStage1DatasetTemporal(n_frames=args.n_frames)
    val_idx_all = deterministic_val_indices(len(full_ds), args.val_frac, seed=args.seed)
    if args.n_val > 0 and len(val_idx_all) > args.n_val:
        rng = random.Random(args.seed)
        val_idx = sorted(rng.sample(val_idx_all, args.n_val))
    else:
        val_idx = val_idx_all
    print(f"[Holdout] total={len(full_ds)}  val_frac={args.val_frac}  "
          f"val_size={len(val_idx)}  (cap={args.n_val})")

    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_stage1_temporal,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    # Resolve checkpoint paths
    ckpt_paths: list[tuple[str, int, str]] = []   # (path, epoch, label)
    for ep in args.epochs:
        p = os.path.join(args.ckpt_dir, f"epoch_{ep:04d}.pth")
        if os.path.exists(p):
            ckpt_paths.append((p, ep, f"epoch_{ep:04d}"))
        else:
            print(f"  missing: {p}")
    if args.also_best:
        for name in ["best.pth", "final.pth"]:
            p = os.path.join(args.ckpt_dir, name)
            if os.path.exists(p):
                ckpt_paths.append((p, 9999 if name == "best.pth" else 10000, name.replace(".pth", "")))

    if not ckpt_paths:
        print("No checkpoints to evaluate.")
        return

    # Build model once (architecture inferred from first ckpt that exists)
    first_state = torch.load(ckpt_paths[0][0], map_location="cpu", weights_only=False)
    sd_root     = first_state.get("model", first_state.get("model_state_dict", first_state))
    flags       = infer_arch_flags(sd_root)
    print(f"[Holdout] architecture flags: {flags}")
    model = TrajGazeV2Temporal(n_vis_keyframes=16, **flags).to(device)

    results: list[dict] = []
    for path, epoch, label in ckpt_paths:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            sd   = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
            res  = model.load_state_dict(sd, strict=False)
            if res.missing_keys:
                print(f"  [{label}] missing keys: {len(res.missing_keys)} (first 3: {res.missing_keys[:3]})")
            losses = evaluate_one(model, val_loader, device, args.seed, args.no_visual)
            losses["label"] = label
            losses["epoch"] = epoch
            losses["ckpt"]  = path
            print(f"  {label:>14s}  loss={losses['loss']:.4f}  "
                  f"traj={losses['loss_traj']:.4f}  "
                  f"score_fut={losses['loss_score_fut']:.4f}  "
                  f"score_past={losses['loss_score_past']:.4f}  "
                  f"score_traj={losses['loss_score_traj']:.4f}  "
                  f"(n_batches={losses['n_batches']})")
            results.append(losses)
        except Exception:
            traceback.print_exc()
            continue

    # Save
    summary = {
        "tag": args.tag,
        "ckpt_dir": args.ckpt_dir,
        "val_size": len(val_idx),
        "val_indices": val_idx,
        "seed": args.seed,
        "results": results,
    }
    json_path = os.path.join(RESULTS_DIR, f"holdout_{args.tag}_val_loss.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved -> {json_path}")

    # Plot
    epoch_results = [r for r in results if r["epoch"] < 9999]
    if epoch_results:
        epoch_results.sort(key=lambda r: r["epoch"])
        xs = [r["epoch"] for r in epoch_results]
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for key, color in [
            ("loss", "black"),
            ("loss_traj", "tab:blue"),
            ("loss_score_fut", "tab:orange"),
            ("loss_score_past", "tab:green"),
            ("loss_score_traj", "tab:red"),
        ]:
            ys = [r[key] for r in epoch_results]
            ax.plot(xs, ys, marker="o", label=key, color=color, lw=1.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean val loss (held-out clips, fixed seed)")
        ax.set_title(f"Stage 1 held-out loss vs epoch  [{args.tag}]")
        ax.grid(alpha=0.3); ax.legend()
        # Mark best.pth row if available
        best_row = next((r for r in results if r["label"] == "best"), None)
        if best_row is not None:
            ax.axhline(best_row["loss"], color="gray", ls="--", lw=1,
                       label=f"best.pth = {best_row['loss']:.4f}")
            ax.legend()
        png_path = os.path.join(RESULTS_DIR, f"holdout_{args.tag}_loss_curve.png")
        fig.tight_layout(); fig.savefig(png_path, dpi=130); plt.close(fig)
        print(f"  plot  -> {png_path}")
        # Identify overfit signal
        traj_curve = [r["loss_traj"] for r in epoch_results]
        if len(traj_curve) >= 3:
            best_idx = int(np.argmin(traj_curve))
            print(f"  best epoch by val loss_traj: {epoch_results[best_idx]['label']} "
                  f"(loss_traj={traj_curve[best_idx]:.4f})")
            if best_idx < len(epoch_results) - 1:
                last = epoch_results[-1]["loss_traj"]
                delta = last - traj_curve[best_idx]
                print(f"  val loss_traj rebound: +{delta:.4f} between best and last "
                      f"({epoch_results[best_idx]['label']} -> {epoch_results[-1]['label']})")
                if delta > 0.01:
                    print("  >> Overfitting signal: val loss rebounded after best epoch.")


if __name__ == "__main__":
    main()
