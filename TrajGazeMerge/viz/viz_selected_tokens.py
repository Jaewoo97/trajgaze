"""
Visualize the per-patch importance scores our TrajGazeMerge encoder produces,
with the top-10% "selected" patches outlined. These scores drive
gaze_weighted_merge selection (10% receivers + 90% sources merged in).

For each picked test item:
  - run TrajGazeV2 encoder on (traj, question, traj_frame_paths) → (196,) scores
  - reshape to 14×14
  - render 4 sample frames from the item's traj_frame_paths with the heatmap
    overlaid (alpha=0.5) and the top-19 (≈10% of 196) patches outlined
  - one composite PNG per item

Usage:
    python -m TrajGazeMerge.viz.viz_selected_tokens \
        --ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora_no_kd_combined/best.pth \
        --out  /workspace/trajgaze_st/TrajGazeMerge/viz_outputs/merge_best \
        --n-per-source 4 --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGaze_v2.models.model import TrajGazeV2

PATCH = 14            # 14x14 = 196 patches
COORD = 224           # encoder canvas
TOPK_RATIO = 0.10     # 10% selected (matches --merge-ratio 0.9)
FRAMES_PER_ITEM = 4   # thumbnails to render per sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--gpu",  type=int, default=0)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--n-per-source", type=int, default=4)
    p.add_argument("--n-traj-frames", type=int, default=32)
    p.add_argument("--n-vlm-frames",  type=int, default=128)
    p.add_argument("--sources", nargs="+", default=["sg", "eg", "hd"])
    return p.parse_args()


def render_one(item, scores_196, out_path):
    """Render one composite PNG: title + N frame thumbnails with heatmap overlay."""
    scores = scores_196.detach().float().cpu().numpy().reshape(PATCH, PATCH)
    s_min, s_max = float(scores.min()), float(scores.max())
    s_norm = (scores - s_min) / max(s_max - s_min, 1e-6)
    topk = max(1, int(round(TOPK_RATIO * PATCH * PATCH)))
    flat = scores.reshape(-1)
    sel_idx = np.argsort(flat)[-topk:]
    sel_mask = np.zeros(PATCH * PATCH, dtype=bool)
    sel_mask[sel_idx] = True
    sel_mask = sel_mask.reshape(PATCH, PATCH)

    paths = item["traj_frame_paths"]
    n_show = min(FRAMES_PER_ITEM, len(paths))
    pick = [int(i * (len(paths) - 1) / max(1, n_show - 1)) for i in range(n_show)]
    pick_paths = [paths[i] for i in pick]

    cell = COORD / PATCH
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.4))
    if n_show == 1:
        axes = [axes]
    for ax, p in zip(axes, pick_paths):
        try:
            img = Image.open(p).convert("RGB").resize((COORD, COORD))
        except Exception:
            img = Image.new("RGB", (COORD, COORD), "black")
        ax.imshow(img)
        ax.imshow(s_norm, cmap="jet", alpha=0.45,
                  extent=(0, COORD, COORD, 0), interpolation="nearest")
        for r in range(PATCH):
            for c in range(PATCH):
                if sel_mask[r, c]:
                    ax.add_patch(Rectangle((c * cell, r * cell), cell, cell,
                                           fill=False, edgecolor="lime", linewidth=1.4))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(os.path.basename(p), fontsize=7)

    q = (item.get("question", "") or "").replace("\n", " ")
    if len(q) > 140:
        q = q[:137] + "…"
    fig.suptitle(
        f"[{item['dataset']} / {item['task']}]  answer={item['answer']}  "
        f"opts={len(item['options'])}  selected={topk}/{PATCH*PATCH} "
        f"({100*TOPK_RATIO:.0f}%)\nQ: {q}",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading TrajGazeV2 encoder from {args.ckpt}", flush=True)
    enc = TrajGazeV2().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    enc.load_state_dict(ck["encoder_state"], strict=False)
    enc.eval()

    ds = CombinedMergeDataset(split=args.split,
                              n_vlm_frames=args.n_vlm_frames,
                              n_traj_frames=args.n_traj_frames)

    # pick first N working items per requested source
    picked = {s: [] for s in args.sources}
    for idx in range(len(ds)):
        src, _ = ds.items[idx]
        if src not in picked or len(picked[src]) >= args.n_per_source:
            continue
        if all(len(v) >= args.n_per_source for v in picked.values()):
            break
        it = ds[idx]
        if it is None:
            continue
        picked[src].append((idx, it))

    n_total = sum(len(v) for v in picked.values())
    print(f"Rendering {n_total} items (split={args.split}) → {args.out}", flush=True)

    with torch.no_grad():
        for src in args.sources:
            for i, (idx, item) in enumerate(picked[src]):
                traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
                q_emb  = enc.query_encoder([item["question"]], device)
                v_feat = enc.visual_encoder([item["traj_frame_paths"]], device)
                scores, _ = enc.encoder(traj_batch, q_emb, v_feat)
                scores = scores.squeeze(0)   # (196,)
                fname = f"{src}_{i:02d}_idx{idx:04d}_{item['task']}.png"
                fname = fname.replace("/", "_")
                out_path = os.path.join(args.out, fname)
                render_one(item, scores, out_path)
                print(f"  saved {out_path}", flush=True)

    print(f"Done. {n_total} PNGs at {args.out}", flush=True)


if __name__ == "__main__":
    main()
