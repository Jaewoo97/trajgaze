"""
Visualize which visual tokens the best model (E1 patch-temporal, 68.44%) keeps
after gaze-weighted token merging.

Pipeline (mirrors evaluate() in train_merge_lora_temporal_no_kd.py):
    Qwen vision encoder        → video_embeds (n_video, d), grid_thw (T_merged)
    TrajGaze encoder           → patch_scores (T_traj, 196)
    score_to_qwen_spatiotemporal → scores_all (n_video,)
    gaze_weighted_merge        → receiver_idx (n_video - r,)  ← kept tokens

For each sample, we render a grid of K keyframes; per frame we show:
    [original]  [score heatmap overlay]  [kept-token mask overlay]

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.eval.viz_token_selection \
        --ckpt   /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
        --out    /workspace/trajgaze/TrajGazeMerge/eval_results/viz_E1_keep10 \
        --n-samples 20 \
        --n-keyframes 8 \
        --merge-ratio 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/workspace/trajgaze")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder load (mirrors load_traj_encoder in trainer; infers arch flags)
# ─────────────────────────────────────────────────────────────────────────────

def load_traj_encoder(stage2_ckpt: str, device, n_vis_keyframes: int = 16):
    ckpt  = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state", ckpt.get("model_state_dict", ckpt))

    has_frame_score = any(
        k.startswith("encoder.frame_attn_pool") or k.startswith("encoder.frame_score_head")
        for k in state
    )
    has_post_iframe = any(k.startswith("encoder.inter_frame_post") for k in state)
    has_patch_temporal = any(
        k.startswith("encoder.patch_temporal_query")
        or k.startswith("encoder.patch_temporal_attn")
        or k.startswith("encoder.patch_temporal_head")
        for k in state
    )
    has_iframe_query_cond = any(
        k.startswith("encoder.iframe_query_conditioner") for k in state
    )

    from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal
    model = TrajGazeV2Temporal(
        n_vis_keyframes=n_vis_keyframes,
        use_frame_score_branch=has_frame_score,
        use_post_fusion_iframe=has_post_iframe,
        use_patch_temporal_branch=has_patch_temporal,
        use_iframe_query_conditioning=has_iframe_query_cond,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[TrajEncoder] loaded from {stage2_ckpt}", flush=True)
    print(f"  flags: patch_temporal={has_patch_temporal}, iframe_query_cond={has_iframe_query_cond}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    print(f"  ckpt epoch={ckpt.get('epoch')} step={ckpt.get('step')} acc={ckpt.get('acc')}")
    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Score reshaping (mirrors get_patch_scores_temporal & score_to_qwen_spatiotemporal)
# ─────────────────────────────────────────────────────────────────────────────

def get_patch_scores_temporal(traj_encoder, item, device):
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    with torch.no_grad():
        scores = traj_encoder.get_patch_scores(
            traj_batch,
            queries     = [item["question"]],
            frame_paths = [item["traj_frame_paths"]],
        )
    return scores.squeeze(0)   # (T_traj, 196)


def score_to_qwen_spatiotemporal(scores, n_spatial, T_merged):
    T_traj = scores.shape[0]
    side   = int(n_spatial ** 0.5)
    s2d    = scores.float().reshape(T_traj, 1, 14, 14)
    if side == 8:
        s16 = F.interpolate(s2d, size=(16, 16), mode="nearest")
        s8  = F.avg_pool2d(s16, kernel_size=2, stride=2)
        scores_spatial = s8.reshape(T_traj, n_spatial)
    else:
        out = F.interpolate(s2d, size=(side, side), mode="bilinear", align_corners=False)
        scores_spatial = out.reshape(T_traj, n_spatial)
    if T_traj != T_merged:
        scores_spatial = F.interpolate(
            scores_spatial.T.unsqueeze(0).float(),
            size=T_merged, mode="linear", align_corners=False,
        ).squeeze(0).T
    return scores_spatial.reshape(-1), scores_spatial   # flat (T*n_sp,), 2d (T_merged, n_spatial)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def render_sample(
    out_path:        Path,
    item:            dict,
    keep_mask_2d:    np.ndarray,   # (T_merged, n_spatial) bool — kept (receiver) tokens
    score_2d:        np.ndarray,   # (T_merged, n_spatial) float in [0,1]
    n_keyframes:     int,
    side:            int,
    pred_letter:     str,
    correct:         bool,
):
    """
    Per-sample figure: K keyframes × 3 columns (orig | score heatmap | keep mask).
    """
    vlm_paths = item["vlm_frame_paths"]
    n_vlm     = len(vlm_paths)
    T_merged  = keep_mask_2d.shape[0]

    # K evenly-spaced keyframes; map each VLM frame index → merged time bucket
    frame_idxs = np.linspace(0, n_vlm - 1, n_keyframes, dtype=int)
    keep_pct   = 100.0 * keep_mask_2d.mean()

    fig, axes = plt.subplots(n_keyframes, 3, figsize=(9, 2.6 * n_keyframes))
    if n_keyframes == 1:
        axes = axes[None, :]

    for row, fi in enumerate(frame_idxs):
        # Map VLM frame → merged time bucket
        t_merged = int(round(fi / max(1, n_vlm - 1) * (T_merged - 1)))
        t_merged = min(max(0, t_merged), T_merged - 1)

        img = Image.open(vlm_paths[fi]).convert("RGB")
        img_np = np.asarray(img)
        H, W = img_np.shape[:2]

        score_grid = score_2d[t_merged].reshape(side, side)
        keep_grid  = keep_mask_2d[t_merged].reshape(side, side).astype(np.float32)

        # Resize to image dims (nearest for keep mask, bilinear for score)
        score_up = np.asarray(Image.fromarray((score_grid * 255).astype(np.uint8))
                              .resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        keep_up  = np.asarray(Image.fromarray((keep_grid * 255).astype(np.uint8))
                              .resize((W, H), Image.NEAREST), dtype=np.float32) / 255.0

        # Col 0: original
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"VLM frame #{fi} (t_merged={t_merged})", fontsize=8)
        axes[row, 0].axis("off")

        # Col 1: score heatmap overlay (jet colormap, 0.5 alpha)
        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(score_up, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[row, 1].set_title("TrajGaze score heatmap", fontsize=8)
        axes[row, 1].axis("off")

        # Col 2: keep mask — kept tokens visible, dropped tokens darkened
        kept_overlay = img_np.astype(np.float32) / 255.0
        # Dim non-kept regions
        dim_factor = 0.25 + 0.75 * keep_up[..., None]   # kept=1.0, dropped=0.25
        kept_overlay = kept_overlay * dim_factor
        # Outline kept cells with green grid
        axes[row, 2].imshow(np.clip(kept_overlay, 0, 1))
        for r in range(side):
            for c in range(side):
                if keep_grid[r, c]:
                    rect = Rectangle(
                        (c * W / side, r * H / side),
                        W / side, H / side,
                        edgecolor="lime", facecolor="none", linewidth=1.2,
                    )
                    axes[row, 2].add_patch(rect)
        axes[row, 2].set_title(f"Kept tokens ({int(keep_grid.sum())}/{side*side})", fontsize=8)
        axes[row, 2].axis("off")

    # Header
    qa = (
        f"Q: {item['question'][:120]}\n"
        f"options: A={item['options'][0][:30]} | B={item['options'][1][:30]} | "
        f"C={item['options'][2][:30]} | D={item['options'][3][:30]}\n"
        f"GT={item['answer']}  PRED={pred_letter}  "
        f"{'✓' if correct else '✗'}  |  task={item['task']}  "
        f"|  kept={keep_pct:.1f}%"
    )
    fig.suptitle(qa, fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True, help="Stage 2 best.pth (encoder_state inside)")
    p.add_argument("--out",           required=True, help="Output directory for PNG visualizations")
    p.add_argument("--n-samples",     type=int,   default=20, help="Number of test items to visualize")
    p.add_argument("--n-keyframes",   type=int,   default=8,  help="Keyframes per sample to show")
    p.add_argument("--merge-ratio",   type=float, default=0.9, help="Token merge ratio (0.9 = drop 90%)")
    p.add_argument("--n-vlm-frames",  type=int,   default=128)
    p.add_argument("--n-traj-frames", type=int,   default=128)
    p.add_argument("--task-filter",   type=str,   default=None,
                   help="Comma-separated task names; if set, only sample from these")
    p.add_argument("--load-lora",     action="store_true",
                   help="Also load LoRA weights from ckpt (faithful but heavier)")
    p.add_argument("--seed",          type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-VL ...", flush=True)
    processor, qwen_model = load_qwen_lora(device)
    base_qwen  = qwen_model.get_base_model()
    option_ids = get_option_ids(processor)

    if args.load_lora:
        print("Loading LoRA weights from ckpt ...", flush=True)
        ckpt_full  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        lora_state = ckpt_full.get("lora_state", {})
        miss, unex = qwen_model.load_state_dict(lora_state, strict=False)
        print(f"  lora load: missing={len(miss)} unexpected={len(unex)}")
        del ckpt_full

    print("Loading TrajGaze encoder ...", flush=True)
    traj_encoder = load_traj_encoder(args.ckpt, device)

    print("Loading test set ...", flush=True)
    test_ds = StreamGazeMergeDataset(
        split="test", n_vlm_frames=args.n_vlm_frames, n_traj_frames=args.n_traj_frames,
    )

    # Sample indices
    rng = np.random.RandomState(args.seed)
    if args.task_filter:
        wanted = set(args.task_filter.split(","))
        candidate_idxs = [i for i in range(len(test_ds)) if test_ds.items[i]["task"] in wanted]
    else:
        candidate_idxs = list(range(len(test_ds)))
    rng.shuffle(candidate_idxs)

    summary = []
    n_done = 0

    qwen_model.eval()
    traj_encoder.eval()

    for idx in candidate_idxs:
        if n_done >= args.n_samples:
            break
        item = test_ds[idx]
        if item is None:
            continue
        try:
            cached = preprocess_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device,
            )
            if cached is None:
                continue

            n_video   = cached["video_embeds"].shape[0]
            T_merged  = int(cached["grid_thw"][0, 0].item())
            n_spatial = n_video // max(1, T_merged)
            side      = int(round(n_spatial ** 0.5))
            r         = max(1, int(args.merge_ratio * n_video))

            scores      = get_patch_scores_temporal(traj_encoder, item, device)
            scores_flat, scores_2d = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
            if scores_flat.shape[0] != n_video:
                if scores_flat.shape[0] > n_video:
                    scores_flat = scores_flat[:n_video]
                    scores_2d   = scores_2d[:T_merged]   # best-effort
                else:
                    scores_flat = scores_flat.repeat(
                        (n_video + scores_flat.shape[0] - 1) // scores_flat.shape[0]
                    )[:n_video]

            _, receiver_idx = gaze_weighted_merge(cached["video_embeds"], scores_flat, r)

            # Build keep mask (n_video,) → (T_merged, n_spatial)
            keep_flat = torch.zeros(n_video, dtype=torch.bool, device=device)
            keep_flat[receiver_idx] = True
            keep_mask_2d = keep_flat.reshape(T_merged, n_spatial).cpu().numpy()

            # Normalise scores per-frame for nice heatmap (relative within-frame contrast)
            sc = scores_2d.detach().cpu().numpy()
            if sc.shape[0] != T_merged:
                # Defensive resample
                sc = np.broadcast_to(sc[:1], (T_merged, sc.shape[1])).copy()
            sc_min = sc.min(axis=1, keepdims=True)
            sc_max = sc.max(axis=1, keepdims=True)
            sc_norm = (sc - sc_min) / np.maximum(sc_max - sc_min, 1e-6)

            # Quick prediction (same path the trainer uses, so we can mark ✓/✗)
            from TrajGazeMerge.models.model import build_merged_inputs, forward_logits
            merged_video, recv_idx = gaze_weighted_merge(cached["video_embeds"], scores_flat, r)
            with torch.no_grad():
                logits = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, recv_idx)
                )
            pred_idx    = logits[option_ids].argmax().item()
            pred_letter = ["A", "B", "C", "D"][pred_idx]
            correct     = (pred_letter == item["answer"])

            sample_path = out_dir / f"sample_{n_done:03d}_{item['task']}_{'OK' if correct else 'WRONG'}.png"
            render_sample(
                sample_path, item, keep_mask_2d, sc_norm,
                n_keyframes=args.n_keyframes, side=side,
                pred_letter=pred_letter, correct=correct,
            )

            summary.append({
                "idx":         int(idx),
                "task":        item["task"],
                "dataset":     item["dataset"],
                "question":    item["question"],
                "options":     item["options"],
                "answer":      item["answer"],
                "pred":        pred_letter,
                "correct":     bool(correct),
                "n_video":     int(n_video),
                "T_merged":    int(T_merged),
                "n_spatial":   int(n_spatial),
                "side":        int(side),
                "r_dropped":   int(r),
                "kept_pct":    float(100.0 * keep_mask_2d.mean()),
                "fig":         str(sample_path.name),
            })
            n_done += 1
            print(f"[{n_done}/{args.n_samples}] {item['task']} {'OK' if correct else 'WRONG'} → {sample_path.name}", flush=True)

        except Exception as e:
            print(f"  skip idx={idx}: {type(e).__name__}: {e}", flush=True)
            continue

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"args": vars(args), "samples": summary}, f, indent=2)
    print(f"\nSaved {n_done} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
