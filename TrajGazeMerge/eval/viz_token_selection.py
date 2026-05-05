"""
Paper-quality qualitative visualization for TrajGaze.

Per sample:
  [Question + answer options at top]
  Repeating pairs of rows (default 6 frames per row, expand as needed):
    Row A  trajectory:        original frame + GT (solid line) and predicted
                              (dashed line) trajectory across the full clip.
                              Temporal alpha gradient — segments close in time
                              to the panel's frame are drawn dark; far segments
                              fade out.
    Row B  token selection:   same frame, with patch-level grid; selected
                              tokens stay opaque, dropped tokens are rendered
                              transparently (replaced by light-grey background).

Frame selection:
  Best `n_show // 2` future frames ranked by lowest pred-vs-GT L2, plus
  the rest evenly spaced across the past for context. Sorted by time.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/trajgaze \
    python -m TrajGazeMerge.eval.viz_token_selection \
        --ckpt  /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
        --out   /workspace/trajgaze/TrajGazeMerge/eval_results/viz_E1_keep10 \
        --cuda 0 --n-samples 20 --n-show 6 --n-per-row 6
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
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, "/workspace/trajgaze")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_traj_encoder(stage2_ckpt: str, device, n_vis_keyframes: int = 16):
    ckpt  = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state", ckpt.get("model_state_dict", ckpt))

    has_patch_temporal = any(
        k.startswith("encoder.patch_temporal_query")
        or k.startswith("encoder.patch_temporal_attn")
        or k.startswith("encoder.patch_temporal_head")
        for k in state
    )
    has_frame_score    = any(k.startswith("encoder.frame_attn_pool") for k in state)
    has_post_iframe    = any(k.startswith("encoder.inter_frame_post") for k in state)
    has_iframe_cond    = any(k.startswith("encoder.iframe_query_conditioner") for k in state)

    from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal
    model = TrajGazeV2Temporal(
        n_vis_keyframes=n_vis_keyframes,
        use_frame_score_branch=has_frame_score,
        use_post_fusion_iframe=has_post_iframe,
        use_patch_temporal_branch=has_patch_temporal,
        use_iframe_query_conditioning=has_iframe_cond,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[TrajEncoder] loaded from {stage2_ckpt}")
    print(f"  flags: patch_temporal={has_patch_temporal}, iframe_cond={has_iframe_cond}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    print(f"  ckpt: epoch={ckpt.get('epoch')} step={ckpt.get('step')} acc={ckpt.get('acc'):.2f}%")
    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Core forward: past encoder + trajectory decoder + score head
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_viz_outputs(traj_encoder, item: dict, device, past_ratio: float = 0.6):
    """
    Split traj frames into past (first past_ratio) / future (rest).
    Run encoder on past → patch scores (T_past, 196) + context.
    Run traj_decoder on context → predicted future trajectory (T_future, 6).
    Run score_head on context → patch_scores (T_past, 196).

    Returns dict:
        patch_scores  : (T_past, 196)  float32, [0,1]
        traj_pred     : (T_future, 6)  float32, [0,1]
                          dims: [gaze_x, gaze_y, left_x, left_y, right_x, right_y]
        past_paths    : list[str]  — frame paths for past
        future_paths  : list[str] — frame paths for future (for overlay)
        gt_past       : dict of arrays (T_past,) for gaze/hand GT
        gt_future     : dict of arrays (T_future,) for gaze/hand GT
        T_past, T_future
    """
    traj      = item["traj"]
    all_paths = item["traj_frame_paths"]
    T         = len(all_paths)
    T_past    = max(2, min(T - 2, round(T * past_ratio)))
    T_future  = T - T_past

    past_paths   = all_paths[:T_past]
    future_paths = all_paths[T_past:]

    # Slice traj tensors
    def slice_traj(d, start, end):
        return {k: v[start:end] for k, v in d.items()}

    past_traj = slice_traj(traj, 0, T_past)
    past_batch = {k: v.unsqueeze(0).to(device) for k, v in past_traj.items()}

    # Query embedding (zero — no query text for trajectory stage, matches stage1 train)
    query_emb = torch.zeros(1, traj_encoder.query_encoder.d_model, device=device)

    # Visual features for past frames
    visual_feat = traj_encoder.visual_encoder([past_paths], T_past, device)

    # Encoder
    _, context, _ = traj_encoder.encoder(past_batch, query_emb, visual_feat)
    # context: (1, T_past, 4, D)

    # Patch scores (TrajScoreHead)
    patch_scores = traj_encoder.score_head(context).squeeze(0)   # (T_past, 196)

    # Trajectory prediction
    traj_pred = traj_encoder.traj_decoder(context, T_future).squeeze(0)  # (T_future, 6)

    # GT arrays for past and future (numpy, normalized [0,1])
    def to_np(key):
        return traj[key].numpy()

    gt_past = {
        "gaze_pos":   to_np("gaze_pos")[:T_past],    # (T_past, 2)
        "gaze_mask":  to_np("gaze_mask")[:T_past],   # (T_past,)
        "left_pos":   to_np("left_pos")[:T_past],
        "left_mask":  to_np("left_mask")[:T_past],
        "right_pos":  to_np("right_pos")[:T_past],
        "right_mask": to_np("right_mask")[:T_past],
    }
    gt_future = {
        "gaze_pos":   to_np("gaze_pos")[T_past:T_past + T_future],
        "gaze_mask":  to_np("gaze_mask")[T_past:T_past + T_future],
        "left_pos":   to_np("left_pos")[T_past:T_past + T_future],
        "left_mask":  to_np("left_mask")[T_past:T_past + T_future],
        "right_pos":  to_np("right_pos")[T_past:T_past + T_future],
        "right_mask": to_np("right_mask")[T_past:T_past + T_future],
    }

    return {
        "patch_scores":  patch_scores.cpu().float().numpy(),  # (T_past, 196)
        "traj_pred":     traj_pred.cpu().float().numpy(),     # (T_future, 6)
        "past_paths":    past_paths,
        "future_paths":  future_paths,
        "gt_past":       gt_past,
        "gt_future":     gt_future,
        "T_past":        T_past,
        "T_future":      T_future,
    }


@torch.no_grad()
def get_token_selection(traj_encoder, item: dict, cached: dict, device, merge_ratio: float):
    """
    Run FULL traj sequence through score pipeline (as in Stage 2 eval).
    Returns:
        keep_mask_2d  : (T_merged, n_spatial) bool
        scores_2d_norm: (T_merged, n_spatial) float  — per-frame normalised scores
        side          : int                          — sqrt(n_spatial)
        scores_flat   : (n_video,) float             — raw merged scores
    """
    n_video   = cached["video_embeds"].shape[0]
    T_merged  = int(cached["grid_thw"][0, 0].item())
    n_spatial = n_video // max(1, T_merged)
    side      = int(round(n_spatial ** 0.5))
    r         = max(1, int(merge_ratio * n_video))

    # Full sequence score (for token selection — same as training pipeline)
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    scores_full = traj_encoder.get_patch_scores(
        traj_batch,
        queries=[item["question"]],
        frame_paths=[item["traj_frame_paths"]],
    ).squeeze(0)   # (T_traj, 196)

    T_traj = scores_full.shape[0]
    s2d    = scores_full.float().reshape(T_traj, 1, 14, 14)
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
    scores_flat = scores_spatial.reshape(-1)

    if scores_flat.shape[0] != n_video:
        if scores_flat.shape[0] > n_video:
            scores_flat = scores_flat[:n_video]
        else:
            scores_flat = scores_flat.repeat(
                (n_video + scores_flat.shape[0] - 1) // scores_flat.shape[0]
            )[:n_video]

    _, receiver_idx = gaze_weighted_merge(cached["video_embeds"], scores_flat, r)
    keep_flat = torch.zeros(n_video, dtype=torch.bool, device=device)
    keep_flat[receiver_idx] = True

    km = keep_flat.reshape(T_merged, n_spatial).cpu().numpy()

    sc = scores_spatial.detach().cpu().numpy()   # (T_traj or T_merged, n_spatial)
    if sc.shape[0] != T_merged:
        sc = np.repeat(sc[:1], T_merged, axis=0)
    sc_min  = sc.min(axis=1, keepdims=True)
    sc_max  = sc.max(axis=1, keepdims=True)
    sc_norm = (sc - sc_min) / np.maximum(sc_max - sc_min, 1e-6)

    return km, sc_norm, side, scores_flat


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

GAZE_COLOR  = "#2196F3"   # blue
LEFT_COLOR  = "#FF9800"   # orange
RIGHT_COLOR = "#9C27B0"   # purple — distinct from on-frame red gaze circle

_PAST_TINT   = "#E8F1FF"
_FUTURE_TINT = "#E8F8EC"


def _open_img(path: str):
    return np.asarray(Image.open(path).convert("RGB"))


def _draw_pred_forward(ax, pred_pos: np.ndarray, color: str,
                       current_t: int, T_past: int, T_total: int,
                       fade_distance: int = 25,
                       linewidth: float = 3.2, linestyle: str = "-",
                       marker: str = "o", marker_step: int = 1):
    """
    Draw the predicted trajectory line going FORWARD in time from `current_t`,
    PLUS a marker at every future prediction step (fading out).
    Alpha decays linearly from 1.0 (at current_t) to 0.0 (at +fade_distance).

    The markers make the trajectory visible even when consecutive predicted
    positions are close together in pixel space.
    """
    if not ax.get_images():
        return
    img_h, img_w = ax.get_images()[0].get_array().shape[:2]

    start_t   = max(current_t, T_past)
    start_idx = start_t - T_past
    n         = len(pred_pos)
    if start_idx >= n - 1:
        return

    # Collect visible prediction points (above alpha cutoff)
    pts = []  # (x, y, alpha)
    for i in range(start_idx, n):
        t_pt  = T_past + i
        d     = t_pt - current_t
        alpha = max(0.0, 1.0 - d / float(fade_distance))
        if alpha < 0.02:
            break
        x = pred_pos[i, 0] * img_w
        y = pred_pos[i, 1] * img_h
        pts.append((x, y, alpha))

    if len(pts) < 1:
        return

    # Connecting segments — alpha = mean of endpoint alphas
    for i in range(len(pts) - 1):
        x0, y0, a0 = pts[i]
        x1, y1, a1 = pts[i + 1]
        seg_alpha  = (a0 + a1) * 0.5
        ax.plot([x0, x1], [y0, y1], color=color, alpha=seg_alpha,
                linewidth=linewidth, linestyle=linestyle,
                solid_capstyle="round", zorder=4)

    # Marker at every prediction step (skip i=0 if there's already a marker
    # there; the caller draws the current-frame marker separately)
    for i in range(1, len(pts), marker_step):
        x, y, a = pts[i]
        size_px = 4.5 + 5.0 * a  # 4.5 → 9.5 px radius
        ax.scatter([x], [y], c="white", marker=marker,
                   s=(size_px + 2.0) ** 2, alpha=a,
                   linewidths=0, zorder=5)
        ax.scatter([x], [y], c=color, marker=marker,
                   s=size_px ** 2, alpha=a,
                   edgecolors="white", linewidths=0.7, zorder=6)


def _draw_traj_dot_at_t(ax, positions, masks, color: str,
                        current_t: int, T_offset: int,
                        marker: str = "o", size: int = 11,
                        filled: bool = True):
    """Mark the position at `current_t` (filled = GT, hollow = predicted)."""
    if not ax.get_images():
        return
    img_h, img_w = ax.get_images()[0].get_array().shape[:2]
    idx = current_t - T_offset
    if idx < 0 or idx >= len(positions) or not masks[idx]:
        return
    x = positions[idx, 0] * img_w
    y = positions[idx, 1] * img_h
    if filled:
        ax.scatter([x], [y], c="white", marker=marker, s=(size + 4) ** 2,
                   linewidths=0, zorder=6)
        ax.scatter([x], [y], c=color, marker=marker, s=size ** 2,
                   edgecolors="white", linewidths=1.2, zorder=7)
    else:
        ax.scatter([x], [y], facecolors="none", edgecolors=color,
                   marker=marker, s=size ** 2, linewidths=2.2, zorder=7)


def _draw_token_selection_transparent(
    ax, img_np, keep_grid: np.ndarray, side: int,
    drop_alpha: float = 0.18,
    bg_color=(0.96, 0.96, 0.96),
    grid_color: str = "white",
):
    """
    Selected tokens stay fully opaque. Dropped tokens are rendered
    semi-transparently (low alpha → blended with light bg) so the user can
    still see what was there but selected patches clearly stand out.
    """
    H, W = img_np.shape[:2]
    keep_up = np.asarray(
        Image.fromarray((keep_grid.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST),
        dtype=np.float32,
    ) / 255.0

    # per-pixel image alpha: 1.0 for kept, `drop_alpha` for dropped
    alpha = drop_alpha + (1.0 - drop_alpha) * keep_up

    img = img_np.astype(np.float32) / 255.0
    bg  = np.zeros_like(img)
    bg[..., 0] = bg_color[0]
    bg[..., 1] = bg_color[1]
    bg[..., 2] = bg_color[2]
    composite = bg * (1.0 - alpha[..., None]) + img * alpha[..., None]
    ax.imshow(np.clip(composite, 0, 1))

    # patch-grid lines
    cell_w = W / side
    cell_h = H / side
    for k in range(1, side):
        ax.axvline(k * cell_w, color=grid_color, linewidth=0.4, alpha=0.50, zorder=2)
        ax.axhline(k * cell_h, color=grid_color, linewidth=0.4, alpha=0.50, zorder=2)

    # subtle outline on kept patches for emphasis
    H_f = H; W_f = W
    for r in range(side):
        for c in range(side):
            if keep_grid[r, c]:
                rect = Rectangle(
                    (c * W_f / side, r * H_f / side),
                    W_f / side, H_f / side,
                    edgecolor="#00C853", facecolor="none",
                    linewidth=1.2, zorder=3,
                )
                ax.add_patch(rect)


def _select_frames(
    viz_out: dict,
    n_show: int,
    keep_mask_2d: np.ndarray = None,
    T_merged: int = None,
) -> list:
    """
    Uniformly sample `n_show` frames across the full timeline (T_total frames).
    Sorted in temporal order. Each panel then shows that frame's per-timestep
    state — token selection (bottom row) and trajectory markers (top row).
    The trajectory evolution emerges from comparing panels left → right.
    """
    T_past   = viz_out["T_past"]
    T_future = viz_out["T_future"]
    T_total  = T_past + T_future
    if T_total <= 0:
        return []
    if n_show >= T_total:
        return list(range(T_total))
    return np.linspace(0, T_total - 1, n_show, dtype=int).tolist()


def _frame_border(ax, color: str, lw: float = 2.0):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# Layout per sample:
#   [Title with question + options]
#   For each chunk of `n_per_row` selected frames:
#     Row A : original frame + GT trajectory (solid) + predicted trajectory
#             (dashed); temporal alpha gradient from `current_t`
#     Row B : same frame with token-selection overlay (kept = opaque,
#             dropped = transparent grey)
# ─────────────────────────────────────────────────────────────────────────────

def render_sample(
    out_path:     Path,
    item:         dict,
    viz_out:      dict,
    keep_mask_2d: np.ndarray,
    scores_2d:    np.ndarray,
    side:         int,
    pred_letter:  str,
    correct:      bool,
    n_show:       int = 6,
    n_per_row:    int = 6,
    T_merged:     int = 1,
    n_vlm:        int = 1,
):
    traj_pred    = viz_out["traj_pred"]      # (T_future, 6)
    past_paths   = viz_out["past_paths"]
    future_paths = viz_out["future_paths"]
    gt_past      = viz_out["gt_past"]
    gt_future    = viz_out["gt_future"]
    T_past       = viz_out["T_past"]
    T_future     = viz_out["T_future"]
    T_total      = T_past + T_future
    all_paths    = past_paths + future_paths

    # Full GT timeline (concat past + future)
    gt_full = {
        k: np.concatenate([gt_past[k], gt_future[k]], axis=0)
        for k in ("gaze_pos", "left_pos", "right_pos",
                  "gaze_mask", "left_mask", "right_mask")
    }

    selected = _select_frames(viz_out, n_show, keep_mask_2d, T_merged)
    if not selected:
        return

    n_chunks = (len(selected) + n_per_row - 1) // n_per_row
    n_rows   = n_chunks * 2
    n_cols   = min(n_per_row, len(selected))

    # Adaptive panel size — smaller cells when many frames per row.
    if n_cols >= 16:
        panel_size = 1.9
    elif n_cols >= 12:
        panel_size = 2.4
    elif n_cols >= 8:
        panel_size = 3.0
    else:
        panel_size = 4.0
    fig_w = panel_size * n_cols + 0.9
    fig_h = panel_size * n_rows + 2.4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.05, "hspace": 0.18},
    )

    # Scale marker geometry inversely with panel size so markers stay
    # visible on small cells (e.g. 16 frames/row → panel_size=1.9").
    marker_scale = max(1.0, 2.0 / panel_size)
    sz_gaze   = int(round(9 * marker_scale))    # gaze pred (○)
    sz_hand   = int(round(7 * marker_scale))    # hand GT/pred (◆/◇/■/□)
    lw_pred   = max(1.5, 2.0 * marker_scale)    # hollow-marker outline thickness
    halo_pad  = 3                                # white halo size = sz + halo_pad
    halo_lw   = max(1.0, 1.5 * marker_scale)
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Aggressive fade so only near-future predictions are visible
    fade_distance = max(8, T_future // 2)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_per_row
        end   = min(start + n_per_row, len(selected))
        chunk_frames = selected[start:end]

        traj_row  = chunk_idx * 2
        token_row = chunk_idx * 2 + 1

        if chunk_idx == 0:
            axes[traj_row, 0].annotate(
                "Trajectory\nat each frame\n(GT ●  pred ○)",
                xy=(-0.20, 0.5), xycoords="axes fraction",
                ha="center", va="center", rotation=90,
                fontsize=12, fontweight="bold",
            )
            axes[token_row, 0].annotate(
                "Token Selection\nkept opaque /\ndropped faded",
                xy=(-0.20, 0.5), xycoords="axes fraction",
                ha="center", va="center", rotation=90,
                fontsize=12, fontweight="bold",
            )

        for col, frame_t in enumerate(chunk_frames):
            ax_t = axes[traj_row,  col]
            ax_k = axes[token_row, col]
            img  = _open_img(all_paths[frame_t])

            # ── Trajectory panel ──────────────────────────────────────────
            # Per-frame markers only (no trails / no full trajectory line).
            # Trajectory evolution is read by scanning panels left → right.
            ax_t.imshow(img)
            H, W = img.shape[:2]

            # GT markers for HANDS only (gaze GT is already on-frame as
            # green dot + red circle, so we skip it). All GT: filled triangle (^).
            for kpos, kmask, color in [
                ("left_pos",  "left_mask",  LEFT_COLOR),
                ("right_pos", "right_mask", RIGHT_COLOR),
            ]:
                _draw_traj_dot_at_t(
                    ax_t, gt_full[kpos], gt_full[kmask], color,
                    current_t=frame_t, T_offset=0,
                    marker="^", size=sz_hand, filled=True,
                )

            # Predicted markers (hollow circle) for future frames — gaze, left, right
            if frame_t >= T_past:
                fi = frame_t - T_past
                for j, (color, size) in enumerate([
                    (GAZE_COLOR, sz_gaze),
                    (LEFT_COLOR, sz_hand),
                    (RIGHT_COLOR, sz_hand),
                ]):
                    marker = "o"
                    px = float(traj_pred[fi, j * 2])     * W
                    py = float(traj_pred[fi, j * 2 + 1]) * H
                    # thick white halo so the hollow marker pops on any background
                    ax_t.scatter([px], [py],
                                 facecolors="none", edgecolors="white",
                                 marker=marker, s=(size + halo_pad) ** 2,
                                 linewidths=halo_lw, zorder=6)
                    ax_t.scatter([px], [py],
                                 facecolors="none", edgecolors=color,
                                 marker=marker, s=size ** 2,
                                 linewidths=lw_pred, zorder=7)

            # Frame label + past/future tint border
            if frame_t < T_past:
                lbl  = f"t={frame_t}  (past)"
                tint = "#5C7AA8"
            else:
                lbl  = f"t={frame_t}  (fut +{frame_t - T_past})"
                tint = "#3D8B4F"
            ax_t.set_title(lbl, fontsize=11, color=tint, fontweight="bold", pad=4)
            _frame_border(ax_t, tint, lw=2.0)
            ax_t.set_xticks([]); ax_t.set_yticks([])

            # ── Token-selection panel ─────────────────────────────────────
            t_m = int(round(frame_t / max(1, T_total - 1) * (T_merged - 1)))
            t_m = min(max(0, t_m), T_merged - 1)
            keep_grid_2d = keep_mask_2d[t_m].reshape(side, side)
            _draw_token_selection_transparent(ax_k, img, keep_grid_2d, side)
            kept_n = int(keep_grid_2d.sum())
            ax_k.set_title(f"kept {kept_n}/{side*side}", fontsize=10, color="#444",
                           pad=4)
            _frame_border(ax_k, tint, lw=2.0)
            ax_k.set_xticks([]); ax_k.set_yticks([])

        # Hide trailing empty columns
        for col in range(len(chunk_frames), n_cols):
            axes[traj_row,  col].axis("off")
            axes[token_row, col].axis("off")

    # ── Title (question + GT/PRED text + options) ─────────────────────────
    status     = "✓" if correct else "✗"
    opts       = item.get("options", [])
    gt_letter  = item.get("answer", "?")
    gt_idx     = ord(gt_letter) - ord("A") if isinstance(gt_letter, str) and len(gt_letter) == 1 else -1
    gt_text    = opts[gt_idx] if 0 <= gt_idx < len(opts) else ""
    pred_idx_i = ord(pred_letter) - ord("A") if isinstance(pred_letter, str) and len(pred_letter) == 1 else -1
    pred_text  = opts[pred_idx_i] if 0 <= pred_idx_i < len(opts) else ""

    opt_str = "    ".join(f"({chr(65+i)}) {o[:32]}" for i, o in enumerate(opts))
    title = (
        f"{status}  GT=({gt_letter}) {gt_text[:60]}    "
        f"PRED=({pred_letter}) {pred_text[:60]}\n"
        f"Q: {item['question'][:140]}\n"
        f"{opt_str}"
    )
    fig.suptitle(title, fontsize=14, y=0.998, linespacing=1.45)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=GAZE_COLOR, linestyle="None",
               marker="o", markerfacecolor="none",
               markeredgewidth=2.5, markersize=11,
               label="Gaze pred (○)  [GT on-frame]"),
        Line2D([0], [0], color=LEFT_COLOR, linestyle="None",
               marker="^", markersize=9,
               label="Left hand GT (▲)"),
        Line2D([0], [0], color=LEFT_COLOR, linestyle="None",
               marker="o", markerfacecolor="none",
               markeredgewidth=2.2, markersize=10,
               label="Left hand pred (○)"),
        Line2D([0], [0], color=RIGHT_COLOR, linestyle="None",
               marker="^", markersize=9,
               label="Right hand GT (▲)"),
        Line2D([0], [0], color=RIGHT_COLOR, linestyle="None",
               marker="o", markerfacecolor="none",
               markeredgewidth=2.2, markersize=10,
               label="Right hand pred (○)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=5,
        fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=(0.025, 0.025, 1, 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Compact render  (1 row per sample)
#
# Layout:  [n_past_show past frames]  |  [present frame + GT & Pred overlay]
#                                        [present frame + token selection]
#
# The "present" panel overlays:
#   - GT future trajectory  : dashed line + filled triangle (▲), alpha fades with time
#   - Pred future trajectory: solid  line + hollow  circle  (○), alpha fades with time
#   Color: gaze=blue, left=orange, right=purple
# ─────────────────────────────────────────────────────────────────────────────

def _draw_future_traj(ax, positions, masks, color, marker, filled,
                      H, W, lw, fade_steps):
    """Draw a fading trajectory (line + markers) on ax for future frames."""
    valid = [(t, positions[t]) for t in range(len(positions))
             if t < len(masks) and masks[t]]
    if not valid:
        return
    T_tot = max(1, len(positions))
    # line segments
    for i in range(len(valid) - 1):
        t0, p0 = valid[i]
        t1, p1 = valid[i + 1]
        a = max(0.05, 1.0 - t0 / float(fade_steps))
        x0, y0 = p0[0] * W, p0[1] * H
        x1, y1 = p1[0] * W, p1[1] * H
        ls = "--" if not filled else "-"
        ax.plot([x0, x1], [y0, y1], color=color, alpha=a,
                linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=4)
    # markers
    for t, pos in valid:
        a = max(0.08, 1.0 - t / float(fade_steps))
        x, y = pos[0] * W, pos[1] * H
        sz = 10
        ax.scatter([x], [y], c="white", marker=marker,
                   s=(sz + 3) ** 2, alpha=a, linewidths=0, zorder=5)
        if filled:
            ax.scatter([x], [y], c=color, marker=marker,
                       s=sz ** 2, alpha=a, edgecolors="white",
                       linewidths=0.8, zorder=6)
        else:
            ax.scatter([x], [y], facecolors="none", edgecolors=color,
                       marker=marker, s=sz ** 2, alpha=a,
                       linewidths=1.8, zorder=6)


def _draw_pred_future_traj(ax, traj_pred, j, color, H, W, lw, fade_steps):
    """Draw predicted future trajectory for body-part index j (hollow circles)."""
    T_fut = len(traj_pred)
    pts = [(t, traj_pred[t, j * 2], traj_pred[t, j * 2 + 1]) for t in range(T_fut)]
    for i in range(len(pts) - 1):
        t0, x0, y0 = pts[i]
        t1, x1, y1 = pts[i + 1]
        a = max(0.05, 1.0 - t0 / float(fade_steps))
        ax.plot([x0 * W, x1 * W], [y0 * H, y1 * H],
                color=color, alpha=a, linewidth=lw,
                solid_capstyle="round", zorder=4)
    for t, x, y in pts:
        a = max(0.08, 1.0 - t / float(fade_steps))
        sz = 10
        ax.scatter([x * W], [y * H], c="white", marker="o",
                   s=(sz + 3) ** 2, alpha=a, linewidths=0, zorder=5)
        ax.scatter([x * W], [y * H], facecolors="none", edgecolors=color,
                   marker="o", s=sz ** 2, alpha=a,
                   linewidths=1.8, zorder=6)


def render_sample_compact(
    out_path:      Path,
    item:          dict,
    viz_out:       dict,
    keep_mask_2d:  np.ndarray,   # (T_merged, n_spatial)
    scores_2d:     np.ndarray,
    side:          int,
    pred_letter:   str,
    correct:       bool,
    n_past_show:   int = 4,
    T_merged:      int = 64,
    n_vlm:         int = 128,
):
    """
    1-row compact figure per sample.

    Columns:
      0 … n_past_show-1 : uniformly-sampled past frames (GT hand pos ▲)
      n_past_show        : present frame overlaid with GT future (▲ dashed)
                           and Pred future (○ solid), both fading over time
      n_past_show+1      : same present frame with token-selection overlay
    """
    T_past   = viz_out["T_past"]
    T_future = viz_out["T_future"]
    gt_past  = viz_out["gt_past"]
    gt_fut   = viz_out["gt_future"]
    traj_pred = viz_out["traj_pred"]        # (T_future, 6)
    past_paths   = viz_out["past_paths"]
    future_paths = viz_out["future_paths"]

    n_cols = n_past_show + 2
    panel_h = 3.2
    fig_w   = panel_h * n_cols * 1.05
    fig_h   = panel_h + 1.2    # +1.2 for title
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))

    fade_steps = max(1, T_future)

    # ── Past frames ──────────────────────────────────────────────────────────
    past_sel = np.linspace(0, T_past - 1, n_past_show, dtype=int).tolist() if T_past >= n_past_show else list(range(T_past))
    for col, t in enumerate(past_sel):
        ax = axes[col]
        img = _open_img(past_paths[t])
        H, W = img.shape[:2]
        ax.imshow(img)
        # GT hand markers at this past frame (filled triangle)
        for key_pos, key_mask, color in [
            ("left_pos",  "left_mask",  LEFT_COLOR),
            ("right_pos", "right_mask", RIGHT_COLOR),
        ]:
            if t < len(gt_past[key_mask]) and gt_past[key_mask][t]:
                x = gt_past[key_pos][t, 0] * W
                y = gt_past[key_pos][t, 1] * H
                ax.scatter([x], [y], c="white", marker="^", s=13 ** 2,
                           linewidths=0, zorder=5)
                ax.scatter([x], [y], c=color, marker="^", s=10 ** 2,
                           edgecolors="white", linewidths=0.8, zorder=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t}", fontsize=7, pad=2)
        _frame_border(ax, _PAST_TINT, lw=2.5)

    # ── Present frame: GT + Pred future overlay ───────────────────────────────
    ax_ov = axes[n_past_show]
    # Use last past frame as "present"
    present_path = past_paths[-1] if past_paths else future_paths[0]
    img = _open_img(present_path)
    H, W = img.shape[:2]
    ax_ov.imshow(img)

    lw_line = 1.8
    # GT future (dashed line + filled ▲)
    for key_pos, key_mask, color in [
        ("gaze_pos",  "gaze_mask",  GAZE_COLOR),
        ("left_pos",  "left_mask",  LEFT_COLOR),
        ("right_pos", "right_mask", RIGHT_COLOR),
    ]:
        _draw_future_traj(ax_ov, gt_fut[key_pos], gt_fut[key_mask],
                          color, "^", filled=True, H=H, W=W,
                          lw=lw_line, fade_steps=fade_steps)

    # Pred future (solid line + hollow ○)
    for j, color in enumerate([GAZE_COLOR, LEFT_COLOR, RIGHT_COLOR]):
        _draw_pred_future_traj(ax_ov, traj_pred, j, color, H, W,
                               lw=lw_line, fade_steps=fade_steps)

    ax_ov.set_xticks([]); ax_ov.set_yticks([])
    ax_ov.set_title("Present  (▲=GT  ○=Pred)", fontsize=7, pad=2)
    _frame_border(ax_ov, "#FFD54F", lw=3.0)   # yellow border = present

    # ── Score heatmap + Pred trajectory overlay ──────────────────────────────
    # Shows WHERE the model assigns high gaze scores (bright = high score)
    # overlaid with the predicted trajectory — they should spatially coincide.
    ax_tok = axes[n_past_show + 1]

    # Map present frame to T_merged index
    present_t_ratio = (T_past - 1) / max(1, T_past + T_future - 1)
    t_m = min(T_merged - 1, int(round(present_t_ratio * T_merged)))

    # Use score map nearest to present (average a small window for stability)
    t_lo = max(0, t_m - 2)
    t_hi = min(T_merged - 1, t_m + 2)
    score_grid = scores_2d[t_lo:t_hi + 1].mean(axis=0).reshape(side, side)  # (S,S) [0,1]

    score_up = np.asarray(
        Image.fromarray((score_grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    cmap_heat = plt.cm.get_cmap("hot")
    score_rgb = cmap_heat(score_up)[:, :, :3]   # (H, W, 3)
    img_float = img.astype(np.float32) / 255.0
    alpha_h   = 0.55
    blended   = img_float * (1 - alpha_h) + score_rgb * alpha_h
    ax_tok.imshow(np.clip(blended, 0, 1))

    # Draw predicted future trajectory on top (same as the overlay panel)
    for j, color in enumerate([GAZE_COLOR, LEFT_COLOR, RIGHT_COLOR]):
        _draw_pred_future_traj(ax_tok, traj_pred, j, color, H, W,
                               lw=lw_line, fade_steps=fade_steps)

    # Grid lines to show token granularity
    cell_w, cell_h = W / side, H / side
    for k in range(1, side):
        ax_tok.axvline(k * cell_w, color="white", linewidth=0.35, alpha=0.45, zorder=2)
        ax_tok.axhline(k * cell_h, color="white", linewidth=0.35, alpha=0.45, zorder=2)

    ax_tok.set_xticks([]); ax_tok.set_yticks([])
    ax_tok.set_title("Score heatmap + Pred traj", fontsize=7, pad=2)
    _frame_border(ax_tok, "#80CBC4", lw=2.5)

    # ── Title ─────────────────────────────────────────────────────────────────
    status   = "✓" if correct else "✗"
    gt_idx   = ["A", "B", "C", "D"].index(item["answer"])
    gt_text  = item["options"][gt_idx] if gt_idx < len(item["options"]) else ""
    pred_idx_n = ["A", "B", "C", "D"].index(pred_letter) if pred_letter in "ABCD" else 0
    pred_text  = item["options"][pred_idx_n] if pred_idx_n < len(item["options"]) else ""
    opt_str  = "  ".join(f"({l}) {item['options'][i][:35]}"
                         for i, l in enumerate("ABCD") if i < len(item["options"]))
    title = (
        f"{status}  GT=({item['answer']}) {gt_text[:55]}    PRED=({pred_letter}) {pred_text[:55]}\n"
        f"Q: {item['question'][:130]}\n"
        f"{opt_str}"
    )
    fig.suptitle(title, fontsize=9, y=1.01, linespacing=1.4)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=GAZE_COLOR, linestyle="--", linewidth=1.5,
               marker="^", markerfacecolor=GAZE_COLOR, markersize=7,
               label="Gaze GT (▲ --)"),
        Line2D([0], [0], color=GAZE_COLOR, linestyle="-", linewidth=1.5,
               marker="o", markerfacecolor="none", markeredgewidth=1.5, markersize=7,
               label="Gaze Pred (○ ─)"),
        Line2D([0], [0], color=LEFT_COLOR, linestyle="--", linewidth=1.5,
               marker="^", markerfacecolor=LEFT_COLOR, markersize=7,
               label="Left GT (▲ --)"),
        Line2D([0], [0], color=LEFT_COLOR, linestyle="-", linewidth=1.5,
               marker="o", markerfacecolor="none", markeredgewidth=1.5, markersize=7,
               label="Left Pred (○ ─)"),
        Line2D([0], [0], color=RIGHT_COLOR, linestyle="--", linewidth=1.5,
               marker="^", markerfacecolor=RIGHT_COLOR, markersize=7,
               label="Right GT (▲ --)"),
        Line2D([0], [0], color=RIGHT_COLOR, linestyle="-", linewidth=1.5,
               marker="o", markerfacecolor="none", markeredgewidth=1.5, markersize=7,
               label="Right Pred (○ ─)"),
        mpatches.Patch(facecolor="#FF4500", alpha=0.7, label="Score: high (bright)"),
        mpatches.Patch(facecolor="#1A1A1A", alpha=0.7, label="Score: low (dark)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=8,
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout(rect=(0, 0.0, 1, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

args_ref: dict = {}   # populated in main, used by render_sample


# ─────────────────────────────────────────────────────────────────────────────
# Patch-only render  (token selection grid, no trajectory markers)
#
# Same 128-frame grid as `full` mode, but each chunk has only ONE row
# (the token-selection row). No trajectory markers / no GT or Pred overlays.
# ─────────────────────────────────────────────────────────────────────────────

def render_sample_patch_only(
    out_path:     Path,
    item:         dict,
    viz_out:      dict,
    keep_mask_2d: np.ndarray,
    scores_2d:    np.ndarray,
    side:         int,
    pred_letter:  str,
    correct:      bool,
    n_show:       int = 128,
    n_per_row:    int = 16,
    T_merged:     int = 64,
    n_vlm:        int = 128,
):
    past_paths   = viz_out["past_paths"]
    future_paths = viz_out["future_paths"]
    T_past       = viz_out["T_past"]
    T_future     = viz_out["T_future"]
    T_total      = T_past + T_future
    all_paths    = past_paths + future_paths

    selected = _select_frames(viz_out, n_show, keep_mask_2d, T_merged)
    if not selected:
        return

    n_chunks = (len(selected) + n_per_row - 1) // n_per_row
    n_rows   = n_chunks
    n_cols   = min(n_per_row, len(selected))

    if n_cols >= 16:
        panel_size = 1.9
    elif n_cols >= 12:
        panel_size = 2.4
    elif n_cols >= 8:
        panel_size = 3.0
    else:
        panel_size = 4.0
    fig_w = panel_size * n_cols + 0.6
    fig_h = panel_size * n_rows + 1.6
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.05, "hspace": 0.20},
    )
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_per_row
        end   = min(start + n_per_row, len(selected))
        chunk_frames = selected[start:end]

        for col, frame_t in enumerate(chunk_frames):
            ax_k = axes[chunk_idx, col]
            img  = _open_img(all_paths[frame_t])

            t_m = int(round(frame_t / max(1, T_total - 1) * (T_merged - 1)))
            t_m = min(max(0, t_m), T_merged - 1)
            keep_grid_2d = keep_mask_2d[t_m].reshape(side, side)
            _draw_token_selection_transparent(ax_k, img, keep_grid_2d, side)

            if frame_t < T_past:
                lbl  = f"t={frame_t}  (past)"
                tint = "#5C7AA8"
            else:
                lbl  = f"t={frame_t}  (fut +{frame_t - T_past})"
                tint = "#3D8B4F"
            kept_n = int(keep_grid_2d.sum())
            ax_k.set_title(f"{lbl}   kept {kept_n}/{side*side}",
                           fontsize=9, color=tint, pad=3)
            _frame_border(ax_k, tint, lw=2.0)
            ax_k.set_xticks([]); ax_k.set_yticks([])

        for col in range(len(chunk_frames), n_cols):
            axes[chunk_idx, col].axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    status     = "✓" if correct else "✗"
    opts       = item.get("options", [])
    gt_letter  = item.get("answer", "?")
    gt_idx     = ord(gt_letter) - ord("A") if isinstance(gt_letter, str) and len(gt_letter) == 1 else -1
    gt_text    = opts[gt_idx] if 0 <= gt_idx < len(opts) else ""
    pred_idx_i = ord(pred_letter) - ord("A") if isinstance(pred_letter, str) and len(pred_letter) == 1 else -1
    pred_text  = opts[pred_idx_i] if 0 <= pred_idx_i < len(opts) else ""
    opt_str    = "    ".join(f"({chr(65+i)}) {o[:32]}" for i, o in enumerate(opts))
    title = (
        f"{status}  GT=({gt_letter}) {gt_text[:60]}    "
        f"PRED=({pred_letter}) {pred_text[:60]}\n"
        f"Q: {item['question'][:140]}\n"
        f"{opt_str}"
    )
    fig.suptitle(title, fontsize=14, y=0.998, linespacing=1.45)

    fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal token selection render  (1 row per sample)
#
# Layout:
#   [Bar chart: kept tokens per T_merged frame]  |  [top-4 high-kept frames]
#
# Shows: the model concentrates token budget on specific temporal keyframes
# rather than uniformly across all frames.
# ─────────────────────────────────────────────────────────────────────────────

def render_sample_temporal(
    out_path:      Path,
    item:          dict,
    viz_out:       dict,
    keep_mask_2d:  np.ndarray,   # (T_merged, n_spatial)
    scores_2d:     np.ndarray,
    side:          int,
    pred_letter:   str,
    correct:       bool,
    n_top_frames:  int = 4,
    T_merged:      int = 64,
    n_vlm:         int = 128,
):
    """
    Two-panel layout:
      Left  (wide) : bar chart of kept-token count per T_merged frame
      Right (×4)   : frames with highest kept-token count, with token-sel overlay

    Tells the story: the model focuses its token budget on specific moments,
    not uniformly across time.
    """
    T_past   = viz_out["T_past"]
    T_future = viz_out["T_future"]
    T_total  = T_past + T_future
    all_paths = viz_out["past_paths"] + viz_out["future_paths"]   # full clip

    kept_per_frame = keep_mask_2d.sum(axis=1)   # (T_merged,)
    top_tm_indices = np.argsort(kept_per_frame)[::-1][:n_top_frames]
    top_tm_indices = sorted(top_tm_indices.tolist())   # chronological order

    # ── Figure layout via GridSpec ────────────────────────────────────────────
    panel_h = 3.4
    n_right  = n_top_frames
    # bar chart takes 2× width of a frame panel
    width_ratios = [2.5] + [1.0] * n_right
    fig_w = panel_h * sum(width_ratios) * 0.95
    fig_h = panel_h + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(1, 1 + n_right, width_ratios=width_ratios,
                           wspace=0.06, left=0.04, right=0.98,
                           top=0.88, bottom=0.14)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 0])
    x = np.arange(T_merged)
    colors_bar = ["#42A5F5"] * T_merged
    for tm in top_tm_indices:
        colors_bar[tm] = "#FF7043"   # highlight top frames in orange-red

    ax_bar.bar(x, kept_per_frame, color=colors_bar, width=1.0, edgecolor="none")
    ax_bar.set_xlim(-0.5, T_merged - 0.5)
    ax_bar.set_ylim(0, kept_per_frame.max() * 1.25)
    ax_bar.set_xlabel("Frame index (T_merged)", fontsize=8)
    ax_bar.set_ylabel("# Kept tokens", fontsize=8)
    ax_bar.set_title("Kept tokens per frame  (orange = shown →)", fontsize=8, pad=4)
    ax_bar.tick_params(labelsize=7)

    # Vertical dashed lines at past/future boundary
    past_tm = int(round((T_past / max(1, T_total - 1)) * (T_merged - 1)))
    ax_bar.axvline(past_tm, color="#9C27B0", linewidth=1.2, linestyle="--", alpha=0.7,
                   label="past|future boundary")
    ax_bar.legend(fontsize=7, loc="upper right")

    # Annotate peak kept count
    peak_tm = int(np.argmax(kept_per_frame))
    ax_bar.annotate(
        f"peak={int(kept_per_frame[peak_tm])}",
        xy=(peak_tm, kept_per_frame[peak_tm]),
        xytext=(peak_tm + max(1, T_merged // 10), kept_per_frame[peak_tm] * 1.05),
        fontsize=7, color="#BF360C",
        arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.0),
    )

    stats_txt = (
        f"mean={kept_per_frame.mean():.1f}  "
        f"max={kept_per_frame.max()}  "
        f"min={kept_per_frame.min()}  "
        f"std={kept_per_frame.std():.1f}"
    )
    ax_bar.text(0.02, 0.97, stats_txt, transform=ax_bar.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # ── Top-kept frames with token-selection overlay ──────────────────────────
    for col, tm in enumerate(top_tm_indices):
        ax = fig.add_subplot(gs[0, col + 1])
        # Map T_merged index → clip frame path
        frame_ratio = tm / max(1, T_merged - 1)
        frame_idx   = min(len(all_paths) - 1, int(round(frame_ratio * (len(all_paths) - 1))))
        img = _open_img(all_paths[frame_idx])

        keep_grid = keep_mask_2d[tm].reshape(side, side)
        _draw_token_selection_transparent(ax, img, keep_grid, side, drop_alpha=0.12)

        kept_n  = int(kept_per_frame[tm])
        is_past = frame_idx < T_past
        region  = "past" if is_past else "future"
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={tm}  {kept_n} kept  ({region})", fontsize=7, pad=2,
                     color="#BF360C")
        _frame_border(ax, "#FF7043", lw=2.5)

    # ── Title ─────────────────────────────────────────────────────────────────
    status   = "✓" if correct else "✗"
    gt_idx   = ["A", "B", "C", "D"].index(item["answer"])
    gt_text  = item["options"][gt_idx] if gt_idx < len(item["options"]) else ""
    pred_idx_n = ["A", "B", "C", "D"].index(pred_letter) if pred_letter in "ABCD" else 0
    pred_text  = item["options"][pred_idx_n] if pred_idx_n < len(item["options"]) else ""
    opt_str   = "  ".join(f"({l}) {item['options'][i][:35]}"
                          for i, l in enumerate("ABCD") if i < len(item["options"]))
    title = (
        f"{status}  GT=({item['answer']}) {gt_text[:55]}    PRED=({pred_letter}) {pred_text[:55]}\n"
        f"Q: {item['question'][:130]}\n"
        f"{opt_str}"
    )
    fig.suptitle(title, fontsize=9, y=0.995, linespacing=1.4)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Gaze-token render  (1 row per sample)
#
# Layout:  [n_past_show past frames with token-sel + gaze GT highlight]
#          | [present frame + GT & Pred traj overlay]
#
# Each past panel:
#   - Token selection transparent overlay (kept=opaque, dropped=faded)
#   - Gaze GT position drawn as ★
#   - The 8×8 token cell containing gaze: GREEN border = kept, RED = dropped
#
# Claim being tested: "gaze-attended region's token is preferentially kept"
# ─────────────────────────────────────────────────────────────────────────────

def render_sample_gaze_token(
    out_path:      Path,
    item:          dict,
    viz_out:       dict,
    keep_mask_2d:  np.ndarray,   # (T_merged, n_spatial)
    scores_2d:     np.ndarray,
    side:          int,
    pred_letter:   str,
    correct:       bool,
    n_past_show:   int = 6,
    T_merged:      int = 64,
    n_vlm:         int = 128,
):
    T_past   = viz_out["T_past"]
    T_future = viz_out["T_future"]
    T_total  = T_past + T_future
    gt_past  = viz_out["gt_past"]
    gt_fut   = viz_out["gt_future"]
    traj_pred  = viz_out["traj_pred"]
    past_paths = viz_out["past_paths"]
    future_paths = viz_out["future_paths"]

    # Uniformly sample past frames that have valid gaze annotation
    all_past = list(range(T_past))
    gaze_valid = [t for t in all_past if gt_past["gaze_mask"][t]]
    # Prefer frames with gaze, fallback to all past if not enough
    pool = gaze_valid if len(gaze_valid) >= n_past_show else all_past
    if len(pool) >= n_past_show:
        past_sel = np.linspace(0, len(pool) - 1, n_past_show, dtype=int)
        past_sel = [pool[i] for i in past_sel]
    else:
        past_sel = pool

    n_cols  = len(past_sel) + 1   # past panels + present overlay
    panel_h = 3.4
    fig_w   = panel_h * n_cols * 1.08
    fig_h   = panel_h + 1.2
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        axes = [axes]

    kept_count  = 0   # across all shown panels with valid gaze
    total_count = 0

    # ── Past panels: token selection + gaze GT highlight ─────────────────────
    for col, t in enumerate(past_sel):
        ax = axes[col]
        img = _open_img(past_paths[t])
        H, W = img.shape[:2]

        # Map past frame t → T_merged index
        t_m = min(T_merged - 1, int(round(t / max(1, T_total - 1) * (T_merged - 1))))
        keep_grid = keep_mask_2d[t_m].reshape(side, side)

        # Token selection overlay
        _draw_token_selection_transparent(ax, img, keep_grid, side,
                                          drop_alpha=0.15)

        has_gaze = bool(gt_past["gaze_mask"][t])
        if has_gaze:
            gx = float(gt_past["gaze_pos"][t, 0])
            gy = float(gt_past["gaze_pos"][t, 1])
            # Token cell containing gaze
            gc = min(side - 1, max(0, int(gx * side)))
            gr = min(side - 1, max(0, int(gy * side)))
            token_kept = bool(keep_grid[gr, gc])

            # Highlight the gaze token cell
            cell_w = W / side
            cell_h_px = H / side
            border_color = "#00E676" if token_kept else "#FF1744"  # green / red
            rect = Rectangle(
                (gc * cell_w, gr * cell_h_px), cell_w, cell_h_px,
                edgecolor=border_color, facecolor=border_color,
                alpha=0.35, linewidth=0, zorder=3,
            )
            ax.add_patch(rect)
            rect2 = Rectangle(
                (gc * cell_w, gr * cell_h_px), cell_w, cell_h_px,
                edgecolor=border_color, facecolor="none",
                linewidth=2.5, zorder=4,
            )
            ax.add_patch(rect2)

            # Gaze GT star marker
            px, py = gx * W, gy * H
            ax.scatter([px], [py], c="white", marker="*", s=14 ** 2,
                       linewidths=0, zorder=6)
            ax.scatter([px], [py], c=GAZE_COLOR, marker="*", s=11 ** 2,
                       edgecolors="white", linewidths=0.8, zorder=7)

            kept_count  += int(token_kept)
            total_count += 1

        label = f"t={t}"
        if has_gaze:
            label += "  ✓kept" if token_kept else "  ✗drop"
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=7, pad=2,
                     color=("#00C853" if (has_gaze and token_kept)
                            else "#D50000" if (has_gaze and not token_kept)
                            else "gray"))
        _frame_border(ax, _PAST_TINT, lw=2.5)

    # ── Present frame: GT + Pred future overlay ───────────────────────────────
    ax_ov = axes[-1]
    present_path = past_paths[-1] if past_paths else future_paths[0]
    img = _open_img(present_path)
    H, W = img.shape[:2]
    ax_ov.imshow(img)

    fade_steps = max(1, T_future)
    lw_line = 1.8
    for key_pos, key_mask, color in [
        ("gaze_pos",  "gaze_mask",  GAZE_COLOR),
        ("left_pos",  "left_mask",  LEFT_COLOR),
        ("right_pos", "right_mask", RIGHT_COLOR),
    ]:
        _draw_future_traj(ax_ov, gt_fut[key_pos], gt_fut[key_mask],
                          color, "^", filled=True, H=H, W=W,
                          lw=lw_line, fade_steps=fade_steps)
    for j, color in enumerate([GAZE_COLOR, LEFT_COLOR, RIGHT_COLOR]):
        _draw_pred_future_traj(ax_ov, traj_pred, j, color, H, W,
                               lw=lw_line, fade_steps=fade_steps)

    kept_rate = f"{100*kept_count//total_count}%" if total_count else "N/A"
    ax_ov.set_xticks([]); ax_ov.set_yticks([])
    ax_ov.set_title("Present  (▲=GT  ○=Pred)", fontsize=7, pad=2)
    _frame_border(ax_ov, "#FFD54F", lw=3.0)

    # ── Title ─────────────────────────────────────────────────────────────────
    status   = "✓" if correct else "✗"
    gt_idx   = ["A", "B", "C", "D"].index(item["answer"])
    gt_text  = item["options"][gt_idx] if gt_idx < len(item["options"]) else ""
    pred_idx_n = ["A", "B", "C", "D"].index(pred_letter) if pred_letter in "ABCD" else 0
    pred_text  = item["options"][pred_idx_n] if pred_idx_n < len(item["options"]) else ""
    opt_str  = "  ".join(f"({l}) {item['options'][i][:35]}"
                         for i, l in enumerate("ABCD") if i < len(item["options"]))
    title = (
        f"{status}  GT=({item['answer']}) {gt_text[:55]}    PRED=({pred_letter}) {pred_text[:55]}"
        f"    [gaze token kept: {kept_count}/{total_count} = {kept_rate}]\n"
        f"Q: {item['question'][:130]}\n"
        f"{opt_str}"
    )
    fig.suptitle(title, fontsize=9, y=1.01, linespacing=1.4)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=GAZE_COLOR, marker="*", markersize=9,
               linestyle="None", markerfacecolor=GAZE_COLOR, label="Gaze GT (★)"),
        mpatches.Patch(facecolor="#00E676", alpha=0.6, label="Gaze token: KEPT"),
        mpatches.Patch(facecolor="#FF1744", alpha=0.6, label="Gaze token: DROPPED"),
        mpatches.Patch(facecolor="#00C853", alpha=0.8, label="Token kept (opaque)"),
        mpatches.Patch(facecolor="#BBBBBB", alpha=0.5, label="Token dropped (faded)"),
        Line2D([0], [0], color=GAZE_COLOR, linestyle="--", linewidth=1.5,
               marker="^", markerfacecolor=GAZE_COLOR, markersize=7,
               label="Gaze GT future (▲--)"),
        Line2D([0], [0], color=GAZE_COLOR, linestyle="-", linewidth=1.5,
               marker="o", markerfacecolor="none", markeredgewidth=1.5, markersize=7,
               label="Gaze Pred future (○─)"),
        Line2D([0], [0], color=LEFT_COLOR, linestyle="-", linewidth=1.5,
               marker="o", markerfacecolor="none", markeredgewidth=1.5, markersize=7,
               label="Left Pred (○─)"),
        Line2D([0], [0], color=RIGHT_COLOR, linestyle="-", linewidth=1.5,
               marker="o", markerfacecolor="none", markeredgewidth=1.5, markersize=7,
               label="Right Pred (○─)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=9,
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout(rect=(0, 0.0, 1, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return kept_count, total_count


def _traj_score_corr(traj_pred: np.ndarray, scores_2d: np.ndarray,
                     t_m: int, side: int) -> float:
    """
    Measure how well the predicted future trajectory aligns with high-score tokens.

    For each predicted future position (gaze / left / right), look up the
    normalised patch score at that spatial location.  Return the ratio of the
    mean score AT predicted positions vs the global mean score for that frame.

    ratio > 1  → model assigns high score where it predicts trajectory to go
    ratio ≈ 1  → no correlation
    ratio < 1  → prediction lands in low-score regions
    """
    t_lo = max(0, t_m - 2)
    t_hi = min(scores_2d.shape[0] - 1, t_m + 2)
    score_row = scores_2d[t_lo:t_hi + 1].mean(axis=0)  # (n_spatial,)
    mean_s = float(score_row.mean())
    if mean_s < 1e-6:
        return 1.0

    vals = []
    T_fut = len(traj_pred)
    for t in range(T_fut):
        for j in range(3):   # gaze, left, right
            x = float(traj_pred[t, j * 2])
            y = float(traj_pred[t, j * 2 + 1])
            r = min(side - 1, max(0, int(y * side)))
            c = min(side - 1, max(0, int(x * side)))
            vals.append(score_row[r * side + c])

    return float(np.mean(vals) / mean_s) if vals else 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--out",           default="/workspace/trajgaze/TrajGazeMerge/eval_results")
    p.add_argument("--split",         choices=["train", "test"], default="test",
                   help="'train' = egoexolearn+holoassist, 'test' = egtea")
    p.add_argument("--n-samples",     type=int,   default=20)
    p.add_argument("--mode",
                   choices=["full", "compact", "gaze_token", "temporal", "patch_only"],
                   default="compact",
                   help="'compact'    = 1 row: past frames + present GT/Pred overlay; "
                        "'temporal'   = bar chart of kept/frame + top-kept frame panels; "
                        "'gaze_token' = token-sel + gaze GT highlight per past frame; "
                        "'patch_only' = 128-frame grid, token selection only (no traj); "
                        "'full'       = 128-frame grid (traj + token selection)")
    p.add_argument("--n-past-show",   type=int,   default=4,
                   help="[compact mode] number of past context frames to show")
    p.add_argument("--n-show",        type=int,   default=128,
                   help="[full mode] Total frames to display per figure.")
    p.add_argument("--n-per-row",     type=int,   default=16,
                   help="[full mode] Frames per row.")
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--past-ratio",    type=float, default=0.6,
                   help="Fraction of clip used as past for trajectory viz")
    p.add_argument("--n-vlm-frames",  type=int,   default=128)
    p.add_argument("--n-traj-frames", type=int,   default=128)
    p.add_argument("--n-vis-kf",      type=int,   default=16)
    p.add_argument("--task-filter",   type=str,   default=None)
    p.add_argument("--load-lora",     action="store_true",
                   help="Load LoRA weights for faithful QA prediction")
    p.add_argument("--cuda",          type=int,   default=0,
                   help="GPU index to use")
    p.add_argument("--seed",          type=int,   default=0)
    return p.parse_args()


def main():
    global args_ref
    args    = parse_args()
    args_ref = vars(args)
    device  = torch.device(f"cuda:{args.cuda}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] split={args.split}  gpu=cuda:{args.cuda}  n_samples={args.n_samples}", flush=True)

    print("Loading Qwen2.5-VL ...", flush=True)
    processor, qwen_model = load_qwen_lora(device)
    base_qwen  = qwen_model.get_base_model()
    option_ids = get_option_ids(processor)

    if args.load_lora:
        print("Loading LoRA weights ...", flush=True)
        ckpt_full  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        miss, unex = qwen_model.load_state_dict(ckpt_full.get("lora_state", {}), strict=False)
        print(f"  LoRA: missing={len(miss)} unexpected={len(unex)}")
        del ckpt_full

    print("Loading TrajGaze encoder ...", flush=True)
    traj_encoder = load_traj_encoder(args.ckpt, device, args.n_vis_kf)

    print(f"Loading {args.split} dataset ...", flush=True)
    dataset = StreamGazeMergeDataset(
        split=args.split, n_vlm_frames=args.n_vlm_frames, n_traj_frames=args.n_traj_frames,
    )

    rng = np.random.RandomState(args.seed)
    if args.task_filter:
        wanted = set(args.task_filter.split(","))
        cands  = [i for i in range(len(dataset)) if dataset.items[i]["task"] in wanted]
    else:
        cands = list(range(len(dataset)))
    rng.shuffle(cands)

    summary = []
    n_done  = 0
    qwen_model.eval()

    for idx in cands:
        if n_done >= args.n_samples:
            break
        item = dataset[idx]
        if item is None:
            continue
        try:
            # ── Qwen preprocessing (for token selection + QA) ─────────────
            cached = preprocess_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device,
            )
            if cached is None:
                continue

            n_video  = cached["video_embeds"].shape[0]
            T_merged = int(cached["grid_thw"][0, 0].item())
            n_spatial = n_video // max(1, T_merged)
            side      = int(round(n_spatial ** 0.5))

            # ── Token selection (full-sequence, mirrors Stage 2 eval) ──────
            keep_mask_2d, scores_2d, side, scores_flat = get_token_selection(
                traj_encoder, item, cached, device, args.merge_ratio,
            )

            # ── Trajectory prediction (past/future split) ──────────────────
            viz_out = get_viz_outputs(traj_encoder, item, device, args.past_ratio)

            # ── QA prediction ──────────────────────────────────────────────
            r = max(1, int(args.merge_ratio * n_video))
            merged_video, recv_idx = gaze_weighted_merge(cached["video_embeds"], scores_flat, r)
            with torch.no_grad():
                logits = forward_logits(
                    qwen_model, build_merged_inputs(base_qwen, cached, merged_video, recv_idx)
                )
            pred_idx    = int(logits[option_ids].argmax().item())
            pred_letter = ["A", "B", "C", "D"][pred_idx]
            correct     = (pred_letter == item["answer"])

            # ── Render ─────────────────────────────────────────────────────
            tag = f"sample_{n_done:03d}_{item['task']}_{'OK' if correct else 'WRONG'}"
            out_path = out_dir / f"{tag}.png"
            if args.mode == "compact":
                render_sample_compact(
                    out_path, item, viz_out,
                    keep_mask_2d, scores_2d, side,
                    pred_letter, correct,
                    n_past_show=args.n_past_show,
                    T_merged=T_merged, n_vlm=len(item["vlm_frame_paths"]),
                )
            elif args.mode == "gaze_token":
                render_sample_gaze_token(
                    out_path, item, viz_out,
                    keep_mask_2d, scores_2d, side,
                    pred_letter, correct,
                    n_past_show=args.n_past_show,
                    T_merged=T_merged, n_vlm=len(item["vlm_frame_paths"]),
                )
            elif args.mode == "temporal":
                render_sample_temporal(
                    out_path, item, viz_out,
                    keep_mask_2d, scores_2d, side,
                    pred_letter, correct,
                    n_top_frames=args.n_past_show,
                    T_merged=T_merged, n_vlm=len(item["vlm_frame_paths"]),
                )
            elif args.mode == "patch_only":
                render_sample_patch_only(
                    out_path, item, viz_out,
                    keep_mask_2d, scores_2d, side,
                    pred_letter, correct,
                    n_show=args.n_show, n_per_row=args.n_per_row,
                    T_merged=T_merged, n_vlm=len(item["vlm_frame_paths"]),
                )
            else:
                render_sample(
                    out_path, item, viz_out,
                    keep_mask_2d, scores_2d, side,
                    pred_letter, correct,
                    n_show=args.n_show, n_per_row=args.n_per_row,
                    T_merged=T_merged, n_vlm=len(item["vlm_frame_paths"]),
                )

            # Trajectory error (simple L2 at each future step)
            tp  = viz_out["traj_pred"]                       # (T_future, 6)
            gf  = viz_out["gt_future"]
            gaze_err  = np.linalg.norm(
                tp[:, :2] - gf["gaze_pos"], axis=1
            )[gf["gaze_mask"]].mean() if gf["gaze_mask"].any() else float("nan")
            left_err  = np.linalg.norm(
                tp[:, 2:4] - gf["left_pos"], axis=1
            )[gf["left_mask"]].mean() if gf["left_mask"].any() else float("nan")
            right_err = np.linalg.norm(
                tp[:, 4:6] - gf["right_pos"], axis=1
            )[gf["right_mask"]].mean() if gf["right_mask"].any() else float("nan")

            # Traj–score correlation: ratio of mean score at predicted positions vs overall mean
            present_t_ratio = (viz_out["T_past"] - 1) / max(1, viz_out["T_past"] + viz_out["T_future"] - 1)
            t_m_corr = min(T_merged - 1, int(round(present_t_ratio * T_merged)))
            traj_corr = _traj_score_corr(tp, scores_2d, t_m_corr, side)

            entry = {
                "idx":       int(idx),
                "task":      item["task"],
                "dataset":   item["dataset"],
                "answer":    item["answer"],
                "pred":      pred_letter,
                "correct":   bool(correct),
                "T_past":    viz_out["T_past"],
                "T_future":  viz_out["T_future"],
                "T_merged":  int(T_merged),
                "n_spatial": int(n_spatial),
                "kept_pct":  float(100.0 * keep_mask_2d.mean()),
                "gaze_L2":   float(gaze_err),
                "left_L2":   float(left_err),
                "right_L2":  float(right_err),
                "traj_corr": float(traj_corr),
                "fig":       out_path.name,
            }
            summary.append(entry)
            n_done += 1
            print(
                f"[{n_done}/{args.n_samples}] {item['task']} "
                f"{'✓' if correct else '✗'}  gaze_L2={gaze_err:.3f}  "
                f"corr={traj_corr:.2f}  → {out_path.name}",
                flush=True,
            )

        except Exception as e:
            import traceback
            print(f"  skip idx={idx}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"args": vars(args), "samples": summary}, f, indent=2)

    if summary:
        g_errs = [s["gaze_L2"]  for s in summary if not np.isnan(s["gaze_L2"])]
        l_errs = [s["left_L2"]  for s in summary if not np.isnan(s["left_L2"])]
        r_errs = [s["right_L2"] for s in summary if not np.isnan(s["right_L2"])]
        acc    = 100.0 * sum(s["correct"] for s in summary) / len(summary)
        print(f"\n── Summary ({n_done} samples) ──────────────────")
        print(f"  QA acc:     {acc:.1f}%")
        print(f"  Gaze L2:    {np.mean(g_errs):.4f}")
        print(f"  Left L2:    {np.mean(l_errs):.4f}")
        print(f"  Right L2:   {np.mean(r_errs):.4f}")
        print(f"  Output dir: {out_dir}")

        # ── Top samples by traj-score correlation ──────────────────────────
        top = sorted(summary, key=lambda s: s["traj_corr"], reverse=True)[:20]
        print(f"\n── Top-20 by traj-score correlation (ratio ↑ = better) ──")
        for rank, s in enumerate(top, 1):
            print(
                f"  {rank:2d}. corr={s['traj_corr']:.3f}  "
                f"gaze_L2={s['gaze_L2']:.3f}  "
                f"{'✓' if s['correct'] else '✗'}  "
                f"{s['fig']}"
            )

        # Save ranked list separately for easy reference
        with open(out_dir / "top_corr.json", "w") as f:
            json.dump(top, f, indent=2)
        print(f"\n  Ranked list saved → {out_dir / 'top_corr.json'}")


if __name__ == "__main__":
    main()
