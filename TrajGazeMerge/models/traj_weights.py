"""Compute spatial + temporal weights from gaze/hand trajectory.

Used by VisionZip-traj to bias the dominant-pool top-K toward patches under the
human's gaze/hand (spatial), and toward frames with active manipulation
(temporal). Zero learnable params — pure deterministic functions of the
measured trajectory in `item["traj"]`.

Recipe 1+3 from the design discussion:
    Recipe 1 (spatial): Gaussian heatmap around gaze + L/R hand patch coords
    Recipe 3 (temporal): per-frame interaction score from
        fixation (low gaze velocity)
      + hand presence
      + gaze-on-hand (low gaze↔hand distance)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


SIGMA_GAZE     = 2.0      # patches: Gaussian width around gaze
SIGMA_HAND     = 3.0      # patches: Gaussian width around hand
ALPHA_HAND     = 0.7      # weight of hand prior relative to gaze
SIGMA_VEL      = 0.05     # normalized: gaze velocity threshold for fixation
SIGMA_GH_DIST  = 0.10     # normalized: gaze-on-hand decay
BETA_FIX       = 1.0
BETA_HAND_PRES = 1.0
BETA_GAZE_HAND = 1.0


def _pool_to_T(pos: torch.Tensor, mask: torch.Tensor, T_target: int):
    """Map (T_src, 2) positions + (T_src,) mask to (T_target, 2) + (T_target,)."""
    T_src = pos.shape[0]
    if T_src == T_target:
        return pos.float(), mask.bool()
    p = pos.t().unsqueeze(0).float()
    p_t = F.interpolate(p, size=T_target, mode="linear", align_corners=False)
    p_t = p_t.squeeze(0).t().contiguous()
    m = mask.float().view(1, 1, T_src)
    m_t = F.interpolate(m, size=T_target, mode="nearest").view(T_target) > 0.5
    return p_t, m_t


@torch.no_grad()
def compute_traj_weights(
    traj_dict: dict,
    T_merged: int,
    H: int,
    W: int,
    device: torch.device,
    sigma_g: float = SIGMA_GAZE,
    sigma_h: float = SIGMA_HAND,
    alpha_hand: float = ALPHA_HAND,
    sigma_v: float = SIGMA_VEL,
    sigma_gh: float = SIGMA_GH_DIST,
    beta_f: float = BETA_FIX,
    beta_h: float = BETA_HAND_PRES,
    beta_gh: float = BETA_GAZE_HAND,
):
    """Compute spatial_w (N=T*H*W,) and temporal_w (T_merged,) from trajectory.

    spatial_w[i]   in [0, 1+alpha_hand+alpha_hand] — Gaussian sum around anchors.
    temporal_w[t]  — softmax over frames * T_merged, so mean = 1.

    Both contain zero-fallbacks for invalid frames (no gaze tracker, hand off
    screen, etc.): the corresponding term contributes 0, leaving the rest intact.
    """
    gaze_pos,  gaze_mask  = _pool_to_T(traj_dict["gaze_pos"],  traj_dict["gaze_mask"],  T_merged)
    left_pos,  left_mask  = _pool_to_T(traj_dict["left_pos"],  traj_dict["left_mask"],  T_merged)
    right_pos, right_mask = _pool_to_T(traj_dict["right_pos"], traj_dict["right_mask"], T_merged)

    gaze_pos  = gaze_pos.to(device);  left_pos  = left_pos.to(device);  right_pos = right_pos.to(device)
    gaze_mask = gaze_mask.to(device).float()
    left_mask = left_mask.to(device).float()
    right_mask = right_mask.to(device).float()

    # ── Spatial weight: Gaussian heatmaps around gaze + L/R hand ──────────────
    py = torch.arange(H, device=device).float()
    px = torch.arange(W, device=device).float()
    py_grid, px_grid = torch.meshgrid(py, px, indexing="ij")
    py_grid = py_grid.reshape(-1)                    # (HW,)
    px_grid = px_grid.reshape(-1)

    # gaze/hand_pos normalized in [0,1]; project to patch coords
    gx = gaze_pos[:, 0] * W;  gy = gaze_pos[:, 1] * H
    lx = left_pos[:, 0] * W;  ly = left_pos[:, 1] * H
    rx = right_pos[:, 0] * W; ry = right_pos[:, 1] * H

    # squared distances (T_merged, HW)
    d_gaze = (px_grid.unsqueeze(0) - gx.unsqueeze(1))**2 + (py_grid.unsqueeze(0) - gy.unsqueeze(1))**2
    d_left = (px_grid.unsqueeze(0) - lx.unsqueeze(1))**2 + (py_grid.unsqueeze(0) - ly.unsqueeze(1))**2
    d_right = (px_grid.unsqueeze(0) - rx.unsqueeze(1))**2 + (py_grid.unsqueeze(0) - ry.unsqueeze(1))**2

    spatial_w = (
        gaze_mask.unsqueeze(1)  * torch.exp(-d_gaze  / (2 * sigma_g**2))
      + alpha_hand * left_mask.unsqueeze(1)  * torch.exp(-d_left  / (2 * sigma_h**2))
      + alpha_hand * right_mask.unsqueeze(1) * torch.exp(-d_right / (2 * sigma_h**2))
    )                                                # (T, HW)
    spatial_w = spatial_w.reshape(-1)                # (T*HW,)

    # ── Temporal weight: interaction score per frame ──────────────────────────
    gaze_speed = torch.zeros(T_merged, device=device)
    if T_merged > 1:
        d = (gaze_pos[1:] - gaze_pos[:-1]).norm(dim=-1)
        valid = gaze_mask[1:] * gaze_mask[:-1]
        gaze_speed[1:] = d * valid
    fixation = 1.0 / (1.0 + gaze_speed / sigma_v)    # high when gaze fixated

    hand_present = 0.5 * (left_mask + right_mask)    # 0 / 0.5 / 1.0

    # gaze on nearer hand
    d_gh_l = (gaze_pos - left_pos).norm(dim=-1)
    d_gh_r = (gaze_pos - right_pos).norm(dim=-1)
    BIG = torch.full_like(d_gh_l, 1e6)
    d_gh = torch.minimum(
        torch.where(left_mask > 0, d_gh_l, BIG),
        torch.where(right_mask > 0, d_gh_r, BIG),
    )
    any_hand = ((left_mask + right_mask) > 0).float()
    gaze_on_hand = torch.exp(-d_gh / sigma_gh) * gaze_mask * any_hand

    interaction = beta_f * fixation + beta_h * hand_present + beta_gh * gaze_on_hand
    temporal_w = F.softmax(interaction, dim=0) * T_merged   # mean 1

    return spatial_w, temporal_w


def _solve_spatial_dims(S_per_frame: int, H_grid: int, W_grid: int) -> tuple[int, int]:
    """Find (s_h, s_w) such that s_h*s_w == S_per_frame and s_h/s_w ~= H_grid/W_grid.

    VisionZip's video_embeds count is N = T_grid * S_per_frame, but S_per_frame
    may differ from H_grid*W_grid because Qwen's grid_thw is the post-spatial-
    merge size while video_embeds is pre-merge. We just need the per-frame
    spatial layout, preserving the grid_thw aspect ratio.
    """
    target_ratio = H_grid / max(1, W_grid)
    best = (1, S_per_frame, float("inf"))
    for h in range(1, S_per_frame + 1):
        if S_per_frame % h == 0:
            w = S_per_frame // h
            diff = abs(h / w - target_ratio)
            if diff < best[2]:
                best = (h, w, diff)
    return best[0], best[1]


def traj_weighted_attn_scores(
    attn_scores: torch.Tensor,            # (N,)
    grid_thw: torch.Tensor,               # (1, 3): (T, H_grid, W_grid)
    traj_dict: dict,
    device: torch.device,
    **kw,
) -> torch.Tensor:
    """Return attn_scores * (1 + spatial_w) * temporal_w_per_token, shape (N,)."""
    N = attn_scores.shape[0]
    T = int(grid_thw[0, 0].item())
    H_grid = int(grid_thw[0, 1].item())
    W_grid = int(grid_thw[0, 2].item())

    S_per_frame = N // T
    s_h, s_w = _solve_spatial_dims(S_per_frame, H_grid, W_grid)

    spatial_w, temporal_w = compute_traj_weights(traj_dict, T, s_h, s_w, device, **kw)
    # spatial_w shape: (T * s_h * s_w,) == (N,) by construction
    temporal_w_per_token = temporal_w.repeat_interleave(s_h * s_w)

    return attn_scores * (1.0 + spatial_w) * temporal_w_per_token
