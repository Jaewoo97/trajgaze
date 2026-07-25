"""Anticipatory trajectory-tube token scores (parameter-free).

Method 2 of the complementary-selection study. Where VisionZip-traj reweights
attention by a Gaussian around the *current* gaze/hand position, this scorer
extrapolates each anchor forward by its per-frame velocity and lays a Gaussian
"tube" over BOTH the current and the predicted-future position. The premise:
in egocentric video the gaze leads the action, so the patch you are about to
fixate / manipulate is more task-relevant than the one you fixate now.

The returned (N,) score ranks every VisionZip video token by anticipatory
trajectory salience; it is used to pick a complementary token pool that is
unioned with VisionZip's content selection (see select_complementary).

Zero learnable params. Reuses _pool_to_T / _solve_spatial_dims from
traj_weights so the spatial layout matches VisionZip's video-token grid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from TrajGazeMerge.models.traj_weights import _pool_to_T, _solve_spatial_dims


@torch.no_grad()
def anticipatory_token_scores(
    traj_dict: dict,
    n_video: int,
    grid_thw: torch.Tensor,        # (1, 3): (T, H_grid, W_grid)
    device: torch.device,
    horizon: float = 2.0,          # frames ahead to extrapolate
    sigma_g: float = 2.0,
    sigma_h: float = 3.0,
    alpha_hand: float = 0.7,
    sigma_v: float = 0.05,
    sigma_gh: float = 0.10,
) -> torch.Tensor:
    """Return (N=n_video,) anticipatory spatiotemporal salience per video token."""
    T = int(grid_thw[0, 0].item())
    H_grid = int(grid_thw[0, 1].item())
    W_grid = int(grid_thw[0, 2].item())
    S_per_frame = n_video // max(1, T)
    H, W = _solve_spatial_dims(S_per_frame, H_grid, W_grid)

    gaze_pos,  gaze_mask  = _pool_to_T(traj_dict["gaze_pos"],  traj_dict["gaze_mask"],  T)
    left_pos,  left_mask  = _pool_to_T(traj_dict["left_pos"],  traj_dict["left_mask"],  T)
    right_pos, right_mask = _pool_to_T(traj_dict["right_pos"], traj_dict["right_mask"], T)
    gaze_pos  = gaze_pos.to(device);  left_pos  = left_pos.to(device);  right_pos = right_pos.to(device)
    gaze_mask = gaze_mask.to(device).float()
    left_mask = left_mask.to(device).float()
    right_mask = right_mask.to(device).float()

    def velocity(pos, mask):
        v = torch.zeros_like(pos)
        if T > 1:
            d = pos[1:] - pos[:-1]
            valid = (mask[1:] * mask[:-1]).unsqueeze(-1)
            v[1:] = d * valid
        return v

    gaze_fut  = (gaze_pos  + horizon * velocity(gaze_pos,  gaze_mask)).clamp(0, 1)
    left_fut  = (left_pos  + horizon * velocity(left_pos,  left_mask)).clamp(0, 1)
    right_fut = (right_pos + horizon * velocity(right_pos, right_mask)).clamp(0, 1)

    py = torch.arange(H, device=device).float()
    px = torch.arange(W, device=device).float()
    pyg, pxg = torch.meshgrid(py, px, indexing="ij")
    pyg = pyg.reshape(-1)
    pxg = pxg.reshape(-1)

    def gauss(pos, mask, sigma):
        x = pos[:, 0] * W
        y = pos[:, 1] * H
        d = (pxg.unsqueeze(0) - x.unsqueeze(1)) ** 2 + (pyg.unsqueeze(0) - y.unsqueeze(1)) ** 2
        return mask.unsqueeze(1) * torch.exp(-d / (2 * sigma ** 2))   # (T, HW)

    # tube = current ∪ anticipated anchors, for gaze + both hands
    spatial = (
        gauss(gaze_pos, gaze_mask, sigma_g) + gauss(gaze_fut, gaze_mask, sigma_g)
        + alpha_hand * (gauss(left_pos,  left_mask,  sigma_h) + gauss(left_fut,  left_mask,  sigma_h))
        + alpha_hand * (gauss(right_pos, right_mask, sigma_h) + gauss(right_fut, right_mask, sigma_h))
    )                                                # (T, HW)

    # temporal interaction weight (fixation + hand presence + gaze-on-hand)
    gaze_speed = torch.zeros(T, device=device)
    if T > 1:
        d = (gaze_pos[1:] - gaze_pos[:-1]).norm(dim=-1)
        valid = gaze_mask[1:] * gaze_mask[:-1]
        gaze_speed[1:] = d * valid
    fixation = 1.0 / (1.0 + gaze_speed / sigma_v)
    hand_present = 0.5 * (left_mask + right_mask)
    d_gh_l = (gaze_pos - left_pos).norm(dim=-1)
    d_gh_r = (gaze_pos - right_pos).norm(dim=-1)
    BIG = torch.full_like(d_gh_l, 1e6)
    d_gh = torch.minimum(
        torch.where(left_mask > 0, d_gh_l, BIG),
        torch.where(right_mask > 0, d_gh_r, BIG),
    )
    any_hand = ((left_mask + right_mask) > 0).float()
    gaze_on_hand = torch.exp(-d_gh / sigma_gh) * gaze_mask * any_hand
    interaction = fixation + hand_present + gaze_on_hand
    temporal_w = F.softmax(interaction, dim=0) * T   # (T,), mean 1

    score = spatial * temporal_w.unsqueeze(1)        # (T, HW)
    return score.reshape(-1)                         # (N,)
