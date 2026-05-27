"""
Trajectory-grounding utilities shared by:
  - Stage 1 encoder TAS (Trajectory-Anchored Scoring)
  - Stage 2 ATR (Auxiliary Trajectory Reconstruction)
  - Stage 2 CGM (Counterfactual Gaze-Mask) loss
  - Eval-side mask_gaze / mask_hand variants

All functions are pure / shape-tested. See `tests/test_trajectory_grounding.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# TAS — Gaussian patch maps over a 14×14 (or generic side) grid
# ----------------------------------------------------------------------

def gaussian_patch_map(
    positions: torch.Tensor,   # (B, T, K, 2) ∈ [0, 1]  (K anchors per frame, e.g. K=1 gaze, K=2 hand L+R)
    mask:      torch.Tensor,   # (B, T, K) bool — True where anchor is valid
    sigma:     torch.Tensor,   # () or (1,) scalar in *normalized* coords (e.g. 0.21 ~ 3 patch units / 14)
    grid:      int = 14,
) -> torch.Tensor:
    """
    Build per-frame Gaussian maps centered on anchor positions.

    Returns:
        scores : (B, T, grid*grid) in (0, 1]  (peak = 1 at the anchor centre,
                 0 for frames with no valid anchor)
    """
    B, T, K, _ = positions.shape
    device = positions.device

    # Patch centres in normalized coords
    ax = (torch.arange(grid, device=device, dtype=positions.dtype) + 0.5) / grid  # (grid,)
    ay = ax
    # (grid, grid, 2)
    grid_xy = torch.stack(torch.meshgrid(ax, ay, indexing="xy"), dim=-1)
    grid_xy = grid_xy.reshape(-1, 2)                                # (grid*grid, 2)

    # (B, T, K, 1, 2) − (1, 1, 1, grid*grid, 2) → (B, T, K, grid*grid, 2)
    diff  = positions.unsqueeze(-2) - grid_xy.view(1, 1, 1, -1, 2)
    dist2 = (diff * diff).sum(-1)                                    # (B, T, K, grid*grid)

    sigma2 = sigma * sigma
    g = torch.exp(-0.5 * dist2 / sigma2.clamp(min=1e-6))            # (B, T, K, grid*grid)

    # Zero-out anchors marked invalid
    g = g * mask.unsqueeze(-1).to(g.dtype)

    # Combine multiple anchors (gaze=1, hand=2) by max — overlapping Gaussians don't
    # double-count (additive would). Could also be sum + clamp; max keeps peaks ≤ 1.
    scores = g.max(dim=2).values                                     # (B, T, grid*grid)
    return scores


# ----------------------------------------------------------------------
# ATR — receiver-to-frame pooling + trajectory regression head
# ----------------------------------------------------------------------

def receivers_to_frame_pool(
    merged_video: torch.Tensor,   # (N_keep, d)
    receiver_idx: torch.Tensor,   # (N_keep,) — indices into the pre-merge n_video grid
    n_spatial:    int,
    T_merged:     int,
) -> torch.Tensor:
    """
    Mean-pool receivers per Qwen-merged frame.

    Returns:
        pooled : (T_merged, d). Frames with no receivers get zeros.
    """
    d = merged_video.shape[-1]
    frame_of_recv = (receiver_idx // n_spatial).long()
    pooled = torch.zeros(T_merged, d, dtype=merged_video.dtype, device=merged_video.device)
    counts = torch.zeros(T_merged, dtype=merged_video.dtype, device=merged_video.device)
    pooled.index_add_(0, frame_of_recv, merged_video)
    counts.index_add_(0, frame_of_recv, torch.ones_like(frame_of_recv, dtype=counts.dtype))
    pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
    return pooled, counts > 0   # (T_merged, d), (T_merged,) presence mask


class TrajectoryReconstructionHead(nn.Module):
    """
    Tiny MLP that maps per-frame pooled receiver features → (gaze_xy, left_xy, right_xy).

    Outputs sigmoid-bounded to [0, 1] (image coords).
    """

    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.gaze = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, 2), nn.Sigmoid(),
        )
        self.hand = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, 4), nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pooled : (T_merged, d) in any dtype.
        Returns (pred_gaze [T, 2], pred_hand [T, 4]) in float32.
        """
        x = pooled.float()
        return self.gaze(x), self.hand(x)


def downsample_traj_to_T_merged(
    x: torch.Tensor,             # (T_traj, ...) any trailing shape
    T_merged: int,
) -> torch.Tensor:
    """
    Subsample a per-traj-frame tensor onto T_merged frames by proportional indexing.
    Matches `score_to_qwen_spatiotemporal`'s temporal mapping.
    """
    T_traj = x.shape[0]
    if T_traj == T_merged:
        return x
    idx = (torch.arange(T_merged, device=x.device, dtype=torch.float32)
           * T_traj / max(1, T_merged)).long().clamp_max(T_traj - 1)
    return x.index_select(0, idx)


def atr_regression_loss(
    pred_gaze: torch.Tensor,    # (T_merged, 2)
    pred_hand: torch.Tensor,    # (T_merged, 4)
    gt_gaze:   torch.Tensor,    # (T_merged, 2) ∈ [0,1]
    gt_left:   torch.Tensor,    # (T_merged, 2)
    gt_right:  torch.Tensor,    # (T_merged, 2)
    gaze_mask:  torch.Tensor,   # (T_merged,) bool
    left_mask:  torch.Tensor,   # (T_merged,) bool
    right_mask: torch.Tensor,   # (T_merged,) bool
    receiver_present: torch.Tensor,   # (T_merged,) bool — pooled was non-empty
) -> torch.Tensor:
    """
    Smooth-L1 regression loss. Frames are weighted equally;
    invalid anchors (mask=False) contribute 0.

    Returns a scalar loss (mean over valid frame-anchor pairs).
    """
    valid = receiver_present.to(pred_gaze.dtype)
    gm = (gaze_mask  & receiver_present).to(pred_gaze.dtype)
    lm = (left_mask  & receiver_present).to(pred_gaze.dtype)
    rm = (right_mask & receiver_present).to(pred_gaze.dtype)

    l_gaze  = F.smooth_l1_loss(pred_gaze,  gt_gaze.float(),  reduction="none").sum(-1) * gm
    l_left  = F.smooth_l1_loss(pred_hand[:, :2], gt_left.float(),  reduction="none").sum(-1) * lm
    l_right = F.smooth_l1_loss(pred_hand[:, 2:], gt_right.float(), reduction="none").sum(-1) * rm

    denom = (gm.sum() + lm.sum() + rm.sum()).clamp(min=1.0)
    return (l_gaze.sum() + l_left.sum() + l_right.sum()) / denom


# ----------------------------------------------------------------------
# CGM — identify receivers near gaze / hand to zero out at training time
# ----------------------------------------------------------------------

def trajectory_region_receiver_mask(
    receiver_idx: torch.Tensor,    # (N_keep,)
    T_merged: int, n_spatial: int,
    targets: torch.Tensor,         # (T_traj, K, 2) ∈ [0,1]
    target_mask: torch.Tensor,     # (T_traj, K) bool
    radius: float,
) -> torch.Tensor:
    """
    Return a bool tensor (N_keep,) marking receivers whose patch location is within
    `radius` of any valid target anchor at the receiver's frame (mapped T_merged → T_traj).
    """
    side = int(round(n_spatial ** 0.5))
    device = receiver_idx.device

    recv_frame = (receiver_idx // n_spatial).long()
    recv_patch = (receiver_idx %  n_spatial).long()
    recv_row   = recv_patch // side
    recv_col   = recv_patch %  side
    recv_xy    = torch.stack([
        (recv_col.float() + 0.5) / side,
        (recv_row.float() + 0.5) / side,
    ], dim=-1)                                                # (N_keep, 2)

    T_traj = targets.shape[0]
    traj_frame = (recv_frame.float() * T_traj / max(1, T_merged)).long().clamp_max(T_traj - 1)

    tgt   = targets[traj_frame].to(device)                    # (N_keep, K, 2)
    tmask = target_mask[traj_frame].to(device)                # (N_keep, K)
    diff  = tgt - recv_xy.unsqueeze(1)                        # (N_keep, K, 2)
    dist  = diff.norm(dim=-1)                                 # (N_keep, K)
    dist  = torch.where(tmask, dist, torch.full_like(dist, float("inf")))
    return dist.min(dim=1).values < radius                    # (N_keep,)


class TrajectoryAnchorModule(nn.Module):
    """
    Multiplicative Gaussian prior on top of per-frame patch scores.

    Built so that at init it is identity (alpha=0): existing Stage 1 ckpts pass
    through unchanged. As training progresses, learnable alpha_{gaze,hand} can
    open the gate, amplifying scores at GT gaze / hand patch locations.

    Inputs (forward):
        scores       : (B, T, grid*grid) — existing per-frame patch scores in [0, 1]-ish
        traj_batch   : dict with keys
                         'gaze_pos'  (B, T, 2), 'gaze_mask'  (B, T)
                         'left_pos'  (B, T, 2), 'left_mask'  (B, T)
                         'right_pos' (B, T, 2), 'right_mask' (B, T)

    Output:
        anchored_scores : (B, T, grid*grid) = scores * (1 + a_g * G_gaze + a_h * G_hand)
    """

    def __init__(
        self,
        grid: int = 14,
        sigma_gaze_init: float = 0.21,   # ≈ 3 patch units / 14
        sigma_hand_init: float = 0.29,   # ≈ 4 patch units / 14
        # zero-init pre-tanh → tanh(0)=0 → identity at step 0
    ):
        super().__init__()
        self.grid = grid
        # softplus(log(σ)) — keep sigma strictly positive, log-parameterized for stability
        self.log_sigma_gaze = nn.Parameter(torch.log(torch.tensor(sigma_gaze_init)))
        self.log_sigma_hand = nn.Parameter(torch.log(torch.tensor(sigma_hand_init)))
        # Multiplicative amplitude — tanh-gated; identity at step 0
        self.amp_gaze_raw = nn.Parameter(torch.zeros(1))
        self.amp_hand_raw = nn.Parameter(torch.zeros(1))
        # Max amplitude (so prior factor ∈ [1, 1 + 2*amp_max]). 1.5 ≈ 4× boost.
        self.amp_max = 1.5

    def forward(self, scores: torch.Tensor, traj_batch: dict) -> torch.Tensor:
        device = scores.device
        gaze_pos  = traj_batch["gaze_pos" ].to(device)        # (B, T, 2)
        gaze_mask = traj_batch["gaze_mask"].to(device).bool() # (B, T)
        left_pos  = traj_batch["left_pos" ].to(device)
        left_mask = traj_batch["left_mask"].to(device).bool()
        right_pos = traj_batch["right_pos"].to(device)
        right_mask= traj_batch["right_mask"].to(device).bool()

        # Hand anchors stacked as K=2
        hand_pos  = torch.stack([left_pos,  right_pos],  dim=2)   # (B, T, 2, 2)
        hand_mask = torch.stack([left_mask, right_mask], dim=2)   # (B, T, 2)

        gaze_pos_k  = gaze_pos.unsqueeze(2)                       # (B, T, 1, 2)
        gaze_mask_k = gaze_mask.unsqueeze(2)                      # (B, T, 1)

        sigma_g = self.log_sigma_gaze.exp()
        sigma_h = self.log_sigma_hand.exp()

        G_gaze = gaussian_patch_map(gaze_pos_k, gaze_mask_k, sigma_g, grid=self.grid)
        G_hand = gaussian_patch_map(hand_pos,   hand_mask,   sigma_h, grid=self.grid)

        # tanh-gated amplitude in (-amp_max, +amp_max). Init at tanh(0)=0 →
        # identity prior at step 0. Allowing the sign to be learned avoids the
        # dead-gradient floor that `.clamp(min=0)` introduced (an early small
        # negative push would freeze amp at 0 permanently, since clamp's
        # gradient is zero in the clamped region).
        amp_g = torch.tanh(self.amp_gaze_raw) * self.amp_max
        amp_h = torch.tanh(self.amp_hand_raw) * self.amp_max

        prior = 1.0 + amp_g * G_gaze.to(scores.dtype) + amp_h * G_hand.to(scores.dtype)
        return scores * prior

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (
                f"sigma_gaze={self.log_sigma_gaze.exp().item():.3f}, "
                f"sigma_hand={self.log_sigma_hand.exp().item():.3f}, "
                f"amp_gaze={torch.tanh(self.amp_gaze_raw).item() * self.amp_max:+.3f}, "
                f"amp_hand={torch.tanh(self.amp_hand_raw).item() * self.amp_max:+.3f}"
            )


__all__ = [
    "gaussian_patch_map",
    "TrajectoryAnchorModule",
    "receivers_to_frame_pool",
    "TrajectoryReconstructionHead",
    "downsample_traj_to_T_merged",
    "atr_regression_loss",
    "trajectory_region_receiver_mask",
]
