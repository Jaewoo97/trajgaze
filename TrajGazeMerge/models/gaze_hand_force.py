"""
Force-include gaze and hand visual tokens.

Given the StreamGaze trajectory dict (per-frame normalized gaze + L/R hand
positions, all in [0,1]) and the Qwen video grid, return a boolean mask
marking the flat post-merge token positions that contain a valid gaze or
hand fixation. The trainer adds a large constant to `scores_all` at these
positions, which guarantees they sort into the receiver set in
`gaze_weighted_merge` while leaving the rest of the budget to be filled
by TrajGazeV2 scores.

Layout assumptions (Qwen2.5-VL default):
  - grid_thw[0] = (T_pre, H_pre, W_pre) — pre-merge grid (per Qwen processor).
  - Spatial merge size = 2, so post-merge spatial = (H_pre/2, W_pre/2).
  - video_embeds is laid out [t0_row0_col0, ..., t0_row(H_post-1)_col(W_post-1),
    t1_row0_col0, ...] — natural spatial-temporal order.
"""

from __future__ import annotations

import torch


_TRAJ_KEYS = (("gaze_pos", "gaze_mask"), ("left_pos", "left_mask"), ("right_pos", "right_mask"))


def gaze_hand_force_mask(
    traj:      dict,
    grid_thw:  torch.Tensor,   # (1, 3) int64 — Qwen processor output, pre-merge dims
    n_video:   int,            # total post-merge video tokens
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """
    Build a (n_video,) bool mask: True at flat indices covering gaze/L/R-hand
    positions for any trajectory frame.

    Args:
        traj: dict from StreamGazeMergeDataset.__getitem__()["traj"] with keys
              gaze_pos (T_traj, 2), gaze_mask (T_traj,), left_pos, left_mask,
              right_pos, right_mask. Positions are float in [0,1].
        grid_thw: (1, 3) int64 tensor (T_pre, H_pre, W_pre) from Qwen processor.
        n_video: cached["video_embeds"].shape[0]. Used for shape sanity.
        spatial_merge_size: Qwen ViT's spatial_merge_size (=2 for 2.5-VL).

    Returns:
        force_mask: (n_video,) bool on CPU. Mover to device before use.
    """
    T_pre = int(grid_thw[0, 0].item())
    H_pre = int(grid_thw[0, 1].item())
    W_pre = int(grid_thw[0, 2].item())
    H_post = H_pre // spatial_merge_size
    W_post = W_pre // spatial_merge_size
    n_spatial = H_post * W_post
    expected = T_pre * n_spatial
    # Defensive: if Qwen ever changes layout the trainer's score-shape fallback
    # already handles n_video != T*H*W; we mirror that by clamping to n_video.
    assert expected <= n_video + 1 and expected >= n_video - 1, (
        f"grid_thw mismatch: T*H_post*W_post={expected} vs n_video={n_video}"
    )

    mask = torch.zeros(n_video, dtype=torch.bool)

    gaze = traj.get("gaze_pos")
    if gaze is None:
        return mask
    T_traj = int(gaze.shape[0])
    if T_traj == 0:
        return mask

    # Each trajectory frame t maps to one Qwen temporal slot t_merged.
    # Two trajectory frames sharing a slot just OR their force-positions.
    t_merged_for = (torch.arange(T_traj) * T_pre // T_traj).clamp_(max=T_pre - 1)

    for pos_key, mask_key in _TRAJ_KEYS:
        pos  = traj.get(pos_key)
        mflg = traj.get(mask_key)
        if pos is None or mflg is None:
            continue
        # pos: (T_traj, 2), mflg: (T_traj,) bool
        valid = mflg.bool()
        if not valid.any():
            continue
        xs = pos[:, 0].float()
        ys = pos[:, 1].float()
        # Drop NaN / out-of-frame
        finite = torch.isfinite(xs) & torch.isfinite(ys)
        in_box = (xs >= 0) & (xs < 1.0) & (ys >= 0) & (ys < 1.0)
        valid = valid & finite & in_box
        if not valid.any():
            continue

        cols = (xs * W_post).long().clamp_(0, W_post - 1)
        rows = (ys * H_post).long().clamp_(0, H_post - 1)
        t_idx = t_merged_for
        flat = t_idx * n_spatial + rows * W_post + cols
        flat = flat[valid].clamp_(0, n_video - 1)
        mask[flat] = True

    return mask
