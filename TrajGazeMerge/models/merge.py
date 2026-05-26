"""
Gaze-Weighted Bipartite Token Merging.

gaze_weighted_merge(tokens, patch_scores, r):
    - High-score patches → receivers (protected)
    - Low-score patches  → sources (merged into best-matching receiver)
    - Merge is a weighted average: numerator = w_r * r + Σ w_s * s
    - Differentiable: gradients flow through w_r, w_s → patch_scores → encoder
    - Uses scatter_add to correctly handle multiple sources per receiver

score_to_qwen_spatial(patch_scores, n_spatial):
    - Interpolates TrajGaze 14×14 scores to Qwen's spatial token grid
    - Matches the 14→16→8 pipeline used in stage2.py for mask mapping

compute_anchor_mask_for_qwen(traj, n_spatial):
    - Builds a (n_spatial,) {0,1} mask marking spatial cells that contain a
      gaze, left-hand, or right-hand position in ANY trajectory frame.
    - Used by the anchored TrajGazeMerge variant to force gaze/hand cells
      into the receiver set ahead of the score-based top-k.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_anchor_mask_for_qwen(traj: dict, n_spatial: int) -> torch.Tensor:
    """
    Per-item Qwen-grid anchor mask.

    Binning: every (x, y) ∈ [0,1] from gaze_pos / left_pos / right_pos that has
    its corresponding mask=True is mapped to its spatial cell (row, col) on a
    side×side grid (side = √n_spatial) and OR'd into the anchor mask.

    Returns:
        (n_spatial,) float32 with values in {0.0, 1.0}.
    """
    side = int(round(n_spatial ** 0.5))
    if side * side != n_spatial:
        # Non-square fallback: drop the anchor (returns all-zeros) — the
        # caller then degrades to pure score-based selection for this item.
        return torch.zeros(n_spatial, dtype=torch.float32,
                           device=traj["gaze_pos"].device)
    device = traj["gaze_pos"].device
    anchor = torch.zeros(n_spatial, dtype=torch.float32, device=device)
    for pos_key, mask_key in (("gaze_pos",  "gaze_mask"),
                              ("left_pos",  "left_mask"),
                              ("right_pos", "right_mask")):
        pos = traj.get(pos_key)
        msk = traj.get(mask_key)
        if pos is None or msk is None:
            continue
        if msk.dtype != torch.bool:
            msk = msk.bool()
        if msk.sum().item() == 0:
            continue
        pp = pos[msk]                                              # (Tk, 2) in [0,1]
        cc = (pp[:, 0] * side).long().clamp(0, side - 1)
        rr = (pp[:, 1] * side).long().clamp(0, side - 1)
        anchor[rr * side + cc] = 1.0
    return anchor


def score_to_qwen_spatial(patch_scores: torch.Tensor, n_spatial: int) -> torch.Tensor:
    """
    Interpolate TrajGaze 14×14 patch scores to Qwen's spatial token grid.

    Follows the same 14→16→8 pipeline used in stage2.py for mask mapping:
        14×14 → (nearest) → 16×16 → (avg_pool 2×2) → 8×8

    For grids other than 8×8, falls back to bilinear interpolation.

    Args:
        patch_scores : (196,) float — TrajGaze output (14×14)
        n_spatial    : int — number of spatial tokens per frame in Qwen

    Returns:
        scores_flat  : (n_spatial,) float — one score per Qwen spatial token
    """
    side = int(n_spatial ** 0.5)

    scores_2d = patch_scores.float().reshape(1, 1, 14, 14)

    if side == 8:
        # Standard path matching stage2.py:  14 → 16 (nearest) → 8 (avg pool)
        s16 = F.interpolate(scores_2d, size=(16, 16), mode="nearest")
        s8  = F.avg_pool2d(s16, kernel_size=2, stride=2)
        return s8.squeeze().flatten()          # (64,)
    else:
        # Generic fallback: bilinear interpolation directly to target size
        out = F.interpolate(scores_2d, size=(side, side),
                            mode="bilinear", align_corners=False)
        return out.squeeze().flatten()         # (n_spatial,)


def gaze_weighted_merge(
    tokens:       torch.Tensor,   # (N, d)
    patch_scores: torch.Tensor,   # (N,) — one score per token
    r:            int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gaze-guided bipartite token merging.

    Assigns the top (N - r) tokens by gaze score as *receivers* (protected),
    and the bottom r tokens as *sources* (to be merged into nearest receiver).

    Each source is matched to its most cosine-similar receiver (by token value).
    The merge is a weighted average over all sources that map to the same receiver:

        merged_receiver = (w_r * r_embed + Σ_{s→r} w_s * s_embed)
                        / (w_r           + Σ_{s→r} w_s)

    Gradients flow through w_r and w_s back to patch_scores → encoder.
    Gradients through best_match (argmax) are zero — standard in ToMe.

    Args:
        tokens       : (N, d) visual token embeddings (float or bfloat16)
        patch_scores : (N,) gaze importance scores (higher = more attended)
        r            : number of tokens to remove (sequence shortens by r)

    Returns:
        merged        : (N - r, d) merged token embeddings
        receiver_idx  : (N - r,) indices into original N tokens that are receivers
    """
    N, d = tokens.shape
    assert 0 < r < N, f"r={r} out of range for N={N}"

    # Sort descending by score: top (N-r) = receivers, bottom r = sources
    sorted_idx    = torch.argsort(patch_scores, descending=True)
    receiver_idx  = sorted_idx[:N - r]          # (N-r,)
    source_idx    = sorted_idx[N - r:]          # (r,)

    receivers = tokens[receiver_idx]             # (N-r, d)
    sources   = tokens[source_idx]              # (r, d)
    w_r = patch_scores[receiver_idx]            # (N-r,) — receiver weights
    w_s = patch_scores[source_idx]              # (r,)   — source weights

    # Cosine-similarity matching (in float32 for numerical stability)
    r_f = F.normalize(receivers.float(), dim=-1)   # (N-r, d)
    s_f = F.normalize(sources.float(),   dim=-1)   # (r,   d)
    sim = s_f @ r_f.T                              # (r, N-r)
    best_match = sim.argmax(dim=-1)                # (r,) — index into receivers

    # Weighted merge via scatter_add (handles multiple sources per receiver)
    # numerator:   w_r * r_embed  +  Σ_{s→r} w_s * s_embed
    # denominator: w_r            +  Σ_{s→r} w_s
    numerator   = receivers * w_r.unsqueeze(-1)                   # (N-r, d)
    w_s_contrib = sources   * w_s.unsqueeze(-1)                   # (r, d)
    numerator.scatter_add_(
        0,
        best_match.unsqueeze(-1).expand(-1, d),
        w_s_contrib,
    )

    denominator = w_r.clone()                                      # (N-r,)
    denominator.scatter_add_(0, best_match, w_s)

    merged = numerator / denominator.unsqueeze(-1).clamp(min=1e-6)
    merged = merged.to(tokens.dtype)

    return merged, receiver_idx
