"""Trajectory-guided temporal token-budget allocation for VisionZip.

Motivation (overlay-data analysis):
    On GAZE_OVERLAY frames the gaze marker is drawn INTO the pixels, so
    VisionZip's native attention already attends to the gaze region spatially.
    Re-biasing selection toward that region with a spatial Gaussian prior
    (VisionZip-traj / QC-Gate) is therefore REDUNDANT, and because the token
    budget is a fixed 10%, the extra concentration STEALS budget from scene
    context — which measurably hurts the context-needing tasks plain VisionZip
    wins. Empirically the QC-Gate controller never moved off its init and
    plain VisionZip beat it (+1.17pp at epoch 1).

    What the static overlay marker does NOT convey is WHICH FRAMES matter
    temporally (gaze fixation + hand manipulation over time). This module keeps
    VisionZip's WITHIN-FRAME spatial selection completely unchanged (top-K by
    attention dominant + cosine-key cluster-center contextual) and only
    reallocates HOW MANY of the 10% tokens each frame receives, using the
    gaze/hand trajectory's per-frame interaction signal.

Per-frame budget:
    share_t = sum_j attn[t, j] / sum(attn)              # VZ's own attention mass
    traj_t  = softmax(interaction_t / tau)              # trajectory temporal signal
    w_t     = (1 - traj_weight) * share_t + traj_weight * traj_t
    K_t     = apportion(w_t, total_keep, cap=S)         # integer, sum == total_keep
Within frame t: top-(k_dom) dominant by attention + (K_t - k_dom) contextual
cluster centers among that frame's non-dominant tokens — the IDENTICAL rule used
by visionzip_select_tokens, just restricted to one frame's S tokens.

Zero learnable params: LoRA adapts to this selection exactly as for plain
VisionZip (no controller, so the QC-Gate dead-gradient failure mode is absent).
`interaction` and the softmax match VisionZip-traj's temporal_w recipe exactly
(BETA_FIX=BETA_HAND_PRES=BETA_GAZE_HAND=1.0; see traj_weights.compute_traj_weights),
so this is purely a redirection of that same signal from score-multiply to
budget-allocation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from TrajGazeMerge.models.traj_weights import (
    _pool_to_T,
    SIGMA_VEL, SIGMA_GH_DIST,
    BETA_FIX, BETA_HAND_PRES, BETA_GAZE_HAND,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, CONTEXTUAL_RATIO, visionzip_select_tokens,
)


@torch.no_grad()
def compute_temporal_interaction(
    traj_dict: dict,
    T: int,
    device: torch.device,
    sigma_v: float = SIGMA_VEL,
    sigma_gh: float = SIGMA_GH_DIST,
) -> torch.Tensor:
    """(T,) per-frame interaction = fixation + hand_present + gaze_on_hand.

    Identical (pre-softmax) to the `interaction` in
    traj_weights.compute_traj_weights, but computes NO spatial kernels.
    Robust to absent hands / untracked gaze (those terms contribute 0).
    """
    gaze_pos,  gaze_mask  = _pool_to_T(traj_dict["gaze_pos"],  traj_dict["gaze_mask"],  T)
    left_pos,  left_mask  = _pool_to_T(traj_dict["left_pos"],  traj_dict["left_mask"],  T)
    right_pos, right_mask = _pool_to_T(traj_dict["right_pos"], traj_dict["right_mask"], T)
    gaze_pos  = gaze_pos.to(device);  left_pos = left_pos.to(device);  right_pos = right_pos.to(device)
    gaze_mask = gaze_mask.to(device).float()
    left_mask = left_mask.to(device).float()
    right_mask = right_mask.to(device).float()

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

    interaction = (
        BETA_FIX * fixation
        + BETA_HAND_PRES * hand_present
        + BETA_GAZE_HAND * gaze_on_hand
    )
    return interaction  # (T,)


def _apportion(weights: torch.Tensor, total: int, cap: int) -> torch.Tensor:
    """Integer apportionment (Hamilton / largest-remainder) with a per-frame cap.

    Returns counts (T,) long with sum == min(total, T*cap) and each count <= cap.
    """
    T = weights.shape[0]
    total = int(min(int(total), T * cap))
    if total <= 0:
        return torch.zeros(T, dtype=torch.long, device=weights.device)

    w = weights.clamp(min=0).float()
    s = w.sum()
    if s <= 0:
        w = torch.ones(T, device=weights.device)
        s = w.sum()
    raw = w / s * total
    base = torch.floor(raw).to(torch.long).clamp(max=cap)
    remainder = total - int(base.sum().item())

    if remainder > 0:
        frac = raw - torch.floor(raw)
        frac = torch.where(base < cap, frac, torch.full_like(frac, -1.0))
        while remainder > 0 and bool((base < cap).any()):
            order = torch.argsort(frac, descending=True)
            for idx in order.tolist():
                if remainder == 0:
                    break
                if base[idx] < cap:
                    base[idx] += 1
                    remainder -= 1
                    if base[idx] >= cap:
                        frac[idx] = -1.0
    return base


def temporal_budget_select_tokens(
    video_embeds: torch.Tensor,   # (N, d)
    attn_scores:  torch.Tensor,   # (N,)
    attn_key:     torch.Tensor,   # (1, N, d_k)
    grid_thw:     torch.Tensor,   # (1, 3): (T, H_grid, W_grid)
    traj_dict:    dict,
    *,
    tau:              float = 1.0,
    traj_weight:      float = 0.5,
    dominant_ratio:   float = DOMINANT_RATIO,
    contextual_ratio: float = CONTEXTUAL_RATIO,
    sigma_v:          float = SIGMA_VEL,
    sigma_gh:         float = SIGMA_GH_DIST,
):
    """VisionZip within-frame selection with trajectory-guided per-frame budget.

    Returns (selected_embeds (n_keep, d), receiver_idx (n_keep,) sorted asc,
             per_frame_budget (T,) long) — the budget vector is for logging.
    Falls back to global visionzip_select_tokens if the (T, S) layout is invalid.
    """
    device = video_embeds.device
    N, d = video_embeds.shape
    T = int(grid_thw[0, 0].item())

    total_keep = max(1, int((dominant_ratio + contextual_ratio) * N))
    total_keep = min(total_keep, N)

    if T <= 0 or N % T != 0:
        se, ri = visionzip_select_tokens(
            video_embeds, attn_scores, attn_key, dominant_ratio, contextual_ratio
        )
        return se, ri, None
    S = N // T

    # ── per-frame weight: blend VZ attention mass with trajectory interaction ──
    attn_2d = attn_scores.reshape(T, S).float()
    share = attn_2d.sum(dim=1)
    share = share / share.sum().clamp(min=1e-8)

    interaction = compute_temporal_interaction(traj_dict, T, device, sigma_v, sigma_gh)
    traj_w = F.softmax(interaction / max(tau, 1e-3), dim=0)

    w = (1.0 - traj_weight) * share + traj_weight * traj_w          # (T,)
    K = _apportion(w, total_keep, cap=S)                            # (T,) long

    frac_dom = dominant_ratio / max(1e-8, dominant_ratio + contextual_ratio)

    embeds_parts: list[torch.Tensor] = []
    idx_parts:    list[torch.Tensor] = []

    for t in range(T):
        kt = int(K[t].item())
        if kt <= 0:
            continue
        base = t * S
        a   = attn_2d[t]                             # (S,)
        emb = video_embeds[base:base + S]            # (S, d)
        key = attn_key[0, base:base + S]             # (S, d_k)

        k_dom = min(S, int(kt * frac_dom + 0.5))     # favor dominant on ties
        k_ctx = kt - k_dom

        if k_dom > 0:
            _, topk = torch.topk(a, k_dom)
            topk_sorted, _ = topk.sort()
            embeds_parts.append(emb[topk_sorted].float())
            idx_parts.append(base + topk_sorted)
            dom_mask = torch.zeros(S, dtype=torch.bool, device=device)
            dom_mask[topk] = True
            non_dom = (~dom_mask).nonzero(as_tuple=True)[0]
        else:
            non_dom = torch.arange(S, device=device)

        if k_ctx > 0 and non_dom.numel() > 0:
            nd_emb   = emb[non_dom]                              # (n_nd, d)
            nd_key_n = F.normalize(key[non_dom].float(), dim=-1) # (n_nd, d_k)
            n_nd     = non_dom.shape[0]
            step     = max(1, n_nd // k_ctx)
            center_local = torch.arange(0, n_nd, step, device=device)[:k_ctx]
            n_ctx    = center_local.shape[0]
            center_key_n = nd_key_n[center_local]
            assign   = (nd_key_n @ center_key_n.T).argmax(dim=-1)  # (n_nd,)
            is_center = torch.zeros(n_nd, dtype=torch.bool, device=device)
            is_center[center_local] = True
            non_center = (~is_center).nonzero(as_tuple=True)[0]

            center_embeds = nd_emb[center_local].float().clone()  # (n_ctx, d)
            counts = torch.ones(n_ctx, device=device)
            if non_center.numel() > 0:
                assign_nc = assign[non_center]
                center_embeds.scatter_add_(
                    0, assign_nc.unsqueeze(-1).expand(-1, d), nd_emb[non_center].float()
                )
                counts.scatter_add_(
                    0, assign_nc, torch.ones(non_center.shape[0], device=device)
                )
            center_embeds = center_embeds / counts.unsqueeze(-1)
            embeds_parts.append(center_embeds)
            idx_parts.append(base + non_dom[center_local])

    if not embeds_parts:
        se, ri = visionzip_select_tokens(
            video_embeds, attn_scores, attn_key, dominant_ratio, contextual_ratio
        )
        return se, ri, K

    all_embeds = torch.cat(embeds_parts, dim=0)
    all_idx    = torch.cat(idx_parts, dim=0)
    sort_order = all_idx.argsort()
    receiver_idx    = all_idx[sort_order]
    selected_embeds = all_embeds[sort_order].to(video_embeds.dtype)
    return selected_embeds, receiver_idx, K
