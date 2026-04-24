"""
Gaze-Weighted Bipartite Token Merging (with optional drop + gaze-aware matching).

gaze_weighted_merge(tokens, patch_scores, r_merge, r_drop=0, ...):
    - Top (N - r_merge - r_drop) patches → receivers (protected)
    - Middle r_merge patches            → sources (merged into best-matching receiver)
    - Bottom r_drop patches             → fully dropped (removed from sequence)
    - Merge is a weighted average: numerator = w_r * r + Σ w_s * s
    - Differentiable: gradients flow through w_r, w_s → patch_scores → encoder
    - Uses scatter_add to correctly handle multiple sources per receiver
    - Matching similarity can be biased by gaze-score gap (score-aware matching)

score_to_qwen_spatial(patch_scores, n_spatial):
    - Interpolates TrajGaze 14×14 scores to Qwen's spatial token grid
    - Matches the 14→16→8 pipeline used in stage2.py for mask mapping
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
        s16 = F.interpolate(scores_2d, size=(16, 16), mode="nearest")
        s8  = F.avg_pool2d(s16, kernel_size=2, stride=2)
        return s8.squeeze().flatten()          # (64,)
    else:
        out = F.interpolate(scores_2d, size=(side, side),
                            mode="bilinear", align_corners=False)
        return out.squeeze().flatten()         # (n_spatial,)


def _apply_score_transform(scores: torch.Tensor, transform: str) -> torch.Tensor:
    """
    Normalize TrajGaze raw scores so they are safe to use as merge weights.

    sigmoid  → (0, 1)         — default, guarantees positive bounded weights
    softplus → (0, ∞)          — keeps relative magnitude, no saturation at 1
    none     → passthrough    — matches legacy behavior (may be negative)
    """
    if transform == "sigmoid":
        return torch.sigmoid(scores)
    if transform == "softplus":
        return F.softplus(scores)
    if transform == "none":
        return scores
    raise ValueError(f"Unknown score_transform: {transform!r}")


def gaze_weighted_merge(
    tokens:              torch.Tensor,
    patch_scores:        torch.Tensor,
    r_merge:             int,
    r_drop:              int = 0,
    score_transform:     str = "sigmoid",
    match_score_penalty: float = 0.0,
    match_score_hard_gap: float | None = None,
    eps:                 float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gaze-guided bipartite token merging with optional drop path.

    Partitions the N tokens by gaze score (descending) into three groups:
        receivers : top (N - r_merge - r_drop) — kept, protected
        sources   : next r_merge              — merged into nearest receiver
        drops     : bottom r_drop             — fully removed from sequence

    Source → receiver matching is cosine-similarity based, optionally biased by
    the gaze-score gap: a source is discouraged from matching receivers whose
    score is very different (prevents dilution of high-score receivers by noise).

    Args:
        tokens               : (N, d) visual token embeddings
        patch_scores         : (N,) gaze importance scores (higher = more attended)
        r_merge              : # of tokens to merge away
        r_drop               : # of tokens to drop completely (no merge)
        score_transform      : how to normalize raw scores before use ("sigmoid"/"softplus"/"none")
        match_score_penalty  : λ in sim_eff = sim - λ · |w_r - w_s|  (soft constraint)
        match_score_hard_gap : if set, receivers with (w_r - w_s) > gap are excluded
        eps                  : clamp for denominator / weights

    Returns:
        merged       : (N - r_merge - r_drop, d) merged/surviving receiver embeddings
        receiver_idx : (N - r_merge - r_drop,) indices into original N tokens
        drop_idx     : (r_drop,) indices of dropped tokens (empty tensor if r_drop=0)
    """
    N, d = tokens.shape
    assert r_merge >= 0 and r_drop >= 0, f"r_merge={r_merge}, r_drop={r_drop}"
    assert r_merge + r_drop < N, (
        f"r_merge + r_drop = {r_merge + r_drop} must be < N = {N}"
    )

    scores_t = _apply_score_transform(patch_scores, score_transform)

    # Sort descending: receivers | sources | drops
    sorted_idx   = torch.argsort(scores_t, descending=True)
    n_recv       = N - r_merge - r_drop
    receiver_idx = sorted_idx[:n_recv]
    source_idx   = sorted_idx[n_recv : n_recv + r_merge]
    drop_idx     = sorted_idx[n_recv + r_merge :]

    receivers = tokens[receiver_idx]             # (n_recv, d)
    w_r = scores_t[receiver_idx].clamp(min=eps)  # (n_recv,)

    # Fast path: no merge needed
    if r_merge == 0:
        merged = receivers
        return merged, receiver_idx, drop_idx

    sources = tokens[source_idx]                 # (r_merge, d)
    w_s = scores_t[source_idx].clamp(min=eps)    # (r_merge,)

    # Cosine-similarity matching (in float32 for numerical stability)
    r_f = F.normalize(receivers.float(), dim=-1)   # (n_recv, d)
    s_f = F.normalize(sources.float(),   dim=-1)   # (r_merge, d)
    sim = s_f @ r_f.T                              # (r_merge, n_recv)

    # Score-aware matching bias (soft)
    if match_score_penalty != 0.0:
        gap_abs = (w_s.unsqueeze(1) - w_r.unsqueeze(0)).abs()  # (r_merge, n_recv)
        sim = sim - match_score_penalty * gap_abs.float()

    # Score-aware hard exclusion: receiver much higher-score than source → blocked
    if match_score_hard_gap is not None:
        gap = w_r.unsqueeze(0) - w_s.unsqueeze(1)          # (r_merge, n_recv)
        blocked = gap > match_score_hard_gap
        if blocked.all(dim=-1).any():
            # at least one source has no candidate → fall back to cosine-only for it
            fallback_rows = blocked.all(dim=-1)
            sim = sim.masked_fill(blocked & ~fallback_rows.unsqueeze(-1), float("-inf"))
        else:
            sim = sim.masked_fill(blocked, float("-inf"))

    best_match = sim.argmax(dim=-1)                # (r_merge,) → receiver index

    # Weighted merge via scatter_add
    numerator   = receivers * w_r.unsqueeze(-1)                   # (n_recv, d)
    w_s_contrib = sources   * w_s.unsqueeze(-1)                   # (r_merge, d)
    numerator.scatter_add_(
        0,
        best_match.unsqueeze(-1).expand(-1, d),
        w_s_contrib,
    )

    denominator = w_r.clone()                                      # (n_recv,)
    denominator.scatter_add_(0, best_match, w_s)

    merged = numerator / denominator.unsqueeze(-1).clamp(min=eps)
    merged = merged.to(tokens.dtype)

    return merged, receiver_idx, drop_idx


# ── Frame-scoped variants (축 1) ──────────────────────────────────────────────
# These call gaze_weighted_merge once per frame, so merge partners are
# automatically restricted to the same frame (intra-frame match) — no
# temporal smearing.  The caller supplies (T, N_spatial) layout info.


def _allocate_per_frame_budget(
    T:            int,
    n_spatial:    int,
    r_merge_total: int,
    r_drop_total:  int,
    frame_scores: torch.Tensor | None,
    k_min:        int = 1,
    budget_temp:  float = 1.0,
) -> tuple[list[int], list[int]]:
    """
    Decide per-frame (receiver_count, source_count+drop_count) given a global budget.

    Returns (receiver_counts, removed_counts) — both length T, entries non-negative,
    Σ receiver_counts = T·n_spatial − r_merge_total − r_drop_total,
    Σ removed_counts  = r_merge_total + r_drop_total.

    If frame_scores is None, the removed budget is split evenly across frames
    (per_frame mode).  Otherwise allocated proportional to softmax(frame_scores / τ)
    inverted: frames with LOWER importance give up MORE tokens. k_min enforces a
    per-frame receiver floor to preserve time-axis continuity (avoid dropping an
    entire frame, which would punch holes in Qwen M-RoPE temporal coords).
    """
    N_total      = T * n_spatial
    remove_total = r_merge_total + r_drop_total
    keep_total   = N_total - remove_total
    assert keep_total >= T * k_min, (
        f"keep_total={keep_total} < T·k_min={T*k_min}; "
        f"reduce merge/drop or lower k_min."
    )

    if frame_scores is None:
        # Equal allocation.  Use integer division + remainder distribution.
        base_keep = keep_total // T
        extra     = keep_total - base_keep * T
        receiver_counts = [base_keep + (1 if i < extra else 0) for i in range(T)]
    else:
        assert frame_scores.shape == (T,), (
            f"frame_scores shape {tuple(frame_scores.shape)} != (T={T},)"
        )
        # Softmax over frame importance → fraction of keep_total per frame.
        w = torch.softmax(frame_scores.float() / max(budget_temp, 1e-6), dim=0)
        raw = (w * (keep_total - T * k_min)).tolist()   # extra budget on top of k_min
        floor_vals = [int(x) for x in raw]
        residual = keep_total - T * k_min - sum(floor_vals)
        # Distribute residual to frames with largest fractional parts.
        frac = sorted(
            range(T), key=lambda i: raw[i] - floor_vals[i], reverse=True
        )
        for i in frac[:residual]:
            floor_vals[i] += 1
        receiver_counts = [k_min + v for v in floor_vals]
        # Cap each frame by n_spatial
        for i in range(T):
            if receiver_counts[i] > n_spatial:
                overflow = receiver_counts[i] - n_spatial
                receiver_counts[i] = n_spatial
                # spill overflow to the next frames (round-robin) that can absorb
                j = 0
                while overflow > 0 and j < T:
                    if i != j and receiver_counts[j] < n_spatial:
                        receiver_counts[j] += 1
                        overflow -= 1
                    j += 1

    removed_counts = [n_spatial - k for k in receiver_counts]
    return receiver_counts, removed_counts


def gaze_weighted_merge_per_frame(
    tokens_tn:   torch.Tensor,          # (T, n_spatial, d)
    scores_tn:   torch.Tensor,          # (T, n_spatial)
    r_merge_total: int,
    r_drop_total:  int = 0,
    frame_scores: torch.Tensor | None = None,
    k_min:        int = 1,
    budget_temp:  float = 1.0,
    **merge_kwargs,                     # forwarded to gaze_weighted_merge
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply gaze_weighted_merge independently to each of the T frames, with a
    user-chosen per-frame budget.

    ★ Merge partner match is strictly intra-frame — sources from frame t can
    only be absorbed by receivers in frame t. This preserves temporal physics
    (t=2 hand feature cannot be smeared into a t=15 background token).

    Args:
        tokens_tn     : (T, n_spatial, d)
        scores_tn     : (T, n_spatial) — patch importance per spatial slot per frame
        r_merge_total : total number of source tokens across ALL frames
        r_drop_total  : total number of drop tokens across ALL frames
        frame_scores  : (T,) optional frame-level importance for budget allocation.
                        If None, budget is split evenly across frames (per_frame mode).
                        If provided, allocated proportional to softmax(scores/τ) (global mode).
        k_min         : minimum number of receivers per frame (prevents frame holes)
        budget_temp   : temperature τ for softmax (higher = more uniform)
        merge_kwargs  : passed through to gaze_weighted_merge (score_transform, etc.)

    Returns:
        merged          : (N_total - r_merge_total - r_drop_total, d)
        receiver_idx    : (keep,) indices into the FLATTENED (T·n_spatial, d) tokens
        drop_idx        : (r_drop_total,) flat indices of dropped tokens
    """
    T, n_spatial, d = tokens_tn.shape
    assert scores_tn.shape == (T, n_spatial), (
        f"scores_tn shape {tuple(scores_tn.shape)} != (T={T}, n_spatial={n_spatial})"
    )

    recv_counts, removed_counts = _allocate_per_frame_budget(
        T, n_spatial, r_merge_total, r_drop_total, frame_scores, k_min, budget_temp
    )

    # Distribute the global r_drop across frames proportional to removed_counts.
    # drop_counts_per_frame ≈ removed_counts[i] * r_drop_total / (r_merge_total + r_drop_total)
    total_removed = r_merge_total + r_drop_total
    drop_counts = [0] * T
    if r_drop_total > 0 and total_removed > 0:
        raw_d = [rc * r_drop_total / total_removed for rc in removed_counts]
        drop_counts = [int(x) for x in raw_d]
        residual = r_drop_total - sum(drop_counts)
        frac = sorted(range(T), key=lambda i: raw_d[i] - drop_counts[i], reverse=True)
        for i in frac[:residual]:
            drop_counts[i] += 1
        # Ensure drop_counts[i] ≤ removed_counts[i]
        for i in range(T):
            if drop_counts[i] > removed_counts[i]:
                overflow = drop_counts[i] - removed_counts[i]
                drop_counts[i] = removed_counts[i]
                j = 0
                while overflow > 0 and j < T:
                    room = removed_counts[j] - drop_counts[j]
                    if i != j and room > 0:
                        take = min(room, overflow)
                        drop_counts[j] += take
                        overflow      -= take
                    j += 1

    merged_chunks = []
    recv_idx_chunks = []
    drop_idx_chunks = []
    for t in range(T):
        r_merge_t = removed_counts[t] - drop_counts[t]
        r_drop_t  = drop_counts[t]
        if recv_counts[t] == 0:
            # Entire frame is removed. (Only possible if k_min=0 and caller allows it.)
            # Still return dropped indices for completeness.
            frame_drop = torch.arange(
                n_spatial, device=tokens_tn.device, dtype=torch.long
            ) + t * n_spatial
            drop_idx_chunks.append(frame_drop)
            continue

        merged_t, recv_idx_t, drop_idx_t = gaze_weighted_merge(
            tokens_tn[t],
            scores_tn[t],
            r_merge_t,
            r_drop=r_drop_t,
            **merge_kwargs,
        )
        merged_chunks.append(merged_t)
        recv_idx_chunks.append(recv_idx_t + t * n_spatial)
        if r_drop_t > 0:
            drop_idx_chunks.append(drop_idx_t + t * n_spatial)

    merged       = torch.cat(merged_chunks, dim=0) if merged_chunks else \
                   torch.empty(0, d, dtype=tokens_tn.dtype, device=tokens_tn.device)
    receiver_idx = torch.cat(recv_idx_chunks, dim=0) if recv_idx_chunks else \
                   torch.empty(0, dtype=torch.long, device=tokens_tn.device)
    drop_idx     = torch.cat(drop_idx_chunks, dim=0) if drop_idx_chunks else \
                   torch.empty(0, dtype=torch.long, device=tokens_tn.device)

    return merged, receiver_idx, drop_idx
