"""Unit tests for gaze_weighted_merge — drop/merge split, score transform, parity."""

from __future__ import annotations

import torch

from TrajGazeMerge.models.merge import gaze_weighted_merge


def _legacy_gaze_weighted_merge(tokens, patch_scores, r):
    """Pre-change reference implementation (no drop, no transform, no gaze-aware matching)."""
    import torch.nn.functional as F
    N, d = tokens.shape
    sorted_idx   = torch.argsort(patch_scores, descending=True)
    receiver_idx = sorted_idx[:N - r]
    source_idx   = sorted_idx[N - r:]
    receivers = tokens[receiver_idx]
    sources   = tokens[source_idx]
    w_r = patch_scores[receiver_idx]
    w_s = patch_scores[source_idx]
    r_f = F.normalize(receivers.float(), dim=-1)
    s_f = F.normalize(sources.float(),   dim=-1)
    sim = s_f @ r_f.T
    best_match = sim.argmax(dim=-1)
    numerator   = receivers * w_r.unsqueeze(-1)
    w_s_contrib = sources   * w_s.unsqueeze(-1)
    numerator.scatter_add_(0, best_match.unsqueeze(-1).expand(-1, d), w_s_contrib)
    denominator = w_r.clone()
    denominator.scatter_add_(0, best_match, w_s)
    merged = numerator / denominator.unsqueeze(-1).clamp(min=1e-6)
    return merged.to(tokens.dtype), receiver_idx


def test_parity_with_legacy():
    """score_transform=none + r_drop=0 + penalty=0 → identical to legacy."""
    torch.manual_seed(0)
    N, d = 64, 32
    tokens = torch.randn(N, d)
    scores = torch.rand(N).abs() + 0.1   # positive so legacy doesn't misbehave
    r = 40

    merged_new, recv_new, drop_new = gaze_weighted_merge(
        tokens, scores, r_merge=r, r_drop=0,
        score_transform="none", match_score_penalty=0.0,
    )
    merged_legacy, recv_legacy = _legacy_gaze_weighted_merge(tokens, scores, r)

    assert drop_new.numel() == 0
    assert torch.equal(recv_new, recv_legacy)
    assert torch.allclose(merged_new, merged_legacy, atol=1e-5, rtol=1e-5)


def test_drop_merge_shapes():
    """Drop + merge split yields expected shapes and disjoint index sets."""
    torch.manual_seed(1)
    N, d = 100, 16
    tokens = torch.randn(N, d)
    scores = torch.randn(N)  # includes negatives on purpose
    r_merge, r_drop = 40, 10

    merged, recv_idx, drop_idx = gaze_weighted_merge(
        tokens, scores, r_merge=r_merge, r_drop=r_drop,
        score_transform="sigmoid",
    )

    assert merged.shape == (N - r_merge - r_drop, d)
    assert recv_idx.numel() == N - r_merge - r_drop
    assert drop_idx.numel() == r_drop

    all_idx = torch.cat([recv_idx, drop_idx])
    assert all_idx.unique().numel() == all_idx.numel(), "recv and drop overlap"

    # Drop indices should correspond to the lowest (transformed) scores.
    sig = torch.sigmoid(scores)
    drop_vals = sig[drop_idx]
    recv_vals = sig[recv_idx]
    assert drop_vals.max() <= recv_vals.min() + 1e-6


def test_negative_scores_stable():
    """Raw negative scores must not produce NaN under sigmoid transform."""
    torch.manual_seed(2)
    tokens = torch.randn(50, 8)
    scores = torch.randn(50) * 5.0 - 2.0   # heavy tails, negatives likely
    merged, _, _ = gaze_weighted_merge(
        tokens, scores, r_merge=30, r_drop=5,
        score_transform="sigmoid",
    )
    assert torch.isfinite(merged).all()


def test_softplus_transform():
    torch.manual_seed(3)
    tokens = torch.randn(30, 8)
    scores = torch.randn(30)
    merged, _, _ = gaze_weighted_merge(
        tokens, scores, r_merge=20, r_drop=0,
        score_transform="softplus",
    )
    assert torch.isfinite(merged).all()


def test_zero_merge_only_drop():
    """r_merge=0, r_drop>0: merged = receivers unchanged."""
    torch.manual_seed(4)
    N, d = 40, 6
    tokens = torch.randn(N, d)
    scores = torch.randn(N)
    r_drop = 10
    merged, recv_idx, drop_idx = gaze_weighted_merge(
        tokens, scores, r_merge=0, r_drop=r_drop, score_transform="sigmoid",
    )
    assert merged.shape == (N - r_drop, d)
    assert drop_idx.numel() == r_drop
    assert torch.allclose(merged, tokens[recv_idx])


def test_gradient_flow():
    """Gradients must flow from merged output back to patch_scores (and thus encoder)."""
    torch.manual_seed(5)
    tokens = torch.randn(20, 8)
    scores = torch.randn(20, requires_grad=True)
    merged, _, _ = gaze_weighted_merge(
        tokens, scores, r_merge=12, r_drop=3, score_transform="sigmoid",
    )
    merged.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert (scores.grad.abs() > 0).any()


def test_match_score_penalty_changes_assignment():
    """Large λ should (often) change best_match vs λ=0."""
    torch.manual_seed(6)
    N, d = 30, 8
    tokens = torch.randn(N, d)
    scores = torch.rand(N)
    m0, r0, _ = gaze_weighted_merge(tokens, scores, r_merge=20, r_drop=0,
                                    score_transform="none",
                                    match_score_penalty=0.0)
    m1, r1, _ = gaze_weighted_merge(tokens, scores, r_merge=20, r_drop=0,
                                    score_transform="none",
                                    match_score_penalty=5.0)
    # Shape parity holds
    assert m0.shape == m1.shape
    assert torch.equal(r0, r1)
    # Values differ under a non-trivial penalty (statistically almost certain
    # on random inputs; deterministic via seed).
    assert not torch.allclose(m0, m1)


if __name__ == "__main__":
    test_parity_with_legacy()
    test_drop_merge_shapes()
    test_negative_scores_stable()
    test_softplus_transform()
    test_zero_merge_only_drop()
    test_gradient_flow()
    test_match_score_penalty_changes_assignment()
    print("All merge tests passed.")
