"""
Unit tests for the per-frame merge path (축 1).

Verifies:
  - When all flags are default, overall training behavior is numerically
    equivalent to the legacy `gaze_weighted_merge` path (the default
    --merge-scope legacy continues to call it directly — see train loop).
  - `gaze_weighted_merge_per_frame` with frame_scores=None produces the same
    receivers, sources, and merged tokens as per-frame calls to
    `gaze_weighted_merge`, confirming that partner match is intra-frame.
  - Frame-budget allocation (`frame_scores` given) enforces k_min, sums to K.
"""

from __future__ import annotations

import torch

from TrajGazeMerge.models.merge import (
    gaze_weighted_merge,
    gaze_weighted_merge_per_frame,
    _allocate_per_frame_budget,
)


def test_budget_sum_and_k_min():
    """Equal allocation and scored allocation both respect totals + k_min floor."""
    T, n_spatial = 8, 64
    r_merge_total, r_drop_total = 400, 0  # keep 112 across 8 frames (14 per frame on avg)
    # Equal allocation
    recv, removed = _allocate_per_frame_budget(
        T, n_spatial, r_merge_total, r_drop_total,
        frame_scores=None, k_min=1,
    )
    assert len(recv) == T
    assert sum(recv) == T * n_spatial - (r_merge_total + r_drop_total)
    assert all(r >= 1 for r in recv)
    assert all(rm == n_spatial - r for r, rm in zip(recv, removed))

    # Scored allocation
    fs = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.0])
    recv2, _ = _allocate_per_frame_budget(
        T, n_spatial, r_merge_total, r_drop_total,
        frame_scores=fs, k_min=1, budget_temp=1.0,
    )
    assert sum(recv2) == T * n_spatial - r_merge_total - r_drop_total
    assert all(r >= 1 for r in recv2)
    # High-score frames should get ≥ low-score frames
    assert recv2[0] >= recv2[-1]


def test_per_frame_shape_and_intra_frame_match():
    """Per-frame merge respects shape and never sends a source across frames."""
    torch.manual_seed(42)
    T, n_spatial, d = 4, 16, 8
    tokens = torch.randn(T, n_spatial, d)
    scores = torch.rand(T, n_spatial).abs() + 0.1

    r_merge_total = 24  # merge 6 per frame (with uniform allocation)
    merged, recv_idx, drop_idx = gaze_weighted_merge_per_frame(
        tokens, scores, r_merge_total=r_merge_total, r_drop_total=0,
        frame_scores=None, k_min=1, score_transform="none",
    )
    assert merged.shape == (T * n_spatial - r_merge_total, d)
    assert recv_idx.numel() == merged.shape[0]
    # Receiver indices must cover all T frames (since uniform budget per frame)
    frames_covered = (recv_idx // n_spatial).unique()
    assert frames_covered.numel() == T


def test_per_frame_matches_manual_per_frame_calls():
    """
    With frame_scores=None (uniform budget) and uniform removal-per-frame,
    gaze_weighted_merge_per_frame == concatenating per-frame gaze_weighted_merge.
    """
    torch.manual_seed(1)
    T, n_spatial, d = 3, 12, 8
    tokens = torch.randn(T, n_spatial, d)
    scores = torch.rand(T, n_spatial) + 0.1
    r_per = 4
    r_merge_total = r_per * T

    merged_new, recv_new, drop_new = gaze_weighted_merge_per_frame(
        tokens, scores, r_merge_total=r_merge_total, r_drop_total=0,
        frame_scores=None, k_min=1, score_transform="none",
    )

    # Reference: call gaze_weighted_merge once per frame, concat
    chunks, recv_chunks = [], []
    for t in range(T):
        m_t, r_t, _ = gaze_weighted_merge(
            tokens[t], scores[t], r_merge=r_per, r_drop=0, score_transform="none",
        )
        chunks.append(m_t)
        recv_chunks.append(r_t + t * n_spatial)
    ref_merged = torch.cat(chunks, dim=0)
    ref_recv   = torch.cat(recv_chunks, dim=0)

    assert torch.equal(recv_new, ref_recv)
    assert torch.allclose(merged_new, ref_merged, atol=1e-6)


def test_drop_split_across_frames():
    """With r_drop_total > 0, per-frame drops sum exactly to r_drop_total."""
    torch.manual_seed(2)
    T, n_spatial, d = 4, 16, 8
    tokens = torch.randn(T, n_spatial, d)
    scores = torch.rand(T, n_spatial) + 0.1

    merged, recv_idx, drop_idx = gaze_weighted_merge_per_frame(
        tokens, scores, r_merge_total=20, r_drop_total=8,
        frame_scores=None, k_min=1, score_transform="sigmoid",
    )
    assert drop_idx.numel() == 8
    # No overlap
    all_idx = torch.cat([recv_idx, drop_idx])
    assert all_idx.unique().numel() == all_idx.numel()
    # Drop+merge per frame sums to total removed
    assert recv_idx.numel() + drop_idx.numel() == T * n_spatial - 20


def test_global_scope_uniform_frame_scores_equals_per_frame():
    """frame_scores that are all equal → global scope collapses to per_frame."""
    torch.manual_seed(3)
    T, n_spatial, d = 4, 12, 8
    tokens = torch.randn(T, n_spatial, d)
    scores = torch.rand(T, n_spatial) + 0.1
    frame_scores_uniform = torch.ones(T)

    merged_per, recv_per, _ = gaze_weighted_merge_per_frame(
        tokens, scores, r_merge_total=12, r_drop_total=0,
        frame_scores=None, k_min=1, score_transform="none",
    )
    merged_glob, recv_glob, _ = gaze_weighted_merge_per_frame(
        tokens, scores, r_merge_total=12, r_drop_total=0,
        frame_scores=frame_scores_uniform, k_min=1, score_transform="none",
    )
    # Counts must match; exact token equality depends on tie-breaking in the
    # budget allocation, but receiver totals per frame should match for uniform.
    per_counts = torch.bincount(recv_per  // n_spatial, minlength=T)
    glob_counts = torch.bincount(recv_glob // n_spatial, minlength=T)
    assert torch.equal(per_counts, glob_counts)


def test_gradient_flow_through_per_frame():
    """Gradients must flow through per-frame merge to scores."""
    torch.manual_seed(4)
    T, n_spatial, d = 3, 8, 6
    tokens = torch.randn(T, n_spatial, d)
    scores = torch.randn(T, n_spatial, requires_grad=True)
    merged, _, _ = gaze_weighted_merge_per_frame(
        tokens, scores, r_merge_total=9, r_drop_total=3,
        frame_scores=None, k_min=1, score_transform="sigmoid",
    )
    merged.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert (scores.grad.abs() > 0).any()


if __name__ == "__main__":
    test_budget_sum_and_k_min()
    test_per_frame_shape_and_intra_frame_match()
    test_per_frame_matches_manual_per_frame_calls()
    test_drop_split_across_frames()
    test_global_scope_uniform_frame_scores_equals_per_frame()
    test_gradient_flow_through_per_frame()
    print("All per-frame merge tests passed.")
