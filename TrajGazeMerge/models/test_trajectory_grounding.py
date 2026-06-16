"""
Unit tests for TrajGazeMerge/models/trajectory_grounding.py.

Run:
    cd /workspace/trajgaze
    PYTHONPATH=. python -m TrajGazeMerge.models.test_trajectory_grounding
"""

from __future__ import annotations

import torch

from TrajGazeMerge.models.trajectory_grounding import (
    gaussian_patch_map,
    TrajectoryAnchorModule,
    receivers_to_frame_pool,
    TrajectoryReconstructionHead,
    downsample_traj_to_T_merged,
    atr_regression_loss,
    trajectory_region_receiver_mask,
)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def test_gaussian_patch_map():
    print("[test] gaussian_patch_map")
    B, T, K, grid = 2, 4, 1, 14
    # Anchor at the centre of every frame
    positions = torch.full((B, T, K, 2), 0.5)
    mask      = torch.ones (B, T, K, dtype=torch.bool)
    sigma     = torch.tensor(0.2)                # broad enough that nearest patch ≈ 0.97
    g = gaussian_patch_map(positions, mask, sigma, grid=grid)
    _check(g.shape == (B, T, grid * grid),       f"shape {g.shape}")
    peak = g.max(dim=-1).values
    _check((peak > 0.95).all() and (peak <= 1.0 + 1e-6).all(),
           f"peak ∈ (0.95, 1], got min={peak.min().item():.3f}, max={peak.max().item():.3f}")
    # Peak should be at the centre patch (or one of its 4 neighbours due to 14×14)
    centre_idx_options = {7 + 7 * grid, 6 + 7 * grid, 7 + 6 * grid, 6 + 6 * grid}
    argmax = int(g[0, 0].argmax().item())
    _check(argmax in centre_idx_options,         f"peak at centre, got idx {argmax}")

    # Invalid mask → zero everywhere
    mask_zero = torch.zeros(B, T, K, dtype=torch.bool)
    g_zero = gaussian_patch_map(positions, mask_zero, sigma, grid=grid)
    _check(torch.equal(g_zero, torch.zeros_like(g_zero)), "invalid mask → all zeros")

    # K=2 (hand) — max-combine, not sum
    positions2 = torch.zeros(1, 1, 2, 2)
    positions2[0, 0, 0] = torch.tensor([0.5, 0.5])
    positions2[0, 0, 1] = torch.tensor([0.5, 0.5])    # same point
    mask2 = torch.ones(1, 1, 2, dtype=torch.bool)
    g2 = gaussian_patch_map(positions2, mask2, torch.tensor(0.1), grid=grid)
    _check(g2.max().item() <= 1.001, "K=2 max-combined peak still ≤ 1")


def test_receivers_to_frame_pool():
    print("[test] receivers_to_frame_pool")
    n_spatial = 64
    T_merged  = 4
    n_video   = T_merged * n_spatial            # 256
    d = 8
    # Build deterministic tokens where token i has constant value i / n_video
    tokens = torch.arange(n_video, dtype=torch.float32).unsqueeze(1).expand(-1, d).clone()
    # Keep all tokens (no merge) → receiver_idx = arange(n_video)
    receiver_idx = torch.arange(n_video)
    pooled, present = receivers_to_frame_pool(tokens, receiver_idx, n_spatial, T_merged)
    _check(pooled.shape == (T_merged, d), f"pooled shape {pooled.shape}")
    _check(present.all(),                  "all frames present when keeping all tokens")
    # Frame f has tokens [f*n_spatial .. (f+1)*n_spatial), so mean = f*n_spatial + (n_spatial-1)/2
    expected_f0 = 0 * n_spatial + (n_spatial - 1) / 2
    _check(abs(pooled[0, 0].item() - expected_f0) < 1e-3,
           f"frame 0 mean ≈ {expected_f0}, got {pooled[0,0].item():.3f}")
    expected_f3 = 3 * n_spatial + (n_spatial - 1) / 2
    _check(abs(pooled[3, 0].item() - expected_f3) < 1e-3,
           f"frame 3 mean ≈ {expected_f3}, got {pooled[3,0].item():.3f}")

    # Empty frame (no receivers in frame 2) → present[2]=False
    keep = torch.cat([torch.arange(2 * n_spatial), torch.arange(3 * n_spatial, n_video)])
    pooled2, present2 = receivers_to_frame_pool(tokens[keep], keep, n_spatial, T_merged)
    _check(not bool(present2[2]),         "frame 2 empty → present=False")
    _check(bool(present2[0]) and bool(present2[1]) and bool(present2[3]),
           "frames 0,1,3 present")


def test_atr_head_and_loss():
    print("[test] TrajectoryReconstructionHead + atr_regression_loss")
    d = 32
    T_merged = 6
    pooled = torch.randn(T_merged, d)
    head = TrajectoryReconstructionHead(d)
    pred_gaze, pred_hand = head(pooled)
    _check(pred_gaze.shape == (T_merged, 2),     "pred_gaze shape")
    _check(pred_hand.shape == (T_merged, 4),     "pred_hand shape")
    _check((pred_gaze >= 0).all() and (pred_gaze <= 1).all(), "pred_gaze in [0,1]")

    # GT with some invalid frames
    gt_gaze  = torch.full((T_merged, 2), 0.5)
    gt_left  = torch.full((T_merged, 2), 0.3)
    gt_right = torch.full((T_merged, 2), 0.7)
    gaze_mask  = torch.tensor([True, True, False, True, True, False])
    left_mask  = torch.tensor([True, False, False, True, True, True])
    right_mask = torch.tensor([False, False, True, True, True, True])
    recv_present = torch.tensor([True, True, True, True, True, True])

    loss = atr_regression_loss(
        pred_gaze, pred_hand, gt_gaze, gt_left, gt_right,
        gaze_mask, left_mask, right_mask, recv_present,
    )
    _check(loss.item() > 0,                      f"loss > 0 (got {loss.item():.3f})")
    _check(loss.requires_grad,                   "loss is differentiable through head")
    loss.backward()
    _check(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()),
           "gradient flows into head parameters")


def test_downsample_traj():
    print("[test] downsample_traj_to_T_merged")
    T_traj = 128
    T_merged = 64
    x = torch.arange(T_traj, dtype=torch.float32)
    ds = downsample_traj_to_T_merged(x, T_merged)
    _check(ds.shape == (T_merged,), f"shape {ds.shape}")
    _check(ds[0].item() == 0.0,                          "first frame = 0")
    _check(ds[-1].item() <= T_traj - 1,                  "last frame in range")
    # Should be monotone non-decreasing for proportional mapping
    _check(((ds[1:] - ds[:-1]) >= 0).all().item(),       "monotone")


def test_trajectory_region_receiver_mask():
    print("[test] trajectory_region_receiver_mask")
    n_spatial = 64               # 8×8
    T_merged  = 4
    T_traj    = 16
    n_video   = T_merged * n_spatial

    # GT trajectory: all frames have a single anchor at the upper-left patch (0.0625, 0.0625)
    # (corresponds to patch index 0 on the 8×8 grid)
    targets = torch.full((T_traj, 1, 2), 0.0625)
    target_mask = torch.ones(T_traj, 1, dtype=torch.bool)

    # Keep all tokens — for each frame, only the patch near (0,0) corner should match
    receiver_idx = torch.arange(n_video)
    is_in_region = trajectory_region_receiver_mask(
        receiver_idx, T_merged, n_spatial, targets, target_mask, radius=0.1,
    )
    # On 8×8: only patch (0,0) per frame is within 0.1 of (0.0625, 0.0625)
    n_match = is_in_region.sum().item()
    _check(n_match == T_merged,
           f"exactly 1 receiver per frame in region, got {n_match}")
    # First receiver of each frame should be in-region
    expected = (receiver_idx % n_spatial == 0)
    _check(torch.equal(is_in_region, expected),
           "in-region mask hits patch index 0 in each frame")

    # Invalid mask → no matches
    targets_invalid = torch.zeros(T_traj, 1, 2)
    target_mask_invalid = torch.zeros(T_traj, 1, dtype=torch.bool)
    none_match = trajectory_region_receiver_mask(
        receiver_idx, T_merged, n_spatial, targets_invalid, target_mask_invalid, radius=1.0,
    )
    _check(not none_match.any(), "invalid mask → no matches even at radius=1.0")


def test_trajectory_anchor_module():
    print("[test] TrajectoryAnchorModule")
    grid = 14
    B, T = 2, 5
    scores = torch.ones(B, T, grid * grid) * 0.5
    traj_batch = {
        "gaze_pos":   torch.full((B, T, 2), 0.5),
        "gaze_mask":  torch.ones(B, T, dtype=torch.bool),
        "left_pos":   torch.full((B, T, 2), 0.3),
        "left_mask":  torch.ones(B, T, dtype=torch.bool),
        "right_pos":  torch.full((B, T, 2), 0.7),
        "right_mask": torch.ones(B, T, dtype=torch.bool),
    }
    mod = TrajectoryAnchorModule(grid=grid)

    # At init: amp gates are tanh(0)=0 → prior should be identity (output == input)
    out0 = mod(scores, traj_batch)
    _check(torch.allclose(out0, scores, atol=1e-5),
           "amp_*=0 at init → identity (output == scores)")

    # Open the gates manually and check that anchor centres get amplified
    with torch.no_grad():
        mod.amp_gaze_raw.fill_(2.0)   # tanh(2) ≈ 0.96, * amp_max(1.5) ≈ 1.45
        mod.amp_hand_raw.fill_(2.0)
    out1 = mod(scores, traj_batch)
    _check((out1.max(dim=-1).values > scores.max(dim=-1).values * 1.5).all(),
           f"open amp → centres boosted (got max ratio {(out1.max(-1).values / scores.max(-1).values).min().item():.2f})")

    # Negative amp dampens (sanity: sign is learnable, no clamp dead zone)
    with torch.no_grad():
        mod.amp_gaze_raw.fill_(-2.0)
        mod.amp_hand_raw.fill_(-2.0)
    out_neg = mod(scores, traj_batch)
    _check((out_neg.min(dim=-1).values < scores.min(dim=-1).values + 1e-5).all(),
           "negative amp dampens at anchor centres (clamp removed)")

    # Differentiability — exercise both positive and negative regimes
    with torch.no_grad():
        mod.amp_gaze_raw.fill_(0.5)
        mod.amp_hand_raw.fill_(-0.5)
    out2 = mod(scores, traj_batch)
    out2.sum().backward()
    _check(mod.amp_gaze_raw.grad is not None and mod.amp_gaze_raw.grad.abs() > 0,
           "amp_gaze_raw receives gradient (positive regime)")
    _check(mod.amp_hand_raw.grad is not None and mod.amp_hand_raw.grad.abs() > 0,
           "amp_hand_raw receives gradient (negative regime — would be zero w/ old clamp)")
    _check(mod.log_sigma_gaze.grad is not None,
           "log_sigma_gaze receives gradient")


def main():
    test_gaussian_patch_map()
    test_receivers_to_frame_pool()
    test_atr_head_and_loss()
    test_downsample_traj()
    test_trajectory_region_receiver_mask()
    test_trajectory_anchor_module()
    print("\nAll trajectory-grounding unit tests passed.")


if __name__ == "__main__":
    main()
