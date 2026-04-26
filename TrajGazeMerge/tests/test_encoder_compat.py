"""Regression test for SpatiotemporalEncoder.forward backward-compat.

Verifies:
1. Legacy 2-tuple return is numerically identical between return_extras=False
   (default) and explicit False — guarantees no accidental change to existing
   callers (eval_per_task, train_merge_lora).
2. return_extras=True returns (patch_scores, context, extras) with
   - patch_scores numerically identical to the legacy return
   - context numerically identical to the legacy return
   - extras["frame_attend"] shape == (B, T), values in [0, 1]
   - extras["traj_embeds"] shape == (B, T, 4, D_enc) and equal to context
3. Trajectory-only fallback (visual_feat=None) returns frame_attend_src="uniform"
   with uniform 1/T values, no exception.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/workspace/trajgaze_msk")

from TrajGaze_v2.models.encoder import SpatiotemporalEncoder, D_ENC, N_TOKENS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_batch(B: int, T: int) -> dict:
    return {
        "gaze_pos":     torch.randn(B, T, 2),
        "gaze_speed":   torch.randn(B, T, 1),
        "gaze_mask":    torch.ones(B, T, dtype=torch.bool),
        "left_pos":     torch.randn(B, T, 2),
        "left_vel":     torch.randn(B, T, 2),
        "left_mask":    torch.ones(B, T, dtype=torch.bool),
        "right_pos":    torch.randn(B, T, 2),
        "right_vel":    torch.randn(B, T, 2),
        "right_mask":   torch.ones(B, T, dtype=torch.bool),
        "d_left":       torch.randn(B, T, 3),
        "d_right":      torch.randn(B, T, 3),
        "v_rel_left":   torch.randn(B, T, 2),
        "v_rel_right":  torch.randn(B, T, 2),
        "convergence":  torch.randn(B, T),
        "lead_lag":     torch.randn(B, T),
    }


def test_legacy_parity_with_default():
    """Default and explicit False give numerically identical 2-tuples."""
    torch.manual_seed(0)
    enc = SpatiotemporalEncoder().eval()
    B, T = 2, 8
    batch = _make_batch(B, T)
    visual = torch.randn(B, 196, 256)
    query  = torch.randn(B, 128)

    with torch.no_grad():
        ps_a, ctx_a = enc(batch, query, visual)
        ps_b, ctx_b = enc(batch, query, visual, return_extras=False)

    assert torch.allclose(ps_a, ps_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ctx_a, ctx_b, atol=1e-6, rtol=1e-6)


def test_extras_shapes_and_parity():
    """return_extras=True keeps patch_scores+context bit-equal to legacy."""
    torch.manual_seed(1)
    enc = SpatiotemporalEncoder().eval()
    B, T = 2, 8
    batch = _make_batch(B, T)
    visual = torch.randn(B, 196, 256)
    query  = torch.randn(B, 128)

    with torch.no_grad():
        ps_a, ctx_a = enc(batch, query, visual)
        ps_b, ctx_b, extras = enc(batch, query, visual, return_extras=True)

    assert torch.allclose(ps_a, ps_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ctx_a, ctx_b, atol=1e-6, rtol=1e-6)

    fa = extras["frame_attend"]
    te = extras["traj_embeds"]
    assert fa.shape == (B, T)
    assert te.shape == (B, T, N_TOKENS, D_ENC)
    assert torch.allclose(te, ctx_b)
    assert (fa >= 0).all() and (fa <= 1).all(), \
        "frame_attend (max-attn-mean) must lie in [0,1]"
    assert extras["frame_attend_src"] == "fusion"


def test_traj_only_fallback():
    """Visual=None: frame_attend uniform 1/T, src='uniform', no exception."""
    torch.manual_seed(2)
    enc = SpatiotemporalEncoder().eval()
    B, T = 2, 6
    batch = _make_batch(B, T)
    query = torch.randn(B, 128)

    with torch.no_grad():
        ps, ctx, extras = enc(batch, query, None, return_extras=True)

    assert ps.shape == (B, 196)
    assert ctx.shape == (B, T, N_TOKENS, D_ENC)
    assert extras["frame_attend_src"] == "uniform"
    expected = torch.full((B, T), 1.0 / T)
    assert torch.allclose(extras["frame_attend"], expected, atol=1e-6)


if __name__ == "__main__":
    test_legacy_parity_with_default()
    test_extras_shapes_and_parity()
    test_traj_only_fallback()
    print("All encoder compat tests passed.")
