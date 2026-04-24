"""
Unit tests for the trajectory hint-token injection helper (축 5a).
"""

from __future__ import annotations

import torch

from TrajGazeMerge.models.model import (
    TrajHintProjection,
    _insert_hint_tokens,
)


def test_hint_proj_zero_init_produces_zero_output():
    """Default zero-init on projection ⇒ hint embeddings are numerically zero
    at start (ensures safe warm-start when flag is first enabled)."""
    hp = TrajHintProjection(in_dim=32, out_dim=64, n_tokens=4)
    x = torch.randn(32)
    y = hp(x)
    assert y.shape == (4, 64)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_insert_hint_tokens_shapes_and_positions():
    """K=0 case is a caller-side guard; here we exercise K>0 path and confirm
    that the inserted block carries consecutive M-RoPE positions and that
    subsequent positions are shifted by +K in all 3 M-RoPE dims."""
    L, d, K = 20, 8, 3
    inputs_embeds  = torch.randn(1, L, d)
    attention_mask = torch.ones(1, L, dtype=torch.long)
    position_ids   = torch.arange(L).view(1, 1, L).expand(3, 1, L).contiguous()
    hint = torch.randn(K, d)

    ins_at = 7
    new_emb, new_mask, new_pos = _insert_hint_tokens(
        inputs_embeds, attention_mask, position_ids, hint, ins_at,
    )
    assert new_emb.shape  == (1, L + K, d)
    assert new_mask.shape == (1, L + K)
    assert new_pos.shape  == (3, 1, L + K)

    # Before insertion unchanged
    assert torch.allclose(new_emb[:, :ins_at], inputs_embeds[:, :ins_at])
    # Inserted embeddings match `hint`
    assert torch.allclose(new_emb[0, ins_at:ins_at + K], hint)
    # After insertion: same content, shifted positions
    assert torch.allclose(new_emb[:, ins_at + K:], inputs_embeds[:, ins_at:])
    # Positions for hints: anchor + 1..K across all 3 dims
    anchor = position_ids[:, :, ins_at - 1]                      # (3, 1)
    expected = anchor.unsqueeze(-1) + torch.arange(1, K + 1).view(1, 1, K)
    assert torch.equal(new_pos[:, :, ins_at:ins_at + K], expected)
    # Positions after: shifted by +K
    assert torch.equal(
        new_pos[:, :, ins_at + K:], position_ids[:, :, ins_at:] + K
    )


if __name__ == "__main__":
    test_hint_proj_zero_init_produces_zero_output()
    test_insert_hint_tokens_shapes_and_positions()
    print("All hint-injection tests passed.")
