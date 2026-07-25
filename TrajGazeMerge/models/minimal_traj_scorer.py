"""Minimal trajectory-prediction-based token scorer.

Philosophy: same as TAS — predict gaze/hand trajectories from visual
features and use the attention map over patches as a per-token importance
score for gaze_weighted_merge. Difference: ~200K params (vs TAS's 35.8M)
and reuses Qwen's existing video_embeds — no separate ViT/DINOv2 forward.

Architecture (per item):
    video_embeds (N=T*S, d_in) [Qwen's already-computed tokens]
        → Linear d_in → d_hidden        (one projection)
        → LayerNorm
        → cross-attention with 3 learnable queries (gaze, hand_L, hand_R)
        → per-query (T, S) attention map
              coords = attention-weighted sum of V → small linear head → (x,y) sigmoid
              scores = max over queries → (N,) importance

For Qwen2.5-VL-7B: d_in = 3584, default d_hidden = 128 → ~470K params.
With d_hidden = 64 → ~240K params.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MinimalTrajScorer(nn.Module):
    """Tiny gaze/hand predictor that scores Qwen tokens by predictor attention."""

    def __init__(self, d_in: int = 3584, d_hidden: int = 128):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden

        self.proj = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

        # 3 learnable query tokens: gaze, left_hand, right_hand
        self.queries = nn.Parameter(torch.randn(3, d_hidden) * 0.02)

        self.k_proj = nn.Linear(d_hidden, d_hidden)
        self.v_proj = nn.Linear(d_hidden, d_hidden)

        self.coord_head = nn.Linear(d_hidden, 2)
        nn.init.zeros_(self.coord_head.bias)
        nn.init.normal_(self.coord_head.weight, std=0.01)

    def forward(self, video_embeds: torch.Tensor, T: int, S: int):
        """
        Args:
            video_embeds: (N, d_in) where N = T * S
            T, S: temporal and spatial dims (T = grid_thw[0,0], S = N // T)
        Returns:
            coords: (T, 3, 2)  predicted (x,y) in [0,1] per query per frame
            scores: (N,)       per-token importance from query attention
        """
        N = video_embeds.shape[0]
        assert N == T * S, f"video_embeds N={N} != T*S={T}*{S}={T*S}"

        feat = self.proj(video_embeds.float()).view(T, S, self.d_hidden)
        feat = self.norm(feat)

        K = self.k_proj(feat)
        V = self.v_proj(feat)

        # cross-attention: queries (3, d_h) attend to (T, S, d_h)
        attn_logits = torch.einsum("qd,tsd->qts", self.queries, K) / (self.d_hidden ** 0.5)
        attn_weights = attn_logits.softmax(dim=-1)  # (3, T, S)

        ctx = torch.einsum("qts,tsd->qtd", attn_weights, V)  # (3, T, d_h)
        coords = self.coord_head(ctx).sigmoid()              # (3, T, 2) ∈ [0,1]
        coords = coords.permute(1, 0, 2).contiguous()        # (T, 3, 2)

        # importance score per token = max over the 3 query attention maps
        scores = attn_weights.amax(dim=0).reshape(N)

        return coords, scores

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
