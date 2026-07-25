"""RGB-only trajectory-salience predictor — the distillation student head.

The teacher salience (`_traj_scores` in the complement trainer) reads the gaze/hand
streams through a frozen TAS encoder and produces a per-token behavioural-salience
field over the video tokens. That field is what selects the 3% "trajectory complement"
re-added on top of VisionZip's 7% content set. Because it needs gaze/hand at
inference, the deployed model would require an eye-tracker.

This module removes that dependency. It predicts the same per-token salience from
ONLY VisionZip's content-side features — the video token embeddings, the ViT
importance scores, and frame position — with no gaze/hand input. Trained by
distillation (BCE toward the teacher's top-k membership), it lets the student pick
the complement itself, so no gaze/hand stream is read at test time.

It is a small head over FROZEN vision features (video_embeds come from the frozen
ViT under no_grad), so its only gradient signal is the selection-distillation loss;
the LoRA is trained separately by the task cross-entropy on the student-selected
tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajSaliencePredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, use_frame_context: bool = True):
        super().__init__()
        self.use_frame_context = use_frame_context
        self.tok_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        # ViT importance (RGB-derived, available at inference) as a scalar feature.
        self.attn_proj = nn.Linear(1, hidden)
        if use_frame_context:
            # Per-frame mean of token embeddings → cheap global context (O(N), no
            # N×N attention over the ~10k+ tokens).
            self.ctx_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video_embeds: torch.Tensor, attn_scores: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        """video_embeds (N, D), attn_scores (N,), grid_thw (1, 3) = [T_merged, H, W].
        Returns per-token salience logits (N,). No gaze/hand input."""
        N, D = video_embeds.shape
        x = video_embeds.float()
        h = self.tok_proj(x)                                    # (N, hidden)

        a = attn_scores.float().view(N, 1)
        a = (a - a.mean()) / (a.std() + 1e-5)                   # per-video standardize
        h = h + self.attn_proj(a)

        if self.use_frame_context:
            T = int(grid_thw[0, 0].item())
            n_spatial = N // max(1, T)
            if T * n_spatial == N:                              # regular grid → add frame ctx
                ctx = x.view(T, n_spatial, D).mean(dim=1)       # (T, D)
                ctx = self.ctx_proj(ctx)                        # (T, hidden)
                ctx = ctx.unsqueeze(1).expand(T, n_spatial, -1).reshape(N, -1)
                h = h + ctx

        return self.head(h).squeeze(-1)                         # (N,)
