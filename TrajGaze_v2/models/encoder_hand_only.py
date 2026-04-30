"""
SpatiotemporalEncoderHandOnly: ablation using only hand trajectory (no gaze).

3 tokens/frame:
  - left hand  : [left_pos(2), left_vel(2), left_mask(1)]  → Linear(5, d_traj)
  - right hand : [right_pos(2), right_vel(2), right_mask(1)] → Linear(5, d_traj)
  - bimanual   : [d_lr(2), ||d_lr||(1), v_rel_lr(2), speed_l(1), speed_r(1)] → Linear(7, d_traj)

Gaze fields in the traj dict are silently ignored.
Visual input (DINOv2 patches) is retained unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    D_TRAJ, D_ENC, D_VIS, D_QUERY,
    N_HEADS_L1, N_HEADS_L2, N_LAYERS_L1, N_LAYERS_L2,
    N_PATCHES,
    SinusoidalPE, FiLM, VisualTrajectoryFusion,
)

N_TOKENS_HAND = 3   # left, right, bimanual-interaction


class TrajectoryTokenizerHandOnly(nn.Module):
    DIM_HAND_RAW     = 5   # pos(2) + vel(2) + mask(1)
    DIM_BIMANUAL_RAW = 7   # d_lr(2) + ||d_lr||(1) + v_rel_lr(2) + speed_l(1) + speed_r(1)

    def __init__(self, d_model: int = D_TRAJ):
        super().__init__()
        self.d_model = d_model
        self.missing_left  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.missing_right = nn.Parameter(torch.randn(d_model) * 0.02)

        self.proj_left     = nn.Linear(self.DIM_HAND_RAW,     d_model)
        self.proj_right    = nn.Linear(self.DIM_HAND_RAW,     d_model)
        self.proj_bimanual = nn.Linear(self.DIM_BIMANUAL_RAW, d_model)

        self.norm_left     = nn.LayerNorm(d_model)
        self.norm_right    = nn.LayerNorm(d_model)
        self.norm_bimanual = nn.LayerNorm(d_model)

    def forward(self, batch: dict):
        """
        batch keys used: left_pos, left_vel, left_mask, right_pos, right_vel, right_mask
        gaze keys are present but ignored.
        Returns: tok_left, tok_right, tok_bimanual — each (B, T, d_model)
        """
        B, T = batch["left_mask"].shape
        device = batch["left_pos"].device

        # Left hand token
        left_feat = torch.cat([
            batch["left_pos"],
            batch["left_vel"],
            batch["left_mask"].unsqueeze(-1).float(),
        ], dim=-1)   # (B, T, 5)
        tok_left = self.norm_left(self.proj_left(left_feat))
        mask_l   = batch["left_mask"].unsqueeze(-1).float()
        tok_left = mask_l * tok_left + (1 - mask_l) * self.missing_left.view(1, 1, -1)

        # Right hand token
        right_feat = torch.cat([
            batch["right_pos"],
            batch["right_vel"],
            batch["right_mask"].unsqueeze(-1).float(),
        ], dim=-1)   # (B, T, 5)
        tok_right = self.norm_right(self.proj_right(right_feat))
        mask_r    = batch["right_mask"].unsqueeze(-1).float()
        tok_right = mask_r * tok_right + (1 - mask_r) * self.missing_right.view(1, 1, -1)

        # Bimanual interaction token: hand-to-hand geometry (no gaze)
        both_visible = (batch["left_mask"] & batch["right_mask"]).unsqueeze(-1).float()  # (B,T,1)
        d_lr   = batch["left_pos"] - batch["right_pos"]              # (B, T, 2)
        d_norm = d_lr.norm(dim=-1, keepdim=True)                     # (B, T, 1)
        v_rel  = batch["left_vel"] - batch["right_vel"]              # (B, T, 2)
        spd_l  = batch["left_vel"].norm(dim=-1, keepdim=True)        # (B, T, 1)
        spd_r  = batch["right_vel"].norm(dim=-1, keepdim=True)       # (B, T, 1)

        bimanual_feat = torch.cat([d_lr, d_norm, v_rel, spd_l, spd_r], dim=-1)  # (B, T, 7)
        bimanual_feat = bimanual_feat * both_visible   # zero out when either hand absent
        tok_bimanual  = self.norm_bimanual(self.proj_bimanual(bimanual_feat))

        return tok_left, tok_right, tok_bimanual


class IntraFrameBlockHandOnly(nn.Module):
    """Self-attention over the 3 hand tokens within each frame."""
    def __init__(self, d_model: int = D_TRAJ, n_heads: int = N_HEADS_L1,
                 n_layers: int = N_LAYERS_L1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])

    def forward(self, tok_left, tok_right, tok_bimanual) -> torch.Tensor:
        B, T, D = tok_left.shape
        x = torch.stack([tok_left, tok_right, tok_bimanual], dim=2)  # (B, T, 3, D)
        x = x.reshape(B * T, N_TOKENS_HAND, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, T, N_TOKENS_HAND, D)


class SpatiotemporalEncoderHandOnly(nn.Module):
    """
    Hand-only spatiotemporal encoder.

    Identical pipeline to SpatiotemporalEncoder but:
      - 3 tokens/frame (left, right, bimanual) instead of 4
      - gaze fields in the batch dict are silently ignored
      - visual input (DINOv2) retained unchanged
    """

    def __init__(
        self,
        d_traj:      int = D_TRAJ,
        d_enc:       int = D_ENC,
        d_vis:       int = D_VIS,
        d_query:     int = D_QUERY,
        n_layers_l2: int = N_LAYERS_L2,
        n_heads_l2:  int = N_HEADS_L2,
    ):
        super().__init__()
        self.tokenizer   = TrajectoryTokenizerHandOnly(d_traj)
        self.intra_frame = IntraFrameBlockHandOnly(d_traj)
        self.proj        = nn.Linear(d_traj, d_enc)
        self.pe          = SinusoidalPE(d_enc)
        self.inter_frame = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_enc, nhead=n_heads_l2, dim_feedforward=d_enc * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=n_layers_l2,
        )
        self.film     = FiLM(d_query, d_enc)
        self.vt_fusion = VisualTrajectoryFusion(d_enc, d_vis)
        self.norm_out  = nn.LayerNorm(d_enc)

        # Trajectory-only fallback (no visual features)
        self.traj_patch_embed = nn.Embedding(N_PATCHES, d_enc)
        patch_idx = torch.arange(N_PATCHES)
        self.register_buffer("patch_idx", patch_idx)
        self.traj_score_head = nn.Sequential(
            nn.Linear(d_enc, 1),
            nn.Sigmoid(),
        )

    def _trajectory_only_scores(self, traj_context: torch.Tensor) -> torch.Tensor:
        B = traj_context.shape[0]
        scale = D_ENC ** -0.5
        q = self.traj_patch_embed(self.patch_idx).unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(q, traj_context.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn, traj_context)
        return self.traj_score_head(attended).squeeze(-1)

    def forward(
        self,
        batch:       dict,
        query_emb:   Optional[torch.Tensor] = None,
        visual_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = batch["left_mask"].shape[0]
        T = batch["left_mask"].shape[1]

        tok_l, tok_r, tok_b = self.tokenizer(batch)                  # each (B, T, D_TRAJ)
        enriched = self.intra_frame(tok_l, tok_r, tok_b)             # (B, T, 3, D_TRAJ)

        x = self.proj(enriched.reshape(B * T * N_TOKENS_HAND, -1))
        x = x.reshape(B, T * N_TOKENS_HAND, D_ENC)
        x = self.pe(x)                                               # (B, T*3, D_ENC)
        x = self.inter_frame(x)                                      # (B, T*3, D_ENC)

        if query_emb is None:
            query_emb = torch.zeros(B, self.film.proj_scale.in_features, device=x.device)
        x = self.film(x, query_emb)

        if visual_feat is not None:
            enriched_context, patch_scores = self.vt_fusion(x, visual_feat)
        else:
            enriched_context = x
            patch_scores = self._trajectory_only_scores(x)

        context = self.norm_out(enriched_context).reshape(B, T, N_TOKENS_HAND, D_ENC)
        return patch_scores, context
