"""
SpatiotemporalEncoderTemporalHandOnly — temporal, hand-only ablation.

Identical pipeline to SpatiotemporalEncoderTemporal except:
  - 3 hand tokens per frame (left, right, bimanual) — gaze inputs ignored
  - IntraFrameBlock over 3 tokens
  - TemporalVisualTrajFusion: Q=(B*T, 3, D) instead of (B*T, 4, D)

Outputs:
  per_frame_scores : (B, T, 196)
  context          : (B, T, 3, D_ENC)
  enc_attn         : (B, T, 3, 196) or None
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    D_TRAJ, D_ENC, D_VIS, D_QUERY,
    N_HEADS_L1, N_HEADS_L2, N_LAYERS_L1, N_LAYERS_L2,
    N_PATCHES,
    SinusoidalPE, FiLM,
)
from .encoder_temporal import TemporalVisualTrajFusion, PatchTemporalBranch

N_TOKENS_HAND = 3


class TrajectoryTokenizerHandOnlyTemporal(nn.Module):
    DIM_HAND_RAW     = 5   # pos(2) + vel(2) + mask(1)
    DIM_BIMANUAL_RAW = 7   # d_lr(2) + ||d_lr||(1) + v_rel_lr(2) + speed_l(1) + speed_r(1)

    def __init__(self, d_model: int = D_TRAJ):
        super().__init__()
        self.missing_left  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.missing_right = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj_left     = nn.Linear(self.DIM_HAND_RAW,     d_model)
        self.proj_right    = nn.Linear(self.DIM_HAND_RAW,     d_model)
        self.proj_bimanual = nn.Linear(self.DIM_BIMANUAL_RAW, d_model)
        self.norm_left     = nn.LayerNorm(d_model)
        self.norm_right    = nn.LayerNorm(d_model)
        self.norm_bimanual = nn.LayerNorm(d_model)

    def forward(self, batch: dict):
        left_feat = torch.cat([batch["left_pos"], batch["left_vel"],
                                batch["left_mask"].unsqueeze(-1).float()], dim=-1)
        tok_left = self.norm_left(self.proj_left(left_feat))
        mask_l   = batch["left_mask"].unsqueeze(-1).float()
        tok_left = mask_l * tok_left + (1 - mask_l) * self.missing_left.view(1, 1, -1)

        right_feat = torch.cat([batch["right_pos"], batch["right_vel"],
                                 batch["right_mask"].unsqueeze(-1).float()], dim=-1)
        tok_right = self.norm_right(self.proj_right(right_feat))
        mask_r    = batch["right_mask"].unsqueeze(-1).float()
        tok_right = mask_r * tok_right + (1 - mask_r) * self.missing_right.view(1, 1, -1)

        both_visible  = (batch["left_mask"] & batch["right_mask"]).unsqueeze(-1).float()
        d_lr          = batch["left_pos"] - batch["right_pos"]
        d_norm        = d_lr.norm(dim=-1, keepdim=True)
        v_rel         = batch["left_vel"] - batch["right_vel"]
        spd_l         = batch["left_vel"].norm(dim=-1, keepdim=True)
        spd_r         = batch["right_vel"].norm(dim=-1, keepdim=True)
        bimanual_feat = torch.cat([d_lr, d_norm, v_rel, spd_l, spd_r], dim=-1) * both_visible
        tok_bimanual  = self.norm_bimanual(self.proj_bimanual(bimanual_feat))

        return tok_left, tok_right, tok_bimanual


class IntraFrameBlockHandOnlyTemporal(nn.Module):
    """Self-attention over 3 hand tokens within each frame."""
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
        x = torch.stack([tok_left, tok_right, tok_bimanual], dim=2).reshape(B * T, N_TOKENS_HAND, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, T, N_TOKENS_HAND, D)


class SpatiotemporalEncoderTemporalHandOnly(nn.Module):
    """
    Hand-only temporal encoder: 3 tokens per frame, per-frame (T, 196) scores.
    Gaze fields in batch are silently ignored.
    """

    def __init__(
        self,
        d_traj:                    int  = D_TRAJ,
        d_enc:                     int  = D_ENC,
        d_vis:                     int  = D_VIS,
        d_query:                   int  = D_QUERY,
        n_layers_l2:               int  = N_LAYERS_L2,
        n_heads_l2:                int  = N_HEADS_L2,
        use_patch_temporal_branch: bool = False,
    ):
        super().__init__()
        self.tokenizer   = TrajectoryTokenizerHandOnlyTemporal(d_traj)
        self.intra_frame = IntraFrameBlockHandOnlyTemporal(d_traj)
        self.proj        = nn.Linear(d_traj, d_enc)
        self.pe          = SinusoidalPE(d_enc, max_len=4096)
        self.inter_frame = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_enc, nhead=n_heads_l2, dim_feedforward=d_enc * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=n_layers_l2,
        )
        self.film      = FiLM(d_query, d_enc)
        self.vt_fusion = TemporalVisualTrajFusion(d_enc, d_vis, n_tokens=N_TOKENS_HAND)
        self.norm_out  = nn.LayerNorm(d_enc)

        self.traj_patch_embed = nn.Embedding(N_PATCHES, d_enc)
        self.register_buffer("patch_idx", torch.arange(N_PATCHES))
        self.traj_score_head  = nn.Sequential(nn.Linear(d_enc, 1), nn.Sigmoid())

        # E1: 196 patch queries cross-attend to inter_frame output per frame
        self.use_patch_temporal_branch = use_patch_temporal_branch
        if use_patch_temporal_branch:
            self.patch_temporal_branch = PatchTemporalBranch(
                d_enc=d_enc, n_heads=n_heads_l2,
                n_patches=N_PATCHES, n_tokens=N_TOKENS_HAND,
            )

    def _trajectory_only_scores(self, traj_context: torch.Tensor) -> torch.Tensor:
        B, T, N_tok, D = traj_context.shape
        ctx_flat = traj_context.reshape(B * T, N_tok, D)
        q = self.traj_patch_embed(self.patch_idx).unsqueeze(0).expand(B * T, -1, -1)
        attn     = torch.bmm(q, ctx_flat.transpose(1, 2)) * (D ** -0.5)
        attn     = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn, ctx_flat)
        return self.traj_score_head(attended).squeeze(-1).reshape(B, T, N_PATCHES)

    def forward(
        self,
        batch:       dict,
        query_emb:   Optional[torch.Tensor] = None,
        visual_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = batch["left_mask"].shape[0]
        T = batch["left_mask"].shape[1]

        tok_l, tok_r, tok_b = self.tokenizer(batch)
        enriched = self.intra_frame(tok_l, tok_r, tok_b)             # (B, T, 3, D_traj)

        x = self.proj(enriched.reshape(B * T * N_TOKENS_HAND, -1)).reshape(B, T * N_TOKENS_HAND, D_ENC)
        x = self.pe(x)

        # Save inter_frame output as side-channel for patch_temporal_branch
        x_inter = self.inter_frame(x)   # (B, T*3, D_enc)

        if query_emb is None:
            query_emb = torch.zeros(B, self.film.proj_scale.in_features, device=x_inter.device)
        x = self.film(x_inter, query_emb)

        x_framed = x.reshape(B, T, N_TOKENS_HAND, D_ENC)

        if visual_feat is not None:
            per_frame_scores, enriched_context, enc_attn = self.vt_fusion(x_framed, visual_feat)
        else:
            enriched_context = x_framed
            per_frame_scores = self._trajectory_only_scores(x_framed)
            enc_attn         = None

        # E1: per-patch temporal modulation from inter_frame output
        if self.use_patch_temporal_branch:
            patch_temporal_scores = self.patch_temporal_branch(x_inter, B, T)
            per_frame_scores = per_frame_scores * patch_temporal_scores

        context = self.norm_out(enriched_context)
        return per_frame_scores, context, enc_attn
