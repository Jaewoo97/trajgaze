"""
Gaze-only SpatiotemporalEncoder for ablation study.
Identical to SpatiotemporalEncoder except only 1 gaze token per frame
(no left/right hand or interaction tokens).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    D_TRAJ, D_ENC, D_VIS, D_QUERY,
    N_HEADS_L2, N_LAYERS_L2, N_PATCHES,
    SinusoidalPE, FiLM, VisualTrajectoryFusion,
)

N_TOKENS_GAZE = 1


class TrajectoryTokenizerGazeOnly(nn.Module):
    """Projects only gaze features → 1 token per frame."""
    DIM_GAZE_RAW = 3  # gaze_pos(2) + gaze_speed(1)

    def __init__(self, d_model: int = D_TRAJ):
        super().__init__()
        self.d_model = d_model
        self.missing_gaze = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj_gaze = nn.Linear(self.DIM_GAZE_RAW, d_model)
        self.norm_gaze = nn.LayerNorm(d_model)

    def forward(self, batch: dict) -> torch.Tensor:
        gaze_feat = torch.cat([batch["gaze_pos"], batch["gaze_speed"]], dim=-1)
        tok_gaze = self.norm_gaze(self.proj_gaze(gaze_feat))
        mask_g = batch["gaze_mask"].unsqueeze(-1).float()
        return mask_g * tok_gaze + (1 - mask_g) * self.missing_gaze.view(1, 1, -1)


class SpatiotemporalEncoderGazeOnly(nn.Module):
    """
    Gaze-only spatiotemporal encoder: 1 token per frame.
    No intra-frame attention (only 1 token), all other modules identical
    to SpatiotemporalEncoder (same inter-frame transformer, FiLM, VT-fusion).

    Returns:
        patch_scores : (B, 196)
        context      : (B, T, 1, D_ENC)
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
        self.tokenizer  = TrajectoryTokenizerGazeOnly(d_traj)
        self.proj       = nn.Linear(d_traj, d_enc)
        self.pe         = SinusoidalPE(d_enc)
        self.inter_frame = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_enc, nhead=n_heads_l2, dim_feedforward=d_enc * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=n_layers_l2,
        )
        self.film       = FiLM(d_query, d_enc)
        self.vt_fusion  = VisualTrajectoryFusion(d_enc, d_vis)
        self.norm_out   = nn.LayerNorm(d_enc)

        # Trajectory-only fallback scorer (when no visual features)
        self.traj_patch_embed = nn.Embedding(N_PATCHES, d_enc)
        self.register_buffer("patch_idx", torch.arange(N_PATCHES))
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
        """
        Returns:
            patch_scores : (B, 196)
            context      : (B, T, 1, D_ENC)
        """
        B = batch["gaze_mask"].shape[0]
        T = batch["gaze_mask"].shape[1]

        # 1. Tokenize gaze only → (B, T, D_TRAJ)
        tok_g = self.tokenizer(batch)

        # 2. Project to D_ENC + PE (no intra-frame block: only 1 token per frame)
        x = self.proj(tok_g.reshape(B * T, -1)).reshape(B, T, D_ENC)
        x = self.pe(x)  # (B, T, D_ENC) — T*N_TOKENS_GAZE = T*1 = T

        # 3. Inter-frame transformer
        x = self.inter_frame(x)  # (B, T, D_ENC)

        # 4. FiLM: query conditions trajectory context
        if query_emb is None:
            query_emb = torch.zeros(B, self.film.proj_scale.in_features, device=x.device)
        x = self.film(x, query_emb)  # (B, T, D_ENC)

        # 5. Visual-Trajectory Fusion
        if visual_feat is not None:
            enriched_context, patch_scores = self.vt_fusion(x, visual_feat)
        else:
            enriched_context = x
            patch_scores = self._trajectory_only_scores(x)

        # 6. Reshape for decoders: (B, T, 1, D_ENC)
        context = self.norm_out(enriched_context).reshape(B, T, N_TOKENS_GAZE, D_ENC)

        return patch_scores, context
