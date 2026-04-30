"""
SpatiotemporalEncoderTemporalGazeOnly — temporal, gaze-only ablation.

Identical pipeline to SpatiotemporalEncoderTemporal except:
  - 1 gaze token per frame (no hand or interaction tokens)
  - No IntraFrameBlock (only 1 token per frame, no cross-modal interaction)
  - TemporalVisualTrajFusion: Q=(B*T, 1, D) instead of (B*T, 4, D)

Outputs:
  per_frame_scores : (B, T, 196)
  context          : (B, T, 1, D_ENC)
  enc_attn         : (B, T, 1, 196) or None
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    D_TRAJ, D_ENC, D_VIS, D_QUERY,
    N_HEADS_L2, N_LAYERS_L2, N_PATCHES,
    SinusoidalPE, FiLM,
)
from .encoder_temporal import TemporalVisualTrajFusion

N_TOKENS_GAZE = 1


class TrajectoryTokenizerGazeOnlyTemporal(nn.Module):
    """Projects gaze features → 1 token per frame."""
    DIM_GAZE_RAW = 3  # gaze_pos(2) + gaze_speed(1)

    def __init__(self, d_model: int = D_TRAJ):
        super().__init__()
        self.missing_gaze = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj_gaze    = nn.Linear(self.DIM_GAZE_RAW, d_model)
        self.norm_gaze    = nn.LayerNorm(d_model)

    def forward(self, batch: dict) -> torch.Tensor:
        gaze_feat = torch.cat([batch["gaze_pos"], batch["gaze_speed"]], dim=-1)
        tok = self.norm_gaze(self.proj_gaze(gaze_feat))
        mask = batch["gaze_mask"].unsqueeze(-1).float()
        return mask * tok + (1 - mask) * self.missing_gaze.view(1, 1, -1)  # (B, T, D)


class SpatiotemporalEncoderTemporalGazeOnly(nn.Module):
    """
    Gaze-only temporal encoder: 1 token per frame, per-frame (T, 196) scores.
    No IntraFrameBlock. All other components identical to the full temporal encoder.
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
        self.tokenizer   = TrajectoryTokenizerGazeOnlyTemporal(d_traj)
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
        self.vt_fusion = TemporalVisualTrajFusion(d_enc, d_vis, n_tokens=N_TOKENS_GAZE)
        self.norm_out  = nn.LayerNorm(d_enc)

        self.traj_patch_embed = nn.Embedding(N_PATCHES, d_enc)
        self.register_buffer("patch_idx", torch.arange(N_PATCHES))
        self.traj_score_head  = nn.Sequential(nn.Linear(d_enc, 1), nn.Sigmoid())

    def _trajectory_only_scores(self, traj_context: torch.Tensor) -> torch.Tensor:
        B, T, D = traj_context.shape
        scale   = D ** -0.5
        ctx     = traj_context.reshape(B * T, 1, D)
        q = self.traj_patch_embed(self.patch_idx).unsqueeze(0).expand(B * T, -1, -1)
        attn     = torch.bmm(q, ctx.transpose(1, 2)) * scale
        attn     = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn, ctx)
        return self.traj_score_head(attended).squeeze(-1).reshape(B, T, N_PATCHES)

    def forward(
        self,
        batch:       dict,
        query_emb:   Optional[torch.Tensor] = None,
        visual_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = batch["gaze_mask"].shape[0]
        T = batch["gaze_mask"].shape[1]

        # 1. Tokenize gaze → (B, T, D_traj)
        tok_g = self.tokenizer(batch)

        # 2. Project + PE (T*1 sequence)
        x = self.proj(tok_g.reshape(B * T, -1)).reshape(B, T, D_ENC)
        x = self.pe(x)  # (B, T, D_enc)

        # 3. Inter-frame transformer
        x = self.inter_frame(x)  # (B, T, D_enc)

        # 4. FiLM
        if query_emb is None:
            query_emb = torch.zeros(B, self.film.proj_scale.in_features, device=x.device)
        x = self.film(x, query_emb)

        # 5. Reshape to (B, T, 1, D_enc) for TemporalVisualTrajFusion
        x_framed = x.reshape(B, T, N_TOKENS_GAZE, D_ENC)

        # 6. Per-frame visual fusion
        if visual_feat is not None:
            per_frame_scores, enriched_context, enc_attn = self.vt_fusion(x_framed, visual_feat)
        else:
            enriched_context = x_framed
            per_frame_scores = self._trajectory_only_scores(x)
            enc_attn         = None

        context = self.norm_out(enriched_context)
        return per_frame_scores, context, enc_attn
