"""
SpatiotemporalEncoderTemporal for TrajGaze_v2.

Extension of SpatiotemporalEncoder that produces per-frame patch score maps
(B, T, 196) instead of a single clip-level (B, 196) map.

Pipeline:
  1. TrajectoryTokenizer  : gaze/hand features → (B, T, 4, D_traj)
  2. IntraFrameBlock      : self-attn across 4 tokens per frame → (B, T, 4, D_traj)
  3. Project + SinusoidalPE → (B, T*4, D_enc)
  4. InterFrameTransformer: global temporal context → (B, T*4, D_enc)
  5. FiLM                 : query conditioning → (B, T*4, D_enc)
  6. Reshape              : → (B, T, 4, D_enc)
  7. TemporalVisualTrajFusion:
       Q = trajectory tokens per frame  (B*T, 4, D_enc)
       K, V = visual patches per frame  (B*T, 196, D_vis)
       → per-frame cross-attention      (B*T, 4, 196) attn weights
       → mean over 4 trajectory tokens  (B*T, 196)
       → score_head + reshape           (B, T, 196) per-frame scores
       → enriched_context (residual)    (B, T, 4, D_enc)

Outputs:
  patch_scores : (B, T, 196) — per-frame patch importance maps
  context      : (B, T, 4, D_enc) — enriched trajectory context for decoders

Training supervision:
  Encoder past scores supervised with GT past frame I_scores.
  Decoder future scores supervised with GT future frame I_scores.
  "The patches you attend to at each frame should match the trajectory-relevant
   patches at that frame."
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    TrajectoryTokenizer, IntraFrameBlock, SinusoidalPE, FiLM,
    D_TRAJ, D_ENC, D_VIS, D_QUERY,
    N_HEADS_L1, N_HEADS_L2, N_LAYERS_L1, N_LAYERS_L2,
    N_PATCHES, N_TOKENS,
)


class TemporalVisualTrajFusion(nn.Module):
    """
    Per-frame cross-attention: each frame's trajectory tokens attend independently
    to that frame's visual patch features.

    Q = trajectory context, reshaped to (B*T, 4, D_enc)
    K, V = per-frame visual patches,    (B*T, 196, D_vis)

    Each frame produces its own 196-dim score map, capturing how gaze/hand
    attention evolves spatially over time.

    Outputs:
        per_frame_scores : (B, T, 196) ∈ [0, 1]
        enriched_context : (B, T, 4, D_enc) — trajectory context + attended visual
    """

    def __init__(
        self,
        d_enc:     int = D_ENC,
        d_vis:     int = D_VIS,
        n_heads:   int = 8,
        n_patches: int = N_PATCHES,
        n_tokens:  int = N_TOKENS,
    ):
        super().__init__()
        self.n_heads   = n_heads
        self.d_head    = d_enc // n_heads
        self.scale     = self.d_head ** -0.5
        self.n_patches = n_patches
        self.n_tokens  = n_tokens

        self.q_proj   = nn.Linear(d_enc, d_enc)
        self.k_proj   = nn.Linear(d_vis, d_enc)
        self.v_proj   = nn.Linear(d_vis, d_enc)
        self.out_proj = nn.Linear(d_enc, d_enc)

        self.score_head = nn.Sequential(
            nn.LayerNorm(d_enc),
            nn.Linear(d_enc, 1),
            nn.Sigmoid(),
        )

        self.norm_traj = nn.LayerNorm(d_enc)
        self.norm_vis  = nn.LayerNorm(d_vis)
        self.norm_out  = nn.LayerNorm(d_enc)

    def forward(
        self,
        traj_context: torch.Tensor,   # (B, T, N_tok, D_enc)
        visual_feat:  torch.Tensor,   # (B, T, 196, D_vis)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            per_frame_scores : (B, T, 196) per-frame importance maps
            enriched_context : (B, T, N_tok, D_enc)
        """
        B, T, N_tok, D = traj_context.shape
        N = self.n_patches

        # Flatten batch×time for parallel per-frame cross-attention
        Q_flat  = traj_context.reshape(B * T, N_tok, D)         # (B*T, 4, D)
        vis_flat = visual_feat.reshape(B * T, N, visual_feat.shape[-1])  # (B*T, 196, D_vis)

        Q = self.q_proj(self.norm_traj(Q_flat))      # (B*T, 4, D)
        K = self.k_proj(self.norm_vis(vis_flat))     # (B*T, 196, D)
        V = self.v_proj(vis_flat)                    # (B*T, 196, D)

        BT = B * T

        def split_heads(x, L):
            return x.view(BT, L, self.n_heads, self.d_head).transpose(1, 2)

        Q_h = split_heads(Q, N_tok)   # (B*T, H, 4, d_head)
        K_h = split_heads(K, N)       # (B*T, H, 196, d_head)
        V_h = split_heads(V, N)       # (B*T, H, 196, d_head)

        attn = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale  # (B*T, H, 4, 196)
        attn_weights = F.softmax(attn, dim=-1)                          # (B*T, H, 4, 196)

        # Attended visual features for each trajectory token
        attended = torch.matmul(attn_weights, V_h)                      # (B*T, H, 4, d_head)
        attended = attended.transpose(1, 2).reshape(BT, N_tok, D)       # (B*T, 4, D)

        # Residual enrichment
        enriched_flat = Q_flat + self.out_proj(self.norm_out(attended))  # (B*T, 4, D)

        # Per-frame patch scores: mean attention over heads and trajectory tokens
        # attn_weights: (B*T, H, 4, 196) → mean over H → (B*T, 4, 196)
        attn_per_token = attn_weights.mean(dim=1)                     # (B*T, 4, 196)
        mean_attn      = attn_per_token.mean(dim=1)                   # (B*T, 196)

        # Weight visual patches by mean trajectory attention, pass through score head
        V_mean = V.reshape(BT, N, D)                                  # (B*T, 196, D)
        patch_feat  = mean_attn.unsqueeze(-1) * V_mean                # (B*T, 196, D)
        scores_flat = self.score_head(patch_feat).squeeze(-1)         # (B*T, 196)

        # Reshape back to (B, T, ...)
        per_frame_scores  = scores_flat.reshape(B, T, N)              # (B, T, 196)
        enriched_context  = enriched_flat.reshape(B, T, N_tok, D)     # (B, T, 4, D)
        attn_per_token_bt = attn_per_token.reshape(B, T, N_tok, N)    # (B, T, 4, 196)

        return per_frame_scores, enriched_context, attn_per_token_bt


class SpatiotemporalEncoderTemporal(nn.Module):
    """
    Spatiotemporal encoder that outputs per-frame (T, 196) patch score maps.

    Identical trajectory tokenization and inter-frame transformer as
    SpatiotemporalEncoder, but replaces the clip-level VisualTrajectoryFusion
    with TemporalVisualTrajFusion to produce one score map per input frame.

    forward(batch, query_emb, visual_feat) →
        per_frame_scores : (B, T, 196)    — per-frame patch importance
        context          : (B, T, 4, D)   — enriched context for decoders
    """

    def __init__(
        self,
        d_traj:                  int  = D_TRAJ,
        d_enc:                   int  = D_ENC,
        d_vis:                   int  = D_VIS,
        d_query:                 int  = D_QUERY,
        n_layers_l2:             int  = N_LAYERS_L2,
        n_heads_l2:              int  = N_HEADS_L2,
        use_frame_score_branch:  bool = False,
        use_post_fusion_iframe:  bool = False,
        use_patch_temporal_branch: bool = False,
    ):
        super().__init__()
        assert not (use_frame_score_branch and use_patch_temporal_branch), (
            "use_frame_score_branch and use_patch_temporal_branch are mutually exclusive — "
            "both modulate per_frame_scores from x_iframe."
        )
        self.tokenizer   = TrajectoryTokenizer(d_traj)
        self.intra_frame = IntraFrameBlock(d_traj)
        self.proj        = nn.Linear(d_traj, d_enc)
        self.pe          = SinusoidalPE(d_enc, max_len=4096)
        self.inter_frame = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_enc, nhead=n_heads_l2, dim_feedforward=d_enc * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=n_layers_l2,
        )
        # Gated residual: at gate=0 the inter-frame transformer is bypassed entirely,
        # so the encoder starts identical to the spatial-only ablation (which converges).
        # The gate learns to open as joint training progresses, avoiding the non-stationary
        # query distribution that destabilises random-init joint training.
        self.inter_frame_gate = nn.Parameter(torch.zeros(1))
        self.film       = FiLM(d_query, d_enc)
        self.vt_fusion  = TemporalVisualTrajFusion(d_enc, d_vis)
        self.norm_out   = nn.LayerNorm(d_enc)

        # (a)+(b1): parallel frame-level score branch sourced from inter_frame output.
        # Output (B, T) frame importance broadcast-multiplied into per-patch scores.
        self.use_frame_score_branch = use_frame_score_branch
        if use_frame_score_branch:
            self.frame_attn_pool = nn.Linear(d_enc, 1)
            self.frame_score_head = nn.Sequential(
                nn.LayerNorm(d_enc),
                nn.Linear(d_enc, 1),
                nn.Sigmoid(),
            )

        # E1: patch-level temporal modulation branch.
        # Replaces the frame-uniform (B, T) scalar with per-patch (B, T, 196) modulation.
        # 196 learned patch queries cross-attend to x_iframe (4 tokens per frame) →
        # (B*T, 196, D_enc) → sigmoid → (B, T, 196) ∈ [0, 1] elementwise multiplier.
        self.use_patch_temporal_branch = use_patch_temporal_branch
        if use_patch_temporal_branch:
            self.patch_temporal_query = nn.Embedding(N_PATCHES, d_enc)
            self.patch_temporal_attn  = nn.MultiheadAttention(
                embed_dim=d_enc, num_heads=4, dropout=0.1, batch_first=True,
            )
            self.patch_temporal_head  = nn.Sequential(
                nn.LayerNorm(d_enc),
                nn.Linear(d_enc, 1),
                nn.Sigmoid(),
            )

        # (b2): post-fusion InterFrameTransformer that mixes enriched_context across frames
        # AFTER per-frame visual cross-attention is computed. Avoids the non-stationary
        # query problem because visual fusion sees clean per-frame queries.
        self.use_post_fusion_iframe = use_post_fusion_iframe
        if use_post_fusion_iframe:
            self.inter_frame_post = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_enc, nhead=n_heads_l2, dim_feedforward=d_enc * 4,
                    dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
                ),
                num_layers=n_layers_l2,
            )
            self.inter_frame_post_gate = nn.Parameter(torch.zeros(1))

        # Trajectory-only fallback when no visual features provided
        self.traj_patch_embed = nn.Embedding(N_PATCHES, d_enc)
        patch_idx = torch.arange(N_PATCHES)
        self.register_buffer("patch_idx", patch_idx)
        self.traj_score_head = nn.Sequential(
            nn.Linear(d_enc, 1),
            nn.Sigmoid(),
        )

    def _trajectory_only_scores(self, traj_context: torch.Tensor) -> torch.Tensor:
        """
        Fallback per-frame scores from trajectory context only (no visual info).
        Each frame's context cross-attends to learned patch position embeddings.
        Returns (B, T, 196).
        """
        B, T, N_tok, D = traj_context.shape
        scale = D ** -0.5
        ctx_flat = traj_context.reshape(B * T, N_tok, D)   # (B*T, 4, D)
        q = self.traj_patch_embed(self.patch_idx)          # (196, D)
        q = q.unsqueeze(0).expand(B * T, -1, -1)          # (B*T, 196, D)
        attn = torch.bmm(q, ctx_flat.transpose(1, 2)) * scale  # (B*T, 196, 4)
        attn = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn, ctx_flat)               # (B*T, 196, D)
        scores = self.traj_score_head(attended).squeeze(-1)  # (B*T, 196)
        return scores.reshape(B, T, N_PATCHES)

    def forward(
        self,
        batch:       dict,
        query_emb:   Optional[torch.Tensor] = None,   # (B, D_QUERY)
        visual_feat: Optional[torch.Tensor] = None,   # (B, T, 196, D_VIS)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            per_frame_scores : (B, T, 196) — per-frame importance maps
            context          : (B, T, 4, D_ENC) — enriched context for decoders
        """
        B = batch["gaze_mask"].shape[0]
        T = batch["gaze_mask"].shape[1]

        # 1. Tokenize trajectory
        tok_g, tok_l, tok_r, tok_i = self.tokenizer(batch)

        # 2. Intra-frame attention (cross-modality at each timestep)
        enriched = self.intra_frame(tok_g, tok_l, tok_r, tok_i)  # (B, T, 4, D_traj)

        # 3. Project to D_enc + sinusoidal temporal PE
        x = self.proj(enriched.reshape(B * T * N_TOKENS, -1)).reshape(B, T * N_TOKENS, D_ENC)
        x = self.pe(x)  # (B, T*4, D_enc)

        # 4. Inter-frame transformer — global temporal context over all T frames,
        #    mixed in via tanh-gated residual (gate inits at 0 → bypass at start).
        x_iframe = self.inter_frame(x)
        gate     = torch.tanh(self.inter_frame_gate)
        x_main   = x + gate * (x_iframe - x)  # (B, T*4, D_enc) — main path

        # 4b. Optional parallel frame-score branch (from x_iframe before gating).
        frame_scores = None
        if self.use_frame_score_branch:
            x_iframe_per_frame = x_iframe.reshape(B, T, N_TOKENS, D_ENC)
            attn_w = torch.softmax(self.frame_attn_pool(x_iframe_per_frame), dim=2)
            pooled = (x_iframe_per_frame * attn_w).sum(dim=2)        # (B, T, D_enc)
            frame_scores = self.frame_score_head(pooled).squeeze(-1) # (B, T) ∈ [0, 1]

        # 4c. E1: patch-level temporal modulation (mutually exclusive with 4b).
        patch_modulation = None
        if self.use_patch_temporal_branch:
            x_iframe_per_frame = x_iframe.reshape(B, T, N_TOKENS, D_ENC)
            flat_kv = x_iframe_per_frame.reshape(B * T, N_TOKENS, D_ENC)        # (B*T, 4, D)
            q = self.patch_temporal_query(self.patch_idx)                       # (196, D)
            q = q.unsqueeze(0).expand(B * T, -1, -1)                            # (B*T, 196, D)
            attended, _ = self.patch_temporal_attn(q, flat_kv, flat_kv)         # (B*T, 196, D)
            patch_mod = self.patch_temporal_head(attended).squeeze(-1)          # (B*T, 196)
            patch_modulation = patch_mod.reshape(B, T, N_PATCHES)               # ∈ [0, 1]

        # 5. FiLM: query shifts which patches the trajectory finds important
        if query_emb is None:
            query_emb = torch.zeros(B, self.film.proj_scale.in_features, device=x_main.device)
        x = self.film(x_main, query_emb)  # (B, T*4, D_enc)

        # 6. Reshape to per-frame tokens
        x_framed = x.reshape(B, T, N_TOKENS, D_ENC)  # (B, T, 4, D_enc)

        # 7. Per-frame visual-trajectory fusion
        if visual_feat is not None:
            per_frame_scores, enriched_context, attn_per_token = self.vt_fusion(x_framed, visual_feat)
        else:
            enriched_context = x_framed
            per_frame_scores = self._trajectory_only_scores(x_framed)
            attn_per_token   = None   # (B, T, 4, 196) unavailable without visual feat

        # 7b. Optional post-fusion InterFrameTransformer over enriched_context.
        if self.use_post_fusion_iframe:
            ec_flat = enriched_context.reshape(B, T * N_TOKENS, D_ENC)
            ec_post = self.inter_frame_post(ec_flat)
            gate_post = torch.tanh(self.inter_frame_post_gate)
            ec_flat = ec_flat + gate_post * (ec_post - ec_flat)
            enriched_context = ec_flat.reshape(B, T, N_TOKENS, D_ENC)

        # 7c. Fuse frame-level score (broadcast-multiply into per-patch scores).
        if frame_scores is not None:
            per_frame_scores = per_frame_scores * frame_scores.unsqueeze(-1)

        # 7d. E1: fuse patch-level modulation (elementwise multiply).
        if patch_modulation is not None:
            per_frame_scores = per_frame_scores * patch_modulation

        # 8. Normalise context
        context = self.norm_out(enriched_context)  # (B, T, 4, D_enc)

        return per_frame_scores, context, attn_per_token
