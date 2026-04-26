"""
SpatiotemporalEncoder for TrajGaze_v2.

Pipeline:
  1. TrajectoryTokenizer: gaze/hand features → (B, T, 4, 128) per-frame tokens
  2. IntraFrameBlock (2L): self-attn across 4 tokens per frame → (B, T, 4, 128)
  3. Project 128→256 + sinusoidal PE
  4. InterFrameTransformer (6L, 8H): temporal context → (B, T*4, 256)
  5. FiLM: condition trajectory context on text query embedding
  6. VisualTrajectoryFusion:
       Q = trajectory context   (B, T*4, 256)
       K, V = visual patches    (B, 196, 256)
       → attn_weights (B, T*4, 196): trajectory tokens attend to visual patches
       → enriched_context = traj_context + attn_weights @ visual_patches  (B, T*4, 256)
       → patch_scores = mean_pool(attn_weights, dim=T*4) → score_head → (B, 196)
             "which visual patches were most attended during trajectory prediction?"

Outputs:
  patch_scores : (B, 196) ∈ [0, 1] — patch importance; directly used for top-K selection
  context      : (B, T, 4, 256) — enriched context (trajectory + visual); passed to decoders

Stage 1 supervision of patch_scores:
  L_attn = MSE(patch_scores, I_scores_future.mean(dim=1))
  "The patches you attended to should match the patches that are trajectory-relevant in future frames"
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dimensions ────────────────────────────────────────────────────────────────
D_TRAJ     = 128
D_ENC      = 256
D_VIS      = 256   # visual feature dim (after projection from DINOv2-384)
D_QUERY    = 128
N_HEADS_L1 = 4
N_HEADS_L2 = 8
N_LAYERS_L1 = 2
N_LAYERS_L2 = 6
N_PATCHES   = 196  # 14×14
N_TOKENS    = 4    # gaze, left, right, interaction


# ── Trajectory Tokenizer ──────────────────────────────────────────────────────

class TrajectoryTokenizer(nn.Module):
    DIM_GAZE_RAW     = 3
    DIM_HAND_RAW     = 5
    DIM_INTERACT_RAW = 12

    def __init__(self, d_model: int = D_TRAJ):
        super().__init__()
        self.d_model = d_model
        self.missing_gaze  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.missing_left  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.missing_right = nn.Parameter(torch.randn(d_model) * 0.02)

        self.proj_gaze     = nn.Linear(self.DIM_GAZE_RAW,     d_model)
        self.proj_left     = nn.Linear(self.DIM_HAND_RAW,     d_model)
        self.proj_right    = nn.Linear(self.DIM_HAND_RAW,     d_model)
        self.proj_interact = nn.Linear(self.DIM_INTERACT_RAW, d_model)

        self.norm_gaze     = nn.LayerNorm(d_model)
        self.norm_left     = nn.LayerNorm(d_model)
        self.norm_right    = nn.LayerNorm(d_model)
        self.norm_interact = nn.LayerNorm(d_model)

    def forward(self, batch: dict):
        B, T = batch["gaze_mask"].shape

        gaze_feat = torch.cat([batch["gaze_pos"], batch["gaze_speed"]], dim=-1)
        tok_gaze  = self.norm_gaze(self.proj_gaze(gaze_feat))
        mask_g    = batch["gaze_mask"].unsqueeze(-1).float()
        tok_gaze  = mask_g * tok_gaze + (1 - mask_g) * self.missing_gaze.view(1, 1, -1)

        left_feat = torch.cat([batch["left_pos"], batch["left_vel"],
                               batch["left_mask"].unsqueeze(-1).float()], dim=-1)
        tok_left  = self.norm_left(self.proj_left(left_feat))
        mask_l    = batch["left_mask"].unsqueeze(-1).float()
        tok_left  = mask_l * tok_left + (1 - mask_l) * self.missing_left.view(1, 1, -1)

        right_feat = torch.cat([batch["right_pos"], batch["right_vel"],
                                batch["right_mask"].unsqueeze(-1).float()], dim=-1)
        tok_right  = self.norm_right(self.proj_right(right_feat))
        mask_r     = batch["right_mask"].unsqueeze(-1).float()
        tok_right  = mask_r * tok_right + (1 - mask_r) * self.missing_right.view(1, 1, -1)

        interact_feat = torch.cat([
            batch["d_left"], batch["d_right"],
            batch["v_rel_left"], batch["v_rel_right"],
            batch["convergence"].unsqueeze(-1),
            batch["lead_lag"].unsqueeze(-1),
        ], dim=-1)
        tok_interact = self.norm_interact(self.proj_interact(interact_feat))

        return tok_gaze, tok_left, tok_right, tok_interact


# ── Sinusoidal PE ─────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


# ── Intra-frame attention ─────────────────────────────────────────────────────

class IntraFrameBlock(nn.Module):
    def __init__(self, d_model: int = D_TRAJ, n_heads: int = N_HEADS_L1,
                 n_layers: int = N_LAYERS_L1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])

    def forward(self, tok_gaze, tok_left, tok_right, tok_interact) -> torch.Tensor:
        B, T, D = tok_gaze.shape
        x = torch.stack([tok_gaze, tok_left, tok_right, tok_interact], dim=2)
        x = x.reshape(B * T, N_TOKENS, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, T, N_TOKENS, D)


# ── FiLM conditioning ─────────────────────────────────────────────────────────

class FiLM(nn.Module):
    def __init__(self, d_cond: int = D_QUERY, d_feat: int = D_ENC):
        super().__init__()
        self.proj_scale = nn.Linear(d_cond, d_feat)
        self.proj_shift = nn.Linear(d_cond, d_feat)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.proj_scale(cond).unsqueeze(1)
        shift = self.proj_shift(cond).unsqueeze(1)
        return x * (1 + scale) + shift


# ── Visual-Trajectory Fusion ──────────────────────────────────────────────────

class VisualTrajectoryFusion(nn.Module):
    """
    Cross-attention where trajectory tokens attend to visual patch features.

    Q = trajectory context  (B, T*4, D_enc) — what the trajectory "wants to know"
    K = visual patches      (B, 196, D_vis)  — what is in each spatial location
    V = visual patches      (B, 196, D_vis)

    Each trajectory token attends over 196 spatial patches to gather visual context.
    The attention weight A[t, p] encodes: "how relevant is patch p for explaining
    the trajectory at time step t?"

    Outputs:
      enriched_context : (B, T*4, D_enc)
          Trajectory context enriched with visual information.
          Used by decoders to predict future trajectory + future patch scores.

      patch_scores : (B, 196)
          Mean attention weight across all trajectory tokens, projected through a
          score head → sigmoid. High score = patch was consistently attended to
          during trajectory prediction = important for the VLM.

    Stage 1 training signal:
      patch_scores is supervised by MSE against mean(I_scores_future, dim=time),
      so the model learns to attend to patches that are actually trajectory-relevant
      in future frames.

    At inference with query conditioning:
      FiLM modulates trajectory context before this module, so the query shifts
      which visual patches trajectory tokens attend to — enabling query-conditioned
      token selection.
    """

    def __init__(self, d_enc: int = D_ENC, d_vis: int = D_VIS,
                 n_heads: int = 8, n_patches: int = N_PATCHES):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_enc // n_heads
        self.scale    = self.d_head ** -0.5
        self.n_patches = n_patches

        # Cross-attention projections
        self.q_proj = nn.Linear(d_enc, d_enc)   # trajectory → Q
        self.k_proj = nn.Linear(d_vis, d_enc)   # visual → K
        self.v_proj = nn.Linear(d_vis, d_enc)   # visual → V
        self.out_proj = nn.Linear(d_enc, d_enc)  # attended visual → D_enc

        # Patch score head: aggregated trajectory-visual attention → scalar score
        # Input: (B, 196, D_enc) — visual patch features weighted by trajectory attention
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
        traj_context: torch.Tensor,           # (B, T*4, D_enc)
        visual_feat:  torch.Tensor,           # (B, 196, D_vis)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            enriched_context : (B, T*4, D_enc) — for decoders
            patch_scores     : (B, 196) ∈ [0, 1] — for top-K token selection

        Side effect: caches the multi-head cross-attention weights in
        ``self.last_attn_weights`` (shape (B, H, T*4, 196)). The encoder may
        read this after a forward pass to derive per-frame saliency without
        changing the public return signature.
        """
        B, L_traj, D = traj_context.shape
        N = self.n_patches

        # ── Project Q, K, V ───────────────────────────────────────────────────
        Q = self.q_proj(self.norm_traj(traj_context))   # (B, T*4, D_enc)
        K = self.k_proj(self.norm_vis(visual_feat))     # (B, 196, D_enc)
        V = self.v_proj(visual_feat)                    # (B, 196, D_enc)

        # ── Multi-head cross-attention ────────────────────────────────────────
        # Reshape for multi-head: (B, n_heads, L, d_head)
        def split_heads(x, L):
            return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        Q_h = split_heads(Q, L_traj)    # (B, H, T*4, d_head)
        K_h = split_heads(K, N)         # (B, H, 196, d_head)
        V_h = split_heads(V, N)         # (B, H, 196, d_head)

        # Attention: each trajectory token attends over 196 patches
        attn = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale  # (B, H, T*4, 196)
        attn_weights = F.softmax(attn, dim=-1)                          # (B, H, T*4, 196)
        # Cache (detached) for downstream extras consumers — see docstring.
        self.last_attn_weights = attn_weights.detach()

        # Attended visual features for each trajectory token
        attended = torch.matmul(attn_weights, V_h)                      # (B, H, T*4, d_head)
        attended = attended.transpose(1, 2).reshape(B, L_traj, D)       # (B, T*4, D_enc)

        # ── Enrich trajectory context (residual) ──────────────────────────────
        enriched_context = traj_context + self.out_proj(self.norm_out(attended))

        # ── Patch scores ──────────────────────────────────────────────────────
        # For each patch p: how much did the trajectory attend to it, on average?
        # attn_weights: (B, H, T*4, 196) → mean over H and T*4 → (B, 196)
        mean_attn = attn_weights.mean(dim=1).mean(dim=1)   # (B, 196)

        # Per-patch attended visual feature (weighted by how much trajectory looked at it)
        # attn_weights transposed: (B, H, 196, T*4) @ V_traj → (B, H, 196, d_head)
        # Simpler: use mean_attn as weights over V directly
        # patch_attended[b, p] = sum_t( mean_attn[b,t,p] * V[b,p] ) → just use V itself
        # Actually: visual patch features weighted by mean trajectory attention
        patch_feat = mean_attn.unsqueeze(-1) * V           # (B, 196, D_enc)
        patch_scores = self.score_head(patch_feat).squeeze(-1)  # (B, 196)

        return enriched_context, patch_scores


# ── Main Encoder ──────────────────────────────────────────────────────────────

class SpatiotemporalEncoder(nn.Module):
    """
    Full spatiotemporal encoder with visual-trajectory fusion.

    forward(batch, query_emb, visual_feat) →
        patch_scores : (B, 196) — patch importance scores for top-K selection
        context      : (B, T, 4, 256) — enriched context for decoders
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
        self.tokenizer     = TrajectoryTokenizer(d_traj)
        self.intra_frame   = IntraFrameBlock(d_traj)
        self.proj          = nn.Linear(d_traj, d_enc)
        self.pe            = SinusoidalPE(d_enc)
        self.inter_frame   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_enc, nhead=n_heads_l2, dim_feedforward=d_enc * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=n_layers_l2,
        )
        self.film          = FiLM(d_query, d_enc)
        self.vt_fusion     = VisualTrajectoryFusion(d_enc, d_vis)
        self.norm_out      = nn.LayerNorm(d_enc)

        # Fallback patch scorer used when no visual features are available
        # (trajectory-only mode; produces scores from trajectory context alone)
        self.traj_patch_embed = nn.Embedding(N_PATCHES, d_enc)
        patch_idx = torch.arange(N_PATCHES)
        self.register_buffer("patch_idx", patch_idx)
        self.traj_score_head  = nn.Sequential(
            nn.Linear(d_enc, 1),
            nn.Sigmoid(),
        )

    def _trajectory_only_scores(self, traj_context: torch.Tensor) -> torch.Tensor:
        """
        Fallback: score patches from trajectory context only (no visual info).
        Used when visual_feat is None.
        Q = learned patch position embeddings, K/V = trajectory context.
        """
        B = traj_context.shape[0]
        scale = D_ENC ** -0.5
        # Q: (B, 196, D_enc), K: (B, T*4, D_enc)
        q = self.traj_patch_embed(self.patch_idx).unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(q, traj_context.transpose(1, 2)) * scale  # (B, 196, T*4)
        attn = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn, traj_context)  # (B, 196, D_enc)
        return self.traj_score_head(attended).squeeze(-1)  # (B, 196)

    def forward(
        self,
        batch:       dict,
        query_emb:   Optional[torch.Tensor] = None,  # (B, D_QUERY)
        visual_feat: Optional[torch.Tensor] = None,  # (B, 196, D_VIS) from DINOv2
        return_extras: bool = False,
    ):
        """
        Returns (legacy, ``return_extras=False``):
            patch_scores : (B, 196) — importance scores per patch
            context      : (B, T, 4, D_ENC) — enriched trajectory context for decoders

        Returns (``return_extras=True``):
            patch_scores, context, extras
            where ``extras`` is a dict with:
                ``frame_attend`` : (B, T) per-frame saliency derived from cross-attn
                                   peakiness (mean of per-token max attention over
                                   patches, averaged over heads and the 4 tokens of
                                   each frame). Higher = frame's trajectory tokens
                                   anchored on a specific visual region.
                ``traj_embeds``  : (B, T, 4, D_ENC) post-fusion trajectory context;
                                   identical to ``context``. Provided so callers can
                                   pick top-K frames by ``frame_attend`` without
                                   needing to re-derive it from the legacy return.
                ``frame_attend_src`` : str — "fusion" if visual_feat was given,
                                       "uniform" if trajectory-only fallback path.
        """
        B = batch["gaze_mask"].shape[0]
        T = batch["gaze_mask"].shape[1]

        # 1. Tokenize trajectory
        tok_g, tok_l, tok_r, tok_i = self.tokenizer(batch)

        # 2. Intra-frame attention
        enriched = self.intra_frame(tok_g, tok_l, tok_r, tok_i)  # (B, T, 4, D_TRAJ)

        # 3. Project to D_ENC + temporal PE
        x = self.proj(enriched.reshape(B * T * N_TOKENS, -1)).reshape(B, T * N_TOKENS, D_ENC)
        x = self.pe(x)  # (B, T*4, D_ENC)

        # 4. Inter-frame transformer
        x = self.inter_frame(x)  # (B, T*4, D_ENC)

        # 5. FiLM: query conditions the trajectory context
        #    (at inference: query shifts which patches trajectory finds important)
        if query_emb is None:
            query_emb = torch.zeros(B, self.film.proj_scale.in_features, device=x.device)
        x = self.film(x, query_emb)  # (B, T*4, D_ENC)

        # 6. Visual-Trajectory Fusion (if visual features are available)
        if visual_feat is not None:
            # Trajectory context cross-attends to visual patches:
            #   "Which visual patches explain the trajectory?"
            # → enriched_context: trajectory enriched with visual info (used by decoders)
            # → patch_scores: which patches had high trajectory attention
            enriched_context, patch_scores = self.vt_fusion(x, visual_feat)
        else:
            # Trajectory-only fallback (used in ablations or when frames are unavailable)
            enriched_context = x
            patch_scores = self._trajectory_only_scores(x)

        # 7. Normalise and reshape context for decoders
        context = self.norm_out(enriched_context).reshape(B, T, N_TOKENS, D_ENC)

        if not return_extras:
            return patch_scores, context

        # ── Build extras for axes 1 (frame budget) and 5a (hint tokens) ───────
        if visual_feat is not None:
            # Cross-attn cached on vt_fusion: (B, H, T*4, 196). Per-token peakiness
            # = max attention weight; high value → token has a clear visual anchor.
            attn = self.vt_fusion.last_attn_weights            # (B, H, T*4, 196)
            peak = attn.max(dim=-1).values                     # (B, H, T*4)
            peak = peak.mean(dim=1)                            # (B, T*4)
            frame_attend = peak.view(B, T, N_TOKENS).mean(-1)  # (B, T)
            src = "fusion"
        else:
            frame_attend = torch.full(
                (B, T), 1.0 / max(1, T), device=context.device, dtype=context.dtype
            )
            src = "uniform"

        extras = {
            "frame_attend":     frame_attend,
            "traj_embeds":      context,
            "frame_attend_src": src,
        }
        return patch_scores, context, extras
