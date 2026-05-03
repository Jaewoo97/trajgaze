"""
TrajGazeV2Temporal — Option C: trajectory-prediction-driven patch scoring.

Score map design:
  At inference, patch scores come from TrajScoreHead(enriched_context), NOT from
  raw encoder cross-attention weights. TrajScoreHead is trained to reproduce
  traj_driven_scores — the chain of decoder × encoder attention — so the output
  directly reflects which patches contributed to future trajectory prediction.

Training losses:
  L_traj         : trajectory position prediction (MSE) — shapes enc cross-attn
  L_score_future : future per-frame score map prediction (MSE)
  L_score_traj   : TrajScoreHead output vs traj_driven (dec_attn × enc_attn)
                   — the primary score supervision, trajectory-prediction-driven
  L_score_past   : raw encoder attn scores vs GT gaze-hand maps (auxiliary)
                   — spatial grounding to help enc cross-attn converge

Inference:
  get_patch_scores(traj_batch, queries, frame_paths)
    → encoder → enriched_context
    → TrajScoreHead(enriched_context)
    → (B, T, 196) per-frame scores, adaptive to any T
  Decoder is NOT called at inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_encoder              import QueryEncoder, D_QUERY
from .encoder                    import D_ENC, N_PATCHES
from .encoder_temporal           import SpatiotemporalEncoderTemporal
from .decoders                   import (
    CrossAttentionDecoder, CrossAttentionDecoderTracked,
    score_loss, traj_loss,
)
from .visual_encoder_temporal    import VisualPatchEncoderTemporal

T_FUTURE_MAX = 128


# ── Trajectory-driven score head ─────────────────────────────────────────────

class TrajScoreHead(nn.Module):
    """
    Maps enriched_context (B, T, 4, D) → per-frame patch scores (B, T, 196).

    enriched_context already has visual information baked in from the encoder's
    visual cross-attention, so the head can extract patch importance from it.

    Uses learned attention pooling over the 4 trajectory tokens per frame,
    then projects to 196 patch scores.

    Trained to reproduce traj_driven_scores = dec_attn × enc_attn, so the
    output reflects which patches contributed to future trajectory prediction.
    """

    def __init__(self, d_enc: int = D_ENC, n_patches: int = N_PATCHES):
        super().__init__()
        # Learned attention pooling: compress 4 trajectory tokens → 1 per frame
        self.token_attn = nn.Sequential(
            nn.LayerNorm(d_enc),
            nn.Linear(d_enc, 1),
        )
        # Project pooled representation → patch scores
        self.head = nn.Sequential(
            nn.LayerNorm(d_enc),
            nn.Linear(d_enc, d_enc),
            nn.GELU(),
            nn.Linear(d_enc, n_patches),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, T, 4, D)
        Returns: (B, T, 196)
        """
        # Attention pooling over 4 tokens
        attn_w = self.token_attn(context)                    # (B, T, 4, 1)
        attn_w = torch.softmax(attn_w, dim=2)                # (B, T, 4, 1)
        pooled = (context * attn_w).sum(dim=2)               # (B, T, D)
        return self.head(pooled)                             # (B, T, 196)


# ── Decoder wrappers ─────────────────────────────────────────────────────────

class TrajectoryDecoderTemporal(nn.Module):
    TRAJ_DIM = 6

    def __init__(self, d_model: int = D_ENC, n_future: int = T_FUTURE_MAX):
        super().__init__()
        self.decoder = CrossAttentionDecoderTracked(
            d_model=d_model, out_dim=self.TRAJ_DIM,
            n_future=n_future, n_layers=3, n_heads=8,
        )

    def forward(self, context, T_future, return_cross_weights=False):
        out = self.decoder(context, T_future, return_cross_weights=return_cross_weights)
        if return_cross_weights:
            raw, cross_w = out
            return torch.sigmoid(raw), cross_w
        return torch.sigmoid(out)


class ScoreDecoderTemporal(nn.Module):
    def __init__(self, d_model: int = D_ENC, n_future: int = T_FUTURE_MAX,
                 n_patches: int = N_PATCHES):
        super().__init__()
        self.decoder = CrossAttentionDecoder(
            d_model=d_model, out_dim=d_model,
            n_future=n_future, n_layers=3, n_heads=8,
        )
        self.head = nn.Sequential(nn.GELU(), nn.Linear(d_model, n_patches))

    def forward(self, context, T_future):
        return torch.sigmoid(self.head(self.decoder(context, T_future)))


# ── Loss ─────────────────────────────────────────────────────────────────────

def score_loss_temporal(pred, gt, T_len):
    """MSE with valid-length masking. pred/gt: (B, T, 196), T_len: (B,)."""
    B, T_pred, N = pred.shape
    T_max  = min(T_pred, gt.shape[1])
    device = pred.device
    mask   = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    for i, T_i in enumerate(T_len):
        mask[i, :min(int(T_i.item()), T_max)] = True
    msk = mask.unsqueeze(-1).float()
    return ((pred[:, :T_max] - gt[:, :T_max]) ** 2 * msk).sum() / (msk.sum() * N + 1e-6)


# ── Main model ────────────────────────────────────────────────────────────────

class TrajGazeV2Temporal(nn.Module):

    def __init__(
        self,
        d_traj:                    int  = 128,
        d_enc:                     int  = D_ENC,
        d_query:                   int  = D_QUERY,
        n_layers_l2:               int  = 6,
        n_heads_l2:                int  = 8,
        t_future_max:              int  = T_FUTURE_MAX,
        n_vis_keyframes:           int  = 16,
        use_frame_score_branch:    bool = False,
        use_post_fusion_iframe:    bool = False,
        use_patch_temporal_branch: bool = False,
    ):
        super().__init__()
        self.query_encoder  = QueryEncoder(d_model=d_query)
        self.visual_encoder = VisualPatchEncoderTemporal(
            n_keyframes=n_vis_keyframes, out_dim=d_enc,
        )
        self.encoder = SpatiotemporalEncoderTemporal(
            d_traj=d_traj, d_enc=d_enc, d_vis=d_enc,
            d_query=d_query, n_layers_l2=n_layers_l2, n_heads_l2=n_heads_l2,
            use_frame_score_branch=use_frame_score_branch,
            use_post_fusion_iframe=use_post_fusion_iframe,
            use_patch_temporal_branch=use_patch_temporal_branch,
        )
        self.traj_decoder  = TrajectoryDecoderTemporal(d_model=d_enc, n_future=t_future_max)
        self.score_decoder = ScoreDecoderTemporal(d_model=d_enc,      n_future=t_future_max)
        self.score_head    = TrajScoreHead(d_enc=d_enc, n_patches=N_PATCHES)

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Four-term loss:
          L_traj         : trajectory prediction (shapes encoder cross-attn)
          L_score_future : future score map prediction
          L_score_traj   : TrajScoreHead vs traj_driven (primary score signal)
          L_score_past   : raw encoder attn vs GT gaze-hand maps (auxiliary)
        """
        past    = batch["past"]
        future  = batch["future"]
        T_past  = batch["T_past"]
        T_f_max = int(batch["T_future"].max().item())
        T_p_max = int(T_past.max().item())
        device  = past["left_pos"].device
        B       = past["left_pos"].shape[0]

        query_emb = torch.zeros(B, self.query_encoder.d_model, device=device)

        # Visual features for past frames
        visual_feat = None
        if (fps_batch := batch.get("frame_paths")) is not None:
            past_paths  = [fps[:int(T_past[i].item())] for i, fps in enumerate(fps_batch)]
            visual_feat = self.visual_encoder(past_paths, T_p_max, device)

        # Encoder → past scores + enriched context + per-token visual attn
        past_scores, context, enc_attn = self.encoder(past, query_emb, visual_feat)
        # past_scores : (B, T_past, 196)  — raw encoder attn readout
        # context     : (B, T_past, 4, D) — enriched with visual features
        # enc_attn    : (B, T_past, 4, 196) or None

        # TrajScoreHead — primary inference-time score output
        score_head_out = self.score_head(context)   # (B, T_past, 196)

        # Trajectory decoder (with cross-attn weights)
        traj_pred, dec_attn = self.traj_decoder(context, T_f_max, return_cross_weights=True)
        # dec_attn : (B, T_f_max, T_past*4) — which past tokens drove future prediction

        # Score decoder
        score_pred = self.score_decoder(context, T_f_max)  # (B, T_f_max, 196)

        # ── Trajectory-driven scores (chain dec_attn × enc_attn) ─────────────
        l_score_traj = torch.tensor(0.0, device=device)
        if enc_attn is not None:
            # dec_attn: (B, T_f_max, T_past*4) → mean over future → (B, T_past, 4)
            dec_importance = dec_attn.mean(dim=1).reshape(B, T_p_max, 4)

            # Weighted sum: which patches did trajectory-important tokens attend to?
            traj_driven = (enc_attn * dec_importance.unsqueeze(-1)).sum(dim=2)  # (B, T_past, 196)
            t_max = traj_driven.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            traj_driven = (traj_driven / t_max).detach()   # normalise + stop gradient

            # Supervise TrajScoreHead to reproduce trajectory-driven scores
            l_score_traj = score_loss_temporal(score_head_out, traj_driven, batch["T_past"])

        # ── Standard losses ───────────────────────────────────────────────────
        l_traj         = traj_loss(traj_pred, future, batch["T_future"])
        l_score_future = score_loss_temporal(score_pred,  batch["I_scores_future"], batch["T_future"])
        l_score_past   = score_loss_temporal(past_scores, batch["I_scores_past"],   batch["T_past"])

        total = l_traj + l_score_future + l_score_past + l_score_traj

        return {
            "loss":            total,
            "loss_traj":       l_traj,
            "loss_score_fut":  l_score_future,
            "loss_score_past": l_score_past,
            "loss_score_traj": l_score_traj,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_patch_scores(
        self,
        traj_batch:  dict,
        queries:     Optional[list[str]]       = None,
        frame_paths: Optional[list[list[str]]] = None,
    ) -> torch.Tensor:
        """
        Returns (B, T, 196) trajectory-prediction-driven per-frame patch scores.
        Decoder is not called — scores come from TrajScoreHead(enriched_context).
        """
        device    = traj_batch["left_pos"].device
        T         = traj_batch["left_pos"].shape[1]
        query_emb = self.query_encoder(queries, device)

        visual_feat = None
        if frame_paths is not None:
            visual_feat = self.visual_encoder(frame_paths, T, device)

        _, context, _ = self.encoder(traj_batch, query_emb, visual_feat)
        return self.score_head(context)   # (B, T, 196)

    def save(self, path: str):
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TrajGazeV2Temporal":
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
        return model
