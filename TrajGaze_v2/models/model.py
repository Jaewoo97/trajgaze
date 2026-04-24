"""
TrajGazeV2: full model — visual + trajectory + query → patch selection.

Stage 1 (no QA labels):
    Past frames (visual + trajectory) → predict FUTURE trajectory + FUTURE patch scores
    Loss: L_traj + L_score (both MSE against ground-truth future values)

Stage 2 / Inference:
    Full clip (visual + trajectory) + text query → top-K patch mask
    Uses PatchScorer: visual patches (query-gated) cross-attend over trajectory context
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_encoder  import QueryEncoder, D_QUERY
from .encoder        import SpatiotemporalEncoder, D_ENC, N_PATCHES
from .decoders       import TrajectoryDecoder, ScoreDecoder, traj_loss, score_loss
from .visual_encoder import VisualPatchEncoder

GAZING_RATIO = 0.10
N_KEEP       = max(1, int(math.ceil(GAZING_RATIO * N_PATCHES)))   # 20 patches = 10%


class TrajGazeV2(nn.Module):
    """
    Full TrajGaze_v2 model.

    Stage 1 usage:
        loss_dict = model.stage1_forward(batch)
        # batch must contain 'past', 'future', 'I_scores_future', 'T_past',
        #   'T_future', and optionally 'frame_paths' (list[list[str]])

    Stage 2 / inference usage:
        mask = model.select_tokens(traj_batch, queries, frame_paths)
        # mask: (B, 196) bool — True = selected patch
    """

    def __init__(
        self,
        d_traj:          int = 128,
        d_enc:           int = D_ENC,
        d_query:         int = D_QUERY,
        n_layers_l2:     int = 6,
        n_heads_l2:      int = 8,
        t_future_max:    int = 32,
        n_vis_keyframes: int = 8,
    ):
        super().__init__()
        self.query_encoder  = QueryEncoder(d_model=d_query)
        self.visual_encoder = VisualPatchEncoder(n_keyframes=n_vis_keyframes, out_dim=d_enc)
        self.encoder        = SpatiotemporalEncoder(
            d_traj=d_traj, d_enc=d_enc, d_query=d_query,
            n_layers_l2=n_layers_l2, n_heads_l2=n_heads_l2,
        )
        self.traj_decoder  = TrajectoryDecoder(d_model=d_enc, n_future=t_future_max)
        self.score_decoder = ScoreDecoder(d_model=d_enc,     n_future=t_future_max)

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Stage 1: predict FUTURE trajectory positions and FUTURE patch interaction
        scores from past trajectory + past visual frames.

        batch keys:
            past             : dict of trajectory tensors (B, T_past, ...)
            future           : dict of trajectory tensors (B, T_future_max, ...)
            I_scores_future  : (B, T_future_max, 196)  — ground-truth future scores
            T_past           : (B,) long
            T_future         : (B,) long
            frame_paths      : list[list[str]] or absent — past frame paths for DINOv2

        Returns dict with 'loss', 'loss_traj', 'loss_score'.
        """
        past    = batch["past"]
        future  = batch["future"]
        T_f_max = int(batch["T_future"].max().item())

        device = past["gaze_pos"].device
        B      = past["gaze_pos"].shape[0]

        # Null query for Stage 1 (no QA labels available yet)
        query_emb = torch.zeros(B, self.query_encoder.d_model, device=device)

        # Encode past visual frames — visual projection weights trained here via L_attn
        frame_paths_batch = batch.get("frame_paths")
        visual_feat = None
        if frame_paths_batch is not None:
            visual_feat = self.visual_encoder(frame_paths_batch, device)  # (B, 196, D_enc)

        # Encode past trajectory + past visual → context
        # patch_scores: which visual patches the trajectory finds important
        patch_scores, context = self.encoder(past, query_emb, visual_feat)
        # context: (B, T_past, 4, D_enc)

        # Decode: predict FUTURE frames from past context
        traj_pred  = self.traj_decoder(context, T_f_max)           # (B, T_f_max, 6)
        score_pred = self.score_decoder(context, T_f_max)          # (B, T_f_max, 196)

        # Loss against ground-truth future values
        l_traj  = traj_loss(traj_pred,  future,                   batch["T_future"])
        l_score = score_loss(score_pred, batch["I_scores_future"], batch["T_future"])

        # Supervise patch attention scores against mean future interaction scores.
        # This teaches the visual-trajectory fusion to attend to patches that will
        # be important in future frames — the core Stage 1 signal for patch selection.
        I_mean  = batch["I_scores_future"].mean(dim=1)             # (B, 196)
        l_attn  = F.mse_loss(patch_scores, I_mean)

        return {
            "loss":       l_traj + l_score + l_attn,
            "loss_traj":  l_traj,
            "loss_score": l_score,
            "loss_attn":  l_attn,
        }

    # ── Stage 2 / Inference ───────────────────────────────────────────────────

    def select_tokens(
        self,
        traj_batch:   dict,
        queries:      Optional[list[str]]       = None,
        frame_paths:  Optional[list[list[str]]] = None,
        gumbel_noise: bool  = False,
        gumbel_temp:  float = 0.5,
        n_keep:       int   = N_KEEP,
    ) -> torch.Tensor:
        """
        Select the top-K most important patches using visual + query + trajectory.

        Args:
            traj_batch  : dict of trajectory tensors (B, T, ...) for the FULL clip
            queries     : list of B question strings (or None for null query)
            frame_paths : list of B lists of frame paths for DINOv2 visual encoding
            gumbel_noise: add Gumbel noise for stochastic exploration (Stage 2 GRPO)
            gumbel_temp : Gumbel temperature
            n_keep      : number of patches to select (default 20 = 10% of 196)

        Returns:
            mask : (B, 196) bool — True = selected patch
        """
        device = traj_batch["gaze_pos"].device
        B      = traj_batch["gaze_pos"].shape[0]

        # Encode query
        query_emb = self.query_encoder(queries, device)            # (B, D_QUERY)

        # Encode visual patches from frame images
        visual_feat = None
        if frame_paths is not None:
            visual_feat = self.visual_encoder(frame_paths, device) # (B, 196, D_enc)

        # PatchScorer: visual (query-gated) × trajectory → (B, 196) scores
        patch_scores, _ = self.encoder(traj_batch, query_emb, visual_feat)

        # Optional Gumbel noise for exploration during GRPO
        if gumbel_noise:
            gumbel = -torch.log(-torch.log(torch.rand_like(patch_scores) + 1e-8) + 1e-8)
            patch_scores = patch_scores + gumbel_temp * gumbel

        # Top-K selection
        _, topk_idx = torch.topk(patch_scores, k=n_keep, dim=-1)
        mask = torch.zeros(B, N_PATCHES, dtype=torch.bool, device=device)
        mask.scatter_(1, topk_idx, True)
        return mask    # (B, 196)

    def get_patch_scores(
        self,
        traj_batch:  dict,
        queries:     Optional[list[str]]       = None,
        frame_paths: Optional[list[list[str]]] = None,
    ) -> torch.Tensor:
        """Return raw (B, 196) patch scores for analysis."""
        device      = traj_batch["gaze_pos"].device
        query_emb   = self.query_encoder(queries, device)
        visual_feat = None
        if frame_paths is not None:
            visual_feat = self.visual_encoder(frame_paths, device)
        scores, _   = self.encoder(traj_batch, query_emb, visual_feat)
        return scores  # (B, 196)

    def save(self, path: str):
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TrajGazeV2":
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
        return model
