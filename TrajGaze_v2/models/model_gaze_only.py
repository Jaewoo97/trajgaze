"""
TrajGazeV2GazeOnly: ablation model using only gaze trajectory (no hand/interaction).

Drop-in replacement for TrajGazeV2 with:
  - SpatiotemporalEncoderGazeOnly (1 token/frame vs 4)
  - TrajectoryDecoderGazeOnly     (2-dim output: gaze x,y only)
  - traj_loss_gaze_only           (MSE on gaze position only)
All other modules (ScoreDecoder, VisualPatchEncoder, QueryEncoder) unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_encoder  import QueryEncoder, D_QUERY
from .encoder        import D_ENC, N_PATCHES
from .encoder_gaze_only import SpatiotemporalEncoderGazeOnly
from .decoders       import CrossAttentionDecoder, ScoreDecoder, score_loss
from .visual_encoder import VisualPatchEncoder

GAZING_RATIO = 0.10
N_KEEP       = max(1, int(math.ceil(GAZING_RATIO * N_PATCHES)))  # 20 patches = 10%
T_FUTURE_MAX = 32


class TrajectoryDecoderGazeOnly(nn.Module):
    """Predicts future gaze positions only. Output: (B, T_future, 2), values in [0,1]."""
    TRAJ_DIM = 2  # gaze_x, gaze_y

    def __init__(self, d_model: int = D_ENC, n_future: int = T_FUTURE_MAX):
        super().__init__()
        self.decoder = CrossAttentionDecoder(
            d_model   = d_model,
            out_dim   = self.TRAJ_DIM,
            n_future  = n_future,
            n_layers  = 3,
            n_heads   = 8,
        )

    def forward(self, context: torch.Tensor, T_future: int) -> torch.Tensor:
        """context: (B, T_past, 1, D_ENC) → (B, T_future, 2)"""
        return torch.sigmoid(self.decoder(context, T_future))


def traj_loss_gaze_only(
    pred:             torch.Tensor,       # (B, T_future, 2) — predicted gaze positions
    future:           dict,               # dict with gaze_pos, gaze_mask (+ unused hand keys)
    T_future_tensor:  torch.Tensor,       # (B,)
) -> torch.Tensor:
    """Masked MSE on future gaze position prediction."""
    B            = pred.shape[0]
    T_future_max = pred.shape[1]
    device       = pred.device

    len_mask = torch.zeros(B, T_future_max, dtype=torch.bool, device=device)
    for i, T_f in enumerate(T_future_tensor):
        len_mask[i, :T_f.item()] = True

    gaze_mask = future["gaze_mask"] & len_mask   # (B, T_future_max)
    if not gaze_mask.any():
        return torch.tensor(0.0, device=device)

    gaze_gt  = future["gaze_pos"][:, :T_future_max]   # (B, T, 2)
    gaze_msk = gaze_mask.unsqueeze(-1).float()
    return ((pred - gaze_gt) ** 2 * gaze_msk).sum() / (gaze_msk.sum() * 2 + 1e-6)


class TrajGazeV2GazeOnly(nn.Module):
    """
    Gaze-only ablation of TrajGazeV2.

    Stage 1: predict future gaze positions + future patch scores from past gaze trajectory.
    Inference: same interface as TrajGazeV2 (get_patch_scores, query_encoder, visual_encoder).

    Fully compatible with train_merge_lora_gaze_only.py — traj_batch dicts can contain
    hand fields; the encoder will silently ignore them.
    """

    def __init__(
        self,
        d_traj:          int = 128,
        d_enc:           int = D_ENC,
        d_query:         int = D_QUERY,
        n_layers_l2:     int = 6,
        n_heads_l2:      int = 8,
        t_future_max:    int = T_FUTURE_MAX,
        n_vis_keyframes: int = 8,
    ):
        super().__init__()
        self.query_encoder  = QueryEncoder(d_model=d_query)
        self.visual_encoder = VisualPatchEncoder(n_keyframes=n_vis_keyframes, out_dim=d_enc)
        self.encoder        = SpatiotemporalEncoderGazeOnly(
            d_traj=d_traj, d_enc=d_enc, d_query=d_query,
            n_layers_l2=n_layers_l2, n_heads_l2=n_heads_l2,
        )
        self.traj_decoder  = TrajectoryDecoderGazeOnly(d_model=d_enc, n_future=t_future_max)
        self.score_decoder = ScoreDecoder(d_model=d_enc,              n_future=t_future_max)

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_forward(self, batch: dict) -> dict[str, torch.Tensor]:
        past    = batch["past"]
        future  = batch["future"]
        T_f_max = int(batch["T_future"].max().item())

        device = past["gaze_pos"].device
        B      = past["gaze_pos"].shape[0]

        query_emb = torch.zeros(B, self.query_encoder.d_model, device=device)

        frame_paths_batch = batch.get("frame_paths")
        visual_feat = None
        if frame_paths_batch is not None:
            visual_feat = self.visual_encoder(frame_paths_batch, device)

        patch_scores, context = self.encoder(past, query_emb, visual_feat)

        traj_pred  = self.traj_decoder(context, T_f_max)   # (B, T_f_max, 2)
        score_pred = self.score_decoder(context, T_f_max)  # (B, T_f_max, 196)

        l_traj  = traj_loss_gaze_only(traj_pred, future, batch["T_future"])
        l_score = score_loss(score_pred, batch["I_scores_future"], batch["T_future"])
        I_mean  = batch["I_scores_future"].mean(dim=1)
        l_attn  = F.mse_loss(patch_scores, I_mean)

        return {
            "loss":       l_traj + l_score + l_attn,
            "loss_traj":  l_traj,
            "loss_score": l_score,
            "loss_attn":  l_attn,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_patch_scores(
        self,
        traj_batch:  dict,
        queries:     Optional[list[str]]       = None,
        frame_paths: Optional[list[list[str]]] = None,
    ) -> torch.Tensor:
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
    def load(cls, path: str, **kwargs) -> "TrajGazeV2GazeOnly":
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
        return model
