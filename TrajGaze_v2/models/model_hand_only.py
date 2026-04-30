"""
TrajGazeV2HandOnly: ablation model using only hand trajectory (no gaze).

Drop-in replacement for TrajGazeV2 with:
  - SpatiotemporalEncoderHandOnly (3 tokens/frame: left, right, bimanual)
  - TrajectoryDecoderHandOnly (4-dim output: left x,y + right x,y)
  - traj_loss_hand_only (MSE on hand positions only)
Visual input is retained unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_encoder        import QueryEncoder, D_QUERY
from .encoder              import D_ENC, N_PATCHES
from .encoder_hand_only    import SpatiotemporalEncoderHandOnly
from .decoders             import CrossAttentionDecoder, ScoreDecoder, score_loss
from .visual_encoder       import VisualPatchEncoder

GAZING_RATIO = 0.10
N_KEEP       = max(1, int(math.ceil(GAZING_RATIO * N_PATCHES)))
T_FUTURE_MAX = 32


class TrajectoryDecoderHandOnly(nn.Module):
    """Predicts future left and right hand positions. Output: (B, T_future, 4)."""
    TRAJ_DIM = 4  # left_x, left_y, right_x, right_y

    def __init__(self, d_model: int = D_ENC, n_future: int = T_FUTURE_MAX):
        super().__init__()
        self.decoder = CrossAttentionDecoder(
            d_model  = d_model,
            out_dim  = self.TRAJ_DIM,
            n_future = n_future,
            n_layers = 3,
            n_heads  = 8,
        )

    def forward(self, context: torch.Tensor, T_future: int) -> torch.Tensor:
        """context: (B, T_past, 3, D_ENC) → (B, T_future, 4)"""
        return torch.sigmoid(self.decoder(context, T_future))


def traj_loss_hand_only(
    pred:            torch.Tensor,   # (B, T_future, 4) — left(2) + right(2)
    future:          dict,
    T_future_tensor: torch.Tensor,   # (B,)
) -> torch.Tensor:
    """Masked MSE on future left and right hand positions."""
    B            = pred.shape[0]
    T_future_max = pred.shape[1]
    device       = pred.device

    len_mask = torch.zeros(B, T_future_max, dtype=torch.bool, device=device)
    for i, T_f in enumerate(T_future_tensor):
        len_mask[i, :T_f.item()] = True

    loss    = torch.tensor(0.0, device=device)
    n_terms = 0

    left_mask = future["left_mask"] & len_mask
    if left_mask.any():
        left_pred = pred[:, :, 0:2]
        left_gt   = future["left_pos"][:, :T_future_max]
        left_msk  = left_mask.unsqueeze(-1).float()
        loss     += ((left_pred - left_gt) ** 2 * left_msk).sum() / (left_msk.sum() * 2 + 1e-6)
        n_terms  += 1

    right_mask = future["right_mask"] & len_mask
    if right_mask.any():
        right_pred = pred[:, :, 2:4]
        right_gt   = future["right_pos"][:, :T_future_max]
        right_msk  = right_mask.unsqueeze(-1).float()
        loss      += ((right_pred - right_gt) ** 2 * right_msk).sum() / (right_msk.sum() * 2 + 1e-6)
        n_terms   += 1

    return loss / max(1, n_terms)


class TrajGazeV2HandOnly(nn.Module):
    """
    Hand-only ablation of TrajGazeV2.

    Stage 1: predict future hand positions + future patch scores from past hand trajectory.
    Inference: same interface as TrajGazeV2 (get_patch_scores, query_encoder, visual_encoder).

    Gaze fields in traj_batch dicts are silently ignored by the encoder.
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
        self.encoder        = SpatiotemporalEncoderHandOnly(
            d_traj=d_traj, d_enc=d_enc, d_query=d_query,
            n_layers_l2=n_layers_l2, n_heads_l2=n_heads_l2,
        )
        self.traj_decoder  = TrajectoryDecoderHandOnly(d_model=d_enc, n_future=t_future_max)
        self.score_decoder = ScoreDecoder(d_model=d_enc,              n_future=t_future_max)

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_forward(self, batch: dict) -> dict[str, torch.Tensor]:
        past    = batch["past"]
        future  = batch["future"]
        T_f_max = int(batch["T_future"].max().item())

        device = past["left_pos"].device
        B      = past["left_pos"].shape[0]

        query_emb = torch.zeros(B, self.query_encoder.d_model, device=device)

        frame_paths_batch = batch.get("frame_paths")
        visual_feat = None
        if frame_paths_batch is not None:
            visual_feat = self.visual_encoder(frame_paths_batch, device)

        patch_scores, context = self.encoder(past, query_emb, visual_feat)

        traj_pred  = self.traj_decoder(context, T_f_max)   # (B, T_f_max, 4)
        score_pred = self.score_decoder(context, T_f_max)  # (B, T_f_max, 196)

        l_traj  = traj_loss_hand_only(traj_pred, future, batch["T_future"])
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
        device      = traj_batch["left_pos"].device
        query_emb   = self.query_encoder(queries, device)
        visual_feat = None
        if frame_paths is not None:
            visual_feat = self.visual_encoder(frame_paths, device)
        scores, _   = self.encoder(traj_batch, query_emb, visual_feat)
        return scores   # (B, 196)

    def save(self, path: str):
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TrajGazeV2HandOnly":
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
        return model
