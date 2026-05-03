"""
TrajGazeV2TemporalGazeOnly — gaze-only ablation of TrajGazeV2Temporal.

Identical to TrajGazeV2Temporal (Option C, TrajScoreHead) except:
  - SpatiotemporalEncoderTemporalGazeOnly: 1 token/frame (gaze only)
  - TrajScoreHead pools over 1 token instead of 4
  - TrajectoryDecoder cross-attn memory: (B, T_past*1, D)

All 4 training losses and inference path are identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .query_encoder                  import QueryEncoder, D_QUERY
from .encoder                        import D_ENC, N_PATCHES
from .encoder_temporal_gaze_only     import SpatiotemporalEncoderTemporalGazeOnly
from .decoders                       import (
    CrossAttentionDecoder, CrossAttentionDecoderTracked,
    score_loss, traj_loss,
)
from .visual_encoder_temporal        import VisualPatchEncoderTemporal
from .model_temporal                 import (
    TrajScoreHead, TrajectoryDecoderTemporal, ScoreDecoderTemporal,
    score_loss_temporal, T_FUTURE_MAX,
)


class TrajGazeV2TemporalGazeOnly(nn.Module):

    def __init__(
        self,
        d_traj:                    int  = 128,
        d_enc:                     int  = D_ENC,
        d_query:                   int  = D_QUERY,
        n_layers_l2:               int  = 6,
        n_heads_l2:                int  = 8,
        t_future_max:              int  = T_FUTURE_MAX,
        n_vis_keyframes:           int  = 16,
        use_patch_temporal_branch: bool = False,
    ):
        super().__init__()
        self.query_encoder  = QueryEncoder(d_model=d_query)
        self.visual_encoder = VisualPatchEncoderTemporal(n_keyframes=n_vis_keyframes, out_dim=d_enc)
        self.encoder = SpatiotemporalEncoderTemporalGazeOnly(
            d_traj=d_traj, d_enc=d_enc, d_vis=d_enc,
            d_query=d_query, n_layers_l2=n_layers_l2, n_heads_l2=n_heads_l2,
            use_patch_temporal_branch=use_patch_temporal_branch,
        )
        self.traj_decoder  = TrajectoryDecoderTemporal(d_model=d_enc, n_future=t_future_max)
        self.score_decoder = ScoreDecoderTemporal(d_model=d_enc, n_future=t_future_max)
        self.score_head    = TrajScoreHead(d_enc=d_enc, n_patches=N_PATCHES)

    def stage1_forward(self, batch: dict) -> dict[str, torch.Tensor]:
        past    = batch["past"]
        future  = batch["future"]
        T_past  = batch["T_past"]
        T_f_max = int(batch["T_future"].max().item())
        T_p_max = int(T_past.max().item())
        device  = past["left_pos"].device
        B       = past["left_pos"].shape[0]

        query_emb   = torch.zeros(B, self.query_encoder.d_model, device=device)
        visual_feat = None
        if (fps_batch := batch.get("frame_paths")) is not None:
            past_paths  = [fps[:int(T_past[i].item())] for i, fps in enumerate(fps_batch)]
            visual_feat = self.visual_encoder(past_paths, T_p_max, device)

        past_scores, context, enc_attn = self.encoder(past, query_emb, visual_feat)
        score_head_out = self.score_head(context)

        traj_pred, dec_attn = self.traj_decoder(context, T_f_max, return_cross_weights=True)
        score_pred = self.score_decoder(context, T_f_max)

        l_score_traj = torch.tensor(0.0, device=device)
        if enc_attn is not None:
            n_tok = context.shape[2]  # 1 for gaze-only
            dec_importance = dec_attn.mean(dim=1).reshape(B, T_p_max, n_tok)
            traj_driven = (enc_attn * dec_importance.unsqueeze(-1)).sum(dim=2)
            t_max = traj_driven.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            traj_driven = (traj_driven / t_max).detach()
            l_score_traj = score_loss_temporal(score_head_out, traj_driven, batch["T_past"])

        l_traj         = traj_loss(traj_pred, future, batch["T_future"])
        l_score_future = score_loss_temporal(score_pred,  batch["I_scores_future"], batch["T_future"])
        l_score_past   = score_loss_temporal(past_scores, batch["I_scores_past"],   batch["T_past"])
        total = l_traj + l_score_future + l_score_past + l_score_traj

        return {
            "loss": total, "loss_traj": l_traj,
            "loss_score_fut": l_score_future, "loss_score_past": l_score_past,
            "loss_score_traj": l_score_traj,
        }

    def get_patch_scores(
        self,
        traj_batch:  dict,
        queries:     Optional[list[str]]       = None,
        frame_paths: Optional[list[list[str]]] = None,
    ) -> torch.Tensor:
        device    = traj_batch["left_pos"].device
        T         = traj_batch["left_pos"].shape[1]
        query_emb = self.query_encoder(queries, device)
        visual_feat = None
        if frame_paths is not None:
            visual_feat = self.visual_encoder(frame_paths, T, device)
        _, context, _ = self.encoder(traj_batch, query_emb, visual_feat)
        return self.score_head(context)

    def save(self, path: str):
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TrajGazeV2TemporalGazeOnly":
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
        return model
