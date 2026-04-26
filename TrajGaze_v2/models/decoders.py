"""
Trajectory and Score decoders for TrajGaze_v2.

Both use non-autoregressive (parallel) cross-attention decoder:
  learned future-step queries × past trajectory context → predictions

TrajectoryDecoder: → (B, T_future, 6)  [gaze_x,y; left_x,y; right_x,y]
ScoreDecoder:      → (B, T_future, 196) [per-patch interaction score]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

D_ENC        = 256
T_FUTURE_MAX = 32   # max future steps (equals T_SAMPLE)
N_PATCHES    = 196


class CrossAttentionDecoder(nn.Module):
    """
    Non-autoregressive decoder using cross-attention.

    Learned queries for T_future_max future steps cross-attend to
    per-frame trajectory context (flattened over past timesteps).
    """

    def __init__(
        self,
        d_model:    int,
        out_dim:    int,
        n_future:   int = T_FUTURE_MAX,
        n_layers:   int = 3,
        n_heads:    int = 8,
    ):
        super().__init__()
        self.n_future = n_future

        # Learnable queries for each future time step
        self.future_queries = nn.Parameter(torch.randn(n_future, d_model) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model         = d_model,
                nhead           = n_heads,
                dim_feedforward = d_model * 4,
                dropout         = 0.1,
                activation      = "gelu",
                batch_first     = True,
                norm_first      = True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, context: torch.Tensor, T_future: int) -> torch.Tensor:
        """
        context: (B, T_past, 4, D_ENC) — past trajectory context
        T_future: number of future steps to predict

        Returns: (B, T_future, out_dim)
        """
        B = context.shape[0]

        # Flatten context: (B, T_past*4, D_ENC)
        memory = context.reshape(B, -1, context.shape[-1])  # (B, T_past*4, D_ENC)

        # Expand future queries: (B, T_future, D)
        queries = self.future_queries[:T_future].unsqueeze(0).expand(B, -1, -1)  # (B, T_future, D)

        x = queries
        for layer in self.layers:
            x = layer(x, memory)

        x = self.norm(x)                  # (B, T_future, D)
        return self.head(x)               # (B, T_future, out_dim)


class TrajectoryDecoder(nn.Module):
    """
    Predicts future gaze and hand positions.

    Output: (B, T_future, 6)
        dim 0,1: gaze_x, gaze_y
        dim 2,3: left_x, left_y
        dim 4,5: right_x, right_y
    All in [0, 1] after sigmoid.
    """

    TRAJ_DIM = 6  # gaze(2) + left(2) + right(2)

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
        """
        context: (B, T_past, 4, D_ENC)
        Returns: (B, T_future, 6), values in [0,1]
        """
        out = self.decoder(context, T_future)  # (B, T_future, 6)
        return torch.sigmoid(out)


class ScoreDecoder(nn.Module):
    """
    Predicts future per-patch interaction scores.

    Output: (B, T_future, 196), values in [0,1] after sigmoid.
    """

    def __init__(self, d_model: int = D_ENC, n_future: int = T_FUTURE_MAX, n_patches: int = N_PATCHES):
        super().__init__()
        # Use bottleneck: predict to smaller dim then expand
        self.decoder = CrossAttentionDecoder(
            d_model   = d_model,
            out_dim   = d_model,     # intermediate
            n_future  = n_future,
            n_layers  = 3,
            n_heads   = 8,
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, n_patches),
        )

    def forward(self, context: torch.Tensor, T_future: int) -> torch.Tensor:
        """
        context: (B, T_past, 4, D_ENC)
        Returns: (B, T_future, 196), values in [0,1]
        """
        x = self.decoder(context, T_future)   # (B, T_future, D_ENC)
        return torch.sigmoid(self.head(x))    # (B, T_future, 196)


# ── Loss functions ────────────────────────────────────────────────────────────

def traj_loss(
    pred: torch.Tensor,      # (B, T_future, 6) — predicted positions
    future: dict,            # dict with keys gaze_pos, left_pos, right_pos, *_mask
    T_future_tensor: torch.Tensor,  # (B,) actual future lengths
) -> torch.Tensor:
    """
    Masked MSE loss on future trajectory prediction.

    Pred layout: [..., 0:2] = gaze, [..., 2:4] = left, [..., 4:6] = right
    Masks from future dict gate which frames/coordinates are valid.
    """
    B = pred.shape[0]
    T_future_max = pred.shape[1]
    device = pred.device

    # Build valid-length mask (B, T_future_max)
    len_mask = torch.zeros(B, T_future_max, dtype=torch.bool, device=device)
    for i, T_f in enumerate(T_future_tensor):
        len_mask[i, :T_f.item()] = True

    loss = torch.tensor(0.0, device=device)
    n_terms = 0

    # Gaze loss
    gaze_mask = future["gaze_mask"] & len_mask  # (B, T_future_max)
    if gaze_mask.any():
        gaze_pred = pred[:, :, 0:2]   # (B, T, 2)
        gaze_gt   = future["gaze_pos"][:, :T_future_max]  # (B, T, 2)
        gaze_msk  = gaze_mask.unsqueeze(-1).float()
        gaze_loss = ((gaze_pred - gaze_gt) ** 2 * gaze_msk).sum() / (gaze_msk.sum() * 2 + 1e-6)
        loss     = loss + gaze_loss
        n_terms  += 1

    # Left hand loss
    left_mask = future["left_mask"] & len_mask
    if left_mask.any():
        left_pred = pred[:, :, 2:4]
        left_gt   = future["left_pos"][:, :T_future_max]
        left_msk  = left_mask.unsqueeze(-1).float()
        left_loss = ((left_pred - left_gt) ** 2 * left_msk).sum() / (left_msk.sum() * 2 + 1e-6)
        loss     = loss + left_loss
        n_terms  += 1

    # Right hand loss
    right_mask = future["right_mask"] & len_mask
    if right_mask.any():
        right_pred = pred[:, :, 4:6]
        right_gt   = future["right_pos"][:, :T_future_max]
        right_msk  = right_mask.unsqueeze(-1).float()
        right_loss = ((right_pred - right_gt) ** 2 * right_msk).sum() / (right_msk.sum() * 2 + 1e-6)
        loss       = loss + right_loss
        n_terms    += 1

    return loss / max(1, n_terms)


def score_loss(
    pred: torch.Tensor,             # (B, T_future, 196)
    I_scores_gt: torch.Tensor,      # (B, T_future_max, 196)
    T_future_tensor: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """MSE loss on future interaction score prediction with length masking."""
    B, T_future_pred, N = pred.shape
    T_future_max = min(T_future_pred, I_scores_gt.shape[1])
    device = pred.device

    # Length mask (B, T_future_max)
    len_mask = torch.zeros(B, T_future_max, dtype=torch.bool, device=device)
    for i, T_f in enumerate(T_future_tensor):
        T_f_val = min(T_f.item(), T_future_max)
        len_mask[i, :T_f_val] = True

    gt   = I_scores_gt[:, :T_future_max]  # (B, T, 196)
    p    = pred[:, :T_future_max]         # (B, T, 196)
    msk  = len_mask.unsqueeze(-1).float() # (B, T, 1)
    return ((p - gt) ** 2 * msk).sum() / (msk.sum() * N + 1e-6)
