"""
Step 3b — Causal video encoder.

Encodes attended frames (RGB + interaction heatmap as 4th channel) into
patch embeddings.

Architecture:
  - Input:  (B, T_att, 4, H, W) — RGB + interaction heatmap
  - 2D conv for spatial patch embedding (stride 16 → 14×14 = 196 patches)
  - Causal 3D conv: temporal kernel sees current + 2 previous attended frames
  - Output: (B, T_att·N_patches, d_model=192)

The 4th channel couples the visual stream to the same gaze-hand interaction
signal that the trajectory encoder sees.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

D_MODEL   = 192
N_PATCHES = 14 * 14   # 196 for 224×224 input with stride 16
PATCH_DIM = 16        # spatial stride in pixels


class VideoEncoder(nn.Module):
    """
    Causal video encoder for attended frames only.

    Input:  (B, T_att, 4, H=224, W=224)
    Output: (B, T_att · N_patches, d_model=192)
    """

    def __init__(
        self,
        in_channels: int = 4,       # RGB + interaction heatmap
        d_model:     int = D_MODEL,
        patch_size:  int = PATCH_DIM,
    ):
        super().__init__()
        self.d_model    = d_model
        self.patch_size = patch_size

        # Spatial patch embedding: each 16×16 region → d_model vector
        # kernel_size = patch_size, stride = patch_size → no overlap
        self.patch_embed = nn.Conv2d(
            in_channels = in_channels,
            out_channels = d_model,
            kernel_size  = patch_size,
            stride       = patch_size,
        )  # (B, d_model, 14, 14)

        # Causal temporal aggregation:
        # 3D conv with temporal kernel=3, pad=0 (causal via manual left-pad)
        # Sees: [t-2, t-1, t] → output at t
        self.temporal_conv = nn.Conv3d(
            in_channels  = d_model,
            out_channels = d_model,
            kernel_size  = (3, 1, 1),
            stride       = 1,
            padding      = 0,
            bias         = False,
        )

        # LayerNorm over patch dimension
        self.norm = nn.LayerNorm(d_model)

        # Learnable positional embedding for patch positions (196 patches)
        self.pos_embed = nn.Parameter(
            torch.randn(1, N_PATCHES, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_att, 4, H, W)

        Returns:
            (B, T_att · N_patches, d_model)
        """
        B, T, C, H, W = x.shape

        # Spatial patch embedding: process all frames at once
        x_flat = x.view(B * T, C, H, W)
        patches = self.patch_embed(x_flat)          # (B·T, d_model, 14, 14)
        _, D, Ph, Pw = patches.shape

        # Reshape for 3D causal conv: (B, d_model, T, Ph, Pw)
        x_3d = patches.view(B, T, D, Ph, Pw).permute(0, 2, 1, 3, 4)  # (B, D, T, Ph, Pw)

        # Causal left-padding: pad 2 frames on the left of T dimension
        # This ensures temporal conv sees [t-2, t-1, t] without future leakage
        x_3d_padded = torch.nn.functional.pad(x_3d, (0, 0, 0, 0, 2, 0))  # pad T dim

        # 3D conv: temporal kernel=3 sees exactly [t-2, t-1, t]
        x_3d = self.temporal_conv(x_3d_padded)      # (B, D, T, Ph, Pw)

        # Reshape to (B, T, Ph·Pw, D)
        x_3d = x_3d.permute(0, 2, 3, 4, 1)          # (B, T, Ph, Pw, D)
        x_3d = x_3d.reshape(B, T, Ph * Pw, D)        # (B, T, N_patches, D)

        # Add 2D positional embedding
        x_3d = x_3d + self.pos_embed                 # (B, T, N_patches, D)

        # Flatten time + patches: (B, T·N_patches, D)
        x_out = x_3d.reshape(B, T * Ph * Pw, D)
        x_out = self.norm(x_out)

        return x_out
