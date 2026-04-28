"""
VisualPatchEncoderTemporal for TrajGaze_v2.

Extracts frozen DINOv2-S/14 patch features per-frame, returning a temporal
sequence of spatial feature maps rather than a single mean-pooled map.

Pipeline:
  1. Sample K evenly-spaced keyframes from the clip
  2. Run DINOv2-S/14 on all K frames in one batch → (K, 196, 384)
  3. Linearly interpolate along the time axis to T_target → (T_target, 196, 384)
  4. Trainable projection 384 → out_dim with LayerNorm
  5. Output: (B, T_target, 196, out_dim)

DINOv2 backbone is fully frozen; only the linear projection is trained.
Frames resized to 196×196 so DINOv2 (patch_size=14) yields exactly 196 patches.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PATCHES = 196   # 14×14
DINO_DIM  = 384   # DINOv2-S/14 patch embedding dimension
IMG_SIZE  = 196   # 196/14 = 14 patches per side


class VisualPatchEncoderTemporal(nn.Module):
    """
    Frozen DINOv2-S/14 + trainable projection → (B, T, 196, out_dim).

    Usage:
        enc = VisualPatchEncoderTemporal(n_keyframes=16, out_dim=256)
        vis_feat = enc(frame_paths_batch, T_target=128, device=device)
        # → (B, 128, 196, 256)
    """

    def __init__(self, n_keyframes: int = 16, out_dim: int = 256):
        super().__init__()
        self.n_keyframes = n_keyframes
        self.out_dim     = out_dim

        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        for p in self.dino.parameters():
            p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Linear(DINO_DIM, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _sample_indices(self, n: int) -> list[int]:
        if n <= self.n_keyframes:
            return list(range(n))
        step = n / self.n_keyframes
        return [int(i * step) for i in range(self.n_keyframes)]

    def _load_frame(self, path: str) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return TF.to_tensor(img)

    def _encode_clip_temporal(
        self,
        frame_paths: list[str],
        T_target:    int,
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Encode one clip → (T_target, 196, DINO_DIM) with linear time interpolation.
        Runs entirely under torch.inference_mode(); output is detached.
        """
        indices = self._sample_indices(len(frame_paths))
        imgs = []
        for i in indices:
            try:
                imgs.append(self._load_frame(frame_paths[i]))
            except Exception:
                imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        K = len(imgs)
        if K == 0:
            return torch.zeros(T_target, N_PATCHES, DINO_DIM, device=device)

        with torch.inference_mode():
            imgs_t = torch.stack(imgs).to(device)           # (K, 3, H, W)
            imgs_t = (imgs_t - self.mean) / self.std

            patch_feats = self.dino.get_intermediate_layers(
                imgs_t, n=1, return_class_token=False
            )[0]                                            # (K, 196, 384)

            if K == T_target:
                return patch_feats   # detached

            # Temporal linear interpolation: (K, 196*384) → (T_target, 196*384)
            feats_flat = patch_feats.reshape(1, K, -1).permute(0, 2, 1)  # (1, 196*384, K)
            feats_interp = F.interpolate(
                feats_flat.float(), size=T_target,
                mode="linear", align_corners=False,
            )                                               # (1, 196*384, T_target)
            feats_out = feats_interp.squeeze(0).permute(1, 0).reshape(
                T_target, N_PATCHES, DINO_DIM
            )                                              # (T_target, 196, 384)

        return feats_out   # detached

    def forward(
        self,
        frame_paths_batch: list[list[str]],
        T_target:          int,
        device:            torch.device,
    ) -> torch.Tensor:
        """
        Args:
            frame_paths_batch : list of B clip frame-path lists
            T_target          : number of temporal steps to output
            device            : compute device

        Returns:
            (B, T_target, 196, out_dim) — per-frame visual patch features,
            differentiable through the projection layer only.
        """
        raw = [self._encode_clip_temporal(fps, T_target, device)
               for fps in frame_paths_batch]
        raw = torch.stack(raw)      # (B, T_target, 196, 384) — no grad through DINOv2
        return self.proj(raw)       # (B, T_target, 196, out_dim) — grad through proj
