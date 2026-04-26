"""
VisualPatchEncoder for TrajGaze_v2.

Extracts frozen DINOv2-S/14 patch features from video frames.

Key design:
- Input frames resized to 196×196 so that DINOv2 (patch_size=14) produces
  exactly 14×14 = 196 patches — matching N_PATCHES throughout TrajGaze.
- K evenly-sampled keyframes per clip, temporally mean-pooled.
- DINOv2 backbone is frozen; only the linear projection is trained.
- Output fused with trajectory patch embeddings in PatchCrossAttention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

N_PATCHES = 196    # 14 × 14
DINO_DIM  = 384    # DINOv2-S/14 patch embedding dimension
IMG_SIZE  = 196    # 196 / 14 = 14 patches per side → 196 patches total


class VisualPatchEncoder(nn.Module):
    """
    Frozen DINOv2-S/14 + trainable projection → (B, 196, out_dim).

    Usage:
        enc = VisualPatchEncoder(n_keyframes=8, out_dim=256)
        vis_feat = enc(frame_paths_batch, device)   # (B, 196, 256)
    """

    def __init__(self, n_keyframes: int = 8, out_dim: int = 256):
        super().__init__()
        self.n_keyframes = n_keyframes
        self.out_dim     = out_dim

        # ── Frozen DINOv2-S/14 ────────────────────────────────────────────────
        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        for p in self.dino.parameters():
            p.requires_grad_(False)

        # ── Trainable projection 384 → out_dim ────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(DINO_DIM, out_dim),
            nn.LayerNorm(out_dim),
        )

        # ImageNet normalisation (fixed)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_indices(self, n: int) -> list[int]:
        if n <= self.n_keyframes:
            return list(range(n))
        step = n / self.n_keyframes
        return [int(i * step) for i in range(self.n_keyframes)]

    def _load_frame(self, path: str) -> torch.Tensor:
        """Load one frame → (3, IMG_SIZE, IMG_SIZE) float32 in [0, 1]."""
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return TF.to_tensor(img)

    # ── Per-clip encoding (frozen, no gradient) ────────────────────────────────

    def _encode_clip(self, frame_paths: list[str], device: torch.device) -> torch.Tensor:
        """
        Encode one clip → (196, DINO_DIM) mean-pooled patch features.
        Runs entirely under torch.inference_mode(); output is detached.
        """
        indices = self._sample_indices(len(frame_paths))
        imgs = []
        for i in indices:
            try:
                imgs.append(self._load_frame(frame_paths[i]))
            except Exception:
                imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        if not imgs:
            return torch.zeros(N_PATCHES, DINO_DIM, device=device)

        with torch.inference_mode():
            imgs_t = torch.stack(imgs).to(device)          # (K, 3, H, W)
            imgs_t = (imgs_t - self.mean) / self.std       # ImageNet normalise

            # DINOv2 patch features: list of length 1, each (K, 196, 384)
            patch_feats = self.dino.get_intermediate_layers(
                imgs_t, n=1, return_class_token=False
            )[0]                                           # (K, 196, 384)

            clip_feat = patch_feats.mean(dim=0)            # (196, 384)

        return clip_feat   # detached from autograd graph

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        frame_paths_batch: list[list[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            frame_paths_batch: list of B lists of frame paths
            device: target device for DINOv2 computation

        Returns:
            (B, 196, out_dim) — visual patch features, differentiable
            through the projection layer only.
        """
        raw = [self._encode_clip(fps, device) for fps in frame_paths_batch]
        raw = torch.stack(raw)          # (B, 196, 384) — no grad through DINOv2
        return self.proj(raw)           # (B, 196, out_dim) — grad through proj
