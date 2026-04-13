"""
Step 2b — Frame selector.

A 2-layer causal transformer over all T interaction token embeddings.
Produces a per-frame importance score s(t) ∈ (0, 1) that decides
attend/skip for each frame.

Architecture:
  - Input: (B, T, d_model=128) interaction tokens only (no visual features)
  - Causal sinusoidal positional encoding
  - 2-layer causal self-attention (4 heads, d_model=128, d_ff=256)
  - MLP head per frame → sigmoid → s(t)
  - Training: L_frame_BCE against attend(t) labels

Causal design ensures s(t) depends only on t' ≤ t.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Additive sinusoidal positional encoding. Supports dynamic lengths."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, : x.size(1)]


class CausalTransformerLayer(nn.Module):
    """Single causal self-attention + FFN layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, T, d_model)
        causal_mask: (T, T) bool — True where position should be ignored
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.drop(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.drop(self.ff(x)) + residual
        return x


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular bool mask: True = masked (future positions blocked)."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class FrameSelector(nn.Module):
    """
    Causal frame selector.

    Ingests interaction tokens for all T frames and outputs per-frame
    attend probability s(t) ∈ (0, 1).

    At training: soft gating, L_frame_BCE supervises all T positions.
    At inference: hard threshold → binary attend/skip.
    """

    def __init__(
        self,
        d_model:   int = 128,
        n_heads:   int = 4,
        d_ff:      int = 256,
        n_layers:  int = 2,
        dropout:   float = 0.1,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model   = d_model
        self.threshold = threshold

        self.pos_enc = SinusoidalPE(d_model)
        self.layers  = nn.ModuleList([
            CausalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        # Scalar head: d_model → 64 → 1
        self.head    = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, tok_interact: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tok_interact: (B, T, d_model) — interaction tokens for all T frames

        Returns:
            scores: (B, T) float — per-frame attend probabilities ∈ (0, 1)
        """
        B, T, _ = tok_interact.shape
        x = self.pos_enc(tok_interact)

        mask = _causal_mask(T, tok_interact.device)
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)                       # (B, T, d_model)
        logits = self.head(x).squeeze(-1)      # (B, T)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def select_frames(
        self,
        tok_interact: torch.Tensor,
        ratio: float | None = None,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Hard frame selection at inference time.

        Uses top-k selection when ratio is given (precise budget control),
        otherwise falls back to threshold-based selection.

        Args:
            tok_interact: (B, T, d_model)
            ratio:        select exactly this fraction of frames (e.g. 0.50)
            threshold:    fallback — select frames with score ≥ threshold

        Returns:
            scores:       (B, T) attend probabilities
            attended_idx: list of lists — attended frame indices per batch item
        """
        scores = self.forward(tok_interact)   # (B, T)
        B, T = scores.shape

        if ratio is not None:
            k = max(1, round(ratio * T))
            attended_idx = [
                scores[b].topk(k).indices.sort().values.tolist()
                for b in range(B)
            ]
        else:
            thresh = threshold if threshold is not None else self.threshold
            attended_idx = [
                torch.nonzero(scores[b] >= thresh, as_tuple=False).squeeze(-1).tolist()
                for b in range(B)
            ]
        return scores, attended_idx


def frame_selector_loss(
    scores:  torch.Tensor,   # (B, T)  predicted attend probabilities
    targets: torch.Tensor,   # (B, T)  int8/float {0, 1}
    mask:    torch.Tensor | None = None,  # (B, T) bool — True = valid
) -> torch.Tensor:
    """
    Binary cross-entropy loss over all T frames.
    Optional padding mask (for variable-length batches).
    """
    targets = targets.float()
    # F.binary_cross_entropy raises an error under any autocast context.
    # Disable autocast locally and compute in float32.
    with torch.amp.autocast("cuda", enabled=False):
        loss = F.binary_cross_entropy(scores.float(), targets, reduction="none")  # (B, T)
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-8)
    return loss.mean()
