"""Per-token gaze/hand salience tag — additive annotation of retained tokens.

A different axis from the scanpath channel (which adds sequence-level *intent*
tokens). Here we keep M1's exact token selection and count, and add a small
learned embedding to EACH retained visual token encoding *how gaze/hand-relevant
that token is* — i.e. tell the LLM which of the patches it sees the person
actually looked at or manipulated. The salience is the same trajectory score M1
uses to pick its complement; binning it and learning a per-bin embedding turns a
selection signal into a per-token annotation the LLM can read.

Strictly additive (no token added/removed, selection unchanged) -> escapes the
fixed-budget zero-sum. Flamingo-style gate (small init) so the tag fades in."""
from __future__ import annotations

import torch
import torch.nn as nn


class GazeTagEmbedding(nn.Module):
    """Map a per-token salience in [0,1] -> additive (M, out_dim) tag embedding."""

    def __init__(self, out_dim: int, n_bins: int = 16, gate_init: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        self.table = nn.Embedding(n_bins, out_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, sal01: torch.Tensor) -> torch.Tensor:
        """sal01: (M,) normalized salience in [0,1]. Returns (M, out_dim)."""
        sal01 = sal01.detach().clamp(0.0, 1.0)          # salience is a fixed signal
        bins = torch.clamp((sal01 * self.n_bins).long(), 0, self.n_bins - 1)
        return self.gate * self.table(bins)
