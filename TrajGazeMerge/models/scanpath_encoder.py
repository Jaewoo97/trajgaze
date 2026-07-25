"""Gaze-scanpath → LLM token channel.

Every prior method in this repo (VisionZip, VZ-traj, M1 complement, anchored, TAS,
coverage, fusion) uses gaze to decide *which video tokens to keep* — a selection
mask over the fixed 10% budget. Three converging null/falsified results show that
re-selection is zero-sum: emphasising gaze tokens just trades scene-context tokens,
and the weak spatial/temporal/future-action tasks don't move because they aren't
bottlenecked on *which* patches are kept — they lack a *behavioural* signal.

This module takes the orthogonal route: treat the gaze trajectory as its own
modality. It encodes the ordered (x, y) fixation sequence (+ both hands) into a
small set of K tokens and inserts them into the LLM sequence ALONGSIDE M1's
unchanged token selection. The scanpath is a predictive trace of intent (gaze
leads action by ~0.5-1 s), so these tokens add information no patch reshuffle can —
targeting exactly the reasoning tasks, at near-zero budget cost (K≈8 vs hundreds
of kept video tokens).

Output is Flamingo-style gated (small init) so the channel fades in during
training instead of perturbing the pretrained model from step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from TrajGazeMerge.models.traj_weights import _pool_to_T

# gaze(x,y) + gaze_valid + gaze_vel(dx,dy) + left(x,y)+valid + right(x,y)+valid
FEAT_DIM = 11


def _traj_features(traj: dict | None, t_scan: int, device: torch.device) -> torch.Tensor:
    """Resample the gaze/hand scanpath to a fixed length and stack per-step features.

    Returns (t_scan, FEAT_DIM). Missing / empty / malformed trajectory -> all-zero
    features (a valid "no gaze" input the encoder still maps to K tokens)."""
    keys = ("gaze_pos", "gaze_mask", "left_pos", "left_mask", "right_pos", "right_mask")
    if not isinstance(traj, dict) or any(k not in traj for k in keys):
        return torch.zeros(t_scan, FEAT_DIM, device=device)

    def grab(pkey: str, mkey: str):
        try:
            pos = torch.as_tensor(traj[pkey], dtype=torch.float32)
            msk = torch.as_tensor(traj[mkey])
        except Exception:
            return torch.zeros(t_scan, 2), torch.zeros(t_scan)
        if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] != 2:
            return torch.zeros(t_scan, 2), torch.zeros(t_scan)
        p_t, m_t = _pool_to_T(pos, msk, t_scan)
        return p_t.float(), m_t.float()

    g_pos, g_m = grab("gaze_pos", "gaze_mask")
    l_pos, l_m = grab("left_pos", "left_mask")
    r_pos, r_m = grab("right_pos", "right_mask")
    g_pos, l_pos, r_pos = g_pos.to(device), l_pos.to(device), r_pos.to(device)
    g_m, l_m, r_m = g_m.to(device), l_m.to(device), r_m.to(device)

    g_vel = torch.zeros_like(g_pos)
    if t_scan > 1:
        g_vel[1:] = (g_pos[1:] - g_pos[:-1]) * (g_m[1:] * g_m[:-1]).unsqueeze(-1)

    return torch.cat([
        g_pos, g_m.unsqueeze(-1), g_vel,
        l_pos, l_m.unsqueeze(-1),
        r_pos, r_m.unsqueeze(-1),
    ], dim=-1)  # (t_scan, FEAT_DIM)


class ScanpathEncoder(nn.Module):
    """Scanpath (gaze+hand sequence) -> K LLM tokens at the model hidden dim.

    Transformer encoder over the resampled scanpath, then K learned queries
    cross-attend to it (Perceiver-style pooling) -> K tokens. A LayerNorm + a
    single learnable gate (small init) control the output scale so the new tokens
    enter the residual stream gently."""

    def __init__(self, out_dim: int, hidden: int = 256, n_layers: int = 2,
                 n_heads: int = 4, n_tokens: int = 8, t_scan: int = 32,
                 gate_init: float = 0.1):
        super().__init__()
        self.t_scan = t_scan
        self.n_tokens = n_tokens
        self.in_proj = nn.Linear(FEAT_DIM, hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, t_scan, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.randn(n_tokens, hidden) * 0.02)
        self.cross = nn.MultiheadAttention(hidden, n_heads, dropout=0.0, batch_first=True)
        self.out_proj = nn.Linear(hidden, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, traj: dict | None, device: torch.device) -> torch.Tensor:
        """Return (n_tokens, out_dim) gaze tokens (float32)."""
        feats = _traj_features(traj, self.t_scan, device)        # (T, FEAT_DIM)
        x = self.in_proj(feats).unsqueeze(0) + self.pos_emb       # (1, T, hidden)
        h = self.encoder(x)                                       # (1, T, hidden)
        q = self.query.unsqueeze(0)                               # (1, K, hidden)
        pooled, _ = self.cross(q, h, h)                           # (1, K, hidden)
        out = self.out_norm(self.out_proj(pooled.squeeze(0)))     # (K, out_dim)
        return self.gate * out
