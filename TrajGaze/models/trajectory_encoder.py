"""
Step 3a — Two-level trajectory encoder.

Level 1 — Intra-frame attention (per timestep, 2-layer block):
  The 4 tokens {gaze, hand_left, hand_right, interaction} at each timestep t
  attend fully to each other, resolving "at this moment, how does gaze relate
  to each hand?" before any temporal modeling.
  Output: 4 interaction-enriched tokens per frame.

Level 2 — Inter-frame causal transformer (6 layers, 8 heads, d_model=256):
  Operates across T_window=16 timesteps centered on each attended frame.
  Temporal dynamics modeled on top of already-interaction-aware representations.
  Asymmetric key-padding mask: missing tokens attend to present ones but
  are excluded as keys.
  Output: (B, 4·T_window, d_model=256) trajectory context per attended frame.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from TrajGaze.models.frame_selector import SinusoidalPE, _causal_mask

D_MODEL_L1 = 128   # matches tokenizer output
D_MODEL_L2 = 256   # inter-frame transformer
T_WINDOW   = 16    # temporal context window around each attended frame
N_TOKENS   = 4     # tokens per frame: gaze, left, right, interaction


# ── Level 1: Intra-frame full attention block ──────────────────────────────────

class IntraFrameBlock(nn.Module):
    """
    2-layer full self-attention across the 4 tokens of a single timestep.

    Input:  (B, T, 4, d_model=128)
    Output: (B, T, 4, d_model=128)

    Processes all T frames in parallel — the 4 tokens at each position
    attend only to each other (not across timesteps).
    """

    def __init__(
        self,
        d_model: int = D_MODEL_L1,
        n_heads: int = 4,
        d_ff:    int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers  = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model     = d_model,
                nhead       = n_heads,
                dim_feedforward = d_ff,
                dropout     = dropout,
                activation  = "gelu",
                batch_first = True,   # (B·T, 4, d_model) when reshaped
                norm_first  = True,   # pre-norm
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        tok_gaze:     torch.Tensor,   # (B, T, d_model)
        tok_left:     torch.Tensor,   # (B, T, d_model)
        tok_right:    torch.Tensor,   # (B, T, d_model)
        tok_interact: torch.Tensor,   # (B, T, d_model)
    ) -> torch.Tensor:
        """
        Returns enriched tokens concatenated as (B, T, 4, d_model).
        """
        B, T, D = tok_gaze.shape

        # Stack 4 tokens per frame: (B, T, 4, D) → (B·T, 4, D)
        x = torch.stack([tok_gaze, tok_left, tok_right, tok_interact], dim=2)  # (B, T, 4, D)
        x = x.view(B * T, N_TOKENS, D)

        for layer in self.layers:
            x = layer(x)

        x = x.view(B, T, N_TOKENS, D)   # (B, T, 4, D)
        return x


# ── Level 2: Inter-frame causal transformer ────────────────────────────────────

class InterFrameTransformer(nn.Module):
    """
    6-layer causal transformer over T_window frames × 4 tokens each.

    Input:  (B, T_window, 4, d_model_in=128) — from IntraFrameBlock
    Output: (B, 4·T_window, d_model_out=256) — trajectory context

    Missing-token handling:
      A key-padding mask excludes missing tokens as keys (preventing them
      from providing context) while still allowing them to attend to
      present tokens (receiving context).
    """

    def __init__(
        self,
        d_in:     int = D_MODEL_L1,
        d_model:  int = D_MODEL_L2,
        n_heads:  int = 8,
        d_ff:     int = 512,
        n_layers: int = 6,
        dropout:  float = 0.1,
        t_window: int = T_WINDOW,
    ):
        super().__init__()
        self.d_model  = d_model
        self.t_window = t_window
        seq_len       = t_window * N_TOKENS

        # Project from d_in to d_model
        self.proj  = nn.Linear(d_in, d_model)
        self.pos   = SinusoidalPE(d_model, max_len=seq_len + 16)
        self.norm_in = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = n_heads,
                dim_feedforward = d_ff,
                dropout         = dropout,
                activation      = "gelu",
                batch_first     = True,
                norm_first      = True,
            )
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x:       torch.Tensor,         # (B, T_window, 4, d_in) from IntraFrameBlock
        present: torch.Tensor,         # (B, T_window) bool — True if frame data present
    ) -> torch.Tensor:
        """
        Returns: (B, 4·T_window, d_model)
        """
        B, Tw, _, D_in = x.shape
        seq_len = Tw * N_TOKENS

        # Flatten to (B, seq_len, d_in) then project
        x = x.reshape(B, seq_len, D_in)
        x = self.proj(x)                      # (B, seq_len, d_model)
        x = self.pos(x)
        x = self.norm_in(x)

        # Build causal mask: (seq_len, seq_len)
        causal = _causal_mask(seq_len, x.device)

        # Use only the causal mask. Missing-frame tokens already carry
        # learned MISSING embeddings from L1 so the model can attend to
        # them safely. Applying key_padding_mask would make all keys masked
        # for padded boundary positions (causal + missing → all -inf →
        # softmax NaN that propagates through the whole clip).
        for layer in self.layers:
            x = layer(x, src_mask=causal)

        x = self.norm_out(x)   # (B, 4·T_window, d_model)
        return x


# ── Unified two-level encoder ──────────────────────────────────────────────────

class TwoLevelTrajectoryEncoder(nn.Module):
    """
    Full two-level trajectory encoder.

    For each attended frame, extracts a T_window context window of interaction
    tokens and produces trajectory context (B, 4·T_window, d_model_out=256).

    At batch inference, multiple attended frames can be processed in parallel
    by stacking their context windows as independent batch items.
    """

    def __init__(
        self,
        d_model_l1: int = D_MODEL_L1,
        d_model_l2: int = D_MODEL_L2,
        t_window:   int = T_WINDOW,
        intra_heads:    int = 4,
        intra_layers:   int = 2,
        inter_heads:    int = 8,
        inter_layers:   int = 6,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.t_window = t_window
        self.intra    = IntraFrameBlock(
            d_model  = d_model_l1,
            n_heads  = intra_heads,
            n_layers = intra_layers,
            dropout  = dropout,
        )
        self.inter    = InterFrameTransformer(
            d_in    = d_model_l1,
            d_model = d_model_l2,
            n_heads = inter_heads,
            n_layers = inter_layers,
            dropout  = dropout,
            t_window = t_window,
        )

    def forward(
        self,
        tok_gaze:     torch.Tensor,   # (B, T, d_model_l1)
        tok_left:     torch.Tensor,
        tok_right:    torch.Tensor,
        tok_interact: torch.Tensor,
        present:      torch.Tensor,   # (B, T) bool
        attended_idx: list[int],      # which frames to process (within [0, T))
    ) -> torch.Tensor:
        """
        Process attended frames only. Extracts a T_window context window
        around each attended frame, runs both encoder levels.

        Returns: (len(attended_idx), 4·T_window, d_model_l2)
                 One context vector per attended frame.

        If attended_idx is empty, returns an empty tensor.
        """
        if not attended_idx:
            dev = tok_interact.device
            return torch.zeros(0, self.t_window * N_TOKENS, self.inter.d_model, device=dev)

        B = tok_gaze.shape[0]
        assert B == 1, "TwoLevelTrajectoryEncoder processes one clip at a time (B=1)"

        T   = tok_gaze.shape[1]
        Tw  = self.t_window
        half = Tw // 2

        # Run Level 1 on the full sequence (all T frames in parallel)
        enriched = self.intra(tok_gaze, tok_left, tok_right, tok_interact)  # (1, T, 4, D)

        # Extract T_window windows for each attended frame
        windows      = []
        windows_pres = []
        for t_att in attended_idx:
            t_lo = max(0, t_att - half)
            t_hi = t_lo + Tw
            if t_hi > T:
                t_hi = T
                t_lo = max(0, t_hi - Tw)

            win = enriched[0, t_lo:t_hi]    # (actual_w, 4, D)
            pres = present[0, t_lo:t_hi]    # (actual_w,)
            actual_w = win.shape[0]

            # Pad to Tw if at boundary
            if actual_w < Tw:
                pad = torch.zeros(Tw - actual_w, N_TOKENS, enriched.shape[-1],
                                  device=tok_gaze.device)
                win  = torch.cat([pad, win], dim=0)   # left-pad
                pres_pad = torch.zeros(Tw - actual_w, dtype=torch.bool, device=tok_gaze.device)
                pres = torch.cat([pres_pad, pres], dim=0)

            windows.append(win)              # (Tw, 4, D)
            windows_pres.append(pres)        # (Tw,)

        # Stack into batch: (n_att, Tw, 4, D)
        win_batch  = torch.stack(windows,      dim=0)   # (n_att, Tw, 4, D)
        pres_batch = torch.stack(windows_pres, dim=0)   # (n_att, Tw)

        # Level 2: inter-frame causal transformer
        context = self.inter(win_batch, pres_batch)  # (n_att, 4·Tw, d_model_l2)
        return context
