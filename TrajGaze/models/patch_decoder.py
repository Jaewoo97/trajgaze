"""
Step 4 — AR Patch Decoder.

Selects patches autoregressively for each attended frame using:
  - Causal self-attention across patch selection history within the frame
  - Primary cross-attention to visual memory (B, T_att·N_patches, 192)
  - Interaction-gated secondary cross-attention to trajectory context (B, 4·T_window, 256→192)

Architecture:
  - 4-layer decoder, d_model=192
  - Vocabulary: 197 tokens (196 single-scale 14×14 patches + padding token 196)
  - Single shared lm_head (position-aware via causal self-attention + patch pos embed)
  - Selects top-20% = 40 patches per attended frame
  - Loss prediction head: scalar estimate of remaining importance (auto-stopping)

Token budget design:
  - Frame selector attends ~50% of frames (temporal selection)
  - Patch decoder selects 20% of 196 patches = 40 patches per attended frame
  - Total tokens = 50% × 20% = 10% of all T×196 visual tokens

Frame positional embedding added to distinguish which attended frame is being
processed so the decoder knows the temporal context.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ──────────────────────────────────────────────────────────────────

N_PATCHES       = 196   # single-scale 14×14 patch grid (224px / 16px stride)
PAD_TOKEN       = 196   # padding index, never predicted
VOCAB_SIZE      = 197   # N_PATCHES + 1 (pad)

D_MODEL = 192
D_TRAJ  = 256           # trajectory encoder output dim
N_HEADS = 6
N_LAYERS = 4
D_FF    = 384

# 20% of 196 patches = 39.2 → 40 patches per attended frame
# Combined with 50% frame selection: 50% × 20% = 10% total visual tokens
MAX_PATCHES_PER_FRAME = 40


# ── Interaction gate ───────────────────────────────────────────────────────────

class InteractionGate(nn.Module):
    """
    Per-timestep gate ∈ (0, 1) applied to trajectory cross-attention.

    gate(t) = sigmoid(MLP(interaction_token(t))) where high convergence
    and gaze-leading → gate → 1.

    The gate is computed from the raw interaction token (d_model_l1=128),
    not the trajectory encoder output.
    """

    def __init__(self, d_interact: int = 128, d_traj: int = D_TRAJ):
        super().__init__()
        # Project trajectory context to d_model for cross-attention
        self.proj_traj = nn.Linear(d_traj, D_MODEL)
        # Gate MLP: per-token gate from mean of trajectory context
        self.gate_mlp  = nn.Sequential(
            nn.Linear(d_interact, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        traj_context:  torch.Tensor,   # (B_att, 4·T_window, D_TRAJ)
        interact_mean: torch.Tensor,   # (B_att, d_interact) — mean of interact tokens for frame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          traj_proj:  (B_att, 4·T_window, D_MODEL) — projected trajectory context
          gate:       (B_att, 1)                    — scalar gate per attended frame
        """
        traj_proj = self.proj_traj(traj_context)        # (B_att, 4·Tw, D_MODEL)
        gate      = torch.sigmoid(self.gate_mlp(interact_mean))  # (B_att, 1)
        return traj_proj, gate


# ── Decoder layer ──────────────────────────────────────────────────────────────

class PatchDecoderLayer(nn.Module):
    """
    Single decoder layer with:
      1. Causal self-attention (across patch selection history)
      2. Primary cross-attention to visual memory
      3. Interaction-gated secondary cross-attention to trajectory context
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff:    int = D_FF,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention to visual memory
        self.cross_visual = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention to trajectory context
        self.cross_traj = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x:           torch.Tensor,   # (B_att, S, D_MODEL) — patch query sequence
        visual_mem:  torch.Tensor,   # (B_att, T_att·N, D_MODEL)
        traj_proj:   torch.Tensor,   # (B_att, 4·Tw, D_MODEL)
        gate:        torch.Tensor,   # (B_att, 1)
        causal_mask: torch.Tensor,   # (S, S) bool
    ) -> torch.Tensor:
        S = x.shape[1]
        # 1. Causal self-attention
        r = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.drop(x) + r

        # 2. Cross-attention to visual memory
        r = x
        x = self.norm2(x)
        x, _ = self.cross_visual(x, visual_mem, visual_mem, need_weights=False)
        x = self.drop(x) + r

        # 3. Interaction-gated cross-attention to trajectory context
        r = x
        x = self.norm3(x)
        traj_out, _ = self.cross_traj(x, traj_proj, traj_proj, need_weights=False)
        # gate: (B_att, 1) → (B_att, 1, 1) for broadcasting over S
        x = self.drop(traj_out) * gate.unsqueeze(1) + r

        # 4. FFN
        r = x
        x = self.norm4(x)
        x = self.drop(self.ff(x)) + r
        return x


# ── AR Patch Decoder ───────────────────────────────────────────────────────────

class ARPatchDecoder(nn.Module):
    """
    Autoregressive patch decoder.

    For each attended frame, selects up to MAX_PATCHES_PER_FRAME patches
    autoregressively from the multi-scale vocabulary.

    Multi-token prediction: 10 heads, each predicting the next patch token
    from a different "head position" in the sequence, all decoded in parallel
    (teacher-forcing during training).

    Auto-stopping: a loss-prediction head estimates remaining importance;
    the decoder stops when remaining_importance < threshold.
    """

    def __init__(
        self,
        d_model:          int = D_MODEL,
        d_traj:           int = D_TRAJ,
        d_interact:       int = 128,
        n_heads:          int = N_HEADS,
        n_layers:         int = N_LAYERS,
        d_ff:             int = D_FF,
        dropout:          float = 0.1,
        max_patches:      int = MAX_PATCHES_PER_FRAME,
        vocab_size:       int = VOCAB_SIZE,
        max_attended_frames: int = 2048,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_patches = max_patches

        # Token embedding for patch vocabulary
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)

        # Frame positional embedding (distinguishes which attended frame)
        self.frame_pos_embed = nn.Embedding(max_attended_frames, d_model)

        # Patch positional embedding (within the selection sequence)
        self.patch_pos_embed = nn.Embedding(max_patches + 2, d_model)

        # Interaction gate
        self.gate = InteractionGate(d_interact=d_interact, d_traj=d_traj)

        # Decoder layers
        self.layers = nn.ModuleList([
            PatchDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Single shared lm_head: predicts patch index at each autoregressive step.
        # Position-specificity comes from causal self-attention + patch positional
        # embedding — no need for separate per-position heads.
        # Output size = N_PATCHES (exclude pad from predictions).
        self.lm_head = nn.Linear(d_model, N_PATCHES, bias=False)

        # Loss prediction head: estimate remaining importance (auto-stopping)
        self.remaining_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        target_patches:  torch.Tensor,   # (B_att, max_patches) int64 — teacher-forced targets
        frame_idx:       torch.Tensor,   # (B_att,) int64 — which attended frame (ordinal)
        visual_mem:      torch.Tensor,   # (B_att, T_att·N, D_MODEL) — from VideoEncoder
        traj_context:    torch.Tensor,   # (B_att, 4·Tw, D_TRAJ) — from TrajEncoder
        interact_mean:   torch.Tensor,   # (B_att, d_interact) — mean interact token for frame
    ) -> dict[str, torch.Tensor]:
        """
        Teacher-forced forward pass for Stage 1 training.

        Returns:
          logits:      (B_att, max_patches, N_PATCHES_TOTAL) — per-position logits
          remaining:   (B_att, max_patches) — remaining importance estimates
          gate_values: (B_att,) — gate scalars for this frame
        """
        B_att, S = target_patches.shape
        device = target_patches.device

        # Interaction gate (project traj + compute gate scalar)
        traj_proj, gate_vals = self.gate(traj_context, interact_mean)  # (B_att, 4·Tw, D_MODEL), (B_att, 1)

        # Build input sequence: embed previous tokens (shifted right by 1)
        # Input at position k = token k-1; position 0 = frame embedding (BOS)
        # Prepend a BOS-like frame embedding, then shift target patches left
        bos = self.frame_pos_embed(frame_idx).unsqueeze(1)  # (B_att, 1, D_MODEL)
        tok_emb = self.tok_embed(target_patches)              # (B_att, S, D_MODEL)

        # Input: [frame_bos, tok_0, tok_1, ..., tok_{S-1}] (length S+1)
        # but we only need S outputs, so input is [frame_bos, tok_0..tok_{S-2}]
        inp = torch.cat([bos, tok_emb[:, :-1, :]], dim=1)   # (B_att, S, D_MODEL)

        # Add patch position embeddings
        patch_pos = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
        inp = inp + self.patch_pos_embed(patch_pos)               # (B_att, S, D_MODEL)

        # Causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
        )

        # Run decoder layers
        x = inp
        for layer in self.layers:
            x = layer(x, visual_mem, traj_proj, gate_vals, causal_mask)
        x = self.norm(x)   # (B_att, S, D_MODEL)

        # Shared lm_head applied across all S positions
        logits = self.lm_head(x)   # (B_att, S, N_PATCHES)

        # Remaining importance
        remaining = self.remaining_head(x).squeeze(-1)  # (B_att, S)

        return {
            "logits":      logits,
            "remaining":   remaining,
            "gate_values": gate_vals.squeeze(-1),  # (B_att,)
        }

    @torch.no_grad()
    def decode_greedy(
        self,
        frame_idx:       torch.Tensor,
        visual_mem:      torch.Tensor,
        traj_context:    torch.Tensor,
        interact_mean:   torch.Tensor,
        n_patches:       int = MAX_PATCHES_PER_FRAME,
        stop_threshold:  float = 0.1,
    ) -> list[int]:
        """
        Greedy autoregressive decoding at inference.

        Returns list of selected patch indices (length ≤ n_patches).
        Stops early when remaining_importance < stop_threshold.
        """
        device = frame_idx.device
        assert frame_idx.shape[0] == 1, "decode_greedy processes one frame at a time"

        traj_proj, gate_vals = self.gate(traj_context, interact_mean)

        bos = self.frame_pos_embed(frame_idx).unsqueeze(1)  # (1, 1, D_MODEL)
        selected = []
        inp = bos

        for step in range(n_patches):
            S = inp.shape[1]
            causal_mask = torch.triu(
                torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1
            )
            patch_pos = torch.arange(S, device=device).unsqueeze(0)
            x = inp + self.patch_pos_embed(patch_pos)

            for layer in self.layers:
                x = layer(x, visual_mem, traj_proj, gate_vals, causal_mask)
            x = self.norm(x)

            # Predict next patch from last position; mask already-selected patches
            logit = self.lm_head(x[:, -1, :]).clone()  # (1, N_PATCHES)
            for already in selected:
                logit[0, already] = -float("inf")
            pred  = logit.argmax(dim=-1).item()         # scalar
            selected.append(pred)

            # Check remaining importance
            rem = self.remaining_head(x[:, -1:, :]).item()
            if rem < stop_threshold and step >= 1:
                break

            # Append new token to input
            next_tok = torch.tensor([[pred]], device=device, dtype=torch.long)
            next_emb = self.tok_embed(next_tok)
            inp = torch.cat([inp, next_emb], dim=1)

        return selected


# ── Trajectory prediction head (auxiliary for Stage 1 L_traj) ─────────────────

class TrajectoryPredictor(nn.Module):
    """
    Auxiliary head for L_traj. Discarded after Stage 1.

    Predicts future gaze and hand positions from mean-pooled selected
    patch embeddings of an attended frame.

    Input:  mean-pooled selected patch embeddings (B_att, D_MODEL=192)
    Output: predicted {gaze(t+1:Δ), hand_left(t+1:Δ), hand_right(t+1:Δ)}
            as (B_att, Δ, 6) — [gaze_x, gaze_y, lh_x, lh_y, rh_x, rh_y]
    """

    def __init__(self, d_model: int = D_MODEL, horizon: int = 8):
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon * 6),
        )

    def forward(self, patch_emb_mean: torch.Tensor) -> torch.Tensor:
        """
        patch_emb_mean: (B_att, D_MODEL)
        Returns: (B_att, horizon, 6)
        """
        B = patch_emb_mean.shape[0]
        out = self.mlp(patch_emb_mean)          # (B_att, horizon*6)
        return out.view(B, self.horizon, 6)


def patch_ntp_loss(
    logits:   torch.Tensor,   # (B_att, max_patches, N_PATCHES)
    targets:  torch.Tensor,   # (B_att, max_patches) int64 — ranked patch indices [0,195]
    pad_mask: torch.Tensor,   # (B_att, max_patches) bool — True = valid
) -> torch.Tensor:
    """
    Next-token prediction loss on patch indices.
    Uses cross-entropy, masked by pad_mask.
    """
    B, S, V = logits.shape
    logits_flat  = logits.reshape(B * S, V)
    targets_flat = targets.reshape(B * S)
    valid_flat   = pad_mask.reshape(B * S)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # (B·S,)
    loss = (loss * valid_flat.float()).sum() / (valid_flat.float().sum() + 1e-8)
    return loss


def traj_prediction_loss(
    pred:      torch.Tensor,   # (B_att, Δ, 6) predicted future positions
    target:    torch.Tensor,   # (B_att, Δ, 6) ground truth
    step_mask: torch.Tensor,   # (B_att, Δ) bool — True = step has valid ground truth
) -> torch.Tensor:
    """
    L2 trajectory prediction loss, masked for null future steps.
    Applied only on attended frames (B_att already filtered).
    """
    loss = F.mse_loss(pred, target, reduction="none")   # (B_att, Δ, 6)
    loss = loss.mean(dim=-1)                             # (B_att, Δ)
    loss = (loss * step_mask.float()).sum() / (step_mask.float().sum() + 1e-8)
    return loss
