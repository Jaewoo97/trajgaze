"""
Step 2a — Trajectory tokenizer.

Converts per-frame interaction features (from interaction.py) into
interaction token embeddings for all T frames. Handles missing detections
via learned MISSING_GAZE / MISSING_LEFT / MISSING_RIGHT embeddings.

Input per frame:
  d_left      (3,)  gaze→hand_left vector or zeros + mask bit
  d_right     (3,)  gaze→hand_right vector or zeros + mask bit
  v_rel_left  (2,)  relative velocity or zeros + mask bit
  v_rel_right (2,)  relative velocity or zeros + mask bit
  convergence (1,)  dD/dt scalar
  lead_lag    (1,)  ±1 / 0 scalar

Output: (T, d_model=128) interaction token embeddings.

Each frame has 4 raw tokens (gaze, hand_left, hand_right, interaction)
that are then fed to the two-level trajectory encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Raw feature dimensions ─────────────────────────────────────────────────────
#
# Gaze token input:   gaze position (2) + gaze speed (1) = 3
# Hand token input:   position (2) + speed (2) + present bit (1) = 5
# Interaction token:  d_left(3) + d_right(3) + v_rel_left(2) + v_rel_right(2)
#                   + convergence(1) + lead_lag(1) = 12

DIM_GAZE_RAW    = 3    # [x, y, speed]
DIM_HAND_RAW    = 5    # [x, y, vx, vy, present]
DIM_INTERACT_RAW = 12  # full interaction vector

D_MODEL = 128   # token embedding dimension


class TrajectoryTokenizer(nn.Module):
    """
    Projects raw trajectory signals to D_MODEL embeddings.

    For each frame, produces 4 tokens:
      tok_gaze        (D_MODEL,)
      tok_hand_left   (D_MODEL,)
      tok_hand_right  (D_MODEL,)
      tok_interaction (D_MODEL,)

    Missing signals are replaced by their learned MISSING embeddings.
    The interaction token degrades gracefully under partial missingness.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model

        # Learned MISSING embeddings (one per signal type)
        self.missing_gaze  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.missing_left  = nn.Parameter(torch.randn(d_model) * 0.02)
        self.missing_right = nn.Parameter(torch.randn(d_model) * 0.02)

        # Linear projections for each token type
        self.proj_gaze     = nn.Linear(DIM_GAZE_RAW,    d_model)
        self.proj_left     = nn.Linear(DIM_HAND_RAW,    d_model)
        self.proj_right    = nn.Linear(DIM_HAND_RAW,    d_model)
        self.proj_interact = nn.Linear(DIM_INTERACT_RAW, d_model)

        # Layer norms for stability
        self.norm_gaze     = nn.LayerNorm(d_model)
        self.norm_left     = nn.LayerNorm(d_model)
        self.norm_right    = nn.LayerNorm(d_model)
        self.norm_interact = nn.LayerNorm(d_model)

    def forward(
        self,
        gaze_pos:     torch.Tensor,   # (B, T, 2) float32 — 0.0 if missing
        gaze_speed:   torch.Tensor,   # (B, T, 1) float32 — 0.0 if missing
        gaze_mask:    torch.Tensor,   # (B, T)    bool    — True = present
        left_pos:     torch.Tensor,   # (B, T, 2) float32
        left_vel:     torch.Tensor,   # (B, T, 2) float32
        left_mask:    torch.Tensor,   # (B, T)    bool
        right_pos:    torch.Tensor,   # (B, T, 2) float32
        right_vel:    torch.Tensor,   # (B, T, 2) float32
        right_mask:   torch.Tensor,   # (B, T)    bool
        d_left:       torch.Tensor,   # (B, T, 3) float32 — 0.0 if missing
        d_right:      torch.Tensor,   # (B, T, 3) float32
        v_rel_left:   torch.Tensor,   # (B, T, 2) float32
        v_rel_right:  torch.Tensor,   # (B, T, 2) float32
        convergence:  torch.Tensor,   # (B, T, 1) float32
        lead_lag:     torch.Tensor,   # (B, T, 1) float32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tok_gaze:      (B, T, d_model)
            tok_left:      (B, T, d_model)
            tok_right:     (B, T, d_model)
            tok_interact:  (B, T, d_model)
        """
        B, T = gaze_mask.shape

        # ── Gaze token ────────────────────────────────────────────────────────
        gaze_feat = torch.cat([gaze_pos, gaze_speed], dim=-1)  # (B, T, 3)
        tok_gaze  = self.norm_gaze(self.proj_gaze(gaze_feat))  # (B, T, d_model)
        # Replace missing frames with MISSING embedding
        missing_g = self.missing_gaze.view(1, 1, -1).expand(B, T, -1)
        mask_g    = gaze_mask.unsqueeze(-1).float()
        tok_gaze  = mask_g * tok_gaze + (1 - mask_g) * missing_g

        # ── Hand left token ───────────────────────────────────────────────────
        left_feat  = torch.cat(
            [left_pos, left_vel, left_mask.unsqueeze(-1).float()], dim=-1
        )  # (B, T, 5)
        tok_left   = self.norm_left(self.proj_left(left_feat))
        missing_l  = self.missing_left.view(1, 1, -1).expand(B, T, -1)
        mask_l     = left_mask.unsqueeze(-1).float()
        tok_left   = mask_l * tok_left + (1 - mask_l) * missing_l

        # ── Hand right token ──────────────────────────────────────────────────
        right_feat = torch.cat(
            [right_pos, right_vel, right_mask.unsqueeze(-1).float()], dim=-1
        )  # (B, T, 5)
        tok_right  = self.norm_right(self.proj_right(right_feat))
        missing_r  = self.missing_right.view(1, 1, -1).expand(B, T, -1)
        mask_r     = right_mask.unsqueeze(-1).float()
        tok_right  = mask_r * tok_right + (1 - mask_r) * missing_r

        # ── Interaction token ─────────────────────────────────────────────────
        # Degrades gracefully: missing hand → d_left/v_rel_left are zeros
        interact_feat = torch.cat(
            [d_left, d_right, v_rel_left, v_rel_right, convergence, lead_lag],
            dim=-1,
        )  # (B, T, 12)
        tok_interact = self.norm_interact(self.proj_interact(interact_feat))

        return tok_gaze, tok_left, tok_right, tok_interact


class BatchFromNpz:
    """
    Converts a loaded .npz interaction file + adapted JSON into
    the input tensors expected by TrajectoryTokenizer.

    Handles coordinate normalization to [−1, 1] and velocity
    clamping to avoid outliers.

    Usage:
        npz    = np.load(npz_path)
        adapted = load_adapted(adapted_path)
        frame_names = sorted(adapted['frames'].keys())
        frames = get_frame_sequence(adapted, frame_names)
        batch  = BatchFromNpz.make(frames, npz)
        toks   = tokenizer(**batch)
    """

    COORD_MAX = 224.0   # normalize position to [0, 1]
    VEL_CLAMP = 30.0    # clamp velocity magnitude

    @staticmethod
    def make(frames: list[dict], npz) -> dict[str, torch.Tensor]:
        """
        Args:
            frames: list of dicts with gaze/hand_left/hand_right (from adapter)
            npz: loaded np.load object with keys from interaction.compute_and_save

        Returns: dict of float32 tensors, each with a leading batch dimension of 1.
        """
        import numpy as np

        T = len(frames)
        CM = BatchFromNpz.COORD_MAX
        VC = BatchFromNpz.VEL_CLAMP

        # Position arrays
        gaze_pos   = np.zeros((T, 2), dtype=np.float32)
        gaze_speed = np.zeros((T, 1), dtype=np.float32)
        gaze_mask  = np.zeros(T,       dtype=bool)

        left_pos   = np.zeros((T, 2), dtype=np.float32)
        left_vel   = np.zeros((T, 2), dtype=np.float32)
        left_mask  = np.zeros(T,       dtype=bool)

        right_pos  = np.zeros((T, 2), dtype=np.float32)
        right_vel  = np.zeros((T, 2), dtype=np.float32)
        right_mask = np.zeros(T,       dtype=bool)

        for t, f in enumerate(frames):
            if f["gaze"] is not None:
                gaze_pos[t]  = np.array(f["gaze"], dtype=np.float32) / CM
                gaze_mask[t] = True
            if f["hand_left"] is not None:
                left_pos[t]  = np.array(f["hand_left"],  dtype=np.float32) / CM
                left_mask[t] = True
            if f["hand_right"] is not None:
                right_pos[t]  = np.array(f["hand_right"], dtype=np.float32) / CM
                right_mask[t] = True

        # Velocities from finite diff on positions
        def fdiff(pos, mask):
            vel = np.zeros_like(pos)
            for t in range(T):
                if not mask[t]:
                    continue
                p_prev = pos[t - 1] if (t > 0 and mask[t - 1]) else None
                p_next = pos[t + 1] if (t < T - 1 and mask[t + 1]) else None
                if p_prev is not None and p_next is not None:
                    vel[t] = (p_next - p_prev) / 2.0
                elif p_next is not None:
                    vel[t] = p_next - pos[t]
                elif p_prev is not None:
                    vel[t] = pos[t] - p_prev
            return np.clip(vel, -VC / CM, VC / CM)

        left_vel  = fdiff(left_pos,  left_mask)
        right_vel = fdiff(right_pos, right_mask)

        # Gaze speed
        gaze_vel  = fdiff(gaze_pos, gaze_mask)
        gaze_speed = np.linalg.norm(gaze_vel, axis=1, keepdims=True)

        # Interaction features from npz (already computed)
        d_left      = npz["d_left"].astype(np.float32)      / CM   # (T, 3)
        d_right     = npz["d_right"].astype(np.float32)     / CM
        v_rel_left  = np.clip(npz["v_rel_left"].astype(np.float32)  / CM, -VC/CM, VC/CM)
        v_rel_right = np.clip(npz["v_rel_right"].astype(np.float32) / CM, -VC/CM, VC/CM)
        convergence = npz["convergence"].reshape(-1, 1).astype(np.float32) / CM
        lead_lag    = npz["lead_lag"].reshape(-1, 1).astype(np.float32)

        def t(x):
            return torch.from_numpy(x).unsqueeze(0)  # (1, ...)

        return {
            "gaze_pos":    t(gaze_pos),
            "gaze_speed":  t(gaze_speed),
            "gaze_mask":   torch.from_numpy(gaze_mask).unsqueeze(0),
            "left_pos":    t(left_pos),
            "left_vel":    t(left_vel),
            "left_mask":   torch.from_numpy(left_mask).unsqueeze(0),
            "right_pos":   t(right_pos),
            "right_vel":   t(right_vel),
            "right_mask":  torch.from_numpy(right_mask).unsqueeze(0),
            "d_left":      t(d_left),
            "d_right":     t(d_right),
            "v_rel_left":  t(v_rel_left),
            "v_rel_right": t(v_rel_right),
            "convergence": t(convergence),
            "lead_lag":    t(lead_lag),
        }
