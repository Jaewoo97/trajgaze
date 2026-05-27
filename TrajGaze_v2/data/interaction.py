"""
Patch-level gaze-hand interaction score computation.

Adapted from TrajGaze/data/interaction.py for TrajGaze_v2.

I(p, t) = G(p, t) · H(p, t) · φ(τ*) · ψ(dD/dt)

Grid: 14×14 = 196 patches (224px / 16px stride).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

COORD_SIZE = 224
PATCH_GRID = 14
N_PATCHES  = PATCH_GRID * PATCH_GRID   # 196
SIGMA_GAZE = 6.0
SIGMA_HAND = 8.0
WINDOW_W   = 8
EPSILON    = 0.05
LAG_MAX    = 4


def _patch_centers() -> np.ndarray:
    stride = COORD_SIZE / PATCH_GRID  # 16.0
    idx = np.arange(PATCH_GRID)
    cx = (idx + 0.5) * stride
    cy = (idx + 0.5) * stride
    xx, yy = np.meshgrid(cx, cy)
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # (196, 2)


_PATCH_CENTERS = _patch_centers()


def _gaussian(centers: np.ndarray, pos: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum((centers - pos) ** 2, axis=1)
    return np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)


def _finite_diff(seq: list, t: int) -> Optional[np.ndarray]:
    prev = seq[t - 1] if t > 0 else None
    curr = seq[t]
    nxt  = seq[t + 1] if t < len(seq) - 1 else None
    if curr is None:
        return None
    if nxt is not None and prev is not None:
        return (np.array(nxt) - np.array(prev)) / 2.0
    if nxt is not None:
        return np.array(nxt) - np.array(curr)
    if prev is not None:
        return np.array(curr) - np.array(prev)
    return np.zeros(2, dtype=np.float32)


def _speed(v: Optional[np.ndarray]) -> Optional[float]:
    return float(np.linalg.norm(v)) if v is not None else None


def _lead_lag(gaze_speeds: list, hand_speeds: list, t: int, lag_max: int = LAG_MAX) -> float:
    half = lag_max
    lo = max(0, t - half)
    hi = min(len(gaze_speeds) - 1, t + half)
    g = [v for v in gaze_speeds[lo:hi+1] if v is not None]
    h = [v for v in hand_speeds[lo:hi+1]  if v is not None]
    n = min(len(g), len(h))
    if n < 3:
        return 0.0
    g_arr = np.array(g[:n])
    h_arr = np.array(h[:n])
    if g_arr.std() < 1e-6 or h_arr.std() < 1e-6:
        return 0.0
    g_z = (g_arr - g_arr.mean()) / (g_arr.std() + 1e-8)
    h_z = (h_arr - h_arr.mean()) / (h_arr.std() + 1e-8)
    best_lag = 0
    best_cc  = 0.0
    for lag in range(-lag_max, lag_max + 1):
        if lag >= 0:
            a, b = g_z[:n - lag], h_z[lag:]
        else:
            a, b = g_z[-lag:], h_z[:n + lag]
        m = min(len(a), len(b))
        if m < 2:
            continue
        cc = float(np.corrcoef(a[:m], b[:m])[0, 1])
        if abs(cc) > abs(best_cc):
            best_cc  = cc
            best_lag = lag
    return float(np.sign(best_lag)) if abs(best_cc) > 0.2 else 0.0


def compute_importance_scores(
    gaze_pos_list:  list[Optional[tuple[float, float]]],  # T elements, each (gx, gy) in [0,1] or None
    left_pos_list:  list[Optional[tuple[float, float]]],  # T elements, (lx, ly) in [0,1] or None
    right_pos_list: list[Optional[tuple[float, float]]],  # T elements, (rx, ry) in [0,1] or None
    window: int = WINDOW_W,
) -> np.ndarray:
    """
    Compute I(p, t) for T frames and 196 patches.

    Inputs are normalized [0,1] → convert to [0, 224) for Gaussian computation.

    Returns:
        I_scores: np.ndarray (T, 196) float32 in [0, 1]
    """
    T = len(gaze_pos_list)
    centers = _PATCH_CENTERS  # (196, 2) in [0, 224)

    def to_pixel(coord):
        if coord is None:
            return None
        return np.array([coord[0] * COORD_SIZE, coord[1] * COORD_SIZE], dtype=np.float32)

    gaze_px  = [to_pixel(p) for p in gaze_pos_list]
    left_px  = [to_pixel(p) for p in left_pos_list]
    right_px = [to_pixel(p) for p in right_pos_list]

    # Precompute Gaussian maps and hand velocity
    gaze_maps = np.zeros((T, N_PATCHES), dtype=np.float32)
    hand_maps = np.zeros((T, N_PATCHES), dtype=np.float32)
    hand_vel  = np.zeros(T, dtype=np.float32)

    for t in range(T):
        if gaze_px[t] is not None:
            gaze_maps[t] = _gaussian(centers, gaze_px[t], SIGMA_GAZE)
        hp_list = [p for p in (left_px[t], right_px[t]) if p is not None]
        if hp_list:
            hm = np.zeros(N_PATCHES, dtype=np.float32)
            for hp in hp_list:
                hm = np.maximum(hm, _gaussian(centers, hp, SIGMA_HAND))
            hand_maps[t] = hm
        # Hand speed
        speeds = []
        for pos_seq in (left_px, right_px):
            p_curr = pos_seq[t]
            if p_curr is not None:
                p_nxt  = pos_seq[t + 1] if t < T - 1 else None
                p_prev = pos_seq[t - 1] if t > 0 else None
                if p_nxt is not None:
                    speeds.append(float(np.linalg.norm(p_nxt - p_curr)))
                elif p_prev is not None:
                    speeds.append(float(np.linalg.norm(p_curr - p_prev)))
        hand_vel[t] = max(speeds) if speeds else 0.0

    vmax = hand_vel.max()
    hand_vel_norm = hand_vel / vmax if vmax > 1e-6 else hand_vel

    # Dominant gaze-to-hand distance for convergence
    dist_seq: list[Optional[float]] = []
    for t in range(T):
        g = gaze_px[t]
        dists = []
        for hp in (left_px[t], right_px[t]):
            if g is not None and hp is not None:
                dists.append(float(np.linalg.norm(g - hp)))
        dist_seq.append(min(dists) if dists else None)

    # Lead-lag
    gaze_vel_ll  = [_finite_diff(gaze_px,  t) for t in range(T)]
    left_vel_ll  = [_finite_diff(left_px,  t) for t in range(T)]
    right_vel_ll = [_finite_diff(right_px, t) for t in range(T)]
    gaze_speeds  = [_speed(gaze_vel_ll[t]) for t in range(T)]
    hand_speeds  = [
        _speed(left_vel_ll[t]) if left_vel_ll[t] is not None
        else _speed(right_vel_ll[t])
        for t in range(T)
    ]

    phi = np.ones(T, dtype=np.float32)
    psi = np.ones(T, dtype=np.float32)
    for t in range(T):
        ll = _lead_lag(gaze_speeds, hand_speeds, t)
        phi[t] = 1.0 + 0.2 * max(0.0, ll)
        d_prev = dist_seq[t - 1] if t > 0 else None
        d_curr = dist_seq[t]
        d_next = dist_seq[t + 1] if t < T - 1 else None
        if d_curr is not None:
            if d_next is not None and d_prev is not None:
                dDdt = (d_next - d_prev) / 2.0
            elif d_next is not None:
                dDdt = d_next - d_curr
            elif d_prev is not None:
                dDdt = d_curr - d_prev
            else:
                dDdt = 0.0
            psi[t] = 1.0 + 0.3 * max(0.0, -dDdt / (COORD_SIZE + 1e-6))

    # Accumulate over temporal window
    I_scores = np.zeros((T, N_PATCHES), dtype=np.float32)
    for t in range(T):
        t_lo = max(0, t - window // 2)
        t_hi = min(T, t + window // 2 + 1)
        for s in range(t_lo, t_hi):
            G_s = gaze_maps[s]
            H_s = hand_maps[s] * (1.0 + hand_vel_norm[s])
            I_scores[t] += G_s * H_s * phi[t] * psi[t]
        I_scores[t] /= max(1, t_hi - t_lo)

    score_max = I_scores.max()
    if score_max > 1e-9:
        I_scores /= score_max

    return I_scores  # (T, 196)


def compute_traj_features(
    gaze_pos_list:  list[Optional[tuple[float, float]]],
    left_pos_list:  list[Optional[tuple[float, float]]],
    right_pos_list: list[Optional[tuple[float, float]]],
) -> dict:
    """
    Compute raw trajectory features for TrajectoryTokenizer.

    All positions assumed to be in [0,1].

    Returns dict with keys:
        gaze_pos, gaze_speed, gaze_mask,
        left_pos, left_vel, left_mask,
        right_pos, right_vel, right_mask,
        d_left, d_right, v_rel_left, v_rel_right, convergence, lead_lag
    All as float32 numpy arrays (T, ...).
    """
    import numpy as np

    T = len(gaze_pos_list)
    CM = 1.0   # already normalized [0,1]
    VC = 0.5   # clamp velocity at 0.5 (half-frame in one step)

    def to_arr(coord):
        return np.array(coord, dtype=np.float32) if coord is not None else None

    gaze_np  = [to_arr(p) for p in gaze_pos_list]
    left_np  = [to_arr(p) for p in left_pos_list]
    right_np = [to_arr(p) for p in right_pos_list]

    # Masks
    gaze_mask  = np.array([p is not None for p in gaze_np],  dtype=bool)
    left_mask  = np.array([p is not None for p in left_np],  dtype=bool)
    right_mask = np.array([p is not None for p in right_np], dtype=bool)

    # Position arrays (0 where missing)
    gaze_pos  = np.array([p if p is not None else np.zeros(2, np.float32) for p in gaze_np],  dtype=np.float32)
    left_pos  = np.array([p if p is not None else np.zeros(2, np.float32) for p in left_np],  dtype=np.float32)
    right_pos = np.array([p if p is not None else np.zeros(2, np.float32) for p in right_np], dtype=np.float32)

    # Velocities via finite diff
    def fdiff(positions, mask):
        vel = np.zeros_like(positions)
        for t in range(T):
            if not mask[t]:
                continue
            p_prev = positions[t - 1] if (t > 0 and mask[t - 1]) else None
            p_next = positions[t + 1] if (t < T - 1 and mask[t + 1]) else None
            if p_prev is not None and p_next is not None:
                vel[t] = (p_next - p_prev) / 2.0
            elif p_next is not None:
                vel[t] = p_next - positions[t]
            elif p_prev is not None:
                vel[t] = positions[t] - p_prev
        return np.clip(vel, -VC, VC)

    gaze_vel  = fdiff(gaze_pos,  gaze_mask)
    left_vel  = fdiff(left_pos,  left_mask)
    right_vel = fdiff(right_pos, right_mask)

    gaze_speed = np.linalg.norm(gaze_vel, axis=1, keepdims=True).astype(np.float32)  # (T, 1)

    # Interaction features
    d_left      = np.zeros((T, 3), dtype=np.float32)
    d_right     = np.zeros((T, 3), dtype=np.float32)
    v_rel_left  = np.zeros((T, 2), dtype=np.float32)
    v_rel_right = np.zeros((T, 2), dtype=np.float32)
    convergence = np.zeros(T, dtype=np.float32)
    lead_lag    = np.zeros(T, dtype=np.float32)

    # Dominant distance sequence for convergence
    dist_seq: list[Optional[float]] = []
    for t in range(T):
        g = gaze_np[t]
        dists = []
        for hp in (left_np[t], right_np[t]):
            if g is not None and hp is not None:
                dists.append(float(np.linalg.norm(np.array(g) - np.array(hp))))
        dist_seq.append(min(dists) if dists else None)

    gaze_speeds_ll = [float(np.linalg.norm(gaze_vel[t])) if gaze_mask[t] else None for t in range(T)]
    hand_speeds_ll = [
        float(np.linalg.norm(left_vel[t]))  if left_mask[t]  else
        float(np.linalg.norm(right_vel[t])) if right_mask[t] else
        None
        for t in range(T)
    ]

    for t in range(T):
        g = gaze_np[t]
        lh = left_np[t]
        rh = right_np[t]

        if g is not None and lh is not None:
            diff = np.array(lh) - np.array(g)
            d_left[t] = [diff[0], diff[1], float(np.linalg.norm(diff))]

        if g is not None and rh is not None:
            diff = np.array(rh) - np.array(g)
            d_right[t] = [diff[0], diff[1], float(np.linalg.norm(diff))]

        if left_mask[t] and gaze_mask[t]:
            v_rel_left[t] = np.clip(left_vel[t] - gaze_vel[t], -VC, VC)

        if right_mask[t] and gaze_mask[t]:
            v_rel_right[t] = np.clip(right_vel[t] - gaze_vel[t], -VC, VC)

        # Convergence
        d_prev = dist_seq[t - 1] if t > 0 else None
        d_curr = dist_seq[t]
        d_next = dist_seq[t + 1] if t < T - 1 else None
        if d_curr is not None:
            if d_next is not None and d_prev is not None:
                convergence[t] = float((d_next - d_prev) / 2.0)
            elif d_next is not None:
                convergence[t] = float(d_next - d_curr)
            elif d_prev is not None:
                convergence[t] = float(d_curr - d_prev)

        lead_lag[t] = _lead_lag(gaze_speeds_ll, hand_speeds_ll, t)

    return {
        "gaze_pos":    gaze_pos,       # (T, 2)
        "gaze_speed":  gaze_speed,     # (T, 1)
        "gaze_mask":   gaze_mask,      # (T,) bool
        "left_pos":    left_pos,       # (T, 2)
        "left_vel":    left_vel,       # (T, 2)
        "left_mask":   left_mask,      # (T,) bool
        "right_pos":   right_pos,      # (T, 2)
        "right_vel":   right_vel,      # (T, 2)
        "right_mask":  right_mask,     # (T,) bool
        "d_left":      d_left,         # (T, 3)
        "d_right":     d_right,        # (T, 3)
        "v_rel_left":  v_rel_left,     # (T, 2)
        "v_rel_right": v_rel_right,    # (T, 2)
        "convergence": convergence,    # (T,)
        "lead_lag":    lead_lag,       # (T,)
    }
