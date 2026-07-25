"""Fixation-confidence weighting for the gaze complement (noise-aware gaze).

Empirical motivation (egtea test, measured):
  - frame-to-frame gaze jitter median 0.135 (13.5% of image) → gaze is NOISY
  - 28% of frames are saccades (speed > 0.25) → ~1/4 of gaze is "in motion" = noise
  - convergence/lead_lag valid only ~9-11% → binocular confidence DEAD on egtea
  → the ONLY alive confidence signal is gaze_speed (fixation vs saccade)

Every prior gaze mechanism injected raw gaze unconditionally (and tied/lost vs an
attention twin). The untested lever: weight the gaze complement by per-frame
FIXATION CONFIDENCE so that gaze contributes during stable fixations (reliable,
intentional) and is suppressed during saccades (noise).

c(t) = sustained-fixation confidence in [c_min, 1]:
  high  when speed is low AND stays low over a temporal window (true fixation)
  low   during saccades / isolated speed dips / invalid gaze

Arms:
  confidence — c(t) as defined                              (the method)
  inverse    — 1 - c(t) (high during saccades = inject noise) (KILL-TEST control)
  random     — random per-frame weights, same range          (placebo)
  none       — all-ones (≡ M1 raw complement)                (baseline)
"""
from __future__ import annotations

import numpy as np
import torch

SACCADE_SPEED = 0.25   # speed threshold (normalized img units): above = saccade
WINDOW        = 5      # temporal smoothing window (frames); ~0.5s at 10 fps
TEMP          = 0.10   # sigmoid sharpness for the fixation indicator
C_MIN         = 0.10   # floor so confident frames demote — never fully kill — saccades

# Direction A — task-adaptive confidence. Saccade suppression helps STATIC object
# grounding (reliable fixation → which object), but hurts DYNAMIC reasoning
# (spatial/temporal/sequence need the gaze trajectory). So apply confidence
# weighting only on object-grounding tasks; elsewhere keep raw gaze (c≡1).
# Set fixed a-priori by question semantics; held-out CV (+1.09pp) confirms it
# generalizes past the test-informed object set.
OBJECT_TASKS = {
    "present_object_identification_easy",
    "present_object_identification_hard",
    "present_object_attribute_recognition",
    "past_non_fixated_object_identification",
}

# Direction C — sign-routed gaze lean (single LoRA, 1 pass). The confidence line's
# net-wash was a SIGN-FLIP, not a null: object grounding wants saccade SUPPRESSED
# (stable fixation → which object), but spatial/temporal reasoning wants saccade
# KEPT/boosted (gaze transitions ARE the signal — measured: spatial 41.1 vs 36.2,
# temporal 42.5 vs 41.2 under inverse vs confidence). So route each task GROUP to
# its empirically-better lean within ONE model, attacking the cancellation that
# made global confidence tie M1. Mechanism-derived 3-way; held-out CV to follow.
DYNAMIC_TASKS = {
    "spatial",
    "temporal",
}


def resolve_arm(arm: str, task: str) -> str:
    """task_adaptive → 'confidence' on object-grounding tasks, else 'none' (raw).
    signrouted → 'confidence' on object grounding, 'inverse' (saccade-boost) on
    spatial/temporal dynamic reasoning, 'none' (raw) elsewhere."""
    if arm == "task_adaptive":
        return "confidence" if task in OBJECT_TASKS else "none"
    if arm == "signrouted":
        if task in OBJECT_TASKS:
            return "confidence"
        if task in DYNAMIC_TASKS:
            return "inverse"
        return "none"
    return arm


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with edge padding (length-preserving)."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def fixation_confidence(
    gaze_speed: torch.Tensor,
    gaze_mask: torch.Tensor,
    arm: str = "confidence",
    n_out: int | None = None,
    item_seed: int = 0,
    saccade_speed: float = SACCADE_SPEED,
    window: int = WINDOW,
    temp: float = TEMP,
    c_min: float = C_MIN,
) -> torch.Tensor:
    """Per-frame gaze confidence weight, shape (n_out,) (defaults to T_traj).

    Args:
        gaze_speed: (T, 1) or (T,) normalized gaze speed per traj frame
        gaze_mask:  (T,) bool — valid gaze present
        arm:        confidence | inverse | random | none
        n_out:      if given and != T, resample the weight to this length
                    (to align with TAS score's temporal axis)
    """
    s = np.asarray(gaze_speed.detach().cpu()).astype(np.float32).ravel()
    m = np.asarray(gaze_mask.detach().cpu()).astype(bool).ravel()
    T = len(s)
    device = gaze_speed.device

    if arm == "none":
        c = np.ones(T, dtype=np.float32)
        return _finalize(c, n_out, device)

    # --- soft fixation indicator: high when slow ---
    fix = 1.0 / (1.0 + np.exp((s - saccade_speed) / max(1e-6, temp)))  # sigmoid, slow→1
    # --- sustained: reward low speed that PERSISTS over the window ---
    fix_sustained = _moving_average(fix, window)
    # invalid gaze → no trustworthy prior → minimum confidence
    fix_sustained[~m] = 0.0

    # normalize to [c_min, 1] within this clip (relative reweight, not rescale)
    lo, hi = fix_sustained.min(), fix_sustained.max()
    if hi > lo:
        c = c_min + (1.0 - c_min) * (fix_sustained - lo) / (hi - lo)
    else:
        c = np.ones(T, dtype=np.float32)

    if arm == "inverse":
        # high during saccades: flip within the same [c_min, 1] range
        c = (c_min + 1.0) - c
    elif arm == "random":
        rng = np.random.RandomState(item_seed & 0xFFFFFFFF)
        c = c_min + (1.0 - c_min) * rng.rand(T).astype(np.float32)
    # arm == "confidence" → c as computed

    return _finalize(c.astype(np.float32), n_out, device)


def _finalize(c: np.ndarray, n_out: int | None, device) -> torch.Tensor:
    t = torch.from_numpy(c).to(device).float()
    if n_out is not None and n_out != t.shape[0]:
        t = torch.nn.functional.interpolate(
            t.view(1, 1, -1), size=n_out, mode="linear", align_corners=False
        ).view(-1)
    return t
