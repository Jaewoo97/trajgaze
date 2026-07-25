"""Gaze-event-driven temporal frame sampling.

VisionZip samples 128 frames UNIFORMLY from the video.
This module replaces that with EVENT-DRIVEN sampling:
    n_event frames at gaze fixation-onset / saccade-peak times (high information)
  + n_uniform frames uniformly spaced (scene coverage)

The hypothesis: gaze events (fixation onsets) mark the moments when the person's
visual system detected something important. Concentrating frame budget there should
help temporal/action tasks while maintaining global coverage.

Contrast with tbudget (failed):
  - tbudget: same 128 uniform frames, just reallocates tokens within frames
  - here:    DIFFERENT 128 frames drawn from the full frame pool (~400-2000 frames)

Arms:
  gaze_event   — event detection from actual gaze trajectory  (the method)
  random_event — same structure, random event times           (placebo)
  uniform      — identical to current M1 sampling             (baseline)
"""
from __future__ import annotations

import numpy as np
from typing import List

SACCADE_SPEED  = 0.25   # speed threshold: above = saccade, below = fixation
NMS_FRAC       = 0.5    # temporal NMS radius as fraction of (N_total / n_event)


def _detect_event_scores(gaze_speed: np.ndarray, gaze_mask: np.ndarray) -> np.ndarray:
    """Score each traj frame by gaze-event significance.

    High score = fixation onset (saccade→fixation transition)
    Medium score = saccade peak (local speed maximum while saccading)
    Zero = stable fixation, invalid frames
    """
    T = len(gaze_speed)
    scores = np.zeros(T, dtype=np.float32)

    for i in range(1, T):
        if not gaze_mask[i]:
            continue
        # Fixation onset: crossed from high to low speed
        if gaze_mask[i - 1] and gaze_speed[i - 1] > SACCADE_SPEED and gaze_speed[i] <= SACCADE_SPEED:
            scores[i] += 2.0                # fixation onset — highest priority

    for i in range(1, T - 1):
        if not gaze_mask[i]:
            continue
        # Saccade peak: local speed maximum above threshold
        if (gaze_speed[i] > SACCADE_SPEED
                and gaze_speed[i] >= gaze_speed[i - 1]
                and gaze_speed[i] >= gaze_speed[i + 1]):
            scores[i] += gaze_speed[i]      # proportional to saccade magnitude

    scores[~gaze_mask] = 0.0
    return scores


def gaze_event_sample_paths(
    full_frame_paths: List[str],
    traj: dict,
    n_frames: int = 128,
    n_event_frac: float = 0.5,
) -> List[str]:
    """Select n_frames from full_frame_paths using gaze events.

    Args:
        full_frame_paths: all extracted frames for this clip (unsorted assumed sorted)
        traj: gaze trajectory dict (keys: gaze_speed, gaze_mask, ...)
        n_frames: total frames to select
        n_event_frac: fraction of n_frames to fill from gaze events (rest = uniform)

    Returns:
        List of n_frames frame paths, time-sorted.
    """
    N_total = len(full_frame_paths)
    if N_total <= n_frames:
        return full_frame_paths[:]

    n_event   = int(n_frames * n_event_frac)
    n_uniform = n_frames - n_event

    gaze_speed = np.array(traj.get("gaze_speed", []), dtype=np.float32).ravel()
    gaze_mask  = np.array(traj.get("gaze_mask",  []), dtype=bool)
    n_traj     = len(gaze_speed)

    # --- map traj indices → full-frame-pool indices ---
    # traj is uniformly sampled from the same pool, so traj_idx → pool_idx
    def traj_to_pool(ti: int) -> int:
        return int(ti * N_total / max(1, n_traj))

    event_scores = _detect_event_scores(gaze_speed, gaze_mask)

    # temporal NMS radius in full-frame-pool space
    nms_radius = max(1, int(NMS_FRAC * N_total / max(1, n_event)))
    suppressed = np.zeros(N_total, dtype=bool)
    event_pool_indices: List[int] = []

    for ti in np.argsort(event_scores)[::-1]:
        if len(event_pool_indices) >= n_event:
            break
        if event_scores[ti] <= 0:
            break
        fi = traj_to_pool(int(ti))
        fi = min(fi, N_total - 1)
        if suppressed[fi]:
            continue
        event_pool_indices.append(fi)
        lo = max(0, fi - nms_radius)
        hi = min(N_total, fi + nms_radius + 1)
        suppressed[lo:hi] = True

    # --- uniform frames for global coverage ---
    event_set = set(event_pool_indices)
    step = N_total / n_uniform
    uniform_pool_indices = [
        min(N_total - 1, int(i * step))
        for i in range(n_uniform)
        if min(N_total - 1, int(i * step)) not in event_set
    ][:n_uniform]

    # --- combine, sort, deduplicate ---
    all_indices = sorted(set(event_pool_indices) | set(uniform_pool_indices))

    # pad to n_frames if dedup reduced count
    if len(all_indices) < n_frames:
        existing = set(all_indices)
        extras = [i for i in range(N_total) if i not in existing]
        step2 = max(1, len(extras) // (n_frames - len(all_indices)))
        all_indices += extras[::step2][: n_frames - len(all_indices)]
        all_indices = sorted(set(all_indices))

    return [full_frame_paths[i] for i in all_indices[:n_frames]]


def random_event_sample_paths(
    full_frame_paths: List[str],
    n_frames: int = 128,
    n_event_frac: float = 0.5,
    seed: int = 0,
) -> List[str]:
    """Same structure as gaze_event_sample_paths but with random event times.

    Placebo control: tests whether ANY non-uniform temporal sampling helps,
    independent of whether events come from gaze.
    """
    N_total = len(full_frame_paths)
    if N_total <= n_frames:
        return full_frame_paths[:]

    n_event   = int(n_frames * n_event_frac)
    n_uniform = n_frames - n_event

    rng = np.random.RandomState(seed)
    event_indices = sorted(rng.choice(N_total, size=n_event, replace=False).tolist())

    event_set = set(event_indices)
    step = N_total / n_uniform
    uniform_indices = [
        min(N_total - 1, int(i * step))
        for i in range(n_uniform)
        if min(N_total - 1, int(i * step)) not in event_set
    ][:n_uniform]

    all_indices = sorted(set(event_indices) | set(uniform_indices))
    if len(all_indices) < n_frames:
        existing = set(all_indices)
        extras = [i for i in range(N_total) if i not in existing]
        all_indices += extras[: n_frames - len(all_indices)]
        all_indices = sorted(set(all_indices))

    return [full_frame_paths[i] for i in all_indices[:n_frames]]


def select_frame_paths(
    arm: str,
    full_frame_paths: List[str],
    traj: dict,
    n_frames: int = 128,
    n_event_frac: float = 0.5,
    item_seed: int = 0,
) -> List[str]:
    """Dispatch to the correct sampler for `arm`."""
    if arm == "gaze_event":
        return gaze_event_sample_paths(full_frame_paths, traj, n_frames, n_event_frac)
    if arm == "random_event":
        return random_event_sample_paths(full_frame_paths, n_frames, n_event_frac, seed=item_seed)
    # "uniform" — identical to M1
    from TrajGazeMerge.data.dataset import _sample_paths
    return _sample_paths(full_frame_paths, n_frames)
