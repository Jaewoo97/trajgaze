"""Step-aware loss curriculum for joint stage-1 training of TrajGazeV2Temporal.

Random-init joint training of the full encoder destabilises because the
InterFrameTransformer rewrites the trajectory-token distribution while
TemporalVisualTrajFusion is simultaneously trying to fit visual cross-attention
on top of those moving queries. Stacking 4 supervisory losses on this
non-stationary state yields val loss ~0.1274 vs ~0.0188 for either ablation
alone.

This module ramps the four losses in over training rather than firing all four
from step 0:

  | Step ratio  | l_traj | l_score_past | l_score_future | l_score_traj |
  |-------------|--------|--------------|----------------|--------------|
  | 0   - 10%   | 1.0    | 0.0          | 0.0            | 0.0          |
  | 10  - 25%   | 1.0    | linear 0->1  | 0.0            | 0.0          |
  | 25  - 50%   | 1.0    | 1.0          | linear 0->1    | 0.0          |
  | 50  - 100%  | 1.0    | 1.0          | 1.0            | linear 0->1  |

Ordering rationale: l_traj is the only loss that does not fight the encoder's
visual cross-attention, so it leads. l_score_past directly grounds attention
maps. l_score_future depends on the score decoder, which needs context first.
l_score_traj chains decoder x encoder attention via chain rule and is the
noisiest signal -- it is only useful once encoder attention has stabilised.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    traj: float
    score_past: float
    score_future: float
    score_traj: float


def _ramp(progress: float, start: float, end: float) -> float:
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / (end - start)


def curriculum_weights(step: int, total_steps: int) -> LossWeights:
    """Return per-loss weights for the given step in a curriculum of total_steps.

    `step` is 0-indexed. `total_steps` should be the planned total optimiser
    steps for stage-1; if unknown, pass an estimate -- the curriculum is
    monotone so over- or under-shooting only affects how much of each phase
    is held at the saturated weight.
    """
    if total_steps <= 0:
        return LossWeights(1.0, 1.0, 1.0, 1.0)

    progress = min(max(step / total_steps, 0.0), 1.0)

    return LossWeights(
        traj=1.0,
        score_past=_ramp(progress, 0.10, 0.25),
        score_future=_ramp(progress, 0.25, 0.50),
        score_traj=_ramp(progress, 0.50, 1.00),
    )


def total_from_weighted(loss_terms: dict, weights: LossWeights):
    """Apply weights to the named loss terms and return the weighted sum.

    Expects keys: loss_traj, loss_score_past, loss_score_fut, loss_score_traj.
    Missing keys contribute zero. Returned tensor preserves device/dtype of
    the first present term.
    """
    pairs = [
        (loss_terms.get("loss_traj"),       weights.traj),
        (loss_terms.get("loss_score_past"), weights.score_past),
        (loss_terms.get("loss_score_fut"),  weights.score_future),
        (loss_terms.get("loss_score_traj"), weights.score_traj),
    ]
    total = None
    for term, w in pairs:
        if term is None or w == 0.0:
            continue
        contrib = term * w
        total = contrib if total is None else total + contrib
    if total is None:
        # Fallback: if every weight is 0, return a zero tensor on the same
        # device as one of the terms (just to keep .backward() callable).
        for term, _ in pairs:
            if term is not None:
                return term * 0.0
        raise ValueError("No loss terms provided to total_from_weighted.")
    return total
