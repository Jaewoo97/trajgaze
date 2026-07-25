"""Gaze-grounded language reference (lever ② / mechanism ③).

Turns a gaze fixation into a NON-LEAKY natural-language localization the LLM can
reason over: it says WHERE the person looks (region, or relation to a hand) but
never WHAT the object is — so it disambiguates the referent without solving an
object-ID task for free.

  region:   "The person's gaze is on the upper-left of the scene."
  relation: "The person's gaze is on the object near their right hand."  (gaze-hand OR)

The SAME builder serves three conditions by swapping the input point:
  - gaze     → dominant fixation point          (the method)
  - saliency → VisionZip attention-peak point    (content-only attention twin = control)
  - random   → a random point                    (placebo control)
Holding everything else (M1 token selection, gaze overlay) constant, the contrast
gaze − saliency isolates the gaze contribution; gaze − random rules out "any hint".

Parameter-free, CPU-only. Granularity (grid) is a knob to refine after the
object_id_hard de-risk. Coordinates are normalized [0,1] (x=horizontal, y=vertical),
matching traj_anticipate / scanpath.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_VERT = ["upper", "middle", "lower"]
_HORIZ = ["left", "center", "right"]


@dataclass
class GazeGroundingConfig:
    grid: int = 3                 # NxN region words (3 → upper/middle/lower × left/center/right)
    use_hand: bool = True         # enable the gaze-hand relation branch
    hand_dist: float = 0.20       # gaze-hand proximity (normalized) to trigger relation
    fixation_frac: float = 0.30   # min gaze validity to emit a sentence at all

    def describe(self) -> str:
        return f"Grounding[grid={self.grid} hand={self.use_hand}@{self.hand_dist}]"


def _as2d(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    return t if (t.ndim == 2 and t.shape[1] == 2) else None


def region_word(x: float, y: float, grid: int = 3) -> str:
    """Normalized (x,y) → coarse region phrase. 3×3 uses upper/middle/lower ×
    left/center/right; center-center collapses to 'center'."""
    def bucket(v, labels):
        i = min(len(labels) - 1, max(0, int(v * len(labels))))
        return labels[i]
    if grid == 3:
        v, h = bucket(y, _VERT), bucket(x, _HORIZ)
        if v == "middle" and h == "center":
            return "center"
        if v == "middle":
            return h                      # "left"/"right" (mid row)
        if h == "center":
            return f"{v} center"
        return f"{v}-{h}"                  # "upper-left", "lower-right", ...
    # generic fallback: fractional cell
    cx = min(grid - 1, int(x * grid)); cy = min(grid - 1, int(y * grid))
    return f"cell ({cy},{cx}) of a {grid}x{grid} grid"


def _nearest_hand(point, traj, valid_thresh: float = 0.5):
    """Return (side, dist) of the nearest valid hand to `point`, averaged over
    frames where that hand is valid. None if no hand is usable."""
    px, py = float(point[0]), float(point[1])
    best = None
    for side, pkey, mkey in (("left", "left_pos", "left_mask"),
                             ("right", "right_pos", "right_mask")):
        pos = _as2d(traj.get(pkey)) if isinstance(traj, dict) else None
        if pos is None:
            continue
        msk = torch.as_tensor(traj.get(mkey, torch.zeros(pos.shape[0])), dtype=torch.float32).view(-1)
        if msk.shape[0] != pos.shape[0] or msk.mean() < valid_thresh:
            continue
        m = msk > 0
        if m.sum() == 0:
            continue
        hc = pos[m].mean(0)               # mean hand position over valid frames
        d = ((hc[0] - px) ** 2 + (hc[1] - py) ** 2) ** 0.5
        if best is None or d < best[1]:
            best = (side, float(d))
    return best


def dominant_gaze_point(traj, cfg: GazeGroundingConfig):
    """Dominant fixation point = centroid of gaze in its most-occupied region.
    Returns (x,y) or None if gaze is missing/too invalid."""
    if not isinstance(traj, dict):
        return None
    g = _as2d(traj.get("gaze_pos"))
    if g is None or g.shape[0] == 0:
        return None
    L = g.shape[0]
    m = torch.as_tensor(traj.get("gaze_mask", torch.ones(L)), dtype=torch.float32).view(-1)
    if m.shape[0] != L:
        m = torch.ones(L)
    valid = m > 0
    if valid.float().mean() < cfg.fixation_frac or valid.sum() == 0:
        return None
    gv = g[valid]
    # mode region, then centroid within it
    keys = [region_word(float(p[0]), float(p[1]), cfg.grid) for p in gv]
    from collections import Counter
    mode = Counter(keys).most_common(1)[0][0]
    sel = [i for i, k in enumerate(keys) if k == mode]
    c = gv[sel].mean(0)
    return (float(c[0]), float(c[1]))


def grounding_sentence(point, traj, cfg: GazeGroundingConfig) -> str:
    """Build the non-leaky localization sentence for a given point (gaze/saliency/
    random). gaze-hand relation takes priority when a hand is near the point."""
    if point is None:
        return ""
    if cfg.use_hand:
        nh = _nearest_hand(point, traj)
        if nh is not None and nh[1] <= cfg.hand_dist:
            return f"The person's gaze is on the object near their {nh[0]} hand."
    return f"The person's gaze is on the {region_word(point[0], point[1], cfg.grid)} of the scene."


def gaze_grounding_sentence(traj, cfg: GazeGroundingConfig) -> str:
    """Convenience: dominant-gaze → sentence (the method condition)."""
    return grounding_sentence(dominant_gaze_point(traj, cfg), traj, cfg)
