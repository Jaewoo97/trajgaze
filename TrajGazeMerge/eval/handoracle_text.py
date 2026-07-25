"""HandOracleDeRisk — leakage-safe hand-kinematics oracle text (+ placebo + audit).

Per the Seed (HandOracleDeRisk): the oracle encodes hand KINEMATICS only
(side, image-normalized 2D position, velocity) built from HD-EPIC GT hand
centroids. It deliberately omits: action verbs, action/step ORDER, contact-object
names, and any answer-option substring — so a gain cannot be an answer leak.

Three arms share this module:
  gt_hand : build_hand_text(traj, frame_paths)
  placebo : build_placebo_text(gt_text)  (same length/format, scrambled numbers)
  baseline: no injection (empty string)

leakage_audit() is the acceptance gate: it must find ZERO ordering words and ZERO
shared content-word with any option, for every injected string.
"""
from __future__ import annotations
import hashlib
import random
import re

# ── leakage vocabulary ────────────────────────────────────────────────────────
# Ordering / sequence words that could encode the action-order answer.
ORDER_WORDS = {
    "then", "after", "before", "next", "first", "second", "third", "fourth",
    "fifth", "last", "finally", "begin", "begins", "beginning", "start", "starts",
    "started", "starting", "end", "ends", "ended", "ending", "while", "during",
    "followed", "following", "prior", "subsequently", "earlier", "later", "order",
    "sequence", "step", "steps", "once", "when", "precede", "precedes", "preceding",
}
# Fixed template vocabulary the oracle IS allowed to use (kinematics only).
_TEMPLATE_STOP = {
    "hand", "hands", "left", "right", "camera", "wearer", "kinematics", "image",
    "normalized", "frame", "frames", "position", "positions", "velocity", "l", "r",
    "and", "the", "of", "a", "to", "in", "at", "no", "detected", "motion", "static",
    "moving", "near", "together", "apart", "top", "bottom", "center", "field",
    "view", "coordinates", "coord", "coords", "x", "y", "dx", "dy", "unit", "units",
    "px", "pixel", "pixels", "over", "sampled", "uniformly", "per", "value", "values",
}
_STOP = _TEMPLATE_STOP | {"is", "are", "with", "for", "on", "by", "or"}
_WORD = re.compile(r"[a-zA-Z]+")


def _content_words(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s) if len(w) > 2 and w.lower() not in _STOP}


def _bucket(x: float, y: float) -> str:
    """Coarse spatial bucket (kinematics, not object identity)."""
    hx = "left" if x < 0.4 else ("right" if x > 0.6 else "center")
    hy = "top" if y < 0.4 else ("bottom" if y > 0.6 else "middle")
    return f"{hy}-{hx}"


def build_hand_text(traj: dict, max_frames: int = 12) -> str:
    """Compact, leakage-safe per-frame hand-kinematics description.

    Uses left/right pos+vel+mask from the HD-EPIC traj dict (image-normalized).
    Deterministic, numeric; no verbs, no order words, no object names."""
    import torch

    def _to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    lp, lv, lm = _to_np(traj["left_pos"]),  _to_np(traj["left_vel"]),  _to_np(traj["left_mask"])
    rp, rv, rm = _to_np(traj["right_pos"]), _to_np(traj["right_vel"]), _to_np(traj["right_mask"])
    T = len(lp)
    if T == 0:
        return "Hand kinematics: no hand detected."
    idx = [int(i * T / min(max_frames, T)) for i in range(min(max_frames, T))]
    rows = []
    for k, t in enumerate(idx):
        parts = []
        if lm[t]:
            sp = float((lv[t, 0] ** 2 + lv[t, 1] ** 2) ** 0.5)
            mo = "static" if sp < 0.01 else "moving"
            parts.append(f"L {_bucket(lp[t,0],lp[t,1])} ({lp[t,0]:.2f},{lp[t,1]:.2f}) {mo}")
        if rm[t]:
            sp = float((rv[t, 0] ** 2 + rv[t, 1] ** 2) ** 0.5)
            mo = "static" if sp < 0.01 else "moving"
            parts.append(f"R {_bucket(rp[t,0],rp[t,1])} ({rp[t,0]:.2f},{rp[t,1]:.2f}) {mo}")
        rows.append(f"f{k}: " + ("; ".join(parts) if parts else "no hand"))
    return ("Camera-wearer hand kinematics (image-normalized x,y; L=left R=right; "
            "frames sampled uniformly):\n" + " | ".join(rows))


def build_placebo_text(gt_text: str, key: str = "") -> str:
    """Same template/length as gt_text, but numeric values scrambled and hand
    rows shuffled — destroys the hand signal while holding format+length fixed."""
    rng = random.Random(int(hashlib.md5((gt_text + key).encode()).hexdigest()[:8], 16))
    header, _, body = gt_text.partition("):\n")
    if not body:
        return gt_text
    rows = body.split(" | ")
    # scramble every float in place, then shuffle row order
    def scramble(m):
        return f"{rng.random():.2f}"
    rows = [re.sub(r"\d\.\d\d", scramble, r) for r in rows]
    rng.shuffle(rows)
    # re-label f-indices so headers still read f0..fN (length preserved)
    rows = [re.sub(r"^f\d+", f"f{i}", r) for i, r in enumerate(rows)]
    return header + "):\n" + " | ".join(rows)


def leakage_audit(text: str, options: list[str], answer: str) -> tuple[bool, list[str]]:
    """Return (ok, violations). ok iff no ordering word and no content-word shared
    with any option appears in `text`."""
    v = []
    tw = _content_words(text)
    low = text.lower()
    for w in ORDER_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", low):
            v.append(f"order-word:{w}")
    for opt in options:
        shared = tw & _content_words(opt)
        if shared:
            v.append(f"option-overlap:{sorted(shared)}")
    return (len(v) == 0, v)
