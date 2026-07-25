"""Convert gaze trajectory → compact fixation-sequence text prefix.

The text captures TEMPORAL ORDER of fixations (region, duration, hand co-location)
— information that per-frame visual token selection cannot encode. Prepend to
the question so the LLM sees gaze dynamics as language, orthogonal to M1's
token selection.

Arms:
  traj_to_text(traj)     — actual gaze trajectory → fixation sequence (the method)
  random_to_text(n, rng) — random region sequence  → placebo control
  attn_to_text(scores)   — attention peaks          → attn-twin control (future)

Format example:
  [Gaze] center(5f) → upper-left(3f+hand) → center(6f+hand) → lower-right(2f)
"""
from __future__ import annotations

import numpy as np

_VERT  = ["upper", "middle", "lower"]
_HORIZ = ["left", "center", "right"]

SACCADE_SPEED  = 0.25   # gaze_speed threshold: above → saccade, below → fixation
MIN_FIX_FRAMES = 1      # minimum consecutive low-speed frames to form a fixation
HAND_DIST      = 0.20   # normalized distance threshold for gaze-hand co-location
MAX_CLUSTERS   = 5      # cap on fixation clusters reported in text
MIN_VALID_FRAC = 0.30   # below this, mark text with validity warning


def _region_word(x: float, y: float) -> str:
    """Normalized (x,y) → coarse 3×3 region label."""
    def bucket(v, labels):
        return labels[min(len(labels) - 1, max(0, int(v * len(labels))))]
    v = bucket(float(y), _VERT)
    h = bucket(float(x), _HORIZ)
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    if h == "center":
        return f"{v} center"
    return f"{v}-{h}"


def _detect_fixation_clusters(gaze_pos, gaze_speed, gaze_mask):
    """Group consecutive low-speed valid frames into fixation clusters.
    Returns list of dicts: {start, end, center, n_frames}."""
    T = len(gaze_pos)
    clusters = []
    i = 0
    while i < T:
        if not gaze_mask[i] or gaze_speed[i] > SACCADE_SPEED:
            i += 1
            continue
        j = i
        while j < T and gaze_mask[j] and gaze_speed[j] <= SACCADE_SPEED:
            j += 1
        if j - i >= MIN_FIX_FRAMES:
            frames = gaze_pos[i:j][gaze_mask[i:j]]
            if len(frames) > 0:
                clusters.append({
                    'start': i, 'end': j,
                    'center': frames.mean(axis=0),
                    'n_frames': j - i,
                })
        i = j if j > i else i + 1
    return clusters


def _hand_near(cluster_start, cluster_end, cluster_center, traj) -> bool:
    """True if any valid hand is within HAND_DIST of the fixation center."""
    for pkey, mkey in (("right_pos", "right_mask"), ("left_pos", "left_mask")):
        hp = np.array(traj.get(pkey, []), dtype=np.float32)
        hm = np.array(traj.get(mkey, []), dtype=bool)
        if hp.shape[0] == 0:
            continue
        seg_m = hm[cluster_start:cluster_end]
        seg_p = hp[cluster_start:cluster_end]
        if seg_m.any():
            h_pos = seg_p[seg_m].mean(axis=0)
            if float(np.linalg.norm(cluster_center - h_pos)) < HAND_DIST:
                return True
    return False


def traj_to_text(traj: dict, n_frames: int = 128) -> str:
    """Gaze trajectory dict → fixation-sequence text prefix.

    Returns empty string if gaze data is absent or has no valid frames.
    """
    gaze_pos   = np.array(traj.get("gaze_pos",   []), dtype=np.float32)   # (T,2)
    gaze_speed = np.array(traj.get("gaze_speed", []), dtype=np.float32).ravel()  # (T,)
    gaze_mask  = np.array(traj.get("gaze_mask",  []), dtype=bool)           # (T,)

    T = len(gaze_pos)
    if T < 2 or int(gaze_mask.sum()) < 2:
        return ""

    if len(gaze_speed) < T:                       # pad if shape mismatch
        gaze_speed = np.pad(gaze_speed, (0, T - len(gaze_speed)))

    valid_frac = float(gaze_mask.mean())
    clusters   = _detect_fixation_clusters(gaze_pos, gaze_speed, gaze_mask)

    if not clusters:
        # fallback: describe dominant region only
        pos  = gaze_pos[gaze_mask].mean(axis=0)
        word = _region_word(pos[0], pos[1])
        pfx  = f"[Gaze~{int(valid_frac*100)}%]" if valid_frac < MIN_VALID_FRAC else "[Gaze]"
        return f"{pfx} mainly {word}"

    # Merge adjacent clusters with the same region word
    merged: list[dict] = []
    for c in clusters:
        word     = _region_word(c["center"][0], c["center"][1])
        has_hand = _hand_near(c["start"], c["end"], c["center"], traj)
        if merged and merged[-1]["word"] == word:
            merged[-1]["n_frames"] += c["n_frames"]
            merged[-1]["has_hand"]  = merged[-1]["has_hand"] or has_hand
        else:
            merged.append({"word": word, "n_frames": c["n_frames"], "has_hand": has_hand})

    # Render at most MAX_CLUSTERS clusters
    parts = []
    for m in merged[:MAX_CLUSTERS]:
        tag = "+hand" if m["has_hand"] else ""
        parts.append(f"{m['word']}({m['n_frames']}f{tag})")

    suffix = "..." if len(merged) > MAX_CLUSTERS else ""
    pfx    = f"[Gaze~{int(valid_frac*100)}%]" if valid_frac < MIN_VALID_FRAC else "[Gaze]"
    return f"{pfx} {' → '.join(parts)}{suffix}"


def random_to_text(n_clusters: int = 4, rng: np.random.RandomState = None) -> str:
    """Random region-sequence placebo — same format as traj_to_text."""
    if rng is None:
        rng = np.random.RandomState(42)
    parts = []
    for _ in range(n_clusters):
        x = float(rng.uniform(0.0, 1.0))
        y = float(rng.uniform(0.0, 1.0))
        n = int(rng.randint(2, 9))
        parts.append(f"{_region_word(x, y)}({n}f)")
    return f"[Gaze] {' → '.join(parts)}"


def build_question_prefix(arm: str, traj: dict, n_frames: int = 128,
                           item_seed: int = 0) -> str:
    """Dispatch to the correct prefix builder for `arm`."""
    if arm == "gaze":
        return traj_to_text(traj, n_frames)
    if arm == "random":
        rng = np.random.RandomState(item_seed & 0xFFFFFFFF)
        # Use same number of clusters as gaze arm would (heuristic: 4)
        return random_to_text(n_clusters=4, rng=rng)
    return ""   # "none" → no prefix
