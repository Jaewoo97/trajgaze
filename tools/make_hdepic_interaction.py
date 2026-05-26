"""
Generate StreamGaze-schema interaction.npz per HD-EPIC recording, using the
same compute_traj_features + compute_importance_scores algorithm as
StreamGaze_v2 and the EgoGazeVQA generator I wrote earlier.

For HD-EPIC, gaze is absent (Aria MPS gaze is in CPF radians; projecting
requires the Aria SDK — out of scope here). The algorithm gracefully degrades:
gaze-dependent terms (d_left/d_right, v_rel, convergence, lead_lag) reduce to
zeros wherever gaze is None; I_scores still respond to hand-only signals.

Output: /workspace/HD-EPIC/interaction/{stem}.npz
        keys: frame_names, I_scores, attend, d_left, d_right,
              v_rel_left, v_rel_right, convergence, lead_lag, present
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/workspace/trajgaze_st")
from TrajGaze_v2.data.interaction import (
    compute_traj_features,
    compute_importance_scores,
    EPSILON,
)

ROOT       = "/workspace/HD-EPIC"
FRAMES     = os.path.join(ROOT, "frames_extracted")
HANDS      = os.path.join(ROOT, "hand_locations")
OUT_DIR    = os.path.join(ROOT, "interaction")

IMG_W = IMG_H = 224   # frames are pre-extracted at 224×224


def _participant_of(stem: str) -> str:
    return stem.split("-", 1)[0]


def _frames_for_recording(stem: str) -> list[str]:
    d = os.path.join(FRAMES, _participant_of(stem), stem)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".jpg"))


def process_one(stem: str, out_path: str) -> tuple[int, int]:
    frames = _frames_for_recording(stem)
    T = len(frames)
    if T == 0:
        return 0, 0
    hand_path = os.path.join(HANDS, f"{stem}.json")
    hand_json: dict = {}
    if os.path.exists(hand_path):
        try:
            hand_json = json.load(open(hand_path))
        except Exception:
            hand_json = {}

    gaze_list: list = [None] * T
    left_list:  list = []
    right_list: list = []
    n_hand = 0
    for fname in frames:
        h = hand_json.get(fname)
        if isinstance(h, dict):
            lh = h.get("left")
            rh = h.get("right")
            left_list.append((lh[0] / IMG_W, lh[1] / IMG_H) if (lh and lh[0] is not None) else None)
            right_list.append((rh[0] / IMG_W, rh[1] / IMG_H) if (rh and rh[0] is not None) else None)
            if lh or rh:
                n_hand += 1
        else:
            left_list.append(None)
            right_list.append(None)

    feats   = compute_traj_features(gaze_list, left_list, right_list)
    I_scores = compute_importance_scores(gaze_list, left_list, right_list)
    attend   = (I_scores.max(axis=1) > EPSILON).astype(np.int8)
    present  = np.ones(T, dtype=bool)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_names=np.array(frames, dtype="<U32"),
        I_scores=I_scores.astype(np.float32),
        attend=attend,
        d_left=feats["d_left"].astype(np.float32),
        d_right=feats["d_right"].astype(np.float32),
        v_rel_left=feats["v_rel_left"].astype(np.float32),
        v_rel_right=feats["v_rel_right"].astype(np.float32),
        convergence=feats["convergence"].astype(np.float32),
        lead_lag=feats["lead_lag"].astype(np.float32),
        present=present,
    )
    return T, n_hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    hand_files = sorted(glob.glob(os.path.join(HANDS, "*.json")))
    stems = [os.path.basename(f)[:-len(".json")] for f in hand_files]
    print(f"found {len(stems)} HD-EPIC recordings with hand_locations", flush=True)

    t0 = time.time()
    n_done = n_skip = n_err = 0
    for i, stem in enumerate(stems, 1):
        out = os.path.join(OUT_DIR, f"{stem}.npz")
        if os.path.exists(out) and not args.overwrite:
            n_skip += 1
            continue
        try:
            T, nh = process_one(stem, out)
            n_done += 1
            if i % 10 == 0 or i == len(stems):
                print(f"  [{i}/{len(stems)}] {stem}: T={T} hand_frames={nh}  "
                      f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            n_err += 1
            print(f"  [{i}/{len(stems)}] ERR {stem}: {e!r}", flush=True)
    print(f"\nDONE. written={n_done}  skipped={n_skip}  errors={n_err}  → {OUT_DIR}")


if __name__ == "__main__":
    main()
