"""
Generate EgoGazeVQA interaction .npz files, using the exact same algorithm as
StreamGaze (TrajGaze_v2/data/interaction.py).

Per subclip (one gaze_mapping CSV = one QA video clip):
  inputs:
    gaze:  /workspace/EgoGazeVQA/{ds}/gaze_mapping/{rec_id}/{subclip}_mapping.csv
           columns: frame_idx,gaze_frame_num,matched_gaze_frame,gaze_x,gaze_y,is_exact_match
           gaze_x/gaze_y already normalized [0,1]
    hand:  /workspace/EgoGazeVQA/{ds}/hand_locations/{rec_id}.json
           { "{subclip}_{gfn}.jpg": {"left":[px,py]|null, "right":[px,py]|null} }
           hand coords are PIXEL space -> normalized here by per-clip frame (W,H)
    frames:/workspace/EgoGazeVQA/{ds}/no_gaze/{rec_id}/{subclip}_{gfn}.jpg
  output:
    /workspace/EgoGazeVQA/interaction/{ds}/{rec_id}/{subclip}.npz
    keys identical to StreamGaze: frame_names, I_scores, attend, d_left, d_right,
    v_rel_left, v_rel_right, convergence, lead_lag, present

Notes:
  - frame order / T follow the gaze_mapping CSV (ordered by frame_idx). gaze is
    always present in these CSVs; hand may be null/absent.
  - `present[t]` = the {subclip}_{gfn}.jpg exists on disk.
  - `attend` is not produced by interaction.py and is unused by TrajGazeMerge
    training; computed here as a documented proxy int8(I_scores[t].max() > EPSILON)
    so the npz is schema-identical to StreamGaze.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import glob
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/trajgaze_st")
from TrajGaze_v2.data.interaction import (
    compute_traj_features,
    compute_importance_scores,
    EPSILON,
)

ROOT = "/workspace/EgoGazeVQA"
DATASETS = ["ego4d", "egoexo", "egtea"]


def _img_wh(path: str, fallback=(640, 480)):
    try:
        with Image.open(path) as im:
            return im.size  # (W, H)
    except Exception:
        return fallback


def _none_xy(x, y):
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return None
    return (fx, fy)


def process_subclip(ds: str, rec_id: str, subclip: str,
                     csv_path: str, hand_json: dict) -> dict | None:
    rec_frame_dir = os.path.join(ROOT, ds, "no_gaze", rec_id)

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["frame_idx"]))

    # Per-clip image size from the first existing frame jpg
    img_w, img_h = 640, 480
    for r in rows:
        gfn = r["gaze_frame_num"]
        jp = os.path.join(rec_frame_dir, f"{subclip}_{gfn}.jpg")
        if os.path.exists(jp):
            img_w, img_h = _img_wh(jp)
            break

    frame_names, gaze_list, left_list, right_list, present = [], [], [], [], []
    for r in rows:
        gfn = r["gaze_frame_num"]
        fname = f"{subclip}_{gfn}.jpg"
        frame_names.append(fname)
        present.append(os.path.exists(os.path.join(rec_frame_dir, fname)))

        gaze_list.append(_none_xy(r.get("gaze_x"), r.get("gaze_y")))  # already [0,1]

        h = hand_json.get(fname, {}) if isinstance(hand_json, dict) else {}
        lh = h.get("left") if isinstance(h, dict) else None
        rh = h.get("right") if isinstance(h, dict) else None
        left_list.append((lh[0] / img_w, lh[1] / img_h)
                          if lh and lh[0] is not None else None)
        right_list.append((rh[0] / img_w, rh[1] / img_h)
                           if rh and rh[0] is not None else None)

    feats = compute_traj_features(gaze_list, left_list, right_list)
    I_scores = compute_importance_scores(gaze_list, left_list, right_list)  # (T,196)
    attend = (I_scores.max(axis=1) > EPSILON).astype(np.int8)

    return {
        "frame_names": np.array(frame_names, dtype="<U32"),
        "I_scores":    I_scores.astype(np.float32),
        "attend":      attend,
        "d_left":      feats["d_left"].astype(np.float32),
        "d_right":     feats["d_right"].astype(np.float32),
        "v_rel_left":  feats["v_rel_left"].astype(np.float32),
        "v_rel_right": feats["v_rel_right"].astype(np.float32),
        "convergence": feats["convergence"].astype(np.float32),
        "lead_lag":    feats["lead_lag"].astype(np.float32),
        "present":     np.array(present, dtype=bool),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N subclips total (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    total, done, skipped, errors = 0, 0, 0, 0
    for ds in args.datasets:
        gm_root = os.path.join(ROOT, ds, "gaze_mapping")
        if not os.path.isdir(gm_root):
            print(f"[skip ds] {ds}: no gaze_mapping", flush=True)
            continue
        for rec_id in sorted(os.listdir(gm_root)):
            rec_dir = os.path.join(gm_root, rec_id)
            if not os.path.isdir(rec_dir):
                continue
            hand_path = os.path.join(ROOT, ds, "hand_locations", f"{rec_id}.json")
            try:
                hand_json = json.load(open(hand_path)) if os.path.exists(hand_path) else {}
            except Exception:
                hand_json = {}
            for csv_file in sorted(glob.glob(os.path.join(rec_dir, "*_mapping.csv"))):
                subclip = os.path.basename(csv_file)[: -len("_mapping.csv")]
                out_dir = os.path.join(ROOT, "interaction", ds, rec_id)
                out_path = os.path.join(out_dir, f"{subclip}.npz")
                total += 1
                if args.limit and total > args.limit:
                    break
                if os.path.exists(out_path) and not args.overwrite:
                    skipped += 1
                    continue
                try:
                    data = process_subclip(ds, rec_id, subclip, csv_file, hand_json)
                    if data is None:
                        skipped += 1
                        continue
                    os.makedirs(out_dir, exist_ok=True)
                    np.savez_compressed(out_path, **data)
                    done += 1
                    if done % 50 == 0:
                        print(f"[{ds}] done={done} skipped={skipped} "
                              f"errors={errors} (last {rec_id}/{subclip})", flush=True)
                except Exception as e:
                    errors += 1
                    print(f"[ERR] {ds}/{rec_id}/{subclip}: {e!r}", flush=True)
            if args.limit and total > args.limit:
                break

    print(f"\nDONE. total={total} written={done} skipped={skipped} errors={errors}",
          flush=True)


if __name__ == "__main__":
    main()
