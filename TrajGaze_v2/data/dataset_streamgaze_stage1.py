"""Raw gaze + hand position loader for StreamGaze_v2 stage-1 datasets.

Reconstructed minimal helper that the temporal stage-1 dataset
(`dataset_temporal.py`) imports as `_load_raw_positions`. The original module
file was never committed to this repository (lives only on the training
server); the loading semantics are reverse-engineered from the still-present
TrajGazeMerge equivalent in `TrajGazeMerge/data/dataset.py::_load_traj`,
which reads the same on-disk layout.

Returned lists are in normalised [0, 1] image coordinates so they can be fed
directly to `interaction.compute_traj_features` and
`interaction.compute_importance_scores`.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from PIL import Image

from TrajGazeMerge.data.dataset import GAZE_BASE, HAND_BASE


_PosOpt = Optional[tuple[float, float]]


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _image_size(path: str) -> tuple[int, int]:
    try:
        img = Image.open(path)
        return img.width, img.height
    except Exception:
        return 1920, 1080  # fallback matches TrajGazeMerge default


def _load_raw_positions(
    stem: str,
    dataset: str,
    sampled_paths: list[str],
) -> tuple[list[_PosOpt], list[_PosOpt], list[_PosOpt]]:
    """Return per-sampled-frame normalised (gaze, left_hand, right_hand) positions.

    Each output list has len(sampled_paths) entries. An entry is either
    ``(x, y)`` in [0, 1] (or near it) or ``None`` when the modality is missing
    for that frame.

    Mirrors the gaze + hand parts of
    `TrajGazeMerge.data.dataset._load_traj` (lines reading ``gaze_json[fname]``
    and ``hand_json[fname]``), but skips the trajectory feature aggregation --
    the caller (``compute_traj_features`` / ``compute_importance_scores``) does
    that itself.
    """
    gaze_json = _read_json(os.path.join(GAZE_BASE, dataset, "viz", f"{stem}.json"))
    hand_json = _read_json(os.path.join(HAND_BASE, dataset, "viz", f"{stem}.json"))

    if not sampled_paths:
        return [], [], []
    img_w, img_h = _image_size(sampled_paths[0])

    gaze_list:  list[_PosOpt] = []
    left_list:  list[_PosOpt] = []
    right_list: list[_PosOpt] = []

    for fp in sampled_paths:
        fname = os.path.basename(fp)

        # gaze
        gxy = gaze_json.get(fname)
        if gxy and len(gxy) == 2 and gxy[0] is not None:
            gaze_list.append((float(gxy[0]) / img_w, float(gxy[1]) / img_h))
        else:
            gaze_list.append(None)

        # hands
        h_entry = hand_json.get(fname)
        if isinstance(h_entry, dict):
            lh = h_entry.get("left")
            if lh and len(lh) == 2 and lh[0] is not None:
                left_list.append((float(lh[0]) / img_w, float(lh[1]) / img_h))
            else:
                left_list.append(None)
            rh = h_entry.get("right")
            if rh and len(rh) == 2 and rh[0] is not None:
                right_list.append((float(rh[0]) / img_w, float(rh[1]) / img_h))
            else:
                right_list.append(None)
        else:
            left_list.append(None)
            right_list.append(None)

    return gaze_list, left_list, right_list
