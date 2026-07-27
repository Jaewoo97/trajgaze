"""
StreamGaze_v2 dataset for TrajGazeMerge training and evaluation.

split='train'  →  egoexolearn videos
split='test'   →  egtea videos

Returns per-item:
    frame_paths       : list[str]  — VLM video frames up to timestamp
    traj_frame_paths  : list[str]  — sampled frames for TrajGaze encoder
    traj              : dict[str, Tensor]  — trajectory features
    question          : str
    options           : list[str]
    answer            : str  — 'A'/'B'/'C'/'D'
    task              : str
    dataset           : str
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SG_ROOT          = os.environ.get("SG_ROOT", "/workspace/datasets/StreamGaze_v2")
FRAMES_BASE      = os.path.join(SG_ROOT, "frames")
GAZE_BASE        = os.path.join(SG_ROOT, "gaze")
HAND_BASE        = os.path.join(SG_ROOT, "hand")
INTERACTION_BASE = os.path.join(SG_ROOT, "interaction")
QA_BASE          = os.path.join(SG_ROOT, "qa")
EXTRACTED_FPS    = 10.0
DATASETS         = ["egtea", "egoexolearn", "holoassist"]

MCQ_TASKS = [
    "past_gaze_sequence_matching",
    "past_non_fixated_object_identification",
    "past_object_transition_prediction",
    "past_scene_recall",
    "present_future_action_prediction",
    "present_object_attribute_recognition",
    "present_object_identification_easy",
    "present_object_identification_hard",
]


def _parse_ts(ts: str) -> float:
    if not ts:
        return 9999.0
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


# Frame variant, mirroring _EG_FRAME_SUB in egogaze_dataset.py. "viz" has the gaze
# marker (red circle + green dot) DRAWN INTO THE PIXELS; "original" is the same
# footage without it. That marker is the only gaze signal a "gaze-free" student
# would otherwise still receive — removing the trajectory-coordinate stream does
# not remove it, and drawing it needs an eye-tracker.
#
# The two consumers are separated so the teacher can keep the overlay while the
# student does not:
#   _SG_FRAME_SUB      → traj_frame_paths, read by the frozen TAS encoder (TRAIN
#                        only). Keep at "viz": stage1_tas_3way_overlay was trained
#                        on overlay frames, and retraining it is 100 epochs × 4 GPUs.
#   _SG_VLM_FRAME_SUB  → vlm_frame_paths, the student's actual input. Set
#                        VLM_GAZE_OVERLAY=0 for a genuinely gaze-free student.
# VLM_GAZE_OVERLAY defaults to GAZE_OVERLAY, so unset behaviour is unchanged.
#
# The gaze/hand/interaction paths below keep their own "viz" directory name —
# that is a layout convention for the trajectory data, not an overlay variant.
_GAZE_OVERLAY     = os.environ.get("GAZE_OVERLAY", "1")
_SG_FRAME_SUB     = "viz" if _GAZE_OVERLAY == "1" else "original"
_SG_VLM_FRAME_SUB = ("viz" if os.environ.get("VLM_GAZE_OVERLAY", _GAZE_OVERLAY) == "1"
                     else "original")


def _find_dataset(stem: str) -> Optional[str]:
    # Teacher variant: always present, so dataset resolution never depends on
    # whether the non-overlay frames have been extracted for this split.
    for ds in DATASETS:
        if os.path.isdir(os.path.join(FRAMES_BASE, ds, _SG_FRAME_SUB, stem)):
            return ds
    return None


def _get_frame_paths(stem: str, dataset: str, ts_sec: float,
                     sub: Optional[str] = None) -> list[str]:
    frame_dir = os.path.join(FRAMES_BASE, dataset, sub or _SG_FRAME_SUB, stem)
    if not os.path.isdir(frame_dir):
        return []
    cutoff = max(1, int(ts_sec * EXTRACTED_FPS))
    paths = []
    for fname in sorted(os.listdir(frame_dir)):
        m = re.match(r"frame_(\d+)\.jpg", fname)
        if m and int(m.group(1)) <= cutoff:
            paths.append(os.path.join(frame_dir, fname))
    return paths


def _sample_paths(paths: list[str], n: int) -> list[str]:
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    indices = [int(i * len(paths) / n) for i in range(n)]
    return [paths[i] for i in indices]


def _get_image_wh(path: str):
    from PIL import Image
    img = Image.open(path)
    return img.width, img.height


def _load_traj(stem: str, dataset: str, frame_paths: list[str],
               n_traj_frames: int) -> tuple[dict, list[str]]:
    """
    Load trajectory features for TrajGaze encoder.

    Returns:
        traj           : dict[str, Tensor]  — all feature tensors (T, ...)
        sampled_paths  : list[str]          — frame paths used (for visual encoder)
    """
    sampled = _sample_paths(frame_paths, n_traj_frames)
    T = len(sampled)
    if T == 0:
        return {}, []

    # --- gaze ---
    gaze_file = os.path.join(GAZE_BASE, dataset, "viz", f"{stem}.json")
    gaze_json: dict = {}
    if os.path.exists(gaze_file):
        with open(gaze_file) as f:
            gaze_json = json.load(f)

    # --- hand ---
    hand_file = os.path.join(HAND_BASE, dataset, "viz", f"{stem}.json")
    hand_json: dict = {}
    if os.path.exists(hand_file):
        with open(hand_file) as f:
            hand_json = json.load(f)

    # --- interaction NPZ ---
    npz_file = os.path.join(INTERACTION_BASE, dataset, "viz", f"{stem}.npz")
    npz = np.load(npz_file) if os.path.exists(npz_file) else None
    int_idx: dict[str, int] = {}
    if npz is not None:
        for i, fn in enumerate(npz["frame_names"]):
            int_idx[str(fn)] = int(i)

    # --- image size (for normalization) ---
    try:
        img_w, img_h = _get_image_wh(sampled[0])
    except Exception:
        img_w, img_h = 640, 480

    # --- allocate arrays ---
    gaze_pos   = np.zeros((T, 2), np.float32)
    gaze_speed = np.zeros((T, 1), np.float32)
    gaze_mask  = np.zeros(T, bool)
    left_pos   = np.zeros((T, 2), np.float32)
    left_vel   = np.zeros((T, 2), np.float32)
    left_mask  = np.zeros(T, bool)
    right_pos  = np.zeros((T, 2), np.float32)
    right_vel  = np.zeros((T, 2), np.float32)
    right_mask = np.zeros(T, bool)
    d_left     = np.zeros((T, 3), np.float32)
    d_right    = np.zeros((T, 3), np.float32)
    v_rel_left  = np.zeros((T, 2), np.float32)
    v_rel_right = np.zeros((T, 2), np.float32)
    convergence = np.zeros(T, np.float32)
    lead_lag    = np.zeros(T, np.float32)

    for t, fpath in enumerate(sampled):
        fname = os.path.basename(fpath)

        # gaze
        if fname in gaze_json:
            xy = gaze_json[fname]
            if xy and len(xy) == 2 and xy[0] is not None:
                gaze_pos[t, 0] = float(xy[0]) / img_w
                gaze_pos[t, 1] = float(xy[1]) / img_h
                gaze_mask[t] = True

        # hand
        if fname in hand_json:
            h = hand_json[fname]
            if isinstance(h, dict):
                lh = h.get("left")
                if lh and len(lh) == 2 and lh[0] is not None:
                    left_pos[t, 0] = float(lh[0]) / img_w
                    left_pos[t, 1] = float(lh[1]) / img_h
                    left_mask[t] = True
                rh = h.get("right")
                if rh and len(rh) == 2 and rh[0] is not None:
                    right_pos[t, 0] = float(rh[0]) / img_w
                    right_pos[t, 1] = float(rh[1]) / img_h
                    right_mask[t] = True

        # interaction features
        if npz is not None and fname in int_idx:
            ii = int_idx[fname]
            d_left[t]      = npz["d_left"][ii]
            d_right[t]     = npz["d_right"][ii]
            v_rel_left[t]  = npz["v_rel_left"][ii]
            v_rel_right[t] = npz["v_rel_right"][ii]
            convergence[t] = float(npz["convergence"][ii])
            lead_lag[t]    = float(npz["lead_lag"][ii])

    # gaze_speed from consecutive positions
    for t in range(1, T):
        if gaze_mask[t] and gaze_mask[t - 1]:
            gaze_speed[t, 0] = float(np.linalg.norm(gaze_pos[t] - gaze_pos[t - 1]))

    # hand velocity from consecutive positions
    for t in range(1, T):
        if left_mask[t] and left_mask[t - 1]:
            left_vel[t] = left_pos[t] - left_pos[t - 1]
        if right_mask[t] and right_mask[t - 1]:
            right_vel[t] = right_pos[t] - right_pos[t - 1]

    traj = {
        "gaze_pos":    torch.from_numpy(gaze_pos),
        "gaze_speed":  torch.from_numpy(gaze_speed),
        "gaze_mask":   torch.from_numpy(gaze_mask),
        "left_pos":    torch.from_numpy(left_pos),
        "left_vel":    torch.from_numpy(left_vel),
        "left_mask":   torch.from_numpy(left_mask),
        "right_pos":   torch.from_numpy(right_pos),
        "right_vel":   torch.from_numpy(right_vel),
        "right_mask":  torch.from_numpy(right_mask),
        "d_left":      torch.from_numpy(d_left),
        "d_right":     torch.from_numpy(d_right),
        "v_rel_left":  torch.from_numpy(v_rel_left),
        "v_rel_right": torch.from_numpy(v_rel_right),
        "convergence": torch.from_numpy(convergence),
        "lead_lag":    torch.from_numpy(lead_lag),
    }
    return traj, sampled


class StreamGazeMergeDataset(Dataset):
    """
    StreamGaze_v2 dataset filtered by dataset split.

    Args:
        split         : 'train' (egoexolearn + holoassist) or 'test' (egtea)
        n_vlm_frames  : frames to sample for VLM input
        n_traj_frames : frames to sample for TrajGaze encoder
        tasks         : list of task names; defaults to all 8 MCQ tasks
    """

    def __init__(
        self,
        split: str = "train",
        n_vlm_frames: int = 128,
        n_traj_frames: int = 32,
        tasks: Optional[list[str]] = None,
    ):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
        self.split         = split
        self.n_vlm_frames  = n_vlm_frames
        self.n_traj_frames = n_traj_frames
        self.filter_ds     = ["egoexolearn", "holoassist"] if split == "train" else ["egtea"]
        self.tasks_list    = tasks if tasks is not None else MCQ_TASKS

        self.items: list[dict] = []
        for task in self.tasks_list:
            qa_path = os.path.join(QA_BASE, f"{task}.json")
            if not os.path.exists(qa_path):
                continue
            with open(qa_path) as f:
                qa_data = json.load(f)
            for entry in qa_data:
                stem = os.path.splitext(entry["video_path"])[0]
                ds   = _find_dataset(stem)
                if ds not in self.filter_ds:
                    continue
                for q in entry.get("questions", []):
                    if not q.get("options"):
                        continue
                    self.items.append({
                        "stem":       stem,
                        "dataset":    ds,
                        "question":   q["question"],
                        "options":    q["options"],
                        "answer":     q["answer"].strip().upper(),
                        "time_stamp": q.get("time_stamp", ""),
                        "task":       task,
                    })

        ds_label = "+".join(self.filter_ds)
        print(f"[StreamGazeMergeDataset] split={split} ({ds_label}) "
              f"→ {len(self.items)} items across {len(self.tasks_list)} tasks")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[dict]:
        item = self.items[idx]
        ts_sec      = _parse_ts(item["time_stamp"])
        # Teacher stream — what the frozen TAS encoder reads (overlay by default).
        frame_paths = _get_frame_paths(item["stem"], item["dataset"], ts_sec)
        if not frame_paths:
            return None

        try:
            traj, traj_frame_paths = _load_traj(
                item["stem"], item["dataset"], frame_paths, self.n_traj_frames
            )
        except Exception:
            return None

        if not traj:
            return None

        # Student stream — the VLM's actual input, possibly a different frame
        # variant. Both directories hold identical per-video frame counts, so the
        # same timestamp cutoff and sampling indices keep the two streams aligned.
        if _SG_VLM_FRAME_SUB == _SG_FRAME_SUB:
            vlm_source = frame_paths                     # avoid a second listdir
        else:
            vlm_source = _get_frame_paths(item["stem"], item["dataset"], ts_sec,
                                          sub=_SG_VLM_FRAME_SUB)
            if not vlm_source:
                return None
        vlm_frame_paths = _sample_paths(vlm_source, self.n_vlm_frames)

        return {
            "vlm_frame_paths":  vlm_frame_paths,
            "full_frame_paths": vlm_source,
            "traj_frame_paths": traj_frame_paths,
            "traj":             traj,
            "question":         item["question"],
            "options":          item["options"],
            "answer":           item["answer"],
            "task":             item["task"],
            "dataset":          item["dataset"],
        }
