"""
EgoGazeVQA dataset for TrajGazeMerge — same per-item contract as
StreamGazeMergeDataset (frames + traj + MCQ), so it can be concatenated
with StreamGaze_v2 for joint training.

Split:
    split='train' → dataset in {ego4d, egoexo}
    split='test'  → dataset == egtea

Source:
    /workspace/EgoGazeVQA/metadata.csv
        file_name,video_id,dataset,qa_type,question,answer_options,correct_answer
        answer_options : "A: ...|B: ...|...|E: ..."  (5-option MCQ, A–E)
    frames      : {ds}/no_gaze/{video_id}/{subclip}_{gfn}.jpg
    gaze        : {ds}/gaze_mapping/{video_id}/{subclip}_mapping.csv  (gaze_x/y ∈ [0,1])
    hand        : {ds}/hand_locations/{video_id}.json                  (pixel coords)
    interaction : interaction/{ds}/{video_id}/{subclip}.npz            (generated, StreamGaze schema)

`subclip` = stem of the mp4 in file_name (e.g. "123_1205").
`task`    = qa_type (causal / spatial / temporal).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from TrajGazeMerge.data.dataset import _sample_paths, _get_image_wh

ROOT = "/workspace/EgoGazeVQA"
METADATA = os.path.join(ROOT, "metadata.csv")


def _parse_options(raw: str) -> list[str]:
    """'A: foo|B: bar|...' → ['A. foo', 'B. bar', ...] (StreamGaze option style)."""
    out = []
    for part in raw.split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        letter, text = part.split(":", 1)
        out.append(f"{letter.strip()}. {text.strip()}")
    return out


def _frame_gfn(fname: str, subclip: str) -> Optional[int]:
    # "{subclip}_{gfn}.jpg" → gfn int
    base = os.path.splitext(fname)[0]
    if not base.startswith(subclip + "_"):
        return None
    try:
        return int(base[len(subclip) + 1:])
    except ValueError:
        return None


def _load_traj_ego(ds: str, video_id: str, subclip: str,
                   frame_paths: list[str], n_traj_frames: int) -> tuple[dict, list[str]]:
    """
    Build the same traj dict as StreamGaze _load_traj, for an EgoGazeVQA subclip.

    Positions/velocities/masks computed on the fly; d_left/d_right/v_rel_*/
    convergence/lead_lag read from the pre-generated interaction npz keyed
    by frame basename — exactly mirroring StreamGaze's loader.
    """
    sampled = _sample_paths(frame_paths, n_traj_frames)
    T = len(sampled)
    if T == 0:
        return {}, []

    # gaze: gaze_mapping CSV → {gaze_frame_num: (x, y)} already normalized [0,1]
    gaze_csv = os.path.join(ROOT, ds, "gaze_mapping", video_id, f"{subclip}_mapping.csv")
    gaze_by_gfn: dict[int, tuple] = {}
    if os.path.exists(gaze_csv):
        for r in csv.DictReader(open(gaze_csv)):
            try:
                gx, gy = float(r["gaze_x"]), float(r["gaze_y"])
                if np.isfinite(gx) and np.isfinite(gy):
                    gaze_by_gfn[int(r["gaze_frame_num"])] = (gx, gy)
            except (TypeError, ValueError):
                pass

    # hand: one json per recording, keyed by "{subclip}_{gfn}.jpg", pixel coords
    hand_path = os.path.join(ROOT, ds, "hand_locations", f"{video_id}.json")
    hand_json: dict = {}
    if os.path.exists(hand_path):
        try:
            hand_json = json.load(open(hand_path))
        except Exception:
            hand_json = {}

    # interaction npz (StreamGaze schema) keyed by frame basename
    npz_path = os.path.join(ROOT, "interaction", ds, video_id, f"{subclip}.npz")
    npz = np.load(npz_path) if os.path.exists(npz_path) else None
    int_idx: dict[str, int] = {}
    if npz is not None:
        for i, fn in enumerate(npz["frame_names"]):
            int_idx[str(fn)] = int(i)

    try:
        img_w, img_h = _get_image_wh(sampled[0])
    except Exception:
        img_w, img_h = 640, 480

    gaze_pos   = np.zeros((T, 2), np.float32)
    gaze_speed = np.zeros((T, 1), np.float32)
    gaze_mask  = np.zeros(T, bool)
    left_pos   = np.zeros((T, 2), np.float32)
    left_vel   = np.zeros((T, 2), np.float32)
    left_mask  = np.zeros(T, bool)
    right_pos  = np.zeros((T, 2), np.float32)
    right_vel  = np.zeros((T, 2), np.float32)
    right_mask = np.zeros(T, bool)
    d_left      = np.zeros((T, 3), np.float32)
    d_right     = np.zeros((T, 3), np.float32)
    v_rel_left  = np.zeros((T, 2), np.float32)
    v_rel_right = np.zeros((T, 2), np.float32)
    convergence = np.zeros(T, np.float32)
    lead_lag    = np.zeros(T, np.float32)

    for t, fpath in enumerate(sampled):
        fname = os.path.basename(fpath)
        gfn = _frame_gfn(fname, subclip)

        if gfn is not None and gfn in gaze_by_gfn:
            gx, gy = gaze_by_gfn[gfn]          # already [0,1]
            gaze_pos[t, 0] = gx
            gaze_pos[t, 1] = gy
            gaze_mask[t] = True

        h = hand_json.get(fname)
        if isinstance(h, dict):
            lh = h.get("left")
            if lh and lh[0] is not None:
                left_pos[t, 0] = float(lh[0]) / img_w
                left_pos[t, 1] = float(lh[1]) / img_h
                left_mask[t] = True
            rh = h.get("right")
            if rh and rh[0] is not None:
                right_pos[t, 0] = float(rh[0]) / img_w
                right_pos[t, 1] = float(rh[1]) / img_h
                right_mask[t] = True

        if npz is not None and fname in int_idx:
            ii = int_idx[fname]
            d_left[t]      = npz["d_left"][ii]
            d_right[t]     = npz["d_right"][ii]
            v_rel_left[t]  = npz["v_rel_left"][ii]
            v_rel_right[t] = npz["v_rel_right"][ii]
            convergence[t] = float(npz["convergence"][ii])
            lead_lag[t]    = float(npz["lead_lag"][ii])

    for t in range(1, T):
        if gaze_mask[t] and gaze_mask[t - 1]:
            gaze_speed[t, 0] = float(np.linalg.norm(gaze_pos[t] - gaze_pos[t - 1]))
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


class EgoGazeVQADataset(Dataset):
    """EgoGazeVQA, same per-item contract as StreamGazeMergeDataset."""

    TRAIN_DS = ["ego4d", "egoexo"]
    VAL_DS   = ["egtea"]

    def __init__(
        self,
        split: str = "train",
        n_vlm_frames: int = 128,
        n_traj_frames: int = 32,
        tasks: Optional[list[str]] = None,
    ):
        assert split in ("train", "test")
        self.split         = split
        self.n_vlm_frames  = n_vlm_frames
        self.n_traj_frames = n_traj_frames
        self.filter_ds     = self.TRAIN_DS if split == "train" else self.VAL_DS
        self.tasks_list    = tasks  # None = all qa_types

        self.items: list[dict] = []
        with open(METADATA) as f:
            for r in csv.DictReader(f):
                ds = r["dataset"]
                if ds not in self.filter_ds:
                    continue
                if self.tasks_list is not None and r["qa_type"] not in self.tasks_list:
                    continue
                opts = _parse_options(r["answer_options"])
                ans  = r["correct_answer"].strip().upper()
                if not opts or len(ans) != 1:
                    continue
                letters = [chr(65 + i) for i in range(len(opts))]
                if ans not in letters:
                    continue
                subclip = os.path.splitext(os.path.basename(r["file_name"]))[0]
                self.items.append({
                    "ds":       ds,
                    "video_id": r["video_id"],
                    "subclip":  subclip,
                    "question": r["question"],
                    "options":  opts,
                    "answer":   ans,
                    "task":     r["qa_type"],
                })

        ds_label = "+".join(self.filter_ds)
        print(f"[EgoGazeVQADataset] split={split} ({ds_label}) "
              f"→ {len(self.items)} items")

    def __len__(self) -> int:
        return len(self.items)

    def _frame_paths(self, ds: str, video_id: str, subclip: str) -> list[str]:
        d = os.path.join(ROOT, ds, "no_gaze", video_id)
        if not os.path.isdir(d):
            return []
        prefix = subclip + "_"
        pairs = []
        for fn in os.listdir(d):
            if fn.startswith(prefix) and fn.endswith(".jpg"):
                g = _frame_gfn(fn, subclip)
                if g is not None:
                    pairs.append((g, os.path.join(d, fn)))
        pairs.sort(key=lambda x: x[0])
        return [p for _, p in pairs]

    def __getitem__(self, idx: int) -> Optional[dict]:
        it = self.items[idx]
        frame_paths = self._frame_paths(it["ds"], it["video_id"], it["subclip"])
        if not frame_paths:
            return None
        try:
            traj, traj_frame_paths = _load_traj_ego(
                it["ds"], it["video_id"], it["subclip"],
                frame_paths, self.n_traj_frames,
            )
        except Exception:
            return None
        if not traj:
            return None

        return {
            "vlm_frame_paths":  _sample_paths(frame_paths, self.n_vlm_frames),
            "traj_frame_paths": traj_frame_paths,
            "traj":             traj,
            "question":         it["question"],
            "options":          it["options"],
            "answer":           it["answer"],
            "task":             it["task"],
            "dataset":          it["ds"],
        }
