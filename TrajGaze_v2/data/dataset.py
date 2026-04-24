"""
TrajGaze_v2 Dataset.

Returns per-clip trajectory features + interaction score ground truth.
Used for Stage 1 training (ego4d + egoexo) and Stage 2 (all datasets with QA).

Stage 1 item:
    traj features (T_sample, ...) + I_scores GT (T_sample, 196)
    split_idx: random integer in [T_past_min, T_past_max)

Stage 2 item (QADataset):
    same trajectory features + question string + correct_answer letter
"""

from __future__ import annotations

import csv
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .loader import (
    list_clips, get_clip_frame_paths, get_image_size,
    load_gaze_csv, load_hand_json, normalize_hand_coords,
    get_frame_num_from_path,
)
from .interaction import compute_importance_scores, compute_traj_features

DATASET_ROOT  = "/workspace/datasets/EgoGazeVQA"
T_SAMPLE      = 32      # frames sampled per clip
PAST_RATIO_LO = 0.4
PAST_RATIO_HI = 0.6


def _sample_indices(n_total: int, n_sample: int) -> list[int]:
    """Uniformly sample n_sample indices from [0, n_total)."""
    if n_total <= n_sample:
        return list(range(n_total))
    step = n_total / n_sample
    return [int(i * step) for i in range(n_sample)]


def _load_clip_traj(clip: dict, n_frames: int = T_SAMPLE) -> Optional[dict]:
    """
    Load and process gaze + hand trajectory for one clip.

    Returns dict with traj features and I_scores, or None on failure.
    """
    # Frame paths
    frame_paths = get_clip_frame_paths(clip["frame_dir"], clip["clip_stem"])
    if len(frame_paths) < 4:
        return None

    # Sample frames
    indices = _sample_indices(len(frame_paths), n_frames)
    frame_paths = [frame_paths[i] for i in indices]
    T = len(frame_paths)

    # Gaze: CSV keyed by original video frame number
    # Image filenames encode the frame number: e.g. "12668_13163_12671.jpg" → 12671
    gaze_by_fnum = load_gaze_csv(clip["gaze_csv"])
    gaze_sel: list[Optional[tuple[float, float]]] = []
    for fp in frame_paths:
        fnum = get_frame_num_from_path(fp)
        gaze_sel.append(gaze_by_fnum.get(fnum))

    # Hand: JSON keyed by "{clip_stem}_{frame_num}.jpg"
    hand_all = load_hand_json(clip["hand_json"])

    # Get image size for hand normalization (from first frame)
    try:
        img_w, img_h = get_image_size(frame_paths[0])
    except Exception:
        img_w, img_h = 1920, 1080  # fallback

    left_sel:  list[Optional[tuple[float, float]]] = []
    right_sel: list[Optional[tuple[float, float]]] = []
    for fp in frame_paths:
        key = os.path.basename(fp)
        entry = hand_all.get(key, {})
        left_sel.append(normalize_hand_coords(entry.get("left"),  img_w, img_h))
        right_sel.append(normalize_hand_coords(entry.get("right"), img_w, img_h))

    # Compute trajectory features
    traj = compute_traj_features(gaze_sel, left_sel, right_sel)

    # Compute interaction scores GT
    I_scores = compute_importance_scores(gaze_sel, left_sel, right_sel)  # (T, 196)

    return {
        "frame_paths": frame_paths,
        "traj":        traj,
        "I_scores":    I_scores,     # (T, 196) float32
    }


def _traj_to_tensors(traj: dict) -> dict[str, torch.Tensor]:
    """Convert numpy traj features to float32 tensors."""
    return {
        "gaze_pos":    torch.from_numpy(traj["gaze_pos"]).float(),     # (T, 2)
        "gaze_speed":  torch.from_numpy(traj["gaze_speed"]).float(),   # (T, 1)
        "gaze_mask":   torch.from_numpy(traj["gaze_mask"]),            # (T,) bool
        "left_pos":    torch.from_numpy(traj["left_pos"]).float(),     # (T, 2)
        "left_vel":    torch.from_numpy(traj["left_vel"]).float(),     # (T, 2)
        "left_mask":   torch.from_numpy(traj["left_mask"]),            # (T,) bool
        "right_pos":   torch.from_numpy(traj["right_pos"]).float(),    # (T, 2)
        "right_vel":   torch.from_numpy(traj["right_vel"]).float(),    # (T, 2)
        "right_mask":  torch.from_numpy(traj["right_mask"]),           # (T,) bool
        "d_left":      torch.from_numpy(traj["d_left"]).float(),       # (T, 3)
        "d_right":     torch.from_numpy(traj["d_right"]).float(),      # (T, 3)
        "v_rel_left":  torch.from_numpy(traj["v_rel_left"]).float(),   # (T, 2)
        "v_rel_right": torch.from_numpy(traj["v_rel_right"]).float(),  # (T, 2)
        "convergence": torch.from_numpy(traj["convergence"]).float(),  # (T,)
        "lead_lag":    torch.from_numpy(traj["lead_lag"]).float(),     # (T,)
    }


# ── Stage 1 Dataset ────────────────────────────────────────────────────────────

class TrajGazeV2Stage1Dataset(Dataset):
    """
    Stage 1 dataset: trajectory prediction from past → future.

    Returns:
        past_*: trajectory features for frames [0:T_past]
        future_*: ground truth for frames [T_past:T]
        T_past: int
        I_scores_future: (T_future, 196) float32
    """

    def __init__(
        self,
        datasets: list[str] = ("ego4d", "egoexo"),
        dataset_root: str = DATASET_ROOT,
        n_frames: int = T_SAMPLE,
        past_lo: float = PAST_RATIO_LO,
        past_hi: float = PAST_RATIO_HI,
        cache: bool = True,
    ):
        self.n_frames  = n_frames
        self.past_lo   = past_lo
        self.past_hi   = past_hi
        self.cache     = cache
        self._cache: dict[int, Optional[dict]] = {}

        self.clips = []
        for ds in datasets:
            self.clips.extend(list_clips(dataset_root, ds))

        print(f"[TrajGazeV2Stage1Dataset] Found {len(self.clips)} clips "
              f"from {datasets} (T_sample={n_frames})")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        # Load (with optional caching)
        if self.cache and idx in self._cache:
            data = self._cache[idx]
        else:
            data = _load_clip_traj(self.clips[idx], self.n_frames)
            if self.cache:
                self._cache[idx] = data

        if data is None:
            # Return a dummy item; collate_fn will filter these out
            return None

        T = len(data["frame_paths"])
        if T < 4:
            return None

        # Random past/future split
        ratio  = random.uniform(self.past_lo, self.past_hi)
        T_past = max(2, min(T - 2, round(T * ratio)))

        tensors = _traj_to_tensors(data["traj"])
        I_scores = torch.from_numpy(data["I_scores"]).float()  # (T, 196)

        # Split into past and future
        past = {k: v[:T_past] for k, v in tensors.items()}
        future = {
            "gaze_pos":    tensors["gaze_pos"][T_past:],
            "gaze_mask":   tensors["gaze_mask"][T_past:],
            "left_pos":    tensors["left_pos"][T_past:],
            "left_mask":   tensors["left_mask"][T_past:],
            "right_pos":   tensors["right_pos"][T_past:],
            "right_mask":  tensors["right_mask"][T_past:],
        }
        I_scores_future = I_scores[T_past:]   # (T_future, 196) — Stage 1 prediction target

        return {
            "past":            past,
            "future":          future,
            "I_scores_future": I_scores_future,
            "T_past":          T_past,
            "T_future":        T - T_past,
            "frame_paths":     data["frame_paths"],   # list[str] — for visual encoder
        }


def collate_stage1(batch: list) -> Optional[dict]:
    """
    Collate variable-length clips into batches.
    Pads past/future sequences to max length in batch.
    Skips None items.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    B = len(batch)
    T_past_max   = max(b["T_past"]   for b in batch)
    T_future_max = max(b["T_future"] for b in batch)

    def pad_seq(tensors_list, T_max, dim0_key=None):
        """Stack sequence tensors with zero-padding along time dim."""
        result = {}
        for key in tensors_list[0].keys():
            chunks = []
            for t in tensors_list:
                arr = t[key]   # (T_i, ...)
                T_i = arr.shape[0]
                if T_i < T_max:
                    pad_shape = list(arr.shape)
                    pad_shape[0] = T_max - T_i
                    pad = torch.zeros(pad_shape, dtype=arr.dtype)
                    arr = torch.cat([arr, pad], dim=0)
                chunks.append(arr.unsqueeze(0))
            result[key] = torch.cat(chunks, dim=0)  # (B, T_max, ...)
        return result

    past_tensors   = pad_seq([b["past"]   for b in batch], T_past_max)
    future_tensors = pad_seq([b["future"] for b in batch], T_future_max)

    # I_scores_future: Stage 1 ScoreDecoder prediction target
    I_chunks = []
    for b in batch:
        arr = b["I_scores_future"]  # (T_f, 196)
        T_f = arr.shape[0]
        if T_f < T_future_max:
            pad = torch.zeros(T_future_max - T_f, 196)
            arr = torch.cat([arr, pad], dim=0)
        I_chunks.append(arr.unsqueeze(0))
    I_scores_future = torch.cat(I_chunks, dim=0)  # (B, T_future_max, 196)

    T_past_tensor   = torch.tensor([b["T_past"]   for b in batch], dtype=torch.long)
    T_future_tensor = torch.tensor([b["T_future"] for b in batch], dtype=torch.long)

    # frame_paths: list of B lists of strings (cannot be tensorized — pass through as-is)
    frame_paths_batch = [b["frame_paths"] for b in batch]

    return {
        "past":            past_tensors,
        "future":          future_tensors,
        "I_scores_future": I_scores_future,
        "T_past":          T_past_tensor,
        "T_future":        T_future_tensor,
        "frame_paths":     frame_paths_batch,   # list[list[str]] — for visual encoder
    }


# ── Stage 2 QA Dataset ─────────────────────────────────────────────────────────

class TrajGazeV2QADataset(Dataset):
    """
    QA dataset for Stage 2 GRPO.

    Each item: full clip trajectory + question text + correct answer.
    Loads from metadata.csv, supports filtering by dataset.
    """

    def __init__(
        self,
        datasets: list[str] = ("ego4d", "egoexo", "egtea"),
        dataset_root: str = DATASET_ROOT,
        metadata_csv: str = None,
        n_frames: int = T_SAMPLE,
    ):
        self.n_frames     = n_frames
        self.dataset_root = dataset_root

        if metadata_csv is None:
            metadata_csv = os.path.join(dataset_root, "metadata.csv")

        # Build clip lookup: (dataset, video_id, clip_stem) -> clip dict
        clip_lookup: dict[tuple, dict] = {}
        for ds in datasets:
            for clip in list_clips(dataset_root, ds):
                key = (clip["dataset"], clip["video_id"], clip["clip_stem"])
                clip_lookup[key] = clip

        # Load QA from metadata.csv
        self.items = []
        with open(metadata_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row.get("dataset", "")
                if ds not in datasets:
                    continue
                video_id = row["video_id"]
                # file_name format: "ego4d/{video_id}/{start}_{end}.mp4"
                fname = row["file_name"]
                try:
                    clip_stem = os.path.splitext(os.path.basename(fname))[0]
                except Exception:
                    continue
                key = (ds, video_id, clip_stem)
                clip = clip_lookup.get(key)
                if clip is None:
                    continue

                # Parse options: "A: ...|B: ...|C: ...|D: ...|E: ..."
                options_raw = row.get("answer_options", "")
                options = [o.strip() for o in options_raw.split("|") if o.strip()]

                self.items.append({
                    "clip":    clip,
                    "qa_type": row.get("qa_type", ""),
                    "question": row.get("question", ""),
                    "options":  options,
                    "answer":   row.get("correct_answer", "").strip(),
                })

        print(f"[TrajGazeV2QADataset] Found {len(self.items)} QA items from {datasets}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        data = _load_clip_traj(item["clip"], self.n_frames)
        if data is None:
            return None

        tensors = _traj_to_tensors(data["traj"])
        I_scores = torch.from_numpy(data["I_scores"]).float()

        return {
            "traj":        tensors,
            "I_scores":    I_scores,
            "question":    item["question"],
            "options":     item["options"],
            "answer":      item["answer"],
            "qa_type":     item["qa_type"],
            "frame_paths": data["frame_paths"],
        }
