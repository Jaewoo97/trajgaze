"""
Temporal Stage 1 dataset for TrajGazeV2Temporal.

Built from StreamGaze_v2 clips (egoexolearn + holoassist).

Key differences from StreamGazeStage1Dataset:
  - Returns I_scores_past (T_past, 196) for encoder supervision (NEW).
  - Returns I_scores_future (T_future, 196) for decoder supervision.
  - Returns frame_paths for ALL T frames (past + future) so the visual
    encoder can produce per-frame DINOv2 features at the correct temporal
    resolution.
  - n_frames defaults to 128 to match Qwen2.5-VL.
  - Past/future split is still random 40–60% each iteration.

Collate:
  collate_stage1_temporal handles variable T_past / T_future across batch items
  by zero-padding all variable-length tensors.
"""

from __future__ import annotations

import json
import os
import re
import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from TrajGazeMerge.data.dataset import (
    FRAMES_BASE, QA_BASE, MCQ_TASKS, _find_dataset, _sample_paths,
)
from TrajGaze_v2.data.dataset import _traj_to_tensors
from TrajGaze_v2.data.interaction import compute_importance_scores, compute_traj_features
from TrajGaze_v2.data.dataset_streamgaze_stage1 import _load_raw_positions

T_SAMPLE      = 128
PAST_RATIO_LO = 0.4
PAST_RATIO_HI = 0.6
TRAIN_DATASETS = ("egoexolearn", "holoassist")


class StreamGazeStage1DatasetTemporal(Dataset):
    """
    Temporal Stage 1 dataset on StreamGaze_v2 clips.

    Each item provides:
        past            : trajectory tensors for T_past frames
        future          : trajectory tensors for T_future frames
        I_scores_past   : (T_past,   196) GT interaction scores for encoder supervision
        I_scores_future : (T_future, 196) GT interaction scores for decoder supervision
        T_past          : int
        T_future        : int
        frame_paths     : list[str] of all T frame paths (past + future)
    """

    def __init__(
        self,
        n_frames:  int   = T_SAMPLE,
        past_lo:   float = PAST_RATIO_LO,
        past_hi:   float = PAST_RATIO_HI,
    ):
        self.n_frames = n_frames
        self.past_lo  = past_lo
        self.past_hi  = past_hi

        seen: set[tuple] = set()
        self.clips: list[dict] = []

        for task in MCQ_TASKS:
            qa_path = os.path.join(QA_BASE, f"{task}.json")
            if not os.path.exists(qa_path):
                continue
            with open(qa_path) as f:
                qa_data = json.load(f)
            for entry in qa_data:
                stem = os.path.splitext(entry["video_path"])[0]
                ds   = _find_dataset(stem)
                if ds not in TRAIN_DATASETS:
                    continue
                # Pick the first non-empty question for this clip+task as a
                # sample-specific text signal for the QueryEncoder. Fallback
                # to a task-name prompt so query_emb is never sample-agnostic.
                first_q = ""
                for q in entry.get("questions", []):
                    if q.get("question"):
                        first_q = q["question"]
                        break
                if not first_q:
                    first_q = f"What is happening in this {task.replace('_', ' ')} clip?"
                if (stem, ds) not in seen:
                    seen.add((stem, ds))
                    self.clips.append({"stem": stem, "dataset": ds, "question": first_q})

        print(f"[StreamGazeStage1DatasetTemporal] {len(self.clips)} unique clips "
              f"(egoexolearn+holoassist, T_sample={n_frames})")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Optional[dict]:
        clip = self.clips[idx]
        stem = clip["stem"]
        ds   = clip["dataset"]

        frame_dir = os.path.join(FRAMES_BASE, ds, "viz", stem)
        if not os.path.isdir(frame_dir):
            return None

        all_frames = sorted([
            os.path.join(frame_dir, fname)
            for fname in os.listdir(frame_dir)
            if re.match(r"frame_\d+\.jpg", fname)
        ])
        if len(all_frames) < 4:
            return None

        sampled = _sample_paths(all_frames, self.n_frames)
        T = len(sampled)
        if T < 4:
            return None

        try:
            gaze_pos_list, left_pos_list, right_pos_list = _load_raw_positions(
                stem, ds, sampled
            )
        except Exception:
            return None

        traj     = compute_traj_features(gaze_pos_list, left_pos_list, right_pos_list)
        I_scores = compute_importance_scores(gaze_pos_list, left_pos_list, right_pos_list)

        tensors    = _traj_to_tensors(traj)
        I_scores_t = torch.from_numpy(I_scores).float()  # (T, 196)

        ratio  = random.uniform(self.past_lo, self.past_hi)
        T_past = max(2, min(T - 2, round(T * ratio)))

        past = {k: v[:T_past] for k, v in tensors.items()}
        future = {
            "gaze_pos":   tensors["gaze_pos"][T_past:],
            "gaze_mask":  tensors["gaze_mask"][T_past:],
            "left_pos":   tensors["left_pos"][T_past:],
            "left_mask":  tensors["left_mask"][T_past:],
            "right_pos":  tensors["right_pos"][T_past:],
            "right_mask": tensors["right_mask"][T_past:],
        }

        return {
            "past":            past,
            "future":          future,
            "I_scores_past":   I_scores_t[:T_past],       # (T_past,   196)
            "I_scores_future": I_scores_t[T_past:],       # (T_future, 196)
            "T_past":          T_past,
            "T_future":        T - T_past,
            "frame_paths":     sampled,                   # all T paths
            "question":        clip.get("question", ""),  # sample-specific text for QueryEncoder
        }


def collate_stage1_temporal(batch: list) -> Optional[dict]:
    """
    Collate variable-length items into batched tensors.

    Pads past/future sequences and score maps to the max lengths in the batch.
    frame_paths is passed through as a list[list[str]] (cannot be tensorized).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    T_past_max   = max(b["T_past"]   for b in batch)
    T_future_max = max(b["T_future"] for b in batch)

    def pad_traj(tensors_list, T_max):
        result = {}
        for key in tensors_list[0].keys():
            chunks = []
            for t in tensors_list:
                arr  = t[key]                     # (T_i, ...)
                T_i  = arr.shape[0]
                if T_i < T_max:
                    pad = torch.zeros(T_max - T_i, *arr.shape[1:], dtype=arr.dtype)
                    arr = torch.cat([arr, pad], dim=0)
                chunks.append(arr.unsqueeze(0))
            result[key] = torch.cat(chunks, dim=0)  # (B, T_max, ...)
        return result

    def pad_scores(scores_list, T_max):
        chunks = []
        for arr in scores_list:
            T_i = arr.shape[0]
            if T_i < T_max:
                pad = torch.zeros(T_max - T_i, 196)
                arr = torch.cat([arr, pad], dim=0)
            chunks.append(arr.unsqueeze(0))
        return torch.cat(chunks, dim=0)  # (B, T_max, 196)

    past_tensors   = pad_traj([b["past"]   for b in batch], T_past_max)
    future_tensors = pad_traj([b["future"] for b in batch], T_future_max)

    I_scores_past   = pad_scores([b["I_scores_past"]   for b in batch], T_past_max)
    I_scores_future = pad_scores([b["I_scores_future"] for b in batch], T_future_max)

    return {
        "past":            past_tensors,
        "future":          future_tensors,
        "I_scores_past":   I_scores_past,
        "I_scores_future": I_scores_future,
        "T_past":          torch.tensor([b["T_past"]   for b in batch], dtype=torch.long),
        "T_future":        torch.tensor([b["T_future"] for b in batch], dtype=torch.long),
        "frame_paths":     [b["frame_paths"] for b in batch],
        "questions":       [b.get("question", "") for b in batch],
    }


__all__ = [
    "StreamGazeStage1DatasetTemporal",
    "collate_stage1_temporal",
]
