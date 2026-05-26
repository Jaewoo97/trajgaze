"""
CombinedSimpleDataset = StreamGaze_v2 ∪ EgoGazeVQA, frames + QA only (no traj).
Same egtea=val / rest=train split for both sources. Used by the VisionZip
trainer (token selection is ViT-attention-based, no trajectory needed).

Per-item: {vlm_frame_paths, question, options, answer, task, dataset}
`options` keep the "A. " letter prefix (identical to StreamGazeSimpleDataset
and to what baseline/merge consumed) so the prompt format is identical
across all conditions.

Exposes flat sliceable `.items` of (src, local_idx) like the other combined ds.
"""
from __future__ import annotations

import os
from typing import Optional

from torch.utils.data import Dataset

from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
from TrajGazeMerge.data.egogaze_dataset import (
    EgoGazeVQADataset, _parse_options, _frame_gfn, ROOT,
)
from TrajGazeMerge.data.hdepic_dataset import HDEpicDataset


class EgoGazeSimpleDataset(Dataset):
    """EgoGazeVQA frames + QA only (no traj); same split + options as the full adapter."""

    def __init__(self, split: str = "train", n_vlm_frames: int = 128):
        # reuse the full adapter only for its parsed item list (metadata.csv)
        meta = EgoGazeVQADataset(split=split, n_vlm_frames=n_vlm_frames,
                                 n_traj_frames=8)
        self.items = meta.items
        self.n_vlm_frames = n_vlm_frames
        from TrajGazeMerge.data.dataset import _sample_paths
        self._sample = _sample_paths

    def __len__(self) -> int:
        return len(self.items)

    def _frame_paths(self, ds, video_id, subclip):
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
        fp = self._frame_paths(it["ds"], it["video_id"], it["subclip"])
        if not fp:
            return None
        vlm = self._sample(fp, self.n_vlm_frames)
        if not vlm:
            return None
        return {
            "vlm_frame_paths": vlm,
            "question":        it["question"],
            "options":         it["options"],   # already "A. ..." prefixed
            "answer":          it["answer"],
            "task":            it["task"],
            "dataset":         it["ds"],
        }


class CombinedSimpleDataset(Dataset):
    def __init__(self, split: str = "train", n_vlm_frames: int = 128,
                 include_hdepic: bool = True):
        self.sg = StreamGazeSimpleDataset(split=split, n_vlm_frames=n_vlm_frames)
        self.eg = EgoGazeSimpleDataset(split=split, n_vlm_frames=n_vlm_frames)
        self._src = {"sg": self.sg, "eg": self.eg}
        self.items = ([("sg", i) for i in range(len(self.sg))] +
                      [("eg", i) for i in range(len(self.eg))])
        if include_hdepic:
            self.hd = HDEpicDataset(split=split, n_vlm_frames=n_vlm_frames, simple=True)
            self._src["hd"] = self.hd
            self.items += [("hd", i) for i in range(len(self.hd))]
        parts = ", ".join(f"{k}={len(self._src[k])}" for k in self._src)
        print(f"[CombinedSimpleDataset] split={split} → {len(self.items)} items ({parts})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[dict]:
        src, local = self.items[idx]
        return self._src[src][local]
