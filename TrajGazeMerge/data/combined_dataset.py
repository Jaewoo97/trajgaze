"""
CombinedMergeDataset = StreamGaze_v2 ∪ EgoGazeVQA, with the same per-item
contract and the same egtea=val / rest=train split for both sources.

    split='train' → StreamGaze(egoexolearn+holoassist) + EgoGazeVQA(ego4d+egoexo)
    split='test'  → StreamGaze(egtea)                    + EgoGazeVQA(egtea)

Exposes a flat `.items` list of (src, local_idx) so callers can slice it
(e.g. evaluate(): `ds.items = ds.items[:max_items]`) exactly like the
single-source datasets.
"""
from __future__ import annotations

from typing import Optional

from torch.utils.data import Dataset

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.data.egogaze_dataset import EgoGazeVQADataset
from TrajGazeMerge.data.hdepic_dataset import HDEpicDataset


class CombinedMergeDataset(Dataset):
    """StreamGaze ∪ EgoGazeVQA ∪ HD-EPIC, same per-item contract.

    Train  = StreamGaze(egoexolearn+holoassist) + EgoGazeVQA(ego4d+egoexo) + HD-EPIC(non-P09)
    Val    = StreamGaze(egtea) + EgoGazeVQA(egtea) + HD-EPIC(P09)

    `include_hdepic=False` reproduces the older 2-source dataset for backward compat.
    """

    def __init__(
        self,
        split: str = "train",
        n_vlm_frames: int = 128,
        n_traj_frames: int = 32,
        include_hdepic: bool = True,
    ):
        self.sg = StreamGazeMergeDataset(
            split=split, n_vlm_frames=n_vlm_frames, n_traj_frames=n_traj_frames
        )
        self.eg = EgoGazeVQADataset(
            split=split, n_vlm_frames=n_vlm_frames, n_traj_frames=n_traj_frames
        )
        self._src = {"sg": self.sg, "eg": self.eg}
        self.items = ([("sg", i) for i in range(len(self.sg))] +
                      [("eg", i) for i in range(len(self.eg))])

        if include_hdepic:
            self.hd = HDEpicDataset(
                split=split, n_vlm_frames=n_vlm_frames,
                n_traj_frames=n_traj_frames, simple=False,
            )
            self._src["hd"] = self.hd
            self.items += [("hd", i) for i in range(len(self.hd))]

        parts = ", ".join(f"{k}={len(self._src[k])}" for k in self._src)
        print(f"[CombinedMergeDataset] split={split} → {len(self.items)} items ({parts})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[dict]:
        src, local = self.items[idx]
        return self._src[src][local]
