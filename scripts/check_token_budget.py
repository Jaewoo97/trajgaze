"""Sanity check: VZ and VZ-traj select the SAME number of dominant tokens.

Prints, for one item, the global dominant count for each method (must be equal)
plus a per-frame histogram showing the redistribution.
"""
from __future__ import annotations
import sys
import torch
sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.traj_weights import _solve_spatial_dims, traj_weighted_attn_scores
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, CONTEXTUAL_RATIO, load_visionzip_lora, preprocess_visionzip_item,
)

device = torch.device("cuda:0"); torch.cuda.set_device(0)
processor, model = load_visionzip_lora(device)
base_qwen = model.get_base_model()
ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True)

IDX = 1  # sg_idx1 (the flip)
item = ds[IDX]
cached = preprocess_visionzip_item(processor, base_qwen, item["vlm_frame_paths"],
                                   item["question"], item["options"], device)
attn = cached["attn_scores"]; grid = cached["grid_thw"]
N = cached["video_embeds"].shape[0]
T = int(grid[0, 0]); S = N // T
s_h, s_w = _solve_spatial_dims(S, int(grid[0, 1]), int(grid[0, 2]))
k_dom = max(1, int(DOMINANT_RATIO * N))
k_ctx = max(1, int(CONTEXTUAL_RATIO * N))

vz_idx = torch.topk(attn, k_dom).indices
w = traj_weighted_attn_scores(attn, grid, item["traj"], device)
tr_idx = torch.topk(w, k_dom).indices

print(f"\nitem idx={IDX}  N={N} tokens  T={T} frames  S={S}/frame  grid=({s_h}x{s_w})")
print(f"budget: dominant={k_dom} ({DOMINANT_RATIO*100:.0f}%) + "
      f"contextual={k_ctx} ({CONTEXTUAL_RATIO*100:.0f}%) = {k_dom+k_ctx} ({(k_dom+k_ctx)/N*100:.1f}%)")
print(f"\nGLOBAL dominant selected:  VZ={vz_idx.numel()}   VZ-traj={tr_idx.numel()}   "
      f"(equal? {vz_idx.numel()==tr_idx.numel()})")

vz_per = torch.bincount(vz_idx // S, minlength=T)
tr_per = torch.bincount(tr_idx // S, minlength=T)
print(f"\nper-frame dominant counts (top 12 frames by |traj-vz|):")
diff = (tr_per - vz_per).abs()
order = torch.argsort(diff, descending=True)[:12].tolist()
print("  frame :  VZ  traj  (delta)")
for t in sorted(order):
    print(f"   t={t:3d} : {vz_per[t].item():3d}  {tr_per[t].item():3d}   ({tr_per[t].item()-vz_per[t].item():+d})")
print(f"\nsum over ALL frames:  VZ={vz_per.sum().item()}  traj={tr_per.sum().item()}  (identical)")
