"""Step 0 for the ViT selection-distillation experiment — measure before committing GPU days.

Four things, in order of how badly getting them wrong would hurt:

  1. REGRESSION. The refactor of the score loop in qwen2_5vl_visionzip.py must be
     arithmetically identical on the default (no-grad, full-query) path, or every
     baseline in docs/kd_handoff_v2.md silently stops applying. Checked two ways:
     the original loop reimplemented here vs. the shipped one on the same tensors,
     and the full visual() run twice for determinism.
  2. SHAPE. N (merged video tokens) and T = 4N (pre-merge patches) set the memory
     cost of making the score differentiable. Nothing downstream is sizeable
     without them.
  3. COST. Frozen forward vs. grad forward+backward, at query_frac 1.0 and 0.25.
     Decides whether --score-query-frac is needed to finish in a day.
  4. FRAMES. Which directory the item actually came from, so the raw/overlay
     settings are known to differ before 30 GPU-hours are spent on them.

Usage:
    source env.sh && CUDA_VISIBLE_DEVICES=0 python scripts/measure_vitkd_step0.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"))

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, VIDEO_KWARGS,
)


def original_colsum(qh, kh, scale, CH=2048):
    """The score loop exactly as it stood before the grad_logits refactor."""
    H, T, D = qh.shape
    colsum = torch.empty(H, T, device=qh.device, dtype=torch.float32)
    for h in range(H):
        kt = kh[h].transpose(0, 1)
        acc = torch.zeros(T, device=qh.device, dtype=torch.float32)
        for c in range(0, T, CH):
            lg = torch.matmul(qh[h, c:c + CH], kt) / scale
            acc += F.softmax(lg, dim=-1).sum(dim=0)
        colsum[h] = acc
    return colsum


def refactored_colsum(qh, kh, scale, CH=2048):
    """What the shipped code now does on the default path (stack instead of
    preallocate). Must agree bit-for-bit: 0.0 + x == x in IEEE754, and the
    accumulation order is unchanged."""
    from qwen2_5vl_visionzip import _score_chunk_colsum
    H, T, D = qh.shape
    per_head = []
    for h in range(H):
        kt = kh[h].transpose(0, 1)
        acc = torch.zeros(T, device=qh.device, dtype=torch.float32)
        for c in range(0, T, CH):
            acc = acc + _score_chunk_colsum(qh[h, c:c + CH], kt, scale)
        per_head.append(acc)
    return torch.stack(per_head, dim=0)


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    gz = os.environ.get("GAZE_OVERLAY", "1")
    vgz = os.environ.get("VLM_GAZE_OVERLAY", gz)
    print(f"\n=== 0. env ===\nGAZE_OVERLAY={gz}  VLM_GAZE_OVERLAY={vgz}", flush=True)

    # ---- 1. arithmetic regression on the score loop, no model needed ----------
    print("\n=== 1. score-loop regression (synthetic) ===", flush=True)
    torch.manual_seed(0)
    for (H, T, D) in [(16, 4096, 80), (16, 9216, 80)]:
        qh = torch.randn(H, T, D, device=device, dtype=torch.float32)
        kh = torch.randn(H, T, D, device=device, dtype=torch.float32)
        a = original_colsum(qh, kh, D ** 0.5)
        b = refactored_colsum(qh, kh, D ** 0.5)
        same = torch.equal(a, b)
        print(f"  H={H} T={T}: bit-identical={same}  max|Δ|={(a-b).abs().max().item():.3e}",
              flush=True)
        if not same:
            raise SystemExit("REGRESSION: refactored score loop is not bit-identical")
        del qh, kh, a, b
        torch.cuda.empty_cache()

    # ---- model ---------------------------------------------------------------
    print("\n=== loading model ===", flush=True)
    processor, model = load_visionzip_lora(device)
    base = model.get_base_model()
    base.eval()

    ds = CombinedMergeDataset(split="train", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    item = next(ds[i] for i in range(len(ds)) if ds[i] is not None)

    # ---- 4. which frames did this actually read? -----------------------------
    p = item["vlm_frame_paths"][0]
    tp = item["traj_frame_paths"][0]
    print("\n=== 4. frame variants ===")
    print(f"  student VLM : {os.path.basename(os.path.dirname(os.path.dirname(p)))}/"
          f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}")
    print(f"  teacher TAS : {os.path.basename(os.path.dirname(os.path.dirname(tp)))}/"
          f"{os.path.basename(os.path.dirname(tp))}/{os.path.basename(tp)}", flush=True)

    cached = preprocess_visionzip_item(
        processor, base, item["vlm_frame_paths"], item["question"],
        item["options"], device)
    assert cached is not None, "preprocess returned None on the probe item"

    N = cached["video_embeds"].shape[0]
    grid = cached["grid_thw"]
    T_merged = int(grid[0, 0])
    print("\n=== 2. shape ===")
    print(f"  grid_thw = {grid.tolist()}")
    print(f"  N (merged video tokens) = {N}")
    print(f"  T (pre-merge patches)   = {4 * N}")
    print(f"  spatial tokens/frame    = {N // max(1, T_merged)}")
    print(f"  10% budget              = {int(0.10 * N)} tokens")
    print(f"  score matrix per head   = {4*N} x {4*N} = {(4*N)**2/1e9:.2f}G entries",
          flush=True)

    # rebuild inputs for direct visual() timing
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "video", "video": item["vlm_frame_paths"],
         "max_pixels": VIDEO_KWARGS["max_pixels"],
         "min_pixels": VIDEO_KWARGS["min_pixels"], "fps": VIDEO_KWARGS["fps"]},
        {"type": "text", "text": "x"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ii, vi, vk = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=ii, videos=vi, **vk, return_tensors="pt")
    vis_dev = base.visual.patch_embed.proj.weight.device
    pv = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    gthw = inputs["video_grid_thw"].to(vis_dev)

    # ---- 1b. determinism of the shipped default path -------------------------
    print("\n=== 1b. visual() default-path determinism ===", flush=True)
    with torch.no_grad():
        _, s1, _ = base.visual(pv, grid_thw=gthw)
        _, s2, _ = base.visual(pv, grid_thw=gthw)
    print(f"  two runs identical = {torch.equal(s1, s2)}  "
          f"max|Δ| = {(s1-s2).abs().max().item():.3e}", flush=True)

    # ---- 3. cost -------------------------------------------------------------
    print("\n=== 3. cost per item ===", flush=True)

    def timed(fn, label):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        dt = time.time() - t0
        pk = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {label:38s} {dt:6.2f}s   peak {pk:5.1f} GB", flush=True)
        return dt

    def frozen():
        with torch.no_grad():
            base.visual(pv, grid_thw=gthw)
    timed(frozen, "frozen forward (default path)")

    # make block 31 trainable so a backward is meaningful
    blk = base.visual.blocks[-1]
    for prm in blk.attn.parameters():
        prm.requires_grad_(True)

    for frac in (1.0, 0.5, 0.25):
        def grad_step(frac=frac):
            ve, sc, _ = base.visual(pv, grid_thw=gthw, grad_last_block=True,
                                    score_query_frac=frac)
            z = (sc - sc.mean()) / (sc.std() + 1e-6)
            tgt = torch.zeros_like(z)
            tgt[torch.topk(z.detach(), max(1, int(0.10 * z.numel()))).indices] = 1.0
            loss = F.binary_cross_entropy_with_logits(z, tgt) + ve.float().pow(2).mean() * 0
            loss.backward()
            base.zero_grad(set_to_none=True)
        timed(grad_step, f"grad fwd+bwd (query_frac={frac})")
        torch.cuda.empty_cache()

    print("\nDone. Use the grad/frozen ratio to pick --score-query-frac:")
    print("  SG epoch = 2900 micro-steps/rank; budget ~3h/epoch.", flush=True)


if __name__ == "__main__":
    main()
