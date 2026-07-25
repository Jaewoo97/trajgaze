"""Smoke test for temporal-budget selection.

  python scripts/smoke_tbudget.py            # synthetic logic test (CPU, instant)
  python scripts/smoke_tbudget.py --real --gpu 0   # full pipeline on 1 item/source
"""
import argparse, os, sys
os.environ.setdefault("GAZE_OVERLAY", "1")
sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

import torch
from TrajGazeMerge.models.temporal_budget import (
    temporal_budget_select_tokens, compute_temporal_interaction, _apportion,
)


def synthetic():
    dev = torch.device("cpu")
    print("== synthetic logic test ==")

    # _apportion: sum preserved, cap respected
    for total, cap, T in [(128, 50, 64), (10, 4, 8), (100, 256, 32), (0, 5, 4)]:
        w = torch.rand(T)
        K = _apportion(w, total, cap)
        assert int(K.sum()) == min(total, T * cap), (int(K.sum()), total, cap, T)
        assert int(K.max()) <= cap
        assert int(K.min()) >= 0
    print("  _apportion sum/cap OK")

    # full selection on synthetic VZ tensors
    T, S, d, dk = 64, 64, 32, 16
    N = T * S
    video_embeds = torch.randn(N, d)
    attn_scores = torch.rand(N)
    attn_key = torch.randn(1, N, dk)
    grid_thw = torch.tensor([[T, 8, 8]])
    # synthetic traj: gaze fixates mid-clip, hands present late
    Tt = 100
    traj = {
        "gaze_pos":  torch.rand(Tt, 2),
        "gaze_mask": (torch.rand(Tt) > 0.3),
        "left_pos":  torch.rand(Tt, 2),
        "left_mask": (torch.arange(Tt) > 60),
        "right_pos": torch.rand(Tt, 2),
        "right_mask": (torch.arange(Tt) > 70),
    }
    inter = compute_temporal_interaction(traj, T, dev)
    assert inter.shape == (T,) and torch.isfinite(inter).all()
    print(f"  interaction (T,): min={inter.min():.3f} max={inter.max():.3f} mean={inter.mean():.3f}")

    se, ri, K = temporal_budget_select_tokens(
        video_embeds, attn_scores, attn_key, grid_thw, traj, tau=1.0, traj_weight=0.5,
    )
    total_keep = int(0.10 * N)
    assert ri.dim() == 1 and se.shape[0] == ri.shape[0]
    assert ri.unique().numel() == ri.numel(), "receiver_idx not unique"
    assert bool((ri[1:] >= ri[:-1]).all()), "receiver_idx not sorted asc"
    assert ri.max() < N and ri.min() >= 0
    kept = ri.numel()
    print(f"  N={N} total_keep≈{total_keep} kept={kept} ({100*kept/N:.1f}%)")
    print(f"  budget K: active_frames={(K>0).sum().item()}/{T} "
          f"max={K.max().item()} min={K.min().item()} sum={K.sum().item()}")

    # compare to pure-VZ (traj_weight=0 should track attention share)
    se0, ri0, K0 = temporal_budget_select_tokens(
        video_embeds, attn_scores, attn_key, grid_thw, traj, tau=1.0, traj_weight=0.0,
    )
    print(f"  traj_weight=0: active_frames={(K0>0).sum().item()} max={K0.max().item()}")
    se1, ri1, K1 = temporal_budget_select_tokens(
        video_embeds, attn_scores, attn_key, grid_thw, traj, tau=0.3, traj_weight=1.0,
    )
    print(f"  traj_weight=1,tau=.3: active_frames={(K1>0).sum().item()} max={K1.max().item()}")

    # degenerate: N % T != 0 -> fallback
    bad_grid = torch.tensor([[7, 8, 8]])  # 7 doesn't divide 4096
    se2, ri2, K2 = temporal_budget_select_tokens(
        video_embeds, attn_scores, attn_key, bad_grid, traj,
    )
    assert K2 is None and ri2.numel() > 0
    print(f"  N%T!=0 fallback OK (kept={ri2.numel()})")
    print("SYNTHETIC: PASS")


def real(gpu):
    print(f"== real pipeline test (gpu {gpu}, GAZE_OVERLAY={os.environ.get('GAZE_OVERLAY')}) ==")
    import torch.nn.functional as F
    from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
    from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits
    from TrajGazeMerge.training.train_visionzip_lora import (
        load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
    )

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    option_ids = get_option_ids(processor, 5)
    model.train()

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True)
    seen = set()
    n_ok = 0
    for idx in range(len(ds)):
        src = ds.items[idx][0]
        if src in seen:
            continue
        item = ds[idx]
        if item is None:
            continue
        cached = preprocess_visionzip_item(
            processor, base_qwen, item["vlm_frame_paths"],
            item["question"], item["options"], device,
        )
        if cached is None:
            continue
        seen.add(src)
        N = cached["video_embeds"].shape[0]

        se, ri, K = temporal_budget_select_tokens(
            cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
            cached["grid_thw"], item["traj"], tau=1.0, traj_weight=0.5,
        )
        se_vz, ri_vz = visionzip_select_tokens(
            cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
        )
        T = int(cached["grid_thw"][0, 0].item())
        print(f"\n[{src}] task={item['task']}  N={N} T={T} S={N//T}")
        print(f"  tbudget kept={ri.numel()} ({100*ri.numel()/N:.1f}%)  vz kept={ri_vz.numel()} ({100*ri_vz.numel()/N:.1f}%)")
        if K is not None:
            print(f"  budget: active_frames={(K>0).sum().item()}/{T} max={K.max().item()} "
                  f"({100*K.max().item()/max(1,K.sum().item()):.1f}% of kept)")

        # forward + backward to confirm grad flows to LoRA
        inputs_dict = build_merged_inputs(base_qwen, cached, se, ri)
        logits = forward_logits(model, inputs_dict)
        n_opt = len(item["options"])
        letters = [chr(65 + i) for i in range(n_opt)]
        if item["answer"] in letters:
            gt = letters.index(item["answer"])
            loss = F.cross_entropy(logits[option_ids[:n_opt]].unsqueeze(0),
                                   torch.tensor([gt], device=device))
            loss.backward()
            g = sum(p.grad.abs().sum().item() for p in model.parameters()
                    if p.requires_grad and p.grad is not None)
            print(f"  loss={loss.item():.4f}  sum|grad|(LoRA)={g:.3e}")
            assert g > 0, "no gradient reached LoRA params!"
            model.zero_grad()
        n_ok += 1
        if len(seen) >= 3:
            break
    print(f"\nREAL: PASS ({n_ok} sources)" if n_ok else "REAL: no items ran")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    synthetic()
    if a.real:
        real(a.gpu)
