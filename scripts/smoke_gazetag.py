#!/usr/bin/env python
"""Single-GPU smoke for the per-token gaze-tag channel — validates the custom
path (select_m1_with_salience -> GazeTagEmbedding -> sel+tag -> build_merged_inputs
-> forward -> backward reaches tag table + gate) on a few real items, before
committing the 4-GPU run. No DDP."""
import os
import sys
import traceback

os.environ.setdefault("GAZE_OVERLAY", "1")

import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits
from TrajGazeMerge.models.gaze_tag import GazeTagEmbedding
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import STAGE1_DEFAULT
from TrajGazeMerge.training.train_visionzip_gazetag_lora import select_m1_with_salience
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder
from torch.optim import AdamW
import torch.nn.functional as F

N_STEPS = 3


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    hidden = base_qwen.config.hidden_size
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", STAGE1_DEFAULT, device, 16)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    gaze_tag = GazeTagEmbedding(out_dim=hidden, n_bins=16).to(device)
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0,
              alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)
    opt = AdamW(list(gaze_tag.parameters()) +
                [p for p in model.parameters() if p.requires_grad], lr=1e-3)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    model.train(); gaze_tag.train()
    done = 0
    gate0 = float(gaze_tag.gate.detach().item())
    for item in ds:
        if done >= N_STEPS:
            break
        if item is None:
            continue
        try:
            with torch.no_grad():
                cached = preprocess_visionzip_item(
                    processor, base_qwen, item["vlm_frame_paths"],
                    item["question"], item["options"], device)
            if cached is None:
                continue
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters:
                continue
            with torch.no_grad():
                sel, idx, sal_kept = select_m1_with_salience(
                    cached, item, device, "learned", encoder, hp, 0.07, 0.03,
                    tag_norm="minmax")
            n_base = idx.shape[0]
            assert sal_kept.shape == (n_base,), (sal_kept.shape, n_base)
            assert float(sal_kept.min()) >= 0.0 and float(sal_kept.max()) <= 1.0
            # tag + build OUTSIDE no_grad (trainable path)
            tag = gaze_tag(sal_kept)
            assert tag.shape == (n_base, hidden), tag.shape
            sel_tagged = sel + tag.to(sel.dtype)
            inputs = build_merged_inputs(base_qwen, cached, sel_tagged, idx)
            seq_len = inputs["inputs_embeds"].shape[1]
            assert inputs["position_ids"].shape[2] == seq_len
            assert inputs["attention_mask"].shape[1] == seq_len
            logits = forward_logits(model, inputs)
            gt = letters.index(item["answer"])
            loss = F.cross_entropy(logits[option_ids[:n_opt]].unsqueeze(0),
                                   torch.tensor([gt], device=device))
            opt.zero_grad()
            loss.backward()
            g_grad = gaze_tag.gate.grad
            t_grad = gaze_tag.table.weight.grad
            opt.step()
            assert torch.isfinite(loss), "loss not finite"
            assert g_grad is not None and torch.isfinite(g_grad).all(), "no/NaN gate grad"
            assert t_grad is not None and t_grad.abs().sum() > 0, "no table grad"
            done += 1
            print(f"[smoke] step {done}: loss={loss.item():.4f} kept_video={n_base} "
                  f"seq_len={seq_len} gate_grad={float(g_grad.item()):.3e} "
                  f"table_grad_l1={float(t_grad.abs().sum()):.3e}", flush=True)
        except Exception:
            traceback.print_exc()
            print("SMOKE_FAIL", flush=True)
            return
    gate1 = float(gaze_tag.gate.detach().item())
    if done == N_STEPS:
        print(f"[smoke] gate {gate0:.4f} -> {gate1:.4f} (moved={abs(gate1-gate0):.2e})",
              flush=True)
        print("SMOKE_OK", flush=True)
    else:
        print(f"SMOKE_FAIL only {done}/{N_STEPS} steps ran", flush=True)


if __name__ == "__main__":
    main()
