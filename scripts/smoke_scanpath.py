#!/usr/bin/env python
"""Single-GPU smoke for the gaze-scanpath channel — validates the custom path
(ScanpathEncoder -> build_inputs_with_gaze -> forward -> backward reaches gaze
params + gate) on a few real items, before committing the 4-GPU run. No DDP."""
import os
import sys
import traceback

os.environ.setdefault("GAZE_OVERLAY", "1")

import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import get_option_ids, forward_logits
from TrajGazeMerge.models.scanpath_encoder import ScanpathEncoder
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import (
    select_complementary, STAGE1_DEFAULT,
)
from TrajGazeMerge.training.train_visionzip_scanpath_lora import build_inputs_with_gaze
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

    scan = ScanpathEncoder(out_dim=hidden, n_tokens=8, t_scan=32).to(device)
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0,
              alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)
    opt = AdamW(list(scan.parameters()) +
                [p for p in model.parameters() if p.requires_grad], lr=1e-3)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    model.train(); scan.train()
    done = 0
    gate0 = float(scan.gate.detach().item())
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
                sel, idx = select_complementary(
                    cached, item, device, "learned", encoder, hp, 0.07, 0.03,
                    complement_mode="topk")
            n_base = idx.shape[0]
            gaze_tokens = scan(item.get("traj"), device)
            assert gaze_tokens.shape == (8, hidden), gaze_tokens.shape
            inp = build_inputs_with_gaze(base_qwen, cached, sel, idx, gaze_tokens)
            seq_len = inp["inputs_embeds"].shape[1]
            assert inp["position_ids"].shape[2] == seq_len
            assert inp["attention_mask"].shape[1] == seq_len
            logits = forward_logits(model, inp)
            gt = letters.index(item["answer"])
            loss = F.cross_entropy(logits[option_ids[:n_opt]].unsqueeze(0),
                                   torch.tensor([gt], device=device))
            opt.zero_grad()
            loss.backward()
            g_grad = scan.gate.grad
            q_grad = scan.query.grad
            opt.step()
            assert torch.isfinite(loss), "loss not finite"
            assert g_grad is not None and torch.isfinite(g_grad).all(), "no/NaN gate grad"
            assert q_grad is not None and q_grad.abs().sum() > 0, "no query grad"
            done += 1
            print(f"[smoke] step {done}: loss={loss.item():.4f} kept_video={n_base} "
                  f"seq_len={seq_len} gate_grad={float(g_grad.item()):.3e}", flush=True)
        except Exception:
            traceback.print_exc()
            print("SMOKE_FAIL", flush=True)
            return
    gate1 = float(scan.gate.detach().item())
    if done == N_STEPS:
        print(f"[smoke] gate {gate0:.4f} -> {gate1:.4f} (moved={abs(gate1-gate0):.2e})",
              flush=True)
        print("SMOKE_OK", flush=True)
    else:
        print(f"SMOKE_FAIL only {done}/{N_STEPS} steps ran", flush=True)


if __name__ == "__main__":
    main()
