"""Per-item eval dump for temporal-budget models (tbudget_w00/w05/w10).

Usage:
    python -m TrajGazeMerge.eval.eval_dump_tbudget \
        --ckpt .../tbudget_w10_2way/best.pth --traj-weight 1.0 \
        --dump .../dumps/tbudget_w10.jsonl --gpu 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import build_merged_inputs, forward_logits, get_option_ids
from TrajGazeMerge.models.temporal_budget import temporal_budget_select_tokens
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)


def item_key(item) -> str:
    s = "|".join([
        str(item.get("task", "")),
        str(item.get("question", "")),
        "||".join(item.get("options", [])),
        str(item.get("answer", "")),
    ])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--dump",        required=True)
    p.add_argument("--traj-weight", type=float, default=1.0)
    p.add_argument("--tau",         type=float, default=1.0)
    p.add_argument("--sigma-v",     type=float, default=0.05)
    p.add_argument("--sigma-gh",    type=float, default=0.10)
    p.add_argument("--gpu",         type=int,   default=1)
    p.add_argument("--n-frames",    type=int,   default=128)
    p.add_argument("--include-hdepic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[eval_dump_tbudget] ckpt={args.ckpt}  w={args.traj_weight}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = qwen.load_state_dict(ckpt["lora_state"], strict=False)
    print(f"  epoch={ckpt.get('epoch')} acc={ckpt.get('acc')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=args.n_frames,
                              n_traj_frames=args.n_frames,
                              include_hdepic=args.include_hdepic)
    print(f"  n_items={len(ds)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    correct = 0; total = 0
    by_task: dict[str, list] = {}

    with open(args.dump, "w") as fout, torch.no_grad():
        for idx in range(len(ds)):
            try:
                src, _ = ds.items[idx]
            except Exception:
                src = "?"
            try:
                item = ds[idx]
                if item is None: continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device)
                if cached is None: continue

                sel, recv, _ = temporal_budget_select_tokens(
                    cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
                    cached["grid_thw"], item["traj"],
                    tau=args.tau, traj_weight=args.traj_weight,
                    sigma_v=args.sigma_v, sigma_gh=args.sigma_gh,
                )
                inputs = build_merged_inputs(base_qwen, cached, sel, recv)
                logits = forward_logits(qwen, inputs)
                pred = logits[option_ids[:n_opt]].argmax().item()
                gt   = letters.index(item["answer"])
                ok   = int(pred == gt)
                correct += ok; total += 1
                opts = item["options"]
                by_task.setdefault(item["task"], []).append(ok)
                fout.write(json.dumps({
                    "key": item_key(item), "src": src, "task": item["task"],
                    "pred": pred, "gt": gt, "ok": ok, "n_opt": n_opt,
                    "options": opts,
                    "pred_text": opts[pred] if 0 <= pred < len(opts) else "",
                    "gt_text":   opts[gt]   if 0 <= gt   < len(opts) else "",
                }) + "\n")
            except Exception as e:
                print(f"  idx={idx} ERR {e!r}", flush=True)

    overall = 100.0 * correct / max(1, total)
    print(f"\n[eval_dump_tbudget] Overall: {overall:.2f}%  (n={total})  → {args.dump}", flush=True)
    for t, v in sorted(by_task.items()):
        print(f"    {t:42s} n={len(v):4d}  {100.0*sum(v)/max(1,len(v)):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
