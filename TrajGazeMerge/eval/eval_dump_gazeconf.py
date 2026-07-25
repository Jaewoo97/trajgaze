"""Per-item eval dump for fixation-confidence-weighted gaze complement models.

Usage:
    python -m TrajGazeMerge.eval.eval_dump_gazeconf \\
        --ckpt .../gazeconf_confidence/best.pth --arm confidence \\
        --dump .../dumps/gazeconf_confidence.jsonl --gpu 0
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
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_gazeconf_lora import select_complementary_conf
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"


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
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--dump",     required=True)
    p.add_argument("--arm",
                   choices=["confidence", "inverse", "random", "none",
                            "task_adaptive", "signrouted", "dual"],
                   default="confidence")
    p.add_argument("--fix-ratio", type=float, default=0.02)
    p.add_argument("--sac-ratio", type=float, default=0.01)
    p.add_argument("--stage1-ckpt",   default=STAGE1_DEFAULT)
    p.add_argument("--saccade-speed", type=float, default=0.25)
    p.add_argument("--window",        type=int,   default=5)
    p.add_argument("--c-min",         type=float, default=0.10)
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--gpu",      type=int, default=0)
    p.add_argument("--include-hdepic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[eval_dump_gazeconf] arm={args.arm}  ckpt={args.ckpt}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = qwen.load_state_dict(ckpt["lora_state"], strict=False)
    print(f"  epoch={ckpt.get('epoch')} acc={ckpt.get('acc')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

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
                item = ds[idx]
                if item is None: continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device)
                if cached is None: continue

                sel_embeds, recv_idx = select_complementary_conf(
                    cached, item, device, encoder, args.arm,
                    args.saccade_speed, args.window, args.c_min,
                    fix_ratio=args.fix_ratio, sac_ratio=args.sac_ratio)
                inputs = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                logits = forward_logits(qwen, inputs)
                pred = logits[option_ids[:n_opt]].argmax().item()
                gt   = letters.index(item["answer"])
                ok   = int(pred == gt)
                correct += ok; total += 1
                opts = item["options"]
                by_task.setdefault(item["task"], []).append(ok)
                fout.write(json.dumps({
                    "key": item_key(item), "task": item["task"],
                    "pred": pred, "gt": gt, "ok": ok, "n_opt": n_opt,
                    "options": opts,
                    "pred_text": opts[pred] if 0 <= pred < len(opts) else "",
                    "gt_text":   opts[gt]   if 0 <= gt   < len(opts) else "",
                    "arm": args.arm,
                }) + "\n")
            except Exception as e:
                print(f"  idx={idx} ERR {e!r}", flush=True)

    overall = 100.0 * correct / max(1, total)
    print(f"\n[eval_dump_gazeconf] Overall: {overall:.2f}%  (n={total})  → {args.dump}",
          flush=True)
    for t, v in sorted(by_task.items()):
        print(f"    {t:42s} n={len(v):4d}  {100.0*sum(v)/max(1,len(v)):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
