"""Per-item eval dump for Gaze-Text prefix models.

Loads a gazetext LoRA checkpoint, runs egtea test set with the same
arm's text prefix used at training time, writes one JSONL line per item.
Stable key = md5(task|question|options|answer) — pairs with m1.jsonl.

Usage:
    python -m TrajGazeMerge.eval.eval_dump_gazetext \\
        --ckpt .../gazetext_gaze/best.pth --arm gaze \\
        --dump .../dumps/gazetext_gaze.jsonl --gpu 0
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
from TrajGazeMerge.models.gaze_text import build_question_prefix
from TrajGazeMerge.models.model import build_merged_inputs, forward_logits, get_option_ids
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import select_complementary
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"
CONTENT_RATIO  = 0.07
TRAJ_RATIO     = 0.03


def item_key(item) -> str:
    s = "|".join([
        str(item.get("task", "")),
        str(item.get("question", "")),
        "||".join(item.get("options", [])),
        str(item.get("answer", "")),
    ])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _item_seed(item) -> int:
    s = "|".join([str(item.get("task", "")), str(item.get("question", ""))])
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--dump",    required=True, help="output per-item jsonl path")
    p.add_argument("--arm",     choices=["gaze", "random", "none"], default="gaze")
    p.add_argument("--gpu",     type=int, default=0)
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    p.add_argument("--n-frames",    type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--include-hdepic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[eval_dump_gazetext] loading {args.ckpt}  arm={args.arm}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = qwen.load_state_dict(ckpt["lora_state"], strict=False)
    print(f"  ckpt epoch={ckpt.get('epoch')} acc={ckpt.get('acc')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0, alpha_hand=0.7,
              sigma_v=0.05, sigma_gh=0.10)

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

                prefix = build_question_prefix(args.arm, item["traj"],
                                               args.n_frames, _item_seed(item))
                question = f"{prefix}\n{item['question']}" if prefix else item["question"]

                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], question, item["options"], device)
                if cached is None: continue

                sel, recv = select_complementary(
                    cached, item, device, "learned", encoder, hp,
                    CONTENT_RATIO, TRAJ_RATIO, complement_mode="topk")
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
                    "gaze_prefix": prefix,
                    "question": item.get("question", ""),
                    "options": opts,
                    "pred_text": opts[pred] if 0 <= pred < len(opts) else "",
                    "gt_text":   opts[gt]   if 0 <= gt   < len(opts) else "",
                }) + "\n")
            except Exception as e:
                print(f"[eval_dump_gazetext] idx={idx} ERR {e!r}", flush=True)
                continue

    overall = 100.0 * correct / max(1, total)
    print(f"\n[eval_dump_gazetext] Overall: {overall:.2f}%  (n={total})  → {args.dump}", flush=True)
    for t, v in sorted(by_task.items()):
        print(f"    {t:42s} n={len(v):4d}  {100.0 * sum(v) / max(1, len(v)):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
