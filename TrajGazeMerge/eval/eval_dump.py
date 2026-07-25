"""Eval-only PER-ITEM dump for VisionZip-complement models (M1 / query_gaze / controls).

Loads a trained LoRA checkpoint, runs the egtea test set under a given complement
config, and writes ONE json line per item:
    {"key", "src", "task", "pred", "gt", "ok", "n_opt"}
where key = md5(task|question|options|answer) is a STABLE id that pairs the same
eval item across two models (robust to skipped items / ordering). Also prints
overall + per-task accuracy WITH per-task n.

This is the per-item granularity the training-time evaluate() throws away; it is
what mcnemar.py needs for paired significance testing at the <1%p effect sizes we
work at. Reuses select_complementary, so it supports every mode (topk / query_gaze
cosine|random|shuffle / coverage / fusion) with no extra code.

Usage (single GPU):
    # M1 reference (best.pth already trained)
    python -m TrajGazeMerge.eval.eval_dump --gpu 0 \
        --ckpt .../visionzip_complement_learned_overlay/best.pth \
        --complement-mode topk --content-ratio 0.07 --traj-ratio 0.03 \
        --dump .../dumps/m1.jsonl
    # query_gaze cosine
    python -m TrajGazeMerge.eval.eval_dump --gpu 0 \
        --ckpt .../visionzip_query_gaze_overlay/best.pth \
        --complement-mode query_gaze --content-ratio 0.07 --query-ratio 0.01 \
        --traj-ratio 0.02 --query-mode cosine --dump .../dumps/qg_cosine.jsonl
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
from TrajGazeMerge.models.model import (
    build_merged_inputs, forward_logits, get_option_ids,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import select_complementary
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"


def item_key(item) -> str:
    """Stable id for PAIRING the same eval item across two models."""
    s = "|".join([
        str(item.get("task", "")),
        str(item.get("question", "")),
        "||".join(item.get("options", [])),
        str(item.get("answer", "")),
    ])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dump", required=True, help="output per-item jsonl path")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    p.add_argument("--traj-pool-mode", choices=["learned", "anticipatory"], default="learned")
    p.add_argument("--complement-mode", choices=["topk", "coverage", "fusion", "query_gaze"],
                   default="topk")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio",    type=float, default=0.03)
    p.add_argument("--query-ratio",   type=float, default=0.0)
    p.add_argument("--query-mode",    choices=["cosine", "random", "shuffle"], default="cosine")
    p.add_argument("--nms-radius",    type=int,   default=1)
    p.add_argument("--fusion-lambda", type=float, default=1.0)
    p.add_argument("--fusion-norm",   choices=["minmax", "rank"], default="minmax")
    p.add_argument("--n-frames",      type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--include-hdepic", action="store_true",
                   help="default OFF → egtea 2-way n=1011 (matches the trained models).")
    # anticipatory hp (only used by --traj-pool-mode anticipatory)
    p.add_argument("--horizon",   type=float, default=2.0)
    p.add_argument("--sigma-g",   type=float, default=2.0)
    p.add_argument("--sigma-h",   type=float, default=3.0)
    p.add_argument("--alpha-hand", type=float, default=0.7)
    p.add_argument("--sigma-v",   type=float, default=0.05)
    p.add_argument("--sigma-gh",  type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    hp = dict(horizon=args.horizon, sigma_g=args.sigma_g, sigma_h=args.sigma_h,
              alpha_hand=args.alpha_hand, sigma_v=args.sigma_v, sigma_gh=args.sigma_gh)

    print(f"[eval_dump] loading {args.ckpt}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = qwen.load_state_dict(ckpt["lora_state"], strict=False)
    print(f"[eval_dump] ckpt epoch={ckpt.get('epoch')} acc={ckpt.get('acc')} "
          f"| missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)

    encoder = None
    if args.traj_pool_mode == "learned":
        encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
        encoder.eval()
        for prm in encoder.parameters():
            prm.requires_grad_(False)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=args.n_frames,
                              n_traj_frames=args.n_frames, include_hdepic=args.include_hdepic)
    print(f"[eval_dump] mode={args.complement_mode} "
          f"content={args.content_ratio} query={args.query_ratio}[{args.query_mode}] "
          f"gaze={args.traj_ratio} | n_items={len(ds)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    correct = 0
    total = 0
    by_task: dict[str, list] = {}
    with open(args.dump, "w") as fout, torch.no_grad():
        for idx in range(len(ds)):
            try:
                src, _ = ds.items[idx]
            except Exception:
                src = "?"
            try:
                item = ds[idx]
                if item is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device)
                if cached is None:
                    continue
                sel, recv = select_complementary(
                    cached, item, device, args.traj_pool_mode, encoder, hp,
                    args.content_ratio, args.traj_ratio,
                    complement_mode=args.complement_mode, nms_radius=args.nms_radius,
                    fusion_lambda=args.fusion_lambda, fusion_norm=args.fusion_norm,
                    query_ratio=args.query_ratio, query_mode=args.query_mode)
                inputs = build_merged_inputs(base_qwen, cached, sel, recv)
                logits = forward_logits(qwen, inputs)
                pred = logits[option_ids[:n_opt]].argmax().item()
                gt = letters.index(item["answer"])
                ok = int(pred == gt)
                correct += ok
                total += 1
                by_task.setdefault(item["task"], []).append(ok)
                opts = item["options"]
                fout.write(json.dumps({
                    "key": item_key(item), "src": src, "task": item["task"],
                    "pred": pred, "gt": gt, "ok": ok, "n_opt": n_opt,
                    "question": item.get("question", ""),
                    "options": opts,
                    "pred_text": opts[pred] if 0 <= pred < len(opts) else "",
                    "gt_text": opts[gt] if 0 <= gt < len(opts) else "",
                }) + "\n")
            except Exception as e:
                print(f"[eval_dump] idx={idx} ERR {e!r}", flush=True)
                continue

    overall = 100.0 * correct / max(1, total)
    print(f"\n[eval_dump] Overall: {overall:.2f}%  (n={total})  → {args.dump}", flush=True)
    for t, v in sorted(by_task.items()):
        print(f"    {t:42s} n={len(v):4d}  {100.0 * sum(v) / max(1, len(v)):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
