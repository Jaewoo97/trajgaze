"""Eval-only PER-ITEM dump for ④ Foveated-ROI models (gaze / attn-twin / random / none).

Fork of eval_dump.py: same M1 selection (select_complementary) + the foveal injection
the foveal trainer uses (compute_foveal_embeds → build_inputs_with_foveal). Writes one
json line per item with a STABLE `key` (md5 of task|question|options|answer) so the same
eval item pairs across arms for mcnemar.py. The --roi-* flags MUST match the training run
(default crop_frac 0.35 / margin 0.08 / K 32 = the launched arms).

Usage (one GPU per arm, run concurrently):
    python -m TrajGazeMerge.eval.eval_dump_foveal --gpu 2 --roi-arm gaze \
        --ckpt .../foveal_gaze/best.pth   --dump .../dumps/foveal_gaze.jsonl
    python -m TrajGazeMerge.eval.eval_dump_foveal --gpu 3 --roi-arm attn \
        --ckpt .../foveal_attn/best.pth   --dump .../dumps/foveal_attn.jsonl
    python -m TrajGazeMerge.eval.eval_dump_foveal --gpu 1 --roi-arm random \
        --ckpt .../foveal_random/best.pth --dump .../dumps/foveal_random.jsonl
Then:
    python -m TrajGazeMerge.eval.mcnemar --a dumps/foveal_attn.jsonl --label-a attn \
        --b dumps/foveal_gaze.jsonl --label-b gaze
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
from TrajGazeMerge.models.model import forward_logits, get_option_ids
from TrajGazeMerge.models.foveal_roi import roi_config_from_args, build_inputs_with_foveal
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import select_complementary
from TrajGazeMerge.training.train_visionzip_foveal_lora import compute_foveal_embeds
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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dump", required=True, help="output per-item jsonl path")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    # ROI arm + knobs (MUST match training)
    p.add_argument("--roi-arm", choices=["gaze", "attn", "random", "none", "anticipatory"],
                   default="gaze")
    p.add_argument("--roi-antic_horizon", type=int, default=12)
    p.add_argument("--roi-n_fix_frames",   type=int,   default=4)
    p.add_argument("--roi-crop_frac",      type=float, default=0.35)
    p.add_argument("--roi-margin_frac",    type=float, default=0.08)
    p.add_argument("--roi-max_crop_frac",  type=float, default=0.50)
    p.add_argument("--roi-foveal_k",       type=int,   default=32)
    p.add_argument("--roi-roi_max_pixels", type=int,   default=256 * 28 * 28)
    p.add_argument("--roi-min_frame_gap",  type=float, default=0.08)
    p.add_argument("--roi-pool", choices=["uniform", "topk_attn", "all"], default="uniform")
    # selection (M1)
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
    p.add_argument("--horizon",   type=float, default=2.0)
    p.add_argument("--sigma-g",   type=float, default=2.0)
    p.add_argument("--sigma-h",   type=float, default=3.0)
    p.add_argument("--alpha-hand", type=float, default=0.7)
    p.add_argument("--sigma-v",   type=float, default=0.05)
    p.add_argument("--sigma-gh",  type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = roi_config_from_args(args)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    hp = dict(horizon=args.horizon, sigma_g=args.sigma_g, sigma_h=args.sigma_h,
              alpha_hand=args.alpha_hand, sigma_v=args.sigma_v, sigma_gh=args.sigma_gh)

    print(f"[eval_dump_foveal] loading {args.ckpt} | arm={args.roi_arm} {cfg.describe()}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = qwen.load_state_dict(ckpt["lora_state"], strict=False)
    print(f"[eval_dump_foveal] ckpt epoch={ckpt.get('epoch')} acc={ckpt.get('acc')} "
          f"roi_arm={ckpt.get('roi_arm')} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
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
    print(f"[eval_dump_foveal] arm={args.roi_arm} content={args.content_ratio} "
          f"gaze={args.traj_ratio} | n_items={len(ds)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    correct = 0; total = 0; n_foveal = 0
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
                foveal = compute_foveal_embeds(processor, base_qwen, cached, item, cfg, args.roi_arm, device)
                if foveal is not None:
                    n_foveal += 1
                inputs = build_inputs_with_foveal(base_qwen, cached, sel, recv, foveal)
                logits = forward_logits(qwen, inputs)
                pred = logits[option_ids[:n_opt]].argmax().item()
                gt = letters.index(item["answer"])
                ok = int(pred == gt)
                correct += ok; total += 1
                by_task.setdefault(item["task"], []).append(ok)
                opts = item["options"]
                fout.write(json.dumps({
                    "key": item_key(item), "src": src, "task": item["task"],
                    "pred": pred, "gt": gt, "ok": ok, "n_opt": n_opt,
                    "foveal_k": (0 if foveal is None else int(foveal.shape[0])),
                    "question": item.get("question", ""),
                    "options": opts,
                    "pred_text": opts[pred] if 0 <= pred < len(opts) else "",
                    "gt_text": opts[gt] if 0 <= gt < len(opts) else "",
                }) + "\n")
            except Exception as e:
                print(f"[eval_dump_foveal] idx={idx} ERR {e!r}", flush=True)
                continue

    overall = 100.0 * correct / max(1, total)
    print(f"\n[eval_dump_foveal] arm={args.roi_arm} Overall: {overall:.2f}%  (n={total}, "
          f"foveal_hit={n_foveal}/{total})  → {args.dump}", flush=True)
    for t, v in sorted(by_task.items()):
        print(f"    {t:42s} n={len(v):4d}  {100.0 * sum(v) / max(1, len(v)):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
