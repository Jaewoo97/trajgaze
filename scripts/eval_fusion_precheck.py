#!/usr/bin/env python
"""Inference-only pre-check for the soft-fusion selection rule.

Loads a trained VZ-complement LoRA checkpoint (default: M1 =
visionzip_complement_learned_overlay/best.pth) + the frozen TAS Stage-1 encoder,
and evaluates one or more selection configs on the egtea 2-way test set WITHOUT
any training. The point is a cheap directional signal: does fusion beat M1's raw
top-k (63.01) before we spend ~13h training it?

Each item is preprocessed once (VisionZip ViT forward), then every config is run
on it (selection + LLM forward), so adding configs is cheap apart from the LLM
pass. Configs are comma-separated:  topk , coverage , fusion@<lambda>[@<norm>]
e.g.  --configs "topk,fusion@0.5,fusion@1.0,fusion@2.0"

Run with GAZE_OVERLAY=1 to match the M1 training/eval frame base.
"""
import argparse
import os
import sys

os.environ.setdefault("GAZE_OVERLAY", "1")

import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import (
    select_complementary, STAGE1_DEFAULT,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder


def parse_configs(spec: str, default_norm: str):
    """'topk,fusion@1.0,fusion@2.0@rank' -> [(name, mode, lam, norm), ...]"""
    cfgs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split("@")
        mode = parts[0]
        if mode == "fusion":
            lam = float(parts[1]) if len(parts) > 1 else 1.0
            norm = parts[2] if len(parts) > 2 else default_norm
            name = f"fusion@{lam:g}/{norm}"
        else:
            lam, norm = 1.0, default_norm
            name = mode
        cfgs.append((name, mode, lam, norm))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/EgoGazeVQA/TrajGazeMerge/"
                    "checkpoints/visionzip_complement_learned_overlay/best.pth")
    ap.add_argument("--configs", default="topk,fusion@1.0")
    ap.add_argument("--content-ratio", type=float, default=0.07)
    ap.add_argument("--traj-ratio", type=float, default=0.03)
    ap.add_argument("--fusion-norm", default="minmax", choices=["minmax", "rank"],
                    help="default norm for fusion configs that don't specify one")
    ap.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every k-th item (representative subset across sources)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no cap")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="partition the test set across this many parallel processes")
    ap.add_argument("--shard-id", type=int, default=0, help="this process's shard index")
    ap.add_argument("--out", default="", help="if set, dump this shard's tallies as JSON here")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--include-hdepic", action="store_true",
                    help="3-way (default off = egtea 2-way, matches M1's 63.01).")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    configs = parse_configs(args.configs, args.fusion_norm)
    print(f"[precheck] ckpt={args.ckpt}", flush=True)
    print(f"[precheck] configs={[c[0] for c in configs]} "
          f"content={args.content_ratio} traj={args.traj_ratio} "
          f"stride={args.stride} limit={args.limit} overlay={os.environ.get('GAZE_OVERLAY')}",
          flush=True)

    processor, model = load_visionzip_lora(device)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck.get("lora_state", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[precheck] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)} "
              f"(epoch={ck.get('epoch')}, acc={ck.get('acc')})", flush=True)
    model.eval()
    base_qwen = model.get_base_model()
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, 16)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0,
              alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=args.include_hdepic)

    stats = {name: {"c": 0, "n": 0, "task": {}} for (name, *_rest) in configs}
    n_seen = 0
    with torch.no_grad():
        for i, item in enumerate(ds):
            if item is None:
                continue
            if args.stride > 1 and (i % args.stride) != 0:
                continue
            if args.num_shards > 1 and (i % args.num_shards) != args.shard_id:
                continue
            if args.limit and n_seen >= args.limit:
                break
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen, item["vlm_frame_paths"],
                    item["question"], item["options"], device)
                if cached is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + j) for j in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                gt = letters.index(item["answer"])
            except Exception:
                continue

            n_seen += 1
            for (name, mode, lam, norm) in configs:
                try:
                    sel, idx = select_complementary(
                        cached, item, device, "learned", encoder, hp,
                        args.content_ratio, args.traj_ratio,
                        complement_mode=mode, nms_radius=1,
                        fusion_lambda=lam, fusion_norm=norm)
                    inp = build_merged_inputs(base_qwen, cached, sel, idx)
                    logits = forward_logits(model, inp)
                    ok = int(logits[option_ids[:n_opt]].argmax().item() == gt)
                except Exception:
                    continue
                s = stats[name]
                s["c"] += ok
                s["n"] += 1
                s["task"].setdefault(item["task"], []).append(ok)

            if n_seen % 50 == 0:
                line = " | ".join(
                    f"{name} {100.0*stats[name]['c']/max(1,stats[name]['n']):.2f}%"
                    for (name, *_ ) in configs)
                print(f"[precheck] seen={n_seen} :: {line}", flush=True)

    print("\n==== RESULTS (n_items={}) ====".format(n_seen), flush=True)
    for (name, *_ ) in configs:
        s = stats[name]
        acc = 100.0 * s["c"] / max(1, s["n"])
        print(f"RESULT {name:18s} Overall {acc:.2f}%  (n={s['n']})", flush=True)
        for t, v in sorted(s["task"].items()):
            print(f"    {t}: {100.0*sum(v)/len(v):.2f}% (n={len(v)})", flush=True)

    if args.out:
        import json
        dump = {name: {"c": s["c"], "n": s["n"],
                       "task": {t: [sum(v), len(v)] for t, v in s["task"].items()}}
                for name, s in stats.items()}
        with open(args.out, "w") as f:
            json.dump(dump, f)
        print(f"[precheck] wrote {args.out}", flush=True)
    print("PRECHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
