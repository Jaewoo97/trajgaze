"""Full per-source eval for minimal-traj LoRA ckpts.

Same selection rule training uses: MinimalTrajScorer(video_embeds) → scores
→ gaze_weighted_merge. Saves JSON next to ckpt + per-item timing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.minimal_traj_scorer import MinimalTrajScorer
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--merge-ratio", type=float, default=0.9)
    p.add_argument("--d-hidden", type=int, default=128)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[full_eval_min] loading Qwen2.5-VL-7B + LoRA on cuda:{args.gpu}", flush=True)
    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    option_ids = get_option_ids(processor, 5)

    d_in = base_qwen.config.hidden_size
    scorer = MinimalTrajScorer(d_in=d_in, d_hidden=args.d_hidden).to(device)
    print(f"[full_eval_min] scorer params={scorer.param_count()/1e3:.1f}K", flush=True)

    print(f"[full_eval_min] loading ckpt: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "lora_state" in ckpt:
        qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
    if "scorer_state" in ckpt:
        scorer.load_state_dict(ckpt["scorer_state"], strict=False)
    qwen_model.eval(); scorer.eval()

    print("[full_eval_min] loading full 3-way test set", flush=True)
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True,
    )
    print(f"[full_eval_min] {len(test_ds)} items", flush=True)

    SG_END, EG_END = 526, 526 + 485
    def src_for_idx(i):
        if i < SG_END: return "sg"
        if i < EG_END: return "eg"
        return "hd"

    per_source = {"sg": {"ok": 0, "n": 0}, "eg": {"ok": 0, "n": 0}, "hd": {"ok": 0, "n": 0}}
    per_task: dict[str, dict] = {}
    timings_pre, timings_cmp = [], []
    end_idx = len(test_ds) if args.end_idx < 0 else args.end_idx
    t0 = time.time(); n_seen = 0

    for idx in range(args.start_idx, end_idx):
        try:
            item = test_ds[idx]
        except Exception:
            continue
        if item is None: continue
        src = src_for_idx(idx)
        try:
            t_pre0 = time.perf_counter()
            cached = preprocess_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device,
            )
            torch.cuda.synchronize(device)
            t_pre1 = time.perf_counter()
            if cached is None: continue
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters: continue

            video_b = cached["video_embeds"]
            N_b = video_b.shape[0]
            T_b = int(cached["grid_thw"][0, 0].item())
            S_b = N_b // max(1, T_b)
            r_b = max(1, int(args.merge_ratio * N_b))

            _, scores = scorer(video_b, T_b, S_b)
            merged_video, recv = gaze_weighted_merge(video_b, scores, r_b)
            inputs_dict = build_merged_inputs(base_qwen, cached, merged_video, recv)
            logits = forward_logits(qwen_model, inputs_dict)
            torch.cuda.synchronize(device)
            t_cmp1 = time.perf_counter()
            timings_pre.append(t_pre1 - t_pre0)
            timings_cmp.append(t_cmp1 - t_pre1)

            pred_idx = logits[option_ids[:n_opt]].argmax().item()
            gt_idx = letters.index(item["answer"])
            ok = int(pred_idx == gt_idx)
            per_source[src]["ok"] += ok
            per_source[src]["n"]  += 1
            task = item.get("task", "unknown")
            per_task.setdefault(task, {"ok": 0, "n": 0})
            per_task[task]["ok"] += ok
            per_task[task]["n"] += 1
            n_seen += 1
            if n_seen % 200 == 0:
                elapsed = time.time() - t0
                so_far = {k: 100.0 * v["ok"] / max(1, v["n"]) for k, v in per_source.items()}
                print(f"  ... {n_seen}/{end_idx - args.start_idx}  ({elapsed:.0f}s)  "
                      f"sg={so_far['sg']:.2f}% eg={so_far['eg']:.2f}% hd={so_far['hd']:.2f}%",
                      flush=True)
        except Exception as e:
            print(f"  err idx={idx}: {e}", flush=True)
            continue

    print("\n=== FINAL FULL EVAL (Minimal-traj) ===", flush=True)
    for k, v in per_source.items():
        acc = 100.0 * v["ok"] / max(1, v["n"])
        print(f"  {k}: {acc:.2f}%  ({v['n']} items)", flush=True)
    total_ok = sum(v["ok"] for v in per_source.values())
    total_n  = sum(v["n"]  for v in per_source.values())
    overall = 100.0 * total_ok / max(1, total_n)
    print(f"  Combined: {overall:.2f}%  ({total_n} items)", flush=True)

    def stats(xs):
        if not xs: return {"n": 0, "mean_ms": 0, "median_ms": 0}
        s = sorted(xs)
        return {"n": len(xs), "mean_ms": 1000.0 * sum(xs)/len(xs),
                "median_ms": 1000.0 * s[len(s)//2]}
    pre_stats = stats(timings_pre); cmp_stats = stats(timings_cmp)
    print("\nPer-item timing (batch=1):", flush=True)
    print(f"  preprocess: mean {pre_stats['mean_ms']:.1f}ms / median {pre_stats['median_ms']:.1f}ms", flush=True)
    print(f"  compute:    mean {cmp_stats['mean_ms']:.1f}ms / median {cmp_stats['median_ms']:.1f}ms", flush=True)

    out = {
        "ckpt": args.ckpt,
        "merge_ratio": args.merge_ratio,
        "per_source": {k: {"acc": 100.0 * v["ok"] / max(1, v["n"]),
                            "ok": v["ok"], "n": v["n"]} for k, v in per_source.items()},
        "combined": {"acc": overall, "ok": total_ok, "n": total_n},
        "per_task": {t: {"acc": 100.0 * v["ok"] / max(1, v["n"]),
                          "ok": v["ok"], "n": v["n"]} for t, v in per_task.items()},
        "timing_per_item_batch1": {"preprocess": pre_stats, "compute": cmp_stats},
    }
    out_path = os.path.splitext(args.ckpt)[0] + ".full_eval.json"
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
