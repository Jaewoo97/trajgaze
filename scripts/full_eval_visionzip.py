"""Full per-source eval for VisionZip best.pth — same protocol as TAS full eval
but using VisionZip's attention-based selection. Output JSON next to ckpt."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# /workspace/trajgaze_st must be first so TrajGazeMerge resolves to the current
# branch; train_visionzip_lora self-inserts the VisionZip Qwen2_5_VL path.
sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.combined_simple_dataset import CombinedSimpleDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    p.add_argument("--out", default="", help="override output JSON path (for sharded runs)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[full_eval_visionzip] loading VisionZip Qwen2.5-VL-7B + LoRA on cuda:{args.gpu}",
          flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    option_ids = get_option_ids(processor, 5)

    print(f"[full_eval_visionzip] loading LoRA ckpt: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "lora_state" in ckpt:
        model.load_state_dict(ckpt["lora_state"], strict=False)
    model.eval()

    print("[full_eval_visionzip] loading full 3-way test set", flush=True)
    test_ds = CombinedSimpleDataset(split="test", n_vlm_frames=128)
    print(f"[full_eval_visionzip] {len(test_ds)} items", flush=True)

    # SG (526) + EG (485) + HD (3925) concatenation
    SG_END = 526
    EG_END = SG_END + 485

    def src_for_idx(i):
        if i < SG_END: return "sg"
        if i < EG_END: return "eg"
        return "hd"

    per_source = {"sg": {"ok": 0, "n": 0}, "eg": {"ok": 0, "n": 0}, "hd": {"ok": 0, "n": 0}}
    per_task: dict[str, dict] = {}
    timings_preprocess: list[float] = []
    timings_compute:    list[float] = []
    end_idx = len(test_ds) if args.end_idx < 0 else args.end_idx
    t0 = time.time()
    n_seen = 0

    for idx in range(args.start_idx, end_idx):
        try:
            item = test_ds[idx]
        except Exception:
            continue
        if item is None: continue
        src_key = src_for_idx(idx)
        try:
            t_pre0 = time.perf_counter()
            cached = preprocess_visionzip_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device,
            )
            torch.cuda.synchronize(device)
            t_pre1 = time.perf_counter()
            if cached is None: continue
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters: continue

            selected_embeds, receiver_idx = visionzip_select_tokens(
                cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
            )
            inputs_dict = build_merged_inputs(
                base_qwen, cached, selected_embeds, receiver_idx
            )
            logits = forward_logits(model, inputs_dict)
            torch.cuda.synchronize(device)
            t_cmp1 = time.perf_counter()
            timings_preprocess.append(t_pre1 - t_pre0)
            timings_compute.append(t_cmp1 - t_pre1)
            pred_idx = logits[option_ids[:n_opt]].argmax().item()
            gt_idx = letters.index(item["answer"])
            ok = int(pred_idx == gt_idx)
            per_source[src_key]["ok"] += ok
            per_source[src_key]["n"] += 1
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

    print("\n=== FINAL FULL EVAL (VisionZip) ===", flush=True)
    for k, v in per_source.items():
        acc = 100.0 * v["ok"] / max(1, v["n"])
        print(f"  {k}: {acc:.2f}%  ({v['n']} items)", flush=True)
    total_ok = sum(v["ok"] for v in per_source.values())
    total_n = sum(v["n"] for v in per_source.values())
    overall = 100.0 * total_ok / max(1, total_n)
    print(f"  Combined: {overall:.2f}%  ({total_n} items)", flush=True)

    print("\nPer-task:", flush=True)
    for t, v in sorted(per_task.items()):
        acc = 100.0 * v["ok"] / max(1, v["n"])
        print(f"  {t}: {acc:.2f}%  ({v['n']})", flush=True)

    def stats(xs):
        if not xs: return {"n": 0, "mean_ms": 0, "median_ms": 0}
        sorted_xs = sorted(xs)
        return {
            "n": len(xs),
            "mean_ms": 1000.0 * sum(xs) / len(xs),
            "median_ms": 1000.0 * sorted_xs[len(sorted_xs) // 2],
        }
    pre_stats = stats(timings_preprocess)
    cmp_stats = stats(timings_compute)
    print("\nPer-item timing (batch=1):", flush=True)
    print(f"  preprocess(load+ViT+tokenize): mean {pre_stats['mean_ms']:.1f}ms / median {pre_stats['median_ms']:.1f}ms  (n={pre_stats['n']})", flush=True)
    print(f"  compute(select+VLM):           mean {cmp_stats['mean_ms']:.1f}ms / median {cmp_stats['median_ms']:.1f}ms  (n={cmp_stats['n']})", flush=True)

    out = {
        "ckpt": args.ckpt,
        "per_source": {k: {"acc": 100.0 * v["ok"] / max(1, v["n"]),
                            "ok": v["ok"], "n": v["n"]} for k, v in per_source.items()},
        "combined": {"acc": overall, "ok": total_ok, "n": total_n},
        "per_task": {t: {"acc": 100.0 * v["ok"] / max(1, v["n"]),
                          "ok": v["ok"], "n": v["n"]} for t, v in per_task.items()},
        "timing_per_item_batch1": {
            "preprocess": pre_stats,
            "compute": cmp_stats,
        },
    }
    out_path = args.out if args.out else (os.path.splitext(args.ckpt)[0] + ".full_eval.json")
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
