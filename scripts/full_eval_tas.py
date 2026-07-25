"""Full per-source eval for TAS Stage-2 ckpt — runs the same selection rule
(TrajGazeV2Temporal encoder → score → gaze_weighted_merge) on the full 4936-item
CombinedMergeDataset(test). Mirrors the trainer's evaluate_per_source but with
no max_items cap.

Usage:
    CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/trajgaze/bin/python -m scripts.full_eval_tas \\
        --stage2-ckpt /workspace/.../tas_lora_3way_v3upright/best.pth \\
        --stage1-ckpt /workspace/.../stage1_tas_v3_upright/best.pth \\
        --merge-ratio 0.9
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
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal, score_to_qwen_spatiotemporal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-ckpt", required=True)
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--merge-ratio", type=float, default=0.9)
    p.add_argument("--model-type", default="full")
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[full_eval_tas] loading models on cuda:{args.gpu}", flush=True)
    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    option_ids = get_option_ids(processor, 5)

    print(f"[full_eval_tas] loading traj encoder ({args.model_type})", flush=True)
    traj_encoder = load_traj_encoder(
        args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes
    )

    print(f"[full_eval_tas] loading Stage-2 LoRA: {args.stage2_ckpt}", flush=True)
    ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    if "lora_state" in ckpt:
        qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
    if "encoder_state" in ckpt:
        traj_encoder.load_state_dict(ckpt["encoder_state"], strict=False)

    qwen_model.eval()
    traj_encoder.eval()

    print("[full_eval_tas] loading full 3-way test set", flush=True)
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
        include_hdepic=True,
    )
    print(f"[full_eval_tas] {len(test_ds)} test items", flush=True)

    per_source = {"sg": {"ok": 0, "n": 0}, "eg": {"ok": 0, "n": 0}, "hd": {"ok": 0, "n": 0}}
    per_task: dict[str, dict] = {}
    # Per-item timing (seconds), batch size 1
    timings_preprocess: list[float] = []   # load frames + ViT + tokenize
    timings_compute:    list[float] = []   # encoder + selection + VLM forward
    t0 = time.time()
    n_seen = 0

    # CombinedMergeDataset concatenation order: SG (526) + EG (485) + HD (3925)
    SG_END = 526
    EG_END = SG_END + 485   # 1011

    def src_for_idx(i: int) -> str:
        if i < SG_END: return "sg"
        if i < EG_END: return "eg"
        return "hd"

    end_idx = len(test_ds) if args.end_idx < 0 else args.end_idx
    for idx in range(args.start_idx, end_idx):
        try:
            item = test_ds[idx]
        except Exception:
            continue
        if item is None: continue
        src_key = src_for_idx(idx)

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

            n_video = cached["video_embeds"].shape[0]
            T_merged = int(cached["grid_thw"][0, 0].item())
            n_spatial = n_video // max(1, T_merged)
            r = max(1, int(args.merge_ratio * n_video))

            scores = get_patch_scores_temporal(traj_encoder, item, device)
            scores_all = score_to_qwen_spatiotemporal(scores, n_spatial, T_merged)
            if scores_all.shape[0] != n_video:
                if scores_all.shape[0] > n_video:
                    scores_all = scores_all[:n_video]
                else:
                    rep = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                    scores_all = scores_all.repeat(rep)[:n_video]

            merged_video, receiver_idx = gaze_weighted_merge(
                cached["video_embeds"], scores_all, r
            )
            logits = forward_logits(
                qwen_model,
                build_merged_inputs(base_qwen, cached, merged_video, receiver_idx),
            )
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
                print(f"  ... {n_seen}/{len(test_ds)}  ({elapsed:.0f}s)  "
                      f"sg={so_far['sg']:.2f}% eg={so_far['eg']:.2f}% hd={so_far['hd']:.2f}%",
                      flush=True)
        except Exception as e:
            print(f"  err idx={idx}: {e}", flush=True)
            continue

    print("\n=== FINAL FULL EVAL ===", flush=True)
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

    # Timing summary
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
    print(f"  compute(encoder+select+VLM):   mean {cmp_stats['mean_ms']:.1f}ms / median {cmp_stats['median_ms']:.1f}ms  (n={cmp_stats['n']})", flush=True)

    # Persist to JSON
    out = {
        "stage2_ckpt": args.stage2_ckpt,
        "stage1_ckpt": args.stage1_ckpt,
        "merge_ratio": args.merge_ratio,
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
    out_path = os.path.splitext(args.stage2_ckpt)[0] + ".full_eval.json"
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
