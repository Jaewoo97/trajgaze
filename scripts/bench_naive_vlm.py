"""Naive VLM latency benchmark: no token reduction, all visual tokens fed to
Qwen2.5-VL-7B. Same dataset / preprocess as VisionZip and TAS evals so the
per-item timing is directly comparable. Sample size kept small (default 50)
because each item takes 1-5s."""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.combined_simple_dataset import CombinedSimpleDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"[naive_bench] loading Qwen2.5-VL-7B on cuda:{args.gpu}", flush=True)
    processor, model = load_qwen_lora(device)
    base_qwen = model.get_base_model()
    option_ids = get_option_ids(processor, 5)
    model.eval()

    test_ds = CombinedSimpleDataset(split="test", n_vlm_frames=128)
    print(f"[naive_bench] test_ds size = {len(test_ds)}", flush=True)

    timings_pre, timings_cmp = [], []
    n_tokens_seen = []
    n_done = 0
    n_correct = 0
    idx = args.start_idx
    while n_done < args.n_items and idx < len(test_ds):
        try:
            item = test_ds[idx]
            idx += 1
            if item is None:
                continue
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters:
                continue

            t_pre0 = time.perf_counter()
            cached = preprocess_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device,
            )
            torch.cuda.synchronize(device)
            t_pre1 = time.perf_counter()
            if cached is None:
                continue

            # NAIVE: skip selection. Feed all video_embeds + identity receiver_idx.
            N = cached["video_embeds"].shape[0]
            receiver_idx = torch.arange(N, device=device)
            inputs_dict = build_merged_inputs(
                base_qwen, cached, cached["video_embeds"], receiver_idx
            )
            logits = forward_logits(model, inputs_dict)
            torch.cuda.synchronize(device)
            t_cmp1 = time.perf_counter()

            timings_pre.append(t_pre1 - t_pre0)
            timings_cmp.append(t_cmp1 - t_pre1)
            n_tokens_seen.append(N)
            pred_idx = logits[option_ids[:n_opt]].argmax().item()
            gt_idx = letters.index(item["answer"])
            n_correct += int(pred_idx == gt_idx)
            n_done += 1
            if n_done % 5 == 0:
                pre_ms = 1000 * timings_pre[-1]
                cmp_ms = 1000 * timings_cmp[-1]
                print(f"  ... {n_done}/{args.n_items}  idx={idx-1}  "
                      f"N_tok={N}  pre={pre_ms:.0f}ms cmp={cmp_ms:.0f}ms",
                      flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at idx={idx-1}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  err idx={idx-1}: {e}", flush=True)
            continue

    def stats(xs):
        if not xs: return (0,0,0)
        return (sum(xs)/len(xs), sorted(xs)[len(xs)//2], min(xs), max(xs))

    pre_mean, pre_med, pre_min, pre_max = (*stats(timings_pre),)[:4]
    cmp_mean, cmp_med, cmp_min, cmp_max = (*stats(timings_cmp),)[:4]
    tok_mean = sum(n_tokens_seen)/len(n_tokens_seen)

    print()
    print("=" * 60, flush=True)
    print(f"NAIVE VLM (FULL TOKENS) — n={n_done}", flush=True)
    print(f"  mean tokens kept = {tok_mean:.0f}  (vs 627 in 10% methods)", flush=True)
    print(f"  preprocess (load+ViT+tokenize):", flush=True)
    print(f"    mean {1000*pre_mean:.0f}ms / median {1000*pre_med:.0f}ms  "
          f"[{1000*pre_min:.0f}-{1000*pre_max:.0f}]", flush=True)
    print(f"  compute (VLM forward only, no selection):", flush=True)
    print(f"    mean {1000*cmp_mean:.0f}ms / median {1000*cmp_med:.0f}ms  "
          f"[{1000*cmp_min:.0f}-{1000*cmp_max:.0f}]", flush=True)
    print(f"  acc on this subset: {100*n_correct/max(1,n_done):.2f}%  (no LoRA-finetuned for full-token)", flush=True)


if __name__ == "__main__":
    main()
