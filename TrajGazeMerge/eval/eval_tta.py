"""Axis-1 variance reduction: test-time option-order symmetrization (+ bias report).

A 2-way MCQ model usually has a POSITION bias (prefers 'A' or 'B' regardless of
content). That bias is pure variance on top of the real signal. Symmetrization
removes it: present the options in every cyclic order, un-permute each forward's
option logits back to canonical option indices, and average. The averaged argmax
is position-invariant.

This needs NO retraining — it runs on an existing checkpoint and reuses
select_complementary, so it works for M1 / query_gaze / any complement config.

Reports, per task and overall:
  - base   : single forward, canonical order (argmax) — the usual number
  - sym    : order-symmetrized (averaged over n_opt cyclic shifts)
  - posA%  : how often the model picks POSITION 0 ('A'). Compare to gtA% (how often
             the correct answer is at position 0). |posA% - gtA%| is the bias sym removes.

Usage (single GPU; --max-items for a quick read):
    python -m TrajGazeMerge.eval.eval_tta --gpu 3 --max-items 100 \
        --ckpt .../visionzip_complement_learned_overlay/best.pth \
        --complement-mode topk --content-ratio 0.07 --traj-ratio 0.03
"""
from __future__ import annotations

import argparse
import re
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
_PREFIX = re.compile(r"^[A-E][\.\)]\s*")


def _strip(opt: str) -> str:
    """Remove a leading 'A. ' / 'B) ' style label so options can be re-lettered."""
    return _PREFIX.sub("", opt.strip())


def _relabel(texts: list[str]) -> list[str]:
    return [f"{chr(65 + j)}. {t}" for j, t in enumerate(texts)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0, help="0 = full set")
    p.add_argument("--symmetrize", choices=["order", "none"], default="order")
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    p.add_argument("--traj-pool-mode", choices=["learned", "anticipatory"], default="learned")
    p.add_argument("--complement-mode", choices=["topk", "coverage", "fusion", "query_gaze"],
                   default="topk")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio",    type=float, default=0.03)
    p.add_argument("--query-ratio",   type=float, default=0.0)
    p.add_argument("--query-mode",    choices=["cosine", "random", "shuffle"], default="cosine")
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--include-hdepic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0, alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)

    print(f"[tta] loading {args.ckpt}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    qwen.load_state_dict(ckpt["lora_state"], strict=False)
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
    n_total = len(ds) if args.max_items <= 0 else min(args.max_items, len(ds))
    print(f"[tta] symmetrize={args.symmetrize}  items={n_total}", flush=True)

    # counters
    c_base = c_sym = total = 0
    pickA_base = gt_at_A = 0
    by_task = {}   # task -> [n, base_correct, sym_correct]

    def run_once(item, options_text_list):
        """Forward with a given option ORDER; return option logits over positions."""
        cached = preprocess_visionzip_item(
            processor, base_qwen, item["vlm_frame_paths"],
            item["question"], options_text_list, device)
        if cached is None:
            return None
        sel, recv = select_complementary(
            cached, item, device, args.traj_pool_mode, encoder, hp,
            args.content_ratio, args.traj_ratio,
            complement_mode=args.complement_mode, query_ratio=args.query_ratio,
            query_mode=args.query_mode)
        inp = build_merged_inputs(base_qwen, cached, sel, recv)
        logits = forward_logits(qwen, inp)
        return logits

    with torch.no_grad():
        for idx in range(n_total):
            try:
                item = ds[idx]
                if item is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                gt = letters.index(item["answer"])           # canonical gt index
                texts = [_strip(o) for o in item["options"]]  # canonical option texts

                # canonical order (= baseline) + cyclic shifts for symmetrization
                shifts = range(n_opt) if args.symmetrize == "order" else [0]
                canon_logits = torch.zeros(n_opt, device=device)
                base_pos_logits = None
                ok = True
                for s in shifts:
                    perm = [(j + s) % n_opt for j in range(n_opt)]    # position j shows canonical perm[j]
                    ordered = _relabel([texts[perm[j]] for j in range(n_opt)])
                    lg = run_once(item, ordered)
                    if lg is None:
                        ok = False
                        break
                    pos_logits = lg[option_ids[:n_opt]].float()       # logit per POSITION
                    for j in range(n_opt):
                        canon_logits[perm[j]] += pos_logits[j]        # un-permute → canonical
                    if s == 0:
                        base_pos_logits = pos_logits
                if not ok or base_pos_logits is None:
                    continue

                pred_base = int(base_pos_logits.argmax().item())      # position-space (canonical order)
                pred_sym = int(canon_logits.argmax().item())          # canonical-space, symmetrized
                total += 1
                c_base += int(pred_base == gt)
                c_sym += int(pred_sym == gt)
                pickA_base += int(pred_base == 0)
                gt_at_A += int(gt == 0)
                t = item["task"]
                rec = by_task.setdefault(t, [0, 0, 0])
                rec[0] += 1; rec[1] += int(pred_base == gt); rec[2] += int(pred_sym == gt)
            except Exception as e:
                print(f"[tta] idx={idx} ERR {e!r}", flush=True)
                continue

    def pct(a, b): return 100.0 * a / max(1, b)
    print(f"\n[tta] n={total}", flush=True)
    print(f"  base (single)      : {pct(c_base, total):.2f}%", flush=True)
    print(f"  sym  (order-avg)   : {pct(c_sym, total):.2f}%   (Δ {pct(c_sym,total)-pct(c_base,total):+.2f})", flush=True)
    print(f"  posA% (model picks 0) = {pct(pickA_base, total):.1f}%   "
          f"gtA% (correct at 0) = {pct(gt_at_A, total):.1f}%   "
          f"→ bias = {pct(pickA_base,total)-pct(gt_at_A,total):+.1f}pp", flush=True)
    print("  per-task (n | base → sym):", flush=True)
    for t, (n, cb, cs) in sorted(by_task.items()):
        print(f"    {t:42s} n={n:4d}  {pct(cb,n):6.2f} → {pct(cs,n):6.2f}  ({pct(cs,n)-pct(cb,n):+.2f})", flush=True)


if __name__ == "__main__":
    main()
