"""Per-source full eval of a Temporal-Budget checkpoint (SG / EG / HD-EPIC / Combined).

Mirrors scripts/eval_qcgate_persource.py but for the zero-param temporal-budget
selection (no controller). Reuses the EXACT training-time selection path
(temporal_budget_select_tokens → build_merged_inputs → forward_logits) over the
SAME full val set (CombinedMergeDataset split=test), broken down by source via
ds.items[idx][0] ∈ {sg, eg, hd}.

tau / traj_weight / sigma_v / sigma_gh default to the values stored in the ckpt
(falling back to the trainer defaults), and can be overridden on the CLI.

GAZE_OVERLAY must match training (overlay run => "1"); set it before running.

Usage:
    GAZE_OVERLAY=1 python -m scripts.eval_tbudget_persource \\
      --ckpt /workspace/.../tbudget_lora_3way_overlay/epoch_03.pth --gpu 0 \\
      --out  /workspace/.../tbudget_lora_3way_overlay/epoch_03.full_eval.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.temporal_budget import temporal_budget_select_tokens
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)

SRC_NAME = {"sg": "StreamGaze(egtea)", "eg": "EgoGazeVQA(egtea)", "hd": "HD-EPIC(P09)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--traj-weight", type=float, default=None)
    p.add_argument("--sigma-v", type=float, default=None)
    p.add_argument("--sigma-gh", type=float, default=None)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or (os.path.splitext(args.ckpt)[0] + ".full_eval.json")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    overlay = os.environ.get("GAZE_OVERLAY", "1")

    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()

    ckpt = torch.load(args.ckpt, map_location=device)
    miss, unexp = model.load_state_dict(ckpt["lora_state"], strict=False)
    miss = [k for k in miss if "lora" in k.lower()]
    # hp: CLI override > ckpt-stored > trainer default
    tau         = args.tau         if args.tau         is not None else ckpt.get("tau", 1.0)
    traj_weight = args.traj_weight if args.traj_weight is not None else ckpt.get("traj_weight", 0.5)
    sigma_v     = args.sigma_v     if args.sigma_v     is not None else ckpt.get("sigma_v", 0.05)
    sigma_gh    = args.sigma_gh    if args.sigma_gh    is not None else ckpt.get("sigma_gh", 0.10)
    print(f"[tbudget-eval] ckpt={args.ckpt}", flush=True)
    print(f"[tbudget-eval] gpu={args.gpu} GAZE_OVERLAY={overlay} "
          f"tau={tau} traj_weight={traj_weight} sigma_v={sigma_v} sigma_gh={sigma_gh}", flush=True)
    print(f"[tbudget-eval] loaded ckpt (epoch={ckpt.get('epoch')} acc_in_ckpt={ckpt.get('acc')}); "
          f"lora-missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if miss:
        print(f"[tbudget-eval] WARNING missing LoRA keys (first 5): {miss[:5]}", flush=True)

    option_ids = get_option_ids(processor, 5)
    ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True,
    )
    model.eval()

    by_src = {}
    by_task = {}
    correct = 0; total = 0
    t0 = time.time()

    with torch.no_grad():
        for idx in range(args.start_idx, len(ds) if args.end_idx < 0 else args.end_idx):
            src = ds.items[idx][0]
            item = ds[idx]
            if item is None:
                continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue

                sel_embeds, recv_idx, _ = temporal_budget_select_tokens(
                    cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
                    cached["grid_thw"], item["traj"],
                    tau=tau, traj_weight=traj_weight, sigma_v=sigma_v, sigma_gh=sigma_gh,
                )
                inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                logits = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)

                correct += ok; total += 1
                by_src.setdefault(src, []).append(ok)
                by_task.setdefault(item["task"], []).append(ok)
            except Exception as e:
                if total < 3:
                    print(f"[tbudget-eval] item {idx} err: {e}", flush=True)

            if (idx + 1) % args.log_every == 0:
                run = " ".join(
                    f"{k}={100.0*sum(v)/len(v):.1f}%({len(v)})"
                    for k, v in sorted(by_src.items())
                )
                el = time.time() - t0
                print(f"[tbudget-eval] {idx+1}/{len(ds)} "
                      f"overall={100.0*correct/max(1,total):.2f}% | {run} | t={el:.0f}s",
                      flush=True)

    per_source = {
        k: {"acc": 100.0 * sum(v) / max(1, len(v)), "ok": sum(v), "n": len(v)}
        for k, v in sorted(by_src.items())
    }
    per_task = {
        t: {"acc": 100.0 * sum(v) / max(1, len(v)), "ok": sum(v), "n": len(v)}
        for t, v in sorted(by_task.items())
    }
    result = {
        "ckpt": args.ckpt,
        "gaze_overlay": overlay,
        "epoch_in_ckpt": ckpt.get("epoch"),
        "tau": tau, "traj_weight": traj_weight, "sigma_v": sigma_v, "sigma_gh": sigma_gh,
        "overall_acc": 100.0 * correct / max(1, total),
        "n": total,
        "per_source": per_source,
        "per_task": per_task,
        "elapsed_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n========== PER-SOURCE FULL EVAL (temporal-budget) ==========", flush=True)
    print(f"ckpt: {args.ckpt}  (epoch {ckpt.get('epoch')}, GAZE_OVERLAY={overlay})", flush=True)
    for k, v in per_source.items():
        print(f"  {SRC_NAME.get(k, k):20s} {v['acc']:.2f}%  (n={v['n']})", flush=True)
    print(f"  {'Combined':20s} {result['overall_acc']:.2f}%  (n={total})", flush=True)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
