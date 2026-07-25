"""Per-source full eval of a QC-Gate checkpoint (SG / EG / HD-EPIC / Combined).

The in-script trainer eval (`evaluate()` in train_visionzip_qcgate_lora_3way.py)
groups only by task, not by source. This standalone single-GPU eval reuses the
EXACT same controller-aware selection path (qcgate_select → build_merged_inputs →
forward_logits) and the SAME full val set (CombinedMergeDataset split=test), but
breaks accuracy down by source via ds.items[idx][0] ∈ {sg, eg, hd}.

GAZE_OVERLAY is honored by the dataset adapters at import time — set it before
running to match the checkpoint's training condition (overlay run => "1").

Usage:
    GAZE_OVERLAY=1 python -m scripts.eval_qcgate_persource \\
      --ckpt /workspace/.../qcgate_lora_3way_overlay/best.pth --gpu 0 \\
      --out  /workspace/.../qcgate_lora_3way_overlay/best.full_eval.json
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
from TrajGazeMerge.models.qc_gate import QCGateController
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_qcgate_lora_3way import qcgate_select

SRC_NAME = {"sg": "StreamGaze(egtea)", "eg": "EgoGazeVQA(egtea)", "hd": "HD-EPIC(P09)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSON output path (default: <ckpt>.full_eval.json)")
    # kernel scales — MUST match training (trainer defaults; overlay run used defaults)
    p.add_argument("--sigma-g",  type=float, default=2.0)
    p.add_argument("--sigma-h",  type=float, default=3.0)
    p.add_argument("--sigma-v",  type=float, default=0.05)
    p.add_argument("--sigma-gh", type=float, default=0.10)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or (os.path.splitext(args.ckpt)[0] + ".full_eval.json")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    overlay = os.environ.get("GAZE_OVERLAY", "1")
    print(f"[persrc-eval] ckpt={args.ckpt}", flush=True)
    print(f"[persrc-eval] gpu={args.gpu}  GAZE_OVERLAY={overlay}  out={out_path}", flush=True)

    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    controller = QCGateController().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    miss, unexp = model.load_state_dict(ckpt["lora_state"], strict=False)
    miss = [k for k in miss if "lora" in k.lower()]      # frozen base keys are fine to skip
    controller.load_state_dict(ckpt["controller_state"])
    print(f"[persrc-eval] loaded ckpt (epoch={ckpt.get('epoch')} "
          f"acc_in_ckpt={ckpt.get('acc')}); lora-missing={len(miss)} unexpected={len(unexp)}",
          flush=True)
    if miss:
        print(f"[persrc-eval] WARNING missing LoRA keys (first 5): {miss[:5]}", flush=True)

    option_ids = get_option_ids(processor, 5)
    hp = dict(sigma_g=args.sigma_g, sigma_h=args.sigma_h,
              sigma_v=args.sigma_v, sigma_gh=args.sigma_gh)

    ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True,
    )
    model.eval()
    controller.eval()

    by_src = {}
    by_task = {}
    g_sum = [0.0, 0.0, 0.0]
    g_n = 0
    correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for idx in range(len(ds)):
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

                sel_embeds, recv_idx, gm = qcgate_select(
                    cached, item, controller, base_qwen, device, hp, training=False
                )
                inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                logits = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)

                correct += ok
                total += 1
                by_src.setdefault(src, []).append(ok)
                by_task.setdefault(item["task"], []).append(ok)
                for i in range(3):
                    g_sum[i] += gm[i]
                g_n += 1
            except Exception as e:
                if total < 3:
                    print(f"[persrc-eval] item {idx} err: {e}", flush=True)

            if (idx + 1) % args.log_every == 0:
                run = " ".join(
                    f"{k}={100.0*sum(v)/len(v):.1f}%({len(v)})"
                    for k, v in sorted(by_src.items())
                )
                el = time.time() - t0
                print(f"[persrc-eval] {idx+1}/{len(ds)} "
                      f"overall={100.0*correct/max(1,total):.2f}% | {run} | t={el:.0f}s",
                      flush=True)

    per_source = {
        k: {"acc": 100.0 * sum(v) / max(1, len(v)), "n": len(v)}
        for k, v in sorted(by_src.items())
    }
    per_task = {
        t: {"acc": 100.0 * sum(v) / max(1, len(v)), "n": len(v)}
        for t, v in sorted(by_task.items())
    }
    gates_mean = [g_sum[i] / max(1, g_n) for i in range(3)]
    result = {
        "ckpt": args.ckpt,
        "gaze_overlay": overlay,
        "epoch_in_ckpt": ckpt.get("epoch"),
        "overall_acc": 100.0 * correct / max(1, total),
        "n": total,
        "per_source": per_source,
        "per_task": per_task,
        "gates_mean": {"a_gaze": gates_mean[0], "a_hand": gates_mean[1], "a_temp": gates_mean[2]},
        "elapsed_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n========== PER-SOURCE FULL EVAL ==========", flush=True)
    print(f"ckpt: {args.ckpt}  (epoch {ckpt.get('epoch')}, GAZE_OVERLAY={overlay})", flush=True)
    for k, v in per_source.items():
        print(f"  {SRC_NAME.get(k, k):20s} {v['acc']:.2f}%  (n={v['n']})", flush=True)
    print(f"  {'Combined':20s} {result['overall_acc']:.2f}%  (n={total})", flush=True)
    print(f"  gates(mean): a_gaze={gates_mean[0]:+.3f} a_hand={gates_mean[1]:+.3f} "
          f"a_temp={gates_mean[2]:+.3f}", flush=True)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
