"""HandOracleDeRisk — 3-arm paired oracle de-risk on HD-EPIC (training-free).

Per the Seed: for each HD-EPIC target/control item, run THREE arms that differ
ONLY by injected question text (vision + token selection identical, paired):
  baseline : question as-is
  gt_hand  : GT hand-kinematics text prepended  (handoracle_text.build_hand_text)
  placebo  : same-length scrambled hand text     (build_placebo_text)

One jsonl line per item carries all three predictions + a leakage flag, so
handoracle_analysis.py can run paired McNemar(gt vs placebo) on target and
control with no cross-run pairing ambiguity.

Token selection = content-only VisionZip (dominant 0.05 + contextual 0.05 = 10%),
NO gaze complement (HD-EPIC gaze is absent). Same M1 LoRA for every arm.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.hdepic_dataset import HDEpicDataset
from TrajGazeMerge.models.model import build_merged_inputs, forward_logits, get_option_ids
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens)
from TrajGazeMerge.eval.handoracle_text import (
    build_hand_text, build_placebo_text, leakage_audit)

TARGET = ["fine_grained_action_recognition", "fine_grained_action_localization",
          "fine_grained_how_recognition", "fine_grained_why_recognition"]
CONTROL = ["3d_perception_object_location", "3d_perception_fixture_location",
           "object_motion_stationary_object_localization"]
DOMINANT, CONTEXTUAL = 0.05, 0.05   # content-only 10% budget


def item_key(it) -> str:
    s = "|".join([it["task"], it["question"], "||".join(it["options"]), it["answer"]])
    return hashlib.md5(s.encode()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/"
                                     "visionzip_complement_learned_overlay/best.pth")
    p.add_argument("--dump", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/"
                                     "dumps/handoracle.jsonl")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--n-traj", type=int, default=32)
    p.add_argument("--max-per-task", type=int, default=300,
                   help="cap large tasks (action_rec/loc) for tractable de-risk; "
                        "small tasks (how/why/controls) are always fully included")
    return p.parse_args()


@torch.no_grad()
def run_arm(processor, qwen, base_qwen, option_ids, device, frame_paths, question, options):
    cached = preprocess_visionzip_item(processor, base_qwen, frame_paths, question, options, device)
    if cached is None:
        return None
    content_embeds, content_idx = visionzip_select_tokens(
        cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
        dominant_ratio=DOMINANT, contextual_ratio=CONTEXTUAL)
    inputs = build_merged_inputs(base_qwen, cached, content_embeds, content_idx)
    logits = forward_logits(qwen, inputs)
    n_opt = len(options)
    return int(logits[option_ids[:n_opt]].argmax().item())


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    keep = set(TARGET) | set(CONTROL)

    print(f"[handoracle] loading {args.ckpt}", flush=True)
    processor, qwen = load_visionzip_lora(device)
    base_qwen = qwen.get_base_model()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    qwen.load_state_dict(ckpt["lora_state"], strict=False)
    qwen.eval()
    option_ids = get_option_ids(processor, 5)

    ds = HDEpicDataset(split="test", n_vlm_frames=args.n_frames,
                       n_traj_frames=args.n_traj, simple=False)
    print(f"[handoracle] target={TARGET} control={CONTROL} max_per_task={args.max_per_task}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    per_task_n = {}
    tot = {"base": 0, "gt": 0, "pb": 0}; total = 0; leak_fail = 0
    with open(args.dump, "w") as fout:
        for idx in range(len(ds)):
            try:
                it = ds[idx]
                if it is None or it["task"] not in keep:
                    continue
                if per_task_n.get(it["task"], 0) >= args.max_per_task:
                    continue
                opts = it["options"]; n_opt = len(opts)
                letters = [chr(65 + i) for i in range(n_opt)]
                if it["answer"] not in letters:
                    continue
                gt_idx = letters.index(it["answer"])
                fp, q = it["vlm_frame_paths"], it["question"]

                hand_txt = build_hand_text(it["traj"])
                pb_txt = build_placebo_text(hand_txt, item_key(it))
                leak_ok, viol = leakage_audit(hand_txt, opts, it["answer"])
                if not leak_ok:
                    leak_fail += 1

                base_pred = run_arm(processor, qwen, base_qwen, option_ids, device, fp, q, opts)
                if base_pred is None:
                    continue
                gt_pred = run_arm(processor, qwen, base_qwen, option_ids, device, fp,
                                  hand_txt + "\n\n" + q, opts)
                pb_pred = run_arm(processor, qwen, base_qwen, option_ids, device, fp,
                                  pb_txt + "\n\n" + q, opts)

                ok_b = int(base_pred == gt_idx); ok_g = int(gt_pred == gt_idx); ok_p = int(pb_pred == gt_idx)
                tot["base"] += ok_b; tot["gt"] += ok_g; tot["pb"] += ok_p
                total += 1; per_task_n[it["task"]] = per_task_n.get(it["task"], 0) + 1
                fout.write(json.dumps({
                    "key": item_key(it), "task": it["task"],
                    "group": "target" if it["task"] in TARGET else "control",
                    "gt": gt_idx, "n_opt": n_opt,
                    "pred_base": base_pred, "pred_gt": gt_pred, "pred_pb": pb_pred,
                    "ok_base": ok_b, "ok_gt": ok_g, "ok_pb": ok_p,
                    "leak_ok": int(leak_ok), "leak_viol": viol[:3],
                }) + "\n")
                fout.flush()
                if total % 25 == 0:
                    print(f"  [{total}] base={100*tot['base']/total:.2f} "
                          f"gt={100*tot['gt']/total:.2f} pb={100*tot['pb']/total:.2f} "
                          f"leak_fail={leak_fail}", flush=True)
            except Exception as e:
                print(f"  idx={idx} ERR {e!r}", flush=True)

    print(f"\n[handoracle] n={total}  base={100*tot['base']/max(1,total):.2f} "
          f"gt={100*tot['gt']/max(1,total):.2f} pb={100*tot['pb']/max(1,total):.2f}  "
          f"leak_fail={leak_fail}  → {args.dump}", flush=True)
    print(f"per-task n: {per_task_n}", flush=True)


if __name__ == "__main__":
    main()
