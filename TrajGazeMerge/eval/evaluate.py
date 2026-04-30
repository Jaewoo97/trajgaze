"""
Evaluate TrajGazeMerge (and baselines) on StreamGaze_v2 egtea test split.

Conditions evaluated per run:
  --condition baseline_frozen : frozen Qwen, full tokens (zero-shot upper bound)
  --condition baseline_lora   : Qwen + LoRA, full tokens (trained baseline)
  --condition merge_frozen    : frozen Qwen, gaze-merged tokens (zero-shot TrajGazeMerge)
  --condition merge_lora      : Qwen + LoRA, gaze-merged tokens (full TrajGazeMerge)

Usage:
    # Zero-shot merge (no training needed):
    /opt/conda/envs/gaze/bin/python -m TrajGazeMerge.eval.evaluate \
        --condition merge_frozen --gpu 0 \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth

    # Trained baseline LoRA:
    /opt/conda/envs/gaze/bin/python -m TrajGazeMerge.eval.evaluate \
        --condition baseline_lora --gpu 0 \
        --lora-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth

    # Trained merge + LoRA:
    /opt/conda/envs/gaze/bin/python -m TrajGazeMerge.eval.evaluate \
        --condition merge_lora --gpu 0 \
        --lora-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora/best.pth \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

import torch

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge, score_to_qwen_spatial
from TrajGazeMerge.models.model import (
    load_qwen_lora, load_qwen_frozen, get_option_ids,
    preprocess_item, build_full_inputs, build_merged_inputs, forward_logits,
)

RESULTS_DIR = "/workspace/EgoGazeVQA/TrajGazeMerge/eval_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition",    required=True,
                   choices=["baseline_frozen", "baseline_lora",
                            "merge_frozen", "merge_lora"])
    p.add_argument("--gpu",          type=int,   default=0)
    p.add_argument("--merge-ratio",  type=float, default=0.5)
    p.add_argument("--n-frames",     type=int,   default=128)
    p.add_argument("--n-traj-frames", type=int,  default=32)
    p.add_argument("--stage1-ckpt",
        default="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth")
    p.add_argument("--lora-ckpt",    default=None,
        help="Path to saved LoRA checkpoint (for baseline_lora / merge_lora)")
    p.add_argument("--model",        default="full",
        choices=["full", "gaze-only"],
        help="full = TrajGazeV2 (4 tokens) | gaze-only = TrajGazeV2GazeOnly (1 token)")
    return p.parse_args()


def extract_answer(response: str) -> str:
    response = response.strip()
    for pat in [r"^([A-D])\b", r"\b([A-D])\b", r"\(([A-D])\)", r"Answer[:\s]+([A-D])"]:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def compute_metrics(results: list[dict]) -> dict:
    by_task: dict[str, list] = {}
    correct_all = 0
    for r in results:
        ok = r.get("correct", False)
        correct_all += int(ok)
        by_task.setdefault(r["task"], []).append(ok)
    overall = 100.0 * correct_all / max(1, len(results))
    per_task = {
        t: {"accuracy": 100.0 * sum(f) / max(1, len(f)), "n_samples": len(f)}
        for t, f in sorted(by_task.items())
    }
    return {"overall": {"accuracy": overall, "n_samples": len(results)},
            "per_task": per_task}


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    needs_lora   = args.condition in ("baseline_lora", "merge_lora")
    needs_merge  = args.condition in ("merge_frozen", "merge_lora")
    needs_frozen = args.condition in ("baseline_frozen", "merge_frozen")

    # ── Load Qwen ─────────────────────────────────────────────────────────────
    if needs_lora:
        print("Loading Qwen + LoRA ...")
        processor, qwen_model = load_qwen_lora(device)
        base_qwen = qwen_model.get_base_model()
        if args.lora_ckpt and os.path.exists(args.lora_ckpt):
            ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
            qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
            print(f"Loaded LoRA from: {args.lora_ckpt}")
    else:
        print("Loading frozen Qwen ...")
        processor, qwen_model = load_qwen_frozen(device)
        base_qwen = qwen_model

    qwen_model.eval()

    # ── Load TrajGaze encoder ─────────────────────────────────────────────────
    traj_encoder = None
    if needs_merge:
        if args.model == "gaze-only":
            from TrajGaze_v2.models.model_gaze_only import TrajGazeV2GazeOnly
            traj_encoder = TrajGazeV2GazeOnly().to(device)
        else:
            from TrajGaze_v2.models.model import TrajGazeV2
            traj_encoder = TrajGazeV2().to(device)
        if os.path.exists(args.stage1_ckpt):
            ckpt  = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
            state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
            # For merge_lora, also load encoder state from lora_ckpt
            if args.condition == "merge_lora" and args.lora_ckpt and os.path.exists(args.lora_ckpt):
                merge_ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
                if "encoder_state" in merge_ckpt:
                    state = merge_ckpt["encoder_state"]
                    print(f"Loaded encoder from merge ckpt: {args.lora_ckpt}")
            traj_encoder.load_state_dict(state, strict=False)
            print(f"TrajGaze encoder loaded from: {args.stage1_ckpt}")
        traj_encoder.eval()

    option_ids = get_option_ids(processor)

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = StreamGazeMergeDataset(
        split="test", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    print(f"Test items: {len(test_ds)}")

    results = []
    for idx, item in enumerate(test_ds):
        if item is None:
            continue
        try:
            cached = preprocess_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device
            )
            if cached is None:
                raise RuntimeError("preprocess returned None")

            if needs_merge:
                # Compute gaze-weighted merge
                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)
                r         = max(1, int(args.merge_ratio * n_video))

                with torch.no_grad():
                    traj_batch = {k: v.unsqueeze(0).to(device)
                                  for k, v in item["traj"].items()}
                    q_emb   = traj_encoder.query_encoder([item["question"]], device)
                    vis_f   = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
                    scores, _ = traj_encoder.encoder(traj_batch, q_emb, vis_f)
                    scores = scores.squeeze(0)   # (196,)

                    scores_q   = score_to_qwen_spatial(scores, n_spatial)
                    scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                    if scores_all.shape[0] != n_video:
                        reps = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        scores_all = scores_all.repeat(reps)[:n_video]

                    merged_video, receiver_idx = gaze_weighted_merge(
                        cached["video_embeds"], scores_all, r
                    )
                    inputs_dict = build_merged_inputs(
                        base_qwen, cached, merged_video, receiver_idx
                    )
            else:
                with torch.no_grad():
                    inputs_dict = build_full_inputs(base_qwen, cached)

            with torch.no_grad():
                if needs_lora and needs_frozen:
                    # Should not happen given choices, but guard anyway
                    logits = forward_logits(qwen_model, inputs_dict)
                elif needs_lora:
                    logits = forward_logits(qwen_model, inputs_dict)
                else:
                    logits = forward_logits(qwen_model, inputs_dict)

            option_logits = logits[option_ids]
            pred_idx  = option_logits.argmax().item()
            pred_char = "ABCD"[pred_idx]
            correct   = (pred_char == item["answer"])

            results.append({
                "task":       item["task"],
                "question":   item["question"],
                "answer":     item["answer"],
                "prediction": pred_char,
                "correct":    correct,
            })

            if (idx + 1) % 50 == 0:
                n_correct = sum(r["correct"] for r in results)
                print(f"  [{idx+1}/{len(test_ds)}] acc={100*n_correct/len(results):.1f}%")

        except Exception:
            traceback.print_exc()
            results.append({
                "task":       item.get("task", ""),
                "question":   item.get("question", ""),
                "answer":     item.get("answer", ""),
                "prediction": "",
                "correct":    False,
            })

    tag = f"streamgaze_merge_{args.condition}_merge{int(args.merge_ratio*100)}pct"
    with open(os.path.join(RESULTS_DIR, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    metrics = compute_metrics(results)
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ov = metrics["overall"]
    print(f"\n{'='*60}")
    print(f"  TrajGazeMerge [{args.condition}] | merge={args.merge_ratio*100:.0f}%")
    print(f"{'='*60}")
    print(f"  Overall: {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"{'─'*60}")
    for task, tm in sorted(metrics["per_task"].items()):
        print(f"  {task[:50]:50s}: {tm['accuracy']:.2f}%  (n={tm['n_samples']})")
    print(f"\nResults → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
