"""
Training-free VisionZip eval: apply VisionZip token selection (5% dominant + 5% contextual)
at inference time to the baseline LoRA checkpoint (trained on full tokens).

Mirrors eval_prunevid_lora.py — no additional training.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.training.eval_visionzip_lora \
        --ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \
        --out  /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/visionzip_baseline_lora_metrics.json
"""

import argparse, json, os, sys, traceback
import torch

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration as VisionZipQwen
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor

from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
from TrajGazeMerge.training.train_visionzip_lora import (
    preprocess_visionzip_item, visionzip_select_tokens,
    VIDEO_KWARGS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
)
from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits

QWEN_MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
BASELINE_CKPT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth"
DOMINANT_RATIO   = 0.05
CONTEXTUAL_RATIO = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=BASELINE_CKPT)
    p.add_argument("--out",  required=True)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0")

    print("Loading VisionZip Qwen2.5-VL-7B ...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL, **VIDEO_KWARGS)
    base_model = VisionZipQwen.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor)
    model.eval()

    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    print(f"Evaluating {len(test_ds)} items with training-free VisionZip "
          f"(dominant={DOMINANT_RATIO*100:.0f}% + contextual={CONTEXTUAL_RATIO*100:.0f}% = 10%) ...")

    correct = 0
    total   = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for i, item in enumerate(test_ds):
            if item is None:
                continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue

                selected_embeds, receiver_idx = visionzip_select_tokens(
                    cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
                    dominant_ratio=DOMINANT_RATIO, contextual_ratio=CONTEXTUAL_RATIO,
                )
                inputs_dict   = build_merged_inputs(base_qwen, cached, selected_embeds, receiver_idx)
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(test_ds)}] acc={100.*correct/max(1,total):.2f}%")

            except Exception:
                traceback.print_exc()
                continue

    acc      = 100.0 * correct / max(1, total)
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}

    print(f"\n=== Training-free VisionZip (10% tokens, no retraining) ===")
    print(f"Overall: {acc:.2f}%  (n={total})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%  (n={len(by_task[t])})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "acc": acc, "n": total, "per_task": per_task,
            "ckpt": args.ckpt,
            "dominant_ratio": DOMINANT_RATIO,
            "contextual_ratio": CONTEXTUAL_RATIO,
            "method": "visionzip_training_free",
        }, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
