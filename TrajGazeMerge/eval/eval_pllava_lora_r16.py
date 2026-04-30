"""
Standalone eval for PLLaVA-7B + LoRA r=16 (baseline fine-tuned model).
Loads epoch_03_delta.pth and runs generation-based eval on egtea test split.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.eval.eval_pllava_lora_r16 \
        --ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora_r16/epoch_03_delta.pth \
        --output /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/pllava_lora_r16_epoch03_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/prunevid")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset

PLLAVA_HF     = "ermu2001/pllava-7b"
POOLING_SHAPE = (16, 12, 12)
FRAME_SHAPE   = (24, 24)
IMAGE_TOKEN   = "<image>"

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)


def _sample_paths(paths, n):
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    return [paths[int(i * len(paths) / n)] for i in range(n)]


def build_prompt(question, options):
    return f"{question}\nOptions:\n" + "\n".join(options)


def _load_pllava_peft_ckpt(model, hf_path, lora_alpha=256, lora_r=128):
    from safetensors import safe_open
    if not os.path.isdir(hf_path):
        from huggingface_hub import snapshot_download
        hf_path = snapshot_download(hf_path)
    raw = {}
    for sf in sorted(glob.glob(os.path.join(hf_path, "*.safetensors"))):
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    scale = lora_alpha / lora_r
    remapped = {}
    for k, v in raw.items():
        if k.startswith("language_model.base_model.model."):
            new_k = k.replace("language_model.base_model.model.", "language_model.", 1)
            if ".base_layer.weight" in new_k:
                proj_prefix = k[: k.rfind(".base_layer.weight")]
                lora_a = raw.get(proj_prefix + ".lora_A.default.weight")
                lora_b = raw.get(proj_prefix + ".lora_B.default.weight")
                if lora_a is not None and lora_b is not None:
                    v = v + scale * (lora_b.float() @ lora_a.float()).to(v.dtype)
                final_k = new_k.replace(".base_layer.weight", ".weight")
                remapped[final_k] = v
            elif ".lora_A." in k or ".lora_B." in k:
                continue
            else:
                remapped[new_k] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[PEFT ckpt] loaded {len(remapped)} keys | missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_model(ckpt_path, device):
    from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor
    processor = PllavaProcessor.from_pretrained(PLLAVA_HF)
    config = PllavaConfig.from_pretrained(
        PLLAVA_HF,
        pooling_method="avg",
        use_pooling=True,
        frame_shape=FRAME_SHAPE,
        pooling_shape=POOLING_SHAPE,
        torch_dtype=torch.bfloat16,
        selected_layer=99,
        tau=1.0,
        cluster_ratio=1.0,
        temporal_segment_ratio=1.0,
    )
    model = PllavaForConditionalGeneration.from_pretrained(
        PLLAVA_HF, config=config, torch_dtype=torch.bfloat16
    )
    _load_pllava_peft_ckpt(model, PLLAVA_HF)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)

    # Load finetuned delta
    delta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[delta ckpt] loaded {len(delta)} keys | missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model, processor


def evaluate(model, processor, device, n_frames=16):
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=n_frames, n_traj_frames=32)
    model_dtype = next(model.language_model.parameters()).dtype
    pad_id = processor.tokenizer.pad_token_id or 0

    per_task: dict[str, dict] = {}
    correct = total = 0

    with torch.no_grad():
        for i, item in enumerate(test_ds):
            if item is None:
                continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < n_frames:
                    pil_frames.append(pil_frames[-1])

                qtext = build_prompt(item["question"], item["options"])
                prompt = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:Best option:("

                inputs = processor(text=prompt, images=pil_frames, return_tensors="pt")
                if inputs.get("pixel_values") is None:
                    continue
                pixel_values = inputs["pixel_values"].to(device).to(model_dtype)
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs["attention_mask"].to(device)

                out_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    media_type="video",
                    do_sample=False,
                    max_new_tokens=10,
                    pad_token_id=pad_id,
                )
                text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                text = text.split("ASSISTANT:")[-1].split("Best option:(")[-1].strip()
                pred = text[0].upper() if text else "A"

                gt = item["answer"].upper()
                hit = int(pred == gt)
                correct += hit
                total += 1

                task = item.get("task", "unknown")
                if task not in per_task:
                    per_task[task] = {"correct": 0, "total": 0}
                per_task[task]["correct"] += hit
                per_task[task]["total"] += 1

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(test_ds)}] running acc={100.*correct/max(1,total):.2f}%")

            except Exception:
                traceback.print_exc()
                continue

    overall = 100.0 * correct / max(1, total)
    per_task_acc = {t: 100.0 * v["correct"] / max(1, v["total"]) for t, v in per_task.items()}
    per_task_n   = {t: v["total"] for t, v in per_task.items()}
    return overall, total, per_task_acc, per_task_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora_r16/epoch_03_delta.pth")
    p.add_argument("--output", default="/workspace/EgoGazeVQA/TrajGazeMerge/eval_results/pllava_lora_r16_epoch03_metrics.json")
    p.add_argument("--n-frames", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda:0")
    print(f"Loading model from {args.ckpt} ...")
    model, processor = load_model(args.ckpt, device)

    print("Running eval on egtea test split ...")
    overall, n_total, per_task_acc, per_task_n = evaluate(model, processor, device, n_frames=args.n_frames)

    print(f"\n=== PLLaVA LoRA r=16 (epoch_03) ===")
    print(f"Overall: {overall:.2f}% (n={n_total})")
    for t in sorted(per_task_acc):
        print(f"  {t}: {per_task_acc[t]:.2f}% (n={per_task_n[t]})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        "model": "pllava_lora_r16_epoch03",
        "ckpt": args.ckpt,
        "overall_acc": round(overall, 4),
        "n_total": n_total,
        "per_task_acc": {t: round(v, 4) for t, v in per_task_acc.items()},
        "per_task_n": per_task_n,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
