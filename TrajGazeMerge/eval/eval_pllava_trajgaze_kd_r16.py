"""
Standalone eval for TrajGazeMerge (10% tokens) + PLLaVA LoRA r=16 (KD fine-tuned).
Loads epoch_03.pth (lora_state + encoder_state) and runs merged-token eval on egtea.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m TrajGazeMerge.eval.eval_pllava_trajgaze_kd_r16 \
        --ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_trajgaze_kd_r16/epoch_03.pth \
        --output /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/pllava_trajgaze_kd_r16_epoch03_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/prunevid")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge

PLLAVA_HF      = "ermu2001/pllava-7b"
POOLING_SHAPE  = (16, 12, 12)
FRAME_SHAPE    = (24, 24)
IMAGE_TOKEN    = "<image>"
IMAGE_TOKEN_ID = 32000
KEEP_RATIO     = 0.10
MERGE_RATIO    = 1.0 - KEEP_RATIO
TOTAL_VIS_TOKENS = POOLING_SHAPE[0] * POOLING_SHAPE[1] * POOLING_SHAPE[2]

STAGE1_CKPT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"

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


def load_model_and_encoder(ckpt_path, device):
    from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor
    from TrajGaze_v2.models.model import TrajGazeV2

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

    # Load finetuned checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state   = ckpt.get("lora_state", ckpt)
    encoder_state = ckpt.get("encoder_state", None)

    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f"[LoRA ckpt] loaded {len(lora_state)} keys | missing={len(missing)} unexpected={len(unexpected)}")

    # TrajGaze encoder
    traj_encoder = TrajGazeV2().to(device)
    if encoder_state is not None:
        m, u = traj_encoder.load_state_dict(encoder_state, strict=False)
        print(f"[TrajEnc ckpt] loaded | missing={len(m)} unexpected={len(u)}")
    else:
        print("[TrajEnc] WARNING: no encoder_state in ckpt, using random init")

    model = model.to(device)
    model.eval()
    traj_encoder.eval()
    return model, processor, traj_encoder


def score_to_pllava_spatial(patch_scores, n_spatial):
    side = int(n_spatial ** 0.5)
    scores_2d = patch_scores.float().reshape(1, 1, 14, 14)
    out = F.interpolate(scores_2d, size=(side, side), mode="bilinear", align_corners=False)
    return out.squeeze().flatten()


def get_pllava_image_features(model, pixel_values, media_type="video"):
    model_dtype = next(model.language_model.parameters()).dtype
    pixel_values = pixel_values.to(model_dtype)
    batch_size = 1
    num_videos = pixel_values.shape[0] // model.config.num_frames // batch_size

    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
    vision_feature_layer = model.config.vision_feature_layer
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    selected_image_feature = selected_image_feature[:, 1:]

    image_features = model.multi_modal_projector(
        selected_image_feature,
        media_type,
        batch_size=batch_size,
        num_videos=num_videos,
        num_frames=model.config.num_frames,
    )
    return image_features


def get_patch_scores(traj_encoder, item, device):
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb = traj_encoder.query_encoder([item["question"]], device)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)
    return scores_raw.squeeze(0)  # (196,)


def evaluate_merged(model, processor, traj_encoder, device, n_frames=16):
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

                proc_out = processor(text=prompt, images=pil_frames, return_tensors="pt")
                if proc_out.get("pixel_values") is None:
                    continue
                pixel_values = proc_out["pixel_values"].to(device)

                full_features = get_pllava_image_features(model, pixel_values)
                N_vis = full_features.shape[1]

                scores = get_patch_scores(traj_encoder, item, device)
                n_spatial = POOLING_SHAPE[1] * POOLING_SHAPE[2]
                n_temporal = POOLING_SHAPE[0]
                scores_spatial = score_to_pllava_spatial(scores, n_spatial)
                scores_all = scores_spatial.unsqueeze(0).expand(n_temporal, -1).reshape(-1)
                if scores_all.shape[0] != N_vis:
                    scores_all = scores_all[:N_vis] if scores_all.shape[0] > N_vis \
                        else scores_all.repeat_interleave((N_vis + scores_all.shape[0] - 1) // scores_all.shape[0])[:N_vis]

                r = max(1, int(MERGE_RATIO * N_vis))
                merged_features, _ = gaze_weighted_merge(
                    full_features[0].detach(), scores_all.detach(), r
                )
                merged_features = merged_features.unsqueeze(0)

                enc = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)

                no_img_ids = torch.where(
                    input_ids != IMAGE_TOKEN_ID,
                    input_ids,
                    torch.full_like(input_ids, pad_id),
                )
                inputs_embeds = model.get_input_embeddings()(no_img_ids).to(model_dtype)
                merged_features = merged_features.to(model_dtype)

                inputs_embeds_m, attn_mask_m, _, _, _ = model._merge_input_ids_with_image_features(
                    merged_features, inputs_embeds, input_ids, attn_mask, labels=None
                )

                out_ids = model.language_model.generate(
                    inputs_embeds=inputs_embeds_m,
                    attention_mask=attn_mask_m,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
                text = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
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
    p.add_argument("--ckpt", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_trajgaze_kd_r16/epoch_03.pth")
    p.add_argument("--output", default="/workspace/EgoGazeVQA/TrajGazeMerge/eval_results/pllava_trajgaze_kd_r16_epoch03_metrics.json")
    p.add_argument("--n-frames", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda:0")
    print(f"Loading model from {args.ckpt} ...")
    model, processor, traj_encoder = load_model_and_encoder(args.ckpt, device)

    print("Running merged-token eval on egtea test split ...")
    overall, n_total, per_task_acc, per_task_n = evaluate_merged(
        model, processor, traj_encoder, device, n_frames=args.n_frames
    )

    print(f"\n=== TrajGaze KD r=16 (epoch_03) ===")
    print(f"Overall: {overall:.2f}% (n={n_total})")
    for t in sorted(per_task_acc):
        print(f"  {t}: {per_task_acc[t]:.2f}% (n={per_task_n[t]})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        "model": "pllava_trajgaze_kd_r16_epoch03",
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
