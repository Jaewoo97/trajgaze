"""
PLLaVA (60.46% LoRA) + PruneVid-style token pruning — zero-shot eval.

Same principle as eval_prunevid_lora.py for Qwen:
  - Score 2304 projected visual tokens by cosine similarity with a
    mean-pooled question+option text query (LLM embedding space).
  - Hard-drop bottom 90%, keep top 10% (~230 tokens).
  - Forward → logit-based MCQ at ASSISTANT: position.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/EgoGazeVQA \
        python -m TrajGazeMerge.training.eval_pllava_prunevid \
        --lora-ckpt .../pllava_baseline_lora/best_delta.pth \
        --out .../eval_results/pllava_prunevid.json
"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/prunevid")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset

PLLAVA_HF      = "ermu2001/pllava-7b"
POOLING_SHAPE  = (16, 12, 12)
FRAME_SHAPE    = (24, 24)
IMAGE_TOKEN    = "<image>"
IMAGE_TOKEN_ID = 32000
NUM_FRAMES     = 16
KEEP_RATIO     = 0.10
OPTION_LETTERS = ["A", "B", "C", "D"]

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)

LORA_CKPT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora/best_delta.pth"


def _sample_paths(paths, n):
    if not paths: return []
    if len(paths) <= n: return paths
    return [paths[int(i * len(paths) / n)] for i in range(n)]


def build_prompt(question, options):
    return f"{question}\nOptions:\n" + "\n".join(options)


def _load_pllava_peft_ckpt(model, hf_path, lora_alpha=256, lora_r=128):
    import glob
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
                remapped[new_k.replace(".base_layer.weight", ".weight")] = v
            elif ".lora_A." in k or ".lora_B." in k:
                continue
            else:
                remapped[new_k] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[PEFT ckpt] loaded {len(remapped)} keys | missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_pllava(device, lora_ckpt):
    from peft import LoraConfig, TaskType, get_peft_model
    from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor
    processor = PllavaProcessor.from_pretrained(PLLAVA_HF)
    config = PllavaConfig.from_pretrained(
        PLLAVA_HF, pooling_method="avg", use_pooling=True,
        frame_shape=FRAME_SHAPE, pooling_shape=POOLING_SHAPE,
        torch_dtype=torch.bfloat16, selected_layer=99,
        tau=1.0, cluster_ratio=1.0, temporal_segment_ratio=1.0,
    )
    model = PllavaForConditionalGeneration.from_pretrained(
        PLLAVA_HF, config=config, torch_dtype=torch.bfloat16
    )
    _load_pllava_peft_ckpt(model, PLLAVA_HF)
    for p in model.vision_tower.parameters():
        p.requires_grad_(False)
    for p in model.language_model.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    # Load fine-tuned delta
    delta = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[LoRA delta] missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval(), processor


def get_full_features(model, pixel_values):
    model_dtype = next(model.language_model.parameters()).dtype
    pixel_values = pixel_values.to(model_dtype)
    batch_size, num_videos = 1, pixel_values.shape[0] // model.config.num_frames
    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
    sel = image_outputs.hidden_states[model.config.vision_feature_layer][:, 1:]
    feats = model.multi_modal_projector(sel, "video", batch_size=batch_size,
                                        num_videos=num_videos, num_frames=model.config.num_frames)
    feats, _, _, _ = model.merge_frames_dynamic(feats, threshold=model.config.tau, k=7)
    return feats  # (1, 2304, d)


def prunevid_scores(model, full_features, question, options, device):
    """Score 2304 visual tokens by cosine sim with mean-pooled text query (LLM embedding space)."""
    tok = model.processor.tokenizer if hasattr(model, 'processor') else None
    return None  # will be replaced below


def forward_logit(model, processor, image_features, question, options, device):
    model_dtype = next(model.language_model.parameters()).dtype
    tok = processor.tokenizer
    pad_id = tok.pad_token_id or 0
    qtext = build_prompt(question, options)
    prompt = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    no_img_ids = torch.where(input_ids != IMAGE_TOKEN_ID, input_ids,
                             torch.full_like(input_ids, pad_id))
    inputs_embeds = model.get_input_embeddings()(no_img_ids).to(model_dtype)
    image_features = image_features.to(model_dtype)
    inputs_embeds_m, attn_mask_m, _, _, _ = model._merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, attn_mask, labels=None
    )
    out = model.language_model(inputs_embeds=inputs_embeds_m, attention_mask=attn_mask_m, use_cache=False)
    return out.logits[0, -1, :]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-ckpt", default=LORA_CKPT)
    p.add_argument("--out", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/eval_results/pllava_prunevid.json")
    p.add_argument("--n-frames", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    print("Loading PLLaVA + LoRA (60.46% delta) ...")
    model, processor = load_pllava(device, args.lora_ckpt)
    tok = processor.tokenizer
    option_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[-1]
                               for l in OPTION_LETTERS], device=device)
    print(f"Option IDs: {option_ids.tolist()}")

    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=args.n_frames, n_traj_frames=16)
    print(f"Eval on {len(test_ds.items)} items — PruneVid 10% keep (text-guided drop)")

    correct, total = 0, 0
    by_task = defaultdict(list)

    with torch.no_grad():
        for i, item in enumerate(test_ds):
            if item is None: continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], args.n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < args.n_frames:
                    pil_frames.append(pil_frames[-1])

                qtext = build_prompt(item["question"], item["options"])
                dummy = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
                proc_out = processor(text=dummy, images=pil_frames, return_tensors="pt")
                if proc_out.get("pixel_values") is None: continue
                pixel_values = proc_out["pixel_values"].to(device)

                # Full visual features (1, 2304, d)
                full_feats = get_full_features(model, pixel_values)

                # Text query from LLM embeddings (same LLM space as visual features)
                text_enc = tok(qtext, return_tensors="pt", add_special_tokens=False)
                text_ids = text_enc["input_ids"].to(device)
                text_embeds = model.get_input_embeddings()(text_ids).squeeze(0).float()
                query = F.normalize(text_embeds.mean(0), dim=-1)  # (d,)

                # Score visual tokens
                vis = F.normalize(full_feats[0].float(), dim=-1)  # (2304, d)
                scores = vis @ query                               # (2304,)

                # Keep top 10%
                N = full_feats.shape[1]
                n_keep = max(1, int(KEEP_RATIO * N))
                _, top_idx = torch.topk(scores, n_keep, largest=True)
                top_idx_sorted, _ = top_idx.sort()
                kept = full_feats[:, top_idx_sorted, :]  # (1, n_keep, d)

                logit = forward_logit(model, processor, kept,
                                      item["question"], item["options"], device)
                pred = logit[option_ids].argmax().item()
                gt = OPTION_LETTERS.index(item["answer"].upper())
                ok = int(pred == gt)
                correct += ok; total += 1
                by_task[item.get("task", "unknown")].append(ok)

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(test_ds.items)}] acc={100.*correct/max(1,total):.2f}%")

            except Exception:
                traceback.print_exc()
                continue

    acc = 100.0 * correct / max(1, total)
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}

    print(f"\n=== PLLaVA + PruneVid (10% text-guided keep) ===")
    print(f"Overall: {acc:.2f}%  (n={total})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%  (n={len(by_task[t])})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"method": "pllava_prunevid", "acc": acc, "n": total,
                   "per_task": per_task, "keep_ratio": KEEP_RATIO}, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
