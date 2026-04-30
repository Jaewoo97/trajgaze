"""
PLLaVA (60.46% LoRA) + PruMerge — zero-shot eval.

Adapts LLaVA-PruMerge (Shang et al., ICCV 2025) to PLLaVA's architecture:
  PLLaVA uses CLIP ViT-L/14 (24 layers, 336px, 576 patches/frame).
  PruMerge hooks layer-23 k/q projections to compute CLS→patch attention,
  selects top-10% tokens as "receivers", merges remaining into nearest receiver
  by key cosine similarity (weighted by CLS attention), appends one residual token.

Runs on 16 frames × 144 projected tokens = 2304 visual tokens total.
Steps:
  1. Hook CLIP layer 23 k_proj / q_proj during vision-tower forward.
  2. Compute per-frame CLS attention: (16, 576) → avg-pool 24×24→12×12 → (16, 144).
  3. Flatten to (2304,) score vector; normalise.
  4. Select top-10% = 230 receivers; find nearest receiver for each source by
     cosine similarity on projected features (vectorised scatter_add).
  5. Append one extra token = Σ(source × attn_weight).
  6. Forward PLLaVA language model with merged (231, d) features → MCQ logit.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/EgoGazeVQA \
        python -m TrajGazeMerge.training.eval_pllava_prumerge \
        --lora-ckpt .../pllava_baseline_lora/best_delta.pth \
        --out .../eval_results/pllava_prumerge.json
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
POOLING_SHAPE  = (16, 12, 12)   # T=16 frames, 12×12 spatial
FRAME_SHAPE    = (24, 24)
IMAGE_TOKEN    = "<image>"
IMAGE_TOKEN_ID = 32000
NUM_FRAMES     = 16
KEEP_RATIO     = 0.10
OPTION_LETTERS = ["A", "B", "C", "D"]
CLIP_N_PATCHES = 576   # 24×24 per frame before CLS removal
CLIP_LAST_LAYER = 23   # 0-indexed

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)

LORA_CKPT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora/best_delta.pth"

# Module-level buffer for hooks
_hook_buf: dict = {}


def _hook_k(module, inp, out):
    _hook_buf["k"] = out

def _hook_q(module, inp, out):
    _hook_buf["q"] = out


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
                la = raw.get(proj_prefix + ".lora_A.default.weight")
                lb = raw.get(proj_prefix + ".lora_B.default.weight")
                if la is not None and lb is not None:
                    v = v + scale * (lb.float() @ la.float()).to(v.dtype)
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
    delta = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(delta, strict=False)
    print(f"[LoRA delta] missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval(), processor


def prumerge_features(model, pixel_values, device):
    """
    Run vision tower with CLIP layer-23 k/q hooks → compute CLS attention →
    apply PruMerge on the 2304-token projected features.

    Returns (1, n_keep+1, d) merged features.
    """
    model_dtype = next(model.language_model.parameters()).dtype
    pixel_values = pixel_values.to(model_dtype)
    batch_size = 1
    num_videos = pixel_values.shape[0] // model.config.num_frames

    # Register hooks on CLIP layer 23 k/q projections
    last_layer = model.vision_tower.vision_model.encoder.layers[CLIP_LAST_LAYER]
    hk = last_layer.self_attn.k_proj.register_forward_hook(_hook_k)
    hq = last_layer.self_attn.q_proj.register_forward_hook(_hook_q)

    # Forward through vision tower
    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
    sel = image_outputs.hidden_states[model.config.vision_feature_layer][:, 1:]  # (16, 576, 1024)
    hk.remove(); hq.remove()

    k = _hook_buf["k"].float()  # (16, 577, d_clip)
    q = _hook_buf["q"].float()  # (16, 577, d_clip)

    # CLS→patch attention using raw k/q (same formula as PruMerge paper)
    B, N, C = k.shape  # 16, 577, 1024
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:]  # (16, 576) — CLS to non-CLS

    # Average-pool 24×24 → 12×12 to match projected spatial resolution
    cls_attn_2d = cls_attn.view(B, 24, 24)
    cls_attn_pooled = F.avg_pool2d(cls_attn_2d.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    scores = cls_attn_pooled.reshape(-1)  # (16×144,) = (2304,)

    # Get projected features
    feats = model.multi_modal_projector(sel, "video", batch_size=batch_size,
                                        num_videos=num_videos, num_frames=model.config.num_frames)
    feats, _, _, _ = model.merge_frames_dynamic(feats, threshold=model.config.tau, k=7)
    # feats: (1, 2304, d_llm)

    # --- PruMerge on the 2304-token projected sequence ---
    tokens = feats[0]   # (2304, d_llm)
    N_vis, d = tokens.shape
    n_keep = max(1, int(KEEP_RATIO * N_vis))

    # Top-k receivers by CLS attention
    _, top_idx = torch.topk(scores, n_keep, largest=True)
    top_idx_sorted, _ = top_idx.sort()

    # Complement: source tokens to be merged
    mask = torch.ones(N_vis, dtype=torch.bool, device=device)
    mask[top_idx_sorted] = False
    source_idx = mask.nonzero(as_tuple=True)[0]

    receivers = tokens[top_idx_sorted]   # (n_keep, d)
    sources   = tokens[source_idx]       # (N_vis-n_keep, d)
    src_attn  = scores[source_idx]       # (N_vis-n_keep,) weights for merging

    # Nearest receiver for each source by cosine similarity on projected features
    r_norm = F.normalize(receivers.float(), dim=-1)
    s_norm = F.normalize(sources.float(),   dim=-1)
    sim    = s_norm @ r_norm.t()           # (N_vis-n_keep, n_keep)
    nearest = sim.argmax(dim=-1)           # (N_vis-n_keep,)

    # Scatter-add: each source adds its attention-weighted feature to its receiver
    weighted_sources = (sources.float() * src_attn.unsqueeze(-1))  # (N_src, d)
    delta = torch.zeros_like(receivers.float())
    delta.scatter_add_(0, nearest.unsqueeze(-1).expand(-1, d), weighted_sources)
    updated_receivers = (receivers.float() + delta).to(model_dtype)

    # Extra token: weighted sum of all sources (residual)
    extra = (sources.float() * src_attn.unsqueeze(-1)).sum(0, keepdim=True).to(model_dtype)

    merged = torch.cat([updated_receivers, extra], dim=0).unsqueeze(0)  # (1, n_keep+1, d)
    return merged


def forward_logit(model, processor, image_features, question, options, device):
    model_dtype = next(model.language_model.parameters()).dtype
    tok = processor.tokenizer
    pad_id = tok.pad_token_id or 0
    qtext = f"{question}\nOptions:\n" + "\n".join(options)
    prompt = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    no_img_ids = torch.where(input_ids != IMAGE_TOKEN_ID, input_ids,
                             torch.full_like(input_ids, pad_id))
    inputs_embeds = model.get_input_embeddings()(no_img_ids).to(model_dtype)
    image_features = image_features.to(model_dtype)
    embeds_m, mask_m, _, _, _ = model._merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, attn_mask, labels=None
    )
    out = model.language_model(inputs_embeds=embeds_m, attention_mask=mask_m, use_cache=False)
    return out.logits[0, -1, :]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-ckpt", default=LORA_CKPT)
    p.add_argument("--out", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/eval_results/pllava_prumerge.json")
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
    print(f"Eval on {len(test_ds.items)} items — PruMerge 10% keep (CLS attention + merge)")

    def _sample(paths, n):
        if not paths: return []
        if len(paths) <= n: return paths
        return [paths[int(i * len(paths) / n)] for i in range(n)]

    correct, total = 0, 0
    by_task = defaultdict(list)

    with torch.no_grad():
        for i, item in enumerate(test_ds):
            if item is None: continue
            try:
                paths = _sample(item["vlm_frame_paths"], args.n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < args.n_frames:
                    pil_frames.append(pil_frames[-1])

                qtext = f"{item['question']}\nOptions:\n" + "\n".join(item["options"])
                dummy = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
                proc_out = processor(text=dummy, images=pil_frames, return_tensors="pt")
                if proc_out.get("pixel_values") is None: continue
                pixel_values = proc_out["pixel_values"].to(device)

                merged_feats = prumerge_features(model, pixel_values, device)
                logit = forward_logit(model, processor, merged_feats,
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

    print(f"\n=== PLLaVA + PruMerge (10% keep, CLS-attn + weighted merge) ===")
    print(f"Overall: {acc:.2f}%  (n={total})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%  (n={len(by_task[t])})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"method": "pllava_prumerge", "acc": acc, "n": total,
                   "per_task": per_task, "keep_ratio": KEEP_RATIO}, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
