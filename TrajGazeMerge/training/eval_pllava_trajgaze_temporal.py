"""
PLLaVA (60.46% LoRA) + TrajGazeMerge temporal — zero-shot eval.

Loads the 60.46% PLLaVA LoRA and the temporal TrajGaze stage-1 encoder.
For each test item:
  1. Run TrajGazeV2Temporal → (T_traj, 196) per-frame patch scores.
  2. Map to PLLaVA's token layout via score_to_pllava_spatiotemporal:
       T_traj→16 (linear), 14×14→12×12 (bilinear) → (2304,) scores.
  3. gaze_weighted_merge: keep top 10% receivers, merge remaining 90% by
     cosine similarity (same as Qwen stage-3 training).
  4. Forward PLLaVA LM with merged features → MCQ logit.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/EgoGazeVQA \
        python -m TrajGazeMerge.training.eval_pllava_trajgaze_temporal \
        --lora-ckpt  .../pllava_baseline_lora/best_delta.pth \
        --stage1-ckpt .../stage1_temporal/best.pth \
        --out .../eval_results/pllava_trajgaze_temporal.json
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
from TrajGazeMerge.models.merge import gaze_weighted_merge

PLLAVA_HF      = "ermu2001/pllava-7b"
POOLING_SHAPE  = (16, 12, 12)
FRAME_SHAPE    = (24, 24)
IMAGE_TOKEN    = "<image>"
IMAGE_TOKEN_ID = 32000
NUM_FRAMES     = 16
KEEP_RATIO     = 0.10
MERGE_RATIO    = 1.0 - KEEP_RATIO
OPTION_LETTERS = ["A", "B", "C", "D"]

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)

LORA_CKPT   = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_baseline_lora/best_delta.pth"
STAGE1_CKPT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_temporal/best.pth"


def _sample_paths(paths, n):
    if not paths: return []
    if len(paths) <= n: return paths
    return [paths[int(i * len(paths) / n)] for i in range(n)]


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


def load_traj_encoder(ckpt_path, device, n_vis_keyframes=16):
    from TrajGaze_v2.models.model_temporal import TrajGazeV2Temporal
    enc = TrajGazeV2Temporal(n_vis_keyframes=n_vis_keyframes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = enc.load_state_dict(state, strict=False)
    print(f"[TrajEncTemporal] missing={len(missing)} unexpected={len(unexpected)}")
    return enc.eval()


def score_to_pllava_spatiotemporal(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: (T_traj, 196) → (2304,)
    Maps T_traj→16 linear, 14×14→12×12 bilinear to match POOLING_SHAPE=(16,12,12).
    """
    T_traj = scores.shape[0]
    T_vis, n_side = POOLING_SHAPE[0], POOLING_SHAPE[1]   # 16, 12
    n_spatial = n_side * n_side                           # 144

    s = scores.float().view(T_traj, 14, 14)
    s = F.interpolate(s.unsqueeze(1), size=(n_side, n_side),
                      mode="bilinear", align_corners=False).squeeze(1)   # (T_traj, 12, 12)
    s = s.reshape(T_traj, n_spatial).t().unsqueeze(0)                    # (1, 144, T_traj)
    s = F.interpolate(s, size=T_vis, mode="linear", align_corners=False) # (1, 144, T_vis)
    return s.squeeze(0).t().contiguous().view(-1)                        # (2304,)


def get_full_features(model, pixel_values):
    model_dtype = next(model.language_model.parameters()).dtype
    pixel_values = pixel_values.to(model_dtype)
    batch_size = 1
    num_videos = pixel_values.shape[0] // model.config.num_frames
    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
    sel = image_outputs.hidden_states[model.config.vision_feature_layer][:, 1:]
    feats = model.multi_modal_projector(sel, "video", batch_size=batch_size,
                                        num_videos=num_videos, num_frames=model.config.num_frames)
    feats, _, _, _ = model.merge_frames_dynamic(feats, threshold=model.config.tau, k=7)
    return feats  # (1, 2304, d)


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
    p.add_argument("--lora-ckpt",    default=LORA_CKPT)
    p.add_argument("--stage1-ckpt",  default=STAGE1_CKPT)
    p.add_argument("--out", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/eval_results/pllava_trajgaze_temporal.json")
    p.add_argument("--n-frames",        type=int, default=16)
    p.add_argument("--n-traj-frames",   type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
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

    print(f"Loading TrajGazeV2Temporal from {args.stage1_ckpt} ...")
    traj = load_traj_encoder(args.stage1_ckpt, device, args.n_vis_keyframes)

    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=args.n_frames,
                                     n_traj_frames=args.n_traj_frames)
    print(f"Eval on {len(test_ds.items)} items — TrajGaze temporal 10% gaze_weighted_merge")

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

                qtext = f"{item['question']}\nOptions:\n" + "\n".join(item["options"])
                dummy = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
                proc_out = processor(text=dummy, images=pil_frames, return_tensors="pt")
                if proc_out.get("pixel_values") is None: continue
                pixel_values = proc_out["pixel_values"].to(device)

                # Full visual features (1, 2304, d)
                full_feats = get_full_features(model, pixel_values)

                # Temporal TrajGaze scores
                traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
                scores_raw = traj.get_patch_scores(
                    traj_batch,
                    queries     = [item["question"]],
                    frame_paths = [item["traj_frame_paths"]],
                ).squeeze(0)  # (T_traj, 196)

                scores_all = score_to_pllava_spatiotemporal(scores_raw)  # (2304,)
                N = full_feats.shape[1]
                if scores_all.shape[0] != N:
                    if scores_all.shape[0] > N:
                        scores_all = scores_all[:N]
                    else:
                        reps = (N + scores_all.shape[0] - 1) // scores_all.shape[0]
                        scores_all = scores_all.repeat(reps)[:N]

                # gaze_weighted_merge: keep top 10% as receivers, merge rest
                r = max(1, int(MERGE_RATIO * N))
                merged_vis, _ = gaze_weighted_merge(full_feats[0], scores_all, r)
                merged_feats = merged_vis.unsqueeze(0)  # (1, ~230, d)

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

    print(f"\n=== PLLaVA + TrajGaze Temporal (10% gaze_weighted_merge) ===")
    print(f"Overall: {acc:.2f}%  (n={total})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%  (n={len(by_task[t])})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"method": "pllava_trajgaze_temporal", "acc": acc, "n": total,
                   "per_task": per_task, "keep_ratio": KEEP_RATIO,
                   "lora_ckpt": args.lora_ckpt, "stage1_ckpt": args.stage1_ckpt}, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
