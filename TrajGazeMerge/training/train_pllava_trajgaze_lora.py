"""
PLLaVA-7B + TrajGaze encoder (10% tokens) + LoRA r=16, logit-based MCQ training.

Matches the approach of pllava_baseline_lora (64.83%) but with TrajGaze merging:
- Logit-based loss: CE at the last token position (ASSISTANT:) over the full vocabulary,
  ground-truth is the correct option letter token ID (A=319, B=350, C=315, D=360).
- TrajGaze encoder maps patch scores → merge 90% of 2304 visual tokens (keep 10%).
- Trainable: LoRA adapters on q/k/v/o_proj (r=16) + TrajGaze encoder.
- Vision tower frozen; base LLM weights frozen (only LoRA trains).

Train : egoexolearn + holoassist (5799 items)
Eval  : egtea (526 items, logit-based with merged tokens)
GPUs  : 0,1 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/gaze/bin/torchrun --nproc_per_node=2 \
        --master_port=29800 \
        -m TrajGazeMerge.training.train_pllava_trajgaze_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_trajgaze_lora_v2 \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \
        --epochs 3 --lr-lora 1e-4 --lr-enc 1e-5 --grad-accum 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

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
MERGE_RATIO    = 1.0 - KEEP_RATIO   # 90% tokens removed
TOTAL_VIS_TOKENS = POOLING_SHAPE[0] * POOLING_SHAPE[1] * POOLING_SHAPE[2]  # 2304

# A=319, B=350, C=315, D=360 in LLaMA tokenizer
OPTION_LETTERS = ["A", "B", "C", "D"]

STAGE1_CKPT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"

SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)


def _sample_paths(paths: list, n: int) -> list:
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    return [paths[int(i * len(paths) / n)] for i in range(n)]


def build_prompt(question: str, options: list) -> str:
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


def load_pllava(device):
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
    for p in model.vision_tower.parameters():
        p.requires_grad_(False)
    for p in model.language_model.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()
    return model.to(device), processor


def load_traj_encoder(ckpt_path: str, device: torch.device):
    from TrajGaze_v2.models.model import TrajGazeV2
    enc = TrajGazeV2().to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = enc.load_state_dict(state, strict=False)
        print(f"[TrajEnc] Loaded | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[TrajEnc] WARNING: ckpt not found: {ckpt_path}, using random init")
    return enc


def score_to_pllava_spatial(patch_scores: torch.Tensor, n_spatial: int) -> torch.Tensor:
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
    image_features, _, _, _ = model.merge_frames_dynamic(image_features, threshold=model.config.tau, k=7)
    return image_features  # (1, 2304, d)


def get_patch_scores(traj_encoder, item, device):
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb = traj_encoder.query_encoder([item["question"]], device)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)
    return scores_raw.squeeze(0)  # (196,)


def merge_features(full_features, scores, device):
    """Merge 90% of visual tokens guided by TrajGaze scores. Returns (1, ~230, d)."""
    N_vis = full_features.shape[1]
    n_spatial = POOLING_SHAPE[1] * POOLING_SHAPE[2]   # 144
    n_temporal = POOLING_SHAPE[0]                       # 16

    scores_spatial = score_to_pllava_spatial(scores, n_spatial)  # (144,)
    scores_all = scores_spatial.unsqueeze(0).expand(n_temporal, -1).reshape(-1)  # (2304,)
    if scores_all.shape[0] != N_vis:
        if scores_all.shape[0] > N_vis:
            scores_all = scores_all[:N_vis]
        else:
            reps = (N_vis + scores_all.shape[0] - 1) // scores_all.shape[0]
            scores_all = scores_all.repeat(reps)[:N_vis]

    r = max(1, int(MERGE_RATIO * N_vis))
    merged, _ = gaze_weighted_merge(full_features[0].detach(), scores_all, r)
    return merged.unsqueeze(0)  # (1, ~230, d)


def forward_logit(model, processor, image_features, question, options, device):
    """Forward pass with merged image features; returns logit at last position."""
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

    out = model.language_model(
        inputs_embeds=inputs_embeds_m,
        attention_mask=attn_mask_m,
        use_cache=False,
    )
    return out.logits[0, -1, :]  # (vocab_size,)


def evaluate_logit(model, processor, traj_encoder, device, option_ids_t, n_frames=16, max_items=None):
    """Logit-based eval on egtea test split with merged tokens."""
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=n_frames, n_traj_frames=16)
    if max_items:
        test_ds.items = test_ds.items[:max_items]

    model_dtype = next(model.language_model.parameters()).dtype
    model.eval()
    traj_encoder.eval()
    correct = total = 0

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < n_frames:
                    pil_frames.append(pil_frames[-1])

                qtext = build_prompt(item["question"], item["options"])
                dummy_prompt = f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} \nOnly give the best option. ASSISTANT:"
                proc_out = processor(text=dummy_prompt, images=pil_frames, return_tensors="pt")
                if proc_out.get("pixel_values") is None:
                    continue
                pixel_values = proc_out["pixel_values"].to(device)

                full_features = get_pllava_image_features(model, pixel_values)
                scores = get_patch_scores(traj_encoder, item, device)
                merged = merge_features(full_features, scores, device)

                logit = forward_logit(model, processor, merged,
                                      item["question"], item["options"], device)
                pred_idx = logit[option_ids_t].argmax().item()
                gt_idx = OPTION_LETTERS.index(item["answer"].upper())
                correct += int(pred_idx == gt_idx)
                total += 1
            except Exception:
                continue

    model.train()
    traj_encoder.train()
    return 100.0 * correct / max(1, total), total


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",   default=STAGE1_CKPT)
    p.add_argument("--output-dir",    default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/pllava_trajgaze_lora_v2")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--lr-lora",       type=float, default=1e-4)
    p.add_argument("--lr-enc",        type=float, default=1e-5)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--log-every",     type=int,   default=20)
    p.add_argument("--eval-every",    type=int,   default=300)
    p.add_argument("--n-frames",      type=int,   default=16)
    p.add_argument("--n-traj-frames", type=int,   default=16)
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[PLLaVA TrajGaze LoRA] output: {args.output_dir}")
        print(f"GPUs={world_size}, epochs={args.epochs}, "
              f"lr_lora={args.lr_lora}, lr_enc={args.lr_enc}, grad_accum={args.grad_accum}")

    if is_main:
        print("Loading PLLaVA-7B + LoRA r=16 ...")
    model, processor = load_pllava(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        print("Loading TrajGaze encoder ...")
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        lora_n = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        enc_n  = sum(p.numel() for p in traj_encoder.parameters())
        print(f"Trainable LoRA params:  {lora_n:,}")
        print(f"TrajGaze encoder params: {enc_n:,}")

    train_ds = StreamGazeMergeDataset(split="train", n_vlm_frames=args.n_frames,
                                      n_traj_frames=args.n_traj_frames)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    tok = processor.tokenizer
    option_ids = [tok.encode(f"Best option:({l}", add_special_tokens=False)[-1]
                  for l in OPTION_LETTERS]
    option_ids_t = torch.tensor(option_ids, device=device)
    if is_main:
        print(f"Option IDs (A/B/C/D): {option_ids}")

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        traj_encoder.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        n_steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                paths = _sample_paths(item["vlm_frame_paths"], args.n_frames)
                pil_frames = [Image.open(p).convert("RGB") for p in paths]
                while len(pil_frames) < args.n_frames:
                    pil_frames.append(pil_frames[-1])

                # Build prompt for pixel_values extraction
                qtext = build_prompt(item["question"], item["options"])
                dummy_prompt = (f"{SYSTEM} USER: {IMAGE_TOKEN}\n USER: {qtext} "
                                f"\nOnly give the best option. ASSISTANT:")
                proc_out = processor(text=dummy_prompt, images=pil_frames, return_tensors="pt")
                if proc_out.get("pixel_values") is None:
                    continue
                pixel_values = proc_out["pixel_values"].to(device)

                # Extract full visual features (no grad through vision tower)
                with torch.no_grad():
                    full_features = get_pllava_image_features(model.module, pixel_values)

                # TrajGaze scores (with grad through encoder)
                scores = get_patch_scores(traj_encoder.module, item, device)

                # Merge to 10%
                merged = merge_features(full_features, scores, device)

                # Forward pass → logit at ASSISTANT: position
                logit_last = forward_logit(
                    model.module, processor, merged,
                    item["question"], item["options"], device,
                )

                # Logit-based MCQ loss: CE over full vocabulary with correct letter as target
                gt_letter = item["answer"].upper()
                if gt_letter not in OPTION_LETTERS:
                    continue
                gt_token = option_ids[OPTION_LETTERS.index(gt_letter)]
                gt_tensor = torch.tensor([gt_token], device=device, dtype=torch.long)
                loss = F.cross_entropy(logit_last.unsqueeze(0), gt_tensor)

                (loss / args.grad_accum).backward()

                epoch_loss += loss.item()
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "elapsed": elapsed,
                        }) + "\n")

                if is_main and n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate_logit(
                        model.module, processor, traj_encoder.module, device,
                        option_ids_t, n_frames=args.n_frames, max_items=200,
                    )
                    print(f"  → eval_logit egtea (merged): {acc:.2f}% (n={n_eval})")
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({
                            "epoch": epoch, "step": n_steps,
                            "lora_state": {k: v.cpu() for k, v in model.module.state_dict().items()
                                           if "lora_" in k or "multi_modal_projector" in k},
                            "encoder_state": traj_encoder.module.state_dict(),
                            "acc": acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)")
                    model.train()
                    traj_encoder.train()

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_loss = epoch_loss / max(1, n_steps)
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")

            acc, n_eval = evaluate_logit(
                model.module, processor, traj_encoder.module, device,
                option_ids_t, n_frames=args.n_frames,
            )
            print(f"  → eval_acc_logit: {acc:.2f}% (n={n_eval})")

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "avg_loss": avg_loss,
                    "eval_acc_logit": acc, "n_eval": n_eval,
                }) + "\n")

            torch.save({
                "epoch": epoch,
                "lora_state": {k: v.cpu() for k, v in model.module.state_dict().items()
                               if "lora_" in k or "multi_modal_projector" in k},
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg_loss, "acc": acc,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

            model.train()
            traj_encoder.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
