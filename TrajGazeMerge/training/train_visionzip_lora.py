"""
VisionZip + Qwen2.5-VL-7B LoRA fine-tuning on StreamGaze_v2 MCQ tasks.

Token selection: VisionZip (dvlab-research/VisionZip) — uses last ViT block attention
to select 5% dominant tokens + 5% contextual cluster-center tokens = 10% total.
  - Dominant: top-k by mean attention weight across heads in last ViT block
  - Contextual: merge remaining 90% non-dominant tokens into evenly-spaced cluster
    centers using cosine similarity of key vectors

Same LoRA config, dataset, and eval protocol as all other conditions.

Train: egoexolearn + holoassist (all MCQ)
Val:   egtea (full 526-item eval after every epoch)
GPUs:  4 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29604 \
        -m TrajGazeMerge.training.train_visionzip_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_lora \
        --epochs 3 --lr 1e-4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

# VisionZip's modified Qwen2.5-VL (visual encoder returns attn scores + keys)
from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration as VisionZipQwen

from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)

QWEN_MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_RANK     = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
DOMINANT_RATIO   = 0.05   # 5% dominant tokens by attention score
CONTEXTUAL_RATIO = 0.05   # 5% contextual cluster-center tokens
# Total kept: DOMINANT_RATIO + CONTEXTUAL_RATIO = 10%

VIDEO_KWARGS = dict(
    max_pixels=256 * 28 * 28,
    min_pixels=1   * 28 * 28,
    fps=1.0,
)


# ── Model loading ────────────────────────────────────────────────────────────

def load_visionzip_lora(device: torch.device):
    """Load VisionZip's Qwen2.5-VL-7B with LoRA applied to LLM attention layers."""
    processor = AutoProcessor.from_pretrained(QWEN_MODEL, **VIDEO_KWARGS)

    base_model = VisionZipQwen.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    return processor, model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_visionzip_item(
    processor,
    base_qwen,
    frame_paths: list[str],
    question: str,
    options: list[str],
    device: torch.device,
) -> dict | None:
    """
    Preprocess one item and run VisionZip's visual encoder to get attention scores.

    Returns a cached dict with:
        - Standard fields (input_ids, attention_mask, position_ids, rope_deltas,
          video_positions, emb_dev, grid_thw)
        - video_embeds : (N_video, d) — full ViT features
        - attn_scores  : (N_video,)   — VisionZip importance scores from last ViT block
        - attn_key     : (1, N_video, d_k) — ViT key vectors for contextual clustering
    """
    from PIL import Image as _PIL
    from qwen_vl_utils import process_vision_info

    options_text = "\n".join(
        f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
    )
    prompt = (
        f"{question}\n"
        f"Options:\n{options_text}\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frame_paths,
             "max_pixels": VIDEO_KWARGS["max_pixels"],
             "min_pixels": VIDEO_KWARGS["min_pixels"],
             "fps": VIDEO_KWARGS["fps"]},
            {"type": "text", "text": prompt},
        ],
    }]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            **video_kwargs, return_tensors="pt",
        )
    except Exception:
        return None

    if "pixel_values_videos" not in inputs:
        return None

    emb_dev = base_qwen.get_input_embeddings().weight.device
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    # Run VisionZip's visual encoder → returns (video_embeds, attn_scores, attn_key)
    with torch.no_grad():
        video_embeds, attn_scores, attn_key = base_qwen.visual(pv_vid, grid_thw=grid_thw)
        video_embeds = video_embeds.to(emb_dev)
        attn_scores  = attn_scores.to(emb_dev)
        attn_key     = attn_key.to(emb_dev)

        position_ids, rope_deltas = base_qwen.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )

    video_token_id  = base_qwen.config.video_token_id
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "position_ids":    position_ids,
        "rope_deltas":     rope_deltas,
        "grid_thw":        grid_thw,
        "video_embeds":    video_embeds,
        "video_positions": video_positions,
        "attn_scores":     attn_scores,
        "attn_key":        attn_key,
        "emb_dev":         emb_dev,
    }


# ── VisionZip token selection ─────────────────────────────────────────────────

def visionzip_select_tokens(
    video_embeds: torch.Tensor,   # (N, d)
    attn_scores:  torch.Tensor,   # (N,) importance from last ViT block
    attn_key:     torch.Tensor,   # (1, N, d_k) key vectors for clustering
    dominant_ratio:   float = DOMINANT_RATIO,
    contextual_ratio: float = CONTEXTUAL_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    VisionZip two-stage selection:
      1. Dominant: top (dominant_ratio * N) tokens by attention score
      2. Contextual: from non-dominant tokens, pick evenly-spaced cluster centers
         and merge all other non-dominant tokens into their nearest center
         (cosine sim of key vectors)

    Returns:
        selected_embeds : (n_keep, d) — dominant + contextual merged embeddings
        receiver_idx    : (n_keep,)   — original indices, sorted ascending
    """
    N, d = video_embeds.shape
    dominant_num   = max(1, int(dominant_ratio   * N))
    contextual_num = max(1, int(contextual_ratio * N))

    # ── 1. Dominant tokens ────────────────────────────────────────────────────
    _, topk_indices = torch.topk(attn_scores, dominant_num)
    topk_sorted, _  = topk_indices.sort()

    dom_mask       = torch.zeros(N, dtype=torch.bool, device=video_embeds.device)
    dom_mask[topk_indices] = True
    non_dom_orig   = (~dom_mask).nonzero(as_tuple=True)[0]   # (N-dom,) orig indices

    # ── 2. Contextual tokens ──────────────────────────────────────────────────
    n_non_dom       = non_dom_orig.shape[0]
    non_dom_embeds  = video_embeds[non_dom_orig]              # (N-dom, d)
    non_dom_key     = attn_key[0, non_dom_orig, :]            # (N-dom, d_k)
    non_dom_key_n   = F.normalize(non_dom_key.float(), dim=-1)

    # Evenly-spaced cluster centers within non-dominant set
    step            = max(1, n_non_dom // contextual_num)
    center_local    = torch.arange(
        0, n_non_dom, step, device=video_embeds.device
    )[:contextual_num]                                        # (contextual_num,)
    center_key_n    = non_dom_key_n[center_local]             # (contextual_num, d_k)

    # Assign every non-dominant token to its nearest center
    sim             = non_dom_key_n @ center_key_n.T          # (N-dom, contextual_num)
    assign          = sim.argmax(dim=-1)                      # (N-dom,)

    # Non-center non-dominant tokens → merge into assigned center
    is_center       = torch.zeros(n_non_dom, dtype=torch.bool, device=video_embeds.device)
    is_center[center_local] = True
    non_center_local = (~is_center).nonzero(as_tuple=True)[0] # (N-dom-ctx,)

    center_embeds   = non_dom_embeds[center_local].float().clone()   # (contextual_num, d)
    counts          = torch.ones(contextual_num, device=video_embeds.device, dtype=torch.float)

    if non_center_local.numel() > 0:
        assign_nc   = assign[non_center_local]                        # (N-dom-ctx,)
        nc_embeds   = non_dom_embeds[non_center_local].float()        # (N-dom-ctx, d)
        center_embeds.scatter_add_(
            0,
            assign_nc.unsqueeze(-1).expand(-1, d),
            nc_embeds,
        )
        counts.scatter_add_(
            0, assign_nc,
            torch.ones(non_center_local.shape[0], device=video_embeds.device)
        )

    center_embeds = (center_embeds / counts.unsqueeze(-1)).to(video_embeds.dtype)
    contextual_orig_idx = non_dom_orig[center_local]                  # (contextual_num,)

    # ── Combine and sort by original position ─────────────────────────────────
    dom_embeds   = video_embeds[topk_sorted]                          # (dominant_num, d)
    all_embeds   = torch.cat([dom_embeds, center_embeds], dim=0)      # (n_keep, d)
    all_idx      = torch.cat([topk_sorted, contextual_orig_idx])      # (n_keep,)

    sort_order   = all_idx.argsort()
    receiver_idx = all_idx[sort_order]
    selected_embeds = all_embeds[sort_order]

    return selected_embeds, receiver_idx


# ── Training helpers ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",  default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_lora")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad-accum",  type=int,   default=4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=20)
    p.add_argument("--n-frames",    type=int,   default=128)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def evaluate(processor, model, base_qwen, option_ids, device):
    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    model.eval()
    correct = 0
    total   = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                selected_embeds, receiver_idx = visionzip_select_tokens(
                    cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
                )
                inputs_dict = build_merged_inputs(
                    base_qwen, cached, selected_embeds, receiver_idx
                )

                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)
            except Exception:
                pass

    model.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[VisionZip LoRA] output: {args.output_dir}")
        print(f"[VisionZip LoRA] GPUs={world_size}, "
              f"dominant={DOMINANT_RATIO*100:.0f}%+contextual={CONTEXTUAL_RATIO*100:.0f}%=10%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")

    if is_main:
        print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_visionzip_lora(device)
    base_qwen  = model.get_base_model()
    model      = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main:
        print("Model loaded.")

    train_ds = StreamGazeSimpleDataset(split="train", n_vlm_frames=args.n_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_steps    = 0
        t_start    = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                # Preprocess + run VisionZip visual encoder
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device
                    )
                if cached is None:
                    continue

                n_video = cached["video_embeds"].shape[0]

                # VisionZip selection: 5% dominant + 5% contextual
                with torch.no_grad():
                    selected_embeds, receiver_idx = visionzip_select_tokens(
                        cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
                    )
                    inputs_dict = build_merged_inputs(
                        base_qwen, cached, selected_embeds, receiver_idx
                    )

                # Forward + CE loss
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                gt_idx  = ["A", "B", "C", "D"].index(item["answer"])
                loss    = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
                loss = loss / args.grad_accum
                loss.backward()

                epoch_loss += loss.item() * args.grad_accum
                n_steps    += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed  = time.time() - t_start
                    n_kept   = receiver_idx.shape[0]
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed  = time.time() - t_start
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")
            torch.save({
                "epoch": epoch,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on full EGTEA val set ...")
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})")
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
