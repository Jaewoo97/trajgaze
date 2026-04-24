"""
VisionZip projector-only fine-tuning on StreamGaze_v2 MCQ tasks.

Finetunes ONLY the multi-modal projector (visual.merger / PatchMerger, ~44.6M params)
as described in VisionZip paper Sec. 2.4 "Efficient Tuning".
ViT and LLM are fully frozen; only the projector receives gradient updates.

Token selection: VisionZip (5% dominant + 5% contextual = 10% total).

Train: egoexolearn + holoassist (all MCQ)
Val:   egtea (full 526-item eval after every epoch)
GPUs:  4 via torchrun

Gradient flow:
    loss → frozen LLM → inputs_embeds → selected_video_embeds
         → visionzip_select → merger_output → visual.merger params

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29605 \\
        -m TrajGazeMerge.training.train_projector_visionzip \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/projector_visionzip \\
        --epochs 3 --lr 2e-5 --grad-accum 4
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration as VisionZipQwen

from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
from TrajGazeMerge.training.train_visionzip_lora import (
    preprocess_visionzip_item, visionzip_select_tokens,
    VIDEO_KWARGS, DOMINANT_RATIO, CONTEXTUAL_RATIO,
)
from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_projector_model(device: torch.device):
    """Load VisionZipQwen, freeze all params, unfreeze only visual.merger."""
    processor = AutoProcessor.from_pretrained(QWEN_MODEL, **VIDEO_KWARGS)
    model = VisionZipQwen.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.visual.merger.parameters():
        p.requires_grad_(True)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Total params: {n_total/1e6:.1f}M | "
        f"Trainable (merger): {n_train/1e6:.1f}M "
        f"({100.*n_train/n_total:.3f}%)",
        flush=True,
    )
    return processor, model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_item_projector(
    processor,
    model,
    frame_paths: list[str],
    question: str,
    options: list[str],
    device: torch.device,
) -> dict | None:
    """
    Tokenize and run the visual encoder in no_grad, capturing the input to
    visual.merger via a forward hook so we can replay it with grad enabled.

    Extra keys vs. preprocess_visionzip_item:
        pre_merger_input : (T, context_dim) in windowed order — on CPU
        reverse_indices  : (N_merged,) mapping windowed→spatial — on CPU
    """
    from qwen_vl_utils import process_vision_info

    options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
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
             "fps":        VIDEO_KWARGS["fps"]},
            {"type": "text", "text": prompt},
        ],
    }]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    emb_dev = model.get_input_embeddings().weight.device
    vis_dev = model.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    captured = {}
    def _hook(module, inp, out):
        captured["pre_merger"] = inp[0].detach().cpu()

    hook = model.visual.merger.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            video_embeds, attn_scores, attn_key = model.visual(pv_vid, grid_thw=grid_thw)
            window_index, _ = model.visual.get_window_index(grid_thw)
            reverse_indices = torch.argsort(window_index).cpu()

            position_ids, rope_deltas = model.get_rope_index(
                input_ids=input_ids,
                video_grid_thw=grid_thw,
                attention_mask=attention_mask,
            )
    finally:
        hook.remove()

    if "pre_merger" not in captured:
        return None

    video_token_id  = model.config.video_token_id
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]

    return {
        "input_ids":        input_ids,
        "attention_mask":   attention_mask,
        "position_ids":     position_ids,
        "rope_deltas":      rope_deltas,
        "grid_thw":         grid_thw,
        "video_embeds":     video_embeds.to(emb_dev),
        "video_positions":  video_positions,
        "attn_scores":      attn_scores.to(emb_dev),
        "attn_key":         attn_key.to(emb_dev),
        "emb_dev":          emb_dev,
        "pre_merger_input": captured["pre_merger"],   # (T, context_dim) on CPU
        "reverse_indices":  reverse_indices,           # (N_merged,) on CPU
    }


# ── Differentiable input construction ─────────────────────────────────────────

def build_merged_inputs_with_grad(
    base_model,
    cached: dict,
    selected_embeds: torch.Tensor,   # (N_keep, d_llm), requires_grad=True
    receiver_idx: torch.Tensor,      # (N_keep,)
) -> dict:
    """
    Like build_merged_inputs but preserves gradient flow from selected_embeds.
    Uses out-of-place index_put so autograd can trace back to selected_embeds.
    """
    input_ids       = cached["input_ids"]
    attention_mask  = cached["attention_mask"]
    position_ids    = cached["position_ids"]
    rope_deltas     = cached["rope_deltas"]
    video_positions = cached["video_positions"]
    emb_dev         = cached["emb_dev"]
    N_video         = cached["video_embeds"].shape[0]

    is_receiver = torch.zeros(N_video, dtype=torch.bool, device=emb_dev)
    is_receiver[receiver_idx] = True
    source_video_pos = video_positions[~is_receiver]

    keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[source_video_pos] = False

    new_input_ids      = input_ids[:, keep_seq]
    new_attention_mask = attention_mask[:, keep_seq]
    new_position_ids   = position_ids[:, :, keep_seq]

    with torch.no_grad():
        text_embeds = base_model.get_input_embeddings()(new_input_ids).detach()

    video_token_id = base_model.config.video_token_id
    is_video       = (new_input_ids[0] == video_token_id)
    video_pos_new  = is_video.nonzero(as_tuple=True)[0]

    L_new = new_input_ids.shape[1]
    d     = text_embeds.shape[-1]

    # Out-of-place index_put preserves autograd graph for selected_embeds
    batch_idx  = torch.zeros(video_pos_new.shape[0], dtype=torch.long, device=emb_dev)
    video_full = torch.zeros(1, L_new, d, device=emb_dev, dtype=selected_embeds.dtype)
    video_full = video_full.index_put(
        (batch_idx, video_pos_new),
        selected_embeds.to(emb_dev),
        accumulate=False,
    )

    is_video_3d   = is_video.unsqueeze(0).unsqueeze(-1).expand(1, L_new, d)
    inputs_embeds = torch.where(is_video_3d, video_full, text_embeds.to(video_full.dtype))

    return {
        "inputs_embeds":  inputs_embeds,
        "attention_mask": new_attention_mask,
        "position_ids":   new_position_ids,
        "rope_deltas":    rope_deltas,
    }


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(processor, model, option_ids, device):
    """Eval on full EGTEA test set (526 items) using VisionZip selection."""
    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    model.eval()
    correct, total = 0, 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                cached = preprocess_visionzip_item(
                    processor, model,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue
                selected_embeds, receiver_idx = visionzip_select_tokens(
                    cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
                )
                inputs_dict   = build_merged_inputs(model, cached, selected_embeds, receiver_idx)
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/projector_visionzip")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def allreduce_merger_grads(merger_params, world_size):
    for p in merger_params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[ProjFT] output: {args.output_dir}", flush=True)
        print(
            f"[ProjFT] GPUs={world_size}, dominant={DOMINANT_RATIO*100:.0f}%+"
            f"contextual={CONTEXTUAL_RATIO*100:.0f}%=10%, "
            f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}",
            flush=True,
        )

    if is_main:
        print("Loading VisionZip Qwen2.5-VL-7B (projector finetuning) ...", flush=True)
    processor, model = load_projector_model(device)
    merger        = model.visual.merger
    merger_params = list(merger.parameters())
    option_ids    = get_option_ids(processor)
    if is_main:
        print("Model loaded.", flush=True)

    train_ds = StreamGazeSimpleDataset(split="train", n_vlm_frames=args.n_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    optimizer = AdamW(merger_params, lr=args.lr, weight_decay=1e-4)

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
                # Run visual encoder in no_grad; hook captures pre-merger features
                cached = preprocess_item_projector(
                    processor, model,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue

                n_video = cached["video_embeds"].shape[0]

                # Replay merger WITH gradient (only merger params have requires_grad=True)
                pre_merger = cached["pre_merger_input"].to(device).to(torch.bfloat16)
                rev_idx    = cached["reverse_indices"].to(device)
                merger_out        = merger(pre_merger)           # (N_merged, d_llm)
                video_embeds_grad = merger_out[rev_idx]          # spatial order, has grad

                # VisionZip selection — grad flows through indexing and cluster averaging
                selected_embeds, receiver_idx = visionzip_select_tokens(
                    video_embeds_grad,
                    cached["attn_scores"],
                    cached["attn_key"],
                )

                # Build input sequence preserving grad through selected_embeds
                inputs_dict = build_merged_inputs_with_grad(
                    model, cached, selected_embeds, receiver_idx
                )

                # Forward frozen LLM (grad flows back to inputs_embeds → selected_embeds)
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                loss   = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
                (loss / args.grad_accum).backward()

                epoch_loss += loss.item()
                n_steps    += 1

                if n_steps % args.grad_accum == 0:
                    allreduce_merger_grads(merger_params, world_size)
                    torch.nn.utils.clip_grad_norm_(merger_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed  = time.time() - t_start
                    n_kept   = receiver_idx.shape[0]
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    print(
                        f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                        f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s",
                        flush=True,
                    )
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Final optimizer step for remaining accumulated grads
        if n_steps % args.grad_accum != 0:
            allreduce_merger_grads(merger_params, world_size)
            torch.nn.utils.clip_grad_norm_(merger_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed  = time.time() - t_start
        if is_main:
            print(
                f"\n=== Epoch {epoch+1}/{args.epochs} | "
                f"avg_loss={avg_loss:.4f} | t={elapsed:.0f}s ===",
                flush=True,
            )
            torch.save({
                "epoch":        epoch,
                "merger_state": merger.state_dict(),
                "loss":         avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()

        if is_main:
            print(f"Evaluating epoch {epoch+1} on full EGTEA val set ...", flush=True)
            acc, n_eval, per_task = evaluate(processor, model, option_ids, device)
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch":        epoch,
                    "merger_state": merger.state_dict(),
                    "acc":          acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
