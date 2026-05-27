"""
Batched variants of model.py functions for micro-batch training.

The existing `preprocess_item`, `build_merged_inputs`, `forward_logits` stay
unchanged. This module adds batched analogs that process B samples through
ONE Qwen forward pass, giving wall-clock speedup vs the per-sample loop.

Variable-length handling:
  - input_ids length differs per sample (different text + video tokens).
    After token merge, the sequence shortens by r_i = num_dropped_video_tokens.
  - We right-pad each sample's shortened sequence to L_new_max in the batch.
  - attention_mask is 0 at padded positions, so they don't affect compute.
  - Per-sample "last real token" position is tracked for logit extraction.

Identity check vs sequential: for a batch of one sample, batched and per-sample
paths should produce identical logits (modulo bf16 noise <1e-3).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from TrajGazeMerge.models.model import FRAME_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Batched preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_batch(
    processor,
    base_qwen,
    items: list[dict],
    device: torch.device,
) -> Optional[dict]:
    """
    Tokenize a batch of QA items and extract video features in one call.

    Args:
        items: list of dicts each with keys vlm_frame_paths, question, options.

    Returns:
        dict with batched tensors:
            input_ids       : (B, L_max)
            attention_mask  : (B, L_max)
            grid_thw        : (B, 3)
            video_embeds    : (sum N_i, d) concatenated across batch
            sample_offsets  : list[int] of length B+1, slices into video_embeds
            is_video        : (B, L_max) bool — video-token positions per sample
            position_ids    : (3, B, L_max)
            rope_deltas     : (B,)
            emb_dev         : torch.device of input embedding layer
        or None if every item failed to load frames.
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    all_messages, valid_idx = [], []
    for i, item in enumerate(items):
        frames = []
        for p in item["vlm_frame_paths"]:
            try:
                frames.append(Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE)))
            except Exception:
                pass
        if not frames:
            continue
        options_text = "\n".join(item["options"])
        user_text = (
            "You are watching a short first-person (egocentric) video clip.\n"
            f"Question: {item['question']}\n\n{options_text}\n\n"
            "Answer with only the letter (A, B, C, or D) of the correct option."
        )
        all_messages.append([{"role": "user", "content": [
            {"type": "video", "video": frames,
             "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
            {"type": "text",  "text": user_text},
        ]}])
        valid_idx.append(i)

    if not all_messages:
        return None

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    # process_vision_info on each — collect lists for the processor.
    # Scalar kwargs (e.g. `do_sample_frames`) stay scalar; list kwargs
    # (e.g. `fps=[2.0]` per video) extend across samples.
    all_video_inputs = []
    video_kwargs_merged: dict = {}
    for m in all_messages:
        _, vi, vk = process_vision_info(m, return_video_kwargs=True)
        all_video_inputs.extend(vi)
        for k, v in vk.items():
            if isinstance(v, list):
                video_kwargs_merged.setdefault(k, []).extend(v)
            else:
                # Assume scalar value is identical across samples
                video_kwargs_merged[k] = v

    inputs = processor(
        text=texts, images=None, videos=all_video_inputs,
        **video_kwargs_merged, return_tensors="pt", padding=True,
    )

    emb_dev = base_qwen.get_input_embeddings().weight.device
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    with torch.no_grad():
        ve_tuple = base_qwen.model.get_video_features(pv_vid, grid_thw)
        # tuple of (N_i, d). Concatenate + remember per-sample boundaries.
        per_sample_n = [t.shape[0] for t in ve_tuple]
        video_embeds = torch.cat(list(ve_tuple), dim=0).to(emb_dev)
        sample_offsets = [0]
        for n in per_sample_n:
            sample_offsets.append(sample_offsets[-1] + n)

        # get_rope_index handles batched input natively
        position_ids, rope_deltas = base_qwen.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )

    video_token_id = base_qwen.config.video_token_id
    is_video = (input_ids == video_token_id)

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "grid_thw":        grid_thw,
        "video_embeds":    video_embeds,
        "sample_offsets":  sample_offsets,
        "is_video":        is_video,
        "position_ids":    position_ids,
        "rope_deltas":     rope_deltas,
        "emb_dev":         emb_dev,
        "valid_idx":       valid_idx,   # indices into the input `items` list that produced output
    }


def slice_sample_video(batched: dict, b: int) -> torch.Tensor:
    """Return sample b's (N_b, d) video_embeds slice."""
    s, e = batched["sample_offsets"][b], batched["sample_offsets"][b + 1]
    return batched["video_embeds"][s:e]


# ──────────────────────────────────────────────────────────────────────────────
# Batched merged-inputs builder
# ──────────────────────────────────────────────────────────────────────────────

def build_merged_inputs_batch(
    base_qwen,
    batched: dict,
    merged_videos: list[torch.Tensor],   # B tensors of shape (N_keep_i, d)
    receiver_idxs: list[torch.Tensor],   # B tensors of shape (N_keep_i,)
) -> dict:
    """
    Build batched inputs_embeds with merged tokens replacing each sample's
    receiver positions and source positions dropped. Pad to max new length.

    Returns:
        inputs_embeds   : (B, L_new_max, d)
        attention_mask  : (B, L_new_max) — 1 for real tokens, 0 for pad
        position_ids    : (3, B, L_new_max)
        rope_deltas     : (B,)
        last_positions  : (B,) int — index of last real token per sample
    """
    input_ids      = batched["input_ids"]
    attn           = batched["attention_mask"]
    position_ids   = batched["position_ids"]
    rope_deltas    = batched["rope_deltas"]
    is_video       = batched["is_video"]
    emb_dev        = batched["emb_dev"]

    B, L_max = input_ids.shape
    video_token_id = base_qwen.config.video_token_id

    new_ids:   list[torch.Tensor] = []
    new_pos:   list[torch.Tensor] = []
    new_isvid: list[torch.Tensor] = []
    last_pos:  list[int] = []

    for b in range(B):
        video_pos_b = is_video[b].nonzero(as_tuple=True)[0]   # (N_b,)
        N_b = batched["sample_offsets"][b + 1] - batched["sample_offsets"][b]
        # N_b should equal video_pos_b.shape[0]; trust grid_thw boundary.

        is_recv = torch.zeros(N_b, dtype=torch.bool, device=emb_dev)
        is_recv[receiver_idxs[b]] = True
        source_pos_b = video_pos_b[~is_recv]

        keep = attn[b].bool().clone()
        keep[source_pos_b] = False

        new_ids_b   = input_ids[b][keep]          # (L_b,)
        new_pos_b   = position_ids[:, b][:, keep] # (3, L_b)
        new_isvid_b = (new_ids_b == video_token_id)

        new_ids.append(new_ids_b)
        new_pos.append(new_pos_b)
        new_isvid.append(new_isvid_b)
        last_pos.append(new_ids_b.shape[0] - 1)

    L_new_max = max(s.shape[0] for s in new_ids)
    pad_id = (base_qwen.config.pad_token_id
              if base_qwen.config.pad_token_id is not None else 0)

    padded_ids  = torch.full((B, L_new_max), pad_id, dtype=input_ids.dtype, device=emb_dev)
    padded_attn = torch.zeros((B, L_new_max), dtype=attn.dtype, device=emb_dev)
    padded_pos  = torch.zeros((3, B, L_new_max), dtype=position_ids.dtype, device=emb_dev)

    for b in range(B):
        L_b = new_ids[b].shape[0]
        padded_ids[b, :L_b]    = new_ids[b]
        padded_attn[b, :L_b]   = 1
        padded_pos[:, b, :L_b] = new_pos[b]
        # Continue position counter into the pad region (masked by attn anyway)
        if L_b < L_new_max:
            tail_count = L_new_max - L_b
            tail = (torch.arange(1, tail_count + 1, device=emb_dev, dtype=position_ids.dtype)
                    .unsqueeze(0).expand(3, -1))
            padded_pos[:, b, L_b:] = padded_pos[:, b, L_b - 1:L_b] + tail

    inputs_embeds = base_qwen.get_input_embeddings()(padded_ids)  # (B, L_new_max, d)

    # Replace each sample's video positions with that sample's merged_video tokens
    for b in range(B):
        full_mask = torch.zeros(L_new_max, dtype=torch.bool, device=emb_dev)
        full_mask[:new_isvid[b].shape[0]] = new_isvid[b]
        inputs_embeds[b, full_mask] = merged_videos[b].to(inputs_embeds.dtype)

    return {
        "inputs_embeds":  inputs_embeds,
        "attention_mask": padded_attn,
        "position_ids":   padded_pos,
        "rope_deltas":    rope_deltas,
        "last_positions": torch.tensor(last_pos, device=emb_dev, dtype=torch.long),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batched forward
# ──────────────────────────────────────────────────────────────────────────────

def forward_logits_batch(model, batched_inputs: dict) -> torch.Tensor:
    """
    Single Qwen forward over the batch; returns last-token logits per sample.

    Returns:
        logits : (B, vocab_size) — logits at each sample's last real-token position.
    """
    out = model(
        inputs_embeds=batched_inputs["inputs_embeds"],
        attention_mask=batched_inputs["attention_mask"],
        position_ids=batched_inputs["position_ids"],
        rope_deltas=batched_inputs["rope_deltas"],
        use_cache=False,
    )
    B = out.logits.shape[0]
    last_pos = batched_inputs["last_positions"]
    return out.logits[torch.arange(B, device=last_pos.device), last_pos, :]


__all__ = [
    "preprocess_batch",
    "slice_sample_video",
    "build_merged_inputs_batch",
    "forward_logits_batch",
]
