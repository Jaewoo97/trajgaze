"""
TrajGazeMerge model helpers.

Provides functions to:
  - Load Qwen2.5-VL-7B with LoRA on LLM layers
  - Preprocess a StreamGaze item (tokenize + extract video features)
  - Build modified input sequence with gaze-merged visual tokens
  - Forward pass returning MCQ option logits

The merge is applied AFTER the ViT (frozen), BEFORE the LLM.
LoRA adapters are applied to LLM Q, K, V, O projection layers only.

Key design:
  - Teacher pass: model.disable_adapter() → base Qwen, full visual tokens
  - Student pass: model (LoRA enabled) → merged visual tokens
  - Gradient flow: loss → student LLM (LoRA weights)
                         → merged_tokens → merge op → patch_scores → encoder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

QWEN_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
FRAME_SIZE   = 224     # PIL resize before Qwen processor
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
MERGE_RATIO  = 0.50    # fraction of visual tokens to merge away


def load_qwen_lora(device: torch.device):
    """
    Load Qwen2.5-VL-7B-Instruct and apply LoRA to LLM attention layers.

    Returns:
        processor : AutoProcessor
        model     : PeftModel (Qwen2_5_VLForConditionalGeneration + LoRA)

    The base model weights are loaded in bfloat16.
    Only LLM q_proj, k_proj, v_proj, o_proj are targeted by LoRA.
    ViT attention layers use different names (q, k, v, proj) — not targeted.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import get_peft_model, LoraConfig, TaskType

    processor = AutoProcessor.from_pretrained(QWEN_PATH)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return processor, model


def load_qwen_frozen(device: torch.device):
    """
    Load Qwen2.5-VL-7B-Instruct fully frozen (for teacher pass reference / eval).
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(QWEN_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def get_option_ids(processor) -> list[int]:
    """Token IDs for 'A', 'B', 'C', 'D' in Qwen tokenizer."""
    return [
        processor.tokenizer.encode(c, add_special_tokens=False)[0]
        for c in ["A", "B", "C", "D"]
    ]


def preprocess_item(
    processor,
    base_qwen,                  # Qwen2_5_VLForConditionalGeneration (base, no PEFT)
    vlm_frame_paths: list[str],
    question:        str,
    options:         list[str],
    device:          torch.device,
) -> Optional[dict]:
    """
    Tokenize one QA item and extract video features from the frozen ViT.

    Mirrors preprocess_qwen_item from stage2.py.

    Returns cached dict with:
        input_ids        : (1, L)
        attention_mask   : (1, L)
        grid_thw         : video grid shape
        video_embeds     : (N_video, d) — ViT output in LLM embedding space
        video_positions  : (N_video,) int64 — positions of video tokens in seq
        position_ids     : (3, 1, L)
        rope_deltas      : scalar tensor
        emb_dev          : device of embedding layer
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    frames = []
    for p in vlm_frame_paths:
        try:
            img = Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
            frames.append(img)
        except Exception:
            pass
    if not frames:
        return None

    options_text = "\n".join(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
        {"type": "text",  "text": user_text},
    ]}]

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

    emb_dev = base_qwen.get_input_embeddings().weight.device
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    # Extract visual features from frozen ViT (already in LLM embedding space)
    with torch.no_grad():
        video_embeds = base_qwen.model.get_video_features(pv_vid, grid_thw).to(emb_dev)

        # 3D-RoPE position IDs for the original (full) sequence
        position_ids, rope_deltas = base_qwen.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )

    video_token_id  = base_qwen.config.video_token_id
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "grid_thw":        grid_thw,
        "video_embeds":    video_embeds,     # (N_video, d)
        "video_positions": video_positions,  # (N_video,)
        "position_ids":    position_ids,     # (3, 1, L)
        "rope_deltas":     rope_deltas,
        "emb_dev":         emb_dev,
    }


def build_merged_inputs(
    base_qwen,
    cached:        dict,
    merged_video:  torch.Tensor,  # (N_video - r, d)
    receiver_idx:  torch.Tensor,  # (N_video - r,) indices into video_embeds
) -> dict:
    """
    Build the shortened input sequence with merged visual tokens.

    Removes the r "source" video token positions from the sequence and
    replaces the remaining receiver positions with merged_video embeddings.

    Returns:
        inputs_embeds   : (1, L - r, d)
        attention_mask  : (1, L - r)
        position_ids    : (3, 1, L - r)
        rope_deltas     : scalar
    """
    input_ids      = cached["input_ids"]
    attention_mask = cached["attention_mask"]
    position_ids   = cached["position_ids"]
    rope_deltas    = cached["rope_deltas"]
    video_embeds   = cached["video_embeds"]
    video_positions = cached["video_positions"]
    emb_dev        = cached["emb_dev"]

    N_video = video_embeds.shape[0]

    # Mark source positions for removal
    is_receiver = torch.zeros(N_video, dtype=torch.bool, device=emb_dev)
    is_receiver[receiver_idx] = True
    source_video_pos = video_positions[~is_receiver]  # positions in full sequence

    keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[source_video_pos] = False

    # Shorten sequence
    new_input_ids      = input_ids[:, keep_seq]
    new_attention_mask = attention_mask[:, keep_seq]
    new_position_ids   = position_ids[:, :, keep_seq]

    # Text embeddings for the shortened sequence
    new_inputs_embeds = base_qwen.get_input_embeddings()(new_input_ids)  # (1, L-r, d)

    # Replace receiver video token positions with merged embeddings
    video_token_id = base_qwen.config.video_token_id
    new_is_video   = (new_input_ids[0] == video_token_id)
    new_inputs_embeds[0, new_is_video] = merged_video.to(new_inputs_embeds.dtype)

    return {
        "inputs_embeds":  new_inputs_embeds,
        "attention_mask": new_attention_mask,
        "position_ids":   new_position_ids,
        "rope_deltas":    rope_deltas,
    }


def build_full_inputs(base_qwen, cached: dict) -> dict:
    """
    Build input sequence with ALL visual tokens (for teacher pass).
    """
    input_ids      = cached["input_ids"]
    attention_mask = cached["attention_mask"]
    position_ids   = cached["position_ids"]
    rope_deltas    = cached["rope_deltas"]
    video_embeds   = cached["video_embeds"]
    video_positions = cached["video_positions"]

    inputs_embeds = base_qwen.get_input_embeddings()(input_ids)  # (1, L, d)
    video_token_id = base_qwen.config.video_token_id
    is_video = (input_ids[0] == video_token_id)
    inputs_embeds[0, is_video] = video_embeds.to(inputs_embeds.dtype)

    return {
        "inputs_embeds":  inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids":   position_ids,
        "rope_deltas":    rope_deltas,
    }


def forward_logits(model, inputs_dict: dict) -> torch.Tensor:
    """
    Run model forward and return last-position logits (vocab_size,).

    Args:
        model       : Qwen model (or PeftModel)
        inputs_dict : output of build_merged_inputs or build_full_inputs

    Returns:
        logits : (vocab_size,)
    """
    out = model(
        inputs_embeds=inputs_dict["inputs_embeds"],
        attention_mask=inputs_dict["attention_mask"],
        position_ids=inputs_dict["position_ids"],
        rope_deltas=inputs_dict["rope_deltas"],
        use_cache=False,
    )
    return out.logits[0, -1, :]   # next-token logits at last prompt position
