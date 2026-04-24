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


def load_qwen_lora(
    device: torch.device,
    vit_unfreeze_last_n: int = 0,
    vit_lora_rank:       int = 0,
    vit_lora_last_n:     int = 2,
):
    """
    Load Qwen2.5-VL-7B-Instruct and apply LoRA to LLM attention layers.

    Returns:
        processor : AutoProcessor
        model     : PeftModel (Qwen2_5_VLForConditionalGeneration + LoRA)

    The base model weights are loaded in bfloat16.
    Only LLM q_proj, k_proj, v_proj, o_proj are targeted by LoRA.
    ViT attention layers use different names (q, k, v, proj) — not targeted by default.

    ViT adaptation (축 2, all default off):
      vit_unfreeze_last_n : unfreeze the last N vision blocks end-to-end
      vit_lora_rank       : attach a second LoRA adapter to the last N ViT blocks.
                            (0 = off; exclusive with vit_unfreeze_last_n)
      vit_lora_last_n     : number of last ViT blocks to target for LoRA
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import get_peft_model, LoraConfig, TaskType

    assert not (vit_unfreeze_last_n > 0 and vit_lora_rank > 0), (
        "Use either --vit-unfreeze-last-n or --vit-lora-rank, not both."
    )

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

    # ── 축 2: optional ViT adaptation (applied AFTER LLM LoRA wrapping) ────────
    if vit_unfreeze_last_n > 0:
        _unfreeze_vit_last_blocks(model, vit_unfreeze_last_n)
    elif vit_lora_rank > 0:
        _attach_vit_lora(model, rank=vit_lora_rank, last_n=vit_lora_last_n)

    model.print_trainable_parameters()
    return processor, model


def _get_vit_blocks(peft_model):
    """Return the ViT block list from a PeftModel wrapping Qwen2.5-VL."""
    # PeftModel → .base_model.model → Qwen2_5_VLForConditionalGeneration
    base = getattr(peft_model, "base_model", peft_model)
    if hasattr(base, "model") and not hasattr(base, "visual"):
        base = base.model
    if not hasattr(base, "visual"):
        raise RuntimeError(
            "Cannot locate ViT: expected peft_model.base_model.model.visual"
        )
    vis = base.visual
    if not hasattr(vis, "blocks"):
        raise RuntimeError("Expected .visual.blocks in Qwen2.5-VL ViT")
    return vis.blocks


def _unfreeze_vit_last_blocks(peft_model, last_n: int):
    """Unfreeze the last `last_n` vision transformer blocks in-place."""
    blocks = _get_vit_blocks(peft_model)
    n_total = len(blocks)
    last_n = min(last_n, n_total)
    for i in range(n_total - last_n, n_total):
        for p in blocks[i].parameters():
            p.requires_grad_(True)
    print(f"[ViT] Unfroze last {last_n}/{n_total} vision blocks")


def _attach_vit_lora(peft_model, rank: int, last_n: int):
    """
    Attach a second LoRA adapter to the last `last_n` ViT blocks.

    Uses regex target_modules to target only the specified block indices.
    Activates both adapters so LLM LoRA + ViT LoRA are applied simultaneously.
    """
    from peft import LoraConfig
    blocks  = _get_vit_blocks(peft_model)
    n_total = len(blocks)
    last_n  = min(last_n, n_total)
    target_ids = list(range(n_total - last_n, n_total))

    # Inspect module names in a sample block's attention
    sample = blocks[target_ids[0]]
    if not hasattr(sample, "attn"):
        raise RuntimeError("ViT block has no .attn; cannot attach LoRA")

    candidate_names = []
    for sub in ["q_proj", "k_proj", "v_proj", "o_proj", "proj", "qkv"]:
        if hasattr(sample.attn, sub) and isinstance(
            getattr(sample.attn, sub), torch.nn.Linear
        ):
            candidate_names.append(sub)
    if not candidate_names:
        raise RuntimeError(
            "ViT LoRA: no Linear submodules found under .attn. "
            "Qwen2.5-VL internals may have changed."
        )

    id_alt = "|".join(str(i) for i in target_ids)
    sub_alt = "|".join(candidate_names)
    target_regex = rf"visual\.blocks\.({id_alt})\.attn\.({sub_alt})"

    vit_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_regex,
    )
    peft_model.add_adapter("vit_lora", vit_cfg)
    # Keep both adapters active; PEFT applies them additively.
    try:
        peft_model.set_adapter(["default", "vit_lora"])
    except TypeError:
        # Older PEFT: fall back to single-active; user must handle in training
        pass
    print(
        f"[ViT] Attached LoRA rank={rank} to {last_n}/{n_total} blocks "
        f"({candidate_names})"
    )


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


class TrajHintProjection(torch.nn.Module):
    """
    Projects TrajGaze encoder's trajectory summary vectors to Qwen LLM
    embedding space, for use as auxiliary "text hint" tokens (축 5a).

    Kept trainable alongside LoRA / encoder parameters.
    """

    def __init__(self, in_dim: int, out_dim: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj     = torch.nn.Linear(in_dim, out_dim)
        # If n_tokens > 1, a small learned position bias per slot (so K slots
        # can diverge even from a single-vector hint input).
        self.slot_bias = torch.nn.Parameter(torch.zeros(n_tokens, out_dim))
        torch.nn.init.normal_(self.proj.weight, std=0.02)
        torch.nn.init.zeros_(self.proj.weight)  # zero-init for safe warm start
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, traj_feat: torch.Tensor) -> torch.Tensor:
        """
        traj_feat: (D_traj,) or (K, D_traj)
        returns  : (n_tokens, D_qwen)
        """
        if traj_feat.dim() == 1:
            traj_feat = traj_feat.unsqueeze(0).expand(self.n_tokens, -1)
        elif traj_feat.shape[0] == 1:
            traj_feat = traj_feat.expand(self.n_tokens, -1)
        elif traj_feat.shape[0] != self.n_tokens:
            # Pool or pad to match n_tokens
            if traj_feat.shape[0] > self.n_tokens:
                traj_feat = traj_feat[: self.n_tokens]
            else:
                reps = (self.n_tokens + traj_feat.shape[0] - 1) // traj_feat.shape[0]
                traj_feat = traj_feat.repeat(reps, 1)[: self.n_tokens]
        return self.proj(traj_feat) + self.slot_bias


def _insert_hint_tokens(
    inputs_embeds:  torch.Tensor,   # (1, L, d)
    attention_mask: torch.Tensor,   # (1, L)
    position_ids:   torch.Tensor,   # (3, 1, L) for M-RoPE
    hint_embeds:    torch.Tensor,   # (K, d)
    insert_at:      int,            # position in L at which to insert (before this index)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Insert K hint embeddings into the sequence right before `insert_at`.

    Position IDs (M-RoPE 3D) are extended such that:
      - hint tokens continue the text-position of the token just before insert_at
        (all 3 dims get the same incrementing values)
      - all positions AFTER the insertion shift by +K in ALL 3 dims
    This keeps the relative ordering consistent with text insertion.
    """
    K, d = hint_embeds.shape
    _, L, _ = inputs_embeds.shape

    # Embeddings
    pre  = inputs_embeds[:, :insert_at]
    post = inputs_embeds[:,  insert_at:]
    hint = hint_embeds.to(inputs_embeds.dtype).unsqueeze(0)   # (1, K, d)
    new_embeds = torch.cat([pre, hint, post], dim=1)

    # Attention mask
    new_mask = torch.cat([
        attention_mask[:, :insert_at],
        torch.ones(1, K, dtype=attention_mask.dtype, device=attention_mask.device),
        attention_mask[:,  insert_at:],
    ], dim=1)

    # M-RoPE 3D position_ids: compute the position at insert_at-1, add 1..K
    # then shift subsequent by +K.
    if insert_at > 0:
        anchor = position_ids[:, :, insert_at - 1]   # (3, 1)
    else:
        anchor = position_ids[:, :, 0] - 1           # (3, 1) = -1 if at start

    offsets = torch.arange(1, K + 1, device=position_ids.device)
    # anchor: (3, 1) → (3, 1, K) via broadcast with offsets
    hint_pos = anchor.unsqueeze(-1) + offsets.view(1, 1, K)     # (3, 1, K)

    shift = K
    new_pos = torch.cat([
        position_ids[:, :, :insert_at],
        hint_pos,
        position_ids[:, :,  insert_at:] + shift,
    ], dim=2)

    return new_embeds, new_mask, new_pos


def build_merged_inputs(
    base_qwen,
    cached:        dict,
    merged_video:  torch.Tensor,                # (N_video - r_merge - r_drop, d)
    receiver_idx:  torch.Tensor,                # (N_video - r_merge - r_drop,)
    drop_idx:      torch.Tensor | None = None,  # (r_drop,) — optional
    hint_embeds:   torch.Tensor | None = None,  # (K, d) — 축 5a (optional)
    align_text_position_to_teacher: bool = False,  # 축 4b safety (optional)
) -> dict:
    """
    Build the shortened input sequence with merged visual tokens.

    Removes both the "source" (merged-away) and "drop" (pruned) video token
    positions from the sequence and replaces the remaining receiver positions
    with merged_video embeddings. Sources and drops are collapsed to the same
    "removed" set at the sequence level — the difference is only that sources
    contributed to the receivers' merged embedding, while drops did not.

    Returns:
        inputs_embeds   : (1, L - r_merge - r_drop, d)
        attention_mask  : (1, L - r_merge - r_drop)
        position_ids    : (3, 1, L - r_merge - r_drop)
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

    # Mark receiver positions to keep; everything else (sources + drops) is removed.
    is_kept = torch.zeros(N_video, dtype=torch.bool, device=emb_dev)
    is_kept[receiver_idx] = True
    removed_video_pos = video_positions[~is_kept]  # positions in full sequence

    keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[removed_video_pos] = False

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

    # ── 축 4b safety: align text-region M-RoPE to teacher's absolute positions ──
    # When student's video tokens shrink, the default post-video text positions
    # shift forward by (N_video - N_video_kept). If we want teacher-student
    # hidden states (축 4b feature MSE) comparable, we restore text positions to
    # their teacher-equivalent values. Video positions themselves remain in the
    # student's shrunk space.
    if align_text_position_to_teacher:
        post_video_start_full = int(video_positions.max().item()) + 1
        # Count how many of the kept positions are video (i.e., receiver count)
        keep_video_count = int(new_is_video.sum().item())
        shrink = (N_video - keep_video_count)  # amount student text was shifted
        # new sequence length's video-block end in student:
        post_video_start_new = post_video_start_full - shrink
        if post_video_start_new < new_inputs_embeds.shape[1]:
            # Text region indices in the student sequence
            text_idx = torch.arange(
                post_video_start_new, new_inputs_embeds.shape[1],
                device=emb_dev,
            )
            # Teacher's position for the same token (by design, student/teacher
            # share the same original text tokens post-video) = position_ids at
            # the corresponding index in the full-length tensor (+ shrink offset).
            full_text_idx = text_idx + shrink
            # position_ids is (3, 1, L) on original (full) sequence
            teacher_text_pos = position_ids[:, :, full_text_idx]   # (3, 1, N_text)
            new_position_ids[:, :, text_idx] = teacher_text_pos.to(new_position_ids)

    # ── 축 5a: inject trajectory hint tokens right after the video block ──────
    if hint_embeds is not None and hint_embeds.numel() > 0:
        # Insert right after the last video position in the shrunk sequence
        new_is_video_final = (new_input_ids[0] == video_token_id)
        video_pos_in_new = new_is_video_final.nonzero(as_tuple=True)[0]
        if video_pos_in_new.numel() > 0:
            insert_at = int(video_pos_in_new.max().item()) + 1
        else:
            insert_at = 0
        new_inputs_embeds, new_attention_mask, new_position_ids = _insert_hint_tokens(
            new_inputs_embeds, new_attention_mask, new_position_ids,
            hint_embeds, insert_at,
        )

    return {
        "inputs_embeds":  new_inputs_embeds,
        "attention_mask": new_attention_mask,
        "position_ids":   new_position_ids,
        "rope_deltas":    rope_deltas,
    }


def build_full_inputs(
    base_qwen,
    cached:      dict,
    hint_embeds: torch.Tensor | None = None,   # 축 5a (teacher side, optional)
) -> dict:
    """
    Build input sequence with ALL visual tokens (for teacher pass).

    If hint_embeds is provided, the same K hint tokens are injected right
    after the video block — this keeps the teacher's sequence aligned with
    the student's so KL / feature MSE comparisons remain position-consistent.
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

    if hint_embeds is not None and hint_embeds.numel() > 0:
        insert_at = int(video_positions.max().item()) + 1
        inputs_embeds, attention_mask, position_ids = _insert_hint_tokens(
            inputs_embeds, attention_mask, position_ids, hint_embeds, insert_at
        )

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


def forward_logits_ext(
    model,
    inputs_dict:           dict,
    output_hidden_states:  bool = False,
    seq_region:            str = "last",   # "last" | "answer_full"
) -> dict:
    """
    Extended forward supporting hidden-state capture (축 4b) and multi-position
    logit return (축 4a). Returns a dict:
        {"logits_last":  (vocab,),              # always
         "logits_seq":   (K, vocab,) or None,   # if seq_region != "last"
         "hidden_states": tuple[(1, L, d)] or None,
         "L": int,                              # sequence length}

    seq_region="last"        : returns only last-position logits (legacy).
    seq_region="answer_full" : returns last N positions' logits where N is
                               heuristic (=8). The caller decides the mask.
                               For MCQ with 1-token answers, this is a no-op
                               equivalent to "last" in terms of gradient.
    """
    out = model(
        inputs_embeds=inputs_dict["inputs_embeds"],
        attention_mask=inputs_dict["attention_mask"],
        position_ids=inputs_dict["position_ids"],
        rope_deltas=inputs_dict["rope_deltas"],
        use_cache=False,
        output_hidden_states=output_hidden_states,
    )
    logits      = out.logits                        # (1, L, vocab)
    logits_last = logits[0, -1, :]                  # (vocab,)
    logits_seq  = None
    if seq_region == "answer_full":
        # Take the last 8 positions' logits as a rough "answer region" proxy.
        n_tail = min(8, logits.shape[1])
        logits_seq = logits[0, -n_tail:, :]         # (n_tail, vocab)
    hidden_states = out.hidden_states if output_hidden_states else None
    return {
        "logits_last":   logits_last,
        "logits_seq":    logits_seq,
        "hidden_states": hidden_states,
        "L":             logits.shape[1],
    }
