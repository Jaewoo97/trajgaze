"""
Unified baseline token-selection + Qwen2.5-VL-7B LoRA trainer.

ONE trainer, one dataset (CombinedSimpleDataset), one LoRA config, one eval —
only the visual-token *selection rule* changes via --select-mode. This is what
makes the tab:main baseline rows a fair comparison with \\sys: identical
gaze-overlay input frames (GAZE_OVERLAY=1), identical 2-way split (--no-hdepic =
StreamGaze+EgoGazeVQA), identical 3-epoch LoRA (eff-batch 8, early-stop),
identical egtea eval (n=1011 = SG 526 + EG 485), --merge-ratio 0.9 (10% budget).

--select-mode:
    visionzip : ViT-attention 5% dominant + 5% contextual (the VisionZip baseline)
    full      : keep ALL tokens (100% upper bound)
    random    : keep a random 10% (per-item deterministic seed)
    prunevid  : question-guided cosine top-10% (PruneVid)
    fastvid   : DySeg -> STPrune -> DTM density-merge at 10% retention (FastVID,
                faithful port of LunarShen/FastVID modeling_qwen2_5_vl.py)
    autogaze  : frozen AutoGaze GRPO gaze-saliency -> gaze_weighted_merge, keep 10%

Per-source eval: every epoch reports StreamGaze / EgoGazeVQA accuracy separately
(binned from CombinedSimpleDataset.items src tag) so tab:main columns fill directly.

Usage (one mode, 2 GPUs):
    cd /workspace/trajgaze_st
    export PATH="/opt/conda/envs/trajgaze/bin:$PATH"; export GAZE_OVERLAY=1
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29661 \\
        -m TrajGazeMerge.training.train_baseline_select_lora \\
        --select-mode prunevid \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/prunevid_sgeg_overlay \\
        --epochs 3 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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
sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

# VisionZip's modified Qwen2.5-VL (visual encoder returns attn scores + keys).
# Used for ALL modes so preprocessing (frames, ViT features) is byte-identical.
from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration as VisionZipQwen

from TrajGazeMerge.data.combined_simple_dataset import CombinedSimpleDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.merge import gaze_weighted_merge

QWEN_MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_RANK     = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
DOMINANT_RATIO   = 0.05
CONTEXTUAL_RATIO = 0.05
KEEP_RATIO       = 0.10   # random / prunevid / autogaze budget

# FastVID hyperparameters (LunarShen/FastVID defaults; retention set to 0.10 so
# the kept fraction matches the 10% budget of every other row — a fair row).
FASTVID_RETENTION    = 0.10
FASTVID_DYSEG_C      = 8
FASTVID_DYSEG_TAU    = 0.84
FASTVID_DYSEG_IGNORE = 0.95
FASTVID_STPRUNE_D    = 0.4
FASTVID_DTM_P        = 4
FASTVID_DTM_BETA     = 0.6

VIDEO_KWARGS = dict(
    max_pixels=256 * 28 * 28,
    min_pixels=1   * 28 * 28,
    fps=1.0,
)


# ── Model loading ────────────────────────────────────────────────────────────

def load_visionzip_lora(device: torch.device):
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


# ── Preprocessing (identical across all modes) ───────────────────────────────

def preprocess_visionzip_item(processor, base_qwen, frame_paths, question, options, device):
    from qwen_vl_utils import process_vision_info

    options_text = "\n".join(options)
    letters = [chr(65 + i) for i in range(len(options))]
    letters_str = ", ".join(letters[:-1]) + (f", or {letters[-1]}" if len(letters) > 1 else "")
    prompt = (
        f"{question}\n"
        f"Options:\n{options_text}\n"
        f"Answer with a single letter ({letters_str})."
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

    emb_dev = base_qwen.get_input_embeddings().weight.device
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    with torch.no_grad():
        video_embeds, attn_scores, attn_key = base_qwen.visual(pv_vid, grid_thw=grid_thw)
        video_embeds = video_embeds.to(emb_dev)
        attn_scores  = attn_scores.to(emb_dev)
        attn_key     = attn_key.to(emb_dev)
        position_ids, rope_deltas = base_qwen.get_rope_index(
            input_ids=input_ids, video_grid_thw=grid_thw, attention_mask=attention_mask,
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


# ── VisionZip selection ──────────────────────────────────────────────────────

def visionzip_select_tokens(video_embeds, attn_scores, attn_key,
                            dominant_ratio=DOMINANT_RATIO, contextual_ratio=CONTEXTUAL_RATIO,
                            dominant_score=None):
    N, d = video_embeds.shape
    dominant_num   = max(1, int(dominant_ratio   * N))
    contextual_num = max(1, int(contextual_ratio * N))

    dom_score = attn_scores if dominant_score is None else dominant_score
    _, topk_indices = torch.topk(dom_score, dominant_num)
    topk_sorted, _  = topk_indices.sort()

    dom_mask       = torch.zeros(N, dtype=torch.bool, device=video_embeds.device)
    dom_mask[topk_indices] = True
    non_dom_orig   = (~dom_mask).nonzero(as_tuple=True)[0]

    n_non_dom      = non_dom_orig.shape[0]
    non_dom_embeds = video_embeds[non_dom_orig]
    non_dom_key    = attn_key[0, non_dom_orig, :]
    non_dom_key_n  = F.normalize(non_dom_key.float(), dim=-1)

    step         = max(1, n_non_dom // contextual_num)
    center_local = torch.arange(0, n_non_dom, step, device=video_embeds.device)[:contextual_num]
    center_key_n = non_dom_key_n[center_local]

    sim    = non_dom_key_n @ center_key_n.T
    assign = sim.argmax(dim=-1)

    is_center = torch.zeros(n_non_dom, dtype=torch.bool, device=video_embeds.device)
    is_center[center_local] = True
    non_center_local = (~is_center).nonzero(as_tuple=True)[0]

    center_embeds = non_dom_embeds[center_local].float().clone()
    counts        = torch.ones(center_local.shape[0], device=video_embeds.device, dtype=torch.float)

    if non_center_local.numel() > 0:
        assign_nc = assign[non_center_local]
        nc_embeds = non_dom_embeds[non_center_local].float()
        center_embeds.scatter_add_(0, assign_nc.unsqueeze(-1).expand(-1, d), nc_embeds)
        counts.scatter_add_(0, assign_nc,
                            torch.ones(non_center_local.shape[0], device=video_embeds.device))

    center_embeds = (center_embeds / counts.unsqueeze(-1)).to(video_embeds.dtype)
    contextual_orig_idx = non_dom_orig[center_local]

    dom_embeds = video_embeds[topk_sorted]
    all_embeds = torch.cat([dom_embeds, center_embeds], dim=0)
    all_idx    = torch.cat([topk_sorted, contextual_orig_idx])

    sort_order   = all_idx.argsort()
    receiver_idx = all_idx[sort_order]
    return all_embeds[sort_order], receiver_idx


# ── PruneVid selection (question-guided cosine top-k) ────────────────────────

def compute_prunevid_scores(base_qwen, cached, device):
    input_ids    = cached["input_ids"]
    video_embeds = cached["video_embeds"]
    video_token_id = base_qwen.config.video_token_id
    is_text = (input_ids[0] != video_token_id)
    text_ids = input_ids[:, is_text]
    with torch.no_grad():
        text_embeds = base_qwen.get_input_embeddings()(text_ids.to(device)).squeeze(0).float()
    query  = F.normalize(text_embeds.mean(dim=0), dim=-1)
    vf     = F.normalize(video_embeds.float(), dim=-1)
    return vf @ query


def select_prunevid_tokens(base_qwen, cached, keep_ratio, device):
    video_embeds = cached["video_embeds"]
    N = video_embeds.shape[0]
    n_keep = max(1, int(keep_ratio * N))
    scores = compute_prunevid_scores(base_qwen, cached, device)
    top_idx = torch.topk(scores, n_keep, largest=True).indices
    receiver_idx, _ = top_idx.sort()
    return video_embeds[receiver_idx], receiver_idx


# ── Random selection (deterministic per item) ────────────────────────────────

def select_random_tokens(cached, item, keep_ratio):
    video_embeds = cached["video_embeds"]
    N = video_embeds.shape[0]
    n_keep = max(1, int(keep_ratio * N))
    seed = int(hashlib.md5((item["question"] + str(N)).encode()).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    idx  = perm[:n_keep].sort().values.to(video_embeds.device)
    return video_embeds[idx], idx


# ── FastVID selection (DySeg -> STPrune -> DTM) ──────────────────────────────
# Faithful port of LunarShen/FastVID modeling_qwen2_5_vl.py (the video-only core,
# operating on post-2x2-merge Qwen tokens = video_embeds). Salient tokens are kept
# raw (top ViT-attention per frame); context tokens are density-peak cluster
# centers that absorb the rest via a beta-clamped weighted merge.

def select_fastvid_tokens(cached, args, device):
    video_embeds = cached["video_embeds"]     # (N, d)
    attn_scores  = cached["attn_scores"]      # (N,)
    grid_thw     = cached["grid_thw"]
    N, d = video_embeds.shape
    dev = video_embeds.device

    frame_num = int(grid_thw[0, 0].item())
    frame_w   = int(grid_thw[0, 1].item()) // 2
    frame_h   = int(grid_thw[0, 2].item()) // 2
    frm_len   = frame_w * frame_h

    # Fallback keeps eval robust if the grid doesn't factor cleanly.
    if frm_len <= 0 or frame_num * frm_len != N:
        return visionzip_select_tokens(video_embeds, attn_scores, cached["attn_key"],
                                       dominant_ratio=0.05, contextual_ratio=0.05)

    video_hidden = video_embeds.reshape(frame_num, frm_len, d)
    frame_attn   = attn_scores.reshape(frame_num, frm_len)
    fgf_raw      = video_hidden.float().mean(dim=1)                       # (frame_num, d)
    fgf          = fgf_raw / fgf_raw.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # ── DySeg: cut the frame stream into event segments ──────────────────────
    if frame_num > 1:
        sim = (fgf[:-1] * fgf[1:]).sum(dim=1)                             # (frame_num-1,)
        k   = min(max(args.fastvid_DySeg_c - 1, 0), sim.shape[0])
        cut = sim.topk(k, largest=False).indices if k > 0 \
            else torch.empty(0, dtype=torch.long, device=dev)
        cos = torch.nonzero(sim < args.fastvid_DySeg_tau).squeeze(1)
        cut = torch.unique(torch.cat([cut, cos])).sort().values
        if cut.numel() == 0:
            segment_sizes = [frame_num]
        else:
            segment_sizes = [cut[0].item() + 1]
            for i in range(1, cut.shape[0]):
                segment_sizes.append(cut[i].item() - cut[i - 1].item())
            segment_sizes.append(frame_num - cut[-1].item() - 1)
    else:
        segment_sizes = [frame_num]
    segment_sizes = [s for s in segment_sizes if s > 0]

    segments_hidden = torch.split(video_hidden, segment_sizes)
    segments_attn   = torch.split(frame_attn,   segment_sizes)
    segments_global = torch.split(fgf,          segment_sizes)

    frame_retain_num = max(1, int(frm_len * args.fastvid_retention))

    vision_tokens  = []
    keep_idx_parts = []
    seg_index_start = 0

    for seg_i in range(len(segment_sizes)):
        cur_hidden = segments_hidden[seg_i]
        cur_attn   = segments_attn[seg_i]
        cur_global = segments_global[seg_i]
        cur_seg_len, frm_len_ = cur_attn.shape

        seg_retain_num  = frame_retain_num * cur_seg_len
        seg_context_num = max(int(seg_retain_num * args.fastvid_STPrune_d), 1)
        seg_salient_num = max(seg_retain_num - seg_context_num, 0)

        if cur_seg_len == 1:
            frm_salient_num = [seg_salient_num]
            frm_context_num = [seg_context_num]
        else:
            cur_sim = (cur_global[:-1] * cur_global[1:]).sum(dim=1)
            cur_cut = torch.nonzero(cur_sim < args.fastvid_DySeg_ignore).squeeze(1).sort().values
            if cur_cut.shape[0] > 0:
                cur_seg_sizes = [cur_cut[0].item() + 1]
                for i in range(1, cur_cut.shape[0]):
                    cur_seg_sizes.append(cur_cut[i].item() - cur_cut[i - 1].item())
                cur_seg_sizes.append(cur_seg_len - cur_cut[-1].item() - 1)
            else:
                cur_seg_sizes = [cur_seg_len]
            cur_seg_sizes = [s for s in cur_seg_sizes if s > 0]

            valid_seg_len = len(cur_seg_sizes)
            chunk = seg_salient_num // valid_seg_len
            rem   = seg_salient_num %  valid_seg_len
            frm_salient_num = [chunk + (1 if i < rem else 0) for i in range(valid_seg_len)]

            temp_num = (valid_seg_len + args.fastvid_DTM_p - 1) // args.fastvid_DTM_p
            chunk = seg_context_num // temp_num
            rem   = seg_context_num %  temp_num
            temp_context_num = [chunk + (1 if i < rem else 0) for i in range(temp_num)]

            tmp_ctx_idx = 0
            frm_context_num = []
            for tmp_i in range(valid_seg_len):
                if tmp_i % args.fastvid_DTM_p == 0:
                    frm_context_num.append(min(temp_context_num[tmp_ctx_idx], frm_len_ // 2))
                    if temp_context_num[tmp_ctx_idx] + frm_context_num[-1] > frm_len_:
                        temp_context_num[tmp_ctx_idx] = frm_len_ - frm_context_num[-1]
                    tmp_ctx_idx += 1
                else:
                    frm_context_num.append(0)
            frm_context_num.reverse()

            valid_salient, valid_context = [], []
            for j in range(valid_seg_len):
                valid_salient.extend([0] * (cur_seg_sizes[j] - 1))
                valid_context.extend([0] * (cur_seg_sizes[j] - 1))
                if frm_context_num[j] == 0:
                    valid_salient.append(min(frm_salient_num[j], frm_len_))
                    valid_context.append(0)
                elif frm_context_num[j] > 0:
                    valid_context.append(min(frm_context_num[j], frm_len_ // 2))
                    valid_salient.append(min(frm_salient_num[j], frm_len_ - valid_context[-1]))
                else:
                    valid_salient.append(frm_len_ - int(frm_len_ * args.fastvid_STPrune_d))
                    valid_context.append(int(frm_len_ * args.fastvid_STPrune_d))
            frm_salient_num = valid_salient
            frm_context_num = valid_context

        cur_salient_idx, cur_context_idx = [], []
        for fi in range(cur_seg_len):
            if frm_salient_num[fi] > 0:
                top = torch.topk(cur_attn[fi], min(frm_salient_num[fi], frm_len_)).indices
            else:
                top = None

            if frm_context_num[fi] > 0:
                all_i = torch.arange(frm_len_, device=dev)
                remaining = all_i[~torch.isin(all_i, top)] if top is not None else all_i
                tmp = cur_hidden[fi][remaining].unsqueeze(0)             # (1, M, d)
                _, M, D = tmp.shape
                if M == 0:
                    ctx = None
                else:
                    dist_m = torch.cdist(tmp.float(), tmp.float()) / (D ** 0.5)
                    dist_near, _ = torch.topk(dist_m, k=min(4, M), dim=-1, largest=False)
                    density = (-(dist_near ** 2).mean(dim=-1)).exp()
                    density = density + torch.rand(density.shape, device=dev,
                                                   dtype=density.dtype) * 1e-6
                    mask = (density[:, None, :] > density[:, :, None]).type(tmp.dtype)
                    dist_max = dist_m.flatten(1).max(dim=-1)[0][:, None, None]
                    dist_v, _ = (dist_m * mask + dist_max * (1 - mask)).min(dim=-1)
                    score = dist_v * density
                    _, sampled = torch.topk(score, k=min(frm_context_num[fi], M), dim=-1)
                    ctx = remaining[sampled[0].sort().values]
            else:
                ctx = None

            if top is not None:
                cur_salient_idx.append(top + fi * frm_len_)
            if ctx is not None:
                cur_context_idx.append(ctx + fi * frm_len_)

        cur_tome = cur_hidden.reshape(cur_seg_len * frm_len_, -1)

        cur_salient_idx = torch.cat(cur_salient_idx) if cur_salient_idx \
            else torch.empty(0, dtype=torch.long, device=dev)
        cur_salient_hidden = cur_tome[cur_salient_idx]

        all_i    = torch.arange(cur_seg_len * frm_len_, device=dev)
        cur_norm = cur_tome / cur_tome.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ctx_parts = []
        for ctx_i in cur_context_idx:
            retain    = torch.cat([cur_salient_idx, ctx_i])
            merge_idx = all_i[~torch.isin(all_i, retain)]
            target    = cur_norm[ctx_i]
            if merge_idx.shape[0] == 0:
                ctx_parts.append(cur_tome[ctx_i])
                continue
            to_merge = cur_norm[merge_idx]
            sim_ct   = torch.mm(to_merge, target.T)
            oh = torch.zeros(to_merge.shape[0], ctx_i.shape[0],
                             dtype=cur_hidden.dtype, device=dev)
            oh.scatter_(1, sim_ct.argmax(dim=1).unsqueeze(-1), 1)
            avg_w  = (1 / (oh.sum(dim=0).unsqueeze(-1) + 1)).clamp(min=args.fastvid_DTM_beta)
            counts = oh.sum(dim=0).clamp(min=1).unsqueeze(-1)
            agg    = torch.mm(oh.T, cur_tome[merge_idx]) / counts
            tgt    = cur_tome[ctx_i]
            ctx_parts.append(avg_w * tgt + (1 - avg_w) * agg)

        if cur_context_idx:
            cur_context_idx_cat = torch.cat(cur_context_idx)
            cur_context_hidden  = torch.cat(ctx_parts, dim=0)
        else:
            cur_context_idx_cat = torch.empty(0, dtype=torch.long, device=dev)
            cur_context_hidden  = cur_tome[cur_context_idx_cat]

        idx_comb = torch.cat([cur_salient_idx, cur_context_idx_cat])
        hid_comb = torch.cat([cur_salient_hidden, cur_context_hidden], dim=0)
        order = torch.argsort(idx_comb)
        vision_tokens.append(hid_comb[order])
        keep_idx_parts.append(idx_comb[order] + seg_index_start)
        seg_index_start += cur_seg_len * frm_len_

    selected = torch.cat(vision_tokens, dim=0)
    vidx     = torch.cat(keep_idx_parts)
    order    = vidx.argsort()
    return selected[order].to(video_embeds.dtype), vidx[order]


# ── AutoGaze selection ───────────────────────────────────────────────────────

def select_autogaze_tokens(cached, item, ag_model, ag_transform, keep_ratio, device):
    from PIL import Image
    from TrajGazeMerge.training.train_autogaze_lora import (
        compute_ag_scores, _sample_paths, N_AG_FRAMES, FRAME_SIZE,
    )
    video_embeds = cached["video_embeds"]
    n_video   = video_embeds.shape[0]
    T_merged  = int(cached["grid_thw"][0, 0].item())
    H_g       = int(cached["grid_thw"][0, 1].item())
    W_g       = int(cached["grid_thw"][0, 2].item())
    n_spatial = n_video // max(1, T_merged)
    # Qwen spatial grid is (H_g/merge, W_g/merge) and generally non-square.
    merge     = max(1, int(round((H_g * W_g / max(1, n_spatial)) ** 0.5)))
    grid_hw   = (H_g // merge, W_g // merge)

    ag_paths  = _sample_paths(item["vlm_frame_paths"], N_AG_FRAMES)
    ag_frames = [Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE)) for p in ag_paths]
    scores_all = compute_ag_scores(ag_frames, ag_model, ag_transform, T_merged, n_spatial, grid_hw=grid_hw)
    r = max(1, int((1.0 - keep_ratio) * n_video))
    return gaze_weighted_merge(video_embeds, scores_all.to(device), r)


# ── Dispatch ─────────────────────────────────────────────────────────────────

def select_dispatch(mode, base_qwen, cached, item, device, args,
                    ag_model=None, ag_transform=None):
    video_embeds = cached["video_embeds"]
    N = video_embeds.shape[0]
    if mode == "visionzip":
        return visionzip_select_tokens(
            video_embeds, cached["attn_scores"], cached["attn_key"],
            dominant_ratio=args.dominant_ratio, contextual_ratio=args.contextual_ratio,
        )
    if mode == "full":
        idx = torch.arange(N, device=video_embeds.device)
        return video_embeds, idx
    if mode == "random":
        return select_random_tokens(cached, item, args.keep_ratio)
    if mode == "prunevid":
        return select_prunevid_tokens(base_qwen, cached, args.keep_ratio, device)
    if mode == "fastvid":
        return select_fastvid_tokens(cached, args, device)
    if mode == "autogaze":
        return select_autogaze_tokens(cached, item, ag_model, ag_transform,
                                      args.keep_ratio, device)
    raise ValueError(f"unknown select-mode: {mode}")


# ── Training helpers ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--select-mode", required=True,
                   choices=["visionzip", "full", "random", "prunevid", "fastvid", "autogaze"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--grad-accum",  type=int,   default=4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=20)
    p.add_argument("--n-frames",    type=int,   default=128)
    p.add_argument("--dominant-ratio",   type=float, default=DOMINANT_RATIO)
    p.add_argument("--contextual-ratio", type=float, default=CONTEXTUAL_RATIO)
    p.add_argument("--keep-ratio",       type=float, default=KEEP_RATIO)
    p.add_argument("--fastvid-retention",    type=float, default=FASTVID_RETENTION)
    p.add_argument("--fastvid-DySeg-c",      type=int,   default=FASTVID_DYSEG_C)
    p.add_argument("--fastvid-DySeg-tau",    type=float, default=FASTVID_DYSEG_TAU)
    p.add_argument("--fastvid-DySeg-ignore", type=float, default=FASTVID_DYSEG_IGNORE)
    p.add_argument("--fastvid-STPrune-d",    type=float, default=FASTVID_STPRUNE_D)
    p.add_argument("--fastvid-DTM-p",        type=int,   default=FASTVID_DTM_P)
    p.add_argument("--fastvid-DTM-beta",     type=float, default=FASTVID_DTM_BETA)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false")
    p.set_defaults(include_hdepic=True)
    p.add_argument("--source", choices=["sg", "eg", "both"], default="both",
                   help="Train/eval on a single benchmark only (sg=StreamGaze, eg=EgoGazeVQA). "
                        "Filters the combined dataset .items to that source; the per-source "
                        "acc then equals overall acc, driving best.pth + early-stop. "
                        "'both' = unchanged joint protocol (separate/specialist protocol otherwise).")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing (needed for full-token 100%).")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def evaluate(processor, model, base_qwen, option_ids, device, args,
             include_hdepic=True, ag_model=None, ag_transform=None):
    test_ds = CombinedSimpleDataset(split="test", n_vlm_frames=128,
                                    include_hdepic=include_hdepic)
    if args.source in ("sg", "eg"):
        test_ds.items = [it for it in test_ds.items if it[0] == args.source]
    model.eval()
    correct = 0
    total   = 0
    by_task: dict[str, list] = {}
    by_src:  dict[str, list] = {}

    with torch.no_grad():
        for idx in range(len(test_ds)):
            src  = test_ds.items[idx][0]
            item = test_ds[idx]
            if item is None:
                continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue
                selected_embeds, receiver_idx = select_dispatch(
                    args.select_mode, base_qwen, cached, item, device, args,
                    ag_model=ag_model, ag_transform=ag_transform,
                )
                inputs_dict = build_merged_inputs(base_qwen, cached, selected_embeds, receiver_idx)

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                logits   = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx   = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)
                by_src.setdefault(src, []).append(ok)
            except Exception:
                pass

    model.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    per_src  = {s: [100.0 * sum(v) / max(1, len(v)), len(v)] for s, v in sorted(by_src.items())}
    return 100.0 * correct / max(1, total), total, per_task, per_src


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[baseline:{args.select_mode}] output: {args.output_dir}")
        print(f"[baseline:{args.select_mode}] GPUs={world_size}, epochs={args.epochs}, "
              f"lr={args.lr}, grad_accum={args.grad_accum}, hdepic={args.include_hdepic}")

    ag_model = ag_transform = None
    if args.select_mode == "autogaze":
        if is_main:
            print("Loading AutoGaze (frozen) ...")
        from TrajGazeMerge.training.train_autogaze_lora import load_autogaze
        ag_transform, ag_model = load_autogaze(device)
        if is_main:
            print("AutoGaze loaded.")

    if is_main:
        print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_visionzip_lora(device)
    base_qwen  = model.get_base_model()
    if args.grad_checkpoint:
        base_qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        base_qwen.config.use_cache = False
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False, static_graph=True)
        if is_main:
            print("Gradient checkpointing ENABLED (use_reentrant=False, static_graph).")
    else:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)
    if is_main:
        print("Model loaded.")

    train_ds = CombinedSimpleDataset(split="train", n_vlm_frames=args.n_frames,
                                     include_hdepic=args.include_hdepic)
    if args.source in ("sg", "eg"):
        n_before = len(train_ds.items)
        train_ds.items = [it for it in train_ds.items if it[0] == args.source]
        if is_main:
            print(f"[source={args.source}] train filtered {n_before} → "
                  f"{len(train_ds.items)} items", flush=True)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0
    epoch_accs: list[float] = []

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
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device,
                    )
                if cached is None:
                    continue
                n_video = cached["video_embeds"].shape[0]

                with torch.no_grad():
                    selected_embeds, receiver_idx = select_dispatch(
                        args.select_mode, base_qwen, cached, item, device, args,
                        ag_model=ag_model, ag_transform=ag_transform,
                    )
                    inputs_dict = build_merged_inputs(base_qwen, cached, selected_embeds, receiver_idx)

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                loss = F.cross_entropy(option_logits.unsqueeze(0),
                                       torch.tensor([gt_idx], device=device))
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
                    pct_kept = 100.0 * receiver_idx.shape[0] / max(1, n_video)
                    print(f"[{args.select_mode}] Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s",
                          flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"epoch": epoch + 1, "step": n_steps,
                                            "loss": avg_loss, "pct_kept": pct_kept,
                                            "elapsed": elapsed}) + "\n")
            except Exception:
                if is_main:
                    traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed  = time.time() - t_start
        if is_main:
            print(f"\n=== [{args.select_mode}] Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===", flush=True)
            torch.save({"epoch": epoch, "lora_state": model.module.state_dict(),
                        "loss": avg_loss},
                       os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on egtea val ...", flush=True)
            acc, n_eval, per_task, per_src = evaluate(
                processor, model.module, base_qwen, option_ids, device, args,
                include_hdepic=args.include_hdepic,
                ag_model=ag_model, ag_transform=ag_transform,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for s, (sacc, sn) in per_src.items():
                print(f"    [src] {s}: {sacc:.2f}%  (n={sn})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            epoch_accs.append(acc)
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "eval_acc": acc, "n_eval": n_eval,
                                    "per_task": per_task, "per_src": per_src}) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({"epoch": epoch, "lora_state": model.module.state_dict(),
                            "acc": acc}, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
        dist.barrier()

        stop = torch.zeros(1, device=device)
        if is_main and args.early_stop and (epoch + 1) == 2 and len(epoch_accs) >= 2 \
                and epoch_accs[1] <= epoch_accs[0]:
            stop[0] = 1.0
            print(f"  Early stop: epoch2 {epoch_accs[1]:.2f}% <= epoch1 "
                  f"{epoch_accs[0]:.2f}% → skip epoch 3.", flush=True)
        dist.broadcast(stop, src=0)
        if stop.item() > 0:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
