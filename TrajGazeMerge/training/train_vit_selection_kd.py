"""Phase 1 — distil \\sys's 10% token selection into the ViT's own attention.

\\sys (M1) keeps 10% of the visual tokens = 7% VisionZip content ∪ 3% trajectory
complement, where the complement is ranked by a gaze/hand salience field (frozen
TAS encoder). Two things are therefore needed at inference: an eye-tracker, and a
36.85M encoder to run over every frame. The KD student replaces both with a 3.95M
RGB head (train_visionzip_kd_lora.py).

This is the ablation that asks whether the selection can live in the ViT instead of
in ANY extra module. VisionZip's dominant score is the last ViT block's attention
column-sum; if that score can be taught to rank M1's chosen 10% at the top, then at
inference the method collapses to plain VisionZip — zero extra parameters, no
eye-tracker, no TAS encoder.

  teacher target   S_T = content_idx ∪ traj_idx   (M1's exact 10%, frozen ViT)
  student score    the ViT's own attn_scores, with a rank-8 LoRA on block 31
  objective        BCE(standardised score, 1[t ∈ S_T]) + feature anchor

Only block 31 is adapted: the score is a function of that block's q,k alone, so
blocks 0..30 stay bit-identical and the representation cannot drift there. The
anchor loss holds block 31's *output* near the frozen encoder's, and the hard gate
is a 100%-token eval (docs/kd_handoff_v2.md §8's ±4-item noise floor).

Selection is non-differentiable (top-k), so the LLM is not in this graph at all:
the LoRA is warm-started from M1 and frozen, there is no task CE, and no VLM
forward. Phase 2 (train_visionzip_lora.py --vit-lora-ckpt) re-adapts the readout.

Usage (2-GPU DDP-free; only 61k params are trained, grads are all-reduced by hand):
    GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=0 \\
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29671 \\
      -m TrajGazeMerge.training.train_vit_selection_kd \\
      --source sg --warmstart-ckpt "$M1_SGONLY" --stage1-ckpt "$STAGE1_CKPT" \\
      --output-dir .../vitkd_SGonly_nooverlay --epochs 2 --no-hdepic
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"))

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, visionzip_select_tokens, VIDEO_KWARGS,
)
from TrajGazeMerge.training.train_visionzip_kd_lora import (
    content_and_avail, topk_in_avail,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder
from TrajGazeMerge.training.train_visionzip_complement_lora import _traj_scores


# ── ViT LoRA ──────────────────────────────────────────────────────────────────
#
# Hand-rolled rather than peft, for three reasons that all bit at once here:
#   * peft's target_modules would have to be a regex; the plain strings "qkv"/"proj"
#     match all 32 blocks AND patch_embed.proj (a Conv3d).
#   * the LLM already carries a peft adapter, and disable_adapter() is global — the
#     frozen reference pass must disable the ViT adapter ONLY.
#   * PeftModel.state_dict() returns the whole 8.29B backbone (see §12.1); we want a
#     2 MB file so mid-epoch checkpointing is free.

class LoRALinear(nn.Module):
    """y = W x + (B A x) * (alpha/r), with W frozen and B zero-initialised."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # B stays 0 → identity at init
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        # Cast the (tiny) adapter weights to the activation dtype instead of the
        # (huge) activations to fp32 — same result, no [T, 1280] fp32 copy.
        h = F.linear(x, self.lora_A.to(x.dtype))
        return out + F.linear(h, self.lora_B.to(x.dtype)) * self.scaling


def attach_vit_lora(base_qwen, r=8, alpha=16, block_idx=-1):
    """Wrap the last ViT block's qkv and proj. Returns the wrapper list."""
    blk = base_qwen.visual.blocks[block_idx]
    wrappers = []
    for name in ("qkv", "proj"):
        base_lin = getattr(blk.attn, name)
        if isinstance(base_lin, LoRALinear):
            wrappers.append(base_lin)
            continue
        w = LoRALinear(base_lin, r=r, alpha=alpha).to(base_lin.weight.device)
        setattr(blk.attn, name, w)
        wrappers.append(w)
    return wrappers


@contextlib.contextmanager
def vit_lora_disabled(wrappers):
    """Frozen-encoder reference pass: the ViT adapter off, LLM adapter untouched."""
    prev = [w.enabled for w in wrappers]
    for w in wrappers:
        w.enabled = False
    try:
        yield
    finally:
        for w, p in zip(wrappers, prev):
            w.enabled = p


def vit_lora_state(wrappers):
    return {f"{i}.{k}": v.detach().float().cpu()
            for i, w in enumerate(wrappers)
            for k, v in (("lora_A", w.lora_A), ("lora_B", w.lora_B))}


def load_vit_lora_state(wrappers, state):
    for i, w in enumerate(wrappers):
        w.lora_A.data.copy_(state[f"{i}.lora_A"].to(w.lora_A.device, w.lora_A.dtype))
        w.lora_B.data.copy_(state[f"{i}.lora_B"].to(w.lora_B.device, w.lora_B.dtype))


# ── video-only preprocessing ──────────────────────────────────────────────────
#
# Phase 1 never runs the LLM, so unlike preprocess_visionzip_item this keeps the
# pixel tensor (the visual encoder has to be called twice — frozen and adapted)
# and skips rope/position_ids entirely.

def preprocess_video(processor, base_qwen, frame_paths, question, options, device):
    from qwen_vl_utils import process_vision_info

    options_text = "\n".join(options)
    prompt = f"{question}\nOptions:\n{options_text}\nAnswer with a single letter."
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frame_paths,
         "max_pixels": VIDEO_KWARGS["max_pixels"],
         "min_pixels": VIDEO_KWARGS["min_pixels"],
         "fps": VIDEO_KWARGS["fps"]},
        {"type": "text", "text": prompt}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           **video_kwargs, return_tensors="pt")
    except Exception:
        return None
    if "pixel_values_videos" not in inputs:
        return None
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device
    return (inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16),
            inputs["video_grid_thw"].to(vis_dev))


# ── teacher target ────────────────────────────────────────────────────────────

def teacher_selection(cached_frozen, item, device, encoder, hp,
                      content_ratio, traj_ratio):
    """M1's exact 10%: VisionZip content ∪ gaze/hand complement, on the FROZEN ViT.

    Frozen on purpose — if the target moved with the adapter the objective would be
    self-referential and the score could satisfy it by collapsing.
    """
    n_video = cached_frozen["video_embeds"].shape[0]
    _, content_idx, avail_idx = content_and_avail(cached_frozen, content_ratio)
    if avail_idx.numel() == 0:
        return None, None
    k_traj = min(max(1, int(traj_ratio * n_video)), avail_idx.numel())
    s_teacher = _traj_scores(cached_frozen, item, device, "learned", encoder, hp)
    traj_idx, _ = topk_in_avail(s_teacher, avail_idx, k_traj)
    return torch.cat([content_idx, traj_idx]).sort().values, traj_idx


def selection_bce(score, target_idx, n_video):
    """BCE over all N tokens: does the ViT's own score rank S_T at the top?

    The score is a positive column-sum, not a logit, so it is standardised per item
    first — the same convention TrajSaliencePredictor uses for its attn input, and
    the only scale-free choice that keeps pos_weight meaningful.
    """
    z = (score - score.mean()) / (score.std() + 1e-6)
    tgt = torch.zeros_like(z)
    tgt[target_idx] = 1.0
    n_pos = tgt.sum().clamp(min=1.0)
    n_neg = (n_video - n_pos).clamp(min=1.0)
    pos_weight = (n_neg / n_pos).clamp(max=50.0)
    return F.binary_cross_entropy_with_logits(z, tgt, pos_weight=pos_weight)


@torch.no_grad()
def selection_metrics(ve, score, key, target_idx, traj_idx, dom_p, ctx_p):
    """How much of the teacher's 10% the student's selection actually recovers.

    recall_P uses the shipped selector at the primary split (raw 6.5% + merged 3.5%),
    which is what Phase 2 and every reported number will use. recall_S is the pure
    attention top-10%, i.e. the ceiling if the contextual mechanism were dropped.
    recall_traj is the part that matters: the gaze complement, which attention had
    no way to see before distillation.
    """
    n = ve.shape[0]
    tgt = set(target_idx.tolist())
    _, idx_p = visionzip_select_tokens(ve, score, key,
                                       dominant_ratio=dom_p, contextual_ratio=ctx_p)
    k_s = max(1, int((dom_p + ctx_p) * n))
    idx_s = torch.topk(score, k_s).indices
    dom_k = max(1, int(dom_p * n))
    dom_idx = set(torch.topk(score, dom_k).indices.tolist())
    return {
        "recall_P": len(tgt & set(idx_p.tolist())) / max(1, len(tgt)),
        "recall_S": len(tgt & set(idx_s.tolist())) / max(1, len(tgt)),
        "recall_traj": len(set(traj_idx.tolist()) & dom_idx) / max(1, traj_idx.numel()),
    }


# ── checkpointing ─────────────────────────────────────────────────────────────

def atomic_save(obj, path):
    """Write-then-rename: a kill during torch.save must not leave a truncated file
    that --resume would then load. Three runs have already been lost to interruption
    (§13.4), so the recovery path itself has to be crash-safe."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_ckpt(path, wrappers, optimizer, epoch, step, extra=None):
    # r/alpha travel with the weights so Phase 2 cannot silently rebuild the adapter
    # with a different scaling than the one that was trained.
    atomic_save({"vit_lora_state": vit_lora_state(wrappers),
                 "vit_lora_r": wrappers[0].r,
                 "vit_lora_alpha": int(round(wrappers[0].scaling * wrappers[0].r)),
                 "opt_state": optimizer.state_dict(),
                 "epoch": epoch, "step": step, **(extra or {})}, path)


def find_resume(output_dir):
    """Newest usable checkpoint: prefer a mid-epoch step file over the epoch file
    when it is further along."""
    import re
    best, best_key = None, (-1, -1)
    if not os.path.isdir(output_dir):
        return None
    for fn in os.listdir(output_dir):
        m = re.fullmatch(r"epoch_(\d+)\.pth", fn)
        if m:
            key = (int(m.group(1)), 1 << 30)
        else:
            m = re.fullmatch(r"step_latest\.pth", fn)
            if not m:
                continue
            try:
                st = torch.load(os.path.join(output_dir, fn), map_location="cpu")
            except Exception:
                continue
            key = (st.get("epoch", 0), st.get("step", 0))
        if key > best_key:
            best_key, best = key, os.path.join(output_dir, fn)
    return best


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warmstart-ckpt", default="",
                   help="M1 LoRA best.pth. Loaded then FROZEN — Phase 1 never trains it.")
    p.add_argument("--stage1-ckpt", required=True,
                   help="Frozen TAS Stage-1 encoder = the privileged gaze/hand teacher.")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio", type=float, default=0.03)
    p.add_argument("--dom-primary", type=float, default=0.065,
                   help="Student dominant ratio at eval (P split). content/2 + traj.")
    p.add_argument("--ctx-primary", type=float, default=0.035,
                   help="Student contextual ratio at eval (P split). content/2.")
    p.add_argument("--vit-lora-r", type=int, default=8)
    p.add_argument("--vit-lora-alpha", type=int, default=16)
    p.add_argument("--lambda-sel", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=1.0,
                   help="Weight on 1-cos(video_embeds_adapted, video_embeds_frozen). "
                        "Raise if the 100%-token integrity gate fails.")
    p.add_argument("--score-query-frac", type=float, default=1.0,
                   help="Train-time subsampling of query chunks in the score sum. "
                        "Unbiased up to a constant; eval always uses 1.0.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every-steps", type=int, default=200,
                   help="Mid-epoch adapter-only checkpoint (~250 KB). 0 disables. "
                        "The epoch-end-only trainers lose everything on a mid-epoch "
                        "death (§13.4); this is what makes auto-resume possible.")
    p.add_argument("--eval-items", type=int, default=0,
                   help="Cap on test items scored per epoch (0 = all).")
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false")
    p.set_defaults(include_hdepic=True)
    p.add_argument("--source", choices=["sg", "eg", "both"], default="both")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the newest checkpoint in --output-dir. Safe on "
                        "a fresh run (no checkpoint = normal start), so a supervisor "
                        "can relaunch the same command after a crash.")
    p.add_argument("--eval-only", default=None,
                   help="Load this vit_lora ckpt, report selection metrics, exit.")
    return p.parse_args()


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_selection(processor, base_qwen, wrappers, encoder, hp, args, device,
                       max_items=0):
    """Selection-only eval: no LLM forward, so an epoch boundary costs minutes.
    End-task accuracy comes from Phase 2 / the integrity gate, not from here."""
    ds = CombinedMergeDataset(split="test", n_vlm_frames=args.n_frames,
                              n_traj_frames=args.n_frames,
                              include_hdepic=args.include_hdepic)
    if args.source in ("sg", "eg"):
        ds.items = [it for it in ds.items if it[0] == args.source]
    n = len(ds) if not max_items else min(max_items, len(ds))
    agg = {"recall_P": 0.0, "recall_S": 0.0, "recall_traj": 0.0}
    base_agg = dict(agg)
    scored = 0
    for i in range(n):
        item = ds[i]
        if item is None:
            continue
        try:
            pack = preprocess_video(processor, base_qwen, item["vlm_frame_paths"],
                                    item["question"], item["options"], device)
            if pack is None:
                continue
            pv, gthw = pack
            with vit_lora_disabled(wrappers):
                ve_f, sc_f, key_f = base_qwen.visual(pv, grid_thw=gthw)
            cached_f = {"video_embeds": ve_f, "attn_scores": sc_f,
                        "attn_key": key_f, "grid_thw": gthw}
            S_T, traj_idx = teacher_selection(cached_f, item, device, encoder, hp,
                                              args.content_ratio, args.traj_ratio)
            if S_T is None:
                continue
            ve_a, sc_a, key_a = base_qwen.visual(pv, grid_thw=gthw)
            m = selection_metrics(ve_a, sc_a, key_a, S_T, traj_idx,
                                  args.dom_primary, args.ctx_primary)
            b = selection_metrics(ve_f, sc_f, key_f, S_T, traj_idx,
                                  args.dom_primary, args.ctx_primary)
            for k in agg:
                agg[k] += m[k]
                base_agg[k] += b[k]
            scored += 1
        except Exception:
            traceback.print_exc()
            continue
    if scored == 0:
        return None, None, 0
    return ({k: v / scored for k, v in agg.items()},
            {k: v / scored for k, v in base_agg.items()}, scored)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        gz = os.environ.get("GAZE_OVERLAY", "1")
        vgz = os.environ.get("VLM_GAZE_OVERLAY", gz)
        print(f"[ViT-KD] out={args.output_dir}", flush=True)
        print(f"[ViT-KD] GPUs={world_size} source={args.source} seed={args.seed} "
              f"GAZE_OVERLAY={gz} VLM_GAZE_OVERLAY={vgz}", flush=True)
        print(f"[ViT-KD] target = content {args.content_ratio:.3f} ∪ traj {args.traj_ratio:.3f}"
              f" | student eval split P = {args.dom_primary:.3f}+{args.ctx_primary:.3f}",
              flush=True)
        print(f"[ViT-KD] ViT LoRA r={args.vit_lora_r} a={args.vit_lora_alpha} on "
              f"visual.blocks[-1].attn.{{qkv,proj}} | lr={args.lr} "
              f"λ_sel={args.lambda_sel} λ_anchor={args.lambda_anchor} "
              f"query_frac={args.score_query_frac}", flush=True)

    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()

    if args.warmstart_ckpt and os.path.exists(args.warmstart_ckpt):
        st = torch.load(args.warmstart_ckpt, map_location="cpu")
        sd = st["lora_state"] if "lora_state" in st else st
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[ViT-KD] warm-started LLM LoRA from {args.warmstart_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    elif is_main:
        print(f"[ViT-KD] WARNING: no warm-start at '{args.warmstart_ckpt}'", flush=True)

    # Everything frozen; only the ViT adapter below will train.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    wrappers = attach_vit_lora(base_qwen, r=args.vit_lora_r, alpha=args.vit_lora_alpha)
    params = [p for w in wrappers for p in (w.lora_A, w.lora_B)]
    for p in params:
        p.requires_grad_(True)
    if is_main:
        print(f"[ViT-KD] trainable: {sum(p.numel() for p in params)/1e6:.3f}M params "
              f"({len(params)} tensors)", flush=True)

    # No DDP: the trainable set is 61k params living inside a frozen 8.29B graph, so
    # a hand-rolled all-reduce is both simpler and cheaper than wrapping the model.
    for p in params:
        dist.broadcast(p.data, src=0)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, 16)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    hp = dict(mask_modality="none")

    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-4)

    start_epoch, start_step = 0, 0
    if args.resume:
        ck = find_resume(args.output_dir)
        if ck:
            st = torch.load(ck, map_location="cpu")
            load_vit_lora_state(wrappers, st["vit_lora_state"])
            if "opt_state" in st:
                optimizer.load_state_dict(st["opt_state"])
            start_epoch, start_step = st.get("epoch", 0), st.get("step", 0)
            if is_main:
                print(f"[ViT-KD] resumed from {ck} at epoch {start_epoch} step {start_step}",
                      flush=True)
        elif is_main:
            print(f"[ViT-KD] --resume: nothing in {args.output_dir}; fresh start", flush=True)

    if args.eval_only:
        st = torch.load(args.eval_only, map_location="cpu")
        load_vit_lora_state(wrappers, st["vit_lora_state"])
        if is_main:
            tuned, frozen, n = evaluate_selection(processor, base_qwen, wrappers,
                                                  encoder, hp, args, device,
                                                  args.eval_items)
            print(f"[eval-only] n={n}", flush=True)
            for k in ("recall_P", "recall_S", "recall_traj"):
                print(f"[eval-only] {k}: frozen {frozen[k]:.4f} → tuned {tuned[k]:.4f}",
                      flush=True)
        dist.barrier(); dist.destroy_process_group(); return

    train_ds = CombinedMergeDataset(split="train", n_vlm_frames=args.n_frames,
                                    n_traj_frames=args.n_frames,
                                    include_hdepic=args.include_hdepic)
    if args.source in ("sg", "eg"):
        n_before = len(train_ds.items)
        train_ds.items = [it for it in train_ds.items if it[0] == args.source]
        if is_main:
            print(f"[source={args.source}] train {n_before} → {len(train_ds.items)} items",
                  flush=True)

    # Same structural check the KD trainer makes: when VLM_GAZE_OVERLAY differs from
    # GAZE_OVERLAY the two streams must resolve to different directories, or the
    # "marker-free" run is silently training on the marker.
    if is_main:
        probe = next((it for it in (train_ds[i] for i in range(min(50, len(train_ds))))
                      if it is not None), None)
        if probe is None:
            raise RuntimeError("[ViT-KD] stream check: first 50 train items all None")
        s_dir = os.path.basename(os.path.dirname(os.path.dirname(probe["vlm_frame_paths"][0])))
        t_dir = os.path.basename(os.path.dirname(os.path.dirname(probe["traj_frame_paths"][0])))
        print(f"[ViT-KD] frame streams: student VLM='{s_dir}'  teacher TAS='{t_dir}'",
              flush=True)
        want_diff = os.environ.get("VLM_GAZE_OVERLAY",
                                   os.environ.get("GAZE_OVERLAY", "1")) != \
                    os.environ.get("GAZE_OVERLAY", "1")
        if want_diff and s_dir == t_dir:
            raise RuntimeError(
                f"[ViT-KD] VLM_GAZE_OVERLAY differs from GAZE_OVERLAY but both streams "
                f"read '{s_dir}'. The ViT would be distilled on the teacher's frames.")

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best = -1.0

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        ep_sel = ep_anc = ep_rp = ep_rt = 0.0
        n_steps = 0
        # Windowed as well as cumulative: a running mean over a 2900-step epoch is
        # dominated by its own history and cannot show whether the last 20 steps
        # improved, which is the only thing an hourly health check can act on.
        win_sel = win_anc = win_rp = win_rt = 0.0
        win_n = 0
        t0 = time.time()

        for step, item in enumerate(loader):
            # Resume lands mid-epoch: skip what this rank already consumed. The
            # sampler is seeded per epoch, so the order is identical on replay.
            if epoch == start_epoch and step < start_step:
                continue
            if item is None:
                continue
            try:
                pack = preprocess_video(processor, base_qwen, item["vlm_frame_paths"],
                                        item["question"], item["options"], device)
                if pack is None:
                    continue
                pv, gthw = pack

                # 1. frozen reference: teacher target + anchor reference
                with torch.no_grad(), vit_lora_disabled(wrappers):
                    ve_f, sc_f, key_f = base_qwen.visual(pv, grid_thw=gthw)
                cached_f = {"video_embeds": ve_f, "attn_scores": sc_f,
                            "attn_key": key_f, "grid_thw": gthw}
                S_T, traj_idx = teacher_selection(cached_f, item, device, encoder, hp,
                                                  args.content_ratio, args.traj_ratio)
                if S_T is None:
                    continue
                n_video = ve_f.shape[0]

                # 2. adapted pass — the only one carrying gradient
                ve_a, sc_a, key_a = base_qwen.visual(
                    pv, grid_thw=gthw, grad_last_block=True,
                    score_query_frac=args.score_query_frac)

                sel = selection_bce(sc_a, S_T, n_video)
                anchor = 1.0 - F.cosine_similarity(
                    ve_a.float(), ve_f.float(), dim=-1).mean()
                loss = (args.lambda_sel * sel + args.lambda_anchor * anchor) / args.grad_accum
                loss.backward()

                with torch.no_grad():
                    m = selection_metrics(ve_a.detach(), sc_a.detach(), key_a,
                                          S_T, traj_idx, args.dom_primary, args.ctx_primary)
                ep_sel += sel.item(); ep_anc += anchor.item()
                ep_rp += m["recall_P"]; ep_rt += m["recall_traj"]
                win_sel += sel.item(); win_anc += anchor.item()
                win_rp += m["recall_P"]; win_rt += m["recall_traj"]
                n_steps += 1; win_n += 1

                if (step + 1) % args.grad_accum == 0:
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if is_main and (step + 1) % args.log_every == 0:
                    w = max(1, win_n)
                    print(f"Epoch {epoch+1} | step {step+1}/{len(loader)} | "
                          f"sel={win_sel/w:.4f} | anc={win_anc/w:.5f} | "
                          f"recall_P={win_rp/w:.3f} | recall_traj={win_rt/w:.3f} | "
                          f"[cum sel={ep_sel/max(1,n_steps):.4f} "
                          f"rtraj={ep_rt/max(1,n_steps):.3f}] | "
                          f"t={int(time.time()-t0)}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": step + 1,
                            "win_sel": win_sel / w, "win_anchor": win_anc / w,
                            "win_recall_P": win_rp / w, "win_recall_traj": win_rt / w,
                        }) + "\n")
                    win_sel = win_anc = win_rp = win_rt = 0.0
                    win_n = 0

                if (is_main and args.ckpt_every_steps
                        and (step + 1) % args.ckpt_every_steps == 0):
                    save_ckpt(os.path.join(args.output_dir, "step_latest.pth"),
                              wrappers, optimizer, epoch, step + 1)

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[rank{rank}] OOM at step {step}, skipped", flush=True)
                continue
            except Exception:
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)
                continue

        start_step = 0        # only the resumed epoch is partially skipped
        dist.barrier()

        if is_main:
            tuned, frozen, n_ev = evaluate_selection(
                processor, base_qwen, wrappers, encoder, hp, args, device,
                args.eval_items)
            rec = {"epoch": epoch + 1, "n_eval": n_ev,
                   "train_sel": ep_sel / max(1, n_steps),
                   "train_anchor": ep_anc / max(1, n_steps),
                   "tuned": tuned, "frozen": frozen}
            print(f"[ViT-KD] epoch {epoch+1} eval (n={n_ev}):", flush=True)
            for k in ("recall_P", "recall_S", "recall_traj"):
                print(f"[ViT-KD]   {k}: frozen {frozen[k]:.4f} → tuned {tuned[k]:.4f} "
                      f"(Δ {tuned[k]-frozen[k]:+.4f})", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            save_ckpt(os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"),
                      wrappers, optimizer, epoch + 1, 0, extra={"metrics": tuned})
            if tuned["recall_P"] > best:
                best = tuned["recall_P"]
                save_ckpt(os.path.join(args.output_dir, "best.pth"),
                          wrappers, optimizer, epoch + 1, 0, extra={"metrics": tuned})
                print(f"[ViT-KD] new best recall_P={best:.4f} → best.pth", flush=True)
        dist.barrier()

    if is_main:
        print(f"[ViT-KD] TRAINING COMPLETE, best recall_P={best:.4f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
