"""
TrajGazeMerge Stage 2: joint training of TrajGaze encoder + Qwen LoRA.

Mechanism:
  - Teacher pass : fine-tuned baseline LoRA Qwen + full visual tokens → logits_teacher
  - Student pass : Qwen + LoRA + gaze-merged tokens (10% budget)      → logits_student
  - Loss         : α * KL(student || teacher) + (1-α) * CE(student, label)
  - Trainable    : TrajGaze encoder + LoRA adapters (ViT frozen)

Gradient flow:
  loss → student logits → LoRA weights
                        → merged_tokens → merge op (diff. weighted avg)
                                       → patch_scores → TrajGaze encoder

Train : egoexolearn + holoassist
Test  : egtea (periodic eval)
GPUs  : 2 via torchrun

Usage:
    torchrun --nproc_per_node=2 -m TrajGazeMerge.training.train_merge_lora \
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \
        --teacher-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora \
        --epochs 3 --lr-lora 1e-4 --lr-enc 1e-5 --alpha 0.5 \
        --merge-ratio 0.9 --grad-accum 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import (
    gaze_weighted_merge,
    gaze_weighted_merge_per_frame,
    score_to_qwen_spatial,
)
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, build_full_inputs, forward_logits, forward_logits_ext,
    TrajHintProjection,
)
from TrajGaze_v2.models.model import TrajGazeV2

STAGE1_CKPT  = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"
OUTPUT_ROOT  = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/merge_lora"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",  default=STAGE1_CKPT)
    p.add_argument("--teacher-ckpt", default=None,
                   help="Path to trained baseline LoRA checkpoint for teacher")
    p.add_argument("--output-dir",   default=OUTPUT_ROOT)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr-lora",      type=float, default=1e-4)
    p.add_argument("--lr-enc",       type=float, default=1e-5)
    p.add_argument("--alpha",        type=float, default=0.5,
                   help="KL loss weight; (1-alpha) for CE loss")
    p.add_argument("--merge-ratio",  type=float, default=0.9,
                   help="Fraction of visual tokens to merge away (0.9 = keep 10%%)")
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--log-every",    type=int,   default=20)
    p.add_argument("--eval-every",   type=int,   default=200)
    p.add_argument("--n-frames",     type=int,   default=128)
    p.add_argument("--n-traj-frames", type=int,  default=32)
    p.add_argument("--resume-ckpt",  default=None,
                   help="Path to checkpoint to resume from (loads lora_state + encoder_state)")
    p.add_argument("--start-epoch",  type=int,   default=0,
                   help="Epoch to start from when resuming")

    # ── Drop + score-aware merging ────────────────────────────────────────────
    p.add_argument("--drop-ratio",   type=float, default=0.0,
                   help="Fraction of N to fully drop (must satisfy 0 <= drop_ratio <= merge_ratio). "
                        "kept = N - r_merge - r_drop = round((1 - merge_ratio) * N).")
    p.add_argument("--score-transform", type=str, default="sigmoid",
                   choices=["sigmoid", "softplus", "none"],
                   help="Normalize raw TrajGaze scores before using as merge weights.")
    p.add_argument("--match-score-penalty", type=float, default=0.0,
                   help="λ for score-aware matching: sim_eff = sim - λ·|w_r - w_s|.")
    p.add_argument("--match-score-hard-gap", type=float, default=None,
                   help="Hard exclusion: receivers with (w_r - w_s) > gap cannot absorb the source.")

    # ── Dynamic α (teacher confidence-aware KD) ───────────────────────────────
    p.add_argument("--alpha-mode",  type=str, default="static",
                   choices=["static", "entropy", "p_gt"],
                   help="How to set the KL weight α per sample.")
    p.add_argument("--alpha-min",   type=float, default=0.1)
    p.add_argument("--alpha-max",   type=float, default=0.9)

    # ── 축 1: Merge scope (frame budget allocation) ───────────────────────────
    p.add_argument("--merge-scope", type=str, default="legacy",
                   choices=["legacy", "per_frame", "global"],
                   help="legacy (default): existing behavior — global sort, "
                        "inter-frame merge partner allowed. "
                        "per_frame: equal budget per frame, intra-frame match. "
                        "global: per-frame budget from frame_scores, intra-frame match.")
    p.add_argument("--k-min", type=int, default=1,
                   help="Minimum receiver tokens per frame (M-RoPE hole avoidance).")
    p.add_argument("--budget-temp", type=float, default=1.0,
                   help="Softmax temperature for global per-frame budget allocation.")

    # ── 축 2: ViT adapter (opt-in) ────────────────────────────────────────────
    p.add_argument("--vit-unfreeze-last-n", type=int, default=0,
                   help="Unfreeze last N ViT blocks end-to-end (0=off).")
    p.add_argument("--vit-lora-rank", type=int, default=0,
                   help="Attach LoRA rank R to last N ViT blocks (0=off).")
    p.add_argument("--vit-lora-last-n", type=int, default=2,
                   help="Number of last ViT blocks to target when --vit-lora-rank > 0.")
    p.add_argument("--lr-vit", type=float, default=1e-5,
                   help="Learning rate for unfrozen/LoRA ViT params.")

    # ── 축 3: Teacher-transcending KD ─────────────────────────────────────────
    p.add_argument("--kd-gate", type=str, default="none",
                   choices=["none", "correct", "confidence"],
                   help="Gate KL loss on teacher correctness. "
                        "correct: KL only when teacher argmax==GT. "
                        "confidence: KL only when p_t[gt] > --kd-gate-tau.")
    p.add_argument("--kd-gate-tau", type=float, default=0.5,
                   help="Threshold for --kd-gate confidence mode.")
    p.add_argument("--alpha-schedule", type=str, default="static",
                   choices=["static", "warmup_ce", "cosine"],
                   help="Epoch-level α schedule. warmup_ce: KL-heavy early → CE-heavy late.")
    p.add_argument("--kd-antiteacher-weight", type=float, default=0.0,
                   help="Margin-ranking loss weight when teacher is wrong. "
                        "Very conservative; start at 0.01.")
    p.add_argument("--kd-antiteacher-margin", type=float, default=1.0,
                   help="Margin m in max(0, m - (logit_s[gt] - logit_s[teacher_argmax])).")

    # ── 축 4: KD supervision extension ────────────────────────────────────────
    p.add_argument("--kd-seq", type=str, default="last",
                   choices=["last", "answer_full"],
                   help="Logit KD region. last: last-position only (default). "
                        "answer_full: last ~8 positions (useful with CoT answers).")
    p.add_argument("--kd-feat-layers", type=str, default="",
                   help="Comma-separated LLM layer indices for feature-MSE KD, e.g. '16,28'. "
                        "Empty string = off.")
    p.add_argument("--kd-feat-weight", type=float, default=0.5,
                   help="Weight of feature-level MSE loss.")
    p.add_argument("--kd-feat-region", type=str, default="context_only_pooled",
                   choices=["context_only_pooled", "per_token"],
                   help="pooled (safer): mean over text-region hidden states. "
                        "per_token: element-wise MSE (requires --align-text-position).")
    p.add_argument("--align-text-position", type=str, default="none",
                   choices=["none", "teacher"],
                   help="Override student's text-region M-RoPE to match teacher. "
                        "Strongly recommended (auto-enforced) with --kd-feat-region per_token.")

    # ── 축 5: Encoder intermediate aux paths (opt-in) ─────────────────────────
    p.add_argument("--aux-traj-tokens", type=int, default=0,
                   help="Number of trajectory hint tokens to inject into prompt (0=off). "
                        "Requires encoder to expose traj_embeds in its forward extras.")
    p.add_argument("--aux-traj-hidden", type=int, default=256,
                   help="Expected dim of encoder traj_embeds (for TrajHintProjection init).")
    p.add_argument("--aux-traj-forecast-weight", type=float, default=0.0,
                   help="Weight of future-trajectory forecasting aux loss (0=off). "
                        "Requires encoder to expose 'forecast' tensor + GT supervision.")
    return p.parse_args()


def compute_dynamic_alpha(
    logits_teacher: torch.Tensor,   # (num_options,) teacher logits over A/B/C/D
    gt_idx:         int,
    mode:           str,
    alpha_static:   float,
    alpha_min:      float,
    alpha_max:      float,
) -> torch.Tensor:
    """
    Return a scalar α controlling KL vs CE mixing: loss = α·KL + (1-α)·CE.

    - static : return args.alpha unchanged.
    - p_gt   : α scales with teacher's confidence on GT (high conf → more KL).
    - entropy: α scales with (1 - H/log(K)) — high certainty → more KL.
    """
    if mode == "static":
        return torch.tensor(alpha_static, device=logits_teacher.device,
                            dtype=logits_teacher.dtype)

    p_t = torch.softmax(logits_teacher.float(), dim=-1)
    K = p_t.shape[-1]

    if mode == "p_gt":
        conf = p_t[gt_idx]
    else:  # entropy
        H = -(p_t * p_t.clamp(min=1e-8).log()).sum()
        conf = 1.0 - H / math.log(K)

    alpha_t = alpha_min + (alpha_max - alpha_min) * conf
    return alpha_t.clamp(alpha_min, alpha_max).to(logits_teacher.dtype)


def compute_alpha_schedule(
    alpha_base:  float,
    mode:        str,
    epoch:       int,
    total_epochs: int,
) -> float:
    """
    축 3: epoch-level schedule multiplier on α (KL weight).

    static     : return alpha_base
    warmup_ce  : α starts at 0.8 on epoch 0, linearly ends at 0.2 on last epoch.
                 (KL-heavy start → CE-heavy finish.)
    cosine     : α = 0.5*(1+cos(π·epoch/(E-1)))*0.6 + 0.2  → smooth 0.8 → 0.2
    """
    if mode == "static" or total_epochs <= 1:
        return alpha_base
    t = epoch / max(1, total_epochs - 1)        # 0..1
    if mode == "warmup_ce":
        return 0.8 * (1.0 - t) + 0.2 * t
    if mode == "cosine":
        return 0.2 + 0.6 * 0.5 * (1.0 + math.cos(math.pi * t))
    return alpha_base


def kd_gate_multiplier(
    logits_teacher: torch.Tensor,   # (K,) option logits
    gt_idx:         int,
    mode:           str,
    tau:            float,
) -> float:
    """
    축 3: returns {0.0, 1.0} scalar to gate KL loss by teacher reliability.
      none       : always 1.0
      correct    : 1.0 iff teacher's argmax == gt_idx
      confidence : 1.0 iff p_t[gt] > tau
    """
    if mode == "none":
        return 1.0
    if mode == "correct":
        return 1.0 if int(logits_teacher.argmax().item()) == gt_idx else 0.0
    if mode == "confidence":
        p = torch.softmax(logits_teacher.float(), dim=-1)
        return 1.0 if float(p[gt_idx].item()) > tau else 0.0
    return 1.0


def margin_ranking_anti_teacher_loss(
    logits_student: torch.Tensor,   # (K,) option logits with grad
    logits_teacher: torch.Tensor,   # (K,) teacher logits (detached)
    gt_idx:         int,
    margin:         float,
) -> torch.Tensor:
    """
    축 3c: when teacher is wrong, push student's GT logit above teacher's argmax
    logit by at least `margin`. Safer than reverse-KL — explicit direction = GT.
    Returns zero (with grad) when teacher is correct.
    """
    t_argmax = int(logits_teacher.argmax().item())
    if t_argmax == gt_idx:
        return logits_student.sum() * 0.0   # keep graph
    gap = logits_student[gt_idx] - logits_student[t_argmax]
    return torch.clamp(margin - gap, min=0.0)


def feature_mse_kd_loss(
    hs_student:      tuple,          # tuple[(1, Ls, d)]
    hs_teacher:      tuple,          # tuple[(1, Lt, d)]
    layer_indices:   list[int],
    region:          str,            # "context_only_pooled" | "per_token"
    n_video_student: int,
    n_video_teacher: int,
) -> torch.Tensor:
    """
    축 4b: MSE between teacher and student hidden states over the post-video
    (text context) region.

    context_only_pooled : mean-pool text region → MSE over a single d-dim vector
                          per selected layer. Robust to M-RoPE mismatch.
    per_token           : element-wise MSE — only valid when student's text
                          positions were aligned to teacher's (M-RoPE override).
    """
    if not layer_indices:
        return hs_student[0].sum() * 0.0

    # hs_* is tuple of (num_layers + 1) tensors (includes embedding output).
    # Anchor text region by skipping the video block from the right.
    # For student: text region = last (Ls - n_video_student) positions? No —
    # more reliable: text region is the tail AFTER the last video token.
    # But we don't have direct access to video positions here; caller must
    # pass layer tensors already sliced if needed. For simplicity, we rely on
    # the fact that video tokens appear in a contiguous block and text is
    # at the tail — take tail pool.

    mses = []
    for idx in layer_indices:
        if idx >= len(hs_student) or idx >= len(hs_teacher):
            continue
        s = hs_student[idx][0]        # (Ls, d)
        t = hs_teacher[idx][0]        # (Lt, d)
        # Take equal-length tail from both — the trailing text region.
        tail_len = min(
            s.shape[0] - n_video_student,
            t.shape[0] - n_video_teacher,
        )
        if tail_len <= 0:
            continue
        s_tail = s[-tail_len:].float()
        t_tail = t[-tail_len:].float()

        if region == "context_only_pooled":
            s_vec = s_tail.mean(dim=0)
            t_vec = t_tail.mean(dim=0)
            mses.append(F.mse_loss(s_vec, t_vec))
        else:   # per_token
            mses.append(F.mse_loss(s_tail, t_tail))

    if not mses:
        return hs_student[0].sum() * 0.0
    return torch.stack(mses).mean()


def get_patch_scores_with_extras(
    traj_encoder,
    item:   dict,
    device: torch.device,
):
    """
    축 5 interface: returns (scores_raw, extras_or_None).

    Tries to unpack encoder.forward returning (scores, extras_dict). If the
    encoder's second return value is a dict, it's exposed to caller for 5a/5b/5c.
    If it's a tensor or None, extras is set to None for graceful degradation.
    """
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb   = traj_encoder.query_encoder([item["question"]], device)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
    out = traj_encoder.encoder(traj_batch, query_emb, visual_feat)
    if isinstance(out, tuple):
        scores_raw = out[0]
        rest = out[1] if len(out) > 1 else None
    else:
        scores_raw = out
        rest = None
    scores_raw = scores_raw.squeeze(0)
    extras = rest if isinstance(rest, dict) else None
    return scores_raw, extras


def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def load_teacher(teacher_ckpt: str, device: torch.device):
    """
    Load a frozen copy of the fine-tuned baseline LoRA Qwen as the teacher model.
    If teacher_ckpt is None or not found, falls back to base pretrained Qwen (frozen).
    Returns the teacher model (PeftModel or base model) with all params frozen.
    """
    from TrajGazeMerge.models.model import load_qwen_lora, load_qwen_frozen
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        print(f"[Teacher] Loading fine-tuned LoRA from: {teacher_ckpt}")
        _, teacher = load_qwen_lora(device)
        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["lora_state"], strict=False)
        print("[Teacher] LoRA weights loaded.")
    else:
        print("[Teacher] No teacher ckpt found, falling back to base pretrained Qwen.")
        _, teacher = load_qwen_frozen(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_traj_encoder(ckpt_path: str, device: torch.device) -> TrajGazeV2:
    model = TrajGazeV2().to(device)
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[TrajEnc] Loaded {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[TrajEnc] WARNING: ckpt not found: {ckpt_path}, using random init")
    return model


def get_patch_scores(
    traj_encoder,
    item:   dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Run TrajGaze encoder forward to get (196,) patch scores.
    Returns tensor with grad attached (for backprop through merge).
    """
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb   = traj_encoder.query_encoder([item["question"]], device)           # (1, D_Q)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)  # (1, 196, D)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)      # (1, 196)
    return scores_raw.squeeze(0)   # (196,) with grad


def evaluate(processor, qwen_model, base_qwen, traj_encoder,
             option_ids, device, merge_ratio, max_items=200,
             teacher_model=None,
             drop_ratio: float = 0.0,
             score_transform: str = "sigmoid",
             match_score_penalty: float = 0.0,
             match_score_hard_gap: float | None = None,
             merge_scope: str = "legacy",
             k_min: int = 1,
             budget_temp: float = 1.0,
             hint_proj=None,
             aux_traj_hidden: int = 256,
             align_text_position_to_teacher: bool = False):
    """Evaluate TrajGazeMerge on egtea test split.

    Returns:
        acc_merge, acc_full, n, kept_ratio
        where kept_ratio = (N - r_merge - r_drop) / N averaged over processed items.
    """
    from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
    test_ds = StreamGazeMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=32)
    test_ds.items = test_ds.items[:max_items]

    qwen_model.eval()
    traj_encoder.eval()
    correct_merge = 0
    correct_full  = 0
    total         = 0
    kept_ratio_sum = 0.0

    with torch.no_grad():
        for item in test_ds:
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)

                r_drop  = int(drop_ratio  * n_video)
                r_total = max(1, int(merge_ratio * n_video))
                r_merge = max(0, r_total - r_drop)
                # Ensure we keep at least one receiver.
                if r_merge + r_drop >= n_video:
                    r_drop  = max(0, n_video - 1 - r_merge)

                # Encoder output + optional extras
                scores, extras = get_patch_scores_with_extras(traj_encoder, item, device)

                hint_embeds_llm = None
                if hint_proj is not None:
                    if (extras is not None
                            and isinstance(extras.get("traj_embeds", None), torch.Tensor)):
                        traj_feat = extras["traj_embeds"].to(device)
                        if traj_feat.dim() > 1 and traj_feat.shape[0] == 1:
                            traj_feat = traj_feat.squeeze(0)
                    else:
                        traj_feat = torch.zeros(
                            aux_traj_hidden, device=device, dtype=torch.float32
                        )
                    hint_embeds_llm = hint_proj(traj_feat.to(torch.bfloat16))

                frame_scores_enc = None
                if (extras is not None
                        and isinstance(extras.get("frame_attend", None), torch.Tensor)):
                    fa = extras["frame_attend"].to(device)
                    if fa.dim() > 1:
                        fa = fa.squeeze(0)
                    frame_scores_enc = fa

                # Full tokens with teacher (or fall back to base model)
                _teacher = teacher_model if teacher_model is not None else qwen_model
                full_inputs  = build_full_inputs(base_qwen, cached,
                                                 hint_embeds=hint_embeds_llm)
                logits_full  = forward_logits(_teacher, full_inputs)
                pred_full = logits_full[option_ids].argmax().item()

                # Merged tokens (with LoRA)
                scores_q   = score_to_qwen_spatial(scores, n_spatial)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)

                if scores_all.shape[0] != n_video:
                    scores_all = scores_all[:n_video] if scores_all.shape[0] > n_video \
                                 else scores_all.repeat_interleave(
                                     (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                                 )[:n_video]

                if merge_scope == "legacy" or n_video != T_merged * n_spatial:
                    merged_video, receiver_idx, drop_idx = gaze_weighted_merge(
                        cached["video_embeds"], scores_all, r_merge,
                        r_drop=r_drop,
                        score_transform=score_transform,
                        match_score_penalty=match_score_penalty,
                        match_score_hard_gap=match_score_hard_gap,
                    )
                else:
                    tokens_tn = cached["video_embeds"].view(T_merged, n_spatial, -1)
                    scores_tn = scores_all.view(T_merged, n_spatial)
                    frame_scores_use = None
                    if merge_scope == "global":
                        if (frame_scores_enc is not None
                                and frame_scores_enc.shape[0] == T_merged):
                            frame_scores_use = frame_scores_enc
                        else:
                            frame_scores_use = scores_tn.mean(dim=-1)
                    merged_video, receiver_idx, drop_idx = gaze_weighted_merge_per_frame(
                        tokens_tn, scores_tn, r_merge,
                        r_drop_total=r_drop,
                        frame_scores=frame_scores_use,
                        k_min=k_min,
                        budget_temp=budget_temp,
                        score_transform=score_transform,
                        match_score_penalty=match_score_penalty,
                        match_score_hard_gap=match_score_hard_gap,
                    )

                merged_inputs = build_merged_inputs(
                    base_qwen, cached, merged_video, receiver_idx,
                    drop_idx=drop_idx,
                    hint_embeds=hint_embeds_llm,
                    align_text_position_to_teacher=align_text_position_to_teacher,
                )
                logits_merge  = forward_logits(qwen_model, merged_inputs)
                pred_merge    = logits_merge[option_ids].argmax().item()

                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                correct_full  += int(pred_full  == gt_idx)
                correct_merge += int(pred_merge == gt_idx)
                total         += 1
                kept_ratio_sum += (n_video - r_merge - r_drop) / max(1, n_video)
            except Exception:
                pass

    qwen_model.train()
    traj_encoder.train()
    acc_merge  = 100.0 * correct_merge / max(1, total)
    acc_full   = 100.0 * correct_full  / max(1, total)
    kept_ratio = kept_ratio_sum / max(1, total)
    return acc_merge, acc_full, total, kept_ratio


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    assert 0.0 <= args.drop_ratio <= args.merge_ratio, (
        f"--drop-ratio ({args.drop_ratio}) must be in [0, --merge-ratio={args.merge_ratio}]"
    )
    # 축 4b safety: per_token feat MSE requires M-RoPE alignment
    if args.kd_feat_layers and args.kd_feat_region == "per_token":
        assert args.align_text_position == "teacher", (
            "--kd-feat-region per_token requires --align-text-position teacher "
            "(otherwise position-encoding mismatch will destabilize training)."
        )
    # Parse feat layers list once
    feat_layers = [int(x) for x in args.kd_feat_layers.split(",") if x.strip()]

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[TrajGazeMerge] output: {args.output_dir}")
        print(f"[TrajGazeMerge] GPUs={world_size}, epochs={args.epochs}, "
              f"lr_lora={args.lr_lora}, lr_enc={args.lr_enc}, "
              f"alpha={args.alpha}, merge_ratio={args.merge_ratio}")
        print(f"[TrajGazeMerge] drop_ratio={args.drop_ratio}, "
              f"score_transform={args.score_transform}, "
              f"match_score_penalty={args.match_score_penalty}, "
              f"match_score_hard_gap={args.match_score_hard_gap}")
        print(f"[TrajGazeMerge] alpha_mode={args.alpha_mode} "
              f"(min={args.alpha_min}, max={args.alpha_max}), "
              f"alpha_schedule={args.alpha_schedule}")
        print(f"[TrajGazeMerge] merge_scope={args.merge_scope}, "
              f"k_min={args.k_min}, budget_temp={args.budget_temp}")
        print(f"[TrajGazeMerge] kd_gate={args.kd_gate} (tau={args.kd_gate_tau}), "
              f"kd_antiteacher={args.kd_antiteacher_weight} "
              f"(margin={args.kd_antiteacher_margin})")
        print(f"[TrajGazeMerge] kd_seq={args.kd_seq}, feat_layers={feat_layers}, "
              f"feat_region={args.kd_feat_region}, "
              f"align_text={args.align_text_position}")
        print(f"[TrajGazeMerge] aux_traj_tokens={args.aux_traj_tokens}, "
              f"aux_forecast={args.aux_traj_forecast_weight}, "
              f"vit_unfreeze={args.vit_unfreeze_last_n}, "
              f"vit_lora_rank={args.vit_lora_rank}")
        print(f"[TrajGazeMerge] teacher_ckpt={args.teacher_ckpt}")

    # ── Load teacher model (frozen, fine-tuned baseline LoRA) ─────────────────
    if is_main:
        print("Loading teacher model ...")
    teacher_model = load_teacher(args.teacher_ckpt, device)
    if is_main:
        print("Teacher loaded.")

    # ── Load TrajGaze encoder ─────────────────────────────────────────────────
    if is_main:
        print("Loading TrajGaze encoder ...")
    traj_encoder = load_traj_encoder(args.stage1_ckpt, device)
    traj_encoder = DDP(traj_encoder, device_ids=[local_rank],
                       find_unused_parameters=True)
    if is_main:
        print("TrajGaze encoder loaded.")

    # ── Load Qwen + LoRA (+ optional ViT adapter) ─────────────────────────────
    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, qwen_model = load_qwen_lora(
        device,
        vit_unfreeze_last_n=args.vit_unfreeze_last_n,
        vit_lora_rank=args.vit_lora_rank,
        vit_lora_last_n=args.vit_lora_last_n,
    )
    base_qwen = qwen_model.get_base_model()

    # ── TrajHintProjection (축 5a) — optional, lives on same device as LLM ────
    d_qwen = base_qwen.get_input_embeddings().weight.shape[-1]
    hint_proj = None
    if args.aux_traj_tokens > 0:
        hint_proj = TrajHintProjection(
            in_dim=args.aux_traj_hidden,
            out_dim=d_qwen,
            n_tokens=args.aux_traj_tokens,
        ).to(device).to(torch.bfloat16)
        # Wrap in DDP so multi-GPU gradient sync applies to hint_proj too.
        hint_proj = DDP(hint_proj, device_ids=[local_rank],
                        find_unused_parameters=True)
        if is_main:
            print(f"[TrajHint] Injecting K={args.aux_traj_tokens} hint tokens "
                  f"(D_traj={args.aux_traj_hidden} → D_qwen={d_qwen})")

    qwen_model = DDP(qwen_model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main:
        print("Qwen loaded.")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        if is_main:
            print(f"[Resume] Loading weights from {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        qwen_model.module.load_state_dict(ckpt["lora_state"], strict=False)
        traj_encoder.module.load_state_dict(ckpt["encoder_state"], strict=False)
        if hint_proj is not None and "hint_proj_state" in ckpt:
            hint_proj.module.load_state_dict(ckpt["hint_proj_state"], strict=False)
            if is_main:
                print("[Resume] hint_proj weights loaded.")
        if is_main:
            print(f"[Resume] Weights loaded. Starting from epoch {args.start_epoch}.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = StreamGazeMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                 rank=rank, shuffle=True)
    loader  = DataLoader(train_ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b[0], num_workers=2)

    # ── Optimizer: separate LR for encoder vs LoRA (+ optional ViT, HintProj) ──
    # Separate ViT params (if any unfrozen / LoRA-attached) into their own group.
    vit_param_ids = set()
    try:
        vit_blocks = qwen_model.module.base_model.model.visual.blocks  # type: ignore[attr-defined]
        for blk in vit_blocks:
            for p in blk.parameters():
                if p.requires_grad:
                    vit_param_ids.add(id(p))
    except AttributeError:
        pass

    lora_params = []
    vit_params  = []
    for _n, p in qwen_model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in vit_param_ids:
            vit_params.append(p)
        else:
            lora_params.append(p)
    enc_params  = list(traj_encoder.parameters())
    hint_params = list(hint_proj.parameters()) if hint_proj is not None else []

    param_groups = [
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ]
    if vit_params:
        param_groups.append({"params": vit_params, "lr": args.lr_vit})
        if is_main:
            print(f"[Optim] ViT group: {len(vit_params)} params @ lr={args.lr_vit}")
    if hint_params:
        # Train hint projection at LoRA learning rate
        param_groups.append({"params": hint_params, "lr": args.lr_lora})
    optimizer = AdamW(param_groups, weight_decay=1e-4)

    log_path  = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc  = 0.0
    n_steps   = 0

    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        qwen_model.train()
        traj_encoder.train()
        optimizer.zero_grad()

        epoch_loss    = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_kl = 0.0
        epoch_alpha   = 0.0
        epoch_kept_ratio = 0.0
        epoch_gate_rate  = 0.0
        epoch_loss_margin = 0.0
        epoch_loss_feat   = 0.0
        epoch_loss_forecast = 0.0
        steps_this_epoch = 0
        t_start = time.time()

        # Epoch-level α schedule multiplier (축 3b)
        alpha_base_ep = compute_alpha_schedule(
            args.alpha, args.alpha_schedule, epoch, args.epochs
        )

        # Whether to capture hidden states this run
        want_hidden = bool(feat_layers)
        align_flag  = (args.align_text_position == "teacher")

        for step, item in enumerate(loader):
            if item is None:
                continue
            try:
                # ── Preprocess (no grad for ViT) ──────────────────────────────
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)

                # Split total removed (merge_ratio) into drop + merge.
                r_drop  = int(args.drop_ratio  * n_video)
                r_total = max(1, int(args.merge_ratio * n_video))
                r_merge = max(0, r_total - r_drop)
                if r_merge + r_drop >= n_video:
                    r_drop = max(0, n_video - 1 - r_merge)

                gt_idx = ["A", "B", "C", "D"].index(item["answer"])
                gt_tensor = torch.tensor([gt_idx], device=device)

                # ── Encoder forward: scores + optional extras (축 5) ──────────
                scores, extras = get_patch_scores_with_extras(
                    traj_encoder.module, item, device
                )   # scores: (196,) with grad

                # Optional trajectory hint tokens (축 5a) — only when enabled AND
                # encoder exposes traj_embeds.
                hint_embeds_llm = None
                if hint_proj is not None:
                    if (extras is not None
                            and isinstance(extras.get("traj_embeds", None), torch.Tensor)):
                        traj_feat = extras["traj_embeds"].to(device)
                        if traj_feat.dim() > 1 and traj_feat.shape[0] == 1:
                            traj_feat = traj_feat.squeeze(0)
                    else:
                        # Fallback: silent zero-vector → hint_proj(0) with zero-init
                        # weights yields (approximately) zero embeddings, i.e., neutral.
                        traj_feat = torch.zeros(
                            args.aux_traj_hidden, device=device, dtype=torch.float32
                        )
                    hint_embeds_llm = hint_proj(traj_feat.to(torch.bfloat16))  # (K, d)

                # Optional frame-attend prob for global budget (축 1)
                frame_scores_enc = None
                if (extras is not None
                        and isinstance(extras.get("frame_attend", None), torch.Tensor)):
                    fa = extras["frame_attend"].to(device)
                    if fa.dim() > 1:
                        fa = fa.squeeze(0)
                    frame_scores_enc = fa

                # Interpolate 14×14 → Qwen spatial grid, expand across T_merged
                scores_q   = score_to_qwen_spatial(scores, n_spatial)   # (n_spatial,)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                if scores_all.shape[0] != n_video:
                    if scores_all.shape[0] > n_video:
                        scores_all = scores_all[:n_video]
                    else:
                        reps = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        scores_all = scores_all.repeat(reps)[:n_video]

                # ── Gaze-weighted merge (축 1 dispatch) ────────────────────────
                video_embeds_detached = cached["video_embeds"].detach()

                if args.merge_scope == "legacy":
                    merged_video, receiver_idx, drop_idx = gaze_weighted_merge(
                        video_embeds_detached, scores_all, r_merge,
                        r_drop=r_drop,
                        score_transform=args.score_transform,
                        match_score_penalty=args.match_score_penalty,
                        match_score_hard_gap=args.match_score_hard_gap,
                    )
                else:
                    # per_frame / global: reshape to (T, n_spatial, d), intra-frame match
                    if n_video != T_merged * n_spatial:
                        # Fallback to legacy if shape doesn't align cleanly
                        merged_video, receiver_idx, drop_idx = gaze_weighted_merge(
                            video_embeds_detached, scores_all, r_merge,
                            r_drop=r_drop,
                            score_transform=args.score_transform,
                            match_score_penalty=args.match_score_penalty,
                            match_score_hard_gap=args.match_score_hard_gap,
                        )
                    else:
                        tokens_tn = video_embeds_detached.view(T_merged, n_spatial, -1)
                        scores_tn = scores_all.view(T_merged, n_spatial)
                        frame_scores_use = None
                        if args.merge_scope == "global":
                            if (frame_scores_enc is not None
                                    and frame_scores_enc.shape[0] == T_merged):
                                frame_scores_use = frame_scores_enc
                            else:
                                # Fallback: mean over spatial scores per frame.
                                # When spatial map is uniform across frames
                                # (current encoder behavior), this yields uniform
                                # budgets → equivalent to per_frame mode.
                                frame_scores_use = scores_tn.mean(dim=-1)
                        merged_video, receiver_idx, drop_idx = gaze_weighted_merge_per_frame(
                            tokens_tn, scores_tn, r_merge,
                            r_drop_total=r_drop,
                            frame_scores=frame_scores_use,
                            k_min=args.k_min,
                            budget_temp=args.budget_temp,
                            score_transform=args.score_transform,
                            match_score_penalty=args.match_score_penalty,
                            match_score_hard_gap=args.match_score_hard_gap,
                        )

                # ── Teacher forward: full tokens, no grad ──
                with torch.no_grad():
                    full_inputs = build_full_inputs(
                        base_qwen, cached,
                        hint_embeds=hint_embeds_llm.detach() if hint_embeds_llm is not None else None,
                    )
                    if want_hidden:
                        teacher_out = forward_logits_ext(
                            teacher_model, full_inputs,
                            output_hidden_states=True,
                            seq_region=args.kd_seq,
                        )
                        logits_teacher_full = teacher_out["logits_last"].detach()
                        logits_teacher = logits_teacher_full[option_ids]
                        teacher_hs = tuple(h.detach() for h in teacher_out["hidden_states"])
                    else:
                        logits_teacher = forward_logits(
                            teacher_model, full_inputs
                        )[option_ids].detach()
                        teacher_hs = None

                # ── Student forward: merged tokens, LoRA active ──
                merged_inputs = build_merged_inputs(
                    base_qwen, cached, merged_video, receiver_idx,
                    drop_idx=drop_idx,
                    hint_embeds=hint_embeds_llm,
                    align_text_position_to_teacher=align_flag,
                )
                if want_hidden:
                    student_out = forward_logits_ext(
                        qwen_model, merged_inputs,
                        output_hidden_states=True,
                        seq_region=args.kd_seq,
                    )
                    logits_student = student_out["logits_last"][option_ids]
                    student_hs = student_out["hidden_states"]
                else:
                    logits_student = forward_logits(
                        qwen_model, merged_inputs
                    )[option_ids]
                    student_hs = None

                # ── Loss ──
                loss_ce = F.cross_entropy(
                    logits_student.unsqueeze(0), gt_tensor
                )
                loss_kl = F.kl_div(
                    F.log_softmax(logits_student, dim=-1),
                    F.softmax(logits_teacher,     dim=-1),
                    reduction="batchmean",
                )

                # 축 3a: KD gate multiplier
                gate = kd_gate_multiplier(
                    logits_teacher, gt_idx, args.kd_gate, args.kd_gate_tau
                )
                loss_kl_effective = loss_kl * gate

                # 축 3b: α (sample dynamic + epoch schedule)
                alpha_t = compute_dynamic_alpha(
                    logits_teacher, gt_idx,
                    mode=args.alpha_mode,
                    alpha_static=alpha_base_ep,
                    alpha_min=args.alpha_min,
                    alpha_max=args.alpha_max,
                )

                loss = alpha_t * loss_kl_effective + (1.0 - alpha_t) * loss_ce

                # 축 3c: margin ranking anti-teacher loss
                loss_margin_val = 0.0
                if args.kd_antiteacher_weight > 0.0:
                    loss_margin = margin_ranking_anti_teacher_loss(
                        logits_student, logits_teacher, gt_idx,
                        args.kd_antiteacher_margin,
                    )
                    loss = loss + args.kd_antiteacher_weight * loss_margin
                    loss_margin_val = float(loss_margin.detach().item())

                # 축 4b: feature-level MSE KD
                loss_feat_val = 0.0
                if want_hidden and feat_layers:
                    loss_feat = feature_mse_kd_loss(
                        student_hs, teacher_hs, feat_layers,
                        args.kd_feat_region,
                        n_video_student=merged_video.shape[0],
                        n_video_teacher=cached["video_embeds"].shape[0],
                    )
                    loss = loss + args.kd_feat_weight * loss_feat
                    loss_feat_val = float(loss_feat.detach().item())

                # 축 5c: trajectory forecasting aux loss (requires encoder support)
                loss_forecast_val = 0.0
                if args.aux_traj_forecast_weight > 0.0 and extras is not None:
                    fc = extras.get("forecast", None)
                    gt_fc = extras.get("forecast_gt", None)
                    if isinstance(fc, torch.Tensor) and isinstance(gt_fc, torch.Tensor):
                        loss_forecast = F.mse_loss(fc, gt_fc.to(fc))
                        loss = loss + args.aux_traj_forecast_weight * loss_forecast
                        loss_forecast_val = float(loss_forecast.detach().item())

                loss = loss / args.grad_accum
                loss.backward()

                epoch_loss      += loss.item() * args.grad_accum
                epoch_loss_ce   += loss_ce.item()
                epoch_loss_kl   += loss_kl.item()
                epoch_alpha     += float(alpha_t)
                epoch_gate_rate += gate
                epoch_loss_margin   += loss_margin_val
                epoch_loss_feat     += loss_feat_val
                epoch_loss_forecast += loss_forecast_val
                epoch_kept_ratio += (n_video - r_merge - r_drop) / max(1, n_video)
                steps_this_epoch += 1
                n_steps          += 1

                if steps_this_epoch % args.grad_accum == 0:
                    all_trainable = lora_params + enc_params + vit_params + hint_params
                    torch.nn.utils.clip_grad_norm_(
                        all_trainable, args.grad_clip
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and steps_this_epoch % args.log_every == 0:
                    avg_l     = epoch_loss       / steps_this_epoch
                    avg_ce    = epoch_loss_ce    / steps_this_epoch
                    avg_kl    = epoch_loss_kl    / steps_this_epoch
                    avg_alpha = epoch_alpha      / steps_this_epoch
                    avg_keep  = epoch_kept_ratio / steps_this_epoch
                    avg_gate  = epoch_gate_rate  / steps_this_epoch
                    avg_mrg   = epoch_loss_margin   / steps_this_epoch
                    avg_feat  = epoch_loss_feat     / steps_this_epoch
                    avg_fc    = epoch_loss_forecast / steps_this_epoch
                    elapsed = time.time() - t_start
                    print(
                        f"Epoch {epoch+1} | step {steps_this_epoch}/{len(loader)} | "
                        f"loss={avg_l:.4f} ce={avg_ce:.4f} kl={avg_kl:.4f} "
                        f"α={avg_alpha:.3f} keep={avg_keep:.3f} gate={avg_gate:.2f} "
                        f"mrg={avg_mrg:.4f} feat={avg_feat:.4f} fc={avg_fc:.4f} | "
                        f"t={elapsed:.0f}s"
                    )
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": steps_this_epoch,
                            "loss": avg_l, "ce": avg_ce, "kl": avg_kl,
                            "alpha": avg_alpha, "kept_ratio": avg_keep,
                            "gate_rate": avg_gate,
                            "loss_margin": avg_mrg,
                            "loss_feat":   avg_feat,
                            "loss_forecast": avg_fc,
                        }) + "\n")

                if is_main and steps_this_epoch % args.eval_every == 0:
                    acc_m, acc_f, n_eval, keep_ratio = evaluate(
                        processor, qwen_model.module, base_qwen, traj_encoder.module,
                        option_ids, device, args.merge_ratio,
                        teacher_model=teacher_model,
                        drop_ratio=args.drop_ratio,
                        score_transform=args.score_transform,
                        match_score_penalty=args.match_score_penalty,
                        match_score_hard_gap=args.match_score_hard_gap,
                        merge_scope=args.merge_scope,
                        k_min=args.k_min,
                        budget_temp=args.budget_temp,
                        hint_proj=hint_proj,
                        aux_traj_hidden=args.aux_traj_hidden,
                        align_text_position_to_teacher=align_flag,
                    )
                    print(f"  → eval egtea: merge={acc_m:.2f}% full={acc_f:.2f}% "
                          f"keep={keep_ratio:.3f} (n={n_eval})")
                    if acc_m > best_acc:
                        best_acc = acc_m
                        ckpt_save = {
                            "epoch": epoch, "step": steps_this_epoch,
                            "lora_state": qwen_model.module.state_dict(),
                            "encoder_state": traj_encoder.module.state_dict(),
                            "acc_merge": acc_m, "acc_full": acc_f,
                        }
                        if hint_proj is not None:
                            ckpt_save["hint_proj_state"] = hint_proj.module.state_dict()
                        torch.save(ckpt_save, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (merge={acc_m:.2f}%)")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Flush remaining gradients
        if steps_this_epoch % args.grad_accum != 0:
            all_trainable = lora_params + enc_params + vit_params + hint_params
            torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_l   = epoch_loss / max(1, steps_this_epoch)
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_l:.4f} | time={elapsed:.0f}s ===")
            ckpt_save_ep = {
                "epoch": epoch,
                "lora_state":    qwen_model.module.state_dict(),
                "encoder_state": traj_encoder.module.state_dict(),
                "loss": avg_l,
            }
            if hint_proj is not None:
                ckpt_save_ep["hint_proj_state"] = hint_proj.module.state_dict()
            torch.save(ckpt_save_ep,
                       os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    # Final evaluation
    if is_main:
        acc_m, acc_f, n_eval, keep_ratio = evaluate(
            processor, qwen_model.module, base_qwen, traj_encoder.module,
            option_ids, device, args.merge_ratio, max_items=500,
            teacher_model=teacher_model,
            drop_ratio=args.drop_ratio,
            score_transform=args.score_transform,
            match_score_penalty=args.match_score_penalty,
            match_score_hard_gap=args.match_score_hard_gap,
            merge_scope=args.merge_scope,
            k_min=args.k_min,
            budget_temp=args.budget_temp,
            hint_proj=hint_proj,
            aux_traj_hidden=args.aux_traj_hidden,
            align_text_position_to_teacher=align_flag,
        )
        print(f"\n[Final] egtea: merge={acc_m:.2f}%  full={acc_f:.2f}%  "
              f"keep={keep_ratio:.3f}  (n={n_eval})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
