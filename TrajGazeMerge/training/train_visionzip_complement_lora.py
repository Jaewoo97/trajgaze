"""VisionZip + complementary trajectory selection — LoRA training.

Forked from train_visionzip_traj_lora_3way.py. The key difference from VZ+traj:
VZ+traj MULTIPLIES VisionZip's attention by a trajectory weight, so it can only
re-rank *within* the attention-supported set — a token the ViT scored ~0 can
never be resurrected. Here we instead take a COMPLEMENTARY UNION:

    10% budget  =  7% VisionZip content selection  ∪  3% trajectory pool

The trajectory pool is the top-3% tokens (by a trajectory salience score)
*excluded* from VisionZip's content set — i.e. exactly the gaze/hand-relevant
tokens that content attention missed or merged away. Two ways to score that 3%:

  --traj-pool-mode learned       frozen TrajGazeV2Temporal (TAS) encoder →
                                 get_patch_scores → score_to_qwen_spatiotemporal
  --traj-pool-mode anticipatory  parameter-free anticipatory tubes
                                 (gaze/hand velocity-extrapolated Gaussians)

Both selectors are FIXED (no params trained); only LoRA adapts to the resulting
tokens — identical training protocol to VZ / VZ+traj (LoRA-only, eff-batch 8,
3 epochs, epoch1→epoch2 early-stop, gaze-overlay, egtea 1011 val).

Usage (2-GPU DDP):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29654 \\
      -m TrajGazeMerge.training.train_visionzip_complement_lora \\
      --traj-pool-mode learned \\
      --stage1-ckpt .../stage1_tas_3way_overlay/best.pth \\
      --output-dir .../visionzip_complement_learned_overlay \\
      --epochs 3 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --no-mid-eval
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re as _re
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

_LET_PREFIX = _re.compile(r"^\s*[A-Ea-e]\s*[.):]\s*")


def _augment_options(item):
    """Return a shallow copy of `item` with its MC options randomly permuted and
    re-lettered, with `answer` remapped to the option's new position. Strips the
    original 'A. ' prefix and re-prefixes by new position. None on malformed item."""
    opts = item.get("options") or []
    n = len(opts)
    if n < 2:
        return item
    letters = [chr(65 + i) for i in range(n)]
    ans = item.get("answer")
    if ans not in letters:
        return None
    gt_old = letters.index(ans)
    stripped = [_LET_PREFIX.sub("", o).strip() for o in opts]
    perm = list(range(n))
    random.shuffle(perm)                       # perm[j] = old index now at position j
    new_opts = [f"{chr(65 + j)}. {stripped[perm[j]]}" for j in range(n)]
    gt_new = perm.index(gt_old)
    out = dict(item)
    out["options"] = new_opts
    out["answer"] = chr(65 + gt_new)
    return out

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.traj_anticipate import anticipatory_token_scores
from TrajGazeMerge.models.traj_weights import _solve_spatial_dims
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal,
)

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_overlay")
    p.add_argument("--traj-pool-mode", choices=["learned", "anticipatory"], default="learned")
    p.add_argument("--mask-modality",
                   choices=["none", "gaze_only", "hand_only", "no_interact",
                            "drop_coord", "drop_prox", "drop_vel"],
                   default="none",
                   help="Complement modality ablation: zero the OTHER modality's trajectory "
                        "channels before scoring. 'gaze_only' drops hands, 'hand_only' drops "
                        "gaze (cross-modal features zeroed either way). 'no_interact' keeps "
                        "gaze+hand but zeroes ALL interaction-token features (isolates the "
                        "interaction token's marginal contribution). 'drop_coord' / 'drop_prox' "
                        "/ 'drop_vel' zero one sub-group of the interaction feature "
                        "(coordination={convergence,lead_lag}, proximity={d_left,d_right}, "
                        "velocity={v_rel_left,v_rel_right}). 'none' = full (default).")
    p.add_argument("--eval-ckpt", default=None,
                   help="If set, skip training: load this LoRA best.pth, run one evaluate() "
                        "with per-source logging, print, and exit. Used to re-measure a "
                        "trained checkpoint under the current per_src eval pipeline.")
    p.add_argument("--content-ratio", type=float, default=0.07,
                   help="VisionZip content budget (split half dominant / half contextual).")
    p.add_argument("--traj-ratio",    type=float, default=0.03,
                   help="Complementary trajectory-pool budget (top-k of non-content tokens).")
    p.add_argument("--query-ratio",   type=float, default=0.0,
                   help="query_gaze mode: question-relevant complement budget. "
                        "e.g. content 0.07 + query 0.01 + traj 0.02 = 10%% (M1's budget, "
                        "1%% reallocated gaze→query).")
    p.add_argument("--query-mode",    choices=["cosine", "random", "shuffle"], default="cosine",
                   help="query_gaze pool scoring: 'cosine' = this item's question-stem "
                        "relevance; 'random' = random discarded tokens (CONTROL: gain from "
                        "query relevance vs just +1%% tokens); 'shuffle' = a DIFFERENT item's "
                        "question (CONTROL: does the CORRECT question matter, or any question).")
    p.add_argument("--complement-mode", choices=["topk", "coverage", "fusion", "query_gaze"],
                   default="topk",
                   help="How to pick the complement: 'topk' = raw global top-k (M1/M2); "
                        "'coverage' = temporally-distributed (per-frame budget, floor 1 per "
                        "gaze-active frame) + spatially-diverse (intra-frame NMS) — falsified; "
                        "'fusion' = no separate pool; dominant tokens by norm(attn)+λ·norm(traj); "
                        "'query_gaze' = 3-pool disjoint content ∪ query ∪ gaze.")
    p.add_argument("--nms-radius", type=int, default=1,
                   help="Intra-frame spatial NMS radius (grid cells) for coverage mode; "
                        "0 = no suppression (dedup only).")
    p.add_argument("--fusion-lambda", type=float, default=1.0,
                   help="Fusion mode: weight on normalized trajectory salience in "
                        "fused = norm(attn) + lambda*norm(traj).")
    p.add_argument("--fusion-norm", choices=["minmax", "rank"], default="minmax",
                   help="Per-video normalization of attn/traj before fusion.")
    p.add_argument("--stage1-ckpt",   default=STAGE1_DEFAULT,
                   help="Frozen TAS Stage-1 encoder (learned mode only).")
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--horizon",       type=float, default=2.0,
                   help="Anticipatory extrapolation horizon in frames (anticipatory mode).")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    # Trajectory-geometry hyperparameters (anticipatory mode)
    p.add_argument("--sigma-g",    type=float, default=2.0)
    p.add_argument("--sigma-h",    type=float, default=3.0)
    p.add_argument("--alpha-hand", type=float, default=0.7)
    p.add_argument("--sigma-v",    type=float, default=0.05)
    p.add_argument("--sigma-gh",   type=float, default=0.10)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="2-way: StreamGaze + EgoGazeVQA only (exclude HD-EPIC).")
    p.add_argument("--option-aug", action="store_true",
                   help="TRAIN-time option-order augmentation: randomly permute the MC "
                        "options each step (answer remapped) to make the model "
                        "position-invariant. Eval stays on original order. Targets the "
                        "measured option-position bias (acc@B 55.6 vs @D 68.3).")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if epoch-2 val <= epoch-1 val.")
    p.add_argument("--no-mid-eval", action="store_true",
                   help="Disable the mid-epoch eval/checkpoint.")
    p.set_defaults(include_hdepic=True)
    p.add_argument("--source", choices=["sg", "eg", "both"], default="both",
                   help="Train/eval on a single benchmark only (sg=StreamGaze, eg=EgoGazeVQA). "
                        "Filters the combined dataset to that source; per-source acc then equals "
                        "overall acc, driving best.pth + early-stop. 'both' = joint protocol.")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _score_to_qwen_robust(scores, n_spatial, T_merged, grid_thw):
    """Map encoder (T_traj, 196=14x14) patch scores onto VisionZip's (N,) video-
    token layout. Unlike the TAS bridge, this does NOT assume a square per-frame
    layout — VisionZip emits non-square spatial grids (e.g. 216 tokens/frame),
    so we factor n_spatial via the same _solve_spatial_dims used by the geometric
    path, keeping the two modes spatially consistent."""
    T_traj = scores.shape[0]
    H_grid = int(grid_thw[0, 1].item())
    W_grid = int(grid_thw[0, 2].item())
    s_h, s_w = _solve_spatial_dims(n_spatial, H_grid, W_grid)
    s2d = scores.float().reshape(T_traj, 1, 14, 14)
    out = F.interpolate(s2d, size=(s_h, s_w), mode="bilinear", align_corners=False)
    scores_spatial = out.reshape(T_traj, n_spatial)                      # n_spatial = s_h*s_w
    if T_traj != T_merged:
        scores_spatial = F.interpolate(
            scores_spatial.T.unsqueeze(0).float(),
            size=T_merged, mode="linear", align_corners=False,
        ).squeeze(0).T
    return scores_spatial.reshape(-1)                                    # (T_merged*n_spatial,)


def _mask_traj(traj, mode):
    """Zero one modality's trajectory channels for the complement modality ablation.
    The learned TAS encoder gates each anchor on its *_mask (falling back to a trained
    missing-modality token), so zeroing mask+coords makes that modality in-distribution
    absent. Cross-modal gaze<->hand features are dropped either way (they leak the other
    modality)."""
    if mode in (None, "none"):
        return traj
    cross = ["d_left", "d_right", "v_rel_left", "v_rel_right", "convergence", "lead_lag"]
    if mode == "gaze_only":
        drop = ["left_pos", "left_vel", "left_mask",
                "right_pos", "right_vel", "right_mask"] + cross
    elif mode == "hand_only":
        drop = ["gaze_pos", "gaze_speed", "gaze_mask"] + cross
    elif mode == "no_interact":
        drop = cross
    elif mode == "drop_coord":
        drop = ["convergence", "lead_lag"]
    elif mode == "drop_prox":
        drop = ["d_left", "d_right"]
    elif mode == "drop_vel":
        drop = ["v_rel_left", "v_rel_right"]
    else:
        raise ValueError(f"unknown mask_modality: {mode}")
    out = dict(traj)
    for k in drop:
        if k in out:
            out[k] = torch.zeros_like(out[k])
    return out


def _traj_scores(cached, item, device, mode, encoder, hp):
    """Per-video-token trajectory salience (N,) — learned or anticipatory."""
    video_embeds = cached["video_embeds"]
    n_video = video_embeds.shape[0]
    mm = hp.get("mask_modality", "none")
    if mm and mm != "none":
        item = {**item, "traj": _mask_traj(item["traj"], mm)}
    if mode == "learned":
        T_merged  = int(cached["grid_thw"][0, 0].item())
        n_spatial = n_video // max(1, T_merged)
        scores = get_patch_scores_temporal(encoder, item, device)        # (T_traj, 196)
        scores_all = _score_to_qwen_robust(scores, n_spatial, T_merged, cached["grid_thw"])
        if scores_all.shape[0] != n_video:
            scores_all = (scores_all[:n_video] if scores_all.shape[0] > n_video
                          else scores_all.repeat(
                              (n_video + scores_all.shape[0] - 1) // scores_all.shape[0])[:n_video])
        return scores_all.to(device)
    return anticipatory_token_scores(
        item["traj"], n_video, cached["grid_thw"], device,
        horizon=hp["horizon"], sigma_g=hp["sigma_g"], sigma_h=hp["sigma_h"],
        alpha_hand=hp["alpha_hand"], sigma_v=hp["sigma_v"], sigma_gh=hp["sigma_gh"],
    )


def _norm_scores(x: torch.Tensor, mode: str = "minmax") -> torch.Tensor:
    """Per-video normalize a score vector to a comparable [0,1] range so attention
    and trajectory salience can be summed. 'minmax' keeps relative magnitudes;
    'rank' is scale-free / outlier-robust (percentile of each token)."""
    x = x.float()
    if mode == "rank":
        order = torch.argsort(torch.argsort(x))
        return order.float() / max(1, x.numel() - 1)
    lo = x.min(); hi = x.max()
    return (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)


def _query_scores(cached, video_embeds, q_override=None):
    """Per-video-token cosine similarity to a question embedding → (N,).

    Parameter-free, leakage-safe (question only, no options). Both operands live
    in Qwen's LLM input-embedding space, so cosine ranks each discarded video
    token by how relevant it is to the question. Uses cached["query_emb"] (the
    current item's question) unless q_override is given (shuffle control: a
    different item's question). Returns None if no embedding is available."""
    q = q_override if q_override is not None else cached.get("query_emb")
    if q is None:
        return None
    v  = F.normalize(video_embeds.float(), dim=-1)                  # (N, d)
    qn = F.normalize(q.float().to(video_embeds.device), dim=-1)     # (d,)
    return v @ qn                                                   # (N,)


def _coverage_complement(traj_scores, avail_mask, k_traj, T, n_spatial,
                         s_h, s_w, nms_radius, device):
    """Pick the k_traj complement tokens with TEMPORAL and SPATIAL coverage.

    Raw global top-k collapses onto the contiguous patches around the single
    gaze peak in the few hottest frames (the encoder score is smooth in space
    and time). Instead we (1) give every gaze-active frame a floor of >=1 token
    and split the rest of the budget by per-frame trajectory mass, then (2) pick
    each frame's tokens greedily with spatial non-max suppression so they land
    on distinct regions, not adjacent patches. Content tokens are pre-masked.

    Returns a LongTensor of global token indices (<= k_traj)."""
    neg = torch.finfo(traj_scores.dtype).min
    neg_half = neg / 2.0
    scores2d = traj_scores.view(T, n_spatial).clone()
    avail2d = avail_mask.view(T, n_spatial)
    scores2d[~avail2d] = neg

    frame_w = scores2d.clamp(min=0.0).sum(dim=1)      # (T,) positive gaze mass
    active = frame_w > 0
    n_active = int(active.sum().item())
    if n_active == 0:                                  # no positive gaze anywhere
        flat = scores2d.view(-1)
        k = min(k_traj, int(avail2d.sum().item()))
        return torch.topk(flat, max(1, k)).indices if k > 0 else \
            torch.empty(0, dtype=torch.long, device=device)

    # --- per-frame budget: floor 1 for active frames, remainder by mass ---
    quota = torch.zeros(T, dtype=torch.long, device=device)
    base = min(n_active, k_traj)
    if base < n_active:                                # tiny budget: top frames only
        quota[torch.topk(frame_w, base).indices] = 1
    else:
        quota[active] = 1
    remaining = k_traj - int(quota.sum().item())
    if remaining > 0:
        w = frame_w.clone(); w[~active] = 0.0
        if w.sum() > 0:
            frac = w / w.sum() * remaining
            add = torch.floor(frac).long()
            quota += add
            deficit = remaining - int(add.sum().item())
            if deficit > 0:
                rem = frac - add.float(); rem[~active] = -1.0
                quota[torch.topk(rem, deficit).indices] += 1

    # --- intra-frame greedy NMS selection ---
    picked: list[int] = []
    for t in range(T):
        q = int(quota[t].item())
        if q <= 0:
            continue
        fr = scores2d[t]
        order = torch.argsort(fr, descending=True).tolist()
        fr_list = fr.tolist()
        blocked = [[False] * s_w for _ in range(s_h)]
        cnt = 0
        for pos in order:
            if fr_list[pos] <= neg_half:               # reached content/unavailable
                break
            r, c = divmod(pos, s_w)
            if blocked[r][c]:
                continue
            picked.append(t * n_spatial + pos)
            cnt += 1
            if cnt >= q:
                break
            for rr in range(max(0, r - nms_radius), min(s_h, r + nms_radius + 1)):
                for cc in range(max(0, c - nms_radius), min(s_w, c + nms_radius + 1)):
                    blocked[rr][cc] = True

    picked_t = torch.tensor(picked, dtype=torch.long, device=device)
    # --- top up any NMS/quota deficit from the remaining available tokens ---
    if picked_t.numel() < k_traj:
        flat = scores2d.view(-1).clone()
        if picked_t.numel() > 0:
            flat[picked_t] = neg
        n_left = int((flat > neg_half).sum().item())
        need = min(k_traj - picked_t.numel(), n_left)
        if need > 0:
            extra = torch.topk(flat, need).indices
            picked_t = torch.cat([picked_t, extra])
    return picked_t


def select_complementary(cached, item, device, mode, encoder, hp,
                         content_ratio, traj_ratio,
                         complement_mode="topk", nms_radius=1,
                         fusion_lambda=1.0, fusion_norm="minmax",
                         query_ratio=0.0, query_mode="cosine"):
    """Union VisionZip content selection (reduced budget) with a complementary
    top-k trajectory pool drawn from tokens VisionZip did NOT keep.

    complement_mode='fusion' instead drops the disjoint two-pool union: it scores
    every token by norm(attn) + lambda*norm(traj) and lets that single fused score
    pick the dominant (raw) tokens, while VisionZip's contextual merge is unchanged.
    Same 10% composition as 'topk' (6.5% raw + 3.5% merged at 7/3) but the raw
    budget is allocated softly between attention and gaze instead of hard-split."""
    video_embeds = cached["video_embeds"]
    attn_scores  = cached["attn_scores"]
    attn_key     = cached["attn_key"]
    N = video_embeds.shape[0]

    if complement_mode == "fusion":
        traj_scores = _traj_scores(cached, item, device, mode, encoder, hp).to(attn_scores.device)
        fused = _norm_scores(attn_scores, fusion_norm) \
                + fusion_lambda * _norm_scores(traj_scores, fusion_norm)
        dom_ratio = content_ratio / 2.0 + traj_ratio     # fold complement into dominant
        ctx_ratio = content_ratio / 2.0                  # contextual stays VZ-native
        return visionzip_select_tokens(
            video_embeds, attn_scores, attn_key,
            dominant_ratio=dom_ratio, contextual_ratio=ctx_ratio,
            dominant_score=fused,
        )

    half = content_ratio / 2.0
    content_embeds, content_idx = visionzip_select_tokens(
        video_embeds, attn_scores, attn_key,
        dominant_ratio=half, contextual_ratio=half,
    )

    if complement_mode == "query_gaze":
        # 3-pool disjoint union under the SAME fixed budget as M1:
        #   content (C) ∪ query (Q) ∪ gaze (G).
        # Q recovers question-relevant tokens VisionZip discarded; G recovers
        # gaze/hand tokens (M1's role). Q is picked first, then G from what's left,
        # so the pools never overlap. content_embeds are MERGED centroids (keep as-is);
        # the Q/G pools are raw tokens (video_embeds[idx]) — mirrors the 'topk' path.
        avail_mask = torch.ones(N, dtype=torch.bool, device=video_embeds.device)
        avail_mask[content_idx] = False
        extra_idx = []

        # --- Query pool Q (question-stem relevance) ---
        if query_ratio > 0:
            avail = avail_mask.nonzero(as_tuple=True)[0]
            k_q = min(max(1, int(query_ratio * N)), avail.numel())
            if k_q > 0:
                if query_mode == "random":               # CONTROL: arbitrary discarded tokens
                    qscore = None
                elif query_mode == "shuffle":            # CONTROL: a DIFFERENT item's question
                    donor = hp.get("_query_donor") if isinstance(hp, dict) else None
                    qscore = _query_scores(cached, video_embeds, q_override=donor)
                else:                                    # 'cosine': this item's question
                    qscore = _query_scores(cached, video_embeds)
                if qscore is None:                       # random / shuffle cold-start / missing emb
                    perm = torch.randperm(avail.numel(), device=avail.device)
                    q_sel = avail[perm[:k_q]]
                else:
                    q_sel = avail[torch.topk(qscore[avail], k_q).indices]
                avail_mask[q_sel] = False
                extra_idx.append(q_sel)
            # Advance the shuffle donor buffer AFTER reading it (so 'shuffle' always
            # scores against the previous item's question, never the current one).
            if isinstance(hp, dict) and cached.get("query_emb") is not None:
                hp["_query_donor"] = cached["query_emb"]

        # --- Gaze pool G (TAS salience), disjoint from C and Q ---
        if traj_ratio > 0:
            traj_scores = _traj_scores(cached, item, device, mode, encoder, hp)
            avail = avail_mask.nonzero(as_tuple=True)[0]
            k_g = min(max(1, int(traj_ratio * N)), avail.numel())
            if k_g > 0:
                g_sel = avail[torch.topk(traj_scores[avail], k_g).indices]
                extra_idx.append(g_sel)

        if extra_idx:
            all_embeds = torch.cat([content_embeds] + [video_embeds[i] for i in extra_idx], dim=0)
            all_idx    = torch.cat([content_idx] + extra_idx)
        else:
            all_embeds, all_idx = content_embeds, content_idx
        order = all_idx.argsort()
        return all_embeds[order], all_idx[order]

    traj_scores = _traj_scores(cached, item, device, mode, encoder, hp)

    avail_mask = torch.ones(N, dtype=torch.bool, device=video_embeds.device)
    avail_mask[content_idx] = False
    avail = avail_mask.nonzero(as_tuple=True)[0]
    k_traj = min(max(1, int(traj_ratio * N)), avail.numel())

    if k_traj > 0 and avail.numel() > 0:
        traj_idx = None
        if complement_mode == "coverage":
            T = int(cached["grid_thw"][0, 0].item())
            n_spatial = N // max(1, T)
            if T * n_spatial == N:
                H_grid = int(cached["grid_thw"][0, 1].item())
                W_grid = int(cached["grid_thw"][0, 2].item())
                s_h, s_w = _solve_spatial_dims(n_spatial, H_grid, W_grid)
                if s_h * s_w == n_spatial:
                    traj_idx = _coverage_complement(
                        traj_scores, avail_mask, k_traj, T, n_spatial,
                        s_h, s_w, nms_radius, video_embeds.device)
        if traj_idx is None:                            # 'topk' (default) or layout fallback
            top = torch.topk(traj_scores[avail], k_traj).indices
            traj_idx = avail[top]
        all_embeds = torch.cat([content_embeds, video_embeds[traj_idx]], dim=0)
        all_idx    = torch.cat([content_idx, traj_idx])
    else:
        all_embeds, all_idx = content_embeds, content_idx

    order = all_idx.argsort()
    return all_embeds[order], all_idx[order]


def evaluate(processor, model, base_qwen, option_ids, device, mode, encoder, hp,
             content_ratio, traj_ratio, include_hdepic=True,
             complement_mode="topk", nms_radius=1,
             fusion_lambda=1.0, fusion_norm="minmax",
             query_ratio=0.0, query_mode="cosine", source="both"):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic,
    )
    if source in ("sg", "eg"):
        test_ds.items = [it for it in test_ds.items if it[0] == source]
    model.eval()
    correct = 0; total = 0
    by_task: dict[str, list] = {}
    by_src: dict[str, list] = {}
    with torch.no_grad():
        for idx in range(len(test_ds)):
            src = test_ds.items[idx][0]
            item = test_ds[idx]
            if item is None: continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None: continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                sel_embeds, recv_idx = select_complementary(
                    cached, item, device, mode, encoder, hp, content_ratio, traj_ratio,
                    complement_mode=complement_mode, nms_radius=nms_radius,
                    fusion_lambda=fusion_lambda, fusion_norm=fusion_norm,
                    query_ratio=query_ratio, query_mode=query_mode)
                inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                logits = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok; total += 1
                by_task.setdefault(item["task"], []).append(ok)
                by_src.setdefault(src, []).append(ok)
            except Exception:
                pass
    model.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    per_src = {s: [100.0 * sum(v) / max(1, len(v)), len(v)] for s, v in sorted(by_src.items())}
    return 100.0 * correct / max(1, total), total, per_task, per_src


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    hp = dict(
        horizon=args.horizon, sigma_g=args.sigma_g, sigma_h=args.sigma_h,
        alpha_hand=args.alpha_hand, sigma_v=args.sigma_v, sigma_gh=args.sigma_gh,
        mask_modality=args.mask_modality,
    )

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[VZ-complement] output: {args.output_dir}")
        print(f"[VZ-complement] mode={args.traj_pool_mode}, GPUs={world_size}, "
              f"content={args.content_ratio*100:.1f}% ∪ traj={args.traj_ratio*100:.1f}% = "
              f"{(args.content_ratio+args.traj_ratio)*100:.0f}%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}", flush=True)
        if args.traj_pool_mode == "anticipatory":
            print(f"[VZ-complement] anticip hp: horizon={args.horizon} σ_g={args.sigma_g} "
                  f"σ_h={args.sigma_h} α_hand={args.alpha_hand}", flush=True)
        else:
            print(f"[VZ-complement] learned encoder: {args.stage1_ckpt}", flush=True)
        if args.mask_modality != "none":
            print(f"[VZ-complement] *** MODALITY ABLATION: mask_modality={args.mask_modality} ***", flush=True)
        print(f"[VZ-complement] complement_mode={args.complement_mode}"
              + (f" (fusion λ={args.fusion_lambda}, norm={args.fusion_norm})"
                 if args.complement_mode == "fusion" else "")
              + (f" (3-pool: content={args.content_ratio*100:.1f}% ∪ "
                 f"query={args.query_ratio*100:.1f}%[{args.query_mode}] ∪ "
                 f"gaze={args.traj_ratio*100:.1f}%)"
                 if args.complement_mode == "query_gaze" else ""), flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    encoder = None
    if args.traj_pool_mode == "learned":
        encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    if is_main: print("Model loaded.", flush=True)

    if args.eval_ckpt:
        if is_main: print(f"[eval-only] loading LoRA state from {args.eval_ckpt}", flush=True)
        sd = torch.load(args.eval_ckpt, map_location="cpu")["lora_state"]
        missing, unexpected = model.module.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[eval-only] loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
            acc, n_eval, per_task, per_src = evaluate(
                processor, model.module, base_qwen, option_ids, device,
                args.traj_pool_mode, encoder, hp, args.content_ratio, args.traj_ratio,
                include_hdepic=args.include_hdepic,
                complement_mode=args.complement_mode, nms_radius=args.nms_radius,
                fusion_lambda=args.fusion_lambda, fusion_norm=args.fusion_norm,
                query_ratio=args.query_ratio, query_mode=args.query_mode,
                source=args.source,
            )
            print(f"[eval-only] Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for s, (s_acc, s_n) in per_src.items():
                print(f"[eval-only] [src] {s}: {s_acc:.2f}%  (n={s_n})", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic,
    )
    if args.source in ("sg", "eg"):
        n_before = len(train_ds.items)
        train_ds.items = [it for it in train_ds.items if it[0] == args.source]
        if is_main:
            print(f"[source={args.source}] train filtered {n_before} → {len(train_ds.items)} items", flush=True)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0
    epoch_accs: list[float] = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None: continue
            try:
                if args.option_aug:
                    item = _augment_options(item)
                    if item is None: continue
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device,
                    )
                if cached is None: continue
                n_video = cached["video_embeds"].shape[0]

                with torch.no_grad():
                    sel_embeds, recv_idx = select_complementary(
                        cached, item, device, args.traj_pool_mode, encoder, hp,
                        args.content_ratio, args.traj_ratio,
                        complement_mode=args.complement_mode, nms_radius=args.nms_radius,
                        fusion_lambda=args.fusion_lambda, fusion_norm=args.fusion_norm,
                        query_ratio=args.query_ratio, query_mode=args.query_mode)
                    inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue
                logits = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                loss = F.cross_entropy(option_logits.unsqueeze(0),
                                       torch.tensor([gt_idx], device=device))
                loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    pct_kept = 100.0 * recv_idx.shape[0] / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s",
                          flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")
            except Exception:
                if is_main: traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed = time.time() - t_start
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} "
                  f"| time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} on full {label} val set ...", flush=True)
            acc, n_eval, per_task, per_src = evaluate(
                processor, model.module, base_qwen, option_ids, device,
                args.traj_pool_mode, encoder, hp, args.content_ratio, args.traj_ratio,
                include_hdepic=args.include_hdepic,
                complement_mode=args.complement_mode, nms_radius=args.nms_radius,
                fusion_lambda=args.fusion_lambda, fusion_norm=args.fusion_norm,
                query_ratio=args.query_ratio, query_mode=args.query_mode,
                source=args.source,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            for s, (s_acc, s_n) in per_src.items():
                print(f"    [src] {s}: {s_acc:.2f}%  (n={s_n})", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task, "per_src": per_src,
                }) + "\n")
            epoch_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
            if args.early_stop and (epoch + 1) == 2 and len(epoch_accs) >= 2 \
                    and epoch_accs[1] <= epoch_accs[0]:
                stop[0] = 1.0
                print(f"  Early stop: epoch2 {epoch_accs[1]:.2f}% <= epoch1 "
                      f"{epoch_accs[0]:.2f}% → skipping epoch 3.", flush=True)
        dist.barrier()
        dist.broadcast(stop, src=0)
        if stop.item() > 0:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
