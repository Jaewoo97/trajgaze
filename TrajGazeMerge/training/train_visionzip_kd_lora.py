"""Gaze-free student via privileged-information selection distillation.

\sys (M1) keeps 10% of the visual tokens = 7% VisionZip content ∪ 3% trajectory
complement, where the complement is the top-3% of the tokens VisionZip discarded,
ranked by a gaze/hand salience field (frozen TAS encoder). That field needs the
gaze/hand streams at inference — i.e. an eye-tracker at test time.

Here the gaze/hand streams are treated as PRIVILEGED information: available while
training, absent at test time. A small RGB-only head (TrajSaliencePredictor) is
distilled to reproduce the teacher's *selection* — which discarded tokens the
gaze/hand field would pick — from content-side features alone (token embeddings,
ViT importance, frame position). At inference the student uses its own predicted
salience to choose the 3% complement, so NO gaze/hand is read.

Two decoupled objectives (top-k selection is non-differentiable, so they do not
share gradients):
  * predictor  ← selection distillation: BCE(student salience, teacher top-k membership)
                 over the discarded (available) tokens.
  * LoRA       ← task cross-entropy on the student-selected 10% tokens.

The LoRA is warm-started from the M1 checkpoint so the readout begins at
teacher quality and only has to absorb the small student/teacher selection gap.

Usage (4-GPU DDP, eff-batch 8 = 4 GPU × grad-accum 2):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29661 \\
      -m TrajGazeMerge.training.train_visionzip_kd_lora \\
      --warmstart-ckpt .../visionzip_complement_learned_overlay/best.pth \\
      --stage1-ckpt   .../stage1_tas_3way_overlay/best.pth \\
      --output-dir    .../visionzip_kd_selection_overlay \\
      --epochs 3 --lr 1e-4 --pred-lr 1e-3 --grad-accum 2 --no-hdepic --early-stop
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"))

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.traj_salience_predictor import TrajSaliencePredictor
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder,
)
# Teacher salience (gaze/hand → per-token field). Reused verbatim so the student
# distills exactly the signal M1 selects on.
from TrajGazeMerge.training.train_visionzip_complement_lora import _traj_scores

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"
M1_DEFAULT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay/best.pth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_kd_selection_overlay")
    p.add_argument("--warmstart-ckpt", default=M1_DEFAULT,
                   help="M1 LoRA best.pth to warm-start the student readout. '' to skip.")
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT,
                   help="Frozen TAS Stage-1 encoder = the privileged gaze/hand teacher.")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio",    type=float, default=0.03)
    p.add_argument("--lambda-sel",    type=float, default=1.0,
                   help="Weight on the selection-distillation (BCE) loss.")
    p.add_argument("--pred-hidden",   type=int, default=512)
    p.add_argument("--pred-lr",       type=float, default=1e-3,
                   help="LR for the (fresh) salience predictor; LoRA uses --lr.")
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=2)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="2-way: StreamGaze + EgoGazeVQA only (exclude HD-EPIC).")
    p.add_argument("--source", choices=["sg", "eg", "both"], default="both",
                   help="Train/eval on a single benchmark only (sg=StreamGaze, eg=EgoGazeVQA). "
                        "Filters the combined dataset to that source; per-source acc then equals "
                        "overall acc, driving best.pth + early-stop. 'both' = joint protocol.")
    p.add_argument("--balance-sources", action="store_true",
                   help="Joint training only: oversample the minority source so SG and EG "
                        "contribute equally many steps per epoch (train counts are "
                        "SG 5799 : EG 1265 = 4.6:1, which lets SG overfit while EG-temporal "
                        "is still underfit). One model, no routing — eval is unchanged.")
    p.add_argument("--balance-seed", type=int, default=0,
                   help="Seed for choosing which minority items get the remainder repeat.")
    p.add_argument("--freeze-lora", action="store_true",
                   help="Train the salience predictor ONLY, holding the warm-started LoRA "
                        "fixed at teacher quality. Drops the task-CE term and the VLM "
                        "forward/backward from the train step (~3-5x faster per epoch) and "
                        "isolates the selection gap from readout over-training.")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if epoch-2 val <= epoch-1 val.")
    p.add_argument("--eval-ckpt", default=None,
                   help="Skip training: load this student.pth (lora_state + pred_state), "
                        "run one gaze-free evaluate() with per-source logging, exit.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the newest epoch_*.pth in --output-dir, restoring BOTH "
                        "lora_state and pred_state and continuing at the next epoch. Safe to "
                        "pass on a fresh run (no checkpoint = normal start), so a supervisor "
                        "can relaunch the same command after a crash.")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def _latest_epoch_ckpt(output_dir):
    """Newest epoch_NN.pth in output_dir → (path, epoch_number), or (None, 0)."""
    import re
    best = (None, 0)
    if not os.path.isdir(output_dir):
        return best
    for fn in os.listdir(output_dir):
        m = re.fullmatch(r"epoch_(\d+)\.pth", fn)
        if m and int(m.group(1)) > best[1]:
            best = (os.path.join(output_dir, fn), int(m.group(1)))
    return best


def _prior_epoch_accs(log_path):
    """Eval accuracies already recorded in the rank-0 JSONL, ordered by epoch.

    Restores best.pth tracking and the early-stop comparison across a restart;
    without it a resumed run would overwrite a better best.pth with a worse epoch.
    """
    accs: dict[int, float] = {}
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "eval_acc" in rec:
                accs[rec["epoch"]] = rec["eval_acc"]
    return [accs[e] for e in sorted(accs)]


def balance_items(items, seed):
    """Equal SG/EG share of the SAME epoch size — only the source MIX changes.

    Train counts are SG 5799 : EG 1265 (4.6:1), which lets SG overfit while EG's
    temporal/spatial tasks are still underfit. Upsampling the minority to the majority
    count would also grow the epoch 1.6x, confounding "balanced" with "trained longer",
    so both sources are resampled to len(items)//n_src instead.

    Called per epoch with a rotating seed: the majority source is subsampled, and a
    fixed subset would permanently hide ~39% of SG from training.
    """
    from collections import defaultdict
    import random
    by_src = defaultdict(list)
    for it in items:
        by_src[it[0]].append(it)
    n_each = len(items) // len(by_src)
    rng = random.Random(seed)
    out = []
    for s in sorted(by_src):
        lst = by_src[s]
        reps, rem = divmod(n_each, len(lst))
        # Sampled, not a prefix: SG items are grouped by task and EG's by qa_type, so a
        # prefix would silently over-weight whichever task comes first.
        out += lst * reps + rng.sample(lst, rem)
    return out, {s: len(v) for s, v in sorted(by_src.items())}, n_each


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


# ── selection helpers (RGB content set shared by teacher-label + student-pick) ──

def content_and_avail(cached, content_ratio):
    ve, a, k = cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
    N = ve.shape[0]
    half = content_ratio / 2.0
    content_embeds, content_idx = visionzip_select_tokens(
        ve, a, k, dominant_ratio=half, contextual_ratio=half)
    avail = torch.ones(N, dtype=torch.bool, device=ve.device)
    avail[content_idx] = False
    return content_embeds, content_idx, avail.nonzero(as_tuple=True)[0]


def topk_in_avail(scores, avail_idx, k):
    kk = min(max(1, k), avail_idx.numel())
    top = torch.topk(scores[avail_idx], kk).indices
    return avail_idx[top], kk


def union_tokens(cached, content_embeds, content_idx, traj_idx):
    ve = cached["video_embeds"]
    all_embeds = torch.cat([content_embeds, ve[traj_idx]], dim=0)
    all_idx = torch.cat([content_idx, traj_idx])
    order = all_idx.argsort()
    return all_embeds[order], all_idx[order]


def selection_kd_loss(s_student, s_teacher, avail_idx, k):
    """BCE(student salience over discarded tokens, teacher top-k membership).
    Returns (loss, top-k agreement) — agreement = |student_topk ∩ teacher_topk| / k,
    the fraction of the gaze complement recovered from RGB alone."""
    teacher_top, kk = topk_in_avail(s_teacher, avail_idx, k)
    logits_av = s_student[avail_idx]
    tgt_av = torch.zeros_like(logits_av)
    # positions of teacher_top within avail_idx
    pos = torch.searchsorted(avail_idx, teacher_top)
    tgt_av[pos] = 1.0
    n_pos = tgt_av.sum().clamp(min=1.0)
    n_neg = (tgt_av.numel() - n_pos).clamp(min=1.0)
    pos_weight = (n_neg / n_pos).clamp(max=50.0)
    loss = F.binary_cross_entropy_with_logits(logits_av, tgt_av, pos_weight=pos_weight)
    with torch.no_grad():
        student_top, _ = topk_in_avail(s_student.detach(), avail_idx, k)
        inter = torch.isin(student_top, teacher_top).sum().item()
        agree = inter / max(1, kk)
    return loss, agree


@torch.no_grad()
def evaluate(processor, model, predictor, base_qwen, option_ids, device,
             content_ratio, traj_ratio, include_hdepic=True, source="both"):
    """Gaze-free eval: complement chosen by the predictor, NO gaze/hand read."""
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic)
    if source in ("sg", "eg"):
        test_ds.items = [it for it in test_ds.items if it[0] == source]
    model.eval(); predictor.eval()
    correct = 0; total = 0
    by_task: dict[str, list] = {}
    by_src: dict[str, list] = {}
    for idx in range(len(test_ds)):
        src = test_ds.items[idx][0]
        item = test_ds[idx]
        if item is None: continue
        try:
            cached = preprocess_visionzip_item(
                processor, base_qwen,
                item["vlm_frame_paths"], item["question"], item["options"], device)
            if cached is None: continue
            n_opt = len(item["options"])
            letters = [chr(65 + i) for i in range(n_opt)]
            if item["answer"] not in letters: continue

            content_embeds, content_idx, avail_idx = content_and_avail(cached, content_ratio)
            N = cached["video_embeds"].shape[0]
            k = min(max(1, int(traj_ratio * N)), avail_idx.numel())
            s_student = predictor(cached["video_embeds"], cached["attn_scores"], cached["grid_thw"])
            traj_idx, _ = topk_in_avail(s_student, avail_idx, k)
            sel_embeds, recv_idx = union_tokens(cached, content_embeds, content_idx, traj_idx)

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
    model.train(); predictor.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    per_src = {s: [100.0 * sum(v) / max(1, len(v)), len(v)] for s, v in sorted(by_src.items())}
    return 100.0 * correct / max(1, total), total, per_task, per_src


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    hp = dict(mask_modality="none")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[KD] output: {args.output_dir}", flush=True)
        print(f"[KD] selection distillation: RGB predictor mimics gaze/hand top-k; "
              f"content={args.content_ratio*100:.1f}% ∪ traj={args.traj_ratio*100:.1f}% = "
              f"{(args.content_ratio+args.traj_ratio)*100:.0f}%; λ_sel={args.lambda_sel}, "
              f"lr={args.lr}, pred_lr={args.pred_lr}, grad_accum={args.grad_accum}, "
              f"GPUs={world_size} (eff-batch {world_size*args.grad_accum}), "
              f"source={args.source}", flush=True)
        print("[KD] gaze/hand used at TRAIN only (teacher labels); eval is gaze-free.", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)

    # warm-start LoRA readout from M1 (before DDP wrap so all ranks start identical)
    warm_pred_state = None
    if args.warmstart_ckpt and os.path.exists(args.warmstart_ckpt):
        _warm = torch.load(args.warmstart_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(_warm["lora_state"], strict=False)
        # A STUDENT checkpoint also carries pred_state. Its LoRA is co-adapted to that
        # predictor's selections, so pairing it with a fresh random head would start
        # below the checkpoint's own score. Carry the predictor over too when present
        # (M1 teacher checkpoints have none, so this is a no-op for them).
        warm_pred_state = _warm.get("pred_state")
        if is_main:
            print(f"[KD] warm-started LoRA from {args.warmstart_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
            print(f"[KD] warm-start predictor: "
                  f"{'FOUND pred_state → continuing from it' if warm_pred_state else 'none in ckpt → fresh head'}",
                  flush=True)
        del _warm
    elif is_main:
        print(f"[KD] WARNING: no warm-start ckpt at {args.warmstart_ckpt}; LoRA from scratch.", flush=True)

    base_qwen = model.get_base_model()
    in_dim = base_qwen.get_input_embeddings().weight.shape[1]
    predictor = TrajSaliencePredictor(in_dim, hidden=args.pred_hidden).to(device)
    if warm_pred_state is not None:
        predictor.load_state_dict(warm_pred_state)
        if is_main:
            print("[KD] predictor warm-started from the checkpoint's pred_state.", flush=True)
    if is_main:
        n_pred = sum(p.numel() for p in predictor.parameters())
        print(f"[KD] predictor: in_dim={in_dim}, params={n_pred/1e6:.2f}M", flush=True)

    # Resume overrides the warm-start: an interrupted run must come back with the
    # distilled predictor too, not a fresh head on top of a trained LoRA.
    # Loaded before the DDP wrap so every rank starts from identical weights.
    start_epoch = 0
    if args.resume:
        ck_path, ck_epoch = _latest_epoch_ckpt(args.output_dir)
        if ck_path:
            st = torch.load(ck_path, map_location="cpu")
            model.load_state_dict(st["lora_state"], strict=False)
            predictor.load_state_dict(st["pred_state"])
            start_epoch = ck_epoch
            if is_main:
                print(f"[KD] resumed from {ck_path}; continuing at epoch {start_epoch+1}",
                      flush=True)
        elif is_main:
            print(f"[KD] --resume: no epoch_*.pth in {args.output_dir}; starting fresh.",
                  flush=True)

    if args.freeze_lora:
        # Predictor-only distillation: the warm-started LoRA already sits at teacher
        # quality, so hold it there and let the student differ from the teacher by the
        # SELECTION gap alone. Left unwrapped — DDP refuses a module with no parameter
        # requiring grad, and there is nothing to all-reduce.
        for p in model.parameters():
            p.requires_grad_(False)
        model_core = model
        if is_main:
            print("[KD] --freeze-lora: LoRA held at warm-start; training predictor only "
                  "(no task CE, no VLM backward).", flush=True)
    else:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        model_core = model.module
    # find_unused=True: the frame-context branch is skipped on any irregular grid,
    # which would otherwise deadlock the reducer mid-run.
    predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    # frozen gaze/hand teacher encoder (TRAIN-time only)
    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    if is_main: print("Model + teacher encoder loaded.", flush=True)

    if args.eval_ckpt:
        if is_main: print(f"[eval-only] loading student from {args.eval_ckpt}", flush=True)
        st = torch.load(args.eval_ckpt, map_location="cpu")
        model_core.load_state_dict(st["lora_state"], strict=False)
        predictor.module.load_state_dict(st["pred_state"])
        if is_main:
            acc, n_eval, per_task, per_src = evaluate(
                processor, model_core, predictor.module, base_qwen, option_ids, device,
                args.content_ratio, args.traj_ratio, include_hdepic=args.include_hdepic,
                source=args.source)
            print(f"[eval-only] Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"[eval-only]     {task}: {task_acc:.2f}%", flush=True)
            for s, (s_acc, s_n) in per_src.items():
                print(f"[eval-only] [src] {s}: {s_acc:.2f}%  (n={s_n})", flush=True)
        dist.barrier(); dist.destroy_process_group(); return

    if start_epoch >= args.epochs:
        if is_main:
            print(f"[KD] TRAINING COMPLETE (already at epoch {start_epoch}/{args.epochs})",
                  flush=True)
        dist.barrier(); dist.destroy_process_group(); return

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic)
    if args.source in ("sg", "eg"):
        n_before = len(train_ds.items)
        train_ds.items = [it for it in train_ds.items if it[0] == args.source]
        if is_main:
            print(f"[source={args.source}] train filtered {n_before} → {len(train_ds.items)} items",
                  flush=True)

    # The student's pixels and the teacher's must come from different frame variants when
    # VLM_GAZE_OVERLAY=0, or the "gaze-free" student is silently trained on the overlay —
    # a failure that changes no shape, raises no error, and does not show up in accuracy.
    if is_main and os.environ.get("VLM_GAZE_OVERLAY", os.environ.get("GAZE_OVERLAY", "1")) != \
            os.environ.get("GAZE_OVERLAY", "1"):
        probe = next((it for it in (train_ds[i] for i in range(min(50, len(train_ds))))
                      if it is not None), None)
        if probe is None:
            raise RuntimeError("[KD] stream check: first 50 train items all None")
        s_dir = os.path.basename(os.path.dirname(os.path.dirname(probe["vlm_frame_paths"][0])))
        t_dir = os.path.basename(os.path.dirname(os.path.dirname(probe["traj_frame_paths"][0])))
        print(f"[KD] frame streams: student VLM='{s_dir}'  teacher TAS='{t_dir}'", flush=True)
        if s_dir == t_dir:
            raise RuntimeError(
                f"[KD] VLM_GAZE_OVERLAY differs from GAZE_OVERLAY but both streams read "
                f"'{s_dir}'. The student would train on the teacher's overlay frames.")

    # Source balancing: repeat the minority source's items up to the majority count, so
    # each source contributes the same number of optimizer steps per epoch. Operates on
    # the flat (src, local_idx) list, same insertion point as the --source filter, and
    # must happen BEFORE the sampler is built so every rank sees the same index space.
    do_balance = args.balance_sources and args.source == "both"
    unbalanced_items = list(train_ds.items)
    if do_balance:
        train_ds.items, before, n_each = balance_items(unbalanced_items, args.balance_seed)
        if is_main:
            print(f"[balance-sources] {before} → {n_each} each, "
                  f"{len(train_ds.items)} items/epoch (was {sum(before.values())}); "
                  f"resampled every epoch", flush=True)

    # Length is identical with or without balancing, so the sampler's fixed
    # num_samples/total_size stay valid when .items is re-rolled between epochs.
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{"params": list(predictor.parameters()), "lr": args.pred_lr}]
    if lora_params:                      # empty under --freeze-lora
        param_groups.insert(0, {"params": lora_params, "lr": args.lr})
    optimizer = AdamW(param_groups, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    epoch_accs: list[float] = _prior_epoch_accs(log_path) if args.resume else []
    best_acc = max(epoch_accs) if epoch_accs else 0.0
    if is_main and epoch_accs:
        print(f"[KD] prior epoch accs {epoch_accs} → best_acc={best_acc:.2f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if do_balance:
            # Re-roll so a different SG subset (and EG remainder) is drawn each epoch;
            # every rank uses the same seed, so all ranks see an identical index space.
            train_ds.items, _, _ = balance_items(unbalanced_items,
                                                 args.balance_seed + epoch)
        sampler.set_epoch(epoch)
        model.train(); predictor.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; epoch_kd = 0.0; epoch_agree = 0.0; n_steps = 0
        t_start = time.time()

        for step, item in enumerate(loader):
            if item is None: continue
            try:
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device)
                if cached is None: continue
                n_video = cached["video_embeds"].shape[0]
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                content_embeds, content_idx, avail_idx = content_and_avail(cached, args.content_ratio)
                if avail_idx.numel() == 0: continue
                k_traj = min(max(1, int(args.traj_ratio * n_video)), avail_idx.numel())

                # privileged teacher field (gaze/hand), TRAIN-only
                with torch.no_grad():
                    s_teacher = _traj_scores(cached, item, device, "learned", encoder, hp)

                # RGB-only student salience (trainable)
                s_student = predictor(cached["video_embeds"], cached["attn_scores"], cached["grid_thw"])
                kd_loss, agree = selection_kd_loss(s_student, s_teacher, avail_idx, k_traj)

                # student picks its own complement (topk is non-diff → detach)
                with torch.no_grad():
                    traj_idx, _ = topk_in_avail(s_student.detach(), avail_idx, k_traj)

                if args.freeze_lora:
                    # No task CE: the LoRA is fixed, so the VLM forward/backward would
                    # produce no gradient for anything. Selection KD alone.
                    n_kept = content_idx.numel() + traj_idx.numel()
                    ce_val = 0.0
                    loss = (args.lambda_sel * kd_loss) / args.grad_accum
                else:
                    with torch.no_grad():
                        sel_embeds, recv_idx = union_tokens(
                            cached, content_embeds, content_idx, traj_idx)
                        inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                    n_kept = recv_idx.shape[0]

                    logits = forward_logits(model, inputs_dict)
                    option_logits = logits[option_ids[:n_opt]]
                    gt_idx = letters.index(item["answer"])
                    ce_loss = F.cross_entropy(option_logits.unsqueeze(0),
                                              torch.tensor([gt_idx], device=device))
                    ce_val = ce_loss.item()
                    loss = (ce_loss + args.lambda_sel * kd_loss) / args.grad_accum

                loss.backward()
                epoch_loss += ce_val; epoch_kd += kd_loss.item()
                epoch_agree += agree; n_steps += 1

                if n_steps % args.grad_accum == 0:
                    if lora_params:
                        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    elapsed = time.time() - t_start
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"ce={epoch_loss/n_steps:.4f} | kd={epoch_kd/n_steps:.4f} | "
                          f"agree={epoch_agree/n_steps:.3f} | kept={pct_kept:.1f}% | "
                          f"t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps,
                            "ce": epoch_loss/n_steps, "kd": epoch_kd/n_steps,
                            "agree": epoch_agree/n_steps, "pct_kept": pct_kept,
                            "elapsed": elapsed,
                        }) + "\n")
            except Exception:
                if is_main: traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            if lora_params:
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
            optimizer.step(); optimizer.zero_grad()

        elapsed = time.time() - t_start
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | ce={epoch_loss/max(1,n_steps):.4f} "
                  f"| kd={epoch_kd/max(1,n_steps):.4f} | agree={epoch_agree/max(1,n_steps):.3f} "
                  f"| time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model_core.state_dict(),
                "pred_state": predictor.module.state_dict(),
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} (gaze-free) on full {label} val set ...", flush=True)
            acc, n_eval, per_task, per_src = evaluate(
                processor, model_core, predictor.module, base_qwen, option_ids, device,
                args.content_ratio, args.traj_ratio, include_hdepic=args.include_hdepic,
                source=args.source)
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
                    "lora_state": model_core.state_dict(),
                    "pred_state": predictor.module.state_dict(),
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

    # Explicit terminal marker so a supervising restart loop can tell "finished"
    # from "crashed" — both otherwise just end the process.
    if is_main:
        print(f"[KD] TRAINING COMPLETE (best_acc={best_acc:.2f}%)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
