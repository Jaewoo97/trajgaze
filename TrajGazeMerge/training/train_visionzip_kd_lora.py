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
sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

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
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if epoch-2 val <= epoch-1 val.")
    p.add_argument("--eval-ckpt", default=None,
                   help="Skip training: load this student.pth (lora_state + pred_state), "
                        "run one gaze-free evaluate() with per-source logging, exit.")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


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
             content_ratio, traj_ratio, include_hdepic=True):
    """Gaze-free eval: complement chosen by the predictor, NO gaze/hand read."""
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic)
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
              f"GPUs={world_size} (eff-batch {world_size*args.grad_accum})", flush=True)
        print("[KD] gaze/hand used at TRAIN only (teacher labels); eval is gaze-free.", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)

    # warm-start LoRA readout from M1 (before DDP wrap so all ranks start identical)
    if args.warmstart_ckpt and os.path.exists(args.warmstart_ckpt):
        sd = torch.load(args.warmstart_ckpt, map_location="cpu")["lora_state"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[KD] warm-started LoRA from {args.warmstart_ckpt} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    elif is_main:
        print(f"[KD] WARNING: no warm-start ckpt at {args.warmstart_ckpt}; LoRA from scratch.", flush=True)

    base_qwen = model.get_base_model()
    in_dim = base_qwen.get_input_embeddings().weight.shape[1]
    predictor = TrajSaliencePredictor(in_dim, hidden=args.pred_hidden).to(device)
    if is_main:
        n_pred = sum(p.numel() for p in predictor.parameters())
        print(f"[KD] predictor: in_dim={in_dim}, params={n_pred/1e6:.2f}M", flush=True)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
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
        model.module.load_state_dict(st["lora_state"], strict=False)
        predictor.module.load_state_dict(st["pred_state"])
        if is_main:
            acc, n_eval, per_task, per_src = evaluate(
                processor, model.module, predictor.module, base_qwen, option_ids, device,
                args.content_ratio, args.traj_ratio, include_hdepic=args.include_hdepic)
            print(f"[eval-only] Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for s, (s_acc, s_n) in per_src.items():
                print(f"[eval-only] [src] {s}: {s_acc:.2f}%  (n={s_n})", flush=True)
        dist.barrier(); dist.destroy_process_group(); return

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": list(predictor.parameters()), "lr": args.pred_lr},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0
    epoch_accs: list[float] = []

    for epoch in range(args.epochs):
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
                    sel_embeds, recv_idx = union_tokens(cached, content_embeds, content_idx, traj_idx)
                    inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)

                logits = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                ce_loss = F.cross_entropy(option_logits.unsqueeze(0),
                                          torch.tensor([gt_idx], device=device))

                loss = (ce_loss + args.lambda_sel * kd_loss) / args.grad_accum
                loss.backward()
                epoch_loss += ce_loss.item(); epoch_kd += kd_loss.item()
                epoch_agree += agree; n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    elapsed = time.time() - t_start
                    pct_kept = 100.0 * recv_idx.shape[0] / max(1, n_video)
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
                "lora_state": model.module.state_dict(),
                "pred_state": predictor.module.state_dict(),
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} (gaze-free) on full {label} val set ...", flush=True)
            acc, n_eval, per_task, per_src = evaluate(
                processor, model.module, predictor.module, base_qwen, option_ids, device,
                args.content_ratio, args.traj_ratio, include_hdepic=args.include_hdepic)
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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
