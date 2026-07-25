"""VisionZip-Complement + Fixation-Confidence-weighted gaze complement — LoRA.

Noise-aware gaze (see models/gaze_confidence.py). M1's complement is the raw
top-3% of TAS gaze salience; here we multiply that salience by a per-frame
fixation-confidence c(t) BEFORE the top-k, so gaze contributes during stable
fixations and is suppressed during saccades (28% of frames = noise on egtea).

Everything else is M1 verbatim: 7%C content selection (VisionZip) ∪ 3%G complement
(learned TAS encoder, top-k), GAZE_OVERLAY, egtea val. Only the 3%G pool's scoring
is reweighted by c(t).

Arms (set with --arm):
  confidence — c(t) fixation confidence              (the method)
  inverse    — 1 - c(t), boost saccades = inject noise (KILL-TEST)
  random     — random per-frame weight               (placebo)
  none       — all-ones ≡ M1 raw complement          (in-protocol baseline)

confidence > inverse (significant) ⇒ saccade tokens were hurting M1 and we fixed it.
confidence ≈ none                  ⇒ the learned encoder already denoised implicitly.

Training protocol: single-GPU + grad-accum 8 = eff-batch 8 (matches tbudget /
foveal / gazetext novelty experiments).

Usage:
    GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=<N> torchrun --nproc_per_node=1 \\
        --master_port=<PORT> \\
        -m TrajGazeMerge.training.train_visionzip_gazeconf_lora \\
        --arm confidence \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/gazeconf_confidence \\
        --epochs 3 --lr 1e-4 --grad-accum 8 --no-hdepic --early-stop
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.gaze_confidence import fixation_confidence, resolve_arm
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import _score_to_qwen_robust
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal,
)

STAGE1_DEFAULT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth"
CONTENT_RATIO  = 0.07
TRAJ_RATIO     = 0.03


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm",
                   choices=["confidence", "inverse", "random", "none",
                            "task_adaptive", "signrouted", "dual"],
                   default="confidence")
    p.add_argument("--fix-ratio", type=float, default=0.02,
                   help="dual arm: fixation-pool budget (confidence-weighted).")
    p.add_argument("--sac-ratio", type=float, default=0.01,
                   help="dual arm: saccade-pool budget (inverse-weighted). "
                        "fix+sac should equal TRAJ_RATIO (0.03).")
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/gazeconf_confidence")
    p.add_argument("--stage1-ckpt", default=STAGE1_DEFAULT)
    p.add_argument("--saccade-speed", type=float, default=0.25)
    p.add_argument("--window",        type=int,   default=5)
    p.add_argument("--c-min",         type=float, default=0.10)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=8)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false")
    p.add_argument("--early-stop", action="store_true")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _item_seed(item) -> int:
    s = "|".join([str(item.get("task", "")), str(item.get("question", ""))])
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _conf_weighted_traj_scores(cached, item, device, encoder, arm,
                               saccade_speed, window, c_min):
    """Per-video-token TAS gaze salience reweighted by per-frame fixation
    confidence c(t). c(t) is applied at the (T_traj, 196) stage — the temporal
    axis of the raw encoder output — then mapped to VisionZip's (N,) layout."""
    video_embeds = cached["video_embeds"]
    n_video   = video_embeds.shape[0]
    T_merged  = int(cached["grid_thw"][0, 0].item())
    n_spatial = n_video // max(1, T_merged)

    scores = get_patch_scores_temporal(encoder, item, device)        # (T_traj, 196)
    T_traj = scores.shape[0]

    eff_arm = resolve_arm(arm, item.get("task", ""))                 # task_adaptive → per-task
    c = fixation_confidence(
        item["traj"]["gaze_speed"], item["traj"]["gaze_mask"],
        arm=eff_arm, n_out=T_traj, item_seed=_item_seed(item),
        saccade_speed=saccade_speed, window=window, c_min=c_min,
    ).to(scores.device)                                              # (T_traj,)
    scores = scores * c.view(T_traj, 1)                              # reweight per frame

    scores_all = _score_to_qwen_robust(scores, n_spatial, T_merged, cached["grid_thw"])
    if scores_all.shape[0] != n_video:
        scores_all = (scores_all[:n_video] if scores_all.shape[0] > n_video
                      else scores_all.repeat(
                          (n_video + scores_all.shape[0] - 1) // scores_all.shape[0])[:n_video])
    return scores_all.to(device)


def _dual_traj_scores(cached, item, device, encoder, saccade_speed, window, c_min):
    """Return (fix_mapped, sac_mapped) (N,): TAS salience reweighted by fixation
    confidence AND by its inverse (saccade), sharing ONE encoder call."""
    video_embeds = cached["video_embeds"]
    n_video   = video_embeds.shape[0]
    T_merged  = int(cached["grid_thw"][0, 0].item())
    n_spatial = n_video // max(1, T_merged)

    scores = get_patch_scores_temporal(encoder, item, device)        # (T_traj, 196)
    T_traj = scores.shape[0]
    out = []
    for sub_arm in ("confidence", "inverse"):
        c = fixation_confidence(
            item["traj"]["gaze_speed"], item["traj"]["gaze_mask"],
            arm=sub_arm, n_out=T_traj, item_seed=_item_seed(item),
            saccade_speed=saccade_speed, window=window, c_min=c_min,
        ).to(scores.device)
        s_all = _score_to_qwen_robust(scores * c.view(T_traj, 1),
                                      n_spatial, T_merged, cached["grid_thw"])
        if s_all.shape[0] != n_video:
            s_all = (s_all[:n_video] if s_all.shape[0] > n_video
                     else s_all.repeat(
                         (n_video + s_all.shape[0] - 1) // s_all.shape[0])[:n_video])
        out.append(s_all.to(device))
    return out[0], out[1]


def select_complementary_conf(cached, item, device, encoder, arm,
                              saccade_speed, window, c_min,
                              fix_ratio=0.02, sac_ratio=0.01):
    """M1's content ∪ gaze complement (top-k). arm='dual' splits the complement
    into a fixation pool (confidence-weighted, object grounding) and a DISJOINT
    saccade pool (inverse-weighted, dynamic transitions) so neither regime's
    signal is discarded. Other arms keep the single-pool reweighting."""
    video_embeds = cached["video_embeds"]
    attn_scores  = cached["attn_scores"]
    attn_key     = cached["attn_key"]
    N = video_embeds.shape[0]

    half = CONTENT_RATIO / 2.0
    content_embeds, content_idx = visionzip_select_tokens(
        video_embeds, attn_scores, attn_key,
        dominant_ratio=half, contextual_ratio=half,
    )

    avail_mask = torch.ones(N, dtype=torch.bool, device=video_embeds.device)
    avail_mask[content_idx] = False

    if arm == "dual":
        fix_s, sac_s = _dual_traj_scores(
            cached, item, device, encoder, saccade_speed, window, c_min)
        extra = []
        for scores_vec, ratio in ((fix_s, fix_ratio), (sac_s, sac_ratio)):
            avail = avail_mask.nonzero(as_tuple=True)[0]
            k = min(max(1, int(ratio * N)), avail.numel())
            if k > 0 and avail.numel() > 0:
                sel = avail[torch.topk(scores_vec[avail], k).indices]
                avail_mask[sel] = False                 # keep pools disjoint
                extra.append(sel)
        if extra:
            traj_idx = torch.cat(extra)
            all_embeds = torch.cat([content_embeds, video_embeds[traj_idx]], dim=0)
            all_idx    = torch.cat([content_idx, traj_idx])
        else:
            all_embeds, all_idx = content_embeds, content_idx
        order = all_idx.argsort()
        return all_embeds[order], all_idx[order]

    traj_scores = _conf_weighted_traj_scores(
        cached, item, device, encoder, arm, saccade_speed, window, c_min)
    avail = avail_mask.nonzero(as_tuple=True)[0]
    k_traj = min(max(1, int(TRAJ_RATIO * N)), avail.numel())

    if k_traj > 0 and avail.numel() > 0:
        top = torch.topk(traj_scores[avail], k_traj).indices
        traj_idx = avail[top]
        all_embeds = torch.cat([content_embeds, video_embeds[traj_idx]], dim=0)
        all_idx    = torch.cat([content_idx, traj_idx])
    else:
        all_embeds, all_idx = content_embeds, content_idx

    order = all_idx.argsort()
    return all_embeds[order], all_idx[order]


def evaluate(processor, model, base_qwen, option_ids, device,
             encoder, arm, saccade_speed, window, c_min, n_frames, include_hdepic,
             fix_ratio=0.02, sac_ratio=0.01):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=n_frames, n_traj_frames=n_frames,
        include_hdepic=include_hdepic,
    )
    model.eval()
    correct = 0; total = 0
    by_task: dict[str, list] = {}
    with torch.no_grad():
        for item in test_ds:
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

                sel_embeds, recv_idx = select_complementary_conf(
                    cached, item, device, encoder, arm, saccade_speed, window, c_min,
                    fix_ratio=fix_ratio, sac_ratio=sac_ratio)
                inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
                logits = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok; total += 1
                by_task.setdefault(item["task"], []).append(ok)
            except Exception:
                pass
    model.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[GazeConf] arm={args.arm}  output: {args.output_dir}")
        print(f"[GazeConf] M1 selection {CONTENT_RATIO*100:.0f}%C ∪ {TRAJ_RATIO*100:.0f}%G, "
              f"complement reweighted by fixation confidence "
              f"(saccade_speed={args.saccade_speed}, window={args.window}, "
              f"c_min={args.c_min})")
        print(f"[GazeConf] GPUs={world_size} epochs={args.epochs} lr={args.lr} "
              f"grad_accum={args.grad_accum}", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    if is_main: print("Model + encoder loaded.", flush=True)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic,
    )
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
                with torch.no_grad():
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device,
                    )
                if cached is None: continue
                n_video = cached["video_embeds"].shape[0]

                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters: continue

                with torch.no_grad():
                    sel_embeds, recv_idx = select_complementary_conf(
                        cached, item, device, encoder, args.arm,
                        args.saccade_speed, args.window, args.c_min,
                        fix_ratio=args.fix_ratio, sac_ratio=args.sac_ratio)
                    inputs_dict = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)

                logits = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                loss = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
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
                "loss": avg_loss, "arm": args.arm,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            print(f"Evaluating epoch {epoch+1} (egtea 2-way) ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, option_ids, device,
                encoder, args.arm, args.saccade_speed, args.window, args.c_min,
                args.n_frames, args.include_hdepic,
                fix_ratio=args.fix_ratio, sac_ratio=args.sac_ratio,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            epoch_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "acc": acc, "arm": args.arm,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)

            if args.early_stop and epoch >= 1 and acc <= epoch_accs[-2]:
                print(f"  Early stop: epoch{epoch+1} {acc:.2f}% <= "
                      f"epoch{epoch} {epoch_accs[-2]:.2f}% → stopping.", flush=True)
                stop.fill_(1.0)

        dist.broadcast(stop, src=0)
        dist.barrier()
        if stop.item() > 0:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
