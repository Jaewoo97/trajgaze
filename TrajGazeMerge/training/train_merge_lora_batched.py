"""
Stage 2 trainer — micro-batched variant.

Identical objective to `train_merge_lora_temporal_no_kd.py` but processes
K samples per Qwen forward call instead of one at a time. Wall-clock speedup
~2-3× expected on H100 (current utilization ~20%; batching amortizes LLM forward
better). Effective batch size held constant via `--micro-batch K --grad-accum N`
giving effective batch K*N.

Loss components (mirrors the per-sample script):
  - CE (always)
  - shuffle margin (Sprint 2.1 §C)        — V_shuf component of cf-mask plan
  - ATR regression (trajectory-grounded plan)
  - CGM margin (trajectory-grounded plan) — mask_gaze region only
  - cf-mask-kept margin                   — V_mask (all visual zeroed)
    Direction A from docs/current_state.md. Forces gt_logit on baseline
    to exceed gt_logit on all-zero kept tokens by `--cf-mask-margin`. Attacks
    language-prior dominance on EgoGazeVQA-style datasets directly.

The per-sample script remains intact for backward compatibility and currently
running ablations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids,
)
from TrajGazeMerge.models.model_batched import (
    preprocess_batch, slice_sample_video,
    build_merged_inputs_batch, forward_logits_batch,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal, score_to_qwen_spatiotemporal,
    _evaluate_one,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",      choices=["full", "gaze_only", "hand_only"], required=True)
    p.add_argument("--stage1-ckpt",     required=True)
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--lr-lora",         type=float, default=1e-4)
    p.add_argument("--lr-enc",          type=float, default=1e-5)
    p.add_argument("--merge-ratio",     type=float, default=0.9)
    p.add_argument("--micro-batch",     type=int,   default=4,
                   help="Number of samples per Qwen forward call.")
    p.add_argument("--grad-accum",      type=int,   default=4,
                   help="Optimizer step every grad_accum micro-batches. "
                        "Effective batch = micro_batch * grad_accum.")
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--log-every",       type=int,   default=20)
    p.add_argument("--eval-every",      type=int,   default=400)
    p.add_argument("--n-frames",        type=int,   default=128)
    p.add_argument("--n-traj-frames",   type=int,   default=128)
    p.add_argument("--n-vis-keyframes", type=int,   default=16)
    p.add_argument("--seed",            type=int,   default=42)
    # Shuffle aug
    p.add_argument("--shuffle-aug",     action="store_true")
    p.add_argument("--shuffle-prob",    type=float, default=0.5)
    p.add_argument("--shuffle-margin",  type=float, default=0.5)
    p.add_argument("--shuffle-lambda",  type=float, default=0.5)
    p.add_argument("--shuffle-warmup-steps", type=int, default=200)
    # ATR
    p.add_argument("--use-atr",     action="store_true")
    p.add_argument("--atr-lambda",  type=float, default=0.5)
    p.add_argument("--atr-hidden",  type=int,   default=256)
    # CGM
    p.add_argument("--cgm-aug",     action="store_true")
    p.add_argument("--cgm-target",  choices=["gaze", "hand"], default="gaze")
    p.add_argument("--cgm-lambda",  type=float, default=0.3)
    p.add_argument("--cgm-prob",    type=float, default=0.3)
    p.add_argument("--cgm-radius",  type=float, default=0.2)
    p.add_argument("--cgm-margin",  type=float, default=0.5)
    p.add_argument("--cgm-warmup-steps", type=int, default=600)
    # cf-mask-kept margin (Direction A, plan cf-mask-augmented-training.md)
    p.add_argument("--use-cf-mask",     action="store_true",
                   help="Enable cf-mask-kept margin loss (V_mask). Zeros ALL kept tokens "
                        "with prob `--cf-mask-prob`, penalizes when baseline gt_logit is not "
                        "above the zeroed forward's gt_logit by `--cf-mask-margin`.")
    p.add_argument("--cf-mask-prob",    type=float, default=0.3)
    p.add_argument("--cf-mask-margin",  type=float, default=1.0)
    p.add_argument("--cf-mask-lambda",  type=float, default=0.3)
    p.add_argument("--cf-mask-warmup-steps", type=int, default=600)
    # Datasets
    p.add_argument("--use-egovqa", action="store_true")
    p.add_argument("--use-hd-epic", action="store_true")
    p.add_argument("--eval-egovqa-egtea", action="store_true")
    p.add_argument("--eval-hd-epic", action="store_true")
    p.add_argument("--dataloader-num-workers", type=int, default=8)
    return p.parse_args()


def evaluate(processor, qwen_model, base_qwen, traj_encoder, option_ids, device,
             merge_ratio, eval_egovqa_egtea: bool = False, eval_hd_epic: bool = False):
    """Per-sample eval (existing helper reused — eval batching not needed for speed)."""
    qwen_model.eval()
    traj_encoder.eval()
    val_sets: dict[str, object] = {
        "streamgaze_egtea": StreamGazeMergeDataset(
            split="test", n_vlm_frames=128, n_traj_frames=128,
        ),
    }
    if eval_egovqa_egtea:
        from TrajGazeMerge.data.dataset_egovqa import EgoGazeVQAMergeDataset
        val_sets["egovqa_egtea"] = EgoGazeVQAMergeDataset(
            split="test", n_vlm_frames=128, n_traj_frames=128, datasets=("egtea",),
        )
    if eval_hd_epic:
        from TrajGazeMerge.data.dataset_hdepic import HDEpicMergeDataset
        val_sets["hd_epic_p09"] = HDEpicMergeDataset(
            split="val", n_vlm_frames=128, n_traj_frames=128,
        )
    results = {}
    for name, vs in val_sets.items():
        acc, n, pt = _evaluate_one(
            vs, processor, qwen_model, base_qwen, traj_encoder,
            option_ids, device, merge_ratio,
        )
        results[name] = (acc, n, pt)
    qwen_model.train()
    traj_encoder.train()
    return results


def _filter_valid(items: list) -> list:
    return [x for x in items if x is not None]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    print(f"[Stage2-batched] micro_batch={args.micro_batch}  grad_accum={args.grad_accum}  "
          f"effective_batch={args.micro_batch * args.grad_accum}", flush=True)
    print(f"  stage1_ckpt: {args.stage1_ckpt}")
    print(f"  output_dir:  {args.output_dir}", flush=True)

    print("Loading Qwen2.5-VL + LoRA ...", flush=True)
    processor, qwen_model = load_qwen_lora(device)
    base_qwen  = qwen_model.get_base_model()
    option_ids = get_option_ids(processor)

    print(f"Loading TrajGaze encoder ({args.model_type}) ...", flush=True)
    traj_encoder = load_traj_encoder(args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes)
    print("All models loaded.", flush=True)

    if args.use_egovqa or args.use_hd_epic:
        from TrajGazeMerge.data.dataset_combined import build_combined_train_dataset
        train_ds = build_combined_train_dataset(
            n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
            use_streamgaze=True, use_egovqa=args.use_egovqa,
            use_hd_epic=args.use_hd_epic,
        )
    else:
        train_ds = StreamGazeMergeDataset(
            split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames,
        )

    # Custom collate returns a list (no stacking — shapes differ per sample)
    loader = DataLoader(
        train_ds, batch_size=args.micro_batch, shuffle=True,
        collate_fn=lambda batch: batch,
        num_workers=args.dataloader_num_workers,
    )

    atr_head = None
    lora_params = [p for p in qwen_model.parameters() if p.requires_grad]
    enc_params  = list(traj_encoder.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": enc_params,  "lr": args.lr_enc},
    ], weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(args.epochs):
        qwen_model.train()
        traj_encoder.train()
        if atr_head is not None:
            atr_head.train()
        optimizer.zero_grad()
        epoch_loss = epoch_ce = epoch_shuf = epoch_atr = epoch_cgm = epoch_cf_mask = 0.0
        n_shuf = n_atr = n_cgm = n_cf_mask = 0
        micro_step = 0
        accum_step = 0
        t_start = time.time()

        for raw_batch in loader:
            items = _filter_valid(raw_batch)
            if not items:
                continue
            try:
                batched = preprocess_batch(processor, base_qwen, items, device)
                if batched is None:
                    continue
                # Some items may have failed inside preprocess_batch (e.g. all frames None).
                items = [items[i] for i in batched["valid_idx"]]
                B = len(items)
                if B == 0:
                    continue

                # Per-sample: encoder scores, score-to-spatial, merge
                merged_videos: list[torch.Tensor] = []
                receiver_idxs: list[torch.Tensor] = []
                gt_indices: list[int] = []
                per_sample_meta: list[dict] = []   # for ATR/CGM

                for b, item in enumerate(items):
                    video_b = slice_sample_video(batched, b)        # (N_b, d)
                    N_b = video_b.shape[0]
                    T_merged_b = int(batched["grid_thw"][b, 0].item())
                    n_spatial_b = N_b // max(1, T_merged_b)
                    r_b = max(1, int(args.merge_ratio * N_b))

                    scores = get_patch_scores_temporal(traj_encoder, item, device)
                    scores_all = score_to_qwen_spatiotemporal(scores, n_spatial_b, T_merged_b)
                    if scores_all.shape[0] != N_b:
                        scores_all = (
                            scores_all[:N_b] if scores_all.shape[0] > N_b
                            else scores_all.repeat((N_b + scores_all.shape[0] - 1) // scores_all.shape[0])[:N_b]
                        )

                    merged_video, receiver_idx = gaze_weighted_merge(video_b, scores_all, r_b)
                    merged_videos.append(merged_video)
                    receiver_idxs.append(receiver_idx)
                    gt_indices.append(["A","B","C","D"].index(item["answer"]))
                    per_sample_meta.append({
                        "T_merged": T_merged_b,
                        "n_spatial": n_spatial_b,
                        "N": N_b, "r": r_b,
                    })

                gt_tensor = torch.tensor(gt_indices, device=device, dtype=torch.long)

                # Batched Qwen forward (CE)
                batched_inputs = build_merged_inputs_batch(
                    base_qwen, batched, merged_videos, receiver_idxs,
                )
                logits = forward_logits_batch(qwen_model, batched_inputs)   # (B, vocab)
                opt_logits = logits[:, option_ids]                          # (B, 4)
                loss_ce = F.cross_entropy(opt_logits, gt_tensor)

                # Sprint 2 shuffle aug — one shuffle per sample, batched forward
                gstep = epoch * len(loader) + micro_step
                use_shuffle = (
                    args.shuffle_aug
                    and gstep >= args.shuffle_warmup_steps
                    and random.random() < args.shuffle_prob
                )
                loss_shuf = torch.tensor(0.0, device=device)
                if use_shuffle:
                    merged_shufs = [
                        m[torch.randperm(m.shape[0], device=device)]
                        for m in merged_videos
                    ]
                    shuf_inputs = build_merged_inputs_batch(
                        base_qwen, batched, merged_shufs, receiver_idxs,
                    )
                    logits_shuf = forward_logits_batch(qwen_model, shuf_inputs)
                    gt_logit_norm = logits.detach()[torch.arange(B), gt_tensor]      # detached
                    gt_logit_shuf = logits_shuf[torch.arange(B), gt_tensor]
                    loss_shuf = F.relu(gt_logit_shuf - gt_logit_norm + args.shuffle_margin).mean()

                # ATR — per-sample pool + batched head + loss
                loss_atr = torch.tensor(0.0, device=device)
                if args.use_atr:
                    from TrajGazeMerge.models.trajectory_grounding import (
                        receivers_to_frame_pool, TrajectoryReconstructionHead,
                        downsample_traj_to_T_merged, atr_regression_loss,
                    )
                    if atr_head is None:
                        d_in = batched["video_embeds"].shape[-1]
                        atr_head = TrajectoryReconstructionHead(d_in, args.atr_hidden).to(device)
                        optimizer.add_param_group({
                            "params": list(atr_head.parameters()),
                            "lr":     args.lr_enc,
                            "weight_decay": 1e-4,
                        })
                        print(f"[ATR] head initialized: d_in={d_in}, hidden={args.atr_hidden}", flush=True)

                    atr_losses = []
                    for b, item in enumerate(items):
                        Tm = per_sample_meta[b]["T_merged"]
                        ns = per_sample_meta[b]["n_spatial"]
                        pooled, recv_present = receivers_to_frame_pool(
                            merged_videos[b], receiver_idxs[b], ns, Tm,
                        )
                        pred_gaze, pred_hand = atr_head(pooled)
                        traj = item["traj"]
                        gt_gaze  = downsample_traj_to_T_merged(traj["gaze_pos" ].to(device).float(), Tm)
                        gt_left  = downsample_traj_to_T_merged(traj["left_pos" ].to(device).float(), Tm)
                        gt_right = downsample_traj_to_T_merged(traj["right_pos"].to(device).float(), Tm)
                        gm = downsample_traj_to_T_merged(traj["gaze_mask" ].to(device), Tm).bool()
                        lm = downsample_traj_to_T_merged(traj["left_mask" ].to(device), Tm).bool()
                        rm = downsample_traj_to_T_merged(traj["right_mask"].to(device), Tm).bool()
                        atr_losses.append(atr_regression_loss(
                            pred_gaze, pred_hand, gt_gaze, gt_left, gt_right,
                            gm, lm, rm, recv_present,
                        ))
                    loss_atr = torch.stack(atr_losses).mean()

                # CGM — per-sample region mask + ONE batched forward over masked tokens
                loss_cgm = torch.tensor(0.0, device=device)
                use_cgm = (
                    args.cgm_aug
                    and gstep >= args.cgm_warmup_steps
                    and random.random() < args.cgm_prob
                )
                if use_cgm:
                    from TrajGazeMerge.models.trajectory_grounding import (
                        trajectory_region_receiver_mask,
                    )
                    merged_cgms: list[torch.Tensor] = []
                    any_region = False
                    for b, item in enumerate(items):
                        Tm = per_sample_meta[b]["T_merged"]
                        ns = per_sample_meta[b]["n_spatial"]
                        traj = item["traj"]
                        if args.cgm_target == "gaze":
                            pos  = traj["gaze_pos" ].to(device).float()
                            mask = traj["gaze_mask"].to(device).bool()
                            targets = pos.unsqueeze(1)
                            tmask   = mask.unsqueeze(1)
                        else:
                            l_pos  = traj["left_pos" ].to(device).float()
                            r_pos  = traj["right_pos"].to(device).float()
                            l_mask = traj["left_mask"].to(device).bool()
                            r_mask = traj["right_mask"].to(device).bool()
                            targets = torch.stack([l_pos, r_pos], dim=1)
                            tmask   = torch.stack([l_mask, r_mask], dim=1)
                        in_region = trajectory_region_receiver_mask(
                            receiver_idxs[b], Tm, ns, targets, tmask, args.cgm_radius,
                        )
                        m = merged_videos[b].clone()
                        if in_region.any():
                            m[in_region] = 0.0
                            any_region = True
                        merged_cgms.append(m)
                    if any_region:
                        cgm_inputs = build_merged_inputs_batch(
                            base_qwen, batched, merged_cgms, receiver_idxs,
                        )
                        logits_cgm = forward_logits_batch(qwen_model, cgm_inputs)
                        gt_logit_norm = logits.detach()[torch.arange(B), gt_tensor]
                        gt_logit_cgm  = logits_cgm[torch.arange(B), gt_tensor]
                        loss_cgm = F.relu(gt_logit_cgm - gt_logit_norm + args.cgm_margin).mean()

                # cf-mask-kept margin — zero ALL kept tokens, force gt_logit to drop
                # (Direction A from docs/current_state.md). Mirrors CGM structure but
                # masks the entire kept set instead of just the gaze region.
                loss_cf_mask = torch.tensor(0.0, device=device)
                use_cf_mask = (
                    args.use_cf_mask
                    and gstep >= args.cf_mask_warmup_steps
                    and random.random() < args.cf_mask_prob
                )
                if use_cf_mask:
                    merged_zeros = [torch.zeros_like(m) for m in merged_videos]
                    cf_inputs = build_merged_inputs_batch(
                        base_qwen, batched, merged_zeros, receiver_idxs,
                    )
                    logits_cf = forward_logits_batch(qwen_model, cf_inputs)
                    gt_logit_norm = logits.detach()[torch.arange(B), gt_tensor]
                    gt_logit_cf   = logits_cf[torch.arange(B), gt_tensor]
                    loss_cf_mask = F.relu(gt_logit_cf - gt_logit_norm + args.cf_mask_margin).mean()

                loss = (
                    loss_ce
                    + args.shuffle_lambda * loss_shuf
                    + args.atr_lambda     * loss_atr
                    + args.cgm_lambda     * loss_cgm
                    + args.cf_mask_lambda * loss_cf_mask
                ) / args.grad_accum

                loss.backward()
                epoch_loss += loss_ce.item()
                epoch_ce   += loss_ce.item()
                if use_shuffle:
                    epoch_shuf += loss_shuf.item();  n_shuf += 1
                if args.use_atr:
                    epoch_atr  += float(loss_atr.item()); n_atr  += 1
                if use_cgm and loss_cgm.requires_grad:
                    epoch_cgm  += float(loss_cgm.item()); n_cgm  += 1
                if use_cf_mask:
                    epoch_cf_mask += float(loss_cf_mask.item()); n_cf_mask += 1
                micro_step += 1
                accum_step += 1

                if accum_step >= args.grad_accum:
                    torch.nn.utils.clip_grad_norm_(lora_params + enc_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_step = 0

                if micro_step % args.log_every == 0:
                    avg_l = epoch_loss / micro_step
                    extras = []
                    if args.shuffle_aug: extras.append(f"shuf={epoch_shuf/max(1,n_shuf):.4f} (n={n_shuf})")
                    if args.use_atr:     extras.append(f"atr={epoch_atr/max(1,n_atr):.4f}")
                    if args.cgm_aug:     extras.append(f"cgm={epoch_cgm/max(1,n_cgm):.4f} (n={n_cgm})")
                    if args.use_cf_mask: extras.append(f"cf_mask={epoch_cf_mask/max(1,n_cf_mask):.4f} (n={n_cf_mask})")
                    extra = " | " + " | ".join(extras) if extras else ""
                    print(f"Epoch {epoch+1} | mstep {micro_step}/{len(loader)} | "
                          f"loss={avg_l:.4f}{extra} | t={time.time()-t_start:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": micro_step,
                            "loss":   avg_l,
                            "loss_shuf": epoch_shuf/max(1,n_shuf) if args.shuffle_aug else None,
                            "n_shuf":    n_shuf if args.shuffle_aug else None,
                            "loss_atr":  epoch_atr/max(1,n_atr) if args.use_atr else None,
                            "loss_cgm":  epoch_cgm/max(1,n_cgm) if args.cgm_aug else None,
                            "n_cgm":     n_cgm if args.cgm_aug else None,
                            "loss_cf_mask": epoch_cf_mask/max(1,n_cf_mask) if args.use_cf_mask else None,
                            "n_cf_mask":    n_cf_mask if args.use_cf_mask else None,
                        }) + "\n")

                if micro_step % args.eval_every == 0:
                    results = evaluate(
                        processor, qwen_model, base_qwen, traj_encoder,
                        option_ids, device, args.merge_ratio,
                        eval_egovqa_egtea=args.eval_egovqa_egtea,
                        eval_hd_epic=args.eval_hd_epic,
                    )
                    accs = [r[0] for r in results.values()]
                    mean_acc = sum(accs) / len(accs)
                    for name, (acc, n_eval, per_task) in results.items():
                        print(f"  → eval[{name}]: acc={acc:.2f}% (n={n_eval})")
                    if len(results) > 1:
                        print(f"  → eval[mean]: {mean_acc:.2f}%")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "type": "eval", "epoch": epoch+1, "step": micro_step,
                            "mean_acc": mean_acc,
                            "per_val":  {k: {"acc": r[0], "n_eval": r[1], "per_task": r[2]}
                                         for k, r in results.items()},
                        }) + "\n")
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        save_dict = {
                            "epoch": epoch, "step": micro_step,
                            "lora_state":    qwen_model.state_dict(),
                            "encoder_state": traj_encoder.state_dict(),
                            "acc": mean_acc,
                            "per_val_acc": {k: r[0] for k, r in results.items()},
                        }
                        if atr_head is not None:
                            save_dict["atr_state"] = atr_head.state_dict()
                        torch.save(save_dict, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (mean={mean_acc:.2f}%)", flush=True)

            except Exception:
                traceback.print_exc()
                continue

        # End of epoch
        torch.save({
            "epoch": epoch+1,
            "lora_state":    qwen_model.state_dict(),
            "encoder_state": traj_encoder.state_dict(),
        }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    print(f"\nTraining complete. Best mean acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
