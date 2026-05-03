"""
EA1 ablation training — single-GPU stage-1 for TrajGazeMerge_v2.

Five ablations of the EA1 model (SpatiotemporalEncoderTemporal with
use_frame_score_branch=True, gate=0 frozen):

  score_only   [1.b] Drop l_traj + l_score_traj. Train with l_score_past +
                     l_score_future only. Tests whether trajectory prediction
                     loss is necessary (vs heatmap supervision alone).

  gaze_only    [2.b] 1-token gaze encoder (no hand/IMU). All 4 losses.
                     Tests whether hand modality contributes.

  hand_only    [2.a] 3-token hand encoder (no gaze). All 4 losses.
                     Tests whether gaze modality contributes.

  no_spatial   [3.a] EA1 encoder, visual_feat=None → trajectory-only score
                     path (no TemporalVisualTrajFusion). Tests spatial
                     grounding contribution.

  no_temporal  [3.b] EA1 encoder without frame_score_branch, gate=0 frozen.
                     No temporal signal in any path. Tests temporal dimension
                     contribution.

After stage-1 completes, auto-launches stage-2 (train_merge_lora_temporal or
the modality-specific variant) on 4 GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m TrajGaze_v2.training.stage1_temporal_ea1_ablation \\
        --ablation score_only \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/stage1_ea1_score_only \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/ea1_score_only \\
        --epochs 100 --lr 3e-4 --batch-size 4

    CUDA_VISIBLE_DEVICES=2 python -m TrajGaze_v2.training.stage1_temporal_ea1_ablation \\
        --ablation gaze_only \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/stage1_ea1_gaze_only \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/ea1_gaze_only \\
        --epochs 100 --lr 3e-4 --batch-size 4

    CUDA_VISIBLE_DEVICES=3 python -m TrajGaze_v2.training.stage1_temporal_ea1_ablation \\
        --ablation hand_only \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/stage1_ea1_hand_only \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/ea1_hand_only \\
        --epochs 100 --lr 3e-4 --batch-size 4

    CUDA_VISIBLE_DEVICES=4 python -m TrajGaze_v2.training.stage1_temporal_ea1_ablation \\
        --ablation no_spatial \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/stage1_ea1_no_spatial \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/ea1_no_spatial \\
        --epochs 100 --lr 3e-4 --batch-size 4

    CUDA_VISIBLE_DEVICES=5 python -m TrajGaze_v2.training.stage1_temporal_ea1_ablation \\
        --ablation no_temporal \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/stage1_ea1_no_temporal \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/ea1_no_temporal \\
        --epochs 100 --lr 3e-4 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGaze_v2.data.dataset_temporal import (
    StreamGazeStage1DatasetTemporal,
    collate_stage1_temporal,
)
from TrajGaze_v2.models.model_temporal import (
    TrajGazeV2Temporal,
    score_loss_temporal,
)
from TrajGaze_v2.models.decoders import traj_loss

# ── Stage-2 launcher config ────────────────────────────────────────────────────

STAGE2_SCRIPTS = {
    "score_only":  "TrajGazeMerge.training.train_merge_lora_temporal",
    "gaze_only":   "TrajGazeMerge.training.train_merge_lora_gaze_only",
    "hand_only":   "TrajGazeMerge.training.train_merge_lora_hand_only",
    "no_spatial":  "TrajGazeMerge.training.train_merge_lora_temporal",
    "no_temporal": "TrajGazeMerge.training.train_merge_lora_temporal",
}
STAGE2_PORTS = {
    "score_only":  29614,
    "gaze_only":   29615,
    "hand_only":   29616,
    "no_spatial":  29617,
    "no_temporal": 29618,
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation",
                   choices=["score_only", "gaze_only", "hand_only",
                            "no_spatial", "no_temporal"],
                   required=True,
                   help="Which EA1 ablation to run")
    p.add_argument("--output-dir",        required=True,
                   help="Stage-1 checkpoint output directory")
    p.add_argument("--stage2-output-dir", default=None,
                   help="Stage-2 output directory (omit to skip auto-launch)")
    p.add_argument("--n-frames",          type=int,   default=128)
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--batch-size",        type=int,   default=4)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--workers",           type=int,   default=2)
    p.add_argument("--log-every",         type=int,   default=10)
    p.add_argument("--save-every",        type=int,   default=10)
    p.add_argument("--n-vis-keyframes",   type=int,   default=16)
    p.add_argument("--resume",            type=str,   default=None)
    p.add_argument("--stage2-epochs",     type=int,   default=3)
    p.add_argument("--stage2-eval-every", type=int,   default=400)
    return p.parse_args()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(ablation: str, n_vis_keyframes: int) -> nn.Module:
    """
    Build the right model for each ablation.

    score_only / no_spatial / no_temporal → TrajGazeV2Temporal with EA1 flags
    gaze_only  → TrajGazeV2TemporalGazeOnly (1 gaze token, full 4-loss)
    hand_only  → TrajGazeV2TemporalHandOnly (3 hand tokens, full 4-loss)
    """
    if ablation == "gaze_only":
        from TrajGaze_v2.models.model_temporal_gaze_only import TrajGazeV2TemporalGazeOnly
        return TrajGazeV2TemporalGazeOnly(n_vis_keyframes=n_vis_keyframes)

    if ablation == "hand_only":
        from TrajGaze_v2.models.model_temporal_hand_only import TrajGazeV2TemporalHandOnly
        return TrajGazeV2TemporalHandOnly(n_vis_keyframes=n_vis_keyframes)

    # EA1-family: all use TrajGazeV2Temporal, differing by flags
    use_frame_score_branch = ablation != "no_temporal"
    model = TrajGazeV2Temporal(
        n_vis_keyframes=n_vis_keyframes,
        use_frame_score_branch=use_frame_score_branch,
    )
    # EA1 gate: freeze at 0 so InterFrameTransformer is bypassed in main path.
    # Frame-score branch (if active) still reads x_iframe as a side channel.
    with torch.no_grad():
        model.encoder.inter_frame_gate.fill_(0.0)
    model.encoder.inter_frame_gate.requires_grad_(False)
    return model


# ── Per-ablation forward pass ─────────────────────────────────────────────────

def forward_score_only(model: TrajGazeV2Temporal, batch: dict) -> dict[str, torch.Tensor]:
    """
    1.b — l_score_past + l_score_future only.
    Trajectory decoder and TrajScoreHead are skipped; their outputs are noisy
    without l_traj shaping the decoder attention, so we drop l_traj + l_score_traj.
    """
    past    = batch["past"]
    T_past  = batch["T_past"]
    T_f_max = int(batch["T_future"].max().item())
    T_p_max = int(T_past.max().item())
    device  = past["left_pos"].device
    B       = past["left_pos"].shape[0]

    query_emb = torch.zeros(B, model.query_encoder.d_model, device=device)

    visual_feat = None
    if (fps := batch.get("frame_paths")) is not None:
        past_paths  = [fps[i][:int(T_past[i].item())] for i in range(B)]
        visual_feat = model.visual_encoder(past_paths, T_p_max, device)

    past_scores, context, _ = model.encoder(past, query_emb, visual_feat)
    score_pred = model.score_decoder(context, T_f_max)

    l_score_past   = score_loss_temporal(past_scores, batch["I_scores_past"],   T_past)
    l_score_future = score_loss_temporal(score_pred,  batch["I_scores_future"], batch["T_future"])
    total = l_score_past + l_score_future

    return {
        "loss":            total,
        "loss_traj":       torch.tensor(0.0, device=device),
        "loss_score_past": l_score_past,
        "loss_score_fut":  l_score_future,
        "loss_score_traj": torch.tensor(0.0, device=device),
    }


def forward_full(model: nn.Module, batch: dict, no_visual: bool = False) -> dict[str, torch.Tensor]:
    """
    Standard 4-loss forward: l_traj + l_score_past + l_score_future + l_score_traj.
    Used by: gaze_only, hand_only, no_spatial (no_visual=True), no_temporal.

    When no_visual=True (3.a no_spatial), visual_feat is None → encoder falls
    back to _trajectory_only_scores(), enc_attn=None, l_score_traj=0.
    """
    past    = batch["past"]
    future  = batch["future"]
    T_past  = batch["T_past"]
    T_f_max = int(batch["T_future"].max().item())
    T_p_max = int(T_past.max().item())
    device  = past["left_pos"].device
    B       = past["left_pos"].shape[0]

    query_emb = torch.zeros(B, model.query_encoder.d_model, device=device)

    visual_feat = None
    if not no_visual and (fps := batch.get("frame_paths")) is not None:
        past_paths  = [fps[i][:int(T_past[i].item())] for i in range(B)]
        visual_feat = model.visual_encoder(past_paths, T_p_max, device)

    past_scores, context, enc_attn = model.encoder(past, query_emb, visual_feat)
    score_head_out = model.score_head(context)

    traj_pred, dec_attn = model.traj_decoder(context, T_f_max, return_cross_weights=True)
    score_pred = model.score_decoder(context, T_f_max)

    # l_score_traj: chain dec_attn × enc_attn (zero when visual unavailable)
    l_score_traj = torch.tensor(0.0, device=device)
    if enc_attn is not None:
        n_tok = context.shape[2]
        dec_importance = dec_attn.mean(dim=1).reshape(B, T_p_max, n_tok)
        traj_driven = (enc_attn * dec_importance.unsqueeze(-1)).sum(dim=2)
        t_max = traj_driven.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        traj_driven = (traj_driven / t_max).detach()
        l_score_traj = score_loss_temporal(score_head_out, traj_driven, batch["T_past"])

    l_traj         = traj_loss(traj_pred, future, batch["T_future"])
    l_score_future = score_loss_temporal(score_pred,  batch["I_scores_future"], batch["T_future"])
    l_score_past   = score_loss_temporal(past_scores, batch["I_scores_past"],   batch["T_past"])
    total = l_traj + l_score_future + l_score_past + l_score_traj

    return {
        "loss":            total,
        "loss_traj":       l_traj,
        "loss_score_past": l_score_past,
        "loss_score_fut":  l_score_future,
        "loss_score_traj": l_score_traj,
    }


# ── Stage-2 auto-launch ───────────────────────────────────────────────────────

def launch_stage2(ablation: str, stage1_ckpt: str, stage2_output_dir: str,
                  epochs: int, eval_every: int):
    os.makedirs(stage2_output_dir, exist_ok=True)
    script   = STAGE2_SCRIPTS[ablation]
    port     = STAGE2_PORTS[ablation]
    log_path = os.path.join(stage2_output_dir, "stage2_launch.log")

    cmd = [
        "/opt/conda/envs/gaze/bin/torchrun",
        f"--nproc_per_node=4",
        f"--master_port={port}",
        "-m", script,
        "--stage1-ckpt", stage1_ckpt,
        "--output-dir",  stage2_output_dir,
        "--epochs",      str(epochs),
        "--merge-ratio", "0.9",
        "--eval-every",  str(eval_every),
    ]

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd="/workspace/EgoGazeVQA", env=env,
        )

    print(f"\n[ea1_ablation/{ablation}] Stage-2 launched (PID={proc.pid})")
    print(f"  script:     {script}")
    print(f"  ckpt:       {stage1_ckpt}")
    print(f"  output_dir: {stage2_output_dir}")
    print(f"  log:        {log_path}", flush=True)


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda:0")
    no_visual = (args.ablation == "no_spatial")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[ea1_ablation] ablation={args.ablation}  n_frames={args.n_frames}")
    print(f"[ea1_ablation] output:   {args.output_dir}")
    print(f"[ea1_ablation] lr={args.lr}  bs={args.batch_size}", flush=True)

    dataset = StreamGazeStage1DatasetTemporal(n_frames=args.n_frames)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collate_stage1_temporal,
        num_workers = args.workers,
        pin_memory  = True,
        drop_last   = True,
    )

    model     = build_model(args.ablation, args.n_vis_keyframes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ea1_ablation] params: {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable",
          flush=True)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[ea1_ablation] Resumed from epoch {start_epoch}", flush=True)

    log_path  = os.path.join(args.output_dir, "train_log.jsonl")
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals    = dict(loss=0., traj=0., sp=0., sf=0., st=0.)
        n_batches = 0
        t_start   = time.time()

        for step, batch in enumerate(loader):
            if batch is None:
                continue

            def to_dev(d):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in d.items()}

            batch["past"]            = to_dev(batch["past"])
            batch["future"]          = to_dev(batch["future"])
            batch["I_scores_past"]   = batch["I_scores_past"].to(device)
            batch["I_scores_future"] = batch["I_scores_future"].to(device)
            batch["T_past"]          = batch["T_past"].to(device)
            batch["T_future"]        = batch["T_future"].to(device)

            optimizer.zero_grad()

            if args.ablation == "score_only":
                loss_dict = forward_score_only(model, batch)
            else:
                loss_dict = forward_full(model, batch, no_visual=no_visual)

            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss at step {step}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["traj"] += loss_dict["loss_traj"].item()
            totals["sp"]   += loss_dict["loss_score_past"].item()
            totals["sf"]   += loss_dict["loss_score_fut"].item()
            totals["st"]   += loss_dict["loss_score_traj"].item()
            n_batches += 1

            if (step + 1) % args.log_every == 0:
                n = max(1, n_batches)
                print(
                    f"Epoch {epoch+1:3d} | step {step+1:4d} | "
                    f"loss={totals['loss']/n:.4f} "
                    f"traj={totals['traj']/n:.4f} "
                    f"sp={totals['sp']/n:.4f} "
                    f"sf={totals['sf']/n:.4f} "
                    f"st={totals['st']/n:.4f} "
                    f"| t={time.time()-t_start:.1f}s",
                    flush=True,
                )

        scheduler.step()
        n          = max(1, n_batches)
        epoch_loss = totals["loss"] / n

        print(
            f"\n=== Epoch {epoch+1}/{args.epochs} [{args.ablation}] | "
            f"avg_loss={epoch_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={time.time()-t_start:.1f}s ===\n",
            flush=True,
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch":           epoch + 1,
                "loss":            epoch_loss,
                "loss_traj":       totals["traj"] / n,
                "loss_score_past": totals["sp"]   / n,
                "loss_score_fut":  totals["sf"]   / n,
                "loss_score_traj": totals["st"]   / n,
                "lr":              scheduler.get_last_lr()[0],
            }) + "\n")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss":      epoch_loss,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}", flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {"model": model.state_dict(), "loss": best_loss},
                os.path.join(args.output_dir, "best.pth"),
            )

    torch.save(
        {"model": model.state_dict()},
        os.path.join(args.output_dir, "final.pth"),
    )
    print(f"\n[ea1_ablation/{args.ablation}] Done. Best loss: {best_loss:.4f}", flush=True)

    if args.stage2_output_dir:
        launch_stage2(
            ablation        = args.ablation,
            stage1_ckpt     = os.path.join(args.output_dir, "best.pth"),
            stage2_output_dir = args.stage2_output_dir,
            epochs          = args.stage2_epochs,
            eval_every      = args.stage2_eval_every,
        )


if __name__ == "__main__":
    main()
