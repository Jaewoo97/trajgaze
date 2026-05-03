"""
E1_patch_temporal ablation training — single-GPU stage-1.

Five ablations of the E1_patch_temporal model
(SpatiotemporalEncoderTemporal with use_patch_temporal_branch=True, gate=0 frozen):

  score_only   [1.b] Drop l_traj + l_score_traj; train with l_score_past +
                     l_score_future only. Tests whether trajectory prediction
                     loss is necessary.

  gaze_only    [2.b] 1-token gaze encoder + patch_temporal_branch. All 4 losses.
                     Tests whether hand modality contributes.

  hand_only    [2.a] 3-token hand encoder + patch_temporal_branch. All 4 losses.
                     Tests whether gaze modality contributes.

  no_spatial   [3.a] E1 encoder, visual_feat=None → trajectory-only score path
                     (no TemporalVisualTrajFusion). patch_temporal_branch still
                     applies. Tests spatial grounding contribution.

  no_temporal  [3.b] Full encoder, gate=0 frozen, NO patch_temporal_branch.
                     No temporal signal in any path. Tests temporal contribution.

After stage-1, auto-launches stage-2 on 4 GPUs via train_merge_lora_temporal_no_kd.

Usage:
    # 1.b — without trajectory prediction loss
    CUDA_VISIBLE_DEVICES=1 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \\
        --ablation score_only \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/e1_score_only \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/e1_score_only

    # 2.b — gaze-only modality
    CUDA_VISIBLE_DEVICES=2 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \\
        --ablation gaze_only \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/e1_gaze_only \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/e1_gaze_only

    # 2.a — hand-only modality
    CUDA_VISIBLE_DEVICES=3 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \\
        --ablation hand_only \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/e1_hand_only \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/e1_hand_only

    # 3.a — no spatial (no TemporalVisualTrajFusion)
    CUDA_VISIBLE_DEVICES=4 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \\
        --ablation no_spatial \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/e1_no_spatial \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/e1_no_spatial

    # 3.b — no temporal (no patch_temporal_branch, gate=0 frozen)
    CUDA_VISIBLE_DEVICES=5 python -m TrajGaze_v2.training.stage1_temporal_e1_ablation \\
        --ablation no_temporal \\
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGaze_v2/checkpoints/e1_no_temporal \\
        --stage2-output-dir /workspace/EgoGazeVQA/TrajGazeMerge_v2/TrajGazeMerge/checkpoints/e1_no_temporal
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

sys.path.insert(0, "/workspace/EgoGazeVQA/TrajGazeMerge_v2")

from TrajGaze_v2.data.dataset_temporal import (
    StreamGazeStage1DatasetTemporal,
    collate_stage1_temporal,
)
from TrajGaze_v2.models.model_temporal import (
    TrajGazeV2Temporal,
    score_loss_temporal,
)
from TrajGaze_v2.models.decoders import traj_loss

# ── Stage-2 config ─────────────────────────────────────────────────────────────
# All ablations use train_merge_lora_temporal_no_kd (no KD teacher, CE only).
# gaze/hand use their dedicated --model-type for the correct encoder at stage-2.

STAGE2_MODEL_TYPE = {
    "score_only":  "full",
    "gaze_only":   "gaze_only",
    "hand_only":   "hand_only",
    "no_spatial":  "full",
    "no_temporal": "full",
}
STAGE2_PORTS = {
    "score_only":  29620,
    "gaze_only":   29621,
    "hand_only":   29622,
    "no_spatial":  29623,
    "no_temporal": 29624,
}
STAGE2_SCRIPT = "TrajGazeMerge.training.train_merge_lora_temporal_no_kd"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation",
                   choices=["score_only", "gaze_only", "hand_only",
                            "no_spatial", "no_temporal"],
                   required=True)
    p.add_argument("--output-dir",        required=True)
    p.add_argument("--stage2-output-dir", default=None)
    p.add_argument("--n-frames",          type=int,   default=128)
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--batch-size",        type=int,   default=2)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--workers",           type=int,   default=2)
    p.add_argument("--log-every",         type=int,   default=10)
    p.add_argument("--save-every",        type=int,   default=10)
    p.add_argument("--n-vis-keyframes",   type=int,   default=16)
    p.add_argument("--resume",            type=str,   default=None)
    p.add_argument("--stage2-epochs",     type=int,   default=3)
    p.add_argument("--stage2-eval-every", type=int,   default=400)
    p.add_argument("--grad-accum",        type=int,   default=4,
                   help="Gradient accumulation steps for stage-2 launch")
    return p.parse_args()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(ablation: str, n_vis_keyframes: int) -> nn.Module:
    """
    All ablations use use_patch_temporal_branch=True EXCEPT no_temporal (3.b),
    which tests the model with zero temporal signal (no branch, gate=0 frozen).

    gaze_only / hand_only use their dedicated model classes so the encoder
    architecture matches the modality. The patch_temporal_branch adapts
    automatically to N_TOKENS=1 (gaze) or N_TOKENS=3 (hand).
    """
    if ablation == "gaze_only":
        from TrajGaze_v2.models.model_temporal_gaze_only import TrajGazeV2TemporalGazeOnly
        return TrajGazeV2TemporalGazeOnly(
            n_vis_keyframes=n_vis_keyframes,
            use_patch_temporal_branch=True,
        )

    if ablation == "hand_only":
        from TrajGaze_v2.models.model_temporal_hand_only import TrajGazeV2TemporalHandOnly
        return TrajGazeV2TemporalHandOnly(
            n_vis_keyframes=n_vis_keyframes,
            use_patch_temporal_branch=True,
        )

    # Full E1 family (score_only / no_spatial / no_temporal)
    use_branch = (ablation != "no_temporal")
    model = TrajGazeV2Temporal(
        n_vis_keyframes=n_vis_keyframes,
        use_patch_temporal_branch=use_branch,
    )
    # Freeze gate at 0: InterFrameTransformer bypassed in main path;
    # patch_temporal_branch (when active) reads x_iframe as a clean side-channel.
    with torch.no_grad():
        model.encoder.inter_frame_gate.fill_(0.0)
    model.encoder.inter_frame_gate.requires_grad_(False)
    return model


# ── Forward passes ─────────────────────────────────────────────────────────────

def forward_score_only(model: TrajGazeV2Temporal, batch: dict) -> dict:
    """
    1.b — l_score_past + l_score_future only.
    Skips trajectory decoder (l_traj + l_score_traj) because those losses are
    noisy without a trained decoder shaping the attention.
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

    return {
        "loss":            l_score_past + l_score_future,
        "loss_traj":       torch.tensor(0.0, device=device),
        "loss_score_past": l_score_past,
        "loss_score_fut":  l_score_future,
        "loss_score_traj": torch.tensor(0.0, device=device),
    }


def forward_full(model: nn.Module, batch: dict, no_visual: bool = False) -> dict:
    """
    Standard 4-loss forward used by: gaze_only, hand_only, no_spatial, no_temporal.

    no_visual=True (3.a no_spatial): sets visual_feat=None so encoder falls back
    to _trajectory_only_scores(). enc_attn=None → l_score_traj=0.
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

    return {
        "loss":            l_traj + l_score_future + l_score_past + l_score_traj,
        "loss_traj":       l_traj,
        "loss_score_past": l_score_past,
        "loss_score_fut":  l_score_future,
        "loss_score_traj": l_score_traj,
    }


# ── Stage-2 auto-launch ────────────────────────────────────────────────────────

def launch_stage2(ablation: str, stage1_ckpt: str, stage2_output_dir: str,
                  epochs: int, eval_every: int, grad_accum: int):
    os.makedirs(stage2_output_dir, exist_ok=True)
    port       = STAGE2_PORTS[ablation]
    model_type = STAGE2_MODEL_TYPE[ablation]
    log_path   = os.path.join(stage2_output_dir, "stage2_launch.log")

    cmd = [
        "/opt/conda/envs/gaze/bin/python",
        "-m", STAGE2_SCRIPT,
        "--model-type",  model_type,
        "--stage1-ckpt", stage1_ckpt,
        "--output-dir",  stage2_output_dir,
        "--epochs",      str(epochs),
        "--merge-ratio", "0.9",
        "--grad-accum",  str(grad_accum),
        "--eval-every",  str(eval_every),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/EgoGazeVQA/TrajGazeMerge_v2"
    env.pop("CUDA_VISIBLE_DEVICES", None)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd="/workspace/EgoGazeVQA/TrajGazeMerge_v2", env=env,
        )

    print(f"\n[e1_ablation/{ablation}] Stage-2 launched (PID={proc.pid})")
    print(f"  model_type: {model_type}")
    print(f"  ckpt:       {stage1_ckpt}")
    print(f"  output_dir: {stage2_output_dir}")
    print(f"  log:        {log_path}", flush=True)


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    device    = torch.device("cuda:0")
    no_visual = (args.ablation == "no_spatial")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[e1_ablation] ablation={args.ablation}  n_frames={args.n_frames}")
    print(f"[e1_ablation] output: {args.output_dir}")
    print(f"[e1_ablation] lr={args.lr}  bs={args.batch_size}", flush=True)

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

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[e1_ablation] params: {n_total/1e6:.2f}M total, {n_train/1e6:.2f}M trainable",
          flush=True)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[e1_ablation] Resumed from epoch {start_epoch}", flush=True)

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

    torch.save({"model": model.state_dict()},
               os.path.join(args.output_dir, "final.pth"))
    print(f"\n[e1_ablation/{args.ablation}] Done. Best loss: {best_loss:.4f}", flush=True)

    if args.stage2_output_dir:
        launch_stage2(
            ablation          = args.ablation,
            stage1_ckpt       = os.path.join(args.output_dir, "best.pth"),
            stage2_output_dir = args.stage2_output_dir,
            epochs            = args.stage2_epochs,
            eval_every        = args.stage2_eval_every,
            grad_accum        = args.grad_accum,
        )


if __name__ == "__main__":
    main()
