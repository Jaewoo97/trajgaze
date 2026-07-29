"""
AutoGaze + Qwen2.5-VL-7B LoRA fine-tuning on StreamGaze_v2.

Token selection: AutoGaze (frozen GRPO ckpt) selects ~10% of visual tokens (gazing_ratio=0.10).
Loss: CrossEntropy over 4 MCQ option logits at last prompt position.
Train: egoexolearn + holoassist (all MCQ)
Val:   egtea (full 526-item val set evaluated after every epoch)
GPUs:  4 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        -m TrajGazeMerge.training.train_autogaze_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/autogaze_lora \
        --epochs 3 --lr 1e-4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import traceback
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/AutoGaze_pub")  # public NVlabs/AutoGaze code (autogaze pkg)

from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item, forward_logits, build_merged_inputs
)
from TrajGazeMerge.models.merge import gaze_weighted_merge

# ── Constants ────────────────────────────────────────────────────────────────
AUTOGAZE_CKPT = (
    # Official public NVlabs/AutoGaze weights (nvidia/AutoGaze on HF). NOTE: this is the
    # authors' general pretrained gaze model, NOT the deleted project-internal
    # streamgaze_fold_c_grpo GRPO checkpoint — see table note.
    "/workspace/autogaze_weights/nvidia_AutoGaze"
)
# Env-driven like TrajGazeMerge/data/dataset.py — these were hardcoded to the
# machine-1 layout, so StreamGazeSimpleDataset silently yielded 0 items anywhere
# else (every qa_path missed and was skipped), which made CombinedSimpleDataset
# score EG-only while reporting it as the full set.
SG_ROOT       = os.environ.get("SG_ROOT", "/workspace/datasets/StreamGaze_v2")
# See TrajGazeMerge/data/dataset.py::_SG_FRAME_SUB — "viz" frames have the gaze
# marker burned into the pixels; GAZE_OVERLAY=0 selects the non-overlay "original".
#
# Two variants, split by consumer exactly as in data/dataset.py:
#   _SG_FRAME_SUB     → dataset RESOLUTION only (_find_dataset). The teacher variant
#                       is always present, so resolution never depends on whether the
#                       non-overlay frames were extracted for this split.
#   _SG_VLM_FRAME_SUB → vlm_frame_paths, i.e. what the VLM actually sees.
# This file used to read GAZE_OVERLAY for BOTH, so StreamGazeSimpleDataset — and
# therefore CombinedSimpleDataset and train_visionzip_lora.py — silently ignored
# VLM_GAZE_OVERLAY and fed the student overlay frames. Same class of silent failure
# the KD trainer's stream assertion exists to catch.
_SG_FRAME_SUB     = "viz" if os.environ.get("GAZE_OVERLAY", "1") == "1" else "original"
_SG_VLM_FRAME_SUB = ("viz" if os.environ.get(
                        "VLM_GAZE_OVERLAY", os.environ.get("GAZE_OVERLAY", "1")) == "1"
                     else "original")
FRAMES_BASE   = os.path.join(SG_ROOT, "frames")
QA_BASE       = os.path.join(SG_ROOT, "qa")
DATASETS      = ["egtea", "egoexolearn", "holoassist"]
EXTRACTED_FPS = 10.0
FRAME_SIZE    = 224
N_AG_FRAMES   = 16       # frames fed to AutoGaze
GAZING_RATIO  = 0.10
TASK_LOSS_REQ = 0.7
AG_SCALES     = [56, 112, 196, 224]
AG_PATCH_SIZE = 14        # Qwen patch_size = 14

MCQ_TASKS = [
    "past_gaze_sequence_matching",
    "past_non_fixated_object_identification",
    "past_object_transition_prediction",
    "past_scene_recall",
    "present_future_action_prediction",
    "present_object_attribute_recognition",
    "present_object_identification_easy",
    "present_object_identification_hard",
]


# ── Lightweight dataset (no traj loading) ────────────────────────────────────

def _parse_ts(ts: str) -> float:
    if not ts:
        return 9999.0
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def _find_dataset(stem: str) -> Optional[str]:
    for ds in DATASETS:
        if os.path.isdir(os.path.join(FRAMES_BASE, ds, _SG_FRAME_SUB, stem)):
            return ds
    return None


def _get_frame_paths(stem: str, dataset: str, ts_sec: float) -> list[str]:
    # VLM variant: this list becomes vlm_frame_paths, the model's actual input.
    frame_dir = os.path.join(FRAMES_BASE, dataset, _SG_VLM_FRAME_SUB, stem)
    if not os.path.isdir(frame_dir):
        return []
    cutoff = max(1, int(ts_sec * EXTRACTED_FPS))
    paths = []
    for fname in sorted(os.listdir(frame_dir)):
        m = re.match(r"frame_(\d+)\.jpg", fname)
        if m and int(m.group(1)) <= cutoff:
            paths.append(os.path.join(frame_dir, fname))
    return paths


def _sample_paths(paths: list[str], n: int) -> list[str]:
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    indices = [int(i * len(paths) / n) for i in range(n)]
    return [paths[i] for i in indices]


class StreamGazeSimpleDataset(Dataset):
    """
    Lightweight StreamGaze_v2 dataset — frame paths + QA only, no traj loading.
    split='train' → egoexolearn + holoassist
    split='test'  → egtea
    """

    def __init__(self, split: str = "train", n_vlm_frames: int = 128):
        filter_ds = ["egoexolearn", "holoassist"] if split == "train" else ["egtea"]
        self.n_vlm_frames = n_vlm_frames
        self.items: list[dict] = []

        for task in MCQ_TASKS:
            qa_path = os.path.join(QA_BASE, f"{task}.json")
            if not os.path.exists(qa_path):
                continue
            with open(qa_path) as f:
                qa_data = json.load(f)
            for entry in qa_data:
                stem = os.path.splitext(entry["video_path"])[0]
                ds = _find_dataset(stem)
                if ds not in filter_ds:
                    continue
                for q in entry.get("questions", []):
                    if not q.get("options"):
                        continue
                    self.items.append({
                        "stem":       stem,
                        "dataset":    ds,
                        "question":   q["question"],
                        "options":    q["options"],
                        "answer":     q["answer"].strip().upper(),
                        "time_stamp": q.get("time_stamp", ""),
                        "task":       task,
                    })

        ds_label = "+".join(filter_ds)
        print(f"[StreamGazeSimpleDataset] split={split} ({ds_label}) "
              f"→ {len(self.items)} items")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[dict]:
        item = self.items[idx]
        ts_sec = _parse_ts(item["time_stamp"])
        frame_paths = _get_frame_paths(item["stem"], item["dataset"], ts_sec)
        if not frame_paths:
            return None
        vlm_paths = _sample_paths(frame_paths, self.n_vlm_frames)
        if not vlm_paths:
            return None
        return {
            "vlm_frame_paths": vlm_paths,
            "question":        item["question"],
            "options":         item["options"],
            "answer":          item["answer"],
            "task":            item["task"],
            "dataset":         item["dataset"],
        }


# ── AutoGaze helpers ─────────────────────────────────────────────────────────

def load_autogaze(device: torch.device):
    from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze as _AutoGaze
    ag_transform = AutoGazeImageProcessor.from_pretrained(
        AUTOGAZE_CKPT, size=(FRAME_SIZE, FRAME_SIZE)
    )
    ag_model = _AutoGaze.from_pretrained(AUTOGAZE_CKPT, use_flash_attn=False)
    ag_model = ag_model.to(device).eval()
    for p in ag_model.parameters():
        p.requires_grad_(False)
    return ag_transform, ag_model


def compute_ag_scores(
    ag_frames_pil: list,
    ag_model,
    ag_transform,
    T_merged: int,
    n_spatial: int,
    grid_hw: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Run AutoGaze and return continuous importance scores for all Qwen visual tokens.

    Uses only the finest-scale mask (AG_SCALES[-1]=224 → 16×16=256 tokens/frame)
    averaged across frames, then interpolated to Qwen's spatial grid.
    Keeps exactly (1-GAZING_RATIO) fraction of tokens via gaze_weighted_merge.

    Returns:
        scores_all : (T_merged * n_spatial,) float on CPU
    """
    out = ag_transform(list(ag_frames_pil))
    imgs = out.pixel_values
    inner = imgs[0] if isinstance(imgs[0], list) else imgs
    if isinstance(inner[0], torch.Tensor):
        vt = torch.stack(inner)
    else:
        vt = torch.from_numpy(np.stack(inner))
    if vt.dim() == 5:
        vt = vt.squeeze(0)
    vt = vt.float().unsqueeze(0).to(ag_model.device)   # (1, T=16, C, H, W)

    with torch.no_grad():
        gaze_out = ag_model(
            {"video": vt},
            gazing_ratio=GAZING_RATIO,
            task_loss_requirement=TASK_LOSS_REQ,
            target_scales=AG_SCALES,
            target_patch_size=AG_PATCH_SIZE,
        )

    # Finest scale: AG_SCALES[-1]=224 → 16×16=256 tokens per frame
    mask_fine = gaze_out["gazing_mask"][-1]            # (1, T=16, 256)
    scores_fine = mask_fine.float().squeeze(0).mean(dim=0).cpu()  # (256,)

    # Map 16×16 AutoGaze saliency → Qwen's spatial token grid.
    # Qwen grids are generally NON-square (e.g. 12×18=216 after 2×2 merge), so we
    # must resize to the actual (h_q, w_q); assuming a square side over-/under-counts.
    s16 = scores_fine.reshape(1, 1, 16, 16)
    if grid_hw is not None and grid_hw[0] * grid_hw[1] == n_spatial:
        h_q, w_q = int(grid_hw[0]), int(grid_hw[1])
        scores_q = F.interpolate(
            s16, size=(h_q, w_q), mode="bilinear", align_corners=False
        ).flatten()
    else:
        # Fallback (no grid given / mismatch): 1-D resize the flat saliency to
        # exactly n_spatial so the token count always matches.
        scores_q = F.interpolate(
            scores_fine.reshape(1, 1, -1), size=int(n_spatial),
            mode="linear", align_corners=False,
        ).flatten()

    # Tile across T_merged frames → (T_merged * n_spatial,)
    scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
    return scores_all


# ── Training helpers ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",    default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/autogaze_lora")
    p.add_argument("--autogaze-ckpt", default=AUTOGAZE_CKPT)
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--log-every",     type=int,   default=20)
    p.add_argument("--eval-every",    type=int,   default=200)
    p.add_argument("--n-frames",      type=int,   default=128)
    p.add_argument("--start-epoch",   type=int,   default=0,
                   help="Resume from this epoch (0=start fresh). Loads epoch_XX.pth.")
    p.add_argument("--resume-ckpt",   type=str,   default=None,
                   help="Path to LoRA checkpoint to resume from.")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def evaluate(processor, model, base_qwen, ag_model, ag_transform,
             option_ids, device, max_items=None):
    """Evaluate AutoGaze+LoRA on full egtea test split; returns (accuracy, n_samples, per_task)."""
    from PIL import Image
    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    if max_items is not None:
        test_ds.items = test_ds.items[:max_items]

    model.eval()
    correct  = 0
    total    = 0
    by_task: dict[str, list] = {}

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

                ag_paths  = _sample_paths(item["vlm_frame_paths"], N_AG_FRAMES)
                ag_frames = [
                    Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
                    for p in ag_paths
                ]
                scores_all = compute_ag_scores(
                    ag_frames, ag_model, ag_transform, T_merged, n_spatial
                )
                r = max(1, int(0.90 * n_video))
                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all.to(device), r
                )
                inputs_dict = build_merged_inputs(
                    base_qwen, cached, merged_video, receiver_idx
                )
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)
            except Exception:
                pass

    model.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[AutoGaze LoRA] output: {args.output_dir}")
        print(f"[AutoGaze LoRA] GPUs={world_size}, epochs={args.epochs}, "
              f"lr={args.lr}, grad_accum={args.grad_accum}, "
              f"gazing_ratio={GAZING_RATIO}")

    # ── Load AutoGaze (frozen, single copy per rank) ──────────────────────────
    if is_main:
        print("Loading AutoGaze ...")
    ag_transform, ag_model = load_autogaze(device)
    if is_main:
        print("AutoGaze loaded.")

    # ── Load Qwen + LoRA ──────────────────────────────────────────────────────
    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen = model.get_base_model()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor)
    if is_main:
        print("Qwen loaded.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = StreamGazeSimpleDataset(split="train", n_vlm_frames=args.n_frames)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size,
                                  rank=rank, shuffle=True)
    loader   = DataLoader(train_ds, batch_size=1, sampler=sampler,
                          collate_fn=lambda b: b[0], num_workers=2)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    resume_path = args.resume_ckpt
    if resume_path is None and args.start_epoch > 0:
        resume_path = os.path.join(args.output_dir, f"epoch_{args.start_epoch:02d}.pth")
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        state = ckpt.get("lora_state", ckpt)
        missing, unexpected = model.module.load_state_dict(state, strict=False)
        if is_main:
            print(f"Resumed from {resume_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ── Optimizer: LoRA params only ───────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path    = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc    = 0.0
    global_step = 0

    for epoch in range(args.start_epoch, args.epochs):
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
                from PIL import Image as _PIL_Image

                # ── Preprocess (ViT forward, no grad) ────────────────────────
                with torch.no_grad():
                    cached = preprocess_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"],
                        item["options"], device
                    )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)

                # ── AutoGaze scores → weighted merge (no grad) ───────────────
                with torch.no_grad():
                    ag_paths  = _sample_paths(item["vlm_frame_paths"], N_AG_FRAMES)
                    ag_frames = [
                        _PIL_Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
                        for p in ag_paths
                    ]
                    scores_all = compute_ag_scores(
                        ag_frames, ag_model, ag_transform, T_merged, n_spatial
                    )
                    r = max(1, int(0.90 * n_video))
                    merged_video, receiver_idx = gaze_weighted_merge(
                        cached["video_embeds"], scores_all.to(device), r
                    )
                    inputs_dict = build_merged_inputs(
                        base_qwen, cached, merged_video, receiver_idx
                    )

                # ── Forward through LoRA model ────────────────────────────────
                logits = forward_logits(model, inputs_dict)   # (vocab_size,)

                # ── MCQ CE loss ───────────────────────────────────────────────
                option_logits = logits[option_ids]            # (4,)
                gt_idx  = ["A", "B", "C", "D"].index(item["answer"])
                loss    = F.cross_entropy(
                    option_logits.unsqueeze(0),
                    torch.tensor([gt_idx], device=device),
                )
                loss = loss / args.grad_accum
                loss.backward()

                epoch_loss  += loss.item() * args.grad_accum
                n_steps     += 1
                global_step += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed  = time.time() - t_start
                    n_kept   = n_video - r
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")

                pass  # epoch-end eval only

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Flush remaining gradients at end of epoch
        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed  = time.time() - t_start
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")
            torch.save({
                "epoch": epoch,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        # ── Full val-set evaluation after every epoch (rank 0 only) ──────────
        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on full EGTEA val set ...")
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen,
                ag_model, ag_transform, option_ids, device,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})")
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch,
                    "lora_state": model.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
