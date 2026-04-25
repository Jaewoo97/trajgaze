"""
AutoGaze + Qwen2.5-VL-7B LoRA fine-tuning on StreamGaze_v2.

Token selection: AutoGaze selects ~10% of visual tokens (gazing_ratio=0.10).
Loss: CrossEntropy over 4 MCQ option logits at last prompt position.
Train: egoexolearn + holoassist
Val:   egtea
GPUs:  2 via torchrun

Usage:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
        -m TrajGazeMerge.training.train_autogaze_lora \
        --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/autogaze_lora \
        --epochs 3 --lr 1e-4 --grad-accum 4
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item, forward_logits
)

# ── Constants ────────────────────────────────────────────────────────────────
AUTOGAZE_CKPT = (
    "/workspace/EgoGazeVQA/AutoGaze/exps/streamgaze_fold_c_ntp/checkpoint_latest_gaze"
)
FRAMES_BASE   = "/workspace/datasets/StreamGaze_v2/frames"
QA_BASE       = "/workspace/datasets/StreamGaze_v2/qa"
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
        if os.path.isdir(os.path.join(FRAMES_BASE, ds, "viz", stem)):
            return ds
    return None


def _get_frame_paths(stem: str, dataset: str, ts_sec: float) -> list[str]:
    frame_dir = os.path.join(FRAMES_BASE, dataset, "viz", stem)
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


def compute_keep_mask(
    ag_frames_pil: list,
    ag_model,
    ag_transform,
    T_merged: int,
    n_spatial: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run AutoGaze on ag_frames_pil (16 PIL images) and produce a boolean keep_mask
    of shape (T_merged * n_spatial,) aligned to Qwen's merged video token sequence.

    Args:
        ag_frames_pil : 16 PIL images (224×224)
        T_merged      : temporal dimension of Qwen video token grid (grid_thw[0,0])
        n_spatial     : spatial tokens per timestep after Qwen merge (grid_thw h*w / 4)
    Returns:
        keep_mask : (T_merged * n_spatial,) bool on device
    """
    # Process frames through AutoGaze image processor
    out  = ag_transform(list(ag_frames_pil))
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

    T      = len(ag_frames_pil)     # 16
    H_q    = W_q = FRAME_SIZE // AG_PATCH_SIZE   # 224/14 = 16

    scale_grids = [
        (0, 4,  4),    # scale=56  → 4×4
        (1, 8,  8),    # scale=112 → 8×8
        (2, 14, 14),   # scale=196 → 14×14
        (3, 16, 16),   # scale=224 → 16×16
    ]

    full_mask = torch.zeros(T, H_q, W_q, dtype=torch.bool)
    for scale_idx, H_s, W_s in scale_grids:
        m = gaze_out["gazing_mask"][scale_idx][0].bool().cpu()   # (T, H_s*W_s)
        m2d = m.view(T, H_s, W_s)
        if H_s == H_q:
            up = m2d
        elif H_q % H_s == 0:
            up = (m2d.repeat_interleave(H_q // H_s, dim=1)
                     .repeat_interleave(W_q // W_s, dim=2))
        else:
            up = F.interpolate(
                m2d.float().unsqueeze(0), size=(H_q, W_q), mode="nearest"
            ).squeeze(0).bool()
        full_mask |= up   # (T, 16, 16)

    # Qwen temporal_patch_size=2: merge pairs of frames
    gaze_t = full_mask[0::2] | full_mask[1::2]   # (T//2, 16, 16)  = (8, 16, 16)

    # Qwen spatial_merge_size=2: 2×2 max pool → 8×8
    gaze_s = F.max_pool2d(
        gaze_t.float().unsqueeze(0), kernel_size=2, stride=2
    ).squeeze(0).bool()   # (8, 8, 8)

    T_ag = gaze_s.shape[0]   # 8 for N_AG_FRAMES=16

    # Tile temporally to match T_merged from VLM
    if T_merged == T_ag:
        tiled = gaze_s
    elif T_merged % T_ag == 0:
        tiled = gaze_s.repeat_interleave(T_merged // T_ag, dim=0)
    else:
        reps  = (T_merged + T_ag - 1) // T_ag
        tiled = gaze_s.repeat(reps, 1, 1)[:T_merged]   # (T_merged, 8, 8)

    # Flatten and handle n_spatial != 64 edge case
    keep_flat = tiled.flatten()   # (T_merged * 64,)
    n_video   = T_merged * n_spatial
    if keep_flat.shape[0] > n_video:
        keep_flat = keep_flat[:n_video]
    elif keep_flat.shape[0] < n_video:
        reps = (n_video + keep_flat.shape[0] - 1) // keep_flat.shape[0]
        keep_flat = keep_flat.repeat(reps)[:n_video]

    # Guarantee at least one token is kept
    if not keep_flat.any():
        keep_flat[0] = True

    return keep_flat.to(device)


def build_autogaze_inputs(base_qwen, cached: dict, keep_mask: torch.Tensor) -> dict:
    """
    Build shortened input sequence keeping only AutoGaze-selected visual tokens.

    Args:
        keep_mask : (n_video,) bool — True = keep token
    Returns:
        inputs_embeds, attention_mask, position_ids, rope_deltas
    """
    input_ids       = cached["input_ids"]
    attention_mask  = cached["attention_mask"]
    position_ids    = cached["position_ids"]
    rope_deltas     = cached["rope_deltas"]
    video_embeds    = cached["video_embeds"]     # (n_video, d)
    video_positions = cached["video_positions"]  # (n_video,) positions in full seq
    emb_dev         = cached["emb_dev"]

    # Positions of dropped video tokens in full sequence
    dropped_pos = video_positions[~keep_mask]
    keep_seq    = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[dropped_pos] = False

    new_input_ids      = input_ids[:, keep_seq]
    new_attention_mask = attention_mask[:, keep_seq]
    new_position_ids   = position_ids[:, :, keep_seq]

    new_inputs_embeds  = base_qwen.get_input_embeddings()(new_input_ids)

    # Inject selected video embeddings
    video_embeds_kept = video_embeds[keep_mask]
    video_token_id    = base_qwen.config.video_token_id
    new_is_video      = (new_input_ids[0] == video_token_id)
    new_inputs_embeds[0, new_is_video] = video_embeds_kept.to(new_inputs_embeds.dtype)

    return {
        "inputs_embeds":  new_inputs_embeds,
        "attention_mask": new_attention_mask,
        "position_ids":   new_position_ids,
        "rope_deltas":    rope_deltas,
    }


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
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def evaluate(processor, model, base_qwen, ag_model, ag_transform,
             option_ids, device, max_items=200):
    """Evaluate AutoGaze+LoRA on egtea test split; returns accuracy."""
    from PIL import Image
    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    test_ds.items = test_ds.items[:max_items]

    model.eval()
    correct = 0
    total   = 0

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
                keep_mask = compute_keep_mask(
                    ag_frames, ag_model, ag_transform, T_merged, n_spatial, device
                )

                inputs_dict   = build_autogaze_inputs(base_qwen, cached, keep_mask)
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                correct      += int(pred_idx == gt_idx)
                total        += 1
            except Exception:
                pass

    model.train()
    return 100.0 * correct / max(1, total), total


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

    # ── Optimizer: LoRA params only ───────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    log_path    = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc    = 0.0
    global_step = 0

    for epoch in range(args.epochs):
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

                # ── AutoGaze keep_mask (no grad) ──────────────────────────────
                with torch.no_grad():
                    ag_paths  = _sample_paths(item["vlm_frame_paths"], N_AG_FRAMES)
                    ag_frames = [
                        _PIL_Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
                        for p in ag_paths
                    ]
                    keep_mask = compute_keep_mask(
                        ag_frames, ag_model, ag_transform, T_merged, n_spatial, device
                    )

                # ── Build filtered inputs (no grad for embed lookup) ──────────
                with torch.no_grad():
                    inputs_dict = build_autogaze_inputs(base_qwen, cached, keep_mask)

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
                    n_kept   = int(keep_mask.sum().item())
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | t={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept, "elapsed": elapsed,
                        }) + "\n")

                if is_main and n_steps % args.eval_every == 0:
                    acc, n_eval = evaluate(
                        processor, model.module, base_qwen,
                        ag_model, ag_transform, option_ids, device
                    )
                    print(f"  → eval egtea: {acc:.2f}% (n={n_eval})")
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({
                            "epoch": epoch, "step": n_steps,
                            "lora_state": model.module.state_dict(),
                            "acc": acc,
                        }, os.path.join(args.output_dir, "best.pth"))
                        print(f"  → saved best (acc={acc:.2f}%)")

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

        # Flush remaining gradients at end of epoch
        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_loss = epoch_loss / max(1, n_steps)
            elapsed  = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s ===")
            torch.save({
                "epoch": epoch,
                "lora_state": model.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

    # Final evaluation
    if is_main:
        acc, n_eval = evaluate(
            processor, model.module, base_qwen,
            ag_model, ag_transform, option_ids, device, max_items=500
        )
        print(f"\n[Final] egtea accuracy: {acc:.2f}% (n={n_eval})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
