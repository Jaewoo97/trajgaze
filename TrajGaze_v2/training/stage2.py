"""
TrajGaze_v2 Stage 2 GRPO Training.

Policy: TrajGazeV2 visual-trajectory attention → patch_scores (B, 196) → top-K mask
Reward: binary correctness from frozen Qwen2.5-VL-7B

For each QA sample:
  1. Run TrajGazeV2(traj, visual, query) → patch_scores (196,) with grad
  2. Sample G masks with Gumbel noise from patch_scores
  3. For each mask: filter Qwen visual tokens → run frozen Qwen → reward ∈ {0, 1}
  4. GRPO policy gradient:
       advantage_g = reward_g - mean(rewards)
       surrogate_g = sum_{i ∈ selected_g} patch_scores[i]  ← differentiable
       pg_loss = -1/G * sum_g( advantage_g * surrogate_g )
       ∇_scores pg_loss = -1/G * sum_g( advantage_g * mask_g )
       → selected patches in high-reward groups get higher scores next step

Key fix over previous version:
  - patch_scores_raw has shape (B, 196) from encoder, NOT (B, T, 196).
    Previous code did .mean(dim=1) which collapsed 196→scalar, making
    surrogate = constant for all groups → pg_loss = -constant * Σadvantages = 0.
    Fixed: scores_agg = patch_scores_raw.squeeze(0)  # (196,)

Usage:
    torchrun --nproc_per_node=4 -m TrajGaze_v2.training.stage2 \\
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \\
        --output-dir  /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage2_v2 \\
        --epochs 5 \\
        --lr 1e-5 \\
        --group-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGaze_v2.data.dataset import TrajGazeV2QADataset
from TrajGaze_v2.models.model import TrajGazeV2, N_KEEP, N_PATCHES

QWEN_MODEL_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
FRAME_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt",
                   default="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth")
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage2_v2")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--group-size",   type=int,   default=8,
                   help="GRPO group size: number of Gumbel-sampled masks per QA item")
    p.add_argument("--gumbel-temp",        type=float, default=1.0,
                   help="Initial Gumbel temperature (higher = more exploration)")
    p.add_argument("--gumbel-temp-final",  type=float, default=0.2,
                   help="Final Gumbel temperature after linear decay")
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--log-every",    type=int,   default=5)
    p.add_argument("--save-every",   type=int,   default=1)
    p.add_argument("--n-frames",     type=int,   default=32,
                   help="Frames for trajectory encoder")
    p.add_argument("--n-qwen-frames", type=int,  default=64,
                   help="Frames loaded for Qwen VLM")
    p.add_argument("--datasets",     nargs="+",
                   default=["ego4d", "egoexo"],
                   help="Datasets for GRPO training. EGTEA is held out for evaluation.")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def load_qwen(device):
    """Load frozen Qwen2.5-VL-7B-Instruct."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def load_frames_pil(frame_paths: list[str], n_frames: int, size: int = FRAME_SIZE) -> list:
    """Load n_frames uniformly sampled from paths as PIL images."""
    from PIL import Image
    n_total = len(frame_paths)
    if n_total == 0:
        return []
    indices = [int(i * n_total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        try:
            img = Image.open(frame_paths[min(idx, n_total - 1)]).convert("RGB").resize((size, size))
            frames.append(img)
        except Exception:
            pass
    return frames


def extract_answer(response: str) -> str:
    """Extract letter answer (A-E) from Qwen response."""
    import re
    response = response.strip()
    for pat in [r'^([A-E])\b', r'\b([A-E])\b', r'\(([A-E])\)', r'Answer[:\s]+([A-E])']:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def preprocess_qwen_item(
    qwen_processor,
    qwen_model,
    frame_paths: list[str],
    question: str,
    options: list[str],
    n_qwen_frames: int = 64,
    device: torch.device = None,
) -> Optional[dict]:
    """
    Preprocess one QA item: load frames, tokenize, extract video features once.
    Cached result is reused across all G group members (Gumbel samples).
    """
    from qwen_vl_utils import process_vision_info

    frames = load_frames_pil(frame_paths, n_qwen_frames)
    if not frames:
        return None

    options_text = "\n".join(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames,
             "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
            {"type": "text", "text": user_text},
        ],
    }]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        **video_kwargs, return_tensors="pt",
    )

    emb_dev = qwen_model.get_input_embeddings().weight.device
    vis_dev = qwen_model.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    T_merged = int(grid_thw[0, 0].item())

    with torch.inference_mode():
        video_embeds = qwen_model.model.get_video_features(pv_vid, grid_thw).to(emb_dev)

    video_token_id  = qwen_model.config.video_token_id
    seq_is_video    = (input_ids[0] == video_token_id)
    video_positions = seq_is_video.nonzero(as_tuple=True)[0]

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "grid_thw":        grid_thw,
        "video_embeds":    video_embeds,
        "T_merged":        T_merged,
        "video_positions": video_positions,
        "emb_dev":         emb_dev,
    }


def qwen_generate_with_mask(
    qwen_model,
    cached: dict,
    patch_mask: torch.Tensor,   # (196,) bool — TrajGaze 14×14 selection
) -> Optional[str]:
    """
    Run Qwen generation with a subset of visual tokens selected by patch_mask.

    Maps TrajGaze 14×14 mask (196 patches) → Qwen 8×8 merged spatial grid (64 tokens per frame).
    Removes unselected video token positions from the sequence.
    """
    input_ids      = cached["input_ids"]
    attention_mask = cached["attention_mask"]
    grid_thw       = cached["grid_thw"]
    video_embeds   = cached["video_embeds"]
    T_merged       = cached["T_merged"]
    video_positions = cached["video_positions"]
    emb_dev        = cached["emb_dev"]

    # Map 14×14 TrajGaze patches → 8×8 Qwen spatial grid (via nearest + max pool)
    pm_spatial = patch_mask.float().reshape(1, 1, 14, 14)
    pm_q16 = F.interpolate(pm_spatial, size=(16, 16), mode="nearest").squeeze()
    pm_q8  = F.max_pool2d(pm_q16.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
    pm_flat = pm_q8.bool().flatten()  # (64,) per-frame spatial mask

    n_total_video = T_merged * 64
    if video_embeds.shape[0] != n_total_video:
        # Qwen temporal merging changed the expected count — keep all tokens
        video_embeds_kept = video_embeds
        keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    else:
        # Expand spatial mask across all T_merged temporal groups
        pm_full = pm_flat.unsqueeze(0).expand(T_merged, -1).reshape(-1)   # (T_merged * 64,)
        pm_full = pm_full.to(emb_dev)
        video_embeds_kept = video_embeds[pm_full]                          # keep selected

        # Remove unselected video positions from the sequence
        keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
        if video_positions.shape[0] == pm_full.shape[0]:
            keep_seq[video_positions[~pm_full]] = False

    import time as _time
    with torch.inference_mode():
        position_ids, rope_deltas = qwen_model.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )
        new_input_ids      = input_ids[:, keep_seq]
        new_attention_mask = attention_mask[:, keep_seq]
        new_position_ids   = position_ids[:, :, keep_seq]

        new_inputs_embeds = qwen_model.get_input_embeddings()(new_input_ids)
        new_is_video      = (new_input_ids[0] == qwen_model.config.video_token_id)
        new_inputs_embeds[0, new_is_video] = video_embeds_kept.to(new_inputs_embeds.dtype)

        n_input_tokens = new_inputs_embeds.shape[1]

        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        output_ids = qwen_model.generate(
            inputs_embeds=new_inputs_embeds,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            rope_deltas=rope_deltas,
            max_new_tokens=16,
            do_sample=False,
        )
        torch.cuda.synchronize()
        generate_time = _time.perf_counter() - t0

    gen_ids = output_ids if output_ids.shape[1] <= 16 else output_ids[:, -16:]
    return gen_ids, generate_time, n_input_tokens


def qwen_inference_with_mask(
    qwen_processor,
    qwen_model,
    frame_paths: list[str],
    question: str,
    options: list[str],
    patch_mask: torch.Tensor,
    n_qwen_frames: int = 64,
    device: torch.device = None,
) -> str:
    """Convenience wrapper for evaluate.py: preprocess + generate → decoded string."""
    cached = preprocess_qwen_item(
        qwen_processor, qwen_model,
        frame_paths, question, options,
        n_qwen_frames=n_qwen_frames, device=device,
    )
    if cached is None:
        return "", 0.0, 0
    gen_ids, generate_time, n_tokens = qwen_generate_with_mask(qwen_model, cached, patch_mask)
    response = qwen_processor.batch_decode(gen_ids, skip_special_tokens=True)
    return (response[0] if response else ""), generate_time, n_tokens


def compute_pg_loss(
    patch_scores: torch.Tensor,      # (196,) — WITH gradient
    masks:        list[torch.Tensor], # G × (196,) bool
    advantages:   torch.Tensor,       # (G,)
) -> torch.Tensor:
    """
    GRPO policy gradient loss for top-K selection.

    Surrogate for group g:
        surrogate_g = Σ_{i ∈ selected_g} patch_scores[i]
                    = (patch_scores * mask_g).sum()

    ∇_{score_i} surrogate_g = mask_g[i]   (1 if selected, 0 otherwise)

    Policy gradient:
        pg_loss = -1/G * Σ_g ( advantage_g * surrogate_g )
        ∇ pg_loss = -1/G * Σ_g ( advantage_g * mask_g )

    Effect:
        - Patches selected in high-reward groups → score increases
        - Patches selected in low-reward groups  → score decreases
        - Unselected patches: zero gradient (can only learn by being selected)

    Note: advantages sum to 0 by GRPO construction, so when all masks are identical
    pg_loss = 0. Gumbel exploration ensures mask diversity.
    """
    G = len(masks)
    loss = torch.tensor(0.0, device=patch_scores.device)
    for g in range(G):
        surrogate = (patch_scores * masks[g].float()).sum()
        loss = loss - advantages[g] * surrogate
    return loss / G


def collate_qa_single(batch):
    batch = [b for b in batch if b is not None]
    return batch[0] if batch else None


def main():
    args      = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main   = rank == 0
    device    = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Stage 2 GRPO] Output: {args.output_dir}")
        print(f"[Stage 2 GRPO] Group size G={args.group_size}, "
              f"GPUs={world_size}, Epochs={args.epochs}, lr={args.lr}")
        print(f"[Stage 2 GRPO] Gumbel temp: {args.gumbel_temp} → {args.gumbel_temp_final}")

    # ── Load TrajGaze_v2 (policy, trainable) ──────────────────────────────────
    model = TrajGazeV2().to(device)
    if os.path.exists(args.stage1_ckpt):
        ckpt  = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            print(f"[Stage 2] Loaded Stage 1 ckpt: {args.stage1_ckpt}")
            if missing:
                print(f"  Missing keys: {len(missing)}: {missing[:3]}")
    else:
        if is_main:
            print(f"[Stage 2] WARNING: Stage 1 ckpt not found: {args.stage1_ckpt}")

    model = DDP(model, device_ids=[local_rank])

    # ── Load frozen Qwen2.5-VL-7B ─────────────────────────────────────────────
    if is_main:
        print("[Stage 2] Loading frozen Qwen2.5-VL-7B ...")
    qwen_processor, qwen_model = load_qwen(device)
    if is_main:
        print("[Stage 2] Qwen loaded.")

    # ── QA Dataset ────────────────────────────────────────────────────────────
    dataset = TrajGazeV2QADataset(
        datasets=args.datasets,
        n_frames=args.n_frames,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size  = 1,
        sampler     = sampler,
        collate_fn  = collate_qa_single,
        num_workers = 2,
    )

    # Only train the selector model (DINOv2 already frozen inside visual_encoder)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    log_path    = os.path.join(args.output_dir, f"grpo_log_rank{rank}.jsonl")
    best_reward = -float("inf")
    total_steps = len(loader) * args.epochs

    def gumbel_temp_at(step: int) -> float:
        alpha = step / max(1, total_steps - 1)
        return args.gumbel_temp * (1 - alpha) + args.gumbel_temp_final * alpha

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        epoch_reward = 0.0
        epoch_loss   = 0.0
        n_steps      = 0
        n_nonzero_advantage = 0
        t_start      = time.time()

        for step, item in enumerate(loader):
            if item is None:
                continue

            global_step = epoch * len(loader) + step
            g_temp      = gumbel_temp_at(global_step)

            # ── Build trajectory batch (B=1) ──────────────────────────────────
            traj = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in item["traj"].items()}
            question    = item["question"]
            options     = item["options"]
            answer_gt   = item["answer"]
            frame_paths = item["frame_paths"]   # list[str]

            # ── Forward: patch_scores (196,) WITH gradient ─────────────────────
            # This is the core policy: patch scores from visual-trajectory attention
            # under the text query conditioning.
            query_emb   = model.module.query_encoder([question], device)         # (1, D_Q)
            visual_feat = model.module.visual_encoder([frame_paths], device)     # (1, 196, D_enc)
            patch_scores_raw, _ = model.module.encoder(traj, query_emb, visual_feat)
            # patch_scores_raw: (B=1, 196) — encoder output is patch scores, NOT time-series
            scores_agg = patch_scores_raw.squeeze(0)   # (196,)  ← critical fix

            # ── Preprocess Qwen inputs once for all G group members ────────────
            cached = None
            try:
                cached = preprocess_qwen_item(
                    qwen_processor, qwen_model,
                    frame_paths, question, options,
                    n_qwen_frames=args.n_qwen_frames,
                    device=device,
                )
            except Exception:
                if is_main:
                    traceback.print_exc()

            # ── Sample G masks + collect rewards ──────────────────────────────
            rewards = []
            masks   = []

            with torch.no_grad():
                for g in range(args.group_size):
                    # Gumbel-top-K for stochastic exploration
                    noise  = -torch.log(
                        -torch.log(torch.rand(N_PATCHES, device=device) + 1e-8) + 1e-8
                    )
                    scores_noisy = scores_agg.detach() + g_temp * noise
                    _, topk_idx = torch.topk(scores_noisy, k=N_KEEP, dim=-1)
                    mask = torch.zeros(N_PATCHES, dtype=torch.bool, device=device)
                    mask.scatter_(0, topk_idx, True)
                    masks.append(mask)

                    # Run Qwen with this mask → binary reward
                    if cached is None:
                        reward = 0.0
                    else:
                        try:
                            gen_ids = qwen_generate_with_mask(qwen_model, cached, mask)
                            response = qwen_processor.batch_decode(
                                gen_ids, skip_special_tokens=True
                            )[0].strip()
                            pred   = extract_answer(response)
                            reward = 1.0 if pred == answer_gt else 0.0
                        except Exception:
                            if is_main:
                                traceback.print_exc()
                            reward = 0.0
                    rewards.append(reward)

            rewards_t  = torch.tensor(rewards, device=device, dtype=torch.float32)
            group_mean = rewards_t.mean()
            advantages = rewards_t - group_mean   # (G,) — sum to 0 by construction

            # Track whether this step has any learning signal
            has_signal = (advantages.abs() > 1e-6).any()
            if has_signal:
                n_nonzero_advantage += 1

            # ── Policy gradient loss ───────────────────────────────────────────
            # scores_agg retains computation graph from model forward above.
            # gradient flows: pg_loss → scores_agg → encoder → visual_encoder.proj
            #                                                 → vt_fusion weights
            #                                                 → query_encoder
            pg_loss = compute_pg_loss(scores_agg, masks, advantages)

            optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_reward += group_mean.item()
            epoch_loss   += pg_loss.item()
            n_steps      += 1

            if is_main and n_steps % args.log_every == 0:
                avg_r  = epoch_reward / n_steps
                avg_l  = epoch_loss   / n_steps
                pct_sig = 100 * n_nonzero_advantage / n_steps
                elapsed = time.time() - t_start
                print(f"Epoch {epoch+1} | step {n_steps:4d}/{len(loader)} | "
                      f"reward={avg_r:.3f} | pg_loss={avg_l:.4f} | "
                      f"signal={pct_sig:.0f}% | temp={g_temp:.3f} | t={elapsed:.0f}s")
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "epoch": epoch + 1, "step": n_steps,
                        "avg_reward": avg_r, "pg_loss": avg_l,
                        "group_rewards": rewards,
                        "gumbel_temp": g_temp,
                        "pct_signal": pct_sig,
                    }) + "\n")

        # ── End of epoch ──────────────────────────────────────────────────────
        if is_main:
            avg_r   = epoch_reward / max(1, n_steps)
            avg_l   = epoch_loss   / max(1, n_steps)
            elapsed = time.time() - t_start
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | "
                  f"avg_reward={avg_r:.3f} | avg_pg_loss={avg_l:.4f} | "
                  f"signal_steps={n_nonzero_advantage}/{n_steps} | "
                  f"time={elapsed:.0f}s ===\n")

            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth")
                torch.save({
                    "epoch":  epoch,
                    "model":  model.module.state_dict(),
                    "reward": avg_r,
                }, ckpt_path)
                print(f"Saved: {ckpt_path}")

            if avg_r > best_reward:
                best_reward = avg_r
                torch.save({"model": model.module.state_dict(), "reward": best_reward},
                           os.path.join(args.output_dir, "best.pth"))
                print(f"New best reward: {best_reward:.3f}")

    if is_main:
        print(f"\n[Stage 2 GRPO] Complete. Best reward: {best_reward:.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
