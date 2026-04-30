"""
Zero-shot TrajGazeMerge eval: use TrajGazeV2 stage1 gaze scores to drive
gaze_weighted_merge (same as joint training), fed to the LoRA-finetuned baseline Qwen VLM.

No joint training — the stage1 encoder and baseline_lora are used as-is.
Gaze-weighted merge: top 10% kept as receivers, bottom 90% merged into nearest
receiver by cosine similarity (weighted average) — same merge op as train_merge_lora.

Eval on EGTEA test set (526 items, all MCQ tasks), 4 GPUs via torchrun.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29606 \\
        -m TrajGazeMerge.training.eval_trajgaze_zeroshot \\
        --stage1-ckpt /workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth \\
        --lora-ckpt   /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth \\
        --out         /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/trajgaze_zeroshot_metrics.json \\
        --keep-ratio  0.10
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import score_to_qwen_spatial, gaze_weighted_merge
from TrajGazeMerge.models.model import (
    get_option_ids, preprocess_item, build_merged_inputs, forward_logits,
    QWEN_PATH, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
)
from TrajGaze_v2.models.model import TrajGazeV2

STAGE1_CKPT = "/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_v3/best.pth"
LORA_CKPT   = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth"
KEEP_RATIO  = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", default=STAGE1_CKPT)
    p.add_argument("--lora-ckpt",   default=LORA_CKPT)
    p.add_argument("--out",         required=True)
    p.add_argument("--keep-ratio",  type=float, default=KEEP_RATIO)
    p.add_argument("--n-vlm-frames",  type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=32)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def load_lora_model(device):
    from transformers import Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(QWEN_PATH)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    return processor, model


def load_stage1(ckpt_path, device):
    enc = TrajGazeV2().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("encoder_state", ckpt))
    enc.load_state_dict(state, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def get_patch_scores(traj_encoder, item, device):
    """Run TrajGazeV2 forward → (196,) patch importance scores."""
    traj_batch = {k: v.unsqueeze(0).to(device) for k, v in item["traj"].items()}
    query_emb   = traj_encoder.query_encoder([item["question"]], device)
    visual_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
    scores_raw, _ = traj_encoder.encoder(traj_batch, query_emb, visual_feat)
    return scores_raw.squeeze(0)   # (196,)




def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"[ZeroShot] keep={args.keep_ratio*100:.0f}% via gaze_weighted_merge (90% merged in)", flush=True)
        print(f"[ZeroShot] stage1: {args.stage1_ckpt}", flush=True)
        print(f"[ZeroShot] VLM:    {args.lora_ckpt}", flush=True)

    # Load LoRA VLM
    if is_main:
        print("Loading baseline LoRA VLM ...", flush=True)
    processor, model = load_lora_model(device)
    ckpt  = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor)

    # Load TrajGazeV2 stage1
    if is_main:
        print("Loading TrajGazeV2 stage1 encoder ...", flush=True)
    traj_encoder = load_stage1(args.stage1_ckpt, device)

    if is_main:
        print("Models loaded.", flush=True)

    # Dataset — needs trajectory data
    test_ds = StreamGazeMergeDataset(
        split="test",
        n_vlm_frames=args.n_vlm_frames,
        n_traj_frames=args.n_traj_frames,
    )
    sampler = DistributedSampler(
        test_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    loader = DataLoader(
        test_ds, batch_size=1, sampler=sampler,
        collate_fn=lambda b: b[0], num_workers=2,
    )

    if is_main:
        print(f"Evaluating {len(test_ds)} items across {world_size} GPUs ...", flush=True)

    correct = 0
    total   = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for i, item in enumerate(loader):
            if item is None:
                continue
            try:
                # Standard Qwen preprocessing (same as baseline_lora training)
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
                )
                if cached is None:
                    continue

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)

                # TrajGazeV2 gaze scores → top-k token selection
                scores    = get_patch_scores(traj_encoder, item, device)       # (196,)
                scores_q  = score_to_qwen_spatial(scores, n_spatial)           # (n_spatial,)
                scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                if scores_all.shape[0] != n_video:
                    if scores_all.shape[0] > n_video:
                        scores_all = scores_all[:n_video]
                    else:
                        reps = (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                        scores_all = scores_all.repeat(reps)[:n_video]

                r = max(1, int((1.0 - args.keep_ratio) * n_video))
                selected_embeds, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all, r
                )

                inputs_dict   = build_merged_inputs(base_qwen, cached, selected_embeds, receiver_idx)
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)

                if is_main and (i + 1) % 30 == 0:
                    print(f"  [{i+1}/{len(loader)}] acc={100.*correct/max(1,total):.2f}%",
                          flush=True)

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

    # Gather results from all ranks
    stats = torch.tensor([correct, total], dtype=torch.long, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_correct = stats[0].item()
    total_n       = stats[1].item()

    # Gather per-task results
    all_task_data = [None] * world_size
    dist.all_gather_object(all_task_data, by_task)

    if is_main:
        merged_tasks: dict[str, list] = {}
        for td in all_task_data:
            for task, vals in td.items():
                merged_tasks.setdefault(task, []).extend(vals)

        acc      = 100.0 * total_correct / max(1, total_n)
        per_task = {t: 100.0 * sum(v) / max(1, len(v))
                    for t, v in sorted(merged_tasks.items())}

        print(f"\n=== Zero-shot TrajGazeMerge (gaze_weighted_merge, keep {args.keep_ratio*100:.0f}%) ===",
              flush=True)
        print(f"Overall: {acc:.2f}%  (n={total_n})", flush=True)
        for t, a in per_task.items():
            print(f"  {t}: {a:.2f}%  (n={len(merged_tasks[t])})", flush=True)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "acc": acc, "n": total_n, "per_task": per_task,
                "stage1_ckpt": args.stage1_ckpt,
                "lora_ckpt": args.lora_ckpt,
                "keep_ratio": args.keep_ratio,
                "method": "trajgaze_zeroshot_weighted_merge",
            }, f, indent=2)
        print(f"Saved → {args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
