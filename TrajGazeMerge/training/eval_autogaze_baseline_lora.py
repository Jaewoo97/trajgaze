"""
Zero-shot AutoGaze 10% pruning applied to baseline_lora at inference.

Uses frozen AutoGaze to score visual tokens, keeps top 10% via gaze_weighted_merge,
feeds to the LoRA-finetuned baseline Qwen VLM. No joint training.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29610 \\
        -m TrajGazeMerge.training.eval_autogaze_baseline_lora \\
        --out /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/autogaze_zeroshot_baseline_lora_metrics.json
"""

from __future__ import annotations

import argparse, datetime, json, os, sys, traceback
import torch
import torch.distributed as dist
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

from TrajGazeMerge.training.train_autogaze_lora import (
    StreamGazeSimpleDataset, _sample_paths,
    load_autogaze, compute_ag_scores,
    N_AG_FRAMES, FRAME_SIZE, AUTOGAZE_CKPT,
)
from TrajGazeMerge.models.model import (
    get_option_ids, preprocess_item, build_merged_inputs, forward_logits,
    QWEN_PATH, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
)
from TrajGazeMerge.models.merge import gaze_weighted_merge

BASELINE_CKPT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth"
KEEP_RATIO    = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          default=BASELINE_CKPT)
    p.add_argument("--out",           required=True)
    p.add_argument("--autogaze-ckpt", default=AUTOGAZE_CKPT)
    p.add_argument("--keep-ratio",    type=float, default=KEEP_RATIO)
    p.add_argument("--n-frames",      type=int,   default=128)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        print(f"[AutoGaze ZeroShot] keep={args.keep_ratio*100:.0f}% via gaze_weighted_merge", flush=True)
        print(f"[AutoGaze ZeroShot] ckpt={args.ckpt}", flush=True)

    # Load AutoGaze (frozen)
    if is_main:
        print("Loading AutoGaze ...", flush=True)
    ag_transform, ag_model = load_autogaze(device)

    # Load baseline LoRA VLM
    if is_main:
        print("Loading Qwen2.5-VL-7B + LoRA ...", flush=True)
    from transformers import Qwen2_5_VLForConditionalGeneration
    processor  = AutoProcessor.from_pretrained(QWEN_PATH)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("lora_state", ckpt), strict=False)
    model.eval()
    base_qwen  = model.get_base_model()
    option_ids = get_option_ids(processor)

    if is_main:
        print("Models loaded.", flush=True)

    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=args.n_frames)
    sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader  = DataLoader(test_ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b[0], num_workers=2)

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
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device,
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
                r = max(1, int((1.0 - args.keep_ratio) * n_video))
                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all.to(device), r
                )
                inputs_dict   = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                logits        = forward_logits(model, inputs_dict)
                pred_idx      = logits[option_ids].argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)

                if is_main and (i + 1) % 30 == 0:
                    print(f"  [{i+1}/{len(loader)}] acc={100.*correct/max(1,total):.2f}%", flush=True)

            except Exception:
                if is_main:
                    traceback.print_exc()
                continue

    stats = torch.tensor([correct, total], dtype=torch.long, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    all_tasks = [None] * world_size
    dist.all_gather_object(all_tasks, by_task)

    if is_main:
        merged: dict[str, list] = {}
        for td in all_tasks:
            for t, v in td.items():
                merged.setdefault(t, []).extend(v)

        acc      = 100.0 * stats[0].item() / max(1, stats[1].item())
        per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(merged.items())}

        print(f"\n=== Zero-shot AutoGaze (gaze_weighted_merge, keep {args.keep_ratio*100:.0f}%) on baseline_lora ===", flush=True)
        print(f"Overall: {acc:.2f}%  (n={stats[1].item()})", flush=True)
        for t, a in per_task.items():
            print(f"  {t}: {a:.2f}%  (n={len(merged[t])})", flush=True)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "acc": acc, "n": stats[1].item(), "per_task": per_task,
                "ckpt": args.ckpt, "autogaze_ckpt": args.autogaze_ckpt,
                "keep_ratio": args.keep_ratio,
                "method": "autogaze_zeroshot_weighted_merge_baseline_lora",
            }, f, indent=2)
        print(f"Saved → {args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
