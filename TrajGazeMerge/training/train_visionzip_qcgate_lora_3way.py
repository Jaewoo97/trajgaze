"""QC-Gate (Query-Conditioned Coordination Gating) — 3-way LoRA training.

Builds on train_visionzip_traj_lora_3way.py. Same VisionZip backbone, same 10%
token budget (5% dominant + 5% contextual), same DDP / dataset / LoRA protocol.
The ONLY addition is a small learned controller (TrajGazeMerge/models/qc_gate.py)
that emits signed per-frame gates (alpha_gaze, alpha_hand, alpha_temp) from the
question embedding + per-frame coordination state, replacing VZ-traj's fixed
+1.0 gaze / +0.7 hand / 0 temporal prior. Initialized to equal VZ-traj at step 0.

Gradient reaches the controller via a straight-through gate on the selected
dominant tokens, so selection/merge run IN-GRAPH here (only the ViT/preprocess
stays under no_grad). The controller is wrapped in its own DDP and its params are
optimized jointly with the LoRA params; checkpoints carry controller_state.

Usage (4-GPU DDP, matching VZ/VZ-traj protocol):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29826 \\
      -m TrajGazeMerge.training.train_visionzip_qcgate_lora_3way \\
      --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/qcgate_lora_3way \\
      --epochs 3 --lr 1e-4 --grad-accum 4
"""
from __future__ import annotations

import argparse
import datetime
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
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.qc_gate import (
    QCGateController, qcgate_reweight_scores, qcgate_select_tokens,
    pooled_question_embedding,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, CONTEXTUAL_RATIO,
    load_visionzip_lora, preprocess_visionzip_item,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/qcgate_lora_3way")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    # Gaussian-kernel / coordination-feature scales (gate AMPLITUDE is learned)
    p.add_argument("--sigma-g",    type=float, default=2.0)
    p.add_argument("--sigma-h",    type=float, default=3.0)
    p.add_argument("--sigma-v",    type=float, default=0.05)
    p.add_argument("--sigma-gh",   type=float, default=0.10)
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def qcgate_select(cached, item, controller, base_qwen, device, hp, training):
    """QC-Gate reweighting + straight-through selection.

    Returns (sel_embeds, recv_idx, gate_means) where gate_means = (g, h, t) are
    detached per-frame-mean gate values for logging. `controller` is the DDP wrap
    in training (so DDP backward hooks fire) and the bare module in eval.
    """
    q_vec = pooled_question_embedding(base_qwen, cached)
    s, gates = qcgate_reweight_scores(
        cached["attn_scores"], cached["grid_thw"], item["traj"], q_vec, controller, device,
        sigma_g=hp["sigma_g"], sigma_h=hp["sigma_h"], sigma_v=hp["sigma_v"], sigma_gh=hp["sigma_gh"],
    )
    sel_embeds, recv_idx = qcgate_select_tokens(
        cached["video_embeds"], s, cached["attn_key"], training=training,
    )
    a_g, a_h, a_t = gates
    gate_means = (a_g.detach().mean().item(),
                  a_h.detach().mean().item(),
                  a_t.detach().mean().item())
    return sel_embeds, recv_idx, gate_means


def evaluate(processor, model, base_qwen, controller, option_ids, device, hp):
    """Full 3-way val eval (controller in inference mode, no straight-through gate)."""
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=True,
    )
    model.eval()
    controller.eval()
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

                sel_embeds, recv_idx, _ = qcgate_select(
                    cached, item, controller, base_qwen, device, hp, training=False
                )
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
    controller.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[QC-Gate 3way] output: {args.output_dir}")
        print(f"[QC-Gate 3way] GPUs={world_size}, dominant={DOMINANT_RATIO*100:.0f}% + "
              f"contextual={CONTEXTUAL_RATIO*100:.0f}% = 10%, "
              f"epochs={args.epochs}, lr={args.lr}, grad_accum={args.grad_accum}")
        print(f"[QC-Gate 3way] kernels: sigma_g={args.sigma_g} sigma_h={args.sigma_h} "
              f"sigma_v={args.sigma_v} sigma_gh={args.sigma_gh}  (gate amplitude is LEARNED)")

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # QC-Gate controller — its own DDP. find_unused_parameters=True is defensive:
    # at step 0 the zero-init output layer blocks grad to fc1/q_proj (they still
    # participate, but this avoids any reduction-mismatch edge case mid-run).
    controller = QCGateController().to(device)
    controller = DDP(controller, device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        n_ctrl = sum(p.numel() for p in controller.parameters())
        print(f"[QC-Gate 3way] controller params: {n_ctrl:,} "
              f"(init == VZ-traj: alpha_gaze~1.0, alpha_hand~0.7, alpha_temp~0)")

    option_ids = get_option_ids(processor, 5)
    if is_main: print("Model loaded.")

    hp = dict(sigma_g=args.sigma_g, sigma_h=args.sigma_h,
              sigma_v=args.sigma_v, sigma_gh=args.sigma_gh)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=True,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    ctrl_params = [p for p in controller.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params + ctrl_params, lr=args.lr, weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        controller.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_steps = 0
        gm_sum = [0.0, 0.0, 0.0]
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

                # QC-Gate selection + merge run IN-GRAPH (controller + ST gate)
                sel_embeds, recv_idx, gate_means = qcgate_select(
                    cached, item, controller, base_qwen, device, hp, training=True
                )
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
                for i in range(3): gm_sum[i] += gate_means[i]

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + ctrl_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    n_kept = recv_idx.shape[0]
                    pct_kept = 100.0 * n_kept / max(1, n_video)
                    gm = [gm_sum[i] / n_steps for i in range(3)]
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | "
                          f"a_gaze={gm[0]:+.3f} a_hand={gm[1]:+.3f} a_temp={gm[2]:+.3f} | "
                          f"t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps,
                            "loss": avg_loss, "pct_kept": pct_kept,
                            "a_gaze": gm[0], "a_hand": gm[1], "a_temp": gm[2],
                            "elapsed": elapsed,
                        }) + "\n")
            except Exception:
                if is_main: traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + ctrl_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed = time.time() - t_start
        if is_main:
            gm = [gm_sum[i] / max(1, n_steps) for i in range(3)]
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | "
                  f"a_gaze={gm[0]:+.3f} a_hand={gm[1]:+.3f} a_temp={gm[2]:+.3f} | "
                  f"time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model.module.state_dict(),
                "controller_state": controller.module.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        if is_main:
            print(f"Evaluating epoch {epoch+1} on full 3-way val set ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, controller.module, option_ids, device, hp,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc,
                    "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "controller_state": controller.module.state_dict(),
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
