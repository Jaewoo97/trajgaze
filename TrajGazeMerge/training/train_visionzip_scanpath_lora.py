"""OUR METHOD — gaze-scanpath token channel on top of M1's frozen selection.

M1 (VZ-complement, learned top-k) = 63.01% is the best 10%-token method, but three
re-selection studies (coverage, fusion, anchored) all netted flat-to-negative: at a
fixed budget, emphasising gaze just trades scene-context tokens. The per-task data
shows the weak tasks (spatial ~40, temporal ~42, future-action ~47) don't move under
any reshuffle — they're missing a *behavioural* signal, not a different patch subset.

So this trainer does NOT touch the selection. It reuses M1's exact
`select_complementary(..., complement_mode="topk", mode="learned")` to pick the same
6.5% raw + 3.5% merged tokens, then ADDS K gaze-scanpath tokens (ScanpathEncoder)
into the sequence right after the video block. Strictly additive (K≈8 vs hundreds of
kept video tokens) → escapes the zero-sum. The frozen TAS Stage-1 encoder still drives
the complement selection (unchanged); the ScanpathEncoder is a *separate*, trainable
module that turns the ordered gaze/hand trajectory into intent tokens.

Trained jointly with LoRA on the same protocol (eff-batch 8, 3 epochs, epoch1->epoch2
early-stop, gaze-overlay, egtea 2-way val). Question this answers: does M1-selection +
gaze-channel beat M1-selection alone (63.01)?

Usage (4-GPU DDP):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29662 \\
      -m TrajGazeMerge.training.train_visionzip_scanpath_lora \\
      --stage1-ckpt .../stage1_tas_3way_overlay/best.pth \\
      --output-dir  .../checkpoints/ours_scanpath \\
      --epochs 3 --lr 1e-4 --scan-lr 1e-3 --grad-accum 2 \\
      --no-hdepic --early-stop --no-mid-eval
"""
from __future__ import annotations

import argparse
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
from TrajGazeMerge.models.model import get_option_ids, forward_logits
from TrajGazeMerge.models.scanpath_encoder import ScanpathEncoder
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.training.train_visionzip_complement_lora import (
    select_complementary, setup_ddp, STAGE1_DEFAULT,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_scanpath")
    # --- M1 selection (frozen behaviour; defaults reproduce M1 exactly) ---
    p.add_argument("--traj-pool-mode", choices=["learned", "anticipatory"], default="learned")
    p.add_argument("--content-ratio", type=float, default=0.07)
    p.add_argument("--traj-ratio",    type=float, default=0.03)
    p.add_argument("--stage1-ckpt",   default=STAGE1_DEFAULT,
                   help="Frozen TAS Stage-1 encoder driving the complement selection.")
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    # --- gaze-scanpath channel ---
    p.add_argument("--gaze-tokens", type=int, default=8, help="K tokens added to the sequence.")
    p.add_argument("--t-scan",      type=int, default=32, help="Scanpath resample length.")
    p.add_argument("--scan-hidden", type=int, default=256)
    p.add_argument("--scan-layers", type=int, default=2)
    p.add_argument("--scan-lr",     type=float, default=1e-3,
                   help="LR for the (fresh) scanpath encoder; LoRA uses --lr.")
    # --- training ---
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int,   default=2)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--log-every",  type=int,   default=20)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--no-hdepic", dest="include_hdepic", action="store_false",
                   help="2-way: StreamGaze + EgoGazeVQA only (exclude HD-EPIC).")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop after epoch 2 if epoch-2 val <= epoch-1 val.")
    p.add_argument("--no-mid-eval", action="store_true")
    p.set_defaults(include_hdepic=True)
    return p.parse_args()


def build_inputs_with_gaze(base_qwen, cached, merged_video, receiver_idx, gaze_tokens):
    """Like model.build_merged_inputs, but inserts K trainable gaze tokens right
    after the video block. Position ids for the inserted tokens continue the
    (text-like) post-video positions; everything after is shifted by K.

    Must run OUTSIDE torch.no_grad so gradients reach gaze_tokens -> ScanpathEncoder.
    """
    input_ids       = cached["input_ids"]
    attention_mask  = cached["attention_mask"]
    position_ids    = cached["position_ids"]
    rope_deltas     = cached["rope_deltas"]
    video_embeds    = cached["video_embeds"]
    video_positions = cached["video_positions"]
    emb_dev         = cached["emb_dev"]
    N_video = video_embeds.shape[0]

    is_receiver = torch.zeros(N_video, dtype=torch.bool, device=emb_dev)
    is_receiver[receiver_idx] = True
    source_video_pos = video_positions[~is_receiver]

    keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[source_video_pos] = False

    new_input_ids    = input_ids[:, keep_seq]
    new_attn_mask    = attention_mask[:, keep_seq]
    new_position_ids = position_ids[:, :, keep_seq]                       # (3,1,L')

    new_inputs_embeds = base_qwen.get_input_embeddings()(new_input_ids)   # (1,L',d)
    video_token_id = base_qwen.config.video_token_id
    new_is_video   = (new_input_ids[0] == video_token_id)
    new_inputs_embeds[0, new_is_video] = merged_video.to(new_inputs_embeds.dtype)

    # ── insert K gaze tokens right after the last video token ──
    K = gaze_tokens.shape[0]
    last_vid = int(new_is_video.nonzero(as_tuple=True)[0][-1].item())
    ins = last_vid + 1
    gt = gaze_tokens.to(new_inputs_embeds.dtype).unsqueeze(0)             # (1,K,d)
    emb = torch.cat([new_inputs_embeds[:, :ins], gt, new_inputs_embeds[:, ins:]], dim=1)

    ones = torch.ones(1, K, dtype=new_attn_mask.dtype, device=emb_dev)
    am = torch.cat([new_attn_mask[:, :ins], ones, new_attn_mask[:, ins:]], dim=1)

    L = new_position_ids.shape[2]
    if ins < L:
        s = new_position_ids[:, :, ins:ins + 1]                          # (3,1,1)
    else:
        s = new_position_ids[:, :, -1:] + 1
    gaze_pid = s + torch.arange(K, device=emb_dev).view(1, 1, K)         # (3,1,K)
    tail = new_position_ids[:, :, ins:] + K
    new_pid = torch.cat([new_position_ids[:, :, :ins], gaze_pid, tail], dim=2)

    return {
        "inputs_embeds":  emb,
        "attention_mask": am,
        "position_ids":   new_pid,
        "rope_deltas":    rope_deltas,
    }


def evaluate(processor, model, base_qwen, scan_encoder, option_ids, device,
             mode, encoder, hp, content_ratio, traj_ratio, include_hdepic=True):
    test_ds = CombinedMergeDataset(
        split="test", n_vlm_frames=128, n_traj_frames=128, include_hdepic=include_hdepic,
    )
    model.eval()
    scan_encoder.eval()
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

                sel_embeds, recv_idx = select_complementary(
                    cached, item, device, mode, encoder, hp, content_ratio, traj_ratio,
                    complement_mode="topk")
                gaze_tokens = scan_encoder(item.get("traj"), device)
                inputs_dict = build_inputs_with_gaze(
                    base_qwen, cached, sel_embeds, recv_idx, gaze_tokens)
                logits = forward_logits(model, inputs_dict)
                pred_idx = logits[option_ids[:n_opt]].argmax().item()
                gt_idx = letters.index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok; total += 1
                by_task.setdefault(item["task"], []).append(ok)
            except Exception:
                pass
    model.train()
    scan_encoder.train()
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}
    return 100.0 * correct / max(1, total), total, per_task


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    hp = dict(horizon=2.0, sigma_g=2.0, sigma_h=3.0,
              alpha_hand=0.7, sigma_v=0.05, sigma_gh=0.10)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[scanpath] output: {args.output_dir}", flush=True)
        print(f"[scanpath] M1 selection (mode={args.traj_pool_mode}, topk complement, "
              f"content={args.content_ratio*100:.1f}% ∪ traj={args.traj_ratio*100:.1f}%) "
              f"+ gaze channel K={args.gaze_tokens}, t_scan={args.t_scan}", flush=True)
        print(f"[scanpath] GPUs={world_size}, epochs={args.epochs}, lora_lr={args.lr}, "
              f"scan_lr={args.scan_lr}, grad_accum={args.grad_accum}, "
              f"eff_batch={args.grad_accum*world_size}", flush=True)

    if is_main: print("Loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    hidden = base_qwen.config.hidden_size
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    option_ids = get_option_ids(processor, 5)

    # frozen TAS encoder for the (unchanged) M1 complement selection
    encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # trainable scanpath encoder (its own DDP wrapper; all params used every step)
    scan_encoder = ScanpathEncoder(
        out_dim=hidden, hidden=args.scan_hidden, n_layers=args.scan_layers,
        n_tokens=args.gaze_tokens, t_scan=args.t_scan,
    ).to(device)
    scan_encoder = DDP(scan_encoder, device_ids=[local_rank])
    if is_main:
        n_scan = sum(p.numel() for p in scan_encoder.parameters())
        print(f"Model loaded. ScanpathEncoder params: {n_scan/1e6:.2f}M", flush=True)

    train_ds = CombinedMergeDataset(
        split="train", n_vlm_frames=args.n_frames, n_traj_frames=args.n_frames,
        include_hdepic=args.include_hdepic,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                        collate_fn=lambda b: b[0], num_workers=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    scan_params = list(scan_encoder.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": scan_params, "lr": args.scan_lr},
    ], weight_decay=1e-4)

    log_path = os.path.join(args.output_dir, f"train_log_rank{rank}.jsonl")
    best_acc = 0.0
    epoch_accs: list[float] = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train(); scan_encoder.train()
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
                    sel_embeds, recv_idx = select_complementary(
                        cached, item, device, args.traj_pool_mode, encoder, hp,
                        args.content_ratio, args.traj_ratio, complement_mode="topk")

                # gaze tokens + input build OUTSIDE no_grad (trainable path)
                gaze_tokens = scan_encoder(item.get("traj"), device)
                inputs_dict = build_inputs_with_gaze(
                    base_qwen, cached, sel_embeds, recv_idx, gaze_tokens)

                logits = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids[:n_opt]]
                gt_idx = letters.index(item["answer"])
                loss = F.cross_entropy(option_logits.unsqueeze(0),
                                       torch.tensor([gt_idx], device=device))
                loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if n_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + scan_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                if is_main and n_steps % args.log_every == 0:
                    avg_loss = epoch_loss / n_steps
                    elapsed = time.time() - t_start
                    pct_kept = 100.0 * recv_idx.shape[0] / max(1, n_video)
                    gate = float(scan_encoder.module.gate.detach().item())
                    print(f"Epoch {epoch+1} | step {n_steps}/{len(loader)} | "
                          f"loss={avg_loss:.4f} | kept={pct_kept:.1f}% | gate={gate:.3f} | "
                          f"t={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch+1, "step": n_steps, "loss": avg_loss,
                            "pct_kept": pct_kept, "gate": gate, "elapsed": elapsed,
                        }) + "\n")
            except Exception:
                if is_main: traceback.print_exc()
                continue

        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(lora_params + scan_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed = time.time() - t_start
        scan_cfg = dict(out_dim=hidden, hidden=args.scan_hidden, n_layers=args.scan_layers,
                        n_tokens=args.gaze_tokens, t_scan=args.t_scan)
        if is_main:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} "
                  f"| time={elapsed:.0f}s ===", flush=True)
            torch.save({
                "epoch": epoch+1,
                "lora_state": model.module.state_dict(),
                "scan_state": scan_encoder.module.state_dict(),
                "scan_cfg": scan_cfg,
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch+1:02d}.pth"))

        dist.barrier()
        stop = torch.zeros(1, device=device)
        if is_main:
            label = "3-way" if args.include_hdepic else "egtea 2-way"
            print(f"Evaluating epoch {epoch+1} on full {label} val set ...", flush=True)
            acc, n_eval, per_task = evaluate(
                processor, model.module, base_qwen, scan_encoder.module, option_ids, device,
                args.traj_pool_mode, encoder, hp, args.content_ratio, args.traj_ratio,
                include_hdepic=args.include_hdepic,
            )
            print(f"  Overall: {acc:.2f}%  (n={n_eval})", flush=True)
            for task, task_acc in per_task.items():
                print(f"    {task}: {task_acc:.2f}%", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch+1, "eval_acc": acc, "n_eval": n_eval, "per_task": per_task,
                }) + "\n")
            epoch_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch": epoch+1,
                    "lora_state": model.module.state_dict(),
                    "scan_state": scan_encoder.module.state_dict(),
                    "scan_cfg": scan_cfg,
                    "acc": acc,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  → saved best (acc={acc:.2f}%)", flush=True)
            if args.early_stop and (epoch + 1) == 2 and len(epoch_accs) >= 2 \
                    and epoch_accs[1] <= epoch_accs[0]:
                stop[0] = 1.0
                print(f"  Early stop: epoch2 {epoch_accs[1]:.2f}% <= epoch1 "
                      f"{epoch_accs[0]:.2f}% → skipping epoch 3.", flush=True)
        dist.barrier()
        dist.broadcast(stop, src=0)
        if stop.item() > 0:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
