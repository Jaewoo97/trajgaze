"""
Corrected evaluation: ALL 1011 combined egtea items (no [:max_items] truncation),
with per-source breakdown (StreamGaze egtea vs EgoGazeVQA egtea) + combined.

Reuses the exact eval logic from the training scripts.

Usage (single GPU):
    python -m TrajGazeMerge.eval.eval_full_1011 \
        --stage baseline --ckpt .../baseline_lora_combined/best.pth --gpu 0
    python -m TrajGazeMerge.eval.eval_full_1011 \
        --stage merge --ckpt .../merge_lora_no_kd_combined/best.pth \
        --merge-ratio 0.9 --gpu 2
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_full_inputs, build_merged_inputs, forward_logits,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["baseline", "merge", "visionzip", "anchored_merge", "random_merge"], required=True)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--merge-ratio", type=float, default=0.9)
    p.add_argument("--stage1-ckpt",
                   default="/workspace/trajgaze_st/TrajGaze_v2/checkpoints/stage1_v3/best.pth")
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    tag = f"[{args.stage}:{os.path.basename(os.path.dirname(args.ckpt))}/{os.path.basename(args.ckpt)}]"

    # ── Model + dataset (stage-specific) ──────────────────────────────────────
    if args.stage == "visionzip":
        from TrajGazeMerge.training.train_visionzip_lora import (
            load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
        )
        from TrajGazeMerge.data.combined_simple_dataset import CombinedSimpleDataset
        processor, qwen = load_visionzip_lora(device)
        ds_cls, ds_kw = CombinedSimpleDataset, dict(split="test", n_vlm_frames=args.n_frames)
    else:
        processor, qwen = load_qwen_lora(device)
        ds_cls, ds_kw = CombinedMergeDataset, dict(split="test",
                                                    n_vlm_frames=args.n_frames,
                                                    n_traj_frames=args.n_traj_frames)
    base_qwen = qwen.get_base_model()
    option_ids = get_option_ids(processor, 5)   # A–E pool, sliced per item

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    qwen.load_state_dict(ckpt["lora_state"], strict=False)
    qwen.eval()

    if args.stage == "random_merge":
        # need merge ops + a deterministic per-item RNG
        from TrajGazeMerge.models.merge import gaze_weighted_merge

    traj_encoder = None
    anchor_boost = None
    if args.stage in ("merge", "anchored_merge"):
        from TrajGaze_v2.models.model import TrajGazeV2
        from TrajGazeMerge.models.merge import gaze_weighted_merge, score_to_qwen_spatial
        traj_encoder = TrajGazeV2().to(device)
        traj_encoder.load_state_dict(ckpt["encoder_state"], strict=False)
        traj_encoder.eval()
        if args.stage == "anchored_merge":
            from TrajGazeMerge.models.merge import compute_anchor_mask_for_qwen
            anchor_boost = 1.0e6

    ds = ds_cls(**ds_kw)

    # per-source counters (sg = StreamGaze, eg = EgoGazeVQA, hd = HD-EPIC P09)
    stats = {"sg": {"c_full": 0, "c_merge": 0, "n": 0},
             "eg": {"c_full": 0, "c_merge": 0, "n": 0},
             "hd": {"c_full": 0, "c_merge": 0, "n": 0}}

    with torch.no_grad():
        for idx in range(len(ds)):
            src, _ = ds.items[idx]
            try:
                item = ds[idx]
                if item is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i) for i in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                oids = option_ids[:n_opt]
                gt = letters.index(item["answer"])

                if args.stage == "visionzip":
                    cached = preprocess_visionzip_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device
                    )
                else:
                    cached = preprocess_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device
                    )
                if cached is None:
                    continue

                full_inputs = build_full_inputs(base_qwen, cached)
                pred_full = forward_logits(qwen, full_inputs)[oids].argmax().item()

                if args.stage == "visionzip":
                    sel, recv = visionzip_select_tokens(
                        cached["video_embeds"], cached["attn_scores"], cached["attn_key"]
                    )
                    merged_inputs = build_merged_inputs(base_qwen, cached, sel, recv)
                    pred_merge = forward_logits(qwen, merged_inputs)[oids].argmax().item()
                elif args.stage == "random_merge":
                    n_video = cached["video_embeds"].shape[0]
                    r = max(1, int(args.merge_ratio * n_video))
                    gen = torch.Generator(device=device).manual_seed(args.random_seed + idx)
                    scores_all = torch.rand(n_video, generator=gen, device=device, dtype=torch.float32)
                    video_det = cached["video_embeds"].detach()
                    merged_video, recv = gaze_weighted_merge(video_det, scores_all, r)
                    merged_inputs = build_merged_inputs(base_qwen, cached, merged_video, recv)
                    pred_merge = forward_logits(qwen, merged_inputs)[oids].argmax().item()
                elif args.stage in ("merge", "anchored_merge"):
                    n_video   = cached["video_embeds"].shape[0]
                    T_merged  = int(cached["grid_thw"][0, 0].item())
                    n_spatial = n_video // max(1, T_merged)
                    r = max(1, int(args.merge_ratio * n_video))
                    traj_batch = {k: v.unsqueeze(0).to(device)
                                  for k, v in item["traj"].items()}
                    q_emb = traj_encoder.query_encoder([item["question"]], device)
                    v_feat = traj_encoder.visual_encoder([item["traj_frame_paths"]], device)
                    scores, _ = traj_encoder.encoder(traj_batch, q_emb, v_feat)
                    scores = scores.squeeze(0)
                    scores_q = score_to_qwen_spatial(scores, n_spatial)
                    scores_all = scores_q.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                    if scores_all.shape[0] != n_video:
                        if scores_all.shape[0] > n_video:
                            scores_all = scores_all[:n_video]
                        else:
                            scores_all = scores_all.repeat(
                                (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                            )[:n_video]
                    if anchor_boost is not None:
                        anchor_spat = compute_anchor_mask_for_qwen(item["traj"], n_spatial).to(scores_all.device)
                        anchor_all = anchor_spat.unsqueeze(0).expand(T_merged, -1).reshape(-1)
                        if anchor_all.shape[0] != n_video:
                            anchor_all = (anchor_all[:n_video] if anchor_all.shape[0] > n_video
                                          else anchor_all.repeat((n_video + anchor_all.shape[0] - 1) // anchor_all.shape[0])[:n_video])
                        scores_all = scores_all + anchor_boost * anchor_all
                    video_det = cached["video_embeds"].detach()
                    merged_video, recv = gaze_weighted_merge(video_det, scores_all, r)
                    merged_inputs = build_merged_inputs(base_qwen, cached, merged_video, recv)
                    pred_merge = forward_logits(qwen, merged_inputs)[oids].argmax().item()
                else:
                    pred_merge = pred_full

                s = stats[src]
                s["n"] += 1
                s["c_full"]  += int(pred_full  == gt)
                s["c_merge"] += int(pred_merge == gt)
            except Exception as e:
                print(f"{tag} idx={idx} src={src} ERR {e!r}", flush=True)
                continue

    def acc(c, n): return 100.0 * c / max(1, n)
    sg, eg, hd = stats["sg"], stats["eg"], stats["hd"]
    tot_n  = sg["n"] + eg["n"] + hd["n"]
    tot_cf = sg["c_full"] + eg["c_full"] + hd["c_full"]
    tot_cm = sg["c_merge"] + eg["c_merge"] + hd["c_merge"]

    print(f"\n==================== {tag} ====================", flush=True)
    print(f"  StreamGaze egtea  : n={sg['n']:5d}  "
          f"full={acc(sg['c_full'],sg['n']):.2f}%  merge={acc(sg['c_merge'],sg['n']):.2f}%")
    print(f"  EgoGazeVQA egtea  : n={eg['n']:5d}  "
          f"full={acc(eg['c_full'],eg['n']):.2f}%  merge={acc(eg['c_merge'],eg['n']):.2f}%")
    print(f"  HD-EPIC P09       : n={hd['n']:5d}  "
          f"full={acc(hd['c_full'],hd['n']):.2f}%  merge={acc(hd['c_merge'],hd['n']):.2f}%")
    print(f"  COMBINED          : n={tot_n:5d}  "
          f"full={acc(tot_cf,tot_n):.2f}%  merge={acc(tot_cm,tot_n):.2f}%")
    print(f"==============================================\n", flush=True)


if __name__ == "__main__":
    main()
