"""
Evaluate TrajGaze with a RANDOMLY INITIALIZED model (no trained weights).

This gives the baseline: what accuracy does the pipeline achieve when the
frame selector and patch decoder are untrained (random)?

Runs on a single GPU (no DDP needed).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m TrajGaze.training.eval_random_baseline \
        --adapted-dir  /workspace/datasets/StreamGaze_v2/adapted \
        --interact-dir /workspace/datasets/StreamGaze_v2/interaction \
        --frames-dir   /workspace/datasets/StreamGaze_v2/frames \
        --qa-file      /workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json \
        --val-datasets egtea \
        [--ckpt        path/to/checkpoint.pt]   # optional: load a specific ckpt instead of random
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

import torch
from tqdm import tqdm

from TrajGaze.training.stage2 import (
    Stage2Dataset,
    VLMOracle,
    _select_patches,
    _timeout_handler,
    VAL_DATASETS,
)
from TrajGaze.training.stage1 import TrajGazeModel


def _validate_single(model, ds, oracle, device, args, max_samples=100):
    model.eval()
    n_correct = n_total = 0
    latencies = []

    indices = list(range(min(max_samples, len(ds))))

    for idx in tqdm(indices, desc="Val"):
        try:
            item = ds[idx]
        except Exception as e:
            print(f"  skip idx={idx}: {e}")
            continue

        tokenizer_keys = [
            "gaze_pos", "gaze_speed", "gaze_mask",
            "left_pos", "left_vel", "left_mask",
            "right_pos", "right_vel", "right_mask",
            "d_left", "d_right", "v_rel_left", "v_rel_right",
            "convergence", "lead_lag",
        ]

        tok_in = {k: item[k].to(device) for k in tokenizer_keys}
        tok_gaze, tok_left, tok_right, tok_interact = model.tokenizer(**tok_in)
        present = tok_in["gaze_mask"] | tok_in["left_mask"] | tok_in["right_mask"]

        # Same frame selection ratio as validation in stage2.py
        _, attended_idx = model.selector.select_frames(tok_interact, ratio=args.frame_ratio)
        attended_idx = attended_idx[0]

        if oracle.model is not None and attended_idx:
            all_patches = _select_patches(
                model, attended_idx, item,
                tok_gaze, tok_left, tok_right, tok_interact, present, device,
            )

            eval_paths   = []
            eval_patches = []
            for pos, t in enumerate(attended_idx):
                if t < len(item["frame_names"]):
                    fpath = os.path.join(item["frames_dir"], item["frame_names"][t])
                    if os.path.exists(fpath):
                        eval_paths.append(fpath)
                        eval_patches.append(all_patches[pos])

            if eval_paths:
                prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(120)
                try:
                    c = oracle.score(
                        eval_paths, eval_patches,
                        item["question"], item["options"], item["gt_answer"],
                    )
                    n_correct += c
                    gen_time = getattr(oracle, "_last_generate_time", None)
                    if gen_time is not None:
                        latencies.append(gen_time)
                except TimeoutError:
                    print(f"  WARNING: sample {idx} timed out, skipping.")
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev_handler)

        n_total += 1

    acc = n_correct / max(1, n_total)
    if latencies:
        import numpy as np
        lats = sorted(latencies)
        print(f"VLM latency: mean={sum(lats)/len(lats):.2f}s  "
              f"median={lats[len(lats)//2]:.2f}s  "
              f"p90={lats[int(len(lats)*0.9)]:.2f}s  (n={len(lats)})")
    print(f"\nResult: {n_correct}/{n_total} correct = {100*acc:.1f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-dir",  required=True)
    parser.add_argument("--interact-dir", required=True)
    parser.add_argument("--frames-dir",   required=True)
    parser.add_argument("--qa-file",      required=True)
    parser.add_argument("--val-datasets", nargs="+", default=VAL_DATASETS)
    parser.add_argument("--vlm",          default="nvila", choices=["nvila", "qwen", "none"])
    parser.add_argument("--max-frames",   type=int, default=200)
    parser.add_argument("--max-samples",  type=int, default=100)
    parser.add_argument("--frame-ratio",  type=float, default=0.50,
                        help="Fraction of frames to attend (default: 0.50)")
    parser.add_argument("--ckpt",         default=None,
                        help="Optional checkpoint to load (default: random init)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build model
    model = TrajGazeModel(args).to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("Using RANDOMLY INITIALIZED model (no checkpoint loaded).")

    # VLM oracle
    os.environ.setdefault("AUTOGAZE_MODEL_ID", "/workspace/EgoGazeVQA/AutoGaze/ckpt")
    oracle = VLMOracle(backend=args.vlm, local_rank=0)
    if args.vlm != "none":
        oracle.load()

    # Val dataset
    val_ds = Stage2Dataset(
        qa_file      = args.qa_file,
        adapted_dir  = args.adapted_dir,
        interact_dir = args.interact_dir,
        frames_dir   = args.frames_dir,
        datasets     = args.val_datasets,
        max_frames   = args.max_frames,
    )
    print(f"Val dataset: {len(val_ds)} samples (evaluating up to {args.max_samples})")

    label = "Random init" if not args.ckpt else f"Checkpoint: {os.path.basename(args.ckpt)}"
    print(f"=== {label} ===")
    _validate_single(model, val_ds, oracle, device, args, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
