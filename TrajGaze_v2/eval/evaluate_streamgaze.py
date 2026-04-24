"""
Evaluate Qwen2.5-VL-7B on StreamGaze_v2 — Future Action Prediction task.

Conditions:
  - baseline   : 100% visual tokens (all frames)
  - no_visual  : text-only (no video input)
  - random     : random K% spatial token selection (--keep-ratio, default 0.10)

Reports accuracy per dataset (egtea, egoexolearn, holoassist) and overall.

Usage:
    /opt/conda/envs/gaze/bin/python -m TrajGaze_v2.eval.evaluate_streamgaze \
        --condition baseline --gpu 0
    /opt/conda/envs/gaze/bin/python -m TrajGaze_v2.eval.evaluate_streamgaze \
        --condition no_visual --gpu 0
    /opt/conda/envs/gaze/bin/python -m TrajGaze_v2.eval.evaluate_streamgaze \
        --condition random --keep-ratio 0.10 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, "/workspace/EgoGazeVQA")

import torch
from PIL import Image
from tqdm import tqdm

import math
from TrajGaze_v2.training.stage2 import (
    load_qwen, FRAME_SIZE, qwen_generate_with_mask,
)
from TrajGaze_v2.models.model import N_PATCHES

QA_FILE    = "/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json"
FRAMES_BASE = "/workspace/datasets/StreamGaze_v2/frames"
RESULTS_DIR = "/workspace/EgoGazeVQA/TrajGaze_v2/eval_results/streamgaze"
DATASETS    = ["egtea", "egoexolearn", "holoassist"]

# Frames are extracted at ~10 fps from original video
EXTRACTED_FPS = 10.0


def parse_ts(ts: str) -> float:
    """Convert 'MM:SS' or 'HH:MM:SS' timestamp to seconds."""
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def find_dataset(stem: str) -> str:
    for ds in DATASETS:
        if os.path.isdir(os.path.join(FRAMES_BASE, ds, "viz", stem)):
            return ds
    return "unknown"


def get_frame_paths_up_to(stem: str, dataset: str, ts_sec: float) -> list[str]:
    """Return all extracted frame paths up to the given timestamp."""
    frame_dir = os.path.join(FRAMES_BASE, dataset, "viz", stem)
    if not os.path.isdir(frame_dir):
        return []
    all_frames = sorted(
        f for f in os.listdir(frame_dir) if f.endswith(".jpg")
    )
    # Frame number is 1-indexed; convert timestamp to last frame index
    last_frame_idx = int(ts_sec * EXTRACTED_FPS)  # number of frames up to timestamp
    # Frame files are named frame_000001.jpg → index 1
    cutoff = last_frame_idx  # keep frames with index <= cutoff
    paths = []
    for fname in all_frames:
        m = re.match(r"frame_(\d+)\.jpg", fname)
        if m and int(m.group(1)) <= max(cutoff, 1):
            paths.append(os.path.join(frame_dir, fname))
    return paths


def sample_frames(frame_paths: list[str], n: int) -> list[str]:
    """Uniformly sample n paths from a list."""
    if not frame_paths:
        return []
    total = len(frame_paths)
    indices = [int(i * total / n) for i in range(n)]
    return [frame_paths[min(i, total - 1)] for i in indices]


def extract_answer(response: str) -> str:
    response = response.strip()
    for pat in [r"^([A-D])\b", r"\b([A-D])\b", r"\(([A-D])\)", r"Answer[:\s]+([A-D])"]:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def build_options_text(options: list[str]) -> str:
    """Options may already have 'A. ...' prefix or may be bare strings."""
    lines = []
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        # If already prefixed with letter, keep as-is; otherwise add prefix
        if re.match(r"^[A-D][.\)]", opt.strip()):
            lines.append(opt.strip())
        else:
            lines.append(f"{letter}. {opt.strip()}")
    return "\n".join(lines)


def preprocess_streamgaze_item(qwen_processor, qwen_model, frame_paths, question,
                               options, n_frames=32, device=None):
    """
    Tokenise one StreamGaze item and cache video embeddings for reuse with masks.
    Returns a cached dict (same format as preprocess_qwen_item in stage2.py).
    """
    from qwen_vl_utils import process_vision_info

    sampled = sample_frames(frame_paths, n_frames)
    frames = []
    for p in sampled:
        try:
            img = Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
            frames.append(img)
        except Exception:
            pass
    if not frames:
        return None

    options_text = build_options_text(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
        {"type": "text", "text": user_text},
    ]}]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        **video_kwargs, return_tensors="pt"
    )
    dev = qwen_model.get_input_embeddings().weight.device
    input_ids      = inputs["input_ids"].to(dev)
    attention_mask = inputs["attention_mask"].to(dev)
    pv_vid   = inputs["pixel_values_videos"].to(dev, torch.bfloat16)
    grid_thw = inputs["video_grid_thw"].to(dev)

    with torch.inference_mode():
        video_embeds = qwen_model.model.visual(pv_vid, grid_thw=grid_thw)

    T_merged = int(grid_thw[:, 0].sum().item())
    video_token_id = qwen_model.config.video_token_id
    flat_ids = input_ids[0]
    video_positions = (flat_ids == video_token_id).nonzero(as_tuple=True)[0]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "grid_thw":       grid_thw,
        "video_embeds":   video_embeds,
        "T_merged":       T_merged,
        "video_positions": video_positions,
        "emb_dev":        dev,
    }


def run_baseline(qwen_processor, qwen_model, frame_paths, question, options,
                 n_frames=32, device=None):
    """Run Qwen with all visual tokens. Returns (pred, generate_time, n_tokens)."""
    from qwen_vl_utils import process_vision_info

    sampled = sample_frames(frame_paths, n_frames)
    frames = []
    for p in sampled:
        try:
            img = Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
            frames.append(img)
        except Exception:
            pass
    if not frames:
        return "", 0.0, 0

    options_text = build_options_text(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
        {"type": "text", "text": user_text},
    ]}]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        **video_kwargs, return_tensors="pt"
    )
    dev = qwen_model.get_input_embeddings().weight.device
    input_ids      = inputs["input_ids"].to(dev)
    attention_mask = inputs["attention_mask"].to(dev)
    pv_vid   = inputs["pixel_values_videos"].to(dev, torch.bfloat16)
    grid_thw = inputs["video_grid_thw"].to(dev)
    n_tokens = input_ids.shape[1]

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = qwen_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values_videos=pv_vid, video_grid_thw=grid_thw,
            max_new_tokens=16, do_sample=False,
        )
        torch.cuda.synchronize()
        generate_time = time.perf_counter() - t0

    gen = out[:, input_ids.shape[1]:]
    resp = qwen_processor.batch_decode(gen, skip_special_tokens=True)
    return (resp[0] if resp else ""), generate_time, n_tokens


def run_no_visual(qwen_processor, qwen_model, question, options, device=None):
    """Run Qwen with text only (no video). Returns (pred, generate_time, n_tokens)."""
    options_text = build_options_text(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": user_text},
    ]}]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_processor(text=[text], return_tensors="pt")
    dev = qwen_model.get_input_embeddings().weight.device
    input_ids      = inputs["input_ids"].to(dev)
    attention_mask = inputs["attention_mask"].to(dev)
    n_tokens = input_ids.shape[1]

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = qwen_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=16, do_sample=False,
        )
        torch.cuda.synchronize()
        generate_time = time.perf_counter() - t0

    gen = out[:, input_ids.shape[1]:]
    resp = qwen_processor.batch_decode(gen, skip_special_tokens=True)
    return (resp[0] if resp else ""), generate_time, n_tokens


def compute_metrics(results: list[dict]) -> dict:
    by_ds: dict[str, list] = {}
    correct_all = 0
    for r in results:
        ok = r.get("correct", False)
        correct_all += int(ok)
        by_ds.setdefault(r["dataset"], []).append(ok)

    overall_acc = 100.0 * correct_all / max(1, len(results))
    avg_gen     = sum(r.get("generate_time", 0) for r in results) / max(1, len(results))
    avg_tokens  = sum(r.get("n_input_tokens", 0) for r in results) / max(1, len(results))

    per_ds = {
        ds: {"accuracy": 100.0 * sum(f) / max(1, len(f)), "n_samples": len(f)}
        for ds, f in sorted(by_ds.items())
    }
    return {
        "overall": {
            "accuracy": overall_acc,
            "n_samples": len(results),
            "avg_generate_time_sec": round(avg_gen, 4),
            "avg_input_tokens": round(avg_tokens, 1),
        },
        "per_dataset": per_ds,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["baseline", "no_visual", "random"], required=True)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--n-frames",   type=int,   default=32)
    p.add_argument("--keep-ratio", type=float, default=0.10,
                   help="Fraction of spatial patches to keep for --condition random")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--qa-file",    type=str,   default=QA_FILE,
                   help="Path to QA JSON file (default: future action prediction)")
    p.add_argument("--task-name",  type=str,   default=None,
                   help="Short task name used in output filenames (inferred from qa-file if not set)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.condition == "random":
        n_keep = max(1, int(math.ceil(args.keep_ratio * N_PATCHES)))
        pct    = int(round(args.keep_ratio * 100))
        print(f"Random selection: {n_keep}/{N_PATCHES} spatial patches ({pct}%) per frame")

    print("Loading Qwen2.5-VL-7B ...")
    qwen_processor, qwen_model = load_qwen(device)
    print("Qwen loaded.")

    task_name = args.task_name or os.path.splitext(os.path.basename(args.qa_file))[0]

    with open(args.qa_file) as f:
        qa_data = json.load(f)

    # Build flat list of items: one per question
    items = []
    for entry in qa_data:
        stem    = os.path.splitext(entry["video_path"])[0]
        dataset = find_dataset(stem)
        for q in entry["questions"]:
            items.append({
                "stem":    stem,
                "dataset": dataset,
                "question": q["question"],
                "options":  q["options"],
                "answer":   q["answer"].strip().upper(),
                "time_stamp": q.get("time_stamp", ""),
            })

    print(f"Total items: {len(items)}")
    for ds in DATASETS:
        n = sum(1 for it in items if it["dataset"] == ds)
        print(f"  {ds}: {n}")

    results = []
    desc = f"StreamGaze [{args.condition}]"
    for item in tqdm(items, desc=desc):
        try:
            ts_sec = parse_ts(item["time_stamp"]) if item["time_stamp"] else 9999.0
            frame_paths = get_frame_paths_up_to(item["stem"], item["dataset"], ts_sec)

            if args.condition == "baseline":
                response, gen_time, n_tok = run_baseline(
                    qwen_processor, qwen_model,
                    frame_paths, item["question"], item["options"],
                    n_frames=args.n_frames, device=device,
                )
            elif args.condition == "no_visual":
                response, gen_time, n_tok = run_no_visual(
                    qwen_processor, qwen_model,
                    item["question"], item["options"], device=device,
                )
            else:  # random
                cached = preprocess_streamgaze_item(
                    qwen_processor, qwen_model,
                    frame_paths, item["question"], item["options"],
                    n_frames=args.n_frames, device=device,
                )
                if cached is None:
                    raise RuntimeError("preprocess returned None")
                perm = torch.randperm(N_PATCHES, device=device)
                mask = torch.zeros(N_PATCHES, dtype=torch.bool, device=device)
                mask[perm[:n_keep]] = True
                gen_ids, gen_time, n_tok = qwen_generate_with_mask(qwen_model, cached, mask)
                resp_list = qwen_processor.batch_decode(gen_ids, skip_special_tokens=True)
                response  = resp_list[0] if resp_list else ""

            pred    = extract_answer(response)
            correct = (pred == item["answer"])
            results.append({
                "dataset":       item["dataset"],
                "question":      item["question"],
                "answer":        item["answer"],
                "prediction":    pred,
                "correct":       correct,
                "generate_time": gen_time,
                "n_input_tokens": n_tok,
            })
        except Exception:
            traceback.print_exc()
            results.append({
                "dataset":       item["dataset"],
                "question":      item["question"],
                "answer":        item["answer"],
                "prediction":    "",
                "correct":       False,
                "generate_time": 0.0,
                "n_input_tokens": 0,
            })

    if args.condition == "random":
        tag = f"{task_name}_random_{int(round(args.keep_ratio*100))}pct_{args.n_frames}f"
    elif args.condition == "baseline":
        tag = f"{task_name}_baseline_{args.n_frames}f"
    else:
        tag = f"{task_name}_{args.condition}"
    with open(os.path.join(RESULTS_DIR, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    metrics = compute_metrics(results)
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ov = metrics["overall"]
    print(f"\n{'='*55}")
    print(f"  StreamGaze — {task_name} [{args.condition}]")
    print(f"{'='*55}")
    print(f"  Overall       : {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"  Generate time : {ov['avg_generate_time_sec']:.4f}s")
    print(f"  Avg tokens    : {ov['avg_input_tokens']:.0f}")
    print(f"{'─'*55}")
    for ds, dm in sorted(metrics["per_dataset"].items()):
        print(f"  {ds:15s}: {dm['accuracy']:.2f}%  (n={dm['n_samples']})")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
