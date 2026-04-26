"""
Evaluate NVILA-8B-HD-Video on StreamGaze_v2 — Future Action Prediction task.

Conditions:
  - baseline  : 100% visual tokens (128 frames)
  - no_visual : text-only (no video)
  - random    : randomly select --keep-ratio fraction of frames (temporal selection)

Uses frames UP TO each question's time_stamp (no future frame leakage).
Reports accuracy per dataset (egtea, egoexolearn, holoassist) and overall.

Usage:
    /workspace/vila_eval_env/bin/python -m TrajGaze_v2.eval.evaluate_streamgaze_nvila \
        --condition baseline --gpu 1
    /workspace/vila_eval_env/bin/python -m TrajGaze_v2.eval.evaluate_streamgaze_nvila \
        --condition no_visual --gpu 3
    /workspace/vila_eval_env/bin/python -m TrajGaze_v2.eval.evaluate_streamgaze_nvila \
        --condition random --keep-ratio 0.10 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback

sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze/VILA")
sys.path.insert(0, "/workspace/EgoGazeVQA")

os.environ.setdefault(
    "AUTOGAZE_MODEL_ID",
    "/workspace/EgoGazeVQA/AutoGaze/exps/streamgaze_fold_b_ntp/checkpoint_latest_gaze",
)

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


MODEL_PATH  = "/home/irteam/.cache/huggingface/hub/nvidia_NVILA-8B-HD-Video"
QA_FILE     = "/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json"
FRAMES_BASE = "/workspace/datasets/StreamGaze_v2/frames"
RESULTS_DIR = "/workspace/EgoGazeVQA/TrajGaze_v2/eval_results/streamgaze"
DATASETS    = ["egtea", "egoexolearn", "holoassist"]
EXTRACTED_FPS = 10.0


# ── Frame helpers ──────────────────────────────────────────────────────────────

def parse_ts(ts: str) -> float:
    parts = [float(p) for p in ts.strip().split(":")]
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


def get_frames_up_to(stem: str, dataset: str, ts_sec: float) -> list[str]:
    """Return sorted frame paths up to ts_sec."""
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


def make_segment_dir(frame_paths: list[str], tmp_dir: str, key: str) -> str | None:
    """Symlink selected frame_paths into a temp directory for the NVILA processor."""
    if not frame_paths:
        return None
    seg_dir = os.path.join(tmp_dir, key)
    os.makedirs(seg_dir, exist_ok=True)
    for i, src in enumerate(frame_paths):
        dst = os.path.join(seg_dir, f"frame_{i:06d}.jpg")
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    return seg_dir


def sample_frames(frame_paths: list[str], n: int) -> list[str]:
    """Uniformly sample n paths from a list."""
    if not frame_paths:
        return []
    total = len(frame_paths)
    if total <= n:
        return frame_paths
    indices = [int(i * total / n) for i in range(n)]
    return [frame_paths[min(i, total - 1)] for i in indices]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(device: torch.device, n_frames: int = 128):
    print("Loading NVILA-8B-HD-Video processor ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        num_video_frames=n_frames,
        num_video_frames_thumbnail=n_frames,
        max_tiles_video=1,
        gazing_ratio_tile=1.0,
        gazing_ratio_thumbnail=1.0,
        task_loss_requirement_tile=None,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
    )
    print("Loading NVILA-8B-HD-Video model ...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        max_batch_size_siglip=16,
    )
    model.eval()
    print("Model loaded.")
    return processor, model


# ── Inference ──────────────────────────────────────────────────────────────────

def build_prompt(processor, question: str, options: list[str],
                 with_video: bool = True) -> str:
    video_token  = processor.tokenizer.video_token
    options_text = "\n".join(options)
    prefix = f"{video_token}\n\n" if with_video else ""
    return (
        f"{prefix}"
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )


def run_baseline(processor, model, seg_dir: str, question: str,
                 options: list[str]) -> tuple[str, float, int]:
    prompt = build_prompt(processor, question, options, with_video=True)
    inputs = processor(text=prompt, videos=seg_dir, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0
    resp = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    return resp, gen_time, n_tokens


def run_no_visual(processor, model, question: str,
                  options: list[str]) -> tuple[str, float, int]:
    prompt = build_prompt(processor, question, options, with_video=False)
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0
    resp = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    return resp, gen_time, n_tokens


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_answer(response: str) -> str:
    response = response.strip()
    for pat in [r"^([A-D])\b", r"\b([A-D])\b", r"\(([A-D])\)", r"Answer[:\s]+([A-D])"]:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


# ── Metrics ────────────────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["baseline", "no_visual", "random"], required=True)
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--n-frames",   type=int,   default=128)
    p.add_argument("--keep-ratio", type=float, default=0.10,
                   help="Fraction of frames to keep for --condition random")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.condition == "random":
        n_keep = max(1, int(math.ceil(args.keep_ratio * args.n_frames)))
        pct    = int(round(args.keep_ratio * 100))
        print(f"Random frame selection: {n_keep}/{args.n_frames} frames ({pct}%)")

    processor, model = load_model(device, n_frames=args.n_frames)

    with open(QA_FILE) as f:
        qa_data = json.load(f)

    # Flat list of items
    items = []
    for entry in qa_data:
        stem    = os.path.splitext(entry["video_path"])[0]
        dataset = find_dataset(stem)
        for q in entry["questions"]:
            items.append({
                "stem":       stem,
                "dataset":    dataset,
                "question":   q["question"],
                "options":    q["options"],
                "answer":     q["answer"].strip().upper(),
                "time_stamp": q.get("time_stamp", ""),
            })

    print(f"Total items: {len(items)}")
    for ds in DATASETS:
        print(f"  {ds}: {sum(1 for it in items if it['dataset'] == ds)}")

    results = []
    desc = f"NVILA-8B-HD [{args.condition}]"
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, item in enumerate(tqdm(items, desc=desc)):
            try:
                ts_sec      = parse_ts(item["time_stamp"]) if item["time_stamp"] else 9999.0
                frame_paths = get_frames_up_to(item["stem"], item["dataset"], ts_sec)

                if args.condition == "baseline":
                    # Sample n_frames uniformly from available frames
                    sampled = sample_frames(frame_paths, args.n_frames)
                    seg_dir = make_segment_dir(sampled, tmp_dir, f"{idx}_base")
                    if seg_dir is None:
                        raise RuntimeError("no frames")
                    response, gen_time, n_tok = run_baseline(
                        processor, model, seg_dir, item["question"], item["options"]
                    )

                elif args.condition == "no_visual":
                    response, gen_time, n_tok = run_no_visual(
                        processor, model, item["question"], item["options"]
                    )

                else:  # random
                    # Randomly select n_keep frames from available
                    if frame_paths:
                        indices = torch.randperm(len(frame_paths))[:n_keep].sort().values.tolist()
                        sampled = [frame_paths[i] for i in indices]
                    else:
                        sampled = []
                    seg_dir = make_segment_dir(sampled, tmp_dir, f"{idx}_rand")
                    if seg_dir is None:
                        raise RuntimeError("no frames")
                    response, gen_time, n_tok = run_baseline(
                        processor, model, seg_dir, item["question"], item["options"]
                    )

                pred    = extract_answer(response)
                correct = (pred == item["answer"])
                results.append({
                    "dataset":        item["dataset"],
                    "question":       item["question"],
                    "answer":         item["answer"],
                    "prediction":     pred,
                    "correct":        correct,
                    "generate_time":  gen_time,
                    "n_input_tokens": n_tok,
                })

            except Exception:
                traceback.print_exc()
                results.append({
                    "dataset":        item["dataset"],
                    "question":       item["question"],
                    "answer":         item["answer"],
                    "prediction":     "",
                    "correct":        False,
                    "generate_time":  0.0,
                    "n_input_tokens": 0,
                })

    if args.condition == "random":
        tag = f"streamgaze_nvila_random_{int(round(args.keep_ratio*100))}pct_{args.n_frames}f"
    elif args.condition == "baseline":
        tag = f"streamgaze_nvila_baseline_{args.n_frames}f"
    else:
        tag = "streamgaze_nvila_no_visual"

    with open(os.path.join(RESULTS_DIR, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    metrics = compute_metrics(results)
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ov = metrics["overall"]
    print(f"\n{'='*60}")
    print(f"  NVILA-8B-HD — Future Action Prediction [{args.condition}]")
    print(f"{'='*60}")
    print(f"  Overall       : {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"  Generate time : {ov['avg_generate_time_sec']:.4f}s")
    print(f"  Avg tokens    : {ov['avg_input_tokens']:.0f}")
    print(f"{'─'*60}")
    for ds, dm in sorted(metrics["per_dataset"].items()):
        print(f"  {ds:15s}: {dm['accuracy']:.2f}%  (n={dm['n_samples']})")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
