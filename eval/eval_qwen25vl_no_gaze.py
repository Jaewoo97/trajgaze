"""
Evaluate Qwen2.5-VL-7B-Instruct WITHOUT AutoGaze on StreamGaze
present_future_action_prediction — all samples.

Standard Qwen2.5-VL inference: all video tokens kept, no spatial pruning.
16 frames at 224×224. Logs per-sample VLM inference time and
per-dataset accuracy (EGTEA, EgoExoLearn, HoloAssist).

Usage:
  /workspace/vila_eval_env/bin/python eval/eval_qwen25vl_no_gaze.py \
      --frames-dir /workspace/datasets/StreamGaze_v2/frames \
      --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(__file__))
from metrics import save_results, build_frames_map

# ── Constants ─────────────────────────────────────────────────────────────────
QWEN_MODEL_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
QA_FILE = "/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json"

N_FRAMES       = 64    # frames per clip
FRAME_SIZE     = 224   # 224×224 → 16×16=256 patches per frame

FRAMES_MAP: dict[str, str] = {}


def parse_response_time(response_time: str) -> tuple[float, float]:
    times = re.findall(r"(\d+):(\d+)", response_time)
    start = int(times[0][0]) * 60 + int(times[0][1])
    end   = int(times[1][0]) * 60 + int(times[1][1])
    return float(start), float(end)


def get_segment_dir(video_path: str, start: float, end: float, tmp_dir: str) -> str | None:
    frame_dir = FRAMES_MAP.get(video_path)
    if frame_dir is None:
        return None

    start_idx = max(0, int(start * 10))
    end_idx   = int(end * 10)

    all_frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    selected = [
        f for f in all_frames
        if start_idx <= (int(f.split("_")[1].split(".")[0]) - 1) <= end_idx
    ]
    if not selected:
        return None

    stem    = video_path.replace(".mp4", "")
    seg_dir = os.path.join(tmp_dir, f"{stem}_{int(start)}_{int(end)}")
    os.makedirs(seg_dir, exist_ok=True)
    for i, fname in enumerate(selected):
        dst = os.path.join(seg_dir, f"frame_{i:06d}.jpg")
        if not os.path.lexists(dst):
            os.symlink(os.path.join(frame_dir, fname), dst)
    return seg_dir


def load_frames_pil(seg_dir: str, n: int = 16, size: int = 224) -> list[Image.Image]:
    """Load n evenly-sampled frames from seg_dir, resized to size×size."""
    paths = sorted(p for p in os.listdir(seg_dir) if p.endswith(".jpg"))
    if not paths:
        raise ValueError(f"No frames in {seg_dir}")
    indices = np.round(np.linspace(0, len(paths) - 1, n)).astype(int)

    def _load(i: int) -> Image.Image:
        return Image.open(os.path.join(seg_dir, paths[i])).convert("RGB").resize(
            (size, size), Image.BILINEAR
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        frames = list(pool.map(_load, indices))
    return frames


def load_model():
    print(f"Loading Qwen2.5-VL-7B from {QWEN_MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
    return processor, model


def run_inference(
    processor,
    model,
    seg_dir: str,
    question: str,
    options: list[str],
) -> tuple[str, float]:
    # Load 16 frames at 224×224
    frames = load_frames_pil(seg_dir, n=N_FRAMES, size=FRAME_SIZE)

    options_text = "\n".join(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "resized_height": FRAME_SIZE,
                    "resized_width":  FRAME_SIZE,
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        return_tensors="pt",
    )

    dev = model.get_input_embeddings().weight.device
    inputs = {
        k: v.to(dev) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - t0

    # Slice off prompt tokens — generate returns input + generated tokens
    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return response, inference_time


def main():
    global FRAMES_MAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir",  required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    FRAMES_MAP = build_frames_map(args.frames_dir)
    print(f"Loaded frame map: {len(FRAMES_MAP)} videos from {args.frames_dir}")

    out_dir = os.path.abspath(args.output)

    with open(QA_FILE) as f:
        samples = json.load(f)
    print(f"Total samples: {len(samples)}")
    if args.max_samples:
        samples = samples[: args.max_samples]

    processor, model = load_model()

    results, skipped, inference_times = [], 0, []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for sample in tqdm(samples, desc="Evaluating (Qwen2.5-VL no gaze)"):
            start, end = parse_response_time(sample["response_time"])
            seg_dir = get_segment_dir(sample["video_path"], start, end, tmp_dir)
            if seg_dir is None:
                skipped += len(sample["questions"])
                continue
            for qa in sample["questions"]:
                try:
                    prediction, inf_time = run_inference(
                        processor, model, seg_dir, qa["question"], qa["options"]
                    )
                    inference_times.append(inf_time)
                except Exception:
                    traceback.print_exc()
                    prediction, inf_time = "", 0.0
                results.append({
                    "video_path":     sample["video_path"],
                    "timestamp":      qa["time_stamp"],
                    "response_time":  sample["response_time"],
                    "question":       qa["question"],
                    "options":        qa["options"],
                    "answer":         qa["answer"],
                    "prediction":     prediction,
                    "inference_time": inf_time,
                })

    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print(f"Done. Evaluated: {len(results)}, Skipped: {skipped}")
    if inference_times:
        print(
            f"Inference time — avg: {avg_time:.3f}s  "
            f"min: {min(inference_times):.3f}s  max: {max(inference_times):.3f}s"
        )

    save_results(
        results, out_dir, "qwen25vl_no_gaze_64f",
        frames_map=FRAMES_MAP,
        extra={
            "qwen_model":             QWEN_MODEL_PATH,
            "n_frames":               N_FRAMES,
            "frame_size":             FRAME_SIZE,
            "avg_inference_time_sec": round(avg_time, 4),
            "skipped":                skipped,
        },
    )


if __name__ == "__main__":
    main()
