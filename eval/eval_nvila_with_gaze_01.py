"""
Evaluate NVILA-8B-HD-Video WITH AutoGaze pruning (ratio=0.1) on StreamGaze
present_future_action_prediction — all samples.

gazing_ratio_tile=0.1, gazing_ratio_thumbnail=0.1 (prune both streams).
Logs per-sample VLM inference time.

Usage:
  /workspace/vila_eval_env/bin/python eval/eval_nvila_with_gaze_01.py \
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

sys.path.insert(0, os.path.dirname(__file__))
from metrics import save_results, build_frames_map

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

AUTOGAZE_CKPT = "/workspace/EgoGazeVQA/AutoGaze/ckpt"

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/irteam/.cache/huggingface/hub/nvidia_NVILA-8B-HD-Video"
QA_FILE    = "/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json"

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


def load_model():
    os.environ["AUTOGAZE_MODEL_ID"] = AUTOGAZE_CKPT
    print(f"AutoGaze checkpoint: {AUTOGAZE_CKPT}")
    print("Loading NVILA-HD-Video processor (gaze ratio=0.1 on tiles+thumbnails) ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        num_video_frames=128,
        num_video_frames_thumbnail=128,
        max_tiles_video=1,
        gazing_ratio_tile=0.1,
        gazing_ratio_thumbnail=1.0,
        task_loss_requirement_tile=0.7,
        task_loss_requirement_thumbnail=0.7,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
    )

    print("Loading NVILA-HD-Video model ...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_batch_size_siglip=16,
    )
    model.eval()
    print("Model loaded.")
    return processor, model


def run_inference(processor, model, seg_dir: str, question: str, options: list[str]) -> tuple[str, float]:
    video_token  = processor.tokenizer.video_token
    options_text = "\n".join(options)
    prompt = (
        f"{video_token}\n\n"
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )

    inputs = processor(text=prompt, videos=seg_dir, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - t0

    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return response, inference_time


def main():
    global FRAMES_MAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", required=True)
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
        for sample in tqdm(samples, desc="Evaluating (gaze=0.1)"):
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
    print(f"Inference time — avg: {avg_time:.3f}s  "
          f"min: {min(inference_times):.3f}s  max: {max(inference_times):.3f}s")

    save_results(results, out_dir, "nvila_with_gaze_01_24",
                 extra={
                     "autogaze": AUTOGAZE_CKPT,
                     "gazing_ratio_tile": 0.1,
                     "gazing_ratio_thumbnail": 1.0,
                     "task_loss_requirement": 0.7,
                     "avg_inference_time_sec": round(avg_time, 4),
                     "skipped": skipped,
                 })


if __name__ == "__main__":
    main()
