"""
Evaluate NVILA-8B-Video (Efficient-Large-Model, no AutoGaze/HD) on StreamGaze
present_future_action_prediction — all samples (EgoExoLearn + HoloAssist + EGTEA).

Requires pre-extracted frames from preprocess_frames.py.
All frames in the response_time window at 10 FPS are used; the model samples
num_video_frames uniformly from them.

Usage:
  /workspace/vila_eval_env/bin/python eval/eval_nvila8b_video.py \
      --frames-dir /workspace/datasets/StreamGaze_v2/frames \
      --output /workspace/EgoGazeVQA/AutoGaze/results/present_future_action_prediction \
      [--max-samples 20]
"""

import argparse
import json
import os
import re
import sys
import tempfile
import traceback

import torch
from tqdm import tqdm
from transformers import GenerationConfig

sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze/VILA")
sys.path.insert(0, "/workspace/EgoGazeVQA/eval")
import llava
from metrics import save_results, build_frames_map

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/irteam/.cache/huggingface/hub/Efficient-Large-Model_NVILA-8B-Video"
QA_FILE    = "/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json"

FRAMES_MAP: dict[str, str] = {}  # video filename → frame directory


def parse_response_time(response_time: str) -> tuple[float, float]:
    """Parse '[MM:SS - MM:SS]' → (start_sec, end_sec)."""
    times = re.findall(r"(\d+):(\d+)", response_time)
    start = int(times[0][0]) * 60 + int(times[0][1])
    end   = int(times[1][0]) * 60 + int(times[1][1])
    return float(start), float(end)


def get_segment_dir(video_path: str, start: float, end: float, tmp_dir: str) -> str | None:
    """Create a temp dir with symlinks to frames in [start, end] at 10 FPS.

    Frame N (1-indexed in filename) corresponds to time (N-1)/10 seconds.
    Returns the temp dir path, or None if no frames are found.
    """
    frame_dir = FRAMES_MAP.get(video_path)
    if frame_dir is None:
        return None

    start_idx = max(0, int(start * 10))  # 0-indexed, inclusive
    end_idx   = int(end * 10)            # 0-indexed, inclusive

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
    print("Loading NVILA-8B-Video ...")
    model = llava.load(MODEL_PATH, device_map="auto")
    model.config.num_video_frames = 128
    model.eval()
    print("Model loaded.")
    return model


def run_inference(model, seg_dir: str, question: str, options: list[str]) -> str:
    options_text = "\n".join(options)
    prompt = [
        llava.Video(seg_dir),
        (
            "You are watching a short first-person (egocentric) video clip.\n"
            f"Question: {question}\n\n"
            f"{options_text}\n\n"
            "Answer with only the letter (A, B, C, or D) of the correct option."
        ),
    ]
    generation_config = GenerationConfig(max_new_tokens=16, do_sample=False)
    return model.generate_content(prompt, generation_config=generation_config)


def main():
    global FRAMES_MAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True,
                        help="Pre-extracted frame root from preprocess_frames.py "
                             "(StreamGaze_v2/frames).")
    parser.add_argument("--output",
                        default="results/present_future_action_prediction/nvila8b_video",
                        help="Output directory for predictions and metrics.")
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

    model = load_model()

    results, skipped = [], 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for sample in tqdm(samples, desc="Evaluating"):
            start, end = parse_response_time(sample["response_time"])
            seg_dir = get_segment_dir(sample["video_path"], start, end, tmp_dir)
            if seg_dir is None:
                skipped += len(sample["questions"])
                continue
            for qa in sample["questions"]:
                try:
                    prediction = run_inference(model, seg_dir, qa["question"], qa["options"])
                except Exception:
                    traceback.print_exc()
                    prediction = ""
                results.append({
                    "video_path":    sample["video_path"],
                    "timestamp":     qa["time_stamp"],
                    "response_time": sample["response_time"],
                    "question":      qa["question"],
                    "options":       qa["options"],
                    "answer":        qa["answer"],
                    "prediction":    prediction,
                })

    print(f"Done. Evaluated: {len(results)}, Skipped: {skipped}")
    save_results(results, out_dir, "nvila8b_video",
                 extra={"model": "NVILA-8B-Video", "skipped": skipped})


if __name__ == "__main__":
    main()
