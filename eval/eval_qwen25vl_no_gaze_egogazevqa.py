"""
Evaluate Qwen2.5-VL-7B-Instruct on EgoGazeVQA (gaze-overlay frames).

All three datasets (ego4d, egoexo, egtea) from all_gaze_v1.
64 frames at 224×224. Reports per-dataset × per-qa_type accuracy table.

Supports --visual-token-ratio to randomly keep only a fraction of visual
tokens (applied after the ViT encoder + patch merger, before the LLM).

Usage:
  python eval/eval_qwen25vl_gaze_egogazevqa.py \
      --data-root /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
      --output results/egogazevqa

  # Keep only 10% of visual tokens (random):
  python eval/eval_qwen25vl_gaze_egogazevqa.py \
      --data-root /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
      --output results/egogazevqa_vt10 \
      --visual-token-ratio 0.1
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(__file__))
from metrics import extract_choice

# ── Constants ─────────────────────────────────────────────────────────────────
QWEN_MODEL_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_DATA_ROOT = "/home/yujin/dataset/EgoGazeVQA/all_gaze_v1"

N_FRAMES   = 64
FRAME_SIZE = 224


# ── Data loading ──────────────────────────────────────────────────────────────
def load_samples(data_root: str) -> list[dict]:
    """Parse metadata.csv and return a flat list of QA samples."""
    csv_path = os.path.join(data_root, "metadata.csv")
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # file_name: ego4d/b70b1c34-.../123_1205.mp4
            parts = row["file_name"].split("/")
            dataset = parts[0]              # ego4d / egoexo / egtea
            video_id = parts[1]             # UUID or folder name
            clip_stem = parts[2].replace(".mp4", "")  # 123_1205

            frame_dir = os.path.join(data_root, dataset, "no_gaze", video_id)
            clip_prefix = clip_stem + "_"   # 123_1205_

            options = row["answer_options"].split("|")

            samples.append({
                "file_name":      row["file_name"],
                "video_id":       video_id,
                "dataset":        dataset,
                "qa_type":        row["qa_type"],
                "question":       row["question"],
                "options":        options,
                "correct_answer": row["correct_answer"].strip().upper(),
                "frame_dir":      frame_dir,
                "clip_prefix":    clip_prefix,
            })
    return samples


def get_clip_frames(frame_dir: str, clip_prefix: str) -> list[str]:
    """Return sorted list of full paths for frames matching clip_prefix."""
    if not os.path.isdir(frame_dir):
        return []
    frames = [
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.startswith(clip_prefix) and f.endswith(".jpg")
    ]
    # Sort by the frame number (last numeric part before .jpg)
    frames.sort(key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
    return frames


# ── Frame loading ─────────────────────────────────────────────────────────────
def load_frames_pil(
    frame_paths: list[str], n: int = 64, size: int = 224
) -> list[Image.Image]:
    """Uniformly sample n frames from frame_paths, resized to size×size."""
    if not frame_paths:
        raise ValueError("No frame paths provided")
    indices = np.round(np.linspace(0, len(frame_paths) - 1, n)).astype(int)

    def _load(i: int) -> Image.Image:
        return (
            Image.open(frame_paths[i])
            .convert("RGB")
            .resize((size, size), Image.BILINEAR)
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        frames = list(pool.map(_load, indices))
    return frames


# ── Visual token pruning ──────────────────────────────────────────────────────
def _make_visual_token_prune_hook(keep_ratio: float):
    """Return a forward hook that zeros out (1 - keep_ratio) of visual tokens.

    Attached to model.model.visual (the ViT encoder).  The hook modifies
    ``pooler_output`` — the merged visual tokens — by zeroing out a random
    subset.  Token count stays the same (placeholder alignment preserved),
    but zeroed tokens carry no visual information to the LLM.
    """
    def hook(_module, _input, output):
        pooler = output.pooler_output          # (N_total_tokens, hidden)
        n = pooler.shape[0]
        n_keep = max(1, int(n * keep_ratio))
        if n_keep >= n:
            return output

        perm = torch.randperm(n, device=pooler.device)
        mask = torch.zeros(n, 1, device=pooler.device, dtype=pooler.dtype)
        mask[perm[:n_keep]] = 1.0
        output.pooler_output = pooler * mask
        return output

    return hook


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(visual_token_ratio: float = 1.0):
    print(f"Loading Qwen2.5-VL-7B from {QWEN_MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if visual_token_ratio < 1.0:
        hook = _make_visual_token_prune_hook(visual_token_ratio)
        model.model.visual.register_forward_hook(hook)
        print(f"Visual token pruning enabled: keeping {visual_token_ratio*100:.0f}% of tokens (random).")

    print("Model loaded.")
    return processor, model


def run_inference(
    processor,
    model,
    frames: list[Image.Image],
    question: str,
    options: list[str],
) -> tuple[str, float]:
    options_text = "\n".join(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
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
    # process_vision_info may return list-wrapped scalars (e.g. fps=[2.0]);
    # the processor expects plain scalars.
    for k, v in video_kwargs.items():
        if isinstance(v, list) and len(v) == 1:
            video_kwargs[k] = v[0]
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

    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return response, inference_time


# ── Metrics & reporting ───────────────────────────────────────────────────────
def compute_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(
        1
        for r in results
        if extract_choice(r["prediction"]) == r["correct_answer"]
    )
    return round(correct / len(results) * 100, 2)


def print_accuracy_table(results: list[dict]) -> dict:
    """Print a dataset × qa_type accuracy table and return metrics dict."""
    datasets = sorted({r["dataset"] for r in results})
    qa_types = ["causal", "spatial", "temporal"]

    # Build buckets: (dataset, qa_type) -> list[result]
    buckets = defaultdict(list)
    for r in results:
        buckets[(r["dataset"], r["qa_type"])].append(r)
        buckets[(r["dataset"], "overall")].append(r)
        buckets[("total", r["qa_type"])].append(r)
        buckets[("total", "overall")].append(r)

    cols = ["overall"] + qa_types
    col_w = 10
    ds_w = max(len(d) for d in datasets + ["total", "Dataset"]) + 2

    # Header
    header = f"{'Dataset':<{ds_w}}" + "".join(f"{c:>{col_w}}" for c in ["Overall"] + [t.capitalize() for t in qa_types])
    sep = "─" * len(header)
    print(f"\n{'=== Accuracy Results ===':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    metrics = {}
    for ds in datasets:
        row = f"{ds:<{ds_w}}"
        ds_metrics = {}
        for col in cols:
            acc = compute_accuracy(buckets[(ds, col)])
            n = len(buckets[(ds, col)])
            row += f"{f'{acc:.2f}%':>{col_w}}"
            ds_metrics[col] = {"accuracy": acc, "n_samples": n}
        print(row)
        metrics[ds] = ds_metrics

    print(sep)
    # Total row
    row = f"{'Total':<{ds_w}}"
    total_metrics = {}
    for col in cols:
        acc = compute_accuracy(buckets[("total", col)])
        n = len(buckets[("total", col)])
        row += f"{f'{acc:.2f}%':>{col_w}}"
        total_metrics[col] = {"accuracy": acc, "n_samples": n}
    print(row)
    print(sep + "\n")
    metrics["total"] = total_metrics

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--visual-token-ratio", type=float, default=1.0,
        help="Fraction of visual tokens to keep (0.0-1.0). "
             "E.g. 0.1 = randomly keep 10%%, zero out the rest.",
    )
    args = parser.parse_args()

    vt_ratio = args.visual_token_ratio
    assert 0.0 < vt_ratio <= 1.0, "--visual-token-ratio must be in (0, 1]"

    samples = load_samples(args.data_root)
    print(f"Total QA samples: {len(samples)}")
    if args.max_samples:
        samples = samples[: args.max_samples]

    processor, model = load_model(visual_token_ratio=vt_ratio)

    out_dir = os.path.abspath(args.output)
    results, skipped, inference_times = [], 0, []

    for sample in tqdm(samples, desc="Evaluating (Qwen2.5-VL gaze EgoGazeVQA)"):
        frame_paths = get_clip_frames(sample["frame_dir"], sample["clip_prefix"])
        if not frame_paths:
            skipped += 1
            continue

        try:
            frames = load_frames_pil(frame_paths, n=N_FRAMES, size=FRAME_SIZE)
            prediction, inf_time = run_inference(
                processor, model, frames, sample["question"], sample["options"]
            )
            inference_times.append(inf_time)
        except Exception:
            traceback.print_exc()
            prediction, inf_time = "", 0.0

        results.append({
            "file_name":      sample["file_name"],
            "video_id":       sample["video_id"],
            "dataset":        sample["dataset"],
            "qa_type":        sample["qa_type"],
            "question":       sample["question"],
            "options":        sample["options"],
            "correct_answer": sample["correct_answer"],
            "prediction":     prediction,
            "inference_time": inf_time,
        })

    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print(f"\nDone. Evaluated: {len(results)}, Skipped: {skipped}")
    if inference_times:
        print(
            f"Inference time — avg: {avg_time:.3f}s  "
            f"min: {min(inference_times):.3f}s  max: {max(inference_times):.3f}s"
        )

    # Print accuracy table
    metrics = print_accuracy_table(results)

    # Save results
    vt_pct = int(vt_ratio * 100)
    stem = f"qwen25vl_gaze_egogazevqa_64f_vt{vt_pct}"
    os.makedirs(out_dir, exist_ok=True)

    pred_path = os.path.join(out_dir, f"{stem}_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions → {pred_path}")

    metrics_out = {
        "accuracy_table": metrics,
        "qwen_model":             QWEN_MODEL_PATH,
        "n_frames":               N_FRAMES,
        "frame_size":             FRAME_SIZE,
        "visual_token_ratio":     vt_ratio,
        "avg_inference_time_sec": round(avg_time, 4),
        "total_evaluated":        len(results),
        "skipped":                skipped,
    }
    metrics_path = os.path.join(out_dir, f"{stem}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics    → {metrics_path}")


if __name__ == "__main__":
    main()
