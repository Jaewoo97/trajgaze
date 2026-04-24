"""
Baseline: Qwen2.5-VL-7B on EGTEA with ALL visual tokens (no selector).
Times only the generate() call to isolate VLM inference cost.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
sys.path.insert(0, "/workspace/EgoGazeVQA")

import torch
from tqdm import tqdm
from TrajGaze_v2.data.dataset import TrajGazeV2QADataset
from TrajGaze_v2.training.stage2 import (
    load_qwen, load_frames_pil, extract_answer, QWEN_MODEL_PATH, FRAME_SIZE
)
from qwen_vl_utils import process_vision_info

RESULTS_DIR = "/workspace/EgoGazeVQA/TrajGaze_v2/eval_results"


def qwen_full_tokens(qwen_processor, qwen_model, frame_paths, question, options,
                     n_qwen_frames=64, device=None):
    """Run Qwen with all visual tokens. Returns (response, generate_time, n_input_tokens)."""
    frames = load_frames_pil(frame_paths, n_qwen_frames)
    if not frames:
        return "", 0.0, 0
    options_text = "\n".join(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n{options_text}\n\n"
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames, "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
        {"type": "text",  "text": user_text},
    ]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs,
                            **video_kwargs, return_tensors="pt")
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


def compute_metrics(results):
    by_type = {}
    correct_all = 0
    for r in results:
        ok = r.get("correct", False)
        correct_all += int(ok)
        by_type.setdefault(r.get("qa_type", "?"), []).append(ok)
    overall_acc = 100.0 * correct_all / max(1, len(results))
    avg_gen     = sum(r.get("generate_time", 0) for r in results) / max(1, len(results))
    avg_tokens  = sum(r.get("n_input_tokens", 0) for r in results) / max(1, len(results))
    per_type = {qt: {"accuracy": 100.0 * sum(f) / max(1, len(f)), "n_samples": len(f)}
                for qt, f in sorted(by_type.items())}
    return {"overall": {"accuracy": overall_acc, "n_samples": len(results),
                        "avg_generate_time_sec": round(avg_gen, 4),
                        "avg_input_tokens": round(avg_tokens, 1)},
            "per_qa_type": per_type}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=32)
    p.add_argument("--n-qwen-frames", type=int, default=64)
    p.add_argument("--output-tag",    default="baseline_full_tokens_egtea_v2")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Qwen2.5-VL-7B ...")
    qwen_processor, qwen_model = load_qwen(device)
    print("Qwen loaded.")

    dataset = TrajGazeV2QADataset(datasets=["egtea"], n_frames=args.n_frames)

    results = []
    for item in tqdm(dataset, desc="Baseline (100% tokens)"):
        if item is None:
            continue
        try:
            response, generate_time, n_tokens = qwen_full_tokens(
                qwen_processor, qwen_model,
                item["frame_paths"], item["question"], item["options"],
                n_qwen_frames=args.n_qwen_frames, device=device,
            )
            pred    = extract_answer(response)
            correct = (pred == item["answer"])
            results.append({"question": item["question"], "answer": item["answer"],
                            "prediction": pred, "correct": correct,
                            "qa_type": item["qa_type"],
                            "generate_time": generate_time,
                            "n_input_tokens": n_tokens})
        except Exception:
            traceback.print_exc()
            results.append({"question": item.get("question", ""), "answer": item.get("answer", ""),
                            "prediction": "", "correct": False,
                            "qa_type": item.get("qa_type", ""),
                            "generate_time": 0.0, "n_input_tokens": 0})

    tag = args.output_tag
    with open(os.path.join(RESULTS_DIR, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    metrics = compute_metrics(results)
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ov = metrics["overall"]
    print(f"\n{'='*50}")
    print(f"  {tag}")
    print(f"{'='*50}")
    print(f"  Overall       : {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"  Generate time : {ov['avg_generate_time_sec']:.4f}s  (VLM generate() only)")
    print(f"  Avg tokens    : {ov['avg_input_tokens']:.0f}")
    for qt, qm in sorted(metrics["per_qa_type"].items()):
        print(f"  {qt:10s}: {qm['accuracy']:.2f}%  (n={qm['n_samples']})")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
