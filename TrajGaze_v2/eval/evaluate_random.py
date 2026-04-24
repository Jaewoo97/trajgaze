"""
Random baseline: Qwen2.5-VL-7B on EGTEA with randomly selected K% visual tokens.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, math
sys.path.insert(0, "/workspace/EgoGazeVQA")

import torch
from tqdm import tqdm
from TrajGaze_v2.data.dataset import TrajGazeV2QADataset
from TrajGaze_v2.models.model import N_PATCHES
from TrajGaze_v2.training.stage2 import (
    load_qwen, preprocess_qwen_item, qwen_generate_with_mask,
    extract_answer, FRAME_SIZE
)

RESULTS_DIR = "/workspace/EgoGazeVQA/TrajGaze_v2/eval_results"


def compute_metrics(results):
    by_type = {}
    correct_all = 0
    for r in results:
        ok = r.get("correct", False)
        correct_all += int(ok)
        by_type.setdefault(r.get("qa_type", "?"), []).append(ok)
    overall_acc = 100.0 * correct_all / max(1, len(results))
    avg_gen    = sum(r.get("generate_time", 0) for r in results) / max(1, len(results))
    avg_tokens = sum(r.get("n_input_tokens", 0) for r in results) / max(1, len(results))
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
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--keep-ratio",    type=float, default=0.10,
                   help="Fraction of patches to keep (e.g. 0.05 for 5%%)")
    p.add_argument("--output-tag",    default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_keep = max(1, int(math.ceil(args.keep_ratio * N_PATCHES)))
    pct    = int(round(args.keep_ratio * 100))
    tag    = args.output_tag or f"random_{pct}pct_egtea"
    print(f"Random selection: {n_keep}/{N_PATCHES} patches ({args.keep_ratio*100:.0f}%)")

    print("Loading Qwen2.5-VL-7B ...")
    qwen_processor, qwen_model = load_qwen(device)
    print("Qwen loaded.")

    dataset = TrajGazeV2QADataset(datasets=["egtea"], n_frames=args.n_frames)

    results = []
    for item in tqdm(dataset, desc=f"Random {pct}% tokens"):
        if item is None:
            continue
        try:
            # Random mask: uniformly sample n_keep patches out of N_PATCHES
            perm = torch.randperm(N_PATCHES, device=device)
            mask = torch.zeros(N_PATCHES, dtype=torch.bool, device=device)
            mask[perm[:n_keep]] = True

            cached = preprocess_qwen_item(
                qwen_processor, qwen_model,
                item["frame_paths"], item["question"], item["options"],
                n_qwen_frames=args.n_qwen_frames, device=device,
            )
            if cached is None:
                raise RuntimeError("preprocess returned None")

            gen_ids, generate_time, n_tokens = qwen_generate_with_mask(
                qwen_model, cached, mask
            )
            response = qwen_processor.batch_decode(gen_ids, skip_special_tokens=True)
            response = response[0] if response else ""
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

    with open(os.path.join(RESULTS_DIR, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    metrics = compute_metrics(results)
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ov = metrics["overall"]
    print(f"\n{'='*50}")
    print(f"  {tag}  ({n_keep}/{N_PATCHES} patches = {pct}%)")
    print(f"{'='*50}")
    print(f"  Overall       : {ov['accuracy']:.2f}%  (n={ov['n_samples']})")
    print(f"  Generate time : {ov['avg_generate_time_sec']:.4f}s  (VLM generate() only)")
    print(f"  Avg tokens    : {ov['avg_input_tokens']:.0f}")
    for qt, qm in sorted(metrics["per_qa_type"].items()):
        print(f"  {qt:10s}: {qm['accuracy']:.2f}%  (n={qm['n_samples']})")
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
