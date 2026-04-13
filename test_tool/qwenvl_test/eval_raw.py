import os
import json
import torch
import csv
import gc
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DATASET_ROOT = "/workspace/datasets/EgoGazeVQA_dataset"
METADATA_CSV = os.path.join(DATASET_ROOT, "metadata.csv")
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
EXP_NAME = "batch1"
OUTPUT_CSV = f"/workspace/EgoGazeVQA/results/{EXP_NAME}-{MODEL_NAME}-eval.csv"
OUTPUT_JSON_DIR = f"/workspace/EgoGazeVQA/results/{EXP_NAME}-{MODEL_NAME}-samples"
BATCH_SIZE = 1       # samples per GPU call
PREFETCH = 16        # number of preprocessed batches to keep ready
DECODE_WORKERS = 16   # parallel video decode threads

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 448 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
    )
    processor.tokenizer.padding_side = "left"
    return model, processor


def build_prompt(question, answer_options):
    options_text = answer_options.replace("|", "\n")
    return (
        f"Given the visual sequence and associated question:\n"
        f"{question}\nOptions:\n{options_text}\n"
        "Choose the most appropriate option. Return only the letter of the correct option."
    )


_video_cache: dict = {}
_video_cache_lock = threading.Lock()


def get_video_frames(file_name):
    """Decode video frames, using cache to avoid re-reading duplicate videos."""
    with _video_cache_lock:
        if file_name in _video_cache:
            return _video_cache[file_name]

    video_path = os.path.join(DATASET_ROOT, file_name)
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": video_path, "max_frames": 128}],
    }]
    _, video_inputs = process_vision_info(messages)
    frames = video_inputs[0] if video_inputs else None  # tensor [T, C, H, W]

    with _video_cache_lock:
        _video_cache[file_name] = frames
    return frames


def preprocess_batch(processor, batch_rows):
    """CPU-side: decode videos and tokenize a batch. Returns (inputs_dict, meta_list)."""
    texts = []
    all_video_inputs = []
    meta = []

    for row in batch_rows:
        video_path = os.path.join(DATASET_ROOT, row["file_name"])
        prompt = build_prompt(row["question"], row["answer_options"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_frames": 128},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        frames = get_video_frames(row["file_name"])
        texts.append(text)
        if frames is not None:
            all_video_inputs.append(frames)
        meta.append({
            "file_name": row["file_name"],
            "dataset": row["dataset"],
            "qa_type": row["qa_type"] if row["qa_type"] else "unknown",
            "question": row["question"],
            "answer_options": row["answer_options"],
            "correct_answer": row["correct_answer"].strip().upper(),
        })

    inputs = processor(
        text=texts,
        images=None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    return inputs, meta


def prefetch_worker(processor, batches_of_rows, out_queue):
    """Background thread: preprocess batches and push to queue.
    Video decoding is parallelised across DECODE_WORKERS threads."""
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        for batch_rows in batches_of_rows:
            # Kick off video decoding for all rows in the batch in parallel
            futures = {row["file_name"]: pool.submit(get_video_frames, row["file_name"])
                       for row in batch_rows}
            # Wait for all futures, then build the batch
            for f in futures.values():
                f.result()  # re-raises any decode exception
            inputs, meta = preprocess_batch(processor, batch_rows)
            out_queue.put((inputs, meta))
    out_queue.put(None)  # sentinel


def extract_letter(response):
    for ch in response:
        if ch.upper() in "ABCDE":
            return ch.upper()
    return ""


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, processor = load_model()

    rows = []
    with open(METADATA_CSV, "r") as f:
        for row in csv.DictReader(f):
            if not row["question"] or not row["correct_answer"]:
                continue
            video_path = os.path.join(DATASET_ROOT, row["file_name"])
            if not os.path.exists(video_path):
                print(f"MISSING: {video_path}")
                continue
            rows.append(row)

    print(f"Total samples: {len(rows)}")

    # Split into batches
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    # Start prefetch thread
    prefetch_queue = queue.Queue(maxsize=PREFETCH)
    worker = threading.Thread(
        target=prefetch_worker, args=(processor, batches, prefetch_queue), daemon=True
    )
    worker.start()

    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    results = []
    processed = 0
    start_time = time.time()

    while True:
        item = prefetch_queue.get()
        if item is None:
            break

        inputs, meta = item
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        del inputs
        torch.cuda.empty_cache()
        gc.collect()

        for m, response in zip(meta, output_texts):
            predicted = extract_letter(response.strip())
            is_correct = predicted == m["correct_answer"]

            stats[m["dataset"]][m["qa_type"]][1] += 1
            stats[m["dataset"]]["overall"][1] += 1
            if is_correct:
                stats[m["dataset"]][m["qa_type"]][0] += 1
                stats[m["dataset"]]["overall"][0] += 1

            record = {
                "file_name": m["file_name"],
                "dataset": m["dataset"],
                "qa_type": m["qa_type"],
                "question": m["question"],
                "answer_options": m["answer_options"],
                "correct_answer": m["correct_answer"],
                "model_answer": response.strip(),
                "predicted_letter": predicted,
                "correct": is_correct,
            }
            results.append(record)

            # Save individual JSON — name derived from the video file path
            sample_id = m["file_name"].replace("/", "_").replace(".mp4", "")
            json_path = os.path.join(OUTPUT_JSON_DIR, f"{sample_id}.json")
            with open(json_path, "w") as jf:
                json.dump(record, jf, indent=2)

        processed += len(meta)
        if processed % 20 == 0 or processed == len(rows):
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (len(rows) - processed) / rate if rate > 0 else 0
            print(f"[{processed}/{len(rows)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s ({rate:.2f} samples/s)")

    worker.join()

    # Save results
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Print accuracy report
    print("\n=== Accuracy Report ===")
    all_correct = 0
    all_total = 0
    for dataset in sorted(stats.keys()):
        overall = stats[dataset]["overall"]
        acc = overall[0] / overall[1] * 100 if overall[1] > 0 else 0
        print(f"\n[{dataset}] overall: {overall[0]}/{overall[1]} = {acc:.1f}%")
        all_correct += overall[0]
        all_total += overall[1]
        for qa_type in sorted(k for k in stats[dataset] if k != "overall"):
            c, t = stats[dataset][qa_type]
            a = c / t * 100 if t > 0 else 0
            print(f"  {qa_type}: {c}/{t} = {a:.1f}%")
    if all_total > 0:
        print(f"\n[ALL] {all_correct}/{all_total} = {all_correct/all_total*100:.1f}%")


if __name__ == "__main__":
    main()
