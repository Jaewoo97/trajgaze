"""
GPU 0 — NVILA-8B-HD-Video without AutoGaze on EgoGazeVQA fold-c val (EGTEA).
All visual tokens kept (gazing_ratio_tile=1.0).
Latency = model.generate() call (includes ViT encoding).
"""

import os, sys, time, tempfile, traceback

# Force single GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(__file__))
from egogaze_data import (
    load_egtea_samples, make_clip_dir, extract_choice, save_eval_results,
)

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_PATH    = "/home/irteam/.cache/huggingface/hub/nvidia_NVILA-8B-HD-Video"
AUTOGAZE_DUMMY = "/workspace/EgoGazeVQA/AutoGaze/exps/egogaze_ntp/checkpoint_latest_gaze"

os.environ.setdefault("AUTOGAZE_MODEL_ID", AUTOGAZE_DUMMY)


def load_model():
    print("Loading NVILA processor (no gaze, ratio=1.0) ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        num_video_frames=32,
        num_video_frames_thumbnail=32,
        max_tiles_video=1,
        gazing_ratio_tile=1.0,
        gazing_ratio_thumbnail=1.0,
        task_loss_requirement_tile=None,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
    )
    print("Loading NVILA model ...")
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


def run_inference(processor, model, seg_dir, question, options):
    video_token  = processor.tokenizer.video_token
    options_text = "\n".join(options)
    prompt = (
        f"{video_token}\n\n"
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
    )
    inputs = processor(text=prompt, videos=seg_dir, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return response, elapsed


def main():
    samples = load_egtea_samples()
    print(f"Loaded {len(samples)} EGTEA samples")

    processor, model = load_model()

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for s in tqdm(samples, desc="NVILA no-gaze"):
            try:
                seg_dir = make_clip_dir(s["frame_dir"], s["clip_stem"], tmp_dir)
                pred, inf_time = run_inference(
                    processor, model, seg_dir, s["question"], s["options"]
                )
            except Exception:
                traceback.print_exc()
                pred, inf_time = "", 0.0
            results.append({
                "file_name":      s["file_name"],
                "qa_type":        s["qa_type"],
                "question":       s["question"],
                "answer":         s["answer"],
                "prediction":     pred,
                "inference_time": inf_time,
            })

    save_eval_results(results, "nvila_no_gaze")


if __name__ == "__main__":
    main()
