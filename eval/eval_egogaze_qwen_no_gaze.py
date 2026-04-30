"""
GPU 2 — Qwen2.5-VL-7B-Instruct without AutoGaze on EgoGazeVQA fold-c val (EGTEA).
All visual tokens kept.
Latency = get_video_features() + LLM generate() (full vision+language forward).
"""

import os, sys, time, traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

sys.path.insert(0, os.path.dirname(__file__))
from egogaze_data import (
    load_egtea_samples, load_clip_frames, save_eval_results,
)

sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

QWEN_MODEL_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
N_FRAMES    = 64
FRAME_SIZE  = 224


def load_model():
    print(f"Loading Qwen2.5-VL from {QWEN_MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Qwen model loaded.")
    return processor, model


def run_inference(processor, model, frames, question, options):
    options_text = "\n".join(options)
    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "Answer with only the letter (A, B, C, D, or E) of the correct option."
    )
    messages = [{
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
    }]
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

    emb_dev = model.get_input_embeddings().weight.device
    vis_dev = model.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        # Vision encoding
        video_embeds = model.model.get_video_features(pv_vid, grid_thw).to(emb_dev)
        # Build inputs_embeds
        inputs_embeds = model.get_input_embeddings()(input_ids)
        video_mask    = (input_ids[0] == model.config.video_token_id)
        inputs_embeds[0, video_mask] = video_embeds.to(inputs_embeds.dtype)
        # Position IDs
        position_ids, rope_deltas = model.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            max_new_tokens=16,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids  = output_ids if output_ids.shape[1] <= 16 else output_ids[:, -16:]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return response, elapsed


def main():
    samples = load_egtea_samples()
    print(f"Loaded {len(samples)} EGTEA samples")

    processor, model = load_model()

    results = []
    for s in tqdm(samples, desc="Qwen no-gaze"):
        try:
            frames = load_clip_frames(s["frame_dir"], s["clip_stem"], n=N_FRAMES, size=FRAME_SIZE)
            pred, inf_time = run_inference(processor, model, frames, s["question"], s["options"])
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

    save_eval_results(results, "qwen_no_gaze")


if __name__ == "__main__":
    main()
