"""
Evaluate Qwen2.5-VL-7B-Instruct WITH AutoGaze spatial pruning on StreamGaze
present_future_action_prediction — all samples.

Integration approach:
  1. Load 16 frames at 224×224 (shared input for AutoGaze and Qwen).
  2. AutoGaze predicts spatial gaze at 16×16 per frame (224×224 / patch_size=14),
     using target_scales=[56,112,196,224] and target_patch_size=14.
  3. Map gaze mask to Qwen's post-merger LLM token space:
       - Temporal merge: OR masks for frame pairs → T_merged = T // 2
       - Spatial merge:  2×2 any-pool → (T_merged, H//2, W//2) keep_mask
  4. Run Qwen's full ViT to get all N_total post-merger video embeddings.
  5. Filter embeddings with keep_mask and rebuild the input sequence.
  6. Position IDs are derived from the original full sequence and filtered
     in sync with the video tokens (preserves 3D spatial / temporal structure).
  7. Run Qwen LLM with filtered inputs_embeds.

gazing_ratio=0.1, task_loss_requirement=0.7, 64 Qwen frames / 16 AutoGaze frames at 224×224.

Usage:
  /workspace/vila_eval_env/bin/python eval/eval_qwen25vl_with_autogaze.py \
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ── AutoGaze ──────────────────────────────────────────────────────────────────
AUTOGAZE_ROOT = "/workspace/EgoGazeVQA/AutoGaze"
sys.path.insert(0, AUTOGAZE_ROOT)
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

# ── Qwen ──────────────────────────────────────────────────────────────────────
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Eval utilities ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from metrics import save_results, build_frames_map

# ── Constants ─────────────────────────────────────────────────────────────────
AUTOGAZE_CKPT = "/workspace/EgoGazeVQA/AutoGaze/ckpt"
QWEN_MODEL_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
QA_FILE = "/workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json"

# AutoGaze: 4 scales ending at 224 (all divisible by 14).
# Largest scale must match the input video resolution.
# At 224-scale: (224/14)^2 = 16^2 = 256 patches per frame — matches Qwen's spatial grid.
AUTOGAZE_TARGET_SCALES = [56, 112, 196, 224]
AUTOGAZE_TARGET_PATCH_SIZE = 14   # Qwen patch_size = 14
AUTOGAZE_N_FRAMES = 16            # AutoGaze trained on 16-frame chunks
N_QWEN_FRAMES     = 64            # frames fed to Qwen (independent of AutoGaze)

GAZING_RATIO    = 0.1
TASK_LOSS_REQ   = 0.7

QWEN_FRAME_SIZE = 224  # 224 / 14 = 16 patches per side

FRAMES_MAP: dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with other eval scripts
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Frame loading
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# AutoGaze → Qwen keep_mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_keep_mask(
    frames_224: list[Image.Image],
    autogaze_model,
    autogaze_transform,
    T_merged: int,
    H_merged: int,
    W_merged: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run AutoGaze and derive a bool keep_mask for Qwen's post-merger LLM tokens.

    Mapping pipeline:
      AutoGaze 224-scale: 16×16=256 patches per frame
        → temporal merge (OR frame pairs):  (T_merged, 16, 16)
        → spatial merge  (2×2 any-pool):    (T_merged, H_merged, W_merged)
        → flatten:                          (N_total = T_merged*H_merged*W_merged,)

    Args:
        frames_224:  list of AUTOGAZE_N_FRAMES PIL images at 224×224 (for AutoGaze).
        T_merged:    Qwen temporal grid = N_QWEN_FRAMES // temporal_patch_size (=2).
        H_merged:    Qwen height grid   = H_patches // spatial_merge_size  (=2).
        W_merged:    Qwen width grid    = W_patches // spatial_merge_size  (=2).
        device:      target device for the returned mask.

    Returns:
        keep_mask (T_merged*H_merged*W_merged,) bool tensor on `device`.
        AutoGaze mask from AUTOGAZE_N_FRAMES frames is tiled temporally to
        match Qwen's T_merged when N_QWEN_FRAMES > AUTOGAZE_N_FRAMES.
    """
    T = len(frames_224)   # AUTOGAZE_N_FRAMES = 16

    # Build (1, T, C, H, W) float32 tensor for AutoGaze.
    # AutoGazeImageProcessor returns pixel_values as [[frame0, frame1, ...]] —
    # a list of one list of T (C,H,W) numpy arrays.
    outputs = autogaze_transform(list(frames_224))
    imgs = outputs.pixel_values   # [[arr0, arr1, ..., arr15]]
    frames_inner = imgs[0] if isinstance(imgs[0], list) else imgs  # [arr0, ..., arr15]
    if isinstance(frames_inner[0], torch.Tensor):
        video_tensor = torch.stack(frames_inner)                    # (T, C, H, W)
    else:
        video_tensor = torch.from_numpy(np.stack(frames_inner))     # (T, C, H, W)
    # Ensure exactly (T, C, H, W) before adding batch dim — guard against extra dim
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)                      # (T, C, H, W)
    video_tensor = video_tensor.float().unsqueeze(0)                # (1, T, C, H, W)
    video_tensor = video_tensor.to(autogaze_model.device)

    with torch.inference_mode():
        gaze_outputs = autogaze_model(
            {"video": video_tensor},
            gazing_ratio=GAZING_RATIO,
            task_loss_requirement=TASK_LOSS_REQ,
            target_scales=AUTOGAZE_TARGET_SCALES,
            target_patch_size=AUTOGAZE_TARGET_PATCH_SIZE,
        )

    # 224-scale mask is the last entry in gazing_mask for AUTOGAZE_TARGET_SCALES=[56,112,196,224]
    # Shape: (B=1, T=16, 256) — 256 = (224/14)^2 = 16^2 patches per frame
    gaze_mask_224 = gaze_outputs["gazing_mask"][-1][0].bool()   # (T, 256)
    H_p = W_p = QWEN_FRAME_SIZE // AUTOGAZE_TARGET_PATCH_SIZE   # 16
    gaze_2d = gaze_mask_224.view(T, H_p, W_p)                   # (T, 16, 16)

    # Temporal merge: OR consecutive frame pairs (Qwen temporal_patch_size=2)
    gaze_temporal = gaze_2d[0::2] | gaze_2d[1::2]               # (T_merged, 16, 16)

    # Spatial merge: 2×2 max-pool (any patch in block → keep block)
    # Treat T_ag as channels: (1, T_ag, 16, 16) → (1, T_ag, H_m, W_m)
    gaze_pooled = F.max_pool2d(
        gaze_temporal.float().unsqueeze(0),
        kernel_size=2, stride=2,
    ).squeeze(0)                                                  # (T_ag, H_m, W_m)

    # Tile temporally to match Qwen's T_merged (N_QWEN_FRAMES may differ from AUTOGAZE_N_FRAMES)
    T_ag = gaze_pooled.shape[0]   # = AUTOGAZE_N_FRAMES // 2
    if T_merged != T_ag:
        repeat = T_merged // T_ag
        gaze_pooled = gaze_pooled.repeat_interleave(repeat, dim=0)  # (T_merged, H_m, W_m)

    keep_mask = gaze_pooled.bool().flatten().to(device)           # (N_total,)
    return keep_mask


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading AutoGaze from {AUTOGAZE_CKPT} ...")
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(
        AUTOGAZE_CKPT, size=(QWEN_FRAME_SIZE, QWEN_FRAME_SIZE)
    )
    autogaze_model = AutoGaze.from_pretrained(AUTOGAZE_CKPT, use_flash_attn=False)
    autogaze_model = autogaze_model.to("cuda:0").eval()
    print("AutoGaze loaded.")

    print(f"Loading Qwen2.5-VL-7B from {QWEN_MODEL_PATH} ...")
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_model.eval()
    print("Qwen model loaded.")

    return autogaze_transform, autogaze_model, qwen_processor, qwen_model


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    autogaze_transform,
    autogaze_model,
    qwen_processor,
    qwen_model,
    seg_dir: str,
    question: str,
    options: list[str],
) -> tuple[str, float]:
    # 1. Load frames: 64 for Qwen, 16 for AutoGaze (trained on 16-frame chunks)
    frames_qwen = load_frames_pil(seg_dir, n=N_QWEN_FRAMES,     size=QWEN_FRAME_SIZE)
    frames_ag   = load_frames_pil(seg_dir, n=AUTOGAZE_N_FRAMES, size=QWEN_FRAME_SIZE)

    # 2. Build Qwen message and get full inputs
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
                    "video": frames_qwen,
                    "resized_height": QWEN_FRAME_SIZE,
                    "resized_width":  QWEN_FRAME_SIZE,
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        return_tensors="pt",
    )

    # Determine embedding device (first non-meta param of embedding layer)
    emb_dev = qwen_model.get_input_embeddings().weight.device
    vis_dev = qwen_model.visual.patch_embed.proj.weight.device

    input_ids        = inputs["input_ids"].to(emb_dev)
    attention_mask   = inputs["attention_mask"].to(emb_dev)
    pixel_values_vid = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    video_grid_thw   = inputs["video_grid_thw"].to(vis_dev)

    # 3. Full Qwen ViT forward → video_embeds (N_total, D)
    # get_video_features lives on the inner Qwen2_5_VLModel, not the outer wrapper
    video_embeds = qwen_model.model.get_video_features(pixel_values_vid, video_grid_thw)
    # Move to embedding device for later injection
    video_embeds = video_embeds.to(emb_dev)

    # Grid sizes after Qwen's ViT merger (spatial_merge_size=2, temporal_patch_size=2)
    T_merged = int(video_grid_thw[0, 0].item())
    H_merged = int(video_grid_thw[0, 1].item()) // 2   # spatial_merge_size=2
    W_merged = int(video_grid_thw[0, 2].item()) // 2
    N_total  = T_merged * H_merged * W_merged           # = video_embeds.shape[0]

    # 4. AutoGaze keep_mask (N_total,) bool — derived from 16 AutoGaze frames,
    #    tiled temporally to cover Qwen's T_merged from 64 frames.
    keep_mask = compute_keep_mask(
        frames_ag, autogaze_model, autogaze_transform,
        T_merged, H_merged, W_merged,
        device=emb_dev,
    )
    N_kept = int(keep_mask.sum().item())

    # 5. Filter video embeddings
    video_embeds_kept = video_embeds[keep_mask]         # (N_kept, D)

    # 6. Compute 3-D position IDs for the FULL (unfiltered) sequence
    #    get_rope_index lives on the inner Qwen2_5_VLModel
    position_ids, rope_deltas = qwen_model.model.get_rope_index(
        input_ids=input_ids,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    # position_ids: (3, batch=1, seq_len)   rope_deltas: (batch=1,)

    # 7. Build filtered token sequence (keep all text + only gazed video tokens)
    video_token_id  = qwen_model.config.video_token_id
    seq_is_video    = (input_ids[0] == video_token_id)           # (seq_len,)
    video_positions = seq_is_video.nonzero(as_tuple=True)[0]    # (N_total,)

    keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[video_positions[~keep_mask]] = False       # drop non-gazed video slots

    new_input_ids      = input_ids[:, keep_seq]         # (1, new_seq_len)
    new_attention_mask = attention_mask[:, keep_seq]    # (1, new_seq_len)
    new_position_ids   = position_ids[:, :, keep_seq]   # (3, 1, new_seq_len)

    # 8. Build inputs_embeds: text tokens from embedding table + filtered video features
    new_inputs_embeds = qwen_model.get_input_embeddings()(new_input_ids)  # (1, new_seq_len, D)
    new_is_video = (new_input_ids[0] == video_token_id)
    new_inputs_embeds[0, new_is_video] = video_embeds_kept.to(new_inputs_embeds.dtype)

    # 9. Generate answer
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = qwen_model.generate(
            inputs_embeds=new_inputs_embeds,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            rope_deltas=rope_deltas,
            max_new_tokens=16,
            do_sample=False,
        )
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - t0

    # generate with inputs_embeds returns full sequence (prompt tokens not tracked),
    # but generate may prepend a decoder start token; strip prompt length to be safe.
    # Since inputs_embeds is used, output_ids contains only generated tokens.
    gen_ids = output_ids if output_ids.shape[1] <= 16 else output_ids[:, -16:]
    response = qwen_processor.batch_decode(
        gen_ids, skip_special_tokens=True
    )[0].strip()
    return response, inference_time


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    autogaze_transform, autogaze_model, qwen_processor, qwen_model = load_model()

    results, skipped, inference_times = [], 0, []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for sample in tqdm(samples, desc="Evaluating (Qwen2.5-VL+AutoGaze 0.3)"):
            start, end = parse_response_time(sample["response_time"])
            seg_dir = get_segment_dir(sample["video_path"], start, end, tmp_dir)
            if seg_dir is None:
                skipped += len(sample["questions"])
                continue
            for qa in sample["questions"]:
                try:
                    prediction, inf_time = run_inference(
                        autogaze_transform, autogaze_model,
                        qwen_processor, qwen_model,
                        seg_dir, qa["question"], qa["options"],
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
    print(f"AutoGaze keep ratio: ~{GAZING_RATIO*100:.0f}% patches "
          f"(gazing_ratio={GAZING_RATIO}, task_loss_req={TASK_LOSS_REQ})")

    save_results(
        results, out_dir, "qwen25vl_with_autogaze_01_64f",
        frames_map=FRAMES_MAP,
        extra={
            "autogaze_ckpt":               AUTOGAZE_CKPT,
            "qwen_model":                  QWEN_MODEL_PATH,
            "gazing_ratio":                GAZING_RATIO,
            "task_loss_requirement":       TASK_LOSS_REQ,
            "autogaze_target_scales":      AUTOGAZE_TARGET_SCALES,
            "autogaze_target_patch_size":  AUTOGAZE_TARGET_PATCH_SIZE,
            "autogaze_n_frames":           AUTOGAZE_N_FRAMES,
            "n_qwen_frames":               N_QWEN_FRAMES,
            "qwen_frame_size":             QWEN_FRAME_SIZE,
            "avg_inference_time_sec":      round(avg_time, 4),
            "skipped":                     skipped,
        },
    )


if __name__ == "__main__":
    main()
