"""
GPU 3 — Qwen2.5-VL-7B-Instruct WITH AutoGaze (GRPO checkpoint, 10% tokens)
on EgoGazeVQA fold-c val (EGTEA).

Integration:
  1. AutoGaze runs on 16 frames → spatial keep_mask
  2. Qwen ViT encodes 64 frames → video_embeds
  3. keep_mask filters video_embeds → 10% tokens pass to LLM

Latency = get_video_features() + token filter + LLM generate()
  (AutoGaze forward is NOT counted in latency per user spec — it is the
   vision feature extractor that is timed, i.e., the VLM forward pass.)
"""

import os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

AUTOGAZE_ROOT = "/workspace/EgoGazeVQA/AutoGaze"
sys.path.insert(0, AUTOGAZE_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from egogaze_data import (
    load_egtea_samples, load_clip_frames, save_eval_results,
)
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

AUTOGAZE_CKPT   = "/workspace/EgoGazeVQA/AutoGaze/exps/egogaze_grpo/checkpoint_latest_gaze"
QWEN_MODEL_PATH = (
    "/home/irteam/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)

N_AG_FRAMES  = 16
N_QWEN_FRAMES = 64
FRAME_SIZE    = 224
GAZING_RATIO  = 0.10
TASK_LOSS_REQ = 0.7

AG_SCALES     = [56, 112, 196, 224]
AG_PATCH_SIZE = 14   # Qwen patch_size = 14


def load_model():
    print(f"Loading AutoGaze from {AUTOGAZE_CKPT} ...")
    ag_transform = AutoGazeImageProcessor.from_pretrained(
        AUTOGAZE_CKPT, size=(FRAME_SIZE, FRAME_SIZE)
    )
    # AutoGaze goes on cuda:0 (the only visible GPU inside CUDA_VISIBLE_DEVICES=3)
    ag_model = AutoGaze.from_pretrained(AUTOGAZE_CKPT, use_flash_attn=False)
    ag_model  = ag_model.to("cuda:0").eval()
    print("AutoGaze loaded.")

    print(f"Loading Qwen2.5-VL from {QWEN_MODEL_PATH} ...")
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_model.eval()
    print("Qwen model loaded.")
    return ag_transform, ag_model, qwen_processor, qwen_model


def compute_keep_mask(frames_ag, ag_model, ag_transform, T_merged, H_merged, W_merged, device):
    T   = len(frames_ag)
    out = ag_transform(list(frames_ag))
    imgs = out.pixel_values
    inner = imgs[0] if isinstance(imgs[0], list) else imgs
    if isinstance(inner[0], torch.Tensor):
        vt = torch.stack(inner)
    else:
        vt = torch.from_numpy(np.stack(inner))
    if vt.dim() == 5:
        vt = vt.squeeze(0)
    vt = vt.float().unsqueeze(0).to(ag_model.device)

    with torch.inference_mode():
        gaze_out = ag_model(
            {"video": vt},
            gazing_ratio=GAZING_RATIO,
            task_loss_requirement=TASK_LOSS_REQ,
            target_scales=AG_SCALES,
            target_patch_size=AG_PATCH_SIZE,
        )

    # AutoGaze selects patches across 4 scales: [32, 64, 112, 224].
    # At 10% budget, most tokens come from coarser scales (scale=224 mask is ~0).
    # Project each scale mask to Qwen's 16×16 spatial grid, then OR them.
    #
    # Scale → spatial grid → Qwen 16×16 mapping:
    #   scale 32  → 4×4   grid (patch=8px), each cell covers 4×4 Qwen patches
    #   scale 64  → 8×8   grid (patch=8px), each cell covers 2×2 Qwen patches
    #   scale 112 → 14×14 grid (patch=8px), interpolate to 16×16
    #   scale 224 → 16×16 grid (patch=14px), 1-to-1
    H_q = W_q = FRAME_SIZE // AG_PATCH_SIZE   # 16

    scale_grids = [
        (0, 4,  4),   # scale=32  → 4×4
        (1, 8,  8),   # scale=64  → 8×8
        (2, 14, 14),  # scale=112 → 14×14
        (3, 16, 16),  # scale=224 → 16×16
    ]

    full_mask = torch.zeros(T, H_q, W_q, dtype=torch.bool, device="cpu")
    for idx, H_s, W_s in scale_grids:
        m = gaze_out["gazing_mask"][idx][0].bool().cpu()   # (T, H_s*W_s)
        m2d = m.view(T, H_s, W_s)
        if H_s == H_q:
            up = m2d
        elif H_q % H_s == 0:
            # Exact integer upscale via repeat_interleave
            up = m2d.repeat_interleave(H_q // H_s, dim=1).repeat_interleave(W_q // W_s, dim=2)
        else:
            # Non-integer (14→16): nearest-neighbour interpolate
            up = F.interpolate(
                m2d.float().unsqueeze(0), size=(H_q, W_q), mode="nearest"
            ).squeeze(0).bool()
        full_mask |= up                                    # (T, 16, 16)

    # Temporal merge (Qwen temporal_patch_size=2)
    gaze_t = full_mask[0::2] | full_mask[1::2]            # (T//2, 16, 16)

    # Spatial merge (Qwen spatial_merge_size=2)
    gaze_s = F.max_pool2d(
        gaze_t.float().unsqueeze(0), kernel_size=2, stride=2
    ).squeeze(0)                                           # (T//2, 8, 8)

    # Tile temporally to match Qwen's T_merged
    T_ag = gaze_s.shape[0]
    if T_merged != T_ag:
        gaze_s = gaze_s.repeat_interleave(T_merged // T_ag, dim=0)

    return gaze_s.bool().flatten().to(device)


def run_inference(ag_transform, ag_model, qwen_processor, qwen_model, frames_qwen, frames_ag, question, options):
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
                "video": frames_qwen,
                "resized_height": FRAME_SIZE,
                "resized_width":  FRAME_SIZE,
            },
            {"type": "text", "text": user_text},
        ],
    }]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        **video_kwargs, return_tensors="pt",
    )

    emb_dev = qwen_model.get_input_embeddings().weight.device
    vis_dev = qwen_model.visual.patch_embed.proj.weight.device

    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)

    T_merged = int(grid_thw[0, 0].item())
    H_merged = int(grid_thw[0, 1].item()) // 2
    W_merged = int(grid_thw[0, 2].item()) // 2

    # AutoGaze keep_mask (not counted in latency — computed before timer)
    keep_mask = compute_keep_mask(
        frames_ag, ag_model, ag_transform,
        T_merged, H_merged, W_merged, device=emb_dev,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        # Vision encoding (timed)
        video_embeds = qwen_model.model.get_video_features(pv_vid, grid_thw).to(emb_dev)

        # Filter tokens
        video_embeds_kept = video_embeds[keep_mask]

        position_ids, rope_deltas = qwen_model.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )

        video_token_id  = qwen_model.config.video_token_id
        seq_is_video    = (input_ids[0] == video_token_id)
        video_positions = seq_is_video.nonzero(as_tuple=True)[0]

        keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
        keep_seq[video_positions[~keep_mask]] = False

        new_input_ids      = input_ids[:, keep_seq]
        new_attention_mask = attention_mask[:, keep_seq]
        new_position_ids   = position_ids[:, :, keep_seq]

        new_inputs_embeds = qwen_model.get_input_embeddings()(new_input_ids)
        new_is_video = (new_input_ids[0] == video_token_id)
        new_inputs_embeds[0, new_is_video] = video_embeds_kept.to(new_inputs_embeds.dtype)

        output_ids = qwen_model.generate(
            inputs_embeds=new_inputs_embeds,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            rope_deltas=rope_deltas,
            max_new_tokens=16,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids  = output_ids if output_ids.shape[1] <= 16 else output_ids[:, -16:]
    response = qwen_processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return response, elapsed


def main():
    samples = load_egtea_samples()
    print(f"Loaded {len(samples)} EGTEA samples")

    ag_transform, ag_model, qwen_processor, qwen_model = load_model()

    results = []
    for s in tqdm(samples, desc="Qwen+AutoGaze"):
        try:
            frames_qwen = load_clip_frames(s["frame_dir"], s["clip_stem"], n=N_QWEN_FRAMES, size=FRAME_SIZE)
            frames_ag   = load_clip_frames(s["frame_dir"], s["clip_stem"], n=N_AG_FRAMES,   size=FRAME_SIZE)
            pred, inf_time = run_inference(
                ag_transform, ag_model, qwen_processor, qwen_model,
                frames_qwen, frames_ag, s["question"], s["options"],
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

    save_eval_results(results, "qwen_autogaze_10pct")


if __name__ == "__main__":
    main()
