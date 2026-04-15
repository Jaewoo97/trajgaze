"""
Step 6 — Stage 2 GRPO training loop.

Jointly fine-tunes the frame selector and patch decoder using Group Relative
Policy Optimization (GRPO) with the frozen VLM as reward oracle.

For each clip:
  1. Frame selector samples K=12 different attend/skip sequences (stochastic)
  2. Select top-40 patches per attended frame via ARPatchDecoder
  3. Inject all attended frames (multi-scale) into frozen NVILA → action prediction
  4. Reward r = correct(pred == gt) − δ·token_ratio
  5. GRPO normalises advantages within the group of 12, updates both selector
     and patch decoder via policy gradient

Key GRPO details (Dr. GRPO style, no KL regularization):
  - Group size K=12
  - Temperature annealed 1.0 → 0.01 during rollout
  - Binary correctness: exact letter match against ground truth
  - token_ratio = n_attended_frames / T  (penalises attending too many frames)

Usage:
  python -m TrajGaze.training.stage2 \
      --adapted-dir    /workspace/datasets/StreamGaze_v2/adapted \
      --interact-dir   /workspace/datasets/StreamGaze_v2/interaction \
      --frames-dir     /workspace/datasets/StreamGaze_v2/frames \
      --qa-file        /workspace/datasets/StreamGaze_v2/qa/present_future_action_prediction.json \
      --stage1-ckpt    /workspace/EgoGazeVQA/TrajGaze/checkpoints/stage1/epoch_0150.pt \
      --output-dir     /workspace/EgoGazeVQA/TrajGaze/checkpoints/stage2 \
      --vlm            nvila \
      --train-datasets egoexolearn holoassist \
      --val-datasets   egtea \
      --epochs         3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import tempfile
from typing import Optional


def _timeout_handler(signum, frame):
    raise TimeoutError("operation timed out")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from TrajGaze.data.adapter import load_adapted, get_frame_sequence
from TrajGaze.data.interaction import compute_heatmap
from TrajGaze.models.tokenizer import BatchFromNpz
from TrajGaze.training.stage1 import (
    TrajGazeModel,
    load_attended_frames_4ch,
    MAX_VIS_FRAMES,
)

# ── GRPO hyperparameters ───────────────────────────────────────────────────────

GROUP_SIZE  = 12      # rollouts per clip
TEMP_START  = 1.0     # sampling temperature at step 0
TEMP_END    = 0.01    # sampling temperature at end of rollout
DELTA_TOKEN = 0.1     # penalty coefficient for token_ratio
ATTEND_THRESH = 0.5   # frame selection threshold (training forward pass)


# ── Dataset splits ─────────────────────────────────────────────────────────────

TRAIN_DATASETS = ["egoexolearn", "holoassist"]
VAL_DATASETS   = ["egtea"]


def _infer_dataset(video_stem: str) -> str:
    """Infer dataset name from video stem."""
    s = video_stem.replace(".mp4", "")
    parts = s.split("-")
    if len(parts) == 5:
        return "egoexolearn"
    if s.startswith("OP"):
        return "egtea"
    return "holoassist"


# ── Patch selection helpers ────────────────────────────────────────────────────

PATCH_SIZE  = 16   # pixels per patch (stride 16 in VideoEncoder)
PATCH_GRID  = 14   # 14×14 = 196 patches per 224×224 frame
N_PATCHES   = 40   # patches per attended frame (20% of 196; matches Stage 1 training)

# NVILA SigLIP multi-scale patch constants
# target_scales=[56,112,224,448], target_patch_size=16
# grid sizes: 56//16=3, 112//16=7, 224//16=14, 448//16=28
# patches per scale: 9, 49, 196, 784  →  total 1038
# cumulative offsets: scale-56→0, scale-112→9, scale-224→58, scale-448→254
NVILA_NUM_FRAMES    = 128   # num_video_frames (processor config)
NVILA_TOTAL_PATCHES = 1038  # 9+49+196+784


def _trajgaze_to_nvila_patches(patch_indices_196: list[int]) -> list[int]:
    """
    Map TrajGaze 14×14 patch indices to NVILA multi-scale for thumbnails.
    Uses scale-56 + scale-112 + scale-224 + scale-448 (top-left child).
    ≈ 9+27+40+40 = 116/1038 ≈ 11% per frame.
    """
    result: set[int] = set()
    for p in patch_indices_196:
        row = p // PATCH_GRID
        col = p % PATCH_GRID
        result.add((row * 3 // PATCH_GRID) * 3 + (col * 3 // PATCH_GRID))
        result.add(9 + (row * 7 // PATCH_GRID) * 7 + (col * 7 // PATCH_GRID))
        result.add(58 + p)
        result.add(254 + (row * 2) * 28 + (col * 2))
    return sorted(result)


def _trajgaze_to_nvila_patches_coarse(patch_indices_196: list[int]) -> list[int]:
    """
    Coarse mapping for tiles: scale-56 + scale-112 only.
    ≈ 9+27 = 36/1038 ≈ 3.5% per frame — keeps global gaze-hand context
    without inflating the tile token budget.
    """
    result: set[int] = set()
    for p in patch_indices_196:
        row = p // PATCH_GRID
        col = p % PATCH_GRID
        result.add((row * 3 // PATCH_GRID) * 3 + (col * 3 // PATCH_GRID))
        result.add(9 + (row * 7 // PATCH_GRID) * 7 + (col * 7 // PATCH_GRID))
    return sorted(result)


def _make_masked_frame(frame_path: str, patch_indices: list[int]) -> "PIL.Image.Image":
    """
    Return a full-resolution RGB PIL Image with only the selected patches visible.
    Used for Qwen inference (which lacks AutoGaze-style token pruning).
    """
    from PIL import Image
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    import numpy as np
    arr = np.array(img, dtype=np.uint8)
    masked = np.zeros_like(arr)
    for idx in patch_indices:
        gr = idx // PATCH_GRID
        gc = idx % PATCH_GRID
        r0 = int(gr * H / PATCH_GRID)
        r1 = int((gr + 1) * H / PATCH_GRID)
        c0 = int(gc * W / PATCH_GRID)
        c1 = int((gc + 1) * W / PATCH_GRID)
        masked[r0:r1, c0:c1] = arr[r0:r1, c0:c1]
    return Image.fromarray(masked)


@torch.no_grad()
def _select_patches(
    model:        "TrajGazeModel",
    attended_idx: list[int],
    item:         dict,
    tok_gaze:     torch.Tensor,   # (1, T, 128)
    tok_left:     torch.Tensor,   # (1, T, 128)
    tok_right:    torch.Tensor,   # (1, T, 128)
    tok_interact: torch.Tensor,   # (1, T, 128)
    present:      torch.Tensor,   # (1, T) bool
    device:       torch.device,
) -> list[list[int]]:
    """
    Run VideoEncoder + TwoLevelTrajectoryEncoder + ARPatchDecoder (greedy)
    for each attended frame.

    Returns: list of length len(attended_idx), each element is a list of
             up to N_PATCHES=40 patch indices for that frame.
    """
    if not attended_idx:
        return []

    frames = item["frames"]

    traj_context = model.traj_enc(
        tok_gaze, tok_left, tok_right, tok_interact,
        present, attended_idx,
    )  # (n_att, 4·T_window, 256)

    # Video encoder — 4-channel frames for attended frames only
    visual_frames = load_attended_frames_4ch(item, attended_idx, frames)
    if visual_frames is not None:
        visual_frames = visual_frames.to(device).unsqueeze(0)  # (1, T_att, 4, 224, 224)
        T_att_full = visual_frames.shape[1]
        if T_att_full > MAX_VIS_FRAMES:
            step = T_att_full / MAX_VIS_FRAMES
            vis_idx = [int(i * step) for i in range(MAX_VIS_FRAMES)]
            visual_frames = visual_frames[:, vis_idx]
        visual_mem = model.video_enc(visual_frames)  # (1, K*196, 192)
    else:
        visual_mem = torch.zeros(1, MAX_VIS_FRAMES * 196, 192, device=device)

    # Run patch decoder greedily, one attended frame at a time
    n_att = len(attended_idx)
    interact_mean = tok_interact[0][attended_idx]  # (n_att, 128)
    max_frame_embed = model.decoder.frame_pos_embed.num_embeddings
    visual_mem_exp = visual_mem.expand(n_att, -1, -1)

    all_patches = []
    for i in range(n_att):
        frame_idx_t = torch.tensor([min(i, max_frame_embed - 1)], device=device)
        patches = model.decoder.decode_greedy(
            frame_idx    = frame_idx_t,
            visual_mem   = visual_mem_exp[i:i+1],
            traj_context = traj_context[i:i+1],
            interact_mean= interact_mean[i:i+1],
            n_patches    = N_PATCHES,
        )
        all_patches.append(patches)

    return all_patches


# ── VLM reward oracle ──────────────────────────────────────────────────────────

import re as _re
_LETTER_RE = _re.compile(r'\b([A-D])\b')

def _extract_letter(pred: str) -> str:
    """
    Extract the answer letter (A-D) from a VLM prediction string.

    Handles formats like: 'A', 'A.', '(A)', 'The answer is A', 'A. some text'.
    Returns the first matched letter, uppercased, or '' if none found.
    """
    m = _LETTER_RE.search(pred.upper())
    return m.group(1) if m else ""

class VLMOracle:
    """
    Wraps a frozen VLM to score frame selections.

    Two backends: 'nvila' (NVILA-8B-HD-Video) or 'qwen' (Qwen2.5-VL-7B).
    The VLM is loaded once and reused across all rollouts.
    """

    def __init__(self, backend: str = "nvila", local_rank: int = 0):
        self.backend    = backend
        self.local_rank = local_rank
        self.model      = None
        self.processor  = None
        self._pil_cache: dict[str, "PIL.Image.Image"] = {}  # path → PIL Image

    def clear_frame_cache(self) -> None:
        """Clear the per-clip PIL image cache between clips."""
        self._pil_cache.clear()

    def load(self):
        if self.backend == "nvila":
            self._load_nvila()
        elif self.backend == "qwen":
            self._load_qwen()
        else:
            raise ValueError(f"Unknown VLM backend: {self.backend}")

    def _load_nvila(self):
        from transformers import AutoModel, AutoProcessor
        MODEL_PATH = "/home/irteam/.cache/huggingface/hub/nvidia_NVILA-8B-HD-Video"
        os.environ["AUTOGAZE_MODEL_ID"] = "/workspace/EgoGazeVQA/AutoGaze/ckpt"
        print(f"[rank {self.local_rank}] Loading NVILA-8B-HD-Video ...")
        # gazing_ratio=1.0 → keep all patches (no AutoGaze pruning).
        # TrajGaze does frame selection; NVILA sees all patches of selected frames.
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            num_video_frames=128,
            num_video_frames_thumbnail=128,
            max_tiles_video=1,
            gazing_ratio_tile=1.0,
            gazing_ratio_thumbnail=1.0,
            task_loss_requirement_tile=None,
            task_loss_requirement_thumbnail=None,
            max_batch_size_autogaze=16,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map=f"cuda:{self.local_rank}",
            torch_dtype=torch.bfloat16,
            max_batch_size_siglip=16,
        )
        self.model.eval()
        print(f"[rank {self.local_rank}] NVILA loaded on cuda:{self.local_rank}.")

    def _load_qwen(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        MODEL_PATH = "/home/irteam/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
        print(f"[rank {self.local_rank}] Loading Qwen2.5-VL-7B ...")
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.local_rank}",
        )
        self.model.eval()
        print(f"[rank {self.local_rank}] Qwen2.5-VL loaded on cuda:{self.local_rank}.")

    @torch.no_grad()
    def score(
        self,
        frame_paths:           list[str],        # paths to attended frame JPEGs
        patch_indices_per_frame: list[list[int]], # 40 patch indices per frame (14×14 grid, 0-195)
        question:              str,
        options:               list[str],
        gt_answer:             str,
    ) -> float:
        """
        Score a frame+patch selection by querying the frozen VLM.

        For NVILA: injects TrajGaze patch indices into SigLIP embeddings across all
                   4 NVILA scales (56, 112, 196, 392); same mechanism as AutoGaze
                   mask_with_gazing, extended to multi-scale (~11% of 1060 patches).
        For Qwen:  uses masked images (best approximation without AutoGaze integration).

        Returns: 1.0 if VLM prediction matches gt_answer, else 0.0
        """
        if not frame_paths:
            return 0.0

        options_text = "\n".join(options)
        if self.backend == "nvila":
            pred = self._infer_nvila_trajgaze(frame_paths, patch_indices_per_frame,
                                              question, options_text)
        else:
            pred = self._infer_qwen_masked(frame_paths, patch_indices_per_frame,
                                           question, options_text)

        letter    = _extract_letter(pred)
        gt_letter = gt_answer.strip().upper()
        return 1.0 if letter == gt_letter else 0.0

    def _infer_nvila_trajgaze(
        self,
        frame_paths:             list[str],
        patch_indices_per_frame: list[list[int]],
        question:                str,
        options_text:            str,
    ) -> str:
        """
        Run NVILA inference with true token pruning via gazing_info injection.

        Instead of visual masking, we inject a custom gazing_info dict into the
        NVILA processor so that SigLIP only processes TrajGaze-selected patches —
        the same mechanism AutoGaze uses, bypassing the AutoGaze model.

        Frame selection: attended frames symlinked to tmpdir; NVILA subsamples
        128 uniformly from those.  Patch selection: TrajGaze's 40 per-frame
        indices (14×14 grid) are expanded to all 4 SigLIP scales and injected
        as gazing_info, reducing processed tokens from 1038 to ~249 per frame.
        """
        import contextlib, io

        video_token = self.processor.tokenizer.video_token
        prompt = (
            f"{video_token}\n\n"
            "You are watching a short first-person (egocentric) video clip.\n"
            f"Question: {question}\n\n"
            f"{options_text}\n\n"
            "Answer with only the letter (A, B, C, or D) of the correct option."
        )

        from PIL import Image as _PIL

        device = next(self.model.parameters()).device
        n_att  = len(frame_paths)

        # Load attended frames via per-clip cache (avoids re-opening same JPEG
        # across the K=12 rollouts of one clip).
        pil_frames = []
        for fp in frame_paths:
            if fp not in self._pil_cache:
                self._pil_cache[fp] = _PIL.open(fp).convert("RGB")
            pil_frames.append(self._pil_cache[fp])

        # Replicate NVILA's linspace subsampling exactly (same logic as
        # _load_video_frames in processing_nvila.py for a directory input).
        nvila_indices = np.round(
            np.linspace(0, n_att - 1, NVILA_NUM_FRAMES)
        ).astype(int)
        sampled_frames = [pil_frames[i] for i in nvila_indices]

        # Exact nvila_to_traj mapping — no approximation needed since we control
        # the subsampling ourselves.
        nvila_to_traj = nvila_indices

        # Expand TrajGaze 14×14 patches to NVILA's 1038-patch multi-scale space
        per_nvila_patches: list[list[int]] = [
            _trajgaze_to_nvila_patches(patch_indices_per_frame[nvila_to_traj[k]])
            for k in range(NVILA_NUM_FRAMES)
        ]
        # Coarse patch set for tiles (scale-56 + scale-112 only, ~36/1038 ≈ 3.5%)
        per_nvila_patches_tile: list[list[int]] = [
            _trajgaze_to_nvila_patches_coarse(patch_indices_per_frame[nvila_to_traj[k]])
            for k in range(NVILA_NUM_FRAMES)
        ]
        n_max_thumb = max((len(p) for p in per_nvila_patches),      default=1)
        n_max_tile  = max((len(p) for p in per_nvila_patches_tile),  default=1)

        def _make_gazing_info(videos_inputs: dict) -> dict:
            """
            Build gazing_info from TrajGaze patch selections.
            Called instead of AutoGaze by the monkey-patched processor.
            Tiles use coarse patches (scale-56+112), thumbnails use fine patches.
            """
            siglip_tiles  = videos_inputs["pixel_values_videos_tiles"]
            siglip_thumbs = videos_inputs["pixel_values_videos_thumbnails"]
            num_tiles_per_vid  = [t.shape[0] for t in siglip_tiles]
            num_thumbs_per_vid = [t.shape[0] for t in siglip_thumbs]
            T_tile = siglip_tiles[0].shape[1]   # frames per temporal chunk

            total_tiles  = sum(num_tiles_per_vid)
            total_thumbs = sum(num_thumbs_per_vid)
            N_tile       = T_tile * n_max_tile    # flat size per tile (coarse)

            # ── Tiles (coarse: scale-56 + scale-112) ───────────────────────────
            tiles_pos    = torch.zeros(total_tiles, N_tile, dtype=torch.long)
            tiles_padded = torch.ones( total_tiles, N_tile, dtype=torch.bool)
            tiles_count  = torch.full((total_tiles, T_tile), n_max_tile, dtype=torch.long)

            for tile_idx in range(total_tiles):
                for f in range(T_tile):
                    k       = tile_idx * T_tile + f   # absolute NVILA frame
                    patches = per_nvila_patches_tile[k]
                    offset  = f * n_max_tile
                    n       = len(patches)
                    tiles_pos[tile_idx, offset:offset + n] = torch.tensor(
                        patches, dtype=torch.long
                    )
                    tiles_padded[tile_idx, offset:offset + n] = False

            # ── Thumbnails (fine: scale-56+112+224+448) ────────────────────────
            thumbs_pos    = torch.zeros(total_thumbs, n_max_thumb, dtype=torch.long)
            thumbs_padded = torch.ones( total_thumbs, n_max_thumb, dtype=torch.bool)
            thumbs_count  = torch.full((total_thumbs, 1), n_max_thumb, dtype=torch.long)

            for th in range(total_thumbs):
                patches = per_nvila_patches[th]
                n       = len(patches)
                thumbs_pos[th, :n] = torch.tensor(patches, dtype=torch.long)
                thumbs_padded[th, :n] = False

            return {
                "gazing_pos_tiles":
                    list(torch.split(tiles_pos,    num_tiles_per_vid, dim=0)),
                "num_gazing_each_frame_tiles":
                    list(torch.split(tiles_count,  num_tiles_per_vid, dim=0)),
                "if_padded_gazing_tiles":
                    list(torch.split(tiles_padded, num_tiles_per_vid, dim=0)),
                "gazing_pos_thumbnails":
                    list(torch.split(thumbs_pos,    num_thumbs_per_vid, dim=0)),
                "num_gazing_each_frame_thumbnails":
                    list(torch.split(thumbs_count,  num_thumbs_per_vid, dim=0)),
                "if_padded_gazing_thumbnails":
                    list(torch.split(thumbs_padded, num_thumbs_per_vid, dim=0)),
            }

        # Pass PIL images directly — eliminates tmpdir I/O and the processor's
        # internal JPEG re-opens (128 reads per rollout eliminated).
        _sentinel = object()
        _orig = self.processor.__dict__.get("_get_gazing_info_from_videos", _sentinel)
        self.processor._get_gazing_info_from_videos = _make_gazing_info
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                inputs = self.processor(
                    text=prompt, videos=[sampled_frames], return_tensors="pt"
                )
        finally:
            if _orig is _sentinel:
                self.processor.__dict__.pop("_get_gazing_info_from_videos", None)
            else:
                self.processor._get_gazing_info_from_videos = _orig

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        import time as _time
        with torch.inference_mode(), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            self._last_generate_time = _time.perf_counter() - _t0

        return self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

    def _infer_qwen_masked(
        self,
        frame_paths:             list[str],
        patch_indices_per_frame: list[list[int]],
        question:                str,
        options_text:            str,
    ) -> str:
        """
        Qwen2.5-VL inference using patch-masked frames.
        Qwen lacks AutoGaze-style embedding injection, so we construct masked
        PIL images where only the selected 16×16 patches are visible.
        """
        from qwen_vl_utils import process_vision_info
        images = [
            _make_masked_frame(p, idxs)
            for p, idxs in zip(frame_paths, patch_indices_per_frame)
        ]
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text":
            f"You are watching a first-person video clip.\n"
            f"Question: {question}\n\n{options_text}\n\n"
            "Answer with only the letter (A, B, C, or D)."
        })
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]


# ── QA dataset for Stage 2 ────────────────────────────────────────────────────

class Stage2Dataset:
    """
    Pairs QA annotations with trajectory data for Stage 2 GRPO.

    Each item yields:
      - trajectory batch tensors (same as Stage 1)
      - QA info: question, options, gt_answer, timestamp
      - video_stem for frame path lookup
    """

    def __init__(
        self,
        qa_file:      str,
        adapted_dir:  str,
        interact_dir: str,
        frames_dir:   str,
        datasets:     list[str],   # which datasets to include
        max_frames:   Optional[int] = 200,
    ):
        self.adapted_dir  = adapted_dir
        self.interact_dir = interact_dir
        self.frames_dir   = frames_dir
        self.max_frames   = max_frames

        with open(qa_file) as f:
            qa_raw = json.load(f)

        self.samples = []
        for item in qa_raw:
            stem = item["video_path"].replace(".mp4", "")
            ds   = _infer_dataset(stem)
            if ds not in datasets:
                continue
            for q in item["questions"]:
                adapted_path = os.path.join(adapted_dir, ds, "viz", f"{stem}.json")
                npz_path     = os.path.join(interact_dir, ds, "viz", f"{stem}.npz")
                if not (os.path.exists(adapted_path) and os.path.exists(npz_path)):
                    continue
                self.samples.append({
                    "dataset":       ds,
                    "stem":          stem,
                    "response_time": item.get("response_time", ""),
                    "timestamp":     q["time_stamp"],
                    "question":      q["question"],
                    "options":       q["options"],
                    "gt_answer":     q["answer"],
                    "adapted_path":  adapted_path,
                    "npz_path":      npz_path,
                    "frames_dir":    os.path.join(frames_dir, ds, "viz", stem),
                })

        print(f"Stage2Dataset: {len(self.samples)} QA samples "
              f"from {datasets}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        adapted = load_adapted(s["adapted_path"])
        npz     = np.load(s["npz_path"])

        frame_names = list(npz["frame_names"])
        T_full = len(frame_names)

        # Use the response_time window [start - end], exactly as the eval scripts do.
        # response_time format: "[MM:SS - MM:SS]"
        import re as _re
        rt_times = _re.findall(r"(\d+):(\d+)", s.get("response_time", ""))
        if len(rt_times) >= 2:
            start_sec = int(rt_times[0][0]) * 60 + int(rt_times[0][1])
            end_sec   = int(rt_times[1][0]) * 60 + int(rt_times[1][1])
            t_start   = min(int(start_sec * 10), T_full)
            t_end     = min(int(end_sec   * 10), T_full)
        else:
            # Fallback: use timestamp as end, take max_frames window before it
            mm, ss = map(int, s["timestamp"].split(":"))
            t_sec  = mm * 60 + ss
            t_end  = min(t_sec * 10, T_full)
            t_start = max(0, t_end - (self.max_frames or t_end))

        # Cap to max_frames (take the end of the window, where the action is)
        if self.max_frames and (t_end - t_start) > self.max_frames:
            t_start = t_end - self.max_frames

        if t_end <= t_start:
            t_start, t_end = 0, min(50, T_full)

        frame_names = frame_names[t_start:t_end]
        T = len(frame_names)

        class _NpzSlice:
            def __init__(self, npz, s, e):
                self._d = {k: npz[k][s:e] for k in npz.files if npz[k].ndim >= 1}
            def __getitem__(self, k): return self._d[k]

        npz_slice = _NpzSlice(npz, t_start, t_end)
        frames    = get_frame_sequence(adapted, frame_names)
        batch     = BatchFromNpz.make(frames, npz_slice)

        attend    = torch.from_numpy(npz_slice["attend"].astype(np.int8))

        return {
            **s,           # dataset, stem, question, options, gt_answer, frames_dir
            "frame_names": frame_names,
            "T":           T,
            "frames":      frames,
            **batch,       # trajectory tensors (1, T, ...)
            "attend":      attend,
        }


# ── GRPO rollout ───────────────────────────────────────────────────────────────

def _sample_attended(
    scores:    torch.Tensor,  # (1, T) ∈ (0, 1)
    temp:      float,
    min_frames: int = 1,
) -> list[int]:
    """
    Sample a binary attend/skip decision per frame using temperature.
    At temp=1.0: Bernoulli(scores[t]).
    At temp→0:   hard threshold at 0.5.
    """
    T = scores.shape[1]
    s = scores[0]  # (T,)
    # Temperature-scaled logit: logit / temp
    logits = torch.log(s + 1e-8) - torch.log(1 - s + 1e-8)
    scaled = torch.sigmoid(logits / (temp + 1e-8))
    attend = torch.bernoulli(scaled).bool()
    idx = attend.nonzero(as_tuple=False).squeeze(-1).tolist()
    if not idx:
        # Always attend at least one frame (highest-score)
        idx = [int(scores[0].argmax().item())]
    return idx


@torch.no_grad()
def rollout_group(
    model:       TrajGazeModel,
    item:        dict,
    oracle:      VLMOracle,
    device:      torch.device,
    group_size:  int   = GROUP_SIZE,
    delta_token: float = DELTA_TOKEN,
) -> list[dict]:
    """
    Generate K=group_size rollouts for one clip (no gradients).

    Each rollout samples attended frames stochastically, selects 40 patches per
    attended frame (multi-scale SigLIP injection into NVILA), and queries the VLM
    with the full 10% visual token budget for the reward signal.

    Log-probs are NOT stored here — they are recomputed with gradients in
    grpo_loss() using the stored attended_idx (the sampled binary actions).

    Returns list of rollout dicts with keys:
      attended_idx — sampled binary attend decision (list of int)
      reward       — scalar reward
      token_ratio  — fraction of frames attended
      correctness  — VLM correctness (0.0 or 1.0)
    """
    tokenizer_keys = [
        "gaze_pos", "gaze_speed", "gaze_mask",
        "left_pos", "left_vel", "left_mask",
        "right_pos", "right_vel", "right_mask",
        "d_left", "d_right", "v_rel_left", "v_rel_right",
        "convergence", "lead_lag",
    ]
    tok_in = {k: item[k].to(device) for k in tokenizer_keys}
    T = item["T"]

    tok_gaze, tok_left, tok_right, tok_interact = model.tokenizer(**tok_in)
    present = tok_in["gaze_mask"] | tok_in["left_mask"] | tok_in["right_mask"]  # (1, T)
    scores = model.selector(tok_interact)  # (1, T)

    rollouts = []
    for k in range(group_size):
        temp = TEMP_START + (TEMP_END - TEMP_START) * k / max(1, group_size - 1)
        attended_idx = _sample_attended(scores, temp)
        token_ratio  = len(attended_idx) / max(1, T)

        # Binary attend vector
        att = torch.zeros(T, device=device)
        att[attended_idx] = 1.0

        # VLM scoring — full 10% visual token budget (all attended frames, multi-scale)
        if oracle.model is not None and attended_idx:
            # Run patch decoder (frozen) to select top-40 patches per attended frame
            all_patches = _select_patches(
                model, attended_idx, item,
                tok_gaze, tok_left, tok_right, tok_interact, present, device,
            )

            # Pass ALL attended frames with their multi-scale patches to NVILA.
            # This is the complete 10% token budget used for the reward signal.
            eval_paths   = []
            eval_patches = []
            for pos, t in enumerate(attended_idx):
                if t < len(item["frame_names"]):
                    fpath = os.path.join(item["frames_dir"], item["frame_names"][t])
                    if os.path.exists(fpath):
                        eval_paths.append(fpath)
                        eval_patches.append(all_patches[pos])

            correctness = oracle.score(
                eval_paths, eval_patches, item["question"], item["options"], item["gt_answer"]
            ) if eval_paths else 0.0
        else:
            # Proxy: attend label accuracy (no VLM, no patch selection needed)
            attend_labels = item["attend"].float().to(device)
            correctness = float((att == attend_labels).float().mean().item())

        reward = correctness - delta_token * token_ratio

        rollouts.append({
            "attended_idx": attended_idx,   # stored action (list[int])
            "att_vec":      att.cpu(),       # (T,) binary tensor
            "reward":       reward,
            "token_ratio":  token_ratio,
            "correctness":  correctness,
        })

    return rollouts


def grpo_loss(
    rollouts:     list[dict],
    model:        TrajGazeModel,
    item:         dict,
    device:       torch.device,
) -> torch.Tensor:
    """
    GRPO policy gradient loss (Dr. GRPO style, no KL regularization).

    Recomputes log-probs under the current policy (with gradients) for
    the stored sampled actions, then applies normalized advantages.

    Advantage = (reward - mean_reward) / (std_reward + ε)
    Loss = -mean(advantage · log_prob)
    """
    # Recompute frame selector scores WITH gradients
    tokenizer_keys = [
        "gaze_pos", "gaze_speed", "gaze_mask",
        "left_pos", "left_vel", "left_mask",
        "right_pos", "right_vel", "right_mask",
        "d_left", "d_right", "v_rel_left", "v_rel_right",
        "convergence", "lead_lag",
    ]
    tok_in = {k: item[k].to(device) for k in tokenizer_keys}
    _, _, _, tok_interact = model.tokenizer(**tok_in)
    scores = model.selector(tok_interact)  # (1, T) — has grad_fn
    s = scores[0]                          # (T,)

    rewards   = torch.tensor([r["reward"] for r in rollouts])
    mean_r    = rewards.mean()
    std_r     = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r   # (K,)

    # Recompute log-probs with gradients for each sampled action
    log_probs = []
    for r in rollouts:
        att = r["att_vec"].to(device)                    # (T,) binary, detached
        lp  = att * torch.log(s + 1e-8) + (1 - att) * torch.log(1 - s + 1e-8)
        log_probs.append(lp.sum())
    log_probs_t = torch.stack(log_probs)                 # (K,) — has grad_fn

    loss = -(advantages.to(device) * log_probs_t).mean()
    return loss


# ── Distributed helpers ────────────────────────────────────────────────────────

def _ddp_avg_grads(model: nn.Module, world_size: int) -> None:
    """All-reduce gradients across ranks and divide by world_size."""
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted-dir",  required=True)
    parser.add_argument("--interact-dir", required=True)
    parser.add_argument("--frames-dir",   required=True)
    parser.add_argument("--qa-file",      required=True)
    parser.add_argument("--stage1-ckpt",  required=True,
                        help="Stage 1 checkpoint to initialize from")
    parser.add_argument("--resume-ckpt",  default=None,
                        help="Resume from a mid-run step_XXXXXX.pt checkpoint")
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--train-datasets", nargs="+", default=TRAIN_DATASETS)
    parser.add_argument("--val-datasets",   nargs="+", default=VAL_DATASETS)
    parser.add_argument("--vlm",          default="nvila",
                        choices=["nvila", "qwen", "none"],
                        help="'none' uses attend-label accuracy as proxy reward")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--max-frames",   type=int,   default=200)
    parser.add_argument("--group-size",   type=int,   default=GROUP_SIZE)
    parser.add_argument("--delta-token",  type=float, default=DELTA_TOKEN)
    parser.add_argument("--save-every",   type=int,   default=1)
    args = parser.parse_args()

    # ── Distributed init ────────────────────────────────────────────────────────
    import datetime
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"Distributed: {world_size} ranks  |  "
              f"Train: {args.train_datasets}  Val: {args.val_datasets}")

    # ── TrajGaze model ──────────────────────────────────────────────────────────
    model = TrajGazeModel(args).to(device)
    ckpt  = torch.load(args.stage1_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    if is_main:
        print(f"Loaded Stage 1 weights from {args.stage1_ckpt}")

    resume_step  = 0
    resume_epoch = 0
    if args.resume_ckpt:
        resume_ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(resume_ckpt["model"])
        resume_step  = resume_ckpt["global_step"]
        resume_epoch = resume_ckpt.get("epoch", 0)
        if is_main:
            print(f"Resumed from {args.resume_ckpt} "
                  f"(step={resume_step}, epoch={resume_epoch})")

    # Freeze trajectory predictor (Stage 2 doesn't use L_traj)
    for p in model.traj_pred.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ── VLM oracle (one per rank, pinned to this rank's GPU) ───────────────────
    oracle = VLMOracle(
        backend    = args.vlm if args.vlm != "none" else "none",
        local_rank = local_rank,
    )
    if args.vlm != "none":
        oracle.load()
    elif is_main:
        print("VLM oracle disabled — using attend-label accuracy as proxy reward.")

    # Stagger NVILA loads to avoid OOM from simultaneous HF downloads/init
    dist.barrier()

    # ── Datasets ────────────────────────────────────────────────────────────────
    train_ds = Stage2Dataset(
        qa_file      = args.qa_file,
        adapted_dir  = args.adapted_dir,
        interact_dir = args.interact_dir,
        frames_dir   = args.frames_dir,
        datasets     = args.train_datasets,
        max_frames   = args.max_frames,
    )
    val_ds = Stage2Dataset(
        qa_file      = args.qa_file,
        adapted_dir  = args.adapted_dir,
        interact_dir = args.interact_dir,
        frames_dir   = args.frames_dir,
        datasets     = args.val_datasets,
        max_frames   = args.max_frames,
    )

    if is_main:
        print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Initial validation (before any training, skip when resuming) ────────────
    if not args.resume_ckpt:
        if is_main:
            print("Running initial validation ...")
        val_acc = _validate(model, val_ds, oracle, device, args,
                            local_rank=local_rank, world_size=world_size)
        if is_main:
            print(f"Val accuracy (step 0, pre-training): {100*val_acc:.1f}%")

    # ── Training loop ───────────────────────────────────────────────────────────
    global_step = resume_step

    for epoch in range(resume_epoch, args.epochs):
        model.train()

        # Each rank processes a disjoint subset of clips (strided split).
        # Seed shuffle by epoch so we can reproduce the order on resume.
        all_indices = list(range(len(train_ds)))
        rng = random.Random(epoch)
        rng.shuffle(all_indices)
        rank_indices = all_indices[local_rank::world_size]

        # On resume: skip steps already completed in this epoch
        steps_done_this_epoch = max(0, resume_step - epoch * len(rank_indices))
        if steps_done_this_epoch > 0:
            rank_indices = rank_indices[steps_done_this_epoch:]
            if is_main:
                print(f"Resuming epoch {epoch+1}: skipping first "
                      f"{steps_done_this_epoch} steps (already done)")

        # Pad all ranks to the same step count so _ddp_avg_grads calls stay in sync
        n_local = torch.tensor(len(rank_indices), dtype=torch.long, device=device)
        dist.all_reduce(n_local, op=dist.ReduceOp.MAX)
        max_steps = int(n_local.item())
        while len(rank_indices) < max_steps:
            rank_indices.append(rank_indices[-1])

        total_reward = total_correct = total_loss = 0.0
        n_steps = 0

        pbar = tqdm(rank_indices,
                    desc=f"Epoch {epoch+1}/{args.epochs} [rank {local_rank}]",
                    disable=not is_main)
        for idx in pbar:
            try:
                item = train_ds[idx]
            except Exception:
                continue

            optimizer.zero_grad()

            _prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(300)  # 5-minute per-clip hard limit
            _clip_ok = True
            try:
                rollouts = rollout_group(
                    model        = model,
                    item         = item,
                    oracle       = oracle,
                    device       = device,
                    group_size   = args.group_size,
                    delta_token  = args.delta_token,
                )

                loss = grpo_loss(rollouts, model, item, device)
                loss.backward()
            except TimeoutError:
                print(f"[rank {local_rank}] WARNING: clip {idx} timed out, skipping.")
                _clip_ok = False
                optimizer.zero_grad()   # discard any partial grads
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, _prev_handler)

            # All ranks must hit this barrier together — timed-out rank contributes zeros
            _ddp_avg_grads(model, world_size)
            if not _clip_ok:
                optimizer.zero_grad()
                global_step += 1
                continue

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            mean_r   = sum(r["reward"]      for r in rollouts) / len(rollouts)
            mean_cor = sum(r["correctness"] for r in rollouts) / len(rollouts)
            total_reward  += mean_r
            total_correct += mean_cor
            total_loss    += loss.item()
            n_steps += 1
            global_step += 1

            if is_main:
                pbar.set_postfix({
                    "loss": f"{total_loss/n_steps:.3f}",
                    "rew":  f"{total_reward/n_steps:.3f}",
                    "acc":  f"{100*total_correct/n_steps:.1f}%",
                })

            # Mid-epoch validation + checkpoint every 50 steps
            if global_step % 50 == 0:
                if is_main:
                    print(f"\n[step {global_step}] Running mid-epoch validation ...")
                val_acc = _validate(model, val_ds, oracle, device, args,
                                    local_rank=local_rank, world_size=world_size)
                if is_main:
                    print(f"[step {global_step}] Val accuracy: {100*val_acc:.1f}%")
                    ckpt_path = os.path.join(
                        args.output_dir, f"step_{global_step:06d}.pt"
                    )
                    torch.save({
                        "global_step":    global_step,
                        "epoch":          epoch,
                        "model":          model.state_dict(),
                        "optimizer":      optimizer.state_dict(),
                        "val_acc":        val_acc,
                        "train_datasets": args.train_datasets,
                        "vlm":            args.vlm,
                    }, ckpt_path)
                    print(f"Saved mid-epoch checkpoint: {ckpt_path}")
                model.train()

        # Log epoch stats from rank 0 only (no AllReduce — avoids sync issues)
        if is_main:
            print(f"Epoch {epoch+1} done: loss={total_loss/max(1,n_steps):.4f}  "
                  f"mean_reward={total_reward/max(1,n_steps):.4f}  "
                  f"accuracy={100*total_correct/max(1,n_steps):.1f}%")

    if is_main:
        print("Stage 2 GRPO training complete.")
    dist.destroy_process_group()


@torch.no_grad()
def _validate(
    model:      TrajGazeModel,
    ds:         Stage2Dataset,
    oracle:     VLMOracle,
    device:     torch.device,
    args:       argparse.Namespace,
    local_rank: int = 0,
    world_size: int = 1,
    max_samples: int = 100,
) -> float:
    """
    Run greedy selection and VLM evaluation on val set.

    Each rank evaluates a disjoint subset; n_correct / n_total are all_reduced
    so every rank returns the same global accuracy.

    Latency (selector → patch decoder → NVILA) is measured per sample and
    reported by rank 0 as mean / median / p90.
    """
    import time

    model.eval()
    n_correct = n_total = 0
    latencies: list[float] = []   # per-sample wall-clock seconds

    all_indices = list(range(min(max_samples, len(ds))))
    rank_indices = all_indices[local_rank::world_size]

    for idx in tqdm(rank_indices, desc=f"Val [rank {local_rank}]", leave=False):
        try:
            item = ds[idx]
        except Exception:
            continue

        tokenizer_keys = [
            "gaze_pos", "gaze_speed", "gaze_mask",
            "left_pos", "left_vel", "left_mask",
            "right_pos", "right_vel", "right_mask",
            "d_left", "d_right", "v_rel_left", "v_rel_right",
            "convergence", "lead_lag",
        ]

        tok_in = {k: item[k].to(device) for k in tokenizer_keys}
        tok_gaze, tok_left, tok_right, tok_interact = model.tokenizer(**tok_in)
        present = tok_in["gaze_mask"] | tok_in["left_mask"] | tok_in["right_mask"]
        # Top-k frame selection (exact 50% budget, same as inference)
        _, attended_idx = model.selector.select_frames(tok_interact, ratio=0.50)
        attended_idx = attended_idx[0]

        if oracle.model is not None and attended_idx:
            # Run patch decoder for all attended frames (full 10% budget)
            all_patches = _select_patches(
                model, attended_idx, item,
                tok_gaze, tok_left, tok_right, tok_interact, present, device,
            )

            # Collect paths and patch indices for all attended frames
            eval_paths   = []
            eval_patches = []
            for pos, t in enumerate(attended_idx):
                if t < len(item["frame_names"]):
                    fpath = os.path.join(item["frames_dir"], item["frame_names"][t])
                    if os.path.exists(fpath):
                        eval_paths.append(fpath)
                        eval_patches.append(all_patches[pos])

            if eval_paths:
                _prev_val_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(120)  # 2-minute per-sample limit during val
                try:
                    c = oracle.score(eval_paths, eval_patches, item["question"], item["options"], item["gt_answer"])
                    n_correct += c
                    gen_time = getattr(oracle, "_last_generate_time", None)
                    if gen_time is not None:
                        latencies.append(gen_time)
                except TimeoutError:
                    print(f"[rank {local_rank}] WARNING: val sample {idx} timed out, skipping.")
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, _prev_val_handler)

        n_total += 1

    model.train()

    # All-reduce accuracy stats across ranks
    stats = torch.tensor([float(n_correct), float(n_total)], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    g_correct, g_total = stats.tolist()

    # Aggregate and report latency from rank 0 only
    if latencies and local_rank == 0:
        lat        = sorted(latencies)
        mean_lat   = sum(lat) / len(lat)
        median_lat = lat[len(lat) // 2]
        p90_lat    = lat[int(len(lat) * 0.9)]
        print(
            f"Val latency (rank 0) — mean: {mean_lat:.2f}s  "
            f"median: {median_lat:.2f}s  "
            f"p90: {p90_lat:.2f}s  "
            f"(n={len(latencies)})"
        )

    return g_correct / max(1, g_total)


if __name__ == "__main__":
    main()
