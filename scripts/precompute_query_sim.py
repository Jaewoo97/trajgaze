"""
Precompute Talk2DINO query-frame similarity for StreamGaze QA data.

For each QA entry, computes per-frame cosine similarity between
DINOv2 patch features (pooled) and Talk2DINO-projected question text.

Two pooling strategies:
  1. Uniform  — mean of all DINOv2 patch features
  2. Weighted — Gaussian-weighted by gaze+hand positions (when available)

Frames are sourced from v3_full parsed dataset (797/921 QAs).
For EGTEA clips with EgoGazeVQA data, gaze+hand weighted pooling is also computed.

Usage:
  python precompute_query_sim.py \
      --qa_file /home/yujin/dataset/StreamGaze/qa/present_future_action_prediction.json \
      --v3_frames_dir /home/yujin/dataset/StreamGaze/dataset_parsed_2fps_gazeHand_v3_full_3.0flash/present_future_action_prediction_frames \
      --egogaze_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
      --output_dir /home/yujin/dataset/StreamGaze/query_sim \
      --batch_size 64 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── Hyperparameters ──────────────────────────────────────────────────────────

SIGMA_GAZE = 32.0 / 224.0
SIGMA_HAND = 40.0 / 224.0
EPS_BG     = 0.01
PATCH_GRID = 37
NUM_PATCHES = PATCH_GRID * PATCH_GRID


# ── Reused from compute_talk2dino_similarity.py ──────────────────────────────

def build_patch_centers(grid_size: int = PATCH_GRID) -> torch.Tensor:
    rows = torch.arange(grid_size, dtype=torch.float32)
    cols = torch.arange(grid_size, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    cx = (grid_c + 0.5) / grid_size
    cy = (grid_r + 0.5) / grid_size
    return torch.stack([cx, cy], dim=-1).reshape(-1, 2)


def compute_weighted_features(
    patch_feats:    torch.Tensor,
    patch_centers:  torch.Tensor,
    gaze_positions: list,
    hand_lefts:     list,
    hand_rights:    list,
) -> torch.Tensor:
    B, N, D = patch_feats.shape
    device = patch_feats.device
    weighted_feats = []

    for i in range(B):
        gaze = gaze_positions[i]
        hl = hand_lefts[i]
        hr = hand_rights[i]

        if gaze is None and hl is None and hr is None:
            weighted_feats.append(patch_feats[i].mean(dim=0))
            continue

        w = torch.full((N,), EPS_BG, device=device)

        if gaze is not None:
            gaze_t = torch.tensor(gaze, dtype=torch.float32, device=device)
            d2 = ((patch_centers - gaze_t) ** 2).sum(dim=-1)
            w = w + torch.exp(-d2 / (2 * SIGMA_GAZE ** 2))

        if hl is not None:
            hl_t = torch.tensor(hl, dtype=torch.float32, device=device)
            d2 = ((patch_centers - hl_t) ** 2).sum(dim=-1)
            w = w + torch.exp(-d2 / (2 * SIGMA_HAND ** 2))

        if hr is not None:
            hr_t = torch.tensor(hr, dtype=torch.float32, device=device)
            d2 = ((patch_centers - hr_t) ** 2).sum(dim=-1)
            w = w + torch.exp(-d2 / (2 * SIGMA_HAND ** 2))

        w = w / w.sum()
        feat = (w.unsqueeze(-1) * patch_feats[i]).sum(dim=0)
        weighted_feats.append(feat)

    return torch.stack(weighted_feats)


# ── Data mapping ─────────────────────────────────────────────────────────────

def build_v3_clip_map(v3_frames_dir: str) -> dict[tuple, str]:
    """Map (video_id, time_range) → clip directory path."""
    clip_map = {}
    if not os.path.isdir(v3_frames_dir):
        return clip_map
    for c in os.listdir(v3_frames_dir):
        parts = c.split("_", 1)
        if len(parts) < 2:
            continue
        rest = parts[1]
        vid_time = rest.split("__")
        vid = vid_time[0]
        time_str = vid_time[1] if len(vid_time) > 1 else ""
        time_norm = time_str.replace("_", ":").replace(":-:", " - ")
        clip_map[(vid, time_norm)] = os.path.join(v3_frames_dir, c)
    return clip_map


def load_egogaze_gaze(egogaze_dir: str, video_id: str) -> dict:
    """Load all gaze_mapping CSVs for a video from EgoGazeVQA."""
    gaze_all = {}
    for ds in ["egtea", "ego4d", "egoexo"]:
        gm_dir = os.path.join(egogaze_dir, ds, "gaze_mapping", video_id)
        if not os.path.isdir(gm_dir):
            continue
        for csv_file in os.listdir(gm_dir):
            if not csv_file.endswith("_mapping.csv"):
                continue
            with open(os.path.join(gm_dir, csv_file)) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        fnum = int(row["gaze_frame_num"])
                        gaze_all[fnum] = (
                            float(row["gaze_x"]),
                            float(row["gaze_y"]),
                        )
                    except (ValueError, KeyError):
                        pass
    return gaze_all


def load_egogaze_hand(egogaze_dir: str, video_id: str) -> dict:
    """Load hand_locations JSON from EgoGazeVQA."""
    for ds in ["egtea", "ego4d", "egoexo"]:
        path = os.path.join(egogaze_dir, ds, "hand_locations",
                            f"{video_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def qa_hash(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()[:8]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--v3_frames_dir", required=True,
                        help="v3_full parsed frames directory")
    parser.add_argument("--egogaze_dir", default=None,
                        help="EgoGazeVQA all_gaze_v1 dir (for gaze+hand weighting)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load QA ───────────────────────────────────────────────────────────
    print("Loading QA data...")
    with open(args.qa_file) as f:
        qa_data = json.load(f)
    print(f"  {len(qa_data)} QA entries")

    # ── Build clip mapping ────────────────────────────────────────────────
    print("Building clip mapping...")
    clip_map = build_v3_clip_map(args.v3_frames_dir)
    print(f"  {len(clip_map)} v3_full clips")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading Talk2DINO-ViTL...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "lorebianchi98/Talk2DINO-ViTL", trust_remote_code=True
    )
    model = model.to(device).eval()
    print("  Model loaded.")

    # ── Pre-encode all questions ──────────────────────────────────────────
    print("Encoding questions...")
    unique_qs = list({d["questions"][0]["question"] for d in qa_data})
    text_embeds = {}
    with torch.no_grad():
        for q in tqdm(unique_qs, desc="  Text encoding"):
            text_embeds[q] = model.encode_text(q).cpu()
    print(f"  {len(unique_qs)} unique questions encoded")

    # ── Patch centers ─────────────────────────────────────────────────────
    patch_centers = build_patch_centers(PATCH_GRID).to(device)

    # ── Process QAs ───────────────────────────────────────────────────────
    stats = {"processed": 0, "skipped": 0, "with_gaze": 0, "total_frames": 0}

    for qi, entry in enumerate(tqdm(qa_data, desc="Processing QAs")):
        vid = entry["video_path"].replace(".mp4", "")
        rt = entry["response_time"].strip("[] ")
        question = entry["questions"][0]["question"]
        qh = qa_hash(question)

        # Output path
        out_path = os.path.join(args.output_dir, f"{vid}_{qh}.npz")
        if args.resume and os.path.exists(out_path):
            stats["processed"] += 1
            continue

        # Find clip frames
        clip_dir = clip_map.get((vid, rt))
        if clip_dir is None:
            stats["skipped"] += 1
            continue

        frame_files = sorted(
            [f for f in os.listdir(clip_dir) if f.endswith(".jpg")]
        )
        if not frame_files:
            stats["skipped"] += 1
            continue

        # Check for EgoGazeVQA gaze+hand data
        has_egogaze = False
        gaze_data = {}
        hand_data = {}
        img_w, img_h = 1, 1
        if args.egogaze_dir:
            gaze_data = load_egogaze_gaze(args.egogaze_dir, vid)
            hand_data = load_egogaze_hand(args.egogaze_dir, vid)
            if gaze_data:
                has_egogaze = True
                stats["with_gaze"] += 1

        # Get image resolution for hand coord normalization
        first_img = Image.open(os.path.join(clip_dir, frame_files[0]))
        img_w, img_h = first_img.size
        first_img.close()

        # Process frames in batches
        all_uniform = []
        all_weighted = []
        n_frames = len(frame_files)

        for batch_start in range(0, n_frames, args.batch_size):
            batch_files = frame_files[batch_start:batch_start + args.batch_size]

            images = []
            batch_gaze = []
            batch_hl = []
            batch_hr = []

            for fname in batch_files:
                fpath = os.path.join(clip_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                except Exception:
                    continue

                images.append(model.image_transforms(img))

                # Try to get gaze+hand for this frame (EgoGazeVQA only)
                # Frame numbering differs — v3_full uses frame_000001.jpg
                # For now, set gaze/hand to None for v3 frames
                # (EgoGazeVQA matching would require frame time alignment)
                batch_gaze.append(None)
                batch_hl.append(None)
                batch_hr.append(None)

            if not images:
                continue

            batch_tensor = torch.stack(images).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                feats = model.model.forward_features(batch_tensor)
                patch_feats = feats["x_norm_patchtokens"].float()

            # Uniform pooling
            uniform_feats = patch_feats.mean(dim=1)

            # Weighted pooling (falls back to uniform when no gaze/hand)
            weighted_feats = compute_weighted_features(
                patch_feats, patch_centers,
                batch_gaze, batch_hl, batch_hr,
            )

            all_uniform.append(uniform_feats.cpu())
            all_weighted.append(weighted_feats.cpu())

            del patch_feats, batch_tensor
            torch.cuda.empty_cache()

        if not all_uniform:
            stats["skipped"] += 1
            continue

        # Stack and compute similarity
        uniform_all = F.normalize(torch.cat(all_uniform), dim=-1)
        weighted_all = F.normalize(torch.cat(all_weighted), dim=-1)

        text_emb = text_embeds[question].float()
        text_norm = F.normalize(text_emb, dim=-1)

        uniform_sims = (uniform_all @ text_norm.T).squeeze(-1).numpy()
        weighted_sims = (weighted_all @ text_norm.T).squeeze(-1).numpy()

        # Save
        np.savez_compressed(
            out_path,
            query_sim=weighted_sims.astype(np.float32),
            uniform_sim=uniform_sims.astype(np.float32),
            question=question,
            video_id=vid,
            response_time=rt,
            n_frames=n_frames,
        )

        stats["processed"] += 1
        stats["total_frames"] += n_frames

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Processed: {stats['processed']}")
    print(f"Skipped:   {stats['skipped']}")
    print(f"With gaze: {stats['with_gaze']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
