"""
Feasibility check: Talk2DINO frame-question similarity for EgoGazeVQA.

Computes per-frame cosine similarity between DINOv2 patch features and
Talk2DINO-projected question text embeddings. Two pooling strategies:
  1. Uniform   — mean of all patch features
  2. Weighted  — Gaussian-weighted mean around gaze + hand positions

Usage:
  python compute_talk2dino_similarity.py \
      --data_dir /home/yujin/dataset/EgoGazeVQA/all_gaze_v1 \
      --batch_size 64 --device cuda:0 \
      --output /home/yujin/dataset/EgoGazeVQA/all_gaze_v1/talk2dino_similarity_results.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── Hyperparameters ──────────────────────────────────────────────────────────

SIGMA_GAZE = 32.0 / 224.0    # 0.1429 in normalized [0,1] space
SIGMA_HAND = 40.0 / 224.0    # 0.1786 — hands have wider action radius
EPS_BG     = 0.01             # background preservation constant
PATCH_GRID = 37               # DINOv2 ViT-L/14 @ resize=518 → 37×37 patches
NUM_PATCHES = PATCH_GRID * PATCH_GRID  # 1369


# ── Patch center grid ────────────────────────────────────────────────────────

def build_patch_centers(grid_size: int = PATCH_GRID) -> torch.Tensor:
    """
    Build normalized [0,1] patch center coordinates for DINOv2 grid.
    Returns: (grid_size², 2) tensor — (x, y) per patch.
    """
    rows = torch.arange(grid_size, dtype=torch.float32)
    cols = torch.arange(grid_size, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    cx = (grid_c + 0.5) / grid_size  # x = col direction
    cy = (grid_r + 0.5) / grid_size  # y = row direction
    return torch.stack([cx, cy], dim=-1).reshape(-1, 2)  # (1369, 2)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_metadata(data_dir: str) -> list[dict]:
    """Load metadata.csv and return list of QA dicts."""
    csv_path = os.path.join(data_dir, "metadata.csv")
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row["qa_index"] = i
            rows.append(row)
    return rows


def parse_file_name(file_name: str) -> tuple[str, str, str]:
    """
    Parse file_name column: "{dataset}/{video_id}/{clip_name}.mp4"
    Returns: (dataset, video_id, clip_name)
    """
    parts = file_name.strip().split("/")
    dataset = parts[0]
    video_id = parts[1]
    clip_name = parts[2].replace(".mp4", "")
    return dataset, video_id, clip_name


def group_by_clip(metadata: list[dict]) -> dict[str, list[dict]]:
    """Group QA pairs by file_name (clip)."""
    groups = defaultdict(list)
    for row in metadata:
        groups[row["file_name"]].append(row)
    return dict(groups)


def get_clip_frames(data_dir: str, dataset: str, video_id: str,
                    clip_name: str) -> list[str]:
    """
    Find all frame files for a clip, sorted by frame number.
    Pattern: {dataset}/no_gaze/{video_id}/{clip_name}_{frame_num}.jpg
    """
    frame_dir = os.path.join(data_dir, dataset, "no_gaze", video_id)
    if not os.path.isdir(frame_dir):
        return []
    pattern = os.path.join(frame_dir, f"{clip_name}_*.jpg")
    frames = glob.glob(pattern)
    # Sort by the last numeric part (frame_num)
    def sort_key(path):
        base = os.path.basename(path).replace(".jpg", "")
        return int(base.rsplit("_", 1)[-1])
    return sorted(frames, key=sort_key)


def extract_frame_name(path: str) -> str:
    """Get the filename (e.g., '123_1205_125.jpg') from a full path."""
    return os.path.basename(path)


def extract_frame_num(path: str) -> int:
    """Extract the frame number from the last segment of the filename."""
    base = os.path.basename(path).replace(".jpg", "")
    return int(base.rsplit("_", 1)[-1])


def load_gaze_mapping(data_dir: str, dataset: str, video_id: str,
                      clip_name: str) -> dict[int, tuple[float, float]]:
    """
    Load gaze mapping CSV → {gaze_frame_num: (gaze_x, gaze_y)}.
    Coordinates are already normalized [0, 1].
    """
    csv_path = os.path.join(
        data_dir, dataset, "gaze_mapping", video_id,
        f"{clip_name}_mapping.csv"
    )
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fnum = int(row["gaze_frame_num"])
                gx = float(row["gaze_x"])
                gy = float(row["gaze_y"])
                mapping[fnum] = (gx, gy)
            except (ValueError, KeyError):
                continue
    return mapping


def load_hand_positions(data_dir: str, dataset: str,
                        video_id: str) -> dict[str, dict]:
    """
    Load hand_locations JSON → {frame_name: {"left": (x,y)|None, "right": (x,y)|None}}.
    Coordinates are in pixel space — will be normalized later using image size.
    """
    json_path = os.path.join(
        data_dir, dataset, "hand_locations", f"{video_id}.json"
    )
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


# ── Weighted pooling ─────────────────────────────────────────────────────────

def compute_weighted_features(
    patch_feats:    torch.Tensor,   # (B, 1369, D)
    patch_centers:  torch.Tensor,   # (1369, 2) on same device
    gaze_positions: list,           # B elements, each (gx, gy) or None
    hand_lefts:     list,           # B elements, each (x, y) normalized or None
    hand_rights:    list,           # B elements, each (x, y) normalized or None
) -> torch.Tensor:
    """
    Compute gaze+hand weighted pooled features for a batch.
    Returns: (B, D)
    """
    B, N, D = patch_feats.shape
    device = patch_feats.device
    weighted_feats = []

    for i in range(B):
        gaze = gaze_positions[i]
        hl = hand_lefts[i]
        hr = hand_rights[i]

        if gaze is None and hl is None and hr is None:
            # No spatial signal → uniform pooling
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

        w = w / w.sum()  # normalize
        feat = (w.unsqueeze(-1) * patch_feats[i]).sum(dim=0)  # (D,)
        weighted_feats.append(feat)

    return torch.stack(weighted_feats)  # (B, D)


# ── Main processing ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute Talk2DINO frame-question similarity"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to all_gaze_v1/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results")
    parser.add_argument("--max_clips", type=int, default=0,
                        help="Process only N clips (0 = all)")
    parser.add_argument("--save_npz", action="store_true",
                        help="Also save per-QA npz files for training")
    parser.add_argument("--npz_dir", default=None,
                        help="Directory for per-QA npz files")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Load metadata ─────────────────────────────────────────────────────
    print("Loading metadata...")
    metadata = load_metadata(args.data_dir)
    clip_groups = group_by_clip(metadata)
    print(f"  {len(metadata)} QA pairs, {len(clip_groups)} unique clips")

    # ── Resume check ──────────────────────────────────────────────────────
    done_clips = set()
    existing_results = []
    if args.resume and os.path.exists(args.output + ".partial"):
        with open(args.output + ".partial", "r") as f:
            partial = json.load(f)
        existing_results = partial.get("results", [])
        done_clips = {r["file_name"] for r in existing_results}
        print(f"  Resuming: {len(done_clips)} clips already done")

    # ── Load Talk2DINO model ──────────────────────────────────────────────
    print("Loading Talk2DINO-ViTL model...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "lorebianchi98/Talk2DINO-ViTL", trust_remote_code=True
    )
    model = model.to(device).eval()
    print("  Model loaded.")

    # ── Pre-encode all unique questions ───────────────────────────────────
    print("Encoding questions...")
    unique_questions = list({row["question"] for row in metadata})
    text_embeds = {}
    with torch.no_grad():
        for q in tqdm(unique_questions, desc="  Text encoding"):
            emb = model.encode_text(q)  # (1, D)
            text_embeds[q] = emb.cpu()
    print(f"  {len(unique_questions)} unique questions encoded")

    # ── Build patch centers ───────────────────────────────────────────────
    patch_centers = build_patch_centers(PATCH_GRID).to(device)  # (1369, 2)

    # ── Process clips ─────────────────────────────────────────────────────
    results = list(existing_results)
    total_frames_processed = 0
    stats = {"gaze_frames": 0, "hand_frames": 0, "total_frames": 0}

    clip_items = [(fn, qas) for fn, qas in clip_groups.items()
                  if fn not in done_clips]
    if args.max_clips > 0:
        clip_items = clip_items[:args.max_clips]
    print(f"\nProcessing {len(clip_items)} clips...")

    for clip_idx, (file_name, qa_list) in enumerate(
        tqdm(clip_items, desc="Clips")
    ):
        dataset, video_id, clip_name = parse_file_name(file_name)

        # Find frames
        frame_paths = get_clip_frames(
            args.data_dir, dataset, video_id, clip_name
        )
        if not frame_paths:
            continue

        # Load gaze mapping
        gaze_map = load_gaze_mapping(
            args.data_dir, dataset, video_id, clip_name
        )

        # Load hand positions
        hand_data = load_hand_positions(args.data_dir, dataset, video_id)

        # Get image resolution from first frame (for hand coord normalization)
        try:
            first_img = Image.open(frame_paths[0])
            img_w, img_h = first_img.size
            first_img.close()
        except Exception:
            img_w, img_h = 1, 1  # avoid division by zero

        # Process frames in batches
        all_uniform_feats = []
        all_weighted_feats = []
        all_frame_names = []
        all_gaze_avail = []
        all_hand_avail = []

        for batch_start in range(0, len(frame_paths), args.batch_size):
            batch_paths = frame_paths[batch_start:batch_start + args.batch_size]

            # Load and transform images
            images = []
            batch_gaze = []
            batch_hl = []
            batch_hr = []
            valid_paths = []

            for fp in batch_paths:
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    continue
                images.append(model.image_transforms(img))
                valid_paths.append(fp)

                fname = extract_frame_name(fp)
                fnum = extract_frame_num(fp)

                # Gaze (already normalized [0,1])
                gaze = gaze_map.get(fnum)
                batch_gaze.append(gaze)

                # Hand (pixel → normalized)
                hd = hand_data.get(fname, {})
                hl_raw = hd.get("left")
                hr_raw = hd.get("right")
                hl = (hl_raw[0] / img_w, hl_raw[1] / img_h) if hl_raw else None
                hr = (hr_raw[0] / img_w, hr_raw[1] / img_h) if hr_raw else None
                batch_hl.append(hl)
                batch_hr.append(hr)

            if not images:
                continue

            batch_tensor = torch.stack(images).to(device)

            # DINOv2 forward pass
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                feats = model.model.forward_features(batch_tensor)
                patch_feats = feats["x_norm_patchtokens"]  # (B, 1369, 1024)

            # Cast to float32 for similarity computation
            patch_feats = patch_feats.float()

            # Uniform pooling
            uniform_feats = patch_feats.mean(dim=1)  # (B, D)

            # Weighted pooling
            weighted_feats = compute_weighted_features(
                patch_feats, patch_centers,
                batch_gaze, batch_hl, batch_hr
            )

            # Collect
            for i, fp in enumerate(valid_paths):
                fname = extract_frame_name(fp)
                all_frame_names.append(fname)
                all_uniform_feats.append(uniform_feats[i].cpu())
                all_weighted_feats.append(weighted_feats[i].cpu())
                all_gaze_avail.append(batch_gaze[i] is not None)
                has_hand = batch_hl[i] is not None or batch_hr[i] is not None
                all_hand_avail.append(has_hand)

            del patch_feats, batch_tensor
            torch.cuda.empty_cache()

        if not all_uniform_feats:
            continue

        # Stack and normalize
        uniform_all = F.normalize(torch.stack(all_uniform_feats), dim=-1)
        weighted_all = F.normalize(torch.stack(all_weighted_feats), dim=-1)

        # Compute similarity for each QA pair
        for qa in qa_list:
            text_emb = text_embeds[qa["question"]].float()
            text_norm = F.normalize(text_emb, dim=-1)  # (1, D)

            u_sims = (uniform_all @ text_norm.T).squeeze(-1).tolist()
            w_sims = (weighted_all @ text_norm.T).squeeze(-1).tolist()

            results.append({
                "qa_index": qa["qa_index"],
                "file_name": file_name,
                "dataset": dataset,
                "qa_type": qa.get("qa_type", ""),
                "question": qa["question"],
                "num_frames": len(all_frame_names),
                "frame_names": all_frame_names,
                "uniform_sims": [round(s, 6) for s in u_sims],
                "weighted_sims": [round(s, 6) for s in w_sims],
                "gaze_available": all_gaze_avail,
                "hand_available": all_hand_avail,
            })

            # Save per-QA npz for training pipeline
            if args.save_npz and args.npz_dir:
                qh = hashlib.md5(qa["question"].encode()).hexdigest()[:8]
                npz_sub = os.path.join(args.npz_dir, dataset, video_id)
                os.makedirs(npz_sub, exist_ok=True)
                np.savez_compressed(
                    os.path.join(npz_sub, f"{qh}.npz"),
                    query_sim=np.array(w_sims, dtype=np.float32),
                    uniform_sim=np.array(u_sims, dtype=np.float32),
                    question=qa["question"],
                )

        n_frames = len(all_frame_names)
        total_frames_processed += n_frames
        stats["total_frames"] += n_frames
        stats["gaze_frames"] += sum(all_gaze_avail)
        stats["hand_frames"] += sum(all_hand_avail)

        # Checkpoint every 100 clips
        if (clip_idx + 1) % 100 == 0:
            _save_partial(results, args.output, stats)
            tqdm.write(
                f"  Checkpoint: {clip_idx+1} clips, "
                f"{total_frames_processed} frames"
            )

    # ── Final save ────────────────────────────────────────────────────────
    # Compute summary
    all_u = []
    all_w = []
    for r in results:
        all_u.extend(r["uniform_sims"])
        all_w.extend(r["weighted_sims"])

    summary = {
        "total_qa": len(results),
        "total_clips": len({r["file_name"] for r in results}),
        "total_frames": stats["total_frames"],
        "mean_uniform_sim": round(np.mean(all_u), 6) if all_u else 0,
        "mean_weighted_sim": round(np.mean(all_w), 6) if all_w else 0,
        "gaze_coverage": round(
            stats["gaze_frames"] / max(1, stats["total_frames"]), 4
        ),
        "hand_coverage": round(
            stats["hand_frames"] / max(1, stats["total_frames"]), 4
        ),
    }

    output = {
        "config": {
            "model": "Talk2DINO-ViTL",
            "sigma_gaze": round(SIGMA_GAZE, 6),
            "sigma_hand": round(SIGMA_HAND, 6),
            "eps_bg": EPS_BG,
            "patch_grid": PATCH_GRID,
            "num_patches": NUM_PATCHES,
        },
        "summary": summary,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"\nResults saved to {args.output}")

    # Remove partial file if exists
    partial_path = args.output + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Total QA pairs:     {summary['total_qa']}")
    print(f"Total clips:        {summary['total_clips']}")
    print(f"Total frames:       {summary['total_frames']}")
    print(f"Mean uniform sim:   {summary['mean_uniform_sim']:.4f}")
    print(f"Mean weighted sim:  {summary['mean_weighted_sim']:.4f}")
    print(f"Gaze coverage:      {summary['gaze_coverage']:.1%}")
    print(f"Hand coverage:      {summary['hand_coverage']:.1%}")
    print("=" * 60)


def _save_partial(results, output_path, stats):
    partial_path = output_path + ".partial"
    with open(partial_path, "w") as f:
        json.dump({"results": results, "stats": stats}, f)


if __name__ == "__main__":
    main()
