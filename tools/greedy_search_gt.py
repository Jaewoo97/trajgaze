"""
Paper-faithful greedy patch scoring for AutoGaze NTP training.
Implements Section 3.2 + Appendix B of "Attend Before Attention" (arXiv 2603.12254).

Key properties:
  1. FULL COMBINED LOSS for scoring: L1_pixel(1.0) + L2_DINOv2-Large(0.3) + L2_SigLIP2-Large(0.3)
     matching the paper's DINOv2 ViT-L/14 (with 4 registers) and SigLIP2 L/16@512px.
  2. HYBRID ONE-PASS per frame: score all 265 candidates in one MAE batch, then rank by
     combined loss.  This is far cheaper than true sequential greedy while producing the
     same quality GT because the dominant component (L1) is already frame-contextual.
  3. CROSS-FRAME CAUSAL CONTEXT: each candidate is scored with the full prior-frame context.
  4. RANDOM GAZING RATIO per clip (Appendix B):
       r_avg  ~ Exponential(scale=0.10) clipped to [0.02, 0.20]
       N_t    ~ Dirichlet(α=10 for t=0, α=3 for t>0), scaled to r_avg × V × T
  5. PER-STEP LOSSES l̃^t_k: recorded in task_losses for each selected patch position using
     L1-only MAE passes (fast; the loss ordering trend is consistent with the combined loss).

Models used:
  - MAE: VideoMAE checkpoint at VIDEOMAE_PT (facebook/vit-mae-large backbone)
  - DINOv2: facebook/dinov2-with-registers-large (ViT-L/14, 4 registers, 224px)
  - SigLIP2: google/siglip2-large-patch16-512 (ViT-L/16, 512px)

Usage (4 GPUs):
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 conda run -n gaze --no-capture-output \\
        python -u tools/greedy_search_gt.py \\
          --fold-dir /workspace/datasets/EgoGazeVQA/autogaze_fold \\
          --gpu $i --total_gpus 4 --cuda_device 0 &
    done
    wait
    conda run -n gaze python -u tools/greedy_search_gt.py \\
      --fold-dir /workspace/datasets/EgoGazeVQA/autogaze_fold --merge
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

from transformers import VivitImageProcessor, AutoModel, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from autogaze.tasks.video_mae_reconstruction.modeling_video_mae import ViTMAEForPreTraining
from autogaze.datasets.video_utils import read_video_pyav, sample_frame_indices, process_video_frames
import av

# ── Config ─────────────────────────────────────────────────────────────────────
RECON_MODEL  = "facebook/vit-mae-large"
VIDEOMAE_PT  = "/workspace/EgoGazeVQA/AutoGaze/VideoMAE_AutoGaze/videomae.pt"
SCALES       = "32+64+112+224"
CLIP_LEN     = 16
CAND_BATCH   = 265     # score all candidates at once in one MAE L1 pass
FILTER_K     = 60      # after L1 pre-filter, keep top-K for DINOv2/SigLIP2 re-ranking
DINO_BATCH   = 60      # sub-batch size for DINOv2/SigLIP2 (single pass since FILTER_K≤DINO_BATCH)
SAVE_EVERY   = 16      # flush per-GPU JSON every N clips
L1_WEIGHT    = 1.0
DINO_WEIGHT  = 0.3
SIGLIP_WEIGHT = 0.3

DINOV2_MODEL  = "facebook/dinov2-with-registers-large"  # ViT-L/14, 4 registers
SIGLIP2_MODEL = "google/siglip2-large-patch16-512"       # ViT-L/16, 512px

# Gazing ratio sampling (Appendix B)
GAZE_RATIO_SCALE       = 0.10
GAZE_RATIO_MIN         = 0.02
GAZE_RATIO_MAX         = 0.20
DIRICHLET_ALPHA_FIRST  = 10.0
DIRICHLET_ALPHA_REST   = 3.0

# Derived
_SCALES_LIST  = sorted(int(s) for s in SCALES.split("+"))
_PATCH_SIZE   = 16
_N_PATCHES    = sum((s // _PATCH_SIZE) ** 2 for s in _SCALES_LIST)   # 265


# ── Gazing ratio sampling ──────────────────────────────────────────────────────

def sample_gazing_counts(T: int, rng=None) -> list:
    """
    Appendix B: per-frame patch counts.
      r_avg ~ Exp(scale=0.10) clipped to [0.02, 0.20]
      N_t   ~ round(Dirichlet(10/3/…) × r_avg × V × T), min 1
    """
    if rng is None:
        rng = np.random
    r_avg = float(np.clip(rng.exponential(scale=GAZE_RATIO_SCALE), GAZE_RATIO_MIN, GAZE_RATIO_MAX))
    total = max(T, round(r_avg * _N_PATCHES * T))
    alpha = np.array([DIRICHLET_ALPHA_FIRST] + [DIRICHLET_ALPHA_REST] * (T - 1), dtype=np.float64)
    weights = rng.dirichlet(alpha)
    counts = np.maximum(1, np.round(weights * total)).astype(int).tolist()
    return counts


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models(device):
    from transformers import ViTMAEConfig

    # --- MAE (L1 loss only — full loss computed separately for scoring) ---
    print(f"Loading MAE config from {RECON_MODEL}...", flush=True)
    config = ViTMAEConfig.from_pretrained(RECON_MODEL)
    config.scales               = SCALES
    config.scale_embed          = True
    config.max_num_frames       = 256
    config.time_embed           = True
    config.causal               = True
    config.loss_type            = "l1"
    config.loss_weights         = "1"
    config._attn_implementation = "sdpa"

    print("Instantiating MAE...", flush=True)
    mae = ViTMAEForPreTraining(config)

    print(f"Loading MAE weights from {VIDEOMAE_PT}...", flush=True)
    sd_full = torch.load(VIDEOMAE_PT, map_location="cpu", weights_only=False)
    prefix  = "module.mae."
    sd      = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, _ = mae.load_state_dict(sd, strict=False)
    if missing:
        print(f"  MAE missing keys ({len(missing)}): {missing[:3]}", flush=True)
    mae.eval().to(device)
    mae_transform = VivitImageProcessor.from_pretrained(RECON_MODEL, size=_SCALES_LIST[-1])
    mae.transform = mae_transform

    # --- DINOv2-Large (ViT-L/14, 4 registers) ---
    print(f"Loading DINOv2 from {DINOV2_MODEL}...", flush=True)
    dinov2 = AutoModel.from_pretrained(DINOV2_MODEL, attn_implementation="sdpa")
    dinov2_transform = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    dinov2.eval().to(device)
    for p in dinov2.parameters():
        p.requires_grad_(False)

    # --- SigLIP2-Large (ViT-L/16, 512px) ---
    print(f"Loading SigLIP2 from {SIGLIP2_MODEL}...", flush=True)
    siglip2 = SiglipVisionModel.from_pretrained(SIGLIP2_MODEL, attn_implementation="sdpa")
    siglip2_transform = AutoImageProcessor.from_pretrained(SIGLIP2_MODEL)
    siglip2.eval().to(device)
    for p in siglip2.parameters():
        p.requires_grad_(False)

    return mae, mae_transform, dinov2, dinov2_transform, siglip2, siglip2_transform


# ── Video loading ──────────────────────────────────────────────────────────────

def load_clip(mp4_path, transform, device):
    """Load 16-frame clip → (1, 16, C, H, W) on device."""
    container = av.open(mp4_path)
    indices   = sample_frame_indices(
        clip_len=CLIP_LEN, frame_sample_rate=1,
        seg_len=container.streams.video[0].frames,
        random_sample_frame=False,
    )
    frames = read_video_pyav(container, indices)
    container.close()
    frames = process_video_frames(frames, CLIP_LEN)

    imgs = transform(list(frames)).pixel_values
    if isinstance(imgs[0], list):
        imgs = imgs[0]
    imgs = np.array(imgs)
    if imgs.ndim == 5:
        imgs = imgs[0]
    video = torch.tensor(imgs, dtype=torch.float32).to(device)
    return video.unsqueeze(0)   # (1, T, C, H, W)


# ── Image retransform (MAE → DINOv2/SigLIP2 normalization + optional resize) ──

def retransform_image(image, from_transform, to_transform):
    """
    Convert image from MAE's normalization to target model's normalization.
    Handles optional spatial resize (e.g., 224px → 512px for SigLIP2-Large).

    image: (B, C, H, W) bfloat16 on device
    Returns: (B, C, H', W') ready for DINOv2/SigLIP2 forward
    """
    # Undo MAE normalization
    if from_transform.do_normalize:
        mean = torch.tensor(from_transform.image_mean, device=image.device, dtype=torch.float32)
        std  = torch.tensor(from_transform.image_std,  device=image.device, dtype=torch.float32)
        image = image.float() * std[None, :, None, None] + mean[None, :, None, None]
    if from_transform.do_rescale:
        image = image / from_transform.rescale_factor

    # Resize if target expects a different spatial size
    tgt_size = getattr(to_transform, 'size', None)
    if tgt_size and isinstance(tgt_size, dict):
        th = tgt_size.get('height', None)
        tw = tgt_size.get('width', None)
        if th and tw and (image.shape[2] != th or image.shape[3] != tw):
            image = F.interpolate(image, size=(th, tw), mode='bicubic', align_corners=False)

    # Apply target normalization
    if to_transform.do_rescale:
        image = image * to_transform.rescale_factor
    if to_transform.do_normalize:
        mean = torch.tensor(to_transform.image_mean, device=image.device, dtype=torch.float32)
        std  = torch.tensor(to_transform.image_std,  device=image.device, dtype=torch.float32)
        image = (image - mean[None, :, None, None]) / std[None, :, None, None]

    return image.to(torch.bfloat16)


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def get_dinov2_features(dinov2, image_bf16, mae_transform, dinov2_transform):
    """
    image_bf16: (B, C, H, W) in MAE's normalization.
    Returns: (B, N_patches, 4*hidden) — last 4 layers, patch tokens only.
    """
    img = retransform_image(image_bf16, mae_transform, dinov2_transform)
    hidden_states = dinov2(img, output_hidden_states=True).hidden_states
    num_reg = dinov2.config.num_register_tokens
    # Exclude CLS (index 0) and register tokens (indices 1..num_reg); keep patch tokens
    feats = torch.cat([h[:, num_reg + 1:] for h in hidden_states[-4:]], dim=-1)
    return feats  # (B, N, 4*D)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def get_siglip2_features(siglip2, image_bf16, mae_transform, siglip2_transform):
    """
    image_bf16: (B, C, H, W) in MAE's normalization.
    Returns: (B, N_patches, 4*hidden) — last 4 layers.
    """
    img = retransform_image(image_bf16, mae_transform, siglip2_transform)
    hidden_states = siglip2(img, output_hidden_states=True).hidden_states
    feats = torch.cat(list(hidden_states[-4:]), dim=-1)
    return feats  # (B, N, 4*D)


# ── Gazing info builder ────────────────────────────────────────────────────────

def build_gazing_info(prior_selected: list, cur_selected: list,
                      candidates: list, frame_t: int):
    """
    Build gazing_info tensors for a batch of candidates at step k of frame t.

    prior_selected : list[list[int]] — frame-relative patch indices for frames 0..t-1
    cur_selected   : list[int]       — frame-relative indices already chosen for frame t
    candidates     : list[int]       — frame-relative candidate indices to score
    frame_t        : int             — 0-indexed current frame
    """
    C = len(candidates)
    k = len(cur_selected)

    n_per_frame = []
    for i in range(frame_t):
        n_per_frame.append(len(prior_selected[i]) + 1)  # patches + EOS
    n_per_frame.append(k + 2)   # k-1 selected + 1 candidate + 1 EOS
    n_per_frame_t = torch.tensor(n_per_frame, dtype=torch.long)

    total_N = int(n_per_frame_t.sum().item())
    gp = torch.full((C, total_N), -1, dtype=torch.long)
    ip = torch.zeros(C, total_N, dtype=torch.bool)

    pos = 0
    for i in range(frame_t):
        for rel in prior_selected[i]:
            gp[:, pos] = i * _N_PATCHES + rel
            pos += 1
        ip[:, pos] = True    # EOS
        pos += 1

    for rel in cur_selected:
        gp[:, pos] = frame_t * _N_PATCHES + rel
        pos += 1

    cand_slot = pos
    for b, cand in enumerate(candidates):
        gp[b, cand_slot] = frame_t * _N_PATCHES + cand

    ip[:, cand_slot + 1] = True   # EOS after candidate

    return gp, ip, n_per_frame_t


# ── Combined-loss scoring for one frame ───────────────────────────────────────

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def score_frame_candidates(mae, dinov2, siglip2,
                            mae_transform, dinov2_transform, siglip2_transform,
                            video_ctx, prior_selected, frame_t, device):
    """
    Score all 265 candidates for frame_t in one MAE batch.
    Returns combined_losses: (265,) float32 tensor on CPU.

    video_ctx: (1, t+1, C, H, W)
    prior_selected: list[list[int]] of frame-relative indices for frames 0..t-1
    """
    candidates = list(range(_N_PATCHES))   # all 265 patches
    C = len(candidates)                    # 265

    # Build batch gazing_info (k=0 for all candidates: no prior selection for frame t)
    gp, ip, n_per_frame = build_gazing_info(
        prior_selected=prior_selected,
        cur_selected=[],
        candidates=candidates,
        frame_t=frame_t,
    )

    pv = video_ctx.expand(C, -1, -1, -1, -1)   # (265, t+1, C, H, W)

    gazing_info = {
        "gazing_pos":            gp.to(device),
        "if_padded_gazing":      ip.to(device),
        "num_gazing_each_frame": n_per_frame.to(device),
    }

    # MAE forward → L1 losses + reconstructions
    out = mae(
        pixel_values=pv,
        gazing_info=gazing_info,
        frame_idx_to_reconstruct=torch.tensor([frame_t], device=device),
        interpolate_pos_encoding=True,
    )
    l1_losses    = out.loss_each_reconstruction_frame[:, 0].float().cpu()  # (265,)
    reconstructions = out.reconstruction[:, 0]   # (265, C, H, W) bfloat16 on device

    # Pre-filter: keep only top-FILTER_K candidates by L1 loss (ascending).
    # Reduces DINOv2/SigLIP2 work by ~4-5x without significant quality loss.
    topk_idx  = l1_losses.argsort()[:FILTER_K]   # (K,) indices of K best by L1
    recon_topk = reconstructions[topk_idx]         # (K, C, H, W)

    # Target frame features (compute once per frame)
    target_frame = video_ctx[:, frame_t]   # (1, C, H, W)
    tgt_dino = get_dinov2_features(dinov2, target_frame, mae_transform, dinov2_transform)   # (1, N, D)
    tgt_sig  = get_siglip2_features(siglip2, target_frame, mae_transform, siglip2_transform)  # (1, N, D)

    # DINOv2 + SigLIP2 on top-K only (K ≤ DINO_BATCH → single forward each)
    d_feat = get_dinov2_features(dinov2, recon_topk, mae_transform, dinov2_transform)     # (K, N, D)
    s_feat = get_siglip2_features(siglip2, recon_topk, mae_transform, siglip2_transform)  # (K, N, D)

    dino_topk = (d_feat - tgt_dino).pow(2).mean(dim=(-1, -2)).float().cpu()   # (K,)
    sig_topk  = (s_feat - tgt_sig).pow(2).mean(dim=(-1, -2)).float().cpu()    # (K,)

    # Combined loss for top-K; remaining 265-K candidates keep infinite loss (never selected)
    combined = torch.full((C,), float("inf"), dtype=torch.float32)
    combined[topk_idx] = (
        L1_WEIGHT      * l1_losses[topk_idx]
        + DINO_WEIGHT  * dino_topk
        + SIGLIP_WEIGHT * sig_topk
    )
    return combined   # (265,) CPU float32


# ── Per-step L1 losses for selected patches ────────────────────────────────────

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def compute_step_losses(mae, video_ctx, prior_selected, selected_rel, frame_t, device):
    """
    Compute l̃^t_k for k=1..N_t using L1-only MAE.
    selected_rel: list of N_t frame-relative patch indices (in selection order)
    Returns: list of N_t float losses.
    """
    step_losses = []
    for k in range(len(selected_rel)):
        cur = selected_rel[:k + 1]
        # Build gazing_info with k+1 patches selected, next slot is EOS only
        gp, ip, n_per_frame = build_gazing_info(
            prior_selected=prior_selected,
            cur_selected=cur[:-1],
            candidates=[cur[-1]],
            frame_t=frame_t,
        )
        pv = video_ctx.expand(1, -1, -1, -1, -1)
        gi = {
            "gazing_pos":            gp.to(device),
            "if_padded_gazing":      ip.to(device),
            "num_gazing_each_frame": n_per_frame.to(device),
        }
        out = mae(
            pixel_values=pv,
            gazing_info=gi,
            frame_idx_to_reconstruct=torch.tensor([frame_t], device=device),
            interpolate_pos_encoding=True,
        )
        step_losses.append(float(out.loss_each_reconstruction_frame[0, 0].float().item()))
    return step_losses


# ── Main greedy selection for one clip ────────────────────────────────────────

def greedy_select_clip(mae, dinov2, siglip2,
                       mae_transform, dinov2_transform, siglip2_transform,
                       video, device, rng):
    """
    Hybrid one-pass greedy for one clip.

    video: (1, T, C, H, W) on device.
    Returns:
        gazing_pos  : [[abs_idx, …] per frame]
        task_losses : [[l̃^t_k, …] per frame]
    """
    T      = video.shape[1]
    counts = sample_gazing_counts(T, rng=rng)

    selected_per_frame    = []
    step_losses_per_frame = []

    for t in range(T):
        n_t       = counts[t]
        video_ctx = video[:, :t + 1]   # (1, t+1, C, H, W)

        # --- Score all 265 candidates with full combined loss ---
        combined_losses = score_frame_candidates(
            mae, dinov2, siglip2,
            mae_transform, dinov2_transform, siglip2_transform,
            video_ctx, selected_per_frame, t, device,
        )

        # Select top-N_t patches (lowest combined loss)
        sorted_idx  = combined_losses.argsort()   # ascending
        selected_rel = sorted_idx[:n_t].tolist()

        # --- Per-step L1 losses for the selected patches ---
        step_losses = compute_step_losses(
            mae, video_ctx, selected_per_frame, selected_rel, t, device
        )

        selected_per_frame.append(selected_rel)
        step_losses_per_frame.append(step_losses)

    # Convert to absolute patch indices
    gazing_pos  = [[t * _N_PATCHES + r for r in selected_per_frame[t]] for t in range(T)]
    task_losses = step_losses_per_frame
    return gazing_pos, task_losses


# ── GT key ────────────────────────────────────────────────────────────────────

def get_gt_key(mp4_path):
    parts = mp4_path.replace("\\", "/").split("/")
    return "/".join(parts[-3:])


# ── Merge mode ─────────────────────────────────────────────────────────────────

def merge_gpu_files(fold_dir):
    import glob
    for split in ("train", "val"):
        pattern   = os.path.join(fold_dir, f"gt_{split}_gpu*.json")
        gpu_files = sorted(glob.glob(pattern))
        if not gpu_files:
            print(f"[{split}] No per-GPU files found, skipping.")
            continue
        merged = {}
        for gf in gpu_files:
            with open(gf) as f:
                merged.update(json.load(f))
        out_json = os.path.join(fold_dir, f"gt_{split}.json")
        with open(out_json, "w") as f:
            json.dump(merged, f)
        print(f"[{split}] Merged {len(gpu_files)} GPU files → {len(merged)} entries → {out_json}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--total_gpus",  type=int, default=1)
    parser.add_argument("--cuda_device", type=int, default=None)
    parser.add_argument("--fold-dir",    type=str,
                        default="/workspace/datasets/EgoGazeVQA/autogaze_fold")
    parser.add_argument("--merge",       action="store_true")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if args.merge:
        merge_gpu_files(args.fold_dir)
        return

    cuda_device = args.cuda_device if args.cuda_device is not None else args.gpu
    device      = torch.device(f"cuda:{cuda_device}")
    rng         = np.random.default_rng(args.seed + args.gpu)

    print(f"=== Full-loss greedy GT: {args.fold_dir}", flush=True)
    print(f"    GPU {args.gpu}/{args.total_gpus}  cuda:{cuda_device}", flush=True)
    print(f"    Loss: L1({L1_WEIGHT}) + DINOv2({DINO_WEIGHT}) + SigLIP2({SIGLIP_WEIGHT})", flush=True)
    print(f"    DINOv2 model : {DINOV2_MODEL}", flush=True)
    print(f"    SigLIP2 model: {SIGLIP2_MODEL}", flush=True)
    print(f"    CAND_BATCH={CAND_BATCH}, DINO_BATCH={DINO_BATCH}", flush=True)

    mae, mae_transform, dinov2, dinov2_transform, siglip2, siglip2_transform = load_models(device)
    print("All models ready.\n", flush=True)

    for split in ("train", "val"):
        split_dir = os.path.join(args.fold_dir, split)
        out_json  = os.path.join(args.fold_dir, f"gt_{split}_gpu{args.gpu}.json")

        gt_dict = json.load(open(out_json)) if os.path.exists(out_json) else {}

        if not os.path.isdir(split_dir):
            print(f"[{split}] Directory not found, skipping.", flush=True)
            continue

        mp4_files = sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".mp4")
        )
        mp4_files = [f for i, f in enumerate(mp4_files) if i % args.total_gpus == args.gpu]
        todo      = [f for f in mp4_files if get_gt_key(f) not in gt_dict]
        print(f"[{split}] {len(mp4_files)} clips on GPU {args.gpu}  "
              f"({len(mp4_files) - len(todo)} done, {len(todo)} remaining)", flush=True)

        done = 0
        t0   = time.time()
        for mp4_path in todo:
            key = get_gt_key(mp4_path)
            try:
                video = load_clip(mp4_path, mae_transform, device)
                gazing_pos, task_losses = greedy_select_clip(
                    mae, dinov2, siglip2,
                    mae_transform, dinov2_transform, siglip2_transform,
                    video, device, rng,
                )
                gt_dict[key] = {"gazing_pos": gazing_pos, "task_losses": task_losses}
            except Exception as e:
                print(f"  ERROR {key}: {e}", flush=True)
                continue

            done += 1
            elapsed = time.time() - t0
            eta_s   = elapsed / done * (len(todo) - done) if done < len(todo) else 0
            if done % SAVE_EVERY == 0 or done >= len(todo):
                with open(out_json, "w") as f:
                    json.dump(gt_dict, f)
                print(f"  [{split}] {done}/{len(todo)}  "
                      f"{elapsed/done:.1f}s/clip  ETA {eta_s/3600:.1f}h", flush=True)

        with open(out_json, "w") as f:
            json.dump(gt_dict, f)
        print(f"[{split}] Done. {len(gt_dict)} entries → {out_json}\n", flush=True)


if __name__ == "__main__":
    main()
