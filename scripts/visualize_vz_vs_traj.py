"""Visualize the dominant-token selection of vanilla VisionZip vs VisionZip-traj.

For a handful of 3-way val items, render the sampled video frames with the
post-merge patch grid and highlight which patches each method puts in its
top-K dominant pool (5%). The trajectory prior only reweights the FROZEN-ViT
attention scores before top-K, so a single model load is enough to compute both
selections; LoRA weights are swapped only to report each method's answer.

Each frame is shown as two SEPARATE panels side by side:
    left  panel = VisionZip dominant pool      (blue filled patches)
    right panel = VZ-traj dominant pool         (green filled patches)
Both panels show: cyan = gaze, magenta = left hand, orange = right hand.
The right (traj) panel also shows a faint green tint = the spatial_w prior.

Usage:
    python -m scripts.visualize_vz_vs_traj \
      --traj-ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_traj_lora_3way_v3upright/best.pth \
      --vz-ckpt   /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_lora_v3upright/best.pth \
      --gpu 0 --n-per-source 2 --scan-per-source 12
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
from TrajGazeMerge.models.model import (
    get_option_ids, build_merged_inputs, forward_logits,
)
from TrajGazeMerge.models.traj_weights import (
    _pool_to_T, _solve_spatial_dims, compute_traj_weights,
    traj_weighted_attn_scores,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, load_visionzip_lora,
    preprocess_visionzip_item, visionzip_select_tokens,
)

# Test-set per-source index boundaries (SG 526, EG 485, HD-EPIC P09 3925)
SG_END, EG_END = 526, 526 + 485
DISP_W = 360   # rendered frame width in px


def src_for_idx(i: int) -> str:
    if i < SG_END: return "sg"
    if i < EG_END: return "eg"
    return "hd"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-ckpt", required=True)
    p.add_argument("--vz-ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-per-source", type=int, default=2)
    p.add_argument("--scan-per-source", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=6)
    p.add_argument("--out-dir",
                   default="/workspace/trajgaze_st/scripts/viz_vz_vs_traj")
    return p.parse_args()


def dominant_indices(scores: torch.Tensor, N: int) -> set[int]:
    k = max(1, int(DOMINANT_RATIO * N))
    return set(torch.topk(scores, k).indices.tolist())


def idx_to_rc(idx: int, S: int, s_w: int):
    t = idx // S
    s = idx % S
    return t, s // s_w, s % s_w


def selection_payload(cached, traj_dict, device):
    """Compute everything needed to render one item's VZ-vs-traj selection."""
    attn = cached["attn_scores"]
    grid = cached["grid_thw"]
    N = cached["video_embeds"].shape[0]
    T = int(grid[0, 0].item())
    H_grid = int(grid[0, 1].item())
    W_grid = int(grid[0, 2].item())
    S = N // T
    s_h, s_w = _solve_spatial_dims(S, H_grid, W_grid)

    vz_idx = dominant_indices(attn, N)
    w_scores = traj_weighted_attn_scores(attn, grid, traj_dict, device)
    traj_idx = dominant_indices(w_scores, N)

    vz_sets = [set() for _ in range(T)]
    traj_sets = [set() for _ in range(T)]
    for i in vz_idx:
        t, r, c = idx_to_rc(i, S, s_w); vz_sets[t].add((r, c))
    for i in traj_idx:
        t, r, c = idx_to_rc(i, S, s_w); traj_sets[t].add((r, c))

    spatial_w, temporal_w = compute_traj_weights(traj_dict, T, s_h, s_w, device)
    spatial_map = spatial_w.reshape(T, s_h, s_w).float().cpu().numpy()

    gaze_pos, gaze_mask = _pool_to_T(traj_dict["gaze_pos"], traj_dict["gaze_mask"], T)
    left_pos, left_mask = _pool_to_T(traj_dict["left_pos"], traj_dict["left_mask"], T)
    right_pos, right_mask = _pool_to_T(traj_dict["right_pos"], traj_dict["right_mask"], T)

    return {
        "T": T, "s_h": s_h, "s_w": s_w,
        "vz_sets": vz_sets, "traj_sets": traj_sets,
        "spatial_map": spatial_map,
        "temporal_w": temporal_w.float().cpu().numpy(),
        "gaze": (gaze_pos.cpu().numpy(), gaze_mask.cpu().numpy()),
        "left": (left_pos.cpu().numpy(), left_mask.cpu().numpy()),
        "right": (right_pos.cpu().numpy(), right_mask.cpu().numpy()),
    }


def predict(model, base_qwen, cached, sel_embeds, recv_idx, option_ids, n_opt):
    inputs = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
    logits = forward_logits(model, inputs)
    return logits[option_ids[:n_opt]].argmax().item()


def pick_frames(pl, max_frames: int) -> list[int]:
    T = pl["T"]
    added = [len(pl["traj_sets"][t] - pl["vz_sets"][t]) for t in range(T)]
    order = sorted(range(T), key=lambda t: (added[t], pl["temporal_w"][t]), reverse=True)
    chosen = [t for t in order if added[t] > 0][:max_frames]
    if not chosen:  # no diff anywhere — fall back to highest-interaction frames
        chosen = sorted(range(T), key=lambda t: pl["temporal_w"][t], reverse=True)[:max_frames]
    return sorted(chosen)


METHOD_COLOR = {"vz": (40, 130, 255), "traj": (0, 210, 80)}


def render_panel(img_path, pl, t, font, method):
    """Render ONE frame showing only `method`'s dominant patches (+ markers)."""
    base = Image.open(img_path).convert("RGB")
    W0, H0 = base.size
    H = int(DISP_W * H0 / W0)
    base = base.resize((DISP_W, H))
    ov = Image.new("RGBA", (DISP_W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    s_h, s_w = pl["s_h"], pl["s_w"]
    pw, ph = DISP_W / s_w, H / s_h

    # spatial prior tint only on the traj panel (it's what traj uses)
    if method == "traj":
        sm = pl["spatial_map"][t]
        smax = float(sm.max()) or 1.0
        for r in range(s_h):
            for c in range(s_w):
                a = int(70 * sm[r, c] / smax)
                if a > 6:
                    d.rectangle([c*pw, r*ph, (c+1)*pw, (r+1)*ph], fill=(0, 255, 0, a))

    # faint patch grid
    for c in range(s_w + 1):
        d.line([c*pw, 0, c*pw, H], fill=(255, 255, 255, 22))
    for r in range(s_h + 1):
        d.line([0, r*ph, DISP_W, r*ph], fill=(255, 255, 255, 22))

    # this method's dominant patches: filled + outlined in its color
    sel = pl["vz_sets"][t] if method == "vz" else pl["traj_sets"][t]
    col = METHOD_COLOR[method]
    for (r, c) in sel:
        d.rectangle([c*pw, r*ph, (c+1)*pw, (r+1)*ph],
                    fill=col + (70,), outline=col + (255,), width=2)

    def marker(posmask, color, kind):
        pos, mask = posmask
        if not mask[t]:
            return
        x, y = float(pos[t, 0]) * DISP_W, float(pos[t, 1]) * H
        if kind == "gaze":
            d.ellipse([x-7, y-7, x+7, y+7], outline=color, width=3)
            d.line([x-11, y, x+11, y], fill=color, width=2)
            d.line([x, y-11, x, y+11], fill=color, width=2)
        else:
            d.rectangle([x-6, y-6, x+6, y+6], fill=color)
    marker(pl["left"],  (255, 0, 255, 255), "hand")
    marker(pl["right"], (255, 140, 0, 255), "hand")
    marker(pl["gaze"],  (0, 230, 255, 255), "gaze")

    out = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    dd = ImageDraw.Draw(out)
    lbl = f"t={t} w={pl['temporal_w'][t]:.2f} sel={len(sel)}"
    dd.rectangle([0, 0, DISP_W, 14], fill=(0, 0, 0))
    dd.text((3, 2), lbl, fill=(255, 255, 255), font=font)
    return out


def compose(item, pl, frame_paths, preds, font, max_frames):
    """One figure: rows = frames, columns = [VisionZip | VZ-traj] (separate panels)."""
    ts = pick_frames(pl, max_frames)
    L = len(frame_paths)
    T = pl["T"]
    rows = []
    for t in ts:
        vlm_i = 0 if T <= 1 else round(t / (T - 1) * (L - 1))
        vlm_i = min(max(vlm_i, 0), L - 1)
        fp = frame_paths[vlm_i]
        rows.append((render_panel(fp, pl, t, font, "vz"),
                     render_panel(fp, pl, t, font, "traj")))

    tw, th = rows[0][0].size
    pad, head, colhead, gap = 6, 80, 18, 16
    Wc = pad + tw + gap + tw + pad
    Hc = head + colhead + len(rows) * (th + pad) + pad
    canvas = Image.new("RGB", (Wc, Hc), (245, 245, 245))
    d = ImageDraw.Draw(canvas)

    q = item["question"]
    q = q if len(q) <= 100 else q[:97] + "..."
    gt = item["answer"]
    letters = [chr(65 + i) for i in range(len(item["options"]))]
    vz_l = letters[preds["vz"]] if preds["vz"] < len(letters) else "?"
    tr_l = letters[preds["traj"]] if preds["traj"] < len(letters) else "?"
    d.text((6, 4), f"[{item.get('dataset','?')}/{item['task']}]  {q}", fill=(0, 0, 0), font=font)
    d.text((6, 22), f"GT={gt}   vanilla-VZ={vz_l} {'OK' if vz_l==gt else 'X'}"
                    f"   VZ-traj={tr_l} {'OK' if tr_l==gt else 'X'}",
           fill=(0, 0, 0), font=font)
    d.text((6, 40), "blue=VisionZip dominant   green=VZ-traj dominant   "
                    "cyan=gaze  magenta=L-hand  orange=R-hand  greentint=spatial prior",
           fill=(40, 40, 40), font=font)
    d.text((6, 58), "dominant pool = 5% of tokens; frames ranked by how many tokens the prior moved",
           fill=(40, 40, 40), font=font)

    cx_vz = pad + tw // 2
    cx_tr = pad + tw + gap + tw // 2
    d.text((cx_vz - 28, head), "VisionZip", fill=METHOD_COLOR["vz"], font=font)
    d.text((cx_tr - 24, head), "VZ-traj", fill=(0, 150, 60), font=font)

    for k, (a, b) in enumerate(rows):
        y = head + colhead + k * (th + pad)
        canvas.paste(a, (pad, y))
        canvas.paste(b, (pad + tw + gap, y))
    return canvas


@torch.no_grad()
def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    print(f"[viz] loading VisionZip Qwen2.5-VL-7B + LoRA on cuda:{args.gpu}", flush=True)
    processor, model = load_visionzip_lora(device)
    base_qwen = model.get_base_model()
    option_ids = get_option_ids(processor, 5)

    def load_lora(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["lora_state"], strict=False)
        model.eval()

    load_lora(args.traj_ckpt)   # start with traj weights

    test_ds = CombinedMergeDataset(split="test", n_vlm_frames=128,
                                   n_traj_frames=128, include_hdepic=True)
    print(f"[viz] {len(test_ds)} test items", flush=True)

    # ── Scan candidates per source; keep the most illustrative (largest sel diff) ─
    want = {"sg": [], "eg": [], "hd": []}
    starts = {"sg": 0, "eg": SG_END, "hd": EG_END}
    chosen = []
    for src, base_i in starts.items():
        scanned = 0
        scored = []
        i = base_i
        while scanned < args.scan_per_source and i < len(test_ds):
            try:
                item = test_ds[i]
            except Exception:
                i += 1; continue
            if item is None:
                i += 1; continue
            try:
                cached = preprocess_visionzip_item(
                    processor, base_qwen, item["vlm_frame_paths"],
                    item["question"], item["options"], device)
                if cached is None:
                    i += 1; continue
                letters = [chr(65 + j) for j in range(len(item["options"]))]
                if item["answer"] not in letters:
                    i += 1; continue
                pl = selection_payload(cached, item["traj"], device)
                diff = sum(len(pl["traj_sets"][t] ^ pl["vz_sets"][t]) for t in range(pl["T"]))
                scored.append((diff, i, item, pl))
                scanned += 1
            except Exception as e:
                print(f"  scan err idx={i}: {e}", flush=True)
            i += 1
        scored.sort(key=lambda x: x[0], reverse=True)
        for diff, idx, item, pl in scored[:args.n_per_source]:
            chosen.append((src, idx, item, pl, diff))
        print(f"[viz] {src}: scanned {scanned}, picked "
              f"{[(c[1], c[4]) for c in chosen if c[0]==src]}", flush=True)

    # ── Predictions: traj weights already loaded → traj preds ─────────────────────
    traj_pred = {}
    for src, idx, item, pl, diff in chosen:
        item2 = test_ds[idx]
        cached = preprocess_visionzip_item(
            processor, base_qwen, item2["vlm_frame_paths"],
            item2["question"], item2["options"], device)
        w = traj_weighted_attn_scores(cached["attn_scores"], cached["grid_thw"],
                                      item2["traj"], device)
        sel, recv = visionzip_select_tokens(cached["video_embeds"], w, cached["attn_key"])
        traj_pred[idx] = predict(model, base_qwen, cached, sel, recv,
                                 option_ids, len(item2["options"]))

    # ── Swap to vanilla VZ weights → vanilla preds ───────────────────────────────
    load_lora(args.vz_ckpt)
    vz_pred = {}
    for src, idx, item, pl, diff in chosen:
        item2 = test_ds[idx]
        cached = preprocess_visionzip_item(
            processor, base_qwen, item2["vlm_frame_paths"],
            item2["question"], item2["options"], device)
        sel, recv = visionzip_select_tokens(
            cached["video_embeds"], cached["attn_scores"], cached["attn_key"])
        vz_pred[idx] = predict(model, base_qwen, cached, sel, recv,
                               option_ids, len(item2["options"]))

    # ── Render ───────────────────────────────────────────────────────────────────
    saved = []
    for src, idx, item, pl, diff in chosen:
        preds = {"vz": vz_pred[idx], "traj": traj_pred[idx]}
        canvas = compose(item, pl, item["vlm_frame_paths"], preds, font, args.max_frames)
        out = os.path.join(args.out_dir, f"{src}_idx{idx}_diff{diff}.png")
        canvas.save(out)
        saved.append(out)
        gt = item["answer"]
        letters = [chr(65 + i) for i in range(len(item["options"]))]
        print(f"  saved {out}  GT={gt} vz={letters[vz_pred[idx]]} traj={letters[traj_pred[idx]]}",
              flush=True)

    print(f"\n[viz] {len(saved)} figures → {args.out_dir}", flush=True)
    for s in saved:
        print("  " + s, flush=True)


if __name__ == "__main__":
    main()
