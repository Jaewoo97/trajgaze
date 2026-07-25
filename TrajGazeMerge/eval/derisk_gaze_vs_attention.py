"""De-risk for Direction ② (dataset switch): does ViT attention COINCIDE with gaze,
or does gaze carry information attention cannot recover?

egtea ceiling hypothesis: gaze ≈ attention because (a) single interaction object and
(b) egocentric gaze is center-biased and ViT attention is too — so they trivially agree.
A gaze-decisive dataset must show gaze that ROAMS away from center AND from attention.

Metrics per gaze-valid frame (normalized [0,1] coords):
  d_attn_gaze   : ViT attention-argmax patch  ↔ gaze         (low = attention tracks gaze)
  d_center_gaze : image center (0.5,0.5)       ↔ gaze         (low = gaze is center-biased)
  d_attn_center : attention-argmax             ↔ center        (low = attention center-biased)
  d_rand_gaze   : random patch                 ↔ gaze         (chance ~0.38)

Reading:
  If d_attn_gaze ≈ d_center_gaze ≈ small  → both sit at center, agreement is trivial (egtea).
  If d_center_gaze LARGE (gaze roams) and d_attn_gaze ALSO large (attention doesn't follow)
     → gaze independent of attention → GAZE-DECISIVE candidate.

Uses no_gaze frames (raw, no marker) so attention is not told where gaze is.
Same code path on both datasets; only --root/--metadata/--persons differ.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/trajgaze_st")
sys.path.insert(0, "/workspace/EgoGazeVQA/VisionZip/Qwen2_5_VL")

from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item,
)
from TrajGazeMerge.models.traj_weights import _solve_spatial_dims


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--persons", required=True, help="comma list of dataset-col values to keep")
    p.add_argument("--label", required=True)
    p.add_argument("--n-items", type=int, default=60)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--gpu", type=int, default=3)
    return p.parse_args()


def frame_gfn(fname, subclip):
    base = os.path.splitext(fname)[0]
    if not base.startswith(subclip + "_"):
        return None
    try:
        return int(base[len(subclip) + 1:])
    except ValueError:
        return None


def list_frames(root, ds, video_id, subclip):
    d = os.path.join(root, ds, "no_gaze", video_id)
    if not os.path.isdir(d):
        return []
    pairs = []
    for fn in os.listdir(d):
        if fn.endswith(".jpg"):
            g = frame_gfn(fn, subclip)
            if g is not None:
                pairs.append((g, os.path.join(d, fn)))
    pairs.sort(key=lambda x: x[0])
    return pairs  # [(gfn, path)]


def load_gaze(root, ds, video_id, subclip):
    csvp = os.path.join(root, ds, "gaze_mapping", video_id, f"{subclip}_mapping.csv")
    g = {}
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp)):
            try:
                gx, gy = float(r["gaze_x"]), float(r["gaze_y"])
                if np.isfinite(gx) and np.isfinite(gy):
                    g[int(r["gaze_frame_num"])] = (gx, gy)
            except (TypeError, ValueError, KeyError):
                pass
    return g


def main():
    a = parse_args()
    torch.cuda.set_device(a.gpu)
    device = torch.device(f"cuda:{a.gpu}")
    keep = set(a.persons.split(","))

    rows = []
    for r in csv.DictReader(open(a.metadata)):
        if r["dataset"] in keep:
            rows.append(r)
    print(f"[{a.label}] {len(rows)} rows in {sorted(keep)}", flush=True)

    processor, model = load_visionzip_lora(device)
    base = model.get_base_model()
    model.eval()

    rng = np.random.RandomState(0)
    D = {k: [] for k in ("attn_gaze", "center_gaze", "attn_center", "rand_gaze")}
    used = 0
    with torch.no_grad():
        for r in rows:
            if used >= a.n_items:
                break
            ds = r["dataset"]; vid = r["video_id"]
            subclip = os.path.splitext(os.path.basename(r["file_name"]))[0]
            frames = list_frames(a.root, ds, vid, subclip)
            gaze = load_gaze(a.root, ds, vid, subclip)
            if len(frames) < 2 or not gaze:
                continue
            # uniform sample n_frames
            idx = [int(i * len(frames) / a.n_frames) for i in range(a.n_frames)]
            sel = [frames[i] for i in idx]
            paths = [p for _, p in sel]
            gfns = [g for g, _ in sel]
            try:
                cached = preprocess_visionzip_item(
                    processor, base, paths, r["question"], ["A", "B", "C", "D"], device)
            except Exception as e:
                continue
            if cached is None:
                continue
            attn = cached["attn_scores"].float().cpu().numpy()
            N = attn.shape[0]
            T = int(cached["grid_thw"][0, 0].item())
            Hg = int(cached["grid_thw"][0, 1].item()); Wg = int(cached["grid_thw"][0, 2].item())
            n_sp = N // max(1, T)
            s_h, s_w = _solve_spatial_dims(n_sp, Hg, Wg)
            if s_h * s_w != n_sp or T < 1:
                continue
            for t in range(T):
                # align merged slot t -> sampled frame -> gfn -> gaze
                fi = int(round(t * (len(sel) - 1) / max(1, T - 1)))
                gfn = gfns[min(fi, len(gfns) - 1)]
                if gfn not in gaze:
                    continue
                gx, gy = gaze[gfn]
                amap = attn[t * n_sp:(t + 1) * n_sp].reshape(s_h, s_w)
                ar, ac = np.unravel_index(amap.argmax(), amap.shape)
                ax, ay = (ac + 0.5) / s_w, (ar + 0.5) / s_h
                rr, rc = rng.randint(s_h), rng.randint(s_w)
                rx, ry = (rc + 0.5) / s_w, (rr + 0.5) / s_h
                D["attn_gaze"].append(np.hypot(ax - gx, ay - gy))
                D["center_gaze"].append(np.hypot(0.5 - gx, 0.5 - gy))
                D["attn_center"].append(np.hypot(ax - 0.5, ay - 0.5))
                D["rand_gaze"].append(np.hypot(rx - gx, ry - gy))
            used += 1
            if used % 20 == 0:
                print(f"  ...{used} items", flush=True)

    print(f"\n[{a.label}] items={used}  frames={len(D['attn_gaze'])}")
    print(f"{'metric':14s} {'median':>8} {'mean':>8}")
    for k in ("attn_gaze", "center_gaze", "attn_center", "rand_gaze"):
        v = np.array(D[k])
        print(f"{k:14s} {np.median(v):8.3f} {v.mean():8.3f}")
    ag = np.array(D["attn_gaze"]); cg = np.array(D["center_gaze"])
    print(f"\n  gaze center-bias (d_center_gaze<0.15): {100*(cg<0.15).mean():.0f}%")
    print(f"  attention tracks gaze beyond center? "
          f"d_attn_gaze({ag.mean():.3f}) vs d_center_gaze({cg.mean():.3f}): "
          f"{'attn≈center-only' if ag.mean()>=cg.mean()-0.02 else 'attn TRACKS gaze'}")


if __name__ == "__main__":
    main()
