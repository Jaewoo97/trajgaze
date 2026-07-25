"""Foveated ROI tokens — gaze/hand-centered HIGH-RESOLUTION patches injected into
the LLM sequence.

Unlike every prior method here (VisionZip / M1 / query / scanpath), which only
RE-SELECTS existing low-res video tokens, this adds *new pixels*: it crops the
gaze/hand region of fixation frames, re-encodes the crop at higher resolution
through the SAME Qwen ViT, and injects the resulting patch tokens. The crop tokens
live in the ViT embedding space the LLM already understands (unlike scanpath's
abstract trajectory tokens, which gated to ~0), so they are far more likely to be
used. This is the one regime-change lever that supplies information no token
reshuffle can — detail on small interaction targets (button, knife tip, handle).

Two budget regimes (set FovealROIConfig.mode):
  'added'   (V1)  global 10% kept intact + K foveal tokens appended (budget grows).
                  Tests whether high-res foveal info is usable at all. Simpler.
  'replace' (V2)  global 7% + foveal 3% (reduce gaze complement; fixed 10% budget).
                  Tests whether reallocating budget to resolution beats M1.

Everything is parameterized through FovealROIConfig so the sweet spot can be swept
later (see SWEEP_GRID + roi_config_from_args). A future trainer adds the matching
--roi-* flags and calls: detect_fixation_frames → extract_roi_crops →
encode_foveal_tokens → build_inputs_with_foveal.

This module is CURVE-INDEPENDENT (no dependence on the 5%/10% budget-curve result).
The geometry + fixation logic is unit-tested on CPU; encode/inject are validated
when the trainer is wired (they need the live ViT).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ── Sweepable configuration ──────────────────────────────────────────────────

@dataclass
class FovealROIConfig:
    """All ROI knobs in one place so the sweet spot is sweepable. Defaults are the
    recommended V1 starting point (≈1/4-frame fixation crops, K=32, fixation frames)."""
    # WHICH frames
    n_fix_frames:     int   = 4       # number of fixation frames to crop
    min_frame_gap:    float = 0.08    # temporal NMS: min clip-fraction gap between fixations
    fixation_sigma_v: float = 0.05    # gaze-speed scale for fixation weight 1/(1+v/σ)
    hand_boost:       float = 0.5     # extra fixation weight when gaze is near a hand
    sigma_gh:         float = 0.10    # gaze-hand proximity scale for that boost
    # CROP geometry (fractions of frame dimension)
    crop_frac:        float = 0.25    # base crop side as fraction of W/H
    margin_frac:      float = 0.05    # margin added around the union box
    include_hand:     bool  = True    # expand crop toward a nearby active hand
    max_crop_frac:    float = 0.50    # cap so a hand-union crop can't swallow the frame
    # RESOLUTION + token count
    roi_max_pixels:   int   = 256 * 28 * 28   # ViT budget per crop (the resolution knob)
    foveal_k:         int   = 32      # total foveal tokens across all crops
    pool:             str   = "uniform"       # 'uniform' | 'topk_attn' | 'all'
    # ① ANTICIPATORY (where gaze is HEADING, not where it is)
    antic_horizon:    int   = 12      # lookahead steps for gaze velocity extrapolation
    antic_min_speed:  float = 0.01    # min gaze speed/step to count as directed motion (saccade)
    # BUDGET regime
    mode:             str   = "added"         # 'added' (V1) | 'replace' (V2)

    def describe(self) -> str:
        return (f"ROI[{self.mode}] fix={self.n_fix_frames} crop={self.crop_frac} "
                f"K={self.foveal_k} res={self.roi_max_pixels // (28 * 28)}patch pool={self.pool}]")


# Example grid for the eventual sweet-spot sweep (each axis varied around the V1 default).
SWEEP_GRID = {
    "crop_frac":      [0.15, 0.25, 0.35],
    "foveal_k":       [16, 32, 64],
    "roi_max_pixels": [128 * 28 * 28, 256 * 28 * 28, 512 * 28 * 28],
    "n_fix_frames":   [2, 4, 8],
}


def roi_config_from_args(args) -> FovealROIConfig:
    """Build a config from an argparse namespace (the future trainer's --roi-* flags).
    Reads each field with getattr so partial namespaces fall back to defaults."""
    d = FovealROIConfig()
    for f in d.__dataclass_fields__:
        v = getattr(args, f, None)
        if v is None:
            v = getattr(args, "roi_" + f, None)   # also accept --roi-<field> style
        if v is not None:
            setattr(d, f, v)
    return d


# ── Fixation detection (CPU-testable, no model) ──────────────────────────────

def _as2d(x):
    t = torch.as_tensor(x, dtype=torch.float32)
    return t if (t.ndim == 2 and t.shape[1] == 2) else None


def detect_fixation_frames(traj: dict | None, n_frames_total: int,
                           cfg: FovealROIConfig) -> list[dict]:
    """Rank fixation moments in the raw gaze trace and return up to cfg.n_fix_frames
    of them, temporally spread (NMS). Each entry:
        {frac, frame_idx, gaze:(x,y), hand:(x,y)|None, score}
    where frac is clip-time in [0,1] and frame_idx indexes vlm_frame_paths. Returns
    [] for missing/empty/invalid gaze (caller then adds no foveal tokens)."""
    if not isinstance(traj, dict):
        return []
    gaze = _as2d(traj.get("gaze_pos"))
    if gaze is None or gaze.shape[0] == 0:
        return []
    L = gaze.shape[0]
    gmask = torch.as_tensor(traj.get("gaze_mask", torch.ones(L)), dtype=torch.float32).view(-1)
    if gmask.shape[0] != L:
        gmask = torch.ones(L)

    # fixation weight: low gaze speed + valid gaze. Step 0 has no predecessor, so it
    # inherits step 1's speed — otherwise its zero speed makes frame 0 a spurious
    # always-fixation and wastes a foveal slot on the clip's first frame.
    speed = torch.zeros(L)
    if L > 1:
        speed[1:] = (gaze[1:] - gaze[:-1]).norm(dim=-1) * (gmask[1:] * gmask[:-1])
        speed[0] = speed[1]
    score = (1.0 / (1.0 + speed / max(1e-6, cfg.fixation_sigma_v))) * gmask

    # boost when gaze sits on an active hand (interaction moment)
    left, lmask = _as2d(traj.get("left_pos")),  torch.as_tensor(traj.get("left_mask", torch.zeros(L)), dtype=torch.float32).view(-1)
    right, rmask = _as2d(traj.get("right_pos")), torch.as_tensor(traj.get("right_mask", torch.zeros(L)), dtype=torch.float32).view(-1)
    for hpos, hmask in ((left, lmask), (right, rmask)):
        if hpos is not None and hpos.shape[0] == L and hmask.shape[0] == L:
            d = (gaze - hpos).norm(dim=-1)
            score = score + cfg.hand_boost * torch.exp(-d / max(1e-6, cfg.sigma_gh)) * gmask * hmask

    # greedy top-n with temporal NMS (min_frame_gap in clip fraction)
    order = torch.argsort(score, descending=True).tolist()
    gap = max(1, int(cfg.min_frame_gap * L))
    picked: list[int] = []
    for idx in order:
        if score[idx] <= 0:
            break
        if all(abs(idx - p) >= gap for p in picked):
            picked.append(idx)
        if len(picked) >= cfg.n_fix_frames:
            break
    picked.sort()

    def hand_at(i):
        cands = []
        if left is not None and left.shape[0] == L and lmask[i] > 0:
            cands.append(left[i])
        if right is not None and right.shape[0] == L and rmask[i] > 0:
            cands.append(right[i])
        if not cands:
            return None
        # nearest hand to the gaze point
        g = gaze[i]
        return min(cands, key=lambda h: (g - h).norm().item()).tolist()

    out = []
    for i in picked:
        frac = i / max(1, L - 1)
        frame_idx = int(round(frac * max(0, n_frames_total - 1)))
        out.append({
            "frac": frac, "frame_idx": frame_idx,
            "gaze": gaze[i].tolist(), "hand": hand_at(i),
            "score": float(score[i]),
        })
    return out


# ── Crop geometry (CPU-testable, no model) ───────────────────────────────────

def gaze_crop_box(gx: float, gy: float, W: int, H: int, cfg: FovealROIConfig,
                  hand: list | None = None) -> tuple[int, int, int, int]:
    """Map a normalized gaze point (+ optional hand) to an integer pixel crop box
    (left, top, right, bottom), clamped to the frame and capped at max_crop_frac."""
    hw = cfg.crop_frac * W / 2.0
    hh = cfg.crop_frac * H / 2.0
    cx, cy = gx * W, gy * H
    l, t, r, b = cx - hw, cy - hh, cx + hw, cy + hh

    if cfg.include_hand and hand is not None:
        hxp, hyp = hand[0] * W, hand[1] * H
        m = cfg.margin_frac
        l = min(l, hxp - m * W); r = max(r, hxp + m * W)
        t = min(t, hyp - m * H); b = max(b, hyp + m * H)
        # cap size at max_crop_frac, re-centered on the union midpoint
        maxw, maxh = cfg.max_crop_frac * W, cfg.max_crop_frac * H
        if (r - l) > maxw:
            mid = (l + r) / 2.0; l, r = mid - maxw / 2.0, mid + maxw / 2.0
        if (b - t) > maxh:
            mid = (t + b) / 2.0; t, b = mid - maxh / 2.0, mid + maxh / 2.0

    # clamp to frame; guarantee a non-degenerate box
    l = int(max(0, min(l, W - 2))); t = int(max(0, min(t, H - 2)))
    r = int(max(l + 1, min(r, W)));  b = int(max(t + 1, min(b, H)))
    return l, t, r, b


def crops_from_fixations(frame_paths: list[str], fixations: list[dict],
                         cfg: FovealROIConfig):
    """Load each fixation's frame and crop the point-centered box → [(PIL crop, meta), ...].
    Arm-agnostic: `fixations` may come from gaze (detect_fixation_frames), attention
    (detect_attn_frames), or random (detect_random_frames); each entry only needs
    {frame_idx, gaze:(x,y), hand:(x,y)|None}. Missing/unreadable frames are skipped.
    Requires PIL; frames are loaded lazily per fixation."""
    from PIL import Image
    crops = []
    for f in fixations:
        fi = f["frame_idx"]
        if fi < 0 or fi >= len(frame_paths):
            continue
        try:
            img = Image.open(frame_paths[fi]).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        box = gaze_crop_box(f["gaze"][0], f["gaze"][1], W, H, cfg, hand=f.get("hand"))
        crops.append((img.crop(box), {**f, "box": box, "src_wh": (W, H)}))
    return crops


def extract_roi_crops(frame_paths: list[str], traj: dict | None, cfg: FovealROIConfig):
    """Gaze arm (the method): detect gaze fixations and crop them. Empty gaze → []."""
    fix = detect_fixation_frames(traj, len(frame_paths), cfg)
    return crops_from_fixations(frame_paths, fix, cfg)


# ── Control-arm fixations (attention-twin + random placebo) ───────────────────

def detect_attn_frames(attn_scores, t_merged: int, s_h: int, s_w: int,
                       n_frames_total: int, cfg: FovealROIConfig) -> list[dict]:
    """ATTENTION-TWIN control for the ROI arm — the same mechanism as the gaze path
    but driven by VisionZip's content attention instead of gaze. Per-frame attention
    mass ranks frames (temporal NMS, same min_frame_gap), and within each picked frame
    the argmax patch gives the crop center. hand=None on purpose: a content-only twin
    must not borrow gaze-derived hand positions. Returns the same dict format as
    detect_fixation_frames so crops_from_fixations is arm-agnostic.
    attn_scores: (N,) with N = t_merged * s_h * s_w (VisionZip's video-token layout)."""
    n_spatial = s_h * s_w
    N = t_merged * n_spatial
    a = torch.as_tensor(attn_scores, dtype=torch.float32).view(-1)
    if N == 0 or a.numel() < N:
        return []
    a2d = a[:N].view(t_merged, n_spatial)
    frame_mass = a2d.sum(dim=1)                                   # (T,)
    order = torch.argsort(frame_mass, descending=True).tolist()
    gap = max(1, int(cfg.min_frame_gap * t_merged))
    picked: list[int] = []
    for t in order:
        if all(abs(t - p) >= gap for p in picked):
            picked.append(t)
        if len(picked) >= cfg.n_fix_frames:
            break
    picked.sort()
    out = []
    for t in picked:
        pos = int(torch.argmax(a2d[t]).item())
        r, c = divmod(pos, s_w)
        out.append({
            "frac": t / max(1, t_merged - 1),
            "frame_idx": int(round((t / max(1, t_merged - 1)) * max(0, n_frames_total - 1))),
            "gaze": [(c + 0.5) / s_w, (r + 0.5) / s_h], "hand": None,
            "score": float(frame_mass[t]),
        })
    return out


def detect_anticipatory_frames(traj: dict | None, n_frames_total: int,
                               cfg: FovealROIConfig) -> list[dict]:
    """① ANTICIPATORY gaze — crop the CURRENT frame where gaze will land Δ steps LATER.

    Gaze leads action by ~0.5–1s: before the next action is visible, the eye has already
    moved to its target. We use the ACTUAL recorded future fixation gaze[i+Δ] (Δ =
    antic_horizon) as the crop center on the *current* frame i, picking the moments of
    largest anticipatory shift ‖gaze[i+Δ]−gaze[i]‖ (NMS-spread). Unlike velocity
    extrapolation (which overshoots off-frame on this coarse-sampled, fast-moving gaze),
    gaze[i+Δ] is a real on-frame point. A content-attention twin sees only frame i's
    current salience — it has no access to where gaze goes next — so it cannot produce
    this crop; that impossibility is the ① contribution's structural protection. hand=None.

    Returns detect_fixation_frames' dict format (gaze = the *anticipated* future point;
    `cur` = the current gaze). Empty gaze or no valid i→i+Δ pair → []."""
    if not isinstance(traj, dict):
        return []
    gaze = _as2d(traj.get("gaze_pos"))
    if gaze is None or gaze.shape[0] == 0:
        return []
    L = gaze.shape[0]
    gmask = torch.as_tensor(traj.get("gaze_mask", torch.ones(L)), dtype=torch.float32).view(-1)
    if gmask.shape[0] != L:
        gmask = torch.ones(L)
    D = max(1, int(cfg.antic_horizon))

    # candidate moments: current + future fixation both valid; score = anticipatory shift
    cand = []  # (shift, i)
    for i in range(0, L - D):
        if gmask[i] > 0 and gmask[i + D] > 0:
            shift = float((gaze[i + D] - gaze[i]).norm())
            cand.append((shift, i))
    if not cand:
        return []
    cand.sort(reverse=True)                                  # most anticipatory first

    gap = max(1, int(cfg.min_frame_gap * L))
    picked: list[int] = []
    for _, idx in cand:
        if all(abs(idx - p) >= gap for p in picked):
            picked.append(idx)
        if len(picked) >= cfg.n_fix_frames:
            break
    picked.sort()

    out = []
    for i in picked:
        land = gaze[i + D]                                   # actual future fixation (on-frame)
        frac = i / max(1, L - 1)
        out.append({
            "frac": frac, "frame_idx": int(round(frac * max(0, n_frames_total - 1))),
            "gaze": [float(land[0]), float(land[1])], "hand": None,
            "cur": gaze[i].tolist(),
            "score": float((gaze[i + D] - gaze[i]).norm()),
        })
    return out


def detect_random_frames(n_frames_total: int, cfg: FovealROIConfig,
                         seed: int = 0) -> list[dict]:
    """RANDOM placebo control — cfg.n_fix_frames random frames, each cropped at a random
    center. Tests that any gaze-ROI gain comes from WHERE gaze looks, not merely from
    adding K high-res tokens at arbitrary spots. Deterministic given seed; hand=None."""
    if n_frames_total <= 0:
        return []
    g = torch.Generator().manual_seed(int(seed))
    n = min(max(1, cfg.n_fix_frames), n_frames_total)
    frames = torch.randperm(n_frames_total, generator=g)[:n].sort().values.tolist()
    pts = torch.rand(n, 2, generator=g)
    return [{
        "frac": fi / max(1, n_frames_total - 1), "frame_idx": int(fi),
        "gaze": [float(pts[j, 0]), float(pts[j, 1])], "hand": None, "score": 0.0,
    } for j, fi in enumerate(frames)]


# ── ViT re-encoding (needs the live model; validated when the trainer is wired) ──

@torch.no_grad()
def encode_foveal_tokens(processor, base_qwen, crops, cfg: FovealROIConfig,
                         device) -> torch.Tensor | None:
    """Re-encode crop images at cfg.roi_max_pixels through Qwen's ViT and pool to
    cfg.foveal_k tokens (in LLM embedding space). Returns (K, d) or None.

    base_qwen.visual may return either embeds or (embeds, attn, key) (VisionZip
    fork) — both are handled; 'topk_attn' pooling needs the attn variant."""
    if not crops:
        return None
    images = [c for c, _ in crops]
    # Use the image_processor component directly: Qwen2_5_VLProcessor.__call__ requires
    # `text` (text=None raises), but image_processor(images=...) returns the pixel_values
    # + image_grid_thw the ViT needs. max_pixels is the resolution knob (the whole point
    # of foveal re-encoding); fall back to the processor's default cap if it's rejected.
    ip = getattr(processor, "image_processor", processor)
    try:
        inputs = ip(images=images, max_pixels=cfg.roi_max_pixels, return_tensors="pt")
    except Exception:
        try:
            inputs = ip(images=images, return_tensors="pt")
        except Exception:
            return None
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        return None

    vis_dev = base_qwen.visual.patch_embed.proj.weight.device
    emb_dev = base_qwen.get_input_embeddings().weight.device
    pv = inputs["pixel_values"].to(vis_dev, torch.bfloat16)
    grid = inputs["image_grid_thw"].to(vis_dev)

    out = base_qwen.visual(pv, grid_thw=grid)
    if isinstance(out, (tuple, list)):
        embeds = out[0].to(emb_dev)
        attn = out[1].to(emb_dev) if len(out) > 1 and out[1] is not None else None
    else:
        embeds = out.to(emb_dev)
        attn = None
    P = embeds.shape[0]
    K = min(cfg.foveal_k, P)
    if cfg.pool == "all" or K >= P:
        sel = torch.arange(P, device=embeds.device)
    elif cfg.pool == "topk_attn" and attn is not None:
        sel = torch.topk(attn.float().view(-1)[:P], K).indices
    else:  # 'uniform' — evenly spaced across all crop patches
        sel = torch.linspace(0, P - 1, K, device=embeds.device).round().long()
    return embeds[sel]


# ── Sequence injection (adapted verbatim from build_inputs_with_gaze) ─────────

def build_inputs_with_foveal(base_qwen, cached: dict, merged_video: torch.Tensor,
                             receiver_idx: torch.Tensor, foveal_embeds: torch.Tensor) -> dict:
    """Like model.build_merged_inputs, but inserts K foveal ROI tokens right after
    the video block, with position ids continuing the post-video positions and
    everything after shifted by K. foveal_embeds: (K, d). If None/empty, falls back
    to a plain merged build (no foveal tokens)."""
    input_ids       = cached["input_ids"]
    attention_mask  = cached["attention_mask"]
    position_ids    = cached["position_ids"]
    rope_deltas     = cached["rope_deltas"]
    video_embeds    = cached["video_embeds"]
    video_positions = cached["video_positions"]
    emb_dev         = cached["emb_dev"]
    N_video = video_embeds.shape[0]

    is_receiver = torch.zeros(N_video, dtype=torch.bool, device=emb_dev)
    is_receiver[receiver_idx] = True
    source_video_pos = video_positions[~is_receiver]

    keep_seq = torch.ones(input_ids.shape[1], dtype=torch.bool, device=emb_dev)
    keep_seq[source_video_pos] = False

    new_input_ids    = input_ids[:, keep_seq]
    new_attn_mask    = attention_mask[:, keep_seq]
    new_position_ids = position_ids[:, :, keep_seq]

    new_inputs_embeds = base_qwen.get_input_embeddings()(new_input_ids)
    video_token_id = base_qwen.config.video_token_id
    new_is_video   = (new_input_ids[0] == video_token_id)
    new_inputs_embeds[0, new_is_video] = merged_video.to(new_inputs_embeds.dtype)

    if foveal_embeds is None or foveal_embeds.shape[0] == 0:
        return {"inputs_embeds": new_inputs_embeds, "attention_mask": new_attn_mask,
                "position_ids": new_position_ids, "rope_deltas": rope_deltas}

    K = foveal_embeds.shape[0]
    last_vid = int(new_is_video.nonzero(as_tuple=True)[0][-1].item())
    ins = last_vid + 1
    ft = foveal_embeds.to(new_inputs_embeds.dtype).unsqueeze(0)
    emb = torch.cat([new_inputs_embeds[:, :ins], ft, new_inputs_embeds[:, ins:]], dim=1)

    ones = torch.ones(1, K, dtype=new_attn_mask.dtype, device=emb_dev)
    am = torch.cat([new_attn_mask[:, :ins], ones, new_attn_mask[:, ins:]], dim=1)

    L = new_position_ids.shape[2]
    s = new_position_ids[:, :, ins:ins + 1] if ins < L else new_position_ids[:, :, -1:] + 1
    fov_pid = s + torch.arange(K, device=emb_dev).view(1, 1, K)
    tail = new_position_ids[:, :, ins:] + K
    new_pid = torch.cat([new_position_ids[:, :, :ins], fov_pid, tail], dim=2)

    return {"inputs_embeds": emb, "attention_mask": am,
            "position_ids": new_pid, "rope_deltas": rope_deltas}
