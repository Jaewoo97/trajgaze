"""QC-Gate: Query-Conditioned Coordination Gating for VisionZip-traj.

Extends VisionZip-traj's zero-param fixed prior with a small LEARNED controller
g_theta that reads [pooled question embedding (+) per-frame coordination state]
and emits SIGNED per-frame gates alpha_gaze / alpha_hand / alpha_temp. These
modulate the gaze Gaussian kernel, the hand Gaussian kernel, and the temporal
interaction softmax respectively, BEFORE VisionZip's top-K dominant selection.

Novelty over VisionZip-traj (which uses a fixed +1.0 gaze / +0.7 hand prior):
  1. Query-aware      - the gate depends on the question, attacking VisionZip's
                        query-agnostic selection through the human-trajectory bridge.
  2. Coordination-gated - uses the otherwise-unused convergence / lead_lag /
                        relative-velocity signals from the interaction npz.
  3. Signed + learned - alpha can go negative (gaze/hand as a *repulsor*) or ~0
                        (coverage mode), expressing behaviours the fixed prior can't.

Initialization makes QC-Gate == VisionZip-traj EXACTLY at step 0
(alpha_gaze~1.0, alpha_hand~0.7, alpha_temp~0, controller output weights = 0), so
training departs from VZ-traj's operating point rather than from scratch.

Gradient to the controller flows through a straight-through gate on the selected
dominant tokens: hard top-K on the DETACHED reweighted scores (identical selection
rule to VZ/VZ-traj), then each selected dominant embedding is scaled by
gate_i = s_i / s_i.detach() (forward = 1, so the inference distribution is
unchanged; backward carries d s_i / d theta). Only selected dominant tokens carry
gradient (DynamicViT-style; accepted limitation for v1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from TrajGazeMerge.models.traj_weights import (
    _pool_to_T, _solve_spatial_dims,
    SIGMA_GAZE, SIGMA_HAND, SIGMA_VEL, SIGMA_GH_DIST,
    BETA_FIX, BETA_HAND_PRES, BETA_GAZE_HAND,
)
from TrajGazeMerge.training.train_visionzip_lora import (
    DOMINANT_RATIO, CONTEXTUAL_RATIO,
)

COORD_DIM   = 8       # per-frame coordination features (see compute_coord_features)
Q_PROJ_DIM  = 64      # question-embedding bottleneck
HIDDEN_DIM  = 64      # controller MLP hidden width
GATE_AMP    = 1.5     # max |alpha| = GATE_AMP (tanh-saturated)
QWEN_HIDDEN = 3584    # Qwen2.5-VL-7B token embedding dim

# init biases so that amp*tanh(bias) == the VZ-traj fixed prior at step 0
_INIT_ALPHA_GAZE = 1.0
_INIT_ALPHA_HAND = 0.7


def _pool_scalar_to_T(x: torch.Tensor, T_target: int) -> torch.Tensor:
    """Map a (T_src,) scalar trajectory signal to (T_target,) via linear interp."""
    T_src = x.shape[0]
    if T_src == T_target:
        return x.float()
    xx = x.float().view(1, 1, T_src)
    xt = F.interpolate(xx, size=T_target, mode="linear", align_corners=False)
    return xt.view(T_target)


# ── Separated spatial kernels + base temporal interaction ─────────────────────

@torch.no_grad()
def compute_traj_kernels(
    traj_dict: dict,
    T_merged: int,
    H: int,
    W: int,
    device: torch.device,
    sigma_g: float = SIGMA_GAZE,
    sigma_h: float = SIGMA_HAND,
    sigma_v: float = SIGMA_VEL,
    sigma_gh: float = SIGMA_GH_DIST,
):
    """Return gaze/hand spatial kernels SEPARATELY + base temporal interaction.

    K_gaze : (T, HW)  gaze_mask * exp(-d_gaze / 2 sigma_g^2)
    K_hand : (T, HW)  left_mask * exp(-d_left / 2 sigma_h^2)
                    + right_mask * exp(-d_right / 2 sigma_h^2)
    base_interaction : (T,)  fixation + hand_present + gaze_on_hand (pre-softmax)

    1 + 1.0*K_gaze + 0.7*K_hand reproduces VisionZip-traj's (1 + spatial_w);
    softmax(base_interaction)*T reproduces its temporal_w. QC-Gate replaces the
    fixed 1.0 / 0.7 / (alpha_temp=0) with learned signed gates. All tensors are
    float32 (kernel grids are deterministic, no learnable params here).
    """
    gaze_pos,  gaze_mask  = _pool_to_T(traj_dict["gaze_pos"],  traj_dict["gaze_mask"],  T_merged)
    left_pos,  left_mask  = _pool_to_T(traj_dict["left_pos"],  traj_dict["left_mask"],  T_merged)
    right_pos, right_mask = _pool_to_T(traj_dict["right_pos"], traj_dict["right_mask"], T_merged)

    gaze_pos  = gaze_pos.to(device);  left_pos = left_pos.to(device);  right_pos = right_pos.to(device)
    gaze_mask = gaze_mask.to(device).float()
    left_mask = left_mask.to(device).float()
    right_mask = right_mask.to(device).float()

    py = torch.arange(H, device=device).float()
    px = torch.arange(W, device=device).float()
    py_grid, px_grid = torch.meshgrid(py, px, indexing="ij")
    py_grid = py_grid.reshape(-1)
    px_grid = px_grid.reshape(-1)

    gx = gaze_pos[:, 0] * W;  gy = gaze_pos[:, 1] * H
    lx = left_pos[:, 0] * W;  ly = left_pos[:, 1] * H
    rx = right_pos[:, 0] * W; ry = right_pos[:, 1] * H

    d_gaze  = (px_grid.unsqueeze(0) - gx.unsqueeze(1))**2 + (py_grid.unsqueeze(0) - gy.unsqueeze(1))**2
    d_left  = (px_grid.unsqueeze(0) - lx.unsqueeze(1))**2 + (py_grid.unsqueeze(0) - ly.unsqueeze(1))**2
    d_right = (px_grid.unsqueeze(0) - rx.unsqueeze(1))**2 + (py_grid.unsqueeze(0) - ry.unsqueeze(1))**2

    K_gaze = gaze_mask.unsqueeze(1) * torch.exp(-d_gaze / (2 * sigma_g**2))          # (T, HW)
    K_hand = (
        left_mask.unsqueeze(1)  * torch.exp(-d_left  / (2 * sigma_h**2))
      + right_mask.unsqueeze(1) * torch.exp(-d_right / (2 * sigma_h**2))
    )                                                                                # (T, HW)

    # temporal interaction (same recipe as compute_traj_weights, pre-softmax)
    gaze_speed = torch.zeros(T_merged, device=device)
    if T_merged > 1:
        d = (gaze_pos[1:] - gaze_pos[:-1]).norm(dim=-1)
        valid = gaze_mask[1:] * gaze_mask[:-1]
        gaze_speed[1:] = d * valid
    fixation     = 1.0 / (1.0 + gaze_speed / sigma_v)
    hand_present = 0.5 * (left_mask + right_mask)
    d_gh_l = (gaze_pos - left_pos).norm(dim=-1)
    d_gh_r = (gaze_pos - right_pos).norm(dim=-1)
    BIG = torch.full_like(d_gh_l, 1e6)
    d_gh = torch.minimum(
        torch.where(left_mask > 0, d_gh_l, BIG),
        torch.where(right_mask > 0, d_gh_r, BIG),
    )
    any_hand     = ((left_mask + right_mask) > 0).float()
    gaze_on_hand = torch.exp(-d_gh / sigma_gh) * gaze_mask * any_hand
    base_interaction = (
        BETA_FIX * fixation + BETA_HAND_PRES * hand_present + BETA_GAZE_HAND * gaze_on_hand
    )                                                                                # (T,)

    return K_gaze, K_hand, base_interaction


# ── Per-frame coordination features for the controller ────────────────────────

@torch.no_grad()
def compute_coord_features(
    traj_dict: dict,
    T_merged: int,
    device: torch.device,
    sigma_v: float = SIGMA_VEL,
    sigma_gh: float = SIGMA_GH_DIST,
) -> torch.Tensor:
    """(T_merged, COORD_DIM) per-frame coordination state, z-scored per feature.

    Features: [fixation, hand_present, gaze_on_hand, convergence, lead_lag,
               gaze_mask, |v_rel| (nearer hand), d_gh (gaze-hand dist, clamped)].
    Robust to sources whose interaction npz is absent (convergence / lead_lag /
    v_rel default to zeros -> z-score -> 0). All values finite (nan_to_num'd).
    """
    gaze_pos,  gaze_mask  = _pool_to_T(traj_dict["gaze_pos"],  traj_dict["gaze_mask"],  T_merged)
    left_pos,  left_mask  = _pool_to_T(traj_dict["left_pos"],  traj_dict["left_mask"],  T_merged)
    right_pos, right_mask = _pool_to_T(traj_dict["right_pos"], traj_dict["right_mask"], T_merged)
    gaze_pos  = gaze_pos.to(device);  left_pos = left_pos.to(device);  right_pos = right_pos.to(device)
    gaze_mask = gaze_mask.to(device).float()
    left_mask = left_mask.to(device).float()
    right_mask = right_mask.to(device).float()

    gaze_speed = torch.zeros(T_merged, device=device)
    if T_merged > 1:
        d = (gaze_pos[1:] - gaze_pos[:-1]).norm(dim=-1)
        valid = gaze_mask[1:] * gaze_mask[:-1]
        gaze_speed[1:] = d * valid
    fixation     = 1.0 / (1.0 + gaze_speed / sigma_v)
    hand_present = 0.5 * (left_mask + right_mask)

    d_gh_l = (gaze_pos - left_pos).norm(dim=-1)
    d_gh_r = (gaze_pos - right_pos).norm(dim=-1)
    BIG = torch.full_like(d_gh_l, 1e6)
    d_gh_raw = torch.minimum(
        torch.where(left_mask > 0, d_gh_l, BIG),
        torch.where(right_mask > 0, d_gh_r, BIG),
    )
    any_hand     = ((left_mask + right_mask) > 0).float()
    gaze_on_hand = torch.exp(-d_gh_raw / sigma_gh) * gaze_mask * any_hand
    # finite sentinel (max normalized dist ~ sqrt(2)) so z-scoring stays stable
    d_gh = torch.where(any_hand > 0, d_gh_raw.clamp(max=1.5), torch.full_like(d_gh_raw, 1.5))

    def _scalar(key):
        v = traj_dict.get(key)
        if v is None:
            return torch.zeros(T_merged, device=device)
        return _pool_scalar_to_T(v.to(device), T_merged)

    conv     = _scalar("convergence")
    lead_lag = _scalar("lead_lag")

    vrl = traj_dict.get("v_rel_left")
    vrr = traj_dict.get("v_rel_right")
    if vrl is not None and vrr is not None:
        mag = torch.maximum(vrl.float().norm(dim=-1), vrr.float().norm(dim=-1))   # (T_src,)
        v_rel_mag = _pool_scalar_to_T(mag.to(device), T_merged)
    else:
        v_rel_mag = torch.zeros(T_merged, device=device)

    feats = torch.stack([
        fixation,        # 0  gaze stillness
        hand_present,    # 1  fraction of hands visible
        gaze_on_hand,    # 2  gaze proximity to nearer hand (kernel)
        conv,            # 3  gaze<->hand convergence (npz)
        lead_lag,        # 4  gaze/hand temporal lead-lag (npz)
        gaze_mask,       # 5  is gaze tracked this frame
        v_rel_mag,       # 6  relative gaze-hand velocity magnitude
        d_gh,            # 7  gaze-hand distance (clamped)
    ], dim=-1)           # (T, COORD_DIM)

    mean = feats.mean(dim=0, keepdim=True)
    std  = feats.std(dim=0, keepdim=True)
    feats = (feats - mean) / (std + 1e-5)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ── The learned controller g_theta ────────────────────────────────────────────

class QCGateController(nn.Module):
    """g_theta: [q_proj(question) (+) coord_feats[t]] -> signed (alpha_gaze,
    alpha_hand, alpha_temp)[t]. Tiny per-frame MLP shared across frames."""

    def __init__(self, qwen_hidden=QWEN_HIDDEN, coord_dim=COORD_DIM,
                 q_proj_dim=Q_PROJ_DIM, hidden=HIDDEN_DIM, amp=GATE_AMP):
        super().__init__()
        self.amp = amp
        self.q_proj = nn.Linear(qwen_hidden, q_proj_dim)
        self.fc1    = nn.Linear(q_proj_dim + coord_dim, hidden)
        self.act    = nn.GELU()
        self.fc2    = nn.Linear(hidden, 3)

        # Init so QC-Gate == VZ-traj at step 0: zero output weights, biases set so
        # amp*tanh(bias) == (1.0, 0.7, 0.0). With fc2.weight=0 the output is the
        # bias regardless of input, so the gates are exactly the fixed prior until
        # gradient (first through fc2.weight) unlocks input dependence.
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias[0] = math.atanh(_INIT_ALPHA_GAZE / amp)
            self.fc2.bias[1] = math.atanh(_INIT_ALPHA_HAND / amp)
            self.fc2.bias[2] = 0.0

    def forward(self, q_vec: torch.Tensor, coord_feats: torch.Tensor):
        """q_vec: (qwen_hidden,) detached pooled question embedding.
           coord_feats: (T, coord_dim).
           returns alpha_gaze (T,), alpha_hand (T,), alpha_temp (T,)."""
        dtype = self.q_proj.weight.dtype
        q_vec = q_vec.to(dtype)
        coord_feats = coord_feats.to(dtype)
        T = coord_feats.shape[0]
        q = self.q_proj(q_vec).unsqueeze(0).expand(T, -1)        # (T, q_proj_dim)
        x = torch.cat([q, coord_feats], dim=-1)                  # (T, q_proj_dim+coord_dim)
        h = self.act(self.fc1(x))
        out = self.fc2(h)                                        # (T, 3)
        gates = self.amp * torch.tanh(out)
        return gates[:, 0], gates[:, 1], gates[:, 2]


@torch.no_grad()
def pooled_question_embedding(base_qwen, cached: dict) -> torch.Tensor:
    """Mean-pooled text-token embedding of the prompt (system + question +
    options), DETACHED. (qwen_hidden,) on the embedding device. Video-placeholder
    tokens are excluded so only the question text conditions the controller."""
    input_ids = cached["input_ids"][0]                           # (L,)
    video_token_id = base_qwen.config.video_token_id
    emb = base_qwen.get_input_embeddings()(input_ids)            # (L, d)
    text_mask = input_ids != video_token_id
    return emb[text_mask].mean(0).detach()


# ── Reweighting + straight-through selection ──────────────────────────────────

def qcgate_reweight_scores(
    attn_scores: torch.Tensor,      # (N,)
    grid_thw: torch.Tensor,         # (1, 3): (T, H_grid, W_grid)
    traj_dict: dict,
    q_vec: torch.Tensor,            # (qwen_hidden,) detached
    controller,                     # QCGateController (or DDP-wrapped)
    device: torch.device,
    sigma_g: float = SIGMA_GAZE,
    sigma_h: float = SIGMA_HAND,
    sigma_v: float = SIGMA_VEL,
    sigma_gh: float = SIGMA_GH_DIST,
):
    """Differentiable reweighted scores s (N,) with learned signed gates.

        spatial_factor[t] = clamp(1 + a_gaze[t]*K_gaze[t] + a_hand[t]*K_hand[t], min=0.05)
        temporal_factor   = softmax(base_interaction * (1 + a_temp)) * T     (mean 1)
        s = attn * spatial_factor_flat * temporal_factor_per_token

    Returns (s, (a_gaze, a_hand, a_temp)). The gate vectors are returned for
    logging; they carry grad, so detach before reducing.
    """
    N = attn_scores.shape[0]
    T = int(grid_thw[0, 0].item())
    H_grid = int(grid_thw[0, 1].item())
    W_grid = int(grid_thw[0, 2].item())
    S_per_frame = N // T
    s_h, s_w = _solve_spatial_dims(S_per_frame, H_grid, W_grid)
    HW = s_h * s_w

    K_gaze, K_hand, base_interaction = compute_traj_kernels(
        traj_dict, T, s_h, s_w, device, sigma_g, sigma_h, sigma_v, sigma_gh,
    )
    coord_feats = compute_coord_features(traj_dict, T, device, sigma_v, sigma_gh)

    a_gaze, a_hand, a_temp = controller(q_vec, coord_feats)      # (T,) each
    a_gaze = a_gaze.to(K_gaze.dtype)
    a_hand = a_hand.to(K_hand.dtype)
    a_temp = a_temp.to(base_interaction.dtype)

    spatial_factor = 1.0 + a_gaze.unsqueeze(1) * K_gaze + a_hand.unsqueeze(1) * K_hand   # (T, HW)
    spatial_factor = spatial_factor.clamp(min=0.05)
    spatial_flat = spatial_factor.reshape(-1)                    # (T*HW,) == (N,)

    temporal_logits = base_interaction * (1.0 + a_temp)          # (T,)
    temporal_factor = F.softmax(temporal_logits, dim=0) * T      # (T,) mean 1
    temporal_per_token = temporal_factor.repeat_interleave(HW)   # (N,)

    s = attn_scores.to(spatial_flat.dtype) * spatial_flat * temporal_per_token
    return s, (a_gaze, a_hand, a_temp)


def qcgate_select_tokens(
    video_embeds: torch.Tensor,        # (N, d)
    reweighted_scores: torch.Tensor,   # (N,) differentiable through controller
    attn_key: torch.Tensor,            # (1, N, d_k)
    training: bool,
    dominant_ratio: float = DOMINANT_RATIO,
    contextual_ratio: float = CONTEXTUAL_RATIO,
):
    """VisionZip two-stage selection on QC-Gate reweighted scores.

    Selection (top-K + contextual clustering) uses reweighted_scores.detach() so
    it is the IDENTICAL rule to VZ/VZ-traj. During training each selected DOMINANT
    embedding is scaled by a straight-through gate gate_i = s_i / s_i.detach()
    (forward = 1; backward = d s_i / d theta), giving the controller gradient
    through exactly the tokens it up-weighted. Contextual centers are ungated.

    Everything is kept in float32; build_merged_inputs casts to the model dtype at
    assignment, so float32-vs-bf16 here is numerically inert at inference.
    """
    N, d = video_embeds.shape
    dominant_num   = max(1, int(dominant_ratio   * N))
    contextual_num = max(1, int(contextual_ratio * N))

    scores_det = reweighted_scores.detach()
    _, topk_indices = torch.topk(scores_det, dominant_num)
    topk_sorted, _  = topk_indices.sort()

    dom_mask = torch.zeros(N, dtype=torch.bool, device=video_embeds.device)
    dom_mask[topk_indices] = True
    non_dom_orig = (~dom_mask).nonzero(as_tuple=True)[0]

    # ── Contextual cluster centers — identical to visionzip_select_tokens ──────
    n_non_dom      = non_dom_orig.shape[0]
    non_dom_embeds = video_embeds[non_dom_orig]
    non_dom_key    = attn_key[0, non_dom_orig, :]
    non_dom_key_n  = F.normalize(non_dom_key.float(), dim=-1)
    step           = max(1, n_non_dom // contextual_num)
    center_local   = torch.arange(0, n_non_dom, step, device=video_embeds.device)[:contextual_num]
    center_key_n   = non_dom_key_n[center_local]
    sim            = non_dom_key_n @ center_key_n.T
    assign         = sim.argmax(dim=-1)
    is_center      = torch.zeros(n_non_dom, dtype=torch.bool, device=video_embeds.device)
    is_center[center_local] = True
    non_center_local = (~is_center).nonzero(as_tuple=True)[0]
    center_embeds  = non_dom_embeds[center_local].float().clone()
    counts         = torch.ones(contextual_num, device=video_embeds.device, dtype=torch.float)
    if non_center_local.numel() > 0:
        assign_nc = assign[non_center_local]
        nc_embeds = non_dom_embeds[non_center_local].float()
        center_embeds.scatter_add_(0, assign_nc.unsqueeze(-1).expand(-1, d), nc_embeds)
        counts.scatter_add_(0, assign_nc,
                            torch.ones(non_center_local.shape[0], device=video_embeds.device))
    center_embeds = center_embeds / counts.unsqueeze(-1)                  # (contextual_num, d) float32
    contextual_orig_idx = non_dom_orig[center_local]

    # ── Dominant embeds + straight-through gate (training only) ────────────────
    dom_embeds = video_embeds[topk_sorted].float()                       # (dominant_num, d) float32
    if training:
        s_sel = reweighted_scores[topk_sorted].float()                   # grad-carrying
        gate = s_sel / (s_sel.detach() + 1e-8)                           # forward = 1
        dom_embeds = dom_embeds * gate.unsqueeze(-1)

    all_embeds = torch.cat([dom_embeds, center_embeds], dim=0)            # float32
    all_idx    = torch.cat([topk_sorted, contextual_orig_idx])
    sort_order = all_idx.argsort()
    receiver_idx    = all_idx[sort_order]
    selected_embeds = all_embeds[sort_order]
    return selected_embeds, receiver_idx
