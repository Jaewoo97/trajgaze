"""Qualitative token-selection figures for the specialist KD students (docs/kd_handoff_v2.md).

Three rows over one frame strip, all at the SAME 10% visual-token budget (7% content ∪ 3%
complement) on the same frozen Qwen2.5-VL-7B backbone, each row loading its own LoRA adapter:

  1. M1 teacher            complement chosen by gaze/hand through the frozen TAS encoder
  2. KD student, overlay   complement chosen by TrajSaliencePredictor (RGB only)
  3. KD student, no-overlay same predictor, own weights, preprocessed from MARKER-FREE frames

Row 3 reads `original` (SG) / `no_gaze` (EG) instead of `viz` / `gaze`, so its ViT attention —
and therefore its content 7% as well as its complement — legitimately differs from rows 1-2.
That is the row's point: §7.7 measures what removing the pixel marker costs, and this draws it.

The selection code is imported from the trainers, not reimplemented: `content_and_avail`,
`topk_in_avail` and `union_tokens` are literally `train_visionzip_kd_lora.evaluate()`'s body
(train_visionzip_kd_lora.py:268-276), and the teacher's complement is
`train_visionzip_complement_lora._traj_scores` under `--complement-mode topk`. So the boxes are
the tokens that produced the numbers in the handoff, not a re-derivation of them.

Item ranking: how well BOTH students track the teacher's 3%, measured as cell-space IoU (the
sets of (t,row,col) the reader actually sees). Cell space rather than raw token indices because
row 3's `avail` set is computed on different pixels, so index intersection would not be
comparable between the two students.

Usage (from $REPO, after `source env.sh`):

  python -m TrajGazeMerge.viz.qual.qual_kd_render --source sg --gpu 0 \
      --scan-limit 60 --n-figures 3 --out-root TrajGazeMerge/qual

  # full run
  python -m TrajGazeMerge.viz.qual.qual_kd_render --source sg --gpu 0 --n-figures 12
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_REPO, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset          # noqa: E402
from TrajGazeMerge.models.model import (get_option_ids, build_merged_inputs,  # noqa: E402
                                        forward_logits)
from TrajGazeMerge.models.traj_salience_predictor import TrajSaliencePredictor  # noqa: E402
from TrajGazeMerge.models.traj_weights import _solve_spatial_dims            # noqa: E402
from TrajGazeMerge.training.train_visionzip_lora import (                    # noqa: E402
    load_visionzip_lora, preprocess_visionzip_item)
from TrajGazeMerge.training.train_visionzip_kd_lora import (                 # noqa: E402
    content_and_avail, topk_in_avail, union_tokens)
from TrajGazeMerge.training.train_visionzip_complement_lora import _traj_scores  # noqa: E402
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder  # noqa: E402

from pick_fixation_frames import load_episodes, sharpness                    # noqa: E402

CONTENT_RATIO = 0.07
TRAJ_RATIO = 0.03
# `learned` mode ignores every anticipatory field, but _traj_scores reads them unconditionally
# in the other branch, so the full dict is passed exactly as the M1 trainer builds it.
HP = dict(horizon=2.0, sigma_g=0.08, sigma_h=0.08, alpha_hand=0.5,
          sigma_v=0.1, sigma_gh=0.1, mask_modality="none")

# frame-tree variant per source: (teacher/overlay stream, marker-free stream)
FRAME_VARIANTS = {"sg": ("viz", "original"), "eg": ("gaze", "no_gaze")}
BENCH_NAME = {"sg": "StreamGaze", "eg": "EgoGazeVQA"}

# ── design tokens (bundle docs/design.md) ────────────────────────────────────
INK, INK_80, INK_48 = (29, 29, 31), (51, 51, 51), (122, 122, 122)
HAIRLINE, WHITE = (224, 224, 224), (255, 255, 255)
PAGE_BG = WHITE
BLUE = (0, 102, 204)
OK_FG, BAD_FG = (26, 127, 55), (193, 58, 52)
C_CONTENT = (52, 199, 89)       # green   = content-based 7%
C_COMP = (255, 45, 190)         # magenta = 3% complement (whoever chose it)
C_GAZE = (255, 209, 26)         # gaze ring

SS = 2
OUT_SCALE = 1.0
FONT_PATH = None
_fc: dict = {}


def font(size, weight="Regular"):
    key = (size, weight)
    if key not in _fc:
        f = ImageFont.truetype(FONT_PATH, int(size * SS))
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _fc[key] = f
    return _fc[key]


def u(v):
    return int(round(v * SS))


def _wrap(d, text, f, maxw):
    out = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            while d.textlength(w, font=f) > maxw and len(w) > 1:
                lo, hi = 1, len(w)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if d.textlength(w[:mid], font=f) <= maxw:
                        lo = mid
                    else:
                        hi = mid - 1
                if cur:
                    out.append(cur); cur = ""
                out.append(w[:lo]); w = w[lo:]
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) <= maxw or not cur:
                cur = t
            else:
                out.append(cur); cur = w
        out.append(cur)
    return out or [""]


def paste_shadow(canvas, w, h, x, y, r=0, alpha=52):
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(
        [x + u(2), y + u(4), x + u(2) + w, y + u(4) + h], radius=r, fill=(0, 0, 0, alpha))
    canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(u(15))))


def solid_card(canvas, x0, y0, x1, y1, rgb, radius=None):
    fill = tuple(int(c * 0.16 + 255 * 0.84) for c in rgb)
    edge = tuple(int(c * 0.55 + 255 * 0.45) for c in rgb)
    ImageDraw.Draw(canvas).rounded_rectangle(
        [x0, y0, x1, y1], radius=(u(12) if radius is None else radius),
        fill=fill + (255,), outline=edge + (255,), width=u(1))


# ── what the question is anchored to (bundle pitfall 3) ──────────────────────

QUERY_MOMENT_TASKS = {
    "present_object_identification_easy", "present_object_identification_hard",
    "present_object_attribute_recognition", "present_future_action_prediction",
    "proactive_gaze_triggered_alert", "proactive_object_appearance_alert",
}
QUERY_MOMENT_CUES = ("currently", "current fixation", "right now", "at this moment",
                     "at the moment", "do next", "will the user do", "recent fixation",
                     "is the user looking")


def asks_about_query_moment(item, source="sg"):
    """True when the answer hinges on the end of the window rather than its history.

    Only StreamGaze can qualify: it cuts the frame list at the question timestamp, so its last
    frame IS the query moment. EgoGazeVQA has no cutoff — its last frame is just where the
    subclip ends — so the blue chip there would point the reader at an unrelated frame.
    """
    if source != "sg":
        return False
    task = str(item.get("task", ""))
    if task in QUERY_MOMENT_TASKS or task.startswith(("present_", "proactive_")):
        return True
    if task.startswith("past_"):
        return False
    return any(c in item.get("question", "").lower() for c in QUERY_MOMENT_CUES)


# ── drawing ──────────────────────────────────────────────────────────────────

def render_frame(img_path, t, groups, s_h, s_w, gaze_pt, disp_w, query_moment=False,
                 label=None):
    """`label` overrides the corner chip text — the contact sheet needs `t8b` for the second
    frame of group 8, which the figure path never shows."""
    W = u(disp_w)
    base = Image.open(img_path).convert("RGB")
    W0, H0 = base.size
    H = int(W * H0 / W0)
    base = base.resize((W, H), Image.LANCZOS)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    pw, ph = W / s_w, H / s_h
    cr = max(2, int(pw * 0.18))
    for cells, fill_a, rgb, width in groups:
        for (r, c) in cells:
            box = [c * pw + u(1), r * ph + u(1), (c + 1) * pw - u(1), (r + 1) * ph - u(1)]
            d.rounded_rectangle(box, radius=cr, fill=(rgb + (fill_a,) if fill_a > 0 else None),
                                outline=rgb + (255,), width=max(u(1), width))
    out = Image.alpha_composite(base.convert("RGBA"), ov)
    dd = ImageDraw.Draw(out)
    if gaze_pt is not None:
        x, y = gaze_pt[0] * W, gaze_pt[1] * H
        rr = u(9)
        dd.ellipse([x - rr - u(2), y - rr - u(2), x + rr + u(2), y + rr + u(2)],
                   outline=(255, 255, 255, 210), width=u(3))
        dd.ellipse([x - rr, y - rr, x + rr, y + rr], outline=C_GAZE + (255,), width=u(3))
    fts = font(11, "SemiBold")
    base_lbl = label if label is not None else f"t{t}"
    lbl = f"{base_lbl} · final fixation" if query_moment else base_lbl
    tw = dd.textlength(lbl, font=fts)
    dd.rounded_rectangle([u(7), H - u(7) - u(22), u(7) + tw + u(14), H - u(7)],
                         radius=u(9), fill=(BLUE + (235,) if query_moment else (0, 0, 0, 140)))
    dd.text((u(7) + u(7), H - u(7) - u(20)), lbl, font=fts, fill=(255, 255, 255, 245))
    return out


def compose(item, rows, out_path, source, legend, footer_txt, disp_w=210):
    """rows: [dict(name, sub, frames, pred_letter, correct, is_ours)] — one band each."""
    q = item["question"].replace("\n", " ").strip()
    opts, gt = item["options"], item["answer"]
    letters = [chr(65 + i) for i in range(len(opts))]
    n = len(rows[0]["frames"])
    fw, fh = rows[0]["frames"][0].size

    M = u(46); Lw = u(252); Rw = u(340); fgap = u(10); colgap = u(26)
    strip_w = n * fw + (n - 1) * fgap
    Wc = M + Lw + strip_w + colgap + Rw + M
    inner_w = Wc - 2 * M

    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    fe = font(12, "SemiBold"); fq = font(23, "SemiBold")
    fol = font(16, "Regular"); folb = font(16, "SemiBold"); fl = font(17, "Medium")
    fm = font(20, "SemiBold"); fsub = font(12, "Regular"); fvd = font(12, "SemiBold")
    fplab = font(12, "SemiBold"); fpred = font(15, "SemiBold")
    q_lh = int(fq.getmetrics()[0] * 1.34)
    opt_lh = int(fol.getmetrics()[0] * 1.42)
    sub_lh = int(fsub.getmetrics()[0] * 1.32)
    pred_lh = int(fpred.getmetrics()[0] * 1.42)

    qlines = _wrap(d0, q, fq, inner_w)
    opt_indent = u(16)
    opt_blocks = []
    for i, o in enumerate(opts):
        is_gt = letters[i] == gt
        f_ = folb if is_gt else fol
        opt_blocks.append((_wrap(d0, o.strip(), f_, inner_w - opt_indent), is_gt, f_))
    opt_gap = u(9)
    opts_h = sum(len(l) * opt_lh for l, _, _ in opt_blocks) + (len(opt_blocks) - 1) * opt_gap
    y_q = M
    y_opts = y_q + len(qlines) * q_lh + u(16)
    y_legend = y_opts + opts_h + u(18)
    head = y_legend + u(22) + u(20)

    sub_w = Lw - u(48)
    sub_lines_by_row = [_wrap(d0, r["sub"], fsub, sub_w) for r in rows]
    label_h = u(28) + u(6) + max(len(s) for s in sub_lines_by_row) * sub_lh + u(10) + u(18)
    pred_texts = [(opts[letters.index(r["pred_letter"])].strip()
                   if r["pred_letter"] in letters else r["pred_letter"]) for r in rows]
    max_pl = max(len(_wrap(d0, pt, fpred, Rw - u(8))) for pt in pred_texts)
    pred_block_h = u(20) + max_pl * pred_lh
    content_h = max(fh, pred_block_h, label_h + u(20))
    band_h = content_h + 2 * u(16)
    band_gap = u(12)
    bands_end = head + len(rows) * band_h + (len(rows) - 1) * band_gap
    ffoot = font(12, "Regular")
    foot_lh = int(ffoot.getmetrics()[0] * 1.36)
    foot_lines = _wrap(d0, footer_txt, ffoot, inner_w)   # three rows made it too long for one
    Hc = bands_end + u(14) + len(foot_lines) * foot_lh + u(10) + M

    S = Image.new("RGBA", (Wc, Hc), PAGE_BG + (255,))
    d = ImageDraw.Draw(S)
    d.text((M, M - u(26)), f"QUALITATIVE  ·  {BENCH_NAME[source]}", font=fe, fill=BLUE)

    y = y_q
    for ln in qlines:
        d.text((M, y), ln, font=fq, fill=INK); y += q_lh
    y = y_opts
    for lines, is_gt, f_ in opt_blocks:
        blk_h = len(lines) * opt_lh
        if is_gt:
            d.rounded_rectangle([M, y + u(2), M + u(4), y + blk_h - u(4)], radius=u(2), fill=BLUE)
        col = BLUE if is_gt else INK_80
        yy = y
        for ln in lines:
            d.text((M + opt_indent, yy), ln, font=f_, fill=col); yy += opt_lh
        y += blk_h + opt_gap

    lx = M
    cy_leg = y_legend + u(10.5)
    for rgb, lab, is_ring in legend:
        if is_ring:
            cx, rr = lx + u(7.5), u(7.5)
            d.ellipse([cx - rr, cy_leg - rr, cx + rr, cy_leg + rr],
                      outline=tuple(rgb) + (255,), width=u(2.5))
        else:
            d.rounded_rectangle([lx, y_legend + u(3), lx + u(15), y_legend + u(18)],
                                radius=u(4), fill=tuple(rgb) + (255,))
        d.text((lx + u(22), cy_leg), lab, font=fl, fill=INK, anchor="lm")
        lx += u(22) + d.textlength(lab, font=fl) + u(26)

    by = head
    for ri, row in enumerate(rows):
        d.rounded_rectangle([M - u(10), by, Wc - M + u(10), by + band_h], radius=u(16),
                            fill=WHITE + (255,), outline=HAIRLINE + (255,), width=u(1))
        vc = OK_FG if row["correct"] else BAD_FG
        cx0, cy0, cx1, cy1 = M, by + u(12), M + Lw - u(18), by + band_h - u(12)
        solid_card(S, cx0, cy0, cx1, cy1, vc)
        sub_lines = sub_lines_by_row[ri]
        blk_h = u(28) + u(6) + len(sub_lines) * sub_lh + u(10) + u(18)
        ty = cy0 + ((cy1 - cy0) - blk_h) // 2
        # the three row names are longer than the bundle's ("Ours", "VisionZip"), so step the
        # name down a size until it clears the card rather than letting it run under the edge
        fname_ = fm
        for sz in (20, 18, 16, 14):
            fname_ = font(sz, "SemiBold")
            if d0.textlength(row["name"], font=fname_) <= (cx1 - cx0) - u(30):
                break
        d.text((cx0 + u(15), ty), row["name"], font=fname_,
               fill=(BLUE if row["is_ours"] else INK))
        yy = ty + u(28) + u(6)
        for ln in sub_lines:
            d.text((cx0 + u(15), yy), ln, font=fsub, fill=INK_80); yy += sub_lh
        d.text((cx0 + u(15), yy + u(6)), "Correct" if row["correct"] else "Wrong",
               font=fvd, fill=vc)

        fy = by + (band_h - fh) // 2
        fx = M + Lw
        for fr in row["frames"]:
            paste_shadow(S, fw, fh, fx, fy)
            S.alpha_composite(fr, (fx, fy))
            fx += fw + fgap

        px = fx + colgap
        plines = _wrap(d0, pred_texts[ri], fpred, Rw - u(8))
        pblk = u(20) + len(plines) * pred_lh
        py = by + (band_h - pblk) // 2
        d.text((px, py), "PREDICTION", font=fplab, fill=INK_48)
        yy = py + u(20)
        for ln in plines:
            d.text((px, yy), ln, font=fpred, fill=(OK_FG if row["correct"] else BAD_FG))
            yy += pred_lh
        by += band_h + band_gap

    fyy = bands_end + u(14)
    for ln in foot_lines:
        d.text((M, fyy), ln, font=ffoot, fill=INK_48)
        fyy += foot_lh

    out = S.convert("RGB")
    target = (int(round(Wc / SS * OUT_SCALE)), int(round(Hc / SS * OUT_SCALE)))
    if target != out.size:
        out = out.resize(target, Image.LANCZOS)
    out.save(out_path, quality=96, subsampling=0)


# ── geometry / selection ─────────────────────────────────────────────────────

def geometry(cached):
    N = cached["video_embeds"].shape[0]
    grid = cached["grid_thw"]
    T = int(grid[0, 0].item())
    S = N // max(1, T)
    s_h, s_w = _solve_spatial_dims(S, int(grid[0, 1].item()), int(grid[0, 2].item()))
    return T, S, s_h, s_w


def idx_to_rc(idx, S, s_w):
    return idx // S, (idx % S) // s_w, (idx % S) % s_w


def _as_list(idx):
    """Indices may arrive as a tensor (fresh scan) or a list (reloaded from the scan file)."""
    return idx.tolist() if hasattr(idx, "tolist") else list(idx)


def cellmap(idx, T, S, s_w):
    """Global token indices → per-temporal-group sets of (row, col)."""
    m = [set() for _ in range(T)]
    for i in _as_list(idx):
        t, r, c = idx_to_rc(i, S, s_w)
        if t < T:
            m[t].add((r, c))
    return m


def cellset(idx, T, S, s_w):
    """Global token indices → a flat set of (t, row, col), for cross-row comparison."""
    out = set()
    for i in _as_list(idx):
        t, r, c = idx_to_rc(i, S, s_w)
        if t < T:
            out.add((t, r, c))
    return out


def cache_entry(rec):
    """Scan record → the shape the render loop consumes. Identical whether the record was
    just produced or read back from the scan file, so --from-scan and a live scan render the
    same figure."""
    cl = rec["cells"]
    return dict(rec=rec, T=rec["T"], S=rec["S"], s_h=rec["grid"][0], s_w=rec["grid"][1],
                rows=[(k, cl[k]["content"], cl[k]["comp"], rec[j]["pred"], rec[j]["ok"])
                      for k, j in (("teacher", "teacher"),
                                   ("stu_ov", "student_overlay"),
                                   ("stu_nov", "student_nooverlay"))])


def iou(a, b):
    return len(a & b) / max(1, len(a | b))


def load_lora_only(path):
    """Just the LoRA tensors out of a 16.6 GB `lora_state` (which holds the whole 7B model).

    mmap=True keeps the frozen backbone off the heap — only the ~10 M adapter params are
    materialised, and they are cloned so the mapping can be dropped straight away.
    """
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("lora_state", ck) if isinstance(ck, dict) else ck
    out = {k: v.clone() for k, v in sd.items() if "lora" in k.lower()}
    del ck, sd
    gc.collect()
    return out


def load_pred_state(path, in_dim, hidden, device):
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    st = {k: v.clone() for k, v in ck["pred_state"].items()}
    del ck
    gc.collect()
    p = TrajSaliencePredictor(in_dim, hidden=hidden).to(device)
    p.load_state_dict(st)
    p.eval()
    for q in p.parameters():
        q.requires_grad_(False)
    return p


def student_select(cached, predictor):
    """Exactly train_visionzip_kd_lora.evaluate():268-276, with the eval loop stripped out."""
    content_embeds, content_idx, avail_idx = content_and_avail(cached, CONTENT_RATIO)
    N = cached["video_embeds"].shape[0]
    k = min(max(1, int(TRAJ_RATIO * N)), avail_idx.numel())
    s = predictor(cached["video_embeds"], cached["attn_scores"], cached["grid_thw"])
    traj_idx, _ = topk_in_avail(s, avail_idx, k)
    sel, recv = union_tokens(cached, content_embeds, content_idx, traj_idx)
    return sel, recv, content_idx, traj_idx


def teacher_select(cached, item, device, encoder):
    """M1 with --complement-mode topk: same content pool, complement from gaze/hand + TAS."""
    content_embeds, content_idx, avail_idx = content_and_avail(cached, CONTENT_RATIO)
    N = cached["video_embeds"].shape[0]
    k = min(max(1, int(TRAJ_RATIO * N)), avail_idx.numel())
    s = _traj_scores(cached, item, device, "learned", encoder, HP).to(cached["video_embeds"].device)
    traj_idx, _ = topk_in_avail(s, avail_idx, k)
    sel, recv = union_tokens(cached, content_embeds, content_idx, traj_idx)
    return sel, recv, content_idx, traj_idx


@torch.no_grad()
def predict(model, base_qwen, cached, sel_embeds, recv_idx, option_ids, n_opt):
    """(letter_index, margin) — margin = p(top1) - p(top2) over the option letters."""
    inputs = build_merged_inputs(base_qwen, cached, sel_embeds, recv_idx)
    logits = forward_logits(model, inputs)
    opt = logits[option_ids[:n_opt]].float()
    p = torch.softmax(opt, dim=0)
    top = torch.topk(p, min(2, n_opt))
    margin = float(top.values[0] - top.values[1]) if n_opt > 1 else 1.0
    return int(opt.argmax().item()), margin


# ── frame variant swap ───────────────────────────────────────────────────────

def swap_variant(paths, want):
    """.../<dataset>/<variant>/<stem>/<file> → same with <variant> replaced.

    Component surgery rather than str.replace so a stem or dataset directory that happens to
    contain 'gaze' cannot be rewritten by accident.
    """
    out = []
    for p in paths:
        head, fname = os.path.split(p)
        head, stem = os.path.split(head)
        head, _variant = os.path.split(head)
        out.append(os.path.join(head, want, stem, fname))
    return out


# ── frame strips ─────────────────────────────────────────────────────────────

def sg_strip(item, eps, nf, sharp_window=2):
    """Anchor the strip on the clip's annotated fixation episodes; never reads the answer.

    Ported from the bundle's pick_fixation_frames.pick(), but driven off the item the dataset
    actually returned instead of a re-derived item list — the bundle rebuilds the test split
    itself and would drift from `CombinedMergeDataset` the moment either side skips an item.
    Frame times come straight from the sampled filenames (10 fps, `frame_%06d.jpg`).
    """
    paths = item["vlm_frame_paths"]
    L = len(paths)
    if L == 0:
        return []
    stem = os.path.basename(os.path.dirname(paths[0]))
    nums = [int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)) for p in paths]
    T = (L + 1) // 2
    tsec = [nums[min(2 * t, L - 1)] / 10.0 for t in range(T)]
    last_in = nums[L - 1] / 10.0
    if T <= nf:
        return [(t, 0) for t in range(T)]

    anchors = []
    for e in eps.get(stem, []):
        if e["t1"] < tsec[0] or e["t0"] > last_in:
            continue
        mid = (e["t0"] + e["t1"]) / 2
        t = min(range(T), key=lambda t: abs(tsec[t] - mid))
        if t not in anchors:
            anchors.append(t)
    anchors = anchors[-(nf - 1):]                      # the most recent episodes win the space
    keep = sorted(set(anchors + [T - 1]))
    min_gap = max(2, T // 24)
    for f in [int(round(i * (T - 1) / (nf - 1))) for i in range(nf)]:
        if len(keep) >= nf:
            break
        if all(abs(f - k) >= min_gap for k in keep):
            keep = sorted(keep + [f])
    while len(keep) > nf:                              # trim the tightest filler, never an anchor
        cand = [(min(keep[i] - keep[i - 1], keep[i + 1] - keep[i]), i)
                for i in range(1, len(keep) - 1) if keep[i] not in anchors]
        if not cand:
            break
        keep.pop(min(cand)[1])

    if sharp_window:
        # a filler landing mid-saccade shows a smear; slide it to the sharpest frame within
        # +-sharp_window groups, keeping order and the min gap. Anchors never move.
        for i, t in enumerate(keep):
            if t in anchors or t in (0, T - 1):
                continue
            lo = max(keep[i - 1] + min_gap, t - sharp_window) if i else t - sharp_window
            hi = (min(keep[i + 1] - min_gap, t + sharp_window) if i + 1 < len(keep)
                  else t + sharp_window)
            cands = [c for c in range(max(0, lo), min(T - 1, hi) + 1) if c not in keep or c == t]
            if len(cands) > 1:
                keep[i] = max(cands, key=lambda c: sharpness(paths[min(2 * c, L - 1)]))
        keep = sorted(set(keep))
    return [(t, 0) for t in keep]


def eg_strip(T, nf):
    """Uniform over the item's own T. EG has no timestamp cutoff and no episode table, so
    there is nothing to anchor to; T also varies 37-64 across clips, hence per-item spacing."""
    nf = min(nf, T)
    return [(int(round(i * (T - 1) / max(1, nf - 1))), 0) for i in range(nf)]


# ── main ─────────────────────────────────────────────────────────────────────

def parse_idxs(spec, n):
    if spec is None or spec.strip() in ("all", "*"):
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-")
            out += list(range(int(a), min(int(b), n - 1) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


@torch.no_grad()
def main():
    global FONT_PATH, SS, OUT_SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sg", "eg"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(_REPO, "TrajGazeMerge", "qual"))
    ap.add_argument("--font", default=os.path.join(_HERE, "Inter.ttf"))
    ap.add_argument("--idxs", default="all", help="items to scan: 'all' | '0-59' | '3,7'")
    ap.add_argument("--scan-limit", type=int, default=0,
                    help="stop the scan after this many successfully scored items (0 = no cap)")
    ap.add_argument("--n-figures", type=int, default=12)
    ap.add_argument("--per-task-cap", type=int, default=3,
                    help="at most this many figures from any one task / qa_type")
    ap.add_argument("--min-margin", type=float, default=0.05,
                    help="skip items where any row's top-2 option gap is below this. Option "
                         "logits quantise to 1/8 in bf16, so exact ties happen; a tie "
                         "reproduces (argmax breaks it the same way) but the row is indifferent "
                         "between two answers, and drawing it as Correct is luck. 0 disables.")
    ap.add_argument("--rank-mode", choices=["correct-first", "follow"], default="correct-first",
                    help="correct-first: most rows correct, then best teacher-tracking. "
                         "follow: purest teacher-tracking, verdicts only break ties.")
    ap.add_argument("--render-idxs", default=None,
                    help="skip ranking and render exactly these item indices")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing scan file, skipping items already in it")
    ap.add_argument("--from-scan", action="store_true",
                    help="render straight from the scan file: no model, no GPU, seconds")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--disp-w", type=int, default=210)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--out-scale", type=float, default=1.0)
    ap.add_argument("--n-vis-keyframes", type=int, default=16)
    ap.add_argument("--pred-hidden", type=int, default=512)
    ap.add_argument("--stage1-ckpt", default=os.environ.get("STAGE1_CKPT"))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student-overlay", default=None)
    ap.add_argument("--student-nooverlay", default=None)
    ap.add_argument("--tag", default="", help="suffix on the scan file, for repeat runs")
    args = ap.parse_args()

    FONT_PATH = args.font
    SS, OUT_SCALE = args.supersample, args.out_scale
    src = args.source
    ck = os.path.join(_REPO, "TrajGazeMerge", "checkpoints")
    up = src.upper()                       # "SG" / "EG", as the checkpoint names spell it
    teacher_ck = args.teacher or os.environ.get(f"M1_{up}ONLY") or os.path.join(
        ck, f"visionzip_complement_learned_{up}only_overlay", "best.pth")
    stu_ov_ck = args.student_overlay or os.path.join(
        ck, f"visionzip_kd_selection_{up}only_overlay", "best.pth")
    stu_nov_ck = args.student_nooverlay or os.path.join(
        ck, f"visionzip_kd_selection_{up}only_nooverlay", "best.pth")
    for p in (teacher_ck, stu_ov_ck, stu_nov_ck, args.stage1_ckpt):
        if not p or not os.path.exists(p):
            raise SystemExit(f"[qual] missing checkpoint: {p}")

    var_ov, var_nov = FRAME_VARIANTS[src]
    out_dir = os.path.join(args.out_root, src)
    fig_dir, lay_dir = os.path.join(out_dir, "figures"), os.path.join(out_dir, "layout")
    scan_dir = os.path.join(args.out_root, "scan")
    for p in (fig_dir, lay_dir, scan_dir):
        os.makedirs(p, exist_ok=True)

    print(f"[qual] source={src} gpu={args.gpu}\n"
          f"       teacher   {teacher_ck}\n"
          f"       stu(ov)   {stu_ov_ck}\n"
          f"       stu(nov)  {stu_nov_ck}", flush=True)

    device = None
    if not args.from_scan:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        processor, model = load_visionzip_lora(device)
        base_qwen = model.get_base_model()
        option_ids = get_option_ids(processor, 5)
        model.eval()

        in_dim = base_qwen.get_input_embeddings().weight.shape[1]
        lora = {}
        for nm, path in (("teacher", teacher_ck), ("stu_ov", stu_ov_ck), ("stu_nov", stu_nov_ck)):
            lora[nm] = load_lora_only(path)
            print(f"[qual] {nm}: {len(lora[nm])} lora tensors", flush=True)
        pred_ov = load_pred_state(stu_ov_ck, in_dim, args.pred_hidden, device)
        pred_nov = load_pred_state(stu_nov_ck, in_dim, args.pred_hidden, device)

        encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    ds.items = [it for it in ds.items if it[0] == src]
    eps = load_episodes() if src == "sg" else {}

    want = (parse_idxs(args.render_idxs, len(ds)) if args.render_idxs
            else parse_idxs(args.idxs, len(ds)))

    scan_path = os.path.join(scan_dir, f"{src}{('_' + args.tag) if args.tag else ''}.jsonl")
    cache: dict[int, dict] = {}
    # A kill mid-scan must not cost the items already done — this node is shared and killed
    # both full runs at 00:03 when another job landed (§13.4 records three such losses).
    if (args.resume or args.from_scan) and os.path.exists(scan_path):
        for line in open(scan_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "cells" in r:
                cache[r["idx"]] = cache_entry(r)
        print(f"[qual] reloaded {len(cache)} scored items from {scan_path}", flush=True)
        want = [i for i in want if i not in cache]

    n_scored = len(cache)
    n_new = 0                      # this process's own work, for the timing figures
    if args.from_scan:
        want = []
        scan_f = open(os.devnull, "w")
    else:
        scan_f = open(scan_path, "a" if args.resume else "w")
    print(f"[qual] {len(ds)} {src} test items; scanning {len(want)}", flush=True)
    stream_checked = False
    t_start = time.time()

    for idx in want:
        if args.scan_limit and n_scored >= args.scan_limit:
            break
        item = ds[idx]
        if item is None:
            continue
        n_opt = len(item["options"])
        letters = [chr(65 + i) for i in range(n_opt)]
        if item["answer"] not in letters:
            continue
        try:
            paths_ov = item["vlm_frame_paths"]
            paths_nov = swap_variant(paths_ov, var_nov)
            missing = [p for p in paths_nov if not os.path.exists(p)]
            if missing:
                print(f"  idx={idx} skip: {len(missing)} marker-free frames missing", flush=True)
                continue
            if not stream_checked:
                # §7.3's assertion: the two streams must resolve to different directories, or
                # the overlay-free row is silently the overlay one and nothing would show it
                t_sub = os.path.basename(os.path.dirname(os.path.dirname(
                    item["traj_frame_paths"][0])))
                n_sub = os.path.basename(os.path.dirname(os.path.dirname(paths_nov[0])))
                o_sub = os.path.basename(os.path.dirname(os.path.dirname(paths_ov[0])))
                print(f"[qual] frame streams: rows1-2 VLM='{o_sub}'  row3 VLM='{n_sub}'  "
                      f"teacher TAS='{t_sub}'", flush=True)
                if n_sub == o_sub or t_sub != var_ov:
                    raise SystemExit(
                        f"[qual] frame streams collapsed (overlay='{o_sub}' nooverlay='{n_sub}' "
                        f"tas='{t_sub}'); expected '{var_ov}' / '{var_nov}' / '{var_ov}'. "
                        f"Is GAZE_OVERLAY=1 and VLM_GAZE_OVERLAY unset?")
                stream_checked = True

            c_ov = preprocess_visionzip_item(processor, base_qwen, paths_ov,
                                             item["question"], item["options"], device)
            c_nov = preprocess_visionzip_item(processor, base_qwen, paths_nov,
                                              item["question"], item["options"], device)
            if c_ov is None or c_nov is None:
                continue
            T, S, s_h, s_w = geometry(c_ov)
            T2, S2, _, _ = geometry(c_nov)
            if (T, S) != (T2, S2):
                print(f"  idx={idx} skip: grid mismatch {T}x{S} vs {T2}x{S2}", flush=True)
                continue

            model.load_state_dict(lora["teacher"], strict=False)
            t_sel, t_recv, t_content, t_comp = teacher_select(c_ov, item, device, encoder)
            t_pi, t_mg = predict(model, base_qwen, c_ov, t_sel, t_recv, option_ids, n_opt)

            model.load_state_dict(lora["stu_ov"], strict=False)
            o_sel, o_recv, o_content, o_comp = student_select(c_ov, pred_ov)
            o_pi, o_mg = predict(model, base_qwen, c_ov, o_sel, o_recv, option_ids, n_opt)

            model.load_state_dict(lora["stu_nov"], strict=False)
            v_sel, v_recv, v_content, v_comp = student_select(c_nov, pred_nov)
            v_pi, v_mg = predict(model, base_qwen, c_nov, v_sel, v_recv, option_ids, n_opt)

            cs_t = cellset(t_comp.cpu(), T, S, s_w)
            cs_o = cellset(o_comp.cpu(), T, S, s_w)
            cs_v = cellset(v_comp.cpu(), T, S, s_w)
            iou_ov, iou_nov = iou(cs_o, cs_t), iou(cs_v, cs_t)
            follow = min(iou_ov, iou_nov)
            # the content pools too: row 3 selects content from different pixels, so part of
            # any complement divergence is inherited from a different `avail` set rather than
            # from the predictor disagreeing. Without this the two cannot be told apart.
            ct_t = cellset(t_content.cpu(), T, S, s_w)
            iou_c_ov = iou(cellset(o_content.cpu(), T, S, s_w), ct_t)
            iou_c_nov = iou(cellset(v_content.cpu(), T, S, s_w), ct_t)
            # the training-time metric, exact token indices on the same cached item
            agree_ov = len(set(o_comp.tolist()) & set(t_comp.tolist())) / max(1, t_comp.numel())

            gt = letters.index(item["answer"])
            rec = dict(idx=idx, task=item["task"], dataset=item["dataset"], n_opt=n_opt,
                       answer=item["answer"], T=T, S=S, grid=[s_h, s_w],
                       n_kept=int(t_recv.numel()), pct_kept=100.0 * t_recv.numel() / (T * S),
                       teacher=dict(pred=letters[t_pi], ok=t_pi == gt, margin=round(t_mg, 4)),
                       student_overlay=dict(pred=letters[o_pi], ok=o_pi == gt,
                                            margin=round(o_mg, 4)),
                       student_nooverlay=dict(pred=letters[v_pi], ok=v_pi == gt,
                                              margin=round(v_mg, 4)),
                       iou_overlay_vs_teacher=round(iou_ov, 4),
                       iou_nooverlay_vs_teacher=round(iou_nov, 4),
                       iou_content_overlay_vs_teacher=round(iou_c_ov, 4),
                       iou_content_nooverlay_vs_teacher=round(iou_c_nov, 4),
                       follow=round(follow, 4),
                       n_correct=int(t_pi == gt) + int(o_pi == gt) + int(v_pi == gt),
                       agree_overlay_exact=round(agree_ov, 4),
                       # the selections themselves, so a killed run never has to re-scan what
                       # it already did and figures can be redrawn with --from-scan on CPU.
                       # This node is shared and kills jobs (§13.4 lists three such losses).
                       cells=dict(
                           teacher=dict(content=t_content.tolist(), comp=t_comp.tolist()),
                           stu_ov=dict(content=o_content.tolist(), comp=o_comp.tolist()),
                           stu_nov=dict(content=v_content.tolist(), comp=v_comp.tolist())))
            scan_f.write(json.dumps(rec) + "\n"); scan_f.flush()
            n_scored += 1; n_new += 1

            cache[idx] = cache_entry(rec)
            el = time.time() - t_start
            rss = int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30
            print(f"  idx={idx:<4} rss={rss:6.1f}G {item['task'][:38]:<38} "
                  f"T={letters[t_pi]}{'✓' if t_pi == gt else '✗'} "
                  f"O={letters[o_pi]}{'✓' if o_pi == gt else '✗'} "
                  f"N={letters[v_pi]}{'✓' if v_pi == gt else '✗'} GT={item['answer']} "
                  f"IoU ov={iou_ov:.3f} nov={iou_nov:.3f} follow={follow:.3f} "
                  f"[{el / max(1, n_new):.1f}s/item]", flush=True)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  idx={idx} err: {type(e).__name__}: {e}", flush=True)
            continue
    scan_f.close()
    print(f"[qual] {n_new} newly scanned in {time.time() - t_start:.0f}s "
          f"({(time.time() - t_start) / max(1, n_new):.1f}s/item); {n_scored} scored in total "
          f"-> {scan_path}", flush=True)
    if args.scan_only:
        return

    # ── ranking ──
    if args.render_idxs:
        chosen = [i for i in parse_idxs(args.render_idxs, len(ds)) if i in cache]
    else:
        def key(i):
            r = cache[i]["rec"]
            same = (r["teacher"]["pred"] == r["student_overlay"]["pred"] ==
                    r["student_nooverlay"]["pred"])
            mgn = min(r["teacher"]["margin"], r["student_overlay"]["margin"],
                      r["student_nooverlay"]["margin"])
            if args.rank_mode == "follow":
                return (r["follow"], r["n_correct"], same, mgn)
            # correct-first: `follow` alone happily surfaces items where all three rows are
            # wrong — high selection agreement on a question none of them answers is a poor
            # exhibit. Rank on how many rows are right, then on how well they track the
            # teacher within that tier.
            return (r["n_correct"], r["follow"], same, mgn)
        def min_margin(i):
            r = cache[i]["rec"]
            return min(r["teacher"]["margin"], r["student_overlay"]["margin"],
                       r["student_nooverlay"]["margin"])
        eligible = [i for i in cache if min_margin(i) >= args.min_margin]
        if len(eligible) < len(cache):
            print(f"[qual] margin gate (>= {args.min_margin}) dropped "
                  f"{len(cache) - len(eligible)} of {len(cache)} scanned items", flush=True)
        ranked = sorted(eligible, key=key, reverse=True)
        chosen, per_task = [], {}
        for i in ranked:
            t = cache[i]["rec"]["task"]
            if per_task.get(t, 0) >= args.per_task_cap:
                continue
            per_task[t] = per_task.get(t, 0) + 1
            chosen.append(i)
            if len(chosen) >= args.n_figures:
                break
    print(f"[qual] rendering {len(chosen)} figures -> {fig_dir}", flush=True)

    legend = [(list(C_CONTENT), "content-based selection 7%", False),
              (list(C_COMP), "3% complement (each row's own selector)", False),
              (list(C_GAZE), "gaze (annotation, rows 1-2 only)", True)]
    # Inter.ttf has no U+222F glyph, so the set-union sign renders as tofu — spell it out
    footer_txt = ("Same 10% visual-token budget (7% content + 3% complement) · frozen "
                  "Qwen2.5-VL-7B backbone · each row loads its own LoRA adapter · "
                  f"row 3 is preprocessed from marker-free `{var_nov}` frames")
    # names follow §12.5's recommended paper labels — `KD (gaze-overlay)` vs `KD (raw video)` —
    # and are short enough to stay inside the label card at --type-scale 1.6
    ROW_META = {
        "teacher": ("M1 teacher", "3% complement from gaze/hand via the frozen TAS encoder; "
                    "uses gaze at test time", False),
        "stu_ov": ("KD (gaze-overlay)", "3% complement from the RGB predictor (3.95 M); "
                   "marker still in the pixels", True),
        "stu_nov": ("KD (raw video)", "3% complement from the RGB predictor; marker-free input",
                    True),
    }

    manifest_rows = []
    for idx in chosen:
        item = ds[idx]
        if item is None:
            continue
        c = cache[idx]
        T, S, s_h, s_w = c["T"], c["S"], c["s_h"], c["s_w"]
        paths_ov = item["vlm_frame_paths"]
        paths_nov = swap_variant(paths_ov, var_nov)
        L = len(paths_ov)

        tsel = (sg_strip(item, eps, args.max_frames) if src == "sg"
                else eg_strip(T, args.max_frames))
        tsel = [(t, h) for (t, h) in tsel if 0 <= t < T][:args.max_frames]
        graw = item["traj"]["gaze_pos"].float().cpu().numpy()
        mraw = item["traj"]["gaze_mask"].cpu().numpy().astype(bool)
        mark_q = asks_about_query_moment(item, src)

        dump = dict(source=src, idx=idx, task=item["task"], flags="",
                    question=item["question"].strip(), options=item["options"],
                    answer=item["answer"], grid=[s_h, s_w], disp_w=args.disp_w,
                    title=f"QUALITATIVE  ·  {BENCH_NAME[src]}",
                    legend=[dict(rgb=rgb, label=lab, shape=("ring" if ring else "swatch"))
                            for rgb, lab, ring in legend],
                    footer=footer_txt, strip=[], rows=[])
        for t, half in tsel:
            vi = min(2 * t + half, L - 1)
            dump["strip"].append(dict(
                t=t, half=half, path=paths_ov[vi],
                gaze=([float(graw[vi, 0]), float(graw[vi, 1])]
                      if vi < len(mraw) and mraw[vi] else None),
                query_moment=bool(mark_q and t == T - 1)))

        rows = []
        r = c["rec"]
        for key_, content_idx, comp_idx, pred_letter, ok in c["rows"]:
            name, sub, is_ours = ROW_META[key_]
            if key_ == "stu_ov":
                # content pool is identical to the teacher's here (same VisionZip rule, same
                # pixels, adapter-independent), so this IoU is purely predictor-vs-gaze/hand
                sub += f" · complement IoU {r['iou_overlay_vs_teacher']:.2f} vs teacher"
            elif key_ == "stu_nov":
                # NOT purely predictor disagreement: marker-free pixels move VisionZip's own
                # content pick too, so the complement is drawn from a different `avail` set.
                # Quoting one number without the other reads as if the predictor diverged.
                sub += (f" · complement IoU {r['iou_nooverlay_vs_teacher']:.2f}, content pool "
                        f"IoU {r['iou_content_nooverlay_vs_teacher']:.2f} vs teacher")
            cm_content = cellmap(content_idx, T, S, s_w)
            cm_comp = cellmap(comp_idx, T, S, s_w)
            paths = paths_nov if key_ == "stu_nov" else paths_ov

            def groups_for(t):
                return [(cm_content[t], 0, C_CONTENT, u(2)),
                        (cm_comp[t], 120, C_COMP, u(3))]

            frames = []
            row_strip = []
            # the overlay-free row gets NO gaze ring: its pixels have no marker and its model
            # never receives gaze, so a ring there reads as an input the row does not have
            show_gaze = key_ != "stu_nov"
            for t, half in tsel:
                # Qwen2.5-VL merges frames (2t, 2t+1) into group t; either may be shown, but
                # the gaze ring must come from the frame actually drawn, never a pooled track
                vi = min(2 * t + half, L - 1)
                gpt = ((float(graw[vi, 0]), float(graw[vi, 1]))
                       if show_gaze and vi < len(mraw) and mraw[vi] else None)
                qm = bool(mark_q and t == T - 1)
                frames.append(render_frame(paths[vi], t, groups_for(t), s_h, s_w, gpt,
                                           args.disp_w, query_moment=qm))
                row_strip.append(dict(t=t, half=half, path=paths[vi],
                                      gaze=list(gpt) if gpt else None, query_moment=qm))
            rows.append(dict(name=name, sub=sub, frames=frames, pred_letter=pred_letter,
                             correct=bool(ok), is_ours=is_ours))
            dump["rows"].append(dict(
                name=name, sub=sub, pred_letter=pred_letter, correct=bool(ok),
                is_ours=is_ours,
                strip=(row_strip if key_ == "stu_nov" else None),
                groups=[[dict(cells=sorted(map(list, cs)), fill_alpha=fa, rgb=list(rgb),
                              width_u=max(1, w // SS))
                         for cs, fa, rgb, w in groups_for(t)] for t, _ in tsel]))

        flags = ("T" + ("1" if r["teacher"]["ok"] else "0")
                 + "O" + ("1" if r["student_overlay"]["ok"] else "0")
                 + "N" + ("1" if r["student_nooverlay"]["ok"] else "0"))
        dump["flags"] = flags
        note = (f"{len(tsel)} frames anchored on the clip's annotated fixation episodes; "
                "the answer was not used" if src == "sg"
                else f"{len(tsel)} frames sampled uniformly over the clip")
        dump["note"] = note
        stem = f"{src}_{flags}_idx{idx}_{item['task']}".replace("/", "_")
        png = os.path.join(fig_dir, stem + ".png")
        compose(item, rows, png, src, legend, footer_txt + f" · {note}", args.disp_w)
        with open(os.path.join(lay_dir, stem + ".json"), "w") as f:
            json.dump(dump, f)
        print(f"  saved {png}", flush=True)
        manifest_rows.append(dict(
            figure=os.path.relpath(png, args.out_root),
            layout=os.path.relpath(os.path.join(lay_dir, stem + ".json"), args.out_root),
            source=src, idx=idx, task=item["task"], answer=item["answer"], flags=flags,
            follow=r["follow"], iou_overlay=r["iou_overlay_vs_teacher"],
            iou_nooverlay=r["iou_nooverlay_vs_teacher"],
            iou_content_overlay=r["iou_content_overlay_vs_teacher"],
            iou_content_nooverlay=r["iou_content_nooverlay_vs_teacher"],
            n_correct=r["n_correct"],
            agree_overlay_exact=r["agree_overlay_exact"],
            pred_teacher=r["teacher"]["pred"], pred_student_overlay=r["student_overlay"]["pred"],
            pred_student_nooverlay=r["student_nooverlay"]["pred"],
            margin_teacher=r["teacher"]["margin"],
            margin_student_overlay=r["student_overlay"]["margin"],
            margin_student_nooverlay=r["student_nooverlay"]["margin"],
            pct_kept=round(r["pct_kept"], 3), strip=" ".join(str(t) for t, _ in tsel),
            ckpt_teacher=teacher_ck, ckpt_student_overlay=stu_ov_ck,
            ckpt_student_nooverlay=stu_nov_ck))

    man = os.path.join(args.out_root, "manifest.csv")
    old = []
    if os.path.exists(man):
        with open(man) as f:
            old = [r for r in csv.DictReader(f) if r.get("source") != src]
    if manifest_rows:
        cols = list(manifest_rows[0].keys())
        with open(man, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in old + manifest_rows:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"[qual] manifest -> {man}", flush=True)


if __name__ == "__main__":
    main()
