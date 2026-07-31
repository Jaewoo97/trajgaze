"""Contact sheets for the ViT-KD ablation (docs/kd_handoff_v3.md): SG teacher vs ViT-distilled.

One sheet per model per item: every temporal group of the clip, with that model's kept visual
tokens drawn on each frame, labelled with the group index and its wall-clock time. Cells that sit
on an annotated fixation episode get a yellow border, the last frame of the clip a red one. The
header prints the question, the options with the ground truth marked, and each model's verdict —
these are diagnostics for choosing items, not blind strip-selection sheets. `--hide-answer` drops
the ground-truth mark, the verdict and the fixation-episode object names together, for when a sheet
has to be answer-free.

| | M1 teacher sheet | ViT-KD sheet |
|---|---|---|
| frames | `viz` (marker in the pixels) | `original` (raw video) |
| ViT | frozen | rank-8 LoRA on `visual.blocks[31].attn.{qkv,proj}` |
| green outline | content 7% | kept 10%; dominant 6.5% drawn thicker than contextual 3.5% |
| magenta fill | gaze/hand complement 3% (TAS) | the dominant tokens that coincide with the teacher's 3% |
| gaze ring | yes | no (this model has neither the marker nor any gaze input) |

The student's magenta is `recall_traj` drawn in place — v3 §5.2's metric, which went 0.042 → 0.383
under distillation. §5.4's finding is that this recovery bought **zero** GSM items, i.e. the
recovered tokens were not the ones that mattered; these sheets are the way to look at where they
actually landed.

Item ranking: both models correct, then a weighted mix of agreement with the teacher and how
tightly the kept tokens cluster on the annotated gaze point.

Usage (from $REPO, after `source env.sh`):

  python -m TrajGazeMerge.viz.qual.vitkd_contact_sheet --gpu 0 --scan-limit 60 --n-items 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
import re
import sys
import time

import torch
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_REPO, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset            # noqa: E402
from TrajGazeMerge.data.dataset import _parse_ts                                # noqa: E402
from TrajGazeMerge.models.model import (get_option_ids, build_merged_inputs,    # noqa: E402
                                        forward_logits)
from TrajGazeMerge.training.train_visionzip_lora import (                       # noqa: E402
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens)
from TrajGazeMerge.training.train_vit_selection_kd import (                     # noqa: E402
    attach_vit_lora, load_vit_lora_state, vit_lora_disabled,
    teacher_selection, selection_metrics)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder  # noqa: E402

import qual_kd_render as qk                                                     # noqa: E402
from pick_fixation_frames import load_episodes, build_items                     # noqa: E402

# v3 §2.1 target / §2.4 student split. NOT the trainer defaults (0.05/0.05) — the ViT-KD runs
# pass the P split on the command line, so hard-code it here and print it.
CONTENT_RATIO, TRAJ_RATIO = 0.07, 0.03
DOM_P, CTX_P = 0.065, 0.035
HP = dict(mask_modality="none")            # `learned` mode reads nothing else
FPS = 10.0                                 # SG frames are frame_%06d.jpg at 10 fps

RED = (193, 58, 52)
YELLOW = (214, 158, 0)
PARCHMENT = (245, 245, 247)

# For these tasks the annotated object IS the answer, so under --hide-answer the episode label has
# to go too — otherwise the yellow cell's caption reinstates what the flag just removed.
ANSWER_IS_THE_FIXATED_OBJECT = {
    "present_object_identification_easy", "present_object_identification_hard",
    "present_object_attribute_recognition",
}


def frame_seconds(path):
    return int(re.search(r"frame_(\d+)", os.path.basename(path)).group(1)) / FPS


def mmss(sec, precise=False):
    return (f"{int(sec) // 60}:{sec % 60:04.1f}" if precise
            else f"{int(sec) // 60}:{int(sec) % 60:02d}")


# ── metrics ──────────────────────────────────────────────────────────────────

def gaze_concentration(cells_per_t, graw, mraw, T, s_h, s_w, L, radius):
    """Fraction of kept cells landing within `radius` of the gaze point, averaged over frames.

    The proxy for "the key object": EGTEA's gaze is a human annotation of what the person was
    actually looking at, so tokens clustered on it are tokens spent on the thing the question is
    about. Distances are in units of frame width, and a cell counts if its CENTRE is inside the
    disc — cell size is 1/s_w wide, so this is a slightly conservative test.
    """
    tot, n = 0.0, 0
    for t in range(T):
        vi = min(2 * t, L - 1)
        if vi >= len(mraw) or not mraw[vi]:
            continue
        cells = cells_per_t[t]
        if not cells:
            continue
        gx, gy = float(graw[vi, 0]), float(graw[vi, 1])
        near = 0
        for (r, c) in cells:
            cx, cy = (c + 0.5) / s_w, (r + 0.5) / s_h
            if ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 <= radius:
                near += 1
        tot += near / len(cells)
        n += 1
    return tot / max(1, n)


@torch.no_grad()
def predict(model, base_qwen, cached, sel, recv, option_ids, n_opt):
    logits = forward_logits(model, build_merged_inputs(base_qwen, cached, sel, recv))
    opt = logits[option_ids[:n_opt]].float()
    p = torch.softmax(opt, dim=0)
    top = torch.topk(p, min(2, n_opt))
    return int(opt.argmax().item()), (float(top.values[0] - top.values[1]) if n_opt > 1 else 1.0)


# ── sheet ────────────────────────────────────────────────────────────────────

def build_sheet(item, idx, paths, layers_for_t, T, s_h, s_w, graw, mraw, eps, cutoff,
                title, subtitle, legend, out_path, cols=8, cell_w=300, halves="both",
                show_gaze=True, show_answer=True, cell_note=None):
    """`layers_for_t(t)` -> the render_frame group list for that temporal group."""
    u, SS = qk.u, qk.SS
    L = len(paths)
    cells = [(t, h) for t in range(T) for h in (0, 1) if 2 * t + h < L]
    if halves == "first":
        cells = [(t, h) for t, h in cells if h == 0]
    ctime = [frame_seconds(paths[2 * t + h]) for t, h in cells]
    last_in = frame_seconds(paths[L - 1])
    fine = (ctime[-1] - ctime[0]) < 120
    cstep = (ctime[1] - ctime[0]) if len(ctime) > 1 else 0.0

    # annotated fixation episodes -> yellow cells. An episode is attached only to a cell it
    # actually overlaps; one starting after the last input frame is reported as a gap instead of
    # being pinned to the nearest cell, which would label the last group with a fixation that is
    # not in the clip at all.
    hide_names = item["task"] in ANSWER_IS_THE_FIXATED_OBJECT and not show_answer
    stem = os.path.basename(os.path.dirname(paths[0]))
    anchors, anchor_lab, after = set(), {}, []
    for e in eps.get(stem, []):
        if e["t1"] < ctime[0]:
            continue
        if e["t0"] > last_in:
            if e["t0"] - last_in < 60:
                after.append(e)
            continue
        i = min(range(len(cells)), key=lambda i: abs(ctime[i] - (e["t0"] + e["t1"]) / 2))
        if not (e["t0"] - cstep <= ctime[i] <= e["t1"] + cstep):
            continue
        anchors.add(cells[i])
        anchor_lab[cells[i]] = ("fixation" if hide_names else
                                e["obj"] + ("+" + ",".join(e["near"]) if e["near"] else ""))

    im0 = Image.open(paths[0])
    cw = u(cell_w)
    chh = int(cw * im0.height / im0.width)
    cap_h, gap, M = u(22), u(6), u(34)
    rows = (len(cells) + cols - 1) // cols
    Wc = 2 * M + cols * cw + (cols - 1) * gap
    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    fq, fo = qk.font(26, "SemiBold"), qk.font(17, "Regular")
    fc, fl, fn = qk.font(15, "SemiBold"), qk.font(16, "Medium"), qk.font(16, "Medium")

    qlines = qk._wrap(d0, item["question"].replace("\n", " ").strip(), fq, Wc - 2 * M)
    olines = []
    for i, o in enumerate(item["options"]):
        gt = show_answer and chr(65 + i) == item["answer"]
        olines += [(ln, gt) for ln in
                   qk._wrap(d0, o.strip() + ("      <-- ANSWER" if gt else ""), fo, Wc - 2 * M - u(14))]

    notes = [(subtitle, (51, 51, 51))]
    if cutoff:
        notes.append((f"QUERY MOMENT: the clip is cut at {mmss(cutoff, fine)}, and the last of the "
                      f"{L} input frames is {mmss(last_in, fine)}. Every one of them is a cell below.",
                      RED))
    if after:
        task = item["task"]
        why = ("so the moment this question asks about is NOT in the clip."
               if task.startswith(("present_", "proactive_")) else
               "and that is where the third group of every option's sequence sits, so no frame "
               "can show it." if task == "past_gaze_sequence_matching" else
               "so it is not in the clip.")
        notes.append((f"The next annotated fixation starts at {mmss(after[0]['t0'], fine)}, "
                      f"{after[0]['t0'] - last_in:.0f} s after the last input frame, " + why, RED))
    if hide_names:
        notes.append(("Fixation episodes are marked without their object names here, because for "
                      "this task the annotated object is the answer.", qk.INK_48))
    note_lines = [(ln, col) for txt, col in notes for ln in qk._wrap(d0, txt, fn, Wc - 2 * M)]

    head = (M + len(qlines) * u(34) + u(10) + len(olines) * u(25)
            + u(10) + len(note_lines) * u(23) + u(34))
    Hc = head + rows * (chh + cap_h + gap) + M

    S = Image.new("RGBA", (Wc, Hc), PARCHMENT + (255,))
    d = ImageDraw.Draw(S)
    d.text((M, M - u(24)), title, font=qk.font(13, "SemiBold"), fill=qk.BLUE)
    y = M + u(6)
    for ln in qlines:
        d.text((M, y), ln, font=fq, fill=qk.INK); y += u(34)
    y += u(10)
    for ln, gt in olines:
        d.text((M + u(14), y), ln, font=(qk.font(17, "SemiBold") if gt else fo),
               fill=(qk.OK_FG if gt else (51, 51, 51)))
        y += u(25)
    y += u(10)
    for ln, col in note_lines:
        d.text((M, y), ln, font=fn, fill=col); y += u(23)
    lx = M
    for rgb, lab in legend:
        d.rounded_rectangle([lx, y + u(4), lx + u(15), y + u(19)], radius=u(4), fill=tuple(rgb) + (255,))
        d.text((lx + u(22), y + u(11)), lab, font=fl, fill=qk.INK, anchor="lm")
        lx += u(22) + d.textlength(lab, font=fl) + u(24)

    for i, (t, h) in enumerate(cells):
        vi = 2 * t + h
        gpt = ((float(graw[vi, 0]), float(graw[vi, 1]))
               if show_gaze and vi < len(mraw) and mraw[vi] else None)
        tag = f"t{t}" + ("b" if h else "")
        cellim = qk.render_frame(paths[vi], t, layers_for_t(t), s_h, s_w, gpt, cell_w, label=tag)
        cx = M + (i % cols) * (cw + gap)
        cy = head + (i // cols) * (chh + cap_h + gap)
        S.alpha_composite(cellim, (cx, cy))
        last = i == len(cells) - 1
        if (t, h) in anchors:
            d.rectangle([cx, cy, cx + cw - 1, cy + cellim.height - 1],
                        outline=YELLOW + (255,), width=u(3))
        if last:
            d.rectangle([cx + u(3), cy + u(3), cx + cw - u(4), cy + cellim.height - u(4)],
                        outline=RED + (255,), width=u(3))
        lab = f"{tag}  {mmss(ctime[i], fine)}"
        if cell_note:
            lab += cell_note(t)
        if (t, h) in anchors:
            lab += f"  · {anchor_lab[(t, h)][:24]}"
        if last:
            lab += f"  · LAST FRAME{f', cut {mmss(cutoff, fine)}' if cutoff else ''}"
        d.text((cx + u(2), cy + cellim.height + u(3)), lab, font=fc,
               fill=(RED if last else qk.INK if ((t, h) in anchors) else qk.INK_48))

    S.convert("RGB").resize((Wc // SS, Hc // SS), Image.LANCZOS).save(out_path, quality=92)


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


def _task_of(ds, i):
    """Task of item i without materialising it — __getitem__ loads 128 frames."""
    src, local = ds.items[i]
    return ds._src[src].items[local]["task"]


def passes_gates(r, min_margin, max_evidence_gap):
    """Both rows correct, both confident, and — on the tasks that ask about the query moment —
    that moment actually inside the clip. Shared by the scan's early stop and the ranking so the
    two can never disagree."""
    if r.get("n_correct") != 2:
        return False
    if min(r["teacher"]["margin"], r["student"]["margin"]) < min_margin:
        return False
    if max_evidence_gap and r["task"].startswith(("present_", "proactive_")):
        g = r.get("evidence_gap")          # absent in scans written before this gate existed
        if g is not None and g < max_evidence_gap:
            return False
    return True


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(_REPO, "TrajGazeMerge", "qual", "distill_v2"))
    ap.add_argument("--font", default=os.path.join(_HERE, "Inter.ttf"))
    ap.add_argument("--idxs", default="all")
    ap.add_argument("--scan-limit", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=10)
    ap.add_argument("--per-task-cap", type=int, default=0,
                    help="at most this many items per task (0 = ceil(n_items/5), so no single "
                         "task takes more than a fifth)")
    ap.add_argument("--render-idxs", default=None)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-scan", action="store_true")
    ap.add_argument("--min-margin", type=float, default=0.25,
                    help="drop items where either row's top-2 option gap is below this. The "
                         "floor for reproducibility is 0.05 (bf16 logits quantise to 1/8), but "
                         "a 'correct' at margin 0.06 is a coin flip that happened to land — and "
                         "raising it to 0.25 costs ~4 of 34 candidates.")
    ap.add_argument("--per-window-cap", type=int, default=1,
                    help="at most this many items per (clip, input window). Keyed on the window "
                         "rather than the clip because only items sharing BOTH produce identical "
                         "sheets — the split has 35 clips but 251 windows, so a clip-level cap "
                         "would hard-limit the output to 35.")
    ap.add_argument("--target-items", type=int, default=0,
                    help="stop scanning once this many items have passed the gates (0 = scan the "
                         "whole --idxs list). Lets a top-N run skip most of the split.")
    ap.add_argument("--max-evidence-gap", type=float, default=3.0,
                    help="on present_*/proactive_* tasks, drop items whose next annotated "
                         "fixation starts within this many seconds after the input window — the "
                         "question's own moment is then outside the clip. 0 disables.")
    ap.add_argument("--gaze-radius", type=float, default=0.15,
                    help="disc radius for the concentration metric, in units of frame width")
    ap.add_argument("--w-agree", type=float, default=0.5)
    ap.add_argument("--w-conc", type=float, default=0.5)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--halves", choices=["both", "first"], default="both")
    ap.add_argument("--student-gaze-ring", action="store_true")
    ap.add_argument("--hide-answer", action="store_true",
                    help="answer-free sheet: no ground-truth mark, no verdict, and fixation "
                         "episodes lose their object names on the tasks where the object IS "
                         "the answer")
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--teacher", default=os.environ.get("M1_SGONLY"))
    ap.add_argument("--vit-lora-ckpt", default=None)
    ap.add_argument("--student-ckpt", default=None)
    ap.add_argument("--stage1-ckpt", default=os.environ.get("STAGE1_CKPT"))
    ap.add_argument("--n-vis-keyframes", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    qk.FONT_PATH, qk.SS = args.font, args.supersample
    ck = os.path.join(_REPO, "TrajGazeMerge", "checkpoints")
    vit_ck = args.vit_lora_ckpt or os.path.join(ck, "vitkd_p1_sg_raw", "best.pth")
    stu_ck = args.student_ckpt or os.path.join(ck, "vitkd_p2_sg_raw", "epoch_01.pth")
    for p in (args.teacher, vit_ck, stu_ck, args.stage1_ckpt):
        if not p or not os.path.exists(p):
            raise SystemExit(f"[vitkd] missing checkpoint: {p}")

    M_TEACHER, M_STUDENT = "M1_teacher", "ViT-KD_raw_video"
    dirs = {m: os.path.join(args.out_root, m, "sheets") for m in (M_TEACHER, M_STUDENT)}
    scan_dir = os.path.join(args.out_root, "scan")
    for p in list(dirs.values()) + [scan_dir]:
        os.makedirs(p, exist_ok=True)

    print(f"[vitkd] teacher  {args.teacher}\n"
          f"[vitkd] vit ada  {vit_ck}\n"
          f"[vitkd] student  {stu_ck}\n"
          f"[vitkd] target {CONTENT_RATIO}+{TRAJ_RATIO}  student split P {DOM_P}+{CTX_P}",
          flush=True)

    device = None
    if not args.from_scan:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        processor, model = load_visionzip_lora(device)
        base_qwen = model.get_base_model()
        option_ids = get_option_ids(processor, 5)
        model.eval()

        # attach AFTER load_visionzip_lora (wrapping renames visual.blocks.31.attn.qkv.weight ->
        # ...qkv.base.weight) and keep the wrappers for the whole run: the teacher pass disables
        # them rather than removing them.
        vst = torch.load(vit_ck, map_location="cpu")
        vs = vst["vit_lora_state"] if "vit_lora_state" in vst else vst
        r = vst.get("vit_lora_r", vs["0.lora_A"].shape[0])
        alpha = vst.get("vit_lora_alpha", 2 * r)
        wrappers = attach_vit_lora(base_qwen, r=r, alpha=alpha)
        load_vit_lora_state(wrappers, vs)
        print(f"[vitkd] ViT LoRA r={r} alpha={alpha} on visual.blocks[-1].attn.{{qkv,proj}}; "
              f"recorded metrics={vst.get('metrics')}", flush=True)

        lora = {M_TEACHER: qk.load_lora_only(args.teacher),
                M_STUDENT: qk.load_lora_only(stu_ck)}
        for k, v in lora.items():
            print(f"[vitkd] {k}: {len(v)} lora tensors", flush=True)

        encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    ds.items = [it for it in ds.items if it[0] == "sg"]
    eps = load_episodes()
    qa_ts = {i: _parse_ts(it["time_stamp"]) for i, it in enumerate(build_items())}

    want = parse_idxs(args.render_idxs or args.idxs, len(ds))
    if not args.render_idxs:
        # SG item indices are grouped by task (GSM 0-63, NFI 64-131, ... OI-H 462-525), so a
        # partial scan taken in index order would cover only the first few tasks and the
        # "top N of what was scanned" would be silently task-biased. Round-robin across the
        # task blocks instead, so every prefix of the scan is task-balanced.
        by_task = {}
        for i in want:
            by_task.setdefault(_task_of(ds, i), []).append(i)
        order, qs = [], list(by_task.values())
        for k in range(max(len(v) for v in qs) if qs else 0):
            for v in qs:
                if k < len(v):
                    order.append(v[k])
        want = order
        print(f"[vitkd] scan order: round-robin over {len(qs)} tasks", flush=True)
    scan_path = os.path.join(scan_dir, f"sg{('_' + args.tag) if args.tag else ''}.jsonl")
    cache: dict[int, dict] = {}
    if (args.resume or args.from_scan) and os.path.exists(scan_path):
        for line in open(scan_path):
            if line.strip():
                rec = json.loads(line)
                if "cells" in rec:
                    cache[rec["idx"]] = rec
        print(f"[vitkd] reloaded {len(cache)} scored items from {scan_path}", flush=True)
        want = [i for i in want if i not in cache]
    if args.from_scan:
        want = []
    scan_f = open(os.devnull if args.from_scan else scan_path,
                  "a" if (args.resume and not args.from_scan) else "w")
    print(f"[vitkd] {len(ds)} sg test items; scanning {len(want)}", flush=True)

    n_new, t0, checked = 0, time.time(), False
    n_gated = sum(1 for r in cache.values()
                  if passes_gates(r, args.min_margin, args.max_evidence_gap))
    for idx in want:
        if args.scan_limit and n_new >= args.scan_limit:
            break
        if args.target_items and n_gated >= args.target_items:
            print(f"[vitkd] {n_gated} items have passed the gates; stopping the scan early "
                  f"({n_new} scanned this run)", flush=True)
            break
        item = ds[idx]
        if item is None:
            continue
        n_opt = len(item["options"])
        letters = [chr(65 + i) for i in range(n_opt)]
        if item["answer"] not in letters:
            continue
        try:
            p_viz = item["vlm_frame_paths"]
            p_org = qk.swap_variant(p_viz, "original")
            if any(not os.path.exists(p) for p in p_org):
                continue
            if not checked:
                sub = lambda ps: os.path.basename(os.path.dirname(os.path.dirname(ps[0])))
                print(f"[vitkd] frame streams: teacher VLM='{sub(p_viz)}'  "
                      f"student VLM='{sub(p_org)}'  teacher TAS='{sub(item['traj_frame_paths'])}'",
                      flush=True)
                if sub(p_viz) != "viz" or sub(p_org) != "original" \
                        or sub(item["traj_frame_paths"]) != "viz":
                    raise SystemExit("[vitkd] frame streams wrong; need GAZE_OVERLAY=1 with "
                                     "VLM_GAZE_OVERLAY unset")
                checked = True

            # 1. M1 teacher as deployed: frozen ViT, viz frames, its own LoRA.
            with vit_lora_disabled(wrappers):
                c_T = preprocess_visionzip_item(processor, base_qwen, p_viz, item["question"],
                                                item["options"], device)
                if c_T is None:
                    continue
                S_T_viz, traj_viz = teacher_selection(c_T, item, device, encoder, HP,
                                                      CONTENT_RATIO, TRAJ_RATIO)
                # 2. the distillation TARGET: frozen ViT on the student's own frames. Phase 1
                #    distilled against this, not against the viz-frame selection, so this is the
                #    only teacher set `recall_traj` may be measured against.
                c_tgt = preprocess_visionzip_item(processor, base_qwen, p_org, item["question"],
                                                  item["options"], device)
                if c_tgt is None:
                    continue
                S_T_org, traj_org = teacher_selection(c_tgt, item, device, encoder, HP,
                                                      CONTENT_RATIO, TRAJ_RATIO)
            if S_T_viz is None or S_T_org is None:
                continue
            content_viz = S_T_viz[~torch.isin(S_T_viz, traj_viz)]

            # 3. the ViT-KD student: adapter live, original frames, pure VisionZip at the P split.
            c_S = preprocess_visionzip_item(processor, base_qwen, p_org, item["question"],
                                            item["options"], device)
            if c_S is None:
                continue
            sel_S, keep_S = visionzip_select_tokens(
                c_S["video_embeds"], c_S["attn_scores"], c_S["attn_key"],
                dominant_ratio=DOM_P, contextual_ratio=CTX_P)
            N = c_S["video_embeds"].shape[0]
            dom_S = torch.topk(c_S["attn_scores"], max(1, int(DOM_P * N))).indices
            ctx_S = keep_S[~torch.isin(keep_S, dom_S)]
            recovered = dom_S[torch.isin(dom_S, traj_org)]      # what the magenta draws

            rec_m = selection_metrics(c_S["video_embeds"], c_S["attn_scores"], c_S["attn_key"],
                                      S_T_org, traj_org, DOM_P, CTX_P)

            T, Sg, s_h, s_w = qk.geometry(c_S)
            model.load_state_dict(lora[M_TEACHER], strict=False)
            sel_T = c_T["video_embeds"][S_T_viz]
            t_pi, t_mg = predict(model, base_qwen, c_T, sel_T, S_T_viz, option_ids, n_opt)
            model.load_state_dict(lora[M_STUDENT], strict=False)
            s_pi, s_mg = predict(model, base_qwen, c_S, sel_S, keep_S, option_ids, n_opt)

            graw = item["traj"]["gaze_pos"].float().cpu().numpy()
            mraw = item["traj"]["gaze_mask"].cpu().numpy().astype(bool)
            L = len(p_viz)
            cm_T = qk.cellmap(S_T_viz.cpu(), T, Sg, s_w)
            cm_S = qk.cellmap(keep_S.cpu(), T, Sg, s_w)
            conc_T = gaze_concentration(cm_T, graw, mraw, T, s_h, s_w, L, args.gaze_radius)
            conc_S = gaze_concentration(cm_S, graw, mraw, T, s_h, s_w, L, args.gaze_radius)
            iou_kept = qk.iou(qk.cellset(S_T_org.cpu(), T, Sg, s_w),
                              qk.cellset(keep_S.cpu(), T, Sg, s_w))

            # Two facts the ranking needs that are not in the model outputs:
            #  - the clip stem: consecutive SG items often share a video AND a timestamp, so
            #    their token selection is byte-identical and a second sheet adds nothing.
            #  - the gap to the next annotated fixation after the input window. On
            #    present_*/proactive_* tasks a small positive gap means the moment the question
            #    asks about is NOT in the clip (v2 §6 / v3 §5.6) — the sheet says so in red, and
            #    such an item is a poor choice for "tokens on the key object".
            stem = os.path.basename(os.path.dirname(p_viz[0]))
            last_in = frame_seconds(p_viz[-1])
            first_in = frame_seconds(p_viz[0])
            # 35 clips but 251 distinct input windows: items sharing a clip AND a question
            # timestamp read the same 128 frames, so their selection is byte-identical and the
            # second sheet is a duplicate. Same clip at a different timestamp is not.
            gaps = [e["t0"] - last_in for e in eps.get(stem, []) if e["t0"] > last_in]
            ev_gap = round(min(gaps), 2) if gaps else None
            n_eps = sum(1 for e in eps.get(stem, [])
                        if e["t1"] >= first_in and e["t0"] <= last_in)

            gt = letters.index(item["answer"])
            rec = dict(idx=idx, task=item["task"], answer=item["answer"], n_opt=n_opt,
                       stem=stem, window_end=round(last_in, 1),
                       evidence_gap=ev_gap, episodes_in_window=n_eps,
                       T=T, S=Sg, grid=[s_h, s_w], n_kept=int(keep_S.numel()),
                       pct_kept=100.0 * keep_S.numel() / N,
                       teacher=dict(pred=letters[t_pi], ok=t_pi == gt, margin=round(t_mg, 4)),
                       student=dict(pred=letters[s_pi], ok=s_pi == gt, margin=round(s_mg, 4)),
                       n_correct=int(t_pi == gt) + int(s_pi == gt),
                       recall_traj=round(rec_m["recall_traj"], 4),
                       recall_P=round(rec_m["recall_P"], 4),
                       recall_S=round(rec_m["recall_S"], 4),
                       iou_kept=round(iou_kept, 4),
                       conc_teacher=round(conc_T, 4), conc_student=round(conc_S, 4),
                       cells=dict(
                           t_content=content_viz.tolist(), t_traj=traj_viz.tolist(),
                           s_dom=dom_S.tolist(), s_ctx=ctx_S.tolist(),
                           s_recovered=recovered.tolist()))
            scan_f.write(json.dumps(rec) + "\n"); scan_f.flush()
            cache[idx] = rec
            n_new += 1
            if passes_gates(rec, args.min_margin, args.max_evidence_gap):
                n_gated += 1
            print(f"  idx={idx:<4} {item['task'][:34]:<34} "
                  f"T={letters[t_pi]}{'✓' if t_pi == gt else '✗'} "
                  f"S={letters[s_pi]}{'✓' if s_pi == gt else '✗'} GT={item['answer']} "
                  f"rec_traj={rec_m['recall_traj']:.3f} iou={iou_kept:.3f} "
                  f"conc T/S={conc_T:.2f}/{conc_S:.2f} "
                  f"[{(time.time() - t0) / max(1, n_new):.1f}s/item]", flush=True)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  idx={idx} err: {type(e).__name__}: {e}", flush=True)
            continue
    scan_f.close()
    if n_new:
        print(f"[vitkd] {n_new} newly scanned in {time.time() - t0:.0f}s; "
              f"{len(cache)} total -> {scan_path}", flush=True)
    if args.scan_only:
        return

    # ── ranking ──
    if args.render_idxs:
        chosen = [i for i in parse_idxs(args.render_idxs, len(ds)) if i in cache]
    else:
        both = [i for i in cache if cache[i]["n_correct"] == 2]
        elig = [i for i in cache
                if passes_gates(cache[i], args.min_margin, args.max_evidence_gap)]
        print(f"[vitkd] {len(both)} of {len(cache)} have both rows correct; {len(elig)} also pass "
              f"margin >= {args.min_margin} and the evidence gate "
              f"(< {args.max_evidence_gap}s to the next fixation on present_*/proactive_*)",
              flush=True)

        def pct(vals):
            """percentile within the scanned set — agreement and concentration are on different
            scales, so neither can be summed with the other raw."""
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            out = [0.0] * len(vals)
            for rank, i in enumerate(order):
                out[i] = rank / max(1, len(vals) - 1)
            return out
        agree = pct([cache[i]["recall_traj"] for i in elig])
        conc = pct([cache[i]["conc_student"] for i in elig])
        score = {i: args.w_agree * a + args.w_conc * c for i, a, c in zip(elig, agree, conc)}
        task_cap = args.per_task_cap or max(3, math.ceil(args.n_items / 5))
        chosen, per_task, per_win = [], {}, {}
        for i in sorted(elig, key=lambda i: score[i], reverse=True):
            t = cache[i]["task"]
            w = (cache[i].get("stem", f"_{i}"), cache[i].get("window_end"))
            if per_task.get(t, 0) >= task_cap:
                continue
            if per_win.get(w, 0) >= args.per_window_cap:
                continue
            per_task[t] = per_task.get(t, 0) + 1
            per_win[w] = per_win.get(w, 0) + 1
            chosen.append(i)
            if len(chosen) >= args.n_items:
                break

    print(f"[vitkd] rendering {len(chosen)} items x 2 sheets", flush=True)
    man = []
    for idx in chosen:
        item = ds[idx]
        if item is None:
            continue
        r = cache[idx]
        T, Sg, s_h, s_w = r["T"], r["S"], r["grid"][0], r["grid"][1]
        p_viz = item["vlm_frame_paths"]
        p_org = qk.swap_variant(p_viz, "original")
        graw = item["traj"]["gaze_pos"].float().cpu().numpy()
        mraw = item["traj"]["gaze_mask"].cpu().numpy().astype(bool)
        cl = r["cells"]
        cmap = {k: qk.cellmap(v, T, Sg, s_w) for k, v in cl.items()}
        stem = f"sg_idx{idx}_{item['task']}".replace("/", "_")
        cut = qa_ts.get(idx)
        # the verdict discloses the ground truth, so it is printed only on a sheet that marks it
        vsay = ((lambda who: f" · answer {r[who]['pred']} "
                             f"({'correct' if r[who]['ok'] else 'WRONG'}, "
                             f"margin {r[who]['margin']:.2f})") if not args.hide_answer
                else (lambda who: f" · answer {r[who]['pred']} (margin {r[who]['margin']:.2f})"))

        build_sheet(
            item, idx, p_viz,
            lambda t: [(cmap["t_content"][t], 0, qk.C_CONTENT, qk.u(2)),
                       (cmap["t_traj"][t], 120, qk.C_COMP, qk.u(3))],
            T, s_h, s_w, graw, mraw, eps, cut,
            title=f"ALL {T} GROUPS  ·  M1 SG specialist teacher (gaze at test)  ·  sg idx {idx}  "
                  f"·  {item['task']}",
            subtitle=("7% VisionZip content + 3% gaze/hand complement (frozen TAS encoder) on "
                       "`viz` frames" + vsay("teacher")
                      + f" · gaze concentration {r['conc_teacher']:.2f}"),
            legend=[(qk.C_CONTENT, "content 7%"), (qk.C_COMP, "gaze/hand complement 3%"),
                    (qk.C_GAZE, "gaze"), (YELLOW, "annotated fixation episode"),
                    (RED, "last frame of the clip")],
            out_path=os.path.join(dirs[M_TEACHER], stem + ".png"),
            cols=args.cols, cell_w=args.cell, halves=args.halves, show_gaze=True,
            show_answer=not args.hide_answer)

        build_sheet(
            item, idx, p_org,
            lambda t: [(cmap["s_ctx"][t], 0, qk.C_CONTENT, qk.u(1)),
                       (cmap["s_dom"][t], 0, qk.C_CONTENT, qk.u(2)),
                       (cmap["s_recovered"][t], 120, qk.C_COMP, qk.u(3))],
            T, s_h, s_w, graw, mraw, eps, cut,
            title=f"ALL {T} GROUPS  ·  ViT-KD raw video (0 extra params, epoch 1)  ·  sg idx {idx}"
                  f"  ·  {item['task']}",
            subtitle=("pure VisionZip on the distilled ViT, 6.5% dominant + 3.5% contextual, on "
                       "marker-free `original` frames" + vsay("student")
                      + f" · recall_traj {r['recall_traj']:.2f} (magenta)"
                        f" · gaze concentration {r['conc_student']:.2f}"),
            legend=[(qk.C_CONTENT, "kept 10% (thick = dominant 6.5%, thin = contextual 3.5%)"),
                    (qk.C_COMP, "dominant tokens that match the teacher's gaze 3% (recall_traj)"),
                    (YELLOW, "annotated fixation episode"), (RED, "last frame of the clip")],
            out_path=os.path.join(dirs[M_STUDENT], stem + ".png"),
            cols=args.cols, cell_w=args.cell, halves=args.halves,
            show_gaze=args.student_gaze_ring, show_answer=not args.hide_answer)

        print(f"  saved {stem}.png  (both methods)", flush=True)
        man.append(dict(item=stem, idx=idx, task=item["task"], answer=item["answer"],
                        teacher_sheet=os.path.relpath(os.path.join(dirs[M_TEACHER], stem + ".png"),
                                                      args.out_root),
                        student_sheet=os.path.relpath(os.path.join(dirs[M_STUDENT], stem + ".png"),
                                                      args.out_root),
                        pred_teacher=r["teacher"]["pred"], pred_student=r["student"]["pred"],
                        ok_teacher=r["teacher"]["ok"], ok_student=r["student"]["ok"],
                        margin_teacher=r["teacher"]["margin"], margin_student=r["student"]["margin"],
                        recall_traj=r["recall_traj"], recall_P=r["recall_P"], recall_S=r["recall_S"],
                        iou_kept=r["iou_kept"], stem=r.get("stem", ""),
                        window_end=r.get("window_end"),
                        evidence_gap=r.get("evidence_gap"),
                        episodes_in_window=r.get("episodes_in_window"),
                        conc_teacher=r["conc_teacher"],
                        conc_student=r["conc_student"], pct_kept=round(r["pct_kept"], 3),
                        ckpt_teacher=args.teacher, ckpt_vit_lora=vit_ck, ckpt_student=stu_ck))

    if man:
        p = os.path.join(args.out_root, "manifest.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
            w.writeheader(); w.writerows(man)
        print(f"[vitkd] manifest -> {p}", flush=True)


if __name__ == "__main__":
    main()
