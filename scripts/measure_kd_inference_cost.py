"""Deployment-side cost of the three selection mechanisms (supplementary KD table).

Companion to scripts/bench_kd_efficiency.sh, which covers the training side.

    source env.sh && CUDA_VISIBLE_DEVICES=0 python scripts/measure_kd_inference_cost.py

kd_handoff_v3.md §1 states the deployment claim as a parameter count -- 36.85 M for the
M1 teacher, 3.95 M for the KD student, 0 for ViT-KD -- but never times it. This script
measures what those parameters actually cost per item, and how that compares to the
forward passes all three systems share.

Per item the three systems differ only in how the 10% is chosen:

    teacher   7% VisionZip content  u  3% ranked by the frozen TAS encoder  (needs gaze)
    student   7% VisionZip content  u  3% ranked by a 3.95 M RGB head
    ViT-KD    pure VisionZip 6.5% dominant + 3.5% contextual on the distilled ViT

Everything else -- the ViT pass that produces `video_embeds` / `attn_scores`, and the
7B LLM forward over the kept tokens -- is identical, so it is timed once and reported
as the denominator rather than attributed to any row.

Weights do not affect latency, so the predictor is constructed fresh rather than pulled
out of a 16.6 GB checkpoint; its parameter count is the check that it is the right
module (3.95 M). The ViT adapter is read from setting 1's real Phase-1 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"))

import torch  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Wall time and peak allocated memory for one isolated stage.

    The peak is reset per call, so what comes back is this stage's own high-water
    mark, not the run's. GB is 1e9 bytes, matching measure_vitkd_step0.py and the
    "peak 21.7 GB" recorded in kd_handoff_v3.md §5.1.
    """

    def __init__(self):
        self.ms: dict[str, list[float]] = {}
        self.peak_gb: dict[str, float] = {}

    def __call__(self, name, fn, record=True):
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        dt = (time.perf_counter() - t0) * 1e3
        if record:
            self.ms.setdefault(name, []).append(dt)
            pk = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            self.peak_gb[name] = max(self.peak_gb.get(name, 0.0), pk)
        return out

    def stats(self, name) -> dict:
        v = self.ms.get(name, [])
        if not v:
            return {}
        return {
            "n": len(v),
            "mean_ms": statistics.mean(v),
            "median_ms": statistics.median(v),
            "stdev_ms": statistics.stdev(v) if len(v) > 1 else 0.0,
            "peak_gb": self.peak_gb.get(name, 0.0),
        }


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def adapter_params(path: str) -> tuple[int, str]:
    """Sum the tensor elements in a ViT-KD Phase-1 adapter checkpoint."""
    if not os.path.exists(path):
        return 0, f"missing: {path}"
    sd = torch.load(path, map_location="cpu")
    for key in ("lora_state", "vit_lora_state", "state", "model"):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    return n, f"{len(sd)} tensors from {os.path.basename(path)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=24,
                    help="SG test items to time (after --warmup)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="items run but not recorded")
    ap.add_argument("--dominant-ratio", type=float, default=0.065)
    ap.add_argument("--contextual-ratio", type=float, default=0.035)
    ap.add_argument("--content-ratio", type=float, default=0.07)
    ap.add_argument("--traj-ratio", type=float, default=0.03)
    ap.add_argument("--out", default=os.path.join(_REPO, "bench_kd_efficiency",
                                                  "inference.json"))
    args = ap.parse_args()

    from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
    from TrajGazeMerge.models.model import build_merged_inputs, forward_logits
    from TrajGazeMerge.models.traj_salience_predictor import TrajSaliencePredictor
    from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder
    from TrajGazeMerge.training.train_visionzip_complement_lora import _traj_scores
    from TrajGazeMerge.training.train_visionzip_kd_lora import (
        content_and_avail, topk_in_avail, union_tokens,
    )
    from TrajGazeMerge.training.train_visionzip_lora import (
        load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stage1 = os.environ["STAGE1_CKPT"]

    print("[inf] loading VisionZip Qwen2.5-VL-7B + LoRA ...", flush=True)
    processor, model = load_visionzip_lora(device)
    n_llm_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_backbone = sum(p.numel() for p in model.parameters())
    model.eval()
    base_qwen = model.get_base_model()

    print("[inf] loading frozen TAS Stage-1 encoder ...", flush=True)
    encoder = load_traj_encoder("full", stage1, device, 16)
    encoder.eval()
    hp = dict(mask_modality="none")

    in_dim = base_qwen.get_input_embeddings().weight.shape[1]
    predictor = TrajSaliencePredictor(in_dim, hidden=512).to(device)
    predictor.eval()

    n_enc = count_params(encoder)
    n_pred = count_params(predictor)
    enc_dtype = next(encoder.parameters()).dtype
    pred_dtype = next(predictor.parameters()).dtype
    n_vit, vit_note = adapter_params(os.path.join(
        _REPO, "TrajGazeMerge", "checkpoints", "vitkd_p1_sg_raw", "best.pth"))

    print(f"[inf] backbone           {n_backbone:,} params ({n_backbone/1e9:.2f} B)")
    print(f"[inf] LLM LoRA (trainable) {n_llm_lora:,} params ({n_llm_lora/1e6:.2f} M)")
    print(f"[inf] TAS encoder        {n_enc:,} params ({n_enc/1e6:.2f} M, {enc_dtype})")
    print(f"[inf] TrajSalienceP.     {n_pred:,} params ({n_pred/1e6:.2f} M, {pred_dtype})")
    print(f"[inf] ViT-KD adapter     {n_vit:,} params -- {vit_note}")

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    sg_idx = [i for i, (s, _) in enumerate(ds.items) if s == "sg"]
    print(f"[inf] SG test items: {len(sg_idx)} (using {args.warmup}+{args.items})",
          flush=True)

    T = Timer()
    n_done = 0
    n_tokens: list[int] = []

    for i in sg_idx:
        if n_done >= args.warmup + args.items:
            break
        item = ds[i]
        if item is None:
            continue
        # Count successfully processed items, not dataset positions: a None item
        # would otherwise shift the warmup boundary silently.
        rec = n_done >= args.warmup

        with torch.no_grad():
            # ── shared: ViT pass ──────────────────────────────────────────────
            cached = T("shared_vit", lambda: preprocess_visionzip_item(
                processor, base_qwen, item["vlm_frame_paths"],
                item["question"], item["options"], device), record=rec)
            if cached is None:
                continue
            N = cached["video_embeds"].shape[0]
            k = max(1, int(args.traj_ratio * N))

            # ── teacher: TAS encoder is the extra module ──────────────────────
            ce, cidx, avail = T("teacher_content_split",
                                lambda: content_and_avail(cached, args.content_ratio),
                                record=rec)
            s_t = T("teacher_extra_tas",
                    lambda: _traj_scores(cached, item, device, "learned", encoder, hp),
                    record=rec)
            T("teacher_topk_union", lambda: union_tokens(
                cached, ce, cidx, topk_in_avail(s_t, avail, min(k, avail.numel()))[0]),
              record=rec)

            # ── student: the 3.95 M RGB head is the extra module ──────────────
            s_s = T("student_extra_predictor",
                    lambda: predictor(cached["video_embeds"], cached["attn_scores"],
                                      cached["grid_thw"]),
                    record=rec)
            T("student_topk_union", lambda: union_tokens(
                cached, ce, cidx, topk_in_avail(s_s, avail, min(k, avail.numel()))[0]),
              record=rec)

            # ── ViT-KD: no extra module; plain VisionZip on the distilled ViT ──
            sel, recv = T("vitkd_select", lambda: visionzip_select_tokens(
                cached["video_embeds"], cached["attn_scores"], cached["attn_key"],
                dominant_ratio=args.dominant_ratio,
                contextual_ratio=args.contextual_ratio), record=rec)

            # ── shared: LLM forward over the kept ~10% ────────────────────────
            inputs_dict = T("shared_build_inputs",
                            lambda: build_merged_inputs(base_qwen, cached, sel, recv),
                            record=rec)
            T("shared_llm_forward", lambda: forward_logits(model, inputs_dict),
              record=rec)

        if rec:
            n_tokens.append(int(recv.shape[0]))
        n_done += 1
        if n_done % 5 == 0:
            print(f"[inf]   {n_done}/{args.warmup + args.items} items", flush=True)

    names = ["shared_vit", "shared_build_inputs", "shared_llm_forward",
             "teacher_content_split", "teacher_extra_tas", "teacher_topk_union",
             "student_extra_predictor", "student_topk_union", "vitkd_select"]
    per_stage = {n: T.stats(n) for n in names if T.stats(n)}

    shared_ms = sum(per_stage[n]["mean_ms"] for n in
                    ("shared_vit", "shared_build_inputs", "shared_llm_forward")
                    if n in per_stage)

    extra = {
        "teacher": per_stage.get("teacher_extra_tas", {}).get("mean_ms", 0.0),
        "student": per_stage.get("student_extra_predictor", {}).get("mean_ms", 0.0),
        "vitkd": 0.0,
    }

    result = {
        "protocol": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "items_timed": len(n_tokens),
            "warmup_items": args.warmup,
            "vlm_gaze_overlay": os.environ.get("VLM_GAZE_OVERLAY", "<unset>"),
            "gaze_overlay": os.environ.get("GAZE_OVERLAY", "<unset>"),
            "ratios": {"content": args.content_ratio, "traj": args.traj_ratio,
                       "dominant": args.dominant_ratio, "contextual": args.contextual_ratio},
            "kept_tokens_mean": statistics.mean(n_tokens) if n_tokens else 0,
        },
        "params": {
            "backbone": n_backbone,
            "llm_lora_trainable": n_llm_lora,
            "teacher_tas_encoder": n_enc,
            "student_predictor": n_pred,
            "vitkd_adapter_trained": n_vit,
            "vitkd_adapter_at_inference": 0,
            "note": ("ViT-KD's adapter is rank-8 LoRA on visual.blocks[31] and folds "
                     "into the ViT weights, so nothing is added at inference. Report "
                     "the 0 as 'folded', not as 'no module'."),
        },
        "extra_mb": {
            "teacher_tas_encoder": n_enc * enc_dtype.itemsize / 1e6,
            "student_predictor": n_pred * pred_dtype.itemsize / 1e6,
            "vitkd": 0.0,
        },
        "per_stage_ms": per_stage,
        "shared_ms_per_item": shared_ms,
        "extra_ms_per_item": extra,
        "extra_pct_of_shared": {k: (100.0 * v / shared_ms if shared_ms else 0.0)
                                for k, v in extra.items()},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 74)
    print(f"{'stage':26s} {'mean ms':>10s} {'median':>10s} {'peak GB':>9s}")
    print("-" * 74)
    for n in names:
        s = per_stage.get(n)
        if s:
            print(f"{n:26s} {s['mean_ms']:10.2f} {s['median_ms']:10.2f} {s['peak_gb']:9.2f}")
    print("-" * 74)
    print(f"shared per item: {shared_ms:.1f} ms   (ViT + build + LLM forward)")
    for k, v in extra.items():
        pct = 100.0 * v / shared_ms if shared_ms else 0.0
        print(f"  extra {k:8s}: {v:8.2f} ms  = {pct:5.2f}% of shared")
    print("=" * 74)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
