"""Staged smoke test for the ported KD pipeline.

Replaces the handoff's `/tmp/smoke_kd.py`, which lived on the original machine.
Each stage is isolated so a failure names exactly what broke during the port.

    source env.sh && python scripts/smoke_kd.py            # A-G
    source env.sh && python scripts/smoke_kd.py --stage B  # one stage

Stage B is the gate: the handoff (docs/kd_handoff.md, section 4) fixes the eval
split at SG n=526, EG n=485, combined 1011. Wrong counts mean the extraction or
the path remapping is wrong, and nothing downstream is meaningful.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"))

EXPECT_SG_TEST = 526
EXPECT_EG_TEST = 485

_results: list[tuple[str, bool, str]] = []


def stage(name: str, fn):
    print(f"\n{'='*70}\n[{name}] {fn.__doc__.strip()}\n{'='*70}", flush=True)
    try:
        msg = fn() or "ok"
        _results.append((name, True, msg))
        print(f"  PASS: {msg}", flush=True)
        return True
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        traceback.print_exc()
        print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
        return False


# ── A ────────────────────────────────────────────────────────────────────────
def stage_a():
    """VisionZip fork imports and exposes the fields the pipeline reads."""
    import torch
    import transformers
    from qwen2_5vl_visionzip import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    import peft
    return (f"torch={torch.__version__} transformers={transformers.__version__} "
            f"peft={peft.__version__} cuda={torch.cuda.is_available()} "
            f"gpus={torch.cuda.device_count()}")


# ── B (gate) ─────────────────────────────────────────────────────────────────
def stage_b():
    """Eval split sizes match the handoff: SG 526 / EG 485 / 1011."""
    from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    n_sg = sum(1 for s, _ in ds.items if s == "sg")
    n_eg = sum(1 for s, _ in ds.items if s == "eg")
    got = f"SG={n_sg} EG={n_eg} total={len(ds)}"
    if n_sg != EXPECT_SG_TEST or n_eg != EXPECT_EG_TEST:
        raise AssertionError(
            f"{got}  != expected SG={EXPECT_SG_TEST} EG={EXPECT_EG_TEST} "
            f"total={EXPECT_SG_TEST + EXPECT_EG_TEST}")
    return got


# ── C ────────────────────────────────────────────────────────────────────────
def stage_c():
    """Train split builds and is non-empty."""
    from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
    ds = CombinedMergeDataset(split="train", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    n_sg = sum(1 for s, _ in ds.items if s == "sg")
    n_eg = sum(1 for s, _ in ds.items if s == "eg")
    if len(ds) == 0:
        raise AssertionError("train split is empty")
    return f"SG={n_sg} EG={n_eg} total={len(ds)}"


# ── D ────────────────────────────────────────────────────────────────────────
def stage_d():
    """One item per source loads: frames exist on disk, traj tensors are shaped."""
    from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    out = []
    for want in ("sg", "eg"):
        idx = next(i for i, (s, _) in enumerate(ds.items) if s == want)
        item = ds[idx]
        if item is None:
            raise AssertionError(f"{want}: __getitem__ returned None")
        paths = item["vlm_frame_paths"]
        if not paths:
            raise AssertionError(f"{want}: no vlm_frame_paths")
        missing = [p for p in paths[:20] if not os.path.exists(p)]
        if missing:
            raise AssertionError(f"{want}: frame paths do not exist, e.g. {missing[0]}")
        traj_keys = sorted(item["traj"].keys())
        out.append(f"{want}: {len(paths)} frames, {len(item['options'])} opts, "
                   f"answer={item['answer']}, traj_keys={len(traj_keys)}")
    return " | ".join(out)


# ── E ────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}


def _load_model():
    if "m" not in _MODEL_CACHE:
        import torch
        from TrajGazeMerge.training.train_visionzip_lora import load_visionzip_lora
        device = torch.device("cuda:0")
        processor, model = load_visionzip_lora(device)
        _MODEL_CACHE.update(processor=processor, model=model, device=device, m=True)
    return _MODEL_CACHE


def stage_e():
    """VisionZip+LoRA loads and the M1 teacher checkpoint warm-starts cleanly."""
    import torch
    c = _load_model()
    ckpt = os.environ.get("M1_JOINT")
    if not ckpt or not os.path.exists(ckpt):
        raise AssertionError(f"M1_JOINT not found: {ckpt}")
    sd = torch.load(ckpt, map_location="cpu")["lora_state"]
    missing, unexpected = c["model"].load_state_dict(sd, strict=False)
    # The 16.6 GB ckpt carries the whole model state, so `missing` is 0 too.
    # `unexpected` is the real check — nonzero means the ckpt has keys this
    # peft/VisionZip build cannot place, i.e. a version mismatch.
    if len(unexpected) > 0:
        raise AssertionError(
            f"{len(unexpected)} unexpected keys (peft/VisionZip build mismatch), "
            f"e.g. {unexpected[:3]}")
    n_lora_ckpt = sum(1 for k in sd if "lora_" in k)
    n_lora_model = sum(1 for k, _ in c["model"].named_parameters() if "lora_" in k)
    if n_lora_ckpt == 0:
        raise AssertionError("checkpoint contains no lora_ keys")
    return (f"lora keys ckpt={n_lora_ckpt} model={n_lora_model}, "
            f"missing={len(missing)} unexpected={len(unexpected)}")


# ── F ────────────────────────────────────────────────────────────────────────
def stage_f():
    """Frozen TAS Stage-1 encoder (the privileged gaze/hand teacher) loads."""
    import torch
    from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder
    ckpt = os.environ.get("STAGE1_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        raise AssertionError(f"STAGE1_CKPT not found: {ckpt}")
    device = torch.device("cuda:0")
    enc = load_traj_encoder("full", ckpt, device, 16)
    enc.eval()
    n = sum(p.numel() for p in enc.parameters())
    return f"TAS encoder loaded, params={n/1e6:.2f}M"


# ── G ────────────────────────────────────────────────────────────────────────
def stage_g():
    """End-to-end on one item: ViT -> content/avail split -> teacher field -> 10% selection."""
    import torch
    from TrajGazeMerge.data.combined_dataset import CombinedMergeDataset
    from TrajGazeMerge.training.train_visionzip_lora import preprocess_visionzip_item
    from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import load_traj_encoder
    from TrajGazeMerge.training.train_visionzip_complement_lora import _traj_scores
    from TrajGazeMerge.training.train_visionzip_kd_lora import (
        content_and_avail, topk_in_avail, union_tokens, selection_kd_loss,
    )
    from TrajGazeMerge.models.traj_salience_predictor import TrajSaliencePredictor

    c = _load_model()
    processor, model, device = c["processor"], c["model"], c["device"]
    base_qwen = model.get_base_model()
    encoder = load_traj_encoder("full", os.environ["STAGE1_CKPT"], device, 16)
    encoder.eval()

    ds = CombinedMergeDataset(split="test", n_vlm_frames=128, n_traj_frames=128,
                              include_hdepic=False)
    idx = next(i for i, (s, _) in enumerate(ds.items) if s == "sg")
    item = ds[idx]

    cached = preprocess_visionzip_item(
        processor, base_qwen, item["vlm_frame_paths"],
        item["question"], item["options"], device)
    if cached is None:
        raise AssertionError("preprocess_visionzip_item returned None")

    N = cached["video_embeds"].shape[0]
    content_embeds, content_idx, avail_idx = content_and_avail(cached, 0.07)
    k = min(max(1, int(0.03 * N)), avail_idx.numel())

    with torch.no_grad():
        s_teacher = _traj_scores(cached, item, device, "learned", encoder,
                                 dict(mask_modality="none"))
    if s_teacher.shape[0] != N:
        raise AssertionError(f"teacher field {tuple(s_teacher.shape)} != N={N}")

    in_dim = base_qwen.get_input_embeddings().weight.shape[1]
    predictor = TrajSaliencePredictor(in_dim, hidden=512).to(device)
    s_student = predictor(cached["video_embeds"], cached["attn_scores"], cached["grid_thw"])
    loss, agree = selection_kd_loss(s_student, s_teacher, avail_idx, k)

    traj_idx, _ = topk_in_avail(s_teacher, avail_idx, k)
    sel_embeds, recv_idx = union_tokens(cached, content_embeds, content_idx, traj_idx)
    pct = 100.0 * recv_idx.shape[0] / N
    if not (8.0 <= pct <= 12.0):
        raise AssertionError(f"kept {pct:.1f}% of tokens, expected ~10%")
    return (f"N={N} content={content_idx.numel()} avail={avail_idx.numel()} k={k} "
            f"kept={pct:.1f}% kd_loss={loss.item():.4f} agree(untrained)={agree:.3f}")


STAGES = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d,
          "E": stage_e, "F": stage_f, "G": stage_g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, choices=sorted(STAGES),
                    help="run a single stage instead of A-G")
    args = ap.parse_args()

    names = [args.stage] if args.stage else sorted(STAGES)
    for name in names:
        ok = stage(name, STAGES[name])
        if not ok and name == "B":
            print("\nStage B is the gate — stopping. Fix extraction/paths first.", flush=True)
            break

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, ok, msg in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results)-n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
