"""Integrity gate — did distilling the selection into the ViT damage the ViT?

The claim the ablation has to defend is "we fine-tuned the vision encoder without
breaking it". The direct test holds the SELECTION fixed and varies only the
FEATURES:

    baseline : frozen ViT scores choose the tokens, frozen ViT embeddings are sent
    tuned    : frozen ViT scores choose the SAME tokens, TUNED  embeddings are sent

Any accuracy difference is therefore representation drift in block 31 and nothing
else. Comparing the two end-to-end 10%-budget numbers instead (frozen selection vs
tuned selection) would confound drift with the selection change that is the point
of the experiment.

visionzip_select_tokens takes video_embeds and (attn_scores, attn_key) separately,
so "frozen selection, tuned features" is just a matter of which tensors go in which
argument — the merge of contextual tokens is redone with the tuned embeddings, which
is what actually reaches the LLM.

An earlier version of this gate ran a ~100%-token eval instead. That works, but it
hands the 7B model ~13.8k visual tokens per item rather than ~1.4k, and at eight
gate runs it costs more than the experiment.

Pass/fail: docs/kd_handoff_v2.md §8 puts the re-scoring noise floor at 3-4 items on
a 526-item split, so a gap of <=4 items is "not damaged" and anything larger is a
real regression — raise --lambda-anchor or lower the Phase-1 LR and retrain.

Usage:
    GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=0 CUDA_VISIBLE_DEVICES=0 \\
    python scripts/vitkd_integrity_gate.py --source sg \\
        --lora-ckpt "$M1_SGONLY" --vit-lora-ckpt .../vitkd_p1_sg_raw/best.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VisionZip", "Qwen2_5_VL"))

from TrajGazeMerge.data.combined_simple_dataset import CombinedSimpleDataset
from TrajGazeMerge.models.model import get_option_ids, build_merged_inputs, forward_logits
from TrajGazeMerge.training.train_visionzip_lora import (
    load_visionzip_lora, preprocess_visionzip_item, visionzip_select_tokens,
)
from TrajGazeMerge.training.train_vit_selection_kd import (
    attach_vit_lora, load_vit_lora_state, vit_lora_disabled,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["sg", "eg", "both"], default="sg")
    p.add_argument("--lora-ckpt", required=True, help="LLM LoRA (the M1 checkpoint).")
    p.add_argument("--vit-lora-ckpt", required=True, help="Phase-1 ViT adapter.")
    p.add_argument("--dominant-ratio", type=float, default=0.065)
    p.add_argument("--contextual-ratio", type=float, default=0.035)
    p.add_argument("--n-frames", type=int, default=128)
    p.add_argument("--max-items", type=int, default=0, help="0 = the whole split.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    gz = os.environ.get("GAZE_OVERLAY", "1")
    print(f"[gate] GAZE_OVERLAY={gz} VLM_GAZE_OVERLAY={os.environ.get('VLM_GAZE_OVERLAY', gz)}",
          flush=True)

    processor, model = load_visionzip_lora(device)
    base = model.get_base_model()
    st = torch.load(args.lora_ckpt, map_location="cpu")
    sd = st["lora_state"] if "lora_state" in st else st
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[gate] LLM LoRA loaded (missing={len(miss)} unexpected={len(unexp)})", flush=True)

    vst = torch.load(args.vit_lora_ckpt, map_location="cpu")
    vs = vst["vit_lora_state"] if "vit_lora_state" in vst else vst
    r = vst.get("vit_lora_r", vs["0.lora_A"].shape[0])
    alpha = vst.get("vit_lora_alpha", 2 * r)
    wrappers = attach_vit_lora(base, r=r, alpha=alpha)
    load_vit_lora_state(wrappers, vs)
    print(f"[gate] ViT LoRA loaded from {args.vit_lora_ckpt} (r={r}, alpha={alpha})",
          flush=True)

    model.eval()
    option_ids = get_option_ids(processor, 5)

    ds = CombinedSimpleDataset(split="test", n_vlm_frames=args.n_frames,
                               include_hdepic=False)
    if args.source in ("sg", "eg"):
        ds.items = [it for it in ds.items if it[0] == args.source]
    n = len(ds) if not args.max_items else min(args.max_items, len(ds))
    print(f"[gate] {n} items, budget {args.dominant_ratio:.3f}+{args.contextual_ratio:.3f}",
          flush=True)

    ok_base = ok_tuned = total = 0
    cos_sum = 0.0
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            if item is None:
                continue
            try:
                # Adapter ON: preprocess_visionzip_item runs the encoder internally,
                # so this pass yields the TUNED embeddings (and tuned scores, unused).
                cached = preprocess_visionzip_item(
                    processor, base, item["vlm_frame_paths"], item["question"],
                    item["options"], device)
                if cached is None:
                    continue
                n_opt = len(item["options"])
                letters = [chr(65 + i2) for i2 in range(n_opt)]
                if item["answer"] not in letters:
                    continue
                gt = letters.index(item["answer"])
                ve_t = cached["video_embeds"]

                # Adapter OFF: the frozen encoder, which supplies BOTH the selection
                # signal and the baseline features.
                with vit_lora_disabled(wrappers):
                    cached_f = preprocess_visionzip_item(
                        processor, base, item["vlm_frame_paths"], item["question"],
                        item["options"], device)
                if cached_f is None:
                    continue
                ve_f, sc_f, key_f = (cached_f["video_embeds"], cached_f["attn_scores"],
                                     cached_f["attn_key"])

                cos_sum += F.cosine_similarity(ve_t.float(), ve_f.float(),
                                               dim=-1).mean().item()

                for tag, embeds in (("base", ve_f), ("tuned", ve_t)):
                    # SAME scores and keys in both cases -> identical token choice.
                    sel, recv = visionzip_select_tokens(
                        embeds, sc_f, key_f,
                        dominant_ratio=args.dominant_ratio,
                        contextual_ratio=args.contextual_ratio)
                    inp = build_merged_inputs(base, cached_f, sel, recv)
                    logits = forward_logits(model, inp)
                    pred = int(logits[option_ids[:n_opt]].argmax().item())
                    if pred == gt:
                        if tag == "base":
                            ok_base += 1
                        else:
                            ok_tuned += 1
                total += 1
                if total % 50 == 0:
                    print(f"[gate] {total}/{n}  base={ok_base} tuned={ok_tuned} "
                          f"cos={cos_sum/total:.5f}", flush=True)
            except Exception:
                traceback.print_exc()
                continue

    d = ok_tuned - ok_base
    print("\n===== INTEGRITY GATE =====")
    print(f"  n scored           : {total}")
    print(f"  frozen features    : {ok_base} items ({100.0*ok_base/max(1,total):.2f}%)")
    print(f"  tuned  features    : {ok_tuned} items ({100.0*ok_tuned/max(1,total):.2f}%)")
    print(f"  Δ                  : {d:+d} items")
    print(f"  mean cos(ve_tuned, ve_frozen) : {cos_sum/max(1,total):.5f}")
    verdict = "PASS" if abs(d) <= 4 else "FAIL"
    print(f"  verdict            : {verdict}  (|Δ| <= 4 items = §8 noise floor)")
    print("==========================", flush=True)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
