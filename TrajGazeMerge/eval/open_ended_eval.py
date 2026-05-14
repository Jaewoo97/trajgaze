"""
Open-ended generation evaluation.

Tests whether the model relies on the multiple-choice (MC) prompt structure
or actually understands the question. For each sample (subset, default 200):

  Mode "mc"    : same prompt as production (with options A/B/C/D shown),
                 use greedy decoding instead of logit argmax. Should match
                 logit-based accuracy ~ exactly (sanity check).
  Mode "open"  : prompt with NO options shown. Model generates a free-text
                 answer (~30 tokens). Then TF-IDF cosine similarity is used
                 to match the generated text to one of the 4 option texts;
                 predicted = argmax similarity.

If MC accuracy >> Open accuracy → model relies on MC option structure.
If MC ≈ Open                   → model genuinely understands.

Same TrajGazeMerge pipeline (frozen Qwen ViT + TrajGaze encoder + LoRA + merge)
is used for both modes — only the textual prompt + decoding path differ.

Output:
  <tag>_open_ended_per_sample.parquet
  <tag>_open_ended_summary.json

Usage:
  python -m TrajGazeMerge.eval.open_ended_eval \
      --stage1-ckpt /workspace/trajgaze/TrajGaze_v2/checkpoints/E1_patch_temporal/best.pth \
      --lora-ckpt   /workspace/trajgaze/TrajGazeMerge/checkpoints/E1_patch_temporal_keep10_bs4/best.pth \
      --tag E1_keep10_openend \
      --n-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/workspace/trajgaze")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.models.merge import gaze_weighted_merge
from TrajGazeMerge.models.model import (
    FRAME_SIZE,
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)
from TrajGazeMerge.training.train_merge_lora_temporal_no_kd import (
    load_traj_encoder, get_patch_scores_temporal, score_to_qwen_spatiotemporal,
)

RESULTS_DIR = "/workspace/trajgaze/TrajGazeMerge/eval_results"
_PREFIX_RE = re.compile(r"^\s*([A-D])\s*[\.\):]\s*")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type",    choices=["full", "gaze_only", "hand_only"], default="full")
    p.add_argument("--stage1-ckpt",   required=True)
    p.add_argument("--lora-ckpt",     required=True)
    p.add_argument("--merge-ratio",   type=float, default=0.9)
    p.add_argument("--tag",           default="E1_keep10_openend")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--n-frames",      type=int, default=128)
    p.add_argument("--n-traj-frames", type=int, default=128)
    p.add_argument("--n-vis-keyframes", type=int, default=16)
    p.add_argument("--split",         default="test", choices=["test", "train"])
    p.add_argument("--n-samples",     type=int, default=200,
                   help="number of samples to evaluate (stratified by task)")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--modes",         nargs="+", default=["mc", "open"], choices=["mc", "open"])
    return p.parse_args()


def strip_prefix(opt: str) -> str:
    return _PREFIX_RE.sub("", opt).strip()


def preprocess_open(processor, base_qwen, vlm_frame_paths, question, device):
    """Same as preprocess_item but with an open-ended prompt (no options shown)."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    frames = []
    for p in vlm_frame_paths:
        try:
            img = Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
            frames.append(img)
        except Exception:
            pass
    if not frames:
        return None

    user_text = (
        "You are watching a short first-person (egocentric) video clip.\n"
        f"Question: {question}\n\n"
        "Answer the question in a single concise sentence."
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "resized_height": FRAME_SIZE, "resized_width": FRAME_SIZE},
        {"type": "text",  "text": user_text},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       **video_kwargs, return_tensors="pt")
    emb_dev = base_qwen.get_input_embeddings().weight.device
    vis_dev = base_qwen.visual.patch_embed.proj.weight.device
    input_ids      = inputs["input_ids"].to(emb_dev)
    attention_mask = inputs["attention_mask"].to(emb_dev)
    pv_vid         = inputs["pixel_values_videos"].to(vis_dev, torch.bfloat16)
    grid_thw       = inputs["video_grid_thw"].to(vis_dev)
    with torch.no_grad():
        ve = base_qwen.model.get_video_features(pv_vid, grid_thw)
        if isinstance(ve, (tuple, list)):
            ve = torch.cat(ve, dim=0)
        video_embeds = ve.to(emb_dev)
        position_ids, rope_deltas = base_qwen.model.get_rope_index(
            input_ids=input_ids, video_grid_thw=grid_thw, attention_mask=attention_mask,
        )
    video_token_id  = base_qwen.config.video_token_id
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "grid_thw":        grid_thw,
        "video_embeds":    video_embeds,
        "video_positions": video_positions,
        "position_ids":    position_ids,
        "rope_deltas":     rope_deltas,
        "emb_dev":         emb_dev,
    }


@torch.no_grad()
def greedy_decode(qwen_model, processor, inputs_dict, max_new_tokens: int,
                  eos_id: int) -> str:
    """Manual greedy decode from inputs_embeds. No KV cache for simplicity
    (small max_new_tokens, ~200 samples → tolerable)."""
    inputs_embeds = inputs_dict["inputs_embeds"]
    attn_mask     = inputs_dict["attention_mask"]
    position_ids  = inputs_dict["position_ids"]
    base = qwen_model.get_base_model() if hasattr(qwen_model, "get_base_model") else qwen_model
    embed_layer = base.get_input_embeddings()
    device = inputs_embeds.device
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        out = qwen_model(
            inputs_embeds = inputs_embeds,
            attention_mask = attn_mask,
            position_ids   = position_ids,
            rope_deltas    = inputs_dict["rope_deltas"],
            use_cache      = False,
        )
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated_ids.append(next_id)
        if next_id == eos_id:
            break
        next_embed = embed_layer(torch.tensor([[next_id]], device=device))
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1
        )
        # Advance 3D position_ids by 1 along T axis (each of the 3 streams)
        last = position_ids[..., -1:]
        position_ids = torch.cat([position_ids, last + 1], dim=-1)

    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def stratified_sample(items: list, n: int, key_fn, seed: int = 42) -> list:
    """Sample up to n items, stratified roughly equally by key_fn."""
    import random
    rng = random.Random(seed)
    bucket: dict = {}
    for i, it in enumerate(items):
        bucket.setdefault(key_fn(it), []).append(i)
    per_bucket = max(1, n // max(1, len(bucket)))
    out: list = []
    for k, idxs in bucket.items():
        rng.shuffle(idxs)
        out.extend(idxs[:per_bucket])
    rng.shuffle(out)
    return sorted(out[:n])


def match_via_tfidf(generated: str, options: list[str]) -> tuple[int, list[float]]:
    """Return (argmax_idx, similarity_per_option)."""
    option_texts = [strip_prefix(o) for o in options]
    if not generated.strip():
        return 0, [0.0] * 4
    docs = option_texts + [generated]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit(docs)
    mat = vec.transform(docs)
    sims = cosine_similarity(mat[-1:], mat[:-1])[0]
    return int(np.argmax(sims)), sims.tolist()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = os.path.join(RESULTS_DIR, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[OpenEnded] tag={args.tag}  modes={args.modes}  n_samples={args.n_samples}")

    processor, qwen_model = load_qwen_lora(device)
    base_qwen = qwen_model.get_base_model()
    if os.path.exists(args.lora_ckpt):
        ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "lora_state" in ckpt:
            qwen_model.load_state_dict(ckpt["lora_state"], strict=False)
    qwen_model.eval()

    traj_encoder = load_traj_encoder(
        args.model_type, args.stage1_ckpt, device, args.n_vis_keyframes
    )
    if os.path.exists(args.lora_ckpt):
        merge_ckpt = torch.load(args.lora_ckpt, map_location=device, weights_only=False)
        if "encoder_state" in merge_ckpt:
            traj_encoder.load_state_dict(merge_ckpt["encoder_state"], strict=False)
    traj_encoder.eval()

    option_ids = get_option_ids(processor)
    eos_id = processor.tokenizer.eos_token_id

    ds = StreamGazeMergeDataset(
        split=args.split, n_vlm_frames=args.n_frames, n_traj_frames=args.n_traj_frames
    )
    print(f"  {args.split} items total: {len(ds)}")

    # Stratified subset
    selected = stratified_sample(ds.items, args.n_samples,
                                 key_fn=lambda x: x.get("task", "?"), seed=args.seed)
    print(f"  evaluating {len(selected)} stratified samples")

    rows: list[dict] = []
    with torch.no_grad():
        for j, idx in enumerate(selected):
            item = ds[idx]
            if item is None:
                continue
            try:
                # Shared TrajGaze score path
                scores_2d = get_patch_scores_temporal(traj_encoder, item, device)

                row = {
                    "idx": idx,
                    "task": item.get("task", "unknown"),
                    "dataset": item.get("dataset", "unknown"),
                    "question": item["question"][:200],
                    "answer": item["answer"],
                    "options": [strip_prefix(o) for o in item["options"]],
                }

                # MC mode (with options shown, generate to extract letter)
                if "mc" in args.modes:
                    cached = preprocess_item(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], item["options"], device,
                    )
                    if cached is not None:
                        n_video   = cached["video_embeds"].shape[0]
                        T_merged  = int(cached["grid_thw"][0, 0].item())
                        n_spatial = n_video // max(1, T_merged)
                        r         = max(1, int(args.merge_ratio * n_video))
                        scores_all = score_to_qwen_spatiotemporal(scores_2d, n_spatial, T_merged)
                        if scores_all.shape[0] != n_video:
                            scores_all = (
                                scores_all[:n_video] if scores_all.shape[0] > n_video
                                else scores_all.repeat(
                                    (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                                )[:n_video]
                            )
                        merged_video, receiver_idx = gaze_weighted_merge(
                            cached["video_embeds"], scores_all, r
                        )
                        inputs_dict = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)
                        # Logit baseline (for sanity vs mc-generation)
                        logits = forward_logits(qwen_model, inputs_dict)
                        pred_letter_logit = "ABCD"[int(logits[option_ids].argmax().item())]
                        # Generation
                        gen_text = greedy_decode(qwen_model, processor, inputs_dict, 8, eos_id)
                        m = re.search(r"([A-D])", gen_text)
                        pred_letter_gen = m.group(1) if m else ""
                        row.update({
                            "mc_pred_logit": pred_letter_logit,
                            "mc_correct_logit": bool(pred_letter_logit == item["answer"]),
                            "mc_gen_text": gen_text,
                            "mc_pred_gen": pred_letter_gen,
                            "mc_correct_gen": bool(pred_letter_gen == item["answer"]),
                        })

                # Open mode (no options shown, free-text + TF-IDF match)
                if "open" in args.modes:
                    cached_o = preprocess_open(
                        processor, base_qwen,
                        item["vlm_frame_paths"], item["question"], device,
                    )
                    if cached_o is not None:
                        n_video   = cached_o["video_embeds"].shape[0]
                        T_merged  = int(cached_o["grid_thw"][0, 0].item())
                        n_spatial = n_video // max(1, T_merged)
                        r         = max(1, int(args.merge_ratio * n_video))
                        scores_all = score_to_qwen_spatiotemporal(scores_2d, n_spatial, T_merged)
                        if scores_all.shape[0] != n_video:
                            scores_all = (
                                scores_all[:n_video] if scores_all.shape[0] > n_video
                                else scores_all.repeat(
                                    (n_video + scores_all.shape[0] - 1) // scores_all.shape[0]
                                )[:n_video]
                            )
                        merged_video, receiver_idx = gaze_weighted_merge(
                            cached_o["video_embeds"], scores_all, r
                        )
                        inputs_dict = build_merged_inputs(base_qwen, cached_o, merged_video, receiver_idx)
                        gen_text = greedy_decode(
                            qwen_model, processor, inputs_dict, args.max_new_tokens, eos_id
                        )
                        match_idx, sims = match_via_tfidf(gen_text, item["options"])
                        pred_letter_open = "ABCD"[match_idx]
                        row.update({
                            "open_gen_text": gen_text,
                            "open_pred": pred_letter_open,
                            "open_correct": bool(pred_letter_open == item["answer"]),
                            "open_sim_A": float(sims[0]),
                            "open_sim_B": float(sims[1]),
                            "open_sim_C": float(sims[2]),
                            "open_sim_D": float(sims[3]),
                            "open_top_sim": float(max(sims)),
                        })

                rows.append(row)
                if (j + 1) % 10 == 0:
                    if "mc" in args.modes and "open" in args.modes:
                        mc_a = 100*sum(r.get("mc_correct_logit", False) for r in rows) / max(1, len(rows))
                        op_a = 100*sum(r.get("open_correct", False) for r in rows) / max(1, len(rows))
                        print(f"  [{j+1}/{len(selected)}] mc={mc_a:.1f}%  open={op_a:.1f}%")
                    else:
                        print(f"  [{j+1}/{len(selected)}] processed")

            except Exception:
                traceback.print_exc()
                continue

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(out_dir, f"{args.tag}_open_ended_per_sample.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = parquet_path.replace(".parquet", ".jsonl")
        df.to_json(parquet_path, orient="records", lines=True)
    print(f"  saved {len(df)} -> {parquet_path}")

    # Summary
    summary = {"tag": args.tag, "n_samples": int(len(df)), "modes": args.modes}
    if "mc" in args.modes:
        summary["mc"] = {
            "acc_logit": float(100 * df["mc_correct_logit"].mean()),
            "acc_gen":   float(100 * df["mc_correct_gen"].mean()),
            "logit_gen_match_rate": float(100 * (df["mc_pred_logit"] == df["mc_pred_gen"]).mean()),
        }
    if "open" in args.modes:
        summary["open"] = {
            "acc": float(100 * df["open_correct"].mean()),
            "mean_top_sim": float(df["open_top_sim"].mean()),
            "per_task": (
                df.groupby("task")["open_correct"].agg(["mean", "count"]).reset_index()
                  .rename(columns={"mean": "accuracy", "count": "n"})
                  .assign(accuracy=lambda d: d["accuracy"] * 100)
                  .to_dict(orient="records")
            ),
        }
    if "mc" in args.modes and "open" in args.modes:
        summary["delta_mc_vs_open"] = summary["mc"]["acc_logit"] - summary["open"]["acc"]
    summary_path = os.path.join(out_dir, f"{args.tag}_open_ended_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\n  Open-ended Eval [{args.tag}]\n{'='*60}")
    if "mc" in args.modes:
        print(f"  MC (logit):      {summary['mc']['acc_logit']:.2f}%")
        print(f"  MC (generation): {summary['mc']['acc_gen']:.2f}%")
        print(f"  logit/gen match: {summary['mc']['logit_gen_match_rate']:.1f}%")
    if "open" in args.modes:
        print(f"  Open (no options, TF-IDF match): {summary['open']['acc']:.2f}%  (random=25%)")
        print(f"  mean top similarity: {summary['open']['mean_top_sim']:.3f}")
    if "mc" in args.modes and "open" in args.modes:
        print(f"  Δ (MC - Open):   {summary['delta_mc_vs_open']:+.2f}pp")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
