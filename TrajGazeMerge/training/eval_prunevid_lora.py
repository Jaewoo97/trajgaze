"""
PruneVid-inspired eval: apply inference-time question-guided visual token pruning
to the baseline LoRA checkpoint (trained on StreamGaze train set with full tokens).

Adaptation for Qwen2.5-VL (no CLS token, FlashAttn2 in ViT):
  - Score visual tokens by dot-product attention between text token embeddings
    (from LLM embedding layer) and video_embeds — question-guided importance
    without a second forward pass.
  - Hard-drop the bottom (1 - KEEP_RATIO) tokens.

This captures PruneVid's Stage-2 principle (question-guided visual token selection)
adapted for Qwen's architecture.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m TrajGazeMerge.training.eval_prunevid_lora \
        --ckpt .../baseline_lora/best.pth \
        --out  .../eval_results/prunevid_baseline_lora_metrics.json
"""

import argparse, json, os, sys, traceback
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/EgoGazeVQA")

from TrajGazeMerge.training.train_autogaze_lora import StreamGazeSimpleDataset
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item,
    build_merged_inputs, forward_logits,
)

KEEP_RATIO   = 0.10
BASELINE_CKPT = "/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora/best.pth"


def compute_prunevid_scores(
    base_qwen,
    cached: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Score visual tokens by text-guided dot-product attention.

    Uses LLM embedding-space representations of both text and video tokens.
    Text token embeddings are mean-pooled to form a query vector; visual tokens
    are scored by their dot product with this query (question-guided importance).

    Returns:
        scores : (N_video,) float on device
    """
    input_ids    = cached["input_ids"]       # (1, L)
    video_embeds = cached["video_embeds"]    # (N_video, d)

    video_token_id = base_qwen.config.video_token_id
    is_video = (input_ids[0] == video_token_id)          # (L,) bool
    is_text  = ~is_video                                  # (L,) bool — includes special tokens

    # Text token embeddings from LLM embedding layer
    text_ids = input_ids[:, is_text]                     # (1, n_text)
    with torch.no_grad():
        text_embeds = base_qwen.get_input_embeddings()(
            text_ids.to(device)
        ).squeeze(0).float()                             # (n_text, d)

    # Mean-pool text embeddings → query vector
    query = text_embeds.mean(dim=0)                      # (d,)
    query = F.normalize(query, dim=-1)

    # Score each video token by cosine similarity with query
    vf = F.normalize(video_embeds.float(), dim=-1)       # (N_video, d)
    scores = vf @ query                                  # (N_video,)

    return scores


def build_prunevid_inputs(base_qwen, cached: dict, device: torch.device) -> dict:
    """Keep top KEEP_RATIO visual tokens by question-guided score, drop the rest."""
    video_embeds = cached["video_embeds"]
    N_video = video_embeds.shape[0]
    n_keep  = max(1, int(KEEP_RATIO * N_video))
    r       = N_video - n_keep

    scores       = compute_prunevid_scores(base_qwen, cached, device)
    receiver_idx = torch.topk(scores, n_keep, largest=True).indices  # (n_keep,)
    receiver_idx, _ = receiver_idx.sort()

    kept_embeds = video_embeds[receiver_idx]             # (n_keep, d) — hard drop

    return build_merged_inputs(base_qwen, cached, kept_embeds, receiver_idx)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=BASELINE_CKPT)
    p.add_argument("--out",  required=True)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0")

    print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen = model.get_base_model()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt    = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state   = ckpt.get("lora_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

    option_ids = get_option_ids(processor)
    model.eval()

    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    print(f"Evaluating {len(test_ds)} items with PruneVid-style pruning (keep={KEEP_RATIO*100:.0f}%) ...")

    correct = 0
    total   = 0
    by_task: dict[str, list] = {}

    with torch.no_grad():
        for i, item in enumerate(test_ds):
            if item is None:
                continue
            try:
                cached = preprocess_item(
                    processor, base_qwen,
                    item["vlm_frame_paths"], item["question"], item["options"], device
                )
                if cached is None:
                    continue

                inputs_dict   = build_prunevid_inputs(base_qwen, cached, device)
                logits        = forward_logits(model, inputs_dict)
                option_logits = logits[option_ids]
                pred_idx      = option_logits.argmax().item()
                gt_idx        = ["A", "B", "C", "D"].index(item["answer"])
                ok = int(pred_idx == gt_idx)
                correct += ok
                total   += 1
                by_task.setdefault(item["task"], []).append(ok)

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(test_ds)}] acc={100.*correct/max(1,total):.2f}%")

            except Exception:
                traceback.print_exc()
                continue

    acc      = 100.0 * correct / max(1, total)
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}

    print(f"\n=== PruneVid-style (10% keep, question-guided drop) ===")
    print(f"Overall: {acc:.2f}%  (n={total})")
    for t, a in per_task.items():
        print(f"  {t}: {a:.2f}%  (n={len(by_task[t])})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"acc": acc, "n": total, "per_task": per_task,
                   "ckpt": args.ckpt, "keep_ratio": KEEP_RATIO,
                   "method": "prunevid_question_guided_drop"}, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
