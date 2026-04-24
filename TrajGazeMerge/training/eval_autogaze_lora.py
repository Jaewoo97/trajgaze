"""
Standalone eval: AutoGaze LoRA checkpoint on full EGTEA MCQ val set.
Usage:
    CUDA_VISIBLE_DEVICES=0 python -m TrajGazeMerge.training.eval_autogaze_lora \
        --ckpt /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/autogaze_lora/epoch_01.pth \
        --out  /workspace/EgoGazeVQA/TrajGazeMerge/eval_results/autogaze_lora_epoch01_metrics.json
"""
import argparse, json, sys, traceback
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/workspace/EgoGazeVQA")
sys.path.insert(0, "/workspace/EgoGazeVQA/AutoGaze")

from TrajGazeMerge.training.train_autogaze_lora import (
    StreamGazeSimpleDataset, _sample_paths,
    load_autogaze, compute_ag_scores,
    N_AG_FRAMES, FRAME_SIZE, AUTOGAZE_CKPT,
)
from TrajGazeMerge.models.model import (
    load_qwen_lora, get_option_ids, preprocess_item, forward_logits, build_merged_inputs
)
from TrajGazeMerge.models.merge import gaze_weighted_merge

MCQ_TASKS = [
    "past_gaze_sequence_matching",
    "past_non_fixated_object_identification",
    "past_object_transition_prediction",
    "past_scene_recall",
    "present_future_action_prediction",
    "present_object_attribute_recognition",
    "present_object_identification_easy",
    "present_object_identification_hard",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--autogaze-ckpt", default=AUTOGAZE_CKPT)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0")

    print("Loading AutoGaze ...")
    ag_transform, ag_model = load_autogaze(device)

    print("Loading Qwen2.5-VL-7B + LoRA ...")
    processor, model = load_qwen_lora(device)
    base_qwen = model.get_base_model()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

    option_ids = get_option_ids(processor)
    model.eval()

    test_ds = StreamGazeSimpleDataset(split="test", n_vlm_frames=128)
    print(f"Evaluating {len(test_ds)} items ...")

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

                n_video   = cached["video_embeds"].shape[0]
                T_merged  = int(cached["grid_thw"][0, 0].item())
                n_spatial = n_video // max(1, T_merged)

                ag_paths  = _sample_paths(item["vlm_frame_paths"], N_AG_FRAMES)
                ag_frames = [
                    Image.open(p).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
                    for p in ag_paths
                ]
                scores_all = compute_ag_scores(
                    ag_frames, ag_model, ag_transform, T_merged, n_spatial
                )
                r = max(1, int(0.90 * n_video))
                merged_video, receiver_idx = gaze_weighted_merge(
                    cached["video_embeds"], scores_all.to(device), r
                )
                inputs_dict = build_merged_inputs(base_qwen, cached, merged_video, receiver_idx)

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

    acc = 100.0 * correct / max(1, total)
    per_task = {t: 100.0 * sum(v) / max(1, len(v)) for t, v in sorted(by_task.items())}

    print(f"\n=== Overall: {acc:.2f}%  (n={total}) ===")
    for t, a in per_task.items():
        n = len(by_task[t])
        print(f"  {t}: {a:.2f}%  (n={n})")

    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"acc": acc, "n": total, "per_task": per_task,
                   "ckpt": args.ckpt}, f, indent=2)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
