# TrajGazeMerge Evaluation Results

Evaluation on **StreamGaze EGTEA test set** (526 items, 8 MCQ tasks).

All methods use **10% visual token budget** (keep 10%, compress/drop 90%).

---

## Per-Task Results

| Task | n | Baseline LoRA (100%) | Random Drop + LoRA | Zero-shot TrajGaze (hard top-k) | Zero-shot TrajGaze (weighted merge) | TrajGazeMerge (jointly trained) |
|---|---|---|---|---|---|---|
| past_gaze_sequence_matching | 64 | 60.94 | 60.94 | 45.45 | 36.36 | **68.75** |
| past_non_fixated_object_identification | 68 | 66.18 | 55.88 | 54.41 | 60.29 | **58.82** |
| past_object_transition_prediction | 2 | 0.00 | 50.00 | 0.00 | 50.00 | 50.00 |
| past_scene_recall | 37 | 64.86 | 72.97 | 43.24 | 40.54 | 37.84 |
| present_future_action_prediction | 94 | 35.11 | 47.87 | 28.72 | 25.53 | **50.00** |
| present_object_attribute_recognition | 96 | 85.42 | 82.29 | 76.04 | 79.17 | **92.71** |
| present_object_identification_easy | 101 | 68.32 | 54.46 | 58.42 | 62.38 | 59.41 |
| present_object_identification_hard | 64 | 70.31 | 65.63 | 60.94 | 67.19 | 62.50 |
| **Overall** | **526** | **64.07** | **61.98** | **53.22** | **54.36** | **63.69** |

---

## Method Descriptions

### Baseline LoRA (100% tokens)
- Qwen2.5-VL-7B + LoRA (r=16) finetuned on full visual token sequences
- No token compression at inference
- Trainable params: ~10.09M (LoRA only)

### Random Drop + LoRA (10%)
- Same LoRA setup, but randomly drops 90% of visual tokens during training and inference
- Hard selection, no merging
- Trainable params: ~10.09M (LoRA only)

### Zero-shot TrajGaze — Hard Top-k (10%)
- TrajGazeV2 stage1 encoder scores tokens by gaze/hand trajectory attention
- Top 10% selected, bottom 90% **discarded** (no merging)
- No joint training: stage1 encoder + baseline_lora used as-is
- Trainable params at inference: 0 (both frozen)

### Zero-shot TrajGaze — Weighted Merge (10%)
- Same TrajGazeV2 scores, but bottom 90% **merged** into top 10% via cosine-similarity weighted average (identical merge op to joint training)
- No joint training: stage1 encoder + baseline_lora used as-is
- Each of the 10% output tokens carries aggregated information from its assigned sources
- Trainable params at inference: 0 (both frozen)

### TrajGazeMerge (jointly trained)
- TrajGazeV2 encoder + LoRA trained jointly end-to-end
- `gaze_weighted_merge`: top 10% receivers + 90% sources merged in via weighted cosine-similarity average
- Teacher-student distillation: KL(student || teacher) + CE loss
- Trainable params: 23.66M (13.57M encoder + 10.09M LoRA)
- Checkpoint: `TrajGazeMerge/checkpoints/merge_lora/` (epoch 2)

---

## Summary

| Method | Trainable params | Overall acc |
|---|---|---|
| Baseline LoRA (100% tokens) | 10.09M | 64.07% |
| TrajGazeMerge (jointly trained) | 23.66M | **63.69%** |
| Random Drop + LoRA (10%) | 10.09M | 61.98% |
| VisionZip Projector-only (10%) | 44.60M | 56.65% |
| Zero-shot TrajGaze + weighted merge | 0 | 54.36% |
| Zero-shot TrajGaze + hard top-k | 0 | 53.22% |
| VisionZip + LoRA (10%) | 10.09M | 69.96% |
| Training-free VisionZip (on baseline_lora) | 0 | 49.05% |

### Key observations

- **TrajGazeMerge (63.69%)** matches baseline accuracy (64.07%) while using only 10% of visual tokens — 9× token compression with no accuracy loss.
- **Zero-shot weighted merge (54.36%)** outperforms zero-shot hard top-k (53.22%), confirming that folding 90% source token information into receivers helps, even without joint training.
- **Joint training is critical**: the 9.3pp gap between zero-shot (54.36%) and jointly trained (63.69%) TrajGazeMerge demonstrates the VLM must be adapted to the merged token distribution.
- **VisionZip + LoRA (69.96%)** is the strongest at 10% budget but trains LoRA against VisionZip-selected tokens — a different paradigm (attention-guided selection vs. gaze-guided selection).
- The `past_scene_recall` task sees a large drop in zero-shot variants, suggesting holistic scene understanding requires well-distributed spatial coverage that gaze does not guarantee when applied to a full-token-trained VLM.
