d# TrajGazeMerge — Implementation Log

## Dataset Split
- **Train**: egoexolearn + holoassist (across 8 MCQ tasks)
- **Test**:  egtea (526 items across 8 MCQ tasks)
- **Tasks** (8 MCQ, proactive excluded):
  - past_gaze_sequence_matching
  - past_non_fixated_object_identification
  - past_object_transition_prediction
  - past_scene_recall
  - present_future_action_prediction
  - present_object_attribute_recognition
  - present_object_identification_easy
  - present_object_identification_hard

## File Structure

```
TrajGazeMerge/
├── PROPOSAL.md               — method design & baselines
├── IMPLEMENTATION.md         — this file
├── data/
│   └── dataset.py            — StreamGazeMergeDataset (train=egoexolearn+holoassist, test=egtea)
├── models/
│   ├── merge.py              — gaze_weighted_merge(), score_to_qwen_spatial()
│   └── model.py              — load_qwen_lora(), preprocess_item(), build_merged_inputs(), etc.
├── training/
│   ├── train_baseline_lora.py   — Qwen + LoRA, full tokens, CE loss
│   ├── train_merge_lora.py      — TrajGaze encoder + Qwen LoRA, 10% merged tokens, KL+CE
│   │                              Teacher: fine-tuned baseline LoRA (not base pretrained)
│   └── train_autogaze_lora.py   — AutoGaze token selector + Qwen LoRA, ~10% tokens, CE loss
├── eval/
│   └── evaluate.py              — 4 conditions: baseline_frozen/lora, merge_frozen/lora
└── logs/
    └── autogaze_lora.log        — AutoGaze+LoRA training log
```

## Task 1: Baseline LoRA Fine-tuning

**Model**: Qwen2.5-VL-7B-Instruct + LoRA
**LoRA config**: r=16, alpha=32, dropout=0.05, targets=q_proj/k_proj/v_proj/o_proj (LLM only)
**Visual input**: 128 frames at 224×224, 100% visual tokens
**Loss**: CrossEntropy over 4 MCQ option logits at last prompt position
**Optimizer**: AdamW(lr=1e-4, wd=1e-4), grad_accum=4
**GPUs**: 2 (torchrun DDP)
**Train**: egoexolearn
**Eval**: egtea (periodic)

Launch command:
```bash
torchrun --nproc_per_node=2 -m TrajGazeMerge.training.train_baseline_lora \
    --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/baseline_lora \
    --epochs 3 --lr 1e-4 --grad-accum 4
```

## Task 2: TrajGazeMerge Fine-tuning

**Models**: TrajGaze encoder (init from stage1_v3/best.pth) + Qwen2.5-VL-7B + LoRA
**LoRA config**: same as baseline
**Merge**: 90% of visual tokens merged away (keep 10%) using gaze-weighted bipartite matching
**Score interpolation**: TrajGaze 14×14 → Qwen 8×8 via (nearest 14→16) + (avg_pool 16→8)
**Loss**:
  - Teacher: **fine-tuned baseline LoRA** + full tokens → logits_teacher
  - Student: Qwen + LoRA + merged tokens → logits_student
  - L = 0.5 * KL(student || teacher) + 0.5 * CE(student, label)
**Optimizer**: AdamW, lr_lora=1e-4, lr_enc=1e-5, grad_accum=4
**Gradient flow**: loss → LoRA weights + (loss → merged_tokens → merge op → patch_scores → encoder)
**GPUs**: 2 (torchrun DDP, GPU 0+1, launched after baseline LoRA finishes)

Launch command:
```bash
bash /workspace/EgoGazeVQA/TrajGazeMerge/launch_merge_after_baseline.sh
# (already running as watcher process PID 844942)
```

## Task 3: AutoGaze + LoRA Fine-tuning

**Models**: AutoGaze (frozen, streamgaze_fold_c_ntp checkpoint) + Qwen2.5-VL-7B + LoRA
**LoRA config**: same as baseline
**Token selection**: AutoGaze selects ~10% of visual tokens (gazing_ratio=0.10)
**Loss**: CE over 4 option logits (same as baseline, but with filtered tokens)
**Optimizer**: AdamW, lr=1e-4, grad_accum=4
**GPUs**: 2 (torchrun DDP, GPU 2+3, running)

Launch command:
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 \
    -m TrajGazeMerge.training.train_autogaze_lora \
    --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/autogaze_lora \
    --epochs 3 --lr 1e-4 --grad-accum 4
```

## Evaluation

Four conditions, all on egtea test split:

| Condition | Qwen weights | Visual tokens | Notes |
|---|---|---|---|
| baseline_frozen | frozen (no LoRA) | 100% | Zero-shot upper bound |
| baseline_lora   | LoRA fine-tuned  | 100% | Trained baseline |
| merge_frozen    | frozen (no LoRA) | 50% merged | Zero-shot TrajGazeMerge |
| merge_lora      | LoRA fine-tuned  | 50% merged | Full TrajGazeMerge |

Evaluation command:
```bash
/opt/conda/envs/gaze/bin/python -m TrajGazeMerge.eval.evaluate \
    --condition <condition> --gpu <gpu> \
    [--stage1-ckpt ...] [--lora-ckpt ...]
```

Results saved to: `/workspace/EgoGazeVQA/TrajGazeMerge/eval_results/`

## Key Implementation Choices

1. **Merge location**: After frozen ViT, before LLM. No ViT modification.
2. **Score interpolation**: 14×14 → 8×8 matching stage2.py pipeline (avoids mismatch).
3. **Sequence modification**: Remove source video token positions from input_ids, attention_mask, position_ids (same approach as qwen_generate_with_mask in stage2.py).
4. **Teacher pass**: `qwen_model.disable_adapter()` — same base weights, no additional memory for a second model copy.
5. **Gradient through merge**: `gaze_weighted_merge` uses `scatter_add_` on `numerator = receivers * w_r + Σ sources * w_s`. Gradient flows through `w_r`, `w_s` (= `patch_scores`) to encoder. Argmax (best_match) is not differentiable — standard in ToMe.
6. **DDP**: `find_unused_parameters=True` because LoRA leaves some base model params with `requires_grad=False`, and TrajGaze encoder's DINOv2 backbone is frozen.

## Training Status

| Job | GPU | Status |
|---|---|---|
| Baseline LoRA | 0,1 | running (epoch 2/3, ~2h remaining) |
| TrajGazeMerge LoRA | 0,1 | queued (watcher PID 844942, launches when baseline finishes) |
| AutoGaze LoRA | 2,3 | running |

## Key Implementation Choices (Updated)

7. **Teacher for TrajGazeMerge**: Loaded fine-tuned baseline LoRA checkpoint (not base pretrained). Teacher is a separate frozen model instance, no `disable_adapter()`.
8. **Merge ratio**: 0.9 (remove 90% of tokens → keep 10% budget).
9. **AutoGaze integration**: Processes 16-frame subset of VLM frames, OR-union across 4 attention scales, then projects to Qwen's 8×8 spatial grid. Actual keep rate ≈ 50-55% after scale union.
10. **AutoGaze dataset**: `StreamGazeSimpleDataset` (no traj loading) for efficiency.
