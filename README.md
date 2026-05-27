# TrajGazeMerge — Trajectory-Aware Visual Token Selection for Egocentric Video QA

Stage-1 trajectory encoder + Stage-2 LoRA-finetuned Qwen2.5-VL-7B pipeline that compresses egocentric visual tokens **10×** while keeping the kept tokens task-relevant. Trained and evaluated on **StreamGaze**, **EgoGazeVQA**, and **HD-EPIC**.

> Paper draft: `docs/NeurIPS_2026_Gaze_hand_Trajectory_Merging_for_Efficient_Egocentric_Video_Understanding.pdf`
> Latest project status: [`docs/current_state.md`](docs/current_state.md)
> Latest narrative: [`docs/paper_narrative_v3.md`](docs/paper_narrative_v3.md)

---

## 1. Best result

**Setup A — 2 datasets (StreamGaze + EgoGazeVQA)** *(headline)*

| Method | StreamGaze | EgoGazeVQA | **mean** |
|---|---:|---:|---:|
| Sprint-1 baseline | 65.21 | 57.31 | 61.13 |
| **TAS-only** ★ | **67.49** | 57.77 | **62.63** |
| TAS+ATR | 64.26 | ~58 | ~61 |
| TAS+ATR+CGM (FULL) | 61.98 | **59.40** | 60.69 |

Best checkpoint: `TrajGazeMerge/checkpoints/E1_combined_TAS_only/best.pth`
StreamGaze gain over baseline: **+2.28 pp** mean, **+20 pp** on `past_gaze_sequence_matching`.

**Setup B — 3 datasets (+ HD-EPIC)**

| Method | StreamGaze | EgoGazeVQA | HD-EPIC | **mean** |
|---|---:|---:|---:|---:|
| **TAS-only-hdepic** ★ | 63.69 | 55.92 | 50.12 | **56.57** |
| TAS+ATR-hdepic | 60.65 | 54.76 | 50.66 | 55.35 |

Best checkpoint: `TrajGazeMerge/checkpoints/E1_combined_TASonly_hdepic_bs8_mb2/best.pth`

TAS-only is best in **both** setups. ATR/CGM extensions hurt mean accuracy; they are reported as honest ablations rather than headline methods. See `docs/paper_narrative_v3.md` for the full mechanism-by-role reading.

---

## 2. Environment

Single conda env at `/opt/conda/envs/gaze`. Key dependencies (already installed on host):
- PyTorch 2.x + CUDA
- `transformers`, `peft` (LoRA)
- `Qwen2.5-VL-7B-Instruct` base model at `/home/irteam/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/...` (see `TrajGazeMerge/models/model.py:29`)
- Datasets on disk under `/workspace/datasets/...` and project-local paths (see `TrajGazeMerge/data/`)

```bash
# Verify env
/opt/conda/envs/gaze/bin/python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. Datasets — train / val split

| Dataset | Train (n) | Train domain | Val (n) | Val domain |
|---|---:|---|---:|---|
| StreamGaze | 5,799 | egoexolearn + holoassist | 526 | egtea |
| EgoGazeVQA | 647 | ego4d + egoexo | 431 | egtea |
| HD-EPIC | 22,551 | participants P01–P08 | 3,899 | participant P09 |

- StreamGaze & EgoGazeVQA: **cross-domain** split (train ≠ val video source).
- HD-EPIC: **participant-level** split (P09 held out).
- All val sets are evaluated separately by the trainer.

---

## 4. Training — reproduce the best model

### 4a. TAS-only on Setup A (headline)

```bash
cd /workspace/trajgaze
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/conda/envs/gaze/bin/python \
  -m TrajGazeMerge.training.train_merge_lora_batched \
    --model-type full \
    --stage1-ckpt TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth \
    --output-dir TrajGazeMerge/checkpoints/E1_combined_TAS_only \
    --epochs 3 --merge-ratio 0.9 \
    --micro-batch 2 --grad-accum 4 \
    --use-egovqa \
    --eval-egovqa-egtea \
    --dataloader-num-workers 8 --eval-every 400
```

Trainable: **LoRA (r=16, α=32) on `q/k/v/o_proj` + TAS encoder fine-tune**. Trainable params ≈ 10M (0.12 % of 8.3B Qwen).

Effective batch = `micro_batch × grad_accum = 8`. Roughly **12–14 h** on a single GPU.

### 4b. TAS-only on Setup B (+HD-EPIC)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/conda/envs/gaze/bin/python \
  -m TrajGazeMerge.training.train_merge_lora_batched \
    --model-type full \
    --stage1-ckpt TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth \
    --output-dir TrajGazeMerge/checkpoints/E1_combined_TASonly_hdepic_bs8_mb2 \
    --epochs 3 --merge-ratio 0.9 \
    --micro-batch 2 --grad-accum 4 \
    --use-egovqa --use-hd-epic \
    --eval-egovqa-egtea --eval-hd-epic \
    --dataloader-num-workers 8 --eval-every 400
```

### 4c. ATR / CGM / cf-mask ablations

Append the appropriate flags to the Setup B command:

| Method | Flags |
|---|---|
| TAS + ATR | `--use-atr --atr-lambda 0.5` |
| TAS + CGM | `--cgm-aug --cgm-lambda 0.3 --cgm-prob 0.3 --cgm-radius 0.2 --cgm-margin 0.5 --cgm-warmup-steps 600` |
| TAS + cf-mask (Direction A) | `--use-cf-mask --cf-mask-prob 0.3 --cf-mask-margin 1.0 --cf-mask-lambda 0.3 --cf-mask-warmup-steps 600` |
| TAS + shuffle-margin | `--shuffle-aug --shuffle-prob 0.3 --shuffle-margin 1.0 --shuffle-lambda 0.3 --shuffle-warmup-steps 600` |

Combined examples are in `TrajGazeMerge/training/run_*.sh`.

---

## 5. Evaluation — counterfactual mask (cf-mask)

Diagnostic that measures whether the LLM actually consumes the kept visual tokens.

```bash
cd /workspace/trajgaze
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /opt/conda/envs/gaze/bin/python \
  -m TrajGazeMerge.eval.counterfactual_mask_eval \
    --stage1-ckpt TrajGaze_v2/checkpoints/E1_combined_AB_TAS/best.pth \
    --lora-ckpt   TrajGazeMerge/checkpoints/E1_combined_TAS_only/best.pth \
    --val-dataset streamgaze \
    --tag         E1_combined_TAS_only_streamgaze_cfmask
```

Repeat with `--val-dataset egovqa`. Variants reported per run: `baseline`, `mask_kept` (= language-only floor), `mask_kept_early/late`, `shuffle_kept`, `mask_gaze`, `mask_hand`. See [`docs/current_state.md`](docs/current_state.md) §0.2 for what each variant tests.

### Key reading

For the TAS-only ★ checkpoint, expected cf-mask Δ on StreamGaze: `mask_kept = −11.98 pp` (LLM strongly uses kept tokens). On EgoGazeVQA: `mask_kept = +0.93 pp` (dataset is largely solvable from gaze metadata; documented as §Limitations).

---

## 6. Repository layout

```
TrajGazeMerge/
  data/                # StreamGaze, EgoGazeVQA, HD-EPIC, combined loader
  models/              # Qwen-VL + LoRA wiring, batched preprocess
                       # trajectory_grounding.py = ATR head + CGM helpers
  training/
    train_merge_lora_batched.py   # main Stage-2 trainer
    run_*.sh                       # launch scripts per ablation
  eval/
    counterfactual_mask_eval.py    # cf-mask diagnostic
    convergence_watcher*.sh        # auto-stop training on plateau
    run_*_diagnosis.sh             # cf-mask eval orchestrators
TrajGaze_v2/           # Stage-1 trajectory encoder (DINOv2 + transformer)
  models/model_temporal.py
  training/stage1_temporal.py
docs/                  # status docs, paper draft, diagnosis reports
```

---

## 7. Documentation index

| Read | When |
|---|---|
| [`docs/current_state.md`](docs/current_state.md) | one-page status, full result tables, methodology glossary |
| [`docs/paper_narrative_v3.md`](docs/paper_narrative_v3.md) | mechanism-by-role paper reframe (TAS = headline) |
| [`docs/visual_grounding_diagnosis_v2.md`](docs/visual_grounding_diagnosis_v2.md) | cf-mask 4×2 matrix + decision gate |
| [`docs/codebase_overview.md`](docs/codebase_overview.md) | architecture details |
| [`docs/journey_summary.md`](docs/journey_summary.md) | 8-day diagnosis-to-narrative timeline (history) |

Plans (out-of-tree, under `~/.claude/plans/`):
- `zazzy-sprouting-ladybug.md` — Step 1 diagnosis plan (completed).
- `cf-mask-augmented-training.md` — Direction A plan (in flight, CF-1 / CF-3 runs).

---

## 8. Active experiments (as of 2026-05-27)

- **CF-1** (CE + cf-mask margin only) — GPU 0 — `TrajGazeMerge/checkpoints/E1_combined_cf1_hdepic_bs8_mb2/`
- **CF-3** (CE + cf-mask margin + shuffle margin) — GPU 1 — `TrajGazeMerge/checkpoints/E1_combined_cf3_hdepic_bs8_mb2/`

Convergence watcher: `TrajGazeMerge/eval/convergence_watcher_cf.sh`
cf-mask post-eval: `TrajGazeMerge/eval/run_cf_diagnosis.sh`
Decision gate criteria: see `cf-mask-augmented-training.md` §4.2.

---

## 9. Acknowledgements

EgoGazeVQA benchmark used for evaluation: [arXiv:2509.07447](https://arxiv.org/abs/2509.07447) (Peng et al., NeurIPS D&B 2025). Stage-2 LoRA base model: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
