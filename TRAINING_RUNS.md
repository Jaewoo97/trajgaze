# Training runs — 10%/13%-budget gaze-token bake-off

Per-model launch commands for the current "ours method" comparison line: every
approach that selects ~10% (or 13%) of Qwen2.5-VL's visual tokens and trains a
LoRA on the 2-way egtea protocol. One launch script per model lives in `scripts/`;
the exact command is reproduced here too so this file is self-contained.

Earlier research families (PLLaVA, PruneVid, QC-Gate, temporal-budget, random,
gaze-only/hand-only) are **not** listed here — those trainers still exist under
`TrajGazeMerge/training/` but are off the current comparison line.

---

## Shared setup (prepend to every command below)

```bash
cd /workspace/trajgaze_st
export PATH="/opt/conda/envs/trajgaze/bin:$PATH"   # conda env `trajgaze`; the `gaze` env has a broken transformers import
export GAZE_OVERLAY=1                              # gaze-overlay input frames (required for SG+EG)
S1=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth   # frozen TAS Stage-1 encoder
```

**Protocol shared by all rows:** data = StreamGaze train 5799 + EgoGazeVQA train
1265 (2-way, `--no-hdepic`); eval = egtea 2-way **n=1011** every epoch; 3 epochs,
LoRA lr 1e-4, **eff-batch 8**, early-stop after epoch 2 if epoch-2 val ≤ epoch-1.
`best.pth` = best per-epoch egtea val.

---

## Master table

| Model | Budget | Composition (raw / merged + gaze) | Trainer module | Launch script | egtea 2-way |
|---|---|---|---|---|---|
| VisionZip (plain) | 10% | 5 raw + 5 merged | `train_visionzip_lora` | `launch_vz_sgeg_overlay.sh` | 62.51 |
| VZ+traj | 10% | 5/5, attn×traj re-rank | `train_visionzip_traj_lora_3way` | `launch_vztraj_sgeg_overlay.sh` | 62.71 |
| TAS (Stage-2 merge) | 10% | full gaze-weighted merge | `train_merge_lora_tas_3way` | `launch_tas_sgeg_overlay.sh` | 59.64 |
| **M1 — VZ-complement (learned top-k)** | 10% | 6.5 raw + 3.5 merged (3.5/3.5 + 3% gaze) | `train_visionzip_complement_lora` | `launch_vzcomp_learned_overlay.sh` | **63.01 (best)** |
| M2 — VZ-complement (anticipatory) | 10% | 3.5/3.5 + 3% gaze (anticipatory pool) | `train_visionzip_complement_lora` | `launch_vzcomp_anticip_overlay.sh` | archived |
| Scanpath (ours) | 10% + K=8 | M1 + 8 side-channel intent tokens | `train_visionzip_scanpath_lora` | `launch_ours_scanpath.sh` | 63.01 (tie, gated off) |
| Gaze-tag (ours) | 10% | M1 + per-token gaze tag | `train_visionzip_gazetag_lora` | `launch_ours_gazetag.sh` | 61.62 |
| Coverage-A (ours) | 10% | M1 7/3, NMS de-cluster | `train_visionzip_complement_lora` | `launch_ours_coverage_a.sh` | 61.82 |
| Coverage-B (ours) | 10% | M1 6/4, NMS de-cluster | `train_visionzip_complement_lora` | `launch_ours_coverage_b.sh` | 61.42 |
| Fusion (ours) | 10% | M1 soft attn×traj fusion | `train_visionzip_complement_lora` | `launch_ours_fusion.sh` | archived (falsified) |
| **M1+13% (decoupling)** | 13% | 8 raw + 5 merged (10% content ∪ 3% gaze) | `train_visionzip_complement_lora` | `launch_decouple_m1plus_13.sh` | 62.81 |
| VZ-(8/5) (decoupling ctrl) | 13% | 8 raw + 5 merged, no gaze | `train_visionzip_lora` | `launch_decouple_vz_8_5.sh` | 60.63 |
| VZ-(6.5/3.5) (decoupling diag) | 10% | 6.5 raw + 3.5 merged, no gaze | `train_visionzip_lora` | `launch_decouple_vz_6p5_3p5.sh` | 61.62 |

---

## Comparison set

### VisionZip (plain) — 62.51 · GPU 0,1
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29651 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_sgeg_overlay \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop
```

### VZ+traj — 62.71 · GPU 0,1,2,3
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29653 \
  -m TrajGazeMerge.training.train_visionzip_traj_lora_3way \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/vztraj_sgeg_overlay \
  --epochs 3 --lr 1e-4 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval
```

### TAS (Stage-2 gaze-weighted merge) — 59.64 · GPU 2,3
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29652 \
  -m TrajGazeMerge.training.train_merge_lora_tas_3way \
  --model-type full --stage1-ckpt "$S1" \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/tas_lora_sgeg_overlay \
  --epochs 3 --merge-ratio 0.9 \
  --micro-batch 1 --grad-accum 4 \
  --no-hdepic --early-stop --eval-every 0 \
  --dataloader-num-workers 8
```

### M1 — VZ-complement, learned top-k — **63.01 (best)** · GPU 0,1
10% = 7% VZ content (3.5 dom + 3.5 ctx) ∪ 3% top-k TAS complement. See `MODEL_M1_VZ_COMPLEMENT.md`.
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29654 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_learned_overlay \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval
```

### M2 — VZ-complement, anticipatory pool — archived · GPU 2,3
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29655 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode anticipatory --horizon 2.0 --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/visionzip_complement_anticip_overlay \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval
```

---

## "Ours" additive / variant family (all ≤ M1 — see `MODEL_SCANPATH_OURS.md` + memory)

### Scanpath — 63.01 (tie, gate collapses) · GPU 0,1,2,3
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29662 \
  -m TrajGazeMerge.training.train_visionzip_scanpath_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --gaze-tokens 8 --t-scan 32 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_scanpath \
  --epochs 3 --lr 1e-4 --scan-lr 1e-3 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval
```

### Gaze-tag — 61.62 (overfit) · GPU 0,1,2,3
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29663 \
  -m TrajGazeMerge.training.train_visionzip_gazetag_lora \
  --traj-pool-mode learned --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_gazetag \
  --epochs 3 --lr 1e-4 --tag-lr 1e-3 --grad-accum 2 \
  --no-hdepic --early-stop --no-mid-eval
```

### Coverage-A 7/3 — 61.82 · Coverage-B 6/4 — 61.42 (de-clustering falsified)
```bash
# A (GPU 1,2): NMS de-cluster of the M1 complement, content 7% / traj 3%
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29656 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode coverage --nms-radius 1 --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_coverage_a \
  --epochs 3 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --no-mid-eval

# B (GPU 3,0): same but content 6% / traj 4%
CUDA_VISIBLE_DEVICES=3,0 torchrun --nproc_per_node=2 --master_port=29657 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode coverage --nms-radius 1 --stage1-ckpt "$S1" \
  --content-ratio 0.06 --traj-ratio 0.04 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_coverage_b \
  --epochs 3 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --no-mid-eval
```

### Fusion — archived (soft attn×traj fusion, falsified) · GPU 0,1
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29658 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode fusion --fusion-lambda "$LAM" --fusion-norm "$NORM" --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/ours_fusion \
  --epochs 3 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --no-mid-eval
```

---

## Decoupling grid (FINAL — 2026-06-17)

Tests whether adding gaze **on top of** the budget (→13%) beats trading content
for it (M1's 10%). **WIN** if `(M1+13 − VZ-8/5) > (M1-10 − VZ-6.5/3.5)` (gaze
scales past the 10% cap) and/or `M1+13 > 63.01` (new best). VZ-(6.5/3.5) also
isolates the gaze-vs-fidelity confound in M1's original +0.5.

### Results (egtea 2-way, n=1011; best across 3 epochs)

| Config | Budget | Composition | Gaze | ep1 | ep2 | ep3 | **Best** |
|---|---|---|---|---|---|---|---|
| VZ-10% (5/5) | 10% | 5 raw + 5 merged | — | — | — | — | 62.51 |
| **M1-10%** | 10% | 6.5 raw + 3.5 merged | traded-within | 61.62 | **63.01** | 60.83 | **63.01** |
| **M1+13%** | 13% | 8 raw + 5 merged | added-on-top | 61.72 | **62.81** | 62.12 | **62.81** |
| VZ-(8/5) | 13% | 8 raw + 5 merged | — | 60.63 | 60.53 | (early-stop) | 60.63 |
| VZ-(6.5/3.5) | 10% | 6.5 raw + 3.5 merged | — | 60.14 | **61.62** | 59.25 | 61.62 |

**Verdict — WIN via clause 1 (scaling-win), NOT a new SOTA.**
- Clause 1 (does gaze's lift grow added-on-top vs traded-within?): `(M1+13 − VZ-8/5) = +2.18` > `(M1-10 − VZ-6.5/3.5) = +1.39` → **TRUE**. Gaze's marginal value scales *past* the 10% zero-sum cap.
- Clause 2 (new best?): `M1+13 62.81 > 63.01` → **FALSE**.
- Why no SOTA: the 13% no-gaze floor (VZ-8/5 = 60.63) is *below* the 10% no-gaze floor (VZ-6.5/3.5 = 61.62) — the extra 3% budget bought low-value content/merged tokens. Gaze adds more on top (+2.18) but starts from a lower floor, landing at 62.81 < M1's 63.01. **The 10% budget remains the sweet spot; M1 (trade-within) stays best.**

### M1+13% (candidate) — FINAL 62.81 (ep2; ep1 61.72 / ep3 62.12) · GPU 0,1
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29671 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$S1" \
  --content-ratio 0.10 --traj-ratio 0.03 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/m1plus_13pct \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop --no-mid-eval
```

### VZ-(8/5) — 13% gaze-matched control — FINAL 60.63 (ep1; early-stopped, ep2 60.53) · GPU 2,3
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29672 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --dominant-ratio 0.08 --contextual-ratio 0.05 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/vz_8_5 \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop
```

### VZ-(6.5/3.5) — 10% diagnostic — FINAL 61.62 (ep2; ep1 60.14 / ep3 59.25) · GPU 2,3
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29673 \
  -m TrajGazeMerge.training.train_visionzip_lora \
  --dominant-ratio 0.065 --contextual-ratio 0.035 \
  --output-dir /workspace/EgoGazeVQA/TrajGazeMerge/checkpoints/vz_6p5_3p5 \
  --epochs 3 --lr 1e-4 --grad-accum 4 \
  --no-hdepic --early-stop
```

---

## Separate / specialist protocol (per-benchmark) — FINAL 2026-07-25

Unlike the master table above (train on the **SG∪EG union**, 3 epochs, eval on the full EGTEA test n=1011), this grid trains **one LoRA per benchmark** and evaluates it on **that benchmark's test split only** (`--source` filters both the train and the test items). Each run is **2 epochs, best-of-2**. Protocol otherwise identical to the joint runs: 2-GPU grad-accum-4 (eff-batch 8), `--merge-ratio 0.9` (10% token budget), `--no-hdepic` (2-way egtea), LoRA lr 1e-4, `GAZE_OVERLAY=1`, `--early-stop`.

- **SG-only**: train = StreamGaze (egoexolearn+holoassist) 5799; test = EGTEA-SG **n=526**.
- **EG-only**: train = EgoGazeVQA (ego4d+egoexo) 1265; test = EGTEA-EG **n=485**.

Because the test split is source-filtered, each run's `Overall: XX% (n=NNN)` **is** that specialist's accuracy on its own benchmark (not a per-source slice of a joint eval).

| Benchmark | Model | Selection @10% | Test n | ep1 | ep2 | **Best** |
|---|---|---|---|---|---|---|
| SG-only | **M1** (VZ-complement, learned traj) | 7% content ∪ 3% traj | 526 | 65.59 | 69.96 | **69.96** |
| EG-only | **M1** (VZ-complement, learned traj) | 7% content ∪ 3% traj | 485 | 54.85 | 53.81 | **54.85** |
| EG-only | VisionZip | 10% content | 485 | 54.85 | 53.20 | **54.85** |
| EG-only | FastVID | 10% | 485 | 52.99 | 53.20 | **53.20** |
| EG-only | PruneVid | 10% | 485 | 55.05 | 53.61 | **55.05** |

Reference (JOINT protocol, from the master table): joint M1 = 63.01 combined (**SG 69.20 / EG 56.29**); joint VZ = 62.51.

**Takeaway.** On its own benchmark the SG-only M1 (**69.96**) beats the joint model's SG slice (69.20, **+0.76**) — the SG train set is large enough that a dedicated specialist wins. But every EG-only specialist (M1 54.85, VZ 54.85, PruneVid 55.05, FastVID 53.20) **underperforms the joint model's EG slice (56.29)** — union training transfers into the smaller/harder EG domain, so the joint model remains the better EG deployment. Among EG selection methods under the single-domain protocol: **PruneVid 55.05 > M1 = VisionZip 54.85 > FastVID 53.20**; M1's gaze-complement edge that holds under joint training (M1 63.01 > VZ 62.51) does **not** survive in the small EG-only regime (M1 merely ties VisionZip). Net: a specialist helps only where its single-source train set is large (SG); for EG, keep the joint model.

### Launch (specialist runs) — `S1`/`CKPT` as in Shared setup
```bash
# SG-only M1  (Chain A [A1], GPU0,1, master_port=29691)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29691 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$S1" \
  --content-ratio 0.07 --traj-ratio 0.03 --source sg \
  --output-dir "$CKPT/visionzip_complement_learned_SGonly_overlay" \
  --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop --no-mid-eval

# EG-only M1  (Chain B [B1], GPU2,3, master_port=29693): as above but --source eg and
#   --output-dir "$CKPT/visionzip_complement_learned_EGonly_overlay"

# EG-only baselines  (visionzip=Chain A [A2] 29692; fastvid=Chain B [B2] 29694; prunevid=Chain B [B3] 29695)
CUDA_VISIBLE_DEVICES=<gpus> torchrun --nproc_per_node=2 --master_port=<port> \
  -m TrajGazeMerge.training.train_baseline_select_lora \
  --select-mode {visionzip|fastvid|prunevid} --source eg \
  --output-dir "$CKPT/{visionzip|fastvid|prunevid}_EGonly_overlay" \
  --epochs 2 --lr 1e-4 --grad-accum 4 --no-hdepic --early-stop
```
Checkpoints: `$CKPT/{visionzip_complement_learned_SGonly_overlay, visionzip_complement_learned_EGonly_overlay, visionzip_EGonly_overlay, fastvid_EGonly_overlay, prunevid_EGonly_overlay}` — the `_SGonly_`/`_EGonly_` suffixes keep them separate from the joint checkpoints.
