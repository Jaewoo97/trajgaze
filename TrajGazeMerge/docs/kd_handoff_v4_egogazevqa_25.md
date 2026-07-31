# KD Handoff v4 — EgoGazeVQA at a 25% token budget

Written 2026-07-31, for a **different machine with 4 GPUs**. Does not supersede
`kd_handoff_v2.md` or `kd_handoff_v3.md` — v2 remains the task definition, v3 defines the
ViT selection-distillation method and its two-phase protocol. This document is the
operating manual for running that method on **EgoGazeVQA at 25%** and nothing else.

v2 §8 (eval is not deterministic), v3 §2.5 (the two-phase protocol is new in v3) and
v3 §10 (what must be said when reporting) apply verbatim to every number produced here.

---

## 1. What this is

Every row in v3 sits on one token budget: M1 keeps 10% of the visual tokens, 7% VisionZip
content ∪ a 3% gaze/hand complement. The budget was raised to **25% = content 15% ∪ traj
10%** and re-run on StreamGaze; this repeats that on EgoGazeVQA so the budget result can
be read on both sources.

Three stages, in order:

1. **Teacher** — M1 EG-only at 25%, 2 epochs, best-of-2.
2. **Re-score** the best checkpoint 3×, report per task.
3. **Student** — ViT-KD on EG raw video: Phase 1 → integrity gate → Phase 2, 1 epoch each.

One command runs all three: `bash scripts/run_b25_eg_all.sh`. §7 breaks it into the
individual commands if you want to stop between stages.

**The SG result this is being compared against** (same recipe, same budget, measured on
the reference machine 2026-07-31):

| SG teacher @25%, best-of-2 | Avg | items |
|---|---|---|
| re-score max | 72.81% | 383 |
| **re-score mean (n=3)** | **72.49%** | **381** |
| re-score min | 72.24% | 380 |
| 10% M1 SG-only teacher (v2 §8, 4-run mean) | 71.29% | 375 |

Raising the budget bought **+1.20 points / +6 items** on SG, and the *minimum* of the
three re-scores also clears the 10% teacher, so the result does not depend on best-of-N.
Per task the gain concentrated where the larger complement should put it: GSM +5.20 and
FAP +4.61, against SR −3.61 and OAR −1.73.

The SG student completed at this budget too, and it is the closest thing to a prediction
for the EG run — same recipe, one epoch per phase, raw video:

| SG student @25%, 1 epoch per phase | measured | 10% counterpart (v3 §5.4, P1/P2 ep1) |
|---|---|---|
| P1 `recall_traj`, frozen → tuned | 0.1358 → **0.5343** (Δ +0.3986) | 0.0435 → 0.3636 (Δ +0.3202) |
| P1 `recall_P` | 0.4428 → **0.5571** (Δ +0.1143) | 0.3985 → 0.4602 (Δ +0.0618) |
| P1 `recall_S` | 0.4481 → **0.6154** (Δ +0.1673) | 0.4019 → 0.5263 (Δ +0.1244) |
| integrity gate | frozen 364 / tuned 366, **Δ +2**, cos 0.99137 → PASS | frozen 372 / tuned 369, Δ −3, cos 0.9915 |
| **P2 accuracy** | **71.10% — 374 items** | 69.58% — 366 items |

**+8 items at a matched readout budget** (both are P2 epoch 1), above v2 §8's ±4 floor.
That puts the gaze-free student within **1 item** of the *10%* gaze-using teacher (375),
and narrows the gap to its own 25% teacher from 9 items to 7.

Where the +8 came from is worth carrying into the EG run: NFI +5, SR +3, OI-E +3 — and
**GSM went the other way, 45 → 43 items, despite `recall_traj` nearly doubling**. That is
v3 §5.6 reproducing at a second budget: GSM is governed by whether the gaze marker is
visible in the pixels, not by which tokens get selected, so on raw video no amount of
selection fidelity buys it. Expect the same shape on EG — recovered selection paying off
in the object/scene columns, not in the gaze-driven one.

Note the frozen baselines are all **higher** at 25% than at 10% — dominant widens from
6.5% to 17.5%, so more of the teacher's complement falls into it by chance. Read the
delta, never the absolute, and do not compare either against v3 §5.4's 0.383.

**The EG numbers to beat** — 10% budget, v3 §5.7, EGTEA test n=485:

| system | Caus. (162) | Spat. (163) | Temp. (160) | Avg | items |
|---|---|---|---|---|---|
| M1 EG-only teacher (single run) | 85.80 | 39.88 | 36.25 | **54.02** | **262** |
| KD student, raw video (v2 §7.7) | 85.80 | 42.94 | 36.88 | **55.26** | **268** |
| ViT-KD raw video, P2 ep1 | 84.57 | 38.65 | 36.88 | 53.40 | 259 |
| ViT-KD raw video, P2 ep2 | 85.80 | 39.26 | 36.25 | 53.81 | 261 |

EG behaved differently from SG at 10%: the ViT-KD row came within one item of its own
teacher but **missed the KD student by 7 items, six of them in `spatial`**, and Phase 1's
selection never transferred to the test distribution (`recall_traj` 0.520 on EG's own
training window, 0.113 on EGTEA). Whether more budget changes that is the question here.

---

## 2. What this machine needs

| | |
|---|---|
| GPUs | **4**, ≥60 GB each. Measured peak at 25%: **50.0 GB** for the teacher, ~26 GB for Phase 1 |
| Disk | ~120 GB for checkpoints (teacher 3 × 16.6 GB, P2 2 × 16.6 GB, P1 ~1 MB) |
| Environment | **the one already installed** — do not build a new one. It needs torch with flash-attn, `peft`, `qwen_vl_utils`, and the Qwen2.5-VL-7B weights in `HF_HOME` |
| `git lfs` | only if you fetch `*.pth` from the HF mirror |

### 2.1 `OMP_NUM_THREADS=1` — read this before running anything

`torch.distributed.run` sets `OMP_NUM_THREADS=1` **only when `nproc_per_node > 1`**. The
re-scores and the integrity gate are single-rank, so they inherit an unset variable and
torch then parallelises every CPU op across all cores. Measured on the reference machine:

| | s/item | symptom |
|---|---|---|
| one eval, threads uncapped | **6.6** | 13 cores burnt, GPU at 0%; `/proc` showed 218 python threads with ~70 at an identical 129 s of CPU — an OpenMP spin barrier, not work |
| two evals, uncapped | **~60** | 144 spinning threads on 72 cores; 97 KB/s and 0 B/s of disk reads |
| one eval, `OMP_NUM_THREADS=1` | **1.2** | 1.2 cores, GPU bursting normally |

`run_b25_eg_all.sh` sets it for every single-rank job. If you run any command by hand,
set it yourself. A "hung" eval is almost always this.

---

## 3. Bring the code up to date

**Do not clone.** This machine already has `TrajGazeMerge`. Fetch only the files below
from `yujin_v2` and overwrite in place:

```
https://raw.githubusercontent.com/Jaewoo97/trajgaze/yujin_v2/<path>
```
Browse at <https://github.com/Jaewoo97/trajgaze/tree/yujin_v2>.

First check what is already there — if all three print a non-zero count, the v3 machinery
is present and you only need the `scripts/` additions:

```bash
grep -c "grad_last_block"    VisionZip/Qwen2_5_VL/qwen2_5vl_visionzip.py
grep -c "_EG_VLM_FRAME_SUB"  TrajGazeMerge/data/combined_simple_dataset.py
ls                           TrajGazeMerge/training/train_vit_selection_kd.py
```

| Path | If missing / stale |
|---|---|
| `TrajGazeMerge/training/train_vit_selection_kd.py` | Phase 1 does not exist |
| `scripts/vitkd_integrity_gate.py` | no gate — Phase 2 could train on a damaged encoder unnoticed |
| `VisionZip/Qwen2_5_VL/qwen2_5vl_visionzip.py` | no `grad_last_block` / `grad_logits` kwargs → **Phase 1 cannot backpropagate at all** |
| `TrajGazeMerge/training/train_visionzip_lora.py` | no `--vit-lora-ckpt` / `--resume` / `--ckpt-every-steps` / `--seed` → Phase 2 cannot run |
| `TrajGazeMerge/data/combined_simple_dataset.py` | **the one that matters most.** v3 §4.1: without it `VLM_GAZE_OVERLAY=0` is silently ignored and Phase 2 trains and evaluates on gaze-overlay frames. No error, no shape change, nothing visible in the accuracy — the "raw video" row is simply false |
| `TrajGazeMerge/training/train_autogaze_lora.py` | the SG half of that same defect. Not exercised by an EG-only run, but keep the pair consistent |
| `TrajGazeMerge/training/bench_probe.py` + its hooks in the four trainers | optional; without it the logs lose the `s/step \| GB` field that §6 checks |

Then add the 25% scripts (all new, none of them modify anything above):

```
scripts/run_b25_eg_all.sh        the chain in §7
scripts/check_eg_dataset.py      §4
scripts/collect_b25_pertask.py   the per-task table, --source eg
```

---

## 4. Verify the dataset — before downloading anything

This machine already has EgoGazeVQA. The question is not whether to download it but
whether it is **the same data the reference numbers were measured on**. A different frame
set produces numbers that cannot be compared to §1, and nothing downstream would catch it.

```bash
export EG_ROOT=/path/to/EgoGazeVQA
python scripts/check_eg_dataset.py --with-loader
```

That checks, and must print `PASS — all 20 checks match`:

| | expected |
|---|---|
| `metadata.csv` | 1,751 lines · 946,793 bytes · md5 `fdcb4f8424fbcef3fa680a22d20b91e9` |
| `egtea/{gaze,no_gaze}` | 82 videos · **93,755** frames each |
| `ego4d/{gaze,no_gaze}` | 27 videos · **66,017** frames each |
| `egoexo/{gaze,no_gaze}` | 154 videos · **231,928** frames each |
| `gaze` vs `no_gaze` | identical frame basenames, per video |
| loader | `split='train'` → **1,265**, `split='test'` → **485** |

The basename check is not decoration: `egogaze_dataset.py:49-52` picks sampling indices
from one variant's listing while gaze, hand and interaction lookups are keyed by frame
basename, so a mismatch silently misaligns the trajectory streams rather than raising.

**Only if it fails**, re-fetch from
<https://huggingface.co/datasets/Peanuttoad/gaze_dataset_full>:

> The raw video is at the **repository root**, under `EgoGazeVQA/` — *not* under `aaai/`,
> which holds only checkpoints. Fetching `aaai/` alone leaves you with no frames.

`EgoGazeVQA/egtea_no_gaze.tar` (7.02 GB) · `ego4d_no_gaze.tar` (12.5 GB) ·
`egoexo_no_gaze_part1.tar` (41.9 GB) · `egoexo_no_gaze_part2.tar` (49.1 GB), plus the
matching `*_gaze*.tar` and `*_gaze_mapping.tar`, then extract with the root `restore.sh`.

### 4.1 The one checkpoint you do need

`aaai/stage1_tas_3way_overlay/best.pth` (**147 MB**) — the frozen TAS Stage-1 encoder that
supplies the gaze/hand salience field. Every stage below reads it. Point `STAGE1_CKPT` at it.

You do **not** need the 16.6 GB `visionzip_complement_learned_EGonly_overlay/best.pth`:
that is the 10% teacher, and this run trains its own at 25%. Fetch it only to reproduce
the 10% baselines in §1 directly.

---

## 5. Environment variables

Use the existing environment; only these need to be right:

```bash
export REPO=/path/to/trajgaze
export EG_ROOT=$DATA/EgoGazeVQA
export STAGE1_CKPT=$REPO/TrajGaze_v2/checkpoints/stage1_tas_3way_overlay/best.pth
export HF_HOME=...            # Qwen2.5-VL-7B cache
export TORCH_HOME=...         # DINOv2 cache; a cold one races the DDP ranks
export GAZE_OVERLAY=1
export TORCHRUN="python -m torch.distributed.run"
```

`GAZE_OVERLAY=1` throughout. `VLM_GAZE_OVERLAY` is set per stage by the chain script and
must not be exported globally: the teacher needs it **unset** (overlay frames) and the
student needs it **0** (raw video).

---

## 6. Smoke test — 5 minutes, before committing GPU hours

```bash
env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  $TORCHRUN --nproc_per_node=1 --master_port=29951 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.15 --traj-ratio 0.10 --source eg \
  --epochs 1 --lr 1e-4 --grad-accum 2 --no-hdepic --no-mid-eval \
  --output-dir /tmp/dryrun_eg --max-steps 3 --bench-warmup 1
```

Must print:

- `content=15.0% ∪ traj=10.0% = 25%` — if it says 10%, a ratio flag was dropped
- `[source=eg] train filtered 7064 → 1265 items`
- a `[BENCH]` line; on the reference machine `peak_gb` was **50.0**

---

## 7. Run

```bash
cd $REPO && source env.sh
tmux new -s b25eg
bash scripts/run_b25_eg_all.sh
```

tmux is not optional — an ssh drop kills the chain. Progress is in
`vitkd25eg_chain.log`; per-job logs are `vitkd25eg_<job>.log`; `.done` markers under
`vitkd25eg_state/` make re-running skip finished jobs.

**4 GPUs, eff-batch 8.** `--nproc_per_node=4` pairs with `--grad-accum 2` (4 × 2 = 2 × 4),
the effective batch every number on record was produced at. EG train is 1,265 items, so
each rank sees ~317 micro-steps per epoch.

### 7.1 Teacher — 25%, 2 epochs

```bash
env -u VLM_GAZE_OVERLAY GAZE_OVERLAY=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $TORCHRUN --nproc_per_node=4 --master_port=29890 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt "$STAGE1_CKPT" \
  --content-ratio 0.15 --traj-ratio 0.10 --source eg \
  --epochs 2 --lr 1e-4 --grad-accum 2 --no-hdepic --early-stop --no-mid-eval \
  --output-dir "$CKPT/visionzip_complement_learned_EGonly_overlay_b25_2ep"
```

Overlay frames (`VLM_GAZE_OVERLAY` unset → EG `gaze`), matching the 10% teacher, so the
budget is the only thing that changed. **This trainer has no `--resume` and no
`--ckpt-every-steps`** — it writes `epoch_NN.pth` and `best.pth` only, so a death inside an
epoch loses that epoch entirely and the retry starts from zero. Check: log says
`epochs=2`, `[source=eg] ... 1265 items`, and each epoch ends with `Overall: XX% (n=485)`.

### 7.2 Re-score the best checkpoint 3×

**There is no seed to vary.** The eval path contains no RNG — no `random`, `np.random` or
`shuffle` in `data/*.py` or `models/model.py`, and the option permutation at
`train_visionzip_complement_lora.py:68` only fires under `--option-aug`, in the training
loop. This trainer has no `--seed` flag at all. The spread v2 §8 measured across four
re-scores of *identical* weights (71.67 / 71.29 / 71.48 / 70.72, five items) is
bf16/flash-attn kernel nondeterminism: you get it by **running the same command again**.

Three single-rank jobs, one per GPU, in parallel — extra ranks would not help because
`evaluate()` runs on rank 0 only (`train_visionzip_complement_lora.py:805`). Then:

```bash
python scripts/collect_b25_pertask.py --source eg
```

which writes `vitkd25eg_teacher_b25_2ep_pertask.txt`: every run's per-task accuracy with
item counts, the max / mean / min / spread, and the 10% reference column. It reconstructs
each per-task item count and checks the counts sum to that run's own `Overall` — the
check that catches a mis-filtered split.

**Report the mean, and say so.** v2 §8 forbids best-of-N selection as an upward-biased
estimator, and the 10% reference in §1 is itself a mean.

### 7.3 Student — ViT-KD on EG raw video

`p1` → `gate` → `p2`, raw video (`VLM_GAZE_OVERLAY=0` → EG `no_gaze`), 1 epoch per phase.
Ratios: `--content-ratio 0.15 --traj-ratio 0.10` with `--dom-primary 0.175
--ctx-primary 0.075` (P1) and `--dominant-ratio 0.175 --contextual-ratio 0.075`
(gate and P2). All four must agree or the log's `kept=` silently reads 10%.

**The gate must PASS (|Δ| ≤ 4 items) before Phase 2 starts** — it holds the selection
fixed at the frozen ViT's choice and swaps only the features, so any difference is
representation drift in block 31 and nothing else. The chain stops on failure.

**Watch Phase 1's first 200 steps.** `lr 2e-3` was chosen against a 10% target; at 25%
there are 2.5× as many positives (`pos_weight` 9 → 3). If the windowed `recall_traj` is
not climbing, kill it and raise the lr (5e-3, then 1e-2). v3 §5.2 records exactly this
failure at `lr 2e-5`: six hours spent to produce a conclusion about the optimiser.

Note the frozen `recall_traj` baseline is **not** v3's 0.042 at this budget — dominant
widened from 6.5% to 17.5%, so more of the complement lands there by chance. Phase 1's
epoch-end eval prints frozen and tuned together; read the delta, not the absolute. On SG
at this budget the frozen baseline came out at 0.1358; §1 has the full set to compare
against, though EG's own baseline will differ.

One EG-specific caution from v3 §5.7: at 10% the EG Phase 1 fit its *training* window
harder than SG did (`recall_traj` 0.520 vs 0.394) and still reached only 0.113 on EGTEA
— a −0.41 train/test gap, against SG's −0.03. A high windowed `recall_traj` during
training therefore does **not** predict the epoch-end number here. Wait for the eval.

---

## 8. Expected wall-clock — ≈ 2.8 h on 4 B200-class GPUs

| stage | derivation | estimate |
|---|---|---|
| teacher, 2 epochs | 317 steps/rank × ~2.6 s + eval 485 × ~1.9 s, twice | **1.0 h** |
| re-score ×3 (parallel) | 485 × 1.2 s + model load, one round | **0.2 h** |
| P1, 1 epoch | 317 × 4.75 s + eval 485 × 2.38 s | **0.75 h** |
| integrity gate | 485 × ~3.4 s, single GPU | **0.45 h** |
| P2, 1 epoch | 317 × 2.45 s + eval 485 × 1.52 s | **0.42 h** |
| **total** | | **≈ 2.8 h** |

Unit costs come from v3 §5.8's measured EG rates at 10% and the reference machine's 25%
SG run. **Raising the budget barely moves step time**: v3 §5.8 recorded the GPU idling
~28% of wall (frame decoding sits in the training loop, not the DataLoader), and the extra
LLM tokens fill that bubble — the SG teacher measured 2.11 s/step at 25% against ~2.1 at
10%, with peak memory going 29.7 → 47.5 GB.

**Extra GPUs do not shorten the eval half.** `evaluate()` is rank-0 only, and on EG the
training set is small enough that eval dominates: about 920 s of eval against 824 s of
training per teacher epoch. If these GPUs are not B200-class, scale the training rows and
leave the eval rows to measure themselves.

---

## 9. What to report

Per task, in this order, with item counts that sum exactly to the total:

- **Teacher**: `Caus. (162) · Spat. (163) · Temp. (160) · Avg (485)` for both epochs and
  all three re-scores, plus max / mean / spread. `collect_b25_pertask.py --source eg`
  emits exactly this.
- **Student**: the same columns for P2, plus Phase 1's `recall_P` / `recall_S` /
  `recall_traj` (frozen → tuned) and the gate's frozen/tuned item counts and cosine.

Against §1's table: does the 25% teacher clear the 10% teacher's 262 items, and does
ViT-KD@25% close any of the 7-item gap to the KD student's 268 — especially in `spatial`,
where six of those seven items were.

---

## 10. What must be said when reporting

Inherited from v3 §10, adjusted for this run:

1. **"matches", not "beats"**, for any comparison to M1 — v2 §10.4's warm-start confound
   applies unchanged and the equal-budget control was never run.
2. **Single runs.** v2 §8 wants ≥3 evals. Only the teacher gets that here; the student's
   P2 number is one run, and differences under 4 items are not measurements.
3. **The teacher is best-of-2 epochs**, and the headline is a 3-run **mean** — if you
   quote the max, label it best-of-3. The 10% references are means.
4. **P1 and P2 run one epoch each.** That matches v3 §5.7's EG rows (also 1 epoch/phase)
   but *not* v3 §5.4's SG rows, whose P1 ran two epochs. Say which when comparing.
5. **Overlay settings are not gaze-free.** This run is raw video, so it is — but the
   Stage-1 TAS encoder is still overlay-trained (v2 §9). State that as an assumption.
6. **`recall_*` are selection metrics, not accuracy.** High recall with flat accuracy
   means the reproduced tokens were not the ones that mattered — report both.
7. The distilled score is the **unmasked cross-frame** column sum of v3 §4.2, not a
   per-frame saliency. Anyone reproducing this on a stock VisionZip will get other numbers.

---

## 11. Failure playbook

| symptom | do |
|---|---|
| a job looks hung, GPU 0% | check `OMP_NUM_THREADS` first (§2.1). This has been the cause every time |
| OOM in Phase 1 | lower `--score-query-frac` (P1 only; it changes the distilled quantity, so record it) |
| gate FAILs, \|Δ\| > 4 | P1 checkpoints every 200 steps — gate an earlier checkpoint instead of retraining |
| `recall_traj` flat over the first 200 steps | raise lr: 5e-3, then 1e-2 (§7.3) |
| `kept=` reads 10.0% | a ratio flag was dropped; all four must be set together (§7.3) |
| eval reports n≠485 | the `--source eg` filter did not apply |
| NaN loss | stop and report; do not restart blindly |
| teacher died mid-epoch | it has no resume — the retry restarts that epoch from zero. Expected, not a bug |
| chain died entirely | re-run `run_b25_eg_all.sh`; finished jobs are skipped and P1/P2 resume mid-epoch |
