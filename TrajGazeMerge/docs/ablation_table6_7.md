# Re-running Table 6 (Stage-1 pretraining objectives) and Table 7 (spatial vs. temporal selection) for the current \sys method

**Audience:** whoever re-runs these two ablations on a fresh machine.
**Why this exists:** `paper/main.tex` now describes the *current* method — VisionZip-complement
selection at a 10% visual-token budget (7% VZ content ∪ 3% frozen-trajectory complement).
The numbers currently sitting in Tables 6 and 7 are **stale**: they were produced by an
earlier method variant (their "Spatio-temporal / All-losses" rows land at ~67–68%, whereas the
current \sys StreamGaze number is ~69–70%). Both tables must be regenerated with the current
pipeline. This document specifies exactly what each row means for the current method, what code
to add, the exact train/eval commands, and the empty result tables to fill. **It does not contain
any measured numbers — you produce those.**

---

## 0a. STATUS — resolved decisions and machine-specific corrections (2026-07-27)

The code this document asks for **has been implemented** and the runs are in progress. The
decisions left open below (§5) were resolved as follows; where this section disagrees with the
rest of the document, **this section wins**.

| Decision | Resolution |
|---|---|
| LoRA rank (§5.1) | Code (r=16, α=32) is correct; `paper/method.tex` was fixed. No re-runs |
| Epochs (§5.2) | New rows `--epochs 1`, no `--early-stop` (provable no-op at 1 epoch) |
| Train source (§5.3) | **`--source sg`** — every row trains *and* evals on StreamGaze only |
| Shared row (§5.4) | \sys row is **not** re-run: reuse the SG-only specialist's existing per-task evals |
| "Only score loss" | Keep `loss_score_past` **and** `loss_score_traj`; drop `loss_traj`, `loss_score_fut` |
| Table 7 free row (§5.5) | Real two-pool \sys (the reused row above) |
| Avg | **macro-average over 7 tasks**; OTP dropped from *both* tables |

**Why "Only score loss" keeps `loss_score_traj`.** The salience Stage-2 actually consumes is
`score_head` (`model_temporal.py:281`), and `score_head` is supervised by `loss_score_traj`
alone (`:218`, `:243`). `loss_score_past` supervises `past_scores` (`:213`), a raw encoder
attention readout that inference never touches. Keeping only `loss_score_past` — this document's
original recommendation in §2.2 — would ship a *randomly initialised* inference head and collapse
the row onto "No pretrain".

**Corrections to §0's paths and assumptions on this machine:**

- `REPO=/NHNHOME/VILAB/vilab_yj/trajgaze`; **`source env.sh`** supplies `$STAGE1_CKPT`,
  `$M1_JOINT`, `$M1_SGONLY`, `$TORCHRUN`. The `/workspace/...` paths and the
  `/opt/conda/envs/trajgaze` interpreter in §0 do not exist here.
- Always `unset VLM_GAZE_OVERLAY` — `env.sh` does not clear it, and a stale `0` silently swaps
  every VLM frame to non-overlay.
- **2 GPUs, not 4.** The 4-GPU Stage-1 launcher and the "two co-resident 2-GPU runs" in §2.5/§3.2
  are impossible; all rows run serially on GPUs 0+1.
- **`--use-egovqa` / `--use-hd-epic` cannot be used.** `TrajGaze_v2/data/dataset_temporal_egovqa.py`
  and `dataset_temporal_hdepic.py` do not exist on this branch (`stage1_temporal.py:167,173` import
  them lazily). New Stage-1 encoders therefore train on **StreamGaze only (246 clips)**, whereas
  `$S1` was trained on SG + EgoGazeVQA + HD-EPIC — a corpus difference that must be disclosed.
- Line numbers throughout §2–§4 have drifted. Actual: `evaluate()` at `:504`,
  `load_traj_encoder` call at `:605`, `MCQ_TASKS` at `dataset.py:38-47`.

**Deviations from the code snippets in §2.3 and §3.1 (both were buggy as written):**

1. §2.3 routes the drop flags through `total_from_weighted`, which *skips* zero-weight terms and
   so removes `traj_decoder` / `score_decoder` from the autograd graph. Stage-1 wraps the model in
   `DDP` without `find_unused_parameters` (`stage1_temporal.py:213`), so that raises
   "Expected to have finished reduction in the prior iteration". The implementation multiplies each
   term by its weight instead, keeping every parameter in the reducer with a zero gradient.
2. §3.1 computes `k` and never uses it, derives the two geometries' budgets independently, omits
   the `T*n_spatial == N` guard from the code, and leaves `topk` unclamped. The implementation
   derives both budgets from `k` and adds the guard and clamps.

**Realized budgets** (measured; grid is uniform at N=13824, T=64, n_spatial=216, k=1382):
`no_spatial` keeps 6 frames × 216 = **9.38%**, `no_temporal` keeps 64 frames × 22 = **10.19%**.
Frame granularity is 1.56% of the budget, so an exact 10% is unreachable; both are the nearest
achievable value. Report these in the caption.

**Captions must disclose:** (1) ablation rows are 1 epoch while the \sys row is the epoch-2 best
of a 2-epoch run — the training budget favours \sys; (2) the Stage-1 corpus difference above;
(3) Table 7's constrained rows select the whole 10% by fused `s = norm(attn)+norm(s_traj)` and
contain raw tokens only (no VisionZip contextual merge), while the free row is the deployed
two-pool selector; (4) Avg is a macro-7 with OTP (n=2) excluded; (5) LoRA r=16, α=32.

**Reproduce:** `scripts/run_ablation_tab6_tab7.sh` runs the four new rows;
`scripts/collect_ablation_tab6_tab7.py` assembles both tables.

---

## 0. Environment and shared protocol (read once)

```bash
# --- paths that may need adjusting on the target machine ---
REPO=/workspace/trajgaze_st                                   # this repo
CKPT=/workspace/EgoGazeVQA/TrajGazeMerge/checkpoints          # LoRA (Stage-2) output root
S1ROOT=/workspace/EgoGazeVQA/TrajGaze_v2/checkpoints          # Stage-1 encoder output root
S1=$S1ROOT/stage1_tas_3way_overlay/best.pth                   # current (frozen) Stage-1 encoder = "All losses"

# --- environment (mandatory) ---
cd $REPO
export PATH=/opt/conda/envs/trajgaze/bin:$PATH               # the 'trajgaze' conda env
export GAZE_OVERLAY=1                                         # frames must be the gaze-overlay base; REQUIRED
```

Invariants that must hold for every run below (they define "our protocol"):

| Item | Value | Notes |
|---|---|---|
| Backbone | Qwen2.5-VL-7B, **frozen** | only LoRA trains |
| LoRA | **r=16, α=32**, dropout 0.05, targets `q/k/v/o_proj` | `train_visionzip_lora.py:54-56,84`. **NB: paper says r=64/α=128 — discrepancy, see §5.** |
| Budget | 10% = `--content-ratio 0.07` ∪ `--traj-ratio 0.03` | do not change |
| Trainer | `TrajGazeMerge.training.train_visionzip_complement_lora` | Stage-2; **constant LR (no scheduler)** |
| Optim | `--lr 1e-4 --grad-accum 4`, 2-GPU DDP ⇒ eff-batch 8 | matched to all other \sys runs |
| Eval set | EGTEA test, `--source sg` ⇒ **n=526** StreamGaze items | Tables 6 & 7 are StreamGaze-only |
| Data | 2-way (StreamGaze+EgoGazeVQA), `--no-hdepic` | matches the main \sys training recipe |
| Encoder | frozen Stage-1 `TrajGazeV2Temporal` (learned mode) | `--traj-pool-mode learned --stage1-ckpt $S1` |

**Constant-LR consequence (important, saves compute):** `train_visionzip_complement_lora`
has no LR scheduler, so epoch-1 of an N-epoch run is bit-identical to a standalone 1-epoch run.
Recent \sys ablations were run at `--epochs 1` (or `2`, taking best-of-2). Pick one policy and
apply it to *every* row of a table so rows are comparable. This spec uses `--epochs 2` +
best-of-2 as the default (matches how the specialist grid was produced); `--epochs 1` is the
cheap option. **Whatever you pick, keep it identical across all rows of the same table.**

**Eval command (identical for every row).** After a Stage-2 run finishes, its own end-of-epoch
eval already prints `Overall:` and a per-task breakdown filtered to `--source sg`. To (re)measure
any saved checkpoint under the SG-only per-task pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29601 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt $S1 \
  --content-ratio 0.07 --traj-ratio 0.03 --source sg --no-hdepic \
  --eval-ckpt $CKPT/<run-dir>/best.pth
```

`evaluate()` (`train_visionzip_complement_lora.py:497`) returns `per_task` keyed by the internal
task name, already filtered to the requested source. The column↔task mapping is in §4.

---

## 1. What the current method is (so each ablation row is unambiguous)

Per video (T frames × `n_spatial` patches = N tokens) the selector keeps 10%:

1. **Content pool C (7%)** — VisionZip picks by LLM-attention: half "dominant" raw tokens +
   half "contextual" merged centroids (`select_complementary` → `visionzip_select_tokens`,
   `train_visionzip_complement_lora.py:407-411`).
2. **Complement pool G (3%)** — from the tokens VZ did **not** keep (`avail = ¬C`), take the
   global top-k by frozen-encoder trajectory salience `s_traj(f,p)`
   (`train_visionzip_complement_lora.py:465-489`; scores from `get_patch_scores_temporal`).
3. Final set S = C ∪ G (disjoint), fed to the frozen LLM; LoRA adapts the LLM to this budget.

The trajectory salience `s_traj` comes from the **frozen Stage-1 encoder** — that encoder is the
subject of Table 6. The **geometry** of how S is laid out across (frame, patch) is the subject of
Table 7.

---

## 2. Table 6 — Stage-1 pretraining objectives (`tab:pretrain`)

**Claim of the table:** the frozen trajectory encoder's usefulness depends on *how it was
pretrained*. Three encoders are pretrained differently, then each is dropped into the **identical**
Stage-2 pipeline.

### 2.1 Stage-1 recap

- Script: `TrajGaze_v2/training/stage1_temporal.py` (produces the `.pth` the Stage-2 trainer loads).
- Launcher for the current encoder: `scripts/launch_stage1_tas_3way.sh` (4-GPU DDP, 100 ep,
  lr 3e-4, batch/GPU 2, 128 frames, `--use-egovqa --use-hd-epic --use-trajectory-anchor`).
- Four supervised loss terms are emitted (`stage1_temporal.py:355-358`, confirmed in the current
  encoder's `train_log.jsonl`):

  | term | paper loss family | what it supervises |
  |---|---|---|
  | `loss_traj` | (a) masked trajectory prediction | decoder predicts future gaze/hand (x,y), Huber |
  | `loss_score_past` | (b) patch-score regression vs GT relevance I(p,t) | grounds per-patch score on **observed** frames |
  | `loss_score_traj` | (b) patch-score regression (trajectory-decoder path) | noisiest term; chains decoder×encoder attn |
  | `loss_score_fut` | (c) attention–future-relevance alignment | per-patch score on **future** frames |

  Default (non-curriculum) combine is the plain sum of all four (`stage1_temporal.py:300`,
  `loss = loss_dict["loss"]`).

### 2.2 Row definitions

| Row | Encoder pretraining | Loss terms kept |
|---|---|---|
| **No pretrain** | none — random-init encoder, frozen | (Stage-1 skipped entirely) |
| **Only score loss** | Stage-1 with patch-score regression only | `loss_score_past` (drop `loss_traj`, `loss_score_fut`, `loss_score_traj`) |
| **All losses** = \sys | current full Stage-1 | all four (this is the existing `$S1`) |

> Decision point: whether "score loss" includes `loss_score_traj`. Recommended = **exclude** it
> (keep only `loss_score_past`, the clean per-patch regression on observed frames) so the row is a
> pure "score-regression-only" encoder. If you prefer "all score-family terms", keep
> `loss_score_past`+`loss_score_traj`+`loss_score_fut` and only drop `loss_traj`. Pick one and note
> it in the caption. The commands below implement the recommended (minimal) version.

### 2.3 Code to add (Stage-1 loss toggles)

`stage1_temporal.py` already has `--drop-loss-score-traj`. Add two more flags and make the
non-curriculum path honor all three. In `parse_args()` (next to the existing flag, ~line 100):

```python
p.add_argument("--drop-loss-traj", action="store_true",
               help="Zero l_traj (masked trajectory prediction).")
p.add_argument("--drop-loss-score-fut", action="store_true",
               help="Zero l_score_future (attention-future-relevance alignment).")
```

Replace the loss-combination block (`stage1_temporal.py:285-300`) with a single path that respects
the flags with or without curriculum:

```python
from TrajGaze_v2.training.loss_schedule import LossWeights, curriculum_weights, total_from_weighted
if args.use_curriculum:
    w = curriculum_weights(global_step, total_steps)
else:
    w = LossWeights(traj=1.0, score_past=1.0, score_future=1.0, score_traj=1.0)
w = LossWeights(
    traj         = 0.0 if args.drop_loss_traj       else w.traj,
    score_past   = w.score_past,
    score_future = 0.0 if args.drop_loss_score_fut  else w.score_future,
    score_traj   = 0.0 if args.drop_loss_score_traj else w.score_traj,
)
loss = total_from_weighted(loss_dict, w)   # total_from_weighted skips 0-weight terms
```

(`total_from_weighted` already drops zero-weighted terms — `loss_schedule.py`.)

### 2.4 Code to add (Stage-2 "No pretrain" = random encoder)

The Stage-2 trainer loads the encoder via `load_traj_encoder("full", stage1_ckpt, ...)`
(`train_merge_lora_temporal_no_kd.py:120`), which always calls `model.load_state_dict(state)`.
Add a `random_init` path that still **infers architecture flags from the ckpt** (so the score
tensor shapes match the real encoder) but keeps random weights:

```python
# train_merge_lora_temporal_no_kd.py :: load_traj_encoder(...)
def load_traj_encoder(model_type, stage1_ckpt, device, n_vis_keyframes, random_init=False):
    ...  # (build `model` exactly as now, flags inferred from `state`)
    if not random_init:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[TrajEncoder] loaded {model_type} from {stage1_ckpt}")
    else:
        print(f"[TrajEncoder] RANDOM-INIT {model_type} (arch flags from {stage1_ckpt}, weights NOT loaded)")
    return model
```

Then expose it on the Stage-2 trainer. In `train_visionzip_complement_lora.py` `parse_args()`:

```python
p.add_argument("--random-encoder", action="store_true",
               help="Table 6 'No pretrain': build the trajectory encoder with random weights "
                    "(architecture inferred from --stage1-ckpt) instead of loading pretrained weights.")
```

and at the load site (`train_visionzip_complement_lora.py:589`):

```python
encoder = load_traj_encoder("full", args.stage1_ckpt, device, args.n_vis_keyframes,
                            random_init=args.random_encoder)
```

(`--stage1-ckpt` is still passed for the random row — only to read the architecture flags; its
weights are ignored.)

### 2.5 Commands

**Stage-1 pretraining (2 new encoders; "All losses" reuses the existing `$S1`).**
Mirror `launch_stage1_tas_3way.sh`; only the loss flags change.

```bash
# --- "Only score loss" encoder (drop traj + future + score_traj; keep score_past) ---
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29827 \
  -m TrajGaze_v2.training.stage1_temporal \
  --output-dir $S1ROOT/stage1_scoreonly_overlay \
  --epochs 100 --lr 3e-4 --batch-size 2 \
  --use-egovqa --use-hd-epic --use-trajectory-anchor \
  --drop-loss-traj --drop-loss-score-fut --drop-loss-score-traj
#  → produces $S1ROOT/stage1_scoreonly_overlay/best.pth

# --- "No pretrain": no Stage-1 run needed (handled at Stage-2 via --random-encoder) ---
```

**Stage-2 LoRA (3 rows). Same protocol; only the encoder source differs.**

```bash
S1_SCORE=$S1ROOT/stage1_scoreonly_overlay/best.pth

# Row: No pretrain (random encoder; --stage1-ckpt only supplies arch flags)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29631 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt $S1 --random-encoder \
  --content-ratio 0.07 --traj-ratio 0.03 --source both --no-hdepic \
  --epochs 2 --lr 1e-4 --grad-accum 4 --early-stop \
  --output-dir $CKPT/tab6_nopretrain_overlay

# Row: Only score loss
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29632 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt $S1_SCORE \
  --content-ratio 0.07 --traj-ratio 0.03 --source both --no-hdepic \
  --epochs 2 --lr 1e-4 --grad-accum 4 --early-stop \
  --output-dir $CKPT/tab6_scoreonly_overlay

# Row: All losses = current \sys — reuse the existing main \sys StreamGaze result
#   (its encoder is $S1). Only run this if you want a fresh number under identical epochs.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29633 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --complement-mode topk --stage1-ckpt $S1 \
  --content-ratio 0.07 --traj-ratio 0.03 --source both --no-hdepic \
  --epochs 2 --lr 1e-4 --grad-accum 4 --early-stop \
  --output-dir $CKPT/tab6_alllosses_overlay
```

Read the SG per-task numbers from each run's own end-of-epoch `--source sg` eval, or re-measure
with the §0 eval command. Take best-of-2 per row.

### 2.6 Result table to fill (StreamGaze EGTEA, n=526)

| Stage-1 objective | GSM | NFI | OTP | SR | OAR | OI-E | OI-H | FAP | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| No pretrain       |  |  |  |  |  |  |  |  |  |
| Only score loss   |  |  |  |  |  |  |  |  |  |
| All losses (\sys) |  |  |  |  |  |  |  |  |  |

---

## 3. Table 7 — spatial vs. temporal selection (`tab:spatiotemporal`)

**Claim of the table:** the 10% budget must be free to concentrate in **both** space and time.
Constraining the geometry (forcing all frames, or forcing whole frames) hurts. Same frozen
encoder ($S1), same LoRA protocol; **only the geometric constraint on which tokens may be kept
changes.** The scoring is held fixed so the row is a clean geometry ablation.

Let `s(f,p) = norm(attn) + norm(s_traj)` be the per-token fused importance (attention +
frozen-trajectory salience; the same two signals \sys already uses, combined as in the existing
`fusion` mode, `train_visionzip_complement_lora.py:395-405`). Budget `k = round(0.10·N)`.

| Row | Geometry | Selection rule (fixed score `s`, budget `k`) |
|---|---|---|
| **No spatial** (keep whole frames) | temporal-only | rank frames by `Σ_p s(f,p)`; keep the top `⌈0.10·T⌉` frames **in full** (all `n_spatial` patches), drop the rest |
| **No temporal** (retain all frames) | spatial-only | in **every** frame keep its top `⌈0.10·n_spatial⌉` patches by `s`; no frame is dropped |
| **Spatio-temporal** = \sys | free | pick top-`k` tokens by `s` globally (space & time unconstrained) |

Both constrained rows keep the identical budget `k` (frame/patch counts round to ≈0.10·N). "No
temporal" is the existing `coverage` intuition (per-frame floor, `_coverage_complement`,
`train_visionzip_complement_lora.py:292-374`) generalized to the **whole** budget rather than the
3% pool.

### 3.1 Code to add (selection-geometry dispatch)

Add a CLI arg to `train_visionzip_complement_lora.py` `parse_args()`:

```python
p.add_argument("--select-geom", choices=["spatiotemporal", "no_spatial", "no_temporal"],
               default="spatiotemporal",
               help="Table 7 geometry ablation. Fixed fused score s=norm(attn)+norm(traj); "
                    "'no_spatial' keeps whole top frames, 'no_temporal' keeps top patches per frame, "
                    "'spatiotemporal' = free global top-k (= current \\sys geometry).")
```

Handle it at the **top** of `select_complementary(...)` (before the existing content/complement
logic, `train_visionzip_complement_lora.py:390`). Compute `s` once from cached signals, apply the
geometric constraint, and return `(video_embeds[idx], idx)` sorted — same return contract as the
other branches:

```python
geom = hp.get("select_geom", "spatiotemporal")
if geom in ("no_spatial", "no_temporal"):
    N = video_embeds.shape[0]
    T = int(cached["grid_thw"][0, 0].item())
    n_spatial = N // max(1, T)
    traj = _traj_scores(cached, item, device, mode, encoder, hp).to(attn_scores.device)
    s = _norm_scores(attn_scores, "minmax") + _norm_scores(traj, "minmax")      # (N,)
    k = max(1, round((content_ratio + traj_ratio) * N))
    if geom == "no_spatial":                       # keep whole top frames
        n_keep_f = max(1, round((content_ratio + traj_ratio) * T))
        fw = s.view(T, n_spatial).sum(dim=1)                                    # (T,)
        frames = torch.topk(fw, n_keep_f).indices
        base = (frames * n_spatial).view(-1, 1)
        idx = (base + torch.arange(n_spatial, device=s.device)).view(-1)
    else:                                          # no_temporal: top patches per frame
        per_f = max(1, round((content_ratio + traj_ratio) * n_spatial))
        s2d = s.view(T, n_spatial)
        cols = torch.topk(s2d, min(per_f, n_spatial), dim=1).indices            # (T, per_f)
        rows = torch.arange(T, device=s.device).view(-1, 1)
        idx = (rows * n_spatial + cols).view(-1)
    idx = idx.sort().values
    return video_embeds[idx], idx
# else: fall through to the existing spatiotemporal (\sys) path unchanged
```

Plumb `hp["select_geom"] = args.select_geom` where the other `hp` fields are set in `main()`
(search for the `hp = {...}` dict; the anticipatory hyperparameters are assembled there), and pass
`args.select_geom` through the `evaluate(...)` call the same way `complement_mode` is passed
(`train_visionzip_complement_lora.py:526-530`). If the grid isn't a clean `T·n_spatial` (rare
non-square layouts), fall back to the global `spatiotemporal` path (guard with
`if T*n_spatial==N`), exactly as `coverage` does at `:477`.

> Note on the Spatio-temporal row: the cleanest 3-way uses `s`-global-top-k for all three rows
> (identical scoring, only geometry differs). That makes the Spatio-temporal control a
> single-fused-pool number, which can differ slightly from the two-pool \sys headline (C∪G).
> Decide in the caption whether the "Spatio-temporal" row is (i) the `s`-global control (clean
> 3-way, recommended) or (ii) the real two-pool \sys result. Do **not** mix: if the constrained
> rows use fused `s`, the free row should too.

### 3.2 Commands (same frozen encoder $S1, same protocol)

```bash
# Row: No spatial (whole frames)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29641 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt $S1 --select-geom no_spatial \
  --content-ratio 0.07 --traj-ratio 0.03 --source both --no-hdepic \
  --epochs 2 --lr 1e-4 --grad-accum 4 --early-stop \
  --output-dir $CKPT/tab7_nospatial_overlay

# Row: No temporal (all frames)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29642 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt $S1 --select-geom no_temporal \
  --content-ratio 0.07 --traj-ratio 0.03 --source both --no-hdepic \
  --epochs 2 --lr 1e-4 --grad-accum 4 --early-stop \
  --output-dir $CKPT/tab7_notemporal_overlay

# Row: Spatio-temporal control (fused s, global top-k)  [omit if using the real \sys headline]
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29643 \
  -m TrajGazeMerge.training.train_visionzip_complement_lora \
  --traj-pool-mode learned --stage1-ckpt $S1 --select-geom spatiotemporal \
  --content-ratio 0.07 --traj-ratio 0.03 --source both --no-hdepic \
  --epochs 2 --lr 1e-4 --grad-accum 4 --early-stop \
  --output-dir $CKPT/tab7_spatiotemporal_overlay
```

### 3.3 Result table to fill (StreamGaze EGTEA, n=526; OTP dropped as in the paper — 7 tasks)

| Selection | GSM | NFI | SR | OAR | OI-E | OI-H | FAP | **Avg** |
|---|---|---|---|---|---|---|---|---|
| No spatial       |  |  |  |  |  |  |  |  |
| No temporal      |  |  |  |  |  |  |  |  |
| Spatio-temporal  |  |  |  |  |  |  |  |  |

---

## 4. Column ↔ internal task-key mapping

`evaluate(..., source="sg")` returns `per_task` keyed by these internal names
(`TrajGazeMerge/data/dataset.py:37-46`, `MCQ_TASKS`). Map to the paper columns:

| Paper column | Internal task key |
|---|---|
| GSM  | `past_gaze_sequence_matching` |
| NFI  | `past_non_fixated_object_identification` |
| OTP  | `past_object_transition_prediction` |
| SR   | `past_scene_recall` |
| OAR  | `present_object_attribute_recognition` |
| OI-E | `present_object_identification_easy` |
| OI-H | `present_object_identification_hard` |
| FAP  | `present_future_action_prediction` |

OTP is the 8th task; the paper reports it in Table 6 but drops it in Table 7 (and in the main
results). "Avg" in the paper = the per-task-averaged accuracy printed as `Overall:` for the SG
source (n=526), not a task-macro average — confirm which your caption means and compute
consistently across rows.

---

## 5. Caveats and decisions (resolve before you start)

1. **LoRA rank.** Code uses **r=16/α=32** (`train_visionzip_lora.py:54-55`); the paper text says
   r=64/α=128. Either (a) fix the paper to r=16/α=32, or (b) bump `LORA_RANK/LORA_ALPHA` and re-run
   **everything** (main results too) — do not run Tables 6/7 at a different rank than the main table.
2. **Epochs / best-of-N.** Constant LR ⇒ ep1==1-epoch. Recent \sys ablations used `--epochs 1` or
   `2` (best-of-2). Use the **same** policy for every row of a table; state it in the caption.
3. **Train source.** The main \sys model is **joint** (`--source both`); Tables 6/7 evaluate the
   StreamGaze slice (`--source sg`). Above trains joint + evals SG (faithful). Cheaper alternative:
   train `--source sg` (SG-only) — but that changes the model, so don't mix joint and SG-only rows.
4. **Reuse the shared row.** Table 6 "All losses" and Table 7 "Spatio-temporal" are both the \sys
   model. You only strictly need to run the **2 new encoders** (Table 6) and **2 new geometries**
   (Table 7); the third row can reuse the main \sys StreamGaze result — as long as epochs/rank/source
   match. Re-run it fresh only if those don't match.
5. **Table 7 scoring confound.** Keep the fused score `s` identical across all three rows so the
   only variable is geometry (see §3.1 note). If the Spatio-temporal row uses the real two-pool C∪G
   while the constrained rows use fused `s`, that's a confound — avoid it.
6. **Checkpoint hygiene.** All output dirs above use fresh `tab6_*`/`tab7_*`/`stage1_scoreonly_*`
   names. Do **not** overwrite `$S1` or the joint/main-method checkpoints.
7. **`--early-stop` at `--epochs 2`** only prints a cosmetic "skipping epoch 3" line; both epochs
   still run. Drop it if using `--epochs 1`.
8. **No fabricated numbers.** Fill the tables only from real runs. The stale values currently in
   `main.tex` Tables 6/7 are from the old method and must not be carried over.

## 6. Compute budget (rough)

- Table 6: **2** Stage-1 pretrains (4-GPU, 100 ep each) + **2–3** Stage-2 LoRA runs (2-GPU, ≤2 ep).
- Table 7: **2–3** Stage-2 LoRA runs (2-GPU, ≤2 ep) — **no** new Stage-1 (reuses `$S1`).
- Two Stage-2 runs are co-resident on 4 GPUs (0,1 + 2,3), as in the example commands.
