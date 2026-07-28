# Re-running Table 6 (Stage-1 pretraining objectives) and Table 7 (spatial vs. temporal selection) for the current \sys method

**Status: both tables are measured.** Results in **§2.6** (Table 6) and **§3.3** (Table 7),
run log and gates in **§0a**. Regenerate at any time with
`source env.sh && python scripts/collect_ablation_tab6_tab7.py`.

**Audience:** whoever re-runs these two ablations, or defends them in review.
**Why this exists:** `paper/main.tex` describes the *current* method — VisionZip-complement
selection at a 10% visual-token budget (7% VZ content ∪ 3% frozen-trajectory complement) — while
the numbers sitting in its Tables 6 and 7 came from an earlier method variant (their
"Spatio-temporal / All-losses" rows land at ~67–68%, against the current \sys StreamGaze ~69–70%).
This document defines what each row means for the current method, records the code that was added,
the commands as actually run, and the measured results.

**`paper/main.tex` has not been updated yet** — it still holds the old-method values, and the
`paper/` tree is not in this repository. Replacing those two tables with §2.6 / §3.3, plus §0a's
five caption disclosures, is the one remaining task.

> Sections §2.3, §2.4 and §3.1 are written as "code to add". That code **is already merged**
> (commit `9c6a85a`); those blocks are kept as the specification the implementation was checked
> against, and §0a lists the two places the implementation deliberately deviates from them.

---

## 0a. STATUS — resolved decisions and machine-specific corrections (2026-07-27, updated 2026-07-28)

The code this document asks for **has been implemented** (commit `9c6a85a`). The
decisions left open below (§5) were resolved as follows; where this section disagrees with the
rest of the document, **this section wins**.

**Run status (2026-07-28 08:25) — COMPLETE.** Stage-1 for "Only score loss" is done
(`TrajGaze_v2/checkpoints/stage1_scoreonly_overlay/best.pth`, 100 ep, exit=0; the epoch-100 log
line confirms the intended composition, `loss 0.0144 = score_past 0.0006 + score_traj 0.0137`,
i.e. `traj` and `score_fut` are zeroed). All four Stage-2 rows finished and passed the gates —
`exit=0`, `n=526` on `per_src.sg`, and the expected `pct_kept` per geometry. Results are in §2.6
and §3.3; `scripts/collect_ablation_tab6_tab7.py` exits 0 and regenerates both tables.

| row | overall | macro-7 | kept | log |
|---|---|---|---|---|
| `tab6_nopretrain_overlay` | 65.02 | 64.55 | 9.99% | `tab6_nopretrain_overlay.log` |
| `tab6_scoreonly_overlay` | 66.92 | 65.83 | 9.99% | `tab6_scoreonly_overlay.log` |
| `tab7_nospatial_overlay` | 62.93 | 61.88 | 9.38% | `tab7_nospatial_overlay.log` |
| `tab7_notemporal_overlay` | 67.30 | 66.19 | 9.91% | `tab7_notemporal_overlay.log` |

Every row is a **single run evaluated once**. §8 of `kd_handoff_v2.md` puts the eval noise floor at
3–4 items (0.6–0.8 macro points) with per-task columns swinging up to 2.95, so gaps below ~1 point
carry no information — see the readings in §2.6 and §3.3 for which comparisons survive that.

> **Incident — node reprovision, 2026-07-27 23:59.** `/NHNHOME/VILAB` is a symlink on node-local
> nvme pointing at the lustre root `/NHNHOME/WORKSPACE/26msit001_A`. The node was reprovisioned
> around midnight, wiping the local disk and the alias with it, which killed `tab6_nopretrain` at
> step 2280/2900 (no checkpoint; artifacts kept as `*.dead-20260727`) and left `$REPO`, `$DATA`,
> the venv on `$PATH`, `$HF_HOME`, and the four in-repo checkpoint symlinks dangling. No data was
> lost — everything lives on lustre. `env.sh` now recreates the alias if it is missing, so a
> future reprovision self-heals. Long runs must be launched with `setsid` as well: an earlier run
> (`kd_train_sgonly_nooverlay.log`) died on `SignalException: got signal: 15` when its launching
> session ended.

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
`scripts/collect_ablation_tab6_tab7.py` assembles both tables. **Launch with `setsid nohup … &`**
— a plain background launch dies with the session (see the incident note above).

> **Second incident — cold `torch.hub` cache, 2026-07-28 00:22.** With `$HOME/.cache` wiped by the
> same reprovision, both DDP ranks re-downloaded DINOv2 simultaneously and one lost the extract
> race in `torch.hub._get_cache_or_reload`: `OSError: [Errno 39] Directory not empty: 'dinov2'`.
> It killed the first row 33 s in; the driver moved on and the rest ran normally on the
> now-warm cache. `env.sh` pins `TORCH_HOME` to lustre so the cache outlives a reprovision, and
> `scripts/rerun_tab6_nopretrain.sh` re-runs the lost row after the driver's other rows finish.
> Only a *cold* cache races — a warm one needs no download.

---

## 0. Environment and shared protocol (read once)

The `/workspace/…` paths and the `/opt/conda/envs/trajgaze` interpreter this section originally
listed are from machine 1 and do not exist here. On this machine everything comes from `env.sh`:

```bash
cd /NHNHOME/VILAB/vilab_yj/trajgaze && source env.sh
# sets REPO, DATA, SG_ROOT/EG_ROOT/HD_ROOT, GAZE_OVERLAY=1, PATH (venv), TORCH_HOME,
#      HF_HOME, TORCHRUN, and STAGE1_CKPT / M1_JOINT / M1_SGONLY / M1_EGONLY
unset VLM_GAZE_OVERLAY        # env.sh does not clear it; a stale 0 voids every run (§0a)

CKPT=$REPO/TrajGazeMerge/checkpoints      # Stage-2 LoRA output root
S1ROOT=$REPO/TrajGaze_v2/checkpoints      # Stage-1 encoder output root
S1=$STAGE1_CKPT                           # "All losses" encoder (stage1_tas_3way_overlay/best.pth)
```

`$TORCHRUN` is `python -m torch.distributed.run`, not the `torchrun` shim — bare `torchrun`
resolves to the system python, which has no `peft`. Commands below that still say `torchrun`
were written for machine 1; use `$TORCHRUN`.

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
Pick one policy and apply it to *every* row of a table so rows are comparable.
**What was run: `--epochs 1` for all four ablation rows** (§0a). The \sys row is *not* on that
policy — it is the epoch-2 best-of-2 specialist — which is disclosure item (1) in §0a's caption
list, not something to paper over.

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
| **Only score loss** | Stage-1 with patch-score regression only | `loss_score_past` **+ `loss_score_traj`** (drop `loss_traj`, `loss_score_fut`) |
| **All losses** = \sys | current full Stage-1 | all four (this is the existing `$S1`) |

> **Resolved (§0a).** This section originally recommended dropping `loss_score_traj` too. That is
> wrong: `score_head` — the only head Stage-2 consumes (`model_temporal.py:281`) — is supervised by
> `loss_score_traj` alone, so the minimal version would have shipped a randomly initialised
> inference head and collapsed this row onto "No pretrain". The row as run keeps
> `loss_score_past + loss_score_traj`, which the epoch-100 log confirms
> (`loss 0.0144 = 0.0006 + 0.0137`). State it this way in the caption.

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

**Stage-1 pretraining (1 new encoder; "All losses" reuses the existing `$S1`, "No pretrain" needs
no Stage-1 at all — it is handled at Stage-2 by `--random-encoder`).**

As run on this machine, 2026-07-27 22:02–22:32 (`stage1_scoreonly.log`, exit=0, 100 ep, 30 min):

```bash
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29827 \
  -m TrajGaze_v2.training.stage1_temporal \
  --output-dir $S1ROOT/stage1_scoreonly_overlay \
  --epochs 100 --lr 3e-4 --batch-size 4 \
  --use-trajectory-anchor \
  --drop-loss-traj --drop-loss-score-fut
#  → $S1ROOT/stage1_scoreonly_overlay/best.pth
```

Three deliberate differences from `launch_stage1_tas_3way.sh`, all forced by this machine (§0a):
2 GPUs × batch 4 instead of 4 × 2 (same eff-batch 8); **no `--use-egovqa --use-hd-epic`** because
`dataset_temporal_{egovqa,hdepic}.py` do not exist on this branch, so the encoder trains on
StreamGaze only (246 clips) while `$S1` saw all three corpora — **this is caption disclosure (2)**;
and `--drop-loss-score-traj` is *not* passed, per §2.2.

Verify from the log rather than trusting the flags: the epoch-100 line must read
`loss ≈ score_past + score_traj` with `traj`/`score_fut` excluded from the sum. The resulting
encoder must also load into Stage-2 with no missing keys — `tab6_scoreonly_overlay.log` shows
`[TrajEncoder] loaded full` and the same inferred architecture flags as `$S1`
(`use_trajectory_anchor=True`, all other branch flags False).

**Stage-2 LoRA (2 rows; "All losses" reuses `$M1_SGONLY` per §0a).** Do not run these by hand —
`scripts/run_ablation_tab6_tab7.sh` holds the protocol in one `run_row()` so no row can silently
differ from another, which is the whole point of an ablation table. It is reproduced here only to
show what each row is:

```bash
# shared by every row of BOTH tables
--traj-pool-mode learned --complement-mode topk \
--content-ratio 0.07 --traj-ratio 0.03 \
--source sg --no-hdepic --epochs 1 --lr 1e-4 --grad-accum 4     # 2-GPU DDP, eff-batch 8

# Row: No pretrain   (--stage1-ckpt supplies architecture flags only; weights ignored)
--stage1-ckpt "$STAGE1_CKPT" --random-encoder    --output-dir $CKPT/tab6_nopretrain_overlay
# Row: Only score loss
--stage1-ckpt "$S1ROOT/stage1_scoreonly_overlay/best.pth" \
                                                 --output-dir $CKPT/tab6_scoreonly_overlay
```

Two corrections to what this section said before: rows train **`--source sg`**, not `--source both`
(§0a), and at **1 epoch** without `--early-stop`, so there is no best-of-N to take. Each row's own
end-of-epoch eval is the number — it is already filtered to `--source sg` and already writes
`per_task` to `train_log_rank0.jsonl`, so no separate `--eval-ckpt` pass is needed.

### 2.6 Result — measured 2026-07-28 (StreamGaze EGTEA, n=526)

OTP is **not** a column here — §0a drops it from *both* tables (n=2 of 526, so it only ever reads
0/50/100%). Avg is the macro-average over the 7 columns below, as emitted by
`scripts/collect_ablation_tab6_tab7.py`.

| Stage-1 objective | GSM | NFI | SR | OAR | OI-E | OI-H | FAP | **Avg** |
|---|---|---|---|---|---|---|---|---|
| No pretrain       | 60.94 | 61.76 | 64.86 | 87.50 | 65.35 | 62.50 | 48.94 | **64.55** |
| Only score loss   | 60.94 | 64.71 | 59.46 | 88.54 | 68.32 | 65.62 | 53.19 | **65.83** |
| All losses (\sys) | 71.36 | 63.24 | 57.66 | 93.40 | 73.27 | 74.48 | 56.38 | **69.97** |

Measured 2026-07-28. Both ablation rows are single 1-epoch runs, exit=0, n=526, `pct_kept` 9.99%;
item-level `Overall:` was 65.02 / 66.92. The \sys row is the mean of 3 evals of `$M1_SGONLY`.

**Reading.** The ordering the table needs holds — random 64.55 < score-only 65.83 < all-losses
69.97 — but only the gap to \sys is comfortably outside the noise floor (**+5.42** and **+4.14**
macro-7 points). **"No pretrain" vs "Only score loss" is 1.28 points and is *not* evidence**: they
are identical on GSM (60.94 both) and trade SR against OI-E/FAP, and SR alone swings 2.70 across
re-evals of identical weights (§8 of `kd_handoff_v2.md`). Do not claim a monotone three-way
ordering from single runs; claim that pretraining with the full objective beats both alternatives.

The \sys advantage is concentrated where the encoder is supposed to matter: **GSM +10.42** over
both ablated encoders, plus OI-H +8.86 / OI-E +4.95 over score-only. Note "No pretrain" wins SR
(64.86, the best of the three) — with 37 items that is one or two questions and should not be
bolded or discussed.

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

> **Resolved (§0a): the free row is the real two-pool \sys, not an `s`-global control.** That makes
> the scoring differ between the constrained rows (fused `s`, raw tokens only) and the free row
> (VisionZip content ∪ trajectory complement, with the contextual merge) — a known confound, and
> **caption disclosure (3)**. It was chosen so the table's third row is the deployed system rather
> than a control that appears nowhere else in the paper. If a reviewer objects, the clean 3-way is
> one extra row: `--select-geom spatiotemporal` with the same encoder and protocol.

### 3.2 Commands (same frozen encoder `$STAGE1_CKPT`, same protocol as §2.5)

Run via `scripts/run_ablation_tab6_tab7.sh`; the row-specific flags are the only difference:

```bash
# Row: No spatial (keep whole top frames)
--stage1-ckpt "$STAGE1_CKPT" --select-geom no_spatial   --output-dir $CKPT/tab7_nospatial_overlay
# Row: No temporal (keep top patches in every frame)
--stage1-ckpt "$STAGE1_CKPT" --select-geom no_temporal  --output-dir $CKPT/tab7_notemporal_overlay
# Row: Spatio-temporal = \sys — not re-run; reuses $M1_SGONLY's 3 existing evals (§0a)
```

The geometry actually fired if `pct_kept` in `train_log_rank0.jsonl` sits at 9.38% (`no_spatial`,
exact because whole frames divide evenly) or ~9.9–10.2% (`no_temporal`). A row logging 10.0%
throughout is running the unconstrained selector and is not the experiment it claims to be.

### 3.3 Result — measured 2026-07-28 (StreamGaze EGTEA, n=526; OTP dropped — 7 tasks)

| Selection | GSM | NFI | SR | OAR | OI-E | OI-H | FAP | **Avg** |
|---|---|---|---|---|---|---|---|---|
| No spatial       | 57.81 | 61.76 | 56.76 | 89.58 | 59.41 | 57.81 | 50.00 | **61.88** |
| No temporal      | 73.44 | 63.24 | 51.35 | 90.62 | 65.35 | 67.19 | 52.13 | **66.19** |
| Spatio-temporal (\sys) | 71.36 | 63.24 | 57.66 | 93.40 | 73.27 | 74.48 | 56.38 | **69.97** |

Both constrained rows are single 1-epoch runs, exit=0, n=526. Logged `pct_kept` averaged **9.38%**
(`no_spatial`) and **9.91%** (`no_temporal`) over training — `no_spatial` is exactly the uniform-grid
figure in §0a because whole frames divide evenly, while `no_temporal` runs slightly under §0a's
10.19% because that figure is for the uniform N=13824 grid and per-clip grids vary.
The \sys row is the mean of 3 evals of `$M1_SGONLY` (§8 of `kd_handoff_v2.md`),
which is a 2-epoch best-of-2 model — the training budget favours it, as §0a's caption rule requires
disclosing. Item-level `Overall:` for the two constrained rows was 62.93 / 67.30.

**Reading.** Both constraints cost real accuracy (−8.09 / −3.78 macro-7 against a 0.6–0.8-point
noise floor), so the table's claim — the budget must be free in space *and* time — holds, and
forcing whole frames is the worse of the two constraints by 4.31 points.

Two things the caption should not overstate:

- **`no_temporal` beats \sys on GSM** (73.44 vs 71.36, +2.08). Keeping every frame and taking each
  frame's top patches is *better* for gaze-sequence matching than the deployed selector. The row's
  deficit is concentrated in OI-E (−7.92) and OI-H (−7.29) instead.
- **SR is not evidence.** Its 37 items make one question worth 2.70%, and §8 measures SR swinging
  2.70 across re-evals of identical weights, so the 51.35 / 57.66 gap is inside the floor.

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

OTP is the 8th task. **Resolved (§0a): it is dropped from *both* tables** — 2 items of 526, so it
can only ever read 0/50/100%. It is still measured and still sits inside the item-level `Overall:`.

**Avg is the macro-average over the 7 columns above**, computed by
`scripts/collect_ablation_tab6_tab7.py`, *not* the `Overall:` line. The two differ by roughly a
point because the columns have very unequal n (OI-E 101 vs SR 37) — e.g. `no_spatial` reads
macro-7 61.88 against `Overall:` 62.93. Quote one or the other consistently; the tables here are
macro-7 throughout, including the \sys row.

---

## 5. Caveats and decisions — all resolved, kept for the reasoning

Every decision below was settled in §0a before the runs; the resolution is repeated here so this
list is not read as still-open. What each became:

1. **LoRA rank** → the code (r=16/α=32) is correct and `paper/method.tex` was fixed. Nothing re-run.
2. **Epochs / best-of-N** → `--epochs 1`, no `--early-stop`, for all four rows. The \sys row is a
   2-epoch best-of-2 model, which is why the caption must disclose the budget asymmetry.
3. **Train source** → `--source sg`. Every row trains *and* evaluates on StreamGaze only, so no row
   mixes a joint model with an SG-slice eval.
4. **Reuse the shared row** → done: Table 6 "All losses" and Table 7 "Spatio-temporal" are both the
   mean of `$M1_SGONLY`'s 3 existing evals, not a fresh run.
5. **Table 7 scoring confound** → knowingly accepted, see §3.1. The free row is the deployed
   two-pool selector while the constrained rows use fused `s`; this is caption disclosure (3).
6. **Checkpoint hygiene** → held. `$S1` and the main-method checkpoints were never written to; all
   output went to fresh `tab6_*` / `tab7_*` / `stage1_scoreonly_overlay` directories.
7. **`--early-stop`** → not passed, since the rows are 1 epoch.
8. **No fabricated numbers** → every figure in §2.6 and §3.3 comes from a gated run
   (exit=0, n=526, expected `pct_kept`). `main.tex` Tables 6/7 have **not** been touched yet — they
   still hold the old-method values and must be replaced with §2.6 / §3.3.

## 6. Compute budget — measured, this machine (2 × B200)

| stage | what it cost |
|---|---|
| Stage-1 "Only score loss" | 100 ep in **30 min** (2 GPUs, batch/GPU 4, SG-only 246 clips) |
| Stage-2 row, training | 2900 steps in **~1.75 h** (≈2.15 s/step) |
| Stage-2 row, end-of-epoch eval | **~18 min** (526 items) |
| **4 rows, serial** | **~8 h** wall clock (00:22 → 08:25) |

Rows are serial by necessity — 2 GPUs, so the "two co-resident 2-GPU runs" this document assumed
is not possible here. "No pretrain" is not cheaper than the others despite skipping Stage-1: the
random encoder still runs at Stage-2.
