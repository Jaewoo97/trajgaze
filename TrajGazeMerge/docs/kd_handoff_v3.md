# KD Handoff v3 — putting the selection inside the ViT

Written 2026-07-29. **Does not supersede `kd_handoff_v2.md`** — v2 remains the task
definition and the record of the KD student (predictor) line. v3 is a *new experiment
branching off it*: a KD-methodology ablation that asks whether the privileged gaze
signal can live in the ViT's own attention instead of in any extra module.

Everything in v2 still holds. In particular v2 §8 (eval is not deterministic, ±4 items),
v2 §10.4 (the warm-start confound) and v2 §12.5 (what not to claim) apply verbatim to
every number produced here.

**Status at time of writing: run 1 of 4 is training.** All numbers below marked *live*
are from a run in flight and will move.

---

## 1. The question

v2's method keeps 10% of the visual tokens: **7% VisionZip content ∪ 3% trajectory
complement**, where the complement is ranked by a gaze/hand salience field from the
frozen TAS encoder. At inference that costs an eye-tracker plus a 36.85M encoder run
over every frame (v2 §12.3a). The KD student removes the eye-tracker by predicting the
complement with a 3.95M RGB head — but a module is still there.

VisionZip's dominant score *is* the last ViT block's attention column-sum. So there is a
third option nobody has tested:

> **Teach that score to rank M1's chosen 10% at the top, and the method collapses to
> plain VisionZip at inference — zero extra parameters, no eye-tracker, no TAS encoder.**

That is the whole experiment. It is an ablation of the KD methodology, not a replacement
for it: it asks *where* the distilled knowledge should live, not whether distillation works.

| system | selection at eval | gaze at test | extra params at inference |
|---|---|---|---|
| M1 teacher | 7% VZ ∪ 3% TAS | **yes** | 36.85 M |
| KD student (v2 §7.7) | 7% VZ ∪ 3% predictor | no | 3.95 M |
| **ViT-KD (this document)** | **pure VisionZip on a distilled ViT** | no | **0** |
| VisionZip content-only | pure VisionZip, stock ViT | no | 0 |

### 1.1 Bars to beat (v2, overlay-free unless stated)

| setting | bar | source |
|---|---|---|
| SG raw video | **360 items** (68.44%) | v2 §7.7 |
| SG overlay | **369 items** (70.15%) | v2 §2.2a |
| EG raw video | **268 items** (55.26%) | v2 §7.7 |
| EG overlay | **272 items** (56.08%) | v2 §10.3 |

1 item = 0.19% on SG (n=526), 0.21% on EG (n=485). Noise floor ±4 items (v2 §8).

---

## 2. Approach

### 2.1 Target — M1's exact 10%, frozen

```python
_, content_idx, avail_idx = content_and_avail(cached_frozen, 0.07)   # v2's helper, verbatim
s_teacher  = _traj_scores(cached_frozen, item, device, "learned", encoder, hp)
traj_idx,_ = topk_in_avail(s_teacher, avail_idx, int(0.03 * N))
S_T        = cat([content_idx, traj_idx]).sort().values              # the 10%
```

Computed with the **frozen** ViT every step. If the target moved with the adapter the
objective would be self-referential and could be satisfied by collapsing the score.

The three helpers are reused unmodified from the v2 trainers, so the target is
bit-for-bit the selection M1 actually makes.

### 2.2 Student — the ViT's own attention, rank-8 LoRA on block 31

The score depends only on the last block's q,k. Adapting **block 31 alone** means blocks
0..30 stay bit-identical and the representation provably cannot drift there.

- `LoRALinear` on `visual.blocks[31].attn.{qkv,proj}`, r=8, α=16 → **61,440 trainable params**
- hand-rolled, not peft — see §4.2 for why
- LLM LoRA warm-started from `$M1_*` and **frozen**; no task CE, no VLM forward in Phase 1

### 2.3 Loss

```
z        = (attn_tuned - mean) / std                 # score is a column-sum, not a logit
L_sel    = BCE_with_logits(z, 1[t ∈ S_T], pos_weight = n_neg/n_pos, clamped 50)
L_anchor = 1 - mean cos(video_embeds_adapted, video_embeds_frozen)
L        = λ_sel · L_sel + λ_anchor · L_anchor        # both 1.0
```

BCE runs over **all N tokens**, not just the discarded pool. That is deliberate: content
tokens are positives too, so the adapter cannot promote gaze tokens by demoting content
tokens. The live numbers confirm it does not (§5.2 — `recall_P` rises alongside `recall_traj`).

### 2.4 Budget split at eval

M1's 10% is raw 6.5% (attn 3.5% + traj 3%) + **positional** contextual 3.5%. The
contextual centres are `arange(0, n_non_dom, step)` — index-spaced, so semantically
arbitrary. The target is still the full 10% (user decision), but the student's split is
measured two ways from **one** trained adapter:

| split | dominant | contextual | reading |
|---|---|---|---|
| **P (primary)** | 0.065 | 0.035 | matches the teacher's raw/merged mix; Phase 2 trains here |
| S (secondary) | 0.10 | 0 | pure-attention ceiling; contextual merging gone |

### 2.5 Two phases — a protocol change from v2, introduced here

- **Phase 1** `train_vit_selection_kd.py` — distil the selection. LLM frozen, so no VLM
  forward/backward. 2 epochs.
- **integrity gate** — must pass before Phase 2 (§3).
- **Phase 2** `train_visionzip_lora.py --vit-lora-ckpt` — ViT frozen, re-adapt the LLM
  readout to the new selection at the P split. 2 epochs, `--early-stop`.

> **v2's KD student does NOT do this.** It is a *single* run training both objectives at
> once — `loss = ce_loss + λ_sel · kd_loss`, one AdamW over two param groups (predictor at
> `--pred-lr`, LoRA at `--lr`). top-k is non-differentiable so the two gradients never mix,
> but they advance together. The two-phase split is **this document's change**, and it must
> be stated as such wherever the two rows are compared.

**Why split it.** The predictor and the ViT are not the same kind of intervention:

| | predictor (v2) | ViT (v3) |
|---|---|---|
| `video_embeds` | **fixed** — the predictor only emits scores | **change every step** |
| what the LLM sees move | the index set | the index set **and every feature** |

Training jointly would make the LLM chase a moving encoder — a problem that does not exist
in the predictor case. Three reasons follow:

1. **No moving target.** Phase 2's LLM adapts to a selector that is already final.
2. **The gate needs a finished ViT.** Under joint training a gate failure would throw away
   the readout training too.
3. **Cost.** Phase 1 runs no VLM forward/backward (v2 records `--freeze-lora` as "~3-5x
   faster per epoch" for the analogous predictor-only mode).

**What this preserves and what it does not.**
- *Preserved:* the LLM LoRA receives the same extra optimizer budget in both protocols —
  M1 warm-start + 2 epochs (725 steps/epoch on SG = 1450). So v2 §10.4's warm-start confound
  applies **equally** to both rows and does not favour either.
- *Not preserved:* v2's LoRA co-adapted **while** the predictor was still learning; v3's
  adapts to a **finished** selector. Whether co-adaptation helps or hurts is not knowable in
  advance — it could act as a regulariser or as noise. This is a genuine protocol difference,
  not a neutral implementation detail, and belongs in the paper's method description.

---

## 3. The integrity gate — "we fine-tuned the ViT without breaking it"

`scripts/vitkd_integrity_gate.py`. **Selection held fixed, features swapped:**

```
baseline : frozen ViT scores pick the tokens, frozen embeddings are sent to the LLM
tuned    : frozen ViT scores pick the SAME tokens, TUNED embeddings are sent
```

Any difference is representation drift in block 31 and nothing else. Comparing the two
end-to-end 10% numbers instead would confound drift with the selection change that is the
point of the experiment. `visionzip_select_tokens` takes `video_embeds` and
`(attn_scores, attn_key)` as separate arguments, so this is just a matter of which tensor
goes in which slot — and the contextual merge is redone with the tuned embeddings, which
is what actually reaches the LLM.

**Pass = |Δ| ≤ 4 items** (v2 §8). Exits 2 on failure, which stops the chain before Phase 2
spends four hours training a readout on a damaged encoder.

> **This replaces the 100%-token eval originally planned.** That also works, but it hands
> the 7B model ~13,800 visual tokens per item instead of ~1,380 (N measured in §5.1), so
> eight gate runs would have cost more than the experiment. The swap is cheaper *and* a
> cleaner isolation.

Smoke-tested on 24 SG items with a probe adapter: **Δ = +0 items, cos = 0.99254, PASS.**

If a final adapter ever fails, Phase 1 checkpoints every 200 steps — gate an earlier
checkpoint rather than retraining.

---

## 4. Two defects in the existing code, found while building this

### 4.1 `CombinedSimpleDataset` ignored `VLM_GAZE_OVERLAY` — on both sources

- SG: `train_autogaze_lora.py:59` defined its own `_SG_FRAME_SUB` from `GAZE_OVERLAY` only
- EG: `combined_simple_dataset.py:43` used `_EG_FRAME_SUB`, not `_EG_VLM_FRAME_SUB`

`train_visionzip_lora.py` — the Phase 2 trainer, and the VisionZip baseline trainer — uses
that dataset. So `VLM_GAZE_OVERLAY=0` was **silently ignored** and the model trained and
evaluated on overlay frames. No shape change, no error, nothing visible in the accuracy:
exactly the failure class v2 §7.3's stream assertion exists to catch, in the one trainer
that lacked it. (`run_ablation_tab6_tab7.sh:20`'s defensive `unset VLM_GAZE_OVERLAY` was a
workaround for this without naming it.)

**Consequence had it not been found:** this document's raw-video and overlay settings would
have read identical frames and produced two rows that differ only by noise, inviting the
conclusion "the overlay does not matter" from a bug.

Fixed; both sources verified to switch (`viz↔original`, `gaze↔no_gaze`) with item counts
unchanged (526 + 485 = 1011). `train_visionzip_lora.py` now prints the resolved directory
at startup and raises on a mismatch.

#### 4.1a No v1/v2 number is affected — audited, not assumed

The defect was **latent**: it lived in a code path that no overlay-free run has ever used.
Four independent checks, because "probably fine" is not good enough for a result that would
silently invalidate the paper's `KD (raw video)` row.

| # | check | result |
|---|---|---|
| 1 | **Which dataset does each trainer use?** | `train_visionzip_kd_lora` (KD student) and `train_visionzip_complement_lora` (M1 teacher) both use `CombinedMergeDataset` → `data/dataset.py` / `data/egogaze_dataset.py`, which **do** honour `VLM_GAZE_OVERLAY`. The defect is confined to `CombinedSimpleDataset`. |
| 2 | **Which launcher ever set `VLM_GAZE_OVERLAY=0`?** | Exactly two — `run_kd_sg_nooverlay.sh`, `run_kd_eg_nooverlay.sh` — and both invoke `train_visionzip_kd_lora`, i.e. the correct path. Every other launcher `unset`s it. |
| 3 | **What did the overlay-free runs actually read?** | The logs record it: `kd_train_sgonly_nooverlay.log` → `student VLM='original' teacher TAS='viz'`; `kd_train_egonly_nooverlay.log` → `student VLM='no_gaze' teacher TAS='gaze'`. Had the defect applied, either `viz`/`gaze` would appear here or v2 §7.3's assertion would have killed the run. |
| 4 | **§7.2's 354-item off-distribution eval?** | Ran with **`GAZE_OVERLAY=0`**, not `VLM_GAZE_OVERLAY=0` (log header). That switches *both* variants, so it resolves to `original` on either code path. Safe. |

Plus a structural argument: v2 §9 lists the no-overlay VisionZip bar as "Explicitly dropped
— user decision", so the one experiment that would have driven the defective path with
overlay-free frames was never run.

**Therefore v2 §7.7 (360 / 268), §7.2 (354), §2.2a (369) and §10.3 (272) all stand**, and
the paper's `KD (raw video)` row is genuinely raw video.

The first thing ever to drive `train_visionzip_lora.py` with `VLM_GAZE_OVERLAY=0` is **this
document's Phase 2** — which is precisely settings 1 and 2 of §6.

### 4.2 The VisionZip score is computed without the `cu_seqlens` mask

`qwen2_5vl_visionzip.py:216-243` (flash path) computes the score as a column-sum over the
**full T×T softmax**, while the actual attention output uses
`flash_attn_varlen_func(cu_seqlens=…)` — i.e. **block-diagonal per frame**. The eager path
(`:299-310`) does mask, so the two implementations disagree.

Upstream VisionZip behaviour, harmless for LLaVA (one image = one segment). **Not fixed** —
every measurement in v1 and v2 sits on this definition and changing it would void them all.
Recorded because it is the exact quantity this experiment backpropagates into: the score
being distilled is a *cross-frame* global saliency, not a per-frame one.

---

## 5. Measurements taken before committing GPU time

### 5.1 Step 0 (`scripts/measure_vitkd_step0.py`)

| | |
|---|---|
| score-loop refactor | **bit-identical** — synthetic (T=4096, 9216) and `visual()` run twice |
| N (merged video tokens) | **13,824** · T = 55,296 patches · grid `[64, 24, 36]` |
| 10% budget | 1,382 tokens |
| frozen forward | 0.73 s |
| grad fwd+bwd, `query_frac=1.0` | **2.19 s**, peak 21.7 GB |
| grad fwd+bwd, `query_frac=0.25` | 0.76 s |

`query_frac=1.0` fits the budget, so no subsampling — the trained score and the eval score
stay identical. Steady state in the real run is **~4.2 s/step**.

### 5.2 LR probe — the plan's `lr=2e-5` would have produced a false negative

Four LRs, ≥200 micro-steps each. `recall_traj` = fraction of the teacher's *gaze
complement* recovered into the student's top 6.5%; frozen baseline **0.042**.

| lr | recall_traj @ step 200 | recall_P | anc (1−cos) | verdict |
|---|---|---|---|---|
| 2e-5 *(planned)* | ~0.037 | 0.398 | 0.00000 | flat — indistinguishable from doing nothing |
| 1e-4 | 0.042 | 0.398 | 0.00000 | flat |
| 5e-4 | 0.146 | 0.411 | 0.0049 | learns |
| **2e-3** | **0.269** | **0.430** | 0.0082 | **chosen** |

At 2e-5 the experiment would have run 6 hours and reported "the ViT cannot absorb the gaze
signal" — a conclusion about the optimiser, not the method. Drift is bounded by the gate
(§3), not by a small learning rate.

At 2e-3 the anchor's growth rate decayed ~4× between intervals (+0.0054 over 19 optimiser
steps, then +0.0012 over 25), i.e. λ_anchor reaching an equilibrium rather than running away.

### 5.3 Live — SG raw video, Phase 1, epoch 1

Step 740 / 2900 (185 optimiser steps), windowed:

| metric | frozen | live | reading |
|---|---|---|---|
| `recall_traj` | 0.042 | **0.395** | ~9× — the gaze complement is being absorbed |
| `recall_P` | 0.398 | **0.465** | **rises** — content tokens are not being sacrificed |
| `sel` | — | 0.957 ↓ | monotone |
| `anc` | 0 | 0.0082 | cos 0.992, still under the 0.01 threshold |

**Provisional and single-run.** `recall_*` are selection metrics, not accuracy; the
accuracy claim requires Phase 2 and ≥3 evals (v2 §8).

### 5.4 SETTING 1 COMPLETE — SG raw video (2026-07-29, 11h32m end to end)

| stage | result | wall |
|---|---|---|
| P1 | `recall_traj` 0.0435 → **0.3833** · `recall_P` 0.3985 → **0.4632** · `recall_S` 0.4019 → **0.5302** | 7h12m |
| gate | **PASS** — frozen 372 / tuned 369, Δ **−3** items, cos 0.9915 | 30m |
| P2 | ep1 **69.58% (366)** · ep2 **69.77% (367)** | 3h51m |

**Per-task, n=526.** Item counts sum exactly (ep1 366, ep2 367).

| task | n | P2 ep1 | P2 ep2 | Δ21 | KD student §7.7 | ep2−KD | M1 §8 |
|---|---|---|---|---|---|---|---|
| **GSM** | 64 | 70.31 (45) | **60.94 (39)** | **−6** | 70.31 (45) | **−6** | 71.36 |
| NFI | 68 | 61.76 (42) | 63.24 (43) | +1 | 67.65 (46) | −3 | 63.24 |
| OTP | 2 | 50.00 (1) | 50.00 (1) | 0 | 50.00 (1) | 0 | — |
| SR | 37 | 56.76 (21) | 59.46 (22) | +1 | 51.35 (19) | +3 | 57.66 |
| OAR | 96 | 91.67 (88) | 89.58 (86) | −2 | 89.58 (86) | 0 | 93.40 |
| OI-E | 101 | 72.28 (73) | 75.25 (76) | +3 | 71.29 (72) | +4 | 73.27 |
| OI-H | 64 | 70.31 (45) | 73.44 (47) | +2 | 67.19 (43) | +4 | 74.48 |
| FAP | 94 | 54.26 (51) | 56.38 (53) | +2 | 51.06 (48) | +5 | 56.38 |
| **Avg** | 526 | 69.58 (**366**) | 69.77 (**367**) | +1 | 68.44 (**360**) | **+7** | 71.17 (374) |

#### The headline is not the Avg — it is that the mechanism did not fire

**The bar is cleared: +7 items over v2 §7.7's 360**, above the ±4 floor, and this comparison
is *budget-matched* (both are M1 warm-start + 2 epochs of LoRA), so v2 §10.4's confound does
not apply to it. The §2.5 co-adaptation difference still does.

**But the gain is in the wrong columns.** The prediction was explicit: if ViT attention had
absorbed the gaze complement, GSM and NFI — the gaze-driven tasks, and the only two where v2
§2.2a's KD student beat its teacher — should move. They did not.

- **GSM: 45 → 45 (ep1), then 45 → 39 (ep2).** Zero, then a 6-item collapse.
- **NFI: below the KD student in both epochs** (−4, −3).
- Every item of the +7 comes from object/action columns: FAP +5, OI-E +4, OI-H +4, SR +3.

GSM's −6 is **9.37 points on a 64-item column**, against the 1.57-point range v2 §8 measured
across re-evaluations of identical weights. That is not noise.

**So `recall_traj` = 0.383 did not convert.** The ViT genuinely recovers ~38% of the teacher's
gaze complement — verified on held-out data, twice — and that recovery buys nothing on the
tasks the complement exists to serve. Two readings survive and this run cannot separate them:
the recovered 38% may be the least informative part of the complement, or the LLM readout may
be unable to exploit it. What is *not* supported is "the ViT learned the gaze signal".

**Epoch 2 traded GSM for object identification.** Training loss fell 0.851 → 0.560 (−34%)
while Avg moved +1 item and the column profile rearranged wholesale. That is redistribution,
not convergence.

#### best.pth selection is actively misleading here

366 vs 367 is **1 item** — far inside v2 §8's ±4 floor, so "best by Avg" is arbitrary. That
arbitrary choice saves the checkpoint whose GSM is **6 items worse**. v2 §8 warns best-of-N is
upward-biased; here it also *hides a column collapse behind a flat average*.

**Report epoch 1 (366), or report both. Do not report only the best-of-2.**

### 5.5 Paper-ready table, SG raw video

Teacher row = v2 §8 **run 2** (375 items); ViT-KD row = **epoch 1**. Both choices are stated
in the caption rather than hidden — see the notes below.

| system | gaze@test | extra params | GSM | NFI | SR | OAR | OI-E | OI-H | FAP | **Avg** | items |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SG specialist teacher (M1) | **yes** | 36.85 M | 70.31 | 64.71 | **59.46** | **93.75** | **73.27** | **73.44** | **56.38** | **71.29** | **375** |
| KD student, raw video | no | 3.95 M | 70.31 | **67.65** | 51.35 | 89.58 | 71.29 | 67.19 | 51.06 | 68.44 | 360 |
| **ViT distillation** | no | **0** | 70.31 | 61.76 | 56.76 | 91.67 | 72.28 | 70.31 | 54.26 | 69.58 | 366 |

```latex
\sys-T (SG-only teacher, gaze)  & 70.31 & 64.71 & 59.46 & 93.75 & 73.27 & 73.44 & 56.38 & 71.29 \\
KD (raw video)                  & 70.31 & 67.65 & 51.35 & 89.58 & 71.29 & 67.19 & 51.06 & 68.44 \\
ViT-KD (raw video)              & 70.31 & 61.76 & 56.76 & 91.67 & 72.28 & 70.31 & 54.26 & 69.58 \\
```

Item deltas: ViT-KD − KD student = GSM 0 · NFI **−4** · SR +2 · OAR +2 · OI-E +1 · OI-H +2 ·
FAP +3 · **Avg +6**. ViT-KD closes **6 of the 15 items (40%)** between the KD student and the
teacher, at **zero extra inference parameters**.

**Caption must state:** teacher is a single run (v2 §8 run 2; its GSM re-scores 70.31–71.88 on
identical weights), the other two rows are single runs, `past_object_transition_prediction`
(2 items) has no column but is inside every Avg, and ViT-KD uses the two-phase protocol of
§2.5 rather than v2's joint one.

### 5.6 Why GSM is identical (70.31, 45/64) in all three rows

Not one cause but three, and the third is a result:

1. **Partly the run choice.** v2 §8 re-scores the teacher's *identical weights* three times and
   GSM comes out **45 / 46 / 46** items. Run 2 happens to be 45. The 3-run mean (71.36 ≈ 45.7)
   would break the tie. The teacher's side of the coincidence is inside its own noise.
2. **Coarse resolution.** n=64 ⇒ one question = 1.5625 pp, so the only reachable values near
   70% are 44 / 45 / 46. Systems of similar strength land in the same bucket easily.
3. **The two raw-video systems agreeing at 45 is NOT a coincidence — it is the finding.**

   | GSM | items |
   |---|---|
   | KD student, **overlay** (`viz`, marker visible) | **49** (v2 §2.2a) |
   | KD student, **overlay removed** (`original`) | **45** (v2 §7.7) |
   | **ViT-KD, raw video** (`original`) | **45** (this run) |

   With the marker in the pixels, 49. Without it, 45 — and replacing the entire selection
   mechanism (a 3.95M predictor → distilled ViT attention) does not move it off 45.

**GSM is governed by whether the gaze marker is visible in the pixels, not by which tokens are
selected.** Raw video simply does not contain the signal, so no amount of selection fidelity
recovers it — which is exactly why `recall_traj` 0.042 → 0.383 bought zero GSM items (§5.4).
It is also consistent with v2 §7.7, where GSM was the single largest drop when the overlay was
removed.

**Open, cheaply testable:** whether the three systems get the *same* 45 questions right, or
different ones. `TrajGazeMerge/eval/eval_dump.py` can dump per-item predictions; the
intersection separates "shared ceiling" from "coincidental tie". ~30 min once the GPUs free up.

**The decisive test is already running.** Setting 2 (`sg_ovl`) gives the ViT the `viz` frames:
- GSM climbs toward 49 → the ViT *can* absorb gaze when it is present; the limit is the raw
  video's information content, not the method's capacity.
- GSM stays near 45 → selection distillation cannot reach GSM under either condition.

---

## 6. Run matrix — four settings, serial, in this order

| # | setting | env | ViT sees | bar |
|---|---|---|---|---|
| 1 | **SG raw video** | `GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=0` | `original` | 360 |
| 2 | **SG overlay** | `GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=1` | `viz` | 369 |
| 3 | **EG raw video** | `GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=0` | `no_gaze` | 268 |
| 4 | **EG overlay** | `GAZE_OVERLAY=1 VLM_GAZE_OVERLAY=1` | `gaze` | 272 |

Each: `P1 → gate → P2`. Setting *n* completes before *n+1* starts (2 GPUs, both used).

The two variants answer different questions:
- **raw** — can attention learn the gaze signal when the marker is *not* in the pixels?
- **overlay** — can attention *translate* a visible marker into a selection? Expected
  easier, since the teacher's traj 3% correlates with the marker's screen position.
  Overlay results are **not gaze-free** (v2 §7.1, §12.5-5) and must be labelled as such.

**Cost:** ~30 GPU-h total. SG ≈ 10 h/setting, EG ≈ 2.5 h/setting, gates ~25 min each.

---

## 7. Code

### Added
| file | role |
|---|---|
| `TrajGazeMerge/training/train_vit_selection_kd.py` | Phase 1 |
| `scripts/vitkd_integrity_gate.py` | the gate (§3) |
| `scripts/run_vitkd_all.sh` | 12-job serial chain, `.done` markers + `--resume` |
| `scripts/vitkd_status.sh` | one-screen supervision report |
| `scripts/measure_vitkd_step0.py` | §5.1 |

### Changed
| file | change |
|---|---|
| `VisionZip/Qwen2_5_VL/qwen2_5vl_visionzip.py` | `grad_logits` / `grad_last_block` kwargs, **default False → bit-identical** |
| `TrajGazeMerge/training/train_visionzip_lora.py` | `--vit-lora-ckpt`, `--resume`, `--ckpt-every-steps`, `--seed`, frame-variant report + assertion |
| `TrajGazeMerge/data/combined_simple_dataset.py` | §4.1 EG fix |
| `TrajGazeMerge/training/train_autogaze_lora.py` | §4.1 SG fix (`_SG_VLM_FRAME_SUB`) |

### 7.1 Notes for whoever touches this next

- **peft cannot target the ViT here.** `target_modules=["qkv","proj"]` as plain strings
  matches all 32 blocks *and* `patch_embed.proj` (a Conv3d). It also shares
  `disable_adapter()` with the LLM adapter, and `PeftModel.state_dict()` returns the whole
  8.29B backbone (v2 §12.1). Hence the hand-rolled `LoRALinear` — 61k params, a 250 KB
  checkpoint, and an adapter that can be disabled independently for the frozen reference pass.
- **Mid-epoch checkpointing now exists** (`--ckpt-every-steps`, adapter tensors only,
  written to a temp file then `os.replace`d). v2 §13.4 lost three runs to epoch-end-only
  saving; auto-resume is not possible without this.
- **Seeding** is present in both new/edited trainers (`--seed`), closing v2 §5 item 5 for
  this line of work. The other trainers still seed nothing.
- `--resume` is safe on a fresh run, so relaunching the chain is always the correct recovery.

---

## 8. Operations

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/vilab_yj/trajgaze && source env.sh
nohup setsid bash scripts/run_vitkd_all.sh >/dev/null 2>&1 &   # start OR recover
bash scripts/vitkd_status.sh                                    # one-screen status
```

Recovery is idempotent: finished jobs are skipped via `vitkd_state/*.done`, and the job
that died resumes mid-epoch. Hourly supervision watches for *hung* (alive but no log write
in 20 min) as well as dead, since `pgrep` alone cannot tell them apart.

Failure playbook: OOM → lower `--score-query-frac`; SIGTERM / reprovision (v2 §13.1) →
relaunch; immediate `exit=1` → read the traceback, do not blind-retry (v2 §9's
`tab6_nopretrain` died the same way twice); NaN loss → stop and report.

**`wait_gpu` is scoped to this chain's own processes**, not to every CUDA process on the
box — otherwise an unrelated job stalls all 12 jobs for the 15-minute cap each time.

---

## 9. TODO

### Running (updated 2026-07-29 12:20)
- [x] **1 · SG raw video — COMPLETE**, 11h32m. P1 → gate PASS → P2 366/367 items (§5.4, §5.5)
- [ ] **2 · SG overlay** — P1 in flight. Frame streams verified `viz`/`viz` vs setting 1's
      `original`/`viz`, i.e. §4.1's fix is doing its job. **This is the decisive run for §5.6.**
- [ ] 3 · EG raw video — P1 → gate → P2
- [ ] 4 · EG overlay — P1 → gate → P2

### Per setting, gates that must be checked, not assumed
- [ ] integrity gate **PASS** (|Δ| ≤ 4 items) before its Phase 2 is allowed to start
- [ ] `anc` stayed < 0.02 through training (hourly check)
- [ ] Phase 2 reported **n=526** for `--source sg` / **n=485** for `--source eg`
- [ ] `kept` ≈ 10.0% in the Phase 2 log

### After the chain
- [ ] Per-task table in v2 §2.2a / §10.3 format, item counts summing exactly to the total
- [ ] `recall_P` / `recall_S` / `recall_traj`, before vs after, for all four settings —
      the S split is an eval-flag change on the same adapter, no extra training
- [ ] Compute the predictor's `recall_traj` under **this** definition so the ViT-KD and
      KD-student rows are comparable (v2's `agree` is restricted to the avail pool and is not)
- [ ] Repeat every student eval ≥3× (v2 §8) — every number here is currently a single run

### Known gaps, deliberately not scheduled
- [ ] **Equal-budget control** — Phase 2 stacks epochs on a warm-started LLM LoRA, so a gain
      over M1 is not separable from "twice the optimizer steps" (v2 §10.4). User dropped it.
      One row (frozen ViT, same extra epochs, ~4 h) would close it.
- [ ] Rank/depth escalation (r=32, or blocks 28–31) if `recall_traj` plateaus below ~0.6
- [ ] Whether the *positional* contextual 3.5% belongs in the target at all (§2.4) — the
      P-vs-S recall gap quantifies it

---

## 10. What must be said when reporting this

1. **"matches", not "beats"**, for any comparison to M1 — v2 §10.4's warm-start confound
   applies unchanged, and the equal-budget control was dropped.
2. **Single runs.** v2 §8 requires ≥3. Differences under 4 items are not measurements.
3. **Overlay settings are not gaze-free** — the marker is still in the pixels (v2 §7.1).
   Label rows `KD (raw video)` vs `KD (gaze-overlay)` as v2 §12.5-5 requires.
4. **Stage-1 TAS remains overlay-trained** (v2 §9 "Explicitly dropped") — state it as an
   assumption, not a verified equivalence.
5. **`recall_*` are selection metrics, not accuracy.** A high `recall_P` with a flat
   accuracy would mean the reproduced tokens were not the ones that mattered — report both.
6. The distilled score is the **unmasked cross-frame** quantity of §4.2, not a per-frame
   saliency. Anyone reproducing this on a fixed VisionZip will get different numbers.
7. **The two-phase protocol is new here** (§2.5). v2's KD student trains selector and
   readout jointly in one run; v3 trains them in sequence. The LLM's extra optimizer budget
   is matched, but the co-adaptation is not. Do not present the two rows as if they came
   from the same recipe.
