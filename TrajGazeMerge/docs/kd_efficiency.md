# KD efficiency — supplementary tables

Measured 2026-07-30 on b200. Covers the three systems of `kd_handoff_v3.md` §5.5 on
StreamGaze: the M1 SG specialist teacher, the KD student on raw video, and ViT-KD.

Reproduce:

```bash
source env.sh
tmux new -s bench
bash scripts/bench_kd_efficiency.sh          # training side, ~50 min
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 VLM_GAZE_OVERLAY=0 \
    python scripts/measure_kd_inference_cost.py --items 24   # deployment side, ~3 min
python scripts/collect_kd_efficiency.py      # regenerates every table below
```

Instrumentation lives in `TrajGazeMerge/training/bench_probe.py` and is wired into all
four trainers. It is always on and changes no tensor, no RNG draw and no control flow;
`--max-steps` is the one thing that does alter control flow and it defaults to 0.

---

## 1. Protocol

| | |
|---|---|
| hardware | 1 × NVIDIA B200 (the box has 2; jobs run **serially**, one GPU each) |
| batch | 1 item per step — batching is forbidden by §5.8 (it changes the distilled score) |
| effective batch | 8 (`--grad-accum 8` on 1 rank = the recorded runs' `4 × 2 ranks`) |
| measured window | 250 steps, after 50 discarded warmup steps |
| memory | `torch.cuda.max_memory_allocated()`, reset at the warmup boundary; GB = 1e9 B |

**One GPU, serially, is a deliberate constraint.** §5.8 item 2 measured this loop as
dataloader-bound — the GPU idles ~28% of wall time because `preprocess_video` runs in
the training loop with `num_workers=2` of 72 cores. Two concurrent jobs contend for
those cores and the speed column stops meaning anything. Memory would survive; s/step
would not.

### 1.1 `OMP_NUM_THREADS` — do not omit it

`torch.distributed.run` pins `OMP_NUM_THREADS=1` **only when `nproc_per_node > 1`**, so
every recorded 2-GPU run on this box got it implicitly and a `--nproc_per_node=1` run
does not. Left unset, torch takes all 72 cores per process.

The first attempt at this benchmark did exactly that and produced a teacher row of
**8.2 s/step** — against the **2.14 s/step** the same trainer with the same arguments is
on record for in `tab6_nopretrain_overlay.log` (6194 s / 2900 steps, 2 GPUs). The process
was running 99 threads at 1271% CPU with a load average of 291; killing it dropped
runnable threads to 11. That 3.8× was the thread pool spinning, not the method.

`scripts/bench_kd_efficiency.sh` now exports it. Anyone re-running a single-rank job
from this repo has to do the same or the number is meaningless.

---

## 2. Table A — training efficiency

1 × B200, batch 1, effective batch 8.

| System | Speed (it/s) | s/step | GPU Mem (GB) | Trainable params |
|---|--:|--:|--:|--:|
| M1 SG specialist teacher | 0.506 | 1.977 | 29.7 (32.3) | 10.09 M |
| KD student (raw video) | 0.509 | 1.963 | 30.3 (33.0) | 14.04 M |
| ViT-KD **Phase 1** | 0.259 | 3.862 | 22.4 (27.8) | 61,440 |
| ViT-KD **Phase 2** | 0.542 | 1.844 | 29.5 (32.2) | 10.09 M |

GPU Mem is the allocator peak, with the nvidia-smi peak in parentheses — the latter
includes the CUDA context and the caching allocator's slack, and is what a user
watching the card sees.

Trainable params: teacher and Phase 2 train the LLM LoRA only (10,092,544, r=16);
the student adds the 3,950,593-parameter `TrajSaliencePredictor`; Phase 1 trains
**only** the rank-8 ViT adapter, with the LLM LoRA warm-started and frozen (§2.2).

### 2.1 Two readings, both load-bearing

**The KD student is free.** 0.509 vs 0.506 it/s against its own teacher — a 0.6%
difference, inside the run-to-run spread (§4). The distillation loss and the 3.95 M
predictor forward cost **+0.7 GB and no measurable time**. This is the same shape of
result the reference table reports for its weighting strategies, and it is the honest
version of "KD is cheap": cheap *to train*, on top of a teacher you had to train anyway.

**ViT-KD is not.** Its two phases sum to **5.71 s/item** (3.862 + 1.844), i.e. an
effective **0.175 it/s** and **2.9× the student's training compute** at equal epochs.
Report the sum. §10-7 warns against quoting one phase; Phase 2 alone (0.542 it/s) reads
as the *fastest* row in the table, which inverts the actual cost.

### 2.2 Phase 1 is the most expensive step in the pipeline — §2.5's cost argument does not hold

§2.5 gives three reasons for splitting ViT-KD into two phases, and the third is
**Cost**: "Phase 1 runs no VLM forward/backward (v2 records `--freeze-lora` as '~3-5x
faster per epoch' for the analogous predictor-only mode)."

Measured, Phase 1 is **2.0× slower per step than the KD student** (3.862 vs 1.963) and
1.9× slower than Phase 2. Removing the 7B forward/backward does not make the step
cheap, because the ViT case replaces it with something more expensive:

- a **second full ViT trunk pass** — §5.8 item 1 already recorded blocks 0..30 being
  computed twice per step (~0.71 s, 17%), once under `no_grad` inside the adapted pass
  and again as a complete frozen forward, and
- a **backward through the ViT**, which the predictor case never needed.

The `--freeze-lora` speedup v2 measured was for the *predictor* line, where freezing the
LoRA removes work and adds none. It does not transfer to the ViT line, and §2.5 should
not be read as if it does. Note this does not weaken §2.5's other two reasons (no moving
target; the gate needs a finished ViT), which are about correctness, not cost.

The memory column is the visible trace of the same structure: Phase 1 peaks at
**22.4 GB against ~30 GB** for every row that runs the LLM. That is consistent with
§5.1's independently measured "grad fwd+bwd, `query_frac=1.0` → 2.19 s, peak 21.7 GB".

---

## 3. Table B — deployment cost per item

What each system adds at inference, over the forward passes all three share.

| System | Gaze @ test | Extra params | Extra mem (MB) | Extra latency (ms/item) | % of shared |
|---|:--:|--:|--:|--:|--:|
| M1 SG specialist teacher | **yes** | 35.80 M | 143.2 | 57.98 | 3.57% |
| KD student (raw video) | no | 3.95 M | 15.8 | 1.64 | 0.10% |
| **ViT-KD (raw video)** | no | **0** | **0.0** | **0.00** | **0.00%** |

Shared forward = **1624.9 ms/item**, broken down as:

| stage | mean ms | note |
|---|--:|---|
| `preprocess_visionzip_item` | 1579.1 | 128-frame JPEG decode **plus** the ViT pass |
| `build_merged_inputs` | 0.3 | |
| 7B LLM forward over the kept ~10% | 45.4 | one forward, not generation |

**The denominator is host-bound, and that flatters every percentage.** §5.1 measured the
frozen ViT forward alone at 0.73 s, so roughly half of the 1579 ms is frame decoding on
the CPU, not model compute. Against a GPU-only denominator (~0.78 s) the teacher's
selection overhead is ~7.4%, not 3.57%. Quote whichever you like but say which.

ViT-KD's **0 is after folding**: training produces 61,440 rank-8 LoRA parameters on
`visual.blocks[31]`, which merge into the ViT weights. Write "folded into the ViT",
never "no module". The 0.89 ms `visionzip_select_tokens` call it does make is the
selection *every* VisionZip system already performs, so it is in the shared column.

### 3.1 The TAS encoder parameter count does not match the number the paper quotes

Measured with `load_traj_encoder("full", $STAGE1_CKPT, device, 16)`:
**35,795,606 parameters (35.80 M, fp32)**.

`kd_handoff_v3.md` §1 and §12.3a of v2 quote **36.85 M**; v2 §12.3 quotes **35.8 M**; and
v2 §1165 reconciles the two as "the same number: 36,852,576 − 1,048,576" = 35,804,000.
The measurement is 8,394 below even that.

Three numbers are now in circulation for one encoder. **Pick one and use it everywhere**
— this table currently reports the measured 35.80 M, which is *not* what the existing
`extra params` column of v3 §1 says. Resolving which is correct is a paper-consistency
task, not a measurement one, and it is deliberately left open here.

Numbers that did check out exactly: `TrajSaliencePredictor` **3,950,593** = v2's 3.95 M,
and the ViT adapter **61,440** = §2.2's 61,440.

---

## 4. Cross-checks against the recorded 2-GPU runs

Per-rank throughput is 1 item/step either way, so s/step compares directly. One GPU has
no DDP all-reduce, so the measurement should land at or slightly **below** the historic
value — and all three rows that have a history do.

| job | measured (1 GPU) | historic (2 GPU) | delta | source |
|---|--:|--:|--:|---|
| teacher | 1.977 | *(2.14)* | *(−7.6%)* | `tab6_nopretrain_overlay.log`, same trainer + args, different objective |
| student | 1.963 | 2.22 – 2.29 | −12.9% | `kd_train_sgonly_nooverlay.log`, 6650 s / 6440 s over 2900 steps |
| ViT-KD P1 | 3.862 | 4.04 | −4.4% | §5.8 |
| ViT-KD P2 | 1.844 | 2.08 | −11.3% | §5.8 |

**The teacher row has no true history.** Its checkpoint
(`visionzip_complement_learned_SGonly_overlay`) is a symlink into `$DATA/aaai/` and was
trained on a different machine — `scripts/launch_vzcomp_learned_overlay.sh` shows
`/workspace/trajgaze_st` with the log going to `/tmp`. Nothing survives. The 2.14 s/step
above is `tab6_nopretrain_overlay`, which runs the *same trainer with the same
arguments* but a different Stage-1 objective, so it bounds the step cost without being
a rerun of this row. Treat the teacher as a single measurement.

Within-run spread, from the 13 post-warmup log windows of each job:

| job | median s/step | min – max |
|---|--:|---|
| teacher | 1.99 | 1.86 – 2.05 |
| student | 1.96 | 1.84 – 2.06 |
| ViT-KD P1 | 3.85 | 3.62 – 4.04 |
| ViT-KD P2 | 1.85 | 1.73 – 1.96 |

±5% window-to-window. **Differences under ~5% in Table A are not measurements** — which
is precisely why the teacher/student 0.6% gap is reported as parity, not as a win.

---

## 5. LaTeX

```latex
% Table A -- training efficiency (1 x B200, batch 1, effective batch 8)
M1 SG specialist teacher     & 0.506 & 29.7 & 10.09M  \\
KD student (raw video)       & 0.509 & 30.3 & 14.04M  \\
ViT-KD (Phase 1)             & 0.259 & 22.4 & 61{,}440 \\
ViT-KD (Phase 2)             & 0.542 & 29.5 & 10.09M  \\
%
% Table B -- deployment cost per item (shared forward 1624.9 ms)
M1 SG specialist teacher     & yes & 35.80M & 143.2 & 57.98 & 3.57 \\
KD student (raw video)       & no  & 3.95M  & 15.8  &  1.64 & 0.10 \\
ViT-KD (raw video)           & no  & 0      & 0.0   &  0.00 & 0.00 \\
```

---

## 6. What must be said when reporting this

1. **ViT-KD is Phase 1 + Phase 2.** Quote the sum (0.175 it/s effective, 2.9× the
   student). Phase 2 alone is the fastest row in the table and reads as the opposite of
   the truth. This is §10-7 applied to cost.
2. **Frame streams differ by row.** The teacher trains and is evaluated on overlay
   frames (`viz` on both streams, v2 §7.3); the student and ViT-KD rows are raw video
   (`VLM_GAZE_OVERLAY=0`). Compute is identical — it selects a JPEG directory — but
   §10-3 requires the labels.
3. **One GPU, one process, `OMP_NUM_THREADS=1`.** The recorded runs were 2 GPUs; these
   numbers are per-device and land 4–13% below their per-rank history because there is
   no all-reduce. Without the OMP pin they land 3.8× above it (§1.1).
4. **The it/s denominator is host-bound.** §5.8 item 2's ~28% GPU idle applies to every
   row equally, so row-to-row comparison holds, but the absolute rate understates the
   hardware and would move on a machine with a faster frame pipeline.
5. **Say which denominator Table B's percentages use.** Against the measured 1624.9 ms
   (decode + ViT + LLM) the teacher costs 3.57%; against GPU compute alone, ~7.4%.
6. **35.80 M vs 36.85 M is unresolved** (§3.1). Do not mix them across tables.
7. **Single runs.** Every row here is one measurement of one configuration. The ±5%
   window spread bounds within-run jitter, not run-to-run variance.
