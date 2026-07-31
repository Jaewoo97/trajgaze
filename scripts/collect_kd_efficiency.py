"""Turn the benchmark artefacts into the supplementary efficiency tables.

    source env.sh && python scripts/collect_kd_efficiency.py

Reads what scripts/bench_kd_efficiency.sh and scripts/measure_kd_inference_cost.py
left in bench_kd_efficiency/ and prints two markdown tables plus their LaTeX. Nothing
here recomputes a measurement -- if a number is missing the cell says so rather than
being filled in from kd_handoff_v3.md, because §5.8's s/step figures were taken on two
GPUs and are a cross-check, not a substitute.

Training and deployment are separate tables on purpose. ViT-KD is a two-phase protocol
(v3 §2.5) so it occupies two training rows, but it is one deployed system and one
deployment row; forcing both into a single table produces a cell that has to be read
two different ways.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(_REPO, "bench_kd_efficiency")

# tag -> (table label, trainable-param recipe)
TRAIN_ROWS = [
    ("teacher",  "M1 SG specialist teacher", ("llm_lora_trainable",)),
    ("student",  "KD student (raw video)",   ("llm_lora_trainable", "student_predictor")),
    ("vitkd_p1", "ViT-KD Phase 1",           ("vitkd_adapter_trained",)),
    ("vitkd_p2", "ViT-KD Phase 2",           ("llm_lora_trainable",)),
]

# label -> (gaze at test, extra-param key, extra-mb key, extra-latency key)
DEPLOY_ROWS = [
    ("M1 SG specialist teacher", "yes", "teacher_tas_encoder", "teacher_tas_encoder", "teacher"),
    ("KD student (raw video)",   "no",  "student_predictor",   "student_predictor",   "student"),
    ("ViT-KD (raw video)",       "no",  "vitkd_adapter_at_inference", "vitkd",        "vitkd"),
]

# kd_handoff_v3.md §5.8, measured on 2 GPUs. Cross-check only.
HISTORIC_S_PER_STEP = {
    "student":  (2.22, 2.29, "kd_train_sgonly_nooverlay.log, 6650s/6440s over 2900 steps"),
    "vitkd_p1": (4.04, 4.04, "kd_handoff_v3.md §5.8"),
    "vitkd_p2": (2.08, 2.08, "kd_handoff_v3.md §5.8"),
}


def bench_summary(tag: str) -> dict | None:
    path = os.path.join(BENCH, f"{tag}.log")
    if not os.path.exists(path):
        return None
    last = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("[BENCH] "):
                try:
                    last = json.loads(line[len("[BENCH] "):])
                except json.JSONDecodeError:
                    pass
    return last


def smi_peak_gb(tag: str) -> float | None:
    """Max nvidia-smi memory.used, sampled at 1 Hz alongside the job.

    Always above the allocator's max_memory_allocated: it includes the CUDA context
    and the caching allocator's unreturned slack.
    """
    path = os.path.join(BENCH, f"{tag}.smi")
    if not os.path.exists(path):
        return None
    vals = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                vals.append(int(line))
    if not vals:
        return None
    return max(vals) * 1024 * 1024 / 1e9      # MiB -> GB


def window_rates(tag: str, warmup: int) -> list[float]:
    """Per-log-window s/step, post-warmup. Their spread is the steady-state check."""
    path = os.path.join(BENCH, f"ckpt_{tag}", "train_log_rank0.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "s_per_step" in d and d.get("step", 0) > warmup:
                out.append(d["s_per_step"])
    return out


def fmt(v, spec=".2f", dash="--"):
    return dash if v is None else format(v, spec)


def main() -> int:
    inf_path = os.path.join(BENCH, "inference.json")
    inf = json.load(open(inf_path)) if os.path.exists(inf_path) else {}
    params = inf.get("params", {})
    extra_mb = inf.get("extra_mb", {})
    extra_ms = inf.get("extra_ms_per_item", {})
    extra_pct = inf.get("extra_pct_of_shared", {})

    print("# KD efficiency -- collected\n")

    # ── raw, per job ──────────────────────────────────────────────────────────
    print("## Raw benchmark output\n")
    print("| job | steps | s/step | it/s | peak alloc (GB) | peak reserved (GB) "
          "| nvidia-smi peak (GB) | windows median / spread |")
    print("|---|--:|--:|--:|--:|--:|--:|---|")
    summaries: dict[str, dict] = {}
    for tag, label, _ in TRAIN_ROWS:
        s = bench_summary(tag)
        if not s or not s.get("bench_steps"):
            print(f"| `{tag}` | -- | -- | -- | -- | -- | -- | **no [BENCH] summary** |")
            continue
        summaries[tag] = s
        w = window_rates(tag, s.get("warmup_steps", 50))
        if w:
            med = statistics.median(w)
            spread = f"{med:.2f} ({min(w):.2f}-{max(w):.2f}, n={len(w)})"
        else:
            spread = "--"
        print(f"| `{tag}` | {s['bench_steps']} | {s['s_per_step']:.3f} | {s['it_s']:.3f} "
              f"| {s['peak_gb']:.2f} | {fmt(s.get('peak_reserved_gb'))} "
              f"| {fmt(smi_peak_gb(tag))} | {spread} |")

    # ── cross-check against the 2-GPU history ─────────────────────────────────
    print("\n## Cross-check against the recorded 2-GPU runs\n")
    print("One GPU has no DDP all-reduce, so these should land at or slightly below "
          "the historic value. A large gap means something other than rank count "
          "changed.\n")
    print("| job | measured s/step (1 GPU) | historic s/step (2 GPU) | delta | source |")
    print("|---|--:|--:|--:|---|")
    for tag, label, _ in TRAIN_ROWS:
        s = summaries.get(tag)
        h = HISTORIC_S_PER_STEP.get(tag)
        if not s:
            continue
        if not h:
            print(f"| `{tag}` | {s['s_per_step']:.3f} | -- | -- | "
                  f"no b200 training log exists for the teacher |")
            continue
        lo, hi, src = h
        mid = (lo + hi) / 2
        d = 100.0 * (s["s_per_step"] - mid) / mid
        print(f"| `{tag}` | {s['s_per_step']:.3f} | {lo:.2f}-{hi:.2f} | {d:+.1f}% | {src} |")

    # ── Table A: training ─────────────────────────────────────────────────────
    print("\n## Table A -- training efficiency (1x B200, batch 1, eff-batch 8)\n")
    print("| System | Speed (it/s) | GPU Mem (GB) | Trainable params |")
    print("|---|--:|--:|--:|")
    for tag, label, recipe in TRAIN_ROWS:
        s = summaries.get(tag)
        n = sum(params.get(k, 0) for k in recipe)
        n_txt = f"{n/1e6:.2f} M" if n >= 1e6 else (f"{n:,}" if n else "--")
        if not s:
            print(f"| {label} | -- | -- | {n_txt} |")
            continue
        smi = smi_peak_gb(tag)
        mem = f"{s['peak_gb']:.1f}" + (f" ({smi:.1f})" if smi else "")
        print(f"| {label} | {s['it_s']:.2f} | {mem} | {n_txt} |")
    print("\nGPU Mem: allocator peak, with the nvidia-smi peak in parentheses. "
          "ViT-KD's training cost is Phase 1 **plus** Phase 2 -- quoting Phase 2 alone "
          "is the misreading kd_handoff_v3.md §10-7 warns about.")

    # ── Table B: deployment ───────────────────────────────────────────────────
    shared = inf.get("shared_ms_per_item")
    print("\n## Table B -- deployment cost per item\n")
    print("| System | Gaze @ test | Extra params | Extra mem (MB) | "
          "Extra latency (ms/item) | % of shared forward |")
    print("|---|:--:|--:|--:|--:|--:|")
    for label, gaze, pkey, mkey, lkey in DEPLOY_ROWS:
        n = params.get(pkey)
        n_txt = "0" if n == 0 else (f"{n/1e6:.2f} M" if n else "--")
        mb = extra_mb.get(mkey)
        ms = extra_ms.get(lkey)
        pct = extra_pct.get(lkey)
        print(f"| {label} | {gaze} | {n_txt} | {fmt(mb, '.1f')} | "
              f"{fmt(ms, '.2f')} | {fmt(pct, '.2f')} |")
    if shared:
        print(f"\nShared forward (ViT + input build + 7B LLM over the kept ~10%): "
              f"**{shared:.0f} ms/item**. The three systems differ only in how the 10% "
              f"is chosen; everything else is identical and is not attributed to any row.")
    if params.get("vitkd_adapter_trained"):
        print(f"\nViT-KD's 0 is *after folding*: training produces "
              f"{params['vitkd_adapter_trained']:,} rank-8 LoRA parameters on "
              f"`visual.blocks[31]`, which merge into the ViT weights. Say \"folded\", "
              f"not \"no module\".")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    print("\n## LaTeX\n")
    print("```latex")
    print("% Table A -- training")
    for tag, label, recipe in TRAIN_ROWS:
        s = summaries.get(tag)
        n = sum(params.get(k, 0) for k in recipe)
        n_txt = f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n:,}" if n else "--")
        if not s:
            print(f"{label:28s} & -- & -- & {n_txt} \\\\")
        else:
            print(f"{label:28s} & {s['it_s']:.2f} & {s['peak_gb']:.1f} & {n_txt} \\\\")
    print("%")
    print("% Table B -- deployment")
    for label, gaze, pkey, mkey, lkey in DEPLOY_ROWS:
        n = params.get(pkey)
        n_txt = "0" if n == 0 else (f"{n/1e6:.2f}M" if n else "--")
        mb = extra_mb.get(mkey)
        ms = extra_ms.get(lkey)
        print(f"{label:28s} & {gaze} & {n_txt} & {fmt(mb, '.1f')} & {fmt(ms, '.2f')} \\\\")
    print("```")

    missing = [t for t, _, _ in TRAIN_ROWS if t not in summaries]
    if missing:
        print(f"\n> **Incomplete**: no benchmark summary for {', '.join(missing)}.", file=sys.stderr)
        return 1
    if not inf:
        print("\n> **Incomplete**: bench_kd_efficiency/inference.json missing; run "
              "scripts/measure_kd_inference_cost.py.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
