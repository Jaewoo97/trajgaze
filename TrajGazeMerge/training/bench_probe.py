"""Steady-state step-rate and peak-memory probe, shared by the four KD trainers.

Written for the supplementary efficiency table (docs/kd_efficiency.md). The trainers
already log a cumulative `t=<seconds>`, which is enough to recover an epoch time after
the fact but not enough to fill a `Speed (it/s)` / `GPU Mem` row:

  * a cumulative mean over a 2900-step epoch is dominated by its own history, so it
    cannot report the rate the run actually settles at, and
  * `torch.cuda.max_memory_allocated()` was never called anywhere in the training path
    -- the only call in the repo is `scripts/measure_vitkd_step0.py:156-162`.

The probe is always on. Per log window it costs one `time.time()` and one allocator
stat read; it allocates nothing, touches no tensor, draws no RNG and changes no control
flow, so a run with it is bit-identical to a run without. `--max-steps` is the one thing
that *does* change control flow, and it is opt-in (default 0 = off) -- which matters
because `train_visionzip_lora.py` and `train_vit_selection_kd.py` are the files the
paused `sg_ovl_p2` will come back through on `--resume`.

GB here is 1e9 bytes, not 2**30, to match `measure_vitkd_step0.py` and the "peak 21.7 GB"
already recorded in kd_handoff_v3.md §5.1.
"""

from __future__ import annotations

import time

import torch


class BenchProbe:
    """Counts micro-steps; reports windowed s/step and peak allocated memory.

    `warmup` steps are excluded from the summary and the peak-memory counter is reset
    once at that boundary. Model load, the first backward's workspace growth and cuDNN
    autotuning all land before it and none of them recur in steady state, so including
    them would report a startup artefact as the training footprint.
    """

    def __init__(self, warmup: int = 50):
        self.warmup = max(0, int(warmup))
        self.n = 0
        self._t_win = time.time()
        self._n_win = 0
        self._t_bench: float | None = None
        self._n_bench = 0

    # ------------------------------------------------------------------ counting

    def step(self) -> None:
        """One micro-step (= one item, since every trainer runs batch_size=1)."""
        self.n += 1
        self._n_win += 1
        if self.n == max(1, self.warmup):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self._t_bench = time.time()
            self._n_bench = 0
        elif self._t_bench is not None:
            self._n_bench += 1

    def done(self, max_steps: int) -> bool:
        """True once `--max-steps` is reached. Rank-local: single-rank benchmarking only."""
        return bool(max_steps) and self.n >= max_steps

    # ------------------------------------------------------------------ reporting

    def window(self) -> dict:
        """Close the current log window and return its rate + current peak.

        Call exactly once per `log_every` print, so the window boundaries are the
        print boundaries and the two always describe the same span of steps.
        """
        now = time.time()
        s_per_step = (now - self._t_win) / max(1, self._n_win)
        self._t_win = now
        self._n_win = 0
        return {
            "s_per_step": s_per_step,
            "it_s": (1.0 / s_per_step) if s_per_step > 0 else 0.0,
            "peak_gb": self.peak_gb(),
        }

    def summary(self) -> dict:
        """Post-warmup aggregate. This is the number that goes in the table."""
        if self._t_bench is None or self._n_bench == 0:
            return {"bench_steps": 0, "warmup_steps": self.warmup,
                    "note": "stopped before the warmup boundary; no steady-state sample"}
        dt = time.time() - self._t_bench
        return {
            "warmup_steps": self.warmup,
            "bench_steps": self._n_bench,
            "s_per_step": dt / self._n_bench,
            "it_s": self._n_bench / dt,
            "peak_gb": self.peak_gb(),
            "peak_reserved_gb": self.peak_reserved_gb(),
        }

    @staticmethod
    def peak_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1e9

    @staticmethod
    def peak_reserved_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_reserved() / 1e9
