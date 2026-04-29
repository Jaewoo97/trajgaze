#!/usr/bin/env bash
# Chain runner: No-KD v2-temporal → (after eval) → mr-cons v2-temporal.
# Use one detached invocation: `setsid nohup ./run_chain_v2_temporal.sh ...`.
# Total expected: ~9h (No-KD) + ~9h (mr-cons) ≈ 18h.

set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)

echo "[$(date)] === START No-KD v2-temporal ===" > /workspace/trajgaze_v2/TrajGazeMerge/checkpoints/chain_v2_temporal.log
bash "$DIR/run_no_kd_v2_temporal.sh"
echo "[$(date)] === END No-KD v2-temporal ===" >> /workspace/trajgaze_v2/TrajGazeMerge/checkpoints/chain_v2_temporal.log

# mr-cons launching is delegated to conditional_mr_cons_launcher.sh, which
# waits for No-KD eval to finish, then only launches mr-cons if OVERALL
# beats the msk No-KD baseline (64.45). This file used to launch mr-cons
# unconditionally; we removed that to avoid wasting ~13h on a regressed run.
echo "[$(date)] === Chain runner exit (mr-cons gated externally) ===" >> /workspace/trajgaze_v2/TrajGazeMerge/checkpoints/chain_v2_temporal.log
