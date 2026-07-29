#!/usr/bin/env bash
# One-screen status of the ViT selection-distillation chain. Written for the hourly
# supervision loop: everything needed to decide "healthy / hung / dead / done"
# without opening a single log by hand.
#
#   scripts/vitkd_status.sh
#
# HUNG is the failure this exists to catch. A dead process is obvious; a process
# that is alive but has not written a step line in 20 minutes looks identical to a
# healthy one from `pgrep` alone, and that is how a night gets wasted.

set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh >/dev/null 2>&1

STATE=$REPO/vitkd_state
STALE_MIN=${STALE_MIN:-20}
now=$(date +%s)

echo "===== vitkd status $(date -Is) ====="

# ── chain progress ────────────────────────────────────────────────────────────
echo
echo "-- jobs --"
for tag in sg_raw sg_ovl eg_raw eg_ovl; do
    line="  $tag:"
    for job in p1 gate p2; do
        if [ -f "$STATE/${tag}_${job}.done" ]; then line="$line $job=done"
        elif [ -f "$REPO/vitkd_${tag}_${job}.log" ]; then line="$line $job=STARTED"
        else line="$line $job=-"
        fi
    done
    echo "$line"
done

# ── live processes ────────────────────────────────────────────────────────────
echo
echo "-- processes --"
# Bracketed first character so these patterns never match this script's own command
# line (or a shell that merely mentions the name) — otherwise the driver always
# reports "alive" and the rank count is inflated by whoever asked.
if pgrep -f "[r]un_vitkd_all.sh" >/dev/null; then
    echo "  chain driver: alive (pid $(pgrep -f '[r]un_vitkd_all.sh' | head -1))"
else
    echo "  chain driver: NOT RUNNING"
fi
# Counts torchrun parents, workers AND forked DataLoader workers (they inherit the
# command line), so this is a liveness signal, not a rank count.
n=$(pgrep -cf "[t]rain_vit_selection_kd|[t]rain_visionzip_lora" 2>/dev/null || echo 0)
echo "  trainer procs: $n (0 = nothing training)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | sed 's/^/  gpu /'

# ── freshness of the active log ───────────────────────────────────────────────
echo
echo "-- active log --"
newest=$(ls -t "$REPO"/vitkd_*.log 2>/dev/null | head -1)
if [ -z "$newest" ]; then
    echo "  (no vitkd_*.log yet)"
else
    age=$(( (now - $(stat -c %Y "$newest")) / 60 ))
    echo "  $(basename "$newest")  (last write ${age} min ago)"
    if [ "$n" -gt 0 ] && [ "$age" -ge "$STALE_MIN" ]; then
        echo "  *** STALE: process alive but no output for ${age} min — suspect hang ***"
    fi
    echo "  tail:"
    grep -E "step [0-9]+/|recall_|Overall:|exit=|COMPLETE|new best" "$newest" \
        2>/dev/null | tail -4 | sed 's/^/    /'
fi

# ── failures worth acting on ──────────────────────────────────────────────────
echo
echo "-- recent failures --"
found=0
for f in "$REPO"/vitkd_*.log; do
    [ -e "$f" ] || continue
    if grep -qE "exit=[1-9]|Traceback|OutOfMemoryError|CUDA error|Killed" "$f" 2>/dev/null; then
        echo "  $(basename "$f"):"
        grep -E "exit=[1-9]|Traceback|OutOfMemoryError|CUDA error|Killed" "$f" \
            | tail -3 | sed 's/^/    /'
        found=1
    fi
done
[ $found -eq 0 ] && echo "  none"

# ── results so far ────────────────────────────────────────────────────────────
echo
echo "-- results --"
for tag in sg_raw sg_ovl eg_raw eg_ovl; do
    p1log=$REPO/vitkd_${tag}_p1.log
    if [ -f "$p1log" ]; then
        grep -E "recall_(P|S|traj):" "$p1log" | tail -3 | sed "s/^/  $tag P1 /"
    fi
    gl=$REPO/vitkd_${tag}_gate.log
    [ -f "$gl" ] && grep -E "frozen features|tuned  features|Δ  |verdict|mean cos" "$gl" \
        | tail -5 | sed "s/^/  $tag GATE /"
    p2log=$REPO/vitkd_${tag}_p2.log
    [ -f "$p2log" ] && grep -E "Overall: " "$p2log" | tail -2 | sed "s/^/  $tag P2 /"
done

echo
echo "-- bars (docs/kd_handoff_v2.md) --"
echo "  sg_raw 360 items (§7.7) | sg_ovl 369 (§2.2a) | eg_raw 268 (§7.7) | eg_ovl 272 (§10.3)"
echo "  noise floor +-4 items; 1 item = 0.19% on SG (n=526), 0.21% on EG (n=485)"
