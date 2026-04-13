#!/bin/bash
# Automated StreamGaze training pipeline.
#
# Stages:
#   1. Wait for EGTEA patch scoring (fold_a/val) — may already be done
#   2. Merge EGTEA GT → fold_a/gt_val.json
#   3. Re-key EGTEA GT → fold_b/gt_train.json
#   4. Start + wait for EgoExoLearn patch scoring (fold_a/train)
#   5. Merge EgoExoLearn GT → fold_a/gt_train.json
#   6. Full re-key → fold_b/gt_val.json
#   7. Train fold_b (EGTEA train, EgoExoLearn val)
#   8. Train fold_a (EgoExoLearn train, EGTEA val)
#
# Re-entrant: skips stages whose outputs already exist.

set -uo pipefail

TOOLS=/workspace/EgoGazeVQA/tools
AUTOGAZE=/workspace/EgoGazeVQA/AutoGaze
DATA=/workspace/datasets/StreamGaze/autogaze_data
LOGS=/workspace/logs
N_GPUS=4

mkdir -p "$LOGS/greedy_search"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGS/pipeline.log"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }
skip() { log "SKIP: $* (already exists)"; }

# ─── wait until no greedy_search_gt python processes are running ───────────────
wait_scoring_done() {
    # Give processes up to 10s to appear before polling
    sleep 10
    while ps aux | grep greedy_search_gt | grep python | grep -v conda | grep -q .; do
        local entries train val
        train=$(python3 -c "
import json,glob,os
files=glob.glob('$DATA/fold_a/gt_train_gpu*.json')
d={}
for f in files:
    try: d.update(json.load(open(f)))
    except: pass
print(len(d))" 2>/dev/null || echo '?')
        val=$(python3 -c "
import json,glob,os
files=glob.glob('$DATA/fold_a/gt_val_gpu*.json')
d={}
for f in files:
    try: d.update(json.load(open(f)))
    except: pass
print(len(d))" 2>/dev/null || echo '?')
        log "  Scoring in progress — val: $val/92207, train: $train/191614"
        sleep 120
    done
    log "All scoring processes have exited."
}

# ─── Stage 1 + 2: EGTEA scoring + merge ───────────────────────────────────────
if [[ -f "$DATA/fold_a/gt_val.json" ]]; then
    skip "$DATA/fold_a/gt_val.json"
else
    log "=== Stage 1: Waiting for EGTEA scoring (fold_a/val) ==="
    wait_scoring_done
    log "=== Stage 2: Merging EGTEA GT ==="
    conda run -n gaze python -u "$TOOLS/greedy_search_gt.py" --fold fold_a --merge
    [[ -f "$DATA/fold_a/gt_val.json" ]] || die "fold_a/gt_val.json not created"
    log "fold_a/gt_val.json ready."
fi

# ─── Stage 3: Re-key EGTEA → fold_b/gt_train.json ────────────────────────────
if [[ -f "$DATA/fold_b/gt_train.json" ]]; then
    skip "$DATA/fold_b/gt_train.json"
else
    log "=== Stage 3: Re-keying EGTEA GT → fold_b/gt_train.json ==="
    python3 -c "
import json, os
src = json.load(open('$DATA/fold_a/gt_val.json'))
dst = {k.replace('fold_a/val/', 'fold_b/train/'): v for k, v in src.items()}
json.dump(dst, open('$DATA/fold_b/gt_train.json', 'w'))
print(f'fold_b/gt_train.json: {len(dst)} entries')
"
    # Placeholder so the dataloader does not error on missing val GT
    [[ -f "$DATA/fold_b/gt_val.json" ]] || echo '{}' > "$DATA/fold_b/gt_val.json"
    log "fold_b/gt_train.json ready."
fi

# ─── Stage 4: EgoExoLearn scoring (fold_a/train) ─────────────────────────────
if [[ -f "$DATA/fold_a/gt_train.json" ]]; then
    skip "$DATA/fold_a/gt_train.json"
else
    # Only launch new scoring processes if none are already running
    n_running=$(ps aux | grep greedy_search_gt | grep python | grep -v conda | wc -l)
    if [[ "$n_running" -eq 0 ]]; then
        log "=== Stage 4: Starting EgoExoLearn scoring (fold_a/train) ==="
        for i in 0 1 2 3; do
            PYTHONUNBUFFERED=1 conda run -n gaze --no-capture-output \
                python -u "$TOOLS/greedy_search_gt.py" \
                --fold fold_a --gpu "$i" --total_gpus 4 \
                >> "$LOGS/greedy_search/fold_a_gpu${i}.log" 2>&1 &
            log "  GPU $i PID: $!"
        done
    else
        log "=== Stage 4: EgoExoLearn scoring already running ($n_running processes) ==="
    fi
    wait_scoring_done

    log "=== Stage 5: Merging EgoExoLearn GT ==="
    conda run -n gaze python -u "$TOOLS/greedy_search_gt.py" --fold fold_a --merge
    [[ -f "$DATA/fold_a/gt_train.json" ]] || die "fold_a/gt_train.json not created"
    log "fold_a/gt_train.json ready."
fi

# ─── Stage 6: Complete fold_b GT (fold_b/gt_val.json from EgoExoLearn) ────────
# Only overwrite if it's the empty placeholder (size ≤ 3 bytes)
fold_b_val_size=$(stat -c%s "$DATA/fold_b/gt_val.json" 2>/dev/null || echo 0)
if [[ "$fold_b_val_size" -gt 3 ]]; then
    skip "$DATA/fold_b/gt_val.json (already populated)"
else
    log "=== Stage 6: Generating full fold_b GT ==="
    conda run -n gaze python -u "$TOOLS/generate_fold_b_gt.py"
    log "fold_b/gt_val.json ready."
fi

# ─── Stage 7: Train fold_b ────────────────────────────────────────────────────
if ls "$AUTOGAZE/exps/streamgaze_fold_b_ntp/"*.pth 2>/dev/null | grep -q .; then
    skip "fold_b training (checkpoint found)"
else
    log "=== Stage 7: Training fold_b ($N_GPUS GPUs) ==="
    cd "$AUTOGAZE"
    WANDB_MODE=disabled PYTHONUNBUFFERED=1 conda run -n gaze --no-capture-output \
        torchrun --nproc_per_node=$N_GPUS \
        autogaze/train.py \
        --config-name streamgaze_fold_b_ntp \
        >> "$LOGS/train_fold_b.log" 2>&1
    log "fold_b training finished."
fi

# ─── Stage 8: Train fold_a ────────────────────────────────────────────────────
if ls "$AUTOGAZE/exps/streamgaze_fold_a_ntp/"*.pth 2>/dev/null | grep -q .; then
    skip "fold_a training (checkpoint found)"
else
    log "=== Stage 8: Training fold_a ($N_GPUS GPUs) ==="
    cd "$AUTOGAZE"
    WANDB_MODE=disabled PYTHONUNBUFFERED=1 conda run -n gaze --no-capture-output \
        torchrun --nproc_per_node=$N_GPUS \
        autogaze/train.py \
        --config-name streamgaze_fold_a_ntp \
        >> "$LOGS/train_fold_a.log" 2>&1
    log "fold_a training finished."
fi

log "=== Pipeline complete. ==="
