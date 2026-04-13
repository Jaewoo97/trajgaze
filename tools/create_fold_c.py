"""
Create fold_c for AutoGaze training:
  train = EgoExoLearn (fold_a/train) + HoloAssist (clips/holoassist)
  val   = EGTEA       (fold_a/val)

Steps:
  1. Create fold_c/train/ and fold_c/val/ with symlinks.
  2. Re-key fold_a GT JSONs from "fold_a/..." to "fold_c/..." for EgoExoLearn + EGTEA.
  3. Placeholders for HoloAssist GT (filled by greedy_search_gt.py --fold fold_c).

Usage:
    conda run -n gaze python tools/create_fold_c.py
"""

import os
import json

DATA_DIR  = "/workspace/datasets/StreamGaze/autogaze_data"
FOLD_A    = os.path.join(DATA_DIR, "fold_a")
FOLD_C    = os.path.join(DATA_DIR, "fold_c")
HOLO_CLIPS = os.path.join(DATA_DIR, "clips", "holoassist")


def make_symlinks(src_dir: str, dst_dir: str, label: str) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    created = 0
    for fname in os.listdir(src_dir):
        if not fname.endswith(".mp4"):
            continue
        src  = os.path.join(src_dir, fname)
        dest = os.path.join(dst_dir, fname)
        if not os.path.lexists(dest):
            os.symlink(src, dest)
            created += 1
    print(f"  {label}: {created} new symlinks in {dst_dir}")
    total = len([f for f in os.listdir(dst_dir) if f.endswith(".mp4")])
    print(f"          {total} total .mp4 entries")
    return total


def rekey_gt(src_json: str, old_fold: str, new_fold: str) -> dict:
    """Load a GT JSON and replace 'old_fold/' prefix with 'new_fold/'."""
    with open(src_json) as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        new_k = k.replace(f"{old_fold}/", f"{new_fold}/", 1)
        out[new_k] = v
    return out


def main():
    # ── 1. Build fold_c/train/ ─────────────────────────────────────────────────
    print("=== Creating fold_c/train/ ===")
    train_dir = os.path.join(FOLD_C, "train")

    # EgoExoLearn clips (from fold_a/train)
    make_symlinks(os.path.join(FOLD_A, "train"), train_dir, "EgoExoLearn")

    # HoloAssist clips
    if os.path.isdir(HOLO_CLIPS):
        make_symlinks(HOLO_CLIPS, train_dir, "HoloAssist")
    else:
        print(f"  WARNING: HoloAssist clips dir not found: {HOLO_CLIPS}")
        print("  Run prepare_holoassist_autogaze.py first.")

    # ── 2. Build fold_c/val/ ───────────────────────────────────────────────────
    print("\n=== Creating fold_c/val/ ===")
    val_dir = os.path.join(FOLD_C, "val")
    make_symlinks(os.path.join(FOLD_A, "val"), val_dir, "EGTEA")

    # ── 3. Re-key fold_a GT JSONs for EgoExoLearn and EGTEA ───────────────────
    print("\n=== Preparing GT JSON files ===")

    # Training GT: start from fold_a/gt_train.json (EgoExoLearn only)
    fold_a_gt_train = os.path.join(FOLD_A, "gt_train.json")
    fold_c_gt_train = os.path.join(FOLD_C, "gt_train.json")

    if os.path.exists(fold_a_gt_train):
        gt_train = rekey_gt(fold_a_gt_train, "fold_a", "fold_c")
        # HoloAssist entries will be added by greedy_search_gt.py --fold fold_c
        with open(fold_c_gt_train, "w") as f:
            json.dump(gt_train, f)
        print(f"  gt_train.json: {len(gt_train)} EgoExoLearn entries → {fold_c_gt_train}")
        print(f"  (HoloAssist GT will be appended by greedy_search_gt.py --fold fold_c)")
    else:
        print(f"  WARNING: {fold_a_gt_train} not found — run greedy_search_gt.py first")

    # Validation GT: re-key fold_a/gt_val.json (EGTEA)
    fold_a_gt_val = os.path.join(FOLD_A, "gt_val.json")
    fold_c_gt_val = os.path.join(FOLD_C, "gt_val.json")

    if os.path.exists(fold_a_gt_val):
        gt_val = rekey_gt(fold_a_gt_val, "fold_a", "fold_c")
        with open(fold_c_gt_val, "w") as f:
            json.dump(gt_val, f)
        print(f"  gt_val.json:   {len(gt_val)} EGTEA entries → {fold_c_gt_val}")
    else:
        print(f"  WARNING: {fold_a_gt_val} not found")

    print("\n=== fold_c summary ===")
    print(f"  train: {len(os.listdir(train_dir))} clips")
    print(f"  val:   {len(os.listdir(val_dir))} clips")
    print(f"\nNext: run greedy_search_gt.py --fold fold_c to generate HoloAssist GT,")
    print(f"then run: bash AutoGaze/train_streamgaze_custom.sh")


if __name__ == "__main__":
    main()
