"""
Generate fold_b GT JSON files from fold_a's results without rerunning greedy search.

fold_a/train → EgoExoLearn clips   fold_a/gt_train.json  keys: fold_a/train/<clip>.mp4
fold_a/val   → EGTEA clips         fold_a/gt_val.json    keys: fold_a/val/<clip>.mp4

fold_b/train → EGTEA clips         fold_b/gt_train.json  keys: fold_b/train/<clip>.mp4
fold_b/val   → EgoExoLearn clips   fold_b/gt_val.json    keys: fold_b/val/<clip>.mp4

Since fold_b/train and fold_a/val share the same underlying clips (EGTEA),
and fold_b/val and fold_a/train share EgoExoLearn clips, we just need to
re-key the entries.

Usage:
    conda run -n gaze python EgoGazeVQA/tools/generate_fold_b_gt.py
"""

import os
import json

DATA_DIR = "/workspace/datasets/StreamGaze/autogaze_data"


def rekey(src_json, src_prefix, dst_prefix, dst_json):
    """
    Load src_json, replace each key's src_prefix with dst_prefix, write dst_json.
    """
    with open(src_json) as f:
        src = json.load(f)

    dst = {}
    for key, value in src.items():
        new_key = key.replace(src_prefix + "/", dst_prefix + "/", 1)
        dst[new_key] = value

    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(dst, f)
    print(f"  {src_json} ({len(src)} entries) → {dst_json} (re-keyed as {dst_prefix}/...)")


def main():
    # fold_b/gt_train.json  ← fold_a/gt_val.json  (EGTEA: fold_a/val → fold_b/train)
    rekey(
        src_json   = os.path.join(DATA_DIR, "fold_a", "gt_val.json"),
        src_prefix = "fold_a/val",
        dst_prefix = "fold_b/train",
        dst_json   = os.path.join(DATA_DIR, "fold_b", "gt_train.json"),
    )

    # fold_b/gt_val.json  ← fold_a/gt_train.json  (EgoExoLearn: fold_a/train → fold_b/val)
    rekey(
        src_json   = os.path.join(DATA_DIR, "fold_a", "gt_train.json"),
        src_prefix = "fold_a/train",
        dst_prefix = "fold_b/val",
        dst_json   = os.path.join(DATA_DIR, "fold_b", "gt_val.json"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
