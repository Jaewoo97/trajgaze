"""
Organise rendered visualisation PNGs into task / OK|WRONG subfolders.

Filename convention: sample_<NNN>_<task_name>_(OK|WRONG).png
                            ↓
        out_dir/<task_name>/(OK|WRONG)/sample_<NNN>_<task>_<status>.png

Usage:
    python -m TrajGazeMerge.eval.organize_by_task \
        --dir /workspace/trajgaze/TrajGazeMerge/eval_results/patch_only
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PATTERN = re.compile(r"^sample_(\d+)_(.+)_(OK|WRONG)\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing sample_*.png")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of move (default: move)")
    args = ap.parse_args()

    root = Path(args.dir)
    pngs = sorted(root.glob("sample_*.png"))
    if not pngs:
        print(f"No sample_*.png files in {root}")
        return

    counts: dict[tuple[str, str], int] = {}
    skipped: list[str] = []

    for p in pngs:
        m = PATTERN.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        _, task, status = m.groups()
        dst_dir = root / task / status
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / p.name
        if args.copy:
            shutil.copy2(p, dst)
        else:
            shutil.move(str(p), str(dst))
        counts[(task, status)] = counts.get((task, status), 0) + 1

    print(f"\nOrganised {sum(counts.values())} files in {root}")
    print(f"Action: {'copy' if args.copy else 'move'}\n")

    tasks = sorted({t for t, _ in counts})
    for task in tasks:
        ok = counts.get((task, "OK"), 0)
        wr = counts.get((task, "WRONG"), 0)
        total = ok + wr
        acc = 100 * ok / total if total else 0
        print(f"  {task:<45}  OK={ok:3d}  WRONG={wr:3d}  total={total:3d}  acc={acc:5.1f}%")

    if skipped:
        print(f"\nSkipped {len(skipped)} files (name didn't match pattern):")
        for n in skipped[:10]:
            print(f"  {n}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
