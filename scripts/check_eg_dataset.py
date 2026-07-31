#!/usr/bin/env python
"""Is this machine's EgoGazeVQA the same data the 25% runs were measured on?

Written for the port in docs/kd_handoff_v4_egogazevqa_25.md. The receiving machine
already has the dataset, so the question is never "download it" but "is it the same
one" -- a silently different frame set produces numbers that cannot be compared to the
reference runs and nothing else would catch it.

Every EXPECTED value below was measured on the b200 that produced those runs.

Checks, in the order they fail cheapest-first:
  1. metadata.csv        size, md5 and line count. Cheap and decisive: it pins the item
                         set, the splits and the MCQ options in one hash.
  2. video directories   per dataset, per frame variant.
  3. frame files         per dataset, per frame variant.
  4. gaze vs no_gaze     identical basenames per video. egogaze_dataset.py:49-52 depends
                         on this -- the sampler picks indices from one variant's listing
                         while gaze/hand/interaction are keyed by frame basename, so a
                         mismatch silently misaligns the trajectory streams instead of
                         raising.
  5. loader split sizes  train 1265 / test 485, via the real dataset class (--with-loader;
                         off by default because it imports torch).

Usage:
    python scripts/check_eg_dataset.py                 # steps 1-4
    python scripts/check_eg_dataset.py --with-loader   # + step 5
    python scripts/check_eg_dataset.py --quick         # steps 1-2 only, no frame walk
"""
import argparse
import hashlib
import os
import sys

ROOT = os.environ.get("EG_ROOT", "/workspace/EgoGazeVQA")
VARIANTS = ("gaze", "no_gaze")

EXPECTED_METADATA = {"lines": 1751, "bytes": 946793,
                     "md5": "fdcb4f8424fbcef3fa680a22d20b91e9"}
# frames are per variant; gaze and no_gaze hold the same count by construction
EXPECTED = {
    "egtea":  {"videos": 82,  "frames": 93755},
    "ego4d":  {"videos": 27,  "frames": 66017},
    "egoexo": {"videos": 154, "frames": 231928},
}
EXPECTED_SPLITS = {"train": 1265, "test": 485}

OK, BAD = "ok", "MISMATCH"
results = []


def record(label, got, want, note=""):
    good = got == want
    results.append((label, got, want, good, note))
    print(f"  [{OK if good else BAD}] {label}: got {got}"
          + ("" if good else f", expected {want}")
          + (f"  ({note})" if note else ""), flush=True)
    return good


def check_metadata():
    print("metadata.csv")
    path = os.path.join(ROOT, "metadata.csv")
    if not os.path.exists(path):
        record("exists", False, True, path)
        return
    data = open(path, "rb").read()
    record("bytes", len(data), EXPECTED_METADATA["bytes"])
    record("md5", hashlib.md5(data).hexdigest(), EXPECTED_METADATA["md5"])
    record("lines", data.count(b"\n") + (0 if data.endswith(b"\n") else 1),
           EXPECTED_METADATA["lines"], "header + 1750 items")


def listdirs(path):
    if not os.path.isdir(path):
        return None
    with os.scandir(path) as it:
        return sorted(e.name for e in it if e.is_dir())


def frames_in(path):
    """Frame basenames under one video directory."""
    if not os.path.isdir(path):
        return set()
    with os.scandir(path) as it:
        return {e.name for e in it if e.is_file() and e.name.endswith(".jpg")}


def check_dataset(ds, quick):
    print(f"\n{ds}")
    per_variant = {}
    for v in VARIANTS:
        vids = listdirs(os.path.join(ROOT, ds, v))
        if vids is None:
            record(f"{v}/ exists", False, True, os.path.join(ROOT, ds, v))
            continue
        per_variant[v] = vids
        record(f"{v}/ videos", len(vids), EXPECTED[ds]["videos"])

    if quick or len(per_variant) < len(VARIANTS):
        return

    # One walk serves both the frame count and the cross-variant name comparison.
    counts, mismatched = {v: 0 for v in VARIANTS}, []
    for vid in per_variant["no_gaze"]:
        names = {}
        for v in VARIANTS:
            names[v] = frames_in(os.path.join(ROOT, ds, v, vid))
            counts[v] += len(names[v])
        if names["gaze"] != names["no_gaze"]:
            mismatched.append(vid)

    for v in VARIANTS:
        record(f"{v}/ frames", counts[v], EXPECTED[ds]["frames"])
    record("gaze/no_gaze basenames identical", len(mismatched), 0,
           "" if not mismatched else "e.g. " + ", ".join(mismatched[:3]))


def check_loader():
    print("\nloader (EgoGazeVQADataset)")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from TrajGazeMerge.data.egogaze_dataset import EgoGazeVQADataset
    except Exception as e:
        record("import", f"failed: {type(e).__name__}", "ok")
        return
    for split, want in EXPECTED_SPLITS.items():
        try:
            record(f"split={split}", len(EgoGazeVQADataset(split=split)), want)
        except Exception as e:
            record(f"split={split}", f"failed: {type(e).__name__}: {e}", want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the frame walk; metadata and video counts only")
    ap.add_argument("--with-loader", action="store_true",
                    help="also build the dataset class and check split sizes (imports torch)")
    args = ap.parse_args()

    print(f"EG_ROOT = {ROOT}")
    print("reference: the b200 runs behind docs/kd_handoff_v4_egogazevqa_25.md\n")

    check_metadata()
    for ds in EXPECTED:
        check_dataset(ds, args.quick)
    if args.with_loader:
        check_loader()

    bad = [r for r in results if not r[3]]
    print("\n" + "=" * 60)
    if bad:
        print(f"FAIL — {len(bad)} of {len(results)} checks mismatched:")
        for label, got, want, _, _ in bad:
            print(f"  {label}: got {got}, expected {want}")
        print("\nThe frames differ from the reference runs. Re-fetch from")
        print("  https://huggingface.co/datasets/Peanuttoad/gaze_dataset_full")
        print("  -> EgoGazeVQA/{egtea,ego4d,egoexo}_no_gaze*.tar and *_gaze*.tar at the")
        print("     repository ROOT (not under aaai/, which holds only checkpoints),")
        print("     then extract with the repo root restore.sh.")
        return 1
    print(f"PASS — all {len(results)} checks match the reference dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
