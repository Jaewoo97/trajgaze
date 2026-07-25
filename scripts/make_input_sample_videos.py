"""Emit the EXACT data fed to the VLM: for each of the 3 sources, take 3 distinct
clips' vlm_frame_paths (with GAZE_OVERLAY=1, same as training) and encode them to
mp4 — pristine frames, no added labels — plus the question/options/answer text.
"""
import os
os.environ["GAZE_OVERLAY"] = "1"   # must be set before dataset modules import

import sys, json
import numpy as np
from PIL import Image
import imageio.v2 as imageio

sys.path.insert(0, "/workspace/trajgaze_st")

from TrajGazeMerge.data.dataset import StreamGazeMergeDataset
from TrajGazeMerge.data.egogaze_dataset import EgoGazeVQADataset
from TrajGazeMerge.data.hdepic_dataset import HDEpicDataset

OUT = "/tmp/input_samples"
os.makedirs(OUT, exist_ok=True)
W, FPS = 384, 10


def _even(x):
    x = int(round(x))
    return x + (x & 1)


def make_video(frame_paths, out_path):
    w, h = Image.open(frame_paths[0]).convert("RGB").size
    H = _even(W * h / w)
    wr = imageio.get_writer(out_path, fps=FPS, codec="libx264",
                            quality=8, macro_block_size=None)
    for fp in frame_paths:
        wr.append_data(np.asarray(Image.open(fp).convert("RGB").resize((W, H))))
    wr.close()
    return W, H


def sample(ds, n=3):
    """One distinct clip from each of n evenly-spaced regions, for recording variety."""
    out, seen = [], set()
    L = len(ds)
    starts = [int(L * k / n) for k in range(n)]
    for s in starts:
        for i in list(range(s, L)) + list(range(0, s)):
            try:
                it = ds[i]
            except Exception:
                continue
            fp = (it or {}).get("vlm_frame_paths") or []
            if not fp:
                continue
            key = (os.path.dirname(fp[0]), os.path.basename(fp[0]), os.path.basename(fp[-1]))
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
            break
    return out


records = []
sources = [
    ("streamgaze", StreamGazeMergeDataset(split="test")),
    ("egogazevqa", EgoGazeVQADataset(split="test")),
    ("hdepic",     HDEpicDataset(split="test", simple=True)),
]
for name, ds in sources:
    for j, it in enumerate(sample(ds, 3), 1):
        fp = it["vlm_frame_paths"]
        out_path = os.path.join(OUT, f"{name}_{j}.mp4")
        size = make_video(fp, out_path)
        rec = {
            "source": name,
            "video_file": out_path,
            "frame_source_dir": os.path.dirname(fp[0]),
            "first_frame": os.path.basename(fp[0]),
            "last_frame": os.path.basename(fp[-1]),
            "n_frames_fed_to_vlm": len(fp),
            "displayed_size": list(size),
            "dataset_tag": it.get("dataset"),
            "task": it.get("task"),
            "question": it.get("question"),
            "options": it.get("options"),
            "answer": it.get("answer"),
        }
        records.append(rec)
        print(f"\n[{name} #{j}] {len(fp)} frames  ->  {out_path}")
        print(f"    frames from : {rec['frame_source_dir']}")
        print(f"    Q: {rec['question']}")
        print(f"    options: {rec['options']}")
        print(f"    answer: {rec['answer']}   task: {rec['task']}")

with open(os.path.join(OUT, "input_manifest.json"), "w") as f:
    json.dump(records, f, indent=2)
print(f"\nWROTE {os.path.join(OUT, 'input_manifest.json')}  ({len(records)} clips)")
