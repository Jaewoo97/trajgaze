"""
Project HD-EPIC Aria MPS eye gaze (CPF radians) onto camera-rgb image pixels.

For each of the 156 HD-EPIC recordings, emits
  /workspace/HD-EPIC/gaze/{stem}.json  =  {frame_NNNNNN.jpg: [px, py]}
in NORMALIZED [0, 1] coordinates (mirrors EgoGazeVQA's gaze_mapping format,
so dataset code that uses normalized gaze works without changes).

Inputs per recording:
  - /workspace/HD-EPIC/video/{P##}/{stem}_mp4_to_vrs_time_ns.csv
        mp4 frame time -> vrs_device_time_ns (== tracking_timestamp * 1000)
  - /workspace/HD-EPIC/SLAM-and-Gaze/{P##}/GAZE_HAND/mps_{stem}_vrs.zip
        eye_gaze/general_eye_gaze.csv  (yaw/pitch/depth in CPF radians)
  - /workspace/HD-EPIC/SLAM-and-Gaze/{P##}/SLAM/multi/{group}.zip
        slam/online_calibration.jsonl  (per-tracking-timestamp DeviceCalibration)
  - /workspace/HD-EPIC/SLAM-and-Gaze/{P##}/SLAM/multi/vrs_to_multi_slam.json
        recording stem -> multi group ID

Frames in /workspace/HD-EPIC/frames_extracted/{P##}/{stem}/ are indexed
1..N, so frame_000001.jpg <-> MP4 row 0 of the timestamp CSV, etc.

T_Device_CPF is the Aria Gen1 DVT-S CAD constant (extracted from the embedded
CSV inside projectaria_tools' _core_pybinds .so). T_Device_Camera comes from
the per-frame online_calibration entry. Math:
  T_Camera_CPF      = T_Device_Camera.inverse() @ T_Device_CPF  (CAD)
  gaze_pt_cpf       = mps.get_eyegaze_point_at_depth(yaw, pitch, depth)
  gaze_pt_camera    = T_Camera_CPF @ gaze_pt_cpf
  pixel             = rgb_calib.project(gaze_pt_camera)  (None if outside FOV)
  normalized_pixel  = pixel / 2880  (Aria native RGB resolution)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import zipfile

import numpy as np
from projectaria_tools.core import calibration, mps
from projectaria_tools.core.mps.utils import get_eyegaze_point_at_depth
from projectaria_tools.core.sophus import SE3

# Aria RGB sensor is mounted rotated; the SDK's make_upright=True option does
# this same CW90 post-multiply on T_Device_Camera before composing T_Camera_CPF.
# HD-EPIC's MP4 frames are pre-rotated to upright, so we must do the same.
_CW90 = SE3.from_matrix(np.array([[0, 1, 0, 0],
                                    [-1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=float))

ROOT = "/workspace/HD-EPIC"
OUT_DIR = os.path.join(ROOT, "gaze")
ARIA_RGB_SIZE = 2880  # native Aria Gen1 RGB image size

# Aria Gen1 DVT-S CAD: T_Cpf_camera-slam-left (= T_Cpf_Device)
# Source: embedded CSV in projectaria_tools _core_pybinds .so
# Row: label, tx, ty, tz, Yi, Yj, Yk, Zi, Zj, Zk
_SLAM_LEFT_CAD = ("camera-slam-left,"
                  "0.071351,0.002372,0.008454,"
                  "0.793353,0.000000,-0.608761,"
                  "0.607927,-0.052336,0.792266")


def _cad_T_Cpf_Device() -> SE3:
    vals = _SLAM_LEFT_CAD.split(",")
    t  = np.array([float(vals[1]), float(vals[2]), float(vals[3])])
    yA = np.array([float(vals[4]), float(vals[5]), float(vals[6])])
    zA = np.array([float(vals[7]), float(vals[8]), float(vals[9])])
    xA = np.cross(yA, zA)
    R = np.column_stack([xA, yA, zA])
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = t
    return SE3.from_matrix(M)


T_CPF_DEVICE = _cad_T_Cpf_Device()
T_DEVICE_CPF = T_CPF_DEVICE.inverse()


def _participant_of(stem: str) -> str:
    return stem.split("-", 1)[0]


def _list_recordings() -> list[tuple[str, str]]:
    out = []
    fb = os.path.join(ROOT, "frames_extracted")
    for p in sorted(os.listdir(fb)):
        pd = os.path.join(fb, p)
        if not os.path.isdir(pd): continue
        for stem in sorted(os.listdir(pd)):
            if os.path.isdir(os.path.join(pd, stem)):
                out.append((p, stem))
    return out


def _multi_group_for(p: str, stem: str) -> int | None:
    mp = os.path.join(ROOT, "SLAM-and-Gaze", p, "SLAM/multi/vrs_to_multi_slam.json")
    if not os.path.exists(mp): return None
    with open(mp) as f:
        m = json.load(f)
    return m.get(f"{p}/{stem}.vrs")


def _extract_to_tmp(zip_path: str, inner: str, out_path: str) -> bool:
    if not os.path.exists(zip_path): return False
    with zipfile.ZipFile(zip_path) as z:
        if inner not in z.namelist():
            # fallback prefix-match if naming differs
            for name in z.namelist():
                if name.endswith(os.path.basename(inner)):
                    inner = name; break
            else:
                return False
        with z.open(inner) as fin, open(out_path, "wb") as fout:
            fout.write(fin.read())
    return True


def _load_online_calib(p: str, group: int, tmpdir: str) -> list | None:
    base = os.path.join(ROOT, "SLAM-and-Gaze", p, "SLAM/multi")
    zp = os.path.join(base, f"{group}.zip")
    oc_path = os.path.join(tmpdir, f"oc_{p}_{group}.jsonl")
    # Some groups are unpacked as dirs (e.g. P01/SLAM/multi/8/)
    dir_oc = os.path.join(base, str(group), "slam", "online_calibration.jsonl")
    if os.path.exists(dir_oc):
        return mps.read_online_calibration(dir_oc)
    if _extract_to_tmp(zp, f"{group}/slam/online_calibration.jsonl", oc_path):
        return mps.read_online_calibration(oc_path)
    return None


def _load_gaze(p: str, stem: str, tmpdir: str) -> list | None:
    zp = os.path.join(ROOT, "SLAM-and-Gaze", p, f"GAZE_HAND/mps_{stem}_vrs.zip")
    gz_path = os.path.join(tmpdir, f"gz_{stem}.csv")
    inner = f"mps_{stem}_vrs/eye_gaze/general_eye_gaze.csv"
    if not _extract_to_tmp(zp, inner, gz_path):
        return None
    return mps.read_eyegaze(gz_path)


def _build_timestamp_index(items) -> np.ndarray:
    return np.array([int(it.tracking_timestamp.total_seconds() * 1e9)
                     for it in items], dtype=np.int64)


def _nearest(timestamps_ns: np.ndarray, t_query_ns: int) -> int:
    return int(np.searchsorted(timestamps_ns, t_query_ns).clip(1, len(timestamps_ns) - 1) - 1) \
           if False else int(np.argmin(np.abs(timestamps_ns - t_query_ns)))


def _fallback_oc_for_participant(p: str, tmpdir: str) -> list | None:
    """For other_aligned recordings without a multi mapping, fall back to
    the first available multi-group's online_calibration in the same participant."""
    mp = os.path.join(ROOT, "SLAM-and-Gaze", p, "SLAM/multi/vrs_to_multi_slam.json")
    if not os.path.exists(mp): return None
    with open(mp) as f:
        m = json.load(f)
    groups_seen: set[int] = set()
    for _, g in m.items():
        if g in groups_seen: continue
        groups_seen.add(g)
        oc = _load_online_calib(p, g, tmpdir)
        if oc: return oc
    return None


def project_one_recording(p: str, stem: str, tmpdir: str,
                          overwrite: bool = False) -> dict | None:
    out_path = os.path.join(OUT_DIR, f"{stem}.json")
    if os.path.exists(out_path) and not overwrite:
        return None

    # 1. Load timestamp mapping (mp4 frame -> vrs_device_time_ns).
    # IMPORTANT: extracted frames are at 10 fps (ffmpeg "fps=10" from 30-fps
    # MP4 — see extract_hdepic_frames.sh), so extracted frame k (1-indexed)
    # corresponds to MP4 frame (3k-2) (1-indexed) under round-nearest. We
    # build a list of MP4-frame → vrs_ns and resolve below by multiplying.
    ts_csv = os.path.join(ROOT, "video", p, f"{stem}_mp4_to_vrs_time_ns.csv")
    if not os.path.exists(ts_csv):
        return {"status": "missing_ts_csv", "stem": stem}
    mp4_vrs_ns: list[int] = []   # 0-indexed list of MP4-frame vrs times
    with open(ts_csv) as f:
        for row in csv.DictReader(f):
            mp4_vrs_ns.append(int(row["vrs_device_time_ns"]))

    def extracted_to_vrs_ns(k: int) -> int | None:
        # k is 1-indexed extracted frame number. 30->10 fps decimation = factor 3.
        mp4_idx0 = (k - 1) * 3   # 0-indexed MP4 frame
        if 0 <= mp4_idx0 < len(mp4_vrs_ns):
            return mp4_vrs_ns[mp4_idx0]
        return None

    # 2. Load online_calibration (with fallback for unmapped recordings OR
    # mapped-but-missing-zip cases like P05 groups 17-25 where the multi zip
    # was simply never uploaded).
    group = _multi_group_for(p, stem)
    oc_list = None
    if group is not None:
        oc_list = _load_online_calib(p, group, tmpdir)
    if not oc_list:
        oc_list = _fallback_oc_for_participant(p, tmpdir)
    if not oc_list:
        return {"status": "no_online_calib", "stem": stem}

    # 3. Load gaze. If absent (rare other_aligned recordings), write all-None
    # JSON so downstream dataset code keeps the gaze-missing fallback.
    gz_list = _load_gaze(p, stem, tmpdir)
    if not gz_list:
        fdir = os.path.join(ROOT, "frames_extracted", p, stem)
        frame_files = sorted(fn for fn in os.listdir(fdir)
                             if fn.startswith("frame_") and fn.endswith(".jpg"))
        out = {fn: None for fn in frame_files}
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f)
        return {"status": "ok_no_gaze", "stem": stem, "n_frames": len(frame_files),
                "in_fov": 0, "out_fov": 0, "no_mp4_ts": 0, "group": group}

    oc_ts = _build_timestamp_index(oc_list)
    gz_ts = _build_timestamp_index(gz_list)

    # 4. Iterate over extracted frame files (only frames we actually have)
    fdir = os.path.join(ROOT, "frames_extracted", p, stem)
    frame_files = sorted(fn for fn in os.listdir(fdir) if fn.startswith("frame_") and fn.endswith(".jpg"))

    out: dict[str, list[float] | None] = {}
    n_in_fov = 0
    n_out_fov = 0
    n_no_mp4 = 0
    for fname in frame_files:
        idx = int(fname[len("frame_"):-len(".jpg")])
        t_vrs_ns = extracted_to_vrs_ns(idx)
        if t_vrs_ns is None:
            out[fname] = None
            n_no_mp4 += 1
            continue

        gi = _nearest(gz_ts, t_vrs_ns)
        oi = _nearest(oc_ts, t_vrs_ns)
        g = gz_list[gi]
        rgb_cal = oc_list[oi].get_camera_calib("camera-rgb")

        depth = g.depth if g.depth and g.depth > 0 else 1.0
        gaze_cpf = get_eyegaze_point_at_depth(g.yaw, g.pitch, depth)

        # make_upright: post-multiply T_Device_Camera by CW90 so that the
        # projected pixel ends up in the upright (rotated-90°-CW) image frame
        # that matches HD-EPIC's MP4 export.
        T_Device_Camera_upright = rgb_cal.get_transform_device_camera() @ _CW90
        T_Camera_Cpf = T_Device_Camera_upright.inverse() @ T_DEVICE_CPF
        gaze_cam = T_Camera_Cpf @ gaze_cpf
        pixel = rgb_cal.project(gaze_cam)
        if pixel is None:
            out[fname] = None
            n_out_fov += 1
        else:
            nx = float(pixel[0]) / ARIA_RGB_SIZE
            ny = float(pixel[1]) / ARIA_RGB_SIZE
            # Clip to [0, 1] (extreme gaze near FOV edge can return slightly OOB)
            nx_c = max(0.0, min(1.0, nx))
            ny_c = max(0.0, min(1.0, ny))
            out[fname] = [nx_c, ny_c]
            n_in_fov += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(out, f)
    os.replace(tmp_out, out_path)

    return {
        "status": "ok",
        "stem": stem,
        "n_frames": len(frame_files),
        "in_fov": n_in_fov,
        "out_fov": n_out_fov,
        "no_mp4_ts": n_no_mp4,
        "group": group,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N recordings (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--participant", default=None,
                    help="Process only this participant (e.g. P01)")
    args = ap.parse_args()

    recs = _list_recordings()
    if args.participant:
        recs = [r for r in recs if r[0] == args.participant]
    if args.limit:
        recs = recs[:args.limit]
    print(f"Will project {len(recs)} recordings → {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        ok = skipped = failed = 0
        for i, (p, stem) in enumerate(recs):
            try:
                res = project_one_recording(p, stem, tmpdir, overwrite=args.overwrite)
            except Exception as e:
                res = {"status": "exception", "stem": stem, "err": str(e)}
            if res is None:
                skipped += 1
                msg = "SKIP (exists)"
            elif res["status"] == "ok":
                ok += 1
                msg = f"ok {res['in_fov']}/{res['n_frames']} in FOV (group={res['group']})"
            else:
                failed += 1
                msg = f"FAIL {res['status']}" + (f" — {res.get('err','')}" if 'err' in res else '')
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(recs)}] {stem} → {msg}   ({elapsed:.1f}s)", flush=True)
    print(f"\nDONE: ok={ok} skipped={skipped} failed={failed}  total={len(recs)}")


if __name__ == "__main__":
    main()
