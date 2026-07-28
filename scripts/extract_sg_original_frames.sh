#!/bin/bash
# Extract NON-OVERLAY StreamGaze frames for the true gaze-free experiment.
#
# The shipped frames under frames/{ds}/viz/ have the gaze marker (red circle +
# green dot) drawn into the pixels, so a model evaluated on them still receives
# the gaze location even when the trajectory-coordinate stream is removed. The
# `*_original.tar` videos are the same footage without the overlay.
#
# Extraction must match the viz frames exactly or the two conditions are not
# comparable: 10 fps (verified — 14968 frames over 1496.833 s for
# OP01-R01-PastaSalad), same 640x480, same frame_%06d.jpg naming starting at 1.
#
#   nohup setsid scripts/extract_sg_original_frames.sh egtea &
#
# Arg 1: dataset name (egtea | egoexolearn | holoassist). Default egtea.

set -u
DS=${1:-egtea}
JOBS=${JOBS:-8}                     # leave cores for the GPU jobs' dataloaders

SG=/NHNHOME/VILAB/vilab_yj/datasets/trajgazemerge/StreamGaze_v2
TAR=$SG/videos_tars/videos_${DS}_original.tar
VIZ=$SG/frames/$DS/viz
OUT=$SG/frames/$DS/original
WORK=$SG/videos_tmp_$DS
LOG=/NHNHOME/VILAB/vilab_yj/trajgaze/extract_${DS}_original.log

mkdir -p "$OUT" "$WORK"
echo "=== extract $DS original frames start $(date -Is) ===" >>"$LOG"

# Untar to a scratch dir first: extracting per-file from a 7GB+ tar repeatedly
# would re-scan the archive for every video.
echo "[extract] untarring $TAR -> $WORK" >>"$LOG"
tar -xf "$TAR" -C "$WORK"

one () {
    local mp4="$1"
    local stem
    stem=$(basename "$mp4" .mp4)
    local d="$OUT/$stem"
    local target
    target=$(ls -1 "$VIZ/$stem" 2>/dev/null | wc -l)

    # NEVER skip on a frame-count match. The retiming below derives the output count from
    # the viz count, so count parity is CONSTRUCTED, not observed — a stem extracted at the
    # wrong moments still counts correctly and would then be skipped forever. That guard
    # let 54/180 egoexolearn stems stay wrong across re-runs (kd_handoff_v2.md §7.4a).
    # Alignment is checked afterwards, by pixels: scripts/verify_frame_alignment.py.
    if [ "${FORCE:-0}" != "1" ] && grep -qxF "$stem" "$OUT/.aligned" 2>/dev/null; then
        echo "[extract] skip $stem (pixel-verified marker present)" >>"$LOG"
        return 0
    fi

    # Rebuild timestamps from the frame INDEX at the viz rate, so output frame k is viz
    # frame k. Both encodes hold the same frame sequence, so the rate follows from the viz
    # jpg count: fps_viz = nb_frames * 10 / n_viz_jpgs.
    #
    # setpts, NOT `-r` as an input option. `-r` reinterprets container timestamps, which is
    # a no-op on CFR input but re-spaces the frames of a VFR one. egoexolearn's originals
    # carry a bogus r_frame_rate (235/12) with a true average matching viz (22.30), i.e.
    # effectively VFR — `-r` put 54/180 stems on different moments while producing exactly
    # the right frame count. setpts=N/RATE/TB is index-based and immune to that.
    local vf="fps=10"
    if [ "$target" -gt 0 ]; then
        local nb
        nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
             -of csv=p=0 "$mp4" 2>/dev/null)
        if [ "$nb" -gt 0 ] 2>/dev/null; then
            vf="setpts=N/$(python3 -c "print(f'{$nb*10/$target:.9f}')")/TB,fps=10"
        fi
    fi

    rm -rf "$d"; mkdir -p "$d"
    ffmpeg -nostdin -v error -i "$mp4" -vf "$vf" -q:v 2 "$d/frame_%06d.jpg" \
        || { echo "[extract] FAIL $stem" >>"$LOG"; return 1; }
    local got
    got=$(ls -1 "$d" | wc -l)
    if [ "$target" -gt 0 ] && [ "$got" != "$target" ]; then
        echo "[extract] MISMATCH $stem: got $got want $target" >>"$LOG"
    else
        echo "[extract] ok $stem $got frames" >>"$LOG"
    fi
}
export -f one; export OUT LOG VIZ

find "$WORK" -name '*.mp4' | sort | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}

echo "[extract] done; $(ls -1 "$OUT" | wc -l) video dirs" >>"$LOG"
rm -rf "$WORK"

# Counting frames cannot detect the failure this script is most prone to, so the run is
# not complete until pixels agree. Stems that pass get a .aligned marker, which is what the
# skip guard above keys on — so a re-run repairs exactly the stems that are still wrong.
echo "[extract] verifying alignment by pixels" >>"$LOG"
if python3 "$REPO/scripts/verify_frame_alignment.py" "$DS" >>"$LOG" 2>&1; then
    ls -1 "$OUT" >"$OUT/.aligned"   # sibling manifest, never inside a frame dir
    echo "=== extract $DS original frames DONE — PIXEL-VERIFIED $(date -Is) ===" >>"$LOG"
else
    echo "=== extract $DS original frames DONE but ALIGNMENT FAILED — see above;" \
         "do not train on this tree $(date -Is) ===" >>"$LOG"
    exit 1
fi
