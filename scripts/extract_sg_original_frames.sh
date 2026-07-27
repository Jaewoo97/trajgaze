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

    # Skip only if it already MATCHES viz. A non-empty dir is not enough: the two
    # encodes can declare different frame rates (holoassist does — same nb_frames,
    # 24.46 vs 29.83 fps), and `-vf fps=10` samples by time, so a plain extraction
    # lands on different moments while looking perfectly healthy.
    if [ "$target" -gt 0 ] && [ "$(ls -1 "$d" 2>/dev/null | wc -l)" = "$target" ]; then
        echo "[extract] skip $stem (matches viz: $target)" >>"$LOG"
        return 0
    fi

    # Retime the input to the viz frame rate so frame k is the same moment in both
    # variants. Both encodes hold the same frame sequence, so fps_viz follows from
    # the viz jpg count: fps_viz = nb_frames * 10 / n_viz_jpgs.
    local rate=""
    if [ "$target" -gt 0 ]; then
        local nb
        nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
             -of csv=p=0 "$mp4" 2>/dev/null)
        if [ "$nb" -gt 0 ] 2>/dev/null; then
            rate="-r $(python3 -c "print(f'{$nb*10/$target:.9f}')")"
        fi
    fi

    rm -rf "$d"; mkdir -p "$d"
    ffmpeg -nostdin -v error $rate -i "$mp4" -vf fps=10 -q:v 2 "$d/frame_%06d.jpg" \
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
echo "=== extract $DS original frames DONE $(date -Is) ===" >>"$LOG"
