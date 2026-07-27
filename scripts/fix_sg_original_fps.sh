#!/bin/bash
# Repair non-overlay StreamGaze frames whose count does not match the viz frames.
#
# THE BUG. extract_sg_original_frames.sh extracts both variants with `-vf fps=10`,
# which samples by TIME. For holoassist the two encodes of the same footage declare
# DIFFERENT frame rates — same nb_frames, different duration:
#
#   R005-7July-GoPro   viz: 24.46 fps / 360.5 s / 8817 frames -> 3605 jpgs
#                      org: 29.83 fps / 295.6 s / 8817 frames -> 2956 jpgs
#
# So `original` is not truncated — it holds every frame — but `fps=10` lands on
# different moments. Measured against viz (normalised cross-correlation of the
# same index): 0.79 at frame 300, decaying to -0.34 by frame 2700, i.e. a
# completely different moment. data/dataset.py indexes frames by number and cuts
# at int(ts_sec * 10), so the student's stream would silently desynchronise from
# the teacher's and from the QA timestamps.
#
# THE FIX. Both encodes contain the same frame SEQUENCE, so retiming the original
# to the viz frame rate makes index k the same moment again:
#
#   fps_viz = nb_frames * 10 / (number of viz jpgs)
#   ffmpeg -r $fps_viz -i org.mp4 -vf fps=10 ...
#
# Verified on R005-7July-GoPro: 3605 jpgs, exactly matching viz, NCC 0.946-1.000
# at every probe (was 0.79 down to -0.34).
#
# Only videos whose count differs from viz are touched; aligned ones are skipped,
# so this is idempotent and safe to re-run.
#
#   nohup setsid scripts/fix_sg_original_fps.sh holoassist &
#
# Arg 1: dataset name (egtea | egoexolearn | holoassist). Default holoassist.

set -u
DS=${1:-holoassist}
JOBS=${JOBS:-8}

SG=/NHNHOME/VILAB/vilab_yj/datasets/trajgazemerge/StreamGaze_v2
TAR=$SG/videos_tars/videos_${DS}_original.tar
VIZ=$SG/frames/$DS/viz
OUT=$SG/frames/$DS/original
WORK=$SG/videos_fix_$DS
LOG=/NHNHOME/VILAB/vilab_yj/trajgaze/fix_${DS}_original.log

mkdir -p "$WORK"
echo "=== fix $DS original frames start $(date -Is) ===" >>"$LOG"

# 1. Which videos are misaligned? Compare jpg counts against viz.
BAD=$WORK/bad.txt
: >"$BAD"
for d in "$VIZ"/*/; do
    stem=$(basename "$d")
    nv=$(ls -1 "$VIZ/$stem" 2>/dev/null | wc -l)
    no=$(ls -1 "$OUT/$stem" 2>/dev/null | wc -l)
    [ "$nv" != "$no" ] && echo "$stem $nv $no" >>"$BAD"
done
N=$(wc -l <"$BAD")
echo "[fix] $N/$(ls -1 "$VIZ" | wc -l) videos misaligned" >>"$LOG"
if [ "$N" -eq 0 ]; then
    echo "=== fix $DS NOTHING TO DO $(date -Is) ===" >>"$LOG"; rm -rf "$WORK"; exit 0
fi

# 2. One tar pass for all of them — extracting members one at a time re-scans a 19 GB
#    archive per video.
echo "[fix] untarring $N members from $TAR" >>"$LOG"
awk '{print $1".mp4"}' "$BAD" >"$WORK/members.txt"
tar -xf "$TAR" -C "$WORK" -T "$WORK/members.txt" || {
    echo "[fix] FATAL: tar extraction failed" >>"$LOG"; exit 1; }

# 3. Re-extract each at the viz frame rate, into a staging dir, and only swap in
#    on an exact count match — a partial ffmpeg must never replace good frames.
one () {
    local stem="$1" target="$2"
    local mp4="$WORK/$stem.mp4" stage="$WORK/stage_$stem"
    if [ ! -f "$mp4" ]; then echo "[fix] MISSING mp4 $stem" >>"$LOG"; return 1; fi

    local nb fps
    nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
         -of csv=p=0 "$mp4" 2>/dev/null)
    if ! [ "$nb" -gt 0 ] 2>/dev/null; then
        echo "[fix] FAIL $stem: no nb_frames" >>"$LOG"; return 1; fi
    fps=$(python3 -c "print(f'{$nb*10/$target:.9f}')")

    rm -rf "$stage"; mkdir -p "$stage"
    ffmpeg -nostdin -v error -r "$fps" -i "$mp4" -vf fps=10 -q:v 2 \
           "$stage/frame_%06d.jpg" || {
        echo "[fix] FAIL $stem: ffmpeg" >>"$LOG"; rm -rf "$stage"; return 1; }

    local got
    got=$(ls -1 "$stage" | wc -l)
    if [ "$got" != "$target" ]; then
        echo "[fix] FAIL $stem: got $got want $target (fps=$fps nb=$nb)" >>"$LOG"
        rm -rf "$stage"; return 1
    fi
    rm -rf "$OUT/$stem" && mv "$stage" "$OUT/$stem" \
        && echo "[fix] ok $stem $got frames (fps=$fps)" >>"$LOG"
}
export -f one; export WORK OUT LOG

awk '{print $1, $2}' "$BAD" | xargs -P "$JOBS" -n 2 bash -c 'one "$0" "$1"'

# 4. Re-verify from scratch, then clean up.
REMAIN=0
for d in "$VIZ"/*/; do
    stem=$(basename "$d")
    nv=$(ls -1 "$VIZ/$stem" 2>/dev/null | wc -l)
    no=$(ls -1 "$OUT/$stem" 2>/dev/null | wc -l)
    [ "$nv" != "$no" ] && { REMAIN=$((REMAIN+1)); echo "[fix] STILL BAD $stem $nv vs $no" >>"$LOG"; }
done
echo "[fix] remaining misaligned: $REMAIN" >>"$LOG"
rm -rf "$WORK"
echo "=== fix $DS original frames DONE $(date -Is) ===" >>"$LOG"
