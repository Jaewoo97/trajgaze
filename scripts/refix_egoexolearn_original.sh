#!/usr/bin/env bash
# Re-extract the egoexolearn `original` frames that extract_sg_original_frames.sh got wrong.
#
# ROOT CAUSE. That script retimes with `-r RATE` as an *input* option, which reinterprets
# the container timestamps. egoexolearn's original encodes carry a bogus r_frame_rate tag
# (e.g. 235/12 = 19.58 fps) while their true average matches viz (22.30 fps) — i.e. they are
# effectively VFR. Forcing an input rate on a VFR stream re-spaces the frames, so output
# frame k is no longer the same moment as viz frame k.
#
# It still produced exactly the right NUMBER of frames, because the forced rate is derived
# from the viz jpg count. So the script constructs count parity by design and the verifier
# then checks count parity — the check cannot fail, and 54/180 stems were wrong anyway.
#
# FIX. Rebuild timestamps from the frame INDEX instead: `setpts=N/RATE/TB,fps=10`. This is
# what the published holoassist build used. Measured on beeabf86-...: MAD vs viz drops from
# 46.98 (wrong footage) to 2.80 (JPEG noise = same moment, marker gone).
#
# Verification here is by PIXELS, never by count.
#
#   nohup setsid scripts/refix_egoexolearn_original.sh &
set -u

cd "$(dirname "$0")/.." || exit 1
source env.sh

DS=egoexolearn
TAR=$SG_ROOT/videos_tars/videos_${DS}_original.tar
VIZ=$SG_ROOT/frames/$DS/viz
CUR=$SG_ROOT/frames/$DS/original
WORK=$SG_ROOT/_refix_$DS
STAGE=$WORK/frames
VID=$WORK/videos
LIST=${1:-$WORK/stems.txt}
LOG=$REPO/refix_${DS}_original.log
JOBS=${JOBS:-4}

mkdir -p "$STAGE" "$VID"
echo "=== refix $DS start $(date -Is) ===" >>"$LOG"

n=$(wc -l <"$LIST")
echo "[refix] $n stems to redo" >>"$LOG"

echo "[refix] extracting mp4s from tar" >>"$LOG"
while read -r s; do
    [ -s "$VID/$s.mp4" ] || tar -xf "$TAR" -C "$VID" "$s.mp4" 2>/dev/null \
        || echo "[refix] TAREXTRACT failed $s" >>"$LOG"
done <"$LIST"
echo "[refix] mp4s on disk: $(ls -1 "$VID" | wc -l)" >>"$LOG"

one () {
    local s=$1
    local mp4="$VID/$s.mp4" d="$STAGE/$s"
    [ -f "$mp4" ] || { echo "[refix] NOVIDEO $s" >>"$LOG"; return 1; }
    local target nb rate
    target=$(ls -1 "$VIZ/$s" 2>/dev/null | wc -l)
    [ "$target" -gt 0 ] || { echo "[refix] NOVIZ $s" >>"$LOG"; return 1; }
    nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$mp4" 2>/dev/null)
    [ "$nb" -gt 0 ] 2>/dev/null || { echo "[refix] NONBFRAMES $s" >>"$LOG"; return 1; }
    rate=$(python3 -c "print(f'{$nb*10/$target:.9f}')")

    rm -rf "$d"; mkdir -p "$d"
    ffmpeg -nostdin -v error -i "$mp4" \
        -vf "setpts=N/$rate/TB,fps=10" -q:v 2 "$d/frame_%06d.jpg" \
        || { echo "[refix] FFMPEGFAIL $s" >>"$LOG"; return 1; }
    local got; got=$(ls -1 "$d" | wc -l)
    if [ "$got" != "$target" ]; then
        echo "[refix] COUNT $s: got $got want $target" >>"$LOG"
    else
        echo "[refix] ok $s ($got frames, rate=$rate)" >>"$LOG"
    fi
}
export -f one
export VID STAGE VIZ LOG

xargs -a "$LIST" -P "$JOBS" -I{} bash -c 'one "$@"' _ {}

echo "=== refix $DS extraction DONE $(date -Is); verifying by pixels ===" >>"$LOG"
python3 "$REPO/scripts/verify_frame_alignment.py" "$DS" "$STAGE" >>"$LOG" 2>&1
echo "=== refix $DS DONE $(date -Is) ===" >>"$LOG"
