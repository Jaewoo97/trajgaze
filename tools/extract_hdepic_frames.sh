#!/bin/bash
set -u
SRC=/workspace/HD-EPIC/video
DST=/workspace/HD-EPIC/frames_extracted
LOG=/workspace/hdepic_extract.log
: > "$LOG"

extract_one() {
  local mp4="$1"
  local p=$(basename "$(dirname "$mp4")")
  local stem=$(basename "$mp4" .mp4)
  local out="$DST/$p/$stem"
  if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[skip $(date +%T)] $p/$stem (already extracted)" >> "$LOG"; return
  fi
  mkdir -p "$out"
  if ffmpeg -hide_banner -loglevel error -y -i "$mp4" \
      -vf "fps=10,scale=224:224:flags=area" -q:v 5 \
      "$out/frame_%06d.jpg" 2>>"$LOG"; then
    local n=$(ls "$out" 2>/dev/null | wc -l)
    echo "[done $(date +%T)] $p/$stem ($n frames)" >> "$LOG"
  else
    echo "[FAIL $(date +%T)] $p/$stem" >> "$LOG"
  fi
}
export -f extract_one
export DST LOG

find "$SRC" -name "*.mp4" -type f | sort | xargs -P 16 -I{} bash -c 'extract_one "$@"' _ {}
echo "[ALL DONE $(date +%T)]" >> "$LOG"
