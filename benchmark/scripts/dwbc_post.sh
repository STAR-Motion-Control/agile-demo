#!/bin/bash
# Caption gr00t-wbc bench outputs (4090). --label already tags video filenames +
# framework as gr00t-wbc, so NO rename/framework-rewrite needed.
#  R2/R3 -> results_dwbc/captioned/    arena -> site/videos/arena/gr00t-wbc/
set -u
BASE=/sda/lizhe/g1bench/results_dwbc
CAP_PY=/sda/lizhe/miniforge3/envs/agile/bin/python   # has PIL+imageio+font
CAP=/sda/lizhe/g1bench/caption_videos.py
SITE=/sda/lizhe/g1bench/site
cd /sda/lizhe/g1bench

echo "=== caption R2/R3 ==="
OUT=$BASE/captioned; mkdir -p "$OUT"
for t in walk_speed squat_box circle_pillar speed_sweep \
         walk_speed_psi0 squat_box_psi0 circle_pillar_psi0 \
         squat_sweep goto_ab pipeline_abc; do
  J=$BASE/$t/results.jsonl
  [ -f "$J" ] || { echo "  [skip] $t"; continue; }
  "$CAP_PY" "$CAP" --pairs "$BASE/$t/videos:$J" --out "$OUT" --all 2>&1 | tail -1
done
echo "R2/R3 captioned: $(ls "$OUT"/*_cap.mp4 2>/dev/null | wc -l)"

echo "=== caption arena (parallel) ==="
AOUT=$SITE/videos/arena/gr00t-wbc; mkdir -p "$AOUT"
ls -d "$BASE"/arena/gr00t-wbc/arena_M*/v* 2>/dev/null | xargs -P 10 -I{} bash -c '
  d="{}"; J="$d/results.jsonl"
  [ -f "$J" ] || exit 0
  /sda/lizhe/miniforge3/envs/agile/bin/python /sda/lizhe/g1bench/caption_videos.py \
     --pairs "$d/videos:$J" --out '"$AOUT"' --all >/dev/null 2>&1
'
echo "arena captioned: $(ls "$AOUT"/*_cap.mp4 2>/dev/null | wc -l)"
echo "POST DONE"
