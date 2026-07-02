#!/bin/bash
# Post-process deep-squat (agile model_2999) bench outputs:
#  1) rename agile_*.mp4 -> agile-deepsquat_*.mp4
#  2) caption R2/R3 videos -> results_deepsquat/captioned/
#  3) caption arena videos -> site/videos/arena/agile-deepsquat/
# Caption banner framework is rewritten AGILE -> AGILE-DEEPSQUAT.
set -u
BASE=/sda/lizhe/g1bench/results_deepsquat
PY=/sda/lizhe/miniforge3/envs/agile/bin/python
CAP=/sda/lizhe/g1bench/caption_videos.py
SITE=/sda/lizhe/g1bench/site
cd /sda/lizhe/g1bench

echo "=== 1) rename agile_ -> agile-deepsquat_ ==="
find "$BASE" -name 'agile_*.mp4' | while read -r f; do
  d=$(dirname "$f"); b=$(basename "$f"); nb=${b/agile_/agile-deepsquat_}
  [ "$b" != "$nb" ] && mv -f "$f" "$d/$nb"
done
echo "renamed; remaining agile_ videos: $(find "$BASE" -name 'agile_*.mp4' | wc -l)"

mkjc() { sed 's/"framework": "AGILE"/"framework": "AGILE-DEEPSQUAT"/' "$1" > "$2"; }

echo "=== 2) caption R2/R3 ==="
OUT=$BASE/captioned; mkdir -p "$OUT"
for t in walk_speed squat_box circle_pillar speed_sweep \
         walk_speed_psi0 squat_box_psi0 circle_pillar_psi0 \
         squat_sweep goto_ab pipeline_abc; do
  J=$BASE/$t/agile_${t}_results.jsonl
  [ -f "$J" ] || { echo "  [skip] no jsonl $t"; continue; }
  JC=$BASE/$t/cap_input.jsonl; mkjc "$J" "$JC"
  "$PY" "$CAP" --pairs "$BASE/$t/videos:$JC" --out "$OUT" --all 2>&1 | tail -1
done
echo "R2/R3 captioned: $(ls "$OUT"/*_cap.mp4 2>/dev/null | wc -l)"

echo "=== 3) caption arena (parallel) ==="
AOUT=$SITE/videos/arena/agile-deepsquat; mkdir -p "$AOUT"
ls -d "$BASE"/arena/agile-deepsquat/arena_M*/v* 2>/dev/null | xargs -P 10 -I{} bash -c '
  d="{}"; test=$(basename "$(dirname "$d")")
  J="$d/agile_${test}_results.jsonl"
  [ -f "$J" ] || exit 0
  JC="$d/cap_input.jsonl"
  sed "s/\"framework\": \"AGILE\"/\"framework\": \"AGILE-DEEPSQUAT\"/" "$J" > "$JC"
  /sda/lizhe/miniforge3/envs/agile/bin/python /sda/lizhe/g1bench/caption_videos.py \
     --pairs "$d/videos:$JC" --out '"$AOUT"' --all >/dev/null 2>&1
'
echo "arena captioned: $(ls "$AOUT"/*_cap.mp4 2>/dev/null | wc -l)"
echo "POST DONE"
