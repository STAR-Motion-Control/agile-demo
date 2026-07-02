#!/bin/bash
# Caption ljk-falcon-v5 bench outputs (4090-lab). --label already tags video
# filenames + framework as ljk-falcon-v5, so NO rename/framework-rewrite needed.
#  R2/R3 -> results_v5/captioned/    arena -> results_v5/arena_cap/
set -u
BASE=/hhd2/ljk/g1bench/results_v5
PY=/hhd2/ljk/miniconda3/envs/fcreal/bin/python
CAP=/hhd2/ljk/g1bench/caption_videos.py
cd /hhd2/ljk/g1bench

echo "=== caption R2/R3 ==="
OUT=$BASE/captioned; mkdir -p "$OUT"
for t in walk_speed squat_box circle_pillar speed_sweep \
         walk_speed_psi0 squat_box_psi0 circle_pillar_psi0 \
         squat_sweep goto_ab pipeline_abc; do
  J=$BASE/$t/results.jsonl
  [ -f "$J" ] || { echo "  [skip] no jsonl $t"; continue; }
  "$PY" "$CAP" --pairs "$BASE/$t/videos:$J" --out "$OUT" --all 2>&1 | tail -1
done
echo "R2/R3 captioned: $(ls "$OUT"/*_cap.mp4 2>/dev/null | wc -l)"

echo "=== caption arena (parallel) ==="
AOUT=$BASE/arena_cap; mkdir -p "$AOUT"
ls -d "$BASE"/arena/ljk-falcon-v5/arena_M*/v* 2>/dev/null | xargs -P 8 -I{} bash -c '
  d="{}"; J="$d/results.jsonl"
  [ -f "$J" ] || exit 0
  /hhd2/ljk/miniconda3/envs/fcreal/bin/python /hhd2/ljk/g1bench/caption_videos.py \
     --pairs "$d/videos:$J" --out '"$AOUT"' --all >/dev/null 2>&1
'
echo "arena captioned: $(ls "$AOUT"/*_cap.mp4 2>/dev/null | wc -l)"
echo "POST DONE"
