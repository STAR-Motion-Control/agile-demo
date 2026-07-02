#!/bin/bash
# Single gr00t-wbc (GR00T decoupled-WBC) bench job wrapper (4090).
# Usage: dwbc_job.sh <out_subpath> <bench_dwbc args...>
set -u
export MUJOCO_GL=egl
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
PY=/sda/lizhe/miniforge3/envs/homie/bin/python
BASE=/sda/lizhe/g1bench/results_dwbc
cd /sda/lizhe/g1bench
name=$1; shift
out=$BASE/$name
mkdir -p "$out" "$BASE/logs"
logf="$BASE/logs/$(echo "$name" | tr / _).log"
echo "[$(date +%H:%M:%S)] START $name :: $*" >> "$BASE/logs/driver.log"
"$PY" bench_dwbc.py --label gr00t-wbc --video policy --out-dir "$out" "$@" > "$logf" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] DONE($rc) $name" >> "$BASE/logs/driver.log"
exit $rc
