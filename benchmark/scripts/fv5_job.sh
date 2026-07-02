#!/bin/bash
# Single ljk-falcon-v5 (FALCON g1_29dof_v5_1.onnx) bench job wrapper (4090-lab).
# Usage: fv5_job.sh <out_subpath> <bench_falcon args...>
set -u
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/hhd2/ljk/miniconda3/envs/fcreal/bin/python
BENCH=/hhd2/ljk/g1bench/bench_falcon.py
MODEL=/hhd2/ljk/FALCON/sim2real/models/falcon/g1_29dof_v5_1.onnx
REPO=/hhd2/ljk/FALCON
BASE=/hhd2/ljk/g1bench/results_v5
cd /hhd2/ljk/g1bench
name=$1; shift
out=$BASE/$name
mkdir -p "$out" "$BASE/logs"
logf="$BASE/logs/$(echo "$name" | tr / _).log"
echo "[$(date +%H:%M:%S)] START $name :: $*" >> "$BASE/logs/driver.log"
"$PY" "$BENCH" --label ljk-falcon-v5 --model-path "$MODEL" --repo "$REPO" \
  --video policy --out-dir "$out" "$@" > "$logf" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] DONE($rc) $name" >> "$BASE/logs/driver.log"
exit $rc
