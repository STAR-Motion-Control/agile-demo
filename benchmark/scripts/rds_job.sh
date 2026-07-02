#!/bin/bash
# Single deep-squat (agile model_2999 student) bench job wrapper.
# Usage: rds_job.sh <out_subpath> <bench_agile args...>
set -u
export AGILE_REPO=/sda/lizhe/g1bench/WBC-AGILE
export UNITREE_MUJOCO_SCENE=/sda/lizhe/g1bench/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
export MUJOCO_GL=egl
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
PY=/sda/lizhe/miniforge3/envs/agile/bin/python
CKPT=/sdb/lizhe/g1_deepsquat/student2999_export/policy.pt
CFG=$AGILE_REPO/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml
BASE=/sda/lizhe/g1bench/results_deepsquat
cd /sda/lizhe/g1bench
name=$1; shift
out=$BASE/$name
mkdir -p "$out" "$BASE/logs"
logf="$BASE/logs/$(echo "$name" | tr / _).log"
echo "[$(date +%H:%M:%S)] START $name :: $*" >> "$BASE/logs/driver.log"
"$PY" bench_agile.py --checkpoint "$CKPT" --config "$CFG" --video policy --out-dir "$out" "$@" > "$logf" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] DONE($rc) $name" >> "$BASE/logs/driver.log"
exit $rc
