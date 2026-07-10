#!/bin/bash
# Round 6 (final): WELD box-squat full sweep + circle r=0.4 with back-drift lead.
set -u
cd "$(dirname "$0")"
OUT=~/three_tests_results
DW=/home/wjzh/GR00T-WholeBodyControl/.venv_wbc/bin/python
AG=/home/wjzh/miniconda3/envs/hdmi/bin/python
export MUJOCO_GL=egl

if [[ "$1" == "dwbc" ]]; then
    $DW run_sim_test.py --controller dwbc --test squat_box \
        --h-low-list 0.55,0.50 --pitch-list 0,0.15,0.3 \
        --hug-sp -1.2 --hug-roll 0.15 --hug-elbow 0.7 \
        --box-x 0.30 --box-z 0.92 --box-weld > "$OUT/log_dwbc_squatbox5.txt" 2>&1
    $DW run_sim_test.py --controller dwbc --test circle --radius-list 0.4 \
        --cmd-gain-list 1.8 --bulge-min 0.85 --lead-strafe-s 1.5 \
        > "$OUT/log_dwbc_circle_r04d.txt" 2>&1
    echo DONE > "$OUT/dwbc6.done"
elif [[ "$1" == "agile" ]]; then
    $AG run_sim_test.py --controller agile29 --test squat_box \
        --h-low-list 0.55,0.50 --pitch-list 0,0.15,0.3 \
        --hug-sp -1.2 --hug-roll 0.15 --hug-elbow 0.7 \
        --box-x 0.30 --box-z 0.92 --box-weld > "$OUT/log_agile_squatbox4.txt" 2>&1
    $AG run_sim_test.py --controller agile29 --test circle --radius-list 0.4 \
        --cmd-gain-list 1.5 --bulge-min 0.70 --lead-strafe-s 1.0 \
        > "$OUT/log_agile_circle_r04d.txt" 2>&1
    echo DONE > "$OUT/agile6.done"
fi
