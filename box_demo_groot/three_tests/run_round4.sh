#!/bin/bash
# Round 4: high-forward carry (knee-clearance) box-squat + r=0.4 lead-strafe circle
# + full agile re-runs. Run: bash run_round4.sh {dwbc|agile}
set -u
cd "$(dirname "$0")"
OUT=~/three_tests_results
DW=/home/wjzh/GR00T-WholeBodyControl/.venv_wbc/bin/python
AG=/home/wjzh/miniconda3/envs/hdmi/bin/python
export MUJOCO_GL=egl

if [[ "$1" == "dwbc" ]]; then
    # carry-high hold tune: arms further forward+up so the box rides above knee path
    $DW tune_box_hold.py --controller dwbc \
        --sps="-1.35,-1.2,-1.05" --rolls "0.15,0.22" --elbows "0.7,0.9" \
        --xs "0.30,0.34" --zs "0.92,0.98" > "$OUT/log_boxtune3.txt" 2>&1
    # squat_box with the high-carry region (sp -1.2 el 0.9 as center; sweep pitch)
    $DW run_sim_test.py --controller dwbc --test squat_box \
        --h-low-list 0.55,0.50 --pitch-list 0,0.15,0.3 \
        --hug-sp -1.2 --hug-roll 0.22 --hug-elbow 0.9 \
        --box-x 0.32 --box-z 0.95 > "$OUT/log_dwbc_squatbox3.txt" 2>&1
    # r=0.4 with lead strafe
    $DW run_sim_test.py --controller dwbc --test circle --radius-list 0.4 \
        --cmd-gain-list 1.8 --bulge-min 0.70 --lead-strafe-s 1.5 \
        > "$OUT/log_dwbc_circle_r04b.txt" 2>&1
    echo DONE > "$OUT/dwbc4.done"
elif [[ "$1" == "agile" ]]; then
    $AG run_sim_test.py --controller agile29 --test squat \
        --h-low-list 0.55,0.50,0.45 > "$OUT/log_agile_squat2.txt" 2>&1
    $AG run_sim_test.py --controller agile29 --test squat_box \
        --h-low-list 0.55,0.50 --pitch-list 0,0.15,0.3 \
        --hug-sp -1.2 --hug-roll 0.22 --hug-elbow 0.9 \
        --box-x 0.32 --box-z 0.95 > "$OUT/log_agile_squatbox2.txt" 2>&1
    $AG run_sim_test.py --controller agile29 --test speed --v2-list 1.2 \
        > "$OUT/log_agile_speed3.txt" 2>&1
    $AG run_sim_test.py --controller agile29 --test circle --radius-list 0.4 \
        --cmd-gain-list 1.2,1.5 --bulge-min 0.60 --lead-strafe-s 1.0 \
        > "$OUT/log_agile_circle_r04b.txt" 2>&1
    echo DONE > "$OUT/agile4.done"
fi
