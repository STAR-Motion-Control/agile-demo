#!/usr/bin/env bash
# Launch the AGILE 23-DoF locomotion stack on the real G1 in a tmux session.
#
#   pane 0  merger    : merge_lowcmd_arm_sdk.py   (SOLE rt/lowcmd publisher; keep it alive)
#   pane 1  pipeline  : agile23_lowcmd_pipeline.py (rt/lowstate -> AGILE student -> rt/lowcmd_rl)
#   pane 2  keyboard  : agile23_keyboard_control.py (single writer of /tmp/robojudo_ext_cmd.json)
#   pane 3  box_demo  : box_demo_main.py  (only with --with-box; autonomous box grasp)
#
# Bring-up order (see AGILE23_BOXDEMO_DEPLOY_DESIGN.md §8): keep RoboJuDo start.sh as a hot
# fallback; first run with --dry-run (no publish, no merger) to validate obs/rate; then the
# obs byte-compare GO/NO-GO; then suspended -> ground with conservative --pd-scale.
#
# Usage:
#   ./start_agile23_box.sh                 # merger + pipeline + keyboard
#   ./start_agile23_box.sh --dry-run       # pipeline computes only, no merger, no publish
#   ./start_agile23_box.sh --with-box      # also start box_demo_main (autonomous)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-agile23}"
CONDA_ENV="${CONDA_ENV:-robojudo}"
IFACE="${IFACE:-enP8p1s0}"

# --- artifacts (override via env) ---
AGILE_REPO="${AGILE_REPO:-$HOME/WBC-AGILE}"
CKPT="${CKPT:-$HOME/agile23/policy/student_23dof_checkpoint.pt}"   # the distilled 23-DoF student
CONFIG="${CONFIG:-$HOME/agile23/policy/student_23dof.yaml}"        # re-exported 23-DoF IODescriptor
MERGER="${MERGER:-$HOME/zihou/box_demo_2/merge_lowcmd_arm_sdk.py}"
BOX_MAIN="${BOX_MAIN:-$HOME/zihou/box_demo_2/box_demo_main.py}"

# --- locomotion caps / height domain ---
FWD_MAX="${FWD_MAX:-0.50}"; LAT_MAX="${LAT_MAX:-0.30}"; YAW_MAX="${YAW_MAX:-0.60}"
MIN_H="${MIN_H:-0.20}"; MAX_H="${MAX_H:-0.74}"; STAND_H="${STAND_H:-0.72}"
PD_SCALE="${PD_SCALE:-1.0}"

DRY_RUN=0; WITH_BOX=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --with-box) WITH_BOX=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

ACT="source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate ${CONDA_ENV}"
PIPE_ARGS="--agile-repo ${AGILE_REPO} --checkpoint ${CKPT} --config ${CONFIG} \
  --interface ${IFACE} --fwd-max ${FWD_MAX} --lat-max ${LAT_MAX} --yaw-max ${YAW_MAX} \
  --min-height ${MIN_H} --max-height ${MAX_H} --stand-height ${STAND_H} --pd-scale ${PD_SCALE}"
[ "$DRY_RUN" = 1 ] && PIPE_ARGS="$PIPE_ARGS --dry-run"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n stack

# pane 0: merger (skipped in dry-run so nothing reaches the motors)
if [ "$DRY_RUN" = 1 ]; then
  tmux send-keys -t "$SESSION":0.0 "echo '[dry-run] merger NOT started (no rt/lowcmd to motors)'" C-m
else
  tmux send-keys -t "$SESSION":0.0 "${ACT}; python ${MERGER} --interface ${IFACE}" C-m
fi

# pane 1: pipeline (give the merger a moment first)
tmux split-window -t "$SESSION":0 -v
tmux send-keys -t "$SESSION":0.1 "${ACT}; sleep 2; python ${HERE}/agile23_lowcmd_pipeline.py ${PIPE_ARGS}" C-m

# pane 2: keyboard teleop
tmux split-window -t "$SESSION":0 -h
tmux send-keys -t "$SESSION":0.2 "${ACT}; python ${HERE}/agile23_keyboard_control.py \
  --fwd-max ${FWD_MAX} --lat-max ${LAT_MAX} --yaw-max ${YAW_MAX} \
  --min-height ${MIN_H} --max-height ${MAX_H} --stand-height ${STAND_H}" C-m

# pane 3: optional autonomous box demo (uses the SAME IPC file -> do not run keyboard then)
if [ "$WITH_BOX" = 1 ]; then
  tmux split-window -t "$SESSION":0 -v
  tmux send-keys -t "$SESSION":0.3 "${ACT}; echo 'box_demo: stop the keyboard pane first (single IPC writer)'; \
    echo 'python ${BOX_MAIN} --locomotion remote ...'" C-m
fi

tmux select-layout -t "$SESSION":0 tiled
echo "tmux session '${SESSION}' started (dry_run=${DRY_RUN}, with_box=${WITH_BOX})."
echo "attach: tmux attach -t ${SESSION}    | focus keyboard pane to drive."
