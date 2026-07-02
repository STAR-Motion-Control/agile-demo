#!/bin/bash
# GR00T-WBC lower-body services for nav_uat.
#
# Panes:
#   1) merge_lowcmd_arm_sdk.py        rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd
#   2) groot_wbc_boxdemo_adapter.py   GR00T-WBC -> rt/lowcmd_rl
#   3) agile_http_ipc_server.py       nav_uat HTTP -> /tmp/robojudo_ext_cmd.json
#
# This launcher intentionally does NOT start agile_keyboard_control.py or
# box_demo_main.py. During navigation, the HTTP bridge should be the only writer
# of /tmp/robojudo_ext_cmd.json.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-groot-wbc-nav}"

IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enp130s0}}"
GROOT_REPO="${GROOT_REPO:-$HOME/GR00T-WholeBodyControl}"
WBC_VENV="${WBC_VENV:-$GROOT_REPO/.venv_wbc}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
TORCH_THREADS="${TORCH_THREADS:-1}"
FWD_MAX="${FWD_MAX:-0.30}"
LAT_MAX="${LAT_MAX:-0.15}"
YAW_MAX="${YAW_MAX:-0.30}"
HEIGHT_RATE="${HEIGHT_RATE:-0.10}"
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-5001}"
LEGACY_CMD_FILE="${LEGACY_CMD_FILE:-/tmp/robojudo_ext_cmd.json}"
DRY_RUN=""
AUTO_POLICY_ARGS=""
ATTACH=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface)        IFACE="$2";        shift 2 ;;
        --groot-repo)   GROOT_REPO="$2";   shift 2 ;;
        --wbc-venv)     WBC_VENV="$2";     shift 2 ;;
        --ros-distro)   ROS_DISTRO="$2";   shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max)      FWD_MAX="$2";      shift 2 ;;
        --lat-max)      LAT_MAX="$2";      shift 2 ;;
        --yaw-max)      YAW_MAX="$2";      shift 2 ;;
        --height-rate)  HEIGHT_RATE="$2";  shift 2 ;;
        --http-host)    HTTP_HOST="$2";    shift 2 ;;
        --http-port)    HTTP_PORT="$2";    shift 2 ;;
        --legacy-cmd-file) LEGACY_CMD_FILE="$2"; shift 2 ;;
        --dry-run)      DRY_RUN="--dry-run"; shift ;;
        --no-auto-activate-policy) AUTO_POLICY_ARGS="--no-auto-activate-policy"; shift ;;
        --no-attach)    ATTACH=0;          shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$GROOT_REPO/decoupled_wbc" ]]; then
    echo "GR00T repo not found: $GROOT_REPO"
    exit 1
fi
if [[ ! -f "$WBC_VENV/bin/activate" ]]; then
    echo "WBC venv not found: $WBC_VENV"
    exit 1
fi

echo "========================================"
echo "  G1 GR00T-WBC nav lower-body services"
echo "  iface:        $IFACE"
echo "  GROOT_REPO:   $GROOT_REPO"
echo "  WBC_VENV:     $WBC_VENV"
echo "  ROS_DISTRO:   $ROS_DISTRO"
echo "  torch threads:$TORCH_THREADS"
echo "  limits:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX height_rate=$HEIGHT_RATE"
echo "  HTTP bridge:  http://$HTTP_HOST:$HTTP_PORT -> $LEGACY_CMD_FILE"
echo "  dry-run:      ${DRY_RUN:-no}"
echo "========================================"

if pgrep -f "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|run_g1_control_loop.py|agile_lowcmd_pipeline.py|agile_keyboard_control.py|box_demo_main.py" >/dev/null 2>&1; then
    echo "[WARN] Existing control or IPC writer process detected."
    echo "       Stop old control/keyboard/box_demo processes before live hardware navigation."
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"
export OPENBLAS_NUM_THREADS="$TORCH_THREADS"
export NUMEXPR_NUM_THREADS="$TORCH_THREADS"
export VECLIB_MAXIMUM_THREADS="$TORCH_THREADS"

WBC_SETUP="
    source /opt/ros/$ROS_DISTRO/setup.bash
    source '$WBC_VENV/bin/activate'
    export GROOT_WBC_REPO='$GROOT_REPO'
    export UNITREE_DDS_INTERFACE='$IFACE'
    export TORCH_THREADS='$TORCH_THREADS'
"

if [[ -n "$DRY_RUN" ]]; then
    tmux new-session -d -s "$SESSION" -n dryrun "
        echo '=== pane1: dry-run ==='
        echo 'merger is not started, so this run does not publish rt/lowcmd'
        exec bash"
else
    tmux new-session -d -s "$SESSION" -n merge "
        $WBC_SETUP
        cd '$SIM2REAL_DIR'
        echo '=== pane1: merge rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd ==='
        echo \"python: \$(which python)\"
        python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE'
        echo '[merge exited]'
        exec bash"
fi

tmux split-window -h -t "$SESSION" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 1
    echo '=== pane2: GR00T-WBC adapter -> rt/lowcmd_rl ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' \
        --interface '$IFACE' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        $AUTO_POLICY_ARGS \
        $DRY_RUN
    echo '[GR00T-WBC adapter exited]'
    exec bash"

tmux split-window -v -t "$SESSION:0.1" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 2
    echo '=== pane3: HTTP IPC bridge for nav_uat ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/agile_http_ipc_server.py' \
        --backend legacy-ipc \
        --host '$HTTP_HOST' \
        --port '$HTTP_PORT' \
        --legacy-cmd-file '$LEGACY_CMD_FILE'
    echo '[HTTP IPC bridge exited]'
    exec bash"

tmux select-layout -t "$SESSION" tiled

echo ""
echo "tmux session started: $SESSION"
echo "  Ctrl+B arrows: switch panes"
echo "  Ctrl+B D: detach"
echo "  tmux attach -t $SESSION: reattach"
echo "  nav_uat should use: http://192.168.123.222:$HTTP_PORT"
echo ""

if [[ "$ATTACH" == "1" ]]; then
    sleep 1
    tmux attach -t "$SESSION"
fi
