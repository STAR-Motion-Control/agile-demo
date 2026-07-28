#!/bin/bash
# Start the lean navigation process only after the refactored base runtime is
# healthy.  This script never starts without an explicit human acknowledgement.

set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REFACTOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUS_SOCKET="${GROOT_MOTION_BUS_SOCKET:-/tmp/groot_motion_bus.sock}"
BUS_STATUS="${GROOT_MOTION_BUS_STATUS:-/tmp/groot_motion_bus_status.json}"
ADAPTER_COMMAND_SOCKET="${GROOT_ADAPTER_COMMAND_SOCKET:-/tmp/groot_adapter_command.sock}"
MERGER_COMMAND_SOCKET="${GROOT_MERGER_COMMAND_SOCKET:-/tmp/groot_merger_command.sock}"
ADAPTER_HEALTH_FILE="${GROOT_ADAPTER_HEALTH_FILE:-/tmp/groot_adapter_health.json}"
MERGER_HEALTH_FILE="${GROOT_MERGER_HEALTH_FILE:-/tmp/groot_merger_health.json}"
CONFIG_NAME="${NAV_CONFIG_NAME:-config_bk}"
NATIVE_THREADS="${NAV_NATIVE_THREADS:-1}"
ACK_START=0
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --native-threads) NATIVE_THREADS="$2"; shift 2 ;;
        --human-approved-control-start) ACK_START=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done

case "$CONFIG_NAME" in
    config|config_bk) ;;
    *) echo "--config must be config or config_bk"; exit 2 ;;
esac

if [[ ! "$NATIVE_THREADS" =~ ^[1-8]$ ]]; then
    echo "--native-threads must be an integer in 1..8"
    exit 2
fi
export OMP_NUM_THREADS="$NATIVE_THREADS"
export MKL_NUM_THREADS="$NATIVE_THREADS"
export OPENBLAS_NUM_THREADS="$NATIVE_THREADS"
export NUMEXPR_NUM_THREADS="$NATIVE_THREADS"
export VECLIB_MAXIMUM_THREADS="$NATIVE_THREADS"
export OPENCV_FOR_THREADS_NUM="$NATIVE_THREADS"
export OPENCV_NUM_THREADS="$NATIVE_THREADS"
export NAV_NATIVE_THREADS="$NATIVE_THREADS"

if [[ ! -S "$BUS_SOCKET" ]]; then
    echo "motion bus is not running: $BUS_SOCKET"
    exit 1
fi
python "$SCRIPT_DIR/src/runtime_preflight.py" \
    --status-file "$BUS_STATUS" \
    --repo-root "$REFACTOR_ROOT" \
    --input-socket "$BUS_SOCKET" \
    --required-output-socket "$ADAPTER_COMMAND_SOCKET" \
    --required-output-socket "$MERGER_COMMAND_SOCKET" \
    --required-health-file "$ADAPTER_HEALTH_FILE" \
    --required-health-file "$MERGER_HEALTH_FILE" \
    --max-age-s 2.0

if pgrep -af "python.*run_ros.py" >/dev/null 2>&1; then
    echo "run_ros.py is already running"
    exit 1
fi
python -c "import hydra, rclpy" >/dev/null
echo "navigation preflight passed: config=$CONFIG_NAME bus=$BUS_SOCKET native_threads=$NATIVE_THREADS"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    exit 0
fi
if [[ "$ACK_START" != "1" ]]; then
    echo "navigation start is locked; rerun with --human-approved-control-start"
    exit 3
fi

export GROOT_REFACTOR_ROOT="$REFACTOR_ROOT"
export GROOT_MOTION_BUS_SOCKET="$BUS_SOCKET"
export GROOT_NAV_LAUNCH_AUTHORIZED=1
export NAV_CONFIG_NAME="$CONFIG_NAME"
cd "$SCRIPT_DIR/src"
exec python run_ros.py
