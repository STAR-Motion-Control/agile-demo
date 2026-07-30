#!/bin/bash
# Refactored onboard runtime.  This is intentionally separate from the legacy
# start_g1_onboard_nav*.sh launchers so both versions remain available for A/B.
#
# This launcher never starts control without the explicit
# --human-approved-control-start flag.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REFACTOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-onboard-runtime}"
CTRL="dwbc"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CTRL="$1"
    shift
fi
if [[ "$CTRL" != "dwbc" ]]; then
    echo "unsupported controller: $CTRL"
    exit 2
fi

IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enP8p1s0}}"
DOMAIN="${DOMAIN:-0}"
GROOT_REPO="${GROOT_REPO:-$REFACTOR_ROOT}"
SDK_ROOT="${UNITREE_SDK_ROOT:-$HOME/zihou/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python}"
RCLPY_STUB_ROOT="${RCLPY_STUB_ROOT:-$HOME/zihou/rclpy_stub}"
BUS_SOCKET="${GROOT_MOTION_BUS_SOCKET:-/tmp/groot_motion_bus.sock}"
CMD_FILE="${LEGACY_CMD_FILE:-/tmp/robojudo_ext_cmd.json}"
BUS_STATUS_FILE="${GROOT_MOTION_BUS_STATUS:-/tmp/groot_motion_bus_status.json}"
BUS_SAFETY_FILE="${GROOT_MOTION_BUS_SAFETY:-/tmp/groot_motion_safety_latch.json}"
ADAPTER_CMD_SOCKET="${GROOT_ADAPTER_COMMAND_SOCKET:-/tmp/groot_adapter_command.sock}"
MERGER_CMD_SOCKET="${GROOT_MERGER_COMMAND_SOCKET:-/tmp/groot_merger_command.sock}"
ADAPTER_HEALTH_FILE="${GROOT_ADAPTER_HEALTH_FILE:-/tmp/groot_adapter_health.json}"
MERGER_HEALTH_FILE="${GROOT_MERGER_HEALTH_FILE:-/tmp/groot_merger_health.json}"
ARM_SOCKET="${GROOT_ARM_CONTROL_SOCKET:-/tmp/groot_arm_control.sock}"
ARM_STATUS_FILE="${GROOT_ARM_CONTROL_STATUS:-/tmp/groot_arm_control_status.json}"
ARM_RUNTIME_SOCKET="${GROOT_ARM_RUNTIME_SOCKET:-/tmp/groot_arm_runtime.sock}"
ARM_RUNTIME_STATUS="${GROOT_ARM_RUNTIME_STATUS:-/tmp/groot_arm_runtime_status.json}"
NAV_PROFILE_FILE="${GROOT_NAV_MOTION_PROFILE_FILE:-/tmp/groot_nav_motion_profile.json}"
STAND_HEIGHT="${STAND_HEIGHT:-0.76}"
WALK_FLOOR="${WALK_FLOOR:-0.72}"
FWD_MAX="${FWD_MAX:-0.50}"
LAT_MAX="${LAT_MAX:-0.30}"
LAT_CRUISE="${LAT_CRUISE:-0.20}"
YAW_MAX="${YAW_MAX:-0.60}"
HEIGHT_RATE="${HEIGHT_RATE:-0.20}"
# Match start_g1_onboard_nav.sh exactly.  The preserved launcher hard-codes
# keyboard vx/wz to 0.40 and config_bk.yaml supplies these nav cruise speeds.
FWD_CRUISE="0.40"
BACK_CRUISE="0.20"
YAW_CRUISE="0.40"
TORCH_THREADS="${TORCH_THREADS:-1}"
NATIVE_THREADS="${GROOT_NATIVE_THREADS:-1}"
ORT_INTRA="${GROOT_ORT_INTRA_OP_THREADS:-1}"
ORT_INTER="${GROOT_ORT_INTER_OP_THREADS:-1}"
ORT_MODE="${GROOT_ORT_EXECUTION_MODE:-sequential}"
ORT_SPIN="${GROOT_ORT_ALLOW_SPINNING:-false}"
OUTPUT_HZ="${MOTION_BUS_OUTPUT_HZ:-20}"
HEALTH_P99_MS="${GROOT_HEALTH_P99_MAX_MS:-40}"
HEALTH_MAX_GAP_MS="${GROOT_HEALTH_MAX_GAP_MS:-60}"
RL_STALE_S="${GROOT_RL_STALE_S:-0.12}"
RUNTIME_ID="${GROOT_RUNTIME_ID:-}"
BROKER_STARTUP_TIMEOUT="${GROOT_STARTUP_BROKER_TIMEOUT_S:-10}"
MERGER_STARTUP_TIMEOUT="${GROOT_STARTUP_MERGER_TIMEOUT_S:-30}"
COMPOSITE_STARTUP_TIMEOUT="${GROOT_STARTUP_COMPOSITE_TIMEOUT_S:-180}"
STARTUP_MAX_AGE="${GROOT_STARTUP_MAX_AGE_S:-2.5}"
STARTUP_STABLE_S="${GROOT_STARTUP_STABLE_S:-1.0}"
ACK_START=0
ATTACH=1
KEYBOARD=1
MANIP_INGRESS=1
MANIP_VISION_LOG=0
MANIP_HOST="${MANIP_INGRESS_HOST:-127.0.0.1}"
MANIP_PORT="${MANIP_INGRESS_PORT:-5055}"
PRINT_CONFIG=0
PREFLIGHT_ONLY=0
ADAPTER_EXTRA=()
MANIP_EXTRA=(--no-vision-log)

if [[ -z "${CONDA_ENV:-}" ]]; then
    for candidate in g1_deploy robojudo_zihou2 robojudo; do
        if [[ -d "$HOME/miniconda3/envs/$candidate" ]]; then
            CONDA_ENV="$candidate"
            break
        fi
    done
fi
CONDA_ENV="${CONDA_ENV:?no supported conda environment found}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface) IFACE="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --groot-repo) GROOT_REPO="$2"; shift 2 ;;
        --sdk-root) SDK_ROOT="$2"; shift 2 ;;
        --stand-height) STAND_HEIGHT="$2"; shift 2 ;;
        --walk-height-floor) WALK_FLOOR="$2"; shift 2 ;;
        --fwd-max) FWD_MAX="$2"; shift 2 ;;
        --lat-max) LAT_MAX="$2"; shift 2 ;;
        --lat-cruise) LAT_CRUISE="$2"; shift 2 ;;
        --yaw-max) YAW_MAX="$2"; shift 2 ;;
        --height-rate) HEIGHT_RATE="$2"; shift 2 ;;
        --health-p99-max-ms) HEALTH_P99_MS="$2"; shift 2 ;;
        --health-max-gap-ms) HEALTH_MAX_GAP_MS="$2"; shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --ort-intra-threads) ORT_INTRA="$2"; shift 2 ;;
        --ort-inter-threads) ORT_INTER="$2"; shift 2 ;;
        --human-approved-control-start) ACK_START=1; shift ;;
        --no-keyboard) KEYBOARD=0; shift ;;
        --no-manip-ingress) MANIP_INGRESS=0; shift ;;
        --manip-vision-log) MANIP_VISION_LOG=1; MANIP_EXTRA=(); shift ;;
        --manip-host) MANIP_HOST="$2"; shift 2 ;;
        --manip-port) MANIP_PORT="$2"; shift 2 ;;
        --no-attach) ATTACH=0; shift ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) ADAPTER_EXTRA+=("$1"); shift ;;
    esac
done

for value in "$NATIVE_THREADS" "$TORCH_THREADS" "$ORT_INTRA" "$ORT_INTER"; do
    if [[ ! "$value" =~ ^[1-8]$ ]]; then
        echo "thread counts must be integers in 1..8, got: $value"
        exit 2
    fi
done
if [[ "$ORT_MODE" != "sequential" && "$ORT_MODE" != "parallel" ]]; then
    echo "GROOT_ORT_EXECUTION_MODE must be sequential or parallel"
    exit 2
fi
case "$ORT_SPIN" in
    0|1|true|false|yes|no|on|off) ;;
    *) echo "GROOT_ORT_ALLOW_SPINNING must be boolean"; exit 2 ;;
esac
python3 - "$BROKER_STARTUP_TIMEOUT" "$MERGER_STARTUP_TIMEOUT" \
    "$COMPOSITE_STARTUP_TIMEOUT" "$STARTUP_MAX_AGE" "$STARTUP_STABLE_S" \
    "$HEALTH_P99_MS" "$HEALTH_MAX_GAP_MS" "$RL_STALE_S" <<'PY'
import math
import sys

names = (
    "broker timeout", "merger timeout", "composite timeout", "max age",
    "stable window", "health p99", "health max gap", "RL stale",
)
values = []
for name, raw in zip(names, sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be numeric, got: {raw}")
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite, got: {raw}")
    values.append(value)
if (
    any(value <= 0.0 for value in values[:4])
    or values[4] < 0.0
    or any(value <= 0.0 for value in values[5:])
):
    raise SystemExit(
        "startup timeouts, health limits, and RL stale must be > 0; "
        "stable window must be >= 0"
    )
PY

print_config() {
    echo "controller=$CTRL"
    # Keep this compatibility block identical to the preserved launcher.
    echo "stand_height=$STAND_HEIGHT"
    echo "walk_height_floor=$WALK_FLOOR"
    echo "nav_motion_profile=precise"
    echo "nav_warmup=off"
    echo "nav_warmup_time=0.0"
    echo "nav_warmup_speed=0.15"
    echo "waist_to_rl_on_motion=1"
    echo "limits=fwd:$FWD_MAX,lat:$LAT_MAX,yaw:$YAW_MAX,height_rate:$HEIGHT_RATE"
    echo "nav_lat_cruise=$LAT_CRUISE"
    echo "nav_runtime_config=$NAV_PROFILE_FILE"
    echo "keyboard_speed=vx:$FWD_CRUISE,vy:$LAT_CRUISE,wz:$YAW_CRUISE"
    echo "nav_cruise=fwd:$FWD_CRUISE,back:$BACK_CRUISE,lat:$LAT_CRUISE,yaw:$YAW_CRUISE"
    echo "direction_limits=fwd:$FWD_MAX,back:0.20,lat:$LAT_MAX,yaw:$YAW_MAX"
    echo "refactor_root=$REFACTOR_ROOT"
    echo "groot_repo=$GROOT_REPO"
    echo "sdk_root=$SDK_ROOT"
    echo "rclpy_stub=$RCLPY_STUB_ROOT"
    echo "iface/domain=$IFACE/$DOMAIN"
    echo "motion_bus=$BUS_SOCKET -> UDS[$ADAPTER_CMD_SOCKET,$MERGER_CMD_SOCKET] @ ${OUTPUT_HZ}Hz"
    echo "legacy_command_file=$CMD_FILE (candidate hot-path writes disabled)"
    echo "adapter_health=$ADAPTER_HEALTH_FILE p99<=${HEALTH_P99_MS}ms,max_gap<=${HEALTH_MAX_GAP_MS}ms"
    echo "merger_health=$MERGER_HEALTH_FILE rl_stale=${RL_STALE_S}s,max_gap<=${HEALTH_MAX_GAP_MS}ms"
    echo "runtime_id=${RUNTIME_ID:-<generated-on-start>}"
    echo "arm_control=$ARM_SOCKET"
    echo "arm_runtime=$ARM_RUNTIME_SOCKET (merger is sole DDS owner)"
    echo "threads=native:$NATIVE_THREADS,torch:$TORCH_THREADS,ort_intra:$ORT_INTRA,ort_inter:$ORT_INTER"
    echo "ort_mode=$ORT_MODE,ort_spinning=$ORT_SPIN"
    echo "startup=broker:${BROKER_STARTUP_TIMEOUT}s,merger:${MERGER_STARTUP_TIMEOUT}s,composite:${COMPOSITE_STARTUP_TIMEOUT}s,max_age:${STARTUP_MAX_AGE}s,stable:${STARTUP_STABLE_S}s"
    echo "keyboard=$KEYBOARD"
    echo "manip_ingress=$MANIP_INGRESS ($MANIP_HOST:$MANIP_PORT)"
    echo "manip_vision_log=$MANIP_VISION_LOG"
}

if [[ "$PRINT_CONFIG" == "1" ]]; then
    print_config
    exit 0
fi

for required in \
    "$GROOT_REPO/decoupled_wbc" \
    "$GROOT_REPO/decoupled_wbc/control/policy/g1_gear_wbc_policy.py" \
    "$SDK_ROOT" \
    "$REFACTOR_ROOT/onboard_runtime/motion_bus.py" \
    "$REFACTOR_ROOT/onboard_runtime/command_stream.py" \
    "$REFACTOR_ROOT/onboard_runtime/loop_health.py" \
    "$REFACTOR_ROOT/onboard_runtime/runtime_startup.py"; do
    if [[ ! -e "$required" ]]; then
        echo "missing runtime dependency: $required"
        exit 1
    fi
done
command -v tmux >/dev/null || { echo "tmux not found"; exit 1; }
command -v ip >/dev/null || { echo "ip not found"; exit 1; }

if ! ip -4 addr show "$IFACE" | grep -q "192.168.123.164"; then
    echo "$IFACE does not have 192.168.123.164; refusing to modify networking"
    exit 1
fi
if ! ping -c1 -W1 192.168.123.161 >/dev/null 2>&1; then
    echo "MCU 192.168.123.161 is unreachable"
    exit 1
fi

BUSY="$(pgrep -af \
    "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|run_g1_control_loop|onboard_runtime.motion_bus|box_demo_main.py|box_agent_tools_server.py|run_ros.py|agile_runtime_keyboard.py|agile_http_ipc_server.py|agile_keyboard_control.py|agile_lowcmd_pipeline.py|groot_wbc_keyboard.py|keyboard_arm_hang.py" \
    2>/dev/null | grep -vE '^[0-9]+ +tmux' || true)"
if [[ -n "$BUSY" ]]; then
    echo "existing control/runtime process detected; refusing to overlap:"
    echo "$BUSY"
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    exit 1
fi

print_config
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "preflight passed; nothing started"
    exit 0
fi
if [[ "$ACK_START" != "1" ]]; then
    echo "control start is locked; a human must rerun with --human-approved-control-start"
    exit 3
fi

if [[ -z "$RUNTIME_ID" ]]; then
    RUNTIME_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
fi
if [[ ! "$RUNTIME_ID" =~ ^[A-Za-z0-9.-]{8,64}$ ]]; then
    echo "GROOT_RUNTIME_ID must match [A-Za-z0-9.-]{8,64}"
    exit 2
fi

BUS_PID_FILE="/tmp/groot_runtime_${RUNTIME_ID}_motion_bus.pid"
MERGER_PID_FILE="/tmp/groot_runtime_${RUNTIME_ID}_merger.pid"
ADAPTER_PID_FILE="/tmp/groot_runtime_${RUNTIME_ID}_adapter.pid"
STARTED_AT="$(python3 -c 'import time; print(time.time())')"

python3 "$REFACTOR_ROOT/onboard_runtime/nav_profile.py" write \
    --output "$NAV_PROFILE_FILE" \
    --motion-bus-socket "$BUS_SOCKET" \
    --stand-height "$STAND_HEIGHT" \
    --walk-min-height "$WALK_FLOOR" \
    --fwd-max "$FWD_MAX" \
    --lat-max "$LAT_MAX" \
    --yaw-max "$YAW_MAX" \
    --fwd-cruise "$FWD_CRUISE" \
    --back-cruise "$BACK_CRUISE" \
    --lat-cruise "$LAT_CRUISE" \
    --yaw-cruise "$YAW_CRUISE"

SANITIZE="unset LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY PYTHONPATH"
SETUP="$SANITIZE; source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate '$CONDA_ENV' && export UNITREE_DDS_INTERFACE='$IFACE' GROOT_RUNTIME_ID='$RUNTIME_ID' GROOT_MOTION_BUS_SOCKET='$BUS_SOCKET' GROOT_ADAPTER_COMMAND_SOCKET='$ADAPTER_CMD_SOCKET' GROOT_MERGER_COMMAND_SOCKET='$MERGER_CMD_SOCKET' GROOT_ARM_RUNTIME_SOCKET='$ARM_RUNTIME_SOCKET' OMP_NUM_THREADS='$NATIVE_THREADS' OMP_DYNAMIC='FALSE' MKL_NUM_THREADS='$NATIVE_THREADS' OPENBLAS_NUM_THREADS='$NATIVE_THREADS' NUMEXPR_NUM_THREADS='$NATIVE_THREADS' VECLIB_MAXIMUM_THREADS='$NATIVE_THREADS' OPENCV_FOR_THREADS_NUM='$NATIVE_THREADS' PYTHONPATH='$REFACTOR_ROOT:$SDK_ROOT:$RCLPY_STUB_ROOT'"
ORT_ENV="export OMP_NUM_THREADS='$TORCH_THREADS' MKL_NUM_THREADS='$TORCH_THREADS' OPENBLAS_NUM_THREADS='$TORCH_THREADS' NUMEXPR_NUM_THREADS='$TORCH_THREADS' GROOT_ORT_INTRA_OP_THREADS='$ORT_INTRA' GROOT_ORT_INTER_OP_THREADS='$ORT_INTER' GROOT_ORT_EXECUTION_MODE='$ORT_MODE' GROOT_ORT_ALLOW_SPINNING='$ORT_SPIN'"

# A failed startup owns exactly one cleanup target: the session created below.
SESSION_CREATED=0
STARTUP_COMPLETE=0
cleanup_failed_startup() {
    status=$?
    if [[ "$status" -ne 0 && "$SESSION_CREATED" == "1" && "$STARTUP_COMPLETE" != "1" ]]; then
        echo "startup failed; stopping only newly-created tmux session: $SESSION" >&2
        tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
    fi
    return "$status"
}
trap cleanup_failed_startup EXIT

# STARTUP_STAGE: broker
tmux new-session -d -s "$SESSION" -n runtime "
    set -eu
    $SETUP
    cd '$REFACTOR_ROOT'
    echo '=== motion bus: leases + health gate ==='
    echo \$\$ > '$BUS_PID_FILE'
    exec python -m onboard_runtime.motion_bus \
        --socket '$BUS_SOCKET' \
        --command-file '' \
        --status-file '$BUS_STATUS_FILE' \
        --safety-file '$BUS_SAFETY_FILE' \
        --output-socket '$ADAPTER_CMD_SOCKET' \
        --output-socket '$MERGER_CMD_SOCKET' \
        --output-hz '$OUTPUT_HZ' \
        --default-height '$STAND_HEIGHT' \
        --health-file '$ADAPTER_HEALTH_FILE' \
        --health-file '$MERGER_HEALTH_FILE' \
        --health-stale-s 1.25 \
        --health-runtime-id '$RUNTIME_ID' \
        --require-health"
SESSION_CREATED=1
BUS_PANE="$(tmux display-message -p -t "$SESSION:runtime" '#{pane_id}')"

python3 "$REFACTOR_ROOT/onboard_runtime/runtime_startup.py" wait-broker \
    --status-file "$BUS_STATUS_FILE" \
    --runtime-id "$RUNTIME_ID" \
    --repo-root "$REFACTOR_ROOT" \
    --input-socket "$BUS_SOCKET" \
    --output-socket "$ADAPTER_CMD_SOCKET" \
    --output-socket "$MERGER_CMD_SOCKET" \
    --health-file "$ADAPTER_HEALTH_FILE" \
    --health-file "$MERGER_HEALTH_FILE" \
    --pid-file "$BUS_PID_FILE" \
    --started-after "$STARTED_AT" \
    --max-age "$STARTUP_MAX_AGE" \
    --timeout "$BROKER_STARTUP_TIMEOUT"

# STARTUP_STAGE: merger
MERGER_PANE="$(tmux split-window -h -P -F '#{pane_id}' -t "$BUS_PANE" "
    set -eu
    $SETUP
    cd '$SCRIPT_DIR'
    echo '=== merger: one DDS participant + integrated arm hang ==='
    echo \$\$ > '$MERGER_PID_FILE'
    exec python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' \
        --iface '$IFACE' \
        --waist-to-rl-on-motion \
        --waist-ipc-file '$CMD_FILE' \
        --motion-command-socket '$MERGER_CMD_SOCKET' \
        --rl-stale-s '$RL_STALE_S' \
        --max-tick-gap-ms '$HEALTH_MAX_GAP_MS' \
        --runtime-health-file '$MERGER_HEALTH_FILE' \
        --arm-control-socket '$ARM_SOCKET' \
        --arm-control-status-file '$ARM_STATUS_FILE' \
        --arm-runtime-socket '$ARM_RUNTIME_SOCKET' \
        --arm-runtime-status-file '$ARM_RUNTIME_STATUS'")"

python3 "$REFACTOR_ROOT/onboard_runtime/runtime_startup.py" wait-component \
    --health-file "$MERGER_HEALTH_FILE" \
    --runtime-id "$RUNTIME_ID" \
    --component lowcmd_merger \
    --pid-file "$MERGER_PID_FILE" \
    --process-fragment merge_lowcmd_arm_sdk.py \
    --required-socket "$MERGER_CMD_SOCKET" \
    --required-socket "$ARM_SOCKET" \
    --required-socket "$ARM_RUNTIME_SOCKET" \
    --started-after "$STARTED_AT" \
    --max-age "$STARTUP_MAX_AGE" \
    --timeout "$MERGER_STARTUP_TIMEOUT"

# STARTUP_STAGE: adapter
ADAPTER_PANE="$(tmux split-window -v -P -F '#{pane_id}' -t "$MERGER_PANE" "
    set -eu
    $SETUP; $ORT_ENV
    cd '$SCRIPT_DIR'
    echo '=== bounded GR00T-WBC adapter ==='
    echo \$\$ > '$ADAPTER_PID_FILE'
    exec python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' \
        --interface '$IFACE' \
        --domain '$DOMAIN' \
        --cmd-file '$CMD_FILE' \
        --runtime-command-socket '$ADAPTER_CMD_SOCKET' \
        --runtime-health-file '$ADAPTER_HEALTH_FILE' \
        --health-p99-max-ms '$HEALTH_P99_MS' \
        --health-max-gap-ms '$HEALTH_MAX_GAP_MS' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        --stand-height '$STAND_HEIGHT' \
        --walk-height-floor '$WALK_FLOOR' \
        --shutdown-action damp \
        ${ADAPTER_EXTRA[*]}")"

# STARTUP_STAGE: composite
python3 "$REFACTOR_ROOT/onboard_runtime/runtime_startup.py" wait-composite \
    --status-file "$BUS_STATUS_FILE" \
    --runtime-id "$RUNTIME_ID" \
    --repo-root "$REFACTOR_ROOT" \
    --input-socket "$BUS_SOCKET" \
    --output-socket "$ADAPTER_CMD_SOCKET" \
    --output-socket "$MERGER_CMD_SOCKET" \
    --health-file "$ADAPTER_HEALTH_FILE" \
    --health-file "$MERGER_HEALTH_FILE" \
    --pid-file "$BUS_PID_FILE" \
    --merger-health-file "$MERGER_HEALTH_FILE" \
    --merger-pid-file "$MERGER_PID_FILE" \
    --merger-process-fragment merge_lowcmd_arm_sdk.py \
    --merger-socket "$MERGER_CMD_SOCKET" \
    --merger-socket "$ARM_SOCKET" \
    --merger-socket "$ARM_RUNTIME_SOCKET" \
    --adapter-health-file "$ADAPTER_HEALTH_FILE" \
    --adapter-pid-file "$ADAPTER_PID_FILE" \
    --adapter-process-fragment groot_wbc_boxdemo_adapter.py \
    --adapter-socket "$ADAPTER_CMD_SOCKET" \
    --started-after "$STARTED_AT" \
    --max-age "$STARTUP_MAX_AGE" \
    --stable-s "$STARTUP_STABLE_S" \
    --timeout "$COMPOSITE_STARTUP_TIMEOUT"

if [[ "$KEYBOARD" == "1" ]]; then
    tmux split-window -v -t "$BUS_PANE" "
        $SETUP
        cd '$SCRIPT_DIR'; sleep 1.5
        echo '=== DDS-free runtime keyboard ==='
        python '$SCRIPT_DIR/agile_runtime_keyboard.py' \
            --motion-bus-socket '$BUS_SOCKET' \
            --arm-control-socket '$ARM_SOCKET' \
            --arm-control-status-file '$ARM_STATUS_FILE' \
            --stand-height '$STAND_HEIGHT' \
            --min-height 0.30 --max-height 0.80 \
            --vx '$FWD_CRUISE' --vy '$LAT_CRUISE' --wz '$YAW_CRUISE'
        echo '[keyboard exited]'; exec bash"
fi

if [[ "$MANIP_INGRESS" == "1" ]]; then
    tmux new-window -t "$SESSION" -n manipulation "
        $SETUP
        export GROOT_MOTION_SOURCE='manipulation.box_demo'
        cd '$REFACTOR_ROOT/box_demo_2'
        echo '=== cold manipulation ingress (no DDS/camera/IK while idle) ==='
        python '$REFACTOR_ROOT/box_demo_2/box_agent_tools_server.py' \
            --host '$MANIP_HOST' \
            --port '$MANIP_PORT' \
            --allow-execute \
            --locomotion groot \
            --iface '$IFACE' \
            ${MANIP_EXTRA[*]}
        echo '[manipulation ingress exited]'; exec bash"
fi

tmux select-layout -t "$SESSION:runtime" tiled
tmux select-window -t "$SESSION:runtime"
STARTUP_COMPLETE=1
trap - EXIT
echo "runtime started in tmux: $SESSION"
echo "HTTP bridge is not part of this runtime; nav_uat must use groot_motion_bus."
if [[ "$ATTACH" == "1" ]]; then
    tmux attach -t "$SESSION"
fi
