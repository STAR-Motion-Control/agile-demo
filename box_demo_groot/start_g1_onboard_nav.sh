#!/bin/bash
# G1 本体导航+GR00T-WBC 运控启动器。
#
# 适用拓扑:
#   nav_uat、HTTP IPC bridge、GR00T adapter、merger 全部运行在机器人本体。
#   不经过 5080，不启动键盘，不启动 box_demo_main.py。
#
# Panes:
#   1) merge_lowcmd_arm_sdk.py        rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd
#   2) groot_wbc_boxdemo_adapter.py   GR00T-WBC -> rt/lowcmd_rl
#   3) agile_http_ipc_server.py       nav_uat localhost HTTP -> /tmp/robojudo_ext_cmd.json
#   4) nav_safety_keyboard.py         keyboard stop / DAMP through HTTP bridge
#
# 用法:
#   cd ~/zihou/box_demo_1
#   bash start_g1_onboard_nav.sh
#   bash start_g1_onboard_nav.sh --dry-run --no-attach

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="${SESSION:-g1-onboard-nav}"

IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enP8p1s0}}"
DOMAIN="${DOMAIN:-0}"
GROOT_REPO="${GROOT_REPO:-$HOME/zihou/GR00T-WholeBodyControl}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-5001}"
LEGACY_CMD_FILE="${LEGACY_CMD_FILE:-/tmp/robojudo_ext_cmd.json}"
TORCH_THREADS="${TORCH_THREADS:-2}"
FWD_MAX="${FWD_MAX:-0.30}"
LAT_MAX="${LAT_MAX:-0.15}"
YAW_MAX="${YAW_MAX:-0.30}"
HEIGHT_RATE="${HEIGHT_RATE:-0.10}"
DRY_RUN=""
ATTACH=1
AUTO_POLICY_ARGS=""
ADAPTER_EXTRA=()

if [[ -z "${CONDA_ENV:-}" ]]; then
    for env_name in robojudo_zihou2 robojudo; do
        if [[ -d "$HOME/miniconda3/envs/$env_name" ]]; then
            CONDA_ENV="$env_name"
            break
        fi
    done
fi
CONDA_ENV="${CONDA_ENV:?no conda env found (robojudo_zihou2 / robojudo)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface) IFACE="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --groot-repo) GROOT_REPO="$2"; shift 2 ;;
        --http-host) HTTP_HOST="$2"; shift 2 ;;
        --http-port) HTTP_PORT="$2"; shift 2 ;;
        --legacy-cmd-file) LEGACY_CMD_FILE="$2"; shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max) FWD_MAX="$2"; ADAPTER_EXTRA+=("--fwd-max" "$2"); shift 2 ;;
        --lat-max) LAT_MAX="$2"; ADAPTER_EXTRA+=("--lat-max" "$2"); shift 2 ;;
        --yaw-max) YAW_MAX="$2"; ADAPTER_EXTRA+=("--yaw-max" "$2"); shift 2 ;;
        --height-rate) HEIGHT_RATE="$2"; ADAPTER_EXTRA+=("--height-rate" "$2"); shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --no-auto-activate-policy) AUTO_POLICY_ARGS="--no-auto-activate-policy"; shift ;;
        --no-attach) ATTACH=0; shift ;;
        *) ADAPTER_EXTRA+=("$1"); shift ;;
    esac
done

if [[ ! -d "$GROOT_REPO/decoupled_wbc" ]]; then
    echo "GR00T repo not found: $GROOT_REPO"
    exit 1
fi

# 清理每个 pane 继承到的 ROS/LD_LIBRARY_PATH 污染。真机控制链走 sdk2py DDS，
# 001/G001 上没有完整 /opt/ros 时使用 rclpy_stub 让 decoupled_wbc import 通过。
SANITIZE="unset LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY PYTHONPATH"
SETUP="$SANITIZE; source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && export UNITREE_DDS_INTERFACE='$IFACE'"
if [[ -f /opt/ros/humble/setup.bash && -d "$HOME/cyclonedds-0.10-install" ]]; then
    ROS_SRC="source /opt/ros/humble/setup.bash &&"
    RCLPY_STUB=""
else
    ROS_SRC=""
    RCLPY_STUB="$HOME/zihou/rclpy_stub:"
fi
DWBC_SETUP="$SETUP && $ROS_SRC export PYTHONPATH=\"$GROOT_REPO/external_dependencies/unitree_sdk2_python:$RCLPY_STUB\${PYTHONPATH:-}\""

BUSY=$(pgrep -af "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|agile_lowcmd_pipeline.py|run_g1_control_loop|agile_keyboard_control.py|box_demo_main.py" 2>/dev/null | grep -vE '^[0-9]+ +tmux' || true)
if [[ -n "$BUSY" ]]; then
    echo "[WARN] 检测到已有控制/IPC 写者进程在跑。导航启动前请先停掉:"
    echo "$BUSY"
    exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

if ! ip -4 addr show "$IFACE" | grep -q 192.168.123.164; then
    echo "[FIX] $IFACE 缺 IPv4, 尝试补 192.168.123.164/24 (需要 sudo)"
    echo 123 | sudo -S ip addr add 192.168.123.164/24 dev "$IFACE" 2>/dev/null || true
fi
if ! ping -c1 -W1 192.168.123.161 >/dev/null 2>&1; then
    echo "[WARN] 内网 MCU(192.168.123.161) ping 不通。dry-run 可继续，真机运动前必须修复。"
fi

echo "========================================"
echo "  G1 onboard nav + GR00T-WBC lower body"
echo "  iface/domain: $IFACE / $DOMAIN"
echo "  conda env:    $CONDA_ENV"
echo "  GROOT_REPO:   $GROOT_REPO"
echo "  limits:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX height_rate=$HEIGHT_RATE"
echo "  HTTP bridge:  http://$HTTP_HOST:$HTTP_PORT -> $LEGACY_CMD_FILE"
echo "  dry-run:      ${DRY_RUN:-no}"
echo "========================================"

tmux new-session -d -s "$SESSION" -n nav "
    $SETUP; cd '$SCRIPT_DIR'
    echo '=== pane1: merger rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd ==='
    if [[ -n '$DRY_RUN' ]]; then
        echo 'dry-run: merger is not started, so rt/lowcmd is not published'
    else
        python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE'
    fi
    echo '[merge exited]'; exec bash"

tmux split-window -h -t "$SESSION" "
    $DWBC_SETUP; cd '$SCRIPT_DIR'; sleep 1
    echo '=== pane2: GR00T-WBC adapter -> rt/lowcmd_rl ==='
    python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' \
        --interface '$IFACE' \
        --domain '$DOMAIN' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        $AUTO_POLICY_ARGS \
        ${ADAPTER_EXTRA[*]} \
        $DRY_RUN
    echo '[adapter exited]'; exec bash"

tmux split-window -v -t "$SESSION:0.1" "
    $SETUP; cd '$SCRIPT_DIR'; sleep 2
    echo '=== pane3: local HTTP IPC bridge for nav_uat ==='
    python '$SCRIPT_DIR/agile_http_ipc_server.py' \
        --backend legacy-ipc \
        --host '$HTTP_HOST' \
        --port '$HTTP_PORT' \
        --legacy-cmd-file '$LEGACY_CMD_FILE'
    echo '[HTTP IPC bridge exited]'; exec bash"

tmux select-layout -t "$SESSION" tiled

echo "tmux 已启动: $SESSION"
echo "nav_uat 配置应使用: http://127.0.0.1:$HTTP_PORT"
tmux split-window -v -t "$SESSION:0.0" "
    $SETUP; cd '$SCRIPT_DIR'; sleep 3
    echo '=== pane4: safety keyboard (space/z stop, o/d DAMP, q exit) ==='
    python '$SCRIPT_DIR/nav_safety_keyboard.py' \
        --ipc-url 'http://127.0.0.1:$HTTP_PORT'
    echo '[safety keyboard exited]'; exec bash"

tmux select-layout -t "$SESSION" tiled

echo "注意: 本 session 不启动运动键盘和 box_demo_main.py，HTTP bridge 是 /tmp/robojudo_ext_cmd.json 单写者。"
echo "安全键盘: space/z=速度归零并保持平衡, o/d=DAMP 策略急停, q=退出安全键盘。"

if [[ "$ATTACH" == "1" ]]; then
    tmux attach -t "$SESSION"
fi
