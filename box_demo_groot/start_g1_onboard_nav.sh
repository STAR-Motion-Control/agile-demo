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
#   4) agile_keyboard_control.py      direct IPC keyboard, same as manipulation bring-up
#
# 用法:
#   cd ~/zihou/box_demo_1
#   bash start_g1_onboard_nav.sh dwbc                    # 0.76m, warm-up 默认关
#   bash start_g1_onboard_nav.sh dwbc --warmup on        # 恢复 0.60s 预热
#   bash start_g1_onboard_nav.sh dwbc --warmup-time 0.35 # 自定义预热
#   bash start_g1_onboard_nav.sh dwbc --print-config     # 只检查，不启动
#   bash start_g1_onboard_nav.sh                         # 兼容旧用法，等同 dwbc
#   bash start_g1_onboard_nav.sh --nav-motion-profile keyboard
#   bash start_g1_onboard_nav.sh --dry-run --no-attach

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="${SESSION:-g1-onboard-nav}"

# 兼容旧的不带 controller 参数用法，同时让用户要求的 `... nav.sh dwbc`
# 成为显式且受校验的入口。该 launcher 只支持 DWBC。
CTRL="dwbc"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CTRL="$1"
    shift
fi
if [[ "$CTRL" != "dwbc" ]]; then
    echo "unsupported controller: $CTRL (start_g1_onboard_nav.sh only supports dwbc)"
    exit 2
fi

IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enP8p1s0}}"
DOMAIN="${DOMAIN:-0}"
GROOT_REPO="${GROOT_REPO:-$HOME/zihou/GR00T-WholeBodyControl}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-5001}"
LEGACY_CMD_FILE="${LEGACY_CMD_FILE:-/tmp/robojudo_ext_cmd.json}"
NAV_PROFILE_FILE="${NAV_PROFILE_FILE:-/tmp/groot_nav_motion_profile.json}"
NAV_MOTION_PROFILE="${NAV_MOTION_PROFILE:-precise}"  # precise | keyboard
TORCH_THREADS="${TORCH_THREADS:-2}"
# 底座限幅(与 adapter 内置默认一致; 键盘/nav 发的命令最终都被这层截断)。
# 可用环境变量(FWD_MAX=0.4 bash ...)或旗标(--fwd-max 0.4)覆盖。
# ⚠️ YAW_MAX 别低于 0.4: 真机 yaw 跟踪率 ~78%, 0.30 时有效转速仅 ~0.23rad/s
# 太临界, 叠加轻微不对称会单侧转不动(2026-07-04 001 左转事件)。
FWD_MAX="${FWD_MAX:-0.50}"          # 前向速度上限 m/s (速度指标测试才要 1.3)
LAT_MAX="${LAT_MAX:-0.30}"          # 与手动 DWBC adapter 默认一致
YAW_MAX="${YAW_MAX:-0.60}"          # 转向角速度上限 rad/s
HEIGHT_RATE="${HEIGHT_RATE:-0.20}"  # 高度变化速率上限 m/s
# 稳走关键高度参数(= adapter 默认, 显式写出防默认漂移; sim round11 标定):
STAND_HEIGHT="${STAND_HEIGHT:-0.76}"  # 与当前手动 DWBC 真机验证配置一致
WALK_FLOOR="${WALK_FLOOR:-0.72}"      # 低位拒走 warmup 地板(拦速度打印 [GATE])
WARMUP_MODE="${DWBC_WARMUP_MODE:-off}"
WARMUP_TIME="${DWBC_WARMUP_TIME:-0.60}"
WARMUP_SPEED="${DWBC_WARMUP_SPEED:-0.15}"
WAIST_RL="${WAIST_RL:-1}"
DRY_RUN=""
ATTACH=1
PRINT_CONFIG=0
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
        --nav-motion-profile|--motion-profile) NAV_MOTION_PROFILE="$2"; shift 2 ;;
        --nav-profile-file) NAV_PROFILE_FILE="$2"; shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max) FWD_MAX="$2"; ADAPTER_EXTRA+=("--fwd-max" "$2"); shift 2 ;;
        --lat-max) LAT_MAX="$2"; ADAPTER_EXTRA+=("--lat-max" "$2"); shift 2 ;;
        --yaw-max) YAW_MAX="$2"; ADAPTER_EXTRA+=("--yaw-max" "$2"); shift 2 ;;
        --height-rate) HEIGHT_RATE="$2"; ADAPTER_EXTRA+=("--height-rate" "$2"); shift 2 ;;
        --stand-height) STAND_HEIGHT="$2"; shift 2 ;;
        --walk-height-floor) WALK_FLOOR="$2"; shift 2 ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "--warmup 需要 on 或 off"; exit 2; }
            WARMUP_MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --warmup=*)
            WARMUP_MODE="${1#*=}"
            WARMUP_MODE="$(printf '%s' "$WARMUP_MODE" | tr '[:upper:]' '[:lower:]')"
            shift
            ;;
        --no-warmup) WARMUP_MODE="off"; shift ;;
        --warmup-time)
            [[ $# -ge 2 ]] || { echo "--warmup-time 缺少秒数"; exit 2; }
            WARMUP_TIME="$2"
            WARMUP_MODE="on"
            shift 2
            ;;
        --warmup-time=*)
            WARMUP_TIME="${1#*=}"
            WARMUP_MODE="on"
            shift
            ;;
        --warmup-speed)
            [[ $# -ge 2 ]] || { echo "--warmup-speed 缺少速度"; exit 2; }
            WARMUP_SPEED="$2"
            shift 2
            ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --no-auto-activate-policy) AUTO_POLICY_ARGS="--no-auto-activate-policy"; shift ;;
        --no-attach) ATTACH=0; shift ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        *) ADAPTER_EXTRA+=("$1"); shift ;;
    esac
done

case "$NAV_MOTION_PROFILE" in
    precise|min-step|min_step|reliable|default)
        NAV_MOTION_PROFILE="precise"
        ;;
    keyboard|keyboard-like|keyboard_like|direct|direct-speed|direct_speed)
        NAV_MOTION_PROFILE="keyboard"
        ;;
    *)
        echo "Unknown --nav-motion-profile '$NAV_MOTION_PROFILE' (use precise or keyboard)"
        exit 2
        ;;
esac

case "$WARMUP_MODE" in
    on|true|yes|1) WARMUP_MODE="on" ;;
    off|false|no|0) WARMUP_MODE="off" ;;
    *) echo "非法 --warmup: $WARMUP_MODE（使用 on 或 off）"; exit 2 ;;
esac

if [[ ! "$STAND_HEIGHT" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || ! awk -v h="$STAND_HEIGHT" 'BEGIN { exit !(h >= 0.40 && h <= 0.80) }'; then
    echo "非法 --stand-height: $STAND_HEIGHT（允许 0.40..0.80 米）"
    exit 2
fi
if [[ ! "$WARMUP_TIME" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || [[ ! "$WARMUP_SPEED" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || ! awk -v t="$WARMUP_TIME" -v v="$WARMUP_SPEED" \
            'BEGIN { exit !(t >= 0.0 && t <= 3.0 && v > 0.05 && v <= 0.50) }'; then
    echo "warm-up 超出范围: time 允许 0..3s，speed 允许 (0.05, 0.50]m/s"
    exit 2
fi
if [[ "$WARMUP_MODE" == "on" ]]; then
    WARMUP_EFFECTIVE="$WARMUP_TIME"
else
    WARMUP_EFFECTIVE="0.0"
fi
case "$WAIST_RL" in
    0|1) ;;
    *) echo "WAIST_RL 只允许 0 或 1"; exit 2 ;;
esac

if [[ "$PRINT_CONFIG" == "1" ]]; then
    echo "controller=$CTRL"
    echo "stand_height=$STAND_HEIGHT"
    echo "walk_height_floor=$WALK_FLOOR"
    echo "nav_motion_profile=$NAV_MOTION_PROFILE"
    echo "nav_warmup=$WARMUP_MODE"
    echo "nav_warmup_time=$WARMUP_EFFECTIVE"
    echo "nav_warmup_speed=$WARMUP_SPEED"
    echo "waist_to_rl_on_motion=$WAIST_RL"
    echo "limits=fwd:$FWD_MAX,lat:$LAT_MAX,yaw:$YAW_MAX,height_rate:$HEIGHT_RATE"
    echo "nav_runtime_config=$NAV_PROFILE_FILE"
    exit 0
fi

if [[ ! -d "$GROOT_REPO/decoupled_wbc" ]]; then
    echo "GR00T repo not found: $GROOT_REPO"
    exit 1
fi

# 清理每个 pane 继承到的 ROS/LD_LIBRARY_PATH 污染。真机控制链走 sdk2py DDS，
# 001/G001 上没有完整 /opt/ros 时使用 rclpy_stub 让 decoupled_wbc import 通过。
SANITIZE="unset LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY PYTHONPATH"
SETUP="$SANITIZE; source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && export UNITREE_DDS_INTERFACE='$IFACE' GROOT_NAV_MOTION_PROFILE='$NAV_MOTION_PROFILE' GROOT_NAV_MOTION_PROFILE_FILE='$NAV_PROFILE_FILE'"
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

# run_ros.py 在另一个终端/conda 环境启动，无法继承本 launcher 的 shell
# export。把两边必须一致的值原子写入 profile，motion_backend.py 启动时读取。
python3 - "$NAV_PROFILE_FILE" "$NAV_MOTION_PROFILE" "$STAND_HEIGHT" \
    "$WALK_FLOOR" "$WARMUP_MODE" "$WARMUP_EFFECTIVE" "$WARMUP_SPEED" \
    "$FWD_MAX" "$LAT_MAX" "$YAW_MAX" "$WAIST_RL" <<'PY'
import json
import os
import sys
import tempfile
import time

(
    path,
    motion_profile,
    stand_height,
    walk_floor,
    warmup_mode,
    warmup_time,
    warmup_speed,
    fwd_max,
    lat_max,
    yaw_max,
    waist_rl,
) = sys.argv[1:]
directory = os.path.dirname(path) or "/tmp"
os.makedirs(directory, exist_ok=True)
payload = {
    "schema_version": 2,
    "source": "start_g1_onboard_nav.sh",
    "motion_profile": motion_profile,
    "stand_height": float(stand_height),
    "walk_min_height": float(walk_floor),
    "warmup_enabled": warmup_mode == "on",
    "warmup_time": float(warmup_time),
    "warmup_speed": float(warmup_speed),
    "fwd_max": float(fwd_max),
    "back_max": min(0.20, float(fwd_max)),
    "lat_max": float(lat_max),
    "yaw_max": float(yaw_max),
    "v_floor": 0.12,
    "w_floor": 0.10,
    "waist_to_rl_on_motion": waist_rl == "1",
    "updated_at": time.time(),
}
fd, tmp = tempfile.mkstemp(prefix=".groot_nav_profile.", suffix=".json", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)
except Exception:
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    raise
PY

echo "========================================"
echo "  G1 onboard nav + GR00T-WBC lower body"
echo "  iface/domain: $IFACE / $DOMAIN"
echo "  conda env:    $CONDA_ENV"
echo "  GROOT_REPO:   $GROOT_REPO"
echo "  limits:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX height_rate=$HEIGHT_RATE"
echo "  height:       stand=$STAND_HEIGHT walk_floor=$WALK_FLOOR (低位拒走 warmup)"
echo "  box warm-up:  $WARMUP_MODE (${WARMUP_EFFECTIVE}s @ ${WARMUP_SPEED}m/s)"
echo "  waist owner:  motion->RL=$WAIST_RL"
echo "  HTTP bridge:  http://$HTTP_HOST:$HTTP_PORT -> $LEGACY_CMD_FILE"
echo "  nav profile:  $NAV_MOTION_PROFILE ($NAV_PROFILE_FILE)"
echo "  dry-run:      ${DRY_RUN:-no}"
echo "========================================"

MERGE_EXTRA=""
if [[ "$WAIST_RL" == "1" ]]; then
    MERGE_EXTRA="--waist-to-rl-on-motion"
fi

tmux new-session -d -s "$SESSION" -n nav "
    $SETUP; cd '$SCRIPT_DIR'
    echo '=== pane1: merger rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd (WAIST_RL=$WAIST_RL) ==='
    if [[ -n '$DRY_RUN' ]]; then
        echo 'dry-run: merger is not started, so rt/lowcmd is not published'
    else
        python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE' $MERGE_EXTRA
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
        --stand-height '$STAND_HEIGHT' \
        --walk-height-floor '$WALK_FLOOR' \
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
        --legacy-cmd-file '$LEGACY_CMD_FILE' \
        --stand-height '$STAND_HEIGHT' \
        --min-height 0.40 \
        --max-height 0.80
    echo '[HTTP IPC bridge exited]'; exec bash"

tmux select-layout -t "$SESSION" tiled

echo "tmux 已启动: $SESSION"
echo "nav_uat 配置应使用: http://127.0.0.1:$HTTP_PORT"
echo "nav_uat motion profile: $NAV_MOTION_PROFILE (由 $NAV_PROFILE_FILE 传给 motion_backend.py)"
tmux split-window -v -t "$SESSION:0.0" "
    $SETUP; cd '$SCRIPT_DIR'; sleep 3
    echo '=== pane4: direct IPC keyboard (same as start_g1_onboard.sh) ==='
    echo 'w/s/a/d/q/e move, z/x height, space stop, o DAMP. Do not use during active ROS nav commands.'
    # 统一键速(7-06): 前进0.40(后退被 adapter 硬截0.2), vy 0.25(round12), wz 0.40
    python '$SCRIPT_DIR/agile_keyboard_control.py' --key-timeout 0.25 \
        --vx 0.40 --vy 0.25 --wz 0.40 \
        --stand-height '$STAND_HEIGHT' --min-height 0.40 --max-height 0.80
    echo '[keyboard exited]'; exec bash"

tmux select-layout -t "$SESSION" tiled

echo "注意: 本 session 不启动 box_demo_main.py。pane4 键盘与 ROS 导航命令都写 /tmp/robojudo_ext_cmd.json，二者必须人工互斥。"
echo "键盘: w/s/a/d/q/e 运动, z/x 高度, space=速度归零并保持平衡, o=DAMP 策略急停, Ctrl+C=退出键盘。"

if [[ "$ATTACH" == "1" ]]; then
    tmux attach -t "$SESSION"
fi
