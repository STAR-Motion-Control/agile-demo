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
#   bash start_g1_onboard_nav.sh
#   bash start_g1_onboard_nav.sh --nav-motion-profile keyboard
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
NAV_PROFILE_FILE="${NAV_PROFILE_FILE:-/tmp/groot_nav_motion_profile.json}"
NAV_MOTION_PROFILE="${NAV_MOTION_PROFILE:-precise}"  # precise | keyboard
TORCH_THREADS="${TORCH_THREADS:-2}"
# 底座限幅(与 adapter 内置默认一致; 键盘/nav 发的命令最终都被这层截断)。
# 可用环境变量(FWD_MAX=0.4 bash ...)或旗标(--fwd-max 0.4)覆盖。
# ⚠️ YAW_MAX 别低于 0.4: 真机 yaw 跟踪率 ~78%, 0.30 时有效转速仅 ~0.23rad/s
# 太临界, 叠加轻微不对称会单侧转不动(2026-07-04 001 左转事件)。
FWD_MAX="${FWD_MAX:-0.50}"          # 前向速度上限 m/s (速度指标测试才要 1.3)
LAT_MAX="${LAT_MAX:-0.40}"          # 横向速度上限 m/s
YAW_MAX="${YAW_MAX:-0.60}"          # 转向角速度上限 rad/s
HEIGHT_RATE="${HEIGHT_RATE:-0.20}"  # 高度变化速率上限 m/s
# 稳走关键高度参数(= adapter 默认, 显式写出防默认漂移; sim round11 标定):
STAND_HEIGHT="${STAND_HEIGHT:-0.74}"  # 行走一律站高(0.74 最优, 0.78 不更稳)
WALK_FLOOR="${WALK_FLOOR:-0.72}"      # 低位拒走 warmup 地板(拦速度打印 [GATE])
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
        --nav-motion-profile|--motion-profile) NAV_MOTION_PROFILE="$2"; shift 2 ;;
        --nav-profile-file) NAV_PROFILE_FILE="$2"; shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max) FWD_MAX="$2"; ADAPTER_EXTRA+=("--fwd-max" "$2"); shift 2 ;;
        --lat-max) LAT_MAX="$2"; ADAPTER_EXTRA+=("--lat-max" "$2"); shift 2 ;;
        --yaw-max) YAW_MAX="$2"; ADAPTER_EXTRA+=("--yaw-max" "$2"); shift 2 ;;
        --height-rate) HEIGHT_RATE="$2"; ADAPTER_EXTRA+=("--height-rate" "$2"); shift 2 ;;
        --stand-height) STAND_HEIGHT="$2"; shift 2 ;;
        --walk-height-floor) WALK_FLOOR="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --no-auto-activate-policy) AUTO_POLICY_ARGS="--no-auto-activate-policy"; shift ;;
        --no-attach) ATTACH=0; shift ;;
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

python3 - "$NAV_MOTION_PROFILE" "$NAV_PROFILE_FILE" <<'PY'
import json
import os
import sys
import time

profile, path = sys.argv[1], sys.argv[2]
payload = {
    "motion_profile": profile,
    "taptap_optimized": os.environ.get("GROOT_NAV_TAPTAP_LIMITS", "0").lower()
    in ("1", "true", "yes", "on"),
    "updated_at": time.time(),
    "note": "Read by nav_uat Node/motion_backend.py. precise keeps min_duration/min_distance; keyboard uses keyboard cruise speeds. Warm-up remains enabled in both modes.",
}
tmp = f"{path}.{os.getpid()}.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
PY

echo "========================================"
echo "  G1 onboard nav + GR00T-WBC lower body"
echo "  iface/domain: $IFACE / $DOMAIN"
echo "  conda env:    $CONDA_ENV"
echo "  GROOT_REPO:   $GROOT_REPO"
echo "  limits:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX height_rate=$HEIGHT_RATE"
echo "  height:       stand=$STAND_HEIGHT walk_floor=$WALK_FLOOR (低位拒走 warmup)"
echo "  HTTP bridge:  http://$HTTP_HOST:$HTTP_PORT -> $LEGACY_CMD_FILE"
echo "  nav profile:  $NAV_MOTION_PROFILE ($NAV_PROFILE_FILE)"
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
        --legacy-cmd-file '$LEGACY_CMD_FILE'
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
    python '$SCRIPT_DIR/agile_keyboard_control.py' --key-timeout 0.25 --vx 0.40 --vy 0.25 --wz 0.40
    echo '[keyboard exited]'; exec bash"

tmux select-layout -t "$SESSION" tiled

echo "注意: 本 session 不启动 box_demo_main.py。pane4 键盘与 ROS 导航命令都写 /tmp/robojudo_ext_cmd.json，二者必须人工互斥。"
echo "键盘: w/s/a/d/q/e 运动, z/x 高度, space=速度归零并保持平衡, o=DAMP 策略急停, Ctrl+C=退出键盘。"

if [[ "$ATTACH" == "1" ]]; then
    tmux attach -t "$SESSION"
fi
