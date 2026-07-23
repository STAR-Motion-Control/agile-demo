#!/bin/bash
# G1 本体(unitree-g1-nx, Jetson)启动器 — dwbc / agile 二选一, 全部跑在机器人上.
# 背景: 5080 有线 DDS 口坏了 → 控制器搬上机; DDS 走机器人内网 enP8p1s0(→MCU 123.161).
#
# 用法(机器人上):
#   bash start_g1_onboard.sh dwbc     # GR00T decoupled-WBC 底座，默认站高 0.76m
#   bash start_g1_onboard.sh dwbc --stand-height 0.75  # 临时覆盖站高
#   bash start_g1_onboard.sh dwbc --warmup on          # box 微调启用 0.6s 预热
#   bash start_g1_onboard.sh dwbc --warmup off         # box 微调关闭预热（默认）
#   bash start_g1_onboard.sh dwbc --warmup-time 0.4    # 自定义预热时长并启用
#   bash start_g1_onboard.sh dwbc --stand-height 0.75 --print-config  # 只检查，不启动
#   bash start_g1_onboard.sh agile    # AGILE-29dof 底座
#   BOX=1 bash start_g1_onboard.sh dwbc  # 额外起 HTTP IPC 桥, 供 box_demo_main.py 控制(同学手动开)
#   附加参数原样透传给 adapter/pipeline (如 --fwd-max 1.3 跑速度测试)
#
# 3 个 tmux pane: merger / adapter(pipeline) / 键盘 (BOX=1 时 +HTTP 桥 = 4 个).
# 键盘是 edge-triggered(idle 不写), 与 box_demo/nav 共用 /tmp/robojudo_ext_cmd.json
# (last-write-wins) — box_demo 自主运动时别同时按键盘。测试脚本时 Ctrl-C 掉键盘 pane.
set -eu

CTRL="${1:?用法: start_g1_onboard.sh dwbc|agile [extra adapter args...]}"
shift || true

# DWBC 的站高是 launcher 级参数：同一个值必须同时进入 adapter 和键盘 IPC，
# 否则 adapter 起在新高度后会被键盘首帧写回旧默认值。AGILE 保持自己的 0.72m。
DWBC_STAND_HEIGHT="${DWBC_STAND_HEIGHT:-0.76}"
GROOT_BOX_RUNTIME_CONFIG="${GROOT_BOX_RUNTIME_CONFIG:-/tmp/groot_box_runtime.json}"
DWBC_WARMUP_MODE="${DWBC_WARMUP_MODE:-off}"
DWBC_WARMUP_TIME="${DWBC_WARMUP_TIME:-0.60}"
DWBC_WARMUP_SPEED="${DWBC_WARMUP_SPEED:-0.15}"
PRINT_CONFIG=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stand-height)
            [[ "$CTRL" == "dwbc" ]] || { echo "--stand-height 仅适用于 dwbc"; exit 2; }
            [[ $# -ge 2 ]] || { echo "--stand-height 缺少数值（米）"; exit 2; }
            DWBC_STAND_HEIGHT="$2"
            shift 2
            ;;
        --stand-height=*)
            [[ "$CTRL" == "dwbc" ]] || { echo "--stand-height 仅适用于 dwbc"; exit 2; }
            DWBC_STAND_HEIGHT="${1#*=}"
            shift
            ;;
        --warmup)
            [[ "$CTRL" == "dwbc" ]] || { echo "--warmup 仅适用于 dwbc"; exit 2; }
            [[ $# -ge 2 ]] || { echo "--warmup 需要 on 或 off"; exit 2; }
            DWBC_WARMUP_MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --warmup=*)
            [[ "$CTRL" == "dwbc" ]] || { echo "--warmup 仅适用于 dwbc"; exit 2; }
            DWBC_WARMUP_MODE="${1#*=}"
            DWBC_WARMUP_MODE="$(printf '%s' "$DWBC_WARMUP_MODE" | tr '[:upper:]' '[:lower:]')"
            shift
            ;;
        --no-warmup)
            [[ "$CTRL" == "dwbc" ]] || { echo "--no-warmup 仅适用于 dwbc"; exit 2; }
            DWBC_WARMUP_MODE="off"
            shift
            ;;
        --warmup-time)
            [[ "$CTRL" == "dwbc" ]] || { echo "--warmup-time 仅适用于 dwbc"; exit 2; }
            [[ $# -ge 2 ]] || { echo "--warmup-time 缺少秒数"; exit 2; }
            DWBC_WARMUP_TIME="$2"
            DWBC_WARMUP_MODE="on"
            shift 2
            ;;
        --warmup-time=*)
            [[ "$CTRL" == "dwbc" ]] || { echo "--warmup-time 仅适用于 dwbc"; exit 2; }
            DWBC_WARMUP_TIME="${1#*=}"
            DWBC_WARMUP_MODE="on"
            shift
            ;;
        --print-config)
            PRINT_CONFIG=1
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
EXTRA_ARGS_Q=""
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    printf -v EXTRA_ARGS_Q ' %q' "${EXTRA_ARGS[@]}"
fi

if [[ "$CTRL" == "dwbc" ]]; then
    if [[ ! "$DWBC_STAND_HEIGHT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "非法 --stand-height: ${DWBC_STAND_HEIGHT}（允许 0.40..0.80 米）"
        exit 2
    fi
    if ! awk -v h="$DWBC_STAND_HEIGHT" \
            'BEGIN { exit !(h >= 0.40 && h <= 0.80) }'; then
        echo "非法 --stand-height: ${DWBC_STAND_HEIGHT}（允许 0.40..0.80 米）"
        exit 2
    fi
    if [[ $(awk -v h="$DWBC_STAND_HEIGHT" 'BEGIN { print (h < 0.72) ? 1 : 0 }') == 1 ]]; then
        echo "[WARN] stand-height=$DWBC_STAND_HEIGHT < 行走地板 0.72m；速度会被 adapter 拦截。"
    fi
    case "$DWBC_WARMUP_MODE" in
        on|true|yes|1) DWBC_WARMUP_MODE="on" ;;
        off|false|no|0) DWBC_WARMUP_MODE="off" ;;
        *) echo "非法 --warmup: ${DWBC_WARMUP_MODE}（使用 on 或 off）"; exit 2 ;;
    esac
    if [[ ! "$DWBC_WARMUP_TIME" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "非法 warm-up 参数: time=$DWBC_WARMUP_TIME speed=$DWBC_WARMUP_SPEED"
        exit 2
    fi
    if [[ ! "$DWBC_WARMUP_SPEED" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "非法 warm-up 参数: time=$DWBC_WARMUP_TIME speed=$DWBC_WARMUP_SPEED"
        exit 2
    fi
    if ! awk -v t="$DWBC_WARMUP_TIME" -v v="$DWBC_WARMUP_SPEED" \
            'BEGIN { exit !(t >= 0.0 && t <= 3.0 && v > 0.05 && v <= 0.50) }'; then
        echo "warm-up 超出范围: time 允许 0..3s，speed 允许 (0.05, 0.50]m/s"
        exit 2
    fi
    if [[ "$DWBC_WARMUP_MODE" == "on" ]]; then
        DWBC_WARMUP_EFFECTIVE="$DWBC_WARMUP_TIME"
    else
        DWBC_WARMUP_EFFECTIVE="0.0"
    fi
else
    DWBC_WARMUP_EFFECTIVE="0.0"
fi

if [[ "$PRINT_CONFIG" == "1" ]]; then
    echo "controller=$CTRL"
    if [[ "$CTRL" == "dwbc" ]]; then
        echo "stand_height=$DWBC_STAND_HEIGHT"
        echo "height_range=0.30..0.80"
        echo "walk_height_floor=0.72"
        echo "box_warmup=$DWBC_WARMUP_MODE"
        echo "box_warmup_time=$DWBC_WARMUP_EFFECTIVE"
        echo "box_warmup_speed=$DWBC_WARMUP_SPEED"
        echo "box_runtime_config=$GROOT_BOX_RUNTIME_CONFIG"
    else
        echo "stand_height=0.72"
    fi
    echo "extra_adapter_args=${EXTRA_ARGS_Q:-<none>}"
    exit 0
fi
# BOX=1: 额外起 HTTP IPC 桥(127.0.0.1:5001), 让 box_demo_main.py --locomotion remote 能连;
# --locomotion groot(直接写 /tmp/robojudo_ext_cmd.json)无桥也行, 桥空跑无害。
BOX="${BOX:-0}"
# WAIST_RL=1(默认): dwbc 时给 merger 开"运动窗口"腰仲裁(--waist-to-rl-on-motion)——
# 微调走路时腰(12-14)归 policy(修垂臂后转向前漂/起步晃/back-lean失效), 静止拍照/
# 抓取仍由 arm_sdk 锁零。WAIST_RL=0 = 旧行为(A/B 对比用)。详见 WAIST_HANDOVER_PLAN.md
WAIST_RL="${WAIST_RL:-1}"
if [[ "$BOX" == "1" ]]; then
    # box_demo has its own rt/arm_sdk publisher. Keep the keyboard from becoming
    # a second arm writer in that scene.
    ARM_HANG_EXTRA="--disable-arm-hang-key"
else
    ARM_HANG_EXTRA=""
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"      # = ~/zihou/box_demo_2
IFACE="${IFACE:-enP8p1s0}"
SESSION="g1-onboard-$CTRL"
# 注意: 用 ~/zihou/ 下的 GR00T 副本(=5080 验证过的版本, 从 5080 同步);
# $HOME/GR00T-WholeBodyControl 是别人(zhengye)的新版, sdk2py IDL 不兼容且
# 其 .venv_* 是 x86 的 — 别用。
GROOT_REPO="${GROOT_REPO:-$HOME/zihou/GR00T-WholeBodyControl}"
AGILE_REPO="${AGILE_REPO:-$HOME/zihou/WBC-AGILE}"
# 002=robojudo, 001=robojudo_zihou2 等 自动用存在的那个
if [[ -z "${CONDA_ENV:-}" ]]; then
    for e in robojudo robojudo_zihou2; do
        [[ -d "$HOME/miniconda3/envs/$e" ]] && CONDA_ENV="$e" && break
    done
fi
CONDA_ENV="${CONDA_ENV:?no conda env found (robojudo / robojudo_zihou2)}"

# ⚠️ ~/.bashrc 现在无条件 source ROS(fishros 装的) + unitree_ros2, 而 tmux
# server 从被污染的 shell 起 → pane 直接继承 ROS 的 LD_LIBRARY_PATH。ROS 的
# libddsc 会让 pip cyclonedds 撞 dds_write.c:318 断言(adapter) / 段错误(merger)
# — 2026-07-03 001 实测复现+净化验证。所以每个 pane 先净化(只影响本 pane,
# 不动 .bashrc, 不影响同学正常用 ROS):
SANITIZE="unset LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY PYTHONPATH"
SETUP="$SANITIZE; source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV \
  && export UNITREE_DDS_INTERFACE='$IFACE'"
# dwbc 需要 rclpy。⚠️ 只有存在 cyclonedds 预载守护(~/cyclonedds-0.10-install,
# 5080/002 模式)时才 source 真 ROS(在净化+conda 之后重新引入) — 否则用
# rclpy_stub(真机控制链不经 ROS, stub 已在 001 干跑验证干净)。
# + GR00T 捆绑版 sdk2py(带 OdoState_ IDL, 只在 dwbc pane 前插 PYTHONPATH —
# agile/merger 继续用系统老版 sdk2py, 互不影响)
if [[ -f /opt/ros/humble/setup.bash && -d "$HOME/cyclonedds-0.10-install" ]]; then
    ROS_SRC="source /opt/ros/humble/setup.bash &&"
    RCLPY_STUB=""
else
    ROS_SRC=""
    RCLPY_STUB="$HOME/zihou/rclpy_stub:"
fi
DWBC_SETUP="$SETUP && $ROS_SRC \
  export PYTHONPATH=\"$GROOT_REPO/external_dependencies/unitree_sdk2_python:$RCLPY_STUB\${PYTHONPATH:-}\""

# 先清掉自己的旧 session(崩溃残留的 tmux server argv 会带 pane 命令字符串,
# 不清会让下面的 pgrep 误报"控制进程在跑")
tmux kill-session -t "$SESSION" 2>/dev/null || true
# 安全检查: 别和已有控制进程抢 rt/lowcmd (过滤 tmux server 的 argv 误匹配)
BUSY=$(pgrep -af "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|agile_lowcmd_pipeline.py|run_g1_control_loop" 2>/dev/null | grep -vE '^[0-9]+ +tmux' || true)
if [[ -n "$BUSY" ]]; then
    echo "[WARN] 检测到已有底层控制进程在跑 — 先停掉再启动:"
    echo "$BUSY"
    exit 1
fi

# box_demo 通常在另一个终端启动，shell export 无法跨终端传播。把 launcher 选择
# 原子写到 /tmp，groot_mover 在 import 时读取；显式 --warmup-time 仍优先于此默认。
if [[ "$CTRL" == "dwbc" ]]; then
    python3 - "$GROOT_BOX_RUNTIME_CONFIG" "$DWBC_WARMUP_MODE" \
        "$DWBC_WARMUP_EFFECTIVE" "$DWBC_WARMUP_SPEED" <<'PY'
import json
import os
import sys
import tempfile
import time

path, mode, warmup_time, warmup_speed = sys.argv[1:]
directory = os.path.dirname(path) or "/tmp"
os.makedirs(directory, exist_ok=True)
payload = {
    "schema_version": 1,
    "source": "start_g1_onboard.sh",
    "updated_at": time.time(),
    "warmup_enabled": mode == "on",
    "warmup_time": float(warmup_time),
    "warmup_speed": float(warmup_speed),
}
fd, tmp = tempfile.mkstemp(prefix=".groot_box_runtime.", suffix=".json", dir=directory)
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
fi
# 重启后 enP8p1s0 会丢静态 IPv4(cyclonedds 因此起不来) — 自动补回
if ! ip -4 addr show "$IFACE" | grep -q 192.168.123.164; then
    echo "[FIX] $IFACE 缺 IPv4, 补 192.168.123.164/24 (sudo)"
    echo 123 | sudo -S ip addr add 192.168.123.164/24 dev "$IFACE" 2>/dev/null || true
fi
if ! ping -c1 -W1 192.168.123.161 >/dev/null 2>&1; then
    echo "[WARN] 内网 MCU(192.168.123.161) ping 不通 — 检查 enP8p1s0 / 机器人状态"
fi

if [[ "$CTRL" == "dwbc" ]]; then
    # 稳走关键高度参数由 launcher 统一管理，默认 0.76m（4090-lab MuJoCo 验证）。
    # --stand-height H 同时控制 adapter 初值和键盘 r/首帧，避免两边高度不一致。
    #   --walk-height-floor 0.72 低位拒走 warmup(低于地板拦速度打印 [GATE], r 键回站高放行)
    ADAPTER_CMD="python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' --interface '$IFACE' --torch-threads 2 \
        --stand-height '$DWBC_STAND_HEIGHT' --min-height 0.30 --max-height 0.80 \
        --walk-height-floor 0.72 --height-rate 0.20$EXTRA_ARGS_Q"
    STAND_H="$DWBC_STAND_HEIGHT"
elif [[ "$CTRL" == "agile" ]]; then
    ADAPTER_CMD="python '$SCRIPT_DIR/agile_lowcmd_pipeline.py' \
        --agile-repo '$AGILE_REPO' --iface '$IFACE' --device cpu$EXTRA_ARGS_Q"
    STAND_H=0.72
else
    echo "unknown controller: $CTRL"; exit 1
fi

# 腰仲裁只对 dwbc 开(agile 29dof 的腰路径未按此协议验证)
MERGE_EXTRA=""
if [[ "$CTRL" == "dwbc" && "$WAIST_RL" == "1" ]]; then
    MERGE_EXTRA="--waist-to-rl-on-motion"
fi
tmux new-session -d -s "$SESSION" -n merge "
    $SETUP; cd '$SCRIPT_DIR'
    echo '=== pane1: merger rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd (WAIST_RL=$WAIST_RL) ==='
    python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE' $MERGE_EXTRA
    echo '[merge exited]'; exec bash"
if [[ "$CTRL" == "dwbc" ]]; then PANE2_SETUP="$DWBC_SETUP"; else PANE2_SETUP="$SETUP"; fi
tmux split-window -h -t "$SESSION" "
    $PANE2_SETUP; cd '$SCRIPT_DIR'; sleep 1
    echo '=== pane2: $CTRL adapter -> rt/lowcmd_rl ==='
    $ADAPTER_CMD
    echo '[adapter exited]'; exec bash"
tmux split-window -v -t "$SESSION:0.1" "
    $SETUP; cd '$SCRIPT_DIR'; sleep 2
    echo '=== pane3: 键盘 (w/s/a/d/q/e, z/x 高度, h 双臂自然下垂, space 停, o 急停) ==='
    echo '跑 three_tests 前先 Ctrl-C 掉本 pane (单写者)!'
    # vx 0.40=前进(后退被 adapter 硬截 0.2, 键盘无需分开); vy 0.25=round12 中点;
    # wz 0.40=7-06 现场定(0.15 太慢)。后退硬限在 adapter BACK_VX_MAX, 键盘改不动。
    python '$SCRIPT_DIR/agile_keyboard_control.py' --key-timeout 0.25 \
        --vx 0.40 --vy 0.25 --wz 0.40 \
        --iface '$IFACE' --arm-hang-duration 3.0 --arm-release-duration 2.5 \
        $ARM_HANG_EXTRA \
        --stand-height '$STAND_H' --min-height 0.30 --max-height 0.80
    echo '[keyboard exited]'; exec bash"
if [[ "$BOX" == "1" ]]; then
tmux split-window -v -t "$SESSION:0.0" "
    $SETUP; cd '$SCRIPT_DIR'; sleep 2
    echo '=== pane4: HTTP IPC 桥 (box_demo --locomotion remote -> http://127.0.0.1:5001) ==='
    python '$SCRIPT_DIR/agile_http_ipc_server.py' --backend legacy-ipc \
        --host 127.0.0.1 --port 5001 --legacy-cmd-file /tmp/robojudo_ext_cmd.json \
        --stand-height '$STAND_H' --min-height 0.30 --max-height 0.80
    echo '[http bridge exited]'; exec bash"
fi
tmux select-layout -t "$SESSION" tiled
echo "tmux 已启动: $SESSION  (Ctrl+B 方向键切 pane / Ctrl+B D 退出)"
echo "站立高度: $STAND_H ($CTRL)。测试脚本: cd $SCRIPT_DIR/three_tests && python3 test_speed_ramp.py ..."
if [[ "$CTRL" == "dwbc" ]]; then
    echo "box warm-up: $DWBC_WARMUP_MODE (time=$DWBC_WARMUP_EFFECTIVE s, speed=$DWBC_WARMUP_SPEED m/s; config=$GROOT_BOX_RUNTIME_CONFIG)"
fi
if [[ "$BOX" == "1" ]]; then
    echo ""
    echo "[BOX] 控制器+HTTP桥就绪。同学在另一个终端用封装开 box_demo(已含 ROS 净化, 二选一, 都写同一 IPC):"
    echo "  GROOT_STAND_HEIGHT=$STAND_H bash ~/zihou/box_demo_2/run_box_demo.sh --locomotion groot  --warmup-time $DWBC_WARMUP_EFFECTIVE --iface $IFACE --host <SAM3_IP> --port 5300 --no-confirm --no-vlm"
    echo "  GROOT_STAND_HEIGHT=$STAND_H bash ~/zihou/box_demo_2/run_box_demo.sh --locomotion remote --warmup-time $DWBC_WARMUP_EFFECTIVE --ipc-url http://127.0.0.1:5001 --iface $IFACE --host <SAM3_IP> --port 5300 --no-confirm --no-vlm"
    echo "  ⚠️ 别直接 python box_demo_main.py: ~/.bashrc 污染 LD_LIBRARY_PATH(ROS libddsc)→发 rt/arm_sdk 撞 cyclonedds 断言崩溃; run_box_demo.sh 已净化。"
    echo "  注意: box_demo 自主运动时别同时按键盘 pane(last-write-wins 会抢命令)。"
fi
tmux attach -t "$SESSION"
