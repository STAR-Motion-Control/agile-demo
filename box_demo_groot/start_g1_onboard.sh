#!/bin/bash
# G1 本体(unitree-g1-nx, Jetson)启动器 — dwbc / agile 二选一, 全部跑在机器人上.
# 背景: 5080 有线 DDS 口坏了 → 控制器搬上机; DDS 走机器人内网 enP8p1s0(→MCU 123.161).
#
# 用法(机器人上):
#   bash start_g1_onboard.sh dwbc     # GR00T decoupled-WBC 底座
#   bash start_g1_onboard.sh agile    # AGILE-29dof 底座
#   附加参数原样透传给 adapter/pipeline (如 --fwd-max 1.3 跑速度测试)
#
# 3 个 tmux pane: merger / adapter(pipeline) / 键盘. 测试脚本时 Ctrl-C 掉键盘 pane
# (单写者), 再跑 three_tests/test_*.py.
set -eu

CTRL="${1:?用法: start_g1_onboard.sh dwbc|agile [extra adapter args...]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"      # = ~/zihou/box_demo_2
IFACE="${IFACE:-enP8p1s0}"
SESSION="g1-onboard-$CTRL"
# 注意: 用 ~/zihou/ 下的 GR00T 副本(=5080 验证过的版本, 从 5080 同步);
# $HOME/GR00T-WholeBodyControl 是别人(zhengye)的新版, sdk2py IDL 不兼容且
# 其 .venv_* 是 x86 的 — 别用。
GROOT_REPO="${GROOT_REPO:-$HOME/zihou/GR00T-WholeBodyControl}"
AGILE_REPO="${AGILE_REPO:-$HOME/zihou/WBC-AGILE}"
# 002=robojudo, 001=robojudo_zihou2 — 自动挑存在的那个
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
# 重启后 enP8p1s0 会丢静态 IPv4(cyclonedds 因此起不来) — 自动补回
if ! ip -4 addr show "$IFACE" | grep -q 192.168.123.164; then
    echo "[FIX] $IFACE 缺 IPv4, 补 192.168.123.164/24 (sudo)"
    echo 123 | sudo -S ip addr add 192.168.123.164/24 dev "$IFACE" 2>/dev/null || true
fi
if ! ping -c1 -W1 192.168.123.161 >/dev/null 2>&1; then
    echo "[WARN] 内网 MCU(192.168.123.161) ping 不通 — 检查 enP8p1s0 / 机器人状态"
fi

if [[ "$CTRL" == "dwbc" ]]; then
    ADAPTER_CMD="python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' --interface '$IFACE' --torch-threads 2 $*"
    STAND_H=0.74
elif [[ "$CTRL" == "agile" ]]; then
    ADAPTER_CMD="python '$SCRIPT_DIR/agile_lowcmd_pipeline.py' \
        --agile-repo '$AGILE_REPO' --iface '$IFACE' --device cpu $*"
    STAND_H=0.72
else
    echo "unknown controller: $CTRL"; exit 1
fi

tmux new-session -d -s "$SESSION" -n merge "
    $SETUP; cd '$SCRIPT_DIR'
    echo '=== pane1: merger rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd ==='
    python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE'
    echo '[merge exited]'; exec bash"
if [[ "$CTRL" == "dwbc" ]]; then PANE2_SETUP="$DWBC_SETUP"; else PANE2_SETUP="$SETUP"; fi
tmux split-window -h -t "$SESSION" "
    $PANE2_SETUP; cd '$SCRIPT_DIR'; sleep 1
    echo '=== pane2: $CTRL adapter -> rt/lowcmd_rl ==='
    $ADAPTER_CMD
    echo '[adapter exited]'; exec bash"
tmux split-window -v -t "$SESSION:0.1" "
    $SETUP; cd '$SCRIPT_DIR'; sleep 2
    echo '=== pane3: 键盘 (w/s/a/d/q/e, z/x 高度, space 停, o 急停) ==='
    echo '跑 three_tests 前先 Ctrl-C 掉本 pane (单写者)!'
    python '$SCRIPT_DIR/agile_keyboard_control.py' --key-timeout 0.25
    echo '[keyboard exited]'; exec bash"
tmux select-layout -t "$SESSION" tiled
echo "tmux 已启动: $SESSION  (Ctrl+B 方向键切 pane / Ctrl+B D 退出)"
echo "站立高度: $STAND_H ($CTRL)。测试脚本: cd $SCRIPT_DIR/three_tests && python3 test_speed_ramp.py ..."
tmux attach -t "$SESSION"
