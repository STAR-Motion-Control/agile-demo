#!/bin/bash
# taptap 测试版导航栈启动器 = start_g1_onboard_nav.sh + 停止回正踏步。
#
# 本 wrapper 开启已通过5080仿真验收的横移/转向限幅，以及连续导航命令
# 的速度斜率限制。旧的固定方波回正会放大部分序列的姿态误差，因此关闭。
# 显式零速/急停不经过斜率限制，立即写入 IPC。
#
# 用法:
#   bash start_g1_onboard_nav_taptap.sh                       # 默认参数
export GROOT_NAV_TAPTAP_LIMITS=1
exec bash "$(cd "$(dirname "$0")" && pwd)/start_g1_onboard_nav.sh" \
    --taptap --taptap-recovery off "$@"
