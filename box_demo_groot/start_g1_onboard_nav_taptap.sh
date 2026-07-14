#!/bin/bash
# taptap 测试版导航栈启动器 = start_g1_onboard_nav.sh + 停止回正踏步。
#
# 本 wrapper 开启横移/转向限幅，以及基于双足几何的自适应停止回正。
# 显式零速/急停不经过斜率限制，立即写入 IPC。
#
# 用法:
#   bash start_g1_onboard_nav_taptap.sh                       # 默认参数
export GROOT_NAV_TAPTAP_LIMITS=1
export GROOT_TAPTAP_ADAPTIVE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/start_g1_onboard_nav.sh" \
    --taptap --taptap-recovery adaptive "$@"
