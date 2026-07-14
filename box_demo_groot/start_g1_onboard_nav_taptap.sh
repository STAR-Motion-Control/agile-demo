#!/bin/bash
# taptap 测试版导航栈启动器 = start_g1_onboard_nav.sh + 停止回正踏步。
#
# 所有高度、速度、warm-up 和 profile 参数均继承普通导航启动器。
# 本 wrapper 只增加基于双足几何的自适应停止回正。
#
# 用法:
#   bash start_g1_onboard_nav_taptap.sh                       # 默认参数
export GROOT_TAPTAP_ADAPTIVE=1
controller=()
if [[ $# -gt 0 && "$1" != --* ]]; then
    controller=("$1")
    shift
fi
exec bash "$(cd "$(dirname "$0")" && pwd)/start_g1_onboard_nav.sh" \
    "${controller[@]}" --taptap --taptap-recovery adaptive "$@"
