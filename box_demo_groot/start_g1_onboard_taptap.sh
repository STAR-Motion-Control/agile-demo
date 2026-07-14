#!/bin/bash
# taptap 测试版启动器 = start_g1_onboard.sh dwbc + 停止回正踏步。
#
# 参考宇树官方运控: 每段运动结束后检查双足间距和前后错位，仅在站姿异常时
# 执行对称踏步回正，再执行下一条指令。
# 仅 dwbc(taptap 是 dwbc adapter 的旗标)。
#
# 用法:
#   bash start_g1_onboard_taptap.sh                          # 默认参数
#   bash start_g1_onboard_taptap.sh --taptap-settle-s 1.5    # 参数原样透传 adapter
# 可调: --taptap-settle-s 1.2  --taptap-cmd 0.08  --taptap-period-s 0.4
#       --taptap-debounce-s 0.35  --taptap-min-motion-s 0.4
exec bash "$(cd "$(dirname "$0")" && pwd)/start_g1_onboard.sh" dwbc \
    --taptap --taptap-recovery adaptive "$@"
