#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试1: 最快速度 (目标 1 m/s) — 真机加速脚本.

0 -> 0.5 m/s (软起步) -> 保持 2s -> 缓升到 1.0 m/s -> 保持 --hold2-s -> 停.
防弹射: 软起步 + 缓坡; 防撞墙: space 随时暂停(速度归零), r 继续, s 回站立, o 急停.

前置: 底座 adapter 在跑 (GR00T groot_wbc_boxdemo_adapter 或 agile_lowcmd_pipeline),
键盘 pane 已 Ctrl-C (单写者)! adapter 的 --fwd-max 必须 >= v2 (默认 0.5 挡 1m/s):
  GR00T:  groot_wbc_boxdemo_adapter.py ... --fwd-max 1.3
  AGILE:  agile_lowcmd_pipeline.py     ... --fwd-max 1.3

用法:  python test_speed_ramp.py [--v2 1.0] [--hold2-s 4] [--stand-height 0.72|0.74]
"""
from __future__ import annotations

import argparse

from ipc_ctl import stream_profile
from schedules import speed_profile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=float, default=0.5, help="起步速度 m/s")
    ap.add_argument("--hold1-s", type=float, default=2.0, help="v1 保持时间 (用户规格 2s)")
    ap.add_argument("--v2", type=float, default=1.0, help="目标速度 m/s")
    ap.add_argument("--ramp-s", type=float, default=1.5, help="v1->v2 过渡时间")
    ap.add_argument("--hold2-s", type=float, default=4.0, help="v2 保持时间 (1m/s×4s=4m, 场地要够!)")
    ap.add_argument("--stand-height", type=float, default=0.74,
                    help="GR00T=0.74, AGILE=0.72")
    a = ap.parse_args()

    dist = a.v1 * (a.hold1_s + 0.25) + (a.v1 + a.v2) / 2 * a.ramp_s + a.v2 * a.hold2_s
    print(f"预计前进距离 ~{dist:.1f} m (按命令积分; 实际略少). 确认场地长度足够!")
    total, f = speed_profile(v1=a.v1, hold1_s=a.hold1_s, v2=a.v2, ramp_s=a.ramp_s,
                             hold2_s=a.hold2_s, height=a.stand_height)
    stream_profile(total, f, stand_height=a.stand_height, label="speed")


if __name__ == "__main__":
    main()
