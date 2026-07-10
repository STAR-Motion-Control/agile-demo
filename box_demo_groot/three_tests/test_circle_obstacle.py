#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试3: 绕障 — 固定朝向半圆绕行 (非 benchmark 转圈, 朝向不变).

障碍在正前方 --radius 米处 (0.4 / 0.6 / 0.8 / 1.0). 机器人保持初始朝向,
用 vx/vy 合成半径 radius 的半圆, 绕到障碍正后方 (180°), 终点在障碍前方
2×radius 处, 朝向仍是出发朝向 — 完成后脚本停, 你用 controller 键盘继续直行.

开环 (无里程计): 真机会有航向漂移, 绕行完可用键盘 q/e 修朝向再直行.
控制: space=暂停 r=继续 s=回站立 o=急停 q=退出.

前置: adapter 在跑 (lat-max 要 >= --speed: GR00T 默认 0.30 OK), 键盘 pane 已停.
"""
from __future__ import annotations

import argparse

from ipc_ctl import stream_profile
from schedules import circle_profile, heading_circle_profile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, required=True,
                    help="距障碍距离=绕行半径 m (0.4/0.6/0.8/1.0)")
    ap.add_argument("--mode", choices=("fixed", "heading"), default="fixed",
                    help="fixed=固定朝向(vx/vy 合成, 不转身); "
                         "heading=朝向跟随(转90°→vx+wz 弧线→超越后转回)")
    ap.add_argument("--speed", type=float, default=0.20,
                    help="沿弧线速度 m/s (fixed: 峰值横移=此值, 别超 lat-max)")
    ap.add_argument("--side", choices=("left", "right"), default="left",
                    help="从哪侧绕")
    ap.add_argument("--cmd-gain", type=float, default=1.0,
                    help="fixed 模式速度增益 (dwbc 用 1.4, agile 1.0)")
    ap.add_argument("--yaw-gain", type=float, default=1.0,
                    help="heading 模式 wz 增益 (sim 标定值)")
    ap.add_argument("--fwd-gain", type=float, default=1.0,
                    help="heading 模式 vx 增益 (dwbc ~1.4)")
    ap.add_argument("--turn-rate", type=float, default=0.30,
                    help="heading 模式原地转90°角速度 rad/s")
    ap.add_argument("--turn-comp", type=float, default=1.25,
                    help="原地转90°时长补偿(dwbc 1.25; agile 视 sim 标定)")
    ap.add_argument("--lead-out", type=float, default=0.25,
                    help="heading 模式: 转90°后先直行外扩(米)再入弧")
    ap.add_argument("--stand-height", type=float, default=0.74,
                    help="GR00T=0.74, AGILE=0.72")
    a = ap.parse_args()

    if a.mode == "heading":
        total, f = heading_circle_profile(radius=a.radius, speed=a.speed,
                                          side=a.side, turn_rate=a.turn_rate,
                                          yaw_gain=a.yaw_gain, fwd_gain=a.fwd_gain,
                                          lead_out_m=a.lead_out, turn_comp=a.turn_comp,
                                          height=a.stand_height)
        print(f"绕障(朝向跟随): r={a.radius}m {a.side}侧 — 转90°({f.turn_T:.1f}s) → "
              f"弧线 {f.arc_T:.1f}s(面朝行进方向) → 超越后转回原朝向. "
              f"终点应在出发点前方 {2*a.radius:.1f}m.")
    else:
        total, f = circle_profile(radius=a.radius, speed=a.speed, side=a.side,
                                  height=a.stand_height, cmd_gain=a.cmd_gain)
        print(f"绕障(固定朝向): r={a.radius}m {a.side}侧, v={a.speed}m/s, "
              f"弧线 {f.arc_T:.1f}s, 终点前方 {2*a.radius:.1f}m, 朝向不变.")
    res = stream_profile(total, f, stand_height=a.stand_height, label=f"circle_{a.mode}")
    if res == "done":
        print("绕行完成 — 已认定绕开障碍. 现在用 controller 键盘 (w) 继续直行;"
              " 朝向若有漂移先用 q/e 修正.")


if __name__ == "__main__":
    main()
