#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试2: 下蹲 20 次 (稳 >= 18) — 真机脚本. 两个模式:

  --mode plain  不抱箱直接蹲: 蹲下-缓一下-站起-缓一下 ×20. 不追求快.
  --mode box    抱箱下蹲: 先跑 hold_box_hug.py --box-width <箱宽> 摆好抱姿
                (换不同大小箱子只改 --box-width; AGILE 加 --waist-pitch 0.15),
                人工递箱确认抱稳后再运行本脚本 (它只发高度循环; 抱姿由
                hold_box_hug 进程经 rt/arm_sdk 保持, 本脚本不碰手臂).
                腰前倾由抱姿的 waist_pitch 承担, 本脚本不下发 pitch.

控制: space=暂停(冻结当前高度) r=继续 s=回站立 o=急停DAMP q=退出.

前置: 底座 adapter 在跑, 键盘 pane 已停 (单写者).
"""
from __future__ import annotations

import argparse

from ipc_ctl import stream_profile
from schedules import squat_profile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("plain", "box"), default="plain")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--h-top", type=float, default=None,
                    help="站立高度 (默认: GR00T 0.74 / AGILE 0.72 — 用 --controller 推断)")
    ap.add_argument("--controller", choices=("dwbc", "agile29"), default="dwbc",
                    help="决定默认站立高度")
    ap.add_argument("--h-low", type=float, default=0.50,
                    help="蹲底高度 m (sim 先验证过的值; AGILE 训练下限 0.40)")
    ap.add_argument("--descend-s", type=float, default=2.0, help="下蹲时长(缓)")
    ap.add_argument("--ascend-s", type=float, default=2.0, help="站起时长(缓)")
    ap.add_argument("--dwell-s", type=float, default=1.5, help="底部/顶部缓一下")
    a = ap.parse_args()

    h_top = a.h_top if a.h_top is not None else (0.74 if a.controller == "dwbc" else 0.72)
    if a.mode == "box":
        print("box 模式: 确认 1) arm_sdk 抱姿进程在跑且箱子抱稳; 2) 底座在 RL_FULL/RL_LOWER"
              " 平衡; 然后回车开始高度循环.")
        input("[Enter] 开始抱箱下蹲 ×%d ..." % a.reps)
    total, f = squat_profile(n_reps=a.reps, h_top=h_top, h_low=a.h_low,
                             descend_s=a.descend_s, dwell_low_s=a.dwell_s,
                             ascend_s=a.ascend_s, dwell_top_s=a.dwell_s)
    print(f"{a.mode} squat ×{a.reps}: {h_top:.2f} <-> {a.h_low:.2f} m, "
          f"周期 {f.cycle_s:.1f}s, 共 {total:.0f}s")
    stream_profile(total, f, stand_height=h_top, label=f"squat_{a.mode}")


if __name__ == "__main__":
    main()
