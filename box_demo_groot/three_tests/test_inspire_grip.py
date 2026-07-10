#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspire 灵巧手 握紧/张开 循环测试 (unitree-001, Modbus TCP 双手).

前置: Headless_driver_double.py 在跑(它是 Modbus<->DDS 桥, 订阅
rt/inspire_hand/ctrl/{l,r} 写角度寄存器 1486, 发布 rt/inspire_hand/state/{l,r})。

必须用 driver 同款解释器(inspire_sdkpy 只装在 ws 的 venv 里):

  cd ~/zihou/box_demo_1/three_tests
  ~/inspire_hand_ws/.venv/bin/python test_inspire_grip.py                 # 双手 3 循环
  ~/inspire_hand_ws/.venv/bin/python test_inspire_grip.py --hand right --cycles 5
  ~/inspire_hand_ws/.venv/bin/python test_inspire_grip.py --speed 300     # 更慢更柔

每循环: 握紧 -> 停 hold_s -> 回读 angle_act 判定 -> 张开 -> 停 -> 判定。
判定: 握紧后四指 angle_act 应 < 300, 张开后应 > 700 (0=弯曲 1000=伸直)。
结束/Ctrl+C 都会把手张开再退出。角度序: 小指,无名,中,食,拇指弯,拇指旋。
姿势常量与机上 open_hand.py 一致(拇指旋保持 1000 跨掌, 开合过渡不打指)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _reexec_without_ros_env():
    """001 的 ~/.bashrc 无条件 source ROS -> LD_LIBRARY_PATH 带 ROS libddsc,
    DDS Write 会撞 dds_write.c:318 断言(段错误)。LD_LIBRARY_PATH 进程启动后
    改无效, 检测到污染就用清洁环境 re-exec 自己 (与 hold_box_hug.py 同款)."""
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/opt/ros/" not in ld and "unitree_ros2" not in ld:
        return
    env = {k: v for k, v in os.environ.items()
           if k not in ("LD_LIBRARY_PATH", "AMENT_PREFIX_PATH",
                        "COLCON_PREFIX_PATH", "ROS_DISTRO", "ROS_VERSION",
                        "ROS_PYTHON_VERSION", "ROS_LOCALHOST_ONLY",
                        "PYTHONPATH")}
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_reexec_without_ros_env()

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)

from inspire_sdkpy import inspire_dds, inspire_hand_defaut

OPEN = [1000, 1000, 1000, 1000, 1000, 1000]   # 四指伸直, 拇指伸直+跨掌
CLOSE = [0, 0, 0, 0, 0, 1000]                 # 握拳 (拇指旋保持 1000)
MODE_ANGLE = 0b0001
MODE_FORCE_SPEED = 0b1100
CLOSE_THRESH = 300      # 握紧后四指 angle_act 上限
OPEN_THRESH = 700       # 张开后四指 angle_act 下限


class Hand:
    def __init__(self, lr: str):
        self.lr = lr
        self.state = None
        self.pub = ChannelPublisher(f"rt/inspire_hand/ctrl/{lr}",
                                    inspire_dds.inspire_hand_ctrl)
        self.pub.Init()
        self.sub = ChannelSubscriber(f"rt/inspire_hand/state/{lr}",
                                     inspire_dds.inspire_hand_state)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        self.state = msg

    def send_angles(self, angles):
        cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        cmd.angle_set = list(angles)
        cmd.mode = MODE_ANGLE
        self.pub.Write(cmd)

    def send_speed_force(self, speed: int, force: int):
        cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        cmd.speed_set = [speed] * 6
        cmd.force_set = [force] * 6
        cmd.mode = MODE_FORCE_SPEED
        self.pub.Write(cmd)

    def fingers_act(self):
        """四指 angle_act (不含拇指); 无状态时 None."""
        return None if self.state is None else list(self.state.angle_act)[:4]

    def report(self, phase: str) -> bool:
        act = self.fingers_act()
        full = None if self.state is None else list(self.state.angle_act)
        if act is None:
            print(f"    {self.lr}: 无 state (driver 没发?)")
            return False
        ok = (max(act) < CLOSE_THRESH) if phase == "close" else (min(act) > OPEN_THRESH)
        print(f"    {self.lr}: angle_act={full}  "
              f"{'握紧OK' if phase == 'close' else '张开OK' if ok else ''}"
              f"{'' if ok else ' !! 未达阈值'}")
        return ok


def run_selfcheck():
    """不碰手: 订阅两手 state 2s + Write 一条到惰性 topic 验证写路径."""
    got = {}
    subs = []
    for lr in ("l", "r"):
        def mk(tag):
            return lambda m: got.__setitem__(tag, list(m.angle_act))
        s = ChannelSubscriber(f"rt/inspire_hand/state/{lr}",
                              inspire_dds.inspire_hand_state)
        s.Init(mk(lr), 10)
        subs.append(s)
    p = ChannelPublisher("rt/inspire_hand/selftest", inspire_dds.inspire_hand_ctrl)
    p.Init()
    time.sleep(2.0)
    cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
    cmd.mode = 0                                  # 无操作模式, 且 topic 无人订阅
    ok = p.Write(cmd)
    print(f"state: {got if got else '无 (driver 在跑吗?)'}")
    print(f"DDS write 路径: {'OK' if ok is not False else 'FAIL'}")
    print("自检通过 — 可以正式跑开合测试" if got else "自检未过 — 先起 driver")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", choices=("both", "left", "right"), default="both")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--hold-s", type=float, default=1.5, help="每个姿势保持时间")
    ap.add_argument("--speed", type=int, default=400, help="0-1000, 越小越柔")
    ap.add_argument("--force", type=int, default=500, help="力控阈值 0-1000")
    ap.add_argument("--iface", default="enP8p1s0", help="DDS 网口 (G1 本体)")
    ap.add_argument("--check", action="store_true",
                    help="自检: 只读 state + 往惰性 topic 试写, 手完全不动")
    a = ap.parse_args()

    ChannelFactoryInitialize(0, a.iface)
    if a.check:
        run_selfcheck()
        return
    lrs = {"both": ("l", "r"), "left": ("l",), "right": ("r",)}[a.hand]
    hands = [Hand(lr) for lr in lrs]
    time.sleep(0.8)                              # 等 DDS 匹配 + 首帧 state

    print(f"手: {lrs}  循环: {a.cycles}  speed={a.speed} force={a.force}")
    for h in hands:
        act = h.fingers_act()
        print(f"  初始 {h.lr}: angle_act={None if h.state is None else list(h.state.angle_act)}")
        if act is None:
            print("  !! 收不到 state — 确认 Headless_driver_double.py 在跑 / 网口对")
    for h in hands:
        h.send_speed_force(a.speed, a.force)
    time.sleep(0.3)

    passed = 0
    try:
        for c in range(1, a.cycles + 1):
            print(f"[循环 {c}/{a.cycles}] 握紧 ...")
            for h in hands:
                h.send_angles(CLOSE)
            time.sleep(a.hold_s)
            ok_close = all(h.report("close") for h in hands)

            print(f"[循环 {c}/{a.cycles}] 张开 ...")
            for h in hands:
                h.send_angles(OPEN)
            time.sleep(a.hold_s)
            ok_open = all(h.report("open") for h in hands)
            passed += int(ok_close and ok_open)
    except KeyboardInterrupt:
        print("\nCtrl+C — 张开退出")
    finally:
        for h in hands:
            h.send_angles(OPEN)
        time.sleep(0.5)
    print(f"完成: {passed}/{a.cycles} 循环开合均达阈值")


if __name__ == "__main__":
    main()
