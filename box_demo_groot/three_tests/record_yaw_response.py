#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""左右转跟踪率对比记录 (只读: IPC 命令 wz vs IMU gyro_z, 不发任何命令).

用法: 机器人站立、底座在跑; 起本脚本后, 键盘 pane 里【按住 q 约 4s (左转) →
松开停 2s → 按住 e 约 4s (右转)】, 脚本结束打印两个方向的 实测/命令 比值.

  conda activate robojudo_zihou2
  python record_yaw_response.py --dur 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _reexec_without_ros_env():
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

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

CMD_FILE = "/tmp/robojudo_ext_cmd.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dur", type=float, default=20.0, help="记录时长 s")
    ap.add_argument("--iface", default="enP8p1s0")
    a = ap.parse_args()

    ChannelFactoryInitialize(0, a.iface)
    st = {}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: st.__setitem__("m", m), 10)
    t0 = time.time()
    while "m" not in st:
        if time.time() - t0 > 6:
            raise SystemExit("no lowstate")
        time.sleep(0.05)

    # 手臂姿态对称性快照 (左 15-21 vs 右 22-28, 镜像关节反号比较)
    m = st["m"]
    L = [m.motor_state[15 + i].q for i in range(7)]
    R = [m.motor_state[22 + i].q for i in range(7)]
    mirror = [1, -1, -1, 1, -1, 1, -1]           # roll/yaw 类反号
    asym = [abs(L[i] - mirror[i] * R[i]) for i in range(7)]
    print("臂姿 L:", [round(v, 2) for v in L])
    print("臂姿 R:", [round(v, 2) for v in R])
    print("镜像不对称度:", [round(v, 2) for v in asym],
          (" !! 臂姿明显不对称(影响转动惯量/重心)" if max(asym) > 0.3 else " OK"))

    print(f"\n开始记录 {a.dur:.0f}s — 现在去键盘: 按住q约4s → 停2s → 按住e约4s")
    samples = []                                  # (wz_cmd, gyro_z)
    t0 = time.time()
    while time.time() - t0 < a.dur:
        try:
            with open(CMD_FILE) as f:
                wz_cmd = float(json.load(f)["velocity"]["yaw"])
        except Exception:
            wz_cmd = 0.0
        gz = float(st["m"].imu_state.gyroscope[2])
        samples.append((wz_cmd, gz))
        time.sleep(0.05)

    def seg(sign):
        xs = [(c, g) for c, g in samples if sign * c > 0.05]
        if not xs:
            return None
        mc = sum(c for c, _ in xs) / len(xs)
        mg = sum(g for _, g in xs) / len(xs)
        return len(xs), mc, mg, (mg / mc if abs(mc) > 1e-6 else 0.0)

    print(f"\n{'方向':6s} {'样本':>5s} {'命令wz':>8s} {'实测gyro_z':>10s} {'跟踪率':>7s}")
    for name, sign in (("左(+)", +1), ("右(-)", -1)):
        r = seg(sign)
        if r is None:
            print(f"{name:6s}  (没采到该方向命令)")
        else:
            n, mc, mg, ratio = r
            print(f"{name:6s} {n:5d} {mc:+8.3f} {mg:+10.3f} {ratio:7.0%}")
    print("\n解读: 两方向跟踪率接近(±15%)=对称, 命令链没问题;"
          " 左明显低=机体/地面/策略侧不对称, 与键盘参数无关.")


if __name__ == "__main__":
    main()
