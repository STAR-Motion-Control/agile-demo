#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机抱箱姿保持 — 按【箱宽】推导抱姿并通过 rt/arm_sdk 持续保持.

抱箱下蹲换不同大小箱子的真机接口 (sim 已验证 box_hug.hug_for_box 映射):

  python3 hold_box_hug.py --box-width 0.30                      # 只给宽即可
  python3 hold_box_hug.py --box-width 0.45 --roll 0.21          # 大箱用 sim 修正值
  python3 hold_box_hug.py --box-width 0.35 --waist-pitch 0.15   # AGILE 抱蹲必给

流程 (与 test_squat_cycles.py --mode box 配套):
  1. 底座 adapter + merger 在跑 (start_g1_onboard.sh dwbc|agile), 机器人站稳;
  2. 本脚本: 手臂从当前位置 ramp 进抱姿 (--ramp-s, 默认 6s) 后一直保持;
  3. 人工递箱进怀里 (箱底放掌台, 站立时约 0.80m 高), 用 [ / ] 微调夹紧;
  4. 另开终端跑 test_squat_cycles.py --mode box (只发高度, 不碰手臂);
  5. 测完取走箱子, 本脚本按 q 释放 (先回中性位再淡出 arm_sdk 权重).

保持期间按键:  [ =夹紧(roll-0.01)   ] =放松(roll+0.01)
              , =腰前倾-0.01      . =腰前倾+0.01
              q =释放退出(先确认箱子已取走!)

发布约定与 dual_arm_target_reach.py 完全一致 (rt/arm_sdk, 权重=motor_cmd[29].q,
mode_machine/mode_pr 每帧同步 lowstate, CRC), kp/kd 也用它真机验证过的值.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time
from pathlib import Path


def _reexec_without_ros_env():
    """001 的 ~/.bashrc 无条件 source ROS -> LD_LIBRARY_PATH 带 ROS libddsc,
    pip cyclonedds 会撞 dds_write.c:318 断言。LD_LIBRARY_PATH 进程启动后改
    无效, 所以检测到污染就用清洁环境 re-exec 自己 (2026-07-03 001 实测)."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_hug import (PALM_SHELF_Z, REF, ROLL_MAX, ROLL_MIN, WAIST_PITCH,
                     hug_for_box, hug_pose)
from ipc_ctl import RawKeys

# ROS 的 libddsc 与 pip cyclonedds ABI 冲突守护 (5080/002 需要; 001 无此目录, 跳过)
_ddsc = Path(os.environ.get("CYCLONEDDS_HOME",
                            Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

ARM_SDK_ENABLE = 29           # 权重关节: motor_cmd[29].q ∈ [0,1]
JOINTS = list(range(12, 29))  # 腰 12-14 + 左臂 15-21 + 右臂 22-28
L_ROLL, R_ROLL = 16, 23
WAIST_PITCH_MAX = 0.35

# kp/kd — dual_arm_target_reach.py 真机抱 2kg 箱验证过的值
KP = {12: 250.0, 13: 250.0, 14: 250.0,                 # waist
      15: 150.0, 22: 150.0,                            # shoulder pitch
      16: 120.0, 23: 120.0,                            # shoulder roll
      17: 130.0, 24: 130.0,                            # shoulder yaw
      18: 130.0, 25: 130.0,                            # elbow
      19: 180.0, 20: 180.0, 21: 180.0,                 # L wrist r/p/y
      26: 180.0, 27: 180.0, 28: 180.0}                 # R wrist r/p/y
KD = {j: (5.0 if j <= 14 else 10.0) for j in JOINTS}

RELEASE_Q = {j: 0.0 for j in JOINTS}   # 中性位 = 两套控制器上半身默认 0 位


class HugHolder:

    def __init__(self, pose: dict, hz: float = 50.0):
        self.pose = dict(pose)
        self.dt = 1.0 / hz
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        self.low_state = msg

    def wait_state(self, timeout_s: float = 10.0):
        t0 = time.monotonic()
        while self.low_state is None:
            if time.monotonic() - t0 > timeout_s:
                raise SystemExit("等不到 rt/lowstate — 检查 --iface / 底座是否在跑")
            time.sleep(0.05)

    def read_q(self) -> dict:
        return {j: float(self.low_state.motor_state[j].q) for j in JOINTS}

    def publish(self, q: dict, weight: float = 1.0):
        st = self.low_state
        if st is not None:
            self.low_cmd.mode_machine = st.mode_machine
            self.low_cmd.mode_pr = st.mode_pr
        en = self.low_cmd.motor_cmd[ARM_SDK_ENABLE]
        en.mode, en.q, en.dq, en.kp, en.kd, en.tau = 1, float(weight), 0.0, 0.0, 0.0, 0.0
        for j in JOINTS:
            mc = self.low_cmd.motor_cmd[j]
            mc.mode, mc.tau, mc.dq = 1, 0.0, 0.0
            mc.q = float(q[j])
            mc.kp, mc.kd = KP[j], KD[j]
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    @staticmethod
    def _lerp(qa: dict, qb: dict, r: float) -> dict:
        r = max(0.0, min(1.0, r))
        return {j: qa[j] * (1.0 - r) + qb[j] * r for j in qa}

    def ramp_in(self, ramp_s: float):
        """当前位姿 -> 抱姿, cosine 缓入缓出."""
        q0 = self.read_q()
        n = max(1, int(ramp_s / self.dt))
        for k in range(n + 1):
            r = 0.5 - 0.5 * math.cos(math.pi * k / n)
            self.publish(self._lerp(q0, self.pose, r))
            time.sleep(self.dt)

    def hold(self, waist_pitch: float):
        """保持抱姿; [ ] 调夹紧, , . 调腰前倾, q 释放退出."""
        roll = self.pose[L_ROLL]
        confirm_release = False
        print(f"\n[保持中] roll={roll:+.2f} waist_pitch={waist_pitch:+.2f}   "
              "键: [ 夹紧  ] 放松  , . 腰前倾∓  q 释放退出", flush=True)
        with RawKeys() as keys:
            while True:
                k = keys.get()
                if k == "[":
                    roll = max(ROLL_MIN, roll - 0.01)
                elif k == "]":
                    roll = min(ROLL_MAX, roll + 0.01)
                elif k == ",":
                    waist_pitch = max(0.0, waist_pitch - 0.01)
                elif k == ".":
                    waist_pitch = min(WAIST_PITCH_MAX, waist_pitch + 0.01)
                elif k == "q":
                    if confirm_release:
                        return
                    confirm_release = True
                    print("\n确认箱子已取走/放稳后【再按一次 q】释放; "
                          "按其它键取消", flush=True)
                elif k is not None:
                    confirm_release = False
                if k in ("[", "]", ",", "."):
                    self.pose[L_ROLL] = roll
                    self.pose[R_ROLL] = -roll
                    self.pose[WAIST_PITCH] = waist_pitch
                    print(f"  roll={roll:+.2f}  waist_pitch={waist_pitch:+.2f}",
                          flush=True)
                self.publish(self.pose)
                time.sleep(self.dt)

    def release(self, release_s: float):
        """抱姿 -> 中性0位 (release_s) -> arm_sdk 权重淡出 (release_s)."""
        q0 = dict(self.pose)
        n = max(1, int(release_s / self.dt))
        for k in range(n + 1):                      # 回中性位, 权重仍 1
            self.publish(self._lerp(q0, RELEASE_Q, k / n))
            time.sleep(self.dt)
        for k in range(n + 1):                      # 权重 1 -> 0, 交还控制器
            self.publish(RELEASE_Q, weight=1.0 - k / n)
            time.sleep(self.dt)
        print("[释放完成] 上半身已交还底座控制器", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box-width", type=float, required=True,
                    help="箱宽 m (主参数; 基准箱 0.35)")
    ap.add_argument("--box-depth", type=float, default=None,
                    help="箱深 m (默认基准 0.25; 只在换更深的箱时需要)")
    ap.add_argument("--box-height", type=float, default=None,
                    help="箱高 m (只影响递箱高度提示, 默认 0.25)")
    ap.add_argument("--waist-pitch", type=float, default=0.0,
                    help="腰前倾 rad — AGILE 抱蹲必给 0.15 (sim 甜点); dwbc 可 0")
    ap.add_argument("--sp", type=float, default=None, help="手动覆盖肩pitch")
    ap.add_argument("--roll", type=float, default=None,
                    help="手动覆盖肩roll (如大箱 0.45 宽用 sim 修正值 0.21)")
    ap.add_argument("--elbow", type=float, default=None, help="手动覆盖肘")
    ap.add_argument("--ramp-s", type=float, default=6.0, help="进入抱姿时长")
    ap.add_argument("--release-s", type=float, default=2.5, help="释放各段时长")
    ap.add_argument("--iface", default="enP8p1s0",
                    help="DDS 网口 (G1 本体=enP8p1s0)")
    a = ap.parse_args()

    pose, _, _, info = hug_for_box(a.box_width, a.box_depth, a.box_height)
    if a.sp is not None or a.roll is not None or a.elbow is not None:
        pose = hug_pose(a.sp if a.sp is not None else info["sp"],
                        a.roll if a.roll is not None else info["roll"],
                        a.elbow if a.elbow is not None else REF["elbow"])
    pose[WAIST_PITCH] = a.waist_pitch

    print("=" * 56)
    print(f"  抱箱姿保持  箱宽 {a.box_width:.2f}m"
          f"{'' if a.box_depth is None else f' 深 {a.box_depth:.2f}m'}")
    print(f"  推导抱姿: sp={info['sp']:+.2f}  roll={info['roll']:+.2f}"
          f"  elbow={REF['elbow']:+.2f}  腰前倾={a.waist_pitch:+.2f}")
    if info["clamped"]:
        print("  !! 尺寸超出线性区, 抱姿已限幅 — 先在 sim 用 tune_box_hold.py 验证")
    if a.roll is not None or a.sp is not None:
        print(f"  手动覆盖: sp={a.sp} roll={a.roll} elbow={a.elbow}")
    print(f"  递箱提示: 箱底放掌台 (站立时离地约 {PALM_SHELF_Z:.2f}m)")
    if a.waist_pitch == 0.0:
        print("  提醒: AGILE 抱蹲要 --waist-pitch 0.15, dwbc 可不给")
    print("=" * 56)
    print("\n前置: 底座 adapter+merger 在跑且机器人站稳, 手臂周围无障碍.")
    input("[Enter] 开始 ramp 进抱姿, Ctrl+C 取消 ...")

    ChannelFactoryInitialize(0, a.iface)
    holder = HugHolder(pose)
    holder.wait_state()
    print(f"收到 lowstate, {a.ramp_s:.0f}s ramp 进抱姿 ...", flush=True)
    try:
        holder.ramp_in(a.ramp_s)
        holder.hold(a.waist_pitch)
    except KeyboardInterrupt:
        print("\nCtrl+C — 转入释放流程 (确认箱子已取走!)", flush=True)
    holder.release(a.release_s)


if __name__ == "__main__":
    main()
