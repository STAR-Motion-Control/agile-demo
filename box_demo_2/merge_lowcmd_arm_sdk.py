#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 RoboJuDo 的 rt/lowcmd_rl 与 box_demo / dual_arm 的 rt/arm_sdk 合成为最终 rt/lowcmd。

- 默认：motor 0–11 及未启用 arm_sdk 时 12–34 均来自 RL（下半身 + 其余）。
- 当 rt/arm_sdk 中 motor_cmd[29].q > 阈值 且报文未过期：motor 12–29（腰 + 双臂 + arm_sdk 权重槽）
  整段来自 arm_sdk（kp/kd/q 等与 box_demo 一致）。

[lz 2026-07-10] 腰部"运动窗口"仲裁 (--waist-to-rl-on-motion, 默认关=旧行为):
  背景: arm_natural_hang 垂臂保持 50Hz 常开地经 rt/arm_sdk 锁腰(12-14, kp250)，而
  dwbc policy 的 15 维动作里有 3 维是腰 → 行走时 policy 被抢走腰，站高塌/转向前漂/
  起步晃(甚至 adapter 的 back-lean 腰补偿从未生效)。与操控约定协议: 拍照=腰锁零，
  微调执行=运控拿回腰(手臂仍由操控垂直保持)，微调结束=腰回零交还。
  实现: 侧线程 20Hz 读 /tmp/robojudo_ext_cmd.json(与 adapter read_external_command
  相同语义: timestamp 新鲜度 0.40s / fsm / estop|limp / velocity)，WaistArbiter 状态机
    LOCKED --运动(fresh∧RL_FULL∧|v|>阈值, 连续2采样)--> BLEND_IN(0.3s) --> RL(腰=policy)
    RL --零速持续 handback_delay(0.8s, stale 视同零速)--> BLEND_OUT(0.4s) --> LOCKED
  bad(estop/limp/RL_LOWER 抓取) → 快速 BLEND_OUT(0.2s) 收敛回 LOCKED。
  BLEND 阶段腰三电机在 arm_sdk 与 RL 间线性混合，无 kp 阶跃。
  安全层(2026-07-10 对抗审查修复):
    ①哨兵防护: LIMP/DAMP 帧(q=PosStopF 2.146e9/dq=VelStopF/kp=0)绝不混入位置内容——
      _blend_motor 只从"位置控制有效"(kp>0 且 q/dq 理智)的一侧取 q/dq/tau, kp/kd 仍线性
      过渡(CRITICAL 修复: 否则 estop 中混出会满扭矩拉腰)。
    ②看门线程免疫: read_ipc_flags 对任意损坏内容(非 dict JSON/字段类型错)一律返回安全值,
      _ipc_loop 全包 try/except; 旗标带 monotonic 时间戳, _tick 发现采样 >0.25s 未更新
      (线程死亡)即视为 (False,False) → 收敛回 LOCKED(HIGH 修复)。
    ③dt 夹紧 0.05s: tick 线程卡顿/RL 断流恢复后 alpha 续坡而非跳变(MEDIUM 修复)。
    ④运动判定对齐 adapter: units=agile 用 0.05(=WALK_THRESHOLD_NOMOVE), 连续 2 采样
      (~100ms 防抖)才开窗(MEDIUM 修复)。
    ⑤腰限速器: 仲裁启用时对最终腰 q 施加 1.5 rad/s 转率钳制, 封顶任何来源的摆动
      (keeper 快照陈旧/混合窗内摆幅)(MEDIUM 修复)。
  已知接受特性: estop 后腰需 ~0.25s(检测≤0.1s + bad 混出 0.2s)才完全回到 arm 锁定,
  期间 kp 连续、无位置阶跃, 收敛点=旧版 estop 中的恒 kp250 锁腰; RL_LOWER 写入到
  dual_arm 开始伸臂有 ≥1.0s 间隙, 0.2s bad 混出裕量充足。

启动顺序（同一 DDS 域、同一网卡）：
  1) python merge_lowcmd_arm_sdk.py --iface enx2c16dbaa7742baa7742
  2) python run_pipeline.py -c g1_mjlab_loco_real_merge
  3) python box_demo_main.py --iface enx2c16dbaa7742baa7742

须先于 unitree_sdk2py 加载与 wheel 匹配的 libddsc（与 run_pipeline / box_demo 一致）。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_MOTORS = 35
RESERVE_LEN = 4
# 与 dual_arm_target_reach.G1Joint 一致：腰(12–14) + 双臂(15–28) + ArmSdkEnable(29)
OVERLAY_LO = 12
OVERLAY_HI = 30  # exclusive → 12..29
# 腰三电机（WaistYaw/Roll/Pitch = 12/13/14），运动窗口内交还给 RL(policy)
WAIST_LO = 12
WAIST_HI = 15  # exclusive → 12..14

# IPC 语义与 groot_wbc_boxdemo_adapter.read_external_command 保持一致
DEFAULT_IPC_FILE = "/tmp/robojudo_ext_cmd.json"
DEFAULT_IPC_STALE_S = 0.40   # = adapter --cmd-stale-s 默认
MOTION_EPS = 1e-3            # 非 agile(归一化)写者的运动阈值
MOTION_EPS_AGILE = 0.05      # units=agile(物理量)写者: 对齐 adapter WALK_THRESHOLD_NOMOVE
IPC_FLAGS_MAX_AGE_S = 0.25   # 旗标采样超龄(看门线程死亡)→ 视为 (False,False) 自愈
ARBITER_MAX_DT = 0.05        # dt 夹紧: 卡顿/断流恢复后续坡不跳变
# 位置内容有效性判定(哨兵防护): PosStopF=2.146e9, VelStopF=16000 必须被拒
Q_SANE_RAD = 10.0
DQ_SANE_RAD_S = 50.0
# 腰限速器: 仲裁启用时最终腰 q 的最大变化率
WAIST_SLEW_RAD_S = 1.5


def _copy_motor(dst: LowCmd_, src: LowCmd_, idx: int) -> None:
    dm = dst.motor_cmd[idx]
    sm = src.motor_cmd[idx]
    dm.mode = sm.mode
    dm.q = sm.q
    dm.dq = sm.dq
    dm.tau = sm.tau
    dm.kp = sm.kp
    dm.kd = sm.kd
    dm.reserve = sm.reserve


def _pos_valid(m) -> bool:
    """该侧是否在做有效位置控制(可以安全地取其 q/dq/tau)。
    LIMP/DAMP 帧(kp=0, q=PosStopF, dq=VelStopF)必须判 False。"""
    return (m.kp > 1e-6) and (abs(m.q) < Q_SANE_RAD) and (abs(m.dq) < DQ_SANE_RAD_S)


def _blend_motor(dst: LowCmd_, rl: LowCmd_, arm: LowCmd_, idx: int, rl_weight: float) -> None:
    """腰过渡混合: rl_weight=1 全 RL, =0 全 arm_sdk。dst 预先已是 RL 值。
    kp/kd 恒线性插值(两侧都是真实增益, 中间值安全)。q/dq/tau 只取"位置控制有效"
    的一侧或两侧插值——绝不把 PosStopF/VelStopF 哨兵混进带刚度的命令(CRITICAL 防护)。
    mode/reserve 不可插值, 取权重大的一侧(两侧无效时取 arm 侧=旧行为收敛点)。"""
    dm = dst.motor_cmd[idx]
    rm = rl.motor_cmd[idx]
    am = arm.motor_cmd[idx]
    w = rl_weight
    v = 1.0 - w
    dm.kp = w * rm.kp + v * am.kp
    dm.kd = w * rm.kd + v * am.kd
    rl_ok = _pos_valid(rm)
    arm_ok = _pos_valid(am)
    if rl_ok and arm_ok:
        dm.q = w * rm.q + v * am.q
        dm.dq = w * rm.dq + v * am.dq
        dm.tau = w * rm.tau + v * am.tau
        src = rm if w >= 0.5 else am
    elif arm_ok:
        dm.q = am.q
        dm.dq = am.dq
        dm.tau = am.tau
        src = am
    elif rl_ok:
        dm.q = rm.q
        dm.dq = rm.dq
        dm.tau = rm.tau
        src = rm
    else:
        # 两侧都非位置控制(全 damp/limp): 位置内容取 arm 侧原样(=旧行为端点),
        # 但哨兵 dq 不能配上插值出的 kd → dq/tau 清零, q 保留 arm(kp 混合后≈0)
        dm.q = am.q
        dm.dq = 0.0
        dm.tau = 0.0
        src = am
    dm.mode = src.mode
    dm.reserve = src.reserve


def _copy_full(dst: LowCmd_, src: LowCmd_) -> None:
    dst.mode_pr = src.mode_pr
    dst.mode_machine = src.mode_machine
    for i in range(NUM_MOTORS):
        _copy_motor(dst, src, i)
    for i in range(RESERVE_LEN):
        dst.reserve[i] = src.reserve[i]


def _snapshot(msg: LowCmd_) -> LowCmd_:
    out = unitree_hg_msg_dds__LowCmd_()
    _copy_full(out, msg)
    out.crc = msg.crc
    return out


def read_ipc_flags(path: str, stale_s: float, now: float | None = None) -> tuple[bool, bool]:
    """读运控 IPC 文件 → (motion, bad)。语义对齐 adapter.read_external_command:
    - 文件缺失/任意损坏(非 dict JSON、字段类型错)/stale → (False, False)
      # 不能开运动窗; 已开的窗按零速计时交还 → 永远向 LOCKED 收敛
    - fresh 且 estop/limp/DAMP/LIMP/非 RL_FULL(含 RL_LOWER 抓取) → (False, True)
    - fresh 且 RL_FULL → (|v| 超阈值, False); units=agile 阈值 0.05(对齐 adapter
      WALK_THRESHOLD_NOMOVE), 归一化写者用 MOTION_EPS
    任何异常都不得逃逸(看门线程免疫)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return False, False
        if now is None:
            now = time.time()
        fresh = (now - float(raw.get("timestamp", 0.0))) <= stale_s
        if not fresh:
            return False, False
        fsm = str(raw.get("fsm") or "RL_FULL")
        if bool(raw.get("estop")) or bool(raw.get("limp")) or fsm != "RL_FULL":
            return False, True
        vel = raw.get("velocity")
        if not isinstance(vel, dict):
            vel = {}
        v_sum = (abs(float(vel.get("forward", 0.0)))
                 + abs(float(vel.get("lateral", 0.0)))
                 + abs(float(vel.get("yaw", 0.0))))
        eps = MOTION_EPS_AGILE if raw.get("units") == "agile" else MOTION_EPS
        return v_sum > eps, False
    except Exception:
        return False, False


class WaistArbiter:
    """腰所有权状态机。alpha: 0=arm_sdk 锁腰(旧行为), 1=RL(policy) 持腰。

    LOCKED --motion--> BLEND_IN --alpha=1--> RL
    RL --零速(或 stale)持续 handback_delay--> BLEND_OUT --alpha=0--> LOCKED
    bad(estop/抓取/limp) 在任何状态 → 转 BLEND_OUT 以 bad_blend_out_s(更快)收敛。
    disabled → 恒 LOCKED/alpha=0(与旧版逐字节一致)。
    dt 夹紧 ARBITER_MAX_DT: 转移当拍即消耗一个(夹紧后的) dt 的坡度。
    """

    LOCKED = "LOCKED"
    BLEND_IN = "BLEND_IN"
    RL = "RL"
    BLEND_OUT = "BLEND_OUT"

    def __init__(self, enabled: bool, blend_in_s: float = 0.3,
                 blend_out_s: float = 0.4, handback_delay: float = 0.8,
                 bad_blend_out_s: float = 0.2):
        self.enabled = bool(enabled)
        self.blend_in_s = max(1e-3, float(blend_in_s))
        self.blend_out_s = max(1e-3, float(blend_out_s))
        self.bad_blend_out_s = max(1e-3, float(bad_blend_out_s))
        self.handback_delay = max(0.0, float(handback_delay))
        self.state = self.LOCKED
        self.alpha = 0.0
        self._zero_since: float | None = None
        self._last_t: float | None = None
        self._out_fast = False   # 本次 BLEND_OUT 是否 bad 触发(用快速常数)

    def update(self, now: float, motion: bool, bad: bool) -> float:
        if not self.enabled:
            self.state = self.LOCKED
            self.alpha = 0.0
            self._zero_since = None
            self._last_t = now
            return 0.0

        dt = 0.0 if self._last_t is None else min(ARBITER_MAX_DT, max(0.0, now - self._last_t))
        self._last_t = now

        if bad and self.state in (self.BLEND_IN, self.RL):
            self.state = self.BLEND_OUT
            self._out_fast = True
            self._zero_since = None

        if self.state == self.LOCKED:
            self.alpha = 0.0
            if motion and not bad:
                self.state = self.BLEND_IN
        elif self.state == self.BLEND_IN:
            self.alpha = min(1.0, self.alpha + dt / self.blend_in_s)
            if self.alpha >= 1.0:
                self.state = self.RL
                self._zero_since = None
        elif self.state == self.RL:
            self.alpha = 1.0
            if motion:
                self._zero_since = None
            else:
                if self._zero_since is None:
                    self._zero_since = now
                elif now - self._zero_since >= self.handback_delay:
                    self.state = self.BLEND_OUT
                    self._out_fast = False
                    self._zero_since = None
        if self.state == self.BLEND_OUT:
            if bad:
                self._out_fast = True
            if motion and not bad:
                self.state = self.BLEND_IN
                # alpha 保留当前值, BLEND_IN 从这里继续升
            else:
                out_s = self.bad_blend_out_s if self._out_fast else self.blend_out_s
                self.alpha = max(0.0, self.alpha - dt / out_s)
                if self.alpha <= 0.0:
                    self.state = self.LOCKED
                    self._out_fast = False
        return self.alpha


class Merger:
    def __init__(
        self,
        iface: str,
        hz: float,
        weight_threshold: float,
        arm_stale_s: float,
        rl_stale_s: float,
        waist_to_rl_on_motion: bool = False,
        waist_ipc_file: str = DEFAULT_IPC_FILE,
        waist_ipc_stale_s: float = DEFAULT_IPC_STALE_S,
        waist_blend_in_s: float = 0.3,
        waist_blend_out_s: float = 0.4,
        waist_bad_blend_out_s: float = 0.2,
        waist_handback_delay: float = 0.8,
    ):
        self._iface = iface
        self._period = 1.0 / hz
        self._weight_threshold = weight_threshold
        self._arm_stale_s = arm_stale_s
        self._rl_stale_s = rl_stale_s
        self._lock = threading.Lock()
        self._rl: LowCmd_ | None = None
        self._rl_t = 0.0
        self._arm: LowCmd_ | None = None
        self._arm_t = 0.0
        self._crc = CRC()
        # 须在 ChannelFactoryInitialize 之后再创建 Publisher/Subscriber，否则 participant 为 None
        self._pub = None
        self._sub_rl = None
        self._sub_arm = None
        self._thread: RecurrentThread | None = None
        # --- 腰仲裁 ---
        self._arbiter = WaistArbiter(
            enabled=waist_to_rl_on_motion,
            blend_in_s=waist_blend_in_s,
            blend_out_s=waist_blend_out_s,
            bad_blend_out_s=waist_bad_blend_out_s,
            handback_delay=waist_handback_delay,
        )
        self._ipc_file = waist_ipc_file
        self._ipc_stale_s = waist_ipc_stale_s
        # (motion, bad, sample_monotonic); 元组整体替换, GIL 下读写原子。
        # 初始 ts=0 → 超龄 → (False,False), 看门线程首采样前恒锁腰。
        self._ipc_flags = (False, False, 0.0)
        self._ipc_thread: threading.Thread | None = None
        self._ipc_stop = threading.Event()
        self._last_logged_state = WaistArbiter.LOCKED
        # 腰限速器状态: {motor_idx: (last_q, last_monotonic)}
        self._waist_slew: dict[int, tuple[float, float]] = {}

    # --- DDS 回调 ---
    def _on_rl(self, msg: LowCmd_):
        with self._lock:
            self._rl = _snapshot(msg)
            self._rl_t = time.monotonic()

    def _on_arm(self, msg: LowCmd_):
        with self._lock:
            self._arm = _snapshot(msg)
            self._arm_t = time.monotonic()

    # --- 腰仲裁: IPC 侧线程(免疫: 任何异常不杀线程, 失败即发布安全旗标) ---
    def _ipc_loop(self):
        prev_motion = False
        while not self._ipc_stop.wait(0.05):
            try:
                motion, bad = read_ipc_flags(self._ipc_file, self._ipc_stale_s)
            except Exception:
                motion, bad = False, False
            # 连续 2 采样(~100ms)防抖才承认运动, 单帧毛刺不开窗
            eff_motion = motion and prev_motion
            prev_motion = motion
            self._ipc_flags = (eff_motion, bad, time.monotonic())

    def _tick(self):
        now = time.monotonic()
        with self._lock:
            rl = self._rl
            rl_age = now - self._rl_t
            arm = self._arm
            arm_age = now - self._arm_t if arm is not None else 1e9

        if rl is None or rl_age > self._rl_stale_s:
            return

        out = unitree_hg_msg_dds__LowCmd_()
        _copy_full(out, rl)

        use_arm = (
            arm is not None
            and arm_age <= self._arm_stale_s
            and float(arm.motor_cmd[29].q) > self._weight_threshold
        )

        # 腰仲裁每 tick 都推进(与 use_arm 无关, 保持计时连续)。
        # 旗标采样超龄(看门线程死亡/未启动)→ 安全值 → 收敛 LOCKED。
        motion, bad, flags_t = self._ipc_flags
        if now - flags_t > IPC_FLAGS_MAX_AGE_S:
            motion, bad = False, False
        alpha = self._arbiter.update(now, motion, bad)
        if self._arbiter.enabled and self._arbiter.state != self._last_logged_state:
            print(f"[WAIST] {self._last_logged_state} -> {self._arbiter.state} "
                  f"(alpha={alpha:.2f} motion={motion} bad={bad})", flush=True)
            self._last_logged_state = self._arbiter.state

        if use_arm:
            for i in range(OVERLAY_LO, OVERLAY_HI):
                if WAIST_LO <= i < WAIST_HI:
                    if alpha >= 1.0:
                        pass                                  # RL 持腰: out 已是 RL 值
                    elif alpha <= 0.0:
                        _copy_motor(out, arm, i)              # 锁腰: 旧行为
                    else:
                        _blend_motor(out, rl, arm, i, alpha)
                    if self._arbiter.enabled:
                        self._slew_waist(out, i, now)
                else:
                    _copy_motor(out, arm, i)

        out.crc = self._crc.Crc(out)
        if self._pub is not None:
            self._pub.Write(out)

    def _slew_waist(self, out: LowCmd_, idx: int, now: float) -> None:
        """腰 q 转率钳制(仅仲裁启用时): 封顶任何来源切换/快照陈旧导致的摆动速率。
        对哨兵值(|q|>Q_SANE, 如 LIMP 的 PosStopF)不钳制也不记录——那种帧 kp=0,
        位置无效, 钳制反而会持续输出旧位置。"""
        m = out.motor_cmd[idx]
        if abs(m.q) >= Q_SANE_RAD:
            self._waist_slew.pop(idx, None)
            return
        prev = self._waist_slew.get(idx)
        if prev is not None:
            last_q, last_t = prev
            dt = min(ARBITER_MAX_DT, max(0.0, now - last_t))
            lim = WAIST_SLEW_RAD_S * dt
            dq = m.q - last_q
            if dq > lim:
                m.q = last_q + lim
            elif dq < -lim:
                m.q = last_q - lim
        self._waist_slew[idx] = (m.q, now)

    def run(self):
        ChannelFactoryInitialize(0, self._iface)
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._sub_rl = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
        self._sub_rl.Init(self._on_rl, 10)
        self._sub_arm = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self._sub_arm.Init(self._on_arm, 10)
        if self._arbiter.enabled:
            self._ipc_thread = threading.Thread(target=self._ipc_loop, daemon=True,
                                                name="waist_ipc_watch")
            self._ipc_thread.start()
        self._thread = RecurrentThread(
            interval=self._period,
            target=self._tick,
            name="merge_lowcmd",
        )
        self._thread.Start()
        print(
            f"merge_lowcmd_arm_sdk: iface={self._iface} 合成→rt/lowcmd  "
            f"订阅 rt/lowcmd_rl + rt/arm_sdk  周期={self._period*1000:.1f} ms"
        )
        if self._arbiter.enabled:
            print(f"[WAIST] 运动窗口腰仲裁 ON: ipc={self._ipc_file} "
                  f"(stale {self._ipc_stale_s:.2f}s)  blend_in={self._arbiter.blend_in_s:.2f}s "
                  f"blend_out={self._arbiter.blend_out_s:.2f}s "
                  f"(bad→{self._arbiter.bad_blend_out_s:.2f}s) "
                  f"handback={self._arbiter.handback_delay:.2f}s  "
                  f"slew={WAIST_SLEW_RAD_S:.1f}rad/s  "
                  f"腰({WAIST_LO}-{WAIST_HI-1}) 运动时归 RL, 静止/拍照/抓取归 arm_sdk")
            if self._arbiter.bad_blend_out_s > 0.5:
                print("[WAIST][WARN] bad 混出 >0.5s — RL_LOWER→抓取裕量约 1.0s, "
                      "过慢可能让伸臂先于腰锁定, 建议 ≤0.3s")
        else:
            print("[WAIST] 腰仲裁 OFF (旧行为: arm_sdk 启用时腰恒被锁)")
        print("按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n退出。")
        finally:
            self._ipc_stop.set()
            if self._thread is not None:
                self._thread.Wait(timeout=1.0)


def main():
    p = argparse.ArgumentParser(description="Merge rt/lowcmd_rl + rt/arm_sdk → rt/lowcmd")
    p.add_argument("--iface", type=str, default=os.environ.get("UNITREE_DDS_INTERFACE", "enx2c16dbaa7742baa7742"))
    p.add_argument("--hz", type=float, default=500.0, help="发布 rt/lowcmd 频率")
    p.add_argument("--weight-threshold", type=float, default=1e-3, help="arm_sdk motor[29].q 超过此值则覆盖腰+臂")
    p.add_argument("--arm-stale-s", type=float, default=0.25, help="超过此时间未收到 arm_sdk 则只用 RL 上半身")
    p.add_argument("--rl-stale-s", type=float, default=0.5, help="超过此时间未收到 lowcmd_rl 则停止写 rt/lowcmd")
    p.add_argument("--waist-to-rl-on-motion", action="store_true",
                   help="运动窗口腰仲裁: IPC 有运动命令时腰(12-14)交还 RL policy, "
                        "静止/拍照/抓取仍由 arm_sdk 锁定。默认关=旧行为")
    p.add_argument("--waist-ipc-file", type=str, default=DEFAULT_IPC_FILE,
                   help="运控 IPC 命令文件(与 adapter --cmd-file 一致)")
    p.add_argument("--waist-ipc-stale-s", type=float, default=DEFAULT_IPC_STALE_S,
                   help="IPC 新鲜度阈值(与 adapter --cmd-stale-s 一致)")
    p.add_argument("--waist-blend-in-s", type=float, default=0.3, help="腰 arm→RL 过渡时长")
    p.add_argument("--waist-blend-out-s", type=float, default=0.4, help="腰 RL→arm 过渡时长(零速交还)")
    p.add_argument("--waist-bad-blend-out-s", type=float, default=0.2,
                   help="estop/抓取触发的快速交还时长(建议 ≤0.3, RL_LOWER→伸臂裕量 ~1.0s)")
    p.add_argument("--waist-handback-delay", type=float, default=0.8,
                   help="零速持续该时长后交还腰(GrootMover 停后写 0.4s 零速, 需 >0.4)")
    args = p.parse_args()
    Merger(
        iface=args.iface,
        hz=args.hz,
        weight_threshold=args.weight_threshold,
        arm_stale_s=args.arm_stale_s,
        rl_stale_s=args.rl_stale_s,
        waist_to_rl_on_motion=args.waist_to_rl_on_motion,
        waist_ipc_file=args.waist_ipc_file,
        waist_ipc_stale_s=args.waist_ipc_stale_s,
        waist_blend_in_s=args.waist_blend_in_s,
        waist_blend_out_s=args.waist_blend_out_s,
        waist_bad_blend_out_s=args.waist_bad_blend_out_s,
        waist_handback_delay=args.waist_handback_delay,
    ).run()


if __name__ == "__main__":
    main()
