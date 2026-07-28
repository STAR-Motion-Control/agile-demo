#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 RoboJuDo 的 rt/lowcmd_rl 与 box_demo / dual_arm 的 rt/arm_sdk 合成为最终 rt/lowcmd。

- 默认：motor 0–11 及未启用 arm_sdk 时 12–34 均来自 RL（下半身 + 其余）。
- 当 rt/arm_sdk 中 motor_cmd[29].q > 阈值 且报文未过期：默认沿用操控协议，
  腰 12–14 + 双臂 15–28 来自 arm_sdk。
- 扩展协议：motor_cmd[29].dq >= 0.5 表示 arms-only。此时仅双臂来自 arm_sdk，
  腰在静止和运动中都逐帧保留 RL；旧操控发布者 dq=0，语义不变。

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
import errno
import fcntl
import json
import math
import os
import signal
import socket
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

from keyboard_arm_hang import ARM_JOINTS, ArmHangPlanner, policy_frame_from_lowcmd

REFACTOR_ROOT = Path(__file__).resolve().parent.parent
if str(REFACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(REFACTOR_ROOT))

from onboard_runtime.arm_protocol import (
    ARM_JOINT_INDICES as MANIP_ARM_JOINTS,
    DEFAULT_ARM_RUNTIME_SOCKET,
    DEFAULT_ARM_RUNTIME_STATUS,
    PROFILE_MANIP,
    PROFILE_WAIST_LEGACY,
    ArmFrame,
)
from onboard_runtime.arm_runtime_broker import ManipulationArmBroker
from onboard_runtime.command_stream import (
    DEFAULT_MERGER_COMMAND_SOCKET,
    LatestCommandReceiver,
)
from onboard_runtime.loop_health import RuntimeHealthHeartbeat

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
# arm_sdk 首次接管腰时，从当前 RL 输出平滑过渡。若直接把首帧 arm 目标
# 覆盖到腰上，即使该目标等于实测角，也会撤掉 RL 依靠位置误差产生的
# 重力支撑力矩。
DEFAULT_WAIST_TAKEOVER_BLEND_S = 0.35
DEFAULT_ARM_CONTROL_SOCKET = "/tmp/groot_arm_control.sock"
DEFAULT_ARM_CONTROL_STATUS_FILE = "/tmp/groot_arm_control_status.json"
DEFAULT_MERGER_HEALTH_FILE = "/tmp/groot_merger_health.json"
# arm_sdk enable slot (motor 29) ownership extension.  The slot is metadata,
# not a physical actuator; q remains the official weight and dq marks arms-only.
ARM_SDK_ARMS_ONLY_THRESHOLD = 0.5
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
DEFAULT_SAFETY_DAMPING_KD = 8.0
DEFAULT_MAX_TICK_GAP_S = 0.060
TICK_DEGRADED_HOLD_S = 1.0


def _runtime_profile_gains(profile: str, joint: int) -> tuple[float, float]:
    if profile == PROFILE_WAIST_LEGACY:
        return (20.0, 1.5) if WAIST_LO <= joint < WAIST_HI else (40.0, 2.0)
    if profile != PROFILE_MANIP:
        raise ValueError(f"unsupported arm runtime profile: {profile}")
    if WAIST_LO <= joint < WAIST_HI:
        return 250.0, 5.0
    if joint in (15, 22):
        return 150.0, 10.0
    if joint in (16, 23):
        return 120.0, 10.0
    if joint in (17, 24):
        return 130.0, 10.0
    if joint in (18, 25):
        return 130.0, 10.0
    if joint in (19, 20, 21, 26, 27, 28):
        return 180.0, 10.0
    raise ValueError(f"unsupported upper-body joint: {joint}")


def _runtime_arm_command(
    frame: ArmFrame,
    *,
    mode_machine: int,
    mode_pr: int,
) -> LowCmd_:
    command = unitree_hg_msg_dds__LowCmd_()
    command.mode_machine = int(mode_machine)
    command.mode_pr = int(mode_pr)
    for joint, q in zip(MANIP_ARM_JOINTS, frame.q):
        motor = command.motor_cmd[joint]
        motor.mode = 1
        motor.q = float(q)
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp, motor.kd = _runtime_profile_gains(frame.profile, joint)
    enable = command.motor_cmd[29]
    enable.mode = 1
    enable.q = float(frame.weight)
    enable.dq = 0.0
    enable.tau = 0.0
    enable.kp = 0.0
    enable.kd = 0.0
    return command


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


def _force_safety_frame(
    command: LowCmd_,
    fsm: str,
    *,
    state_q: tuple[float, ...] | None = None,
    damping_kd: float = DEFAULT_SAFETY_DAMPING_KD,
) -> None:
    """Replace every motor field with an explicit whole-body safety command."""
    safety_fsm = str(fsm).upper()
    if safety_fsm not in ("DAMP", "LIMP"):
        raise ValueError(f"unsupported safety fsm: {fsm}")

    for index in range(NUM_MOTORS):
        motor = command.motor_cmd[index]
        motor.tau = 0.0
        motor.kp = 0.0
        if safety_fsm == "LIMP":
            motor.q = 0.0 if index == 29 else POS_STOP_F
            motor.dq = VEL_STOP_F
            motor.kd = 0.0
            continue

        q = None
        if state_q is not None and index < len(state_q):
            candidate = float(state_q[index])
            if math.isfinite(candidate) and abs(candidate) < Q_SANE_RAD:
                q = candidate
        if q is None:
            candidate = float(motor.q)
            q = candidate if math.isfinite(candidate) and abs(candidate) < Q_SANE_RAD else 0.0
        motor.q = 0.0 if index == 29 else q
        motor.dq = 0.0
        motor.kd = float(damping_kd)


def _snapshot(msg: LowCmd_) -> LowCmd_:
    out = unitree_hg_msg_dds__LowCmd_()
    _copy_full(out, msg)
    out.crc = msg.crc
    return out


def _write_arm_control_status(
    path: str,
    planner: ArmHangPlanner,
    *,
    accepted: bool = True,
    message: str = "",
) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": 1,
        "timestamp": time.time(),
        "state": planner.state,
        "active": planner.active,
        "accepted": bool(accepted),
        "message": str(message),
    }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def command_runtime_flags(raw: object, fresh: bool) -> tuple[bool, bool, bool]:
    """Return ``(motion, waist_bad, global_safety)`` for one command.

    ``RL_LOWER`` closes the waist motion window but is not a global safety
    frame. DAMP/LIMP are tracked separately because they must bypass every arm
    and waist overlay in the final publisher.
    """
    try:
        if not isinstance(raw, dict) or not fresh:
            return False, False, False
        fsm = str(raw.get("fsm") or "RL_FULL").upper()
        safety = (
            bool(raw.get("estop"))
            or bool(raw.get("limp"))
            or fsm in ("DAMP", "LIMP")
        )
        if safety or fsm != "RL_FULL":
            return False, True, safety
        vel = raw.get("velocity")
        if not isinstance(vel, dict):
            vel = {}
        v_sum = (
            abs(float(vel.get("forward", 0.0)))
            + abs(float(vel.get("lateral", 0.0)))
            + abs(float(vel.get("yaw", 0.0)))
        )
        eps = MOTION_EPS_AGILE if raw.get("units") == "agile" else MOTION_EPS
        return v_sum > eps, False, False
    except Exception:
        return False, False, False


def command_flags(raw: object, fresh: bool) -> tuple[bool, bool]:
    """Compatibility wrapper returning only waist-arbiter flags."""
    motion, bad, _safety = command_runtime_flags(raw, fresh)
    return motion, bad


def read_ipc_runtime_flags(
    path: str,
    stale_s: float,
    now: float | None = None,
) -> tuple[bool, bool, bool]:
    """读运控 IPC 文件 → (motion, bad, safety)。语义对齐 adapter:
    - 文件缺失/任意损坏(非 dict JSON、字段类型错)/stale → (False, False)
      # 不能开运动窗; 已开的窗按零速计时交还 → 永远向 LOCKED 收敛
    - fresh 且 estop/limp/DAMP/LIMP/非 RL_FULL(含 RL_LOWER 抓取) → (False, True)
    - fresh 且 RL_FULL → (|v| 超阈值, False); units=agile 阈值 0.05(对齐 adapter
      WALK_THRESHOLD_NOMOVE), 归一化写者用 MOTION_EPS
    任何异常都不得逃逸(看门线程免疫)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if now is None:
            now = time.time()
        fresh = isinstance(raw, dict) and (
            now - float(raw.get("timestamp", 0.0))
        ) <= stale_s
        return command_runtime_flags(raw, fresh)
    except Exception:
        return False, False, False


def read_ipc_flags(path: str, stale_s: float, now: float | None = None) -> tuple[bool, bool]:
    """Compatibility wrapper returning only waist-arbiter flags."""
    motion, bad, _safety = read_ipc_runtime_flags(path, stale_s, now)
    return motion, bad


def _rl_frame_is_safety(command: LowCmd_) -> bool:
    """Conservatively identify adapter DAMP/LIMP frames without IPC state.

    Normal policy modes, including RL_LOWER, carry positive position gains.
    Adapter DAMP/LIMP frames set ``kp=0`` on every physical motor, and a fully
    passive frame must never receive a manipulation overlay.
    """
    try:
        gains = [float(command.motor_cmd[index].kp) for index in range(29)]
        return all(math.isfinite(gain) and gain <= 1e-6 for gain in gains)
    except (AttributeError, IndexError, TypeError, ValueError):
        return True


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

    def reset(self, now: float) -> None:
        """Drop all transition history after a global safety frame."""
        self.state = self.LOCKED
        self.alpha = 0.0
        self._zero_since = None
        self._last_t = now
        self._out_fast = False


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
        motion_command_socket: str | None = None,
        waist_blend_in_s: float = 0.3,
        waist_blend_out_s: float = 0.4,
        waist_bad_blend_out_s: float = 0.2,
        waist_handback_delay: float = 0.8,
        waist_takeover_blend_s: float = DEFAULT_WAIST_TAKEOVER_BLEND_S,
        arm_control_socket: str | None = DEFAULT_ARM_CONTROL_SOCKET,
        arm_control_status_file: str = DEFAULT_ARM_CONTROL_STATUS_FILE,
        arm_hang_move_s: float = 3.0,
        arm_hang_release_s: float = 2.5,
        arm_runtime_socket: str | None = DEFAULT_ARM_RUNTIME_SOCKET,
        arm_runtime_status_file: str = DEFAULT_ARM_RUNTIME_STATUS,
        runtime_health_file: str | None = None,
        max_tick_gap_s: float = DEFAULT_MAX_TICK_GAP_S,
        safety_damping_kd: float = DEFAULT_SAFETY_DAMPING_KD,
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
        self._sub_state = None
        self._thread: RecurrentThread | None = None
        self._fatal_error: BaseException | None = None
        self._last_tick_completed = time.monotonic()
        self._last_tick_started: float | None = None
        self._last_tick_gap_s = 0.0
        self._max_tick_gap_s = max(self._period * 2.0, float(max_tick_gap_s))
        self._tick_degraded_until = 0.0
        self._tick_watchdog_s = max(0.10, self._max_tick_gap_s * 2.0)
        self._safety_damping_kd = max(0.0, float(safety_damping_kd))
        self._health_reporter = (
            RuntimeHealthHeartbeat(
                runtime_health_file,
                component="lowcmd_merger",
                report_hz=2.0,
            )
            if runtime_health_file
            else None
        )
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
        self._motion_receiver = (
            LatestCommandReceiver(motion_command_socket)
            if motion_command_socket
            else None
        )
        self._motion_stream_identity: tuple[str, int] | None = None
        self._motion_stream_prev = False
        self._motion_stream_fresh = False
        self._motion_safety_active = False
        self._motion_safety_fsm: str | None = None
        self._motion_health_ok = False
        # (motion, bad, sample_monotonic); 元组整体替换, GIL 下读写原子。
        # 初始 ts=0 → 超龄 → (False,False), 看门线程首采样前恒锁腰。
        self._ipc_flags = (False, False, 0.0)
        self._ipc_thread: threading.Thread | None = None
        self._ipc_stop = threading.Event()
        self._last_logged_state = WaistArbiter.LOCKED
        # 腰限速器状态: {motor_idx: (last_q, last_monotonic)}
        self._waist_slew: dict[int, tuple[float, float]] = {}
        # arm_sdk 腰接管边沿。失权期间仍把缓存同步到最终 RL 输出，避免
        # 下一次接管从上一次前倾目标继续（旧实现会造成一次大幅弹射）。
        self._waist_takeover_blend_s = max(1e-3, float(waist_takeover_blend_s))
        self._waist_takeover_alpha = 0.0
        self._waist_takeover_last_t: float | None = None
        self._arm_was_active = False
        # H-key owns no DDS objects.  The pure planner runs in this merger and
        # reuses the rt/lowcmd_rl frame already received above.
        self._arm_control_path = arm_control_socket
        self._arm_control_status_file = arm_control_status_file
        self._arm_control_sock: socket.socket | None = None
        self._arm_control_lock_stream = None
        self._arm_planner = ArmHangPlanner(
            move_duration=arm_hang_move_s,
            release_duration=arm_hang_release_s,
        )
        self._arm_control_sequence: dict[str, int] = {}
        self._last_arm_planner_state = self._arm_planner.state
        self._last_output: LowCmd_ | None = None
        self._robot_modes = (0, 0)
        self._state_q: tuple[float, ...] | None = None
        self._state_t = 0.0
        self._arm_runtime = (
            ManipulationArmBroker(
                socket_path=arm_runtime_socket,
                status_file=arm_runtime_status_file,
                max_lease_s=min(0.25, self._arm_stale_s),
            )
            if arm_runtime_socket
            else None
        )

    # --- DDS 回调 ---
    def _on_rl(self, msg: LowCmd_):
        now = time.monotonic()
        with self._lock:
            self._rl = _snapshot(msg)
            self._rl_t = now
        if self._arm_runtime is not None:
            try:
                self._arm_runtime.update_policy(
                    tuple(
                        float(msg.motor_cmd[index].q)
                        for index in MANIP_ARM_JOINTS
                    ),
                    received_at=now,
                )
            except Exception:
                pass

    def _on_arm(self, msg: LowCmd_):
        with self._lock:
            self._arm = _snapshot(msg)
            self._arm_t = time.monotonic()

    def _on_state(self, msg) -> None:
        now = time.monotonic()
        try:
            full_q = tuple(
                float(msg.motor_state[index].q) for index in range(NUM_MOTORS)
            )
            if not all(math.isfinite(value) for value in full_q):
                return
            q = tuple(full_q[index] for index in MANIP_ARM_JOINTS)
            tau_est = tuple(
                float(getattr(msg.motor_state[index], "tau_est", 0.0))
                for index in MANIP_ARM_JOINTS
            )
            mode_machine = int(msg.mode_machine)
            mode_pr = int(msg.mode_pr)
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        self._robot_modes = (mode_machine, mode_pr)
        self._state_q = full_q
        self._state_t = now
        if self._arm_runtime is not None:
            try:
                self._arm_runtime.update_robot_state(
                    mode_machine=mode_machine,
                    mode_pr=mode_pr,
                    q=q,
                    tau_est=tau_est,
                    received_at=now,
                )
            except Exception:
                pass

    # --- 腰仲裁: IPC 侧线程(免疫: 任何异常不杀线程, 失败即发布安全旗标) ---
    def _ipc_loop(self):
        prev_motion = False
        while not self._ipc_stop.wait(0.05):
            try:
                motion, bad, safety = read_ipc_runtime_flags(
                    self._ipc_file,
                    self._ipc_stale_s,
                )
            except Exception:
                motion, bad, safety = False, False, False
            # 连续 2 采样(~100ms)防抖才承认运动, 单帧毛刺不开窗
            eff_motion = motion and prev_motion
            prev_motion = motion
            self._ipc_flags = (eff_motion, bad, time.monotonic())
            self._motion_stream_fresh = True
            self._motion_safety_active = safety
            self._motion_safety_fsm = "DAMP" if safety else None

    def _update_stream_flags(self, now: float) -> None:
        receiver = self._motion_receiver
        if receiver is None:
            return
        try:
            receiver.drain()
            raw, fresh = receiver.latest(now=now, stale_s=self._ipc_stale_s)
            self._motion_stream_fresh = bool(fresh and isinstance(raw, dict))
            if not fresh or not isinstance(raw, dict):
                self._motion_stream_prev = False
                # Once observed, a safety command survives stream loss. Only a
                # later fresh non-safety broker frame may release this local
                # fail-safe latch.
                self._motion_safety_active = self._motion_safety_fsm is not None
                self._motion_health_ok = False
                self._ipc_flags = (False, False, now)
                return
            self._motion_health_ok = bool(raw.get("motion_bus_health_ok", True))
            identity = (
                str(raw.get("broker_id", "")),
                int(raw.get("broker_sequence", -1)),
            )
            if identity == self._motion_stream_identity:
                return
            self._motion_stream_identity = identity
            motion, bad, safety = command_runtime_flags(raw, True)
            self._motion_safety_active = safety
            if safety:
                fsm = str(raw.get("fsm") or "DAMP").upper()
                self._motion_safety_fsm = (
                    "LIMP"
                    if bool(raw.get("limp")) or fsm == "LIMP"
                    else "DAMP"
                )
            else:
                self._motion_safety_fsm = None
            # Debounce across two distinct broker frames, not two 500 Hz ticks.
            effective_motion = motion and self._motion_stream_prev
            self._motion_stream_prev = motion
            self._ipc_flags = (effective_motion, bad, now)
        except Exception:
            self._motion_stream_prev = False
            self._motion_stream_fresh = False
            self._motion_safety_active = self._motion_safety_fsm is not None
            self._motion_health_ok = False
            self._ipc_flags = (False, False, now)

    def _report_runtime_health(
        self,
        now: float,
        *,
        rl_age: float,
        safety_active: bool = False,
    ) -> None:
        reporter = self._health_reporter
        if reporter is None:
            return
        if self._pub is None:
            healthy, reason = False, "publisher_missing"
        elif self._rl is None:
            healthy, reason = False, "rl_missing"
        elif rl_age > self._rl_stale_s:
            healthy, reason = False, "rl_stale"
        elif now < self._tick_degraded_until:
            healthy, reason = False, "tick_gap_high"
        elif self._motion_receiver is not None and not self._motion_stream_fresh:
            healthy, reason = False, "motion_stream_stale"
        else:
            healthy, reason = True, "healthy"
        reporter.record(
            healthy=healthy,
            reason=reason,
            now_mono=now,
            rl_age_ms=None if self._rl is None else max(0.0, rl_age) * 1000.0,
            motion_stream_fresh=(
                None
                if self._motion_receiver is None
                else self._motion_stream_fresh
            ),
            safety_active=bool(safety_active),
            tick_gap_ms=max(0.0, self._last_tick_gap_s) * 1000.0,
            max_tick_gap_ms=self._max_tick_gap_s * 1000.0,
            motion_health_ok=(
                None if self._motion_receiver is None else self._motion_health_ok
            ),
        )

    def _publish_final(self, out: LowCmd_) -> None:
        out.crc = self._crc.Crc(out)
        self._last_output = _snapshot(out)
        if self._pub is not None:
            self._pub.Write(out)

    def _publish_fail_safe(self, *, attempts: int = 3) -> int:
        """Best-effort DAMP writes when the worker or watchdog is failing."""
        if self._pub is None:
            return 0
        with self._lock:
            source = self._last_output or self._rl
        out = unitree_hg_msg_dds__LowCmd_()
        if source is not None:
            _copy_full(out, source)
        else:
            out.mode_machine, out.mode_pr = self._robot_modes
        state_q = (
            self._state_q
            if self._state_q is not None
            and time.monotonic() - self._state_t <= self._rl_stale_s
            else None
        )
        _force_safety_frame(
            out,
            "DAMP",
            state_q=state_q,
            damping_kd=self._safety_damping_kd,
        )
        published = 0
        for _ in range(max(1, int(attempts))):
            try:
                self._publish_final(out)
                published += 1
            except BaseException:
                continue
        return published

    def _tick_guarded(self) -> None:
        started = time.monotonic()
        if self._last_tick_started is not None:
            self._last_tick_gap_s = max(0.0, started - self._last_tick_started)
            if self._last_tick_gap_s > self._max_tick_gap_s:
                self._tick_degraded_until = max(
                    self._tick_degraded_until,
                    started + TICK_DEGRADED_HOLD_S,
                )
        self._last_tick_started = started
        try:
            self._tick()
        except BaseException as exc:
            self._fatal_error = exc
            self._publish_fail_safe()
            if self._health_reporter is not None:
                try:
                    self._health_reporter.mark_stopped(
                        f"worker_exception:{type(exc).__name__}"
                    )
                except OSError:
                    pass
            raise
        else:
            self._last_tick_completed = time.monotonic()

    def _tick(self):
        now = time.monotonic()
        self._update_stream_flags(now)
        with self._lock:
            rl = self._rl
            rl_age = now - self._rl_t
            arm = self._arm
            arm_age = now - self._arm_t if arm is not None else 1e9
        state_q = (
            self._state_q
            if self._state_q is not None
            and now - self._state_t <= self._rl_stale_s
            else None
        )

        if rl is None or rl_age > self._rl_stale_s:
            self._report_runtime_health(now, rl_age=rl_age)
            return

        out = unitree_hg_msg_dds__LowCmd_()
        _copy_full(out, rl)

        requested_safety_fsm = self._motion_safety_fsm
        safety_active = self._motion_safety_active or _rl_frame_is_safety(rl)
        supervision_inhibit = self._motion_receiver is not None and (
            not self._motion_stream_fresh or not self._motion_health_ok
        )
        timing_inhibit = now < self._tick_degraded_until
        if safety_active or supervision_inhibit or timing_inhibit:
            # DAMP/LIMP is the final whole-body command. Invalidate every
            # overlay before draining requests. A lost/unhealthy command plane
            # also inhibits overlays while preserving the adapter's zero-speed
            # balance frame.
            self._arbiter.reset(now)
            self._last_logged_state = self._arbiter.state
            self._update_waist_takeover(now, False)
            self._waist_slew.clear()
            self._arm_planner.emergency_release(None)
            if self._arm_runtime is not None:
                self._arm_runtime.activate_safety_stop()
                self._arm_runtime.drain(
                    now=now,
                    external_active=False,
                    safety_active=True,
                )
                self._arm_runtime.write_status(now=now)
            self._process_arm_control(
                now,
                rl,
                external_active=False,
                safety_active=True,
            )
            if self._arm_planner.state != self._last_arm_planner_state:
                self._last_arm_planner_state = self._arm_planner.state
                message = (
                    "base safety active"
                    if safety_active
                    else "base runtime supervision unhealthy"
                )
                self._publish_arm_control_status(message=message)
            final_safety_fsm = requested_safety_fsm
            if final_safety_fsm is None and (supervision_inhibit or timing_inhibit):
                final_safety_fsm = "DAMP"
            if final_safety_fsm is not None:
                _force_safety_frame(
                    out,
                    final_safety_fsm,
                    state_q=state_q,
                    damping_kd=self._safety_damping_kd,
                )
            self._publish_final(out)
            self._report_runtime_health(
                now,
                rl_age=rl_age,
                safety_active=True,
            )
            return

        legacy_weight = 0.0
        if arm is not None and arm_age <= self._arm_stale_s:
            raw_legacy_weight = float(arm.motor_cmd[29].q)
            if math.isfinite(raw_legacy_weight):
                legacy_weight = max(0.0, min(1.0, raw_legacy_weight))
        legacy_active = legacy_weight > self._weight_threshold

        runtime_frame = None
        runtime_arm_active = False
        if self._arm_runtime is not None:
            self._arm_runtime.drain(
                now=now,
                external_active=legacy_active,
                safety_active=False,
            )
            runtime_frame = self._arm_runtime.current_frame(
                now=now,
                external_active=legacy_active,
            )
            self._arm_runtime.write_status(now=now)
        if runtime_frame is not None:
            runtime_arm_active = True
            mode_machine, mode_pr = self._robot_modes
            arm = _runtime_arm_command(
                runtime_frame,
                mode_machine=mode_machine,
                mode_pr=mode_pr,
            )
            arm_age = 0.0
        elif self._arm_runtime is not None:
            # Candidate runtime and legacy rt/arm_sdk are mutually exclusive
            # modes. A legacy publisher appearing while the runtime socket is
            # enabled is observed as a conflict but can never take over later.
            arm = None
            arm_age = 1e9

        arm_weight = 0.0
        arms_only = False
        if arm is not None and arm_age <= self._arm_stale_s:
            raw_weight = float(arm.motor_cmd[29].q)
            if math.isfinite(raw_weight):
                arm_weight = max(0.0, min(1.0, raw_weight))
            raw_arms_only = float(arm.motor_cmd[29].dq)
            if math.isfinite(raw_arms_only):
                arms_only = raw_arms_only >= ARM_SDK_ARMS_ONLY_THRESHOLD
        use_arm = arm_weight > self._weight_threshold
        self._process_arm_control(now, rl, use_arm, safety_active=False)
        if use_arm and self._arm_planner.active:
            self._arm_planner.emergency_release(None)
            self._publish_arm_control_status(
                accepted=False,
                message="manipulation arm owner preempted integrated arm hang",
            )
        arm_owns_waist = use_arm and not arms_only
        takeover = self._update_waist_takeover(now, arm_owns_waist)

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
                    if arms_only:
                        continue  # h natural-hang: waist remains exactly RL
                    # takeover=0 时保持当前 RL 命令；takeover=1 后恢复原有
                    # 运动窗口 alpha 语义。统一混合 q/kp/kd/tau，避免腰 pitch
                    # 在 RL kd=-5 与 arm kd=+5 之间瞬时翻转。
                    rl_weight = 1.0 - takeover * arm_weight * (1.0 - alpha)
                    if rl_weight >= 1.0:
                        pass                                  # RL 持腰: out 已是 RL 值
                    elif rl_weight <= 0.0:
                        _copy_motor(out, arm, i)              # 锁腰: 旧行为
                    else:
                        _blend_motor(out, rl, arm, i, rl_weight)
                    if self._arbiter.enabled:
                        self._slew_waist(out, i, now)
                else:
                    if runtime_arm_active and arm_weight < 1.0:
                        _blend_motor(out, rl, arm, i, 1.0 - arm_weight)
                    else:
                        # Keep the established binary semantics for the explicit
                        # legacy A/B mode.
                        _copy_motor(out, arm, i)
            if arms_only:
                # Keep the future manipulation-waist takeover baseline aligned
                # with the actual RL waist while h owns only the arms.
                self._remember_waist_output(out, now)
        else:
            # 不改变 RL 输出，只把限速参考同步到真正发给电机的腰命令。
            self._remember_waist_output(out, now)

        if not use_arm:
            policy_frame = policy_frame_from_lowcmd(rl)
            hang_frame = self._arm_planner.step(now, policy_frame)
            if hang_frame is not None and hang_frame.weight > self._weight_threshold:
                for joint, source in zip(ARM_JOINTS, hang_frame.motors):
                    motor = out.motor_cmd[joint]
                    motor.mode = source.mode
                    motor.q = source.q
                    motor.dq = source.dq
                    motor.tau = source.tau
                    motor.kp = source.kp
                    motor.kd = source.kd
                    motor.reserve = source.reserve
        if self._arm_planner.state != self._last_arm_planner_state:
            self._last_arm_planner_state = self._arm_planner.state
            self._publish_arm_control_status(message="state changed")

        self._publish_final(out)
        self._report_runtime_health(now, rl_age=rl_age)

    def _publish_arm_control_status(
        self,
        *,
        accepted: bool = True,
        message: str = "",
    ) -> None:
        _write_arm_control_status(
            self._arm_control_status_file,
            self._arm_planner,
            accepted=accepted,
            message=message,
        )

    def _process_arm_control(
        self,
        now: float,
        rl: LowCmd_,
        external_active: bool,
        *,
        safety_active: bool,
    ) -> None:
        sock = self._arm_control_sock
        if sock is None:
            return
        while True:
            try:
                data, address = sock.recvfrom(4096)
            except BlockingIOError:
                return
            except OSError:
                return
            accepted, message = False, "invalid request"
            request_id = ""
            source = ""
            sequence = -1
            try:
                payload = json.loads(data.decode("utf-8"))
                request_id = str(payload.get("request_id", ""))
                source = str(payload.get("source", ""))
                sequence = int(payload.get("sequence", -1))
                if not source or sequence <= self._arm_control_sequence.get(source, -1):
                    raise ValueError("stale sequence")
                self._arm_control_sequence[source] = sequence
                command = str(payload.get("command", "")).lower()
                policy = policy_frame_from_lowcmd(rl)
                if safety_active and command in ("release", "emergency_release"):
                    self._arm_planner.emergency_release(None)
                    accepted, message = True, "arm overlay released; base safety active"
                elif safety_active:
                    message = "base DAMP/LIMP safety is active"
                elif command == "toggle":
                    if self._arm_planner.active:
                        accepted = self._arm_planner.release(policy, now)
                        message = "aligning arms back to policy"
                    elif external_active:
                        message = "external arm_sdk owner is active"
                    else:
                        start_msg = self._last_output or rl
                        accepted = self._arm_planner.activate(
                            policy_frame_from_lowcmd(start_msg),
                            now,
                        )
                        message = "moving arms to natural hang"
                elif command == "release":
                    if self._arm_planner.active:
                        accepted = self._arm_planner.release(policy, now)
                        message = "aligning arms back to policy"
                    else:
                        accepted, message = True, "already released"
                elif command == "emergency_release":
                    self._arm_planner.emergency_release(policy)
                    accepted, message = True, "arm overlay released"
                else:
                    message = f"unknown command: {command}"
            except Exception as exc:
                message = f"request rejected: {exc}"
            self._publish_arm_control_status(accepted=accepted, message=message)
            if address:
                response = {
                    "schema_version": 1,
                    "request_id": request_id,
                    "source": source,
                    "sequence": sequence,
                    "accepted": accepted,
                    "message": message,
                    "state": self._arm_planner.state,
                    "timestamp": time.time(),
                }
                try:
                    sock.sendto(
                        json.dumps(response, separators=(",", ":")).encode(),
                        address,
                    )
                except OSError:
                    pass

    def _bind_arm_control(self) -> None:
        if not self._arm_control_path or self._arm_control_sock is not None:
            return
        path = Path(self._arm_control_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f"{path.name}.lock")
        lock_stream = lock_path.open("a+")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise RuntimeError(
                f"arm control endpoint is already active: {path}"
            ) from exc
        try:
            if path.exists():
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                probe.setblocking(False)
                try:
                    probe.sendto(b'{"command":"probe"}', str(path))
                except OSError as exc:
                    if exc.errno not in {
                        errno.ENOENT,
                        errno.ECONNREFUSED,
                        errno.ECONNRESET,
                    }:
                        raise
                else:
                    raise RuntimeError(f"another process owns arm control: {path}")
                finally:
                    probe.close()
                path.unlink(missing_ok=True)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.bind(str(path))
            os.chmod(path, 0o660)
            sock.setblocking(False)
        except BaseException:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
            raise
        self._arm_control_lock_stream = lock_stream
        self._arm_control_sock = sock
        self._publish_arm_control_status(message="ready")

    def _close_local_endpoints(self) -> None:
        if self._arm_control_sock is not None:
            self._arm_control_sock.close()
            self._arm_control_sock = None
        if self._arm_control_lock_stream is not None:
            if self._arm_control_path:
                try:
                    Path(self._arm_control_path).unlink()
                except FileNotFoundError:
                    pass
            fcntl.flock(
                self._arm_control_lock_stream.fileno(),
                fcntl.LOCK_UN,
            )
            self._arm_control_lock_stream.close()
            self._arm_control_lock_stream = None
        if self._arm_runtime is not None:
            self._arm_runtime.close()
        if self._motion_receiver is not None:
            self._motion_receiver.close()

    def _update_waist_takeover(self, now: float, use_arm: bool) -> float:
        """返回 0..1 的 arm 腰接管进度；失权立即复位但不延迟回到 RL。"""
        if not use_arm:
            self._arm_was_active = False
            self._waist_takeover_alpha = 0.0
            self._waist_takeover_last_t = None
            return 0.0
        if not self._arm_was_active:
            self._arm_was_active = True
            self._waist_takeover_alpha = 0.0
            self._waist_takeover_last_t = now
            return 0.0
        last_t = self._waist_takeover_last_t
        dt = 0.0 if last_t is None else min(ARBITER_MAX_DT, max(0.0, now - last_t))
        self._waist_takeover_last_t = now
        self._waist_takeover_alpha = min(
            1.0,
            self._waist_takeover_alpha + dt / self._waist_takeover_blend_s,
        )
        return self._waist_takeover_alpha

    def _remember_waist_output(self, out: LowCmd_, now: float) -> None:
        for idx in range(WAIST_LO, WAIST_HI):
            q = float(out.motor_cmd[idx].q)
            if abs(q) < Q_SANE_RAD:
                self._waist_slew[idx] = (q, now)
            else:
                self._waist_slew.pop(idx, None)

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
        previous_signal_handlers = {}

        def request_signal_shutdown(signum, _frame) -> None:
            self._publish_fail_safe(attempts=5)
            raise KeyboardInterrupt(f"received signal {signum}")

        try:
            # Claim every local endpoint before creating a DDS participant.
            # Any overlapping candidate instance therefore fails without
            # replacing the running instance's socket.
            if self._arbiter.enabled and self._motion_receiver is not None:
                self._motion_receiver.bind()
            if self._arm_runtime is not None:
                self._arm_runtime.bind()
            self._bind_arm_control()

            ChannelFactoryInitialize(0, self._iface)
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._pub.Init()
            for signum in (signal.SIGTERM, signal.SIGHUP):
                previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_signal_shutdown)
            self._sub_rl = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
            self._sub_rl.Init(self._on_rl, 10)
            self._sub_arm = ChannelSubscriber("rt/arm_sdk", LowCmd_)
            self._sub_arm.Init(self._on_arm, 10)
            if self._arm_runtime is not None:
                self._sub_state = ChannelSubscriber("rt/lowstate", LowState_)
                self._sub_state.Init(self._on_state, 1)
            if self._arbiter.enabled and self._motion_receiver is None:
                self._ipc_thread = threading.Thread(
                    target=self._ipc_loop,
                    daemon=True,
                    name="waist_ipc_watch",
                )
                self._ipc_thread.start()
            self._thread = RecurrentThread(
                interval=self._period,
                target=self._tick_guarded,
                name="merge_lowcmd",
            )
            self._fatal_error = None
            self._last_tick_completed = time.monotonic()
            self._last_tick_started = None
            self._thread.Start()
        except BaseException:
            self._ipc_stop.set()
            self._publish_fail_safe(attempts=5)
            for signum, previous in previous_signal_handlers.items():
                signal.signal(signum, previous)
            self._close_local_endpoints()
            raise
        print(
            f"merge_lowcmd_arm_sdk: iface={self._iface} 合成→rt/lowcmd  "
            f"订阅 rt/lowcmd_rl + rt/arm_sdk  周期={self._period*1000:.1f} ms"
        )
        if self._arm_control_sock is not None:
            print(
                f"[ARM_HANG] integrated planner ON: {self._arm_control_path}; "
                "no extra DDS participant"
            )
        if self._arm_runtime is not None:
            print(
                f"[ARM_RUNTIME] manipulation IPC ON: "
                f"{self._arm_runtime.socket_path}; merger owns DDS/CRC"
            )
        if self._arbiter.enabled:
            command_source = (
                f"stream={self._motion_receiver.socket_path}"
                if self._motion_receiver is not None
                else f"legacy_ipc={self._ipc_file}"
            )
            print(f"[WAIST] 运动窗口腰仲裁 ON: {command_source} "
                  f"(stale {self._ipc_stale_s:.2f}s)  blend_in={self._arbiter.blend_in_s:.2f}s "
                  f"blend_out={self._arbiter.blend_out_s:.2f}s "
                  f"(bad→{self._arbiter.bad_blend_out_s:.2f}s) "
                  f"handback={self._arbiter.handback_delay:.2f}s  "
                  f"takeover={self._waist_takeover_blend_s:.2f}s  "
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
                time.sleep(0.05)
                if self._fatal_error is not None:
                    raise RuntimeError("lowcmd merger worker failed") from self._fatal_error
                stalled_s = time.monotonic() - self._last_tick_completed
                if stalled_s > self._tick_watchdog_s:
                    raise RuntimeError(
                        f"lowcmd merger worker stalled for {stalled_s:.3f}s"
                    )
        except KeyboardInterrupt:
            print("\n退出。")
        finally:
            self._ipc_stop.set()
            self._publish_fail_safe(attempts=5)
            if self._thread is not None:
                self._thread.Wait(timeout=1.0)
            if self._health_reporter is not None:
                try:
                    reason = (
                        "merger_worker_failed"
                        if self._fatal_error is not None
                        else "merger_stopped"
                    )
                    self._health_reporter.mark_stopped(reason)
                except OSError:
                    pass
            for signum, previous in previous_signal_handlers.items():
                signal.signal(signum, previous)
            self._close_local_endpoints()


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
    p.add_argument(
        "--motion-command-socket",
        default=DEFAULT_MERGER_COMMAND_SOCKET,
        help="broker 最新命令流；空字符串回退到 legacy --waist-ipc-file",
    )
    p.add_argument("--waist-blend-in-s", type=float, default=0.3, help="腰 arm→RL 过渡时长")
    p.add_argument("--waist-blend-out-s", type=float, default=0.4, help="腰 RL→arm 过渡时长(零速交还)")
    p.add_argument("--waist-bad-blend-out-s", type=float, default=0.2,
                   help="estop/抓取触发的快速交还时长(建议 ≤0.3, RL_LOWER→伸臂裕量 ~1.0s)")
    p.add_argument("--waist-handback-delay", type=float, default=0.8,
                   help="零速持续该时长后交还腰(GrootMover 停后写 0.4s 零速, 需 >0.4)")
    p.add_argument("--waist-takeover-blend-s", type=float,
                   default=DEFAULT_WAIST_TAKEOVER_BLEND_S,
                   help="arm_sdk 首次接管腰时从当前 RL 命令平滑混入的时长")
    p.add_argument("--arm-control-socket", default=DEFAULT_ARM_CONTROL_SOCKET,
                   help="H 手臂状态机 Unix socket；空字符串关闭")
    p.add_argument("--arm-control-status-file", default=DEFAULT_ARM_CONTROL_STATUS_FILE)
    p.add_argument("--arm-hang-move-s", type=float, default=3.0)
    p.add_argument("--arm-hang-release-s", type=float, default=2.5)
    p.add_argument(
        "--arm-runtime-socket",
        default=DEFAULT_ARM_RUNTIME_SOCKET,
        help="操控 q/state Unix socket；空字符串关闭并仅保留 legacy rt/arm_sdk",
    )
    p.add_argument(
        "--arm-runtime-status-file",
        default=DEFAULT_ARM_RUNTIME_STATUS,
    )
    p.add_argument(
        "--runtime-health-file",
        default=DEFAULT_MERGER_HEALTH_FILE,
        help="merger liveness/consumer heartbeat used by the motion health gate",
    )
    p.add_argument(
        "--max-tick-gap-ms",
        type=float,
        default=DEFAULT_MAX_TICK_GAP_S * 1000.0,
        help="publish DAMP and mark merger unhealthy after this tick gap",
    )
    p.add_argument(
        "--safety-damping-kd",
        type=float,
        default=DEFAULT_SAFETY_DAMPING_KD,
    )
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
        motion_command_socket=args.motion_command_socket or None,
        waist_blend_in_s=args.waist_blend_in_s,
        waist_blend_out_s=args.waist_blend_out_s,
        waist_bad_blend_out_s=args.waist_bad_blend_out_s,
        waist_handback_delay=args.waist_handback_delay,
        waist_takeover_blend_s=args.waist_takeover_blend_s,
        arm_control_socket=args.arm_control_socket or None,
        arm_control_status_file=args.arm_control_status_file,
        arm_hang_move_s=args.arm_hang_move_s,
        arm_hang_release_s=args.arm_hang_release_s,
        arm_runtime_socket=args.arm_runtime_socket or None,
        arm_runtime_status_file=args.arm_runtime_status_file,
        runtime_health_file=args.runtime_health_file or None,
        max_tick_gap_s=args.max_tick_gap_ms / 1000.0,
        safety_damping_kd=args.safety_damping_kd,
    ).run()


if __name__ == "__main__":
    main()
