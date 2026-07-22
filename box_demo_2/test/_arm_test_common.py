#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_all_zero_arms / test_waist_target 共享的交权与清理逻辑。"""

from __future__ import annotations

import threading
import time

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from arm_natural_hang import (
    DT,
    ArmNaturalHangKeeper,
    publish_arm_sdk,
    read_arm_q_from_state,
    read_policy_q,
)
from dual_arm_target_reach import ARM_JOINTS

try:
    from groot_mover import write_command
except ImportError:
    write_command = None

# merge_lowcmd 默认 arm_stale_s=0.25；退出后需等 stale 才真正回到纯 policy
ARM_SDK_STALE_S = 0.35
RL_LOWER_SETTLE_S = 0.5
POST_RL_LOWER_SETTLE_S = 0.2
WAIST_ALIGN_EPS_RAD = 0.06
WAIST_ALIGN_TIMEOUT_S = 6.0
WAIST_ALIGN_MIN_WAIT_S = 0.8
WAIST_RESTORE_EPS_RAD = 0.03
WAIST_RESTORE_DQ_RAD_S = 0.15
WAIST_RESTORE_HOLD_S = 0.25


def _waist_max_err(q_phys: list[float], q_policy: list[float]) -> float:
    return max(abs(q_phys[14 + i] - q_policy[14 + i]) for i in range(3))


def read_policy_q_snapshot(timeout: float = 2.0) -> list[float]:
    """读取一帧 rt/lowcmd_rl 上半身命令；接管前必须用命令而非实测角对齐。"""
    rl_cmd: list[LowCmd_ | None] = [None]
    ready = threading.Event()

    def _on_rl(msg: LowCmd_) -> None:
        rl_cmd[0] = msg
        ready.set()

    sub = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
    sub.Init(_on_rl, 10)
    if not ready.wait(timeout=max(0.1, float(timeout))) or rl_cmd[0] is None:
        raise RuntimeError("未收到 rt/lowcmd_rl，拒绝在未知腰命令下接管")
    return read_policy_q(rl_cmd[0])


def wait_waist_pose_restore(
    arm_keeper: ArmNaturalHangKeeper,
    q_reference: list[float],
    *,
    timeout: float = WAIST_ALIGN_TIMEOUT_S,
    eps: float = WAIST_RESTORE_EPS_RAD,
    dq_eps: float = WAIST_RESTORE_DQ_RAD_S,
    hold_s: float = WAIST_RESTORE_HOLD_S,
) -> bool:
    """等待物理腰回到测试前姿态并稳定；不能用 phys≈policy 代替姿态恢复。"""
    arm_keeper._ensure_dds()
    if arm_keeper.low_state is None:
        print("[恢复] 警告: low_state 未就绪，无法验证测试前腰姿态")
        return False

    print(f"[恢复] 等腰回到测试前姿态 (误差≤{eps:.3f}rad)...")
    deadline = time.time() + max(0.1, float(timeout))
    stable_since = None
    last_err = last_dq = None
    while time.time() < deadline:
        state = arm_keeper.low_state
        if state is None:
            time.sleep(DT)
            continue
        q_phys = [float(state.motor_state[int(j)].q) for j in ARM_JOINTS]
        dq_phys = [float(state.motor_state[int(j)].dq) for j in ARM_JOINTS]
        last_err = max(abs(q_phys[14 + i] - q_reference[14 + i]) for i in range(3))
        last_dq = max(abs(dq_phys[14 + i]) for i in range(3))
        if last_err <= eps and last_dq <= dq_eps:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= hold_s:
                print(
                    f"[恢复] 腰已回到测试前姿态 "
                    f"(err={last_err:.3f}rad, dq={last_dq:.3f}rad/s)"
                )
                return True
        else:
            stable_since = None
        time.sleep(DT)

    print(
        f"[恢复] 警告: 腰未在超时内回到测试前姿态 "
        f"(err={last_err}, dq={last_dq})"
    )
    return False


def wait_waist_policy_settle(
    arm_keeper: ArmNaturalHangKeeper,
    *,
    timeout: float = WAIST_ALIGN_TIMEOUT_S,
    eps: float = WAIST_ALIGN_EPS_RAD,
    min_wait: float = WAIST_ALIGN_MIN_WAIT_S,
    q_reference: list[float] | None = None,
    reference_eps: float = WAIST_RESTORE_EPS_RAD,
) -> None:
    """RL_FULL 下等命令对齐；可同时要求物理腰仍接近测试前姿态。"""
    arm_keeper._ensure_dds()
    if arm_keeper.low_state is None:
        time.sleep(min_wait)
        return

    rl_cmd: list[LowCmd_ | None] = [None]
    rl_ready = threading.Event()

    def _on_rl(msg: LowCmd_) -> None:
        rl_cmd[0] = msg
        if not rl_ready.is_set():
            rl_ready.set()

    sub = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
    sub.Init(_on_rl, 10)

    print(f"[准备] 等 policy 收敛腰角 (≤{timeout:.0f}s)...")
    t0 = time.time()
    deadline = t0 + timeout
    last_err = None

    while time.time() < deadline:
        if not rl_ready.wait(timeout=0.05):
            continue
        rl_ready.clear()
        if arm_keeper.low_state is None or rl_cmd[0] is None:
            continue
        q_phys = [
            float(arm_keeper.low_state.motor_state[int(j)].q) for j in ARM_JOINTS
        ]
        q_pol = read_policy_q(rl_cmd[0])
        err = _waist_max_err(q_phys, q_pol)
        last_err = err
        reference_ok = (
            q_reference is None
            or _waist_max_err(q_phys, q_reference) < reference_eps
        )
        if time.time() - t0 >= min_wait and err < eps and reference_ok:
            print(f"[准备] 腰角已对齐 policy (err={err:.3f} rad)")
            return
        time.sleep(DT)

    if last_err is not None:
        print(f"[准备] 警告: 腰角未完全对齐 policy (err={last_err:.3f} rad)，继续")
    else:
        print(f"[准备] 警告: 未收到 rt/lowcmd_rl，跳过腰角对齐")


def handoff_rl_lower_then_move(
    mover,
    arm_keeper: ArmNaturalHangKeeper,
    runner,
    *,
    move_label: str,
    rl_lower_settle_s: float = RL_LOWER_SETTLE_S,
    post_settle_s: float = POST_RL_LOWER_SETTLE_S,
) -> None:
    """RL_FULL 下 never 发 arm_sdk；写 RL_LOWER 后立即 50Hz 锁快照位姿。"""
    if write_command is None:
        raise RuntimeError("groot_mover.write_command 不可用")

    q_hold = read_arm_q_from_state(arm_keeper)
    print("\n[执行] 切换 RL_LOWER + arm_sdk 接管")
    print("  （不在 RL_FULL 下发 arm_sdk，避免与 policy 抢上半身）")
    write_command(mover.cmd_file, "RL_LOWER", height=mover._height)
    arm_keeper._burst_publish(q_hold)
    runner.start_preflight(q_hold)
    time.sleep(rl_lower_settle_s)
    if post_settle_s > 0:
        time.sleep(post_settle_s)
    runner.begin_move()
    print(f"\n[执行] {move_label}")


def finalize_arm_sdk_off(arm_keeper: ArmNaturalHangKeeper, q_hold: list[float]) -> None:
    """进程退出前确保 merge 不再消费 arm_sdk。"""
    arm_keeper._ensure_dds()
    if arm_keeper._pub is None or arm_keeper.low_state is None:
        return
    for _ in range(20):
        publish_arm_sdk(
            arm_keeper._pub, arm_keeper.low_state, q_hold, sdk_weight=0.0,
        )
        time.sleep(DT)
    time.sleep(ARM_SDK_STALE_S)


def restore_rl_full_pose(
    arm_keeper: ArmNaturalHangKeeper,
    mover,
    *,
    settle_timeout: float = WAIST_ALIGN_TIMEOUT_S,
) -> None:
    """退出：RL_FULL 后等 policy 把腰收回，再关 arm_sdk。"""
    if mover is not None:
        print("[退出] 切换 RL_FULL...")
        try:
            mover.initialize()
            time.sleep(0.3)
        except Exception as exc:
            print(f"  警告: mover.initialize 失败: {exc}")
    wait_waist_policy_settle(arm_keeper, timeout=settle_timeout)


def lerp_waist_only(q_start: list[float], q_target: list[float], ratio: float) -> list[float]:
    """仅插值腰 3 关节，手臂保持 q_start（waist_target 专用）。"""
    q = list(q_start)
    r = max(0.0, min(1.0, float(ratio)))
    for i in range(14, 17):
        q[i] = q_start[i] * (1.0 - r) + q_target[i] * r
    return q


def read_ipc_height(cmd_file: str, default: float) -> float:
    """读 /tmp/robojudo_ext_cmd.json 里的 height（运控 z/x 下蹲后的值）。"""
    import json

    try:
        with open(cmd_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "height" in raw:
            return float(raw["height"])
    except Exception:
        pass
    return float(default)


def sync_mover_height_from_ipc(mover, *, default: float | None = None) -> float:
    """把 GrootMover 内部 height 设为 IPC 当前值，避免 initialize() 抬回 stand。"""
    from groot_mover import MAX_HEIGHT, MIN_HEIGHT, STAND_HEIGHT

    fallback = STAND_HEIGHT if default is None else float(default)
    raw = read_ipc_height(mover.cmd_file, fallback)
    h = max(MIN_HEIGHT, min(MAX_HEIGHT, raw))
    mover._height = h
    if abs(h - raw) > 1e-6:
        print(
            f"[高度] IPC 原始={raw:.2f}m → 钳位后={h:.2f}m "
            f"(groot_mover MIN_HEIGHT={MIN_HEIGHT:.2f})"
        )
    else:
        print(f"[高度] IPC 当前={h:.2f}m（读 {mover.cmd_file}）")
    return h
