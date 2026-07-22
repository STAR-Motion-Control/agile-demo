#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双臂 + 腰 17 关节目标测试（点 A / 点 B 往返）。

点 A / 点 B：腰目标均为 [0, 0, 0.5] rad；点 B 手臂由点 A FK→内收→IK 求得。
各阶段（原位置→A、A→B、B→A、A→交还 policy）腰与手臂均同步插值。

流程:
  1. GrootMover 初始化 RL_FULL
  2. RL_FULL 下平滑插值 → 点 A，50Hz 保持
  3. Enter → 点 B；Enter → 回点 A；Enter → 交还 policy 默认位姿

用法（机器人上，需 merge_lowcmd_arm_sdk + groot adapter 已运行）:
  cd ~/zihou/box_demo_2
  python test/test_all_zero_arms.py --iface enP8p1s0
  python test/test_all_zero_arms.py --y-inward 0.05 --duration 3
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

from arm_natural_hang import (  # noqa: E402
    ArmNaturalHangKeeper,
    DT,
    DEFAULT_MOVE_DURATION,
    lerp_q,
    read_arm_q_from_state,
    smooth_ratio,
)
from dual_arm_target_reach import ARM_JOINTS  # noqa: E402
from arm_kinematics import (  # noqa: E402
    ArmKinematics,
    FULL6D_COL_MIN_DOT,
    FULL6D_ELBOW_MAX_RAD,
    FULL6D_POS_TOL_M,
    FULL6D_SHOULDER_YAW_REF,
    FULL6D_SHOULDER_YAW_REG_WEIGHT,
    FULL6D_TIME_LIMIT_SEC,
    full6d_ik_pass,
)
from _arm_test_common import (  # noqa: E402
    read_policy_q_snapshot,
    sync_mover_height_from_ipc,
    wait_waist_pose_restore,
    wait_waist_policy_settle,
)

try:
    from groot_mover import GrootMover, STAND_HEIGHT, WARMUP_TIME as GROOT_DEFAULT_WARMUP_TIME
except ImportError:
    GrootMover = None
    STAND_HEIGHT = 0.76
    GROOT_DEFAULT_WARMUP_TIME = 0.0

JOINT_LABELS = [
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw", "L_Elbow",
    "L_WristRoll", "L_WristPitch", "L_WristYaw",
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw", "R_Elbow",
    "R_WristRoll", "R_WristPitch", "R_WristYaw",
    "WaistYaw", "WaistRoll", "WaistPitch",
]

# ARM_JOINTS 顺序：左7 + 右7 + 腰3（rad）
ALL_ZERO_TARGET_Q = [
    -1.1, 0.4, 0.0, 1.3, 0.0, 0.0, 0.0,
    -1.1, -0.4, 0.0, 1.3, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.5,
]
POINT_A_Q = ALL_ZERO_TARGET_Q
WAIST_TARGET_Q = [0.0, 0.0, 0.5]
DEFAULT_Y_INWARD_M = 0.05
assert len(ALL_ZERO_TARGET_Q) == len(ARM_JOINTS)
assert ALL_ZERO_TARGET_Q[14:17] == WAIST_TARGET_Q


def _fmt_xyz(p) -> str:
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m"


def compute_point_b_from_a(
    q_a: list[float],
    *,
    y_inward: float = DEFAULT_Y_INWARD_M,
    ik_err_limit: float = 0.100,
) -> tuple[list[float], dict]:
    """由点 A 关节角 FK → 左右 EE 内收 y_inward → full6d IK 得点 B（腰与 A 相同 [0,0,0.5]）。"""
    ik = ArmKinematics()
    left_qa = list(q_a[0:7])
    right_qa = list(q_a[7:14])
    waist = list(WAIST_TARGET_Q)

    left_pos, left_R = ik.forward_kinematics(left_qa, left=True)
    right_pos, right_R = ik.forward_kinematics(right_qa, left=False)

    left_b = left_pos.copy()
    right_b = right_pos.copy()
    left_b[1] -= float(y_inward)
    right_b[1] += float(y_inward)

    left_qb, left_err, left_meta = ik.inverse_kinematics_scipy_full6d(
        left_b, left=True, q0=left_qa, R_des=left_R,
        time_limit_sec=FULL6D_TIME_LIMIT_SEC,
        col_min_dot=FULL6D_COL_MIN_DOT,
        pos_tol=FULL6D_POS_TOL_M,
        shoulder_yaw_reg_weight=FULL6D_SHOULDER_YAW_REG_WEIGHT,
        shoulder_yaw_ref=FULL6D_SHOULDER_YAW_REF,
        elbow_max_rad=FULL6D_ELBOW_MAX_RAD,
    )
    right_qb, right_err, right_meta = ik.inverse_kinematics_scipy_full6d(
        right_b, left=False, q0=right_qa, R_des=right_R,
        time_limit_sec=FULL6D_TIME_LIMIT_SEC,
        col_min_dot=FULL6D_COL_MIN_DOT,
        pos_tol=FULL6D_POS_TOL_M,
        shoulder_yaw_reg_weight=FULL6D_SHOULDER_YAW_REG_WEIGHT,
        shoulder_yaw_ref=FULL6D_SHOULDER_YAW_REF,
        elbow_max_rad=FULL6D_ELBOW_MAX_RAD,
    )

    info = {
        "left_a": left_pos,
        "right_a": right_pos,
        "left_b": left_b,
        "right_b": right_b,
        "left_err_mm": left_err * 1000.0,
        "right_err_mm": right_err * 1000.0,
        "left_meta": left_meta,
        "right_meta": right_meta,
    }

    if (left_err > ik_err_limit or right_err > ik_err_limit
            or not full6d_ik_pass(left_err, left_meta)
            or not full6d_ik_pass(right_err, right_meta)):
        raise ValueError(
            f"点 B IK 不合格 (左:{left_err*1000:.1f}mm, 右:{right_err*1000:.1f}mm)"
        )

    q_b = list(left_qb) + list(right_qb) + waist
    return q_b, info


def _print_ab_summary(q_a: list[float], q_b: list[float], info: dict, *, y_inward: float) -> None:
    print("\n点 A / 点 B（torso 系）:")
    print(f"  左 EE  A: {_fmt_xyz(info['left_a'])}  →  B: {_fmt_xyz(info['left_b'])}  "
          f"(y -{y_inward:.2f}m)")
    print(f"  右 EE  A: {_fmt_xyz(info['right_a'])}  →  B: {_fmt_xyz(info['right_b'])}  "
          f"(y +{y_inward:.2f}m)")
    print(f"  点 B IK 误差: 左 {info['left_err_mm']:.1f}mm  右 {info['right_err_mm']:.1f}mm")
    print(f"  腰目标 (rad): yaw={WAIST_TARGET_Q[0]:+.4f}  roll={WAIST_TARGET_Q[1]:+.4f}  "
          f"pitch={WAIST_TARGET_Q[2]:+.4f}")
    print("\n点 A 关节角 (rad / deg):")
    for i, (name, val) in enumerate(zip(JOINT_LABELS, q_a)):
        print(f"  [{i:2d}] {name:16s}  {val:+.4f} rad  ({val * 57.2958:+.1f}°)")
    print("\n点 B 关节角 (rad / deg):")
    for i, (name, val) in enumerate(zip(JOINT_LABELS, q_b)):
        print(f"  [{i:2d}] {name:16s}  {val:+.4f} rad  ({val * 57.2958:+.1f}°)")


class TargetHoldRunner:
    """50Hz 保持固定 q_target（同 ArmNaturalHangKeeper keepalive，目标可自定义）。"""

    def __init__(self, keeper: ArmNaturalHangKeeper, q_target: list[float]):
        self._keeper = keeper
        self._q_target = list(q_target)
        self._running = False
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def q_target(self) -> list[float]:
        return list(self._q_target)

    def _loop(self) -> None:
        while self._running:
            t0 = time.time()
            self._keeper.publish_once(self._q_target)
            sleep_s = DT - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def start(self) -> None:
        if self.is_running():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="all_zero_hold")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None


def transition_to_target(
    keeper: ArmNaturalHangKeeper,
    holder: TargetHoldRunner | None,
    q_target: list[float],
    duration: float,
    *,
    label: str,
    q_start_override: list[float] | None = None,
) -> TargetHoldRunner:
    """停止当前保持 → 17 关节插值到 q_target → 再 50Hz 保持。"""
    if holder is not None:
        # 轨迹必须从上一条“命令”连续，而不是从带负载静差的物理角重新起步。
        # 否则 q_cmd=物理角会瞬间撤掉腰部的 PD 支撑力矩。
        q_start = holder.q_target
        holder.stop()
    elif q_start_override is not None:
        q_start = list(q_start_override)
    else:
        q_start = None
    keeper._ensure_dds()
    if keeper.low_state is None:
        raise RuntimeError("low_state 未就绪")

    q_goal = list(q_target)
    if q_start is None:
        q_start = read_arm_q_from_state(keeper)
    duration = max(float(duration), 0.1)
    print(f"\n[执行] {label}（{duration:.1f}s 插值，腰+双臂）")

    t = 0.0
    while t < duration:
        q = lerp_q(q_start, q_goal, smooth_ratio(t / duration))
        keeper.publish_once(q)
        time.sleep(DT)
        t += DT
    keeper.publish_once(q_goal)

    new_holder = TargetHoldRunner(keeper, q_goal)
    new_holder.start()
    print(f"[执行] 已到达并保持: {label}")
    return new_holder


def main() -> None:
    parser = argparse.ArgumentParser(description="双臂点 A/B 往返测试（box_demo move_and_start 同款进入/退出）")
    parser.add_argument("--iface", default=os.environ.get("UNITREE_DDS_INTERFACE", "enP8p1s0"))
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_MOVE_DURATION,
        help="各段 RL_FULL 插值时长(s)，默认与 arm_natural_hang 一致",
    )
    parser.add_argument(
        "--y-inward", type=float, default=DEFAULT_Y_INWARD_M,
        help="点 B 相对点 A：左右 EE 沿 torso +Y 内收距离(m)，默认 0.05",
    )
    parser.add_argument("--warmup-time", type=float, default=GROOT_DEFAULT_WARMUP_TIME)
    parser.add_argument("--skip-mover", action="store_true",
                        help="不初始化 GrootMover（仅测 arm_sdk，退出时不切 RL_FULL）")
    parser.add_argument(
        "--stand-height", action="store_true",
        help="initialize 时用默认站高 %.2fm，不保留运控 IPC 当前高度" % STAND_HEIGHT,
    )
    args = parser.parse_args()

    if GrootMover is None and not args.skip_mover:
        raise SystemExit("groot_mover 不可用；请加 --skip-mover 或检查环境")

    print("=" * 60)
    print("  双臂点 A/B 测试 (test_all_zero_arms.py)")
    print(f"  iface={args.iface}  duration={args.duration}s  y_inward={args.y_inward:.2f}m")
    print(f"  腰目标 (A/B): {WAIST_TARGET_Q}")
    print("  交权逻辑: box_demo_main move_and_start / _graceful_shutdown")
    print("=" * 60)

    ChannelFactoryInitialize(0, args.iface)

    mover = None
    if not args.skip_mover:
        mover = GrootMover(warmup_time=args.warmup_time)
        if args.stand_height:
            print(f"[初始化] 使用默认站高 {STAND_HEIGHT:.2f}m")
        else:
            h = sync_mover_height_from_ipc(mover)
            print(
                f"[初始化] 保留运控 IPC 高度 {h:.2f}m"
                f"（避免 initialize 写回默认 stand {STAND_HEIGHT:.2f}m）"
            )
        mover.initialize()
        time.sleep(0.5)

    arm_keeper = ArmNaturalHangKeeper()
    arm_keeper.sync_waist_from_robot()

    q_current = read_arm_q_from_state(arm_keeper)
    # 保存“测试前物理姿态 + 测试前 RL 命令”。接管首帧必须延续 RL 命令，
    # 退出则先让物理腰回到测试前姿态，再交还实时 policy。
    q_pretest_phys = list(q_current)
    try:
        q_pretest_policy = read_policy_q_snapshot()
    except RuntimeError as exc:
        raise SystemExit(f"接管前检查失败: {exc}") from exc
    q_a = list(POINT_A_Q)
    try:
        q_b, ab_info = compute_point_b_from_a(q_a, y_inward=args.y_inward)
    except ValueError as exc:
        raise SystemExit(f"点 B 求解失败: {exc}") from exc

    print("\n当前关节角 (rad):")
    for i, name in enumerate(JOINT_LABELS):
        print(f"  [{i:2d}] {name:16s}  {q_current[i]:+.4f}")
    _print_ab_summary(q_a, q_b, ab_info, y_inward=args.y_inward)

    holder = transition_to_target(
        arm_keeper, None, q_a, args.duration,
        label="运控上半身 → 点 A",
        q_start_override=q_pretest_policy,
    )
    print(f"\n[完成] 已在点 A 保持。")

    shutdown_done = False

    def _graceful_shutdown(*, interrupted: bool = False) -> None:
        nonlocal shutdown_done, holder
        if shutdown_done:
            return
        shutdown_done = True
        if interrupted:
            print("\n[退出] Ctrl+C — 平滑交还 policy...")
        else:
            print("\n[退出] 平滑交还 policy...")
        # 先让 RL_FULL 更新实时 policy，但 holder 继续锁住当前命令，避免空窗。
        if mover is not None:
            try:
                mover.stop()
            except Exception:
                pass
            print("[退出] 切换 RL_FULL...")
            try:
                if not args.stand_height:
                    sync_mover_height_from_ipc(mover)
                mover.initialize()
                time.sleep(0.3)
            except Exception as exc:
                print(f"  警告: mover.initialize 失败: {exc}")
        # 不能直接追随状态依赖的实时 policy：它会在前倾状态形成局部平衡。
        # 先用测试前 RL 命令把物理腰拉回测试前姿态，并让 policy 历史刷新。
        holder = transition_to_target(
            arm_keeper,
            holder,
            q_pretest_policy,
            args.duration,
            label="点 A → 测试前运控位姿",
        )
        wait_waist_pose_restore(arm_keeper, q_pretest_phys)

        q_handoff = holder.q_target
        holder.stop()
        arm_keeper.release_to_policy(q_start=q_handoff)
        # 同时验证 policy 对齐和绝对姿态，不能在残余前倾处误判完成。
        wait_waist_policy_settle(arm_keeper, q_reference=q_pretest_phys)
        print("测试结束。")

    try:
        input("\n按 Enter 执行点 B...")
        holder = transition_to_target(
            arm_keeper, holder, q_b, args.duration, label="点 A → 点 B",
        )

        input("\n按 Enter 回点 A...")
        holder = transition_to_target(
            arm_keeper, holder, q_a, args.duration, label="点 B → 点 A",
        )

        input("\n按 Enter 交还 policy 默认位姿并退出（Ctrl+C 同样交权）...")
        _graceful_shutdown(interrupted=False)
    except KeyboardInterrupt:
        _graceful_shutdown(interrupted=True)
        raise SystemExit(0)


if __name__ == "__main__":
    main()
