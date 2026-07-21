#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T Decoupled WBC adapter for box_demo_2.

This process lets the existing box demo keep its command and arm merge path:

  /tmp/robojudo_ext_cmd.json -> GR00T-WBC policy -> rt/lowcmd_rl
  rt/lowcmd_rl + rt/arm_sdk -> merge_lowcmd_arm_sdk.py -> rt/lowcmd

Run this instead of the official GR00T run_g1_control_loop.py when using
box_demo_2. The official loop publishes rt/lowcmd directly, while this adapter
only publishes rt/lowcmd_rl so the original arm_sdk overlay still works.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_taptap import AdaptiveTapTapController, StanceMetrics

CMD_FILE = "/tmp/robojudo_ext_cmd.json"
TAPTAP_STATUS_FILE = "/tmp/groot_taptap_status.json"

DEFAULT_BASE_HEIGHT = 0.74
DEFAULT_MIN_HEIGHT = 0.30
DEFAULT_MAX_HEIGHT = 0.80
# vx 不对称硬限(2026-07-06 真机): 快速后退(0.4)保不住高度越走越低直至摔;
# 后退安全上限 0.2 — 有意不开放 CLI(别改成旗标); 前进硬顶 1.0(--fwd-max 给
# 更大会被压回并告警; 复跑 1.2 速度指标需临时改这个常量)。
BACK_VX_MAX = 0.20
FWD_VX_HARD = 1.00

# GR00T-WBC auto-switches Balance<->Walk at ||[vx,vy,wz]|| < 0.05
# (g1_gear_wbc_policy.py:223). A command inside (0, 0.05) makes the robot
# silently balance instead of step -> the open-loop distance is wrong with no
# error signal. Commands in that dead-band are snapped to an explicit no-move.
WALK_THRESHOLD_NOMOVE = 0.05

TOTAL_HG_MOTORS = 35
BODY_MOTORS = 29
ARM_SDK_ENABLE_SLOT = 29

TOTAL_HG_MOTORS = 35
BODY_MOTORS = 29
ARM_SDK_ENABLE_SLOT = 29


@dataclass
class ExternalCommand:
    fsm: str
    vx: float
    vy: float
    wz: float
    height: float
    fresh: bool
    estop: bool = False
    allow_recovery: bool = False
    defer_recovery: bool = False


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def write_taptap_status(
    path: Path,
    controller: AdaptiveTapTapController,
    *,
    enabled: bool,
    event: str | None = None,
    stance: StanceMetrics | None = None,
) -> None:
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "state": controller.state if enabled else controller.IDLE,
        "active": bool(enabled and controller.state != controller.IDLE),
        "event": event,
        "timestamp": time.time(),
    }
    if stance is not None:
        payload["stance"] = dataclasses.asdict(stance)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def approach(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


def resolve_motion_limits(
    fwd_max: float,
    lat_max: float,
    yaw_max: float,
) -> tuple[float, float, float]:
    """Apply the same configured limits for standard and taptap launch paths."""
    fwd = min(float(fwd_max), FWD_VX_HARD)
    lat = max(0.0, float(lat_max))
    yaw = max(0.0, float(yaw_max))
    return fwd, lat, yaw


def ensure_cv2_importable() -> None:
    """Reuse Ubuntu's cv2 after conda packages, preserving conda precedence."""
    try:
        __import__("cv2")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise

    system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if system_dist_packages.is_dir() and str(system_dist_packages) not in sys.path:
        sys.path.append(str(system_dist_packages))
    __import__("cv2")


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Heading (yaw) in radians from a (w,x,y,z) quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _proj_gravity(qw: float, qx: float, qy: float, qz: float):
    """Gravity [0,0,-1] expressed in the body frame (matches decoupled_wbc
    get_gravity_orientation). A nonzero gx == pitch tilt, nonzero gy == roll
    tilt; a constant gy at rest points at an IMU roll-mount offset."""
    gx = 2.0 * (qw * qy - qx * qz)
    gy = -2.0 * (qy * qz + qw * qx)
    gz = -(qw * qw - qx * qx - qy * qy + qz * qz)
    return gx, gy, gz


def _load_ddsc() -> None:
    ddsc = (
        Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install"))
        / "lib"
        / "libddsc.so.0"
    )
    if ddsc.is_file():
        ctypes.CDLL(str(ddsc), mode=ctypes.RTLD_GLOBAL)


def read_external_command(
    cmd_file: Path,
    last_height: float,
    stale_s: float,
    fwd_max: float,
    lat_max: float,
    yaw_max: float,
    min_height: float,
    max_height: float,
) -> ExternalCommand:
    now = time.time()
    try:
        with cmd_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return ExternalCommand("RL_FULL", 0.0, 0.0, 0.0, last_height, False)
    except Exception as exc:
        print(f"[WARN] cannot read command file {cmd_file}: {exc}")
        return ExternalCommand("RL_FULL", 0.0, 0.0, 0.0, last_height, False)

    age = now - float(raw.get("timestamp", 0.0))
    fresh = age <= stale_s
    fsm = str(raw.get("fsm") or "RL_FULL")
    height = clamp(float(raw.get("height", last_height)), min_height, max_height)

    if bool(raw.get("limp")) or fsm == "LIMP":
        return ExternalCommand("LIMP", 0.0, 0.0, 0.0, height, fresh, estop=True)

    if bool(raw.get("estop")) or fsm == "DAMP":
        return ExternalCommand("DAMP", 0.0, 0.0, 0.0, height, fresh, estop=True)

    if not fresh or fsm == "RL_LOWER":
        return ExternalCommand(
            fsm, 0.0, 0.0, 0.0, height, fresh,
            allow_recovery=bool(raw.get("allow_recovery", False)),
            defer_recovery=bool(raw.get("defer_recovery", False)),
        )

    vel = raw.get("velocity") or {}
    vx = float(vel.get("forward", 0.0))
    vy = float(vel.get("lateral", 0.0))
    wz = float(vel.get("yaw", 0.0))

    # Existing RobotMover writes normalized values. agile_keyboard_control.py
    # writes physical units and marks them with units=agile.
    if raw.get("units") != "agile":
        vx *= fwd_max
        vy *= lat_max
        wz *= yaw_max

    vx = clamp(vx, -BACK_VX_MAX, fwd_max)   # 后退恒 ≤0.2, 前进走 fwd_max
    vy = clamp(vy, -lat_max, lat_max)
    wz = clamp(wz, -yaw_max, yaw_max)

    # Dead-band guard: snap a sub-Walk-threshold command to a clean no-move so it
    # never masquerades as motion (groot_mover.py floors above this, so its small
    # precise moves are unaffected).
    if 0.0 < math.sqrt(vx * vx + vy * vy + wz * wz) < WALK_THRESHOLD_NOMOVE:
        vx = vy = wz = 0.0

    return ExternalCommand(
        fsm=fsm,
        vx=vx,
        vy=vy,
        wz=wz,
        height=height,
        fresh=fresh,
        allow_recovery=bool(raw.get("allow_recovery", False)),
        defer_recovery=bool(raw.get("defer_recovery", False)),
    )


def import_groot_stack(repo: Path):
    repo = repo.expanduser().resolve()
    if not (repo / "decoupled_wbc").is_dir():
        raise SystemExit(f"GR00T-WholeBodyControl repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    ensure_cv2_importable()
    from decoupled_wbc.control.envs.g1.g1_env import G1Env
    from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
    from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

    return {
        "G1Env": G1Env,
        "ControlLoopConfig": ControlLoopConfig,
        "get_wbc_policy": get_wbc_policy,
        "instantiate_g1_robot_model": instantiate_g1_robot_model,
    }


class LowCmdRlPublisher:
    """Publish GR00T-WBC body targets to rt/lowcmd_rl instead of rt/lowcmd."""

    def __init__(self, wbc_config: dict[str, Any], topic: str = "rt/lowcmd_rl"):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.config = wbc_config
        self.topic = topic
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.pub = ChannelPublisher(topic, LowCmd_)
        self.pub.Init()
        self.crc = CRC()

        self.robot_kp = list(wbc_config["MOTOR_KP"])
        self.robot_kd = list(wbc_config["MOTOR_KD"])
        self.joint2motor = list(wbc_config["JOINT2MOTOR"])
        self.motor2joint = list(wbc_config["MOTOR2JOINT"])
        self.default_motor_angles = list(wbc_config.get("DEFAULT_MOTOR_ANGLES", [0.0] * BODY_MOTORS))
        self.weak_motor_indices = set(wbc_config.get("WeakMotorJointIndex", {}).values())
        self.pos_stop = float(wbc_config["UNITREE_LEGGED_CONST"]["PosStopF"])
        self.vel_stop = float(wbc_config["UNITREE_LEGGED_CONST"]["VelStopF"])
        self.mode_pr = int(wbc_config["UNITREE_LEGGED_CONST"]["MODE_PR"])
        self.mode_machine = int(wbc_config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"])
        self._init_lowcmd()

    def _mode_for_motor(self, motor_index: int) -> int:
        return 0x01 if motor_index in self.weak_motor_indices else 0x0A

    def _init_lowcmd(self) -> None:
        if hasattr(self.low_cmd, "level_flag"):
            self.low_cmd.level_flag = 0xFF
        if hasattr(self.low_cmd, "gpio"):
            self.low_cmd.gpio = 0
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            m.q = self.pos_stop
            m.dq = self.vel_stop
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def publish_body_targets(self, body_q, low_state=None) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine

        # Fill all 35 HG command slots. Body motors 0..28 get GR00T-WBC
        # targets. Extra slots are passive. Slot 29 must stay q=0.0 so the
        # merger never treats the RL stream as arm_sdk ownership.
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            if i < BODY_MOTORS:
                joint_index = self.motor2joint[i]
                motor_index = self.joint2motor[i]
                out = self.low_cmd.motor_cmd[motor_index]
                out.mode = self._mode_for_motor(motor_index)
                if joint_index == -1:
                    out.q = float(self.default_motor_angles[motor_index])
                else:
                    out.q = float(body_q[joint_index])
                out.dq = 0.0
                out.tau = 0.0
                out.kp = float(self.robot_kp[motor_index])
                out.kd = float(self.robot_kd[motor_index])
            else:
                q = 0.0
                if low_state is not None and i != ARM_SDK_ENABLE_SLOT:
                    try:
                        q = float(low_state.motor_state[i].q)
                    except Exception:
                        q = 0.0
                m.q = 0.0 if i == ARM_SDK_ENABLE_SLOT else q
                m.dq = 0.0
                m.tau = 0.0
                m.kp = 0.0
                m.kd = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def publish_damping(self, low_state, kd_value: float) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            if i == ARM_SDK_ENABLE_SLOT:
                q = 0.0
            else:
                try:
                    q = float(low_state.motor_state[i].q)
                except Exception:
                    q = 0.0
            m.q = q
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = float(kd_value)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def publish_limp(self) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            m.q = 0.0 if i == ARM_SDK_ENABLE_SLOT else self.pos_stop
            m.dq = self.vel_stop
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def _activate_policy_once(wbc_policy) -> None:
    if hasattr(wbc_policy, "activate_policy"):
        wbc_policy.activate_policy()
        return
    try:
        wbc_policy.lower_body_policy.use_policy_action = True
    except Exception:
        pass


def _latest_low_state(env):
    try:
        return env.body().body_state_processor.robot_low_state
    except Exception:
        return None


def measure_stance(robot_model, q) -> StanceMetrics:
    """Measure ankle-center geometry in the pelvis yaw frame."""
    robot_model.cache_forward_kinematics(q, auto_clip=False)
    pelvis = robot_model.frame_placement("pelvis")
    left = robot_model.frame_placement("left_ankle_roll_link")
    right = robot_model.frame_placement("right_ankle_roll_link")
    relative = pelvis.rotation.T @ (left.translation - right.translation)
    foot_rotation = left.rotation.T @ right.rotation
    return StanceMetrics(
        width=abs(float(relative[1])),
        stagger=float(relative[0]),
        height_delta=abs(float(relative[2])),
        yaw_error=math.atan2(float(foot_rotation[1, 0]), float(foot_rotation[0, 0])),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GR00T-WBC rt/lowcmd_rl adapter for box_demo_2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--groot-repo", type=Path, default=Path(os.environ.get("GROOT_WBC_REPO", "~/GR00T-WholeBodyControl")))
    p.add_argument("--interface", "--iface", dest="interface", default=os.environ.get("UNITREE_DDS_INTERFACE", "real"))
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--cmd-file", type=Path, default=Path(CMD_FILE))
    p.add_argument("--taptap-status-file", type=Path, default=Path(TAPTAP_STATUS_FILE),
                   help="自适应回正状态文件，供HTTP/local mover条件等待")
    p.add_argument("--publish-topic", default="rt/lowcmd_rl")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--cmd-stale-s", type=float, default=0.40)
    p.add_argument("--fwd-max", type=float, default=0.50)
    p.add_argument("--lat-max", type=float, default=0.30)
    p.add_argument("--yaw-max", type=float, default=0.60)
    p.add_argument("--taptap", action="store_true",
                   help="停止回正踏步(参考宇树官方运控): 每段运动结束后原地踏步"
                        " settle 秒恢复标准站姿+高度回站高, 段间清零开环漂移。"
                        "测试版入口 start_g1_onboard_taptap.sh / _nav_taptap.sh")
    p.add_argument("--taptap-recovery", default="legacy", choices=("legacy", "adaptive", "off"),
                   help="adaptive=仅在足间距过窄/前后错位过大时回正; "
                        "legacy=每次正常停止后固定回正; off=仅使用taptap限幅")
    p.add_argument("--taptap-settle-s", type=float, default=1.2,
                   help="踏步回正窗口时长 s")
    p.add_argument("--taptap-mode", default="wz", choices=("wz", "vx", "vx_taper"),
                   help="踏步激励方式。sim round13/13b: wz(原地小转±方波)最优 — "
                        "前后错位/脚间距/平移漂移三指标全胜, 仅 ~2° yaw 漂移")
    p.add_argument("--taptap-cmd", type=float, default=None,
                   help="踏步激励幅值 (缺省: wz 模式 0.15 rad/s, vx 模式 0.08 m/s;"
                        " 都刚超 Walk 阈值 0.05, 净位移≈0)")
    p.add_argument("--taptap-period-s", type=float, default=0.4,
                   help="踏步激励交替周期 s")
    p.add_argument("--taptap-debounce-s", type=float, default=0.35,
                   help="停止多久后才开始回正(盖过键盘 auto-repeat 空窗)")
    p.add_argument("--taptap-min-motion-s", type=float, default=0.4,
                   help="运动短于此时长不触发回正(过滤指令毛刺)")
    p.add_argument("--taptap-reference-width", type=float, default=0.24,
                   help="自适应回正的固定标准足间距 m")
    p.add_argument("--taptap-reference-stagger", type=float, default=0.08,
                   help="自适应回正的固定标准前后脚差 m")
    p.add_argument("--taptap-auto-calibrate", action="store_true",
                   help="仅已确认初始站姿正常时手动启用自动标定")
    p.add_argument("--taptap-width-margin", type=float, default=0.035,
                   help="实际足间距比标准值窄超过该值时触发回正 m")
    p.add_argument("--taptap-stagger-limit", type=float, default=0.08,
                   help="双脚前后错位触发阈值 m")
    p.add_argument("--taptap-yaw-limit", type=float, default=0.12,
                   help="双脚相对偏航角触发阈值 rad")
    p.add_argument("--taptap-height-delta-max", type=float, default=0.03,
                   help="双脚高度差超过该值时仍视为摆动期, 暂不判断 m")
    p.add_argument("--taptap-confirm-s", type=float, default=0.12,
                   help="停止防抖后用于足姿确认的采样窗口 s")
    p.add_argument("--taptap-adaptive-speed", type=float, default=0.08,
                   help="自适应回正前后对称踏步速度 m/s")
    p.add_argument("--taptap-adaptive-s", type=float, default=1.60,
                   help="自适应回正持续时间 s")
    p.add_argument("--safety-trip-ticks", type=int, default=3,
                   help="关节安全违规需连续 N 帧(50Hz)才触发停机。GR00T 原版单帧"
                        "sys.exit — 7-07 实测后退落地冲击在下垂手臂激起单帧肘部 dq"
                        "尖峰(双肘同时-11.4rad/s)被误杀; 单帧尖峰用安全动作跳过, "
                        "连续超限才 DAMP 缓停(比原版瞬退断流安全)")
    p.add_argument("--walk-height-floor", type=float, default=0.72,
                   help="行走高度地板 m (0=关)。速度命令非零时: 命令高度低于地板"
                        "→拒绝速度(打印提示); 高度仍在从低位恢复(slew)→先归零速度"
                        "等高度到位再走 (= groot_mover 的 WALK_MIN_HEIGHT warmup 机制;"
                        " sim round11: h<0.70 后退只走 60%% 距离、横移几乎不动)")
    p.add_argument("--back-lean-gain", type=float, default=0.3,
                   help="后退自动前倾增益 rad/(m/s), 0=关。默认 0.3 = sim 甜点"
                        "(round10 backward_obscomp: 骨盆后仰/tilt/停稳振荡全改善), "
                        "2026-07-06 真机确认后退稳后设为默认")
    p.add_argument("--back-lean-max", type=float, default=0.15,
                   help="前倾上限 rad (0.3增益 x 0.5m/s = 0.15)")
    p.add_argument("--back-lean-rate", type=float, default=0.30,
                   help="前倾变化速率上限 rad/s (渐入渐出)")
    p.add_argument("--stand-height", type=float, default=DEFAULT_BASE_HEIGHT)
    p.add_argument("--min-height", type=float, default=DEFAULT_MIN_HEIGHT)
    p.add_argument("--max-height", type=float, default=DEFAULT_MAX_HEIGHT)
    p.add_argument("--height-rate", type=float, default=0.20, help="height slew rate in m/s")
    p.add_argument("--wbc-model-path", default="policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx")
    p.add_argument("--control-frequency", type=int, default=50)
    p.add_argument("--sim-frequency", type=int, default=200)
    p.add_argument("--sim-sync-mode", action="store_true", help="step MuJoCo in this process instead of a sim thread")
    p.add_argument("--onscreen", action="store_true", help="show MuJoCo viewer when --interface resolves to sim")
    p.add_argument("--offscreen", action="store_true", help="enable offscreen rendering when supported")
    p.add_argument("--upper-body-joint-speed", type=float, default=1000.0)
    p.add_argument("--enable-waist", action="store_true", help="put waist in both upper/lower groups like official option")
    p.add_argument("--high-elbow-pose", action="store_true")
    p.add_argument("--damping-kd", type=float, default=8.0)
    p.add_argument("--shutdown-action", choices=("limp", "damp", "none"), default="limp",
                   help="command sent when this adapter exits by Ctrl+C")
    p.add_argument("--shutdown-s", type=float, default=1.0, help="seconds to publish shutdown command")
    p.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "1")))
    p.add_argument("--dry-run", action="store_true", help="compute policy but do not publish rt/lowcmd_rl")
    p.add_argument("--no-auto-activate-policy", action="store_true", help="leave WBC lower policy in current-q hold mode")
    p.add_argument("--disable-joint-safety", action="store_true", help="skip GR00T JointSafetyMonitor before publishing")
    p.add_argument("--print-every-s", type=float, default=1.0)
    p.add_argument("--log-pose", type=Path, default=None,
                   help="append sim ground-truth floating_base_pose to a CSV each "
                        "tick (for open-loop precision tests; sim only)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.torch_threads))

    _load_ddsc()

    import numpy as np
    import torch
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    torch.set_num_threads(max(1, int(args.torch_threads)))

    # DDS init 只做一次: 交给 G1Env(带 wbc_config[INTERFACE]=args.interface).
    # 捆绑版 sdk2py 的 ChannelFactoryInitialize 不幂等, 这里再 init 会 create-domain 冲突.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    G = import_groot_stack(args.groot_repo)
    ControlLoopConfig = G["ControlLoopConfig"]
    instantiate_g1_robot_model = G["instantiate_g1_robot_model"]
    G1Env = G["G1Env"]
    get_wbc_policy = G["get_wbc_policy"]

    config = ControlLoopConfig(
        interface=args.interface,
        control_frequency=int(args.control_frequency),
        sim_frequency=int(args.sim_frequency),
        sim_sync_mode=bool(args.sim_sync_mode),
        wbc_model_path=args.wbc_model_path,
        enable_waist=bool(args.enable_waist),
        with_hands=False,
        high_elbow_pose=bool(args.high_elbow_pose),
        upper_body_joint_speed=float(args.upper_body_joint_speed),
        verbose=False,
        enable_onscreen=bool(args.onscreen),
        enable_offscreen=bool(args.offscreen),
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["DOMAIN_ID"] = int(args.domain)
    wbc_config["INTERFACE"] = str(args.interface)  # yaml 默认 lo, 必须跟 --interface 一致

    waist_location = "lower_and_upper_body" if args.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=bool(args.high_elbow_pose),
    )
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()
    wbc_policy = get_wbc_policy(
        "g1",
        robot_model,
        wbc_config,
        init_time=time.monotonic(),
    )
    if not args.no_auto_activate_policy:
        _activate_policy_once(wbc_policy)

    publisher = LowCmdRlPublisher(wbc_config=wbc_config, topic=args.publish_topic)

    if float(args.fwd_max) > FWD_VX_HARD:
        print(f"[WARN] --fwd-max {args.fwd_max:.2f} 超前进硬顶, 压回 {FWD_VX_HARD:.2f} m/s")
    fwd_max_eff, lat_max_eff, yaw_max_eff = resolve_motion_limits(
        args.fwd_max,
        args.lat_max,
        args.yaw_max,
    )

    # 安全监视防抖: env_type 改 "sim" 让 handle_violations 返回 shutdown_required
    # 而不是单帧直接 sys.exit(1)(违规打印/停机由下面 adapter 自己防抖后接管)。
    safety_trip = 0
    if not args.disable_joint_safety:
        try:
            env.safety_monitor.env_type = "sim"
        except Exception as exc:
            print(f"[WARN] 无法接管 safety monitor 停机路径: {exc}")

    # 后退自动前倾(vx<0 时腰pitch前倾弥补重心靠后)。机制 = 动作偏置+观测补偿:
    # 腰pitch动作目标 += lean, 同时喂给策略的 obs["q"] 腰pitch -= lean, 策略
    # 对偏置无感不会反补偿。sim 标定(round8-10): rpy 命令通道无效; 纯动作偏置
    # 被策略从关节观测发现并对抗(骨盆更后仰+位移超冲); obscomp 全指标改善。
    back_lean_gain = max(0.0, float(args.back_lean_gain))
    lean_state = 0.0
    waist_dof = None
    waist_body_idx = -1
    if back_lean_gain > 0.0:
        waist_dof = getattr(robot_model, "joint_to_dof_index", {}).get("waist_pitch_joint")
        try:
            waist_body_idx = int(publisher.motor2joint[14])   # HG motor 14 = WaistPitch
        except Exception:
            waist_body_idx = -1
        if waist_dof is None or waist_body_idx < 0:
            print(f"[WARN] back-lean 已禁用: waist_pitch 不可寻址 "
                  f"(dof={waist_dof}, body_idx={waist_body_idx}) — 检查 --enable-waist")
            back_lean_gain = 0.0

    # taptap 停止回正状态(--taptap 开启时用)
    tap_amp = (float(args.taptap_cmd) if args.taptap_cmd is not None
               else (0.15 if args.taptap_mode == "wz" else 0.08))
    tap_motion_since = None
    tap_zero_since = None
    tap_last_motion_dur = 0.0
    tap_until = 0.0
    tap_t0 = 0.0
    tap_start_sign = 1.0
    adaptive_taptap = AdaptiveTapTapController(
        reference_width=args.taptap_reference_width,
        reference_stagger=args.taptap_reference_stagger,
        width_margin=args.taptap_width_margin,
        stagger_limit=args.taptap_stagger_limit,
        yaw_limit=args.taptap_yaw_limit,
        max_height_delta=args.taptap_height_delta_max,
        debounce_s=args.taptap_debounce_s,
        confirm_s=args.taptap_confirm_s,
        min_motion_s=args.taptap_min_motion_s,
        recovery_s=args.taptap_adaptive_s,
        recovery_speed=args.taptap_adaptive_speed,
        phase_s=args.taptap_period_s,
        auto_calibrate=args.taptap_auto_calibrate,
    )
    last_stance = None
    adaptive_enabled = bool(args.taptap and args.taptap_recovery == "adaptive")
    last_taptap_status_write = 0.0
    write_taptap_status(
        args.taptap_status_file,
        adaptive_taptap,
        enabled=adaptive_enabled,
    )

    dt = 1.0 / float(args.hz)
    height_cmd = clamp(float(args.stand_height), args.min_height, args.max_height)
    last_print = 0.0
    last_gate_print = 0.0

    pose_log = None
    if args.log_pose is not None:
        pose_log = open(args.log_pose, "w")
        pose_log.write(
            "time,x,y,z,qw,qx,qy,qz,fsm,vx,vy,wz,"
            "yaw,gyro_x,gyro_y,gyro_z,grav_x,grav_y,grav_z\n"
        )

    print("=" * 72)
    print("GR00T-WBC adapter for box_demo_2")
    print(f"  repo:        {args.groot_repo.expanduser().resolve()}")
    print(f"  interface:   {config.interface} ({config.env_type}), domain={args.domain}")
    print(f"  cmd file:    {args.cmd_file}")
    print(f"  publish:     {'DRY-RUN' if args.dry_run else args.publish_topic} @ {args.hz:.1f}Hz")
    print(f"  limits:      fwd={fwd_max_eff:.2f}(硬顶{FWD_VX_HARD:.1f}) "
          f"back={BACK_VX_MAX:.2f}(固定) lat={lat_max_eff:.2f} yaw={yaw_max_eff:.2f}")
    if back_lean_gain > 0.0:
        print(f"  back-lean:   gain={back_lean_gain:.2f} rad/(m/s) "
              f"max={args.back_lean_max:.2f} rate={args.back_lean_rate:.2f} (obscomp)")
    if args.taptap and args.taptap_recovery == "legacy":
        print(f"  taptap:      ON — {args.taptap_mode} 停止回正踏步 "
              f"{args.taptap_settle_s:.1f}s ±{tap_amp:.2f} "
              f"周期{args.taptap_period_s:.1f}s (防抖{args.taptap_debounce_s:.2f}s)")
    elif args.taptap and args.taptap_recovery == "adaptive":
        print(f"  taptap:      adaptive vx=±{args.taptap_adaptive_speed:.2f}m/s "
              f"for {args.taptap_adaptive_s:.2f}s; "
              f"width<{args.taptap_reference_width:.3f}-{args.taptap_width_margin:.3f}m "
              f"or stagger>{args.taptap_stagger_limit:.3f}m")
    elif args.taptap:
        print("  taptap:      limits ON; recovery OFF")
    print(f"  height:      {args.min_height:.2f}..{args.max_height:.2f}m, rate={args.height_rate:.2f}m/s")
    if float(args.walk_height_floor) > 0.0:
        print(f"  walk-floor:  {args.walk_height_floor:.2f}m (低位拒走/恢复中warmup拦速度)")
    print(f"  policy:      {'auto-activated' if not args.no_auto_activate_policy else 'current-q hold'}")
    print("  merger:      keep merge_lowcmd_arm_sdk.py running to publish final rt/lowcmd")
    print("=" * 72)

    try:
        while True:
            t0 = time.monotonic()
            if env.sim and config.sim_sync_mode:
                env.step_simulator()

            cmd = read_external_command(
                cmd_file=args.cmd_file,
                last_height=height_cmd,
                stale_s=float(args.cmd_stale_s),
                fwd_max=fwd_max_eff,
                lat_max=lat_max_eff,
                yaw_max=yaw_max_eff,
                min_height=float(args.min_height),
                max_height=float(args.max_height),
            )
            # 行走高度地板(warmup): 低位不走, 恢复到位才放行速度
            # (必须在 taptap 之前: 被 GATE 拦下的"假运动"不算运动, 蹲位不误触发回正)
            floor = float(args.walk_height_floor)
            wants_motion = (abs(cmd.vx) + abs(cmd.vy) + abs(cmd.wz)) > 1e-3
            if floor > 0.0 and wants_motion and height_cmd < floor - 0.01:
                now_g = time.monotonic()
                if now_g - last_gate_print >= 2.0:
                    reason = ("命令高度低于地板" if cmd.height < floor
                              else "高度恢复中(warmup)")
                    print(f"[GATE] 速度已拦: {reason} "
                          f"(height_cmd={height_cmd:.2f} < floor={floor:.2f}) — "
                          f"升高度(r键回站立)后放行")
                    last_gate_print = now_g
                cmd = dataclasses.replace(cmd, vx=0.0, vy=0.0, wz=0.0)

            # --- taptap 停止回正: 显式停止(fresh 零命令)防抖后注入原地踏步窗.
            # 对抗审查修复(7-08): ①运动判定用 GATE 后命令 + 开窗要求高度在地板上
            # (蹲位/被拦的假运动不触发, 不再强制从蹲位站起); ②窗长取偶数个半周期
            # + 起始方向轮换(激励积分净零, 不再每次+3°); ③stale/非 RL_FULL 不触发
            # 不注入(命令黑箱期绝不自主动作); ④毛刺段不覆盖待回正记录, 被打断的
            # 回正下次停止后补做(完成才消费).
            tap_active = False
            if args.taptap and args.taptap_recovery == "legacy":
                now_t = time.monotonic()
                user_moving = (abs(cmd.vx) + abs(cmd.vy) + abs(cmd.wz)) > 1e-3
                bad_state = cmd.estop or cmd.fsm != "RL_FULL"
                if user_moving or bad_state:
                    if tap_until > 0.0:
                        print("[TAPTAP] 中断回正 (新命令/状态切换)")
                    tap_until = 0.0
                    tap_zero_since = None
                    if bad_state:
                        tap_motion_since = None
                        tap_last_motion_dur = 0.0
                    else:
                        tap_motion_since = tap_motion_since or now_t
                else:
                    if tap_motion_since is not None:          # 运动->停 过渡
                        seg = now_t - tap_motion_since
                        tap_motion_since = None
                        if cmd.fresh:                         # 显式停止命令才武装
                            if seg >= float(args.taptap_min_motion_s):
                                tap_last_motion_dur = seg     # 毛刺段不覆盖旧记录
                            tap_zero_since = now_t
                        else:                                 # 命令黑箱(stale): 不回正
                            tap_last_motion_dur = 0.0
                            tap_zero_since = None
                    height_ok = (floor <= 0.0 or height_cmd >= floor - 0.005)
                    if (tap_until == 0.0 and tap_zero_since is not None
                            and now_t - tap_zero_since >= float(args.taptap_debounce_s)
                            and tap_last_motion_dur >= float(args.taptap_min_motion_s)
                            and height_ok):
                        period = max(1e-3, float(args.taptap_period_s))
                        n_half = max(2, int(round(float(args.taptap_settle_s) / period)))
                        n_half += n_half % 2                  # 偶数半周期 -> 净激励零
                        tap_start_sign = -tap_start_sign      # 相邻回正起始方向轮换
                        tap_until = now_t + n_half * period
                        tap_t0 = now_t
                        print(f"[TAPTAP] 回正: {args.taptap_mode} 踏步 "
                              f"{n_half * period:.1f}s (±{tap_amp:.2f}, "
                              f"{n_half}个半周期) 高度回 {args.stand_height:.2f}")
                    if tap_until > 0.0:
                        if now_t < tap_until:
                            tap_active = True
                            period = max(1e-3, float(args.taptap_period_s))
                            ph = int((now_t - tap_t0) / period) % 2
                            amp = tap_amp
                            if args.taptap_mode == "vx_taper":
                                amp *= (1.0 - 0.3 * (now_t - tap_t0)
                                        / max(1e-6, tap_until - tap_t0))
                            sgn_amp = tap_start_sign * (amp if ph == 0 else -amp)
                            if args.taptap_mode == "wz":
                                cmd = dataclasses.replace(cmd, vx=0.0, vy=0.0,
                                                          wz=sgn_amp)
                            else:
                                cmd = dataclasses.replace(cmd, vx=sgn_amp, vy=0.0,
                                                          wz=0.0)
                        else:
                            tap_until = 0.0
                            tap_zero_since = None
                            tap_last_motion_dur = 0.0         # 完成才消费; 打断保留补做
                            print("[TAPTAP] 回正完成")
            elif args.taptap and args.taptap_recovery == "adaptive":
                cmd, tap_active, tap_event = adaptive_taptap.update(
                    time.monotonic(), cmd, last_stance
                )
                now_status = time.monotonic()
                if tap_event is not None or now_status - last_taptap_status_write >= 0.10:
                    write_taptap_status(
                        args.taptap_status_file,
                        adaptive_taptap,
                        enabled=True,
                        event=tap_event,
                        stance=last_stance,
                    )
                    last_taptap_status_write = now_status
                if tap_event == "started":
                    print(f"[TAPTAP] 站姿异常, 开始自适应回正 "
                          f"(width={last_stance.width:.3f}m "
                          f"stagger={last_stance.stagger:.3f}m "
                          f"foot_yaw={last_stance.yaw_error:.3f}rad "
                          f"reference={adaptive_taptap.reference_width:.3f}m)")
                elif tap_event == "checking_initial":
                    print("[TAPTAP] 初始站姿异常, 拦住首条运动并检查回正")
                elif tap_event == "healthy":
                    print(f"[TAPTAP] 站姿正常, 跳过回正 "
                          f"(reference={adaptive_taptap.reference_width:.3f}m)")
                elif tap_event == "completed":
                    print("[TAPTAP] 自适应回正完成")
                elif tap_event == "cancelled":
                    print("[TAPTAP] 安全/状态命令中断回正")

            height_cmd = approach(
                height_cmd,
                float(args.stand_height) if tap_active else cmd.height,
                float(args.height_rate) * dt)

            lean_target = 0.0
            if (back_lean_gain > 0.0 and not tap_active and not cmd.estop
                    and cmd.fsm not in ("DAMP", "LIMP")):
                lean_target = clamp(back_lean_gain * max(0.0, -cmd.vx),
                                    0.0, float(args.back_lean_max))
            lean_state = approach(lean_state, lean_target,
                                  float(args.back_lean_rate) * dt)

            try:
                obs = env.observe()
            except Exception as exc:
                now = time.monotonic()
                if now - last_print >= float(args.print_every_s):
                    print(f"[WAIT] no valid GR00T-WBC observation yet: {exc}")
                    last_print = now
                time.sleep(dt)
                continue

            low_state = _latest_low_state(env)

            try:
                last_stance = measure_stance(robot_model, obs["q"])
            except Exception as exc:
                last_stance = None
                now = time.monotonic()
                if now - last_print >= float(args.print_every_s):
                    print(f"[WARN] cannot measure foot stance: {exc}")
                    last_print = now

            if lean_state > 1e-4 and waist_dof is not None:
                # 观测补偿: 策略看到的腰pitch = 实测值 - 偏置(即它自己命令的角度)
                q_comp = np.array(obs["q"], copy=True)
                q_comp[waist_dof] -= lean_state
                obs = dict(obs)
                obs["q"] = q_comp

            if pose_log is not None:
                try:
                    p7 = obs["floating_base_pose"]
                    qw, qx, qy, qz = float(p7[3]), float(p7[4]), float(p7[5]), float(p7[6])
                    yaw = _yaw_from_quat(qw, qx, qy, qz)
                    gx, gy, gz = _proj_gravity(qw, qx, qy, qz)
                    try:
                        fbv = obs["floating_base_vel"]
                        wgx, wgy, wgz = float(fbv[3]), float(fbv[4]), float(fbv[5])
                    except Exception:
                        wgx = wgy = wgz = 0.0
                    pose_log.write(
                        "%.4f,%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%.4f,%.4f,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                        % (time.time(), float(p7[0]), float(p7[1]), float(p7[2]),
                           qw, qx, qy, qz,
                           cmd.fsm, cmd.vx, cmd.vy, cmd.wz,
                           yaw, wgx, wgy, wgz, gx, gy, gz))
                    pose_log.flush()
                except Exception:
                    pass

            if cmd.fsm == "LIMP":
                if not args.dry_run:
                    publisher.publish_limp()
            elif cmd.estop or cmd.fsm == "DAMP":
                if not args.dry_run and low_state is not None:
                    publisher.publish_damping(low_state, args.damping_kd)
            else:
                t_now = time.monotonic()
                wbc_policy.set_observation(obs)
                goal = {
                    "navigate_cmd": np.array([cmd.vx, cmd.vy, cmd.wz], dtype=np.float32),
                    "base_height_command": np.array([height_cmd], dtype=np.float32),
                    "target_time": t_now + dt,
                    "interpolation_garbage_collection_time": t_now - 2.0 * dt,
                }
                wbc_policy.set_goal(goal)
                action = wbc_policy.get_action(time=t_now)

                if not args.disable_joint_safety:
                    try:
                        sres = env.safety_monitor.handle_violations(obs, action)
                        action = sres["action"]
                        if sres.get("shutdown_required"):
                            safety_trip += 1
                            vio = getattr(env.safety_monitor, "violations", [])
                            print(f"[SAFETY] 违规帧 {safety_trip}/{args.safety_trip_ticks}: "
                                  + "; ".join(f"{v.get('joint')}={v.get('value'):+.1f}rad/s"
                                              for v in vio if v.get("critical", True)))
                            if safety_trip >= int(args.safety_trip_ticks):
                                print("[SAFETY] 连续超限 — DAMP 缓停后退出 "
                                      "(单帧尖峰不会走到这里)")
                                for _ in range(max(1, int(float(args.shutdown_s) * args.hz))):
                                    ls = _latest_low_state(env)
                                    if not args.dry_run and ls is not None:
                                        publisher.publish_damping(ls, args.damping_kd)
                                    time.sleep(dt)
                                sys.exit(1)
                        else:
                            safety_trip = 0
                    except SystemExit:
                        raise
                    except Exception as exc:
                        print(f"[WARN] joint safety monitor failed; using raw action: {exc}")

                body_q = robot_model.get_body_actuated_joints(action["q"])
                if lean_state > 1e-4 and waist_body_idx >= 0:
                    body_q = np.array(body_q, copy=True)
                    body_q[waist_body_idx] += lean_state
                if not args.dry_run:
                    publisher.publish_body_targets(body_q, low_state=low_state)

            now = time.monotonic()
            if now - last_print >= float(args.print_every_s):
                lean_s = f" lean={lean_state:.3f}" if lean_state > 1e-4 else ""
                print(
                    f"[GR00T-WBC] fsm={cmd.fsm:8s} fresh={int(cmd.fresh)} "
                    f"cmd=({cmd.vx:+.2f},{cmd.vy:+.2f},{cmd.wz:+.2f},h={height_cmd:.2f})"
                    f"{lean_s}"
                )
                last_print = now

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print(f"\nStopping GR00T-WBC adapter; shutdown_action={args.shutdown_action}...")
        if not args.dry_run and args.shutdown_action != "none":
            for _ in range(max(1, int(float(args.shutdown_s) * args.hz))):
                if args.shutdown_action == "limp":
                    publisher.publish_limp()
                elif args.shutdown_action == "damp":
                    low_state = _latest_low_state(env)
                    if low_state is not None:
                        publisher.publish_damping(low_state, args.damping_kd)
                time.sleep(dt)
    finally:
        try:
            write_taptap_status(
                args.taptap_status_file,
                adaptive_taptap,
                enabled=False,
                event="shutdown",
                stance=last_stance,
            )
        except Exception as exc:
            print(f"[WARN] 无法清理 taptap 状态文件: {exc}")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
