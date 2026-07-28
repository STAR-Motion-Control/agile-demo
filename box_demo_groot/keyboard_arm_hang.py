#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyboard-triggered, bumpless natural-arm-hang controller.

The lower-body keyboard process owns no arm joints until ``activate()`` is
requested.  On activation, the first rt/arm_sdk frame is copied from the live
rt/lowcmd_rl upper-body command.  Position and gains then follow a minimum-jerk
trajectory to the natural-hang target.  Release performs the inverse handoff:
track the live policy command first, then fade arm_sdk weight to zero.

This helper intentionally owns only the 14 arm joints.  It marks its arm_sdk
frames as ``arms-only`` so the merger leaves all three waist joints on the
GR00T-dWBC policy in both standing and locomotion.
"""

from __future__ import annotations

import math
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Optional


# Target order: left arm 7, right arm 7.  Waist 12..14 always remains on RL.
ARM_JOINTS = tuple(range(15, 29))
ARM_SDK_ENABLE = 29
ARM_SDK_ARMS_ONLY_MARKER = 1.0
RL_LOWER_HANDOFF_Q = (
    0.05, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03,
    0.05, -0.22, 0.0, 0.98, -0.20, 0.03, 0.03,
)

# Use 1.15x the released GR00T-dWBC arm MOTOR_KP, selected by the paired
# non-inferiority sweep; retain the released damping.
# The previous manipulation gains (up to kp=180, kd=10) rigidly coupled arm
# disturbances into the torso and were not the impedance used by locomotion.
TARGET_KP = (
    115.0, 115.0, 46.0, 46.0, 23.0, 23.0, 23.0,
    115.0, 115.0, 46.0, 46.0, 23.0, 23.0, 23.0,
)
TARGET_KD = (
    5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0,
)

CONTROL_DT = 0.02
DEFAULT_MOVE_DURATION = 3.0
DEFAULT_RELEASE_DURATION = 2.5
DEFAULT_WEIGHT_FADE_DURATION = 0.35
DEFAULT_POLICY_STALE_S = 0.20
DEFAULT_ARM_OWNER_STALE_S = 0.28
LOCK_ACQUIRE_TIMEOUT_S = 0.02
WORKER_HEARTBEAT_S = 0.10
WORKER_STALE_S = 0.50
WORKER_REQUEST_TIMEOUT_S = 0.15
# motor 29 is protocol metadata, not a physical actuator. reserve distinguishes
# this helper's loopback samples from another rt/arm_sdk writer.
KEYBOARD_ARM_OWNER_MARKER = 0x4B48
Q_SANE_RAD = 10.0


def minimum_jerk_ratio(value: float) -> float:
    """Fifth-order smoothstep: zero velocity and acceleration at both ends."""
    u = max(0.0, min(1.0, float(value)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


@dataclass(frozen=True)
class MotorFrame:
    mode: int
    q: float
    dq: float
    tau: float
    kp: float
    kd: float
    reserve: int = 0


@dataclass(frozen=True)
class UpperBodyFrame:
    mode_machine: int
    mode_pr: int
    motors: tuple[MotorFrame, ...]
    weight: float = 1.0


def policy_frame_from_lowcmd(msg) -> UpperBodyFrame:
    motors = []
    for joint in ARM_JOINTS:
        src = msg.motor_cmd[joint]
        motors.append(
            MotorFrame(
                mode=int(src.mode),
                q=float(src.q),
                dq=float(src.dq),
                tau=float(src.tau),
                kp=float(src.kp),
                kd=float(src.kd),
                reserve=int(src.reserve),
            )
        )
    return UpperBodyFrame(
        mode_machine=int(msg.mode_machine),
        mode_pr=int(msg.mode_pr),
        motors=tuple(motors),
        weight=1.0,
    )


def _frame_is_safe_policy_start(frame: UpperBodyFrame) -> bool:
    if len(frame.motors) != len(ARM_JOINTS):
        return False
    for motor in frame.motors:
        values = (motor.q, motor.dq, motor.tau, motor.kp, motor.kd)
        if not all(math.isfinite(value) for value in values):
            return False
        if abs(motor.q) >= Q_SANE_RAD or motor.kp <= 1e-6:
            return False
    return True


def _target_frame(header: UpperBodyFrame) -> UpperBodyFrame:
    motors = []
    for i, start in enumerate(header.motors):
        motors.append(
            MotorFrame(
                mode=1,
                q=RL_LOWER_HANDOFF_Q[i],
                dq=0.0,
                tau=0.0,
                kp=TARGET_KP[i],
                kd=TARGET_KD[i],
                reserve=start.reserve,
            )
        )
    return UpperBodyFrame(
        mode_machine=header.mode_machine,
        mode_pr=header.mode_pr,
        motors=tuple(motors),
        weight=1.0,
    )


def _blend_frame(
    source: UpperBodyFrame,
    target: UpperBodyFrame,
    ratio: float,
    *,
    weight: float = 1.0,
) -> UpperBodyFrame:
    r = max(0.0, min(1.0, float(ratio)))
    motors = []
    for left, right in zip(source.motors, target.motors):
        motors.append(
            MotorFrame(
                mode=left.mode if r < 0.5 else right.mode,
                q=left.q + (right.q - left.q) * r,
                dq=left.dq + (right.dq - left.dq) * r,
                tau=left.tau + (right.tau - left.tau) * r,
                kp=left.kp + (right.kp - left.kp) * r,
                kd=left.kd + (right.kd - left.kd) * r,
                reserve=left.reserve if r < 0.5 else right.reserve,
            )
        )
    return UpperBodyFrame(
        mode_machine=target.mode_machine,
        mode_pr=target.mode_pr,
        motors=tuple(motors),
        weight=max(0.0, min(1.0, float(weight))),
    )


class ArmHangPlanner:
    """Pure trajectory state machine; it performs no DDS I/O."""

    IDLE = "idle"
    MOVING = "moving_to_hang"
    HOLDING = "holding"
    RELEASING = "aligning_to_policy"
    FADING = "fading_weight"

    def __init__(
        self,
        move_duration: float = DEFAULT_MOVE_DURATION,
        release_duration: float = DEFAULT_RELEASE_DURATION,
        fade_duration: float = DEFAULT_WEIGHT_FADE_DURATION,
    ):
        self.move_duration = max(CONTROL_DT, float(move_duration))
        self.release_duration = max(CONTROL_DT, float(release_duration))
        self.fade_duration = max(CONTROL_DT, float(fade_duration))
        self.state = self.IDLE
        self._phase_start = 0.0
        self._last_step_time = 0.0
        self._start: Optional[UpperBodyFrame] = None
        self._target: Optional[UpperBodyFrame] = None
        self._last_output: Optional[UpperBodyFrame] = None

    @property
    def active(self) -> bool:
        return self.state != self.IDLE

    @property
    def last_output(self) -> Optional[UpperBodyFrame]:
        return self._last_output

    def activate(self, policy: UpperBodyFrame, now: float) -> bool:
        if not _frame_is_safe_policy_start(policy):
            return False
        start = self._last_output if self.active and self._last_output is not None else policy
        self._start = start
        self._target = _target_frame(policy)
        self._phase_start = float(now)
        self._last_step_time = float(now)
        self.state = self.MOVING
        self._last_output = start
        return True

    def release(self, policy: UpperBodyFrame, now: float) -> bool:
        if not self.active or not _frame_is_safe_policy_start(policy):
            return False
        self._start = self._last_output or _target_frame(policy)
        self._phase_start = float(now)
        self._last_step_time = float(now)
        self.state = self.RELEASING
        return True

    def emergency_release(self, policy: Optional[UpperBodyFrame]) -> Optional[UpperBodyFrame]:
        output = None
        if policy is not None and _frame_is_safe_policy_start(policy):
            output = UpperBodyFrame(
                mode_machine=policy.mode_machine,
                mode_pr=policy.mode_pr,
                motors=policy.motors,
                weight=0.0,
            )
        self.state = self.IDLE
        self._start = None
        self._target = None
        self._last_output = output
        return output

    def step(self, now: float, policy: Optional[UpperBodyFrame]) -> Optional[UpperBodyFrame]:
        now = float(now)
        if self.state == self.IDLE:
            return None
        elapsed_since_step = max(0.0, now - self._last_step_time)
        self._last_step_time = now

        if self.state == self.MOVING:
            assert self._start is not None and self._target is not None
            u = (now - self._phase_start) / self.move_duration
            output = _blend_frame(
                self._start,
                self._target,
                minimum_jerk_ratio(u),
            )
            if u >= 1.0:
                output = self._target
                self.state = self.HOLDING

        elif self.state == self.HOLDING:
            assert self._target is not None
            output = self._target

        elif self.state == self.RELEASING:
            if policy is None or not _frame_is_safe_policy_start(policy):
                # Freeze phase time while policy is stale.  Resuming must not
                # jump to a later interpolation point.
                self._phase_start += elapsed_since_step
                return self._last_output
            assert self._start is not None
            u = (now - self._phase_start) / self.release_duration
            output = _blend_frame(
                self._start,
                policy,
                minimum_jerk_ratio(u),
            )
            if u >= 1.0:
                output = policy
                self._phase_start = now
                self.state = self.FADING

        elif self.state == self.FADING:
            if policy is None or not _frame_is_safe_policy_start(policy):
                self._phase_start += elapsed_since_step
                return self._last_output
            u = (now - self._phase_start) / self.fade_duration
            weight = 1.0 - minimum_jerk_ratio(u)
            output = UpperBodyFrame(
                mode_machine=policy.mode_machine,
                mode_pr=policy.mode_pr,
                motors=policy.motors,
                weight=weight,
            )
            if u >= 1.0:
                output = UpperBodyFrame(
                    mode_machine=policy.mode_machine,
                    mode_pr=policy.mode_pr,
                    motors=policy.motors,
                    weight=0.0,
                )
                self.state = self.IDLE
                self._start = None
                self._target = None
        else:
            raise RuntimeError(f"unknown arm-hang planner state: {self.state}")

        self._last_output = output
        return output


class KeyboardArmHangController:
    """DDS transport around :class:`ArmHangPlanner`.

    Constructing the object is side-effect free. ``start()`` creates DDS
    readers/writer but still publishes nothing until ``toggle()`` activates it.
    """

    def __init__(
        self,
        iface: str,
        *,
        move_duration: float = DEFAULT_MOVE_DURATION,
        release_duration: float = DEFAULT_RELEASE_DURATION,
        policy_stale_s: float = DEFAULT_POLICY_STALE_S,
    ):
        self.iface = str(iface)
        self.policy_stale_s = max(CONTROL_DT, float(policy_stale_s))
        self.planner = ArmHangPlanner(
            move_duration=move_duration,
            release_duration=release_duration,
        )
        self._lock = threading.Lock()
        self._latest_policy: Optional[UpperBodyFrame] = None
        self._latest_policy_t = 0.0
        self._latest_output: Optional[UpperBodyFrame] = None
        self._latest_output_t = 0.0
        self._latest_external_arm_weight = 0.0
        self._latest_external_arm_t = 0.0
        self._publisher = None
        self._subscriber_rl = None
        self._subscriber_output = None
        self._subscriber_arm = None
        self._crc = None
        self._lowcmd_factory = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def state(self) -> str:
        # A display-only read must never block the lower-body keyboard loop.
        # A one-frame stale label is harmless; waiting forever on a wedged DDS
        # callback lock would disable every locomotion key.
        return self.planner.state

    def start(self) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, self.iface)
        self._lowcmd_factory = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._subscriber_rl = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
        self._subscriber_rl.Init(self._on_policy, 10)
        # Activation starts from the command that the merger is actually
        # sending, not merely measured q or a possibly different policy frame.
        self._subscriber_output = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._subscriber_output.Init(self._on_output, 10)
        self._subscriber_arm = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self._subscriber_arm.Init(self._on_arm_sdk, 10)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="keyboard_arm_hang",
        )
        self._thread.start()

    def _on_policy(self, msg) -> None:
        try:
            frame = policy_frame_from_lowcmd(msg)
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._latest_policy = frame
            self._latest_policy_t = time.monotonic()
        finally:
            self._lock.release()

    def _on_output(self, msg) -> None:
        try:
            frame = policy_frame_from_lowcmd(msg)
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._latest_output = frame
            self._latest_output_t = time.monotonic()
        finally:
            self._lock.release()

    def _on_arm_sdk(self, msg) -> None:
        try:
            enable = msg.motor_cmd[ARM_SDK_ENABLE]
            weight = float(enable.q)
            owner_marker = int(enable.reserve)
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not math.isfinite(weight):
            return
        if owner_marker == KEYBOARD_ARM_OWNER_MARKER:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._latest_external_arm_weight = max(0.0, min(1.0, weight))
            self._latest_external_arm_t = time.monotonic()
        finally:
            self._lock.release()

    def _fresh_frame_locked(
        self,
        now: float,
        frame: Optional[UpperBodyFrame],
        timestamp: float,
    ) -> Optional[UpperBodyFrame]:
        if frame is None or now - timestamp > self.policy_stale_s:
            return None
        return frame

    def _fresh_policy_locked(self, now: float) -> Optional[UpperBodyFrame]:
        return self._fresh_frame_locked(
            now,
            self._latest_policy,
            self._latest_policy_t,
        )

    def _fresh_output_locked(self, now: float) -> Optional[UpperBodyFrame]:
        return self._fresh_frame_locked(
            now,
            self._latest_output,
            self._latest_output_t,
        )

    def _external_owner_active_locked(self, now: float) -> bool:
        return (
            now - self._latest_external_arm_t <= DEFAULT_ARM_OWNER_STALE_S
            and self._latest_external_arm_weight > 1e-3
        )

    def toggle(self) -> tuple[bool, str]:
        now = time.monotonic()
        if not self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_S):
            return False, "手臂 DDS 锁忙；腿部键盘保持可用，本次 h 已忽略"
        try:
            policy = self._fresh_policy_locked(now)
            if policy is None:
                return False, "未收到新鲜 rt/lowcmd_rl，拒绝切换双臂所有权"
            if self.planner.active:
                ok = self.planner.release(policy, now)
                first = None
                message = "开始平滑交还双臂给运控 policy"
            else:
                if self._external_owner_active_locked(now):
                    return False, "检测到其他新鲜 arm_sdk 发布者，拒绝双写接管"
                output = self._fresh_output_locked(now)
                if output is None:
                    return False, "未收到新鲜 rt/lowcmd 最终指令，拒绝接管"
                ok = self.planner.activate(output, now)
                first = self.planner.step(now, policy) if ok else None
                message = "开始平滑移动到双臂自然下垂"
        finally:
            self._lock.release()
        # Publish the exact current-command frame immediately. Waiting for the
        # next 50 Hz thread tick would leave an avoidable takeover gap.
        if first is not None:
            self._publish(first)
        return ok, message

    def emergency_release(self) -> None:
        if not self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_S):
            return
        try:
            policy = self._fresh_policy_locked(time.monotonic())
            output = self.planner.emergency_release(policy)
        finally:
            self._lock.release()
        if output is not None:
            self._publish(output)

    def _publish(self, frame: UpperBodyFrame) -> None:
        if self._publisher is None or self._lowcmd_factory is None or self._crc is None:
            return
        msg = self._lowcmd_factory()
        msg.mode_machine = frame.mode_machine
        msg.mode_pr = frame.mode_pr
        enable = msg.motor_cmd[ARM_SDK_ENABLE]
        enable.mode = 1
        enable.q = frame.weight
        # motor 29 is the arm_sdk protocol enable slot and is never copied to
        # a physical actuator by the merger.  dq=1 is our backward-compatible
        # ownership extension: this publisher owns arms 15..28, never waist.
        # Existing manipulation publishers leave dq=0 and retain waist control.
        enable.dq = ARM_SDK_ARMS_ONLY_MARKER
        enable.tau = enable.kp = enable.kd = 0.0
        enable.reserve = KEYBOARD_ARM_OWNER_MARKER
        for joint, source in zip(ARM_JOINTS, frame.motors):
            motor = msg.motor_cmd[joint]
            motor.mode = source.mode
            motor.q = source.q
            motor.dq = source.dq
            motor.tau = source.tau
            motor.kp = source.kp
            motor.kd = source.kd
            motor.reserve = source.reserve
        msg.crc = self._crc.Crc(msg)
        self._publisher.Write(msg)

    def _loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            output = None
            if self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_S):
                try:
                    policy = self._fresh_policy_locked(now)
                    if self.planner.active and self._external_owner_active_locked(now):
                        # A manipulation publisher has taken ownership. Stop
                        # this overlay immediately; the fresh external frame
                        # already provides the handoff target.
                        self.planner.emergency_release(None)
                    else:
                        output = self.planner.step(now, policy)
                finally:
                    self._lock.release()
            if output is not None:
                self._publish(output)
            deadline += CONTROL_DT
            wait_s = deadline - time.monotonic()
            if wait_s <= 0.0:
                deadline = time.monotonic()
            else:
                self._stop.wait(wait_s)

    def shutdown(self, *, graceful: bool = True) -> bool:
        """Stop the helper; align to policy first when an overlay is active."""
        if graceful:
            now = time.monotonic()
            if self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_S):
                try:
                    policy = self._fresh_policy_locked(now)
                    if self.planner.active and policy is not None:
                        self.planner.release(policy, now)
                finally:
                    self._lock.release()
            timeout = (
                self.planner.release_duration
                + self.planner.fade_duration
                + 1.0
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self.planner.active:
                    break
                time.sleep(CONTROL_DT)

        active = self.planner.active
        if active:
            self.emergency_release()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        return not active


def _worker_send(connection, message) -> bool:
    try:
        connection.send(message)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _arm_hang_worker_main(
    connection,
    iface: str,
    move_duration: float,
    release_duration: float,
) -> None:
    """Own all arm DDS objects in a process isolated from the base keyboard."""
    controller = KeyboardArmHangController(
        iface,
        move_duration=move_duration,
        release_duration=release_duration,
    )
    try:
        controller.start()
    except Exception as exc:
        _worker_send(connection, ("ready", False, str(exc), "unavailable"))
        connection.close()
        return

    if not _worker_send(connection, ("ready", True, "", controller.state)):
        return

    last_heartbeat = 0.0
    keep_running = True
    try:
        while keep_running:
            now = time.monotonic()
            if now - last_heartbeat >= WORKER_HEARTBEAT_S:
                if not _worker_send(connection, ("state", controller.state, now)):
                    break
                last_heartbeat = now

            if not connection.poll(0.02):
                continue
            try:
                request_id, command, payload = connection.recv()
            except (EOFError, OSError):
                break

            if command == "toggle":
                ok, message = controller.toggle()
                _worker_send(
                    connection,
                    ("response", request_id, ok, message, controller.state),
                )
            elif command == "emergency_release":
                controller.emergency_release()
                _worker_send(
                    connection,
                    ("response", request_id, True, "released", controller.state),
                )
            elif command == "shutdown":
                graceful = bool(payload.get("graceful", True))
                ok = controller.shutdown(graceful=graceful)
                _worker_send(
                    connection,
                    ("response", request_id, ok, "shutdown", controller.state),
                )
                keep_running = False
            else:
                _worker_send(
                    connection,
                    ("response", request_id, False, f"unknown command: {command}", controller.state),
                )
    finally:
        if keep_running:
            # Pipe closure or a normal worker-side exception: stop publishing.
            controller.shutdown(graceful=False)
        connection.close()


class KeyboardArmHangProcess:
    """Non-blocking parent-side proxy for the isolated arm DDS worker.

    A CycloneDDS/native-thread stall can freeze the child, but it can no longer
    stop the process that reads lower-body keyboard commands.
    """

    def __init__(
        self,
        iface: str,
        *,
        move_duration: float = DEFAULT_MOVE_DURATION,
        release_duration: float = DEFAULT_RELEASE_DURATION,
        request_timeout: float = WORKER_REQUEST_TIMEOUT_S,
    ):
        self.iface = str(iface)
        self.move_duration = float(move_duration)
        self.release_duration = float(release_duration)
        self.request_timeout = max(0.02, float(request_timeout))
        self._connection = None
        self._process = None
        self._request_id = 0
        self._responses = {}
        self._state = "starting"
        self._last_heartbeat = 0.0
        self._ready = None
        self._ready_message = ""

    def start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_arm_hang_worker_main,
            args=(
                child_connection,
                self.iface,
                self.move_duration,
                self.release_duration,
            ),
            daemon=True,
            name="keyboard_arm_hang_dds",
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and self._ready is None:
            self._drain_messages()
            if self._ready is not None:
                break
            if not process.is_alive():
                break
            time.sleep(0.01)
        if self._ready is not True:
            self._terminate_worker()
            raise RuntimeError(self._ready_message or "arm DDS worker startup timed out")

    @property
    def state(self) -> str:
        self._drain_messages()
        process = self._process
        if process is None or not process.is_alive():
            return "unavailable"
        if (
            self._last_heartbeat > 0.0
            and time.monotonic() - self._last_heartbeat > WORKER_STALE_S
        ):
            return "worker_stalled"
        return self._state

    def _drain_messages(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            while connection.poll():
                message = connection.recv()
                kind = message[0]
                if kind == "ready":
                    _, ok, detail, state = message
                    self._ready = bool(ok)
                    self._ready_message = str(detail)
                    self._state = str(state)
                    self._last_heartbeat = time.monotonic()
                elif kind == "state":
                    _, state, _worker_time = message
                    self._state = str(state)
                    self._last_heartbeat = time.monotonic()
                elif kind == "response":
                    _, request_id, ok, detail, state = message
                    self._responses[int(request_id)] = (bool(ok), str(detail))
                    self._state = str(state)
                    self._last_heartbeat = time.monotonic()
        except (BrokenPipeError, EOFError, OSError):
            self._state = "unavailable"

    def _request(self, command: str, payload=None, timeout: Optional[float] = None):
        connection = self._connection
        process = self._process
        if connection is None or process is None or not process.is_alive():
            return False, "手臂子进程不可用；腿部键盘保持可用"
        self._request_id += 1
        request_id = self._request_id
        try:
            connection.send((request_id, command, payload or {}))
        except (BrokenPipeError, EOFError, OSError):
            return False, "手臂子进程连接已断开；腿部键盘保持可用"

        deadline = time.monotonic() + (
            self.request_timeout if timeout is None else max(0.0, float(timeout))
        )
        while time.monotonic() < deadline:
            self._drain_messages()
            response = self._responses.pop(request_id, None)
            if response is not None:
                return response
            time.sleep(0.005)
        return False, "手臂子进程无响应；腿部键盘保持可用"

    def toggle(self):
        return self._request("toggle")

    def emergency_release(self) -> None:
        self._request("emergency_release", timeout=0.05)

    def _terminate_worker(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        connection = self._connection
        if connection is not None:
            connection.close()
        self._connection = None
        self._process = None

    def shutdown(self, *, graceful: bool = True) -> bool:
        process = self._process
        if process is None:
            return True
        timeout = (
            self.release_duration + DEFAULT_WEIGHT_FADE_DURATION + 1.5
            if graceful else 0.3
        )
        ok, _message = self._request(
            "shutdown",
            {"graceful": bool(graceful)},
            timeout=timeout,
        )
        process.join(timeout=0.5)
        if process.is_alive():
            self._terminate_worker()
            return False
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._process = None
        return bool(ok)
