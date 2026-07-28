#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import keyboard_arm_hang as H  # noqa: E402


def make_policy(offset: float = 0.0) -> H.UpperBodyFrame:
    motors = []
    for i in range(len(H.ARM_JOINTS)):
        motors.append(
            H.MotorFrame(
                mode=1,
                q=offset + 0.01 * i,
                dq=0.001 * i,
                tau=-0.002 * i,
                kp=30.0 + i,
                kd=2.0 + 0.1 * i,
                reserve=i,
            )
        )
    return H.UpperBodyFrame(mode_machine=5, mode_pr=1, motors=tuple(motors))


def test_target_is_exact_requested_pose():
    policy = make_policy()
    planner = H.ArmHangPlanner(move_duration=3.0)
    assert planner.activate(policy, 10.0)
    output = planner.step(13.0, policy)
    assert output is not None
    assert tuple(m.q for m in output.motors) == H.RL_LOWER_HANDOFF_Q
    assert tuple(m.kp for m in output.motors) == H.TARGET_KP
    assert tuple(m.kd for m in output.motors) == H.TARGET_KD
    assert planner.state == H.ArmHangPlanner.HOLDING


def test_hang_owns_exactly_fourteen_arm_joints_and_never_waist():
    assert H.ARM_JOINTS == tuple(range(15, 29))
    assert len(H.RL_LOWER_HANDOFF_Q) == 14
    assert len(H.TARGET_KP) == 14
    assert len(H.TARGET_KD) == 14
    assert not {12, 13, 14}.intersection(H.ARM_JOINTS)


def test_hang_gains_match_deployed_gr00t_dwbc_arm_impedance():
    expected_kp = (115.0, 115.0, 46.0, 46.0, 23.0, 23.0, 23.0) * 2
    expected_kd = (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0) * 2
    assert H.TARGET_KP == expected_kp
    assert H.TARGET_KD == expected_kd


def test_hang_uses_locomotion_compatible_shoulder_pitch():
    assert H.RL_LOWER_HANDOFF_Q[0] == 0.05
    assert H.RL_LOWER_HANDOFF_Q[7] == 0.05


def test_state_read_never_waits_for_dds_lock():
    controller = H.KeyboardArmHangController("unused")
    controller._lock.acquire()
    try:
        started = time.monotonic()
        assert controller.state == H.ArmHangPlanner.IDLE
        assert time.monotonic() - started < 0.01
    finally:
        controller._lock.release()


def test_toggle_fails_fast_instead_of_blocking_lower_body_keyboard():
    controller = H.KeyboardArmHangController("unused")
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with controller._lock:
            locked.set()
            release.wait(timeout=1.0)

    worker = threading.Thread(target=hold_lock)
    worker.start()
    assert locked.wait(timeout=0.2)
    try:
        started = time.monotonic()
        ok, message = controller.toggle()
        assert not ok
        assert "腿部键盘保持可用" in message
        assert time.monotonic() - started < 0.20
    finally:
        release.set()
        worker.join(timeout=0.2)


def test_arm_owner_callback_ignores_self_and_tracks_external_writer():
    class Enable:
        q = 1.0
        reserve = H.KEYBOARD_ARM_OWNER_MARKER

    class Message:
        motor_cmd = [object() for _ in range(H.ARM_SDK_ENABLE)] + [Enable()]

    controller = H.KeyboardArmHangController("unused")
    controller._on_arm_sdk(Message())
    assert controller._latest_external_arm_t == 0.0

    Message.motor_cmd[H.ARM_SDK_ENABLE].reserve = 0
    controller._on_arm_sdk(Message())
    assert controller._latest_external_arm_weight == 1.0
    assert controller._latest_external_arm_t > 0.0


def test_stalled_arm_process_never_blocks_parent_keyboard():
    class AliveProcess:
        @staticmethod
        def is_alive():
            return True

    class SilentConnection:
        def __init__(self):
            self.sent = []

        @staticmethod
        def poll():
            return False

        def send(self, message):
            self.sent.append(message)

    proxy = H.KeyboardArmHangProcess("unused", request_timeout=0.02)
    proxy._process = AliveProcess()
    proxy._connection = SilentConnection()
    proxy._state = H.ArmHangPlanner.HOLDING
    proxy._last_heartbeat = time.monotonic() - 1.0

    started = time.monotonic()
    assert proxy.state == "worker_stalled"
    ok, message = proxy.toggle()
    elapsed = time.monotonic() - started

    assert not ok
    assert "腿部键盘保持可用" in message
    assert elapsed < 0.10
    assert len(proxy._connection.sent) == 1


def test_first_takeover_frame_exactly_matches_live_policy_command():
    policy = make_policy(0.2)
    planner = H.ArmHangPlanner(move_duration=3.0)
    assert planner.activate(policy, 20.0)
    first = planner.step(20.0, policy)
    assert first == policy
    assert first.weight == 1.0


def test_minimum_jerk_has_no_endpoint_velocity_or_acceleration():
    eps = 1e-4
    assert H.minimum_jerk_ratio(0.0) == 0.0
    assert H.minimum_jerk_ratio(1.0) == 1.0
    assert H.minimum_jerk_ratio(eps) < 2.0e-11
    assert 1.0 - H.minimum_jerk_ratio(1.0 - eps) < 2.0e-11


def test_commanded_position_rate_is_bounded_and_monotonic():
    policy = make_policy(-0.3)
    planner = H.ArmHangPlanner(move_duration=3.0)
    assert planner.activate(policy, 0.0)
    samples = []
    t = 0.0
    while t <= 3.0:
        output = planner.step(t, policy)
        assert output is not None
        samples.append(output.motors[3].q)
        t += H.CONTROL_DT
    delta = H.RL_LOWER_HANDOFF_Q[3] - policy.motors[3].q
    assert all(a <= b for a, b in zip(samples, samples[1:]))
    max_step = max(abs(b - a) for a, b in zip(samples, samples[1:]))
    theoretical = 1.875 * abs(delta) / 3.0 * H.CONTROL_DT
    assert max_step <= theoretical * 1.01


def test_release_aligns_all_fields_before_binary_arm_handoff():
    first_policy = make_policy(-0.2)
    second_policy = make_policy(0.4)
    planner = H.ArmHangPlanner(
        move_duration=1.0,
        release_duration=2.5,
        fade_duration=0.35,
    )
    assert planner.activate(first_policy, 0.0)
    planner.step(1.0, first_policy)
    assert planner.release(second_policy, 2.0)
    aligned = planner.step(4.5, second_policy)
    assert aligned == second_policy
    assert planner.state == H.ArmHangPlanner.FADING

    released = planner.step(4.86, second_policy)
    assert released is not None
    assert released.motors == second_policy.motors
    assert math.isclose(released.weight, 0.0, abs_tol=1e-12)
    assert planner.state == H.ArmHangPlanner.IDLE


def test_stale_policy_pauses_release_instead_of_catching_up_with_a_jump():
    policy = make_policy(-0.1)
    new_policy = make_policy(0.5)
    planner = H.ArmHangPlanner(move_duration=1.0, release_duration=2.5)
    assert planner.activate(policy, 0.0)
    planner.step(1.0, policy)
    assert planner.release(new_policy, 2.0)
    before_stale = planner.step(2.5, new_policy)
    assert before_stale is not None
    held = planner.step(7.5, None)
    assert held == before_stale
    resumed = planner.step(7.5 + H.CONTROL_DT, new_policy)
    assert resumed is not None
    # Only one 20 ms interpolation step may be added after a 5 s outage.
    max_change = max(
        abs(a.q - b.q) for a, b in zip(before_stale.motors, resumed.motors)
    )
    assert max_change < 0.02


def test_invalid_policy_refuses_takeover():
    policy = make_policy()
    motors = list(policy.motors)
    motors[0] = H.MotorFrame(1, float("nan"), 0.0, 0.0, 30.0, 2.0)
    invalid = H.UpperBodyFrame(5, 1, tuple(motors))
    planner = H.ArmHangPlanner()
    assert not planner.activate(invalid, 0.0)
    assert planner.step(0.0, invalid) is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ALL PASS ({len(tests)} tests)")
