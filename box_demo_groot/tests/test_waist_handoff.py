#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class Motor:
    def __init__(self):
        self.mode = 1
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.reserve = 0


class LowCmd:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 5
        self.motor_cmd = [Motor() for _ in range(35)]
        self.reserve = [0] * 4
        self.crc = 0


class CRC:
    def Crc(self, _msg):
        return 0


class Publisher:
    def __init__(self):
        self.messages = []

    def Write(self, msg):
        self.messages.append(msg)


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def load_merger():
    for name in (
        "unitree_sdk2py",
        "unitree_sdk2py.core",
        "unitree_sdk2py.core.channel",
        "unitree_sdk2py.idl",
        "unitree_sdk2py.idl.default",
        "unitree_sdk2py.idl.unitree_hg",
        "unitree_sdk2py.idl.unitree_hg.msg",
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py.utils",
        "unitree_sdk2py.utils.crc",
        "unitree_sdk2py.utils.thread",
    ):
        _module(name)
    channel = sys.modules["unitree_sdk2py.core.channel"]
    channel.ChannelFactoryInitialize = lambda *_a, **_k: None
    channel.ChannelPublisher = object
    channel.ChannelSubscriber = object
    sys.modules["unitree_sdk2py.idl.default"].unitree_hg_msg_dds__LowCmd_ = LowCmd
    sys.modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"].LowCmd_ = LowCmd
    sys.modules["unitree_sdk2py.utils.crc"].CRC = CRC
    sys.modules["unitree_sdk2py.utils.thread"].RecurrentThread = object

    path = Path(__file__).resolve().parents[1] / "merge_lowcmd_arm_sdk.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("merge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_merger()
_REAL_MONOTONIC = M.time.monotonic


@pytest.fixture(autouse=True)
def restore_global_monotonic():
    yield
    # M.time is the process-global stdlib module, so direct test replacement
    # must not leak into later socket/deadline tests.
    M.time.monotonic = _REAL_MONOTONIC


def make_cmd(
    waist_q: float,
    waist_kd: float,
    *,
    arm_enable: float,
    arms_only: bool = False,
) -> LowCmd:
    cmd = LowCmd()
    for i, motor in enumerate(cmd.motor_cmd):
        motor.q = 0.01 * i
        motor.kp = 20.0
        motor.kd = 2.0
    for i in range(12, 15):
        cmd.motor_cmd[i].q = waist_q
        cmd.motor_cmd[i].kp = 250.0
        cmd.motor_cmd[i].kd = waist_kd
    cmd.motor_cmd[15].q = 1.2
    cmd.motor_cmd[29].q = arm_enable
    cmd.motor_cmd[29].dq = 1.0 if arms_only else 0.0
    return cmd


def make_merger(clock, *, waist_arbiter: bool = True):
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        waist_to_rl_on_motion=waist_arbiter,
        waist_takeover_blend_s=0.35,
        arm_runtime_socket=None,
    )
    merger._pub = Publisher()
    merger._ipc_flags = (False, False, clock[0])
    return merger


def tick(merger, clock, *, dt=0.0):
    clock[0] += dt
    merger._rl_t = clock[0]
    if merger._arm is not None:
        merger._arm_t = clock[0]
    merger._ipc_flags = (False, False, clock[0])
    merger._tick()
    return merger._pub.messages[-1]


class FakeCommandReceiver:
    def __init__(self, raw, fresh=True):
        self.raw = raw
        self.fresh = fresh

    def drain(self):
        return 1

    def latest(self, **_kwargs):
        return self.raw, self.fresh


def stream_command(
    sequence,
    *,
    forward=0.0,
    fsm="RL_FULL",
    health_ok=True,
):
    return {
        "broker_id": "broker-a",
        "broker_sequence": sequence,
        "fsm": fsm,
        "units": "agile",
        "motion_bus_health_ok": health_ok,
        "velocity": {
            "forward": forward,
            "lateral": 0.0,
            "yaw": 0.0,
        },
    }


def test_motion_stream_debounces_distinct_broker_frames_and_fails_closed():
    clock = [50.0]
    merger = make_merger(clock)
    receiver = FakeCommandReceiver(stream_command(1, forward=0.2))
    merger._motion_receiver = receiver

    merger._update_stream_flags(clock[0])
    assert merger._ipc_flags[:2] == (False, False)

    # Re-reading one retained frame at 500 Hz is not a second debounce sample.
    merger._update_stream_flags(clock[0] + 0.002)
    assert merger._ipc_flags[:2] == (False, False)

    receiver.raw = stream_command(2, forward=0.2)
    merger._update_stream_flags(clock[0] + 0.05)
    assert merger._ipc_flags[:2] == (True, False)

    receiver.fresh = False
    merger._update_stream_flags(clock[0] + 0.50)
    assert merger._ipc_flags[:2] == (False, False)


def test_motion_stream_marks_non_rl_full_as_bad_immediately():
    clock = [60.0]
    merger = make_merger(clock)
    merger._motion_receiver = FakeCommandReceiver(stream_command(1, fsm="RL_LOWER"))
    merger._update_stream_flags(clock[0])
    assert merger._ipc_flags[:2] == (False, True)


def test_second_merger_cannot_replace_arm_control_socket(tmp_path):
    socket_path = Path(
        f"/tmp/groot-arm-control-owner-{M.os.getpid()}-{M.time.time_ns()}.sock"
    )
    first = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=str(socket_path),
        arm_control_status_file=str(tmp_path / "first-status.json"),
        arm_runtime_socket=None,
    )
    second = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=str(socket_path),
        arm_control_status_file=str(tmp_path / "second-status.json"),
        arm_runtime_socket=None,
    )
    first._bind_arm_control()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            second._bind_arm_control()
        second._close_local_endpoints()
        assert socket_path.exists()
    finally:
        first._close_local_endpoints()


def test_no_arm_path_is_byte_semantically_unchanged_and_refreshes_cache():
    clock = [100.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    merger._waist_slew[14] = (0.50, 1.0)  # 模拟上一次前倾残留
    out = tick(merger, clock)
    assert out.motor_cmd[14].q == 0.05
    assert out.motor_cmd[14].kd == -5.0
    assert merger._waist_slew[14][0] == 0.05


def test_arm_takeover_starts_exactly_from_current_rl_then_blends_all_fields():
    clock = [200.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    tick(merger, clock)  # 先同步 RL 基准
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0)

    first = tick(merger, clock, dt=0.002)
    assert first.motor_cmd[14].q == 0.05
    assert first.motor_cmd[14].kd == -5.0
    assert first.motor_cmd[15].q == 1.2  # 非腰部仍维持原来的立即 arm overlay 语义

    values = []
    for _ in range(7):
        out = tick(merger, clock, dt=0.05)
        values.append((out.motor_cmd[14].q, out.motor_cmd[14].kd))
    assert all(a[0] <= b[0] for a, b in zip(values, values[1:]))
    assert abs(values[-1][0] - 0.50) < 1e-9
    assert abs(values[-1][1] - 5.0) < 1e-9


def test_motion_window_still_keeps_waist_on_rl_while_arms_use_arm_sdk():
    clock = [300.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.06, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0)
    merger._arbiter.state = M.WaistArbiter.RL
    merger._arbiter.alpha = 1.0
    merger._arm_was_active = True
    merger._waist_takeover_alpha = 1.0
    merger._waist_takeover_last_t = clock[0]
    merger._ipc_flags = (True, False, clock[0])
    # 直接调用时保留 motion=True，避免测试 helper 覆盖。
    merger._rl_t = merger._arm_t = clock[0]
    merger._tick()
    out = merger._pub.messages[-1]
    assert out.motor_cmd[14].q == 0.06
    assert out.motor_cmd[14].kd == -5.0
    assert out.motor_cmd[15].q == 1.2


def test_arms_only_overlay_never_takes_waist_even_without_motion_arbiter():
    clock = [350.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock, waist_arbiter=False)
    merger._rl = make_cmd(0.07, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(
        0.50,
        5.0,
        arm_enable=1.0,
        arms_only=True,
    )

    for _ in range(12):
        out = tick(merger, clock, dt=0.05)
        assert out.motor_cmd[14].q == 0.07
        assert out.motor_cmd[14].kd == -5.0
        assert out.motor_cmd[15].q == 1.2
    assert merger._waist_takeover_alpha == 0.0
    assert merger._waist_slew[14][0] == 0.07


def test_integrated_arm_hang_reuses_rl_frame_without_touching_waist(tmp_path):
    clock = [375.0]
    M.time.monotonic = lambda: clock[0]
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_control_status_file=str(tmp_path / "arm-status.json"),
        arm_hang_move_s=1.0,
    )
    merger._pub = Publisher()
    merger._rl = make_cmd(0.08, -5.0, arm_enable=0.0)
    merger._ipc_flags = (False, False, clock[0])
    policy = M.policy_frame_from_lowcmd(merger._rl)
    assert merger._arm_planner.activate(policy, clock[0])

    out = tick(merger, clock, dt=1.0)
    assert out.motor_cmd[14].q == 0.08
    target = tuple(motor.q for motor in merger._arm_planner.last_output.motors)
    assert tuple(out.motor_cmd[i].q for i in range(15, 29)) == target
    assert merger._arm_planner.state == M.ArmHangPlanner.HOLDING


def test_external_arm_owner_preempts_integrated_arm_hang(tmp_path):
    clock = [390.0]
    M.time.monotonic = lambda: clock[0]
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_control_status_file=str(tmp_path / "arm-status.json"),
        arm_runtime_socket=None,
    )
    merger._pub = Publisher()
    merger._rl = make_cmd(0.08, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0, arms_only=True)
    merger._ipc_flags = (False, False, clock[0])
    policy = M.policy_frame_from_lowcmd(merger._rl)
    assert merger._arm_planner.activate(policy, clock[0])

    out = tick(merger, clock, dt=0.01)
    assert merger._arm_planner.state == M.ArmHangPlanner.IDLE
    assert out.motor_cmd[15].q == 1.2


def test_manipulation_runtime_frame_is_applied_without_rt_arm_sdk(tmp_path):
    clock = [395.0]
    M.time.monotonic = lambda: clock[0]
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_runtime_socket=str(tmp_path / "arm-runtime.sock"),
        arm_runtime_status_file=str(tmp_path / "arm-runtime-status.json"),
    )
    merger._pub = Publisher()
    merger._rl = make_cmd(0.08, -5.0, arm_enable=0.0)
    merger._rl_t = clock[0]
    merger._ipc_flags = (False, False, clock[0])
    q = [0.2 + 0.01 * index for index in range(17)]
    merger._robot_modes = (5, 0)
    merger._arm_runtime.update_robot_state(
        mode_machine=5,
        mode_pr=0,
        q=[0.0] * 17,
        tau_est=[0.0] * 17,
        received_at=clock[0],
    )
    merger._arm_runtime.update_policy(q, received_at=clock[0])
    response = merger._arm_runtime.handle_request(
        {
            "schema_version": 1,
            "op": "acquire",
            "source": "manipulation.test",
            "sequence": 1,
            "request_id": "runtime-test",
            "boot_id": merger._arm_runtime.boot_id,
            "lease_s": 0.25,
            "frame": {
                "q": q,
                "weight": 1.0,
                "profile": "manip_v1",
            },
        },
        now=clock[0],
    )
    assert response["ok"] is True

    out = tick(merger, clock, dt=0.002)
    assert out.motor_cmd[15].q == q[0]
    assert out.motor_cmd[15].kp == 150.0
    assert out.motor_cmd[15].kd == 10.0
    assert out.motor_cmd[28].q == q[13]
    assert out.motor_cmd[28].kp == 180.0
    # First waist takeover frame remains exactly on RL for continuity.
    assert out.motor_cmd[14].q == 0.08
    assert merger._arm is None


def test_arm_release_remains_immediate_rl_and_seeds_next_takeover_from_rl():
    clock = [400.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.04, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0)
    merger._arm_was_active = True
    merger._waist_takeover_alpha = 1.0
    merger._waist_takeover_last_t = clock[0]
    tick(merger, clock)

    merger._arm.motor_cmd[29].q = 0.0
    released = tick(merger, clock, dt=0.002)
    assert released.motor_cmd[14].q == 0.04
    assert merger._waist_slew[14][0] == 0.04

    merger._arm.motor_cmd[29].q = 1.0
    reacquired = tick(merger, clock, dt=0.002)
    assert reacquired.motor_cmd[14].q == 0.04


def test_continuous_weight_fades_waist_but_preserves_binary_arm_semantics():
    clock = [500.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.0, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.40, 5.0, arm_enable=0.5)
    merger._arm_was_active = True
    merger._waist_takeover_alpha = 1.0
    merger._waist_takeover_last_t = clock[0]
    out = tick(merger, clock, dt=0.002)
    assert abs(out.motor_cmd[14].q - 0.20) < 1e-9
    assert abs(out.motor_cmd[14].kd - 0.0) < 1e-9
    assert out.motor_cmd[15].q == 1.2


def test_runtime_weight_blends_arm_gains_and_position_to_policy(tmp_path):
    clock = [550.0]
    M.time.monotonic = lambda: clock[0]
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_runtime_socket=str(tmp_path / "arm-runtime.sock"),
        arm_runtime_status_file=str(tmp_path / "arm-runtime-status.json"),
    )
    merger._pub = Publisher()
    merger._rl = make_cmd(0.0, -5.0, arm_enable=0.0)
    merger._rl_t = clock[0]
    merger._ipc_flags = (False, False, clock[0])
    q = [1.0] * 17
    merger._robot_modes = (5, 0)
    merger._arm_runtime.update_robot_state(
        mode_machine=5,
        mode_pr=0,
        q=[0.0] * 17,
        tau_est=[0.0] * 17,
        received_at=clock[0],
    )
    merger._arm_runtime.update_policy(q, received_at=clock[0])
    merger._arm_runtime.current_frame = lambda **_kwargs: M.ArmFrame(
        q=tuple(q),
        weight=0.5,
        profile="manip_v1",
    )

    out = tick(merger, clock, dt=0.002)
    rl_motor = merger._rl.motor_cmd[15]
    assert out.motor_cmd[15].q == 0.5 * (rl_motor.q + 1.0)
    assert out.motor_cmd[15].kp == 0.5 * (rl_motor.kp + 150.0)
    assert out.motor_cmd[15].kd == 0.5 * (rl_motor.kd + 10.0)


def test_runtime_mode_ignores_legacy_arm_sdk_even_when_runtime_is_idle(tmp_path):
    clock = [575.0]
    M.time.monotonic = lambda: clock[0]
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_runtime_socket=str(tmp_path / "arm-runtime.sock"),
        arm_runtime_status_file=str(tmp_path / "arm-runtime-status.json"),
    )
    merger._pub = Publisher()
    merger._rl = make_cmd(0.02, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0)
    merger._rl_t = merger._arm_t = clock[0]
    merger._ipc_flags = (False, False, clock[0])

    out = tick(merger, clock)
    assert out.motor_cmd[15].q == merger._rl.motor_cmd[15].q
    assert merger._arm_runtime.snapshot(clock[0])["runtime"]["external_conflict"]


def test_nonfinite_arm_weight_fails_closed_to_rl():
    clock = [600.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.03, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=float("nan"))
    out = tick(merger, clock)
    assert out.motor_cmd[14].q == 0.03
    assert out.motor_cmd[15].q == merger._rl.motor_cmd[15].q


def _motor_signature(command):
    return [
        (
            motor.mode,
            motor.q,
            motor.dq,
            motor.tau,
            motor.kp,
            motor.kd,
            motor.reserve,
        )
        for motor in command.motor_cmd
    ]


def test_damp_frame_bypasses_runtime_external_and_integrated_arm_overlays(tmp_path):
    clock = [700.0]
    M.time.monotonic = lambda: clock[0]
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_runtime_socket=str(tmp_path / "arm-runtime.sock"),
        arm_runtime_status_file=str(tmp_path / "arm-runtime-status.json"),
    )
    merger._pub = Publisher()
    normal = make_cmd(0.08, -5.0, arm_enable=0.0)
    merger._rl = normal
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0)
    merger._rl_t = merger._arm_t = clock[0]
    merger._ipc_flags = (True, False, clock[0])

    q = [0.3] * 17
    merger._arm_runtime.update_robot_state(
        mode_machine=5,
        mode_pr=0,
        q=[0.0] * 17,
        tau_est=[0.0] * 17,
        received_at=clock[0],
    )
    merger._arm_runtime.update_policy(q, received_at=clock[0])
    acquired = merger._arm_runtime.handle_request(
        {
            "schema_version": 1,
            "op": "acquire",
            "source": "manipulation.safety-test",
            "sequence": 1,
            "request_id": "safety-test",
            "boot_id": merger._arm_runtime.boot_id,
            "lease_s": 0.25,
            "frame": {"q": q, "weight": 1.0, "profile": "manip_v1"},
        },
        now=clock[0],
    )
    assert acquired["ok"] is True
    assert merger._arm_planner.activate(M.policy_frame_from_lowcmd(normal), clock[0])

    damp = make_cmd(0.11, 8.0, arm_enable=0.0)
    for motor in damp.motor_cmd[:29]:
        motor.kp = 0.0
        motor.kd = 8.0
    expected = _motor_signature(damp)
    merger._rl = damp

    out = tick(merger, clock, dt=0.002)
    assert M._rl_frame_is_safety(damp) is True
    assert _motor_signature(out) == expected
    assert merger._arm_planner.state == M.ArmHangPlanner.IDLE
    assert merger._arm_runtime.current_frame(
        now=clock[0],
        external_active=False,
    ) is None
    runtime = merger._arm_runtime.snapshot(clock[0])["runtime"]
    assert runtime["owner"] is None
    assert merger._arm_runtime._metrics["safety_stops"] == 1


def test_rl_lower_is_not_misclassified_as_global_safety():
    command = make_cmd(0.04, -5.0, arm_enable=0.0)
    motion, bad, safety = M.command_runtime_flags(
        stream_command(1, fsm="RL_LOWER"),
        True,
    )
    assert (motion, bad, safety) == (False, True, False)
    assert M._rl_frame_is_safety(command) is False


def test_unhealthy_motion_plane_forces_whole_body_damp():
    clock = [750.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=1.0)
    merger._motion_receiver = FakeCommandReceiver(
        stream_command(1, health_ok=False)
    )
    policy = M.policy_frame_from_lowcmd(merger._rl)
    assert merger._arm_planner.activate(policy, clock[0])

    out = tick(merger, clock, dt=0.002)
    assert merger._motion_stream_fresh is True
    assert merger._motion_health_ok is False
    assert all(motor.kp == 0.0 for motor in out.motor_cmd)
    assert all(motor.kd == M.DEFAULT_SAFETY_DAMPING_KD for motor in out.motor_cmd)
    assert merger._arm_planner.state == M.ArmHangPlanner.IDLE


def test_stream_damp_replaces_cached_stiff_rl_frame():
    clock = [800.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    merger._motion_receiver = FakeCommandReceiver(
        stream_command(1, fsm="DAMP")
    )

    out = tick(merger, clock, dt=0.002)

    assert merger._motion_safety_fsm == "DAMP"
    assert all(motor.kp == 0.0 for motor in out.motor_cmd)
    assert all(motor.kd == M.DEFAULT_SAFETY_DAMPING_KD for motor in out.motor_cmd)
    assert all(motor.dq == 0.0 for motor in out.motor_cmd)


def test_stream_limp_survives_stream_loss_until_fresh_release():
    clock = [850.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    receiver = FakeCommandReceiver(stream_command(1, fsm="LIMP"))
    merger._motion_receiver = receiver

    first = tick(merger, clock, dt=0.002)
    assert all(motor.kp == 0.0 and motor.kd == 0.0 for motor in first.motor_cmd)
    assert first.motor_cmd[0].q == M.POS_STOP_F
    assert first.motor_cmd[0].dq == M.VEL_STOP_F
    assert first.motor_cmd[29].q == 0.0

    receiver.fresh = False
    retained = tick(merger, clock, dt=0.50)
    assert merger._motion_safety_fsm == "LIMP"
    assert retained.motor_cmd[0].q == M.POS_STOP_F

    receiver.fresh = True
    receiver.raw = stream_command(2)
    released = tick(merger, clock, dt=0.002)
    assert merger._motion_safety_fsm is None
    assert released.motor_cmd[0].kp == merger._rl.motor_cmd[0].kp


def test_tick_gap_forces_damp_and_reports_unhealthy(tmp_path):
    clock = [900.0]
    M.time.monotonic = lambda: clock[0]
    health_path = tmp_path / "merger-health.json"
    merger = M.Merger(
        iface="fake",
        hz=500.0,
        weight_threshold=1e-3,
        arm_stale_s=0.25,
        rl_stale_s=0.5,
        arm_control_socket=None,
        arm_runtime_socket=None,
        runtime_health_file=str(health_path),
        max_tick_gap_s=0.060,
    )
    merger._pub = Publisher()
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    merger._rl_t = clock[0]
    merger._tick_guarded()

    clock[0] += 0.100
    merger._rl_t = clock[0]
    merger._tick_guarded()

    out = merger._pub.messages[-1]
    status = json.loads(health_path.read_text())
    assert all(motor.kp == 0.0 for motor in out.motor_cmd)
    assert status["healthy"] is False
    assert status["reason"] == "tick_gap_high"
    assert status["tick_gap_ms"] == pytest.approx(100.0)


def test_worker_exception_attempts_final_damp_frame():
    clock = [950.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.05, -5.0, arm_enable=0.0)
    merger._rl_t = clock[0]

    def fail_tick():
        raise RuntimeError("synthetic tick failure")

    merger._tick = fail_tick
    with pytest.raises(RuntimeError, match="synthetic tick failure"):
        merger._tick_guarded()

    assert len(merger._pub.messages) == 3
    assert all(motor.kp == 0.0 for motor in merger._pub.messages[-1].motor_cmd)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ALL PASS ({len(tests)} tests)")
