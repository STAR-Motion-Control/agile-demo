#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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
    spec = importlib.util.spec_from_file_location("merge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_merger()


def make_cmd(waist_q: float, waist_kd: float, *, arm_enable: float) -> LowCmd:
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


def test_nonfinite_arm_weight_fails_closed_to_rl():
    clock = [600.0]
    M.time.monotonic = lambda: clock[0]
    merger = make_merger(clock)
    merger._rl = make_cmd(0.03, -5.0, arm_enable=0.0)
    merger._arm = make_cmd(0.50, 5.0, arm_enable=float("nan"))
    out = tick(merger, clock)
    assert out.motor_cmd[14].q == 0.03
    assert out.motor_cmd[15].q == merger._rl.motor_cmd[15].q


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ALL PASS ({len(tests)} tests)")
