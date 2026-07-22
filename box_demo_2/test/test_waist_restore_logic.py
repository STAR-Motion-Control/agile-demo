#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path


ARM_JOINTS = list(range(15, 29)) + [12, 13, 14]


class MotorState:
    def __init__(self, q=0.0, dq=0.0):
        self.q = q
        self.dq = dq


class LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(30)]


class MotorCmd:
    def __init__(self, q=0.0):
        self.q = q


class LowCmd:
    def __init__(self):
        self.motor_cmd = [MotorCmd() for _ in range(35)]


POLICY_MSG = LowCmd()


class ChannelSubscriber:
    def __init__(self, _topic, _type):
        pass

    def Init(self, callback, _queue):
        callback(POLICY_MSG)


class Keeper:
    def __init__(self, state):
        self.low_state = state

    def _ensure_dds(self):
        pass


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def load_common():
    for name in (
        "unitree_sdk2py",
        "unitree_sdk2py.core",
        "unitree_sdk2py.core.channel",
        "unitree_sdk2py.idl",
        "unitree_sdk2py.idl.unitree_hg",
        "unitree_sdk2py.idl.unitree_hg.msg",
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
    ):
        _module(name)
    sys.modules["unitree_sdk2py.core.channel"].ChannelSubscriber = ChannelSubscriber
    sys.modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"].LowCmd_ = LowCmd

    arm = _module("arm_natural_hang")
    arm.DT = 0.001
    arm.ArmNaturalHangKeeper = Keeper
    arm.publish_arm_sdk = lambda *_a, **_k: None
    arm.read_arm_q_from_state = lambda keeper: [
        float(keeper.low_state.motor_state[int(j)].q) for j in ARM_JOINTS
    ]
    arm.read_policy_q = lambda msg: [float(msg.motor_cmd[int(j)].q) for j in ARM_JOINTS]
    dual = _module("dual_arm_target_reach")
    dual.ARM_JOINTS = ARM_JOINTS

    path = Path(__file__).resolve().parent / "_arm_test_common.py"
    spec = importlib.util.spec_from_file_location("arm_common_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C = load_common()


def arm_order_q(state):
    return [float(state.motor_state[int(j)].q) for j in ARM_JOINTS]


def test_pose_restore_accepts_standing_reference():
    state = LowState()
    state.motor_state[14].q = 0.10
    reference = arm_order_q(state)
    assert C.wait_waist_pose_restore(
        Keeper(state), reference, timeout=0.05, hold_s=0.0
    )


def test_pose_restore_accepts_squat_specific_reference_without_forcing_stand():
    state = LowState()
    state.motor_state[12].q = 0.02
    state.motor_state[13].q = -0.03
    state.motor_state[14].q = 0.16
    reference = arm_order_q(state)
    assert C.wait_waist_pose_restore(
        Keeper(state), reference, timeout=0.05, hold_s=0.0
    )


def test_pose_restore_rejects_residual_forward_lean():
    state = LowState()
    state.motor_state[14].q = 0.20
    reference = arm_order_q(state)
    reference[16] = 0.10
    assert not C.wait_waist_pose_restore(
        Keeper(state), reference, timeout=0.01, hold_s=0.0
    )


def test_policy_alignment_cannot_mask_wrong_absolute_pose():
    state = LowState()
    state.motor_state[14].q = 0.50
    POLICY_MSG.motor_cmd[14].q = 0.46  # 旧判据 err=0.04<0.06，会误判成功
    reference = arm_order_q(state)
    reference[16] = 0.10
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        C.wait_waist_policy_settle(
            Keeper(state),
            timeout=0.01,
            min_wait=0.0,
            q_reference=reference,
            reference_eps=0.03,
        )
    assert "腰角已对齐" not in output.getvalue()
    assert "腰角未完全对齐" in output.getvalue()


def test_policy_snapshot_uses_command_not_physical_state():
    POLICY_MSG.motor_cmd[14].q = 0.05
    snapshot = C.read_policy_q_snapshot(timeout=0.05)
    assert snapshot[16] == 0.05


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ALL PASS ({len(tests)} tests)")
