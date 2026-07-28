#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import groot_wbc_boxdemo_adapter as A  # noqa: E402


def test_real_hardware_defaults_to_raw_arm_observation():
    args = A.build_arg_parser().parse_args([])
    assert args.h_arm_observation_compensation == 0.0
    assert not args.disable_h_arm_observation_neutralization


def test_neutralization_changes_only_policy_arm_q_and_dq():
    q = np.arange(29, dtype=np.float32)
    dq = -np.arange(29, dtype=np.float32)
    observation = {
        "q": q,
        "dq": dq,
        "floating_base_pose": np.arange(7, dtype=np.float32),
    }
    arms = np.arange(15, 29)
    defaults = np.linspace(-0.2, 0.2, 14, dtype=np.float32)

    policy = A.neutralize_arm_observation(observation, arms, defaults)

    np.testing.assert_array_equal(policy["q"][:15], q[:15])
    np.testing.assert_array_equal(policy["dq"][:15], dq[:15])
    np.testing.assert_allclose(policy["q"][arms], defaults)
    np.testing.assert_array_equal(policy["dq"][arms], 0.0)
    assert policy["floating_base_pose"] is observation["floating_base_pose"]
    np.testing.assert_array_equal(observation["q"], q)
    np.testing.assert_array_equal(observation["dq"], dq)
    assert policy["q"] is not observation["q"]
    assert policy["dq"] is not observation["dq"]


def test_partial_compensation_retains_required_physical_feedback():
    q = np.ones(29, dtype=np.float32)
    dq = np.full(29, 2.0, dtype=np.float32)
    arms = np.arange(15, 29)
    policy = A.neutralize_arm_observation(
        {"q": q, "dq": dq},
        arms,
        np.zeros(14, dtype=np.float32),
        ratio=0.40,
    )
    np.testing.assert_allclose(policy["q"][arms], 0.60)
    np.testing.assert_allclose(policy["dq"][arms], 1.20)
    np.testing.assert_allclose(policy["q"][:15], 1.0)
    np.testing.assert_allclose(policy["dq"][:15], 2.0)


class _Enable:
    q = 1.0
    dq = 1.0
    reserve = A.KEYBOARD_ARM_OWNER_MARKER


class _Message:
    motor_cmd = [object() for _ in range(A.ARM_SDK_ENABLE_SLOT)] + [_Enable()]


def test_monitor_requires_fresh_keyboard_h_arms_only_owner():
    _Enable.q = 1.0
    _Enable.dq = 1.0
    _Enable.reserve = A.KEYBOARD_ARM_OWNER_MARKER
    monitor = A.KeyboardArmOverlayMonitor(stale_s=0.25)
    monitor._on_arm_sdk(_Message())
    sample_t = monitor._sample[0]
    assert monitor.active(now=sample_t + 0.24)
    assert not monitor.active(now=sample_t + 0.26)

    _Enable.dq = 0.0
    monitor._on_arm_sdk(_Message())
    assert not monitor.active(now=monitor._sample[0])

    _Enable.dq = 1.0
    _Enable.reserve = 0
    monitor._on_arm_sdk(_Message())
    assert not monitor.active(now=monitor._sample[0])

    _Enable.reserve = A.KEYBOARD_ARM_OWNER_MARKER
    _Enable.q = 0.0
    monitor._on_arm_sdk(_Message())
    assert not monitor.active(now=monitor._sample[0])


def test_monitor_ignores_invalid_enable_values():
    _Enable.q = math.nan
    _Enable.dq = 1.0
    _Enable.reserve = A.KEYBOARD_ARM_OWNER_MARKER
    monitor = A.KeyboardArmOverlayMonitor()
    before = monitor._sample
    monitor._on_arm_sdk(_Message())
    assert monitor._sample == before
