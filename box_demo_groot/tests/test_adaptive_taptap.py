#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_taptap import AdaptiveTapTapController, StanceMetrics


@dataclass(frozen=True)
class Cmd:
    fsm: str = "RL_FULL"
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height: float = 0.76
    fresh: bool = True
    estop: bool = False
    allow_recovery: bool = False


GOOD = StanceMetrics(width=0.24, stagger=0.03, height_delta=0.0)
BAD = StanceMetrics(width=0.24, stagger=-0.08, height_delta=0.0)


def controller():
    result = AdaptiveTapTapController(
        reference_width=0.24,
        debounce_s=0.10,
        confirm_s=0.10,
        min_motion_s=0.20,
        recovery_s=0.80,
        recovery_speed=0.08,
        phase_s=0.20,
        calibration_s=10.0,
    )
    result.reference_stagger = GOOD.stagger
    return result


def arm_stop(c, stance):
    c.update(0.0, Cmd(vx=0.4, allow_recovery=True), GOOD)
    c.update(0.25, Cmd(vx=0.4, allow_recovery=True), GOOD)
    out, active, event = c.update(0.30, Cmd(allow_recovery=True), stance)
    assert not active and event == "checking" and out.vx == 0.0


def test_healthy_stop_skips_recovery():
    c = controller()
    arm_stop(c, GOOD)
    c.update(0.41, Cmd(allow_recovery=True), GOOD)
    out, active, event = c.update(0.51, Cmd(allow_recovery=True), GOOD)
    assert not active and event == "healthy" and out.vx == 0.0


def test_zero_stance_calibrates_signed_reference():
    c = AdaptiveTapTapController(calibration_s=0.20)
    neutral = StanceMetrics(width=0.25, stagger=0.04, height_delta=0.0)
    for now in (0.0, 0.10, 0.21):
        c.update(now, Cmd(allow_recovery=True), neutral)
    assert c.reference_width == 0.25
    assert c.reference_stagger == 0.04
    assert not c.stance_bad(neutral)


def test_bad_stance_runs_symmetric_recovery():
    c = controller()
    arm_stop(c, BAD)
    c.update(0.41, Cmd(allow_recovery=True), BAD)
    _, active, event = c.update(0.51, Cmd(allow_recovery=True), BAD)
    assert active and event == "started"
    values = []
    for now in (0.52, 0.72, 0.92, 1.12):
        out, active, _ = c.update(now, Cmd(allow_recovery=True), BAD)
        assert active
        values.append(out.vx)
    assert values == [0.08, -0.08, 0.08, -0.08]
    out, active, event = c.update(1.32, Cmd(allow_recovery=True), GOOD)
    assert not active and event == "completed" and out.vx == 0.0


def test_safety_and_unmarked_zero_never_recover():
    c = controller()
    c.update(0.0, Cmd(vx=0.4, allow_recovery=True), GOOD)
    c.update(0.3, Cmd(vx=0.4, allow_recovery=True), GOOD)
    _, active, event = c.update(0.4, Cmd(allow_recovery=False), BAD)
    assert not active and event is None and c.state == c.IDLE

    arm_stop(c, BAD)
    c.update(0.41, Cmd(allow_recovery=True), BAD)
    c.update(0.51, Cmd(allow_recovery=True), BAD)
    out, active, event = c.update(
        0.60, Cmd(fsm="DAMP", estop=True, fresh=True), BAD
    )
    assert not active and event == "cancelled" and out.estop


def test_new_normal_command_is_held_until_recovery_finishes():
    c = controller()
    arm_stop(c, BAD)
    c.update(0.41, Cmd(allow_recovery=True), BAD)
    c.update(0.51, Cmd(allow_recovery=True), BAD)
    pending = Cmd(vy=0.2, allow_recovery=True)
    out, active, _ = c.update(0.60, pending, BAD)
    assert active and out.vy == 0.0
    out, active, event = c.update(1.32, pending, GOOD)
    assert not active and event == "completed" and out.vy == 0.2


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ALL PASS ({len(tests)} tests)")
