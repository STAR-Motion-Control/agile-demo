#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_taptap import StanceMetrics  # noqa: E402
from lateral_stance_guard import LateralStanceGuard  # noqa: E402


@dataclass(frozen=True)
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


def stance(width: float) -> StanceMetrics:
    return StanceMetrics(width=width, stagger=0.0, height_delta=0.0)


def test_healthy_lateral_command_passes_unchanged():
    guard = LateralStanceGuard(min_width=0.15, guard_margin=0.03)
    cmd = Cmd(vy=0.2)
    out, blocked, event = guard.update(0.0, cmd, stance(0.24))
    assert out == cmd
    assert not blocked
    assert event is None


def test_hard_margin_blocks_and_latches_lateral_only():
    guard = LateralStanceGuard(min_width=0.15, guard_margin=0.03)
    out, blocked, event = guard.update(0.0, Cmd(vx=0.1, vy=0.2), stance(0.18))
    assert blocked and event == "blocked"
    assert out.vx == 0.1 and out.vy == 0.0

    out, blocked, event = guard.update(0.1, Cmd(vy=-0.2), stance(0.19))
    assert blocked and event is None and out.vy == 0.0


def test_shrinking_width_is_blocked_predictively():
    guard = LateralStanceGuard(
        min_width=0.15,
        guard_margin=0.03,
        prediction_s=0.12,
        rate_alpha=1.0,
    )
    guard.update(0.0, Cmd(vy=0.2), stance(0.24))
    out, blocked, event = guard.update(0.1, Cmd(vy=0.2), stance(0.20))
    assert guard.width_rate < 0.0
    assert blocked and event == "blocked" and out.vy == 0.0


def test_release_requires_explicit_zero_lateral_and_safe_width():
    guard = LateralStanceGuard(min_width=0.15, guard_margin=0.03)
    guard.update(0.0, Cmd(vy=0.2), stance(0.17))
    _, blocked, event = guard.update(0.1, Cmd(vy=0.2), stance(0.21))
    assert blocked and event is None
    out, blocked, event = guard.update(0.2, Cmd(), stance(0.21))
    assert out == Cmd()
    assert not blocked and event == "released"


def test_forward_motion_is_not_blocked_by_lateral_guard():
    guard = LateralStanceGuard(min_width=0.15, guard_margin=0.03)
    out, blocked, event = guard.update(0.0, Cmd(vx=0.4), stance(0.10))
    assert out.vx == 0.4
    assert not blocked and event is None
