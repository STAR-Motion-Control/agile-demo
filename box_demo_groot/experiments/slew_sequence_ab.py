#!/usr/bin/env python3
"""A/B-test command slew limiting against abrupt navigation velocity changes."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "box_demo_1" / "three_tests"
sys.path.insert(0, str(TESTS))

from run_sim_test import _gait_metrics, _stance, fell  # noqa: E402
from sim_dwbc_backend import DwbcBackend  # noqa: E402


SCENARIOS = {
    "turn_reversal": [
        (2.0, (0.25, 0.0, 0.40)),
        (2.0, (0.25, 0.0, -0.40)),
        (2.0, (0.25, 0.0, 0.40)),
        (2.0, (0.25, 0.0, -0.40)),
    ],
    "turn_reversal_mirror": [
        (2.0, (0.25, 0.0, -0.40)),
        (2.0, (0.25, 0.0, 0.40)),
        (2.0, (0.25, 0.0, -0.40)),
        (2.0, (0.25, 0.0, 0.40)),
    ],
    "lateral_reversal": [
        (1.5, (0.20, 0.20, 0.0)),
        (1.5, (0.20, -0.20, 0.0)),
        (1.5, (0.20, 0.20, 0.0)),
        (1.5, (0.20, -0.20, 0.0)),
    ],
    "lateral_reversal_mirror": [
        (1.5, (0.20, -0.20, 0.0)),
        (1.5, (0.20, 0.20, 0.0)),
        (1.5, (0.20, -0.20, 0.0)),
        (1.5, (0.20, 0.20, 0.0)),
    ],
    "turn_forward": [
        (1.5, (0.0, 0.0, 0.40)),
        (1.5, (0.40, 0.0, 0.0)),
        (1.5, (0.0, 0.0, -0.40)),
        (1.5, (0.40, 0.0, 0.0)),
        (1.5, (0.0, 0.0, 0.40)),
        (1.5, (0.40, 0.0, 0.0)),
    ],
    "turn_forward_mirror": [
        (1.5, (0.0, 0.0, -0.40)),
        (1.5, (0.40, 0.0, 0.0)),
        (1.5, (0.0, 0.0, 0.40)),
        (1.5, (0.40, 0.0, 0.0)),
        (1.5, (0.0, 0.0, -0.40)),
        (1.5, (0.40, 0.0, 0.0)),
    ],
}

PROFILES = {
    "abrupt": None,
    "slew_0.8_1.2": (0.8, 1.2),
    "slew_0.5_0.8": (0.5, 0.8),
    "linear_0.8": (0.8, math.inf),
    "linear_0.5": (0.5, math.inf),
    "yaw_1.2": (math.inf, 1.2),
    "yaw_0.8": (math.inf, 0.8),
    "slew_0.8_1.2_tap_pos": (0.8, 1.2),
    "slew_0.8_1.2_tap_neg": (0.8, 1.2),
}


def approach(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    return max(target, value - step)


def run_case(name: str, phases, slew, tap_sign=0.0):
    backend = DwbcBackend()
    backend.reset(settle_s=2.5)
    initial = _stance(backend.state())
    command = [0.0, 0.0, 0.0]
    states = []
    command_integral = [0.0, 0.0, 0.0]
    profile_t = 0.0

    tail = [(2.5, (0.0, 0.0, 0.0))]
    if tap_sign:
        tail = [(0.35, (0.0, 0.0, 0.0))]
        tail += [
            (0.4, (0.0, 0.0, tap_sign * 0.15 * (1.0 if index % 2 == 0 else -1.0)))
            for index in range(4)
        ]
        tail += [(0.55, (0.0, 0.0, 0.0))]
    for duration, target in phases + tail:
        for _ in range(round(duration / backend.ctrl_dt)):
            if slew is None:
                command[:] = target
            else:
                linear_rate, yaw_rate = slew
                command[0] = approach(command[0], target[0], linear_rate * backend.ctrl_dt)
                command[1] = approach(command[1], target[1], linear_rate * backend.ctrl_dt)
                command[2] = approach(command[2], target[2], yaw_rate * backend.ctrl_dt)
            backend.step({
                "vx": command[0], "vy": command[1], "wz": command[2],
                "height": backend.stand_height, "pitch": 0.0,
            })
            state = backend.state()
            state["profile_t"] = profile_t
            states.append(state)
            for axis in range(3):
                command_integral[axis] += command[axis] * backend.ctrl_dt
            profile_t += backend.ctrl_dt

    motion_end = sum(duration for duration, _ in phases)
    gait = _gait_metrics(states, 0.0, motion_end)
    final_window = states[-round(0.8 / backend.ctrl_dt):]
    final_stances = [_stance(state) for state in final_window]
    final_stances = [stance for stance in final_stances if stance is not None]
    final_dx = sum(stance[0] for stance in final_stances) / len(final_stances)
    final_dy = sum(stance[1] for stance in final_stances) / len(final_stances)
    widths = [
        abs(_stance(state)[1])
        for state in states[:round(motion_end / backend.ctrl_dt)]
        if state.get("foot_l_contact") and state.get("foot_r_contact")
    ]
    end = states[-1]
    return {
        "scenario": name,
        "posture_error_m": math.hypot(final_dx - initial[0], final_dy - initial[1]),
        "final_dx_m": final_dx,
        "final_dy_m": final_dy,
        "min_double_support_width_m": min(widths),
        "step_interval_cv": gait["step_interval_cv"],
        "stride_length_cv": gait["stride_length_cv"],
        "endpoint": [float(end["base_pos"][0]), float(end["base_pos"][1]), float(end["yaw"])],
        "command_integral": command_integral,
        "tilt_max_deg": max(float(state["tilt_deg"]) for state in states),
        "fell": any(fell(state) for state in states),
    }


def main():
    output = []
    for scenario, phases in SCENARIOS.items():
        for profile, slew in PROFILES.items():
            tap_sign = -1.0 if profile.endswith("tap_neg") else (
                1.0 if profile.endswith("tap_pos") else 0.0
            )
            result = run_case(scenario, phases, slew, tap_sign=tap_sign)
            result["profile"] = profile
            output.append(result)
            print(
                f"{scenario:18s} {profile:14s} "
                f"posture={result['posture_error_m']*100:5.1f}cm "
                f"minW={result['min_double_support_width_m']*100:5.1f}cm "
                f"stepCV={result['step_interval_cv']:.3f} "
                f"strideCV={result['stride_length_cv']:.3f} "
                f"fell={result['fell']}"
            )
    out = ROOT / "sim_results" / "slew_sequence_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
