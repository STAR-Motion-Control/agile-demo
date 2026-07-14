#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THREE_TESTS = HERE
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from adaptive_taptap import AdaptiveTapTapController, StanceMetrics
from schedules import taptap_profile
from sim_dwbc_backend import DwbcBackend


@dataclass(frozen=True)
class Cmd:
    fsm: str = "RL_FULL"
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height: float = 0.74
    fresh: bool = True
    estop: bool = False
    allow_recovery: bool = True


def stance(state) -> StanceMetrics:
    left, right = state["foot_l"], state["foot_r"]
    yaw = float(state["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    world_x = float(left[0] - right[0])
    world_y = float(left[1] - right[1])
    forward = world_x * c + world_y * s
    lateral = -world_x * s + world_y * c

    def quat_yaw(quat) -> float:
        w, x, y, z = (float(value) for value in quat)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    yaw_error = 0.0
    if "foot_l_quat" in state and "foot_r_quat" in state:
        yaw_error = (
            quat_yaw(state["foot_l_quat"]) - quat_yaw(state["foot_r_quat"])
            + math.pi
        ) % (2.0 * math.pi) - math.pi
    return StanceMetrics(
        width=abs(lateral),
        stagger=forward,
        height_delta=abs(float(left[2] - right[2])),
        yaw_error=yaw_error,
    )


def run_case(motion: str, adaptive: bool) -> dict:
    backend = DwbcBackend()
    backend.reset(settle_s=2.5, height=0.74)
    initial_state = backend.state()
    initial = stance(initial_state)
    moves = {
        "fwd": (0.30, 1.0),
        "back": (0.20, 0.8),
        "lat": (0.25, 0.5),
        "turn": (0.40, math.pi / 2),
    }
    speed, amount = moves[motion]
    total, profile = taptap_profile(
        motion=motion,
        speed=speed,
        amount=amount,
        taptap_s=0.0,
        settle_s=2.0,
        soft_start_s=0.6,
        height=0.74,
    )
    move_start, move_end, _ = profile.phases
    total = move_end + 3.0
    controller = AdaptiveTapTapController(
        reference_width=0.24,
        reference_stagger=0.08,
        width_margin=0.035,
        stagger_limit=0.08,
        yaw_limit=0.12,
        debounce_s=0.35,
        confirm_s=0.12,
        min_motion_s=0.40,
        recovery_s=1.60,
        recovery_speed=0.08,
        phase_s=0.40,
        auto_calibrate=False,
    )
    events = []
    recovery_start_state = None
    recovery_end_state = None
    states = []
    steps = int(total / backend.ctrl_dt)
    for tick in range(steps):
        now = tick * backend.ctrl_dt
        requested = profile(now)
        cmd = Cmd(
            vx=float(requested["vx"]),
            vy=float(requested["vy"]),
            wz=float(requested["wz"]),
            height=float(requested["height"]),
        )
        if adaptive:
            cmd, _, event = controller.update(now, cmd, stance(backend.state()))
            if event:
                events.append(event)
                if event == "started":
                    recovery_start_state = backend.state()
                elif event == "completed":
                    recovery_end_state = backend.state()
        backend.step({"vx": cmd.vx, "vy": cmd.vy, "wz": cmd.wz, "height": cmd.height})
        state = backend.state()
        states.append(state)

    final_state = states[-1]
    final = stance(final_state)
    recovery_drift = 0.0
    if recovery_start_state is not None:
        recovery_end_state = recovery_end_state or final_state
        recovery_drift = math.hypot(
            float(recovery_end_state["base_pos"][0] - recovery_start_state["base_pos"][0]),
            float(recovery_end_state["base_pos"][1] - recovery_start_state["base_pos"][1]),
        )
    return {
        "motion": motion,
        "adaptive": adaptive,
        "events": events,
        "triggered": "started" in events,
        "initial_width": initial.width,
        "reference_width": 0.24,
        "reference_stagger": 0.08,
        "final_width": final.width,
        "final_stagger": final.stagger,
        "stagger_error": abs(final.stagger - 0.08),
        "width_error": abs(final.width - 0.24),
        "foot_yaw_error": abs(final.yaw_error),
        "min_width": min(stance(value).width for value in states),
        "recovery_drift": recovery_drift,
        "fell": min(float(value["pelvis_z"]) for value in states) < 0.45,
        "end_x": float(final_state["base_pos"][0]),
        "end_y": float(final_state["base_pos"][1]),
        "end_yaw": float(final_state["yaw"]),
        "move_window": [move_start, move_end],
    }


def run_bad_initial_case() -> dict:
    backend = DwbcBackend()
    backend.reset(settle_s=2.5, height=0.74)
    # Controlled吊架-like bad start: left leg is both too narrow and too far forward.
    backend.d.qpos[7 + 0] -= 0.15
    backend.d.qpos[7 + 1] -= 0.15
    backend.d.qvel[:] = 0.0
    mujoco.mj_forward(backend.m, backend.d)
    initial = stance(backend.state())
    controller = AdaptiveTapTapController(
        reference_width=0.24,
        reference_stagger=0.08,
        width_margin=0.035,
        stagger_limit=0.08,
        yaw_limit=0.12,
        debounce_s=0.35,
        confirm_s=0.12,
        min_motion_s=0.40,
        recovery_s=1.60,
        recovery_speed=0.08,
        phase_s=0.40,
    )
    events = []
    states = []
    for tick in range(int(4.5 / backend.ctrl_dt)):
        now = tick * backend.ctrl_dt
        requested = Cmd(vx=0.30 if now < 2.5 else 0.0)
        effective, _, event = controller.update(now, requested, stance(backend.state()))
        if event:
            events.append(event)
        backend.step({
            "vx": effective.vx,
            "vy": effective.vy,
            "wz": effective.wz,
            "height": effective.height,
        })
        states.append(backend.state())
    final = stance(states[-1])
    return {
        "initial": initial.__dict__,
        "final": final.__dict__,
        "events": events,
        "fell": min(float(value["pelvis_z"]) for value in states) < 0.45,
    }


def main():
    results = []
    for motion in ("fwd", "back", "lat", "turn"):
        baseline = run_case(motion, False)
        adaptive = run_case(motion, True)
        results.extend((baseline, adaptive))
        print(
            f"{motion:4s}: trigger={adaptive['triggered']} "
            f"stagger_err {baseline['stagger_error']:.3f}->{adaptive['stagger_error']:.3f}m "
            f"width_err {baseline['width_error']:.3f}->{adaptive['width_error']:.3f}m "
            f"drift={adaptive['recovery_drift']:.3f}m"
        )

    pairs = {motion: [r for r in results if r["motion"] == motion]
             for motion in ("fwd", "back", "lat", "turn")}

    def bad(item: dict) -> bool:
        return (
            item["width_error"] > 0.035
            or item["stagger_error"] > 0.08
            or item["foot_yaw_error"] > 0.12
        )

    assertions = {
        "no_fall": not any(item["fell"] for item in results),
        "healthy_stops_not_triggered": all(
            not adaptive["triggered"]
            for baseline, adaptive in pairs.values()
            if not bad(baseline)
        ),
        "healthy_path_pose_equivalent": all(
            math.hypot(
                adaptive["end_x"] - baseline["end_x"],
                adaptive["end_y"] - baseline["end_y"],
            ) < 0.005
            and abs(adaptive["end_yaw"] - baseline["end_yaw"]) < 0.01
            for baseline, adaptive in pairs.values()
            if not bad(baseline)
        ),
        "abnormal_stops_triggered": all(
            adaptive["triggered"]
            for baseline, adaptive in pairs.values()
            if bad(baseline)
        ),
        "abnormal_stagger_improved": all(
            adaptive["stagger_error"] < baseline["stagger_error"]
            for baseline, adaptive in pairs.values()
            if bad(baseline)
        ),
        "recovery_drift_bounded": all(
            adaptive["recovery_drift"] < 0.06
            for _, adaptive in pairs.values()
        ),
        "safe_width": min(item["min_width"] for item in results) > 0.18,
    }
    bad_initial = run_bad_initial_case()
    assertions["bad_initial_detected"] = (
        "checking_initial" in bad_initial["events"]
        and "started" in bad_initial["events"]
    )
    assertions["bad_initial_no_fall"] = not bad_initial["fell"]
    assertions["bad_initial_recovered_inside_limits"] = not (
        abs(bad_initial["final"]["width"] - 0.24) > 0.035
        or abs(bad_initial["final"]["stagger"] - 0.08) > 0.08
        or abs(bad_initial["final"]["yaw_error"]) > 0.12
    )
    payload = {
        "results": results,
        "bad_initial": bad_initial,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
    output = HERE / "sim_results" / "adaptive_taptap_v3.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(assertions, indent=2))
    print(f"result={output} passed={payload['passed']}")
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
