#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

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
    return StanceMetrics(
        width=abs(lateral),
        stagger=forward,
        height_delta=abs(float(left[2] - right[2])),
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
        reference_width=initial.width,
        width_margin=0.035,
        stagger_limit=0.08,
        debounce_s=0.35,
        confirm_s=0.12,
        min_motion_s=0.40,
        recovery_s=1.60,
        recovery_speed=0.08,
        phase_s=0.40,
        calibration_s=0.30,
    )
    events = []
    recovery_start_state = None
    recovery_end_state = None
    states = []
    reference_samples = []
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
        if move_start - 0.30 <= now < move_start:
            reference_samples.append(stance(state))

    final_state = states[-1]
    final = stance(final_state)
    reference_stagger = sum(x.stagger for x in reference_samples) / len(reference_samples)
    reference_width = sum(x.width for x in reference_samples) / len(reference_samples)
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
        "reference_width": reference_width,
        "reference_stagger": reference_stagger,
        "final_width": final.width,
        "final_stagger": final.stagger,
        "stagger_error": abs(final.stagger - reference_stagger),
        "width_error": abs(final.width - reference_width),
        "min_width": min(stance(value).width for value in states),
        "recovery_drift": recovery_drift,
        "fell": min(float(value["pelvis_z"]) for value in states) < 0.45,
        "end_x": float(final_state["base_pos"][0]),
        "end_y": float(final_state["base_pos"][1]),
        "end_yaw": float(final_state["yaw"]),
        "move_window": [move_start, move_end],
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
    assertions = {
        "no_fall": not any(item["fell"] for item in results),
        "forward_triggered": pairs["fwd"][1]["triggered"],
        "forward_stagger_improved": (
            pairs["fwd"][1]["stagger_error"]
            < pairs["fwd"][0]["stagger_error"] * 0.80
        ),
        "turn_triggered": pairs["turn"][1]["triggered"],
        "turn_stagger_improved": (
            pairs["turn"][1]["stagger_error"]
            < pairs["turn"][0]["stagger_error"] * 0.90
        ),
        "back_not_triggered": not pairs["back"][1]["triggered"],
        "lateral_not_triggered": not pairs["lat"][1]["triggered"],
        "safe_width": min(item["min_width"] for item in results) > 0.18,
        "recovery_drift_bounded": max(
            item["recovery_drift"] for item in results if item["adaptive"]
        ) < 0.06,
    }
    payload = {"results": results, "assertions": assertions, "passed": all(assertions.values())}
    output = HERE / "sim_results" / "adaptive_taptap_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(assertions, indent=2))
    print(f"result={output} passed={payload['passed']}")
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
