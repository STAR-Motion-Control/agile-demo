#!/usr/bin/env python3
"""Run a deterministic navigation route through GrootMover and MuJoCo.

This is an offline validation harness. It imports the repository's production
``GrootMover`` through ``sim2sim_groot_mover.SimMover`` and sends every motion
command to the GR00T-WBC MuJoCo policy simulation. It does not import or start
DDS, ROS, a robot SDK, a camera, or any robot-control process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


NAVIGATION_MOVER_CONFIG = {
    "fwd_cruise": 0.40,
    "back_cruise": 0.20,
    "lat_cruise": 0.20,
    "yaw_cruise": 0.40,
    "fwd_max": 0.50,
    "back_max": 0.20,
    "lat_max": 0.40,
    "yaw_max": 0.60,
    "v_floor": 0.10,
    "w_floor": 0.10,
    "min_duration": 1.0,
    "min_distance": 0.08,
    "warmup_time": 0.6,
    "warmup_speed": 0.15,
    "settle_before_s": 0.0,
    "stop_hold_s": 0.4,
    "walk_min_height": 0.72,
    "auto_raise_for_walk": False,
    "dist_gain": 1.0,
    "recover_each_move": False,
}

ROUTE = (
    ("move_forward", 0.25),
    ("rotate", 0.35),
    ("move_left", 0.15),
    ("move_forward", -0.12),
    ("rotate", -0.20),
    ("move_left", -0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--initial-settle-s", type=float, default=2.5)
    parser.add_argument("--final-settle-s", type=float, default=2.0)
    return parser.parse_args()


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def rounded(value: float) -> float:
    return round(float(value), 12)


def pose(sim: Any, yaw_from_quat: Any) -> dict[str, float]:
    position = sim.base()
    return {
        "x_m": rounded(position[0]),
        "y_m": rounded(position[1]),
        "z_m": rounded(position[2]),
        "yaw_rad": rounded(yaw_from_quat(sim.d.qpos[3:7])),
    }


def delta_pose(current: dict[str, float], origin: dict[str, float]) -> dict[str, float]:
    return {
        "x_m": rounded(current["x_m"] - origin["x_m"]),
        "y_m": rounded(current["y_m"] - origin["y_m"]),
        "z_m": rounded(current["z_m"] - origin["z_m"]),
        "yaw_rad": rounded(normalize_angle(current["yaw_rad"] - origin["yaw_rad"])),
    }


def plan_payload(plan: Any) -> dict[str, Any]:
    return {
        "speed": rounded(plan.speed),
        "duration_s": rounded(plan.duration),
        "expected": rounded(plan.expected),
        "floored": bool(plan.floored),
    }


def main() -> int:
    args = parse_args()
    if args.initial_settle_s < 0.0 or args.final_settle_s < 0.0:
        raise SystemExit("settle durations must be non-negative")

    repository = args.repo.resolve()
    testbed_dir = repository / "box_demo_groot"
    if not (testbed_dir / "sim2sim_groot_mover.py").is_file():
        raise SystemExit(f"missing simulation testbed: {testbed_dir}")

    os.environ["GROOT_BOX_RUNTIME_CONFIG"] = "/__sim_nav_route_missing__.json"
    os.environ.pop("GROOT_MOTION_BUS_SOCKET", None)
    sys.path.insert(0, str(testbed_dir))

    import sim2sim_groot_mover as testbed

    sim = testbed.S.Sim()
    control_dt = float(testbed.CTRL_DT)
    tracked = {
        "ticks": 0,
        "minimum_pelvis_z_m": math.inf,
    }
    original_step_once = sim.step_once

    def tracked_step_once(command: Any, height: float) -> None:
        original_step_once(command, height)
        tracked["ticks"] += 1
        tracked["minimum_pelvis_z_m"] = min(
            tracked["minimum_pelvis_z_m"],
            float(sim.base()[2]),
        )

    sim.step_once = tracked_step_once
    for _ in range(round(args.initial_settle_s / control_dt)):
        sim.step_once([0.0, 0.0, 0.0], testbed.STAND)

    origin = pose(sim, testbed.S.yaw_from_quat)
    mover = testbed.SimMover(sim, **NAVIGATION_MOVER_CONFIG)
    mover._height = testbed.STAND
    segments = []
    for method_name, request in ROUTE:
        plan = getattr(mover, method_name)(request)
        current = pose(sim, testbed.S.yaw_from_quat)
        segments.append(
            {
                "method": method_name,
                "request": rounded(request),
                "plan": plan_payload(plan),
                "pose_delta_from_start": delta_pose(current, origin),
            }
        )

    for _ in range(round(args.final_settle_s / control_dt)):
        sim.step_once([0.0, 0.0, 0.0], testbed.STAND)

    final_pose = pose(sim, testbed.S.yaw_from_quat)
    report = {
        "schema_version": 1,
        "offline_only": True,
        "opened_sockets": False,
        "imported_ros_dds_robot_sdk": False,
        "control_dt_s": rounded(control_dt),
        "navigation_mover_config": NAVIGATION_MOVER_CONFIG,
        "route": segments,
        "initial_pose": origin,
        "final_pose": final_pose,
        "final_pose_delta": delta_pose(final_pose, origin),
        "simulation_ticks": int(tracked["ticks"]),
        "minimum_pelvis_z_m": rounded(tracked["minimum_pelvis_z_m"]),
        "fell": bool(tracked["minimum_pelvis_z_m"] < 0.40),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["fell"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
