#!/usr/bin/env python3
"""MuJoCo validation for lateral spacing guard and H-compatible taptap."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

import sim2sim_arm_hang as A
from adaptive_taptap import StanceMetrics
from lateral_stance_guard import LateralStanceGuard


@dataclass(frozen=True)
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


def yaw_from_quat(q_wxyz) -> float:
    w, x, y, z = (float(value) for value in q_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GuardSim(A.ArmHangSim):
    def __init__(self):
        super().__init__()
        self.left_ankle = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self.right_ankle = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )

    def stance(self) -> StanceMetrics:
        left = self.d.xpos[self.left_ankle]
        right = self.d.xpos[self.right_ankle]
        relative_world = left - right
        yaw = yaw_from_quat(self.d.qpos[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        relative_x = c * relative_world[0] + s * relative_world[1]
        relative_y = -s * relative_world[0] + c * relative_world[1]
        left_rotation = self.d.xmat[self.left_ankle].reshape(3, 3)
        right_rotation = self.d.xmat[self.right_ankle].reshape(3, 3)
        foot_rotation = left_rotation.T @ right_rotation
        return StanceMetrics(
            width=abs(float(relative_y)),
            stagger=float(relative_x),
            height_delta=abs(float(relative_world[2])),
            yaw_error=math.atan2(
                float(foot_rotation[1, 0]), float(foot_rotation[0, 0])
            ),
        )


def prepare(sim: GuardSim, arm_hang: bool) -> None:
    sim.reset_stand()
    sim.neutralize_arm_observation = 0.40 if arm_hang else 0.0
    target = A.HANG_ARM_Q if arm_hang else A.POLICY_ARM_Q
    target_kp = A.POLICY_ARM_KP * (1.15 if arm_hang else 1.0)
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(round(A.SETTLE_BEFORE_S / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)
    start_q = sim.arm_target.copy()
    start_kp = sim.arm_kp.copy()
    for tick in range(round(A.ARM_TRANSITION_S / A.CTRL_DT)):
        ratio = A.minimum_jerk_ratio(
            (tick + 1) * A.CTRL_DT / A.ARM_TRANSITION_S
        )
        sim.arm_target = start_q + (target - start_q) * ratio
        sim.arm_kp = start_kp + (target_kp - start_kp) * ratio
        sim.step_once(zero, A.STAND_HEIGHT)


def blank_metrics() -> dict:
    return {
        "min_width_m": float("inf"),
        "max_width_m": 0.0,
        "max_roll_deg": 0.0,
        "max_pitch_deg": 0.0,
        "max_omega": 0.0,
        "min_pelvis_z": float("inf"),
        "fell": False,
    }


def sample(sim: GuardSim, metrics: dict) -> None:
    stance = sim.stance()
    roll, pitch = A.quat_to_roll_pitch(sim.d.qpos[3:7])
    metrics["min_width_m"] = min(metrics["min_width_m"], stance.width)
    metrics["max_width_m"] = max(metrics["max_width_m"], stance.width)
    metrics["max_roll_deg"] = max(metrics["max_roll_deg"], abs(math.degrees(roll)))
    metrics["max_pitch_deg"] = max(
        metrics["max_pitch_deg"], abs(math.degrees(pitch))
    )
    metrics["max_omega"] = max(
        metrics["max_omega"], float(np.linalg.norm(sim.d.qvel[3:6]))
    )
    metrics["min_pelvis_z"] = min(metrics["min_pelvis_z"], float(sim.base()[2]))
    metrics["fell"] = metrics["fell"] or float(sim.base()[2]) < 0.40


def run_lateral(
    sim: GuardSim,
    *,
    arm_hang: bool,
    direction: float,
    use_guard: bool,
    duration_s: float = 4.0,
    narrow_start: bool = False,
) -> dict:
    prepare(sim, arm_hang)
    if narrow_start:
        sim.d.qpos[7 + 1] -= 0.08
        sim.d.qpos[7 + 7] += 0.08
        mujoco.mj_forward(sim.m, sim.d)
    guard = LateralStanceGuard(
        min_width=0.15,
        guard_margin=0.03,
        prediction_s=0.0,
    )
    metrics = blank_metrics()
    requested = Cmd(vy=direction * 0.20)
    blocked_at = None
    start_y = float(sim.base()[1])
    for tick in range(round(duration_s / A.CTRL_DT)):
        command = requested
        if use_guard:
            command, blocked, event = guard.update(
                tick * A.CTRL_DT, command, sim.stance()
            )
            if event == "blocked":
                blocked_at = tick * A.CTRL_DT
        sim.step_once(
            np.array([command.vx, command.vy, command.wz], dtype=np.float32),
            A.STAND_HEIGHT,
        )
        sample(sim, metrics)
    for _ in range(round(1.0 / A.CTRL_DT)):
        sim.step_once(np.zeros(3, dtype=np.float32), A.STAND_HEIGHT)
        sample(sim, metrics)
    final = sim.stance()
    net_y = direction * (float(sim.base()[1]) - start_y)
    metrics.update(
        test="lateral",
        arm_hang=arm_hang,
        direction="left" if direction > 0 else "right",
        guard=use_guard,
        narrow_start=narrow_start,
        blocked_at_s=blocked_at,
        net_lateral_m=net_y,
        final_width_m=final.width,
        final_stagger_m=final.stagger,
        final_yaw_deg=math.degrees(final.yaw_error),
    )
    return metrics


def run_taptap(
    sim: GuardSim,
    *,
    arm_hang: bool,
    speed: float,
    phase_s: float,
    pre_lateral_s: float,
) -> dict:
    prepare(sim, arm_hang)
    lateral = np.array([0.0, 0.20, 0.0], dtype=np.float32)
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(round(pre_lateral_s / A.CTRL_DT)):
        sim.step_once(lateral, A.STAND_HEIGHT)
    for _ in range(round(0.40 / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)
    before = sim.stance()
    metrics = blank_metrics()
    for phase in range(4):
        command = np.array(
            [speed if phase % 2 == 0 else -speed, 0.0, 0.0],
            dtype=np.float32,
        )
        for _ in range(round(phase_s / A.CTRL_DT)):
            sim.step_once(command, A.STAND_HEIGHT)
            sample(sim, metrics)
    for _ in range(round(1.0 / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)
        sample(sim, metrics)
    final = sim.stance()
    metrics.update(
        test="taptap",
        arm_hang=arm_hang,
        speed=speed,
        phase_s=phase_s,
        pre_lateral_s=pre_lateral_s,
        initial_width_m=before.width,
        initial_stagger_m=before.stagger,
        initial_yaw_deg=math.degrees(before.yaw_error),
        final_width_m=final.width,
        final_stagger_m=final.stagger,
        final_yaw_deg=math.degrees(final.yaw_error),
    )
    return metrics


def run_repeated_lateral(
    sim: GuardSim,
    *,
    arm_hang: bool,
    use_guard: bool,
) -> dict:
    prepare(sim, arm_hang)
    guard = LateralStanceGuard(min_width=0.15, guard_margin=0.03)
    metrics = blank_metrics()
    blocked_count = 0
    now = 0.0
    for segment in range(6):
        requested = Cmd(vy=0.20 if segment % 2 == 0 else -0.20)
        for _ in range(round(1.5 / A.CTRL_DT)):
            command = requested
            if use_guard:
                command, _, event = guard.update(now, command, sim.stance())
                blocked_count += int(event == "blocked")
            sim.step_once(
                np.array([command.vx, command.vy, command.wz], dtype=np.float32),
                A.STAND_HEIGHT,
            )
            sample(sim, metrics)
            now += A.CTRL_DT
        for _ in range(round(0.4 / A.CTRL_DT)):
            command = Cmd()
            if use_guard:
                command, _, _ = guard.update(now, command, sim.stance())
            sim.step_once(np.zeros(3, dtype=np.float32), A.STAND_HEIGHT)
            sample(sim, metrics)
            now += A.CTRL_DT
    final = sim.stance()
    metrics.update(
        test="repeated_lateral",
        arm_hang=arm_hang,
        guard=use_guard,
        blocked_count=blocked_count,
        final_width_m=final.width,
        final_stagger_m=final.stagger,
        final_yaw_deg=math.degrees(final.yaw_error),
    )
    return metrics


def run_small_lateral_candidate(
    sim: GuardSim,
    *,
    arm_hang: bool,
    direction: float,
    speed: float,
    duration_s: float,
    phase_delay_s: float,
) -> dict:
    prepare(sim, arm_hang)
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(round(phase_delay_s / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)
    start = sim.base().copy()
    metrics = blank_metrics()
    command = np.array([0.0, direction * speed, 0.0], dtype=np.float32)
    for _ in range(round(duration_s / A.CTRL_DT)):
        sim.step_once(command, A.STAND_HEIGHT)
        sample(sim, metrics)
    for _ in range(round(1.0 / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)
        sample(sim, metrics)
    end = sim.base().copy()
    metrics.update(
        test="small_lateral_candidate",
        arm_hang=arm_hang,
        direction="left" if direction > 0 else "right",
        speed=speed,
        duration_s=duration_s,
        phase_delay_s=phase_delay_s,
        net_lateral_m=direction * float(end[1] - start[1]),
    )
    return metrics


def main() -> None:
    sim = GuardSim()
    results = []
    for arm_hang in (False, True):
        for direction in (1.0, -1.0):
            for use_guard in (False, True):
                result = run_lateral(
                    sim,
                    arm_hang=arm_hang,
                    direction=direction,
                    use_guard=use_guard,
                )
                results.append(result)
                print(
                    "LATERAL",
                    "H" if arm_hang else "default",
                    result["direction"],
                    "guard" if use_guard else "raw",
                    f"min_w={result['min_width_m']:.3f}",
                    f"final_w={result['final_width_m']:.3f}",
                    f"net={result['net_lateral_m']:.3f}",
                    f"roll={result['max_roll_deg']:.1f}",
                    f"omega={result['max_omega']:.2f}",
                    f"blocked={result['blocked_at_s']}",
                    f"fell={int(result['fell'])}",
                )
            for use_guard in (False, True):
                result = run_lateral(
                    sim,
                    arm_hang=arm_hang,
                    direction=direction,
                    use_guard=use_guard,
                    narrow_start=True,
                )
                results.append(result)
                print(
                    "LATERAL-NARROW",
                    "H" if arm_hang else "default",
                    result["direction"],
                    "guard" if use_guard else "raw",
                    f"min_w={result['min_width_m']:.3f}",
                    f"final_w={result['final_width_m']:.3f}",
                    f"net={result['net_lateral_m']:.3f}",
                    f"roll={result['max_roll_deg']:.1f}",
                    f"omega={result['max_omega']:.2f}",
                    f"blocked={result['blocked_at_s']}",
                    f"fell={int(result['fell'])}",
                )

    for arm_hang in (False, True):
        for use_guard in (False, True):
            result = run_repeated_lateral(
                sim, arm_hang=arm_hang, use_guard=use_guard
            )
            results.append(result)
            print(
                "REPEATED-LATERAL",
                "H" if arm_hang else "default",
                "guard" if use_guard else "raw",
                f"min_w={result['min_width_m']:.3f}",
                f"final_w={result['final_width_m']:.3f}",
                f"yaw={result['final_yaw_deg']:.1f}",
                f"roll={result['max_roll_deg']:.1f}",
                f"omega={result['max_omega']:.2f}",
                f"blocks={result['blocked_count']}",
                f"fell={int(result['fell'])}",
            )

    candidates = ((0.12, 1.50), (0.10, 1.00), (0.10, 0.80), (0.08, 1.00), (0.08, 0.80))
    for arm_hang in (False, True):
        for direction in (1.0, -1.0):
            for speed, duration_s in candidates:
                group = []
                for phase_delay_s in (0.0, 0.20, 0.40, 0.60, 0.80):
                    result = run_small_lateral_candidate(
                        sim,
                        arm_hang=arm_hang,
                        direction=direction,
                        speed=speed,
                        duration_s=duration_s,
                        phase_delay_s=phase_delay_s,
                    )
                    results.append(result)
                    group.append(result)
                nets = [item["net_lateral_m"] for item in group]
                print(
                    "SMALL-LATERAL",
                    "H" if arm_hang else "default",
                    "left" if direction > 0 else "right",
                    f"{speed:.2f}@{duration_s:.2f}",
                    f"net={min(nets):+.3f}..{max(nets):+.3f}",
                    f"min_w={min(item['min_width_m'] for item in group):.3f}",
                    f"roll={max(item['max_roll_deg'] for item in group):.1f}",
                    f"omega={max(item['max_omega'] for item in group):.2f}",
                    f"falls={sum(int(item['fell']) for item in group)}",
                )

    for arm_hang in (False, True):
        for speed, phase_s in (
            (0.08, 0.40),
            (0.07, 0.50),
            (0.06, 0.60),
        ):
            for pre_lateral_s in (0.8, 1.0, 1.2, 1.4):
                result = run_taptap(
                    sim,
                    arm_hang=arm_hang,
                    speed=speed,
                    phase_s=phase_s,
                    pre_lateral_s=pre_lateral_s,
                )
                results.append(result)
                print(
                    "TAPTAP",
                    "H" if arm_hang else "default",
                    f"{speed:.2f}@{phase_s:.2f}",
                    f"pre={pre_lateral_s:.1f}",
                    f"width={result['initial_width_m']:.3f}->{result['final_width_m']:.3f}",
                    f"yaw={result['initial_yaw_deg']:.1f}->{result['final_yaw_deg']:.1f}",
                    f"roll={result['max_roll_deg']:.1f}",
                    f"omega={result['max_omega']:.2f}",
                    f"fell={int(result['fell'])}",
                )

    output = Path(__file__).with_name("sim2sim_lateral_guard_taptap_results.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
