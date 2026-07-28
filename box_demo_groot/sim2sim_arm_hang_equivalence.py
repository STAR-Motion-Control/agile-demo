#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired non-inferiority test: dWBC reference arms vs natural-hang arms.

Each pair uses the same direction, pre-command hold time and external push.
The comparison covers locomotion and the two-second stop/recovery window.
This is intentionally stricter than a no-fall smoke test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import sim2sim_arm_hang as A


BASELINE = "policy_reference"
PROPOSED = "hang_policy_impedance"
PROPOSED_STATIC_Q = A.HANG_ARM_Q.copy()
PROPOSED_WALK_Q = A.HANG_ARM_Q.copy()
PROPOSED_WALK_KP = A.POLICY_ARM_KP * 1.15
WALK_POSE_TRANSITION_S = 1.0
STATIC_POSE_RESTORE_S = 1.5
HOLD_PHASES_S = (0.0, 0.37, 0.73)
PUSH_FORCES_N = (0.0, 35.0, -35.0)
PUSH_START_S = 1.50
PUSH_DURATION_S = 0.20

# Cost metrics: proposed may not exceed baseline by more than
# max(absolute margin, relative margin * baseline).
COST_MARGINS = {
    "roll_rms_deg": (0.50, 0.15),
    "pitch_rms_deg": (0.50, 0.15),
    "tilt_peak_deg": (1.00, 0.15),
    "omega_rms": (0.05, 0.15),
    "omega_peak": (0.10, 0.15),
    "height_std_m": (0.0015, 0.20),
    "lower_tau_rms_nm": (2.0, 0.15),
    "track_rmse_mps": (0.03, 0.15),
    "orthogonal_drift_m": (0.03, 0.20),
    "recovery_omega_rms": (0.04, 0.20),
}
MIN_HEIGHT_MARGIN_M = 0.010
PROGRESS_RATIO_RANGE = (0.85, 1.15)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def configure_scenario(sim: A.ArmHangSim, scenario: str, extra_hold_s: float) -> None:
    cfg = A.SCENARIOS[scenario]
    target_q = PROPOSED_STATIC_Q if scenario == PROPOSED else cfg["arm_q"]
    zero = np.zeros(3, dtype=np.float32)
    sim.reset_stand()
    for _ in range(round(A.SETTLE_BEFORE_S / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)

    start_q = sim.arm_target.copy()
    start_kp = sim.arm_kp.copy()
    start_kd = sim.arm_kd.copy()
    for tick in range(round(A.ARM_TRANSITION_S / A.CTRL_DT)):
        ratio = A.minimum_jerk_ratio((tick + 1) * A.CTRL_DT / A.ARM_TRANSITION_S)
        sim.arm_target = start_q + (target_q - start_q) * ratio
        sim.arm_kp = start_kp + (cfg["arm_kp"] - start_kp) * ratio
        sim.arm_kd = start_kd + (cfg["arm_kd"] - start_kd) * ratio
        sim.waist_lock_ratio = ratio if cfg["lock_waist"] else 0.0
        sim.step_once(zero, A.STAND_HEIGHT)

    for _ in range(round(extra_hold_s / A.CTRL_DT)):
        sim.step_once(zero, A.STAND_HEIGHT)


def run_trial(
    sim: A.ArmHangSim,
    scenario: str,
    direction: str,
    extra_hold_s: float,
    push_force_n: float,
) -> dict:
    configure_scenario(sim, scenario, extra_hold_s)
    command = A.DIRECTIONS[direction]
    axis = 0 if direction in ("forward", "backward") else 1
    orth_axis = 1 - axis
    sign = 1.0 if command[axis] > 0.0 else -1.0
    push_axis = orth_axis
    start = sim.base().copy()

    move_roll = []
    move_pitch = []
    move_omega = []
    move_height = []
    move_tau = []
    move_axis_vel = []
    all_roll = []
    all_pitch = []
    all_omega = []
    all_height = []
    all_tau = []

    def record(*, moving: bool) -> None:
        roll, pitch = A.quat_to_roll_pitch(sim.d.qpos[3:7])
        omega = float(np.linalg.norm(sim.d.qvel[3:6]))
        height = float(sim.base()[2])
        tau = np.asarray(sim.last_lower_tau, dtype=np.float64)
        all_roll.append(roll)
        all_pitch.append(pitch)
        all_omega.append(omega)
        all_height.append(height)
        all_tau.append(tau)
        if moving:
            move_roll.append(roll)
            move_pitch.append(pitch)
            move_omega.append(omega)
            move_height.append(height)
            move_tau.append(tau)
            move_axis_vel.append(float(sim.d.qvel[axis]) * sign)

    move_ticks = round(A.MOVE_S / A.CTRL_DT)
    for tick in range(move_ticks):
        t = tick * A.CTRL_DT
        if scenario == PROPOSED:
            pose_ratio = A.minimum_jerk_ratio(
                min(1.0, (tick + 1) * A.CTRL_DT / WALK_POSE_TRANSITION_S)
            )
            sim.arm_target = (
                PROPOSED_STATIC_Q
                + (PROPOSED_WALK_Q - PROPOSED_STATIC_Q) * pose_ratio
            )
        sim.external_force[:] = 0.0
        if PUSH_START_S <= t < PUSH_START_S + PUSH_DURATION_S:
            sim.external_force[push_axis] = push_force_n
        sim.step_once(command, A.STAND_HEIGHT)
        record(moving=True)

    command_end = sim.base().copy()
    sim.external_force[:] = 0.0
    settle_ticks = round(A.SETTLE_AFTER_S / A.CTRL_DT)
    for tick in range(settle_ticks):
        if scenario == PROPOSED:
            pose_ratio = A.minimum_jerk_ratio(
                min(1.0, (tick + 1) * A.CTRL_DT / STATIC_POSE_RESTORE_S)
            )
            sim.arm_target = (
                PROPOSED_WALK_Q
                + (PROPOSED_STATIC_Q - PROPOSED_WALK_Q) * pose_ratio
            )
        sim.step_once(np.zeros(3, dtype=np.float32), A.STAND_HEIGHT)
        record(moving=False)
    end = sim.base().copy()

    move_roll_a = np.asarray(move_roll)
    move_pitch_a = np.asarray(move_pitch)
    all_roll_a = np.asarray(all_roll)
    all_pitch_a = np.asarray(all_pitch)
    move_omega_a = np.asarray(move_omega)
    all_omega_a = np.asarray(all_omega)
    move_height_a = np.asarray(move_height)
    move_tau_a = np.asarray(move_tau)
    move_axis_vel_a = np.asarray(move_axis_vel)
    recovery_ticks = max(1, round(0.50 / A.CTRL_DT))

    lfoot = sim.d.xpos[sim.lfoot].copy()
    rfoot = sim.d.xpos[sim.rfoot].copy()
    command_speed = abs(float(command[axis]))
    result = {
        "scenario": scenario,
        "direction": direction,
        "hold_s": extra_hold_s,
        "push_n": push_force_n,
        "fell": bool(np.min(all_height) < 0.40),
        "roll_rms_deg": math.degrees(rms(move_roll_a)),
        "pitch_rms_deg": math.degrees(rms(move_pitch_a)),
        "tilt_peak_deg": math.degrees(
            float(np.max(np.sqrt(all_roll_a**2 + all_pitch_a**2)))
        ),
        "omega_rms": rms(move_omega_a),
        "omega_peak": float(np.max(all_omega_a)),
        "height_std_m": float(np.std(move_height_a)),
        "min_height_m": float(np.min(all_height)),
        "lower_tau_rms_nm": rms(move_tau_a),
        "lower_tau_peak_nm": float(np.max(np.abs(move_tau_a))),
        "track_rmse_mps": rms(move_axis_vel_a - command_speed),
        "orthogonal_drift_m": abs(float(end[orth_axis] - start[orth_axis])),
        "progress_m": float((end[axis] - start[axis]) * sign),
        "progress_at_stop_m": float((command_end[axis] - start[axis]) * sign),
        "recovery_omega_rms": rms(all_omega_a[-recovery_ticks:]),
        "final_stagger_m": abs(float(lfoot[0] - rfoot[0])),
        "final_width_m": abs(float(lfoot[1] - rfoot[1])),
    }
    return result


def evaluate_pair(baseline: dict, proposed: dict) -> list[str]:
    failures = []
    if proposed["fell"]:
        failures.append("fell")
    for metric, (absolute, relative) in COST_MARGINS.items():
        margin = max(absolute, relative * baseline[metric])
        if proposed[metric] > baseline[metric] + margin:
            failures.append(
                f"{metric}:{proposed[metric]:.4f}>"
                f"{baseline[metric] + margin:.4f}"
            )
    if proposed["min_height_m"] < baseline["min_height_m"] - MIN_HEIGHT_MARGIN_M:
        failures.append("min_height_m")
    if baseline["progress_m"] > 0.05:
        ratio = proposed["progress_m"] / baseline["progress_m"]
        if not (PROGRESS_RATIO_RANGE[0] <= ratio <= PROGRESS_RATIO_RANGE[1]):
            failures.append(f"progress_ratio:{ratio:.3f}")
    return failures


def aggregate(rows: list[dict], scenario: str, direction: str) -> dict:
    selected = [
        row for row in rows
        if row["scenario"] == scenario and row["direction"] == direction
    ]
    fields = (
        "roll_rms_deg",
        "pitch_rms_deg",
        "tilt_peak_deg",
        "omega_rms",
        "omega_peak",
        "height_std_m",
        "min_height_m",
        "lower_tau_rms_nm",
        "track_rmse_mps",
        "orthogonal_drift_m",
        "progress_m",
        "recovery_omega_rms",
        "final_stagger_m",
        "final_width_m",
    )
    output = {"scenario": scenario, "direction": direction, "trials": len(selected)}
    for field in fields:
        values = np.asarray([row[field] for row in selected])
        output[f"{field}_mean"] = float(np.mean(values))
        output[f"{field}_worst"] = (
            float(np.min(values)) if field == "min_height_m" else float(np.max(values))
        )
    output["falls"] = sum(int(row["fell"]) for row in selected)
    return output


def main() -> None:
    # Static h remains the exact user pose.  This test evaluates the candidate
    # locomotion sub-pose selected by the component sensitivity sweep.
    A.SCENARIOS[PROPOSED]["arm_q"] = PROPOSED_WALK_Q
    A.SCENARIOS[PROPOSED]["arm_kp"] = PROPOSED_WALK_KP
    A.SCENARIOS[PROPOSED]["arm_kd"] = A.POLICY_ARM_KD
    sim = A.ArmHangSim()
    rows = []
    pair_failures = []
    for direction in A.DIRECTIONS:
        for hold_s in HOLD_PHASES_S:
            for push_n in PUSH_FORCES_N:
                baseline = run_trial(sim, BASELINE, direction, hold_s, push_n)
                proposed = run_trial(sim, PROPOSED, direction, hold_s, push_n)
                rows.extend((baseline, proposed))
                failures = evaluate_pair(baseline, proposed)
                if failures:
                    pair_failures.append(
                        {
                            "direction": direction,
                            "hold_s": hold_s,
                            "push_n": push_n,
                            "failures": failures,
                        }
                    )

    aggregates = []
    print(
        "direction scenario                  n fall "
        "roll_rms pitch_rms tilt_pk omega_rms omega_pk zstd   zmin  "
        "tau_rms track drift progress"
    )
    for direction in A.DIRECTIONS:
        for scenario in (BASELINE, PROPOSED):
            item = aggregate(rows, scenario, direction)
            aggregates.append(item)
            print(
                f"{direction:9s} {scenario:25s} {item['trials']:2d} "
                f"{item['falls']:4d} "
                f"{item['roll_rms_deg_mean']:8.3f} "
                f"{item['pitch_rms_deg_mean']:9.3f} "
                f"{item['tilt_peak_deg_worst']:7.3f} "
                f"{item['omega_rms_mean']:9.3f} "
                f"{item['omega_peak_worst']:8.3f} "
                f"{item['height_std_m_mean']:.4f} "
                f"{item['min_height_m_worst']:.3f} "
                f"{item['lower_tau_rms_nm_mean']:7.2f} "
                f"{item['track_rmse_mps_mean']:.3f} "
                f"{item['orthogonal_drift_m_worst']:.3f} "
                f"{item['progress_m_mean']:.3f}"
            )

    passed = not pair_failures
    output = {
        "passed_noninferiority": passed,
        "criteria": {
            "cost_margins": COST_MARGINS,
            "min_height_margin_m": MIN_HEIGHT_MARGIN_M,
            "progress_ratio_range": PROGRESS_RATIO_RANGE,
        },
        "proposed_walk_q": PROPOSED_WALK_Q.tolist(),
        "proposed_static_q": PROPOSED_STATIC_Q.tolist(),
        "proposed_walk_kp": PROPOSED_WALK_KP.tolist(),
        "proposed_walk_kd": A.POLICY_ARM_KD.tolist(),
        "walk_pose_transition_s": WALK_POSE_TRANSITION_S,
        "static_pose_restore_s": STATIC_POSE_RESTORE_S,
        "pair_failures": pair_failures,
        "aggregates": aggregates,
        "trials": rows,
    }
    path = Path(__file__).with_name("sim2sim_arm_hang_equivalence_results.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"paired trials={len(rows)//2}, failures={len(pair_failures)}")
    print(f"NONINFERIOR={'PASS' if passed else 'FAIL'}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
