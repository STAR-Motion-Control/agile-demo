#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identify which natural-hang pose components change dWBC gait quality."""

from __future__ import annotations

import copy

import numpy as np

import sim2sim_arm_hang as A
import sim2sim_arm_hang_equivalence as E


def symmetric_arm(left):
    left = np.asarray(left, dtype=np.float32)
    right = left.copy()
    right[[1, 4, 6]] *= -1.0
    return np.concatenate((left, right))


USER = np.array([0.29, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03], np.float32)
POSES = {
    "official": A.POLICY_ARM_Q,
    "user_exact": symmetric_arm(USER),
    "no_wrist": symmetric_arm([0.29, 0.22, 0.0, 0.98, 0.0, 0.0, 0.0]),
    "pitch_0p20": symmetric_arm([0.20, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p18": symmetric_arm([0.18, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p15": symmetric_arm([0.15, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p13": symmetric_arm([0.13, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p12": symmetric_arm([0.12, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p10": symmetric_arm([0.10, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p05": symmetric_arm([0.05, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_zero": symmetric_arm([0.0, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "roll_official": symmetric_arm([0.29, 0.30, 0.0, 0.98, 0.20, 0.03, -0.03]),
    "pitch_0p20_no_wrist": symmetric_arm([0.20, 0.22, 0.0, 0.98, 0.0, 0.0, 0.0]),
    "shoulder_elbow_only": symmetric_arm([0.29, 0.22, 0.0, 0.98, 0.0, 0.0, 0.0]),
    "wrist_only_on_official": A.POLICY_ARM_Q
    + symmetric_arm([0.0, 0.0, 0.0, 0.0, 0.20, 0.03, -0.03]),
}
SCREEN_HOLDS = (0.0, 0.37, 0.73)


def main() -> None:
    baseline_sim = A.ArmHangSim()
    baselines = {}
    for direction in A.DIRECTIONS:
        for hold_s in SCREEN_HOLDS:
            baselines[(direction, hold_s)] = E.run_trial(
                baseline_sim, E.BASELINE, direction, hold_s, 0.0
            )

    original = copy.deepcopy(A.SCENARIOS[E.PROPOSED])
    print(
        "pose                      fail pose  back_ratio "
        "left_drift right_drift roll_rms pitch_rms omega_pk"
    )
    for name, pose in POSES.items():
        A.SCENARIOS[E.PROPOSED]["arm_q"] = pose
        A.SCENARIOS[E.PROPOSED]["arm_kp"] = A.POLICY_ARM_KP * 1.15
        A.SCENARIOS[E.PROPOSED]["arm_kd"] = A.POLICY_ARM_KD
        sim = A.ArmHangSim()
        rows = []
        failures = []
        for direction in A.DIRECTIONS:
            for hold_s in SCREEN_HOLDS:
                proposed = E.run_trial(
                    sim, E.PROPOSED, direction, hold_s, 0.0
                )
                rows.append(proposed)
                failures.extend(
                    E.evaluate_pair(baselines[(direction, hold_s)], proposed)
                )
        pose_failures = sum(
            item.startswith(
                ("roll_rms_deg", "pitch_rms_deg", "tilt_peak_deg", "omega")
            )
            for item in failures
        )
        back_ratios = [
            row["progress_m"] / baselines[(row["direction"], row["hold_s"])]["progress_m"]
            for row in rows if row["direction"] == "backward"
        ]
        left_drift = max(
            row["orthogonal_drift_m"] for row in rows if row["direction"] == "left"
        )
        right_drift = max(
            row["orthogonal_drift_m"] for row in rows if row["direction"] == "right"
        )
        print(
            f"{name:25s} {len(failures):4d} {pose_failures:4d} "
            f"{np.mean(back_ratios):10.3f} "
            f"{left_drift:10.3f} {right_drift:11.3f} "
            f"{np.mean([r['roll_rms_deg'] for r in rows]):8.3f} "
            f"{np.mean([r['pitch_rms_deg'] for r in rows]):9.3f} "
            f"{max(r['omega_peak'] for r in rows):8.3f}"
        )

    A.SCENARIOS[E.PROPOSED].update(original)


if __name__ == "__main__":
    main()
