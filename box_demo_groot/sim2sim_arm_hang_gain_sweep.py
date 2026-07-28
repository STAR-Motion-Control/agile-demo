#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen natural-hang arm impedances before the full paired A/B test."""

from __future__ import annotations

import copy

import numpy as np

import sim2sim_arm_hang as A
import sim2sim_arm_hang_equivalence as E


def arm(values):
    return np.asarray(values * 2, dtype=np.float32)


BASE_KP = A.POLICY_ARM_KP
BASE_KD = A.POLICY_ARM_KD
CANDIDATES = {
    "policy": (BASE_KP, BASE_KD),
    "uniform_1p15": (BASE_KP * 1.15, BASE_KD),
    "uniform_1p30": (BASE_KP * 1.30, BASE_KD),
    "uniform_1p50": (BASE_KP * 1.50, BASE_KD),
    "shoulder_120": (arm([120, 120, 40, 40, 20, 20, 20]), BASE_KD),
    "shoulder_140_120": (arm([140, 120, 40, 40, 20, 20, 20]), BASE_KD),
    "proximal_mid": (arm([130, 115, 60, 60, 20, 20, 20]), BASE_KD),
    "proximal_firm": (arm([150, 120, 80, 80, 20, 20, 20]), BASE_KD),
    "policy_kd_1p5": (BASE_KP, BASE_KD * 1.50),
    "shoulder_120_kd_1p5": (
        arm([120, 120, 40, 40, 20, 20, 20]),
        BASE_KD * 1.50,
    ),
}
SCREEN_HOLDS = (0.0, 0.37, 0.73)


def main() -> None:
    baseline_sim = A.ArmHangSim()
    baselines = {}
    for direction in A.DIRECTIONS:
        for hold_s in SCREEN_HOLDS:
            key = (direction, hold_s)
            baselines[key] = E.run_trial(
                baseline_sim,
                E.BASELINE,
                direction,
                hold_s,
                0.0,
            )

    original = copy.deepcopy(A.SCENARIOS[E.PROPOSED])
    print(
        "candidate                 fail cost pose  back_ratio "
        "lat_drift roll_rms pitch_rms omega_pk tau_rms"
    )
    for name, (kp, kd) in CANDIDATES.items():
        A.SCENARIOS[E.PROPOSED]["arm_kp"] = kp
        A.SCENARIOS[E.PROPOSED]["arm_kd"] = kd
        sim = A.ArmHangSim()
        rows = []
        failures = []
        for direction in A.DIRECTIONS:
            for hold_s in SCREEN_HOLDS:
                proposed = E.run_trial(
                    sim,
                    E.PROPOSED,
                    direction,
                    hold_s,
                    0.0,
                )
                rows.append(proposed)
                failures.extend(
                    E.evaluate_pair(baselines[(direction, hold_s)], proposed)
                )

        costs = sum(
            not item.startswith(("progress_ratio", "orthogonal_drift_m"))
            for item in failures
        )
        pose = sum(
            item.startswith(
                ("roll_rms_deg", "pitch_rms_deg", "tilt_peak_deg", "omega")
            )
            for item in failures
        )
        back_ratios = [
            row["progress_m"] / baselines[(row["direction"], row["hold_s"])]["progress_m"]
            for row in rows if row["direction"] == "backward"
        ]
        lateral_drifts = [
            row["orthogonal_drift_m"]
            for row in rows if row["direction"] in ("left", "right")
        ]
        print(
            f"{name:25s} {len(failures):4d} {costs:4d} {pose:4d} "
            f"{np.mean(back_ratios):10.3f} "
            f"{max(lateral_drifts):9.3f} "
            f"{np.mean([r['roll_rms_deg'] for r in rows]):8.3f} "
            f"{np.mean([r['pitch_rms_deg'] for r in rows]):9.3f} "
            f"{max(r['omega_peak'] for r in rows):8.3f} "
            f"{np.mean([r['lower_tau_rms_nm'] for r in rows]):7.2f}"
        )

    A.SCENARIOS[E.PROPOSED].update(original)


if __name__ == "__main__":
    main()
