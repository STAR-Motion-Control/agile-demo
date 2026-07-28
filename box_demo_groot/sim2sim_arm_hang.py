#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless A/B validation for GR00T-dWBC locomotion with natural-hang arms.

This reuses ``sweep_step_limit.Sim`` so the ONNX observation, Balance/Walk
switch, leg/waist actions, PD loop and MuJoCo model match the deployment
regression harness.  It compares:

1. the released dWBC arm pose;
2. the requested natural-hang pose with its measured arm observation;
3. the exact same natural-hang pose while only the policy arm observation is
   restored to its released reference (the proposed fix).

No DDS process is created and no real-robot command is published.

5080:
  MUJOCO_GL=egl ~/GR00T-WholeBodyControl/.venv_wbc/bin/python \
      sim2sim_arm_hang.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import sweep_step_limit as S


CTRL_DT = S.SIM_DT * S.DECIM
STAND_HEIGHT = 0.76
SETTLE_BEFORE_S = 2.0
ARM_TRANSITION_S = 3.0
MOVE_S = 4.0
SETTLE_AFTER_S = 8.0

# instantiate_g1_robot_model(..., high_elbow_pose=False).get_default_body_pose()
# for arms.  The older version of this harness incorrectly used the YAML high
# elbow pose and therefore did not represent the deployed adapter.
POLICY_ARM_Q = np.array(
    [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
HANG_ARM_Q = np.array(
    [0.05, 0.22, 0.0, 0.98, 0.20, 0.03, -0.03,
     0.05, -0.22, 0.0, 0.98, -0.20, 0.03, 0.03],
    dtype=np.float32,
)
POLICY_ARM_KP = np.array(
    [100, 100, 40, 40, 20, 20, 20] * 2,
    dtype=np.float32,
)
POLICY_ARM_KD = np.array(
    [5, 5, 2, 2, 2, 2, 2] * 2,
    dtype=np.float32,
)
OLD_ARM_KP = np.array(
    [150, 120, 130, 130, 180, 180, 180] * 2,
    dtype=np.float32,
)
OLD_ARM_KD = np.array([10] * 14, dtype=np.float32)

DIRECTIONS = {
    "forward": np.array([0.40, 0.0, 0.0], dtype=np.float32),
    "backward": np.array([-0.20, 0.0, 0.0], dtype=np.float32),
    "left": np.array([0.0, 0.20, 0.0], dtype=np.float32),
    "right": np.array([0.0, -0.20, 0.0], dtype=np.float32),
}

SCENARIOS = {
    "policy_reference": {
        "arm_q": POLICY_ARM_Q,
        "arm_kp": POLICY_ARM_KP,
        "arm_kd": POLICY_ARM_KD,
        "lock_waist": False,
        "neutralize_arm_observation": 0.0,
    },
    "hang_raw_observation": {
        "arm_q": HANG_ARM_Q,
        "arm_kp": POLICY_ARM_KP * 1.15,
        "arm_kd": POLICY_ARM_KD,
        "lock_waist": False,
        "neutralize_arm_observation": 0.0,
    },
    **{
        f"hang_obscomp_{ratio:.2f}": {
            "arm_q": HANG_ARM_Q,
            "arm_kp": POLICY_ARM_KP * 1.15,
            "arm_kd": POLICY_ARM_KD,
            "lock_waist": False,
            "neutralize_arm_observation": ratio,
        }
        for ratio in (0.40, 0.50, 0.60, 0.70, 1.00)
    },
    "hang_obscomp_0.40_kp1.00": {
        "arm_q": HANG_ARM_Q,
        "arm_kp": POLICY_ARM_KP,
        "arm_kd": POLICY_ARM_KD,
        "lock_waist": False,
        "neutralize_arm_observation": 0.40,
    },
    "hang_obscomp_0.40_kp0.90": {
        "arm_q": HANG_ARM_Q,
        "arm_kp": POLICY_ARM_KP * 0.90,
        "arm_kd": POLICY_ARM_KD,
        "lock_waist": False,
        "neutralize_arm_observation": 0.40,
    },
}


def minimum_jerk_ratio(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def quat_to_roll_pitch(q_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(v) for v in q_wxyz)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


class ArmHangSim(S.Sim):
    def reset_stand(self):
        super().reset_stand()
        self.arm_target = POLICY_ARM_Q.copy()
        self.arm_kp = POLICY_ARM_KP.copy()
        self.arm_kd = POLICY_ARM_KD.copy()
        self.waist_lock_ratio = 0.0
        self.neutralize_arm_observation = 0.0
        self.external_force = np.zeros(3, dtype=np.float32)
        self.last_lower_tau = np.zeros(S.NUM_ACT, dtype=np.float32)
        self.last_arm_tau = np.zeros(S.N_JOINTS - S.NUM_ACT, dtype=np.float32)
        self.d.qpos[7 + S.NUM_ACT : 7 + S.N_JOINTS] = POLICY_ARM_Q
        self.d.qvel[6 + S.NUM_ACT : 6 + S.N_JOINTS] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def compute_obs(self, loco, height):
        observation = super().compute_obs(loco, height)
        ratio = float(self.neutralize_arm_observation)
        if ratio > 0.0:
            arm_start = 13 + S.NUM_ACT
            arm_obs = observation[arm_start : arm_start + len(POLICY_ARM_Q)]
            observation[arm_start : arm_start + len(POLICY_ARM_Q)] = (
                arm_obs + ratio * (POLICY_ARM_Q - arm_obs)
            )
            dq_start = 13 + S.N_JOINTS + S.NUM_ACT
            observation[dq_start : dq_start + len(POLICY_ARM_Q)] *= 1.0 - ratio
        return observation

    def step_once(self, loco, height):
        self.control(loco, height)
        for _ in range(S.DECIM):
            d = self.d
            lower_target = self.target.copy()
            lower_kp = S.KPS.copy()
            lower_kd = S.KDS.copy()
            if self.waist_lock_ratio > 0.0:
                r = self.waist_lock_ratio
                lower_target[12:15] *= 1.0 - r
                lower_kp[12:15] = (1.0 - r) * lower_kp[12:15] + r * 250.0
                lower_kd[12:15] = (1.0 - r) * lower_kd[12:15] + r * 5.0
            lower_tau = (
                (lower_target - d.qpos[7 : 7 + S.NUM_ACT]) * lower_kp
                - d.qvel[6 : 6 + S.NUM_ACT] * lower_kd
            )
            arm_tau = (
                (self.arm_target - d.qpos[7 + S.NUM_ACT : 7 + S.N_JOINTS])
                * self.arm_kp
                - d.qvel[6 + S.NUM_ACT : 6 + S.N_JOINTS] * self.arm_kd
            )
            self.last_lower_tau = lower_tau.copy()
            self.last_arm_tau = arm_tau.copy()
            d.xfrc_applied[self.pelvis, :] = 0.0
            d.xfrc_applied[self.pelvis, :3] = self.external_force
            d.ctrl[: S.NUM_ACT] = lower_tau
            d.ctrl[S.NUM_ACT :] = arm_tau
            mujoco.mj_step(self.m, self.d)


def sample_metrics(sim: ArmHangSim, state: dict) -> None:
    roll, pitch = quat_to_roll_pitch(sim.d.qpos[3:7])
    state["max_abs_roll_deg"] = max(state["max_abs_roll_deg"], abs(math.degrees(roll)))
    state["max_abs_pitch_deg"] = max(state["max_abs_pitch_deg"], abs(math.degrees(pitch)))
    state["max_base_omega"] = max(
        state["max_base_omega"],
        float(np.linalg.norm(sim.d.qvel[3:6])),
    )
    pelvis_z = float(sim.base()[2])
    state["min_pelvis_z"] = min(state["min_pelvis_z"], pelvis_z)
    state["fell"] = state["fell"] or pelvis_z < 0.40


def run_case(
    sim: ArmHangSim,
    scenario_name: str,
    direction_name: str,
    *,
    lateral_push_n: float = 0.0,
) -> dict:
    cfg = SCENARIOS[scenario_name]
    command = DIRECTIONS[direction_name]
    sim.reset_stand()

    metrics = {
        "max_abs_roll_deg": 0.0,
        "max_abs_pitch_deg": 0.0,
        "max_base_omega": 0.0,
        "min_pelvis_z": float("inf"),
        "fell": False,
    }
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(round(SETTLE_BEFORE_S / CTRL_DT)):
        sim.step_once(zero, STAND_HEIGHT)
        sample_metrics(sim, metrics)

    sim.neutralize_arm_observation = float(cfg["neutralize_arm_observation"])
    start_q = sim.arm_target.copy()
    start_kp = sim.arm_kp.copy()
    start_kd = sim.arm_kd.copy()
    for tick in range(round(ARM_TRANSITION_S / CTRL_DT)):
        ratio = minimum_jerk_ratio((tick + 1) * CTRL_DT / ARM_TRANSITION_S)
        sim.arm_target = start_q + (cfg["arm_q"] - start_q) * ratio
        sim.arm_kp = start_kp + (cfg["arm_kp"] - start_kp) * ratio
        sim.arm_kd = start_kd + (cfg["arm_kd"] - start_kd) * ratio
        sim.waist_lock_ratio = ratio if cfg["lock_waist"] else 0.0
        sim.step_once(zero, STAND_HEIGHT)
        sample_metrics(sim, metrics)

    start = sim.base().copy()
    move_hip_actual = []
    move_hip_target = []
    move_waist_actual = []
    move_waist_target = []
    move_base_pitch = []
    for move_tick in range(round(MOVE_S / CTRL_DT)):
        move_t = move_tick * CTRL_DT
        sim.external_force[1] = (
            float(lateral_push_n) if 1.50 <= move_t < 1.70 else 0.0
        )
        sim.step_once(command, STAND_HEIGHT)
        sample_metrics(sim, metrics)
        move_hip_actual.append(float(np.mean(sim.d.qpos[7 + np.array([0, 6])])))
        move_hip_target.append(float(np.mean(sim.target[[0, 6]])))
        move_waist_actual.append(float(sim.d.qpos[7 + 14]))
        move_waist_target.append(float(sim.target[14]))
        move_base_pitch.append(quat_to_roll_pitch(sim.d.qpos[3:7])[1])
    command_end = sim.base().copy()
    sim.external_force[:] = 0.0
    for _ in range(round(SETTLE_AFTER_S / CTRL_DT)):
        sim.step_once(zero, STAND_HEIGHT)
        sample_metrics(sim, metrics)
    end = sim.base().copy()

    axis = 0 if direction_name in ("forward", "backward") else 1
    sign = 1.0 if command[axis] > 0.0 else -1.0
    metrics.update(
        scenario=scenario_name,
        direction=direction_name,
        lateral_push_n=float(lateral_push_n),
        command_speed=float(command[axis]),
        net_command_axis_m=float((end[axis] - start[axis]) * sign),
        at_command_end_m=float((command_end[axis] - start[axis]) * sign),
        final_pelvis_z=float(end[2]),
        move_hip_actual_mean_deg=math.degrees(float(np.mean(move_hip_actual))),
        move_hip_actual_std_deg=math.degrees(float(np.std(move_hip_actual))),
        move_hip_target_mean_deg=math.degrees(float(np.mean(move_hip_target))),
        move_waist_actual_mean_deg=math.degrees(float(np.mean(move_waist_actual))),
        move_waist_actual_std_deg=math.degrees(float(np.std(move_waist_actual))),
        move_waist_target_mean_deg=math.degrees(float(np.mean(move_waist_target))),
        move_base_pitch_mean_deg=math.degrees(float(np.mean(move_base_pitch))),
        move_base_pitch_std_deg=math.degrees(float(np.std(move_base_pitch))),
    )
    return metrics


def main() -> None:
    sim = ArmHangSim()
    results = []
    print(
        "scenario                         direction  net(m)  "
        "roll°  pitch°  omega  min_z  fell"
    )
    for scenario_name in SCENARIOS:
        for direction_name in DIRECTIONS:
            result = run_case(sim, scenario_name, direction_name)
            results.append(result)
            print(
                f"{scenario_name:32s} {direction_name:9s} "
                f"{result['net_command_axis_m']:+.3f}  "
                f"{result['max_abs_roll_deg']:5.1f}  "
                f"{result['max_abs_pitch_deg']:6.1f}  "
                f"{result['max_base_omega']:5.2f}  "
                f"{result['min_pelvis_z']:.3f}  "
                f"{int(result['fell'])}"
            )

    print("\n100 N / 0.20 s lateral-push regression:")
    for scenario_name in (
        "policy_reference",
        "hang_raw_observation",
        "hang_obscomp_0.40",
        "hang_obscomp_0.40_kp1.00",
        "hang_obscomp_0.40_kp0.90",
    ):
        for direction_name in ("backward", "left"):
            result = run_case(
                sim,
                scenario_name,
                direction_name,
                lateral_push_n=100.0,
            )
            results.append(result)
            print(
                f"{scenario_name:32s} {direction_name:9s} "
                f"{result['net_command_axis_m']:+.3f}  "
                f"{result['max_abs_roll_deg']:5.1f}  "
                f"{result['max_abs_pitch_deg']:6.1f}  "
                f"{result['max_base_omega']:5.2f}  "
                f"{result['min_pelvis_z']:.3f}  "
                f"{int(result['fell'])}"
            )

    output = Path(__file__).with_name("sim2sim_arm_hang_results.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
