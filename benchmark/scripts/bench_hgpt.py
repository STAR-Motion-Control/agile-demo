#!/usr/bin/env python3
"""Humanoid-GPT G1 walk-policy benchmark adapter.

This adapter evaluates the released Humanoid-GPT G1-Walk ONNX policy against
the locomotion-only parts of the local G1 lower-body benchmark. It intentionally
does not implement squat / box-carry tests: the released WalkPolicy accepts
only [vx, vy, wz] and has no base-height command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np


CONTROL_DT = 0.02
SIM_DT = 0.001
SETTLE_S = 2.0
WALK_RAMP_S = 2.0
WALK_HOLD_S = 10.0
WALK_MEAN_WINDOW_S = 8.0
WALK_TARGET_VX = 1.0
SWEEP_SPEEDS = (0.4, 0.6, 0.8, 1.0, 1.2)
CIRCLE_VX = 0.4
CIRCLE_WZ = 0.4
CIRCLE_N_LAPS = 2
CIRCLE_RADIAL_TOL = 0.15
CIRCLE_CENTER = (0.0, 1.0)
GOTO_B = (3.0, 1.0, math.pi / 2)
VLN_FINAL_POS_TOL_M = 0.30
VLN_FINAL_YAW_TOL_RAD = math.radians(15.0)
VLN_SMALL_VX_BAND = (0.05, 0.15)
VLN_STOP_SKIP_S = 0.5
FALL_TILT_RAD = 0.9
FALL_BASE_Z = 0.45
VIDEO_W, VIDEO_H = 640, 480
VIDEO_FPS = 25
VIDEO_EVERY_N_CTRL_STEPS = 2

UNSUPPORTED = {
    "squat_box",
    "squat_sweep",
    "squat_limit",
    "squat_pick_ground",
    "squat_place_psi0",
    "pipeline_abc",
    "squat_box_psi0",
}

# psi0 arm-replay tests supported here (pure locomotion + 14-arm disturbance;
# the WalkPolicy lacks a height command, so only the height-free walk/circle
# psi0 variants are wired up). Same agile29 contract-v1 npz the other harnesses
# use: K=17 (14 arms + 3 waist); only the 14 arm columns are replayed, the 3
# waist columns are dropped (waist stays policy-controlled).
PSI0_TESTS = {"walk_speed_psi0", "circle_pillar_psi0"}
N_ARMS = 14            # arms replayed onto target[15:29] = qpos[22:36]
ARM_QPOS_START = 22    # qpos[22:36] = left arm x7 + right arm x7 (14 arms)
# psi0: smoothly blend the physical arm PD target from the settled home pose to
# the replay over this window at locomotion start, so the arms do not snap
# (a hard step from home elbow ~1.28 to replay elbow ~-0.35 throws the gait).
ARM_BLEND_S = 1.0


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_q(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def body_cmd_to_world(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    return (
        vx * math.cos(yaw) - vy * math.sin(yaw),
        vx * math.sin(yaw) + vy * math.cos(yaw),
    )


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = model.sensor(name).id
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr : adr + dim]


def install_hgpt_imports(repo: Path):
    if not (repo / "deploy" / "walk_policy.py").exists():
        raise SystemExit(f"Humanoid-GPT repo not found or wrong layout: {repo}")
    sys.path.insert(0, str(repo))
    from deploy.constants import KDs_walking, KPs_walking  # noqa: PLC0415
    from deploy.walk_policy import WalkPolicy  # noqa: PLC0415
    from tracking.constants import TORQUE_LIMIT  # noqa: PLC0415

    return WalkPolicy, KPs_walking, KDs_walking, TORQUE_LIMIT


def model_xml_for_test(base_xml: Path, test: str) -> Path:
    """Return an XML path, injecting visible benchmark objects when needed."""
    if test not in ("circle_pillar", "circle_pillar_psi0"):
        return base_xml
    text = base_xml.read_text()
    if "bench_pillar" not in text:
        pillar = (
            f'<geom name="bench_pillar" type="cylinder" size="0.15 0.6" '
            f'pos="{CIRCLE_CENTER[0]} {CIRCLE_CENTER[1]} 0.6" '
            f'rgba="0.45 0.45 0.45 1" contype="0" conaffinity="0"/>'
        )
        text = text.replace("</worldbody>", f"        {pillar}\n    </worldbody>", 1)
    out = base_xml.with_name(".bench_hgpt_circle_scene.xml")
    out.write_text(text)
    return out


# ---------------------------------------------------------------------------
# psi0 upper-body arm replay (contract v1; agile29 npz shared with the other
# harnesses). Mirrors bench_dwbc.load_upper_replay / prepare_upper_replay: the
# npz K columns (14 arms + 3 waist) are mapped by joint NAME onto the 14 arm
# slots; the 3 waist columns are DROPPED (waist stays policy-controlled).
# ---------------------------------------------------------------------------
def load_upper_replay(path: str) -> dict:
    """Load + validate a contract-v1 upper-body replay npz. Returns a plain
    dict; pos_arm/carry_arm (model-order arm mapping) are filled by
    prepare_upper_replay()."""
    if not os.path.isfile(path):
        raise SystemExit(f"upper-replay npz not found: {path}")
    z = np.load(path, allow_pickle=True)
    required = ("t", "upper_names", "upper_pos", "height_cmd",
                "grasp_close_t", "carry_pose", "meta_json")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise SystemExit(f"upper-replay npz missing fields {missing}: {path}")
    meta = json.loads(str(z["meta_json"]))
    t = np.asarray(z["t"], dtype=np.float64)
    pos = np.asarray(z["upper_pos"], dtype=np.float64)
    if len(t) < 2 or abs((t[1] - t[0]) - CONTROL_DT) > 1e-6:
        raise SystemExit(
            f"upper-replay dt {t[1] - t[0] if len(t) > 1 else 'n/a'} != harness "
            f"control dt {CONTROL_DT}: {path}")
    names = [str(n) for n in z["upper_names"]]
    if pos.shape != (t.shape[0], len(names)):
        raise SystemExit(f"upper-replay shape mismatch {pos.shape}: {path}")
    if not np.isfinite(pos).all():
        raise SystemExit(f"upper-replay contains NaN/inf: {path}")
    carry = np.asarray(z["carry_pose"], dtype=np.float64)
    if carry.shape != (len(names),):
        raise SystemExit(
            f"upper-replay carry_pose shape {carry.shape} != (K,): {path}")
    return {
        "path": os.path.abspath(path),
        "names": names,
        "pos": pos,
        "carry_pose": carry,
        "grasp_close_t": float(z["grasp_close_t"]),
        "n": int(pos.shape[0]),
        "duration_s": float(pos.shape[0]) * CONTROL_DT,
        "source": str(meta.get("source", "unknown")),
        "embodiment": str(meta.get("embodiment", "unknown")),
        "pos_arm": None,
        "carry_arm": None,
    }


def prepare_upper_replay(model: mujoco.MjModel, rep: dict) -> dict:
    """Map the replay ARM columns by joint NAME onto the 14 qpos[22:36] arm
    slots; the npz's 3 waist columns are DROPPED (waist stays policy-
    controlled). Returns a NEW dict with pos_arm (N,14) / carry_arm (14,)
    added; unmapped arm slots stay 0 (overwriting that arm with 0 PD target)."""
    idx, cols = [], []      # arm-slot 0..13, corresponding replay column
    for col, name in enumerate(rep["names"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"upper-replay joint missing from model: {name}")
        k = int(model.jnt_qposadr[jid]) - ARM_QPOS_START
        if 0 <= k < N_ARMS:
            idx.append(k)
            cols.append(col)
        # else: waist (k<0) or out-of-range -> dropped silently
    if len(set(idx)) != len(idx):
        raise SystemExit("upper-replay has duplicate arm joint columns")
    if not idx:
        raise SystemExit("upper-replay mapped no arm joints (wrong npz?)")
    pos_arm = np.zeros((rep["n"], N_ARMS), dtype=np.float64)
    pos_arm[:, idx] = rep["pos"][:, cols]
    carry_arm = np.zeros(N_ARMS, dtype=np.float64)
    carry_arm[idx] = rep["carry_pose"][cols]
    return dict(rep, pos_arm=pos_arm, carry_arm=carry_arm)


def replay_arm_at(rep: dict, step_idx: int) -> np.ndarray:
    """14-dim arm PD target for the given control step. Index the replay by
    control step (50 Hz); clamp/hold at the last frame past the replay end."""
    k = min(step_idx, rep["n"] - 1)
    return rep["pos_arm"][k]


def blended_arm_at(rep: dict, step_idx: int, t: float,
                   home_arm: np.ndarray) -> np.ndarray:
    """Arm PD target at locomotion time t, blending from the settled home arm
    pose to the replay over ARM_BLEND_S (linear), then following the clamped
    replay. Avoids a hard arm-target step at t=0."""
    arm = replay_arm_at(rep, step_idx)
    f = min(1.0, t / ARM_BLEND_S) if ARM_BLEND_S > 0 else 1.0
    if f >= 1.0:
        return arm
    return (1.0 - f) * np.asarray(home_arm) + f * arm


class HGPTSim:
    def __init__(self, args):
        repo = Path(args.hgpt_repo).resolve()
        WalkPolicy, kps, kds, torque_limit = install_hgpt_imports(repo)
        self.kps = kps
        self.kds = kds
        self.torque_limit = torque_limit
        self.model_xml = model_xml_for_test(Path(args.mjcf).resolve(), args.test)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.model.opt.timestep = SIM_DT
        self.home_qpos = self.model.keyframe("home").qpos.copy()
        self.policy_cls = WalkPolicy
        self.onnx_path = str(Path(args.onnx_walk).resolve())
        self.substeps = int(round(CONTROL_DT / SIM_DT))
        self.data = None
        self.policy = None
        # Optional per-step 14-dim arm PD-target override (psi0 arm replay).
        # When set, step() replaces target[15:29] (= qpos[22:36] arm slots)
        # with these angles after policy inference; the waist (target[12:15])
        # and legs (target[0:12]) stay policy-controlled.
        self.arm_override = None

    def reset(self, seed: int, arm_init=None):
        rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        # Match local benchmark convention: small joint reset noise.
        self.data.qpos[7:] += rng.uniform(-0.02, 0.02, size=self.data.qpos[7:].shape)
        # psi0: start the 14 arms EXACTLY at replay frame 0 (no noise) so the
        # arm PD targets do not snap when the rollout begins; the policy then
        # settles with a consistent arm posture in its obs (mirrors how the
        # other harnesses pre-position the replayed joints at init).
        if arm_init is not None:
            self.data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARMS] = arm_init
        mujoco.mj_forward(self.model, self.data)
        self.policy = self.policy_cls(self.onnx_path)
        # Hold the initial arm pose through the settle phase; the per-frame
        # stream takes over in run_walk / run_circle.
        self.arm_override = np.asarray(arm_init).copy() if arm_init is not None else None

    def step(self, cmd_vel: tuple[float, float, float]):
        gyro = sensor(self.model, self.data, "gyro_pelvis").copy()
        qpos_obs = self.data.qpos[7:].copy()
        qvel_obs = self.data.qvel[6:].copy()
        if self.arm_override is not None:
            # The released WalkPolicy is LOWER-BODY ONLY: it actuates the 12 leg
            # joints and forces waist+arms to its default pose (it never produces
            # arm targets). Its obs reads 8 arm slots purely as fixed-pose
            # proprioceptive context. The psi0 arm replay is an EXOGENOUS upper-
            # body disturbance the leg policy must reject through dynamics, so the
            # policy's arm proprioception is held at its nominal pose (qpos[15:29]
            # = default; qvel = 0) — the physical arms still move (PD-driven to
            # arm_override below), imparting real mass/inertia disturbance. Feeding
            # the raw OOD replay angles into the obs instead corrupts the leg
            # policy and topples it (verified). Legs + waist obs are untouched.
            qpos_obs[15:29] = self.policy.default_qpos[15:29]
            qvel_obs[15:29] = 0.0
        target = self.policy.infer(
            self.data.qpos[3:7].copy(),
            gyro,
            qpos_obs,
            qvel_obs,
            np.asarray(cmd_vel, dtype=np.float32),
        )
        if self.arm_override is not None:
            target = target.copy()
            target[15:29] = self.arm_override
        for _ in range(self.substeps):
            torques = self.kps * (target - self.data.qpos[7:]) + self.kds * (-self.data.qvel[6:])
            self.data.ctrl[:] = np.clip(torques, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.model, self.data)

    def pose(self):
        return self.data.qpos[:2].copy(), yaw_from_q(self.data.qpos[3:7])

    def tilt(self):
        up = sensor(self.model, self.data, "upvector_pelvis")
        return math.acos(max(-1.0, min(1.0, float(up[2]))))

    def local_vx(self):
        return float(sensor(self.model, self.data, "local_linvel_pelvis")[0])

    def fall(self):
        return bool(self.tilt() > FALL_TILT_RAD or float(self.data.qpos[2]) < FALL_BASE_Z)


class Recorder:
    def __init__(self, model, enabled: bool):
        self.enabled = enabled
        self.frames = []
        self.renderer = None
        self.camera = None
        if enabled:
            self.renderer = mujoco.Renderer(model, width=VIDEO_W, height=VIDEO_H)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 0
            self.camera.azimuth = 90
            self.camera.elevation = -20
            self.camera.distance = 2.5

    def maybe_frame(self, data, step_idx: int):
        if not self.enabled or step_idx % VIDEO_EVERY_N_CTRL_STEPS:
            return
        self.renderer.update_scene(data, self.camera)
        self.frames.append(self.renderer.render())

    def save(self, path: Path):
        if not self.enabled or not self.frames:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=VIDEO_FPS)
        return str(path)


def settle(sim: HGPTSim):
    for _ in range(int(SETTLE_S / CONTROL_DT)):
        sim.step((0.0, 0.0, 0.0))


def run_walk(sim: HGPTSim, target_vx: float, record=False, arm_stream=None):
    rec = Recorder(sim.model, record)
    xs, vxs, tilts, zs = [], [], [], []
    home_arm = sim.home_qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARMS]
    arm_devs = []
    n = int((WALK_RAMP_S + WALK_HOLD_S) / CONTROL_DT)
    for i in range(n):
        t = i * CONTROL_DT
        vx = target_vx * min(1.0, t / WALK_RAMP_S)
        if arm_stream is not None:
            sim.arm_override = blended_arm_at(arm_stream, i, t, home_arm)
        sim.step((vx, 0.0, 0.0))
        rec.maybe_frame(sim.data, i)
        xs.append(sim.data.qpos[:2].copy())
        vxs.append(sim.local_vx())
        tilts.append(sim.tilt())
        zs.append(float(sim.data.qpos[2]))
        arm_devs.append(float(np.max(np.abs(
            sim.data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARMS] - home_arm))))
        if sim.fall():
            break
    tail = max(1, int(WALK_MEAN_WINDOW_S / CONTROL_DT))
    mean_vx = float(np.mean(vxs[-tail:]))
    rmse = float(np.sqrt(np.mean((np.asarray(vxs[-tail:]) - target_vx) ** 2)))
    metrics = {
        "vx_target": target_vx,
        "mean_vx_last8s": mean_vx,
        "vx_rmse": rmse,
        "root_dx": float(xs[-1][0] - xs[0][0]) if len(xs) > 1 else 0.0,
        "root_dy": float(xs[-1][1] - xs[0][1]) if len(xs) > 1 else 0.0,
        "min_base_z": float(min(zs)) if zs else None,
        "max_tilt": float(max(tilts)) if tilts else None,
        "arm_dev_max": float(max(arm_devs)) if arm_devs else None,
        "loco_task_mask_end": float(sim.policy._loco_task_mask),
    }
    return metrics, rec


def run_circle(sim: HGPTSim, direction: str, record=False, arm_stream=None):
    sign = 1.0 if direction == "ccw" else -1.0
    rec = Recorder(sim.model, record)
    radius = CIRCLE_VX / CIRCLE_WZ
    center = np.array([0.0, sign * radius])
    n = int((CIRCLE_N_LAPS * 2 * math.pi / CIRCLE_WZ) / CONTROL_DT)
    radial_errs, center_dists, tilts, zs = [], [], [], []
    home_arm = sim.home_qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARMS]
    arm_devs = []
    start_xy, start_yaw = sim.pose()
    prev_yaw = start_yaw
    yaw_accum = 0.0
    for i in range(n):
        t = i * CONTROL_DT
        if arm_stream is not None:
            sim.arm_override = blended_arm_at(arm_stream, i, t, home_arm)
        sim.step((CIRCLE_VX, 0.0, sign * CIRCLE_WZ))
        rec.maybe_frame(sim.data, i)
        arm_devs.append(float(np.max(np.abs(
            sim.data.qpos[ARM_QPOS_START:ARM_QPOS_START + N_ARMS] - home_arm))))
        xy, yaw = sim.pose()
        dist_to_center = float(np.linalg.norm(xy - center))
        center_dists.append(dist_to_center)
        radial_errs.append(abs(dist_to_center - radius))
        yaw_accum += wrap(yaw - prev_yaw)
        prev_yaw = yaw
        tilts.append(sim.tilt())
        zs.append(float(sim.data.qpos[2]))
        if sim.fall():
            break
    end_xy, end_yaw = sim.pose()
    required_yaw = CIRCLE_N_LAPS * 2.0 * math.pi
    lap_completion = abs(yaw_accum) / required_yaw
    metrics = {
        "circle_vx": CIRCLE_VX,
        "circle_wz": CIRCLE_WZ,
        "direction": direction,
        "radial_err_mean": float(np.mean(radial_errs)) if radial_errs else None,
        "radial_err_max": float(np.max(radial_errs)) if radial_errs else None,
        "root_dx": float(end_xy[0] - start_xy[0]),
        "root_dy": float(end_xy[1] - start_xy[1]),
        "yaw_delta": float(wrap(end_yaw - start_yaw)),
        "yaw_accum": float(yaw_accum),
        "lap_completion": float(lap_completion),
        # The Humanoid-GPT XML uses explicit foot-floor contact pairs, so the
        # added pillar is treated as a geometric proximity gate here.
        "pillar_collision": bool(center_dists and np.min(center_dists) < 0.35),
        "min_base_z": float(min(zs)) if zs else None,
        "max_tilt": float(max(tilts)) if tilts else None,
        "arm_dev_max": float(max(arm_devs)) if arm_devs else None,
        "loco_task_mask_end": float(sim.policy._loco_task_mask),
    }
    return metrics, rec


def nav_cmd(xy, yaw, target, vx_clip, vy_clip, wz_clip):
    dx, dy = target[0] - xy[0], target[1] - xy[1]
    dist = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx) if dist > 0.5 else target[2]
    eyaw = wrap(desired_yaw - yaw)
    ex_body = math.cos(yaw) * dx + math.sin(yaw) * dy
    ey_body = -math.sin(yaw) * dx + math.cos(yaw) * dy
    vx = min(vx_clip, max(-vx_clip, 1.0 * ex_body))
    vy = min(vy_clip, max(-vy_clip, 1.0 * ey_body))
    wz = min(wz_clip, max(-wz_clip, 1.5 * eyaw))
    return vx, vy, wz, dist, abs(wrap(target[2] - yaw))


def run_goto(sim: HGPTSim, record=False):
    rec = Recorder(sim.model, record)
    settle_steps = 0
    nav_ok = cal_ok = fine_hold = False
    t_nav = t_cal = 0.0
    err_pos_nav = err_yaw_nav = err_pos_cal = err_yaw_cal = None
    step_idx = 0
    for phase, timeout, clips in (
        ("nav", 30.0, (0.6, 0.3, 0.6)),
        ("cal", 15.0, (0.10, 0.10, 0.10)),
    ):
        hold = 0
        for _ in range(int(timeout / CONTROL_DT)):
            xy, yaw = sim.pose()
            vx, vy, wz, dist, yaw_err = nav_cmd(xy, yaw, GOTO_B, *clips)
            sim.step((vx, vy, wz))
            rec.maybe_frame(sim.data, step_idx)
            step_idx += 1
            if sim.fall():
                break
            xy2, yaw2 = sim.pose()
            err_pos = math.hypot(GOTO_B[0] - xy2[0], GOTO_B[1] - xy2[1])
            err_yaw = abs(wrap(GOTO_B[2] - yaw2))
            if phase == "nav":
                t_nav += CONTROL_DT
                if err_pos <= 0.30 and err_yaw <= math.radians(15):
                    nav_ok = True
                    err_pos_nav, err_yaw_nav = err_pos, err_yaw
                    break
            else:
                t_cal += CONTROL_DT
                if err_pos <= 0.05 and err_yaw <= math.radians(5):
                    hold += 1
                    if hold * CONTROL_DT >= 1.0:
                        cal_ok = fine_hold = True
                        break
                else:
                    hold = 0
        if phase == "nav" and not nav_ok:
            break
        if sim.fall():
            break
    xy, yaw = sim.pose()
    err_pos_cal = math.hypot(GOTO_B[0] - xy[0], GOTO_B[1] - xy[1])
    err_yaw_cal = abs(wrap(GOTO_B[2] - yaw))
    metrics = {
        "success_coarse": bool(nav_ok),
        "success_fine": bool(cal_ok),
        "err_pos_nav": err_pos_nav,
        "err_yaw_nav": err_yaw_nav,
        "err_pos_cal": err_pos_cal,
        "err_yaw_cal": err_yaw_cal,
        "t_nav": t_nav,
        "t_cal": t_cal,
        "fine_hold": bool(fine_hold),
        "loco_task_mask_end": float(sim.policy._loco_task_mask),
    }
    return metrics, rec


def load_tapes(path: str):
    with open(path) as fp:
        data = json.load(fp)
    tapes = data.get("tapes", [])
    if not tapes:
        raise SystemExit(f"No VLN tapes found in {path}")
    return tapes


def tape_cmd_at(tape, t: float, idx: int):
    cmds = tape["cmds"]
    while idx + 1 < len(cmds) and cmds[idx + 1][0] <= t + 1e-9:
        idx += 1
    return idx, tuple(float(x) for x in cmds[idx][1:4])


def run_vln(sim: HGPTSim, tape, record=False):
    rec = Recorder(sim.model, record)
    start_xy, start_yaw = sim.pose()
    duration = float(tape["ref_xy_yaw"][-1][0])
    cmd_idx = 0
    ref_by_t = {round(float(r[0]), 2): r for r in tape["ref_xy_yaw"]}
    track_errs = []
    small_ratios = []
    stop_speeds = []
    zero_start = None
    step_idx = 0
    n = int((duration + 1.0) / CONTROL_DT)
    for i in range(n):
        t = i * CONTROL_DT
        if t < duration:
            cmd_idx, cmd = tape_cmd_at(tape, t, cmd_idx)
        else:
            cmd = (0.0, 0.0, 0.0)
        sim.step(cmd)
        rec.maybe_frame(sim.data, step_idx)
        step_idx += 1
        if sim.fall():
            break

        xy, yaw = sim.pose()
        rel = xy - start_xy
        rel_yaw = wrap(yaw - start_yaw)
        if abs((t % 1.0)) < CONTROL_DT / 2:
            ref = ref_by_t.get(round(t, 2))
            if ref is not None:
                track_errs.append(float(np.linalg.norm(rel - np.array(ref[1:3]))))
        vx_cmd = cmd[0]
        if VLN_SMALL_VX_BAND[0] <= abs(vx_cmd) <= VLN_SMALL_VX_BAND[1]:
            small_ratios.append(sim.local_vx() / vx_cmd)
        if abs(cmd[0]) < 1e-6 and abs(cmd[1]) < 1e-6 and abs(cmd[2]) < 1e-6:
            if zero_start is None:
                zero_start = t
            if t - zero_start >= VLN_STOP_SKIP_S:
                vx_w, vy_w = body_cmd_to_world(sim.local_vx(), 0.0, yaw)
                stop_speeds.append(math.hypot(vx_w, vy_w))
        else:
            zero_start = None

    xy, yaw = sim.pose()
    rel = xy - start_xy
    rel_yaw = wrap(yaw - start_yaw)
    ref_end = tape["ref_xy_yaw"][-1]
    final_pos_err = float(np.linalg.norm(rel - np.array(ref_end[1:3])))
    final_yaw_err = abs(wrap(rel_yaw - float(ref_end[3])))
    metrics = {
        "tape_id": tape.get("id"),
        "final_pos_err": final_pos_err,
        "final_yaw_err": final_yaw_err,
        "mean_track_err": float(np.mean(track_errs)) if track_errs else None,
        "small_cmd_response": float(np.mean(small_ratios)) if small_ratios else None,
        "stop_settle": float(np.mean(stop_speeds)) if stop_speeds else None,
        "loco_task_mask_end": float(sim.policy._loco_task_mask),
    }
    return metrics, rec


def row_for(args, test, trial, success, metrics, video_path, fall_time=None, fall_phase=None, error=None):
    return {
        "framework": "humanoid-gpt",
        "test": test,
        "trial": trial,
        "seed": trial,
        "success": bool(success),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "harness_error": error,
        "metrics": metrics,
        "video": video_path,
        "notes": "Humanoid-GPT WalkPolicy; no height/squat command; moving gate uses norm(cmd_vel)>0.2.",
    }


def write_summary(rows, path: Path, extra=None):
    vals = [r.get("metrics", {}) for r in rows]
    def mean_key(k):
        xs = [v.get(k) for v in vals if isinstance(v.get(k), (int, float))]
        return sum(xs) / len(xs) if xs else None
    summary = {
        "framework": "humanoid-gpt",
        "n_trials": len(rows),
        "n_success": sum(bool(r.get("success")) for r in rows),
        "n_falls": sum(r.get("fall_time") is not None for r in rows),
        "mean_vx_last8s": mean_key("mean_vx_last8s"),
        "radial_err_mean": mean_key("radial_err_mean"),
        "err_pos_cal": mean_key("err_pos_cal"),
        "err_yaw_cal": mean_key("err_yaw_cal"),
        "arm_dev_max": mean_key("arm_dev_max"),
    }
    if extra:
        summary.update(extra)
    with open(path, "w") as fp:
        json.dump(summary, fp, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video", choices=("none", "policy", "all"), default="none")
    ap.add_argument("--label", default="humanoid-gpt")
    ap.add_argument("--hgpt-repo", default="/sda/lizhe/repos/Humanoid-GPT")
    ap.add_argument("--mjcf", default="/sda/lizhe/repos/Humanoid-GPT/storage/assets/unitree_g1_5010/scene_mjx_loco.xml")
    ap.add_argument("--onnx-walk", default="/sda/lizhe/repos/Humanoid-GPT/storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx")
    ap.add_argument("--custom-vx", type=float, default=None)
    ap.add_argument("--custom-wz", type=float, default=None)
    ap.add_argument("--tapes", default="/sda/lizhe/g1bench/vln_tapes.json")
    ap.add_argument("--upper-replay", default=None,
                    help="psi0 tests (walk_speed_psi0 / circle_pillar_psi0): "
                         "upper_replay_agile29_*.npz (contract v1). The 14 arm "
                         "columns are replayed onto the arm PD targets; the 3 "
                         "waist columns are dropped (waist stays policy-held).")
    args = ap.parse_args()

    if args.test in UNSUPPORTED:
        raise SystemExit(f"{args.test} unsupported by released Humanoid-GPT WalkPolicy (vx/vy/wz only, no height/squat command).")
    supported = {"walk_speed", "speed_sweep", "circle_pillar", "goto_ab",
                 "vln_follow"} | PSI0_TESTS
    if args.test not in supported:
        raise SystemExit(f"Unsupported test for bench_hgpt.py: {args.test}")

    replay = None
    if args.test in PSI0_TESTS:
        if args.upper_replay is None:
            raise SystemExit(
                f"--upper-replay is required for {args.test} "
                "(upper_replay_agile29_*.npz, contract v1).")
        replay = load_upper_replay(os.path.expanduser(args.upper_replay))
    elif args.upper_replay is not None:
        sys.stderr.write("WARNING: --upper-replay ignored (not a psi0 test)\n")

    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.label}_{args.test}.jsonl"
    rows = []

    sim = HGPTSim(args)
    upper_mode = None
    if replay is not None:
        replay = prepare_upper_replay(sim.model, replay)
        upper_mode = "psi0_replay_%s" % replay["source"]
    with open(jsonl_path, "w") as fp:
        trial_id = 0
        if args.test == "speed_sweep":
            cases = [(vx, i) for vx in SWEEP_SPEEDS for i in range(args.trials)]
        elif args.test == "vln_follow":
            tapes = load_tapes(args.tapes)
            cases = [(None, i) for i in range(args.trials)]
        else:
            tapes = None
            cases = [(None, i) for i in range(args.trials)]
        for case_vx, _ in cases:
            # psi0: settle in the HOME arm pose (so the leg policy balances
            # cleanly), then blend the physical arms from home to the replay at
            # locomotion start (see blended_arm_at). arm_init stays None.
            sim.reset(seed=trial_id)
            settle(sim)
            record = args.video != "none"
            try:
                if args.test == "walk_speed":
                    target = args.custom_vx if args.custom_vx is not None else WALK_TARGET_VX
                    metrics, rec = run_walk(sim, target, record=record)
                    success = (not sim.fall()) and metrics["mean_vx_last8s"] >= 0.9 * target
                elif args.test == "speed_sweep":
                    metrics, rec = run_walk(sim, case_vx, record=record)
                    metrics["cmd_vx"] = case_vx
                    success = (not sim.fall()) and metrics["mean_vx_last8s"] >= 0.9 * case_vx
                elif args.test == "circle_pillar":
                    direction = "ccw" if trial_id % 2 == 0 else "cw"
                    metrics, rec = run_circle(sim, direction, record=record)
                    success = (
                        (not sim.fall())
                        and metrics["lap_completion"] >= 0.90
                        and metrics["radial_err_mean"] <= CIRCLE_RADIAL_TOL
                        and not metrics["pillar_collision"]
                    )
                elif args.test == "walk_speed_psi0":
                    # walk_speed profile (target_vx=1.0) + 14-arm psi0 replay.
                    target = WALK_TARGET_VX
                    metrics, rec = run_walk(sim, target, record=record,
                                            arm_stream=replay)
                    metrics["upper_mode"] = upper_mode
                    success = (not sim.fall()) and metrics["mean_vx_last8s"] >= 0.9 * target
                elif args.test == "circle_pillar_psi0":
                    # circle_pillar profile (ccw+cw) + 14-arm psi0 replay.
                    direction = "ccw" if trial_id % 2 == 0 else "cw"
                    metrics, rec = run_circle(sim, direction, record=record,
                                              arm_stream=replay)
                    metrics["upper_mode"] = upper_mode
                    success = (
                        (not sim.fall())
                        and metrics["lap_completion"] >= 0.90
                        and metrics["radial_err_mean"] <= CIRCLE_RADIAL_TOL
                        and not metrics["pillar_collision"]
                    )
                else:  # goto_ab
                    if args.test == "goto_ab":
                        metrics, rec = run_goto(sim, record=record)
                        success = (not sim.fall()) and metrics["success_fine"]
                    else:  # vln_follow
                        tape = tapes[trial_id % len(tapes)]
                        metrics, rec = run_vln(sim, tape, record=record)
                        success = (
                            (not sim.fall())
                            and metrics["final_pos_err"] <= VLN_FINAL_POS_TOL_M
                            and metrics["final_yaw_err"] <= VLN_FINAL_YAW_TOL_RAD
                        )
                tag = "ok" if success else "fail"
                video_path = None
                if record:
                    vpath = video_dir / f"{args.label}_{args.test}_t{trial_id:02d}_{tag}.mp4"
                    video_path = rec.save(vpath)
                row = row_for(args, args.test, trial_id, success, metrics, video_path)
            except Exception as exc:
                row = row_for(args, args.test, trial_id, False, {}, None, error=repr(exc))
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            rows.append(row)
            trial_id += 1

    summary_extra = None
    if replay is not None:
        summary_extra = {
            "upper_mode": upper_mode,
            "upper_replay": replay["path"],
            "upper_replay_n_frames": replay["n"],
            "upper_replay_duration_s": replay["duration_s"],
        }
    write_summary(rows, out_dir / f"{args.label}_{args.test}_summary.json",
                  extra=summary_extra)
    print(f"[ok] wrote {jsonl_path} rows={len(rows)} videos={video_dir if args.video != 'none' else 'none'}")


if __name__ == "__main__":
    main()
