#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GR00T decoupled-WBC sim backend for the three tests (5080, .venv_wbc, EGL).

Mirrors the VALIDATED obs/PD/policy core of ../sweep_step_limit.py (which
reproduces run_mujoco_gear_wbc.py exactly — audited 2026-07-01), and adds:
  * rpy_cmd passthrough into obs command[4:7] (torso lean channel),
  * upper-body override: PD waist+arms to a given pose (= the real merger's
    rt/arm_sdk overlay of motors 12-28; used for the box hug + waist lean),
  * injected scenes (box / pillar) via scene_inject.make_scene,
  * offscreen EGL video frames + base/box/contact metrics.

Interface (shared with sim_agile_backend.SimBackend):
  b = DwbcBackend(scene_xml)
  b.reset(settle_s)
  b.step(cmd: dict)          # one 50Hz control tick; cmd from schedules.py
  b.state() -> dict          # base pos/quat/vel, yaw, box pose, knee-box contact
  b.render() -> ndarray      # RGB frame
"""
from __future__ import annotations

import collections
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import onnxruntime as ort

HOME = Path.home()
G1_DIR = HOME / "GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1"
DEFAULT_SCENE = str(G1_DIR / "scene_29dof.xml")
POLDIR = HOME / "GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
BAL = str(POLDIR / "GR00T-WholeBodyControl-Balance.onnx")
WALK = str(POLDIR / "GR00T-WholeBodyControl-Walk.onnx")

# ---- ground truth from g1_gear_wbc.yaml (same as the real adapter) ----
KPS = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40,
                250, 250, 250], np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], np.float32)
DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
                    0, 0, 0], np.float32)
ANG_VEL_SCALE = 0.5
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], np.float32)
SIM_DT = 0.005
DECIM = 4
NUM_ACT = 15          # 12 legs + 3 waist (RL)
N_JOINTS = 29
OBS_HL = 6
SOBS = 86
WALK_THRESH = 0.05
ARM_KP, ARM_KD = 100.0, 0.5          # decoupled default arm hold
HUG_KP, HUG_KD = 40.0, 2.0           # real arm_sdk hug gains (box_demo)
WAIST_OVR_KP, WAIST_OVR_KD = 60.0, 2.0  # waist lean override (real arm_sdk waist)

CTRL_DT = SIM_DT * DECIM             # 0.02 s

# joint index blocks within the 29 (MJCF order = motor order)
WAIST_IDX = [12, 13, 14]             # yaw, roll, pitch
ARM_IDX = list(range(15, 29))        # L7 + R7


def _quat_rot_inv(q, v):
    w, x, y, z = q
    qc = np.array([w, -x, -y, -z])
    return np.array([
        v[0]*(qc[0]**2+qc[1]**2-qc[2]**2-qc[3]**2)+v[1]*2*(qc[1]*qc[2]-qc[0]*qc[3])+v[2]*2*(qc[1]*qc[3]+qc[0]*qc[2]),
        v[0]*2*(qc[1]*qc[2]+qc[0]*qc[3])+v[1]*(qc[0]**2-qc[1]**2+qc[2]**2-qc[3]**2)+v[2]*2*(qc[2]*qc[3]-qc[0]*qc[1]),
        v[0]*2*(qc[1]*qc[3]-qc[0]*qc[2])+v[1]*2*(qc[2]*qc[3]+qc[0]*qc[1])+v[2]*(qc[0]**2-qc[1]**2-qc[2]**2+qc[3]**2),
    ])


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class DwbcBackend:
    name = "dwbc"
    ctrl_dt = CTRL_DT
    stand_height = 0.74

    def __init__(self, scene_xml: str = DEFAULT_SCENE, render_wh=(720, 480)):
        self.m = mujoco.MjModel.from_xml_path(scene_xml)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = SIM_DT
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.bal = ort.InferenceSession(BAL, sess_options=so, providers=prov)
        self.walk = ort.InferenceSession(WALK, sess_options=so, providers=prov)
        self.bal_in = self.bal.get_inputs()[0].name
        self.walk_in = self.walk.get_inputs()[0].name
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.foot_l = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.foot_r = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.foot_l_geoms = self._foot_geoms("left_ankle_pitch_link")
        self.foot_r_geoms = self._foot_geoms("right_ankle_pitch_link")
        self.box_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "test_box")
        self.box_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "test_box_geom")
        self.pillar_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "test_pillar")
        self.knee_geoms = set()
        for g in range(self.m.ngeom):
            b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) or ""
            if "knee" in b or "hip_pitch" in b:
                self.knee_geoms.add(g)
        # upper-body override (None = decoupled default hold at 0)
        self.upper_override: dict[int, float] | None = None
        # pitch 命令注入方式:
        #   "rpy"          obs command[4:7] 通道(round8 实测: 策略基本不跟)
        #   "waist"        只偏置腰pitch动作目标(round9 实测: 策略从关节观测
        #                  看到偏置会反补偿 -> 骨盆更后仰+位移超冲, 弃用)
        #   "waist_obscomp" 偏置动作 + 观测同步减掉偏置(策略无感) = 推荐,
        #                  真机 adapter --back-lean-gain 按此同构实现
        self.pitch_mode: str = "rpy"
        self._lean_obs_comp: float = 0.0
        self._renderer = None
        self._render_wh = render_wh
        self._cam = mujoco.MjvCamera()
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 3.0, 135.0, -18.0
        self.reset()

    def _foot_geoms(self, root_name: str) -> set[int]:
        """Return geoms on the ankle subtree, used for touchdown detection."""
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, root_name)
        if root < 0:
            return set()
        bodies = {root}
        changed = True
        while changed:
            changed = False
            for body_id in range(self.m.nbody):
                if body_id not in bodies and int(self.m.body_parentid[body_id]) in bodies:
                    bodies.add(body_id)
                    changed = True
        return {
            geom_id
            for geom_id in range(self.m.ngeom)
            if int(self.m.geom_bodyid[geom_id]) in bodies
        }

    # ------------------------------------------------------------- lifecycle
    def reset(self, settle_s: float = 2.5, height: float | None = None):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = 0
        self.d.qpos[2] = 0.793
        self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[7:7+NUM_ACT] = DEFAULT
        if self.box_body >= 0:      # park the box far away until spawn_box()
            adr = self.m.jnt_qposadr[self.m.body_jntadr[self.box_body]]
            self.d.qpos[adr:adr+3] = [5.0, 5.0, BOXPARK_Z]
            self.d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(NUM_ACT, np.float32)
        self.hist = collections.deque([np.zeros(SOBS, np.float32)] * OBS_HL, maxlen=OBS_HL)
        self.obs = np.zeros(SOBS * OBS_HL, np.float32)
        h = self.stand_height if height is None else height
        for _ in range(int(settle_s / CTRL_DT)):
            self.step(dict(vx=0, vy=0, wz=0, height=h, pitch=0))

    def spawn_box(self, x: float = 0.30, z: float = 0.93):
        """Teleport the box into the robot's arms ("递箱子")."""
        assert self.box_body >= 0, "scene has no test_box (use scene_inject box=...)"
        adr = self.m.jnt_qposadr[self.m.body_jntadr[self.box_body]]
        px, py = self.d.xpos[self.pelvis][0], self.d.xpos[self.pelvis][1]
        yaw = yaw_of(self.d.qpos[3:7])
        self.d.qpos[adr+0] = px + x * math.cos(yaw)
        self.d.qpos[adr+1] = py + x * math.sin(yaw)
        self.d.qpos[adr+2] = z
        self.d.qpos[adr+3:adr+7] = self.d.qpos[3:7]   # match base yaw
        vadr = self.m.jnt_dofadr[self.m.body_jntadr[self.box_body]]
        self.d.qvel[vadr:vadr+6] = 0
        mujoco.mj_forward(self.m, self.d)

    def set_upper_override(self, pose: dict[str, float] | None):
        """pose maps 29-joint INDEX -> target rad for waist/arm joints (12..28).
        None restores the decoupled default (arms PD to 0, waist RL)."""
        self.upper_override = None if pose is None else dict(pose)

    def weld_box(self) -> bool:
        """Weld the box to the torso at its CURRENT relative pose (= 确认抱到).
        Emulates the real compliant grasp (rigid-PD squeeze on a rigid box is
        knife-edge fragile in sim); benchmark-standard carried-box mechanism."""
        import scene_inject
        return scene_inject.weld_box_to_torso(mujoco, self.m, self.d)

    # ------------------------------------------------------------- control
    def _compute_obs(self, loco, height, rpy):
        d = self.d
        command = np.zeros(7, np.float32)
        command[:3] = np.asarray(loco, np.float32) * CMD_SCALE
        command[3] = height
        command[4:7] = np.asarray(rpy, np.float32)
        qj = d.qpos[7:7+N_JOINTS].copy()
        # waist_obscomp: 观测里减掉腰pitch偏置 — 策略看到的腰=它自己命令的角度,
        # 不会把前倾当扰动去对抗(纯 waist 模式实测: 策略反补偿 -> 骨盆更后仰+超冲)
        qj[14] -= self._lean_obs_comp
        dqj = d.qvel[6:6+N_JOINTS].copy()
        quat = d.qpos[3:7].copy()
        omega = d.qvel[3:6].copy()
        pad = np.zeros(N_JOINTS, np.float32)
        pad[:NUM_ACT] = DEFAULT
        so = np.zeros(SOBS, np.float32)
        so[0:7] = command
        so[7:10] = omega * ANG_VEL_SCALE
        so[10:13] = _quat_rot_inv(quat, np.array([0., 0., -1.]))
        so[13:13+N_JOINTS] = (qj - pad)
        so[13+N_JOINTS:13+2*N_JOINTS] = dqj * DOF_VEL_SCALE
        so[13+2*N_JOINTS:13+2*N_JOINTS+15] = self.action
        return so

    def step(self, cmd: dict):
        """One 50 Hz control tick. cmd: vx vy wz height pitch (pitch -> waist
        override if upper_override active, else rpy_cmd channel)."""
        loco = [cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("wz", 0.0)]
        height = float(cmd.get("height", self.stand_height))
        pitch = float(cmd.get("pitch", 0.0))
        # lean: with an active upper override the lean goes through the waist
        # override target (real arm_sdk path); otherwise through rpy_cmd.
        use_rpy = self.upper_override is None and self.pitch_mode == "rpy"
        rpy = (0.0, pitch, 0.0) if use_rpy else (0.0, 0.0, 0.0)
        self._lean_obs_comp = (pitch if (self.pitch_mode == "waist_obscomp"
                                         and self.upper_override is None) else 0.0)

        so = self._compute_obs(loco, height, rpy)
        self.hist.append(so)
        for i, h in enumerate(self.hist):
            self.obs[i*SOBS:(i+1)*SOBS] = h
        inp = self.obs[None].astype(np.float32)
        if np.linalg.norm(np.asarray(loco)) <= WALK_THRESH:
            self.action = self.bal.run(None, {self.bal_in: inp})[0].squeeze().astype(np.float32)
        else:
            self.action = self.walk.run(None, {self.walk_in: inp})[0].squeeze().astype(np.float32)
        target15 = self.action * ACTION_SCALE + DEFAULT

        # full 29-joint PD targets/gains
        tgt = np.zeros(N_JOINTS, np.float32)
        kp = np.zeros(N_JOINTS, np.float32)
        kd = np.zeros(N_JOINTS, np.float32)
        tgt[:NUM_ACT] = target15
        if (self.pitch_mode in ("waist", "waist_obscomp")
                and self.upper_override is None and pitch != 0.0):
            tgt[14] += pitch                     # 腰pitch目标偏置(前倾为正)
        kp[:NUM_ACT] = KPS
        kd[:NUM_ACT] = KDS
        kp[15:29] = ARM_KP
        kd[15:29] = ARM_KD          # arms PD to 0 (decoupled default)
        if self.upper_override is not None:
            for j, q in self.upper_override.items():
                tgt[j] = q
                if j in WAIST_IDX:
                    kp[j], kd[j] = WAIST_OVR_KP, WAIST_OVR_KD
                else:
                    kp[j], kd[j] = HUG_KP, HUG_KD
            # apply the commanded lean on the waist-pitch override target
            if pitch != 0.0 and 14 in self.upper_override:
                tgt[14] = self.upper_override[14] + pitch
        for _ in range(DECIM):
            d = self.d
            tau = (tgt - d.qpos[7:7+N_JOINTS]) * kp + (0 - d.qvel[6:6+N_JOINTS]) * kd
            d.ctrl[:N_JOINTS] = tau
            mujoco.mj_step(self.m, self.d)

    # ------------------------------------------------------------- readouts
    def state(self) -> dict:
        d = self.d
        st = dict(
            t=float(d.time),
            base_pos=d.xpos[self.pelvis].copy(),
            base_quat=d.qpos[3:7].copy(),
            yaw=yaw_of(d.qpos[3:7]),
            base_vel_world=d.qvel[0:3].copy(),
            pelvis_z=float(d.xpos[self.pelvis][2]),
        )
        g = _quat_rot_inv(d.qpos[3:7], np.array([0., 0., -1.]))
        st["tilt_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        st["waist_pitch_q"] = float(d.qpos[7 + 14])
        if self.foot_l >= 0 and self.foot_r >= 0:
            st["foot_l"] = d.xpos[self.foot_l].copy()
            st["foot_r"] = d.xpos[self.foot_r].copy()
            st["foot_l_quat"] = d.xquat[self.foot_l].copy()
            st["foot_r_quat"] = d.xquat[self.foot_r].copy()
            left_contact = False
            right_contact = False
            for contact_id in range(d.ncon):
                geom1 = int(d.contact[contact_id].geom1)
                geom2 = int(d.contact[contact_id].geom2)
                if geom1 in self.foot_l_geoms or geom2 in self.foot_l_geoms:
                    left_contact = True
                if geom1 in self.foot_r_geoms or geom2 in self.foot_r_geoms:
                    right_contact = True
            st["foot_l_contact"] = left_contact
            st["foot_r_contact"] = right_contact
        if self.box_body >= 0:
            st["box_pos"] = d.xpos[self.box_body].copy()
            torso = d.xpos[self.torso]
            st["box_torso_dist"] = float(np.linalg.norm(st["box_pos"] - torso))
            knee_hit = False
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                if self.box_geom in (g1, g2) and (g1 in self.knee_geoms or g2 in self.knee_geoms):
                    knee_hit = True
                    break
            st["knee_box_contact"] = knee_hit
        if self.pillar_geom >= 0:
            p = self.m.geom_pos[self.pillar_geom][:2]
            st["pillar_clearance"] = float(np.linalg.norm(st["base_pos"][:2] - p)) - 0.15
            hit = False
            for c in range(d.ncon):
                if self.pillar_geom in (d.contact[c].geom1, d.contact[c].geom2):
                    hit = True
                    break
            st["pillar_contact"] = hit
        return st

    def render(self):
        if self._renderer is None:
            w, h = self._render_wh
            self._renderer = mujoco.Renderer(self.m, height=h, width=w)
        p = self.d.xpos[self.pelvis]
        self._cam.lookat[:] = [float(p[0]), float(p[1]), 0.7]
        self._renderer.update_scene(self.d, self._cam)
        return self._renderer.render()


BOXPARK_Z = 0.13   # parked box rests on the floor far away
