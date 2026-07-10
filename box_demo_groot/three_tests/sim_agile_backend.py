#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AGILE-29dof (velocity-height recurrent student) sim backend (5080, hdmi env).

Wraps WBC-AGILE's own sim2mujoco stack — the SAME ObservationProcessor /
PolicyWrapper classes the real deployment pipeline (agile_lowcmd_pipeline.py)
uses — following the proven agile23_sim2sim_video.py pattern, on the same
scene_29dof.xml as the dwbc backend (joint names match).

Upper-body handling mirrors the real stack: AGILE's action covers the 12 leg
joints; waist+arms are PD-held (pipeline --upper-hold kp40/kd1). The box hug /
waist lean = the merger's rt/arm_sdk overlay -> here an override of the
JointCommand for waist/arm joints.

Interface identical to sim_dwbc_backend.DwbcBackend.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HOME = Path.home()
AGILE_REPO = Path(os.environ.get("AGILE_REPO", HOME / "agile_boxdeploy/WBC-AGILE"))
G1_DIR = HOME / "GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1"
DEFAULT_SCENE = str(G1_DIR / "scene_29dof.xml")
REL_POLICY = "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt"
REL_CONFIG = "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml"

UPPER_HOLD_KP, UPPER_HOLD_KD = 40.0, 1.0     # pipeline --upper-hold defaults
HUG_KP, HUG_KD = 40.0, 2.0                   # arm_sdk hug gains
WAIST_OVR_KP, WAIST_OVR_KD = 60.0, 2.0

WAIST_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_NAMES = [f"{s}_{j}_joint" for s in ("left", "right")
             for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                       "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]


def _import_agile(repo: Path):
    repo = repo.expanduser().resolve()
    for p in (str(repo), str(repo / "agile" / "algorithms" / "rsl_rl")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from agile.sim2mujoco.actions import ActionProcessor
    from agile.sim2mujoco.command_provider import create_command_provider
    from agile.sim2mujoco.observations import ObservationProcessor
    from agile.sim2mujoco.policy import PolicyWrapper
    from agile.sim2mujoco.simulation import MuJocoSimulation
    from agile.sim2mujoco.utils import load_config
    return dict(ActionProcessor=ActionProcessor, create_command_provider=create_command_provider,
                ObservationProcessor=ObservationProcessor, PolicyWrapper=PolicyWrapper,
                MuJocoSimulation=MuJocoSimulation, load_config=load_config)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class AgileBackend:
    name = "agile29"
    stand_height = 0.72          # AGILE trained stand height

    def __init__(self, scene_xml: str = DEFAULT_SCENE, render_wh=(720, 480)):
        import mujoco
        import torch
        self._mujoco = mujoco
        self._torch = torch
        A = _import_agile(AGILE_REPO)
        device = torch.device("cpu")
        config = A["load_config"](str(AGILE_REPO / REL_CONFIG))
        config["mjcf_path"] = scene_xml
        self.policy = A["PolicyWrapper"].from_config(AGILE_REPO / REL_POLICY, config, device)
        self.sim = A["MuJocoSimulation"](config, device, enable_viewer=False,
                                         mjcf_path=Path(scene_xml))
        self.obs_processor = A["ObservationProcessor"](config, self.sim.joint_names, device)
        provider = A["create_command_provider"](config, device,
                                                motion_tracker=self.obs_processor.motion_tracker)
        self.cmd_mgr = provider.manager
        self.obs_processor.command_manager = self.cmd_mgr
        self.sim.command_manager = self.cmd_mgr
        if hasattr(self.cmd_mgr, "height_range"):
            self.cmd_mgr.height_range = (0.40, 0.80)
        # the command manager UI-clamps vx/vy to ±0.5 and wz to ±1.0 — lift so
        # the SPEED test can probe the policy's TRUE tracking ceiling (the real
        # pipeline clamps via --fwd-max flags, not here).
        for attr, rng in (("linear_x_range", (-1.5, 1.5)),
                          ("linear_y_range", (-1.0, 1.0)),
                          ("angular_z_range", (-1.5, 1.5))):
            if hasattr(self.cmd_mgr, attr):
                setattr(self.cmd_mgr, attr, rng)
        self.act_processor = A["ActionProcessor"](config, self.sim.joint_names, device)
        self.ctrl_dt = float(self.sim.dt)

        # AGILE's get_state slices qpos[7:]/qvel[6:] = ALL remaining dofs. An
        # injected free box appends 7 qpos/6 qvel AFTER the robot's 29 joints
        # (robot include precedes the box in the scene), which breaks
        # step()'s PD (29-cmd vs 36-state). Slice the state to the robot's
        # actuated joints once here — fixes both sim.step and the obs path.
        n = int(self.sim.num_joints)
        _orig_get_state = self.sim.get_state

        def _get_state_robot():
            st = _orig_get_state()
            if st.joint_pos.shape[0] > n:
                st.joint_pos = st.joint_pos[:n]
                st.joint_vel = st.joint_vel[:n]
            return st

        self.sim.get_state = _get_state_robot

        # joint-name -> index in sim.joint_names (for the upper-body override)
        self._jidx = {n: i for i, n in enumerate(self.sim.joint_names)}
        self._upper_idx = {n: self._jidx[n] for n in (WAIST_NAMES + ARM_NAMES)
                           if n in self._jidx}
        m = self.sim.mj_model
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "test_box")
        self.box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "test_box_geom")
        self.pillar_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "test_pillar")
        self.knee_geoms = set()
        for g in range(m.ngeom):
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if "knee" in b or "hip_pitch" in b:
                self.knee_geoms.add(g)
        self.upper_override: dict[str, float] | None = None
        self._renderer = None
        self._render_wh = render_wh
        self._cam = mujoco.MjvCamera()
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 3.0, 135.0, -18.0
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self, settle_s: float = 2.5, height: float | None = None):
        self.sim.reset()
        self.obs_processor.reset()
        self.policy.reset()
        d = self.sim.mj_data
        if hasattr(d, "eq_active"):       # clear a stale box weld across configs
            d.eq_active[:] = self.sim.mj_model.eq_active0[:] \
                if hasattr(self.sim.mj_model, "eq_active0") else 0
        if self.box_body >= 0:
            m, d = self.sim.mj_model, self.sim.mj_data
            adr = m.jnt_qposadr[m.body_jntadr[self.box_body]]
            d.qpos[adr:adr+3] = [5.0, 5.0, 0.13]
            d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            self._mujoco.mj_forward(m, d)
        h = self.stand_height if height is None else height
        for _ in range(int(settle_s / self.ctrl_dt)):
            self.step(dict(vx=0, vy=0, wz=0, height=h, pitch=0))

    def spawn_box(self, x: float = 0.30, z: float = 0.91):
        assert self.box_body >= 0, "scene has no test_box"
        m, d = self.sim.mj_model, self.sim.mj_data
        adr = m.jnt_qposadr[m.body_jntadr[self.box_body]]
        px, py = d.xpos[self.pelvis][0], d.xpos[self.pelvis][1]
        yaw = yaw_of(d.qpos[3:7])
        d.qpos[adr+0] = px + x * math.cos(yaw)
        d.qpos[adr+1] = py + x * math.sin(yaw)
        d.qpos[adr+2] = z
        d.qpos[adr+3:adr+7] = d.qpos[3:7]
        vadr = m.jnt_dofadr[m.body_jntadr[self.box_body]]
        d.qvel[vadr:vadr+6] = 0
        self._mujoco.mj_forward(m, d)

    def weld_box(self) -> bool:
        """Weld the box to the torso at its current pose (= 确认抱到)."""
        import scene_inject
        return scene_inject.weld_box_to_torso(self._mujoco, self.sim.mj_model,
                                              self.sim.mj_data)

    def set_upper_override(self, pose: dict | None):
        """pose: joint NAME (or 29-index) -> target rad. None = release."""
        if pose is None:
            self.upper_override = None
            return
        named = {}
        order = (["left_hip_pitch_joint"] * 0)  # noqa: placeholder for clarity
        for k, v in pose.items():
            if isinstance(k, int):   # accept dwbc-style 29-indices too
                k = self.sim.joint_names[k] if k < len(self.sim.joint_names) else None
            if k in self._upper_idx:
                named[k] = float(v)
        self.upper_override = named

    # ------------------------------------------------------------- control
    def step(self, cmd: dict):
        torch = self._torch
        vx, vy, wz = cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("wz", 0.0)
        height = float(cmd.get("height", self.stand_height))
        pitch = float(cmd.get("pitch", 0.0))
        self.cmd_mgr.set_command(vx, vy, wz, height)
        sim_state = self.sim.get_state()
        obs = self.obs_processor.compute(sim_state, noise_scale=0.0)
        with torch.no_grad():
            actions = self.policy(obs)
        self.obs_processor.set_last_action(actions)
        jc = self.act_processor.process(actions)
        if self.upper_override:
            for nm, q in self.upper_override.items():
                i = self._upper_idx[nm]
                tq = q + (pitch if nm == "waist_pitch_joint" else 0.0)
                jc.position[i] = tq
                if nm in WAIST_NAMES:
                    jc.kp[i], jc.kd[i] = WAIST_OVR_KP, WAIST_OVR_KD
                else:
                    jc.kp[i], jc.kd[i] = HUG_KP, HUG_KD
        for _ in range(self.sim.decimation):
            self.sim.step(jc)

    # ------------------------------------------------------------- readouts
    def state(self) -> dict:
        m, d = self.sim.mj_model, self.sim.mj_data
        quat = d.qpos[3:7].copy()
        st = dict(
            t=float(d.time),
            base_pos=d.xpos[self.pelvis].copy(),
            base_quat=quat,
            yaw=yaw_of(quat),
            base_vel_world=d.qvel[0:3].copy(),
            pelvis_z=float(d.xpos[self.pelvis][2]),
        )
        w, x, y, z = quat
        gz = -(w*w - x*x - y*y + z*z)
        st["tilt_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))
        if self.box_body >= 0:
            st["box_pos"] = d.xpos[self.box_body].copy()
            st["box_torso_dist"] = float(np.linalg.norm(st["box_pos"] - d.xpos[self.torso]))
            knee_hit = False
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                if self.box_geom in (g1, g2) and (g1 in self.knee_geoms or g2 in self.knee_geoms):
                    knee_hit = True
                    break
            st["knee_box_contact"] = knee_hit
        if self.pillar_geom >= 0:
            p = m.geom_pos[self.pillar_geom][:2]
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
            self._renderer = self._mujoco.Renderer(self.sim.mj_model, height=h, width=w)
        p = self.sim.mj_data.xpos[self.pelvis]
        self._cam.lookat[:] = [float(p[0]), float(p[1]), 0.7]
        self._renderer.update_scene(self.sim.mj_data, self._cam)
        return self._renderer.render()
