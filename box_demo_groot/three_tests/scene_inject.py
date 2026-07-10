#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inject test props (carry box / obstacle pillar) into scene_29dof.xml.

Works by patching the scene XML text and writing a temp scene NEXT TO the
original (so its <include file="g1_29dof_old.xml"/> still resolves). The same
patched scene drives both the dwbc backend (raw mujoco) and the AGILE backend
(agile.sim2mujoco.MuJocoSimulation with mjcf_path override).

Box = the current benchmark carry box: half-extents 0.125 0.175 0.125
(0.25 x 0.35 x 0.25 m), 2 kg — from 4090 g1bench bench_dwbc/bench_agile.
Pillar = benchmark obstacle: r=0.15 m, h=1.2 m cylinder.
"""
from __future__ import annotations

import os
from pathlib import Path

BOX_HALF = (0.125, 0.175, 0.125)   # 默认: benchmark 箱 0.25x0.35x0.25m (半边长)
BOX_MASS = 2.0                     # kg
PILLAR_R = 0.15                    # m
PILLAR_H = 1.2                     # m


def _box_xml(x: float, z: float, half=BOX_HALF, mass: float = BOX_MASS) -> str:
    hx, hy, hz = half
    return f"""
    <body name="test_box" pos="{x:.3f} 0 {z:.3f}">
      <freejoint name="test_box_free"/>
      <geom name="test_box_geom" type="box" size="{hx} {hy} {hz}" mass="{mass}"
            friction="1.0 0.02 0.0001" condim="4" rgba="0.85 0.65 0.2 1"/>
    </body>
"""


# weld (inactive until the backend activates it after "确认抱到") — same
# mechanism as the 4090 benchmark's carried-box; emulates the real compliant
# grasp that a rigid-PD squeeze cannot reproduce.
BOX_WELD_XML = """
  <equality>
    <weld name="test_box_weld" body1="torso_link" body2="test_box" active="false"/>
  </equality>
"""


def _pillar_xml(x: float, y: float = 0.0) -> str:
    return f"""
    <geom name="test_pillar" type="cylinder" pos="{x:.3f} {y:.3f} {PILLAR_H/2}"
          size="{PILLAR_R} {PILLAR_H/2}" rgba="0.6 0.2 0.2 1"/>
"""


def make_scene(base_xml: str | os.PathLike, *, box: tuple[float, float] | None = None,
               pillar: tuple[float, float] | None = None,
               box_half: tuple[float, float, float] = BOX_HALF,
               box_mass: float = BOX_MASS,
               tag: str = "test") -> str:
    """Return the path of a patched copy of `base_xml` with the requested props.

    box      = (x, z) spawn center of the free carry box (y=0).
    box_half = box half-extents (hx, hy, hz) — 支持不同大小的箱子.
    pillar   = (x, y) of the static obstacle cylinder.
    """
    base = Path(base_xml).expanduser().resolve()
    text = base.read_text()
    # enlarge the offscreen framebuffer for video rendering (default is 640x480)
    if "offwidth" not in text and "<global " in text:
        text = text.replace("<global ", '<global offwidth="1280" offheight="720" ', 1)
    inject = ""
    if box is not None:
        inject += _box_xml(*box, half=box_half, mass=box_mass)
    if pillar is not None:
        inject += _pillar_xml(*pillar)
    if inject:
        assert "</worldbody>" in text, f"no </worldbody> in {base}"
        text = text.replace("</worldbody>", inject + "\n  </worldbody>", 1)
    if box is not None:
        text = text.replace("</mujoco>", BOX_WELD_XML + "\n</mujoco>", 1)
    out = base.parent / f"scene_29dof_{tag}.xml"
    out.write_text(text)
    return str(out)


def weld_box_to_torso(mj, m, d) -> bool:
    """Activate the scene's test_box_weld at the CURRENT torso->box pose.

    MuJoCo weld holds body2 at eq_data's relpose in body1's frame; write the
    live relative pose first, then enable. Shared by both sim backends
    (kept here: stdlib+numpy only, safe to import from any env).
    """
    import numpy as np
    eq = mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, "test_box_weld")
    if eq < 0:
        return False
    b1 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "torso_link")
    b2 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "test_box")
    R1 = d.xmat[b1].reshape(3, 3)
    rel_p = R1.T @ (d.xpos[b2] - d.xpos[b1])
    q1 = np.empty(4)
    q2 = np.empty(4)
    mj.mju_mat2Quat(q1, d.xmat[b1].flatten())
    mj.mju_mat2Quat(q2, d.xmat[b2].flatten())
    q1c = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    rel_q = np.empty(4)
    mj.mju_mulQuat(rel_q, q1c, q2)
    # weld eq_data: anchor(0:3), relpose pos(3:6)+quat(6:10), torquescale(10)
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = rel_p
    m.eq_data[eq, 6:10] = rel_q
    if hasattr(d, "eq_active"):
        d.eq_active[eq] = 1
    else:
        m.eq_active[eq] = 1
    return True
