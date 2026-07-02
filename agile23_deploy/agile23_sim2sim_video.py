#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a MuJoCo sim2sim walking-validation video of the AGILE 23-DoF velocity-height student.

Reuses AGILE's own sim2mujoco stack (same obs/policy/action wiring as sim2mujoco_eval), drives
a scripted command schedule (settle / forward / back / turn-left / turn-right [/ squat]), renders
offscreen (no display needed) with mujoco.Renderer, and writes an mp4. Prints per-phase base
displacement + yaw so walking can be judged quantitatively, not just visually.

Run on a host with mujoco + imageio + torch and the 23-DoF AGILE repo on PYTHONPATH, e.g.:
  /sda/lizhe/miniforge3/envs/agile/bin/python agile23_sim2sim_video.py \
    --agile-repo /sdb/lizhe/g1_deepsquat/WBC-AGILE \
    --checkpoint /sdb/lizhe/g1_deepsquat/student_23dof_1400_export/policy.pt \
    --config     /sdb/lizhe/g1_deepsquat/student_23dof_1400_export/<IO_descriptors>.yaml \
    --mjcf       /sdb/lizhe/g1_deepsquat/g1_23dof_desc/scene_23dof.xml \
    --out /sdb/lizhe/g1_deepsquat/agile23_walk.mp4
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")        # headless offscreen rendering


def import_agile(repo: Path):
    repo = repo.expanduser().resolve()
    for p in (str(repo), str(repo / "agile" / "algorithms" / "rsl_rl")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from agile.sim2mujoco.actions import ActionProcessor
    from agile.sim2mujoco.command_provider import VelocityCommandProvider, create_command_provider
    from agile.sim2mujoco.observations import ObservationProcessor
    from agile.sim2mujoco.policy import PolicyWrapper
    from agile.sim2mujoco.simulation import MuJocoSimulation
    from agile.sim2mujoco.utils import load_config
    return dict(ActionProcessor=ActionProcessor, VelocityCommandProvider=VelocityCommandProvider,
                create_command_provider=create_command_provider, ObservationProcessor=ObservationProcessor,
                PolicyWrapper=PolicyWrapper, MuJocoSimulation=MuJocoSimulation, load_config=load_config)


def yaw_of(qw, qx, qy, qz):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def main():
    p = argparse.ArgumentParser(description="AGILE 23-DoF sim2sim walking video")
    p.add_argument("--agile-repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--height", type=float, default=0.72, help="held base-height cmd (squat not the focus)")
    p.add_argument("--vx", type=float, default=0.4)
    p.add_argument("--vy", type=float, default=0.3)
    p.add_argument("--wz", type=float, default=0.5)
    p.add_argument("--phase-s", type=float, default=4.0)
    p.add_argument("--settle-s", type=float, default=1.5)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height-px", type=int, default=540)
    p.add_argument("--with-squat", action="store_true", help="append a squat phase at the end")
    args = p.parse_args()

    import numpy as np
    import torch
    import mujoco
    import imageio.v2 as imageio

    A = import_agile(args.agile_repo)
    device = torch.device(args.device)
    config = A["load_config"](args.config)
    config["mjcf_path"] = str(args.mjcf)

    policy = A["PolicyWrapper"].from_config(args.checkpoint, config, device)
    sim = A["MuJocoSimulation"](config, device, enable_viewer=False, mjcf_path=args.mjcf)
    obs_processor = A["ObservationProcessor"](config, sim.joint_names, device)
    command_provider = A["create_command_provider"](config, device, motion_tracker=obs_processor.motion_tracker)
    command_manager = command_provider.manager
    obs_processor.command_manager = command_manager
    sim.command_manager = command_manager
    if hasattr(command_manager, "height_range"):
        command_manager.height_range = (0.18, 0.80)        # allow the held/any height
    act_processor = A["ActionProcessor"](config, sim.joint_names, device)

    sim.reset()
    obs_processor.reset()
    policy.reset()

    # command schedule: (label, vx, vy, wz). A short zero-command settle is interleaved
    # between motion phases so each direction is validated from a stable stand (not from the
    # end-state of the previous motion).
    h = args.height
    motions = [
        ("forward",      args.vx, 0.0,      0.0),
        ("back",        -abs(args.vx) * 0.75, 0.0, 0.0),
        ("turn_left",    0.0,     0.0,      args.wz),
        ("turn_right",   0.0,     0.0,     -args.wz),
    ]
    if args.with_squat:
        motions.append(("squat_walk", 0.2, 0.0, 0.0))
    settle = max(1.0, args.settle_s * 0.66)
    phases = [("settle", 0.0, 0.0, 0.0, args.settle_s)]
    for (lbl, vx, vy, wz) in motions:
        phases.append((lbl, vx, vy, wz, args.phase_s))
        phases.append((f"settle_{lbl}", 0.0, 0.0, 0.0, settle))

    renderer = mujoco.Renderer(sim.mj_model, height=args.height_px, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.2, 130.0, -15.0

    control_dt = sim.dt
    render_every = max(1, int(round((1.0 / args.fps) / control_dt)))
    frames = []
    summary = []

    print(f"control_dt={control_dt:.4f}s ({1/control_dt:.0f}Hz)  render_every={render_every}  "
          f"joints={sim.num_joints}")

    for (label, vx, vy, wz, dur) in phases:
        steps = int(dur / control_dt)
        st = sim.get_state()
        x0, y0 = float(st.root_pos[0]), float(st.root_pos[1])
        yaw0 = yaw_of(*[float(v) for v in st.root_quat])
        min_h = 9.9
        for t in range(steps):
            command_manager.set_command(vx, vy, wz, h)
            sim_state = sim.get_state()
            obs = obs_processor.compute(sim_state, noise_scale=0.0)
            with torch.no_grad():
                actions = policy(obs)
            obs_processor.set_last_action(actions)
            joint_cmd = act_processor.process(actions)
            for _ in range(sim.decimation):
                sim.step(joint_cmd)
            min_h = min(min_h, float(sim.get_state().root_pos[2]))
            if (t % render_every) == 0:
                st2 = sim.get_state()
                cam.lookat[:] = [float(st2.root_pos[0]), float(st2.root_pos[1]), 0.6]
                renderer.update_scene(sim.mj_data, cam)
                frames.append(renderer.render())
        st = sim.get_state()
        dx, dy = float(st.root_pos[0]) - x0, float(st.root_pos[1]) - y0
        dyaw = math.degrees(yaw_of(*[float(v) for v in st.root_quat]) - yaw0)
        dyaw = (dyaw + 180) % 360 - 180
        z = float(st.root_pos[2])
        line = (f"[{label:11s}] dx={dx:+.2f}m dy={dy:+.2f}m dyaw={dyaw:+6.1f}deg "
                f"endZ={z:.3f} minZ={min_h:.3f} fell={'YES' if z < 0.3 else 'no'}")
        print(line, flush=True)
        summary.append(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(args.out), frames, fps=args.fps)
    print(f"\nwrote {args.out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
    print("=== walking validation summary ===")
    for s in summary:
        print(" ", s)


if __name__ == "__main__":
    main()
