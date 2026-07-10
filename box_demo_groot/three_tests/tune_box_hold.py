#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search the box hug: (spawn_x, spawn_z, roll, pitch_delta) -> held for 5s?

Quick, no video. Finds configs where the free benchmark box (0.25x0.35x0.25, 2kg)
stays in the arms for 5s of standing. Winners feed run_sim_test --box-x/--box-z/
--hug-roll/--hug-pitch-delta.

  MUJOCO_GL=egl <python> tune_box_hold.py --controller dwbc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_inject
from run_sim_test import hug_pose, make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="dwbc", choices=("dwbc", "agile29"))
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--box-size", default="0.25,0.35,0.25",
                    help="箱子 深,宽,高(米)。给出时 sps/rolls/xs/zs 缺省围绕 hug_for_box 推导值扫")
    ap.add_argument("--box-mass", type=float, default=2.0)
    ap.add_argument("--sps", default=None, help="shoulder pitch (fwd<0); 缺省=推导值±0.15")
    ap.add_argument("--rolls", default=None, help="缺省=推导值±0.05")
    ap.add_argument("--elbows", default="0.7,0.9")
    ap.add_argument("--xs", default=None, help="缺省=推导 spawn_x±0.03")
    ap.add_argument("--zs", default=None, help="缺省=推导 spawn_z±0.04")
    a = ap.parse_args()

    depth, width, height = [float(v) for v in a.box_size.split(",")]
    from run_sim_test import hug_for_box
    _, dx, dz, dinfo = hug_for_box(width, depth, height)
    if a.sps is None:
        a.sps = f"{dinfo['sp']-0.15:.2f},{dinfo['sp']:.2f},{dinfo['sp']+0.15:.2f}"
    if a.rolls is None:
        a.rolls = f"{max(0.03,dinfo['roll']-0.05):.2f},{dinfo['roll']:.2f},{dinfo['roll']+0.05:.2f}"
    if a.xs is None:
        a.xs = f"{dx-0.03:.2f},{dx:.2f},{dx+0.03:.2f}"
    if a.zs is None:
        a.zs = f"{dz-0.04:.2f},{dz:.2f},{dz+0.04:.2f}"
    print(f"box 深{depth}x宽{width}x高{height} 推导中心: {dinfo} spawn=({dx:.2f},{dz:.2f})")

    scene = scene_inject.make_scene(
        Path.home() / "GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml",
        box=(0.35, 0.93), box_half=(depth/2, width/2, height/2), box_mass=a.box_mass,
        tag=f"boxtune_{a.controller}")
    b = make_backend(a.controller, scene)
    winners = []
    for sp in [float(v) for v in a.sps.split(",")]:
        for roll in [float(v) for v in a.rolls.split(",")]:
            for elbow in [float(v) for v in a.elbows.split(",")]:
                for x in [float(v) for v in a.xs.split(",")]:
                    for z in [float(v) for v in a.zs.split(",")]:
                        b.reset(settle_s=2.0)
                        b.set_upper_override(hug_pose(sp, roll, elbow))
                        for _ in range(int(2.5 / b.ctrl_dt)):
                            b.step(dict(vx=0, vy=0, wz=0, height=b.stand_height, pitch=0))
                        b.spawn_box(x=x, z=z)
                        held_t = 0.0
                        for k in range(int(a.hold_s / b.ctrl_dt)):
                            b.step(dict(vx=0, vy=0, wz=0, height=b.stand_height, pitch=0))
                            st = b.state()
                            if st["box_torso_dist"] > 0.55 or st["box_pos"][2] < 0.45 \
                                    or st["pelvis_z"] < 0.35:
                                break
                            held_t = (k + 1) * b.ctrl_dt
                        ok = held_t >= a.hold_s - 1e-6
                        print(f"sp={sp:+.2f} roll={roll:.2f} el={elbow:.1f} "
                              f"x={x:.2f} z={z:.2f} -> held {held_t:.1f}s "
                              f"{'HOLD' if ok else ''}", flush=True)
                        if ok:
                            winners.append((sp, roll, elbow, x, z))
    print("\nwinners (sp, roll, elbow, x, z):")
    for w in winners:
        print(" ", w)


if __name__ == "__main__":
    main()
