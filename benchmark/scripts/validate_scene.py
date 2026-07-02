"""Validate the ManipArena scene generator by rendering variants (§7.1).

Builds a MINIMAL standalone MuJoCo scene -- floor plane + one simple box at the
origin standing in for the robot -- with ``furniture_xml`` injected, for a set
of variants covering all four H_pick levels (seeds 0-3) plus seed 25. For each
variant it renders:
  (a) a top-down orthographic camera, and
  (b) a 3/4 perspective camera,
overlays text labels for each landmark (projected to pixel coords), and writes
PNGs. It also runs geometry sanity checks (legs reaching the floor, box/cube
not floating, nav targets not inside furniture) and prints SceneSpec key fields.

Run on a box with MuJoCo + EGL:
    MUJOCO_GL=egl python validate_scene.py --out /sda/lizhe/g1bench/scene_val
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import mujoco
from PIL import Image, ImageDraw, ImageFont

import manip_scene as ms

# ---------------------------------------------------------------------------
# Minimal world template. A robot-placeholder box at the origin (bit 1, like
# the real robot) plus a ground plane and two cameras. The FLOOR_ANCHOR string
# is the unique injection point (mirrors the harness g1.xml scene worldbody).
# Two free bodies (box/cube) are appended after the placeholder so their coords
# land at the qpos/qvel tail -- exactly the harness layout.
# ---------------------------------------------------------------------------
FLOOR_ANCHOR = ('<geom name="floor" size="0 0 0.05" type="plane" '
                'material="groundplane"/>')

WORLD_TEMPLATE = """<mujoco model="manip_arena_validate">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="groundplane" texture="grid" texrepeat="6 6"
              reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="2 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="robot_placeholder" pos="{sx} {sy} 0.5">
      <freejoint name="robot_free"/>
      <geom name="robot_geom" type="box" size="0.15 0.15 0.5"
            contype="1" conaffinity="1" rgba="0.9 0.2 0.2 1"/>
    </body>
    {floor}
    <camera name="topdown" mode="fixed" pos="1.4 0.2 9.0" euler="0 0 0"
            fovy="38.0"/>
    <camera name="persp" mode="fixed" pos="-1.8 -3.4 3.1"
            xyaxes="0.88 -0.47 0 0.24 0.45 0.86"/>
  </worldbody>
</mujoco>
"""

LANDMARKS_ORDER = [
    ("spawn", "spawn"),
    ("T_pick", "T_pick"),
    ("box", "box"),
    ("T_relay", "T_relay"),
    ("cube", "cube"),
    ("Z_store", "Z_store"),
    ("pillar", "pillar"),
]
TARGET_KEYS = ["P_pick_front", "P_relay_front", "P_cube_touch", "P_store"]


def build_world_xml(spec: ms.SceneSpec) -> str:
    """Assemble the minimal validation world XML for ``spec``.

    Wrist placeholders are replaced with the placeholder box body so the weld
    block compiles (welds stay inactive). The robot placeholder spawns at the
    spec spawn xy.
    """
    frags = ms.furniture_xml(spec, collision_scheme="bit2")
    d = spec.to_dict()
    xml = WORLD_TEMPLATE.format(
        floor=FLOOR_ANCHOR,
        sx=d["spawn"][0], sy=d["spawn"][1],
    )
    # inject furniture + free bodies after the floor anchor
    xml = xml.replace(FLOOR_ANCHOR, FLOOR_ANCHOR + frags["bodies"])
    # inject the weld block before </mujoco>; map wrist placeholders to the
    # validation robot placeholder so it compiles (welds inactive anyway).
    eq = frags["equality"]
    eq = eq.replace(ms.WRIST_L_PLACEHOLDER, "robot_placeholder")
    eq = eq.replace(ms.WRIST_R_PLACEHOLDER, "robot_placeholder")
    xml = xml.replace("</mujoco>", eq + "</mujoco>")
    return xml


def patch_floor_bit2(model: mujoco.MjModel) -> None:
    """OR bit 2 into the floor plane so box/cube collide with the ground
    (mirrors the harness runtime patch). Floor template already has the slab
    tables/zstore at bits 1|2; the plane geom defaults to bit 1."""
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name == "floor":
            model.geom_contype[g] |= 2
            model.geom_conaffinity[g] |= 2
    if hasattr(model, "body_contype"):
        model.body_contype[0] |= 2
        model.body_conaffinity[0] |= 2


# ---------------------------------------------------------------------------
# Projection: world -> pixel, for both cameras, so labels land on geometry.
# ---------------------------------------------------------------------------
def camera_matrices(model, data, cam_name, width, height):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    # mujoco 3.9 has no orthographic camera; both cameras are perspective.
    is_ortho = (bool(model.cam_orthographic[cam_id])
                if hasattr(model, "cam_orthographic") else False)
    fovy = model.cam_fovy[cam_id]
    return cam_pos, cam_mat, is_ortho, fovy


def project_point(world_xyz, cam_pos, cam_mat, is_ortho, fovy, width, height):
    """Project a world point to pixel coords for a MuJoCo fixed camera.

    Camera looks down its local -Z; local +X right, +Y up. Returns (px, py) in
    image pixels (origin top-left) or None if behind the camera."""
    rel = np.asarray(world_xyz) - cam_pos
    local = cam_mat.T @ rel  # into camera frame
    x, y, z = local
    depth = -z  # distance in front of the camera
    if depth <= 1e-6:
        return None
    if is_ortho:
        # For an orthographic camera MuJoCo treats cam_fovy as the full vertical
        # extent in length units. half-height = fovy/2; width scales by aspect.
        half_h = fovy / 2.0
        half_w = half_h * (width / height)
        ndc_x = x / half_w
        ndc_y = y / half_h
    else:
        f = 1.0 / math.tan(math.radians(fovy) / 2.0)
        aspect = width / height
        ndc_x = (x / depth) * f / aspect
        ndc_y = (y / depth) * f
    px = (ndc_x * 0.5 + 0.5) * width
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * height
    return (px, py)


def get_font(size=22):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(img_array, spec, model, data, cam_name, width, height, title):
    img = Image.fromarray(img_array).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(20)
    font_small = get_font(16)
    title_font = get_font(26)

    cam_pos, cam_mat, is_ortho, fovy = camera_matrices(
        model, data, cam_name, width, height)
    d = spec.to_dict()

    def label(world_xyz, text, color):
        proj = project_point(world_xyz, cam_pos, cam_mat, is_ortho, fovy,
                             width, height)
        if proj is None:
            return
        px, py = proj
        if not (0 <= px < width and 0 <= py < height):
            return
        r = 5
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color,
                     outline="white")
        draw.text((px + 8, py - 10), text, fill=color, font=font,
                  stroke_width=2, stroke_fill="black")

    # landmarks (xy at their characteristic z)
    label([d["spawn"][0], d["spawn"][1], 0.5], "spawn", (255, 80, 80))
    label([d["T_pick"]["pos"][0], d["T_pick"]["pos"][1], d["T_pick"]["top_h"]],
          "T_pick h=%.2f" % d["T_pick"]["top_h"], (255, 200, 40))
    label([d["box"]["pos"][0], d["box"]["pos"][1], d["box"]["pos"][2]],
          "box", (255, 230, 60))
    label([d["T_relay"]["pos"][0], d["T_relay"]["pos"][1], d["T_relay"]["top_h"]],
          "T_relay", (180, 140, 80))
    label([d["cube"]["pos"][0], d["cube"]["pos"][1], d["cube"]["pos"][2]],
          "cube", (60, 220, 90))
    label([d["Z_store"]["pos"][0], d["Z_store"]["pos"][1], d["Z_store"]["rim_h"]],
          "Z_store", (60, 160, 230))
    label([d["pillar"]["pos"][0], d["pillar"]["pos"][1], d["pillar"]["h"]],
          "pillar", (170, 170, 170))

    # derived targets (on the ground)
    tcolors = {
        "P_pick_front": (255, 120, 0),
        "P_relay_front": (200, 120, 255),
        "P_cube_touch": (0, 230, 230),
        "P_store": (120, 200, 0),
    }
    for k in TARGET_KEYS:
        t = d["targets"][k]
        label([t[0], t[1], 0.02], k, tcolors[k])

    # title strip
    draw.rectangle([0, 0, width, 40], fill=(0, 0, 0))
    draw.text((10, 6), title, fill=(255, 255, 255), font=title_font)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Geometry sanity checks.
# ---------------------------------------------------------------------------
def geometry_checks(spec: ms.SceneSpec) -> list:
    """Return a list of human-readable problem strings (empty = clean)."""
    problems = []
    d = spec.to_dict()

    # box bottom should sit on T_pick top face
    box_bottom = d["box"]["pos"][2] - ms.BOX_HALF[2]
    if abs(box_bottom - d["T_pick"]["top_h"]) > 1e-6:
        problems.append("box bottom z=%.4f != T_pick top_h=%.4f"
                        % (box_bottom, d["T_pick"]["top_h"]))
    if box_bottom < -1e-6:
        problems.append("box below floor (z_bottom=%.4f)" % box_bottom)

    # cube bottom on T_relay top
    cube_bottom = d["cube"]["pos"][2] - ms.CUBE_HALF[2]
    if abs(cube_bottom - d["T_relay"]["top_h"]) > 1e-6:
        problems.append("cube bottom z=%.4f != T_relay top_h=%.4f"
                        % (cube_bottom, d["T_relay"]["top_h"]))

    # box on the table top face (within half-extents, allowing the 0.05 jitter)
    bx = d["box"]["pos"][0] - d["T_pick"]["pos"][0]
    by = d["box"]["pos"][1] - d["T_pick"]["pos"][1]
    if abs(bx) > d["T_pick"]["size"][0] / 2.0 + 1e-9 or \
       abs(by) > d["T_pick"]["size"][1] / 2.0 + 1e-9:
        problems.append("box off T_pick top face (dx=%.3f dy=%.3f)" % (bx, by))

    # nav targets must NOT fall inside furniture footprints
    targets = d["targets"]
    for k in TARGET_KEYS:
        tx, ty = targets[k][0], targets[k][1]
        # inside pillar?
        pr = math.hypot(tx - d["pillar"]["pos"][0], ty - d["pillar"]["pos"][1])
        if pr < d["pillar"]["r"] + 0.05:
            problems.append("%s inside pillar (r=%.3f)" % (k, pr))
        # inside a table footprint (axis-aligned approx; ignores small yaw)?
        for tname, half in (("T_pick", d["T_pick"]["size"]),
                            ("T_relay", d["T_relay"]["size"])):
            cx, cy = d[tname]["pos"]
            if abs(tx - cx) < half[0] / 2.0 and abs(ty - cy) < half[1] / 2.0:
                # P_cube_touch is allowed near T_relay since the cube is there;
                # only flag if the target is well inside the slab footprint.
                if not (k == "P_cube_touch" and tname == "T_relay"):
                    problems.append("%s inside %s footprint" % (k, tname))

    # nav targets inside the room
    for k in TARGET_KEYS:
        tx, ty = targets[k][0], targets[k][1]
        if abs(tx) > ms.ROOM_HALF or abs(ty) > ms.ROOM_HALF:
            problems.append("%s outside room (%.2f, %.2f)" % (k, tx, ty))

    # P_cube_touch reach sanity: distance to cube ~ ARM_REACH
    pc = targets["P_cube_touch"]
    dc = math.hypot(pc[0] - d["cube"]["pos"][0], pc[1] - d["cube"]["pos"][1])
    if abs(dc - ms.ARM_REACH) > 1e-3:
        problems.append("P_cube_touch dist to cube=%.3f != arm_reach=%.3f"
                        % (dc, ms.ARM_REACH))
    return problems


# ---------------------------------------------------------------------------
# Runtime settle check: drop the scene a few steps, confirm nothing explodes
# and free bodies stay near their start (legs/box not falling through floor).
# ---------------------------------------------------------------------------
def _free_qadr(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def settle_check(model, data, spec) -> list:
    problems = []
    # Resolve free-body qpos addresses by joint name (robust to body order in
    # this validation harness; real harnesses slice the tail per the documented
    # FREE_BODY_QPOS_LAYOUT since they declare free bodies LAST).
    box_q0 = _free_qadr(model, "carried_box_freejoint")
    cube_q0 = _free_qadr(model, "relay_cube_freejoint")
    z_box0 = data.qpos[box_q0 + 2]
    z_cube0 = data.qpos[cube_q0 + 2]
    for _ in range(150):  # ~0.3 s at dt=0.002
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        problems.append("NaN/inf in qpos after settle (constraint blowup)")
        return problems
    z_box1 = data.qpos[box_q0 + 2]
    z_cube1 = data.qpos[cube_q0 + 2]
    if z_box1 < spec.to_dict()["T_pick"]["top_h"] - 0.05:
        problems.append("box fell below T_pick top after settle "
                        "(z %.3f -> %.3f)" % (z_box0, z_box1))
    if z_cube1 < spec.to_dict()["T_relay"]["top_h"] - 0.05:
        problems.append("cube fell below T_relay top after settle "
                        "(z %.3f -> %.3f)" % (z_cube0, z_cube1))
    return problems


def render(model, data, cam_name, width, height):
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam_name)
    pix = renderer.render()
    renderer.close()
    return pix


def summarize(spec: ms.SceneSpec) -> dict:
    d = spec.to_dict()
    return {
        "variant_seed": d["variant_seed"],
        "H_pick": d["H_pick"],
        "scene_spec_hash": d["scene_spec_hash"],
        "spawn": [round(v, 4) for v in d["spawn"]],
        "T_pick_xy": [round(v, 4) for v in d["T_pick"]["pos"]],
        "box_xyz": [round(v, 4) for v in d["box"]["pos"]],
        "T_relay_xy": [round(v, 4) for v in d["T_relay"]["pos"]],
        "cube_xyz": [round(v, 4) for v in d["cube"]["pos"]],
        "Z_store_xy": [round(v, 4) for v in d["Z_store"]["pos"]],
        "pillar_xy": [round(v, 4) for v in d["pillar"]["pos"]],
        "targets": {k: [round(v, 4) for v in d["targets"][k]]
                    for k in TARGET_KEYS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./scene_val")
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=[0, 1, 2, 3, 25])
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=1100)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_summaries = {}
    all_problems = {}
    png_paths = []

    for seed in args.seeds:
        spec = ms.build_variant(seed)
        summ = summarize(spec)
        all_summaries[seed] = summ

        problems = geometry_checks(spec)

        xml = build_world_xml(spec)
        # keep a copy of the assembled XML for debugging
        xml_path = os.path.join(args.out, "variant_%02d.xml" % seed)
        with open(xml_path, "w") as f:
            f.write(xml)

        try:
            model = mujoco.MjModel.from_xml_string(xml)
        except Exception as e:  # noqa: BLE001
            problems.append("XML compile failed: %s" % e)
            all_problems[seed] = problems
            print("=== variant %d (H_pick=%.2f) COMPILE FAIL ===" %
                  (seed, spec.H_pick))
            print(json.dumps(summ, indent=2))
            for p in problems:
                print("  PROBLEM:", p)
            continue

        data = mujoco.MjData(model)
        patch_floor_bit2(model)
        mujoco.mj_forward(model, data)

        problems += settle_check(model, data, spec)
        # re-forward to a clean static pose for rendering (undo settle drift)
        data2 = mujoco.MjData(model)
        mujoco.mj_forward(model, data2)

        for cam, tag in (("topdown", "top"), ("persp", "persp")):
            pix = render(model, data2, cam, args.width, args.height)
            title = "variant %d  H_pick=%.2f  %s  hash=%s" % (
                seed, spec.H_pick, cam, spec.scene_spec_hash[:8])
            pix = annotate(pix, spec, model, data2, cam, args.width,
                           args.height, title)
            out_png = os.path.join(args.out, "variant_%02d_%s.png" % (seed, tag))
            Image.fromarray(pix).save(out_png)
            png_paths.append(out_png)

        all_problems[seed] = problems

        print("=== variant %d (H_pick=%.2f) ===" % (seed, spec.H_pick))
        print(json.dumps(summ, indent=2))
        if problems:
            for p in problems:
                print("  PROBLEM:", p)
        else:
            print("  geometry OK")
        print()

    print("=== PNG OUTPUTS ===")
    for p in png_paths:
        print(p)

    print("\n=== PROBLEM SUMMARY ===")
    any_problem = False
    for seed, probs in all_problems.items():
        if probs:
            any_problem = True
            print("variant %d:" % seed)
            for p in probs:
                print("   -", p)
    if not any_problem:
        print("no geometry problems across all variants")

    # machine-readable dump
    with open(os.path.join(args.out, "summaries.json"), "w") as f:
        json.dump({"summaries": all_summaries, "problems": all_problems},
                  f, indent=2)


if __name__ == "__main__":
    main()
