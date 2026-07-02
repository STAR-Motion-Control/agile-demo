"""ManipArena shared scene generator (BENCHMARK_V2_DESIGN.md §2-§3).

One deterministic scene generator that every harness (homie/agile/falcon/amo)
and every task (L1-L5, M1-M2) shares. Given a ``variant_seed`` it produces:

  (a) a ``SceneSpec`` dict  -- all landmark ground-truth poses + derived
      navigation targets + grasp/weld definitions, JSON-serialisable so each
      JSONL row can carry it and rebuild the scene for a replay.
  (b) a ``furniture_xml`` fragment -- furniture body/geom XML to inject into a
      harness's robot ``<worldbody>`` plus the ``<equality>`` weld block.

Conventions reused verbatim from R3-R5 (bench_homie.py):
  - bit-2 collision scheme: robot keeps bit 1, box/cube get bit 2, furniture &
    floor get bits 1|2 -> box/cube collide with furniture+floor but NOT the
    bit-1 robot until a place flips them on. See ``collision_scheme="bit2"``.
  - free bodies (box, cube) carry a ``<freejoint>`` and are appended AFTER all
    robot bodies, so the robot keeps qpos[0:N]/qvel[0:N-1] and the free-body
    coords land at the END of qpos/qvel. See ``FREE_BODY_QPOS_LAYOUT`` doc.
  - welds are predeclared in ``<equality>`` (active=false) with placeholder
    relposes; each harness re-anchors the relpose at grasp time.

No MuJoCo import here -- pure data + string assembly so it stays importable in
any env. ``validate_scene.py`` does the actual MuJoCo build/render.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Nominal layout constants (BENCHMARK_V2_DESIGN.md §2 table). Top-down frame:
# robot spawns at origin facing +X, 6m x 6m flat room.
# ---------------------------------------------------------------------------
ROOM_HALF = 3.0  # 6m x 6m room -> +/-3 m floor half extent

# H_pick layering: variant_seed % 4 selects the controlled table height.
H_PICK_LEVELS: Tuple[float, ...] = (0.30, 0.45, 0.60, 0.75)

# T_pick (pick table): nominal centre, top-down size of the top face.
T_PICK_POS_NOM = (1.6, 0.0)        # +/-0.15 m, yaw +/-10 deg
T_PICK_TOP_HALF = (0.30, 0.20)     # 0.6 x 0.4 m top face -> half extents

# box (2 kg): sits on T_pick top face centre, jitter +/-0.05 m on the top.
BOX_FULL = (0.30, 0.22, 0.22)      # 0.30 x 0.22 x 0.22 m
BOX_HALF = tuple(v / 2.0 for v in BOX_FULL)
BOX_MASS = 2.0

# T_relay (relay table): nominal centre, fixed top height 0.55 m.
T_RELAY_POS_NOM = (2.4, 1.8)       # +/-0.2 m
T_RELAY_TOP_H = 0.55
T_RELAY_TOP_HALF = (0.25, 0.20)    # 0.5 x 0.4 m top face -> half extents

# cube (0.3 kg): on T_relay top, jitter +/-0.05 m. 0.08 m cube.
CUBE_FULL = (0.08, 0.08, 0.08)
CUBE_HALF = tuple(v / 2.0 for v in CUBE_FULL)
CUBE_MASS = 0.3

# Z_store (placement zone): shallow framed marker pad.
Z_STORE_POS_NOM = (3.2, -1.2)      # +/-0.3 m
Z_STORE_HALF = (0.30, 0.30)        # 0.6 x 0.6 m pad -> half extents
Z_STORE_RIM_H = 0.12               # rim height

# pillar (cylindrical obstacle): shared by nav / circle tasks.
PILLAR_POS_NOM = (1.5, -0.8)       # +/-0.3 m
PILLAR_R = 0.15
PILLAR_H = 1.2

# lane: 4 m straight corridor along +X (walk-speed task). No geometry, just a
# semantic extent recorded in the spec.
LANE_LENGTH = 4.0

# Jitter magnitudes (half-ranges; uniform in [-mag, +mag]).
JIT_SPAWN_XY = 0.10
JIT_SPAWN_YAW = 0.10
JIT_TPICK_XY = 0.15
JIT_TPICK_YAW = math.radians(10.0)
JIT_BOX_ON_TOP = 0.05
JIT_TRELAY_XY = 0.20
JIT_CUBE_ON_TOP = 0.05
JIT_ZSTORE_XY = 0.30
JIT_PILLAR_XY = 0.30

# Grasp / placement abstraction (§3).
D_GRASP = 0.30          # both wrists < this to box surface -> activate weld
CUBE_TOUCH_D = 0.15     # wrist < this to cube -> cube->box-top weld

# Derived nav-target geometry.
FRONT_STANDOFF = 0.55   # P_pick_front / P_relay_front stand 0.55 m off the edge
ARM_REACH = 0.45        # robot arm reach ~0.45 m (P_cube_touch back-solve)

# Table-leg geometry (validation/visualisation only; legs are static geoms).
LEG_HALF = 0.025        # 0.05 m square legs
LEG_INSET = 0.04        # legs inset from the top-face corners

# Fields of SceneSpec hashed into ``scene_spec_hash`` (md5). Order-stable.
HASH_FIELDS = (
    "variant_seed", "H_pick", "spawn",
    "T_pick", "box", "T_relay", "cube", "Z_store", "pillar",
    "targets", "grasp",
)

# ---------------------------------------------------------------------------
# Free-body qpos/qvel layout note (consumed by each harness when slicing).
# ---------------------------------------------------------------------------
# The furniture XML appends the box body THEN the cube body, both after all
# robot bodies. With a robot that occupies qpos[0:NQR] / qvel[0:NVR]:
#     qpos: [ robot 0:NQR | box freejoint (7) | cube freejoint (7) ]
#     qvel: [ robot 0:NVR | box freejoint (6) | cube freejoint (6) ]
# So:  box  qpos = [NQR     : NQR + 7],  qvel = [NVR     : NVR + 6]
#      cube qpos = [NQR + 7 : NQR + 14], qvel = [NVR + 6 : NVR + 12]
# (freejoint qpos = [x y z qw qx qy qz]; qvel = [vx vy vz wx wy wz].)
# A harness reads its own NQR/NVR from the compiled model and slices the tail.
FREE_BODY_QPOS_LAYOUT = {
    "order": ["carried_box", "relay_cube"],
    "box_qpos_offset_from_robot_end": 0,
    "box_qpos_size": 7,
    "cube_qpos_offset_from_robot_end": 7,
    "cube_qpos_size": 7,
    "box_qvel_offset_from_robot_end": 0,
    "box_qvel_size": 6,
    "cube_qvel_offset_from_robot_end": 6,
    "cube_qvel_size": 6,
}


# ---------------------------------------------------------------------------
# SceneSpec dataclass (JSON-serialisable). Mirrors the §2 field list.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneSpec:
    variant_seed: int
    H_pick: float
    spawn: List[float]                      # [x, y, yaw]
    T_pick: Dict                            # {pos[x,y], top_h, size[w,d], yaw}
    box: Dict                               # {pos[x,y,z], size[w,d,h], mass}
    T_relay: Dict                           # {pos[x,y], top_h, size[w,d]}
    cube: Dict                              # {pos[x,y,z], size[w,d,h], mass}
    Z_store: Dict                           # {pos[x,y], size[w,d], rim_h}
    pillar: Dict                            # {pos[x,y], r, h}
    lane: Dict                              # {dir, length}
    targets: Dict                           # {P_pick_front, ...}[x,y,yaw]
    grasp: Dict                             # {box_grasp_h, d_grasp, cube_touch_d}
    free_body_layout: Dict = field(default_factory=lambda: dict(FREE_BODY_QPOS_LAYOUT))
    scene_spec_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Math helpers (yaw-only planar poses; immutable, return new tuples/lists).
# ---------------------------------------------------------------------------
def _jit(rng: random.Random, mag: float) -> float:
    """Uniform jitter in [-mag, +mag]."""
    return rng.uniform(-mag, mag)


def _yaw_to_target(frm: Tuple[float, float], to: Tuple[float, float]) -> float:
    """Planar yaw that points from ``frm`` toward ``to``."""
    return math.atan2(to[1] - frm[1], to[0] - frm[0])


def _yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    """wxyz quaternion for a pure yaw rotation about +Z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _front_pose(center: Tuple[float, float], spawn_xy: Tuple[float, float],
                standoff: float) -> List[float]:
    """A stand-off pose ``standoff`` metres from ``center`` on the side facing
    ``spawn_xy`` (so the robot approaches the near edge), facing ``center``."""
    dx, dy = spawn_xy[0] - center[0], spawn_xy[1] - center[1]
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    px, py = center[0] + ux * standoff, center[1] + uy * standoff
    yaw = _yaw_to_target((px, py), center)  # face the table centre
    return [px, py, yaw]


# ---------------------------------------------------------------------------
# build_variant: deterministic scene from a seed.
# ---------------------------------------------------------------------------
def build_variant(seed: int) -> SceneSpec:
    """Return a fully-resolved :class:`SceneSpec` for ``variant_seed=seed``.

    Determinism: a single ``random.Random(seed)`` drives all jitter draws in a
    fixed order, so the same seed yields byte-identical furniture truth across
    machines/models (only the robot URDF differs per model).
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int, got %r" % (seed,))

    rng = random.Random(seed)

    H_pick = H_PICK_LEVELS[seed % 4]

    # --- spawn ---
    spawn = [
        _jit(rng, JIT_SPAWN_XY),
        _jit(rng, JIT_SPAWN_XY),
        _jit(rng, JIT_SPAWN_YAW),
    ]
    spawn_xy = (spawn[0], spawn[1])

    # --- T_pick (pick table) ---
    tpick_x = T_PICK_POS_NOM[0] + _jit(rng, JIT_TPICK_XY)
    tpick_y = T_PICK_POS_NOM[1] + _jit(rng, JIT_TPICK_XY)
    tpick_yaw = _jit(rng, JIT_TPICK_YAW)
    T_pick = {
        "pos": [tpick_x, tpick_y],
        "top_h": H_pick,
        "size": [T_PICK_TOP_HALF[0] * 2.0, T_PICK_TOP_HALF[1] * 2.0],  # w, d
        "yaw": tpick_yaw,
    }

    # --- box on T_pick top, +/-0.05 m jitter on the top face ---
    box_x = tpick_x + _jit(rng, JIT_BOX_ON_TOP)
    box_y = tpick_y + _jit(rng, JIT_BOX_ON_TOP)
    box_z = H_pick + BOX_HALF[2]  # bottom on the top face -> centre at top+half
    box = {
        "pos": [box_x, box_y, box_z],
        "size": list(BOX_FULL),
        "mass": BOX_MASS,
    }

    # --- T_relay (relay table), fixed top height ---
    trelay_x = T_RELAY_POS_NOM[0] + _jit(rng, JIT_TRELAY_XY)
    trelay_y = T_RELAY_POS_NOM[1] + _jit(rng, JIT_TRELAY_XY)
    T_relay = {
        "pos": [trelay_x, trelay_y],
        "top_h": T_RELAY_TOP_H,
        "size": [T_RELAY_TOP_HALF[0] * 2.0, T_RELAY_TOP_HALF[1] * 2.0],
    }

    # --- cube on T_relay top ---
    cube_x = trelay_x + _jit(rng, JIT_CUBE_ON_TOP)
    cube_y = trelay_y + _jit(rng, JIT_CUBE_ON_TOP)
    cube_z = T_RELAY_TOP_H + CUBE_HALF[2]
    cube = {
        "pos": [cube_x, cube_y, cube_z],
        "size": list(CUBE_FULL),
        "mass": CUBE_MASS,
    }

    # --- Z_store (placement zone) ---
    zstore_x = Z_STORE_POS_NOM[0] + _jit(rng, JIT_ZSTORE_XY)
    zstore_y = Z_STORE_POS_NOM[1] + _jit(rng, JIT_ZSTORE_XY)
    Z_store = {
        "pos": [zstore_x, zstore_y],
        "size": [Z_STORE_HALF[0] * 2.0, Z_STORE_HALF[1] * 2.0],
        "rim_h": Z_STORE_RIM_H,
    }

    # --- pillar ---
    pillar_x = PILLAR_POS_NOM[0] + _jit(rng, JIT_PILLAR_XY)
    pillar_y = PILLAR_POS_NOM[1] + _jit(rng, JIT_PILLAR_XY)
    pillar = {"pos": [pillar_x, pillar_y], "r": PILLAR_R, "h": PILLAR_H}

    lane = {"dir": [1.0, 0.0], "length": LANE_LENGTH}

    # --- derived navigation targets ---
    P_pick_front = _front_pose((tpick_x, tpick_y), spawn_xy, FRONT_STANDOFF)
    P_relay_front = _front_pose((trelay_x, trelay_y), spawn_xy, FRONT_STANDOFF)
    P_store = _front_pose((zstore_x, zstore_y), spawn_xy, FRONT_STANDOFF)

    # P_cube_touch: a fixed pose where the right wrist can reach the cube. Back-
    # solved from the cube using the arm reach: stand ARM_REACH in front of the
    # cube on the side facing spawn, facing the cube. The robot must be precise
    # here -- this is M2's hardest nav leg.
    P_cube_touch = _front_pose((cube_x, cube_y), spawn_xy, ARM_REACH)

    targets = {
        "P_pick_front": P_pick_front,
        "P_relay_front": P_relay_front,
        "P_cube_touch": P_cube_touch,
        "P_store": P_store,
    }

    grasp = {
        "box_grasp_h": box_z,         # box centre height when grasped on table
        "d_grasp": D_GRASP,
        "cube_touch_d": CUBE_TOUCH_D,
        "arm_reach": ARM_REACH,
    }

    spec_no_hash = SceneSpec(
        variant_seed=seed,
        H_pick=H_pick,
        spawn=spawn,
        T_pick=T_pick,
        box=box,
        T_relay=T_relay,
        cube=cube,
        Z_store=Z_store,
        pillar=pillar,
        lane=lane,
        targets=targets,
        grasp=grasp,
    )

    spec_hash = _compute_hash(spec_no_hash)
    # frozen dataclass -> rebuild with the hash filled in.
    data = spec_no_hash.to_dict()
    data["scene_spec_hash"] = spec_hash
    return SceneSpec(**data)


def _compute_hash(spec: SceneSpec) -> str:
    """md5 over the key truth fields (rounded to 1e-6 to be float-stable)."""
    d = spec.to_dict()
    payload = {k: _round_floats(d[k]) for k in HASH_FIELDS}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _round_floats(obj, ndigits: int = 6):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# furniture_xml: injectable scene fragment + predeclared welds.
# ---------------------------------------------------------------------------
# Collision bit scheme ("bit2"):
#   robot   = bit 1 (contype/conaffinity 1, the harness g1.xml default)
#   floor   = bits 1|2 = 3   (collides with robot AND box/cube)
#   tables  = bits 1|2 = 3   (robot can bump them; box/cube rest on them)
#   pillar  = bits 1|2 = 3   (robot navigates around it)
#   Z_store = bits 1|2 = 3
#   box     = bit 2  (collides with furniture+floor, NOT the bit-1 robot)
#   cube    = bit 2
# At place time a harness flips the placed body's geom to also collide as
# needed (same mechanism as bench_homie release_box_collision()).
FURNITURE_CT_CA = 3      # bits 1|2 for static furniture + floor
FREE_BODY_CT_CA = 2      # bit 2 for box/cube

# Placeholder wrist body names. Each harness sed-replaces WRIST_L / WRIST_R
# with its own wrist link names before compiling, e.g.
#   xml = xml.replace("WRIST_L", "left_wrist_yaw_link") \
#            .replace("WRIST_R", "right_wrist_yaw_link")
WRIST_L_PLACEHOLDER = "WRIST_L"
WRIST_R_PLACEHOLDER = "WRIST_R"


def _table_xml(name: str, cx: float, cy: float, top_h: float,
               half_w: float, half_d: float, yaw: float, rgba: str,
               ct_ca: int) -> str:
    """A table = box geom top face on 4 leg geoms, all static (no joint).
    The top face is centred at z=top_h-leg_top_half... actually we model the
    top as a thin slab whose TOP surface sits at z=top_h, with legs reaching
    the floor. Yaw rotates the whole body about +Z."""
    top_half_t = 0.02  # 0.04 m thick slab
    top_cz = top_h - top_half_t
    quat = _yaw_quat(yaw)
    # leg corner offsets (in the table's local frame, inset from edges)
    lx = half_w - LEG_INSET
    ly = half_d - LEG_INSET
    leg_half_h = (top_h - 2 * top_half_t) / 2.0
    leg_cz = leg_half_h  # legs centred so bottom sits on the floor
    legs = []
    for i, (sx, sy) in enumerate(((lx, ly), (lx, -ly), (-lx, ly), (-lx, -ly))):
        legs.append(
            '<geom name="%s_leg%d" type="box" size="%g %g %g" '
            'pos="%g %g %g" contype="%d" conaffinity="%d" '
            'rgba="%s"/>' % (
                name, i, LEG_HALF, LEG_HALF, leg_half_h,
                sx, sy, leg_cz, ct_ca, ct_ca, rgba)
        )
    return (
        '<body name="%s" pos="%g %g 0" quat="%g %g %g %g">'
        '<geom name="%s_top" type="box" size="%g %g %g" pos="0 0 %g" '
        'contype="%d" conaffinity="%d" rgba="%s"/>'
        '%s'
        '</body>'
    ) % (
        name, cx, cy, quat[0], quat[1], quat[2], quat[3],
        name, half_w, half_d, top_half_t, top_cz, ct_ca, ct_ca, rgba,
        "".join(legs),
    )


def _zstore_xml(cx: float, cy: float, half_w: float, half_d: float,
                rim_h: float, ct_ca: int) -> str:
    """Shallow framed pad: a thin floor plate + 4 low rim walls (static)."""
    plate_half_t = 0.01
    rim_half_t = 0.01
    rim_half_h = rim_h / 2.0
    rgba = "0.20 0.55 0.85 0.6"
    plate = (
        '<geom name="zstore_plate" type="box" size="%g %g %g" pos="0 0 %g" '
        'contype="%d" conaffinity="%d" rgba="%s"/>' % (
            half_w, half_d, plate_half_t, plate_half_t,
            ct_ca, ct_ca, rgba)
    )
    rims = []
    # +X / -X walls (run along y), +Y / -Y walls (run along x)
    rims.append('<geom name="zstore_rim_xp" type="box" size="%g %g %g" '
                'pos="%g 0 %g" contype="%d" conaffinity="%d" rgba="%s"/>' % (
                    rim_half_t, half_d, rim_half_h, half_w, rim_half_h,
                    ct_ca, ct_ca, rgba))
    rims.append('<geom name="zstore_rim_xn" type="box" size="%g %g %g" '
                'pos="%g 0 %g" contype="%d" conaffinity="%d" rgba="%s"/>' % (
                    rim_half_t, half_d, rim_half_h, -half_w, rim_half_h,
                    ct_ca, ct_ca, rgba))
    rims.append('<geom name="zstore_rim_yp" type="box" size="%g %g %g" '
                'pos="0 %g %g" contype="%d" conaffinity="%d" rgba="%s"/>' % (
                    half_w, rim_half_t, rim_half_h, half_d, rim_half_h,
                    ct_ca, ct_ca, rgba))
    rims.append('<geom name="zstore_rim_yn" type="box" size="%g %g %g" '
                'pos="0 %g %g" contype="%d" conaffinity="%d" rgba="%s"/>' % (
                    half_w, rim_half_t, rim_half_h, -half_d, rim_half_h,
                    ct_ca, ct_ca, rgba))
    return (
        '<body name="zstore" pos="%g %g 0">%s%s</body>'
        % (cx, cy, plate, "".join(rims))
    )


def _pillar_xml(cx: float, cy: float, r: float, h: float, ct_ca: int) -> str:
    return (
        '<body name="pillar" pos="%g %g 0">'
        '<geom name="pillar_geom" type="cylinder" size="%g %g" pos="0 0 %g" '
        'contype="%d" conaffinity="%d" rgba="0.5 0.5 0.5 1"/>'
        '</body>'
    ) % (cx, cy, r, h / 2.0, h / 2.0, ct_ca, ct_ca)


def _free_box_xml(name: str, geom: str, pos: Tuple[float, float, float],
                  half: Tuple[float, float, float], mass: float,
                  rgba: str, ct_ca: int) -> str:
    return (
        '<body name="%s" pos="%g %g %g">'
        '<freejoint name="%s_freejoint"/>'
        '<geom name="%s" type="box" size="%g %g %g" mass="%g" '
        'contype="%d" conaffinity="%d" rgba="%s"/>'
        '</body>'
    ) % (
        name, pos[0], pos[1], pos[2],
        name,
        geom, half[0], half[1], half[2], mass, ct_ca, ct_ca, rgba,
    )


def _weld_block() -> str:
    """Predeclared <equality> with 4 welds, all active=false at compile time.

    relpose left at identity (placeholder) -- harnesses re-anchor at grasp time
    (see bench_homie configure_box_welds()). solref/solimp kept compliant so the
    two wrist<->box welds don't fight each other (over-constrained free body) or
    the upper-body PD: a stiffer "0.02 1" spikes qacc when standing up from a
    deep squat with the box anchored at near-max reach (low-table variant). The
    softer "0.05 1" + relaxed solimp absorbs that without visibly slipping the
    box. Body names use WRIST_L/WRIST_R placeholders that each harness
    sed-replaces with its own wrist link names.
    """
    sr = 'solref="0.05 1" solimp="0.8 0.95 0.002 0.5 2"'
    return (
        '<equality>'
        '<weld name="box_weld_left" active="false" '
        'body1="%s" body2="carried_box" %s/>'
        '<weld name="box_weld_right" active="false" '
        'body1="%s" body2="carried_box" %s/>'
        '<weld name="cube_weld_box" active="false" '
        'body1="carried_box" body2="relay_cube" %s/>'
        '<weld name="cube_weld_right" active="false" '
        'body1="%s" body2="relay_cube" %s/>'
        '</equality>'
    ) % (WRIST_L_PLACEHOLDER, sr, WRIST_R_PLACEHOLDER, sr, sr,
         WRIST_R_PLACEHOLDER, sr)


def furniture_xml(spec: SceneSpec, collision_scheme: str = "bit2") -> Dict[str, str]:
    """Return injectable XML fragments for ``spec``.

    Result keys:
      ``bodies``   -- furniture + free-body XML to append inside a <worldbody>
                      (append AFTER all robot bodies so free-body coords land at
                      the qpos/qvel tail; see FREE_BODY_QPOS_LAYOUT).
      ``equality`` -- the <equality> weld block to insert before </mujoco>.

    Each harness does, after substituting its wrist link names:
        frags = furniture_xml(spec)
        xml = xml.replace(FLOOR_ANCHOR, FLOOR_ANCHOR + frags["bodies"])
        xml = xml.replace("</mujoco>", frags["equality"] + "</mujoco>")
        xml = xml.replace("WRIST_L", "left_wrist_yaw_link") \\
                 .replace("WRIST_R", "right_wrist_yaw_link")
    and at runtime ORs bit 2 into the floor plane(s) so the box/cube collide
    with the ground (same as bench_homie's bit-2 floor patch).
    """
    if collision_scheme != "bit2":
        raise ValueError("only collision_scheme='bit2' is supported, got %r"
                         % (collision_scheme,))

    d = spec.to_dict()
    tp, tr, zs, pl = d["T_pick"], d["T_relay"], d["Z_store"], d["pillar"]
    bx, cb = d["box"], d["cube"]

    tpick = _table_xml(
        "table_pick", tp["pos"][0], tp["pos"][1], tp["top_h"],
        tp["size"][0] / 2.0, tp["size"][1] / 2.0, tp["yaw"],
        "0.55 0.42 0.26 1", FURNITURE_CT_CA)
    trelay = _table_xml(
        "table_relay", tr["pos"][0], tr["pos"][1], tr["top_h"],
        tr["size"][0] / 2.0, tr["size"][1] / 2.0, 0.0,
        "0.42 0.32 0.20 1", FURNITURE_CT_CA)
    zstore = _zstore_xml(
        zs["pos"][0], zs["pos"][1], zs["size"][0] / 2.0, zs["size"][1] / 2.0,
        zs["rim_h"], FURNITURE_CT_CA)
    pillar = _pillar_xml(pl["pos"][0], pl["pos"][1], pl["r"], pl["h"],
                         FURNITURE_CT_CA)

    box = _free_box_xml(
        "carried_box", "carried_box_geom",
        (bx["pos"][0], bx["pos"][1], bx["pos"][2]),
        (BOX_HALF[0], BOX_HALF[1], BOX_HALF[2]), bx["mass"],
        "1.0 0.85 0.1 1", FREE_BODY_CT_CA)
    cube = _free_box_xml(
        "relay_cube", "relay_cube_geom",
        (cb["pos"][0], cb["pos"][1], cb["pos"][2]),
        (CUBE_HALF[0], CUBE_HALF[1], CUBE_HALF[2]), cb["mass"],
        "0.15 0.7 0.25 1", FREE_BODY_CT_CA)

    # Static furniture first, then the two free bodies LAST (box then cube) so
    # the qpos/qvel tail order matches FREE_BODY_QPOS_LAYOUT.
    bodies = tpick + trelay + zstore + pillar + box + cube
    return {"bodies": bodies, "equality": _weld_block()}


# ---------------------------------------------------------------------------
# CLI: dump a few variants' specs as JSON (numeric sanity).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 25]
    for s in seeds:
        spec = build_variant(s)
        print(spec.to_json())
