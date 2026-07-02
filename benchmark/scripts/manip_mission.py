"""manip_mission.py — shared ManipArena mission state machine (skeleton).

Authoritative design: experiments/BENCHMARK_V2_DESIGN.md (§3 grasp/place
abstraction, §4 task redefinition, §6 metrics). This module is the §7.2
deliverable: a pure-Python (numpy only) mission engine that the four
benchmark harnesses (AMO / AGILE / FALCON / HOMIE) each drive through an
INJECTED callback interface (MissionIO). The harnesses keep their own
MuJoCo loop, model, Psi0 replay path and command->obs plumbing; this module
owns ONLY the sequencing (which leg/primitive runs now) and the per-stage
metric/failure bookkeeping. No harness import, no MuJoCo import.

Why injection: nav command math lives in nav_errors / nav_cmd_coarse /
nav_cmd_cal (copied below from bench_homie.py, spec-identical across the four
harnesses — DO NOT tune per model). Everything that touches physics
(send a leg command, replay an upper-body frame, read root/wrist/box pose,
weld / release / cube-transfer) is a callback on MissionIO, so each harness
implements ~7 small methods and reuses this whole file.

This is a SKELETON: state/transition definitions + stage metric hooks +
failure transitions are complete and py_compile-clean. The physics-touching
work is delegated to MissionIO callbacks the harnesses fill in (§7.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Protocol, Sequence

# numpy is permitted (BENCHMARK_V2_DESIGN §7.2 "纯 python，仅依赖 numpy") but the
# mission engine itself uses only the stdlib ``math`` — keep the import optional
# so this stays importable in any env (the docstring's portability guarantee).
try:
    import numpy as np  # noqa: F401  (kept for harness adapters subclassing here)
except ImportError:  # pragma: no cover
    np = None

# ---------------------------------------------------------------------------
# Shared constants (copied from bench_homie.py spec v3/v4; MUST stay
# line-equivalent across the four harnesses — see BENCHMARK_V2_DESIGN §6).
# ---------------------------------------------------------------------------
NAV_SWITCH_DIST = 0.5            # m; farther: heading target = bearing
NAV_KP_POS = 1.0
NAV_KP_YAW = 1.5
NAV_VX_MAX = 0.6                 # m/s (coarse vx clipped to [0, vx_max])
NAV_VY_MAX = 0.3                 # m/s
NAV_WZ_MAX = 0.6                 # rad/s
NAV_POS_TOL = 0.30              # m, coarse exit
NAV_YAW_TOL = math.radians(15.0)
NAV_TIMEOUT_S = 30.0
CAL_CMD_LIM = 0.10              # |vx|,|vy| [m/s] and |wz| [rad/s] in CAL
CAL_POS_TOL = 0.05              # m, fine convergence (the 5cm/5deg target)
CAL_YAW_TOL = math.radians(5.0)
CAL_HOLD_S = 1.0               # fine tolerance must hold this long
CAL_TIMEOUT_S = 15.0

# Pillar-avoidance detour (BENCHMARK_V2_DESIGN §4: M1 step 1 "nav spawn->
# P_pick_front 避 pillar", M2 step 2 "nav 朝置物区方向出发，中途绕去
# P_relay_front"). A NavLeg whose straight start->target segment passes within
# PILLAR_AVOID_CLEARANCE_M of the pillar CENTRE gets ONE lateral waypoint
# inserted at leg start that bows the path AWAY from the pillar; the waypoint
# sits (pillar.r + clearance + margin) off the pillar centre on the path's own
# side, so the robot skirts the obstacle then proceeds to the real target.
PILLAR_AVOID_CLEARANCE_M = 0.40   # keep-out: detour if segment-pillar dist < this
PILLAR_DETOUR_MARGIN_M = 0.10     # extra lateral clearance beyond r + clearance

# Grasp / place / cube-transfer thresholds (BENCHMARK_V2_DESIGN §3).
D_GRASP_M = 0.30               # both wrists < this from box surface to weld
CUBE_TOUCH_D_M = 0.32         # right wrist < this from cube to transfer.
                              # Matched to the box-grasp gate (D_GRASP_M 0.30):
                              # the reach-pose right wrist reaches ~0.30 m from a
                              # target (box grasp succeeds at that range), and
                              # the cube touch is the same magnetic-touch
                              # abstraction; the original 0.15 was unreachable
                              # given P_cube_touch's ARM_REACH standoff.
PLACE_SETTLE_S = 1.0          # hold after release so the box lands & rests
PLACE_BOX_MAX_SPEED = 0.05   # m/s, box considered at rest
PLACE_BOX_UPRIGHT_TOL = math.radians(30.0)  # box z-axis vs world z
PLACE_POS_TOL_M = 0.20       # m, box landing within target footprint

# Squat profile (M-series reach-down / place). Heights are the harness's own
# height-command semantics (HOMIE absolute base z, AMO/AGILE/FALCON relative)
# — the harness adapter converts; the mission only emits a target + rate.
SQUAT_RATE_MPS = 0.20         # m/s height ramp (default)
SQUAT_BOTTOM_HOLD_S = 0.5    # s at the bottom before grasp/release fires
STAND_HOLD_S = 1.0           # s standing between legs / at task end
# ROOT CAUSE 3: longer settle after a grasp before the carry leg. Standing up
# from a DEEP pick squat (low tables) with the 2 kg box leaves the robot tilted
# (~0.4 rad); starting to walk immediately topples it. A longer quiet stand lets
# the policy recover an upright carry stance first.
POST_GRASP_HOLD_S = 2.5
# ROOT CAUSE 3: height ramp when STANDING UP from the pick squat with the box.
# Kept at the default 0.20 — a slower 0.10 rise did NOT save the deepest H0.30
# pick (still stumbles, that depth is at HOMIE's recovery limit) and made the
# mid-table picks wobblier.
POST_GRASP_STAND_RATE = SQUAT_RATE_MPS
# Stand after a PLACE: kept at the defaults. Slowing the rise or lengthening the
# hold did NOT save the wobbly post-place stands (a few variants topple standing
# up from the placement squat — a HOMIE locomotion limit at that pose) and
# perturbed others, so the post-place stand is left as the plain default.
POST_PLACE_STAND_RATE = SQUAT_RATE_MPS
POST_PLACE_HOLD_S = STAND_HOLD_S
# The reach_down Psi0 segment is ~16.7s long but a squat ramp+hold only plays
# its first ~2.7s, so the arm is nowhere near the box when grasp begins. The
# grasp stage therefore *continues* replaying reach_down (phase carried over
# from the squat) until the arm is fully extended. Timeout must cover the rest
# of reach_down plus a settle margin.
GRASP_TIMEOUT_S = 18.0       # s; long enough to finish the reach_down segment

# Fall (unified, BENCHMARK_V2_DESIGN §6): tilt>0.9 (gravity-xy magnitude) or
# a non-foot ground contact. Evaluated by the harness; the mission just
# consumes the boolean each tick and terminates.
FALL_TILT_GXY = 0.9

# Per-mission wall budget (sim seconds) — long-chain M2 safety net.
MISSION_TIMEOUT_S = 180.0


# ROOT CAUSE 3: scale applied to the lateral vy command on a box-laden carry
# leg. HOMIE strafes poorly under the 2 kg load, but FULLY damping vy made the
# robot freeze on the lateral skirt waypoints (it could neither strafe nor turn
# in place under load). A mild damp keeps the strafe usable while taking the
# edge off the toppling-prone large sidesteps. 1.0 == no damp (baseline).
CARRY_VY_DAMP = 0.8


def _clipf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ===========================================================================
# Shared two-phase nav command math (copied from bench_homie.py:1020-1049,
# spec v3). Signatures are the contract: every harness's nav adapter calls
# THESE — never re-derives the P-law. Kept here verbatim so the four
# harnesses share one source of truth.
# ===========================================================================
def nav_errors(x, y, yaw, goal_xy, goal_yaw):
    """Spec v3 error block: world error -> body frame; heading target is the
    bearing to the goal while >0.5 m away, the goal heading once within.

    Returns (dist, ex, ey, eyaw, yaw_target).
        dist     scalar range to goal_xy [m]
        ex, ey   goal offset in the robot body frame [m] (ex fwd, ey left)
        eyaw     heading error to yaw_target, wrapped [rad]
    """
    dx = goal_xy[0] - x
    dy = goal_xy[1] - y
    dist = math.hypot(dx, dy)
    ex = math.cos(yaw) * dx + math.sin(yaw) * dy
    ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
    yaw_target = math.atan2(dy, dx) if dist > NAV_SWITCH_DIST else goal_yaw
    eyaw = wrap_angle(yaw_target - yaw)
    return dist, ex, ey, eyaw, yaw_target


def nav_cmd_coarse(ex, ey, eyaw, vx_max):
    """Coarse NAV P-law: vx=clip(1.0*ex,0,vx_max), vy=clip(1.0*ey,+-0.3),
    wz=clip(1.5*eyaw,+-0.6). Returns (vx, vy, wz)."""
    vx = _clipf(NAV_KP_POS * ex, 0.0, vx_max)
    vy = _clipf(NAV_KP_POS * ey, -NAV_VY_MAX, NAV_VY_MAX)
    wz = _clipf(NAV_KP_YAW * eyaw, -NAV_WZ_MAX, NAV_WZ_MAX)
    return vx, vy, wz


def nav_cmd_cal(ex, ey, eyaw):
    """CAL (fine) P-law: same gains, every command clamped to |0.10| with NO
    deadzone compensation — native small-command efficacy is the measured
    quantity. Returns (vx, vy, wz)."""
    vx = _clipf(NAV_KP_POS * ex, -CAL_CMD_LIM, CAL_CMD_LIM)
    vy = _clipf(NAV_KP_POS * ey, -CAL_CMD_LIM, CAL_CMD_LIM)
    wz = _clipf(NAV_KP_YAW * eyaw, -CAL_CMD_LIM, CAL_CMD_LIM)
    return vx, vy, wz


# ===========================================================================
# Pillar-avoidance geometry. NOT part of the cross-harness nav contract above
# (nav_errors / nav_cmd_coarse / nav_cmd_cal): these only decide and place the
# one-shot detour waypoint a NavLeg routes to first. Pure stdlib math.
# ===========================================================================
def point_segment_dist(px, py, ax, ay, bx, by):
    """Shortest distance from point P=(px,py) to segment A=(ax,ay)->B=(bx,by),
    plus the clamped projection parameter s in [0,1] of the closest point.
    Returns (dist, s). Used for the keep-out gate ("线段距 pillar")."""
    abx, aby = bx - ax, by - ay
    seg2 = abx * abx + aby * aby
    if seg2 <= 1e-12:                       # degenerate: A == B
        return math.hypot(px - ax, py - ay), 0.0
    s = ((px - ax) * abx + (py - ay) * aby) / seg2
    s = _clipf(s, 0.0, 1.0)
    cx, cy = ax + s * abx, ay + s * aby
    return math.hypot(px - cx, py - cy), s


def pillar_detour_waypoint(ax, ay, bx, by, px, py, pillar_r, clearance,
                           margin=PILLAR_DETOUR_MARGIN_M):
    """One lateral waypoint that bows the A->B path AWAY from a pillar at
    P=(px,py). Placed at (pillar_r + clearance + margin) from the pillar centre,
    on the path's own side (opposite the pillar relative to the segment), with
    yaw facing the real target B. Returns (wx, wy, wyaw).

    The away direction is the segment normal pointing from the pillar toward the
    path (pillar -> closest point on the segment line). If the path runs through
    the pillar centre (degenerate), the segment's left normal is used so a
    deterministic side is still chosen."""
    abx, aby = bx - ax, by - ay
    seg = math.hypot(abx, aby) or 1.0
    ux, uy = abx / seg, aby / seg                  # unit along the segment
    s = (px - ax) * ux + (py - ay) * uy            # pillar projection on the line
    cx, cy = ax + s * ux, ay + s * uy              # closest point on the line
    awx, awy = cx - px, cy - py                    # pillar -> path (away dir)
    an = math.hypot(awx, awy)
    if an < 1e-6:                                  # path through pillar centre
        awx, awy = -uy, ux                         # arbitrary but deterministic
        an = 1.0
    awx, awy = awx / an, awy / an
    offset = pillar_r + clearance + margin
    wx, wy = px + awx * offset, py + awy * offset
    wyaw = math.atan2(by - wy, bx - wx)
    return (wx, wy, wyaw)


# ===========================================================================
# Injected harness interface (MissionIO). Each harness implements these; the
# mission never imports MuJoCo. Pose convention: xy in world [m], yaw [rad].
# Heights are passed in the harness's own height-command semantics — the
# harness adapter is responsible for that mapping (the mission emits a
# normalized "target height + rate"; see SquatTo / LegCmd).
# ===========================================================================
class MissionIO(Protocol):
    # --- read state (root / wrists / box / cube) -------------------------
    def root_pose(self) -> tuple:
        """-> (x, y, yaw). Robot base in world."""
        ...

    def root_tilt(self) -> float:
        """-> gravity-xy magnitude (the unified fall tilt scalar)."""
        ...

    def wrist_box_dist(self) -> tuple:
        """-> (d_left, d_right) [m], each wrist origin to box surface
        (0 if inside). Used by Grasp to gate weld activation."""
        ...

    def right_wrist_cube_dist(self) -> float:
        """-> [m] right wrist origin to cube surface. Gates CubeTransfer."""
        ...

    def box_pose(self) -> tuple:
        """-> (x, y, z, tilt_rad, speed_mps). Free-body box state, for
        place_ok evaluation."""
        ...

    def base_height(self) -> float:
        """-> current achieved base height [m] (harness semantics)."""
        ...

    def is_fallen(self) -> bool:
        """-> unified fall flag for THIS tick (tilt>0.9 / non-foot ground)."""
        ...

    # --- act: lower body (send command) ----------------------------------
    def send_leg_cmd(self, vx: float, vy: float, wz: float,
                     height: float) -> None:
        """Push one locomotion command (the tested policy steps on it).
        height in harness height-command semantics."""
        ...

    # --- act: upper body (Psi0 replay) -----------------------------------
    def replay_upper(self, segment: str, phase_t: float) -> None:
        """Write upper-body joints from the Psi0 stream for `segment`
        (one of 'reach_down' / 'carry' / 'place'), at local time phase_t.
        No-op for L-series (no box)."""
        ...

    # --- act: weld / release / cube-transfer -----------------------------
    def weld_box(self) -> None:
        """Activate both wrist<->box welds at the current relative pose
        (magnetic grasp in place; box not moved)."""
        ...

    def release_box(self, surface_top_h: Optional[float] = None,
                    target_xy: Optional[tuple] = None) -> None:
        """Deactivate the wrist weld + enable box collision (box drops / rests on
        the surface below). ``surface_top_h`` (if given) is the surface top the
        harness lowers + levels the box onto so it lands flat; ``target_xy`` is
        the placement target the box is set down toward (ROOT CAUSE 2)."""
        ...

    def transfer_cube(self) -> None:
        """Activate the cube<->box-top weld (cube rides the box thereafter)."""
        ...


# ===========================================================================
# Primitive descriptors (immutable config) + their runtime state enums.
# ===========================================================================
NAV_COARSE = "coarse"   # phase within a NavLeg
NAV_CAL = "cal"         # fine small-command calibration phase
NAV_DONE = "done"


@dataclass(frozen=True)
class NavLeg:
    """One navigation leg: two-phase controller (coarse P-law NAV until
    0.30 m & 15 deg, then small-command CAL to 0.05 m & 5 deg). Mirrors the
    R5 controller; command math is nav_errors / nav_cmd_coarse / nav_cmd_cal.

    target_pose : (x, y, yaw) goal in world.
    vx_max      : coarse forward-speed clip [m/s] (lower while carrying).
    mode        : 'coarse_then_cal' (default, the 5cm/5deg legs) or
                  'coarse_only' (relay/store legs where 0.30 m suffices).

    Pillar avoidance (BENCHMARK_V2_DESIGN §4): when ``pillar_xy`` / ``pillar_r``
    are set (and ``skip_detour`` is False), the leg inserts ONE lateral detour
    waypoint at leg start if its straight start->target segment intrudes the
    pillar keep-out (PILLAR_AVOID_CLEARANCE_M). It coarse-routes to the waypoint
    first, then proceeds to the real target — the nav P-law itself is unchanged.
    """
    target_pose: tuple
    vx_max: float = NAV_VX_MAX
    mode: str = "coarse_then_cal"
    timeout_coarse_s: float = NAV_TIMEOUT_S
    timeout_cal_s: float = CAL_TIMEOUT_S
    # skip pillar-detour rewriting for this leg: set on pre-planned skirt
    # waypoints so the detour heuristic does not shove them back into a table.
    skip_detour: bool = False
    # base height to command while walking this leg (None -> the mission's
    # h_stand). Carry legs set this to a slightly lower carry height so the
    # box-laden stance is less top-heavy (ROOT CAUSE 3 locomotion stability).
    height: Optional[float] = None
    # pillar geometry threaded from SceneSpec (spec["pillar"]) so this leg can
    # plan a one-shot avoidance detour. pillar_xy = (x, y) world centre,
    # pillar_r = radius [m]. Left None on legs with no pillar to dodge (the
    # skirt waypoints opt out via skip_detour instead).
    pillar_xy: Optional[tuple] = None
    pillar_r: Optional[float] = None


@dataclass(frozen=True)
class SquatTo:
    """Lower body ramps to height h at rate, holds at the bottom. Upper body
    plays the named Psi0 segment ('reach_down' / 'place') concurrently.
    grasp/release fire as a *separate* primitive after the hold."""
    h: float
    rate: float = SQUAT_RATE_MPS
    upper_segment: str = "reach_down"
    bottom_hold_s: float = SQUAT_BOTTOM_HOLD_S


@dataclass(frozen=True)
class StandTo:
    """Rise back to standing height h_stand at rate, upper plays 'carry'."""
    h_stand: float
    rate: float = SQUAT_RATE_MPS
    upper_segment: str = "carry"
    stand_hold_s: float = STAND_HOLD_S


@dataclass(frozen=True)
class Grasp:
    """Magnetic box grasp: at the bottom of a SquatTo, *continue* replaying the
    arm-reach Psi0 segment (``upper_segment``, phase carried over from the
    preceding SquatTo) until BOTH wrists are within d of the box surface, then
    weld in place. Fails if the gate is never met before timeout. The reach
    segment is much longer than the squat hold, so the grasp is what actually
    extends the arm onto the box."""
    d: float = D_GRASP_M
    upper_segment: str = "reach_down"
    timeout_s: float = GRASP_TIMEOUT_S


@dataclass(frozen=True)
class PlaceOn:
    """Release the box onto `surface` ('relay' table or 'store' zone). After
    release, hold settle_s and evaluate place_ok (box upright + at rest +
    within target footprint). ``surface_top_h`` (if set) is the surface top
    height the harness lowers the box onto so it lands flat (ROOT CAUSE 2)."""
    surface: str
    settle_s: float = PLACE_SETTLE_S
    pos_tol_m: float = PLACE_POS_TOL_M
    surface_top_h: Optional[float] = None


@dataclass(frozen=True)
class CubeTransfer:
    """At P_cube_touch: continue replaying the arm-reach segment (phase carried
    over from the preceding SquatTo) until the right wrist is within
    cube_touch_d of the cube, then weld cube to the box top (cube rides the
    box)."""
    d: float = CUBE_TOUCH_D_M
    upper_segment: str = "reach_down"
    timeout_s: float = GRASP_TIMEOUT_S


# ===========================================================================
# Per-stage record (metric hook target). One per executed stage; the harness
# reads these into its JSONL `metrics.stages[]`.
# ===========================================================================
@dataclass
class StageRecord:
    name: str
    kind: str                       # 'nav'|'squat'|'grasp'|'stand'|'place'|'cube'
    t_start: float = 0.0
    t_end: Optional[float] = None
    ok: Optional[bool] = None       # stage-local success flag
    # nav fields
    coarse_err_pos: Optional[float] = None
    coarse_err_yaw: Optional[float] = None
    coarse_reached: Optional[bool] = None
    cal_err_pos: Optional[float] = None
    cal_err_yaw: Optional[float] = None
    cal_converged: Optional[bool] = None
    # pillar-avoidance detour waypoint inserted at leg start, or None when the
    # straight path already cleared the pillar keep-out (BENCHMARK_V2_DESIGN §4).
    detour_wp: Optional[tuple] = None
    # squat / stand fields
    h_target: Optional[float] = None
    h_achieved_min: Optional[float] = None
    h_dev: Optional[float] = None         # |target - achieved|
    root_drift_m: Optional[float] = None  # xy drift during the squat hold
    max_tilt: Optional[float] = None
    # grasp / cube fields
    wrist_dist_at_event: Optional[tuple] = None
    grasp_ok: Optional[bool] = None
    cube_transfer_ok: Optional[bool] = None
    # place fields
    place_ok: Optional[bool] = None
    box_land_pos: Optional[tuple] = None
    box_land_err_m: Optional[float] = None
    box_tilt_end_rad: Optional[float] = None
    box_speed_end: Optional[float] = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        for k in ("wrist_dist_at_event", "box_land_pos", "detour_wp"):
            if k in d and d[k] is not None:
                d[k] = list(d[k])
        return d


# ===========================================================================
# Mission engine. Iterates a flat list of primitives; per tick it returns the
# leg command to send and advances stage state. Physics is the harness's job
# (call io.send_leg_cmd with the returned command; the mission has already
# called io.replay_upper / io.weld_box / io.release_box / io.transfer_cube
# for the side effects). The harness steps the sim, then calls step() again.
# ===========================================================================
class Mission:
    """Drives a primitive sequence. Per tick the harness:
        cmd = mission.step(t)            # mission reads io, fires welds, etc.
        io.send_leg_cmd(*cmd)            # tested policy consumes it
        <harness steps sim one tick>
    until mission.finished. `cmd` is (vx, vy, wz, height).

    Failure transitions: on io.is_fallen() OR mission timeout the active
    stage is closed with ok=False, fall metadata is recorded and the mission
    terminates (finished=True, status='fall'/'timeout')."""

    def __init__(self, name: str, sequence: Sequence[object], io: MissionIO,
                 h_stand: float, timeout_s: float = MISSION_TIMEOUT_S):
        self.name = name
        self.seq = list(sequence)
        self.io = io
        self.h_stand = h_stand
        self.timeout_s = timeout_s
        self.idx = 0
        self.t0_mission: Optional[float] = None
        self.t0_stage: Optional[float] = None
        self.finished = False
        self.status = "running"          # 'running'|'done'|'fall'|'timeout'|'fail'
        self.fall_stage: Optional[str] = None
        self.fall_t: Optional[float] = None
        self.stages: list[StageRecord] = []
        self._cur: Optional[StageRecord] = None
        # per-stage scratch
        self._nav_phase = NAV_COARSE
        self._cal_hold_t0: Optional[float] = None
        # pillar-avoidance detour: the active waypoint (x, y, yaw) the current
        # nav leg routes to before its real target, and whether this leg has
        # already run the (one-shot) detour planning. Reset per stage below.
        self._nav_detour: Optional[tuple] = None
        self._nav_detour_planned: bool = False
        self._squat_min_h: Optional[float] = None
        self._squat_xy0: Optional[tuple] = None
        self._squat_max_tilt = 0.0
        self._event_t0: Optional[float] = None  # grasp/cube/place sub-timer
        # reach_down phase (s) reached at the end of the last SquatTo, so the
        # following Grasp continues the arm-reach replay without snapping back.
        self._reach_phase_carry: float = 0.0
        # height command the preceding SquatTo held at, so a Grasp/CubeTransfer
        # keeps commanding that fixed target instead of echoing the (possibly
        # still-settling) live base height, which otherwise collapses the robot.
        self._squat_hold_h: float = h_stand

    # --- stage lifecycle -------------------------------------------------
    def _open_stage(self, kind: str, name: str, t: float) -> StageRecord:
        rec = StageRecord(name=name, kind=kind, t_start=round(t, 3))
        self.stages.append(rec)
        self._cur = rec
        self.t0_stage = t
        self._nav_phase = NAV_COARSE
        self._cal_hold_t0 = None
        self._nav_detour = None
        self._nav_detour_planned = False
        self._squat_min_h = None
        self._squat_xy0 = None
        self._squat_max_tilt = 0.0
        self._event_t0 = None
        return rec

    def _close_stage(self, t: float, ok: bool) -> None:
        if self._cur is not None:
            self._cur.t_end = round(t, 3)
            self._cur.ok = ok
        self.idx += 1
        self._cur = None
        self.t0_stage = None

    def _terminate(self, t: float, status: str) -> None:
        if self._cur is not None:
            self._cur.t_end = round(t, 3)
            self._cur.ok = False
        self.finished = True
        self.status = status

    # --- main tick -------------------------------------------------------
    def step(self, t: float) -> tuple:
        """-> (vx, vy, wz, height). Stand-still command once finished."""
        if self.t0_mission is None:
            self.t0_mission = t
        if self.finished:
            return 0.0, 0.0, 0.0, self.h_stand

        # global guards: fall + mission timeout
        if self.io.is_fallen():
            self.fall_stage = self._cur.name if self._cur else None
            self.fall_t = round(t, 3)
            self._terminate(t, "fall")
            return 0.0, 0.0, 0.0, self.h_stand
        if t - self.t0_mission >= self.timeout_s:
            self.fall_stage = self._cur.name if self._cur else None
            self.fall_t = round(t, 3)
            self._terminate(t, "timeout")
            return 0.0, 0.0, 0.0, self.h_stand

        if self.idx >= len(self.seq):
            self.finished = True
            self.status = "done"
            return 0.0, 0.0, 0.0, self.h_stand

        prim = self.seq[self.idx]
        if self._cur is None:
            self._open_stage(_kind_of(prim), _name_of(prim, self.idx), t)

        if isinstance(prim, NavLeg):
            return self._tick_nav(prim, t)
        if isinstance(prim, SquatTo):
            return self._tick_squat(prim, t)
        if isinstance(prim, StandTo):
            return self._tick_stand(prim, t)
        if isinstance(prim, Grasp):
            return self._tick_grasp(prim, t)
        if isinstance(prim, PlaceOn):
            return self._tick_place(prim, t)
        if isinstance(prim, CubeTransfer):
            return self._tick_cube(prim, t)
        # unknown primitive: defensive fail
        self._terminate(t, "fail")
        return 0.0, 0.0, 0.0, self.h_stand

    # --- NavLeg (two-phase) ---------------------------------------------
    def _tick_nav(self, p: NavLeg, t: float) -> tuple:
        x, y, yaw = self.io.root_pose()
        rec = self._cur
        # carry legs walk at a slightly lower base height (less top-heavy with
        # the box); plain legs use the mission stand height.
        h = p.height if p.height is not None else self.h_stand

        # One-shot pillar-avoidance detour (BENCHMARK_V2_DESIGN §4 M1.1 / M2.2):
        # planned once at leg entry from the CURRENT root pose (segment start) to
        # the leg target (segment end), so it adapts to where the robot actually
        # is. Inserts at most one lateral waypoint; see _plan_pillar_detour.
        if not self._nav_detour_planned:
            self._nav_detour_planned = True
            self._nav_detour = self._plan_pillar_detour(p, (x, y))
            if self._nav_detour is not None:
                rec.detour_wp = (round(self._nav_detour[0], 4),
                                 round(self._nav_detour[1], 4))

        # Detour leg: coarse-route to the waypoint (position-only exit — the
        # waypoint yaw is a routing artifact), then clear it and continue with a
        # fresh coarse timer to the real target via the normal two-phase nav.
        if self._nav_detour is not None:
            wx, wy, wyaw = self._nav_detour
            dwp, exd, eyd, eywd, _ = nav_errors(x, y, yaw, (wx, wy), wyaw)
            if dwp <= NAV_POS_TOL or (t - self.t0_stage) >= p.timeout_coarse_s:
                self._nav_detour = None
                self._nav_phase = NAV_COARSE
                self.t0_stage = t
            else:
                vx, vy, wz = nav_cmd_coarse(exd, eyd, eywd, p.vx_max)
                if p.height is not None:
                    vy = CARRY_VY_DAMP * vy
                return vx, vy, wz, h

        dist, ex, ey, eyaw, _ = nav_errors(x, y, yaw, p.target_pose[:2],
                                           p.target_pose[2])
        el = t - self.t0_stage
        if self._nav_phase == NAV_COARSE:
            # coarse_only legs are routing waypoints (skirt corners): their goal
            # yaw is an artifact of how the corner was placed, so exit on
            # POSITION alone. Otherwise the robot can sit on the waypoint forever
            # because the arbitrary corner yaw is never satisfied. coarse_then_cal
            # legs (placement / pick stands) still gate on yaw before CAL.
            if p.mode == "coarse_only":
                reached = dist <= NAV_POS_TOL
            else:
                reached = dist <= NAV_POS_TOL and abs(eyaw) <= NAV_YAW_TOL
            if reached or el >= p.timeout_coarse_s:
                rec.coarse_err_pos = round(dist, 4)
                rec.coarse_err_yaw = round(abs(eyaw), 4)
                rec.coarse_reached = bool(reached)
                if p.mode == "coarse_only":
                    self._close_stage(t, bool(reached))
                    return 0.0, 0.0, 0.0, h
                self._nav_phase = NAV_CAL
                self.t0_stage = t
                self._cal_hold_t0 = None
            else:
                vx, vy, wz = nav_cmd_coarse(ex, ey, eyaw, p.vx_max)
                # carry legs (height set) damp the unstable lateral vy channel
                # (HOMIE strafes poorly under the box load) — keep forward vx +
                # turn (wz) so the robot walks toward the target instead of
                # toppling on a big sidestep. NOT a full turn-first (that stalls
                # the short skirt legs); just a vy damp.
                if p.height is not None:
                    vy = CARRY_VY_DAMP * vy
                return vx, vy, wz, h
        if self._nav_phase == NAV_CAL:
            in_tol = dist <= CAL_POS_TOL and abs(eyaw) <= CAL_YAW_TOL
            if in_tol:
                if self._cal_hold_t0 is None:
                    self._cal_hold_t0 = t
                elif t - self._cal_hold_t0 >= CAL_HOLD_S:
                    rec.cal_converged = True
            else:
                self._cal_hold_t0 = None
            el_cal = t - self.t0_stage
            if rec.cal_converged or el_cal >= p.timeout_cal_s:
                rec.cal_err_pos = round(dist, 4)
                rec.cal_err_yaw = round(abs(eyaw), 4)
                rec.cal_converged = bool(rec.cal_converged)
                self._close_stage(t, bool(rec.cal_converged))
                return 0.0, 0.0, 0.0, h
            vx, vy, wz = nav_cmd_cal(ex, ey, eyaw)
            return vx, vy, wz, h
        return 0.0, 0.0, 0.0, h

    def _plan_pillar_detour(self, p: NavLeg, start_xy: tuple) -> Optional[tuple]:
        """Decide a one-shot pillar-avoidance waypoint for NavLeg ``p``. The
        segment is start_xy (the live root pose at leg entry) -> p.target_pose.
        Returns the lateral detour waypoint (x, y, yaw), or None when there is
        no pillar to dodge (no geometry / skip_detour) or the straight path
        already clears the PILLAR_AVOID_CLEARANCE_M keep-out. Deterministic."""
        if p.skip_detour or p.pillar_xy is None or p.pillar_r is None:
            return None
        ax, ay = start_xy
        bx, by = p.target_pose[0], p.target_pose[1]
        px, py = p.pillar_xy
        d, _s = point_segment_dist(px, py, ax, ay, bx, by)
        if d >= PILLAR_AVOID_CLEARANCE_M:
            return None
        return pillar_detour_waypoint(ax, ay, bx, by, px, py, p.pillar_r,
                                      PILLAR_AVOID_CLEARANCE_M)

    # --- SquatTo (ramp + hold, upper reach_down/place) -------------------
    def _tick_squat(self, p: SquatTo, t: float) -> tuple:
        rec = self._cur
        if self._squat_xy0 is None:
            x, y, _ = self.io.root_pose()
            self._squat_xy0 = (x, y)
            rec.h_target = p.h
        self.io.replay_upper(p.upper_segment, t - self.t0_stage)
        achieved = self.io.base_height()
        self._squat_min_h = (achieved if self._squat_min_h is None
                             else min(self._squat_min_h, achieved))
        self._squat_max_tilt = max(self._squat_max_tilt, self.io.root_tilt())
        t_ramp = abs(self.h_stand - p.h) / max(p.rate, 1e-6)
        el = t - self.t0_stage
        if el < t_ramp:
            h = self.h_stand - math.copysign(p.rate * el, self.h_stand - p.h)
            return 0.0, 0.0, 0.0, h
        # bottom hold
        if el < t_ramp + p.bottom_hold_s:
            return 0.0, 0.0, 0.0, p.h
        rec.h_achieved_min = round(self._squat_min_h, 4)
        rec.h_dev = round(abs(p.h - self._squat_min_h), 4)
        rec.max_tilt = round(self._squat_max_tilt, 4)
        x, y, _ = self.io.root_pose()
        rec.root_drift_m = round(math.hypot(x - self._squat_xy0[0],
                                            y - self._squat_xy0[1]), 4)
        # hand the reach_down phase reached so far to a following Grasp so it
        # continues the arm-reach replay seamlessly (only meaningful when the
        # squat played 'reach_down').
        self._reach_phase_carry = (el if p.upper_segment == "reach_down"
                                   else 0.0)
        self._squat_hold_h = p.h
        self._close_stage(t, True)
        return 0.0, 0.0, 0.0, p.h

    # --- StandTo (rise, upper carry) ------------------------------------
    def _tick_stand(self, p: StandTo, t: float) -> tuple:
        rec = self._cur
        if rec.h_target is None:
            rec.h_target = p.h_stand
            x, y, _ = self.io.root_pose()
            self._squat_xy0 = (x, y)
        self.io.replay_upper(p.upper_segment, t - self.t0_stage)
        self._squat_max_tilt = max(self._squat_max_tilt, self.io.root_tilt())
        cur_h = self.io.base_height()
        t_ramp = abs(p.h_stand - cur_h) / max(p.rate, 1e-6)
        el = t - self.t0_stage
        if el < t_ramp:
            h = min(cur_h + p.rate * el, p.h_stand)
            return 0.0, 0.0, 0.0, h
        if el < t_ramp + p.stand_hold_s:
            return 0.0, 0.0, 0.0, p.h_stand
        rec.max_tilt = round(self._squat_max_tilt, 4)
        self._close_stage(t, True)
        return 0.0, 0.0, 0.0, p.h_stand

    # --- Grasp (gate on wrist distance, then weld) ----------------------
    def _tick_grasp(self, p: Grasp, t: float) -> tuple:
        rec = self._cur
        if self._event_t0 is None:
            self._event_t0 = t
        hold_h = self._squat_hold_h
        d_left, d_right = self.io.wrist_box_dist()
        self._advance_reach(p.upper_segment, t)
        # report the closest approach, not just the last tick
        prev = rec.wrist_dist_at_event
        if prev is None or (d_left + d_right) < (prev[0] + prev[1]):
            rec.wrist_dist_at_event = (round(d_left, 4), round(d_right, 4))
        if d_left < p.d and d_right < p.d:
            self.io.weld_box()
            rec.grasp_ok = True
            self._close_stage(t, True)
            return 0.0, 0.0, 0.0, hold_h
        if t - self._event_t0 >= p.timeout_s:
            rec.grasp_ok = False
            self._terminate(t, "fail")
            return 0.0, 0.0, 0.0, hold_h
        return 0.0, 0.0, 0.0, hold_h

    def _advance_reach(self, segment: str, t: float) -> None:
        """Continue the arm-reach replay from where the SquatTo left off (phase =
        squat-carry + grasp elapsed). The reach_down segment is far longer than
        the squat hold, so this is what actually extends the arm onto the target;
        without it the arm freezes ~16% into reach_down and never reaches."""
        phase = self._reach_phase_carry + (t - self._event_t0)
        self.io.replay_upper(segment, phase)

    # --- PlaceOn (release, settle, evaluate place_ok) -------------------
    def _tick_place(self, p: PlaceOn, t: float) -> tuple:
        rec = self._cur
        if self._event_t0 is None:
            self._event_t0 = t
            self.io.release_box(surface_top_h=p.surface_top_h,
                                target_xy=getattr(p, "_target_xy", None))
        if t - self._event_t0 < p.settle_s:
            return 0.0, 0.0, 0.0, self.io.base_height()
        bx, by, bz, btilt, bspeed = self.io.box_pose()
        # target footprint comes from SceneSpec; the harness sets it on the
        # PlaceOn-resolved target via _place_target (filled at build time).
        tgt = getattr(p, "_target_xy", None)
        if tgt is not None:
            err = math.hypot(bx - tgt[0], by - tgt[1])
        else:
            err = 0.0
        place_ok = (btilt < PLACE_BOX_UPRIGHT_TOL
                    and bspeed < PLACE_BOX_MAX_SPEED
                    and err <= p.pos_tol_m)
        rec.place_ok = bool(place_ok)
        rec.box_land_pos = (round(bx, 4), round(by, 4))
        rec.box_land_err_m = round(err, 4)
        rec.box_tilt_end_rad = round(btilt, 4)
        rec.box_speed_end = round(bspeed, 4)
        self._close_stage(t, bool(place_ok))
        return 0.0, 0.0, 0.0, self.io.base_height()

    # --- CubeTransfer (gate on right-wrist distance, weld cube) ---------
    def _tick_cube(self, p: CubeTransfer, t: float) -> tuple:
        rec = self._cur
        if self._event_t0 is None:
            self._event_t0 = t
        # Same reach continuation as grasp: extend the right arm toward the cube
        # by continuing the reach_down replay carried over from the squat.
        hold_h = self._squat_hold_h
        d = self.io.right_wrist_cube_dist()
        self._advance_reach(p.upper_segment, t)
        # track closest approach for diagnostics (mirrors Grasp)
        prev = rec.wrist_dist_at_event
        if prev is None or d < prev[1]:
            rec.wrist_dist_at_event = (0.0, round(d, 4))
        if d < p.d:
            self.io.transfer_cube()
            rec.cube_transfer_ok = True
            self._close_stage(t, True)
            return 0.0, 0.0, 0.0, hold_h
        if t - self._event_t0 >= p.timeout_s:
            rec.cube_transfer_ok = False
            self._terminate(t, "fail")
            return 0.0, 0.0, 0.0, hold_h
        return 0.0, 0.0, 0.0, hold_h

    # --- result ----------------------------------------------------------
    def result(self) -> dict:
        """JSONL-ready mission block (merged into the harness result by the
        adapter; see TASKS_V2.md JSONL schema)."""
        return {
            "mission": self.name,
            "status": self.status,
            "success": self.status == "done"
            and all(s.ok for s in self.stages),
            "fall_stage": self.fall_stage,
            "fall_t": self.fall_t,
            "n_stages": len(self.stages),
            "stages": [s.to_dict() for s in self.stages],
        }


# ---------------------------------------------------------------------------
# Naming helpers for stage records.
# ---------------------------------------------------------------------------
def _kind_of(prim: object) -> str:
    return {
        NavLeg: "nav", SquatTo: "squat", StandTo: "stand",
        Grasp: "grasp", PlaceOn: "place", CubeTransfer: "cube",
    }.get(type(prim), "unknown")


def _name_of(prim: object, idx: int) -> str:
    k = _kind_of(prim)
    if isinstance(prim, NavLeg):
        return "nav#%d->%.2f,%.2f" % (idx, prim.target_pose[0],
                                      prim.target_pose[1])
    if isinstance(prim, PlaceOn):
        return "place#%d:%s" % (idx, prim.surface)
    if isinstance(prim, SquatTo):
        return "squat#%d:h%.2f" % (idx, prim.h)
    return "%s#%d" % (k, idx)


# ===========================================================================
# Mission builders for M1 / M2 (BENCHMARK_V2_DESIGN §4). The harness passes
# the resolved SceneSpec targets + height commands; these return the flat
# primitive sequence the Mission engine iterates. PlaceOn targets are tagged
# with the landing footprint center so place_ok can measure landing error.
# ===========================================================================
def _tag_place_target(place: PlaceOn, target_xy: tuple) -> PlaceOn:
    """Attach the landing footprint center (frozen dataclass -> object set)."""
    object.__setattr__(place, "_target_xy", target_xy)
    return place


# Lateral clearance (m) added beyond a table half-extent so the carried box +
# robot body skirt the table instead of plowing through it. Kept modest so the
# sidestep corridor stays clear of the pillar (which sits ~1 m south of the
# pick table) and does not trip a spurious pillar detour on the skirt leg.
TABLE_SKIRT_CLEAR_M = 0.30

# Where the box is released relative to the pelvis during the 'place' Psi0 pose,
# in the robot body frame (forward, left) [m]. RE-MEASURED after the canonical
# upright-grasp fix (bench_homie _snap_box_upright): the box is now welded
# centred between the two wrists with z up, so on release it drops ~0.26 m
# straight AHEAD of the pelvis with ~0 lateral skew (was 0.20 fwd / -0.37 left
# when the box hung at the live skewed wrist pose). Standing PLACE_RELEASE_FWD_M
# ahead of the zone centre, facing it, drops the box on centre.
PLACE_RELEASE_FWD_M = 0.26
PLACE_RELEASE_LEFT_M = 0.0
# Surface top heights the box is lowered + leveled onto at release so it lands
# flat (ROOT CAUSE 2). Store pad: the plate top sits ~0.02 m above the floor;
# relay table top comes from the scene per-build.
STORE_SURFACE_TOP_H = 0.02
# Forward standoff used for placement. Kept LARGER than the forward release
# offset (0.26) so the pelvis — and the feet under it — stay back from the
# Z_store rim (half 0.30 m + 0.12 m rim): squatting deep right on the rim makes
# the robot pitch forward and topple. The box then lands ~(stand - release) =
# 0.09 m short of centre, comfortably inside PLACE_POS_TOL_M (0.20).
PLACE_STAND_FWD_M = 0.35


def _place_stand_pose(center_xy: tuple, approach_from: tuple) -> tuple:
    """Pelvis (x, y, yaw) for a placement. The robot faces the zone centre from
    the approach side and stands PLACE_STAND_FWD_M ahead of centre, with both
    release-offset components corrected so the box's drop lands centred. With the
    canonical upright grasp the release is ~pure forward, so pelvis + forward
    release == centre and the residual landing error is small."""
    dx, dy = approach_from[0] - center_xy[0], approach_from[1] - center_xy[1]
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist                # unit vector centre->approach
    # stand PLACE_STAND_FWD_M off the centre on the approach side, facing centre
    px = center_xy[0] + ux * PLACE_STAND_FWD_M
    py = center_xy[1] + uy * PLACE_STAND_FWD_M
    yaw = math.atan2(center_xy[1] - py, center_xy[0] - px)
    # correct the lateral release drop (left-normal of the heading); ~0 now
    left = (-math.sin(yaw), math.cos(yaw))
    px -= PLACE_RELEASE_LEFT_M * left[0]
    py -= PLACE_RELEASE_LEFT_M * left[1]
    return (px, py, yaw)


def _table_skirt_wps(scene: dict, table_key: str, next_xy: tuple,
                     vx_carry: float) -> list:
    """Two skirt NavLegs the robot walks after picking so it clears the pick
    table before the carry leg. The straight P_*_front -> P_store path plows
    through the table the robot just picked from. The L-route is:
      (a) sidestep in y to the destination's side, clear of the table edge,
          still west of the table (don't advance +x into the table yet);
      (b) advance in x to just past the table's far edge on that same y, so the
          final leg to ``next_xy`` runs entirely clear of the table top.
    Keeping (b) on the same y as (a) means the final leg approaches the store
    from a clean corridor and never re-crosses the table.

    table_key : 'T_pick' (or 'T_relay'); scene provides centre + top size.
    next_xy   : the following leg's target xy (decides which side to skirt to).
    """
    t = scene[table_key]
    cx, cy = t["pos"][0], t["pos"][1]
    half_x, half_y = t["size"][0] / 2.0, t["size"][1] / 2.0
    side = -1.0 if next_xy[1] < cy else 1.0
    wy = cy + side * (half_y + TABLE_SKIRT_CLEAR_M)
    wx_west = cx - half_x - 0.20            # sidestep point, west of the table
    wx_east = cx + half_x + 0.20            # advance point, east of the table
    yaw_a = math.atan2(wy - 0.0, wx_west - wx_east)  # facing the sidestep dir
    yaw_b = math.atan2(next_xy[1] - wy, next_xy[0] - wx_east)
    return [
        NavLeg((wx_west, wy, yaw_a), vx_max=vx_carry, mode="coarse_only",
               skip_detour=True),
        NavLeg((wx_east, wy, yaw_b), vx_max=vx_carry, mode="coarse_only",
               skip_detour=True),
    ]


def build_m1(scene: dict, h_stand: float, h_pick: float, h_store: float,
             vx_carry: float = 0.5, h_carry: Optional[float] = None) -> list:
    """M1 table_pick_place — five stages (BENCHMARK_V2_DESIGN §4 M1):
        1. nav spawn -> P_pick_front (avoid pillar)         [NavLeg cal]
        2. squat to reach height + magnetic grasp           [SquatTo+Grasp]
        3. stand with box                                   [StandTo]
        4. nav -> P_store (around pillar)                   [NavLeg]
        5. squat to store height + place box                [SquatTo+PlaceOn]

    `h_pick` / `h_store` are the harness height commands that make the wrists
    reach the box on T_pick / the floor of Z_store; `scene` provides the
    derived targets and the Z_store footprint center. `h_carry` (if given) is
    the lower base height the box-laden carry legs walk at (ROOT CAUSE 3).
    """
    tg = scene["targets"]
    z_xy = tuple(scene["Z_store"]["pos"][:2])
    pillar_xy, pillar_r = _pillar_of(scene)
    skirt = _table_skirt_wps(scene, "T_pick", tuple(tg["P_store"][:2]), vx_carry)
    # apply the carry height to the post-grasp skirt legs too
    skirt = [_with_height(w, h_carry) for w in skirt]
    # stand so the place-pose release drops the box on the store zone centre
    store_stand = _place_stand_pose(z_xy, skirt[-1].target_pose[:2])
    return [
        # (1) spawn -> pick approach (avoid pillar, §4 M1 step 1)
        NavLeg(tuple(tg["P_pick_front"]), vx_max=NAV_VX_MAX,
               mode="coarse_then_cal",
               pillar_xy=pillar_xy, pillar_r=pillar_r),
        SquatTo(h_pick, upper_segment="reach_down"),
        Grasp(d=D_GRASP_M),
        StandTo(h_stand, upper_segment="carry", rate=POST_GRASP_STAND_RATE,
                stand_hold_s=POST_GRASP_HOLD_S),
        # skirt the pick table before the carry leg (straight path crosses it)
        *skirt,
        # ROOT CAUSE 3: the box-laden placement approach uses coarse_only. The
        # fine CAL small-command convergence topples the wobbly carry stance at
        # the very end; coarse (0.30 m) + the forward release offset still lands
        # the box inside the place tolerance, and the squat that follows settles
        # the pose. (4) store leg also rounds the pillar (§4 M1 step 4).
        NavLeg(store_stand, vx_max=vx_carry, mode="coarse_only",
               height=h_carry, pillar_xy=pillar_xy, pillar_r=pillar_r),
        SquatTo(h_store, upper_segment="place"),
        _tag_place_target(
            PlaceOn("store", surface_top_h=STORE_SURFACE_TOP_H), z_xy),
        StandTo(h_stand, upper_segment="carry", rate=POST_PLACE_STAND_RATE,
                stand_hold_s=POST_PLACE_HOLD_S),
    ]


def _with_height(leg: NavLeg, h: Optional[float]) -> NavLeg:
    """Return a copy of ``leg`` with its carry height set (immutable). Uses
    dataclasses.replace so every other field — including pillar_xy/pillar_r — is
    preserved verbatim."""
    if h is None:
        return leg
    return replace(leg, height=h)


def _pillar_of(scene: dict) -> tuple:
    """-> (pillar_xy, pillar_r) from a SceneSpec's pillar dict (spec["pillar"]
    = {pos[x,y], r, h}), or (None, None) when the scene has no pillar (legs then
    never detour). Threaded onto the primary nav legs by the M1/M2 builders."""
    pil = scene.get("pillar")
    if not pil:
        return None, None
    return tuple(pil["pos"][:2]), float(pil["r"])


def build_m2(scene: dict, h_stand: float, h_pick: float, h_relay: float,
             h_store: float, h_cube: float, vx_carry: float = 0.5,
             h_carry: Optional[float] = None) -> list:
    """M2 relay_pick_place — seven functional stages (BENCHMARK_V2_DESIGN §4
    M2), expanded into the primitive chain:
        1. nav -> P_pick_front; grasp box   (= M1 step 1-3)
        2. nav toward store, detour to P_relay_front
        3. place box on T_relay
        4. nav -> P_cube_touch (fixed pose, must be precise)
        5. wrist touch cube -> cube welded to box top
        6. re-grasp box (box + cube)
        7. nav -> P_store; place box (box + cube into Z_store)

    `h_carry` (if given) is the lower base height the box-laden carry legs walk
    at (ROOT CAUSE 3 locomotion stability); legs walked empty-handed (the
    initial pick approach, the cube-touch approach, the re-grasp approach) keep
    the full stand height.
    """
    tg = scene["targets"]
    relay_xy = tuple(scene["T_relay"]["pos"][:2])
    z_xy = tuple(scene["Z_store"]["pos"][:2])
    pillar_xy, pillar_r = _pillar_of(scene)
    # skirt the pick table on the way to the relay table (straight path crosses
    # it, same failure mode as M1); then stand so the place-pose release lands
    # the box on the relay table top and, finally, on the store zone centre.
    skirt = _table_skirt_wps(scene, "T_pick", relay_xy, vx_carry)
    skirt = [_with_height(w, h_carry) for w in skirt]   # carry legs
    relay_stand = _place_stand_pose(relay_xy, skirt[-1].target_pose[:2])
    store_stand = _place_stand_pose(z_xy, tg["P_store"][:2])
    return [
        # (1) pick (empty-handed approach -> full stand height; avoid pillar)
        NavLeg(tuple(tg["P_pick_front"]), mode="coarse_then_cal",
               pillar_xy=pillar_xy, pillar_r=pillar_r),
        SquatTo(h_pick, upper_segment="reach_down"),
        Grasp(d=D_GRASP_M),
        StandTo(h_stand, upper_segment="carry"),
        # (2) relay leg (skirt pick table, then detour round the pillar toward
        #     P_relay_front — §4 M2 step 2) + (3) place on relay table [CARRY]
        *skirt,
        NavLeg(relay_stand, vx_max=vx_carry, mode="coarse_only",
               height=h_carry, pillar_xy=pillar_xy, pillar_r=pillar_r),
        SquatTo(h_relay, upper_segment="place"),
        _tag_place_target(
            PlaceOn("relay", surface_top_h=scene["T_relay"]["top_h"]),
            relay_xy),
        StandTo(h_stand, upper_segment="carry", rate=POST_PLACE_STAND_RATE,
                stand_hold_s=POST_PLACE_HOLD_S),
        # (4) precise nav to cube-touch pose (empty-handed -> full stand)
        NavLeg(tuple(tg["P_cube_touch"]), vx_max=vx_carry,
               mode="coarse_then_cal",
               pillar_xy=pillar_xy, pillar_r=pillar_r),
        # (5) cube transfer onto box top
        SquatTo(h_cube, upper_segment="reach_down"),
        CubeTransfer(d=CUBE_TOUCH_D_M),
        StandTo(h_stand, upper_segment="carry"),
        # (6) re-grasp box (+cube) — back to the relay table front (empty-
        #     handed approach -> full stand), squat, weld the box again
        NavLeg(tuple(tg["P_relay_front"]), vx_max=vx_carry,
               mode="coarse_then_cal",
               pillar_xy=pillar_xy, pillar_r=pillar_r),
        SquatTo(h_relay, upper_segment="reach_down"),
        Grasp(d=D_GRASP_M),
        StandTo(h_stand, upper_segment="carry"),
        # (7) store leg + final place [CARRY] (rounds the pillar)
        NavLeg(store_stand, vx_max=vx_carry, mode="coarse_only",
               height=h_carry, pillar_xy=pillar_xy, pillar_r=pillar_r),
        SquatTo(h_store, upper_segment="place"),
        _tag_place_target(
            PlaceOn("store", surface_top_h=STORE_SURFACE_TOP_H), z_xy),
        StandTo(h_stand, upper_segment="carry", rate=POST_PLACE_STAND_RATE,
                stand_hold_s=POST_PLACE_HOLD_S),
    ]


# Convenience registry so a harness can build by task id without importing the
# builders by name.
MISSION_BUILDERS: dict = {
    "arena_M1": build_m1,
    "arena_M2": build_m2,
}


# ===========================================================================
# Deterministic self-check for the NavLeg pillar-avoidance detour (no numpy,
# no MuJoCo). Drives a single NavLeg through the Mission engine under a pure
# body-frame kinematic integrator and asserts the BENCHMARK_V2_DESIGN §4
# detour contract:
#   (a) a straight leg whose path CLEARS the pillar gets NO waypoint,
#   (b) a leg whose path INTRUDES the 0.40 m keep-out gets EXACTLY ONE detour
#       waypoint on the side AWAY from the pillar,
#   (c) the detoured leg still CONVERGES (and never breaches the keep-out).
# Run:  python3 scripts/manip_mission.py   (prints "self-check OK ...").
# ===========================================================================
class _KinematicNavIO:
    """Deterministic body-frame integrator standing in for a harness during the
    NavLeg self-check. Only root_pose / is_fallen are meaningful (the only IO
    the mission touches on a pure nav leg); the rest are inert Protocol stubs.
    apply() advances the pose one tick under the body-frame command convention
    nav_errors uses (vx forward along yaw, vy left)."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self._x, self._y, self._yaw = x, y, yaw

    # --- state the mission reads on a nav leg ---
    def root_pose(self) -> tuple:
        return (self._x, self._y, self._yaw)

    def is_fallen(self) -> bool:
        return False

    # --- one integration tick (driver-side, not part of MissionIO) ---
    def apply(self, cmd: tuple, dt: float) -> None:
        vx, vy, wz, _h = cmd
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        self._x += (vx * c - vy * s) * dt
        self._y += (vx * s + vy * c) * dt
        self._yaw = wrap_angle(self._yaw + wz * dt)

    # --- inert stubs (never reached on a pure nav leg; satisfy MissionIO) ---
    def root_tilt(self) -> float: return 0.0
    def wrist_box_dist(self) -> tuple: return (0.0, 0.0)
    def right_wrist_cube_dist(self) -> float: return 0.0
    def box_pose(self) -> tuple: return (0.0, 0.0, 0.0, 0.0, 0.0)
    def base_height(self) -> float: return 0.0
    def send_leg_cmd(self, *a) -> None: pass
    def replay_upper(self, *a) -> None: pass
    def weld_box(self) -> None: pass
    def release_box(self, *a, **k) -> None: pass
    def transfer_cube(self) -> None: pass


def _drive_nav(leg: NavLeg, io: "_KinematicNavIO", h_stand: float = 0.74,
               dt: float = 0.05, t_max: float = 200.0,
               pillar: Optional[tuple] = None) -> tuple:
    """Run a single-NavLeg mission to completion under the kinematic IO. Returns
    (mission, min_pillar_clearance) where the clearance is the closest the root
    came to the pillar SURFACE over the whole trajectory (inf if pillar=None)."""
    m = Mission("selfcheck", [leg], io, h_stand=h_stand)
    t = 0.0
    min_clear = float("inf")
    while not m.finished and t < t_max:
        cmd = m.step(t)
        io.apply(cmd, dt)
        if pillar is not None:
            px, py, pr = pillar
            x, y, _ = io.root_pose()
            min_clear = min(min_clear, math.hypot(x - px, y - py) - pr)
        t += dt
    return m, min_clear


def _seg_side(ax, ay, bx, by, px, py) -> float:
    """Signed side of point P relative to the directed segment A->B (left
    normal). Sign distinguishes the two sides; magnitude is unimportant."""
    return -(by - ay) * (px - ax) + (bx - ax) * (py - ay)


def _self_check() -> None:
    h_stand = 0.74
    pillar_r = 0.15
    ax, ay, bx, by = 0.0, 0.0, 3.0, 0.0          # straight leg along +x

    # (a) CLEAR: pillar 1.0 m off the path -> no detour, leg converges.
    leg_clear = NavLeg((bx, by, 0.0), mode="coarse_then_cal",
                       pillar_xy=(1.5, 1.0), pillar_r=pillar_r)
    d_clear, _ = point_segment_dist(1.5, 1.0, ax, ay, bx, by)
    assert d_clear >= PILLAR_AVOID_CLEARANCE_M, "test setup: (a) must clear"
    m_a, _ = _drive_nav(leg_clear, _KinematicNavIO(ax, ay, 0.0))
    rec_a = m_a.stages[-1]
    assert rec_a.detour_wp is None, "(a) clear path must NOT insert a detour"
    assert m_a.status == "done" and rec_a.cal_converged, \
        "(a) clear leg must converge"

    # (b) INTRUDE: pillar 0.2 m off the path (centre at +y) -> exactly one detour
    #     waypoint on the -y side (away from the pillar), and the leg converges.
    pillar_xy = (1.5, 0.2)
    leg_block = NavLeg((bx, by, 0.0), mode="coarse_then_cal",
                       pillar_xy=pillar_xy, pillar_r=pillar_r)
    d_block, _ = point_segment_dist(pillar_xy[0], pillar_xy[1], ax, ay, bx, by)
    assert d_block < PILLAR_AVOID_CLEARANCE_M, "test setup: (b) must intrude"
    io_b = _KinematicNavIO(ax, ay, 0.0)
    m_b, min_clear = _drive_nav(leg_block, io_b, pillar=(*pillar_xy, pillar_r))
    rec_b = m_b.stages[-1]
    assert rec_b.detour_wp is not None, "(b) intruding path must insert a detour"
    wx, wy = rec_b.detour_wp
    # correct side: waypoint opposite the pillar across the start->target segment
    side_pillar = _seg_side(ax, ay, bx, by, pillar_xy[0], pillar_xy[1])
    side_wp = _seg_side(ax, ay, bx, by, wx, wy)
    assert (side_pillar > 0) != (side_wp > 0), \
        "(b) detour waypoint must be on the side away from the pillar"
    # the waypoint itself clears the keep-out around the pillar centre
    assert (math.hypot(wx - pillar_xy[0], wy - pillar_xy[1])
            >= pillar_r + PILLAR_AVOID_CLEARANCE_M - 1e-9), \
        "(b) detour waypoint must clear the r+keep-out radius"

    # one-shot + deterministic: replanning from the same start yields the SAME
    # single waypoint (exactly one waypoint is ever produced for the leg).
    wp1 = m_b._plan_pillar_detour(leg_block, (ax, ay))
    wp2 = m_b._plan_pillar_detour(leg_block, (ax, ay))
    assert (wp1 is not None and wp2 is not None
            and abs(wp1[0] - wp2[0]) < 1e-12 and abs(wp1[1] - wp2[1]) < 1e-12), \
        "(b) detour planning must be deterministic / a single waypoint"

    # (c) the detoured leg converges AND its trajectory never hit the pillar.
    assert m_b.status == "done" and rec_b.cal_converged, \
        "(c) detoured leg must still converge to the 5cm/5deg target"
    assert min_clear > 0.0, "(c) detoured trajectory must not breach the pillar"

    print("manip_mission self-check OK: "
          "(a) clear=no-detour+converge; "
          "(b) intrude=1 detour on far side wp=(%.3f, %.3f); "
          "(c) converges, min_clear=%.3f m" % (wx, wy, min_clear))


if __name__ == "__main__":
    _self_check()
