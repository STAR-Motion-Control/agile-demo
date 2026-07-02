#!/usr/bin/env python
# -----------------------------------------------------------------------------
# AMO (OpenTeleVision/AMO, G1 23-DoF) benchmark harness — unified spec v1.
#
# Runs the official MuJoCo sim2sim deployment (play_amo.py / HumanoidEnv)
# headless and measures: T1 walk_speed, T2 squat_box, T3 circle_pillar,
# plus a speed_sweep. Reuses HumanoidEnv VERBATIM (obs/action pipeline is the
# official one); only the GUI viewer is stubbed (sys.modules injection, no
# real glfw / mujoco_viewer import) and the scene XML is patched per test.
#
# squat_box box mechanism v2 ("wrist_weld_v2"): the 2 kg box is a FREE body
# (freejoint) held between the two hands by two soft equality welds
# (box <-> left_rubber_hand, box <-> right_rubber_hand). Load goes through
# the arms; arm sag/oscillation under the 2 kg payload is real physics.
# v1 (box welded rigidly to the torso, no arm load) is superseded.
#
# psi0 replay variants (upper-replay npz contract v1, embodiment amo23, K=11
# = arm 8 + waist 3; mapping authority: 4090:/sda/lizhe/g1bench/psi0_replay/
# README_replay.md). All three need --upper-replay <npz>:
#   walk_speed_psi0    walk_speed cmds + looped real_ep053 upper stream;
#                      cracker box (0.411 kg) dual-wrist-welded from t=0.
#   squat_box_psi0     table-top pick: 2 s settle -> sim_ep035 upper replay
#                      with its height_cmd; welds activate at grasp_close_t
#                      iff both wrists are < 0.12 m from the box surface,
#                      else pick_failed; then carry_pose freeze + 20x5 s
#                      squat cycles. Box collides with table/ground only.
#   circle_pillar_psi0 circle_pillar cmds + looped real_ep053 upper stream;
#                      box welded from t=0. 25 ccw + 25 cw.
# Upper-body drive: arms AND waist pd_targets are overwritten per control
# step from the replay (waist 3 are part of the AMO policy action -> SIMPLE-
# style hack, precedent SIMPLE g1_wholebody.py:271: overwrite pd_target after
# the policy ran, before PD torque; see _apply_upper_replay).
#
# v3 tests (unified spec v3, shared across the 4 harnesses):
#   T4 squat_sweep   depth x ramp-speed grid while carrying the v2 wrist-weld
#                    box: 8 absolute base heights {0.65..0.30} @ 0.2 m/s +
#                    4 ramp speeds {0.1,0.2,0.4,0.8} @ depth 0.45;
#                    --trials N = trials PER GRID POINT (default 5).
#                    Metrics: achieved_depth, depth_err, root_drift_hold
#                    (max XY drift during bottom hold), root_drift_total.
#   T5 goto_ab       A(origin, yaw 0 +- 0.3 rad noise) -> B(3.0, 1.0, +90deg),
#                    no box. Two-phase shared nav controller: NAV (coarse,
#                    P-law, exit 0.30 m/15deg or 30 s) -> CAL (cmds clamped to
#                    |0.10|, converge 5 cm/5deg held 1 s or 15 s timeout).
#                    AMO has no wz: heading goes through the ABSOLUTE yaw
#                    channel commands[1]; in CAL it is slew-limited to
#                    0.10 rad/s. play_amo locks heading tracking when
#                    |vx_cmd| < 0.1 (in-place-stand flag), so CAL yaw
#                    authority is mostly dead -- measured, not worked around
#                    (cal_yaw_locked_frac).
#   T6 pipeline_abc  box carried from t=0 (v2 welds) -> NAV to C(0,-2.5,-90deg)
#                    with vx_max 0.5 -> CAL -> squat to 0.45 @ 0.2 m/s ->
#                    bottom hold 0.5 s -> release both welds + enable box
#                    collision (runtime model writes) -> hold 1 s -> rise ->
#                    stand 1 s. box_place_ok = box static, upright (<30deg),
#                    within 0.8 m of the robot. Videos: first 3 successes +
#                    all failures by default.
#
# v4 tests (unified spec v4, shared across the 4 harnesses):
#   T7 squat_limit   squat-depth limit calibration with the 2 kg wrist-weld
#                    v2 box: 2 s settle -> height cmd ramps 0.75 -> 0.10 m at
#                    a CONSTANT 0.05 m/s -> hold the 0.10 m command 3 s, no
#                    rise. commands[3] goes down to -0.65, far below the
#                    trained height domain, and is sent UNCLIPPED: play_amo
#                    passes commands[3] raw into the adapter/obs
#                    (play_amo.py:248,282 -- no clip in the command layer)
#                    and this harness adds none. Records a 10 Hz
#                    descent_curve [[h_cmd, base_z, drift_xy], ...] plus
#                    fall_h_cmd/fall_base_z, depth_floor (min STABLE base_z),
#                    track_sat_h, drift5_h/drift20_h, max_drift_xy.
#                    success = no fall (saturation without falling expected).
#   T8 squat_place_psi0  place the 2 kg box on the FLOOR, arms coordinated by
#                    the Psi0 real_ep053 trajectory: box wrist-welded (v2)
#                    from t=0 at the standing pose -> 2 s settle -> single
#                    playback of the ep053 upper stream (arm 8 + waist 3,
#                    same replay pathway as the other psi0 tests) with its
#                    height_cmd driving commands[3] -> at t_place =
#                    argmin(height_cmd) (computed at npz load) both welds
#                    release (bit-2 + body-bit mechanism, same as
#                    pipeline_abc) so the box lands in front -> replay rise
#                    segment -> 1 s stand. Lower-body-error metrics:
#                    height_rmse_descent/hold/rise, root_drift_place,
#                    box_place_ok + box_land_dx (box landing point forward of
#                    the release-time foot FRONT EDGE; positive = placed in
#                    front). success = no fall AND box_place_ok.
#                    Needs --upper-replay <real_ep053 npz>.
#
# v5 tests (unified spec v5, shared across the 4 harnesses):
#   T9 squat_pick_ground  ground-level pick: 2 kg box (0.35x0.25x0.25 m,
#                    collision bit 2 -> ground yes / robot never) upright on
#                    the floor 0.45 m ahead; robot starts EMPTY-HANDED, arms
#                    hanging. 2 s settle -> arms blend to the low front-reach
#                    pose (PICK_ARM_POSE) -> height cmd ramps 0.75 -> 0.25 m
#                    at 0.2 m/s (unified command; commands[3] = -0.50 sent
#                    UNCLIPPED, each model saturates per its own ability;
#                    AMO's R4 depth floor ~0.345 m) -> bottom hold: both
#                    wrists < 0.30 m from the box surface => magnetic grasp
#                    (welds anchored in place + enabled, squat_box_psi0
#                    mechanism; box NOT moved); no grasp within 5 s =>
#                    pick_failed, rise anyway -> 0.5 s post-grasp hold ->
#                    rise with the 2 kg load -> 1 s stand.
#                    success = pick AND no fall AND stand_ok.
#   T10 vln_follow   VLN-style irregular command-stream following. Needs
#                    --tapes tapes.json (make_vln_tapes.py output; ALL FOUR
#                    harnesses use the SAME file for fairness). Per tape:
#                    30 s of [t,vx,vy,wz] breakpoints (zero-order hold,
#                    update jitter 0.4-1.2 s, |vx|<=0.35 |vy|<=0.2
#                    |wz|<=0.30, 2 full-stop segments) + ref_xy_yaw = ideal
#                    error-free integral at 1 Hz. 2 s settle -> tape -> 1 s
#                    stand; trial i uses tape i % n_tapes. AMO has no wz:
#                    wz is integrated into the ABSOLUTE heading target
#                    commands[1] (circle/goto realisation), vy ->
#                    commands[2]. success = no fall AND final_pos_err <=
#                    0.30 m AND final_yaw_err <= 15 deg.
#
# RUN ON THE 5080 SERVER (AMO already deployed there):
#   ssh wjzh@10.24.88.193
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate amo
#   pip install "imageio[ffmpeg]"     # one-time, only extra dep (videos)
#   cd ~/AMO                          # cwd MUST be ~/AMO (relative ckpt/xml)
#   # (scp this file into ~/AMO first)
#   MUJOCO_GL=egl python bench_amo.py --test walk_speed    --trials 50 --out-dir bench_out --video policy
#   MUJOCO_GL=egl python bench_amo.py --test speed_sweep   --trials 25 --out-dir bench_out --video none
#   MUJOCO_GL=egl python bench_amo.py --test squat_box     --trials 50 --out-dir bench_out --video policy
#   MUJOCO_GL=egl python bench_amo.py --test circle_pillar --trials 50 --out-dir bench_out --video policy
#   MUJOCO_GL=egl python bench_amo.py --test walk_speed_psi0    --trials 50 --out-dir bench_out --video policy \
#       --upper-replay ~/AMO/psi0_replay/upper_replay_amo23_real_ep053.npz
#   MUJOCO_GL=egl python bench_amo.py --test squat_box_psi0     --trials 50 --out-dir bench_out --video policy \
#       --upper-replay ~/AMO/psi0_replay/upper_replay_amo23_sim_ep035.npz
#   MUJOCO_GL=egl python bench_amo.py --test circle_pillar_psi0 --trials 50 --out-dir bench_out --video policy \
#       --upper-replay ~/AMO/psi0_replay/upper_replay_amo23_real_ep053.npz
#   MUJOCO_GL=egl python bench_amo.py --test squat_sweep   --trials 5  --out-dir bench_out --video policy   # 5/grid pt
#   MUJOCO_GL=egl python bench_amo.py --test goto_ab       --trials 25 --out-dir bench_out --video policy
#   MUJOCO_GL=egl python bench_amo.py --test pipeline_abc  --trials 15 --out-dir bench_out --video policy
#   MUJOCO_GL=egl python bench_amo.py --test squat_limit   --trials 10 --out-dir bench_out --video all   # every trial is a calibration sample
#   MUJOCO_GL=egl python bench_amo.py --test squat_place_psi0 --trials 25 --out-dir bench_out --video policy \
#       --upper-replay ~/AMO/psi0_replay/upper_replay_amo23_real_ep053.npz
#   MUJOCO_GL=egl python bench_amo.py --test squat_pick_ground --trials 15 --out-dir bench_out --video policy
#   MUJOCO_GL=egl python bench_amo.py --test vln_follow --trials 10 --out-dir bench_out --video policy \
#       --tapes ~/AMO/vln_tapes.json
#
# Requirements on server: env `amo` (torch 2.11+cu128, mujoco 3.2.3), healthy
# GPU (JIT traces hardcode cuda:0; after an Xid-154 fault the box must be
# rebooted — see ~/AMO/run_amo.sh pre-flight).
#
# Outputs (in --out-dir):
#   amo_<test>.jsonl                       one line per trial
#   summary.json                           merged summary keyed by test
#   amo_<test>_tNN_{ok|fail}.mp4           per video policy (640x480, 25 fps)
# -----------------------------------------------------------------------------

import argparse
import json
import math
import os
import sys
import time
import types as _types

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco import

import numpy as np

# ---------------------------------------------------------------------------
# GUI stubs: inject fake `glfw` and `mujoco_viewer` modules BEFORE importing
# play_amo, so no real GUI library is ever imported (headless-safe).
# Pattern follows ~/AMO/smoke_test.py, hardened to full module injection.
# ---------------------------------------------------------------------------


class _DummyCam:
    def __init__(self):
        self.distance = 2.5
        self.elevation = 0.0
        self.azimuth = 0.0
        self.lookat = np.zeros(3)


class _DummyViewer:
    """Stands in for mujoco_viewer.MujocoViewer (play_amo.py:163-168)."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.commands = np.zeros(8, dtype=np.float32)
        self.cam = _DummyCam()
        self.window = None

    def render(self):
        pass

    def close(self):
        pass


def _install_gui_stubs():
    fake_glfw = _types.ModuleType("glfw")
    fake_glfw.PRESS = 1
    for key in ("S", "W", "A", "D", "Q", "E", "Z", "X", "J", "U", "K", "I",
                "L", "O", "T", "ESCAPE"):
        setattr(fake_glfw, "KEY_" + key, 0)
    fake_glfw.set_key_callback = lambda *a, **k: None
    fake_glfw.set_window_should_close = lambda *a, **k: None
    sys.modules["glfw"] = fake_glfw

    fake_mv = _types.ModuleType("mujoco_viewer")
    fake_mv.MujocoViewer = _DummyViewer
    sys.modules["mujoco_viewer"] = fake_mv


_install_gui_stubs()

import mujoco  # noqa: E402
import torch   # noqa: E402

_CWD = os.getcwd()
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

try:
    import play_amo  # noqa: E402  (imports our fake glfw/mujoco_viewer)
except ImportError as exc:
    sys.stderr.write(
        "[bench_amo] cannot import play_amo (%s).\n"
        "Run with cwd = ~/AMO (must contain play_amo.py, g1.xml, amo_jit.pt,"
        " adapter_jit.pt, adapter_norm_stats.pt).\n" % exc)
    raise

# ManipArena v2 (BENCHMARK_V2_DESIGN.md §2-§4): shared scene generator + mission
# state machine. The four harnesses share these two modules as the single source
# of truth; this is the AMO adapter (mirrors the reference HOMIE adapter in
# bench_homie.py §run_arena_*). Both files must sit next to bench_amo.py (scp'd
# into ~/AMO alongside it). Imported lazily-safe: the L/M arena tests are the
# only ones that touch them.
try:
    import manip_scene as ms        # noqa: E402
    import manip_mission as mm      # noqa: E402
except ImportError as _arena_exc:   # pragma: no cover - non-arena runs still ok
    ms = None
    mm = None
    _ARENA_IMPORT_ERR = _arena_exc

# ---------------------------------------------------------------------------
# Constants (unified experiment spec v1)
# ---------------------------------------------------------------------------

FRAMEWORK = "amo"
# ManipArena unified-scene tasks (BENCHMARK_V2_DESIGN §2-§4):
#   L1 arena_L1 walk sweep along the lane
#   L2 arena_L2 circle the scene pillar
#   L3 arena_L3 goto P_pick_front (small-command nav calibration)
#   L4 arena_L4 vln tape following inside the arena (--tapes vln_tapes.json)
#   L5 arena_L5 squat-depth limit while carrying the 2 kg box
#   M1 arena_M1 table pick -> place (5 stages)
#   M2 arena_M2 relay pick -> place (7 stages)
ARENA_L_TESTS = ("arena_L1", "arena_L2", "arena_L3", "arena_L4", "arena_L5")
ARENA_M_TESTS = ("arena_M1", "arena_M2")
ARENA_TESTS = ARENA_L_TESTS + ARENA_M_TESTS
TESTS = ("walk_speed", "squat_box", "circle_pillar", "speed_sweep",
         "walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
         "squat_sweep", "goto_ab", "pipeline_abc",
         "squat_limit", "squat_place_psi0",
         "squat_pick_ground", "vln_follow") + ARENA_TESTS
DEFAULT_TRIALS = {"walk_speed": 50, "squat_box": 50,
                  "circle_pillar": 50, "speed_sweep": 25,
                  "walk_speed_psi0": 50, "squat_box_psi0": 50,
                  "circle_pillar_psi0": 50,
                  "squat_sweep": 5,        # per grid point (12 points)
                  "goto_ab": 25, "pipeline_abc": 15,
                  "squat_limit": 10, "squat_place_psi0": 25,
                  "squat_pick_ground": 15, "vln_follow": 10,
                  # arena tests: --trials N re-runs the SAME --variant N times
                  # (default 1, like the HOMIE reference adapter).
                  "arena_L1": 1, "arena_L2": 1, "arena_L3": 1,
                  "arena_L4": 1, "arena_L5": 1,
                  "arena_M1": 1, "arena_M2": 1}
# 2 kg v2 carry box (HUG pose capture):
BOX_TESTS = ("squat_box", "squat_sweep", "pipeline_abc", "squat_limit")
PSI0_TESTS = ("walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
              "squat_place_psi0")

REQUIRED_FILES = ("g1.xml", "play_amo.py", "amo_jit.pt",
                  "adapter_jit.pt", "adapter_norm_stats.pt")

SETTLE_T = 2.0                      # zero-command standing after reset
INIT_JOINT_NOISE = 0.02             # rad, uniform, on all 23 joints
BASE_HEIGHT_DEFAULT = 0.75          # AMO height cmd is delta on 0.75 (play_amo.py:97,248)

FALL_TILT_RAD = 0.9
FALL_HEIGHT_MARGIN = 0.2

SWEEP_SPEEDS = (0.4, 0.6, 0.8, 1.0, 1.2)

SQUAT_HEIGHT_CMD = -0.30            # 0.75 - 0.30 = 0.45 m squat target
SQUAT_RAMP_T = 1.5
SQUAT_HOLD_T = 1.0
SQUAT_CYCLE_T = 5.0
SQUAT_N_CYCLES = 20
ARM_TRANSITION_T = 3.0              # arm blend takes 2 s (blend += 0.01 @50 Hz)

# Hug-the-box arm pose [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow] L/R.
# Tuned by FK on g1.xml (keyframe 'home' base pose): rubber-hand origins end up
# 0.353 m apart laterally (= the 0.35 m box width, hands ~ +-0.175 m off the
# box center), midpoint ~0.23 m in front of the torso at chest-front height.
# TODO(verify-on-server): eyeball one rendered frame (squat_box --video all
# --trials 1) to confirm the hands sit flush on the box sides.
HUG_ARM_POSE = np.array([-0.3, 0.1, 0.0, 1.1,
                         -0.3, -0.1, 0.0, 1.1])

# squat_box box mechanism v2: free box + dual wrist welds.
BOX_MODE = "wrist_weld_v2"
BOX_HAND_BODIES = ("left_rubber_hand", "right_rubber_hand")
BOX_WELD_SOLREF = "0.02 1"          # soft weld so it doesn't fight the arm PD
BOX_KEEP_DIST = 0.6                 # box considered "kept" while |box-torso|<0.6 m
WELD_RESID_TOL = 1e-6               # weld residual at capture config must be ~0
QACC_DIVERGE_LIMIT = 1e8            # rad/s^2; above this => solver divergence

# --- psi0 upper-body replay (contract v1; npz on the 5080: ~/AMO/psi0_replay/)
REPLAY_DT = 0.02                    # contract dt; must equal env.control_dt
PSI0_GRASP_DIST = 0.30              # m, BOTH hands to the box surface at grasp
PSI0_BOX_HALF = (0.036, 0.082, 0.1065)  # cracker_box (YCB 003) half extents
PSI0_BOX_MASS = 0.411               # kg
# Replayed height_cmd -> AMO's nominal height domain. Lower edge 0.45 m per
# the squat_sweep note (below ~0.45 m is OOD); upper = 0.75 m stand default.
# commands[3] = clip(height_cmd) - 0.75 (delta convention, play_amo.py:97,248).
PSI0_HEIGHT_CLIP = (0.45, BASE_HEIGHT_DEFAULT)
# BendPick table scene (psi0_replay/scene_info.json, ep035), x-shifted so the
# pelvis spawns at the origin: tabletop top face z=0.40 m, table near edge at
# x=0.2935 -> pelvis-to-edge ~0.29 m (matches the source scene). Floor z=0.
PSI0_TABLE_HALF = (0.625, 0.395, 0.05)
PSI0_TABLE_CENTER = (0.9185, 0.0, 0.35)
PSI0_BOX_START_XY = (0.2985, -0.0488)    # ep035 target pose, shifted
PSI0_BOX_START_Z = 0.40 + PSI0_BOX_HALF[2] + 0.0005  # upright on the tabletop
PSI0_BOX_START_QUAT = (0.997384, 0.0, 0.0, -0.072316)  # ep035 yaw ~ -0.145 rad
# AMO upper-body joint names (play_amo dof_names[12:23]); npz upper_names
# (amo23, K=11) are matched by NAME ("_joint" suffix tolerated).
AMO_WAIST_NAMES = ("waist_yaw", "waist_roll", "waist_pitch")
AMO_ARM_NAMES = ("left_shoulder_pitch", "left_shoulder_roll",
                 "left_shoulder_yaw", "left_elbow",
                 "right_shoulder_pitch", "right_shoulder_roll",
                 "right_shoulder_yaw", "right_elbow")

CIRCLE_VX = 0.4
CIRCLE_WZ = 0.4                     # rad/s; radius = vx/wz = 1.0 m
CIRCLE_RADIUS = 1.0
PILLAR_RADIUS = 0.15
PILLAR_HALF_H = 0.6                 # height 1.2 m
# Robot collision geoms = pelvis_contour + 8 foot spheres only (arms/torso are
# visual-only meshes), so MuJoCo contact misses arm-pillar hits. Supplementary
# proximity criterion: base center closer than pillar_r + approx body radius.
PROXIMITY_COLLISION_DIST = 0.35
RADIAL_ERR_SUCCESS = 0.15

VIDEO_W, VIDEO_H = 640, 480
VIDEO_EVERY_N_CTRL = 2              # 50 Hz ctrl -> capture every 2nd = 25 fps
VIDEO_FPS = 25
MAX_SUCCESS_VIDEOS = 5
PIPE_MAX_SUCCESS_VIDEOS = 3         # T6 spec: first 3 successes + all failures

# --- T4 squat_sweep (spec v3): depth x ramp-speed grid, v2 box carried ---
SQUAT_SWEEP_HEIGHTS = (0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)
SQUAT_SWEEP_BASE_RATE = 0.2         # m/s, fixed rate for the depth sweep
SQUAT_SWEEP_RATES = (0.1, 0.2, 0.4, 0.8)
SQUAT_SWEEP_FIXED_H = 0.45          # m, fixed depth for the speed sweep
SQUAT_SWEEP_HOLD_T = 3.0            # s at the bottom
SQUAT_SWEEP_FINAL_T = 1.0           # s standing after rise

# --- T7 squat_limit (spec v4): continuous depth-limit calibration ---
SQUAT_LIMIT_RATE = 0.05             # m/s, constant height-cmd descent rate
SQUAT_LIMIT_FLOOR_H = 0.10          # m, final commanded height (far OOD;
                                    # commands[3] = -0.65, sent UNCLIPPED)
SQUAT_LIMIT_HOLD_T = 3.0            # s holding the 0.10 m command (no rise)
SQUAT_LIMIT_CURVE_STRIDE = 5        # 50 Hz ctrl -> 10 Hz descent_curve
SQUAT_LIMIT_TRACK_SAT_M = 0.05      # m, base_z above h_cmd => tracking saturated
SQUAT_LIMIT_DRIFT_MARKS = (0.05, 0.20)  # m, drift5_h / drift20_h thresholds
SQUAT_LIMIT_STABLE_TILT_RAD = 0.5   # depth_floor stable filter: tilt below this
SQUAT_LIMIT_STABLE_VZ = -0.25       # m/s; sinking faster = collapsing, not squatting

# --- T8 squat_place_psi0 (spec v4): ep053-coordinated floor placement ---
PLACE_HOLD_TOL_M = 0.02             # height_cmd <= min + tol => "hold" segment
PLACE_FINAL_STAND_T = 1.0           # s standing after the replay ends
PLACE_BOX_LAND_MAX_Z = 0.30         # m, box center must end below this (landed,
                                    # guards a silently-failed weld release)

# --- T9 squat_pick_ground (spec v5): root-to-the-floor squat + ground pick ---
PICK_BOX_MODE = "ground_pick_weld_v1"
PICK_BOX_DIST = 0.45                # m ahead of the pelvis (rotated by yaw0)
PICK_BOX_HALF = (0.125, 0.175, 0.125)  # same 2 kg / 0.35x0.25x0.25 carry box
PICK_SQUAT_H = 0.25                 # m, UNIFIED commanded bottom height
                                    # (commands[3] = -0.50, sent unclipped;
                                    # each model saturates per its ability)
PICK_SQUAT_RATE = 0.2               # m/s height-cmd ramp (down and up)
PICK_GRASP_DIST = 0.30              # m, BOTH wrists to the box surface
PICK_GRASP_TIMEOUT_T = 5.0          # s at the bottom before pick_failed
PICK_POST_GRASP_HOLD_T = 0.5        # s hold after the grasp, still low
PICK_FINAL_STAND_T = 1.0            # s standing at the end
# Low front-reach hold pose [shoulder_pitch, shoulder_roll, shoulder_yaw,
# elbow] L/R (negative pitch = forward swing, HUG_ARM_POSE convention):
# shoulders pitched far forward-down + slight elbow flexion so the wrists end
# as LOW as possible and just outside the box's +-0.175 m side faces.
# TODO(verify-on-server): eyeball one rendered frame
# (squat_pick_ground --video all --trials 1) before the full batch.
PICK_ARM_POSE = np.array([-0.9, 0.15, 0.0, 0.5,
                          -0.9, -0.15, 0.0, 0.5])

# --- T10 vln_follow (spec v5): VLN-style command-stream following ---
VLN_FINAL_STAND_T = 1.0             # s standing after the tape ends
VLN_SMALL_VX_RANGE = (0.05, 0.15)   # m/s |vx| band for small_cmd_response
VLN_FINAL_POS_TOL = 0.30            # m, success gate on the final pose
VLN_FINAL_YAW_TOL = math.radians(15.0)
VLN_STOP_MIN_T = 0.5                # s, ignore sub-0.5 s stop blips

# --- T5/T6 shared two-phase navigation controller (spec v3, the math below
# is copied verbatim from the spec and MUST stay line-equivalent across the
# four harnesses) ---
NAV_SWITCH_DIST = 0.5               # m; farther: heading = bearing to goal
NAV_KP_POS = 1.0
NAV_KP_YAW = 1.5
NAV_VX_MAX = 0.6                    # m/s (vx clipped to [0, vx_max])
NAV_VY_MAX = 0.3                    # m/s
NAV_WZ_MAX = 0.6                    # rad/s
NAV_POS_TOL = 0.30                  # m, coarse exit
NAV_YAW_TOL = math.radians(15.0)    # coarse exit
NAV_TIMEOUT_T = 30.0                # s
CAL_CMD_LIM = 0.10                  # m/s and rad/s clamp on ALL cal commands
CAL_POS_TOL = 0.05                  # m, fine convergence
CAL_YAW_TOL = math.radians(5.0)     # fine convergence
CAL_HOLD_T = 1.0                    # s the fine tolerance must persist
CAL_TIMEOUT_T = 15.0                # s
AMO_STAND_VX_THRESH = 0.1           # |vx_cmd|<0.1 -> _in_place_stand_flag
                                    # locks dyaw (play_amo); measured in CAL

GOTO_B_XY = (3.0, 1.0)              # T5 goal B (world frame)
GOTO_B_YAW = math.pi / 2.0          # +90 deg
GOTO_INIT_YAW_NOISE = 0.3           # rad, uniform, from seed
GOTO_END_T = 1.0                    # s standing after CAL ends

PIPE_C_XY = (0.0, -2.5)             # T6 goal C (world frame)
PIPE_C_YAW = -math.pi / 2.0         # -90 deg (turn right then walk 2.5 m)
PIPE_NAV_VX_MAX = 0.5               # m/s, conservative while carrying the box
PIPE_SQUAT_H = 0.45                 # m absolute base height at placement
PIPE_SQUAT_RATE = 0.2               # m/s ramp
PIPE_BOTTOM_HOLD_T = 0.5            # s at the bottom before releasing
PIPE_PLACE_HOLD_T = 1.0             # s after the release, still squatting
PIPE_FINAL_T = 1.0                  # s standing at the end
BOX_PLACE_MAX_SPEED = 0.05          # m/s, box considered static
BOX_PLACE_UPRIGHT_TOL = math.radians(30.0)  # box z vs world z
BOX_PLACE_MAX_DIST = 0.8            # m from the robot base at the end

# ---------------------------------------------------------------------------
# ManipArena v2 (BENCHMARK_V2_DESIGN §2-§4) — AMO adapter constants.
# Mirrors the HOMIE reference adapter (bench_homie.py); the few that differ are
# the AMO-specific height/squat numbers. AMO height command is a DELTA on 0.75
# (commands[3] = h_target - 0.75), so the mission emits an ABSOLUTE base-height
# target and the adapter converts. AMO has NO wz channel: the mission's wz is
# integrated into the ABSOLUTE heading target commands[1] (goto/vln/circle
# realisation, same as _NavCalController) with the in-place-stand interlock
# (|vx|<0.1 freezes dyaw) recorded, not worked around.
# ---------------------------------------------------------------------------
# AMO wrist-weld anchor links (no wrist joints; the rubber hands are the last
# arm bodies — same anchors as the squat_box v2 carry box).
ARENA_HAND_BODIES = BOX_HAND_BODIES          # ("left_rubber_hand", "right_rubber_hand")
ARENA_STAND_H = BASE_HEIGHT_DEFAULT          # 0.75 carry/stand absolute base z
# AMO height-command domain. The squat_sweep note pins ~0.45 m as the practical
# lower edge (below is OOD), and R4 measured AMO's depth FLOOR at ~0.345 m on
# the floor with the box. Reach commands are clipped into [ARENA_H_MIN, 0.75];
# the low tables (H_pick 0.30/0.45) sit at/below the floor so reaching them is
# expected to fail or topple — recorded as-is.
ARENA_H_MIN = 0.40
ARENA_REACH_OFFSET = 0.10        # m, base above the surface for a tabletop reach
ARENA_STORE_PLACE_H = 0.58       # shallow, stable store-placement squat (HOMIE
                                 # ROOT CAUSE 1: deep store squats topple under
                                 # the 2 kg box; release levels+lowers the box
                                 # onto the pad regardless of squat depth).
ARENA_RELEASE_FWD_M = 0.35       # box set-down forward offset (matches
                                 # mm.PLACE_STAND_FWD_M so a robot standing that
                                 # far in front of the zone drops the box centred)
ARENA_PLACE_CLAMP_R = 0.12       # clamp the set-down inside the zone footprint
ARENA_VX_CARRY = 0.4             # m/s coarse forward clip while carrying the box
ARENA_SQUAT_RATE = 0.2           # m/s base-height ramp
ARENA_CARRY_H = ARENA_STAND_H    # carry at full stand height (lower carry stance
                                 # destabilises the box-laden walk on AMO too)
ARENA_PILLAR_CLEAR_M = 0.4       # min straight-line clearance to the pillar
ARENA_PILLAR_SIDE_M = 0.7        # lateral offset of an inserted skirt waypoint
# Box/cube half-extents from the SceneSpec so wrist-surface distances use the
# arena box dims (not the cracker/carry defaults).
ARENA_BOX_HALF = None            # filled at import time if ms is available
ARENA_CUBE_HALF = None
if ms is not None:
    ARENA_BOX_HALF = tuple(v / 2.0 for v in ms.BOX_FULL)
    ARENA_CUBE_HALF = tuple(v / 2.0 for v in ms.CUBE_FULL)
# Heading-target slew used to convert the mission's wz command into an absolute
# heading increment per control step (AMO has no wz channel). The mission emits
# a body-frame wz; integrating it onto commands[1] reproduces a turn.
ARENA_YAW_FROM_WZ = True
# arena L-series specifics ---------------------------------------------------
ARENA_L1_RAMP_T = 2.0
ARENA_L1_HOLD_T = 10.0
ARENA_L1_VX = 0.6                # lane sweep forward speed (within nav vx_max)
ARENA_L2_VX = CIRCLE_VX          # 0.4
ARENA_L2_WZ = CIRCLE_WZ          # 0.4 -> radius 1.0 m about the scene pillar
ARENA_L2_RADIUS = CIRCLE_RADIUS  # 1.0
ARENA_L2_LAPS = 2
ARENA_L5_RATE = SQUAT_LIMIT_RATE        # 0.05 m/s continuous descent
ARENA_L5_FLOOR_H = SQUAT_LIMIT_FLOOR_H  # 0.10 m final command (far OOD, unclipped)
ARENA_L5_HOLD_T = SQUAT_LIMIT_HOLD_T    # 3 s hold at the floor command

# ---------------------------------------------------------------------------
# Scene construction: string-patch g1.xml (free box + wrist welds / pillar)
#
# qpos layout with box (IMPORTANT): play_amo.HumanoidEnv reads robot joints
# with NEGATIVE indexing (qpos[-23:], qvel[-23:], play_amo.py:227-228), so the
# box free body MUST be injected BEFORE the pelvis in <worldbody>:
#   qpos = [box free 0:7 | pelvis free 7:14 | 23 hinges 14:37]
#   qvel = [box free 0:6 | pelvis free 6:12 | 23 hinges 12:35]
# All base-pose reads in this harness are therefore pelvis-adr aware.
# ---------------------------------------------------------------------------

_GROUND_ANCHOR = ('<geom name="ground" type="plane" size="0 0 1" '
                  'pos="0.001 0 0" quat="1 0 0 0" material="MatPlane" '
                  "condim=\"1\" conaffinity='15'/>")
_WORLDBODY_OPEN_ANCHOR = "<worldbody>"        # first occurrence = robot tree
_MUJOCO_CLOSE_ANCHOR = "</mujoco>"
_KEYFRAME_QPOS_ANCHOR = 'qpos="0 0 1.0 1 0 0 0'   # keyframe 'home', 30 values

# 0.35(y, between the hands) x 0.25(x) x 0.25(z) m box, 2.0 kg, FREE body.
# No collision vs robot/ground (contype=0 conaffinity=0) per spec; it is held
# purely by the two wrist welds. diaginertia = m/12*(b^2+c^2) etc. for
# (x,y,z)=(0.25,0.35,0.25).
_BOX_FREE_BODY_TMPL = """
    <body name="bench_box" pos="{px} {py} {pz}">
      <freejoint name="bench_box_free"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.030833 0.020833 0.030833"/>
      <geom name="bench_box_geom" type="box" size="0.125 0.175 0.125" contype="0" conaffinity="0" condim="3" group="1" density="0" rgba="0.85 0.65 0.25 0.7"/>
    </body>"""

# Two soft welds "hands hugging the box". relpose = pose of body2 (bench_box)
# in the body1 (hand) frame, captured at the HUG arm pose (verified locally:
# MuJoCo weld residual ~1e-16 with this convention). Soft solref so the welds
# don't fight the arm PD into numerical blowup. active="false" is used by
# squat_box_psi0 (welds only engage at grasp_close_t, via data.eq_active).
_WELD_EQUALITY_TMPL = """
  <equality>
    <weld name="bench_weld_left" body1="left_rubber_hand" body2="bench_box" relpose="{left}" solref="{solref}" active="{active}"/>
    <weld name="bench_weld_right" body1="right_rubber_hand" body2="bench_box" relpose="{right}" solref="{solref}" active="{active}"/>
  </equality>
"""

_PILLAR_GEOM_TMPL = ('    <geom name="pillar" type="cylinder" '
                     'size="{r} {hh}" pos="0 {y} {hh}" condim="3" '
                     'contype="1" conaffinity="1" rgba="0.65 0.3 0.3 1"/>')


def _fmt_vec(v):
    return " ".join("%.10g" % x for x in v)


# psi0 cracker box: FREE body, bit-2 collision. Robot collidable geoms keep
# the default contype/conaffinity 1, the table gets 3 (=1|2) and the ground
# plane ships with conaffinity=15 (bit 2 included) -> the box collides with
# table/ground but NEVER with the robot, per the unified psi0 spec.
_PSI0_BOX_FREE_BODY_TMPL = """
    <body name="bench_box" pos="{{px}} {{py}} {{pz}}">
      <freejoint name="bench_box_free"/>
      <geom name="bench_box_geom" type="box" size="{half}" mass="{mass}" contype="2" conaffinity="2" condim="3" group="1" rgba="0.85 0.3 0.15 0.8"/>
    </body>""".format(half=_fmt_vec(PSI0_BOX_HALF), mass="%.10g" % PSI0_BOX_MASS)

_PSI0_TABLE_GEOM = ('    <geom name="bench_table" type="box" size="%s" '
                    'pos="%s" contype="3" conaffinity="3" condim="3" '
                    'rgba="0.55 0.42 0.26 1"/>'
                    % (_fmt_vec(PSI0_TABLE_HALF), _fmt_vec(PSI0_TABLE_CENTER)))

# T9 ground-pick box: the SAME 2 kg / 0.35x0.25x0.25 m box as the v2 carry
# box but with collision bit 2 (cracker-box scheme): it collides with the
# ground (plane conaffinity=15 includes bit 2) and never with the robot
# (bit 1). It stands upright on the floor and is picked up by in-place weld
# activation (squat_box_psi0 mechanism), so no runtime collision-bit writes
# are ever needed (bits are compile-time correct for MuJoCo>=3.2.4
# body-level broadphase too).
_PICK_BOX_FREE_BODY_TMPL = """
    <body name="bench_box" pos="{px} {py} {pz}">
      <freejoint name="bench_box_free"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.030833 0.020833 0.030833"/>
      <geom name="bench_box_geom" type="box" size="0.125 0.175 0.125" contype="2" conaffinity="2" condim="3" group="1" density="0" rgba="0.85 0.65 0.25 0.9"/>
    </body>"""


def build_scene_xml(with_box, pillar_y, box_pos=None, weld_relposes=None,
                    box_template=_BOX_FREE_BODY_TMPL,
                    box_quat=(1.0, 0.0, 0.0, 0.0), welds_active=True,
                    with_table=False):
    """Return g1.xml text with optional free box (+welds), table and pillar."""
    with open("g1.xml", "r") as f:
        xml_text = f.read()
    if with_box:
        if box_pos is None:
            raise ValueError("with_box requires box_pos")
        for anchor in (_WORLDBODY_OPEN_ANCHOR, _KEYFRAME_QPOS_ANCHOR):
            if anchor not in xml_text:
                raise RuntimeError("g1.xml anchor %r for box injection not "
                                   "found; xml changed upstream?" % anchor)
        body = box_template.format(px="%.10g" % box_pos[0],
                                   py="%.10g" % box_pos[1],
                                   pz="%.10g" % box_pos[2])
        xml_text = xml_text.replace(_WORLDBODY_OPEN_ANCHOR,
                                    _WORLDBODY_OPEN_ANCHOR + body, 1)
        # extend keyframe 'home' qpos: box free joint comes FIRST (see above)
        kf_prefix = ('qpos="%s %s 0 0 1.0 1 0 0 0'
                     % (_fmt_vec(box_pos), _fmt_vec(box_quat)))
        xml_text = xml_text.replace(_KEYFRAME_QPOS_ANCHOR, kf_prefix, 1)
        if weld_relposes is not None:
            eq = _WELD_EQUALITY_TMPL.format(
                left=weld_relposes[0], right=weld_relposes[1],
                solref=BOX_WELD_SOLREF,
                active="true" if welds_active else "false")
            xml_text = xml_text.replace(_MUJOCO_CLOSE_ANCHOR,
                                        eq + _MUJOCO_CLOSE_ANCHOR, 1)
    if with_table:
        if _GROUND_ANCHOR not in xml_text:
            raise RuntimeError("g1.xml anchor for table injection not found "
                               "(ground geom line); xml changed upstream?")
        xml_text = xml_text.replace(_GROUND_ANCHOR,
                                    _GROUND_ANCHOR + "\n" + _PSI0_TABLE_GEOM, 1)
    if pillar_y is not None:
        if _GROUND_ANCHOR not in xml_text:
            raise RuntimeError("g1.xml anchor for pillar injection not found "
                               "(ground geom line); xml changed upstream?")
        pillar = _PILLAR_GEOM_TMPL.format(r=PILLAR_RADIUS, hh=PILLAR_HALF_H,
                                          y=pillar_y)
        xml_text = xml_text.replace(_GROUND_ANCHOR,
                                    _GROUND_ANCHOR + "\n" + pillar, 1)
    return xml_text


def _load_model_from_text(xml_text):
    """Compile xml via a temp file inside cwd (~/AMO) so the relative
    meshdir="meshes" resolves against the XML file location."""
    tmp_path = os.path.abspath("_bench_scene_%d.xml" % os.getpid())
    with open(tmp_path, "w") as f:
        f.write(xml_text)
    try:
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _pelvis_qpos_adr(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis")
    if jid < 0:
        raise RuntimeError("pelvis free joint not found in model")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _set_capture_config(model, data, arm_pose=None, waist_pose=None):
    """Keyframe 'home' base pose with the arms at `arm_pose` (default
    HUG_ARM_POSE) and optionally the waist overridden (psi0: replay frame 0);
    box (if its joint exists in qpos already) is positioned by the caller."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    base_q, _ = _pelvis_qpos_adr(model)
    hinge_adr = base_q + 7               # 23 hinges; waist 12:15, arms 15:23
    if waist_pose is not None:
        data.qpos[hinge_adr + 12:hinge_adr + 15] = waist_pose
    data.qpos[hinge_adr + 15:hinge_adr + 23] = (
        HUG_ARM_POSE if arm_pose is None else arm_pose)
    mujoco.mj_forward(model, data)


def _capture_box_attachment(model, arm_pose=None, waist_pose=None):
    """Stage A (no welds yet): put arms in the capture pose, place the box
    midway between the two hands, and capture the per-hand weld relposes.

    Returns (box_pos_world, [relpose_left_str, relpose_right_str],
    box_offset_from_pelvis).
    """
    data = mujoco.MjData(model)
    _set_capture_config(model, data, arm_pose, waist_pose)
    hand_pos = [data.body(b).xpos.copy() for b in BOX_HAND_BODIES]
    box_pos = 0.5 * (hand_pos[0] + hand_pos[1])

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bench_box_free")
    box_adr = int(model.jnt_qposadr[jid])
    data.qpos[box_adr:box_adr + 3] = box_pos
    data.qpos[box_adr + 3:box_adr + 7] = (1.0, 0.0, 0.0, 0.0)  # yaw 0 capture
    mujoco.mj_forward(model, data)

    relposes = []
    for b in BOX_HAND_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        rot = data.xmat[bid].reshape(3, 3)
        rp_pos = rot.T @ (box_pos - data.xpos[bid])
        rp_quat = _quat_conj(data.xquat[bid])     # box quat is identity
        relposes.append(_fmt_vec(np.concatenate([rp_pos, rp_quat])))

    base_q, _ = _pelvis_qpos_adr(model)
    offset = box_pos - data.qpos[base_q:base_q + 3]
    return box_pos, relposes, offset


def _max_weld_residual(model, data):
    rows = [i for i in range(data.nefc)
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    if not rows:
        raise RuntimeError("no equality-constraint rows found (welds missing?)")
    return float(np.abs(data.efc_pos[rows]).max())


def _build_box_spec(model, offset_from_pelvis, arm_pose=None, waist_pose=None,
                    check_welds=True):
    """Verify the welds close at the capture config and collect ids/adrs.

    check_welds=False for squat_box_psi0: its welds compile INACTIVE with a
    placeholder relpose (they are re-anchored at grasp time), so there is no
    residual to verify."""
    data = mujoco.MjData(model)
    # keyframe already holds the box pose
    _set_capture_config(model, data, arm_pose, waist_pose)
    if check_welds:
        resid = _max_weld_residual(model, data)
        if resid > WELD_RESID_TOL:
            raise RuntimeError(
                "weld relpose verification failed (max residual %.3g > %.1g); "
                "relpose convention / capture config mismatch" %
                (resid, WELD_RESID_TOL))
    base_q, base_v = _pelvis_qpos_adr(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bench_box_free")
    return {
        "base_qpos_adr": base_q,
        "base_qvel_adr": base_v,
        "box_qpos_adr": int(model.jnt_qposadr[jid]),
        "box_qvel_adr": int(model.jnt_dofadr[jid]),
        "offset_from_pelvis": offset_from_pelvis.copy(),
        "box_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         "bench_box"),
        "torso_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           "torso_link"),
    }


_MODEL_CACHE = {}

_BOX_PLACEHOLDER_POS = (0.25, 0.0, 1.0)   # stage-A only; replaced by capture


def get_model(with_box, pillar_y):
    """Load (and cache) the MjModel for a scene variant.

    Returns (model, box_spec); box_spec is None when with_box is False.
    With box: two-stage build — stage A compiles the free box WITHOUT welds
    to capture hand/box relposes at the HUG pose, stage B recompiles with the
    welds baked in (relpose verified to close, residual ~0).
    """
    key = (with_box, pillar_y)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not with_box:
        model = _load_model_from_text(build_scene_xml(False, pillar_y))
        _MODEL_CACHE[key] = (model, None)
        return _MODEL_CACHE[key]
    stage_a = _load_model_from_text(
        build_scene_xml(True, pillar_y, box_pos=_BOX_PLACEHOLDER_POS))
    box_pos, relposes, offset = _capture_box_attachment(stage_a)
    model = _load_model_from_text(
        build_scene_xml(True, pillar_y, box_pos=box_pos,
                        weld_relposes=relposes))
    box_spec = _build_box_spec(model, offset)
    _MODEL_CACHE[key] = (model, box_spec)
    return _MODEL_CACHE[key]


def _psi0_upper_frame0(replay):
    """(arm8, waist3) of replay frame 0, in AMO dof order."""
    row0 = np.asarray(replay.pos[0], dtype=np.float64)
    return row0[replay.arm_cols].copy(), row0[replay.waist_cols].copy()


def _psi0_add_grasp_ids(model, box_spec):
    """Extend a box_spec with hand-body and weld-equality ids (psi0 grasp)."""
    hand_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in BOX_HAND_BODIES)
    eq_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, n)
                   for n in ("bench_weld_left", "bench_weld_right"))
    if min(hand_ids) < 0 or min(eq_ids) < 0:
        raise RuntimeError("psi0: hand bodies / weld equalities missing "
                           "after scene build")
    out = dict(box_spec)
    out["hand_body_ids"] = hand_ids
    out["eq_ids"] = eq_ids
    return out


def get_psi0_model(test, replay, pillar_y):
    """Load (and cache) the psi0 scene for `test` (cracker box variants).

    walk/circle psi0: two-stage build like get_model, but the capture config
    is the REPLAY FRAME 0 upper pose (arms+waist) and the welds are active
    from t=0. squat_box_psi0: single-stage BendPick table scene — the box
    stands upright on the tabletop at the ep035 pose, the welds compile
    INACTIVE (placeholder relpose) and are re-anchored + enabled at
    grasp_close_t at the live pose (the box is not moved).
    squat_place_psi0: like walk/circle but with the 2 kg wrist-weld v2 carry
    box (default template) instead of the cracker box; welds active from t=0
    and released at t_place during the trial.
    """
    key = ("psi0", test, pillar_y, replay.path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    arm0, waist0 = _psi0_upper_frame0(replay)
    if test == "squat_box_psi0":
        box_pos = (PSI0_BOX_START_XY[0], PSI0_BOX_START_XY[1],
                   PSI0_BOX_START_Z)
        ident = _fmt_vec((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        model = _load_model_from_text(build_scene_xml(
            True, pillar_y, box_pos=box_pos, weld_relposes=(ident, ident),
            box_template=_PSI0_BOX_FREE_BODY_TMPL,
            box_quat=PSI0_BOX_START_QUAT, welds_active=False,
            with_table=True))
        base_q, _ = _pelvis_qpos_adr(model)
        data = mujoco.MjData(model)
        _set_capture_config(model, data, arm0, waist0)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                "bench_box_free")
        adr = int(model.jnt_qposadr[jid])
        offset = (data.qpos[adr:adr + 3]
                  - data.qpos[base_q:base_q + 3]).copy()  # informational only
        box_spec = _build_box_spec(model, offset, arm0, waist0,
                                   check_welds=False)
    elif test == "squat_place_psi0":
        # T8: 2 kg v2 box captured between the hands at the ep053 frame-0
        # upper pose (default _BOX_FREE_BODY_TMPL, contype/conaffinity 0
        # until release).
        stage_a = _load_model_from_text(build_scene_xml(
            True, pillar_y, box_pos=_BOX_PLACEHOLDER_POS))
        box_pos, relposes, offset = _capture_box_attachment(stage_a, arm0,
                                                            waist0)
        model = _load_model_from_text(build_scene_xml(
            True, pillar_y, box_pos=box_pos, weld_relposes=relposes))
        box_spec = _build_box_spec(model, offset, arm0, waist0)
    else:
        stage_a = _load_model_from_text(build_scene_xml(
            True, pillar_y, box_pos=_BOX_PLACEHOLDER_POS,
            box_template=_PSI0_BOX_FREE_BODY_TMPL))
        box_pos, relposes, offset = _capture_box_attachment(stage_a, arm0,
                                                            waist0)
        model = _load_model_from_text(build_scene_xml(
            True, pillar_y, box_pos=box_pos, weld_relposes=relposes,
            box_template=_PSI0_BOX_FREE_BODY_TMPL))
        box_spec = _build_box_spec(model, offset, arm0, waist0)
    box_spec = _psi0_add_grasp_ids(model, box_spec)
    _MODEL_CACHE[key] = (model, box_spec)
    return _MODEL_CACHE[key]


def get_pick_model():
    """Load (and cache) the T9 squat_pick_ground scene.

    Single-stage build like squat_box_psi0: the 2 kg bit-2 box stands
    upright on the ground PICK_BOX_DIST m ahead at the keyframe yaw-0 pose
    (apply_initial_state rotates it with the trial yaw via
    offset_from_pelvis; the qpos layout is the standard box-before-pelvis
    one). The welds compile INACTIVE with placeholder relposes and are
    anchored in place + enabled at grasp time (_activate_welds_at_current_
    pose, same as the psi0 pick)."""
    key = ("pick_ground",)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    box_pos = (PICK_BOX_DIST, 0.0, PICK_BOX_HALF[2] + 0.0005)
    ident = _fmt_vec((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    model = _load_model_from_text(build_scene_xml(
        True, None, box_pos=box_pos, weld_relposes=(ident, ident),
        box_template=_PICK_BOX_FREE_BODY_TMPL, welds_active=False))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base_q, _ = _pelvis_qpos_adr(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            "bench_box_free")
    if jid < 0:
        raise RuntimeError("bench_box_free joint missing after T9 scene "
                           "build")
    adr = int(model.jnt_qposadr[jid])
    offset = (data.qpos[adr:adr + 3] - data.qpos[base_q:base_q + 3]).copy()
    box_spec = _build_box_spec(model, offset, check_welds=False)
    box_spec = _psi0_add_grasp_ids(model, box_spec)
    box_spec["box_half"] = np.asarray(PICK_BOX_HALF, dtype=np.float64)
    _MODEL_CACHE[key] = (model, box_spec)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Env construction: reuse play_amo.HumanoidEnv with our pre-built model.
# HumanoidEnv hardcodes mujoco.MjModel.from_xml_path("g1.xml"); we swap the
# module-level `mujoco` binding inside play_amo for a thin proxy during
# __init__ only, then restore it (env.run() then uses the real module).
# ---------------------------------------------------------------------------


def _make_mujoco_proxy(model):
    class _MjModelStub:
        @staticmethod
        def from_xml_path(_path):
            return model

    class _Proxy:
        MjModel = _MjModelStub

        def __getattr__(self, name):
            return getattr(mujoco, name)

    return _Proxy()


def build_env(policy_jit, model, device):
    original_binding = play_amo.mujoco
    play_amo.mujoco = _make_mujoco_proxy(model)
    try:
        env = play_amo.HumanoidEnv(policy_jit=policy_jit, robot_type="g1",
                                   device=device)
    finally:
        play_amo.mujoco = original_binding
    return env


def apply_initial_state(env, init_yaw, box_spec=None, upper_init=None,
                        box_follows_pelvis=True):
    """Reset to keyframe 'home', add +-0.02 rad joint noise, set base yaw.

    With box: the keyframe holds the box at the yaw-0 chest-front pose; for a
    random yaw the box is re-placed by rotating its pelvis-frame offset, so it
    always starts between the hands regardless of heading.

    psi0: upper_init=(arm8, waist3) puts the 11 replayed joints EXACTLY at
    replay frame 0 (noise removed there, so the per-frame PD targets do not
    snap at t=0). box_follows_pelvis=False keeps the box at its keyframe
    world pose (squat_box_psi0: ep035 tabletop pose, independent of the base).
    """
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    base = box_spec["base_qpos_adr"] if box_spec is not None else 0
    nd = env.num_dofs
    noise = np.random.uniform(-INIT_JOINT_NOISE, INIT_JOINT_NOISE, nd)
    env.data.qpos[base + 7:base + 7 + nd] = (
        env.data.qpos[base + 7:base + 7 + nd] + noise)
    if upper_init is not None:
        arm0, waist0 = upper_init
        hinge = base + 7
        env.data.qpos[hinge + 12:hinge + 15] = waist0
        env.data.qpos[hinge + 15:hinge + 23] = arm0
    yaw_quat = np.array([math.cos(init_yaw / 2.0), 0.0, 0.0,
                         math.sin(init_yaw / 2.0)])
    env.data.qpos[base + 3:base + 7] = yaw_quat
    if box_spec is not None and box_follows_pelvis:
        off = box_spec["offset_from_pelvis"]
        cy, sy = math.cos(init_yaw), math.sin(init_yaw)
        a = box_spec["box_qpos_adr"]
        env.data.qpos[a + 0] = env.data.qpos[base + 0] + cy * off[0] - sy * off[1]
        env.data.qpos[a + 1] = env.data.qpos[base + 1] + sy * off[0] + cy * off[1]
        env.data.qpos[a + 2] = env.data.qpos[base + 2] + off[2]
        env.data.qpos[a + 3:a + 7] = yaw_quat
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# psi0 upper-body replay (contract v1): npz loading, joint-name mapping and
# the per-control-step drive. Arms (8) go through play_amo's arm machinery
# (prev/arm_action with arm_blend=1 -> direct drive) AND are written onto the
# live pd_target; the waist (3) is part of the AMO policy action, so its
# pd_target entries are overwritten AFTER the policy ran and BEFORE the PD
# torque (SIMPLE hack, precedent SIMPLE g1_wholebody.py:271-272). play_amo's
# run() builds `pd_target` as a local right before calling viewer.render()
# (= the harness hook), so the hook's caller frame holds the exact array the
# PD loop consumes for the next 10 substeps -- see _apply_upper_replay.
# ---------------------------------------------------------------------------


def _strip_joint_suffix(name):
    return name[:-len("_joint")] if name.endswith("_joint") else name


def _map_upper_columns(names):
    """Map npz upper_names (amo23, K=11) -> (arm_cols, waist_cols) indices."""
    norm = [_strip_joint_suffix(str(n)) for n in names]
    expected = AMO_WAIST_NAMES + AMO_ARM_NAMES
    if len(norm) != len(expected):
        raise SystemExit("[bench_amo] upper-replay K=%d != %d (amo23 "
                         "contract: waist3 + arm8)" % (len(norm),
                                                       len(expected)))
    missing = [n for n in expected if n not in norm]
    if missing:
        raise SystemExit("[bench_amo] upper-replay joints missing for amo23: "
                         "%s (names=%s)" % (missing, norm))
    arm_cols = np.array([norm.index(n) for n in AMO_ARM_NAMES], dtype=int)
    waist_cols = np.array([norm.index(n) for n in AMO_WAIST_NAMES], dtype=int)
    return arm_cols, waist_cols


def load_upper_replay(path):
    """Load + validate a contract-v1 upper-replay npz (embodiment amo23).

    Required fields: t/upper_names/upper_pos/height_cmd/grasp_close_t/
    carry_pose/meta_json; dt must equal the 0.02 s control dt; the K=11
    columns are matched by joint NAME against AMO's waist3+arm8 ("_joint"
    suffix tolerated). arm_cols/waist_cols index arrays are attached.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise SystemExit("[bench_amo] upper-replay npz not found: %s" % path)
    z = np.load(path, allow_pickle=True)
    required = ("t", "upper_names", "upper_pos", "height_cmd",
                "grasp_close_t", "carry_pose", "meta_json")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise SystemExit("[bench_amo] upper-replay npz missing fields %s: %s"
                         % (missing, path))
    t = np.asarray(z["t"], dtype=np.float64)
    pos = np.asarray(z["upper_pos"], dtype=np.float64)
    height = np.asarray(z["height_cmd"], dtype=np.float64)
    carry = np.asarray(z["carry_pose"], dtype=np.float64)
    names = [str(n) for n in z["upper_names"]]
    meta = json.loads(str(z["meta_json"]))
    if len(t) < 2 or abs((t[1] - t[0]) - REPLAY_DT) > 1e-6:
        raise SystemExit("[bench_amo] upper-replay dt %s != control dt %s: %s"
                         % (t[1] - t[0] if len(t) > 1 else "n/a",
                            REPLAY_DT, path))
    if pos.shape != (t.shape[0], len(names)) or carry.shape != (len(names),):
        raise SystemExit("[bench_amo] upper-replay shape mismatch pos=%s "
                         "carry=%s names=%d: %s"
                         % (pos.shape, carry.shape, len(names), path))
    if height.shape != (t.shape[0],):
        raise SystemExit("[bench_amo] upper-replay height_cmd shape %s != "
                         "(%d,): %s" % (height.shape, t.shape[0], path))
    if not (np.isfinite(pos).all() and np.isfinite(height).all()
            and np.isfinite(carry).all()):
        raise SystemExit("[bench_amo] upper-replay contains NaN/inf: %s"
                         % path)
    arm_cols, waist_cols = _map_upper_columns(names)
    place_idx = int(np.argmin(height))   # squat_place_psi0: weld release time
    return _types.SimpleNamespace(
        path=path, names=names, pos=pos, height=height,
        grasp_close_t=float(z["grasp_close_t"]), carry_pose=carry,
        place_idx=place_idx, t_place=place_idx * REPLAY_DT,
        n=int(pos.shape[0]), duration_s=float(pos.shape[0]) * REPLAY_DT,
        source=str(meta.get("source", "unknown")),
        embodiment=str(meta.get("embodiment", "unknown")),
        arm_cols=arm_cols, waist_cols=waist_cols)


class UpperReplayDrive:
    """Per-control-step (arm8, waist3) targets from the replay npz.

    loop mode (walk/circle psi0): real_ep053 stream looped from step 0
    (settle included). squat mode (squat_box_psi0 AND squat_place_psi0):
    frame 0 held during settle, the stream plays ONCE from t=SETTLE_T,
    carry_pose frozen after it ends (squat_box_psi0: the squat cycles run,
    see Psi0SquatSchedule; squat_place_psi0: the 1 s final stand).
    """

    def __init__(self, replay, squat_mode):
        self.replay = replay
        self.squat_mode = squat_mode

    def row(self, t, step):
        rep = self.replay
        if self.squat_mode:
            k = int(math.floor((t - SETTLE_T) / REPLAY_DT + 1e-9))
            if k < 0:
                r = rep.pos[0]
            elif k >= rep.n:
                r = rep.carry_pose
            else:
                r = rep.pos[k]
        else:
            r = rep.pos[step % rep.n]
        return r[rep.arm_cols].copy(), r[rep.waist_cols].copy()


def _apply_upper_replay(env, run_frame, drive, t, step):
    """Overwrite THIS control step's upper-body pd_target from the replay.

    `run_frame` is play_amo.HumanoidEnv.run()'s frame (the hook's caller):
    its local `pd_target` (built from the policy action two lines above the
    viewer.render() call) is mutated in place, so the override reaches the
    PD torque of the current 10-substep block with zero delay. Arms are also
    routed through the arm machinery (blend=1 direct drive) so the env-side
    blend line agrees at the next control step; last_action / observations
    stay untouched (the policy keeps observing its own waist action, exactly
    like the SIMPLE precedent).
    """
    pd_target = run_frame.f_locals.get("pd_target")
    if not isinstance(pd_target, np.ndarray) or pd_target.shape != (23,):
        raise RuntimeError(
            "psi0 upper replay: pd_target not found in play_amo run() frame "
            "(play_amo.py changed upstream?)")
    arm, waist = drive.row(t, step)
    env.prev_arm_action = arm.copy()
    env.arm_action = arm.copy()
    env.arm_blend = 1.0                 # blend=1 -> direct drive next step
    pd_target[15:23] = arm              # arms 8: this control step too
    pd_target[12:15] = waist            # waist 3: SIMPLE-style overwrite


def _box_surface_dist(data, box_spec, hand_body_id):
    """Distance from a hand body origin to the oriented box SURFACE
    (0 when the origin is inside the box). Box half extents come from
    box_spec["box_half"] when present (T9 carry box), else the psi0
    cracker box."""
    half = np.asarray(box_spec.get("box_half", PSI0_BOX_HALF))
    bid = box_spec["box_body_id"]
    rel = data.xpos[hand_body_id] - data.xpos[bid]
    local = data.xmat[bid].reshape(3, 3).T @ rel
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def _activate_welds_at_current_pose(model, data, box_spec):
    """Anchor both wrist welds at the CURRENT hand->box relative pose and
    enable them — the box is NOT moved (squat_box_psi0 grasps it where it
    stands on the table). eq_data layout (mujoco>=3): anchor 0:3,
    relpose 3:10, torquescale 10. The cached model's eq_data is rewritten at
    every grasp; welds start inactive each trial via eq_active0 (XML
    active="false")."""
    bid = box_spec["box_body_id"]
    for eq_id, hand_bid in zip(box_spec["eq_ids"], box_spec["hand_body_ids"]):
        rot = data.xmat[hand_bid].reshape(3, 3)
        relp = rot.T @ (data.xpos[bid] - data.xpos[hand_bid])
        neg = np.zeros(4)
        relq = np.zeros(4)
        mujoco.mju_negQuat(neg, data.xquat[hand_bid])
        mujoco.mju_mulQuat(relq, neg, data.xquat[bid])
        model.eq_data[eq_id, 0:3] = 0.0
        model.eq_data[eq_id, 3:6] = relp
        model.eq_data[eq_id, 6:10] = relq
        model.eq_data[eq_id, 10] = 1.0
        data.eq_active[eq_id] = 1


# ---------------------------------------------------------------------------
# T10 vln_follow tapes (make_vln_tapes.py output, shared across the four
# harnesses so every model gets the IDENTICAL command streams).
# ---------------------------------------------------------------------------


def _tape_stop_segments(cmds, duration):
    """Maximal [t0, t1] intervals where the held command is a full stop
    (vx=vy=wz=0), at least VLN_STOP_MIN_T long."""
    segs = []
    t0 = None
    for i in range(cmds.shape[0]):
        stopped = bool(np.all(np.abs(cmds[i, 1:4]) < 1e-9))
        if stopped and t0 is None:
            t0 = float(cmds[i, 0])
        elif not stopped and t0 is not None:
            segs.append((t0, float(cmds[i, 0])))
            t0 = None
    if t0 is not None:
        segs.append((t0, float(duration)))
    return [s for s in segs if s[1] - s[0] >= VLN_STOP_MIN_T]


def _tape_cmd_at(tape, tau):
    """Zero-order-held (vx, vy, wz) at tape time tau."""
    idx = int(np.searchsorted(tape.cmd_ts, tau + 1e-9)) - 1
    idx = max(0, min(idx, tape.cmds.shape[0] - 1))
    row = tape.cmds[idx]
    return float(row[1]), float(row[2]), float(row[3])


def load_vln_tapes(path):
    """Load + validate tapes.json for T10.

    Format: {"tapes":[{"id":0,"dt":0.02,"cmds":[[t,vx,vy,wz],...]
    (breakpoints, zero-order hold), "ref_xy_yaw":[[t,x,y,yaw],...] (ideal
    error-free integral of the cmds, 1 Hz)}]}. dt must equal the 0.02 s
    control dt. Returns a list of SimpleNamespace tapes with precomputed
    stop segments."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise SystemExit("[bench_amo] tapes file not found: %s" % path)
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit("[bench_amo] cannot read tapes file %s: %r"
                         % (path, exc))
    tapes_raw = raw.get("tapes") if isinstance(raw, dict) else None
    if not isinstance(tapes_raw, list) or not tapes_raw:
        raise SystemExit("[bench_amo] tapes file has no non-empty 'tapes' "
                         "list: %s" % path)
    tapes = []
    for k, tp in enumerate(tapes_raw):
        try:
            cmds = np.asarray(tp["cmds"], dtype=np.float64)
            ref = np.asarray(tp["ref_xy_yaw"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("[bench_amo] tape %d malformed (%r): %s"
                             % (k, exc, path))
        if cmds.ndim != 2 or cmds.shape[1] != 4 or cmds.shape[0] < 1:
            raise SystemExit("[bench_amo] tape %d cmds shape %s != (M,4): %s"
                             % (k, cmds.shape, path))
        if ref.ndim != 2 or ref.shape[1] != 4 or ref.shape[0] < 2:
            raise SystemExit("[bench_amo] tape %d ref_xy_yaw shape %s != "
                             "(R,4): %s" % (k, ref.shape, path))
        if not (np.isfinite(cmds).all() and np.isfinite(ref).all()):
            raise SystemExit("[bench_amo] tape %d contains NaN/inf: %s"
                             % (k, path))
        if cmds[0, 0] < 0 or (np.diff(cmds[:, 0]) <= 0).any():
            raise SystemExit("[bench_amo] tape %d cmd breakpoint times not "
                             "strictly ascending from >= 0: %s" % (k, path))
        dt = float(tp.get("dt", REPLAY_DT))
        if abs(dt - REPLAY_DT) > 1e-9:
            raise SystemExit("[bench_amo] tape %d dt %s != control dt %s: %s"
                             % (k, dt, REPLAY_DT, path))
        duration = max(float(cmds[-1, 0]), float(ref[-1, 0]))
        if duration <= 0:
            raise SystemExit("[bench_amo] tape %d has zero duration: %s"
                             % (k, path))
        tapes.append(_types.SimpleNamespace(
            id=int(tp.get("id", k)), cmds=cmds, cmd_ts=cmds[:, 0].copy(),
            ref=ref, duration=duration,
            stops=_tape_stop_segments(cmds, duration)))
    return tapes


# ---------------------------------------------------------------------------
# Command schedules. Written into env.viewer.commands (8-dim, play_amo.py:164):
#   [0]=vx  [1]=ABSOLUTE target yaw (rad)  [2]=vy  [3]=height delta on 0.75
#   [4..6]=torso ypr  [7]=random-arm toggle (kept 0 always).
# NOTE: hook runs after get_observation, so commands set at control step k take
# effect at step k+1 (20 ms shift, negligible and constant).
# ---------------------------------------------------------------------------


class WalkSchedule:
    """T1 / sweep: settle 2s -> linear ramp vx over 2s -> hold 10s."""

    RAMP_T = 2.0
    HOLD_T = 10.0

    def __init__(self, yaw0, target_vx):
        self.yaw0 = yaw0
        self.target_vx = target_vx
        self.total_t = SETTLE_T + self.RAMP_T + self.HOLD_T

    def apply(self, env, t):
        c = env.viewer.commands
        c[1] = self.yaw0  # hold initial heading the whole trial
        if t < SETTLE_T:
            c[0] = 0.0
            return "settle"
        if t < SETTLE_T + self.RAMP_T:
            c[0] = self.target_vx * (t - SETTLE_T) / self.RAMP_T
            return "ramp"
        c[0] = self.target_vx
        return "hold"


def _squat_cycle_cmd(tc):
    """(height-delta command, phase) at point tc of one 5 s squat cycle
    (shared by squat_box and squat_box_psi0; numerics unchanged from v1)."""
    if tc < SQUAT_RAMP_T:                            # 0.0-1.5 down
        return SQUAT_HEIGHT_CMD * tc / SQUAT_RAMP_T, "down"
    if tc < SQUAT_RAMP_T + SQUAT_HOLD_T:             # 1.5-2.5 hold low
        return SQUAT_HEIGHT_CMD, "hold_low"
    if tc < 2.0 * SQUAT_RAMP_T + SQUAT_HOLD_T:       # 2.5-4.0 up
        return (SQUAT_HEIGHT_CMD * (1.0 - (tc - SQUAT_RAMP_T - SQUAT_HOLD_T)
                                    / SQUAT_RAMP_T), "up")
    return 0.0, "hold_high"                          # 4.0-5.0 hold high


class SquatSchedule:
    """T2: settle 2s -> arm hug settles (until t=5s) -> 20 x 5s squat cycles.

    wrist_weld_v2: the box welds are active from step 0, so the arm blend
    toward HUG_ARM_POSE is started at t=0 (first apply call) instead of after
    settle — minimizes the weld-vs-arm-pose mismatch transient. Phase names
    and the squat_start/total timing are unchanged from v1.
    """

    def __init__(self, yaw0):
        self.yaw0 = yaw0
        self.squat_start = SETTLE_T + ARM_TRANSITION_T
        self.total_t = self.squat_start + SQUAT_CYCLE_T * SQUAT_N_CYCLES
        self._arm_set = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0          # in-place stand the whole time
        c[1] = self.yaw0
        c[7] = 0.0          # keep random-arm machinery OFF
        if not self._arm_set:
            # Direct arm target injection (play_amo.py:317-318 blend path);
            # toggle_arm stays False so play_amo.py:311-316 never overrides.
            env.prev_arm_action = env.dof_pos[15:].copy()
            env.arm_action = HUG_ARM_POSE.copy()
            env.arm_blend = 0.0
            self._arm_set = True
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"
        if t < self.squat_start:
            c[3] = 0.0
            return "arm_transition"
        c[3], phase = _squat_cycle_cmd((t - self.squat_start) % SQUAT_CYCLE_T)
        return phase


class Psi0SquatSchedule:
    """squat_box_psi0: settle 2s -> sim_ep035 replay following its height_cmd
    (clipped to AMO's nominal domain; commands[3] = h - 0.75) -> carry_pose
    freeze + the standard 20 x 5s squat cycles. The upper body itself is
    driven by UpperReplayDrive in the run_trial hook, not here; the grasp /
    weld activation at grasp_close_t also lives in the hook."""

    def __init__(self, replay):
        self.yaw0 = 0.0                  # BendPick table scene alignment
        self.replay = replay
        self.squat_start = SETTLE_T + replay.duration_s
        self.total_t = self.squat_start + SQUAT_CYCLE_T * SQUAT_N_CYCLES

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0          # in-place stand the whole time
        c[1] = self.yaw0
        c[2] = 0.0
        c[7] = 0.0          # keep random-arm machinery OFF
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"
        if t < self.squat_start:
            k = min(int((t - SETTLE_T) / REPLAY_DT + 1e-9), self.replay.n - 1)
            h = _clipf(float(self.replay.height[k]),
                       PSI0_HEIGHT_CLIP[0], PSI0_HEIGHT_CLIP[1])
            c[3] = h - BASE_HEIGHT_DEFAULT
            return "replay"
        c[3], phase = _squat_cycle_cmd((t - self.squat_start) % SQUAT_CYCLE_T)
        return phase


class CircleSchedule:
    """T3: settle 2s -> vx=0.4 with yaw target integrated at wz=+-0.4 rad/s.

    AMO has no wz channel: commands[1] is an ABSOLUTE heading target, so we
    integrate yaw_cmd = dir*wz*(t-settle) (exact for constant wz). |vx|>=0.1
    keeps _in_place_stand_flag false so heading tracking is active.
    """

    def __init__(self, direction_sign):
        self.dir = direction_sign           # +1 ccw, -1 cw
        self.yaw0 = 0.0                     # fixed heading +x (spec: no random yaw)
        self.circle_t = 2.0 * math.pi / CIRCLE_WZ
        self.total_t = SETTLE_T + 2.0 * self.circle_t + 0.5  # buffer for closure 2

    def apply(self, env, t):
        c = env.viewer.commands
        if t < SETTLE_T:
            c[0] = 0.0
            c[1] = self.yaw0
            return "settle"
        c[0] = CIRCLE_VX
        c[1] = self.yaw0 + self.dir * CIRCLE_WZ * (t - SETTLE_T)
        return "circle1" if (t - SETTLE_T) < self.circle_t else "circle2"

    def cmd_yaw_progress(self, t):
        return CIRCLE_WZ * max(0.0, t - SETTLE_T)


class SquatSweepSchedule:
    """T4: settle 2s -> arm hug + v2 box welds (blend from t=0, same as
    SquatSchedule) -> ramp base height 0.75 -> H at `rate` m/s -> hold 3s ->
    ramp back to 0.75 -> 1s stand. commands[3] = h(t) - 0.75 (height delta).
    """

    def __init__(self, yaw0, target_h, rate):
        self.yaw0 = yaw0
        self.target_h = target_h
        self.rate = rate
        self.squat_start = SETTLE_T + ARM_TRANSITION_T
        self.ramp_t = (BASE_HEIGHT_DEFAULT - target_h) / rate
        self.hold_end = self.ramp_t + SQUAT_SWEEP_HOLD_T
        self.up_end = self.hold_end + self.ramp_t
        self.total_t = self.squat_start + self.up_end + SQUAT_SWEEP_FINAL_T
        self._arm_set = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0          # in-place the whole time
        c[1] = self.yaw0
        c[2] = 0.0
        c[7] = 0.0
        if not self._arm_set:
            env.prev_arm_action = env.dof_pos[15:].copy()
            env.arm_action = HUG_ARM_POSE.copy()
            env.arm_blend = 0.0
            self._arm_set = True
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"
        if t < self.squat_start:
            c[3] = 0.0
            return "arm_transition"
        tc = t - self.squat_start
        depth_cmd = self.target_h - BASE_HEIGHT_DEFAULT      # negative delta
        if tc < self.ramp_t:
            c[3] = -self.rate * tc
            return "down"
        if tc < self.hold_end:
            c[3] = depth_cmd
            return "hold"
        if tc < self.up_end:
            c[3] = depth_cmd + self.rate * (tc - self.hold_end)
            return "up"
        c[3] = 0.0
        return "final"


# ---------------------------------------------------------------------------
# Box release / placement helpers (shared by pipeline_abc T6 and
# squat_place_psi0 T8).
# ---------------------------------------------------------------------------


def _release_box_welds(env):
    """Deactivate both wrist welds and make the box collide with the ground
    (geom contype/conaffinity are runtime-writable model fields; they are
    reset per trial because the model is cached)."""
    for name in ("bench_weld_left", "bench_weld_right"):
        eq = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq < 0:
            raise RuntimeError("equality %r not found at release" % name)
        env.data.eq_active[eq] = 0
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                            "bench_box_geom")
    if gid < 0:
        raise RuntimeError("bench_box_geom not found at release")
    env.model.geom_contype[gid] = 2
    env.model.geom_conaffinity[gid] = 2
    # bit-2 only: AMO floor conaffinity=15 already includes bit 2, robot
    # geoms are bit 1 -> box lands on floor, never jams against the palms
    # (full-mask release caused solver depenetration buzz).
    # MuJoCo >= 3.2.4 broadphase culls BODY pairs via body_contype/
    # body_conaffinity (compile-time OR of each body's geom bits). The
    # box compiles 0/0, so geom-level writes alone are invisible to
    # broadphase and the box free-falls through the floor. The world
    # body inherits the ground's conaffinity=15 (bit 2 included), so
    # only the box body bits need updating.
    if hasattr(env.model, "body_contype"):
        bid = int(env.model.geom_bodyid[gid])
        env.model.body_contype[bid] = 2
        env.model.body_conaffinity[bid] = 2


def _box_end_state(env, box_spec):
    """Snapshot the box pose/speed/tilt/robot-distance (end-of-trial)."""
    a = box_spec["box_qpos_adr"]
    v = box_spec["box_qvel_adr"]
    b = box_spec["base_qpos_adr"]
    q = env.data.qpos
    quat = q[a + 3:a + 7]
    r22 = 1.0 - 2.0 * (float(quat[1]) ** 2 + float(quat[2]) ** 2)
    return {
        "pos": [float(q[a]), float(q[a + 1]), float(q[a + 2])],
        "speed": float(np.linalg.norm(env.data.qvel[v:v + 3])),
        "tilt_from_upright_rad": math.acos(_clipf(r22, -1.0, 1.0)),
        "dist_to_robot": math.hypot(float(q[a] - q[b]),
                                    float(q[a + 1] - q[b + 1])),
    }


def _release_anchor(env, base_q):
    """Base pose + foot FRONT EDGE at release time (T8 box_land_dx datum).

    Front edge = max heading-projection of the collidable ankle_roll sphere
    centers + their radius, relative to the base XY. box_land_dx later =
    heading-projection of the box landing point minus this edge (positive =
    box placed in front of the feet)."""
    q = env.data.qpos
    yaw = float(play_amo.quatToEuler(q[base_q + 3:base_q + 7])[2])
    hx, hy = math.cos(yaw), math.sin(yaw)
    bx, by = float(q[base_q]), float(q[base_q + 1])
    m = env.model
    s_front = None
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                      int(m.geom_bodyid[g])) or ""
        if "ankle_roll" not in body_name:
            continue
        s = ((float(env.data.geom_xpos[g][0]) - bx) * hx
             + (float(env.data.geom_xpos[g][1]) - by) * hy
             + float(m.geom_size[g][0]))
        s_front = s if s_front is None else max(s_front, s)
    if s_front is None:
        raise RuntimeError("release anchor: no collidable ankle_roll geoms "
                           "found (g1.xml changed upstream?)")
    return {"x": bx, "y": by, "hx": hx, "hy": hy, "s_front": s_front}


class SquatLimitSchedule:
    """T7: settle 2s (standing, v2 box welded from t=0, arm blend from the
    first apply like SquatSchedule) -> height cmd ramps 0.75 -> 0.10 m at a
    constant 0.05 m/s -> hold the 0.10 m command 3 s -> end (NO rise).

    commands[3] reaches -0.65, far below AMO's trained height domain, and is
    sent UNCLIPPED on purpose (play_amo's command layer has no clip either:
    commands[3] feeds the adapter/obs raw, play_amo.py:248,282) -- the
    saturation/fall boundary is the measurement. A fall aborts the trial
    (run_trial fall machinery).
    """

    def __init__(self, yaw0):
        self.yaw0 = yaw0
        self.descent_start = SETTLE_T
        self.descent_t = ((BASE_HEIGHT_DEFAULT - SQUAT_LIMIT_FLOOR_H)
                          / SQUAT_LIMIT_RATE)          # 13 s
        self.total_t = SETTLE_T + self.descent_t + SQUAT_LIMIT_HOLD_T
        self._arm_set = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0          # in-place stand the whole time
        c[1] = self.yaw0
        c[2] = 0.0
        c[7] = 0.0
        if not self._arm_set:
            env.prev_arm_action = env.dof_pos[15:].copy()
            env.arm_action = HUG_ARM_POSE.copy()
            env.arm_blend = 0.0
            self._arm_set = True
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"
        tc = t - self.descent_start
        if tc < self.descent_t:
            c[3] = -SQUAT_LIMIT_RATE * tc          # h_cmd = 0.75 - 0.05*tc
            return "descend"
        c[3] = SQUAT_LIMIT_FLOOR_H - BASE_HEIGHT_DEFAULT       # -0.65
        return "hold_floor"


class Psi0PlaceSchedule:
    """T8 squat_place_psi0: 2 kg v2 box welded from t=0 (standing pose) ->
    settle 2s -> SINGLE playback of the real_ep053 upper stream (arms+waist
    via UpperReplayDrive squat mode in the run_trial hook) with its
    height_cmd driving commands[3] (clipped to AMO's nominal domain like the
    other psi0 tests) -> at t_place = argmin(height_cmd) both welds release
    (bit-2 + body-bit mechanism, shared with pipeline_abc) -> replay rise
    segment -> 1 s stand. Phase segmentation descent/hold_place/rise comes
    from the height_cmd profile (hold = within PLACE_HOLD_TOL_M of its min).
    """

    def __init__(self, init_yaw, replay, box_spec):
        self.yaw0 = init_yaw
        self.replay = replay
        self.box_spec = box_spec
        self.replay_end = SETTLE_T + replay.duration_s
        self.total_t = self.replay_end + PLACE_FINAL_STAND_T + 1.0
        in_hold = np.flatnonzero(
            replay.height <= float(replay.height.min()) + PLACE_HOLD_TOL_M)
        self.k_hold0 = int(in_hold[0])
        self.k_hold1 = int(in_hold[-1])
        self.release_t = None
        self.release_anchor = None
        self.final_box = None
        self.finished = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0          # in-place stand the whole time
        c[1] = self.yaw0
        c[2] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"
        if t < self.replay_end:
            k = min(int((t - SETTLE_T) / REPLAY_DT + 1e-9), self.replay.n - 1)
            h = _clipf(float(self.replay.height[k]),
                       PSI0_HEIGHT_CLIP[0], PSI0_HEIGHT_CLIP[1])
            c[3] = h - BASE_HEIGHT_DEFAULT
            if self.release_t is None and (t - SETTLE_T) >= self.replay.t_place:
                self.release_anchor = _release_anchor(
                    env, self.box_spec["base_qpos_adr"])
                _release_box_welds(env)
                self.release_t = round(t, 3)
            if k < self.k_hold0:
                return "descent"
            if k <= self.k_hold1:
                return "hold_place"
            return "rise"
        c[3] = 0.0
        if t >= self.replay_end + PLACE_FINAL_STAND_T:
            if self.final_box is None:
                fb = _box_end_state(env, self.box_spec)
                if self.release_anchor is not None:
                    a = self.release_anchor
                    fb["land_dx"] = ((fb["pos"][0] - a["x"]) * a["hx"]
                                     + (fb["pos"][1] - a["y"]) * a["hy"]
                                     - a["s_front"])
                else:
                    fb["land_dx"] = None
                self.final_box = fb
            self.finished = True
        return "stand_final"


class PickGroundSchedule:
    """T9 squat_pick_ground: empty-handed stand (arms at the hanging
    keyframe pose) -> settle 2 s -> arms blend to the low front-reach pose
    (PICK_ARM_POSE, play_amo blend path: ~2 s) -> height cmd ramps
    0.75 -> 0.25 m at 0.2 m/s (commands[3] -> -0.50, sent UNCLIPPED; AMO
    saturates near its R4 depth floor ~0.345 m) -> bottom hold: the first
    control step with BOTH wrists < PICK_GRASP_DIST from the box surface
    triggers the magnetic grasp (welds anchored at the live pose + enabled,
    squat_box_psi0 mechanism; the box is NOT moved) -> 0.5 s post-grasp
    hold -> height ramps back to 0.75 with the 2 kg load -> 1 s stand.
    No grasp within 5 s of bottom-hold start -> pick_failed, rise anyway.
    """

    def __init__(self, init_yaw, box_spec):
        self.yaw0 = init_yaw
        self.box_spec = box_spec
        self.descend_start = SETTLE_T + ARM_TRANSITION_T
        self.ramp_t = (BASE_HEIGHT_DEFAULT - PICK_SQUAT_H) / PICK_SQUAT_RATE
        self.bottom_start = self.descend_start + self.ramp_t
        self.total_t = (self.bottom_start + PICK_GRASP_TIMEOUT_T
                        + PICK_POST_GRASP_HOLD_T + self.ramp_t
                        + PICK_FINAL_STAND_T + 2.0)
        self._arm_set = False
        self.grasp_t = None
        self.grasp_dists = None       # at grasp time, or at timeout
        self.pick_timeout = False
        self.rise_start = None
        self.final_box = None
        self.finished = False

    def _wrist_dists(self, env):
        return tuple(_box_surface_dist(env.data, self.box_spec, hb)
                     for hb in self.box_spec["hand_body_ids"])

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0          # in-place the whole trial
        c[1] = self.yaw0
        c[2] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"             # arms still hanging (keyframe pose)
        if not self._arm_set:
            env.prev_arm_action = env.dof_pos[15:].copy()
            env.arm_action = PICK_ARM_POSE.copy()
            env.arm_blend = 0.0
            self._arm_set = True
        if t < self.descend_start:
            c[3] = 0.0
            return "arm_reach"
        depth_cmd = PICK_SQUAT_H - BASE_HEIGHT_DEFAULT       # -0.50
        if self.rise_start is None:
            tc = t - self.descend_start
            if tc < self.ramp_t:
                c[3] = -PICK_SQUAT_RATE * tc
                return "descend"
            c[3] = depth_cmd
            if self.grasp_t is None:
                dists = self._wrist_dists(env)
                if max(dists) < PICK_GRASP_DIST:
                    self.grasp_dists = dists
                    self.grasp_t = round(t, 3)
                    _activate_welds_at_current_pose(env.model, env.data,
                                                    self.box_spec)
                elif (t - self.bottom_start) >= PICK_GRASP_TIMEOUT_T:
                    self.grasp_dists = dists          # timeout snapshot
                    self.pick_timeout = True
                    self.rise_start = t
                return "bottom_hold"
            if (t - self.grasp_t) < PICK_POST_GRASP_HOLD_T:
                return "hold_grasp"
            self.rise_start = t
        tr = t - self.rise_start
        if tr < self.ramp_t:
            c[3] = depth_cmd + PICK_SQUAT_RATE * tr
            return "rise"
        c[3] = 0.0
        if tr >= self.ramp_t + PICK_FINAL_STAND_T:
            if self.final_box is None:
                self.final_box = _box_end_state(env, self.box_spec)
            self.finished = True
        return "stand_final"


class VlnFollowSchedule:
    """T10 vln_follow: settle 2 s -> tape commands zero-order-held
    (vx -> commands[0], vy -> commands[2]) -> 1 s final stand.

    AMO has NO wz channel: wz is INTEGRATED into the ABSOLUTE heading
    target commands[1] (CircleSchedule/goto_ab realisation, unwrapped like
    CircleSchedule -- play_amo wraps internally). Semantic difference
    (heading tracking vs rate tracking) is recorded, not worked around;
    the in-place-stand interlock freezes heading tracking while
    |vx_cmd| < 0.1, counted in vln_yaw_locked_frac. The tape frame is
    anchored at the robot's ACTUAL pose at tape start (t = SETTLE_T) so
    settle drift does not bias the ref comparison; the integrated heading
    starts from the actual yaw too."""

    def __init__(self, tape):
        self.yaw0 = 0.0                 # tape ref frame: start at yaw 0
        self.tape = tape
        self.tape_end = SETTLE_T + tape.duration
        self.total_t = self.tape_end + VLN_FINAL_STAND_T + 1.0
        self.anchor = None              # (x0, y0, yaw0_actual) at tape start
        self.cmd_yaw = None
        self.tape_steps = 0
        self.locked_steps = 0
        self.finished = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[3] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[0] = 0.0
            c[2] = 0.0
            c[1] = self.yaw0
            return "settle"
        if self.anchor is None:
            q = env.data.qpos
            yaw = float(play_amo.quatToEuler(q[3:7])[2])
            self.anchor = (float(q[0]), float(q[1]), yaw)
            self.cmd_yaw = yaw
        tau = t - SETTLE_T
        if tau < self.tape.duration:
            vx, vy, wz = _tape_cmd_at(self.tape, tau)
            c[0] = vx
            c[2] = vy
            self.cmd_yaw += wz * env.control_dt   # wz -> absolute yaw target
            c[1] = self.cmd_yaw
            self.tape_steps += 1
            if abs(vx) < AMO_STAND_VX_THRESH:
                self.locked_steps += 1
            if abs(vx) < 1e-9 and abs(vy) < 1e-9 and abs(wz) < 1e-9:
                return "stopped"
            return "follow"
        c[0] = 0.0
        c[2] = 0.0
        c[1] = self.cmd_yaw
        if t >= self.tape_end + VLN_FINAL_STAND_T:
            self.finished = True
        return "stand_final"


# ---------------------------------------------------------------------------
# T5/T6 shared navigation controller (spec v3). The error/command math below
# is the cross-harness contract -- copied from the spec, identical in all
# four harnesses. AMO adaptation lives ONLY in how the wz intent is realised:
# commands[1] is an ABSOLUTE heading target (AMO has no wz channel), so NAV
# writes the heading target directly and CAL slew-limits it to 0.10 rad/s.
# ---------------------------------------------------------------------------


def _clipf(v, lo, hi):
    return min(max(v, lo), hi)


def nav_errors(x, y, yaw, goal_xy, goal_yaw):
    """Spec v3 error block: world error -> body frame; heading target is the
    bearing to the goal while >0.5 m away, the goal heading once within."""
    dx = goal_xy[0] - x
    dy = goal_xy[1] - y
    dist = math.hypot(dx, dy)
    ex = math.cos(yaw) * dx + math.sin(yaw) * dy
    ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
    yaw_target = math.atan2(dy, dx) if dist > NAV_SWITCH_DIST else goal_yaw
    eyaw = wrap_angle(yaw_target - yaw)
    return dist, ex, ey, eyaw, yaw_target


def nav_cmd_coarse(ex, ey, eyaw, vx_max):
    """NAV P-law: vx=clip(1.0*ex,0,vx_max), vy=clip(1.0*ey,+-0.3),
    wz=clip(1.5*eyaw,+-0.6)."""
    vx = _clipf(NAV_KP_POS * ex, 0.0, vx_max)
    vy = _clipf(NAV_KP_POS * ey, -NAV_VY_MAX, NAV_VY_MAX)
    wz = _clipf(NAV_KP_YAW * eyaw, -NAV_WZ_MAX, NAV_WZ_MAX)
    return vx, vy, wz


def nav_cmd_cal(ex, ey, eyaw):
    """CAL: same P-law, every command clamped to |0.10| (no deadzone
    compensation on purpose -- small-command efficacy is what is measured)."""
    vx = _clipf(NAV_KP_POS * ex, -CAL_CMD_LIM, CAL_CMD_LIM)
    vy = _clipf(NAV_KP_POS * ey, -CAL_CMD_LIM, CAL_CMD_LIM)
    wz = _clipf(NAV_KP_YAW * eyaw, -CAL_CMD_LIM, CAL_CMD_LIM)
    return vx, vy, wz


class _NavCalController:
    """Two-phase NAV -> CAL state machine driving env.viewer.commands.

    Phase results land in self.result: err_pos_nav/err_yaw_nav/t_nav/
    success_coarse at NAV exit (criterion 0.30 m & 15 deg, or 30 s timeout),
    err_pos_cal/err_yaw_cal/t_cal/success_fine at CAL exit (5 cm & 5 deg held
    1 s, or 15 s timeout), plus cal_yaw_locked_frac = fraction of CAL steps
    with |vx_cmd| < 0.1 where AMO's in-place-stand interlock freezes dyaw.
    """

    def __init__(self, goal_xy, goal_yaw, vx_max, base_qpos_adr):
        self.goal_xy = goal_xy
        self.goal_yaw = goal_yaw
        self.vx_max = vx_max
        self.base_q = base_qpos_adr
        self.phase = "nav"
        self.t0 = None
        self.cal_t0 = None
        self.cmd_yaw = None
        self.in_tol_since = None
        self.cal_steps = 0
        self.cal_locked_steps = 0
        self.result = {}
        self.done = False

    def _pose(self, env):
        q = env.data.qpos
        b = self.base_q
        yaw = float(play_amo.quatToEuler(q[b + 3:b + 7])[2])
        return float(q[b]), float(q[b + 1]), yaw

    def _finish_cal(self, t, dist, eyaw, converged):
        self.result.update({
            "err_pos_cal": dist,
            "err_yaw_cal": abs(eyaw),
            "t_cal": round(t - self.cal_t0, 3),
            "success_fine": bool(converged),
            "cal_yaw_locked_frac": (self.cal_locked_steps / self.cal_steps
                                    if self.cal_steps else None),
        })
        self.phase = "done"
        self.done = True

    def step(self, env, t):
        """One control step; returns the phase label. Keeps commands[3,7]
        untouched (owned by the parent schedule)."""
        if self.t0 is None:
            self.t0 = t
        x, y, yaw = self._pose(env)
        dist, ex, ey, eyaw, yaw_target = nav_errors(
            x, y, yaw, self.goal_xy, self.goal_yaw)
        c = env.viewer.commands

        if self.phase == "nav":
            reached = dist <= NAV_POS_TOL and abs(eyaw) <= NAV_YAW_TOL
            if reached or (t - self.t0) >= NAV_TIMEOUT_T:
                self.result.update({
                    "err_pos_nav": dist,
                    "err_yaw_nav": abs(eyaw),
                    "t_nav": round(t - self.t0, 3),
                    "success_coarse": bool(reached),
                })
                self.phase = "cal"
                self.cal_t0 = t
                self.cmd_yaw = yaw        # CAL slews from the actual heading
            else:
                vx, vy, _wz = nav_cmd_coarse(ex, ey, eyaw, self.vx_max)
                c[0] = vx
                c[2] = vy
                c[1] = yaw_target         # absolute heading channel (no wz)
                return "nav"

        if self.phase == "cal":
            in_tol = dist <= CAL_POS_TOL and abs(eyaw) <= CAL_YAW_TOL
            if in_tol:
                if self.in_tol_since is None:
                    self.in_tol_since = t
                converged = (t - self.in_tol_since) >= CAL_HOLD_T
            else:
                self.in_tol_since = None
                converged = False
            if converged or (t - self.cal_t0) >= CAL_TIMEOUT_T:
                self._finish_cal(t, dist, eyaw, converged)
                c[0] = 0.0
                c[2] = 0.0
                c[1] = self.goal_yaw
                return "cal"
            vx, vy, _wz = nav_cmd_cal(ex, ey, eyaw)
            c[0] = vx
            c[2] = vy
            # "small wz" for AMO = slew-limited absolute heading increments
            # (<= 0.10 rad/s * dt per step). NOTE: with |vx_cmd| < 0.1 the
            # policy's in-place-stand interlock freezes dyaw anyway -- this
            # is recorded (cal_yaw_locked_frac), not worked around.
            dyaw = wrap_angle(yaw_target - self.cmd_yaw)
            max_step = CAL_CMD_LIM * env.control_dt
            self.cmd_yaw = wrap_angle(
                self.cmd_yaw + _clipf(dyaw, -max_step, max_step))
            c[1] = self.cmd_yaw
            self.cal_steps += 1
            if abs(vx) < AMO_STAND_VX_THRESH:
                self.cal_locked_steps += 1
            return "cal"

        c[0] = 0.0
        c[2] = 0.0
        c[1] = self.goal_yaw
        return "done"


class GotoSchedule:
    """T5: settle 2s -> NAV -> CAL -> stand 1s. No box."""

    def __init__(self, init_yaw):
        self.yaw0 = init_yaw
        self.nav = _NavCalController(GOTO_B_XY, GOTO_B_YAW, NAV_VX_MAX,
                                     base_qpos_adr=0)
        self.end_t0 = None
        self.finished = False
        self.total_t = (SETTLE_T + NAV_TIMEOUT_T + CAL_TIMEOUT_T
                        + GOTO_END_T + 2.0)

    def apply(self, env, t):
        c = env.viewer.commands
        c[3] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[0] = 0.0
            c[2] = 0.0
            c[1] = self.yaw0
            return "settle"
        if not self.nav.done:
            return self.nav.step(env, t)
        if self.end_t0 is None:
            self.end_t0 = t
        c[0] = 0.0
        c[2] = 0.0
        c[1] = GOTO_B_YAW
        if t - self.end_t0 >= GOTO_END_T:
            self.finished = True
        return "end"


class PipelineSchedule:
    """T6: box carried from t=0 (v2 welds + arm blend) -> settle 2s -> NAV to
    C (vx_max 0.5) -> CAL -> squat to 0.45 @ 0.2 m/s -> bottom hold 0.5s ->
    release both welds + enable box collision -> hold 1s -> rise -> stand 1s.
    """

    def __init__(self, init_yaw, box_spec):
        self.yaw0 = init_yaw
        self.box_spec = box_spec
        self.nav = _NavCalController(PIPE_C_XY, PIPE_C_YAW, PIPE_NAV_VX_MAX,
                                     base_qpos_adr=box_spec["base_qpos_adr"])
        self.ramp_t = (BASE_HEIGHT_DEFAULT - PIPE_SQUAT_H) / PIPE_SQUAT_RATE
        self.rise_start = self.ramp_t + PIPE_BOTTOM_HOLD_T + PIPE_PLACE_HOLD_T
        self.total_t = (SETTLE_T + NAV_TIMEOUT_T + CAL_TIMEOUT_T
                        + self.rise_start + self.ramp_t + PIPE_FINAL_T + 2.0)
        self._arm_set = False
        self.squat_t0 = None
        self.release_t = None
        self.final_box = None
        self.finished = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[7] = 0.0
        if not self._arm_set:
            env.prev_arm_action = env.dof_pos[15:].copy()
            env.arm_action = HUG_ARM_POSE.copy()
            env.arm_blend = 0.0
            self._arm_set = True
        if t < SETTLE_T:
            c[0] = 0.0
            c[2] = 0.0
            c[1] = self.yaw0
            c[3] = 0.0
            return "settle"
        if not self.nav.done:
            c[3] = 0.0
            return self.nav.step(env, t)
        if self.squat_t0 is None:
            self.squat_t0 = t
        c[0] = 0.0
        c[2] = 0.0
        c[1] = PIPE_C_YAW
        tc = t - self.squat_t0
        depth_cmd = PIPE_SQUAT_H - BASE_HEIGHT_DEFAULT       # -0.30
        if tc < self.ramp_t:
            c[3] = -PIPE_SQUAT_RATE * tc
            return "squat_down"
        if tc < self.ramp_t + PIPE_BOTTOM_HOLD_T:
            c[3] = depth_cmd
            return "bottom_hold"
        if self.release_t is None:
            _release_box_welds(env)            # shared with squat_place_psi0
            self.release_t = round(t, 3)
        if tc < self.rise_start:
            c[3] = depth_cmd
            return "place_hold"
        if tc < self.rise_start + self.ramp_t:
            c[3] = min(0.0, depth_cmd
                       + PIPE_SQUAT_RATE * (tc - self.rise_start))
            return "rise"
        c[3] = 0.0
        if tc >= self.rise_start + self.ramp_t + PIPE_FINAL_T:
            if self.final_box is None:
                self.final_box = _box_end_state(env, self.box_spec)
            self.finished = True
        return "stand_final"


# ---------------------------------------------------------------------------
# Contact / fall machinery
# ---------------------------------------------------------------------------


def build_geom_sets(model):
    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    ground = gid("ground")
    pillar = gid("pillar")        # -1 when absent
    bench_box = gid("bench_box_geom")   # psi0 box collides (bit 2): exclude
    bench_table = gid("bench_table")    # psi0 table (static): exclude
    robot, feet = set(), set()
    for g in range(model.ngeom):
        if g in (ground, pillar, bench_box, bench_table):
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual-only mesh (arms/torso/knees) or the v2 box
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[g])) or ""
        robot.add(g)
        if "ankle_roll" in body_name:
            feet.add(g)
    # expected: 9 collidable robot geoms = pelvis_contour + 8 foot spheres
    return {"ground": ground, "pillar": pillar, "robot": robot, "feet": feet}


def scan_contacts(data, gs):
    """Return (non_foot_ground_contact, pillar_contact)."""
    nonfoot_ground = False
    pillar_hit = False
    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        for a, b in ((g1, g2), (g2, g1)):
            if a == gs["ground"] and b in gs["robot"] and b not in gs["feet"]:
                nonfoot_ground = True
            if gs["pillar"] >= 0 and a == gs["pillar"] and b in gs["robot"]:
                pillar_hit = True
    return nonfoot_ground, pillar_hit


def fall_reason(data, tilt, base_z, height_target, nonfoot_ground):
    """Unified fall criteria. Note: 'non-foot contact' only covers the
    pelvis_contour geom — arms/torso/knees have no collision geoms in g1.xml."""
    if not np.isfinite(data.qpos).all():
        return "nan_state"
    if tilt > FALL_TILT_RAD:
        return "tilt"
    if base_z < height_target - FALL_HEIGHT_MARGIN:
        return "base_height"
    if nonfoot_ground:
        return "non_foot_ground_contact"
    return None


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Trial runner: drives env.run() via the per-control-step render hook.
# ---------------------------------------------------------------------------


class TrialAbort(Exception):
    pass


def _track_circle_closure(schedule, state, t, x, y, yaw):
    if t < SETTLE_T:
        return
    if state["start_pose"] is None:
        state["start_pose"] = (x, y, yaw)
    progress = schedule.cmd_yaw_progress(t)
    n_done = len(state["closures"])
    if n_done < 2 and progress >= 2.0 * math.pi * (n_done + 1):
        sx, sy, syaw = state["start_pose"]
        state["closures"].append({
            "circle": n_done + 1,
            "pos_err_m": math.hypot(x - sx, y - sy),
            "yaw_err_rad": abs(wrap_angle(yaw - syaw)),
        })


_DIVERGENCE_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")


def _solver_diverged(data):
    """True when MuJoCo flagged bad qpos/qvel/qacc (it auto-resets the state
    on those warnings, so the counters are the reliable signal) or qacc blew
    up past QACC_DIVERGE_LIMIT without tripping a warning yet."""
    for wname in _DIVERGENCE_WARNINGS:
        if int(data.warning[getattr(mujoco.mjtWarning, wname)].number) > 0:
            return True
    qacc = data.qacc
    return (not np.isfinite(qacc).all()) or float(np.abs(qacc).max()) > QACC_DIVERGE_LIMIT


def run_trial(env, schedule, geom_sets, capture_frames, pillar_center=None,
              box_spec=None, upper_drive=None, grasp_close_t=None):
    rec = {k: [] for k in ("t", "phase", "x", "y", "z", "yaw", "tilt",
                           "vx_fwd", "cmd_vx", "h_target")}
    state = {
        "step": 0,
        "fall_time": None, "fall_phase": None, "fall_reason": None,
        "pillar_contact": False, "proximity_violation": False,
        "min_pillar_dist": float("inf"),
        "start_pose": None, "closures": [],
        "box_dist_max": None, "box_drop_time": None,
        "pick_checked": False, "pick_success": None, "grasp_dists": None,
        "box_z0": None, "box_z_max": None,
    }
    frames = []
    ctrl_dt = env.control_dt  # 0.02 s (50 Hz policy)
    base_q = box_spec["base_qpos_adr"] if box_spec is not None else 0
    base_v = box_spec["base_qvel_adr"] if box_spec is not None else 0

    def hook():
        i = state["step"]
        state["step"] = i + 1
        t = i * ctrl_dt
        phase = schedule.apply(env, t)

        if upper_drive is not None:
            # caller frame = play_amo.HumanoidEnv.run() (holds pd_target)
            _apply_upper_replay(env, sys._getframe(1), upper_drive, t, i)

        d = env.data
        x = float(d.qpos[base_q + 0])
        y = float(d.qpos[base_q + 1])
        z = float(d.qpos[base_q + 2])
        rpy = play_amo.quatToEuler(d.qpos[base_q + 3:base_q + 7])
        tilt = math.acos(max(-1.0, min(1.0,
                                       math.cos(rpy[0]) * math.cos(rpy[1]))))
        h_target = BASE_HEIGHT_DEFAULT + float(env.viewer.commands[3])
        # world base velocity projected onto current heading
        vx_fwd = float(d.qvel[base_v + 0] * math.cos(rpy[2]) +
                       d.qvel[base_v + 1] * math.sin(rpy[2]))

        rec["t"].append(t)
        rec["phase"].append(phase)
        rec["x"].append(x)
        rec["y"].append(y)
        rec["z"].append(z)
        rec["yaw"].append(float(rpy[2]))
        rec["tilt"].append(tilt)
        rec["vx_fwd"].append(vx_fwd)
        rec["cmd_vx"].append(float(env.viewer.commands[0]))
        rec["h_target"].append(h_target)

        nonfoot_ground, pillar_hit = scan_contacts(d, geom_sets)
        if pillar_center is not None:
            dist = math.hypot(x - pillar_center[0], y - pillar_center[1])
            state["min_pillar_dist"] = min(state["min_pillar_dist"], dist)
            if pillar_hit:
                state["pillar_contact"] = True
            if dist < PROXIMITY_COLLISION_DIST:
                state["proximity_violation"] = True
            _track_circle_closure(schedule, state, t, x, y, float(rpy[2]))

        if box_spec is not None:
            # Divergence check FIRST: a weld-vs-PD blowup must be recorded as
            # harness_error (RuntimeError -> main()'s catch), not as a fall.
            if _solver_diverged(d):
                raise RuntimeError(
                    "solver divergence at t=%.2fs (BADQPOS/QVEL/QACC warning "
                    "or |qacc|>%.0g): wrist-weld vs arm-PD blowup" %
                    (t, QACC_DIVERGE_LIMIT))
            if grasp_close_t is not None:
                # squat_box_psi0: track box height + the one-shot grasp event
                # (grasp_close_t is on the replay clock, which starts at
                # SETTLE_T).
                bz = float(d.xpos[box_spec["box_body_id"]][2])
                state["box_z0"] = bz if state["box_z0"] is None \
                    else state["box_z0"]
                state["box_z_max"] = bz if state["box_z_max"] is None \
                    else max(state["box_z_max"], bz)
                if (not state["pick_checked"]
                        and (t - SETTLE_T) >= grasp_close_t):
                    state["pick_checked"] = True
                    dists = tuple(_box_surface_dist(d, box_spec, hb)
                                  for hb in box_spec["hand_body_ids"])
                    state["grasp_dists"] = dists
                    state["pick_success"] = bool(
                        max(dists) < PSI0_GRASP_DIST)
                    if state["pick_success"]:
                        _activate_welds_at_current_pose(env.model, d,
                                                        box_spec)
            # box_kept tracking: for squat_box_psi0 only once the box is
            # welded (before the grasp it legitimately sits on the table,
            # far from the torso).
            track_box = grasp_close_t is None or state["pick_success"] is True
            if track_box:
                box_dist = float(np.linalg.norm(
                    d.xpos[box_spec["box_body_id"]] -
                    d.xpos[box_spec["torso_body_id"]]))
                state["box_dist_max"] = max(state["box_dist_max"] or 0.0,
                                            box_dist)
                if box_dist >= BOX_KEEP_DIST \
                        and state["box_drop_time"] is None:
                    state["box_drop_time"] = round(t, 3)

        if capture_frames and (i % VIDEO_EVERY_N_CTRL == 0):
            frames.append(d.qpos.copy())

        reason = fall_reason(d, tilt, z, h_target, nonfoot_ground)
        if reason is not None:
            state["fall_time"] = round(t, 3)
            state["fall_phase"] = phase
            state["fall_reason"] = reason
            if capture_frames:
                frames.append(d.qpos.copy())
            raise TrialAbort()

        # State-machine schedules (T5/T6) end themselves before total_t.
        if getattr(schedule, "finished", False):
            if capture_frames:
                frames.append(d.qpos.copy())
            raise TrialAbort()

    env.viewer.render = hook              # called once per control step
    env.sim_duration = schedule.total_t   # overrides 2000 s default
    try:
        env.run()
    except TrialAbort:
        pass
    return rec, state, frames


# ---------------------------------------------------------------------------
# Per-test metrics
# ---------------------------------------------------------------------------


def _rmse(err):
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else None


def compute_walk_metrics(rec, state, schedule):
    t = np.asarray(rec["t"])
    vx = np.asarray(rec["vx_fwd"])
    cmd = np.asarray(rec["cmd_vx"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    test_mask = t >= SETTLE_T
    last8_mask = t >= (schedule.total_t - 8.0)
    mean_vx_last8 = float(vx[last8_mask].mean()) if last8_mask.any() else None
    metrics = {
        "target_vx": schedule.target_vx,
        "init_yaw_rad": round(schedule.yaw0, 4),
        "mean_vx_last8s": mean_vx_last8,
        "vx_rmse_test_seg": _rmse((vx - cmd)[test_mask]),
        "vx_rmse_last8s": _rmse((vx - cmd)[last8_mask]),
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
    }
    success = ((not fell) and mean_vx_last8 is not None
               and mean_vx_last8 >= 0.9 * schedule.target_vx)
    return success, metrics


def compute_squat_metrics(rec, state, schedule):
    t = np.asarray(rec["t"])
    z = np.asarray(rec["z"])
    h_target = np.asarray(rec["h_target"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    if fell:
        ft = state["fall_time"]
        if ft <= schedule.squat_start:
            cycles = 0
        else:
            cycles = int(min(SQUAT_N_CYCLES,
                             (ft - schedule.squat_start) // SQUAT_CYCLE_T))
    else:
        cycles = SQUAT_N_CYCLES

    box_tracked = state["box_dist_max"] is not None
    box_kept = (state["box_drop_time"] is None) if box_tracked else None

    squat_mask = t >= schedule.squat_start
    metrics = {
        "init_yaw_rad": round(schedule.yaw0, 4),
        "cycles_completed": cycles,
        "height_rmse_m": _rmse((z - h_target)[squat_mask]),
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
        "fall_cycle": (int((state["fall_time"] - schedule.squat_start)
                           // SQUAT_CYCLE_T) + 1
                       if fell and state["fall_time"] > schedule.squat_start
                       else None),
        "box_kept": box_kept,
        "box_drop_time": state["box_drop_time"],
        "box_torso_dist_max_m": state["box_dist_max"],
    }
    success = (not fell) and cycles == SQUAT_N_CYCLES and bool(box_kept)
    return success, metrics


def compute_psi0_squat_metrics(rec, state, schedule):
    """squat_box_psi0: table pick (grasp gate at grasp_close_t) + carry_pose
    freeze + standard 20x5 s squat cycles. success := no fall AND pick_success
    AND 20/20 cycles AND box_kept (welded box stays < 0.6 m from the torso).
    pick_failed trials keep running (cycles without the box) but cannot
    succeed."""
    t = np.asarray(rec["t"])
    z = np.asarray(rec["z"])
    h_target = np.asarray(rec["h_target"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    if fell:
        ft = state["fall_time"]
        cycles = (0 if ft <= schedule.squat_start
                  else int(min(SQUAT_N_CYCLES,
                               (ft - schedule.squat_start) // SQUAT_CYCLE_T)))
    else:
        cycles = SQUAT_N_CYCLES

    pick = bool(state["pick_success"])
    box_kept = pick and state["box_drop_time"] is None
    gd = state["grasp_dists"] or (None, None)
    test_mask = t >= SETTLE_T            # replay + squat cycles
    metrics = {
        "cycles_completed": cycles,
        "height_rmse_m": _rmse((z - h_target)[test_mask]),
        "pick_success": pick,
        "pick_checked": bool(state["pick_checked"]),
        "grasp_dist_left_m": gd[0],
        "grasp_dist_right_m": gd[1],
        "box_lift_height_m": (state["box_z_max"] - state["box_z0"]
                              if state["box_z0"] is not None else None),
        "box_kept": box_kept,
        "box_drop_time": state["box_drop_time"],
        "box_torso_dist_max_m": state["box_dist_max"],
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
        "fall_cycle": (int((state["fall_time"] - schedule.squat_start)
                           // SQUAT_CYCLE_T) + 1
                       if fell and state["fall_time"] > schedule.squat_start
                       else None),
    }
    success = (not fell) and pick and cycles == SQUAT_N_CYCLES and box_kept
    return success, metrics


def psi0_box_metrics(success, metrics, state):
    """walk/circle psi0: add the box fields and gate success on box_kept
    (cracker box wrist-welded from t=0). Returns a NEW metrics dict."""
    box_kept = state["box_drop_time"] is None
    out = dict(metrics)
    out["box_kept"] = box_kept
    out["box_drop_time"] = state["box_drop_time"]
    out["box_torso_dist_max_m"] = state["box_dist_max"]
    return success and box_kept, out


def compute_circle_metrics(rec, state, schedule, pillar_center, direction):
    t = np.asarray(rec["t"])
    x = np.asarray(rec["x"])
    y = np.asarray(rec["y"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    test_mask = t >= SETTLE_T
    radial_err = np.abs(np.hypot(x - pillar_center[0], y - pillar_center[1])
                        - CIRCLE_RADIUS)[test_mask]
    mean_radial = float(radial_err.mean()) if len(radial_err) else None
    closures = {c["circle"]: c for c in state["closures"]}
    collision = state["pillar_contact"] or state["proximity_violation"]

    metrics = {
        "direction": direction,
        "radial_err_mean_m": mean_radial,
        "radial_err_max_m": float(radial_err.max()) if len(radial_err) else None,
        "circle1_closure_pos_err_m": closures.get(1, {}).get("pos_err_m"),
        "circle1_closure_yaw_err_rad": closures.get(1, {}).get("yaw_err_rad"),
        "circle2_closure_pos_err_m": closures.get(2, {}).get("pos_err_m"),
        "circle2_closure_yaw_err_rad": closures.get(2, {}).get("yaw_err_rad"),
        "pillar_contact_collision": bool(state["pillar_contact"]),
        "pillar_proximity_collision": bool(state["proximity_violation"]),
        "collision": bool(collision),
        "min_pillar_dist_m": (None if math.isinf(state["min_pillar_dist"])
                              else float(state["min_pillar_dist"])),
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
    }
    success = ((not fell) and (not collision) and mean_radial is not None
               and mean_radial <= RADIAL_ERR_SUCCESS)
    return success, metrics


def compute_squat_sweep_metrics(rec, state, schedule, trial):
    """T4: achieved_depth / depth_err on the bottom hold, root_drift_hold
    (max XY drift during the hold vs its first sample = "didn't squat
    steadily"), root_drift_total (XY at trial end vs pre-squat)."""
    z = np.asarray(rec["z"])
    x = np.asarray(rec["x"])
    y = np.asarray(rec["y"])
    phase = np.asarray(rec["phase"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    hold = phase == "hold"
    achieved = float(z[hold].mean()) if hold.any() else None
    if hold.any():
        hx, hy = float(x[hold][0]), float(y[hold][0])
        drift_hold = float(np.hypot(x[hold] - hx, y[hold] - hy).max())
    else:
        drift_hold = None
    down = phase == "down"
    final = phase == "final"
    if down.any() and final.any() and not fell:
        px, py = float(x[down][0]), float(y[down][0])
        drift_total = math.hypot(float(x[final][-1]) - px,
                                 float(y[final][-1]) - py)
    else:
        drift_total = None

    box_tracked = state["box_dist_max"] is not None
    box_kept = (state["box_drop_time"] is None) if box_tracked else None
    metrics = {
        "sweep": trial["sweep"],
        "target_height": trial["target_height"],
        "ramp_speed": trial["ramp_speed"],
        "init_yaw_rad": round(schedule.yaw0, 4),
        "achieved_depth": achieved,
        "depth_err": (achieved - trial["target_height"]
                      if achieved is not None else None),
        "root_drift_hold": drift_hold,
        "root_drift_total": drift_total,
        "max_tilt": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
        "box_kept": box_kept,
        "box_drop_time": state["box_drop_time"],
    }
    success = (not fell) and bool(box_kept)
    return success, metrics


def compute_squat_limit_metrics(rec, state, schedule):
    """T7: height-tracking-drift calibration curve + boundary scalars.

    descent_curve: 10 Hz [[h_cmd, base_z, drift_xy], ...] over the test
    segment (descent + floor hold); drift_xy is relative to the root XY at
    descent start. depth_floor = min base_z over STABLE samples (tilt <
    SQUAT_LIMIT_STABLE_TILT_RAD and dz/dt > SQUAT_LIMIT_STABLE_VZ -- the
    commanded sink rate is 0.05 m/s, so anything sinking 5x faster is a
    collapse, not a squat). track_sat_h = h_cmd when base_z first lags the
    command by > 5 cm (saturation onset). fall_h_cmd/fall_base_z = command /
    actual height at the fall sample (None when no fall)."""
    t = np.asarray(rec["t"])
    z = np.asarray(rec["z"])
    x = np.asarray(rec["x"])
    y = np.asarray(rec["y"])
    h = np.asarray(rec["h_target"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    seg = t >= schedule.descent_start
    curve = []
    depth_floor = track_sat_h = drift5_h = drift20_h = max_drift = None
    if seg.any():
        zs, hs, ts = z[seg], h[seg], tilt[seg]
        drift = np.hypot(x[seg] - x[seg][0], y[seg] - y[seg][0])
        curve = [[round(float(hs[i]), 4), round(float(zs[i]), 4),
                  round(float(drift[i]), 4)]
                 for i in range(0, len(zs), SQUAT_LIMIT_CURVE_STRIDE)]
        max_drift = float(drift.max())
        vz = np.gradient(zs, t[seg])
        stable = (ts < SQUAT_LIMIT_STABLE_TILT_RAD) \
            & (vz > SQUAT_LIMIT_STABLE_VZ)
        depth_floor = float(zs[stable].min()) if stable.any() else None
        sat = np.flatnonzero((zs - hs) > SQUAT_LIMIT_TRACK_SAT_M)
        track_sat_h = float(hs[sat[0]]) if len(sat) else None
        d5 = np.flatnonzero(drift > SQUAT_LIMIT_DRIFT_MARKS[0])
        d20 = np.flatnonzero(drift > SQUAT_LIMIT_DRIFT_MARKS[1])
        drift5_h = float(hs[d5[0]]) if len(d5) else None
        drift20_h = float(hs[d20[0]]) if len(d20) else None

    box_tracked = state["box_dist_max"] is not None
    metrics = {
        "init_yaw_rad": round(schedule.yaw0, 4),
        "fall_h_cmd": float(h[-1]) if fell and len(h) else None,
        "fall_base_z": float(z[-1]) if fell and len(z) else None,
        "depth_floor": depth_floor,
        "track_sat_h": track_sat_h,
        "drift5_h": drift5_h,
        "drift20_h": drift20_h,
        "max_drift_xy": max_drift,
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
        "box_kept": (state["box_drop_time"] is None) if box_tracked else None,
        "box_drop_time": state["box_drop_time"],
        "descent_curve": curve,
    }
    # spec v4: success := no fall (the 0.10 m command was reached and held;
    # tracking saturation is expected and recorded, not penalized).
    success = not fell
    return success, metrics


def compute_psi0_place_metrics(rec, state, schedule):
    """T8: lower-body error while ep053 drives the upper body. Segmented
    height RMSE (descent/hold_place/rise from the height_cmd profile),
    root_drift_place = max root XY drift over descent+hold (the forward
    reach + release shifts the CoM forward -- the decoupling stress test),
    box placement quality (static, upright, near, LANDED below
    PLACE_BOX_LAND_MAX_Z) and box_land_dx vs the release-time foot front
    edge. success := no fall AND box_place_ok."""
    t = np.asarray(rec["t"])
    z = np.asarray(rec["z"])
    x = np.asarray(rec["x"])
    y = np.asarray(rec["y"])
    h = np.asarray(rec["h_target"])
    tilt = np.asarray(rec["tilt"])
    phase = np.asarray(rec["phase"])
    fell = state["fall_time"] is not None

    err = z - h
    dmask = phase == "descent"
    hmask = phase == "hold_place"
    rmask = phase == "rise"
    pmask = dmask | hmask                    # reach-forward + place segment
    if pmask.any():
        px0, py0 = float(x[pmask][0]), float(y[pmask][0])
        root_drift_place = float(np.hypot(x[pmask] - px0,
                                          y[pmask] - py0).max())
    else:
        root_drift_place = None

    fb = schedule.final_box
    box_place_ok = bool(
        fb is not None
        and fb["speed"] < BOX_PLACE_MAX_SPEED
        and fb["tilt_from_upright_rad"] < BOX_PLACE_UPRIGHT_TOL
        and fb["dist_to_robot"] < BOX_PLACE_MAX_DIST
        and fb["pos"][2] < PLACE_BOX_LAND_MAX_Z)

    drop = state["box_drop_time"]
    release_t = schedule.release_t
    if state["box_dist_max"] is None:
        box_kept = None
    elif drop is None:
        box_kept = True
    else:  # after the release the box leaving the torso is expected
        box_kept = release_t is not None and drop >= release_t - 0.05
    stand_ok = bool((not fell) and (phase == "stand_final").any()
                    and schedule.finished)

    metrics = {
        "init_yaw_rad": round(schedule.yaw0, 4),
        "t_place_s": round(schedule.replay.t_place, 3),
        "release_t": release_t,
        "height_rmse_descent": _rmse(err[dmask]),
        "height_rmse_hold": _rmse(err[hmask]),
        "height_rmse_rise": _rmse(err[rmask]),
        "root_drift_place": root_drift_place,
        "max_tilt": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
        "box_kept_until_release": box_kept,
        "box_place_ok": box_place_ok,
        "box_land_dx": (fb or {}).get("land_dx"),
        "box_z_end": fb["pos"][2] if fb is not None else None,
        "box_speed_end": fb["speed"] if fb is not None else None,
        "box_tilt_end_rad": (fb["tilt_from_upright_rad"]
                             if fb is not None else None),
        "box_dist_to_robot_end": (fb["dist_to_robot"]
                                  if fb is not None else None),
        "stand_ok": stand_ok,
        "total_t": round(rec["t"][-1], 3) if rec["t"] else None,
    }
    success = (not fell) and box_place_ok
    return success, metrics


def compute_pick_ground_metrics(rec, state, schedule):
    """T9: ground-pick outcome. min_root_z = lowest base height from descent
    start (the actually-achieved depth under the unified 0.25 m command);
    root_drift = max XY drift over the bottom-hold + post-grasp-hold samples
    vs the first bottom sample (squat-bottom + grasp segment); rise_fall
    flags a fall during the loaded rise/final stand. success := pick AND
    no fall AND stand_ok (spec v5). Whether the grasp is even reachable is
    dominated by the squat-depth limit (R4 calibration: AMO 0.345 m)."""
    t = np.asarray(rec["t"])
    z = np.asarray(rec["z"])
    x = np.asarray(rec["x"])
    y = np.asarray(rec["y"])
    phase = np.asarray(rec["phase"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None

    pick = schedule.grasp_t is not None
    seg = t >= schedule.descend_start
    min_root_z = float(z[seg].min()) if seg.any() else None
    bottom = (phase == "bottom_hold") | (phase == "hold_grasp")
    if bottom.any():
        bx, by = float(x[bottom][0]), float(y[bottom][0])
        root_drift = float(np.hypot(x[bottom] - bx, y[bottom] - by).max())
    else:
        root_drift = None
    gd = schedule.grasp_dists or (None, None)
    stand_ok = bool((not fell) and schedule.finished)
    rise_fall = bool(fell and state["fall_phase"] in ("rise", "stand_final"))
    fb = schedule.final_box

    metrics = {
        "init_yaw_rad": round(schedule.yaw0, 4),
        "pick_success": pick,
        "pick_timeout": bool(schedule.pick_timeout),
        "grasp_t": schedule.grasp_t,
        "grasp_dist_left_m": gd[0],
        "grasp_dist_right_m": gd[1],
        "min_root_z": min_root_z,
        "root_drift": root_drift,
        "rise_fall": rise_fall,
        "stand_ok": stand_ok,
        "box_z_end": fb["pos"][2] if fb is not None else None,
        "box_dist_to_robot_end": (fb["dist_to_robot"]
                                  if fb is not None else None),
        "pick_arm_pose": [float(v) for v in PICK_ARM_POSE],
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
    }
    success = (not fell) and pick and stand_ok
    return success, metrics


def compute_vln_metrics(rec, state, schedule):
    """T10: pose tracking vs the tape's ideal integral reference.

    All poses are expressed in the TAPE FRAME = the robot's actual pose at
    tape start (anchor), so the ref (which starts at the origin) is directly
    comparable. final_* errors are taken at the LAST TAPE SAMPLE (end of the
    command stream; the 1 s final stand is excluded). mean_track_err = mean
    position error at the 1 Hz ref samples. small_cmd_response = mean of
    (actual forward speed / commanded vx) over tape steps whose HELD
    |vx_cmd| is inside VLN_SMALL_VX_RANGE. stop_settle = mean over full-stop
    tape segments of the max XY drift within the segment."""
    t = np.asarray(rec["t"])
    tilt = np.asarray(rec["tilt"])
    fell = state["fall_time"] is not None
    tape = schedule.tape
    anchor = schedule.anchor

    final_pos_err = final_yaw_err = mean_track_err = None
    small_resp = stop_settle = None
    if anchor is not None and len(t):
        ax, ay, ayaw = anchor
        ca, sa = math.cos(ayaw), math.sin(ayaw)
        dx = np.asarray(rec["x"]) - ax
        dy = np.asarray(rec["y"]) - ay
        px = ca * dx + sa * dy
        py = -sa * dx + ca * dy
        pyaw = np.asarray(rec["yaw"]) - ayaw

        def idx_at(tau):
            i = int(round((SETTLE_T + tau) / REPLAY_DT))
            return i if 0 <= i < len(t) else None

        track_errs = []
        for row in tape.ref:
            i = idx_at(float(row[0]))
            if i is not None:
                track_errs.append(math.hypot(px[i] - float(row[1]),
                                             py[i] - float(row[2])))
        mean_track_err = float(np.mean(track_errs)) if track_errs else None

        i_end = idx_at(tape.duration)
        if i_end is not None and not fell:
            ref_end = tape.ref[-1]
            final_pos_err = math.hypot(px[i_end] - float(ref_end[1]),
                                       py[i_end] - float(ref_end[2]))
            final_yaw_err = abs(wrap_angle(float(pyaw[i_end])
                                           - float(ref_end[3])))

        cmd = np.asarray(rec["cmd_vx"])
        vxf = np.asarray(rec["vx_fwd"])
        in_tape = (t >= SETTLE_T) & (t < SETTLE_T + tape.duration)
        small = (in_tape & (np.abs(cmd) >= VLN_SMALL_VX_RANGE[0])
                 & (np.abs(cmd) <= VLN_SMALL_VX_RANGE[1]))
        if small.any():
            small_resp = float(np.mean(vxf[small] / cmd[small]))
        drifts = []
        for s0, s1 in tape.stops:
            # half-open [s0, s1): the sample where the next non-zero command
            # resumes is excluded from the stop-residual measurement
            m = (t >= SETTLE_T + s0) & (t < SETTLE_T + s1)
            if m.sum() >= 2:
                sx, sy = float(px[m][0]), float(py[m][0])
                drifts.append(float(np.hypot(px[m] - sx, py[m] - sy).max()))
        stop_settle = float(np.mean(drifts)) if drifts else None

    metrics = {
        "tape_id": tape.id,
        "tape_duration_s": tape.duration,
        "final_pos_err": final_pos_err,
        "final_yaw_err": final_yaw_err,
        "mean_track_err": mean_track_err,
        "small_cmd_response": small_resp,
        "stop_settle": stop_settle,
        "n_stop_segments": len(tape.stops),
        "vln_yaw_locked_frac": (schedule.locked_steps / schedule.tape_steps
                                if schedule.tape_steps else None),
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
    }
    success = ((not fell) and final_pos_err is not None
               and final_pos_err <= VLN_FINAL_POS_TOL
               and final_yaw_err is not None
               and final_yaw_err <= VLN_FINAL_YAW_TOL)
    return success, metrics


def _nav_metrics(nav, rec, state):
    """Shared T5/T6 navigation metric block from a _NavCalController."""
    res = nav.result
    tilt = np.asarray(rec["tilt"])
    en, ec = res.get("err_pos_nav"), res.get("err_pos_cal")
    yn, yc = res.get("err_yaw_nav"), res.get("err_yaw_cal")
    return {
        "err_pos_nav": en,
        "err_yaw_nav": yn,
        "err_pos_cal": ec,
        "err_yaw_cal": yc,
        "cal_improve_pos": (en - ec if en is not None and ec is not None
                            else None),
        "cal_improve_yaw": (yn - yc if yn is not None and yc is not None
                            else None),
        "t_nav": res.get("t_nav"),
        "t_cal": res.get("t_cal"),
        "success_coarse": res.get("success_coarse"),
        "success_fine": res.get("success_fine"),
        "cal_yaw_locked_frac": res.get("cal_yaw_locked_frac"),
        "max_tilt": float(tilt.max()) if len(tilt) else None,
        "fall_reason": state["fall_reason"],
    }


def compute_goto_metrics(rec, state, schedule):
    """T5: the project's direct measurement of whether small-command
    calibration converges. Headline success = success_fine and no fall."""
    fell = state["fall_time"] is not None
    metrics = _nav_metrics(schedule.nav, rec, state)
    metrics["init_yaw_rad"] = round(schedule.yaw0, 4)
    success = (not fell) and bool(metrics["success_fine"])
    return success, metrics


def compute_pipeline_metrics(rec, state, schedule):
    """T6: nav metrics + squat/place/rise outcome + box placement quality.
    success := success_coarse AND no fall AND box_place_ok (fine convergence
    recorded but not gating, per spec)."""
    fell = state["fall_time"] is not None
    metrics = _nav_metrics(schedule.nav, rec, state)

    squat_phases = ("squat_down", "bottom_hold", "place_hold", "rise",
                    "stand_final")
    metrics["squat_fall"] = (state["fall_phase"]
                             if fell and state["fall_phase"] in squat_phases
                             else None)

    drop = state["box_drop_time"]
    release_t = schedule.release_t
    if state["box_dist_max"] is None:
        box_kept = None
    elif drop is None:
        box_kept = True
    else:  # after the release the box leaving the torso is expected
        box_kept = release_t is not None and drop >= release_t - 0.05
    metrics["box_kept_until_release"] = box_kept
    metrics["release_t"] = release_t

    fb = schedule.final_box
    box_place_ok = bool(
        fb is not None
        and fb["speed"] < BOX_PLACE_MAX_SPEED
        and fb["tilt_from_upright_rad"] < BOX_PLACE_UPRIGHT_TOL
        and fb["dist_to_robot"] < BOX_PLACE_MAX_DIST)
    metrics["box_place_ok"] = box_place_ok
    if fb is not None:
        metrics["box_land_pos"] = [fb["pos"][0] - PIPE_C_XY[0],
                                   fb["pos"][1] - PIPE_C_XY[1]]
        metrics["box_land_dist_from_c"] = math.hypot(*metrics["box_land_pos"])
        metrics["box_speed_end"] = fb["speed"]
        metrics["box_tilt_end_rad"] = fb["tilt_from_upright_rad"]
        metrics["box_dist_to_robot_end"] = fb["dist_to_robot"]
    else:
        metrics["box_land_pos"] = None
        metrics["box_land_dist_from_c"] = None
    metrics["total_t"] = round(rec["t"][-1], 3) if rec["t"] else None

    success = (bool(metrics["success_coarse"]) and (not fell)
               and box_place_ok)
    return success, metrics


# ---------------------------------------------------------------------------
# Video: zero render cost during sim — qpos snapshots are replayed through a
# fresh MjData + mujoco.Renderer ONLY for trials whose video we keep.
# ---------------------------------------------------------------------------


def write_video(model, qpos_frames, path):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[bench_amo] WARN imageio not installed "
              "(pip install 'imageio[ffmpeg]'); skipping %s" % path,
              flush=True)
        return False
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 135.0
    base_q, _ = _pelvis_qpos_adr(model)   # box (when present) precedes pelvis
    renderer = mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    writer = imageio.get_writer(path, fps=VIDEO_FPS)
    try:
        for qpos in qpos_frames:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.qpos[base_q:base_q + 3]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    return True


# ---------------------------------------------------------------------------
# Trial list / orchestration
# ---------------------------------------------------------------------------


def build_trials(test, n):
    trials = []
    if test in ("walk_speed", "walk_speed_psi0"):
        for i in range(n):
            trials.append({"index": i, "seed": i, "target_vx": 1.0})
    elif test == "speed_sweep":
        per_speed = max(1, n // len(SWEEP_SPEEDS))
        i = 0
        for v in SWEEP_SPEEDS:
            for _ in range(per_speed):
                trials.append({"index": i, "seed": i, "target_vx": v})
                i += 1
    elif test in ("squat_box", "squat_box_psi0", "squat_limit",
                  "squat_place_psi0"):
        for i in range(n):
            trials.append({"index": i, "seed": i})
    elif test == "squat_sweep":
        # n = trials PER GRID POINT: 8 depths @ 0.2 m/s + 4 rates @ 0.45 m
        i = 0
        for h in SQUAT_SWEEP_HEIGHTS:
            for _ in range(n):
                trials.append({"index": i, "seed": i, "sweep": "depth",
                               "target_height": h,
                               "ramp_speed": SQUAT_SWEEP_BASE_RATE})
                i += 1
        for v in SQUAT_SWEEP_RATES:
            for _ in range(n):
                trials.append({"index": i, "seed": i, "sweep": "speed",
                               "target_height": SQUAT_SWEEP_FIXED_H,
                               "ramp_speed": v})
                i += 1
    elif test in ("goto_ab", "pipeline_abc"):
        for i in range(n):
            trials.append({"index": i, "seed": i})
    elif test in ("squat_pick_ground", "vln_follow"):
        # vln_follow: tape i % n_tapes is resolved in run_one_trial.
        for i in range(n):
            trials.append({"index": i, "seed": i})
    elif test in ("circle_pillar", "circle_pillar_psi0"):
        n_ccw = (n + 1) // 2
        for i in range(n):
            trials.append({"index": i, "seed": i,
                           "direction": "ccw" if i < n_ccw else "cw"})
    return trials


def make_schedule(test, trial, init_yaw, box_spec=None, replay=None,
                  tape=None):
    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        return WalkSchedule(init_yaw, trial["target_vx"])
    if test == "squat_box":
        return SquatSchedule(init_yaw)
    if test == "squat_box_psi0":
        return Psi0SquatSchedule(replay)
    if test == "squat_limit":
        return SquatLimitSchedule(init_yaw)
    if test == "squat_place_psi0":
        return Psi0PlaceSchedule(init_yaw, replay, box_spec)
    if test == "squat_pick_ground":
        return PickGroundSchedule(init_yaw, box_spec)
    if test == "vln_follow":
        return VlnFollowSchedule(tape)
    if test == "squat_sweep":
        return SquatSweepSchedule(init_yaw, trial["target_height"],
                                  trial["ramp_speed"])
    if test == "goto_ab":
        return GotoSchedule(init_yaw)
    if test == "pipeline_abc":
        return PipelineSchedule(init_yaw, box_spec)
    return CircleSchedule(1.0 if trial["direction"] == "ccw" else -1.0)


def _reset_released_box_collision(model):
    """T6/T8 trials mutate the cached model at release (box geom contype/
    conaffinity 0 -> 2); undo it before the next trial / geom-set scan."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bench_box_geom")
    if gid < 0:
        raise RuntimeError("bench_box_geom not found in release-test model")
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0
    if hasattr(model, "body_contype"):
        bid = int(model.geom_bodyid[gid])
        model.body_contype[bid] = 0
        model.body_conaffinity[bid] = 0


def run_one_trial(test, trial, policy_jit, capture_frames, replay=None,
                  tapes=None):
    """Returns (result_dict, qpos_frames, model)."""
    np.random.seed(trial["seed"])  # also covers play_amo's global np.random

    pillar_y = None
    pillar_center = None
    if test in ("circle_pillar", "circle_pillar_psi0"):
        init_yaw = 0.0  # spec: no random yaw for the circle test
        pillar_y = CIRCLE_RADIUS if trial["direction"] == "ccw" else -CIRCLE_RADIUS
        pillar_center = (0.0, pillar_y)
    elif test == "goto_ab":
        # spec v3: A = origin, heading 0 + seed-determined +-0.3 rad noise
        init_yaw = float(np.random.uniform(-GOTO_INIT_YAW_NOISE,
                                           GOTO_INIT_YAW_NOISE))
    elif test in ("pipeline_abc", "squat_box_psi0"):
        # pipeline: spec v3 A = origin, heading 0.
        # squat_box_psi0: BendPick table scene alignment (yaw 0).
        init_yaw = 0.0
    elif test == "vln_follow":
        # tape ref frame starts at yaw 0; the schedule re-anchors on the
        # ACTUAL pose at tape start anyway (settle drift removed).
        init_yaw = 0.0
    else:
        init_yaw = float(np.random.uniform(-math.pi, math.pi))

    tape = None
    if test == "vln_follow":
        tape = tapes[trial["index"] % len(tapes)]   # spec: tape i % n_tapes

    upper_drive = None
    grasp_close_t = None
    if test in PSI0_TESTS:
        model, box_spec = get_psi0_model(test, replay, pillar_y)
        upper_drive = UpperReplayDrive(
            replay,
            squat_mode=(test in ("squat_box_psi0", "squat_place_psi0")))
        if test == "squat_box_psi0":
            grasp_close_t = replay.grasp_close_t
    elif test == "squat_pick_ground":
        model, box_spec = get_pick_model()
    else:
        model, box_spec = get_model(with_box=(test in BOX_TESTS),
                                    pillar_y=pillar_y)
    if test in ("pipeline_abc", "squat_place_psi0"):
        _reset_released_box_collision(model)  # cached model: undo release
    geom_sets = build_geom_sets(model)
    env = build_env(policy_jit, model, "cuda")
    if test in PSI0_TESTS:
        apply_initial_state(env, init_yaw, box_spec,
                            upper_init=_psi0_upper_frame0(replay),
                            box_follows_pelvis=(test != "squat_box_psi0"))
    else:
        # squat_pick_ground included: the keyframe holds the box upright on
        # the ground 0.45 m ahead at yaw 0; box_follows_pelvis rotates that
        # pelvis-frame offset with the trial yaw (z stays on the floor:
        # pelvis z is the keyframe value at reset).
        apply_initial_state(env, init_yaw, box_spec)
    schedule = make_schedule(test, trial, init_yaw, box_spec, replay, tape)

    rec, state, frames = run_trial(env, schedule, geom_sets, capture_frames,
                                   pillar_center, box_spec,
                                   upper_drive=upper_drive,
                                   grasp_close_t=grasp_close_t)

    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        success, metrics = compute_walk_metrics(rec, state, schedule)
    elif test == "squat_box":
        success, metrics = compute_squat_metrics(rec, state, schedule)
    elif test == "squat_box_psi0":
        success, metrics = compute_psi0_squat_metrics(rec, state, schedule)
    elif test == "squat_limit":
        success, metrics = compute_squat_limit_metrics(rec, state, schedule)
    elif test == "squat_place_psi0":
        success, metrics = compute_psi0_place_metrics(rec, state, schedule)
    elif test == "squat_pick_ground":
        success, metrics = compute_pick_ground_metrics(rec, state, schedule)
    elif test == "vln_follow":
        success, metrics = compute_vln_metrics(rec, state, schedule)
    elif test == "squat_sweep":
        success, metrics = compute_squat_sweep_metrics(rec, state, schedule,
                                                       trial)
    elif test == "goto_ab":
        success, metrics = compute_goto_metrics(rec, state, schedule)
    elif test == "pipeline_abc":
        success, metrics = compute_pipeline_metrics(rec, state, schedule)
    else:
        success, metrics = compute_circle_metrics(rec, state, schedule,
                                                  pillar_center,
                                                  trial["direction"])
    if test in ("walk_speed_psi0", "circle_pillar_psi0"):
        success, metrics = psi0_box_metrics(success, metrics, state)

    result = {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial["index"],
        "seed": trial["seed"],
        "success": bool(success),
        "fall_time": state["fall_time"],
        "fall_phase": state["fall_phase"],
        "metrics": metrics,
    }
    if test in BOX_TESTS or test in PSI0_TESTS:
        result["box_mode"] = BOX_MODE
    if test == "squat_pick_ground":
        result["box_mode"] = PICK_BOX_MODE
    if test in PSI0_TESTS:
        result["upper_mode"] = "psi0_replay_%s" % replay.source
    return result, frames, model


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def aggregate_metric_stats(results):
    keys = set()
    for r in results:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    stats = {}
    for k in sorted(keys):
        vals = []
        for r in results:
            v = r["metrics"].get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and math.isfinite(v):
                vals.append(float(v))
        if vals:
            stats[k] = {"mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals)}
    return stats


def sweep_breakdown(results):
    table = {}
    for v in SWEEP_SPEEDS:
        rs = [r for r in results
              if abs(r["metrics"].get("target_vx", -1) - v) < 1e-9]
        if not rs:
            continue
        n_ok = sum(1 for r in rs if r["success"])
        means = [r["metrics"]["mean_vx_last8s"] for r in rs
                 if r["metrics"]["mean_vx_last8s"] is not None]
        table["%.1f" % v] = {
            "n": len(rs),
            "n_success": n_ok,
            "mean_vx_last8s": float(np.mean(means)) if means else None,
            "all_stable": n_ok == len(rs),
        }
    stable = [float(k) for k, row in table.items() if row["all_stable"]]
    return table, (max(stable) if stable else None)


def _group_stats(rs):
    """Per-group squat_sweep stats: directly answers 'which heights fall,
    how far does the root drift'."""
    n = len(rs)
    n_ok = sum(1 for r in rs if r["success"])

    def m(key):
        vals = [r["metrics"].get(key) for r in rs]
        vals = [float(v) for v in vals
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    fall_phases = {}
    for r in rs:
        if r["fall_time"] is not None:
            ph = r["fall_phase"] or "unknown"
            fall_phases[ph] = fall_phases.get(ph, 0) + 1
    return {
        "n": n,
        "n_success": n_ok,
        "success_rate": (n_ok / n) if n else None,
        "n_falls": sum(1 for r in rs if r["fall_time"] is not None),
        "fall_phases": fall_phases,
        "mean_achieved_depth": m("achieved_depth"),
        "mean_root_drift_hold": m("root_drift_hold"),
        "mean_root_drift_total": m("root_drift_total"),
    }


def squat_sweep_breakdown(results):
    by_height, by_speed = {}, {}
    for h in SQUAT_SWEEP_HEIGHTS:
        rs = [r for r in results if r["metrics"].get("sweep") == "depth"
              and r["metrics"].get("target_height") == h]
        if rs:
            by_height["%.2f" % h] = _group_stats(rs)
    for v in SQUAT_SWEEP_RATES:
        rs = [r for r in results if r["metrics"].get("sweep") == "speed"
              and r["metrics"].get("ramp_speed") == v]
        if rs:
            by_speed["%.1f" % v] = _group_stats(rs)
    return by_height, by_speed


# ===========================================================================
# ManipArena v2 (BENCHMARK_V2_DESIGN §2-§4). AMO adapter — mirrors the HOMIE
# reference adapter (bench_homie.py §run_arena_*). Key AMO specifics:
#   - box/cube free bodies are injected BEFORE pelvis (play_amo reads joints by
#     NEGATIVE index qpos[-23:]); all base reads are pelvis-adr aware.
#   - height command is a DELTA on 0.75 (commands[3] = h - 0.75).
#   - no wz channel: the mission's wz is integrated into the ABSOLUTE heading
#     target commands[1]; the in-place-stand interlock (|vx|<0.1 freezes dyaw)
#     is recorded as-is.
#   - upper body driven via the existing pd_target overwrite path
#     (_apply_upper_replay style): the IO stores arm8+waist3 each tick.
# ===========================================================================
def _require_arena():
    if ms is None or mm is None:
        raise SystemExit(
            "[bench_amo] arena tests need manip_scene.py + manip_mission.py "
            "next to bench_amo.py (scp them into ~/AMO): %r" % _ARENA_IMPORT_ERR)


def arena_reach_base_h(top_h):
    """AMO absolute base-height target to bring the wrists to a surface at
    height ``top_h``. Linear "lower surface -> lower base" with a fixed reach
    offset, clipped to AMO's trackable domain [ARENA_H_MIN, 0.75]. The mission
    emits this as its squat target; AMO's depth floor (~0.345 m) decides whether
    the low tables are reachable at all (the point of the H_pick sweep)."""
    return float(min(max(top_h + ARENA_REACH_OFFSET, ARENA_H_MIN),
                     ARENA_STAND_H))


# Static furniture goes after the ground geom (no freejoint); the two free
# bodies (box, cube) are injected BEFORE pelvis so their qpos lands ahead of the
# robot's negative-indexed joints:
#   qpos = [ box free (7) | cube free (7) | pelvis free (7) | 23 hinges ]
#   qvel = [ box free (6) | cube free (6) | pelvis free (6) | 23 hinges ]
# play_amo reads qpos[-23:]/qvel[-23:] -> robot joints untouched.
def build_arena_model(spec):
    """Compile g1.xml with the ManipArena furniture + free bodies injected
    (AMO layout: free bodies before pelvis, static furniture after ground)."""
    _require_arena()
    with open("g1.xml", "r") as f:
        xml = f.read()
    d = spec.to_dict()
    tp, tr, zs, pl = d["T_pick"], d["T_relay"], d["Z_store"], d["pillar"]
    bx, cb = d["box"], d["cube"]

    # --- static furniture (after the ground geom) ---
    tpick = ms._table_xml("table_pick", tp["pos"][0], tp["pos"][1], tp["top_h"],
                          tp["size"][0] / 2.0, tp["size"][1] / 2.0, tp["yaw"],
                          "0.55 0.42 0.26 1", ms.FURNITURE_CT_CA)
    trelay = ms._table_xml("table_relay", tr["pos"][0], tr["pos"][1],
                           tr["top_h"], tr["size"][0] / 2.0, tr["size"][1] / 2.0,
                           0.0, "0.42 0.32 0.20 1", ms.FURNITURE_CT_CA)
    zstore = ms._zstore_xml(zs["pos"][0], zs["pos"][1], zs["size"][0] / 2.0,
                            zs["size"][1] / 2.0, zs["rim_h"], ms.FURNITURE_CT_CA)
    pillar = ms._pillar_xml(pl["pos"][0], pl["pos"][1], pl["r"], pl["h"],
                            ms.FURNITURE_CT_CA)
    static_xml = tpick + trelay + zstore + pillar
    if _GROUND_ANCHOR not in xml:
        raise RuntimeError("g1.xml ground anchor for arena furniture not found")
    xml = xml.replace(_GROUND_ANCHOR, _GROUND_ANCHOR + "\n" + static_xml, 1)

    # --- two free bodies (box then cube) before pelvis ---
    box_xml = ms._free_box_xml(
        "carried_box", "carried_box_geom",
        (bx["pos"][0], bx["pos"][1], bx["pos"][2]),
        ARENA_BOX_HALF, bx["mass"], "1.0 0.85 0.1 1", ms.FREE_BODY_CT_CA)
    cube_xml = ms._free_box_xml(
        "relay_cube", "relay_cube_geom",
        (cb["pos"][0], cb["pos"][1], cb["pos"][2]),
        ARENA_CUBE_HALF, cb["mass"], "0.15 0.7 0.25 1", ms.FREE_BODY_CT_CA)
    if _WORLDBODY_OPEN_ANCHOR not in xml:
        raise RuntimeError("g1.xml worldbody anchor for arena free bodies "
                           "not found")
    xml = xml.replace(_WORLDBODY_OPEN_ANCHOR,
                      _WORLDBODY_OPEN_ANCHOR + box_xml + cube_xml, 1)

    # --- weld block (predeclared, inactive) before </mujoco>; sub wrist names ---
    weld_block = ms._weld_block()
    weld_block = weld_block.replace(ms.WRIST_L_PLACEHOLDER, ARENA_HAND_BODIES[0])
    weld_block = weld_block.replace(ms.WRIST_R_PLACEHOLDER, ARENA_HAND_BODIES[1])
    xml = xml.replace(_MUJOCO_CLOSE_ANCHOR, weld_block + _MUJOCO_CLOSE_ANCHOR, 1)

    # --- extend keyframe 'home' qpos: box(7)+cube(7) prefix the 30-value home ---
    box_kf = "%s 1 0 0 0" % _fmt_vec(bx["pos"])
    cube_kf = "%s 1 0 0 0" % _fmt_vec(cb["pos"])
    kf_prefix = 'qpos="%s %s 0 0 1.0 1 0 0 0' % (box_kf, cube_kf)
    if _KEYFRAME_QPOS_ANCHOR not in xml:
        raise RuntimeError("g1.xml keyframe anchor for arena qpos not found")
    xml = xml.replace(_KEYFRAME_QPOS_ANCHOR, kf_prefix, 1)

    model = _load_model_from_text(xml)
    return model


def configure_arena_welds(model):
    """Resolve the arena free-body addresses + weld equality ids. The welds are
    re-anchored at the live wrist<->box pose at grasp time, so here we only look
    up ids and confirm the box/cube freejoints precede the pelvis (AMO layout)."""
    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError("arena joint missing: %s" % name)
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    box_qadr, box_vadr = jq("carried_box_freejoint")
    cube_qadr, cube_vadr = jq("relay_cube_freejoint")
    base_q, base_v = _pelvis_qpos_adr(model)
    # AMO layout: box at 0, cube at 7, pelvis at 14.
    assert box_qadr == 0 and cube_qadr == 7 and base_q == 14, \
        ("arena AMO free-body layout unexpected: box=%d cube=%d pelvis=%d "
         "(want 0/7/14)" % (box_qadr, cube_qadr, base_q))

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def eid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    return {
        "box_qadr": box_qadr, "box_vadr": box_vadr,
        "cube_qadr": cube_qadr, "cube_vadr": cube_vadr,
        "base_qpos_adr": base_q, "base_qvel_adr": base_v,
        "box_body": bid("carried_box"),
        "cube_body": bid("relay_cube"),
        "wrist_left": bid(ARENA_HAND_BODIES[0]),
        "wrist_right": bid(ARENA_HAND_BODIES[1]),
        "box_geom": gid("carried_box_geom"),
        "cube_geom": gid("relay_cube_geom"),
        "torso_body_id": bid("torso_link"),
        "eq_box_left": eid("box_weld_left"),
        "eq_box_right": eid("box_weld_right"),
        "eq_cube_box": eid("cube_weld_box"),
        "eq_cube_right": eid("cube_weld_right"),
    }


def _arena_floor_bit2(m, body_id):
    """OR bit 2 into a free body + the floor plane(s) + world body so the body
    collides with the ground at release (AMO ground conaffinity=15 already
    includes bit 2; the box body compiles bit-2 so this is mostly the body bit)."""
    if hasattr(m, "body_contype"):
        m.body_contype[body_id] |= 2
        m.body_conaffinity[body_id] |= 2
        m.body_contype[0] |= 2
        m.body_conaffinity[0] |= 2
    for g in range(m.ngeom):
        if (m.geom_bodyid[g] == 0
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE):
            m.geom_contype[g] |= 2
            m.geom_conaffinity[g] |= 2


def _arena_weld_at_pose(m, d, eq, body1, body2):
    """Activate weld ``eq`` anchored at the CURRENT relative body2-in-body1 pose
    (no body moved). Same math as _activate_welds_at_current_pose, generalised."""
    rot1 = d.xmat[body1].reshape(3, 3)
    relp = rot1.T @ (d.xpos[body2] - d.xpos[body1])
    q1inv = np.zeros(4)
    relq = np.zeros(4)
    mujoco.mju_negQuat(q1inv, d.xquat[body1])
    mujoco.mju_mulQuat(relq, q1inv, d.xquat[body2])
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = relp
    m.eq_data[eq, 6:10] = relq
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def _arena_snap_box_upright(m, d, box, base_yaw):
    """ROOT CAUSE 2 fix (HOMIE): before welding, move the box into a canonical
    UPRIGHT pose — position = midpoint of the two hands, orientation = pure yaw
    locked to the base (box z == world z). Makes the two wrist<->box welds anchor
    at mutually-consistent relposes and the box rides upright so it lands level
    on release. Mutates d, then mj_forward so the welds anchor against it."""
    bq = box["box_qadr"]
    bv = box["box_vadr"]
    mid = 0.5 * (d.xpos[box["wrist_left"]] + d.xpos[box["wrist_right"]])
    d.qpos[bq:bq + 3] = mid
    d.qpos[bq + 3:bq + 7] = [math.cos(base_yaw / 2.0), 0.0, 0.0,
                             math.sin(base_yaw / 2.0)]
    d.qvel[bv:bv + 6] = 0.0
    mujoco.mj_forward(m, d)


def _arena_weld_box_both(m, d, box):
    """Activate BOTH wrist<->box welds, anchored at the canonical upright pose
    (see _arena_snap_box_upright). Shares the 2 kg load across both arms; solref/
    solimp left at the compliant XML defaults."""
    _arena_weld_at_pose(m, d, box["eq_box_right"], box["wrist_right"],
                        box["box_body"])
    _arena_weld_at_pose(m, d, box["eq_box_left"], box["wrist_left"],
                        box["box_body"])


def _arena_box_surface_dist(d, box, hand_body_id, half_extents):
    """Distance [m] from a hand body origin to the oriented box surface (0 if
    inside)."""
    bid = box["box_body"] if half_extents is ARENA_BOX_HALF else box["cube_body"]
    half = np.asarray(half_extents, dtype=np.float64)
    rel = d.xpos[hand_body_id] - d.xpos[bid]
    local = d.xmat[bid].reshape(3, 3).T @ rel
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


class AmoMissionIO:
    """MissionIO (manip_mission Protocol) bound to a live AMO env. The mission
    owns sequencing + per-stage metrics; this adapter owns the physics side
    effects against env.model/env.data.

    Pose reads are pelvis-adr aware (box+cube precede pelvis in AMO qpos). The
    leg command (vx, vy, wz, height) is stored each tick; the ArenaSchedule
    converts it to AMO's commands[] (wz -> absolute heading slew on commands[1],
    height -> commands[3] = h - 0.75). The upper body is driven by writing the
    latest (arm8, waist3) target the schedule applies via the pd_target hook."""

    def __init__(self, env, box, replay, h_stand=ARENA_STAND_H):
        self.env = env
        self.m = env.model
        self.d = env.data
        self.box = box
        self.replay = replay        # prepared real arena replay dict
        self.h_stand = h_stand
        self.base_q = box["base_qpos_adr"]
        self.base_v = box["base_qvel_adr"]
        # mutable slots the schedule reads:
        self.cmd = (0.0, 0.0, 0.0, h_stand)         # vx, vy, wz, height (abs)
        # current upper target = (arm8, waist3) in AMO dof order:
        if replay is not None:
            self.upper_target = self._frame_upper(0)
        else:
            self.upper_target = (env.default_dof_pos[15:].copy(),
                                 env.default_dof_pos[12:15].copy())
        self.box_grasped = False
        self.cube_on_box = False
        self._fallen = False

    # --- rollout-facing helpers (not part of MissionIO) ------------------
    def set_command(self, vx, vy, wz, height):
        self.cmd = (float(vx), float(vy), float(wz), float(height))

    def set_fallen(self, flag):
        self._fallen = bool(flag)

    def _frame_upper(self, k):
        rep = self.replay
        k = min(max(int(k), 0), rep["n"] - 1)
        row = rep["pos"][k]
        return (row[rep["arm_cols"]].copy(), row[rep["waist_cols"]].copy())

    # --- MissionIO: read state -------------------------------------------
    def root_pose(self):
        q = self.d.qpos
        b = self.base_q
        yaw = float(play_amo.quatToEuler(q[b + 3:b + 7])[2])
        return (float(q[b]), float(q[b + 1]), yaw)

    def root_tilt(self):
        b = self.base_q
        rpy = play_amo.quatToEuler(self.d.qpos[b + 3:b + 7])
        # gravity-xy magnitude == sqrt(1 - (cos r cos p)^2); tilt scalar the
        # mission's fall gate (>0.9) consumes.
        cz = math.cos(rpy[0]) * math.cos(rpy[1])
        return float(math.sqrt(max(0.0, 1.0 - cz * cz)))

    def wrist_box_dist(self):
        d_l = _arena_box_surface_dist(self.d, self.box, self.box["wrist_left"],
                                      ARENA_BOX_HALF)
        d_r = _arena_box_surface_dist(self.d, self.box, self.box["wrist_right"],
                                      ARENA_BOX_HALF)
        return (d_l, d_r)

    def right_wrist_cube_dist(self):
        return _arena_box_surface_dist(self.d, self.box, self.box["wrist_right"],
                                       ARENA_CUBE_HALF)

    def box_pose(self):
        b = self.box["box_body"]
        bx, by, bz = (float(v) for v in self.d.xpos[b])
        zz = float(self.d.xmat[b].reshape(3, 3)[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        speed = float(np.linalg.norm(
            self.d.qvel[self.box["box_vadr"]:self.box["box_vadr"] + 3]))
        return (bx, by, bz, tilt, speed)

    def base_height(self):
        return float(self.d.qpos[self.base_q + 2])

    def is_fallen(self):
        return self._fallen

    # --- MissionIO: act (lower body) -------------------------------------
    def send_leg_cmd(self, vx, vy, wz, height):
        self.set_command(vx, vy, wz, height)

    # --- MissionIO: act (upper body, replay) -----------------------------
    def replay_upper(self, segment, phase_t):
        rep = self.replay
        if rep is None:
            return
        seg = rep.get("arena_segments", {}).get(segment)
        if seg is None:
            self.upper_target = self._frame_upper(0)
            return
        k0, k1 = seg
        n = max(k1 - k0, 1)
        k = k0 + int(min(max(phase_t / REPLAY_DT, 0.0), n - 1))
        self.upper_target = self._frame_upper(k)

    # --- MissionIO: act (weld / release / cube-transfer) -----------------
    def weld_box(self):
        b = self.base_q
        base_yaw = float(play_amo.quatToEuler(self.d.qpos[b + 3:b + 7])[2])
        _arena_snap_box_upright(self.m, self.d, self.box, base_yaw)
        _arena_weld_box_both(self.m, self.d, self.box)
        # while HELD the box collides with NOTHING (it is snapped to the wrist
        # midpoint, which during a deep reach can overlap the pick table; leaving
        # it collidable spikes qacc on stand-up).
        self.m.geom_contype[self.box["box_geom"]] = 0
        self.m.geom_conaffinity[self.box["box_geom"]] = 0
        if hasattr(self.m, "body_contype"):
            self.m.body_contype[self.box["box_body"]] = 0
            self.m.body_conaffinity[self.box["box_body"]] = 0
        self.box_grasped = True

    def release_box(self, surface_top_h=None, target_xy=None):
        bq = self.box["box_qadr"]
        bv = self.box["box_vadr"]
        b = self.base_q
        base_yaw = float(play_amo.quatToEuler(self.d.qpos[b + 3:b + 7])[2])
        px, py = float(self.d.qpos[b]), float(self.d.qpos[b + 1])
        if target_xy is not None:
            dx, dy = target_xy[0] - px, target_xy[1] - py
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dist, dy / dist
            reach = min(ARENA_RELEASE_FWD_M, dist)
            bx = px + reach * ux
            by = py + reach * uy
            rem = math.hypot(target_xy[0] - bx, target_xy[1] - by)
            if rem > ARENA_PLACE_CLAMP_R:
                k = (rem - ARENA_PLACE_CLAMP_R) / rem
                bx += (target_xy[0] - bx) * k
                by += (target_xy[1] - by) * k
        else:
            bx = px + ARENA_RELEASE_FWD_M * math.cos(base_yaw)
            by = py + ARENA_RELEASE_FWD_M * math.sin(base_yaw)
        self.d.qpos[bq] = bx
        self.d.qpos[bq + 1] = by
        self.d.qpos[bq + 3:bq + 7] = [math.cos(base_yaw / 2.0), 0.0, 0.0,
                                      math.sin(base_yaw / 2.0)]
        if surface_top_h is not None:
            self.d.qpos[bq + 2] = surface_top_h + ARENA_BOX_HALF[2] + 0.002
        self.d.qvel[bv:bv + 6] = 0.0
        self.d.eq_active[self.box["eq_box_left"]] = 0
        self.d.eq_active[self.box["eq_box_right"]] = 0
        self.m.geom_contype[self.box["box_geom"]] = ms.FREE_BODY_CT_CA
        self.m.geom_conaffinity[self.box["box_geom"]] = ms.FREE_BODY_CT_CA
        _arena_floor_bit2(self.m, self.box["box_body"])
        mujoco.mj_forward(self.m, self.d)
        self.box_grasped = False

    def transfer_cube(self):
        _arena_weld_at_pose(self.m, self.d, self.box["eq_cube_box"],
                            self.box["box_body"], self.box["cube_body"])
        self.m.geom_contype[self.box["cube_geom"]] = ms.FREE_BODY_CT_CA
        self.m.geom_conaffinity[self.box["cube_geom"]] = ms.FREE_BODY_CT_CA
        self.cube_on_box = True


def prepare_arena_segments(rep):
    """Label the real arena replay into reach_down / carry / place windows.
    Mirrors bench_homie.prepare_arena_segments but on the AMO replay namespace
    (rep is a dict here). reach_down = descent toward the surface, place = the
    set-down + retract, carry = a short standing-tall window."""
    h = rep["height"]
    n = rep["n"]
    k_low = int(np.argmin(h))
    hmin = float(h[k_low])
    first_hold = k_low
    while first_hold > 0 and float(h[first_hold - 1]) <= hmin + PLACE_HOLD_TOL_M:
        first_hold -= 1
    reach = (0, max(first_hold, 1))
    place = (first_hold, n)
    carry = (max(n - max(1, n // 10), 0), n)
    out = dict(rep)
    out["arena_segments"] = {"reach_down": reach, "place": place, "carry": carry}
    out["seg_bottom_k"] = k_low
    return out


def _arena_load_replay_dict(path):
    """Load the arena upper replay into a plain dict (reuses load_upper_replay's
    validation + amo23 column mapping). Arena uses the real_ep053 stream."""
    rep = load_upper_replay(path)
    return {
        "path": rep.path, "names": rep.names, "pos": rep.pos,
        "height": rep.height, "carry_pose": rep.carry_pose,
        "n": rep.n, "duration_s": rep.duration_s,
        "source": rep.source, "embodiment": rep.embodiment,
        "arm_cols": rep.arm_cols, "waist_cols": rep.waist_cols,
    }


# ---------------------------------------------------------------------------
# Pillar-avoidance: insert a lateral skirt waypoint into a nav leg whose
# straight segment passes too close to the pillar (harness-side rewrite of the
# mission's NavLeg list; the mission engine stays physics-free).
# ---------------------------------------------------------------------------
def _arena_seg_point_dist(p0, p1, c):
    ax, ay = p0
    bx, by = p1
    cx, cy = c
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(cx - ax, cy - ay)
    tp = ((cx - ax) * dx + (cy - ay) * dy) / denom
    tp = min(max(tp, 0.0), 1.0)
    px, py = ax + tp * dx, ay + tp * dy
    return math.hypot(cx - px, cy - py)


def insert_pillar_detours(sequence, spawn_xy, pillar_xy,
                          clear_m=ARENA_PILLAR_CLEAR_M,
                          side_m=ARENA_PILLAR_SIDE_M):
    """Return a NEW primitive list where any NavLeg whose straight path passes
    within ``clear_m`` of the pillar is preceded by a side-offset skirt NavLeg."""
    out = []
    prev_xy = tuple(spawn_xy)
    for prim in sequence:
        if isinstance(prim, mm.NavLeg):
            tgt_xy = (prim.target_pose[0], prim.target_pose[1])
            dist = _arena_seg_point_dist(prev_xy, tgt_xy, pillar_xy)
            if dist < clear_m and not getattr(prim, "skip_detour", False):
                mx = 0.5 * (prev_xy[0] + tgt_xy[0])
                my = 0.5 * (prev_xy[1] + tgt_xy[1])
                sx, sy = tgt_xy[0] - prev_xy[0], tgt_xy[1] - prev_xy[1]
                slen = math.hypot(sx, sy) or 1.0
                nx, ny = -sy / slen, sx / slen
                if ((mx + nx - pillar_xy[0]) ** 2 + (my + ny - pillar_xy[1]) ** 2) \
                        < ((mx - nx - pillar_xy[0]) ** 2
                           + (my - ny - pillar_xy[1]) ** 2):
                    nx, ny = -nx, -ny
                wp = (mx + nx * side_m, my + ny * side_m,
                      math.atan2(tgt_xy[1] - (my + ny * side_m),
                                 tgt_xy[0] - (mx + nx * side_m)))
                out.append(mm.NavLeg(wp, vx_max=prim.vx_max, mode="coarse_only"))
            out.append(prim)
            prev_xy = tgt_xy
        else:
            out.append(prim)
    return out


# ---------------------------------------------------------------------------
# Arena schedules. Each .apply(env, t) writes env.viewer.commands and returns a
# phase label; the M-series schedule ticks the shared Mission, the L-series
# schedules reuse the existing command_at/Nav logic against the arena scene.
# wz -> commands[1] integration (AMO has no wz channel): cmd_yaw is slewed by
# wz*dt each step, with the in-place-stand interlock (|vx|<0.1) recorded.
# ---------------------------------------------------------------------------
class ArenaMissionSchedule:
    """M1/M2: drive the shared Mission. Per control step it reads the IO (which
    is bound to the live env), fires welds/release/transfer + replay_upper as
    side effects, and converts the returned (vx, vy, wz, height) into AMO
    commands. Heading: NAV/coarse uses the body-frame wz integrated onto an
    absolute commands[1]; height -> commands[3] = h - 0.75."""

    def __init__(self, mission, io, init_yaw):
        self.mission = mission
        self.io = io
        self.yaw0 = init_yaw
        self.cmd_yaw = init_yaw
        self.finished = False
        self.locked_steps = 0
        self.total_steps = 0
        # generous wall budget; the mission has its own MISSION_TIMEOUT_S.
        self.total_t = mm.MISSION_TIMEOUT_S + 5.0

    def apply(self, env, t):
        c = env.viewer.commands
        c[7] = 0.0
        # mission tick (reads io, fires side effects, returns the leg command)
        vx, vy, wz, h = self.mission.step(t)
        self.io.set_command(vx, vy, wz, h)
        # heading: integrate the body-frame wz onto the absolute target. AMO
        # freezes dyaw when |vx|<0.1 (in-place stand), so a pure-turn command
        # is mostly dead — recorded as locked_frac, not worked around.
        self.cmd_yaw = wrap_angle(self.cmd_yaw + wz * env.control_dt)
        c[0] = vx
        c[2] = vy
        c[1] = self.cmd_yaw
        c[3] = h - BASE_HEIGHT_DEFAULT
        self.total_steps += 1
        if abs(vx) < AMO_STAND_VX_THRESH:
            self.locked_steps += 1
        if self.mission.finished:
            self.finished = True
        return self.mission.stages[-1].name if self.mission.stages else "init"


class ArenaWalkSchedule:
    """L1 arena_L1: settle 2s -> ramp vx -> hold, heading toward the lane (+x).
    No box welds; the box/cube sit on their tables as fixed background."""

    def __init__(self, init_yaw, target_vx=ARENA_L1_VX):
        self.yaw0 = init_yaw
        self.target_vx = target_vx
        self.total_t = SETTLE_T + ARENA_L1_RAMP_T + ARENA_L1_HOLD_T

    def apply(self, env, t):
        c = env.viewer.commands
        c[1] = self.yaw0
        c[2] = 0.0
        c[3] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[0] = 0.0
            return "settle"
        if t < SETTLE_T + ARENA_L1_RAMP_T:
            c[0] = self.target_vx * (t - SETTLE_T) / ARENA_L1_RAMP_T
            return "ramp"
        c[0] = self.target_vx
        return "hold"


class ArenaCircleSchedule:
    """L2 arena_L2: settle 2s -> circle the SCENE pillar at radius 1.0 m,
    vx=0.4, heading integrated at wz=+0.4 rad/s (ccw). Uses the scene pillar
    position from the SceneSpec; closures tracked like circle_pillar."""

    def __init__(self, init_yaw, pillar_xy):
        self.yaw0 = init_yaw
        self.pillar_xy = pillar_xy
        self.circle_t = 2.0 * math.pi / ARENA_L2_WZ
        self.total_t = SETTLE_T + ARENA_L2_LAPS * self.circle_t + 0.5

    def apply(self, env, t):
        c = env.viewer.commands
        c[2] = 0.0
        c[3] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[0] = 0.0
            c[1] = self.yaw0
            return "settle"
        c[0] = ARENA_L2_VX
        c[1] = self.yaw0 + ARENA_L2_WZ * (t - SETTLE_T)
        return "circle1" if (t - SETTLE_T) < self.circle_t else "circle2"

    def cmd_yaw_progress(self, t):
        return ARENA_L2_WZ * max(0.0, t - SETTLE_T)


class ArenaGotoSchedule:
    """L3 arena_L3: settle 2s -> two-phase nav to P_pick_front -> stand 1s.
    Reuses _NavCalController (the shared nav math) against the scene target."""

    def __init__(self, init_yaw, target_pose, base_qpos_adr):
        self.yaw0 = init_yaw
        self.nav = _NavCalController(tuple(target_pose[:2]), float(target_pose[2]),
                                     NAV_VX_MAX, base_qpos_adr=base_qpos_adr)
        self.end_t0 = None
        self.finished = False
        self.total_t = (SETTLE_T + NAV_TIMEOUT_T + CAL_TIMEOUT_T
                        + GOTO_END_T + 2.0)

    def apply(self, env, t):
        c = env.viewer.commands
        c[3] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[0] = 0.0
            c[2] = 0.0
            c[1] = self.yaw0
            return "settle"
        if not self.nav.done:
            return self.nav.step(env, t)
        if self.end_t0 is None:
            self.end_t0 = t
        c[0] = 0.0
        c[2] = 0.0
        c[1] = self.nav.goal_yaw
        if t - self.end_t0 >= GOTO_END_T:
            self.finished = True
        return "end"


class ArenaVlnSchedule:
    """L4 arena_L4: tape following inside the arena. Same realisation as the
    standalone VlnFollowSchedule (vx->commands[0], vy->commands[2], wz
    integrated onto the absolute heading commands[1]), but pelvis-adr aware so
    the anchor reads the robot base (box+cube precede pelvis in AMO qpos)."""

    def __init__(self, tape, base_qpos_adr):
        self.tape = tape
        self.base_q = base_qpos_adr
        self.yaw0 = 0.0
        self.tape_end = SETTLE_T + tape.duration
        self.total_t = self.tape_end + VLN_FINAL_STAND_T + 1.0
        self.anchor = None
        self.cmd_yaw = None
        self.tape_steps = 0
        self.locked_steps = 0
        self.finished = False

    def apply(self, env, t):
        c = env.viewer.commands
        b = self.base_q
        c[3] = 0.0
        c[7] = 0.0
        if t < SETTLE_T:
            c[0] = 0.0
            c[2] = 0.0
            c[1] = self.yaw0
            return "settle"
        if self.anchor is None:
            q = env.data.qpos
            yaw = float(play_amo.quatToEuler(q[b + 3:b + 7])[2])
            self.anchor = (float(q[b]), float(q[b + 1]), yaw)
            self.cmd_yaw = yaw
        tau = t - SETTLE_T
        if tau < self.tape.duration:
            vx, vy, wz = _tape_cmd_at(self.tape, tau)
            c[0] = vx
            c[2] = vy
            self.cmd_yaw += wz * env.control_dt
            c[1] = self.cmd_yaw
            self.tape_steps += 1
            if abs(vx) < AMO_STAND_VX_THRESH:
                self.locked_steps += 1
            return "follow"
        c[0] = 0.0
        c[2] = 0.0
        c[1] = self.cmd_yaw
        if t >= self.tape_end + VLN_FINAL_STAND_T:
            self.finished = True
        return "stand_final"


class ArenaSquatLimitSchedule:
    """L5 arena_L5: 2 s settle (box wrist-welded from t=0, arm blend) -> height
    cmd ramps 0.75 -> 0.10 m at 0.05 m/s -> hold 3 s (no rise). Same as the
    standalone squat_limit but with the box welded via the arena IO at t=0."""

    def __init__(self, init_yaw, io):
        self.yaw0 = init_yaw
        self.io = io
        self.descent_start = SETTLE_T
        self.descent_t = ((BASE_HEIGHT_DEFAULT - ARENA_L5_FLOOR_H)
                          / ARENA_L5_RATE)
        self.total_t = SETTLE_T + self.descent_t + ARENA_L5_HOLD_T
        self._welded = False

    def apply(self, env, t):
        c = env.viewer.commands
        c[0] = 0.0
        c[1] = self.yaw0
        c[2] = 0.0
        c[7] = 0.0
        # weld the box at the first tick (after mj_forward in apply_initial_state)
        if not self._welded:
            self.io.weld_box()
            self._welded = True
        # carry pose (reach_down frame 0 ~ standing carry) on the upper body
        self.io.replay_upper("carry", 0.0)
        if t < SETTLE_T:
            c[3] = 0.0
            return "settle"
        tc = t - self.descent_start
        if tc < self.descent_t:
            c[3] = -ARENA_L5_RATE * tc
            return "descend"
        c[3] = ARENA_L5_FLOOR_H - BASE_HEIGHT_DEFAULT
        return "hold_floor"


# ---------------------------------------------------------------------------
# Arena trial runner. Mirrors run_trial's hook/fall machinery but mission- or
# L-schedule-driven, with the arena IO applying the upper-body target each tick.
# ---------------------------------------------------------------------------
def _arena_apply_upper(env, run_frame, arm, waist):
    """Overwrite this control step's upper-body pd_target from the arena IO's
    current (arm8, waist3) target (same mechanism as _apply_upper_replay)."""
    pd_target = run_frame.f_locals.get("pd_target")
    if not isinstance(pd_target, np.ndarray) or pd_target.shape != (23,):
        raise RuntimeError("arena upper: pd_target not found in play_amo run() "
                           "frame (play_amo.py changed upstream?)")
    env.prev_arm_action = arm.copy()
    env.arm_action = arm.copy()
    env.arm_blend = 1.0
    pd_target[15:23] = arm
    pd_target[12:15] = waist


def run_arena_trial(env, schedule, io, box, geom_sets, capture_frames,
                    drive_upper=True):
    """Drive one arena trial via env.run() + the per-control-step hook.
    Returns (rec, state, frames)."""
    rec = {k: [] for k in ("t", "phase", "x", "y", "z", "yaw", "tilt")}
    state = {"step": 0, "fall_time": None, "fall_phase": None,
             "fall_reason": None, "harness_error": None}
    frames = []
    ctrl_dt = env.control_dt
    base_q = box["base_qpos_adr"]

    def hook():
        i = state["step"]
        state["step"] = i + 1
        t = i * ctrl_dt
        d = env.data

        # fall flag for THIS tick BEFORE the mission ticks (it consumes it)
        rpy = play_amo.quatToEuler(d.qpos[base_q + 3:base_q + 7])
        tilt = math.acos(max(-1.0, min(1.0,
                                       math.cos(rpy[0]) * math.cos(rpy[1]))))
        nonfoot_ground, _pillar = scan_contacts(d, geom_sets)
        gxy = io.root_tilt()
        fall_now = (gxy > FALL_TILT_RAD) or nonfoot_ground
        io.set_fallen(fall_now)

        phase = schedule.apply(env, t)

        if drive_upper:
            arm, waist = io.upper_target
            _arena_apply_upper(env, sys._getframe(1), arm, waist)

        x = float(d.qpos[base_q + 0])
        y = float(d.qpos[base_q + 1])
        z = float(d.qpos[base_q + 2])
        rec["t"].append(t)
        rec["phase"].append(phase)
        rec["x"].append(x)
        rec["y"].append(y)
        rec["z"].append(z)
        rec["yaw"].append(float(rpy[2]))
        rec["tilt"].append(tilt)

        # solver divergence (weld-vs-PD blowup) -> harness_error, not a fall
        if _solver_diverged(d):
            state["harness_error"] = "solver_divergence_t%.2f" % t
            if capture_frames:
                frames.append(d.qpos.copy())
            raise TrialAbort()

        if capture_frames and (i % VIDEO_EVERY_N_CTRL == 0):
            frames.append(d.qpos.copy())

        if fall_now:
            state["fall_time"] = round(t, 3)
            state["fall_phase"] = phase
            state["fall_reason"] = ("tilt" if gxy > FALL_TILT_RAD
                                    else "non_foot_ground_contact")
            if capture_frames:
                frames.append(d.qpos.copy())
            raise TrialAbort()

        if getattr(schedule, "finished", False):
            if capture_frames:
                frames.append(d.qpos.copy())
            raise TrialAbort()

    env.viewer.render = hook
    env.sim_duration = schedule.total_t
    try:
        env.run()
    except TrialAbort:
        pass
    except Exception as exc:  # noqa: BLE001
        state["harness_error"] = "exception: %r" % exc
    return rec, state, frames


def _arena_apply_initial_state(env, spec, box, replay):
    """Reset to keyframe home, spawn at the spec pose (+0.02 rad joint noise),
    place box/cube at their spec ground-truth poses, welds OFF, upper body at
    replay frame 0."""
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    base = box["base_qpos_adr"]
    nd = env.num_dofs
    noise = np.random.uniform(-INIT_JOINT_NOISE, INIT_JOINT_NOISE, nd)
    env.data.qpos[base + 7:base + 7 + nd] = (
        env.data.qpos[base + 7:base + 7 + nd] + noise)
    sd = spec.to_dict()
    spawn = sd["spawn"]
    yaw0 = float(spawn[2])
    yaw_quat = np.array([math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)])
    env.data.qpos[base + 0] = spawn[0]
    env.data.qpos[base + 1] = spawn[1]
    env.data.qpos[base + 3:base + 7] = yaw_quat
    # upper body at replay frame 0 (noise removed there so PD targets don't snap)
    if replay is not None:
        arm0 = replay["pos"][0][replay["arm_cols"]]
        waist0 = replay["pos"][0][replay["waist_cols"]]
        hinge = base + 7
        env.data.qpos[hinge + 12:hinge + 15] = waist0
        env.data.qpos[hinge + 15:hinge + 23] = arm0
    # free bodies at the spec ground-truth poses, welds inactive
    bq, cq = box["box_qadr"], box["cube_qadr"]
    env.data.qpos[bq:bq + 3] = sd["box"]["pos"]
    env.data.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[cq:cq + 3] = sd["cube"]["pos"]
    env.data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    for eq in ("eq_box_left", "eq_box_right", "eq_cube_box", "eq_cube_right"):
        env.data.eq_active[box[eq]] = 0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _arena_reset_free_body_collision(model, box):
    """Undo any release-time collision-bit writes so the next trial starts with
    the box/cube as bit-2 free bodies resting on the furniture."""
    for body, geom in ((box["box_body"], box["box_geom"]),
                       (box["cube_body"], box["cube_geom"])):
        model.geom_contype[geom] = ms.FREE_BODY_CT_CA
        model.geom_conaffinity[geom] = ms.FREE_BODY_CT_CA
        if hasattr(model, "body_contype"):
            model.body_contype[body] = ms.FREE_BODY_CT_CA
            model.body_conaffinity[body] = ms.FREE_BODY_CT_CA


def _arena_stage_flags(stages):
    """Flatten the per-stage records into M-series diagnostic flags."""
    flags = {}
    grasp_seen = 0
    place_seen = 0
    for s in stages:
        kind = s.get("kind")
        if kind == "nav":
            err = s.get("cal_err_pos", s.get("coarse_err_pos"))
            flags["nav_arrival_err_last"] = err
            flags.setdefault("nav_arrival_errs", []).append(err)
        elif kind == "grasp":
            grasp_seen += 1
            ok = bool(s.get("grasp_ok"))
            flags["grasp_ok" if grasp_seen == 1 else "regrasp_ok"] = ok
        elif kind == "place":
            place_seen += 1
            ok = bool(s.get("place_ok"))
            nm = s.get("name", "")
            if "relay" in nm:
                flags["place_ok_relay"] = ok
            elif "store" in nm:
                flags["place_ok_store"] = ok
            else:
                flags["place_ok_%d" % place_seen] = ok
        elif kind == "cube":
            flags["cube_transfer_ok"] = bool(s.get("cube_transfer_ok"))
    return flags


def _arena_build_geom_sets(model, box):
    """Geom sets for fall detection in the arena scene. Only the bit-1 ROBOT
    geoms (pelvis_contour + 8 foot spheres) count; furniture (bits 1|2) and the
    free box/cube (bit 2) are excluded so resting/standing on the floor or
    bumping a table is not mis-read as a fall. Mirrors build_geom_sets but uses
    the arena geom names and a body-id exclusion for furniture/free bodies."""
    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    ground = gid("ground")
    # exclude every geom that belongs to furniture or a free body.
    exclude_bodies = set()
    for nm in ("carried_box", "relay_cube", "table_pick", "table_relay",
               "zstore", "pillar"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            exclude_bodies.add(bid)
    robot, feet = set(), set()
    for g in range(model.ngeom):
        if g == ground:
            continue
        if int(model.geom_bodyid[g]) in exclude_bodies:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual-only meshes (arms/torso/knees)
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[g])) or ""
        robot.add(g)
        if "ankle_roll" in body_name:
            feet.add(g)
    # pillar contact handled separately via the proximity check; keep -1 here so
    # scan_contacts treats the pillar like any furniture (no pillar_hit gating).
    return {"ground": ground, "pillar": -1, "robot": robot, "feet": feet}


def run_arena_m_trial(test, spec, replay, policy_jit, capture_frames):
    """M1/M2 one trial. Returns (result, frames, model)."""
    model = build_arena_model(spec)
    assert model.nu == 23, "arena AMO model nu=%d != 23" % model.nu
    box = configure_arena_welds(model)
    _arena_reset_free_body_collision(model, box)
    geom_sets = _arena_build_geom_sets(model, box)
    env = build_env(policy_jit, model, "cuda")
    _arena_apply_initial_state(env, spec, box, replay)

    io = AmoMissionIO(env, box, replay, h_stand=ARENA_STAND_H)
    sd = spec.to_dict()
    h_pick = arena_reach_base_h(sd["T_pick"]["top_h"])
    h_relay = arena_reach_base_h(sd["T_relay"]["top_h"])
    h_store = ARENA_STORE_PLACE_H
    h_cube = arena_reach_base_h(sd["T_relay"]["top_h"])
    if test == "arena_M1":
        seq = mm.build_m1(sd, ARENA_STAND_H, h_pick, h_store,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    else:
        seq = mm.build_m2(sd, ARENA_STAND_H, h_pick, h_relay, h_store, h_cube,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    seq = insert_pillar_detours(seq, (sd["spawn"][0], sd["spawn"][1]),
                                tuple(sd["pillar"]["pos"]))
    mission = mm.Mission(test, seq, io, ARENA_STAND_H)
    schedule = ArenaMissionSchedule(mission, io, float(sd["spawn"][2]))

    rec, state, frames = run_arena_trial(env, schedule, io, box, geom_sets,
                                         capture_frames, drive_upper=True)

    mres = mission.result()
    tilt = np.asarray(rec["tilt"])
    metrics = {
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "mission_status": mres["status"],
        "mission_success": mres["success"],
        "n_stages": mres["n_stages"],
        "completed_stages": sum(1 for s in mres["stages"]
                                if s.get("ok") is True),
        "fall_stage": mres["fall_stage"],
        "fall_t": mres["fall_t"],
        "stages": mres["stages"],
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "cal_yaw_locked_frac": (schedule.locked_steps / schedule.total_steps
                                if schedule.total_steps else None),
        "sim_time_s": round(rec["t"][-1] if rec["t"] else 0.0, 3),
        "fall_reason": state["fall_reason"],
        "harness_error": state["harness_error"],
    }
    metrics.update(_arena_stage_flags(mres["stages"]))
    success = bool(mres["success"] and state["fall_time"] is None
                   and state["harness_error"] is None)
    result = {
        "framework": FRAMEWORK, "test": test, "trial": spec.variant_seed,
        "seed": spec.variant_seed, "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick, "scene_spec_hash": spec.scene_spec_hash,
        "success": success,
        "fall_time": state["fall_time"], "fall_phase": state["fall_phase"],
        "box_mode": "arena_wrist_weld",
        "upper_mode": "psi0_replay_%s" % (replay["source"] if replay else "none"),
        "metrics": metrics,
    }
    return result, frames, model


def run_arena_l_trial(test, spec, replay, trial, policy_jit, capture_frames,
                      tape=None):
    """L1-L5 one trial. Returns (result, frames, model). The furniture is a
    fixed background; the box is only welded for L5 (squat-limit carry)."""
    model = build_arena_model(spec)
    box = configure_arena_welds(model)
    _arena_reset_free_body_collision(model, box)
    geom_sets = _arena_build_geom_sets(model, box)
    env = build_env(policy_jit, model, "cuda")
    _arena_apply_initial_state(env, spec, box, replay)
    sd = spec.to_dict()
    yaw0 = float(sd["spawn"][2])
    base_q = box["base_qpos_adr"]
    io = AmoMissionIO(env, box, replay, h_stand=ARENA_STAND_H)
    pillar_center = tuple(sd["pillar"]["pos"])

    drive_upper = False
    if test == "arena_L1":
        schedule = ArenaWalkSchedule(yaw0, ARENA_L1_VX)
    elif test == "arena_L2":
        schedule = ArenaCircleSchedule(yaw0, pillar_center)
    elif test == "arena_L3":
        schedule = ArenaGotoSchedule(yaw0, sd["targets"]["P_pick_front"], base_q)
    elif test == "arena_L4":
        schedule = ArenaVlnSchedule(tape, base_q)
    elif test == "arena_L5":
        schedule = ArenaSquatLimitSchedule(yaw0, io)
        drive_upper = True
    else:
        raise ValueError(test)

    rec, state, frames = run_arena_trial(env, schedule, io, box, geom_sets,
                                         capture_frames, drive_upper=drive_upper)

    tilt = np.asarray(rec["tilt"])
    x = np.asarray(rec["x"])
    y = np.asarray(rec["y"])
    z = np.asarray(rec["z"])
    tt = np.asarray(rec["t"])
    fell = state["fall_time"] is not None
    metrics = {
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "max_tilt_rad": float(tilt.max()) if len(tilt) else None,
        "sim_time_s": round(tt[-1] if len(tt) else 0.0, 3),
        "fall_reason": state["fall_reason"],
        "harness_error": state["harness_error"],
    }
    # per-L diagnostics + success
    if test == "arena_L1":
        seg = tt >= SETTLE_T
        last = tt >= (schedule.total_t - 8.0)
        metrics["mean_vx_last8s"] = None
        if last.any() and len(x) > 1:
            vx_fwd = np.gradient(x, tt)
            metrics["mean_vx_last8s"] = float(vx_fwd[last].mean())
        metrics["dist_traveled_m"] = (float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
                                      if len(x) else None)
        success = not fell
    elif test == "arena_L2":
        seg = tt >= SETTLE_T
        radial = np.abs(np.hypot(x - pillar_center[0], y - pillar_center[1])
                        - ARENA_L2_RADIUS)[seg]
        metrics["radial_err_mean_m"] = float(radial.mean()) if len(radial) else None
        metrics["radial_err_max_m"] = float(radial.max()) if len(radial) else None
        success = (not fell and metrics["radial_err_mean_m"] is not None
                   and metrics["radial_err_mean_m"] <= RADIAL_ERR_SUCCESS)
    elif test == "arena_L3":
        res = schedule.nav.result
        metrics.update({
            "err_pos_nav": res.get("err_pos_nav"),
            "err_yaw_nav": res.get("err_yaw_nav"),
            "success_coarse": res.get("success_coarse"),
            "err_pos_cal": res.get("err_pos_cal"),
            "err_yaw_cal": res.get("err_yaw_cal"),
            "success_fine": res.get("success_fine"),
            "cal_yaw_locked_frac": res.get("cal_yaw_locked_frac"),
        })
        success = not fell and bool(res.get("success_coarse"))
    elif test == "arena_L4":
        # reuse the standalone vln metric scaffolding fields where available
        metrics["final_x"] = float(x[-1]) if len(x) else None
        metrics["final_y"] = float(y[-1]) if len(y) else None
        success = not fell
    else:  # arena_L5
        seg = tt >= SETTLE_T
        if seg.any():
            metrics["min_base_z"] = float(z[seg].min())
            metrics["final_base_z"] = float(z[-1])
        success = not fell
    result = {
        "framework": FRAMEWORK, "test": test, "trial": trial["index"],
        "seed": spec.variant_seed, "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick, "scene_spec_hash": spec.scene_spec_hash,
        "success": bool(success),
        "fall_time": state["fall_time"], "fall_phase": state["fall_phase"],
        "metrics": metrics,
    }
    if test == "arena_L5":
        result["box_mode"] = "arena_wrist_weld"
    return result, frames, model


def run_arena_main(args, policy_jit):
    """ManipArena entry. --variant selects the scene; --trials N re-runs the
    SAME variant N times (default 1). Writes amo_<test>.jsonl + summary.json +
    videos."""
    _require_arena()
    test = args.test
    if not (0 <= args.variant <= 49):
        sys.stderr.write("[bench_amo] --variant must be in [0, 49] (got %d)\n"
                         % args.variant)
        return 2
    replay = None
    if test in ("arena_M1", "arena_M2", "arena_L5"):
        if args.upper_replay is None:
            sys.stderr.write("[bench_amo] --upper-replay required for %s "
                             "(real arena upper_replay_amo23_real_ep053.npz)\n"
                             % test)
            return 2
        replay = _arena_load_replay_dict(args.upper_replay)
        if replay["embodiment"] not in ("amo23", "unknown"):
            sys.stderr.write("[bench_amo] arena replay embodiment %r != amo23\n"
                             % replay["embodiment"])
            return 2
        replay = prepare_arena_segments(replay)
        print("[bench_amo] arena replay source=%s N=%d (%.2fs) segments "
              "reach_down=%s carry=%s place=%s"
              % (replay["source"], replay["n"], replay["duration_s"],
                 replay["arena_segments"]["reach_down"],
                 replay["arena_segments"]["carry"],
                 replay["arena_segments"]["place"]), flush=True)

    tape = None
    if test == "arena_L4":
        if args.tapes is None:
            sys.stderr.write("[bench_amo] --tapes required for arena_L4\n")
            return 2
        tapes = load_vln_tapes(args.tapes)
        tape = tapes[args.variant % len(tapes)]

    spec = ms.build_variant(args.variant)
    print("[bench_amo] arena variant_seed=%d H_pick=%.2f hash=%s"
          % (spec.variant_seed, spec.H_pick, spec.scene_spec_hash[:12]),
          flush=True)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "amo_%s.jsonl" % test)
    n = args.trials if args.trials is not None else 1
    capture_frames = args.video != "none"
    max_ok = MAX_SUCCESS_VIDEOS
    results = []
    ok_videos = 0
    with open(jsonl_path, "w") as jf:
        for i in range(n):
            np.random.seed(i)
            trial = {"index": i, "seed": spec.variant_seed}
            t0 = time.time()
            try:
                if test in ARENA_M_TESTS:
                    result, frames, model = run_arena_m_trial(
                        test, spec, replay, policy_jit, capture_frames)
                else:
                    result, frames, model = run_arena_l_trial(
                        test, spec, replay, trial, policy_jit,
                        capture_frames, tape=tape)
            except Exception as exc:  # keep the batch alive
                result = {
                    "framework": FRAMEWORK, "test": test, "trial": i,
                    "seed": spec.variant_seed, "variant_seed": spec.variant_seed,
                    "H_pick": spec.H_pick, "success": False,
                    "fall_time": None, "fall_phase": None,
                    "metrics": {"fall_reason": "harness_error",
                                "error": repr(exc)},
                }
                frames, model = [], None
                print("[bench_amo] arena trial %d ERROR: %r" % (i, exc),
                      flush=True)
            results.append(result)
            jf.write(json.dumps(to_jsonable(result)) + "\n")
            jf.flush()
            keep = False
            if model is not None and frames:
                if args.video == "all":
                    keep = True
                elif args.video == "policy":
                    keep = (not result["success"]) or (ok_videos < max_ok)
            if keep:
                tag = "ok" if result["success"] else "fail"
                vid = os.path.join(out_dir, "%s_%s_v%02d_t%02d_%s.mp4"
                                   % (FRAMEWORK, test, args.variant, i, tag))
                if write_video(model, frames, vid) and result["success"]:
                    ok_videos += 1
            mtr = result["metrics"]
            print("[bench_amo] [%s] v%02d trial %02d -> %s%s%s (status=%s, %ss)"
                  % (test, args.variant, i,
                     "OK" if result["success"] else "FAIL",
                     "" if not result["fall_time"] else
                     " fall@%.1fs(%s)" % (result["fall_time"],
                                          result.get("fall_phase")),
                     "" if not mtr.get("harness_error") else
                     " HARNESS_ERR(%s)" % mtr["harness_error"],
                     mtr.get("mission_status", "n/a"),
                     round(mtr.get("sim_time_s", 0.0), 1)), flush=True)

    summary = {
        "framework": FRAMEWORK, "test": test,
        "variant_seed": spec.variant_seed, "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "scene_spec": spec.to_dict(),
        "n_trials": len(results),
        "success_rate": (float(np.mean([r["success"] for r in results]))
                         if results else None),
        "results": [to_jsonable(r["metrics"]) for r in results],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                merged = json.load(f)
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged[test] = to_jsonable(summary)
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)
    print("[bench_amo] arena DONE success_rate=%s results=%s summary=%s"
          % (summary["success_rate"], jsonl_path, summary_path), flush=True)
    return 0


def summarize(test, results, args):
    n = len(results)
    summary = {
        "framework": FRAMEWORK,
        "test": test,
        "n_trials": n,
        "success_rate": (sum(1 for r in results if r["success"]) / n
                         if n else None),
        "n_falls": sum(1 for r in results if r["fall_time"] is not None),
        "video_mode": args.video,
        "metric_stats": aggregate_metric_stats(results),
        "notes": [
            "AMO height cmd is a delta on 0.75 m base height; squat 0.45 m "
            "=> commands[3]=-0.30.",
            "Robot collision geoms are ONLY pelvis_contour + 8 foot spheres; "
            "arm/torso/knee contacts (pillar & ground) are NOT detectable. "
            "Pillar collision adds a radial-proximity criterion "
            "(base-to-pillar-center < %.2f m)." % PROXIMITY_COLLISION_DIST,
            "vx commands above the trained domain [-0.5,0.5] are sent "
            "unclipped on purpose (spec).",
            "squat_box box mode %s: 2 kg free box held between the hands by "
            "two soft equality welds (solref %s) on left/right_rubber_hand; "
            "load goes through the arms (sag/oscillation is real physics). "
            "success additionally requires box_kept (|box-torso| < %.1f m "
            "throughout); weld/PD solver divergence is recorded as "
            "harness_error." % (BOX_MODE, BOX_WELD_SOLREF, BOX_KEEP_DIST),
        ],
    }
    if test in BOX_TESTS:
        summary["box_mode"] = BOX_MODE
    if test in PSI0_TESTS:
        summary["box_mode"] = BOX_MODE
        modes = sorted({r.get("upper_mode") for r in results
                        if r.get("upper_mode")})
        summary["upper_mode"] = modes[0] if modes else None
        summary["upper_replay"] = getattr(args, "upper_replay", None)
        kept_key = ("box_kept_until_release" if test == "squat_place_psi0"
                    else "box_kept")
        summary["n_box_drops"] = sum(
            1 for r in results if r["metrics"].get(kept_key) is False)
        summary["notes"].append(
            "psi0 upper replay (contract v1, amo23 K=11): arm 8 pd targets "
            "driven per control step through play_amo's arm machinery "
            "(arm_blend=1 direct drive) AND written onto the live pd_target; "
            "waist 3 are part of the policy action and are overwritten on "
            "pd_target after the policy ran, before PD torque (SIMPLE hack, "
            "precedent SIMPLE g1_wholebody.py:271).")
        if test != "squat_place_psi0":
            summary["notes"].append(
                "Cracker box 0.072x0.164x0.213 m / %.3f kg with collision "
                "bit 2 (table 1|2, ground conaffinity 15): collides with "
                "table/ground, never with the robot." % PSI0_BOX_MASS)
    if test == "squat_box_psi0":
        picks = [r["metrics"].get("pick_success") for r in results
                 if isinstance(r["metrics"].get("pick_success"), bool)]
        summary["pick_success_rate"] = (float(np.mean(picks)) if picks
                                        else None)
        summary["notes"].append(
            "squat_box_psi0: BendPick table scene (ep035; tabletop 0.40 m, "
            "pelvis ~0.29 m from the near edge); replay height_cmd clipped "
            "to [%.2f, %.2f] -> commands[3] = h - 0.75; at grasp_close_t "
            "both rubber hands must be < %.2f m from the box surface to "
            "activate the welds in place (box not moved), else pick_failed; "
            "afterwards carry_pose freeze + standard 20x5 s squat cycles."
            % (PSI0_HEIGHT_CLIP[0], PSI0_HEIGHT_CLIP[1], PSI0_GRASP_DIST))
    if test == "squat_limit":
        floors = [r["metrics"].get("depth_floor") for r in results]
        floors = [f for f in floors if isinstance(f, (int, float))
                  and math.isfinite(f)]
        summary["depth_floor_best"] = min(floors) if floors else None
        summary["n_track_saturated"] = sum(
            1 for r in results
            if r["metrics"].get("track_sat_h") is not None)
        summary["notes"].append(
            "squat_limit: height command ramps 0.75 -> 0.10 m at a constant "
            "%.2f m/s then holds %.0f s (no rise); commands[3] reaches "
            "%.2f, far below the trained height domain, and is sent "
            "UNCLIPPED -- play_amo's command layer has no clip (commands[3] "
            "feeds the adapter/obs raw, play_amo.py:248,282) and this "
            "harness adds none. descent_curve = 10 Hz [h_cmd, base_z, "
            "drift_xy]; depth_floor = min base_z over stable samples "
            "(tilt < %.1f rad, dz/dt > %.2f m/s -- collapse excluded); "
            "success = no fall (tracking saturation expected, recorded "
            "via track_sat_h)."
            % (SQUAT_LIMIT_RATE, SQUAT_LIMIT_HOLD_T,
               SQUAT_LIMIT_FLOOR_H - BASE_HEIGHT_DEFAULT,
               SQUAT_LIMIT_STABLE_TILT_RAD, SQUAT_LIMIT_STABLE_VZ))
    if test == "squat_place_psi0":
        summary["n_box_place_ok"] = sum(
            1 for r in results if r["metrics"].get("box_place_ok"))
        summary["notes"].append(
            "squat_place_psi0: 2 kg wrist-weld v2 carry box (NOT the "
            "cracker box) welded from t=0 at the ep053 frame-0 upper pose; "
            "single playback of the real_ep053 stream, its height_cmd "
            "clipped to [%.2f, %.2f] -> commands[3] = h - 0.75 (same psi0 "
            "height convention); at t_place = argmin(height_cmd) both welds "
            "release and the box geom gets collision bit 2 (+ body bits, "
            "pipeline_abc mechanism) so it lands on the floor; box_land_dx "
            "= box landing point forward of the release-time foot front "
            "edge (positive = placed in front). success = no fall AND "
            "box_place_ok (static, upright < %.0f deg, < %.1f m from the "
            "robot, center below %.2f m)."
            % (PSI0_HEIGHT_CLIP[0], PSI0_HEIGHT_CLIP[1],
               math.degrees(BOX_PLACE_UPRIGHT_TOL), BOX_PLACE_MAX_DIST,
               PLACE_BOX_LAND_MAX_Z))
    if test == "squat_pick_ground":
        summary["box_mode"] = PICK_BOX_MODE
        picks = [r["metrics"].get("pick_success") for r in results
                 if isinstance(r["metrics"].get("pick_success"), bool)]
        summary["pick_success_rate"] = (float(np.mean(picks)) if picks
                                        else None)
        zs = [r["metrics"].get("min_root_z") for r in results]
        zs = [float(v) for v in zs if isinstance(v, (int, float))
              and not isinstance(v, bool) and math.isfinite(v)]
        summary["min_root_z_best"] = min(zs) if zs else None
        summary["notes"].append(
            "squat_pick_ground: 2 kg box (0.35x0.25x0.25 m) upright on the "
            "GROUND %.2f m ahead, collision bit 2 (ground conaffinity 15 "
            "includes bit 2, robot is bit 1 -> box never collides with the "
            "robot); robot starts empty-handed, arms blend after settle to "
            "the low front-reach pose %s ([shoulder_pitch, shoulder_roll, "
            "shoulder_yaw, elbow] x L/R, negative pitch = forward). Height "
            "cmd ramps 0.75 -> %.2f m at %.1f m/s (commands[3] = %.2f, "
            "sent UNCLIPPED; AMO saturates near its R4 depth floor "
            "~0.345 m -- the squat-depth limit decides reachability). "
            "Magnetic grasp: BOTH wrists < %.2f m from the box surface "
            "during the bottom hold -> welds anchored in place "
            "(squat_box_psi0 mechanism); no grasp within %.0f s -> "
            "pick_failed, rise anyway. success = pick AND no fall AND "
            "stand_ok."
            % (PICK_BOX_DIST,
               [round(float(v), 2) for v in PICK_ARM_POSE],
               PICK_SQUAT_H, PICK_SQUAT_RATE,
               PICK_SQUAT_H - BASE_HEIGHT_DEFAULT,
               PICK_GRASP_DIST, PICK_GRASP_TIMEOUT_T))
    if test == "vln_follow":
        summary["tapes_file"] = getattr(args, "tapes", None)
        summary["tape_ids"] = sorted({r["metrics"].get("tape_id")
                                      for r in results
                                      if r["metrics"].get("tape_id")
                                      is not None})
        summary["notes"].append(
            "vln_follow: shared tapes.json command streams (zero-order "
            "hold; trial i uses tape i %% n_tapes; identical tapes across "
            "the four harnesses). AMO has NO wz channel: wz is integrated "
            "into the ABSOLUTE heading target commands[1] (circle/goto "
            "realisation; heading-tracking vs rate-tracking semantic "
            "difference recorded, not worked around), vy -> commands[2]. "
            "play_amo's in-place-stand interlock freezes heading tracking "
            "while |vx_cmd| < %.1f -- small-cmd and stop segments sit "
            "inside that deadzone by design (vln_yaw_locked_frac measures "
            "exposure). Tape frame anchored at the actual robot pose at "
            "tape start; final errors taken at the last tape sample. "
            "success = no fall AND final_pos_err <= %.2f m AND "
            "final_yaw_err <= %.0f deg."
            % (AMO_STAND_VX_THRESH, VLN_FINAL_POS_TOL,
               math.degrees(VLN_FINAL_YAW_TOL)))
    if test in ("circle_pillar", "circle_pillar_psi0"):
        summary["n_contact_collisions"] = sum(
            1 for r in results if r["metrics"].get("pillar_contact_collision"))
        summary["n_proximity_collisions"] = sum(
            1 for r in results if r["metrics"].get("pillar_proximity_collision"))
    if test == "speed_sweep":
        table, max_stable = sweep_breakdown(results)
        summary["per_target"] = table
        summary["max_stable_vx"] = max_stable
    if test == "squat_sweep":
        by_height, by_speed = squat_sweep_breakdown(results)
        summary["by_target_height"] = by_height
        summary["by_ramp_speed"] = by_speed
        summary["notes"].append(
            "squat_sweep: absolute height targets below ~0.45 m are outside "
            "AMO's nominal height-command domain; commands[3]=H-0.75 is sent "
            "unclipped on purpose (that is the measurement).")
    if test in ("goto_ab", "pipeline_abc"):
        summary["coarse_success_rate"] = (
            sum(1 for r in results if r["metrics"].get("success_coarse")) / n
            if n else None)
        summary["fine_success_rate"] = (
            sum(1 for r in results if r["metrics"].get("success_fine")) / n
            if n else None)
        summary["notes"].append(
            "AMO has no wz channel: NAV writes the heading target to the "
            "ABSOLUTE yaw channel commands[1]; CAL slews it at <=%.2f rad/s. "
            "play_amo freezes heading tracking (in-place-stand interlock) "
            "when |vx_cmd| < %.1f, and CAL clamps vx to %.2f which sits at "
            "that deadzone edge -- cal_yaw_locked_frac measures how often "
            "yaw authority was frozen (recorded as-is, by spec)."
            % (CAL_CMD_LIM, AMO_STAND_VX_THRESH, CAL_CMD_LIM))
    if test == "pipeline_abc":
        summary["n_box_place_ok"] = sum(
            1 for r in results if r["metrics"].get("box_place_ok"))
        summary["notes"].append(
            "pipeline_abc: at the squat bottom both wrist welds are "
            "deactivated (data.eq_active) and the box geom gets "
            "contype/conaffinity=15 at runtime so it lands on the ground; "
            "the cached model's flags are reset at every trial start. "
            "success = success_coarse AND no fall AND box_place_ok "
            "(fine convergence recorded, not gating).")
    return summary


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="AMO benchmark harness (G1, MuJoCo)")
    p.add_argument("--test", required=True, choices=TESTS)
    p.add_argument("--trials", type=int, default=None,
                   help="number of trials (default: 50; speed_sweep 25; "
                        "goto_ab 25; pipeline_abc 15; squat_limit 10; "
                        "squat_place_psi0 25; squat_pick_ground 15; "
                        "vln_follow 10). For squat_sweep this "
                        "is trials PER GRID POINT (default 5; grid = "
                        "8 depths + 4 ramp speeds = 12 points)")
    p.add_argument("--out-dir", default="bench_out")
    p.add_argument("--variant", type=int, default=0,
                   help="ManipArena (arena_L1-L5/M1/M2): variant_seed 0-49 "
                        "(BENCHMARK_V2_DESIGN §5; default 0). --trials N re-runs "
                        "the SAME variant N times (default 1).")
    p.add_argument("--upper-replay", default=None,
                   help="psi0 + arena_M1/M2/L5 tests: upper_replay_amo23_*.npz "
                        "(contract v1; on the 5080: ~/AMO/psi0_replay/)")
    p.add_argument("--tapes", default=None,
                   help="vln_follow: tapes.json from make_vln_tapes.py "
                        "(REQUIRED for that test; the SAME file must be "
                        "used by all four harnesses)")
    p.add_argument("--video", choices=("none", "policy", "all"),
                   default="policy",
                   help="policy = first %d successes + all failures "
                        "(pipeline_abc: first %d successes)"
                        % (MAX_SUCCESS_VIDEOS, PIPE_MAX_SUCCESS_VIDEOS))
    return p.parse_args()


def preflight():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        sys.stderr.write("[bench_amo] missing files in cwd=%s: %s\n"
                         "cwd must be ~/AMO.\n" % (os.getcwd(), missing))
        return False
    if not torch.cuda.is_available():
        sys.stderr.write(
            "[bench_amo] CUDA unavailable. AMO JIT policies hardcode cuda:0.\n"
            "If the RTX 5080 hit an Xid-154 fault, reboot the box first "
            "(see ~/AMO/run_amo.sh pre-flight).\n")
        return False
    return True


def main():
    args = parse_args()
    if not preflight():
        return 2

    test = args.test

    # ManipArena tasks have their own variant-driven entry point.
    if test in ARENA_TESTS:
        policy_jit = torch.jit.load("amo_jit.pt", map_location="cuda")
        return run_arena_main(args, policy_jit)

    n_trials = args.trials if args.trials is not None else DEFAULT_TRIALS[test]
    trials = build_trials(test, n_trials)

    replay = None
    if test in PSI0_TESTS:
        if args.upper_replay is None:
            sys.stderr.write(
                "[bench_amo] --upper-replay is required for %s "
                "(upper_replay_amo23_*.npz, contract v1; on the 5080: "
                "~/AMO/psi0_replay/)\n" % test)
            return 2
        replay = load_upper_replay(args.upper_replay)
        if replay.embodiment not in ("amo23", "unknown"):
            sys.stderr.write("[bench_amo] upper-replay embodiment %r != "
                             "amo23 -- wrong npz?\n" % replay.embodiment)
            return 2
        want_src = "sim" if test == "squat_box_psi0" else "real"
        if not replay.source.startswith(want_src):
            print("[bench_amo] WARNING: %s normally uses a %s_* replay "
                  "source, got %r" % (test, want_src, replay.source),
                  flush=True)
        print("[bench_amo] upper replay: %s source=%s K=%d N=%d (%.2fs, "
              "grasp_close_t=%.2fs)"
              % (replay.path, replay.source, len(replay.names), replay.n,
                 replay.duration_s, replay.grasp_close_t), flush=True)
    elif args.upper_replay is not None:
        print("[bench_amo] WARNING: --upper-replay ignored (not a psi0 test)",
              flush=True)

    tapes = None
    if test == "vln_follow":
        if args.tapes is None:
            sys.stderr.write(
                "[bench_amo] --tapes is required for vln_follow "
                "(tapes.json from make_vln_tapes.py; all four harnesses "
                "must consume the SAME file)\n")
            return 2
        tapes = load_vln_tapes(args.tapes)
        print("[bench_amo] vln tapes: %s n=%d duration=%.1fs "
              "stops/tape=%s"
              % (args.tapes, len(tapes), tapes[0].duration,
                 [len(tp.stops) for tp in tapes]), flush=True)
    elif args.tapes is not None:
        print("[bench_amo] WARNING: --tapes ignored (not vln_follow)",
              flush=True)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "amo_%s.jsonl" % test)

    print("[bench_amo] test=%s trials=%d video=%s out=%s device=%s"
          % (test, len(trials), args.video, out_dir,
             torch.cuda.get_device_name(0)), flush=True)

    policy_jit = torch.jit.load("amo_jit.pt", map_location="cuda")

    results = []
    success_videos = 0
    capture_frames = args.video != "none"
    max_success_videos = (PIPE_MAX_SUCCESS_VIDEOS if test == "pipeline_abc"
                          else MAX_SUCCESS_VIDEOS)

    with open(jsonl_path, "w") as jf:
        for trial in trials:
            t0 = time.time()
            try:
                result, frames, model = run_one_trial(test, trial, policy_jit,
                                                      capture_frames,
                                                      replay=replay,
                                                      tapes=tapes)
            except Exception as exc:  # keep long runs alive on a bad trial
                err_metrics = {"fall_reason": "harness_error",
                               "error": repr(exc)}
                for k in ("target_vx", "direction", "sweep",
                          "target_height", "ramp_speed"):
                    if k in trial:  # keep sweep grouping keys on error rows
                        err_metrics[k] = trial[k]
                result = {
                    "framework": FRAMEWORK, "test": test,
                    "trial": trial["index"], "seed": trial["seed"],
                    "success": False, "fall_time": None, "fall_phase": None,
                    "metrics": err_metrics,
                }
                if test in BOX_TESTS or test in PSI0_TESTS:
                    result["box_mode"] = BOX_MODE
                if test == "squat_pick_ground":
                    result["box_mode"] = PICK_BOX_MODE
                if test in PSI0_TESTS:
                    result["upper_mode"] = "psi0_replay_%s" % replay.source
                frames, model = [], None
                print("[bench_amo] trial %d ERROR: %r"
                      % (trial["index"], exc), flush=True)

            results.append(result)
            jf.write(json.dumps(to_jsonable(result)) + "\n")
            jf.flush()

            keep_video = False
            if model is not None and frames:
                if args.video == "all":
                    keep_video = True
                elif args.video == "policy":
                    if not result["success"]:
                        keep_video = True
                    elif success_videos < max_success_videos:
                        keep_video = True
            if keep_video:
                tag = "ok" if result["success"] else "fail"
                vid_path = os.path.join(out_dir, "%s_%s_t%02d_%s.mp4"
                                        % (FRAMEWORK, test, trial["index"], tag))
                if write_video(model, frames, vid_path) and result["success"]:
                    success_videos += 1

            print("[bench_amo] trial %02d/%d seed=%d success=%s fall=%s "
                  "phase=%s wall=%.0fs"
                  % (trial["index"] + 1, len(trials), trial["seed"],
                     result["success"], result["fall_time"],
                     result["fall_phase"], time.time() - t0), flush=True)

    summary = summarize(test, results, args)
    summary_path = os.path.join(out_dir, "summary.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                merged = json.load(f)
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged[test] = to_jsonable(summary)
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    print("[bench_amo] DONE  success_rate=%s  results=%s  summary=%s"
          % (summary["success_rate"], jsonl_path, summary_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
