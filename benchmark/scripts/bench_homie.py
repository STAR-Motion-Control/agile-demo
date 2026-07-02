#!/usr/bin/env python3
"""
HOMIE (OpenHomie) G1 benchmark harness — unified experiment spec v1.

Runs the released HomieDeploy/deploy.onnx policy (456-d obs history -> 12 leg
actions, 50 Hz) in MuJoCo (500 Hz physics), fully headless. The obs/action
pipeline replicates OpenHomie/MujocoDeploy/mujoco_deploy_g1.py:59-183 exactly,
with the torch.jit policy swapped for onnxruntime and the GUI viewer swapped
for mujoco.Renderer (EGL offscreen).

Runtime environment (4090 server, headless):
    ssh 4090
    mkdir -p /sda/g1_bench && cd /sda/g1_bench
    git clone https://github.com/InternRobotics/OpenHomie.git
    python3.10 -m venv venv_homie && source venv_homie/bin/activate
    pip install -U pip
    pip install mujoco==3.2.3 numpy==1.26.4 onnxruntime==1.18.1 PyYAML \
                imageio imageio-ffmpeg

Example commands (from /sda/g1_bench, venv active):
    MUJOCO_GL=egl python bench_homie.py --test walk_speed   --trials 50 \
        --out-dir results/homie/walk_speed   --video policy
    MUJOCO_GL=egl python bench_homie.py --test squat_box    --trials 50 \
        --out-dir results/homie/squat_box    --video policy
    MUJOCO_GL=egl python bench_homie.py --test circle_pillar --trials 50 \
        --out-dir results/homie/circle_pillar --video policy
    MUJOCO_GL=egl python bench_homie.py --test speed_sweep  --trials 5 \
        --out-dir results/homie/speed_sweep  --video policy
    # quick smoke test:
    MUJOCO_GL=egl python bench_homie.py --test walk_speed --trials 2 --video none

Outputs in --out-dir:
    results.jsonl   one line per trial
    summary.json    aggregates (success rate, mean+/-std per metric)
    videos/homie_{test}_t{NN}_{ok|fail}.mp4   per video policy

squat_box box mechanism v2 ("wrist_weld_v2"): the 2 kg box is a free body
held between the hands by two soft weld constraints (one per wrist), so the
load is carried through the arm PD chain instead of being rigidly attached
to the torso. New metrics: box_kept / box_drop_time / box_dist_max, plus a
per-trial harness_error field that catches weld-vs-PD divergence (bad qacc).

psi0 replay variants (walk_speed_psi0 / squat_box_psi0 / circle_pillar_psi0):
the upper body replays a Psi0 trajectory frame-by-frame (--upper-replay npz,
contract v1; columns remapped by joint name to homie27 = arms14 + waist_yaw,
waist roll/pitch dropped upstream) through the existing direct upper-body PD
path. walk/circle psi0: real_ep053 looped, replay height_cmd ignored (legs
stay at stand height), cracker_box (0.411 kg) wrist-welded from t=0; success
= base test + box_kept. squat_box_psi0: BendPick scene from psi0_replay/
scene_info.json (table top 0.40 m, cracker_box upright on it, robot pelvis
~0.29 m from the near edge); 2 s settle -> sim_ep035 replay with the legs
following its (pre-smoothed) height_cmd; at grasp_close_t both wrists must
be < 0.12 m from the box surface to activate the welds (else pick_failed);
afterwards the arms freeze at carry_pose and the standard 20x5 s squat
cycles run. success := pick_success + 20/20 cycles + box_kept. New metrics:
pick_success, box_lift_height. Box collides with table/floor (bit 2) but
never with the robot (bit 1), per the unified psi0 spec.

    MUJOCO_GL=egl python bench_homie.py --test squat_box_psi0 --trials 50 \
        --upper-replay upper_replay_homie27_sim_ep035.npz \
        --out-dir results/homie/squat_box_psi0 --video policy

v3 tests (spec v3; none of them need a psi0 npz):
  T4 squat_sweep   depth x ramp-speed grid with the v2 wrist-weld box
      carried throughout. 8 heights {0.65..0.30} @ 0.2 m/s + 4 ramp rates
      {0.1,0.2,0.4,0.8} @ 0.45 m; --trials N = trials PER GRID POINT
      (default 5). Per trial: settle 2 s (standing carry) -> ramp down ->
      hold 3 s -> ramp up -> stand 1 s. Metrics: achieved_depth (hold-mean
      base_z), depth_err, root_drift_hold (max XY drift during the hold =
      "didn't squat steadily"), root_drift_total (XY end vs pre-squat),
      box_kept. summary.by_height / .by_speed answer "which heights fall,
      how far does the root drift".
  T5 goto_ab       A(origin, yaw 0 +- 0.3 rad seeded noise) -> B(3.0, 1.0),
      goal yaw +90 deg, no box. Shared two-phase controller (math identical
      across the four harnesses): NAV coarse P-law until 0.30 m & 15 deg
      (or 30 s timeout), then CAL with all commands clamped to |0.10| (no
      deadzone compensation); success_fine = 5 cm & 5 deg held 1 s within
      15 s. The project's direct measurement of small-command calibration.
      HOMIE note: native wz channel under-tracks (~x0.6-0.8); measured
      as-is in CAL per spec.
  T6 pipeline_abc  box welded from t=0 (standing hug) -> NAV to
      C(0.0, -2.5), goal yaw -90 deg (right turn + 2.5 m walk), vx_max 0.5
      -> CAL -> squat to 0.45 @ 0.2 m/s -> bottom hold 0.5 s -> release
      both welds (box geom contype/conaffinity flipped to 1 at runtime, box
      drops) -> hold 1 s -> rise -> stand 1 s. box_place_ok = box at rest
      (|v|<0.05), upright (<30 deg), within 0.8 m of the robot. success :=
      success_coarse AND no fall AND box_place_ok (fine convergence
      recorded, not gating). Videos: first 3 OK + all failed (spec v3).

    MUJOCO_GL=egl python bench_homie.py --test squat_sweep --trials 5 \
        --out-dir results/homie/squat_sweep --video policy
    MUJOCO_GL=egl python bench_homie.py --test goto_ab --trials 25 \
        --out-dir results/homie/goto_ab --video policy
    MUJOCO_GL=egl python bench_homie.py --test pipeline_abc --trials 15 \
        --out-dir results/homie/pipeline_abc --video policy

v4 tests (spec v4):
  T7 squat_limit   descent-limit calibration with the v2 2 kg wrist-weld
      box: 2 s settle (standing carry) -> height command ramps down
      CONTINUOUSLY at 0.05 m/s from 0.74 to 0.10 m (far below the
      [0.24, 0.74] training domain; deliberately NOT clipped — the point
      is to locate where tracking saturates / the policy falls) -> hold
      the 0.10 command 3 s -> end (no rise). Stops at a fall.
      metrics.descent_curve = [[h_cmd, base_z, drift_xy], ...] @ 10 Hz;
      scalars: fall_h_cmd / fall_base_z (None if no fall), depth_floor
      (min stable base_z = the method's physical squat limit),
      track_sat_h (first cmd height with base_z - h_cmd > 5 cm),
      drift5_h / drift20_h (cmd height at first 5 / 20 cm root XY drift),
      max_drift_xy. success := no fall (reached and held the 0.10
      command). Most models are expected to saturate without falling;
      --video all recommended (every trial is a calibration sample).
  T8 squat_place_psi0  Psi0-coordinated reach-and-place: carry the 2 kg
      v2 box from t=0 (standing), then single-play a real_ep053
      upper-body stream (arms + waist, "two-hand lower-and-place",
      ~22.3 s) through the existing psi0 replay path while the legs
      follow its height_cmd (0.77 -> 0.46 -> 0.75, clipped to the
      [0.24, 0.74] HOMIE domain — only the 0.77 top end is affected).
      At t_place = argmin(height_cmd) (computed at npz load) both wrist
      welds release via the bit-2+body release mechanism and the box
      drops in front of the robot; the replay then rises and the trial
      ends with a 1 s stand. Lower-body-error metrics:
      height_rmse_descent/hold/rise (segments from the replay height
      profile), root_drift_place (root XY drift during reach+place —
      the forward-lean decoupling stress test), max_tilt, box_place_ok +
      box_land_dx (box rest position vs the release-time foot front
      edge along the heading; >0 = placed in front), stand_ok.
      success := no fall AND box_place_ok. Requires
      --upper-replay upper_replay_homie27_real_ep053*.npz.

    MUJOCO_GL=egl python bench_homie.py --test squat_limit --trials 10 \
        --out-dir results/homie/squat_limit --video all
    MUJOCO_GL=egl python bench_homie.py --test squat_place_psi0 --trials 25 \
        --upper-replay upper_replay_homie27_real_ep053.npz \
        --out-dir results/homie/squat_place_psi0 --video policy

v5 tests (spec v5):
  T9 squat_pick_ground  ground-level box pick: the 2 kg v2-size box
      (0.25 x 0.35 x 0.25 m) stands ON THE FLOOR 0.45 m ahead of the robot
      (bit-2 collision: box hits floor, never the robot); the robot starts
      EMPTY-HANDED, arms hanging. 2 s settle -> 1 s arm swing into the low
      forward-reach pose (FK-tuned, see PICK_ARM_POSE + NOTES: sh_pitch
      -0.7, sh_roll +/-0.20, elbow 1.2 -> wrists ~(0.30, +/-0.18) m ahead,
      straddling the box, ~0.07 m above base_z+0.06) -> height cmd ramps
      down 0.2 m/s to 0.25 m (UNIFIED command, each model saturates at its
      own depth limit) -> bottom wait: when BOTH wrists are < 0.30 m from
      the box surface the welds activate in place (same magnetic-grasp
      mechanism as squat_box_psi0) -> 0.5 s grasp hold -> height ramps
      back to 0.74 carrying the 2 kg load -> 1 s stand. 5 s at the bottom
      without a grasp -> pick_failed, rises anyway. Metrics: pick_success,
      min_root_z (lowest root height actually reached), grasp_dist_left/
      right (at grasp or timeout), root_drift_bottom (XY over bottom +
      grasp), stand_ok, box_kept. success := pick_success AND no fall AND
      stand_ok. Squat-depth limits decide this test (R4 depth floors:
      AMO 0.345 / HOMIE ~0.20 / FALCON 0.213 / AGILE ~0.53 — HOMIE is the
      most promising; AGILE is expected to never reach).
  T10 vln_follow  irregular small-command stream following (simulated VLN
      navigation): --tapes tapes.json REQUIRED (make_vln_tapes.py; ONE
      shared set of 10 tapes for all four harnesses, fairness). Tape: 30 s,
      zero-order-hold [t,vx,vy,wz] breakpoints updated at 0.4-1.2 s jitter,
      |vx|<=0.35 |vy|<=0.2 |wz|<=0.30, two 1-2 s full stops, several
      small-command stretches; ref_xy_yaw = ideal error-free integral
      @ 1 Hz. 2 s settle -> tape playback (HOMIE native wz channel taken
      as-is; known to under-track ~x0.6-0.8) -> 1 s stand. Trial i plays
      tape i % n_tapes. Errors measured in the tape-start frame (robot
      pose at settle end = tape origin). Metrics: final_pos_err /
      final_yaw_err_rad (vs ref end pose), mean_track_err (mean 1 Hz
      position error), small_cmd_response (mean actual/commanded vx ratio
      where |cmd vx| in [0.05, 0.15]), stop_settle (mean residual XY
      displacement inside full-stop segments after a 0.5 s decel skip).
      success := no fall AND final_pos_err <= 0.30 m AND final_yaw_err <=
      15 deg.

    MUJOCO_GL=egl python bench_homie.py --test squat_pick_ground --trials 15 \
        --out-dir results/homie/squat_pick_ground --video policy
    MUJOCO_GL=egl python bench_homie.py --test vln_follow --trials 10 \
        --tapes vln_tapes.json --out-dir results/homie/vln_follow --video policy

License note: OpenHomie is CC BY-NC-SA (non-commercial). Internal benchmark
use only.
"""
import argparse
import collections
import json
import math
import os
import sys
import time

# Must be set before importing mujoco for headless EGL rendering.
# Linux-only default: 'egl' is invalid on macOS (use cgl/glfw there).
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import onnxruntime as ort

# ManipArena v2 (BENCHMARK_V2_DESIGN.md). Shared scene generator + mission
# state machine; this harness is the reference adapter (§7.3). Imported from
# the same scripts/ dir so the four harnesses share one source of truth.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manip_scene as ms          # noqa: E402
import manip_mission as mm        # noqa: E402

# ---------------------------------------------------------------------------
# Constants — HOMIE pipeline (sources: MujocoDeploy/g1.yaml, mujoco_deploy_g1.py,
# HomieDeploy/g1_gym_deploy/envs/lcm_agent.py)
# ---------------------------------------------------------------------------
FRAMEWORK = "homie"
XML_REL = "HomieRL/legged_gym/resources/robots/g1_description/g1.xml"
ONNX_REL = "HomieDeploy/deploy.onnx"

SIM_DT = 0.002            # g1.yaml: simulation_dt
DECIMATION = 10           # g1.yaml: control_decimation -> 50 Hz policy
POLICY_DT = SIM_DT * DECIMATION

N_JOINTS = 27             # legs 12 + waist_yaw + 2x7 arms (g1.xml actual)
N_ACTIONS = 12
SINGLE_OBS_DIM = 3 + 1 + 3 + 3 + N_JOINTS + N_JOINTS + 12   # = 76
HIST_LEN = 6
OBS_DIM = SINGLE_OBS_DIM * HIST_LEN                          # = 456

# g1.yaml legs (order: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
LEG_DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float64)
LEG_KP = np.array([100, 100, 100, 150, 40, 40,
                   100, 100, 100, 150, 40, 40], dtype=np.float64)
LEG_KD = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                   2.0, 2.0, 2.0, 4.0, 2.0, 2.0], dtype=np.float64)

# Upper body (waist_yaw, left arm x7, right arm x7) — real-robot gains
# (lcm_agent.py:34-35). NOT the MujocoDeploy demo 100/0.5; wrist_pitch/yaw
# actuatorfrcrange is only +/-5 Nm so kp=20 there.
UPPER_KP = np.array([300,
                     200, 200, 200, 100, 20, 20, 20,
                     200, 200, 200, 100, 20, 20, 20], dtype=np.float64)
UPPER_KD = np.array([5.0,
                     4.0, 4.0, 4.0, 1.0, 0.5, 0.5, 0.5,
                     4.0, 4.0, 4.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float64)

# Obs scales (g1.yaml + lcm_agent.py:71-77). ang_vel scale is a CLI flag
# because the sources disagree (0.25 in g1.yaml vs 0.5 on the real robot).
# A/B MEASURED (mac, mujoco 3.9, circle_pillar wz tracking): 0.25 tracks yaw
# rate ~2x better (radial_err_mean 0.23 vs 0.67 ccw); walk/squat unchanged.
# => default 0.25. TODO(verify-on-server): re-confirm A/B on mujoco 3.2.3.
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
OBS_CLIP = 100.0          # training clip (legged_robot_config.py:205)
ACTION_CLIP = 100.0       # lcm_agent.py:113

STAND_HEIGHT = 0.74       # absolute base height command for standing/walking
SQUAT_HEIGHT = 0.45       # T2 low target (within training domain [0.24, 0.74])
INIT_BASE_Z = 0.78        # TODO(verify-on-server): check feet do not start
                          # penetrating the floor at default leg angles.
JOINT_NOISE = 0.02        # unified spec: +/-0.02 rad init noise

# Fall detection (unified spec)
TILT_FALL_GXY = math.sin(0.9)   # |projected gravity xy| > sin(0.9 rad)
HEIGHT_FALL_MARGIN = 0.2        # base_z < height_cmd - 0.2
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

# Test timing (unified spec)
SETTLE_S = 2.0
WALK_RAMP_S = 2.0
WALK_HOLD_S = 10.0

# --custom-vx render mode (experiment console): walk success threshold
# becomes relative (0.9 * target) instead of the absolute 0.9 m/s spec gate.
# Set once in main() before any trial runs, read-only afterwards.
CUSTOM_VX_REL_THR = False
WALK_LAST_WINDOW_S = 8.0
SQUAT_CYCLES = 20
SQUAT_CYCLE_S = 5.0       # 1.5 down + 1.0 hold + 1.5 up + 1.0 hold
CIRCLE_VX = 0.4
CIRCLE_WZ = 0.4
CIRCLE_RAMP_S = 1.0
CIRCLE_LAPS = 2
CIRCLE_MARGIN_S = 8.0   # slack: policy under-tracks wz, laps take >2pi/0.4 s
SWEEP_SPEEDS = (0.4, 0.6, 0.8, 1.0, 1.2)

# --- T4 squat_sweep (spec v3): depth x ramp-speed grid, v2 box carried ----
SQUAT_SWEEP_HEIGHTS = (0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)
SQUAT_SWEEP_BASE_RATE = 0.2     # m/s, fixed ramp rate for the depth sweep
SQUAT_SWEEP_RATES = (0.1, 0.2, 0.4, 0.8)   # m/s ramp rates, speed sweep
SQUAT_SWEEP_FIXED_H = 0.45      # m, fixed depth for the speed sweep
SQUAT_SWEEP_HOLD_S = 3.0        # s at the bottom
SQUAT_SWEEP_FINAL_S = 1.0       # s standing after the rise

# --- T5/T6 shared two-phase navigation controller (spec v3). The MATH in
# nav_errors/nav_cmd_coarse/nav_cmd_cal is copied from the spec and MUST
# stay line-equivalent across the four harnesses — do not tune per-model.
NAV_SWITCH_DIST = 0.5           # m; farther: heading target = bearing
NAV_KP_POS = 1.0
NAV_KP_YAW = 1.5
NAV_VX_MAX = 0.6                # m/s (NAV vx clipped to [0, vx_max])
NAV_VY_MAX = 0.3                # m/s
NAV_WZ_MAX = 0.6                # rad/s
NAV_POS_TOL = 0.30              # m, coarse exit
NAV_YAW_TOL = math.radians(15.0)
NAV_TIMEOUT_S = 30.0
CAL_CMD_LIM = 0.10              # |vx|,|vy| [m/s] and |wz| [rad/s] in CAL
CAL_POS_TOL = 0.05              # m, fine convergence
CAL_YAW_TOL = math.radians(5.0)
CAL_HOLD_S = 1.0                # fine tolerance must hold this long
CAL_TIMEOUT_S = 15.0
NAV_END_STAND_S = 1.0           # goto_ab: stand quietly before trial end

# T5 goto_ab geometry
GOTO_B_XY = (3.0, 1.0)
GOTO_B_YAW = math.pi / 2.0      # +90 deg
GOTO_YAW_NOISE = 0.3            # +-0.3 rad initial yaw noise (seeded)

# T6 pipeline_abc geometry / squat-place profile
PIPE_C_XY = (0.0, -2.5)
PIPE_C_YAW = -math.pi / 2.0     # -90 deg (right turn, then ~2.5 m walk)
PIPE_NAV_VX_MAX = 0.5           # conservative while carrying the box
PIPE_SQUAT_HEIGHT = 0.45        # absolute base height target (HOMIE abs)
PIPE_SQUAT_SPEED = 0.2          # m/s height ramp
PIPE_BOTTOM_HOLD_S = 0.5        # bottom hold before releasing the box
PIPE_PLACE_HOLD_S = 1.0         # hold after release (box lands)
PIPE_END_STAND_S = 1.0
BOX_PLACE_MAX_SPEED = 0.05      # m/s: box considered at rest
BOX_PLACE_UPRIGHT_TOL = math.radians(30.0)  # box z-axis vs world z
BOX_PLACE_MAX_DIST = 0.8        # m, box-to-robot distance at trial end

# --- T7 squat_limit (spec v4): continuous descent calibration, v2 box ----
SQUAT_LIMIT_MIN_H = 0.10        # m, final height command — far below the
                                # [0.24, 0.74] training domain, NOT clipped
SQUAT_LIMIT_RATE = 0.05         # m/s, continuous downward ramp
SQUAT_LIMIT_HOLD_S = 3.0        # s holding the 0.10 command at the end
SQUAT_LIMIT_CURVE_DECIM = 5     # 50 Hz policy ticks -> 10 Hz descent_curve
SQUAT_LIMIT_SAT_DEV_M = 0.05    # m, base_z - h_cmd gap = tracking saturated
SQUAT_LIMIT_DRIFT5_M = 0.05     # m, drift5_h threshold
SQUAT_LIMIT_DRIFT20_M = 0.20    # m, drift20_h threshold
SQUAT_LIMIT_STABLE_TILT = 0.3   # rad, "stable" gate for depth_floor
SQUAT_LIMIT_FALL_EXCLUDE_S = 0.5  # s before a fall excluded from depth_floor

# --- T8 squat_place_psi0 (spec v4): Psi0 reach-and-place, v2 2 kg box ----
PLACE_HOLD_EPS_M = 0.02         # m, replay hold segment = h_min + this
PLACE_END_STAND_S = 1.0         # s standing after the replay ends
FOOT_TOE_FORWARD_M = 0.125      # m, toe ahead of the ankle_roll origin
                                # (g1.xml front foot sphere x=0.12 + r=0.005,
                                # projected on the heading); box_land_dx ruler
PLACE_BOX_GROUND_Z_MAX = 0.30   # m, box center below this = on the floor
STAND_OK_TILT_RAD = 0.3         # rad, max tilt during the final stand

# --- T9 squat_pick_ground (spec v5): floor-level box pick ------------------
PICK_BOX_DIST_M = 0.45          # m, box center ahead of the robot (heading)
PICK_BOX_HALF = (0.125, 0.175, 0.125)  # same dims as the v2 carry box
PICK_REACH_S = 1.0              # s, arm swing settle->reach blend
PICK_RATE = 0.2                 # m/s height ramp (down AND up)
PICK_TARGET_H = 0.25            # m, unified bottom command (models saturate)
PICK_GRASP_DIST_M = 0.30        # m, BOTH wrists < this from the box surface
PICK_BOTTOM_TIMEOUT_S = 5.0     # s at the bottom without grasp -> pick_failed
PICK_GRASP_HOLD_S = 0.5         # s hold after the grasp before rising
PICK_END_STAND_S = 1.0          # s standing at the end
# Low forward-reach pose (FK-tuned on g1.xml, /tmp probe 2026-06-12; see
# NOTES): wrists land at ~(0.30, +/-0.18, base_z + 0.07) — y straddles the
# 0.35 m box width, surface dist 0.075 m @ base_z 0.25 / 0.12 m @ 0.30.
# Order: waist_yaw, L[sh_p, sh_r, sh_y, elbow, wr x3], R[mirror].
PICK_ARM_POSE = np.array([0.0,
                          -0.7, 0.20, 0.0, 1.2, 0.0, 0.0, 0.0,
                          -0.7, -0.20, 0.0, 1.2, 0.0, 0.0, 0.0],
                         dtype=np.float64)

# --- T10 vln_follow (spec v5): VLN-style command-stream following ----------
VLN_END_STAND_S = 1.0           # s standing after the tape ends
VLN_FINAL_POS_TOL = 0.30        # m, success gate on the end-pose error
VLN_FINAL_YAW_TOL = math.radians(15.0)
VLN_SMALL_VX_LO = 0.05          # m/s, small-command response band (incl.)
VLN_SMALL_VX_HI = 0.15
VLN_STOP_SKIP_S = 0.5           # s skipped at each stop start (deceleration)
VLN_TRACK_MATCH_TOL_S = 0.05    # s, rec-tick vs 1 Hz ref sample matching
VLN_CMD_LIMITS = (0.35, 0.2, 0.30)  # |vx|,|vy|,|wz| sanity bounds at load

CARRY_TESTS = ("squat_box", "squat_sweep", "pipeline_abc",
               "squat_limit")   # hug arms + 2 kg box welded from t=0
PSI0_TESTS = ("walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
              "squat_place_psi0")
# psi0 tests that use the YCB cracker box (0.411 kg) + build-time floor
# collision bits; squat_place_psi0 keeps the v2 2 kg carry box and gets its
# floor bits at release time via release_box().
PSI0_CRACKER_TESTS = ("walk_speed_psi0", "squat_box_psi0",
                      "circle_pillar_psi0")
BOX_TESTS = (CARRY_TESTS + PSI0_TESTS
             + ("squat_pick_ground",))  # tests with a free box + wrist welds
# tests whose box must collide with the floor from t=0 (bit-2 scheme OR-ed
# into the floor plane + world body at build time)
BIT2_FLOOR_TESTS = PSI0_CRACKER_TESTS + ("squat_pick_ground",)
# tests where box_dist tracking only starts once the box is actually grasped
PICK_GATED_TESTS = ("squat_box_psi0", "squat_pick_ground")
NAV_TESTS = ("goto_ab", "pipeline_abc")

# --- ManipArena v2 (BENCHMARK_V2_DESIGN §2-§4): unified-scene M-series -------
ARENA_TESTS = ("arena_M1", "arena_M2")
# HOMIE wrist link names that ms.WRIST_L/WRIST_R placeholders sed-replace to.
# (Literals here because WRIST_BODY_NAMES is defined later in the file; an
# assert below pins them equal so the two never drift.)
ARENA_WRIST_L = "left_wrist_yaw_link"
ARENA_WRIST_R = "right_wrist_yaw_link"
# Box/cube half-extents come from the SceneSpec (BOX_FULL/CUBE_FULL); kept here
# so box_surface_dist gets the arena box dims, not the carry-box default.
ARENA_BOX_HALF = tuple(v / 2.0 for v in ms.BOX_FULL)
ARENA_CUBE_HALF = tuple(v / 2.0 for v in ms.CUBE_FULL)
# Squat-height command mapping (HOMIE absolute base z). A linear "the lower the
# surface, the lower the base must go to reach it" fit, anchored at the carry-box
# reach experience: a 0.75 m table needs ~a light bend, a 0.30 m table a deep
# squat near the HOMIE depth floor (~0.20 m). Clipped to the HOMIE domain so the
# command stays trackable. The mission only emits "target base height + rate".
ARENA_STAND_H = STAND_HEIGHT                     # 0.74 carry/stand
ARENA_H_MIN = 0.24                               # HOMIE training-domain floor
# reach_base_h(top_h): base height command to bring the wrists to a surface at
# height top_h. wrists hang ~0.45 m below the pelvis in the reach pose, so the
# base sits roughly (surface_h + reach_offset); tuned to land inside [0.24,0.74].
ARENA_REACH_OFFSET = 0.10        # m, base above the surface for a tabletop reach
                                 # (the grasp needs the base low enough that the
                                 # reach-pose wrists get within D_GRASP of the
                                 # box; 0.16 was too high and the H0.60 grasp
                                 # missed. The deepest H0.30 pick is at HOMIE's
                                 # squat-recovery limit either way — see report.)
ARENA_STORE_FLOOR_H = ms.Z_STORE_RIM_H           # store-pad floor ~0.12 m
# ROOT CAUSE 1/3: base-height command for the STORE placement squat. The naive
# reach_base_h(0.12)=0.24 is the deepest, most topple-prone squat; under the
# forward-slung 2 kg box the robot pitches forward and falls before reaching it.
# Placing from a moderately shallower base keeps the squat stable — the box's
# short remaining drop onto the 0.12 m pad still lands it upright at rest.
ARENA_STORE_PLACE_H = 0.58
# ^ ROOT CAUSE 1/3: the store placement squat is kept SHALLOW (a light
# bend). Because release_box() levels + lowers the box onto the pad regardless
# of squat depth, the robot does NOT need to squat deep to place on the 0.12 m
# floor — and a deep store squat under the 2 kg box is topple-prone (H0.60 fell
# at the h0.30 store squat). A light bend keeps every variant stable through the
# placement; the box still lands flat at rest on the pad via the release snap.
# Forward offset (m, robot heading) at which the box is set down ahead of the
# pelvis at release. Matched to manip_mission.PLACE_STAND_FWD_M (0.35) so a
# robot standing that far in front of the zone centre, facing it, drops the box
# ON centre; nav error in the placement pose carries straight through to the
# box landing error (the place_ok ruler).
ARENA_RELEASE_FWD_M = 0.35
# Max radius (m) the set-down box may land from the placement target centre.
# Keeps the box clear of the Z_store / relay-table edge (zone half 0.30, box
# half 0.15 -> the box centre must stay within ~0.15 of an edge): clamping a
# small coarse-nav undershoot inside this keeps the box flat in the zone instead
# of toppling on the rim. Well inside PLACE_POS_TOL_M (0.20).
ARENA_PLACE_CLAMP_R = 0.12
ARENA_VX_CARRY = 0.4             # m/s coarse forward clip while carrying
                                # (ROOT CAUSE 3: was 0.5; modestly slower carry
                                # legs are steadier under the 2 kg box load
                                # without stalling like a deeper cut did)
ARENA_SQUAT_RATE = 0.2          # m/s base-height ramp
# Carry base height while holding the box. Kept at the full stand height: the
# policy tracks moderate carry commands FINE at 0.74 m (baseline v3 H0.75 ran
# the whole chain clean here) but freezes/can't strafe when dropped to a lower
# carry stance under the 2 kg load, so a lower carry height HURT stability.
ARENA_CARRY_H = ARENA_STAND_H  # 0.74; see note above
# Pillar avoidance: if a nav leg's straight segment passes within this of the
# pillar centre, insert one lateral side-offset waypoint to skirt it.
ARENA_PILLAR_CLEAR_M = 0.4      # min straight-line clearance to pillar
ARENA_PILLAR_SIDE_M = 0.7       # lateral offset of the inserted skirt waypoint

# Video
VIDEO_W, VIDEO_H = 640, 480
VIDEO_FPS = 30
MAX_OK_VIDEOS = 5
PIPE_MAX_OK_VIDEOS = 3          # T6 spec: first 3 OK + all failed

# Scene injection snippets.
# Box-carry v2 ("wrist_weld_v2"): the 2 kg box is a FREE body (freejoint)
# held between the two hands by one soft weld constraint per wrist
# (left/right *_wrist_yaw_link — last actuated arm body; finger joints in
# g1.xml are commented out, the hand links are fixed to wrist_yaw). The load
# path is box -> welds -> wrists -> arm PD chain; arm sag/oscillation under
# the 2 kg is real physics and is recorded as-is. solref "0.02 1" keeps the
# welds compliant enough not to fight the upper-body PD (numerical safety).
# Collision still disabled per spec (contype=0: no grasp-contact simulation).
BOX_MODE = "wrist_weld_v2"
BOX_KEEP_DIST = 0.6       # box_kept: |box_center - torso| < 0.6 m throughout
WRIST_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# Pin the arena wrist names (declared earlier in the file) to the canonical
# carry-box names so the two never drift apart.
assert (ARENA_WRIST_L, ARENA_WRIST_R) == WRIST_BODY_NAMES
# Half-sizes: 0.25 m deep (x) x 0.35 m wide (y, between the hands) x 0.25 m
# tall. pos is a placeholder; per-trial pose is set from wrist FK.
BOX_FREE_XML = (
    '<body name="carried_box" pos="0.3 0 0.9">'
    '<freejoint name="carried_box_freejoint"/>'
    '<geom name="carried_box_geom" type="box" size="0.125 0.175 0.125" '
    'mass="2.0" rgba="1.0 0.85 0.1 0.5" contype="0" conaffinity="0"/>'
    "</body>"
)
# T9 squat_pick_ground box: identical dims/mass to the v2 carry box but it
# starts ON THE FLOOR, so it compiles with the bit-2 collision scheme
# (collides with the floor — bit 2 OR-ed into the plane at build time —
# never with the bit-1 robot). Same body/geom/joint names as the v2 box so
# configure_box_welds()/BOX_WELD_XML apply unchanged; welds stay disabled
# until the magnetic grasp.
PICK_BOX_XML = (
    '<body name="carried_box" pos="0.45 0 0.126">'
    '<freejoint name="carried_box_freejoint"/>'
    '<geom name="carried_box_geom" type="box" size="%g %g %g" '
    'mass="2.0" rgba="1.0 0.85 0.1 1" contype="2" conaffinity="2"/>'
    "</body>" % PICK_BOX_HALF
)
# relpose is recomputed at the HUG pose in configure_box_welds(); the XML
# default (qpos0 = arms hanging down) would make the two welds fight.
BOX_WELD_XML = (
    "<equality>"
    '<weld name="box_weld_left" body1="left_wrist_yaw_link" '
    'body2="carried_box" solref="0.02 1"/>'
    '<weld name="box_weld_right" body1="right_wrist_yaw_link" '
    'body2="carried_box" solref="0.02 1"/>'
    "</equality>"
)
PILLAR_XML = (
    '<body name="pillar" pos="0 0 0">'
    '<geom name="pillar_geom" type="cylinder" size="0.15 0.6" pos="0 0 0.6" '
    'rgba="0.5 0.5 0.5 1"/>'
    "</body>"
)
# g1.xml has TWO <worldbody> sections (robot + scene); this floor geom inside
# the scene one is a unique injection anchor.
FLOOR_ANCHOR = ('<geom name="floor" size="0 0 0.05" type="plane" '
                'material="groundplane"/>')

# ---- psi0 upper-body replay (contract v1; psi0_replay/make_replay.py) ------
# npz fields: t (N,), upper_names (K,), upper_pos (N,K), height_cmd (N,),
# grasp_close_t (scalar), carry_pose (K,), meta_json; dt = 0.02 s. homie27:
# K=15 = arms14 + waist_yaw (the g1.xml waist roll/pitch joints are commented
# out; the replay maker drops them upstream). Columns mapped by joint NAME to
# the qpos[19:34] upper slots.
PSI0_GRASP_DIST_M = 0.30  # both wrists must be closer than this to the box surface
PSI0_HEIGHT_CLIP = (0.24, STAND_HEIGHT)  # HOMIE height domain for replayed height_cmd
PSI0_BOX_HALF = (0.036, 0.082, 0.1065)  # cracker_box (YCB 003) half extents [m]
PSI0_BOX_MASS_KG = 0.411
# BendPick scene (psi0_replay/scene_info.json, ep035), shifted +0.6185 m in x
# so the robot pelvis spawns at the origin (it was at x=-0.6185). Floor z=0
# here (scene_info uses tabletop z=0 / floor z=-0.40). Pelvis ends up ~0.29 m
# from the table's near edge (x=0.2935), matching the source scene.
PSI0_TABLE_HALF = (0.625, 0.395, 0.05)
PSI0_TABLE_CENTER = (0.9185, 0.0, 0.35)  # top face at z=0.40
PSI0_BOX_START_XY = (0.2985, -0.0488)  # ep035 target pose, shifted
PSI0_BOX_START_Z = 0.40 + PSI0_BOX_HALF[2] + 0.0005  # resting upright on the tabletop
PSI0_BOX_START_QUAT = (0.997384, 0.0, 0.0, -0.072316)  # ep035 yaw ~= -0.145 rad
# Collision masks (unified psi0 spec): robot keeps bit 1 (g1.xml collision
# geoms default contype/conaffinity 1), box gets bit 2, the table bits 1|2 and
# the floor plane gets bit 2 OR-ed in at runtime -> the box collides with
# table/floor but never with the robot. Same body/geom/joint names as the v2
# carry box so configure_box_welds()/BOX_WELD_XML apply unchanged.
PSI0_BOX_XML = (
    '<body name="carried_box" pos="0.3 0 0.9">'
    '<freejoint name="carried_box_freejoint"/>'
    '<geom name="carried_box_geom" type="box" '
    'size="%g %g %g" mass="%g" rgba="0.85 0.3 0.15 0.8" '
    'contype="2" conaffinity="2"/>'
    "</body>" % (PSI0_BOX_HALF[0], PSI0_BOX_HALF[1], PSI0_BOX_HALF[2],
                 PSI0_BOX_MASS_KG)
)
PSI0_TABLE_XML = (
    '<geom name="bench_table" type="box" '
    'size="%g %g %g" pos="%g %g %g" '
    'contype="3" conaffinity="3" rgba="0.55 0.42 0.26 1"/>'
    % (PSI0_TABLE_HALF + PSI0_TABLE_CENTER)
)
UPPER_QPOS_START = 19  # qpos[19:34] = waist_yaw + left arm x7 + right arm x7

# T2 box-carry arm pose (FK-verified: wrist_yaw origins ~0.35 m apart =
# +/-0.175 from the box center matching the 0.35 m box width, ~0.21 m in
# front of the torso). Order: waist_yaw, L[sh_pitch, sh_roll, sh_yaw, elbow,
# wr x3], R[mirror]. NOTE: obs still subtracts the TRAINING defaults (zeros
# for the upper body, lcm_agent.py:30-33) — never these targets.
BOX_ARM_POSE = np.array([0.0,
                         -0.4, 0.12, 0.0, 1.2, 0.0, 0.0, 0.0,
                         -0.4, -0.12, 0.0, 1.2, 0.0, 0.0, 0.0],
                        dtype=np.float64)
ZERO_ARM_POSE = np.zeros(15, dtype=np.float64)

# Obs default subtraction: legs at LEG_DEFAULT, upper body zeros — exactly
# the padded defaults of mujoco_deploy_g1.py:68-72 / lcm_agent.py:30-33.
OBS_DEFAULT_27 = np.concatenate([LEG_DEFAULT, np.zeros(15)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Math helpers (copied semantics from mujoco_deploy_g1.py:31-57; quat is wxyz)
# ---------------------------------------------------------------------------
def quat_rotate_inverse(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    # rotate v by conj(q)
    qc = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
        + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
        + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
        v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
        + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2)
        + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
        v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
        + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
        + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
    ])


def gravity_body_frame(quat):
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Scene building
# ---------------------------------------------------------------------------
def find_repo(arg_repo):
    candidates = [
        arg_repo,
        os.environ.get("HOMIE_REPO"),
        "OpenHomie",
        "/sda/g1_bench/OpenHomie",
        "/Users/lizhe/Project/sim2real/cc/experiments/repos/OpenHomie",
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, XML_REL)):
            return os.path.abspath(c)
    sys.stderr.write(
        "ERROR: OpenHomie repo not found. Pass --repo /path/to/OpenHomie "
        "(must contain %s)\n" % XML_REL
    )
    sys.exit(2)


def patch_and_replace(xml, anchor, replacement):
    if xml.count(anchor) != 1:
        raise RuntimeError("XML anchor not unique/found: %r" % anchor)
    return xml.replace(anchor, replacement)


def build_model(repo, test):
    xml_path = os.path.join(repo, XML_REL)
    with open(xml_path, "r") as f:
        xml = f.read()
    # meshdir is relative ("meshes"); make absolute so from_xml_string works
    # regardless of cwd.
    mesh_abs = os.path.join(os.path.dirname(xml_path), "meshes")
    xml = patch_and_replace(xml, 'meshdir="meshes"', 'meshdir="%s"' % mesh_abs)
    if test in BOX_TESTS:
        # free box in the scene worldbody (AFTER all robot bodies, so the
        # robot keeps qpos[0:34]/qvel[0:33]; box freejoint lands at qpos[34:]).
        # psi0 walk/circle/pick tests use the cracker-box variant (same
        # names, different size/mass/collision bits); squat_place_psi0
        # keeps the v2 2 kg carry box (spec v4); squat_pick_ground uses the
        # v2-size box with floor collision from t=0 (spec v5).
        if test in PSI0_CRACKER_TESTS:
            box_xml = PSI0_BOX_XML
        elif test == "squat_pick_ground":
            box_xml = PICK_BOX_XML
        else:
            box_xml = BOX_FREE_XML
        xml = patch_and_replace(xml, FLOOR_ANCHOR, FLOOR_ANCHOR + box_xml)
        xml = patch_and_replace(xml, "</mujoco>", BOX_WELD_XML + "</mujoco>")
    if test == "squat_box_psi0":
        xml = patch_and_replace(xml, FLOOR_ANCHOR, FLOOR_ANCHOR + PSI0_TABLE_XML)
    if test in ("circle_pillar", "circle_pillar_psi0"):
        xml = patch_and_replace(xml, FLOOR_ANCHOR, FLOOR_ANCHOR + PILLAR_XML)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = SIM_DT
    if test in BIT2_FLOOR_TESTS:
        # Unified bit-2 spec: floor planes get bit 2 OR-ed in so the box
        # (contype 2) collides with the ground; the robot stays bit 1 ->
        # no box-robot contact.
        for g in range(model.ngeom):
            if (model.geom_bodyid[g] == 0
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE):
                model.geom_contype[g] |= 2
                model.geom_conaffinity[g] |= 2
        # MuJoCo >= 3.2.4 also culls body pairs via the compile-time
        # body_contype/body_conaffinity aggregates; OR bit 2 into the world
        # body so the box (body bits 2/2 from its geom) can reach the floor.
        if hasattr(model, "body_contype"):
            model.body_contype[0] |= 2
            model.body_conaffinity[0] |= 2
    return model


def lookup_ids(model):
    ids = {}
    ids["floor_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ids["foot_bodies"] = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in FOOT_BODY_NAMES
    )
    pg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_geom")
    ids["pillar_geom"] = pg if pg >= 0 else None
    pb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pillar")
    ids["pillar_body"] = pb if pb >= 0 else None
    ids["torso_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          "torso_link")
    return ids


def configure_box_welds(model, upper_pose=BOX_ARM_POSE):
    """squat_box v2 setup. Rewrites the two wrist<->box weld relposes so they
    are mutually CONSISTENT in the carry arm pose (the compiler captures qpos0,
    where the arms hang down — those two relposes would fight each other in
    the carry pose and inject large bogus constraint forces).

    `upper_pose` is the 15-dim qpos[19:34] pose the welds are anchored at:
    the HUG pose for the v2 carry tests, replay frame 0 for walk/circle psi0
    (squat_box_psi0 overwrites eq_data at grasp time anyway).

    Mutates `model.eq_data` once at setup. Returns a dict with the box
    freejoint addresses, body ids and the left-wrist relpose used for
    per-trial box placement.

    eq_data weld layout (verified against compiler output, mujoco 3.9):
    [anchor(3, body2 frame), relpos(3), relquat(4), torquescale(1)] where
    relpos/relquat = pose of body2 (box) in the body1 (wrist) frame.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            "carried_box_freejoint")
    qadr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    assert qadr == 34 and vadr == 33, \
        "box freejoint must come after the 34/33 robot coords (got %d/%d)" \
        % (qadr, vadr)
    box_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carried_box")
    wrist_b = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for side, name in zip(("left", "right"), WRIST_BODY_NAMES)}

    # nominal noise-free standing pose, arms at the anchor pose
    d = mujoco.MjData(model)
    d.qpos[2] = INIT_BASE_Z
    d.qpos[3] = 1.0
    d.qpos[7:19] = LEG_DEFAULT
    d.qpos[19:34] = upper_pose
    d.qpos[qadr + 3] = 1.0
    mujoco.mj_forward(model, d)

    # box nominal pose: centered between the wrists, axis-aligned with the
    # (yaw=0) base so its 0.35 m width spans the two hands
    d.qpos[qadr:qadr + 3] = 0.5 * (d.xpos[wrist_b["left"]]
                                   + d.xpos[wrist_b["right"]])
    mujoco.mj_forward(model, d)

    relpose = {}
    eq_ids = {}
    for side in ("left", "right"):
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                               "box_weld_%s" % side)
        eq_ids[side] = eq
        q1inv = np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[wrist_b[side]])
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, q1inv, d.xquat[box_b])
        relp = np.zeros(3)
        mujoco.mju_rotVecQuat(
            relp, d.xpos[box_b] - d.xpos[wrist_b[side]], q1inv)
        model.eq_data[eq, 0:3] = 0.0       # anchor at box origin
        model.eq_data[eq, 3:6] = relp
        model.eq_data[eq, 6:10] = relq
        model.eq_data[eq, 10] = 1.0        # torquescale
        relpose[side] = (relp, relq)

    return {
        "qadr": qadr,
        "vadr": vadr,
        "body": box_b,
        "wrist_left": wrist_b["left"],
        "wrist_right": wrist_b["right"],
        "relp_left": relpose["left"][0],
        "relq_left": relpose["left"][1],
        "eq_left": eq_ids["left"],
        "eq_right": eq_ids["right"],
        "geom": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  "carried_box_geom"),
    }


def release_box(m, d, box):
    """pipeline_abc bottom-of-squat release: deactivate both wrist welds
    and make the box collide with EVERYTHING (geom contype/conaffinity are
    runtime-writable model fields), so it drops to the floor — contact with
    the robot on the way down is real physics. The model-level collision
    flags persist across trials and are re-disabled at trial init."""
    d.eq_active[box["eq_left"]] = 0
    d.eq_active[box["eq_right"]] = 0
    m.geom_contype[box["geom"]] = 2
    m.geom_conaffinity[box["geom"]] = 2
    # bit-2 scheme: collide with floor (OR-ed below), never the robot (bit 1).
    # Full contype=1 jammed the box between the palms at release (solver
    # depenetration buzz, |v|~34 m/s) -- it never fell.
    for _g in range(m.ngeom):
        if (m.geom_bodyid[_g] == 0
                and m.geom_type[_g] == mujoco.mjtGeom.mjGEOM_PLANE):
            m.geom_contype[_g] |= 2
            m.geom_conaffinity[_g] |= 2
    # MuJoCo >= 3.2.4 broadphase culls BODY pairs via body_contype/
    # body_conaffinity (compile-time OR of each body's geom bits). The box
    # compiles 0/0 and the world body 1/1, so geom-level writes alone are
    # invisible to broadphase and the box free-falls through the floor
    # (verified mujoco 3.9: ncon stays 0). Update the body bits too.
    if hasattr(m, "body_contype"):
        m.body_contype[box["body"]] = 2
        m.body_conaffinity[box["body"]] = 2
        m.body_contype[0] |= 2
        m.body_conaffinity[0] |= 2


# ---------------------------------------------------------------------------
# psi0 upper-body replay (contract v1)
# ---------------------------------------------------------------------------
def load_upper_replay(path):
    """Load + validate a contract-v1 upper-body replay npz
    (psi0_replay/make_replay.py). Returns a plain dict; the model-order
    mapping (pos15/carry15) is filled by prepare_upper_replay()."""
    if not os.path.isfile(path):
        raise SystemExit("upper-replay npz not found: %s" % path)
    z = np.load(path, allow_pickle=True)
    required = ("t", "upper_names", "upper_pos", "height_cmd",
                "grasp_close_t", "carry_pose", "meta_json")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise SystemExit("upper-replay npz missing fields %s: %s"
                         % (missing, path))
    meta = json.loads(str(z["meta_json"]))
    t = np.asarray(z["t"], dtype=np.float64)
    pos = np.asarray(z["upper_pos"], dtype=np.float64)
    if len(t) < 2 or abs((t[1] - t[0]) - POLICY_DT) > 1e-6:
        raise SystemExit(
            "upper-replay dt %s != harness control dt %s: %s"
            % (t[1] - t[0] if len(t) > 1 else "n/a", POLICY_DT, path))
    names = [str(n) for n in z["upper_names"]]
    if pos.shape != (t.shape[0], len(names)):
        raise SystemExit("upper-replay shape mismatch %s: %s"
                         % (pos.shape, path))
    if not np.isfinite(pos).all():
        raise SystemExit("upper-replay contains NaN/inf: %s" % path)
    carry = np.asarray(z["carry_pose"], dtype=np.float64)
    if carry.shape != (len(names),):
        raise SystemExit("upper-replay carry_pose shape %s != (K,): %s"
                         % (carry.shape, path))
    return {
        "path": os.path.abspath(path),
        "names": names,
        "pos": pos,
        "height": np.asarray(z["height_cmd"], dtype=np.float64),
        "grasp_close_t": float(z["grasp_close_t"]),
        "carry_pose": carry,
        "n": int(pos.shape[0]),
        "duration_s": float(pos.shape[0]) * POLICY_DT,
        "source": str(meta.get("source", "unknown")),
        "embodiment": str(meta.get("embodiment", "unknown")),
        "pos15": None,
        "carry15": None,
    }


def prepare_upper_replay(model, rep):
    """Map the replay columns by joint NAME onto the 15 qpos[19:34] upper
    slots (waist_yaw + arms14; HOMIE direct upper-body PD order). Returns a
    NEW dict with pos15 (N,15) / carry15 (15,) added; unmapped slots stay at
    the training default 0 (homie27 K=15 covers all of them)."""
    idx = []
    for name in rep["names"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit("upper-replay joint missing from model: %s" % name)
        k = int(model.jnt_qposadr[jid]) - UPPER_QPOS_START
        if not 0 <= k < 15:
            raise SystemExit("upper-replay joint %s is not in the upper-body "
                             "qpos block [19:34]" % name)
        idx.append(k)
    if len(set(idx)) != len(idx):
        raise SystemExit("upper-replay has duplicate joint columns")
    pos15 = np.zeros((rep["n"], 15), dtype=np.float64)
    pos15[:, idx] = rep["pos"]
    carry15 = np.zeros(15, dtype=np.float64)
    carry15[idx] = rep["carry_pose"]
    return dict(rep, pos15=pos15, carry15=carry15)


def prepare_place_segments(rep):
    """squat_place_psi0: locate the place event and label the replay frames.
    t_place = argmin(height_cmd) (spec v4, computed at npz load time); the
    'hold' segment is the contiguous run around it with height <= min +
    PLACE_HOLD_EPS_M, 'descent' before it, 'rise' after. Returns a NEW dict
    with t_place / place_k / seg_label (list[str], len N) added."""
    h = rep["height"]
    k = int(np.argmin(h))
    hmin = float(h[k])
    first = k
    while first > 0 and float(h[first - 1]) <= hmin + PLACE_HOLD_EPS_M:
        first -= 1
    last = k
    while (last < rep["n"] - 1
           and float(h[last + 1]) <= hmin + PLACE_HOLD_EPS_M):
        last += 1
    labels = ["descent" if i < first else ("hold" if i <= last else "rise")
              for i in range(rep["n"])]
    return dict(rep, place_k=k, t_place=k * POLICY_DT, seg_label=labels)


def box_surface_dist(d, box, wrist_body_id, half_extents=PSI0_BOX_HALF):
    """Distance [m] from a wrist body origin to the oriented box surface
    (0 if inside). half_extents: PSI0_BOX_HALF (cracker box, default) or
    PICK_BOX_HALF (T9 ground box)."""
    rel = d.xpos[wrist_body_id] - d.xpos[box["body"]]
    local = d.xmat[box["body"]].reshape(3, 3).T @ rel
    half = np.asarray(half_extents, dtype=np.float64)
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def activate_welds_at_current_pose(m, d, box):
    """squat_box_psi0 grasp: anchor both wrist welds at the CURRENT relative
    wrist->box pose and enable them. The box is NOT moved — it is grasped
    where it stands on the table."""
    for eq, wb in ((box["eq_left"], box["wrist_left"]),
                   (box["eq_right"], box["wrist_right"])):
        rot_w = d.xmat[wb].reshape(3, 3)
        relp = rot_w.T @ (d.xpos[box["body"]] - d.xpos[wb])
        q1inv = np.zeros(4)
        relq = np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[wb])
        mujoco.mju_mulQuat(relq, q1inv, d.xquat[box["body"]])
        m.eq_data[eq, 0:3] = 0.0       # anchor at box origin
        m.eq_data[eq, 3:6] = relp
        m.eq_data[eq, 6:10] = relq
        m.eq_data[eq, 10] = 1.0        # torquescale
        d.eq_active[eq] = 1


# ---------------------------------------------------------------------------
# Command profiles. Return (vx, vy, wz, height_cmd_abs, phase, cycle_idx).
# height is ABSOLUTE base height in meters (stand = 0.74); HOMIE obs takes it
# raw at obs[3] (mujoco_deploy_g1.py:86).
# ---------------------------------------------------------------------------
def command_at(test, t, vx_target, direction, target_height=None,
               ramp_speed=None, replay=None, tape=None):
    if t < SETTLE_S:
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "settle", None
    tt = t - SETTLE_S
    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        if tt < WALK_RAMP_S:
            return vx_target * tt / WALK_RAMP_S, 0.0, 0.0, STAND_HEIGHT, "ramp", None
        return vx_target, 0.0, 0.0, STAND_HEIGHT, "hold", None
    if test == "vln_follow":
        # T10 (spec v5): zero-order-hold replay of the tape breakpoints
        # through HOMIE's NATIVE vx/vy/wz channels (wz is a yaw-rate
        # command here, taken as-is — unlike AMO, no target-yaw
        # integration), then a short final stand.
        if tt < tape["duration_s"]:
            k = int(np.searchsorted(tape["cmd_t"], tt + 1e-9, "right")) - 1
            if k < 0:
                return 0.0, 0.0, 0.0, STAND_HEIGHT, "follow", None
            vx, vy, wz = (float(v) for v in tape["cmd_v"][k])
            return vx, vy, wz, STAND_HEIGHT, "follow", None
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "end_stand", None
    if test == "squat_box_psi0":
        # Replay segment: legs follow the (pre-smoothed) replayed height_cmd,
        # given to HOMIE as an absolute base height, clipped to its training
        # domain; afterwards the standard 20x5 s squat cycles run.
        if tt < replay["duration_s"]:
            k = min(int(tt / POLICY_DT), replay["n"] - 1)
            h = min(max(float(replay["height"][k]), PSI0_HEIGHT_CLIP[0]),
                    PSI0_HEIGHT_CLIP[1])
            return 0.0, 0.0, 0.0, h, "replay", None
        return command_at("squat_box", t - replay["duration_s"], vx_target,
                          direction)
    if test == "squat_place_psi0":
        # T8: legs follow the replayed height_cmd (HOMIE absolute height
        # semantics, clipped to the training domain — real_ep053 only
        # touches the 0.77 top end); phase labels come from the
        # precomputed replay segmentation (descent / hold / rise).
        if tt < replay["duration_s"]:
            k = min(int(tt / POLICY_DT), replay["n"] - 1)
            h = min(max(float(replay["height"][k]), PSI0_HEIGHT_CLIP[0]),
                    PSI0_HEIGHT_CLIP[1])
            return 0.0, 0.0, 0.0, h, replay["seg_label"][k], None
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "end_stand", None
    if test == "squat_limit":
        # T7 (spec v4): continuous 0.05 m/s descent to 0.10 m, then hold —
        # deliberately NOT clipped to the [0.24, 0.74] training domain
        # (the test calibrates where tracking saturates / the policy
        # falls). No rise segment.
        t_ramp = (STAND_HEIGHT - SQUAT_LIMIT_MIN_H) / SQUAT_LIMIT_RATE
        if tt < t_ramp:
            h = max(STAND_HEIGHT - SQUAT_LIMIT_RATE * tt, SQUAT_LIMIT_MIN_H)
            return 0.0, 0.0, 0.0, h, "descend", None
        return 0.0, 0.0, 0.0, SQUAT_LIMIT_MIN_H, "bottom_hold", None
    if test == "squat_box":
        cyc = min(int(tt // SQUAT_CYCLE_S), SQUAT_CYCLES - 1)
        tc = tt - cyc * SQUAT_CYCLE_S
        if tc < 1.5:
            h = STAND_HEIGHT - (STAND_HEIGHT - SQUAT_HEIGHT) * tc / 1.5
            ph = "down"
        elif tc < 2.5:
            h, ph = SQUAT_HEIGHT, "hold_low"
        elif tc < 4.0:
            h = SQUAT_HEIGHT + (STAND_HEIGHT - SQUAT_HEIGHT) * (tc - 2.5) / 1.5
            ph = "up"
        else:
            h, ph = STAND_HEIGHT, "hold_high"
        return 0.0, 0.0, 0.0, h, ph, cyc
    if test == "squat_sweep":
        # settle -> ramp down at ramp_speed -> hold 3 s -> ramp up -> 1 s
        t_ramp = (STAND_HEIGHT - target_height) / ramp_speed
        if tt < t_ramp:
            h = max(STAND_HEIGHT - ramp_speed * tt, target_height)
            return 0.0, 0.0, 0.0, h, "down", None
        if tt < t_ramp + SQUAT_SWEEP_HOLD_S:
            return 0.0, 0.0, 0.0, target_height, "hold", None
        if tt < 2.0 * t_ramp + SQUAT_SWEEP_HOLD_S:
            h = min(target_height
                    + ramp_speed * (tt - t_ramp - SQUAT_SWEEP_HOLD_S),
                    STAND_HEIGHT)
            return 0.0, 0.0, 0.0, h, "up", None
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "final", None
    if test in ("circle_pillar", "circle_pillar_psi0"):
        sign = 1.0 if direction == "ccw" else -1.0
        f = min(tt / CIRCLE_RAMP_S, 1.0)
        ph = "ramp" if tt < CIRCLE_RAMP_S else "circle"
        return CIRCLE_VX * f, 0.0, sign * CIRCLE_WZ * f, STAND_HEIGHT, ph, None
    raise ValueError(test)


def trial_duration(test, target_height=None, ramp_speed=None, replay=None,
                   tape=None):
    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        return SETTLE_S + WALK_RAMP_S + WALK_HOLD_S
    if test == "vln_follow":
        return SETTLE_S + tape["duration_s"] + VLN_END_STAND_S
    if test == "squat_pick_ground":
        # worst case (grasp never happens); the loop exits early on
        # mission.finished
        t_ramp = (STAND_HEIGHT - PICK_TARGET_H) / PICK_RATE
        return (SETTLE_S + PICK_REACH_S + 2.0 * t_ramp
                + PICK_BOTTOM_TIMEOUT_S + PICK_GRASP_HOLD_S
                + PICK_END_STAND_S + 0.5)
    if test == "squat_box":
        return SETTLE_S + SQUAT_CYCLES * SQUAT_CYCLE_S
    if test == "squat_box_psi0":
        return SETTLE_S + replay["duration_s"] + SQUAT_CYCLES * SQUAT_CYCLE_S
    if test == "squat_place_psi0":
        return SETTLE_S + replay["duration_s"] + PLACE_END_STAND_S
    if test == "squat_limit":
        return (SETTLE_S
                + (STAND_HEIGHT - SQUAT_LIMIT_MIN_H) / SQUAT_LIMIT_RATE
                + SQUAT_LIMIT_HOLD_S)
    if test in ("circle_pillar", "circle_pillar_psi0"):
        return (SETTLE_S + CIRCLE_RAMP_S
                + CIRCLE_LAPS * 2.0 * math.pi / CIRCLE_WZ + CIRCLE_MARGIN_S)
    if test == "squat_sweep":
        t_ramp = (STAND_HEIGHT - target_height) / ramp_speed
        return (SETTLE_S + 2.0 * t_ramp + SQUAT_SWEEP_HOLD_S
                + SQUAT_SWEEP_FINAL_S)
    # NAV tests: worst-case budget; the loop exits early on mission.finished
    if test == "goto_ab":
        return (SETTLE_S + NAV_TIMEOUT_S + CAL_TIMEOUT_S + NAV_END_STAND_S
                + 1.0)
    if test == "pipeline_abc":
        t_ramp = (STAND_HEIGHT - PIPE_SQUAT_HEIGHT) / PIPE_SQUAT_SPEED
        return (SETTLE_S + NAV_TIMEOUT_S + CAL_TIMEOUT_S + 2.0 * t_ramp
                + PIPE_BOTTOM_HOLD_S + PIPE_PLACE_HOLD_S + PIPE_END_STAND_S
                + 1.0)
    raise ValueError(test)


# ---------------------------------------------------------------------------
# T5/T6 shared two-phase navigation controller (spec v3 — the error/command
# math below is the cross-harness contract, copied from the spec and
# identical in all four harnesses; do not improvise).
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
    """CAL: same P-law, every command clamped to |0.10| (deliberately NO
    deadzone compensation — native small-command efficacy is the thing
    being measured)."""
    vx = _clipf(NAV_KP_POS * ex, -CAL_CMD_LIM, CAL_CMD_LIM)
    vy = _clipf(NAV_KP_POS * ey, -CAL_CMD_LIM, CAL_CMD_LIM)
    wz = _clipf(NAV_KP_YAW * eyaw, -CAL_CMD_LIM, CAL_CMD_LIM)
    return vx, vy, wz


class NavMission:
    """Phase machine for goto_ab / pipeline_abc.

    Sequencing lives here; ALL command math is in nav_errors /
    nav_cmd_coarse / nav_cmd_cal (shared, spec-identical). HOMIE has a
    native wz channel that is known to under-track (~x0.6-0.8 of the
    command); the CAL phase measures that as-is, per spec. Heights are
    ABSOLUTE base heights (HOMIE semantics, stand = 0.74)."""

    def __init__(self, test):
        self.test = test
        if test == "goto_ab":
            self.goal_xy, self.goal_yaw = GOTO_B_XY, GOTO_B_YAW
            self.vx_max = NAV_VX_MAX
        else:
            self.goal_xy, self.goal_yaw = PIPE_C_XY, PIPE_C_YAW
            self.vx_max = PIPE_NAV_VX_MAX
        self.squat_t_ramp = ((STAND_HEIGHT - PIPE_SQUAT_HEIGHT)
                             / PIPE_SQUAT_SPEED)
        self.phase = "settle"
        self.phase_t0 = 0.0
        self.cal_hold_t0 = None
        self.fine_converged = False
        self.nav_end = None    # {err_pos, err_yaw, t, reached}
        self.cal_end = None    # {err_pos, err_yaw, t}
        self.release_pending = False
        self.release_t = None
        self.finished = False

    def _to(self, phase, t):
        self.phase = phase
        self.phase_t0 = t

    def command(self, t, base_xy, yaw):
        """-> (vx, vy, wz, height_abs, phase). Falls through phase
        transitions so the new phase issues its command in the same tick."""
        dist, ex, ey, eyaw, _yaw_target = nav_errors(
            base_xy[0], base_xy[1], yaw, self.goal_xy, self.goal_yaw)
        if self.phase == "settle":
            if t < SETTLE_S:
                return 0.0, 0.0, 0.0, STAND_HEIGHT, "settle"
            self._to("nav", t)
        if self.phase == "nav":
            el = t - self.phase_t0
            reached = dist <= NAV_POS_TOL and abs(eyaw) <= NAV_YAW_TOL
            if reached or el >= NAV_TIMEOUT_S:
                self.nav_end = {"err_pos": dist, "err_yaw": abs(eyaw),
                                "t": el, "reached": bool(reached)}
                self._to("cal", t)
            else:
                vx, vy, wz = nav_cmd_coarse(ex, ey, eyaw, self.vx_max)
                return vx, vy, wz, STAND_HEIGHT, "nav"
        if self.phase == "cal":
            el = t - self.phase_t0
            in_tol = dist <= CAL_POS_TOL and abs(eyaw) <= CAL_YAW_TOL
            if in_tol:
                if self.cal_hold_t0 is None:
                    self.cal_hold_t0 = t
                elif t - self.cal_hold_t0 >= CAL_HOLD_S:
                    self.fine_converged = True
            else:
                self.cal_hold_t0 = None
            if self.fine_converged or el >= CAL_TIMEOUT_S:
                self.cal_end = {"err_pos": dist, "err_yaw": abs(eyaw),
                                "t": el}
                self._to("end_stand" if self.test == "goto_ab"
                         else "squat_down", t)
            else:
                vx, vy, wz = nav_cmd_cal(ex, ey, eyaw)
                return vx, vy, wz, STAND_HEIGHT, "cal"
        if self.phase == "squat_down":
            el = t - self.phase_t0
            if el < self.squat_t_ramp:
                h = STAND_HEIGHT - PIPE_SQUAT_SPEED * el
                return 0.0, 0.0, 0.0, h, "squat_down"
            self._to("squat_hold", t)
        if self.phase == "squat_hold":
            if t - self.phase_t0 < PIPE_BOTTOM_HOLD_S:
                return 0.0, 0.0, 0.0, PIPE_SQUAT_HEIGHT, "squat_hold"
            self.release_pending = True   # executed by the rollout loop
            self.release_t = t
            self._to("place_hold", t)
        if self.phase == "place_hold":
            if t - self.phase_t0 < PIPE_PLACE_HOLD_S:
                return 0.0, 0.0, 0.0, PIPE_SQUAT_HEIGHT, "place_hold"
            self._to("squat_up", t)
        if self.phase == "squat_up":
            el = t - self.phase_t0
            if el < self.squat_t_ramp:
                h = PIPE_SQUAT_HEIGHT + PIPE_SQUAT_SPEED * el
                return 0.0, 0.0, 0.0, h, "squat_up"
            self._to("end_stand", t)
        end_s = (NAV_END_STAND_S if self.test == "goto_ab"
                 else PIPE_END_STAND_S)
        if t - self.phase_t0 >= end_s:
            self.finished = True
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "end_stand"


def fill_nav_metrics(metrics, mission):
    """Common T5/T6 navigation metric block (AMO-compatible field names).
    success_coarse = NAV exited on criterion (0.30 m & 15 deg, not by
    timeout; False if the trial died mid-NAV); success_fine = CAL held the
    5 cm & 5 deg tolerance for 1 s."""
    ne, ce = mission.nav_end, mission.cal_end
    metrics["err_pos_nav"] = ne["err_pos"] if ne else None
    metrics["err_yaw_nav"] = ne["err_yaw"] if ne else None
    metrics["t_nav"] = ne["t"] if ne else None
    metrics["nav_timeout"] = (not ne["reached"]) if ne else None
    metrics["err_pos_cal"] = ce["err_pos"] if ce else None
    metrics["err_yaw_cal"] = ce["err_yaw"] if ce else None
    metrics["t_cal"] = ce["t"] if ce else None
    if ne and ce:
        metrics["cal_improve_pos"] = ne["err_pos"] - ce["err_pos"]
        metrics["cal_improve_yaw"] = ne["err_yaw"] - ce["err_yaw"]
    else:
        metrics["cal_improve_pos"] = None
        metrics["cal_improve_yaw"] = None
    metrics["success_coarse"] = bool(ne is not None and ne["reached"])
    metrics["success_fine"] = bool(mission.fine_converged)


class PickGroundMission:
    """T9 squat_pick_ground phase machine (spec v5).

    Time-driven phases: settle -> reach -> descend -> bottom_wait; the
    bottom_wait exit is EVENT-driven from the rollout loop (on_grasp /
    on_timeout — the grasp needs wrist-to-box distances only the rollout
    can compute), then grasp_hold -> rise -> stand -> finished. Heights are
    ABSOLUTE base heights (HOMIE semantics, stand = 0.74)."""

    def __init__(self):
        self.t_ramp = (STAND_HEIGHT - PICK_TARGET_H) / PICK_RATE
        self.phase = "settle"
        self.phase_t0 = 0.0
        self.grasp_t = None
        self.pick_failed = False
        self.finished = False

    def _to(self, phase, t):
        self.phase = phase
        self.phase_t0 = t

    def on_grasp(self, t):
        """Rollout: both wrists within PICK_GRASP_DIST_M, welds activated."""
        self.grasp_t = t
        self._to("grasp_hold", t)

    def on_timeout(self, t):
        """Rollout: PICK_BOTTOM_TIMEOUT_S at the bottom without a grasp."""
        self.pick_failed = True
        self._to("rise", t)

    def command(self, t):
        """-> (height_abs, phase). Falls through time-driven transitions so
        the new phase issues its command in the same tick."""
        if self.phase == "settle":
            if t < SETTLE_S:
                return STAND_HEIGHT, "settle"
            self._to("reach", t)
        if self.phase == "reach":
            if t - self.phase_t0 < PICK_REACH_S:
                return STAND_HEIGHT, "reach"
            self._to("descend", t)
        if self.phase == "descend":
            el = t - self.phase_t0
            if el < self.t_ramp:
                return STAND_HEIGHT - PICK_RATE * el, "descend"
            self._to("bottom_wait", t)
        if self.phase == "bottom_wait":
            return PICK_TARGET_H, "bottom_wait"
        if self.phase == "grasp_hold":
            if t - self.phase_t0 < PICK_GRASP_HOLD_S:
                return PICK_TARGET_H, "grasp_hold"
            self._to("rise", t)
        if self.phase == "rise":
            el = t - self.phase_t0
            if el < self.t_ramp:
                return min(PICK_TARGET_H + PICK_RATE * el, STAND_HEIGHT), "rise"
            self._to("stand", t)
        if t - self.phase_t0 >= PICK_END_STAND_S:
            self.finished = True
        return STAND_HEIGHT, "stand"


# ---------------------------------------------------------------------------
# T10 vln_follow tape loading (spec v5; tapes.json from make_vln_tapes.py,
# ONE shared file across all four harnesses)
# ---------------------------------------------------------------------------
def load_vln_tapes(path):
    """Load + validate a vln tapes.json. Format: {"tapes": [{"id": int,
    "dt": float, "cmds": [[t, vx, vy, wz], ...] (zero-order-hold
    breakpoints), "ref_xy_yaw": [[t, x, y, yaw], ...] (ideal integral,
    ~1 Hz)}]}. Returns a list of dicts with numpy arrays + precomputed
    full-stop segments [(t0, t1), ...]."""
    if not os.path.isfile(path):
        raise SystemExit("vln tapes file not found: %s" % path)
    with open(path, "r") as f:
        try:
            raw = json.load(f)
        except ValueError as e:
            raise SystemExit("vln tapes file is not valid JSON: %s (%s)"
                             % (path, e))
    if not isinstance(raw, dict) or not isinstance(raw.get("tapes"), list) \
            or not raw["tapes"]:
        raise SystemExit('vln tapes file needs a non-empty "tapes" list: %s'
                         % path)
    tapes = []
    for ti, tp in enumerate(raw["tapes"]):
        where = "%s tape[%d]" % (path, ti)
        cmds = np.asarray(tp.get("cmds", []), dtype=np.float64)
        ref = np.asarray(tp.get("ref_xy_yaw", []), dtype=np.float64)
        if cmds.ndim != 2 or cmds.shape[1] != 4 or cmds.shape[0] < 2:
            raise SystemExit("%s: cmds must be [[t,vx,vy,wz],...]" % where)
        if ref.ndim != 2 or ref.shape[1] != 4 or ref.shape[0] < 2:
            raise SystemExit("%s: ref_xy_yaw must be [[t,x,y,yaw],...]"
                             % where)
        if not (np.isfinite(cmds).all() and np.isfinite(ref).all()):
            raise SystemExit("%s: NaN/inf in cmds or ref" % where)
        if (np.diff(cmds[:, 0]) <= 0).any() or (np.diff(ref[:, 0]) <= 0).any():
            raise SystemExit("%s: t columns must be strictly increasing"
                             % where)
        for col, lim, nm in zip((1, 2, 3), VLN_CMD_LIMITS,
                                ("vx", "vy", "wz")):
            if np.abs(cmds[:, col]).max() > lim + 1e-6:
                sys.stderr.write("WARNING: %s: |%s| %.3f exceeds spec "
                                 "limit %.2f\n" % (where, nm,
                                                   np.abs(cmds[:, col]).max(),
                                                   lim))
        duration = float(max(cmds[-1, 0], ref[-1, 0]))
        # full-stop segments: consecutive all-zero breakpoints, ZOH until
        # the next non-zero breakpoint (or tape end)
        stops = []
        zero = np.all(np.abs(cmds[:, 1:4]) < 1e-9, axis=1)
        k = 0
        while k < len(cmds):
            if zero[k]:
                j = k
                while j + 1 < len(cmds) and zero[j + 1]:
                    j += 1
                t1 = float(cmds[j + 1, 0]) if j + 1 < len(cmds) else duration
                stops.append((float(cmds[k, 0]), t1))
                k = j + 1
            else:
                k += 1
        tapes.append({
            "id": int(tp.get("id", ti)),
            "dt": float(tp.get("dt", POLICY_DT)),
            "cmd_t": cmds[:, 0].copy(),
            "cmd_v": cmds[:, 1:4].copy(),
            "ref_t": ref[:, 0].copy(),
            "ref_xy": ref[:, 1:3].copy(),
            "ref_yaw": ref[:, 3].copy(),
            "duration_s": duration,
            "stops": stops,
        })
    return tapes


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------
def run_trial(ctx, trial_idx, seed, vx_target=1.0, direction="ccw",
              target_height=None, ramp_speed=None, sweep=None,
              tape_id=None, video_path=None):
    """One trial. Deterministic given seed (video rendering does not affect
    physics, so a failed/selected trial can be re-run with video)."""
    test = ctx["test"]
    m = ctx["model"]
    ids = ctx["ids"]
    sess = ctx["sess"]
    in_name, out_name = ctx["in_name"], ctx["out_name"]
    ang_vel_scale = ctx["ang_vel_scale"]
    rep = ctx.get("upper")        # psi0 upper-body replay dict, else None
    tape = (ctx["tapes"][tape_id] if test == "vln_follow" else None)

    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)

    # --- initial state ----------------------------------------------------
    if test in ("circle_pillar", "circle_pillar_psi0"):
        # On the circle around the pillar at origin (radius = vx/wz; 1 m for
        # the spec 0.4/0.4, overridable via --custom-vx/--custom-wz), facing
        # the tangent. No yaw randomization for this test (unified spec).
        yaw0 = math.pi / 2.0 if direction == "ccw" else -math.pi / 2.0
        base_xy = (CIRCLE_VX / CIRCLE_WZ, 0.0)
    elif test == "squat_box_psi0":
        yaw0 = 0.0  # BendPick table scene alignment (scene_info: identity heading)
        base_xy = (0.0, 0.0)
    elif test == "goto_ab":
        # A = origin, heading 0 + seeded yaw noise +-0.3 rad (spec v3)
        yaw0 = float(rng.uniform(-GOTO_YAW_NOISE, GOTO_YAW_NOISE))
        base_xy = (0.0, 0.0)
    elif test == "pipeline_abc":
        yaw0 = 0.0                      # A = origin, heading 0 (spec v3)
        base_xy = (0.0, 0.0)
    else:
        yaw0 = float(rng.uniform(-math.pi, math.pi))
        base_xy = (0.0, 0.0)
    if test in PSI0_TESTS:
        upper_target = rep["pos15"][0]   # replay frame 0 (PD targets follow)
    elif test in CARRY_TESTS:
        upper_target = BOX_ARM_POSE
    else:
        upper_target = ZERO_ARM_POSE

    box = ctx.get("box")
    if test in ("pipeline_abc", "squat_place_psi0"):
        # model-level fields persist across trials (and video re-runs):
        # re-disable box collision that a previous release switched on
        # (geom AND body bits — broadphase filters on the body level).
        m.geom_contype[box["geom"]] = 0
        m.geom_conaffinity[box["geom"]] = 0
        if hasattr(m, "body_contype"):
            m.body_contype[box["body"]] = 0
            m.body_conaffinity[box["body"]] = 0
    mission = NavMission(test) if test in NAV_TESTS else None
    pick_mission = (PickGroundMission() if test == "squat_pick_ground"
                    else None)

    d.qpos[:] = 0.0
    d.qpos[0] = base_xy[0]
    d.qpos[1] = base_xy[1]
    d.qpos[2] = INIT_BASE_Z
    d.qpos[3:7] = [math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)]
    d.qpos[7:19] = LEG_DEFAULT + rng.uniform(-JOINT_NOISE, JOINT_NOISE, 12)
    if test in PSI0_TESTS:
        # replayed joints start exactly at replay frame 0 (no noise) so the
        # per-frame PD targets do not snap at t=0.
        d.qpos[19:34] = upper_target
    else:
        d.qpos[19:34] = upper_target + rng.uniform(-JOINT_NOISE, JOINT_NOISE, 15)
    if box is not None:
        d.qpos[box["qadr"] + 3] = 1.0      # valid quat before first FK
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    if test == "squat_box_psi0":
        # Box upright on the table (BendPick ep035 pose); welds OFF until
        # grasp_close_t (the XML welds compile active -> disable explicitly).
        adr = box["qadr"]
        d.qpos[adr:adr + 2] = PSI0_BOX_START_XY
        d.qpos[adr + 2] = PSI0_BOX_START_Z
        bq = np.asarray(PSI0_BOX_START_QUAT, dtype=np.float64)
        d.qpos[adr + 3:adr + 7] = bq / np.linalg.norm(bq)
        d.qvel[box["vadr"]:box["vadr"] + 6] = 0.0
        d.eq_active[box["eq_left"]] = 0
        d.eq_active[box["eq_right"]] = 0
        mujoco.mj_forward(m, d)
    elif test == "squat_pick_ground":
        # T9: box upright ON THE FLOOR, PICK_BOX_DIST_M ahead along the
        # (seeded random) heading, long side facing the hands; welds OFF
        # until the magnetic grasp (XML welds compile active).
        adr = box["qadr"]
        d.qpos[adr] = base_xy[0] + PICK_BOX_DIST_M * math.cos(yaw0)
        d.qpos[adr + 1] = base_xy[1] + PICK_BOX_DIST_M * math.sin(yaw0)
        d.qpos[adr + 2] = PICK_BOX_HALF[2] + 0.0005
        d.qpos[adr + 3:adr + 7] = [math.cos(yaw0 / 2.0), 0.0, 0.0,
                                   math.sin(yaw0 / 2.0)]
        d.qvel[box["vadr"]:box["vadr"] + 6] = 0.0
        d.eq_active[box["eq_left"]] = 0
        d.eq_active[box["eq_right"]] = 0
        mujoco.mj_forward(m, d)
    elif box is not None:
        # Place the box exactly at the left weld target (wrist FK at the
        # noisy hug pose / psi0 replay frame 0); the right weld is off only
        # by the +/-0.02 rad joint noise, absorbed by the soft solref.
        qw = d.xquat[box["wrist_left"]]
        boxq = np.zeros(4)
        mujoco.mju_mulQuat(boxq, qw, box["relq_left"])
        off = np.zeros(3)
        mujoco.mju_rotVecQuat(off, box["relp_left"], qw)
        d.qpos[box["qadr"]:box["qadr"] + 3] = d.xpos[box["wrist_left"]] + off
        d.qpos[box["qadr"] + 3:box["qadr"] + 7] = boxq
        mujoco.mj_forward(m, d)

    # --- policy state (mirrors mujoco_deploy_g1.py:109-121) ----------------
    action = np.zeros(N_ACTIONS, dtype=np.float32)
    target_dof_pos = LEG_DEFAULT.copy()
    obs_hist = collections.deque(maxlen=HIST_LEN)
    for _ in range(HIST_LEN):
        obs_hist.append(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))

    # --- video ------------------------------------------------------------
    writer = None
    cam = None
    next_frame_t = 0.0
    if video_path is not None and ctx["renderer"] is not None:
        import imageio
        writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264",
                                    quality=8)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.elevation, cam.azimuth = 3.5, -20.0, 135.0

    # --- records ----------------------------------------------------------
    rec = collections.defaultdict(list)   # per policy tick
    fall = False
    fall_time = fall_phase = fall_reason = None
    fall_cycle = None
    box_drop_time = None
    harness_error = None
    pick_checked = False          # squat_box_psi0 grasp state
    pick_success = None
    grasp_dists = None
    place_released = False        # squat_place_psi0 release state
    place_release_t = None
    place_frame = None            # (heading_x, heading_y, foot_front_s)
    fall_cmd_h = None             # cmd height at the fall instant (T7)
    fall_base_z_v = None          # actual base_z at the fall instant (T7)
    box_z0 = box_z_max = None
    pillar_collision = False
    pillar_collision_time = None
    laps = []                 # [{"lap","pos_err_m","yaw_err_rad","t"}]
    accum_angle = 0.0
    prev_pos_angle = None
    start_xy = None
    start_yaw = None

    duration = trial_duration(test, target_height=target_height,
                              ramp_speed=ramp_speed, replay=rep, tape=tape)
    max_steps = int(round(duration / SIM_DT))
    nonfoot_ground = False    # accumulated between policy ticks
    t_end = 0.0
    wall_t0 = time.time()

    floor_g = ids["floor_geom"]
    pillar_g = ids["pillar_geom"]
    pillar_b = ids["pillar_body"]
    foot_bodies = ids["foot_bodies"]
    box_b = box["body"] if box is not None else None

    try:
        for step in range(max_steps):
            # PD every sim step (500 Hz), identical split to official script:
            # legs from policy targets, upper body position-held.
            d.ctrl[0:12] = ((target_dof_pos - d.qpos[7:19]) * LEG_KP
                            - d.qvel[6:18] * LEG_KD)
            d.ctrl[12:27] = ((upper_target - d.qpos[19:34]) * UPPER_KP
                             - d.qvel[18:33] * UPPER_KD)
            mujoco.mj_step(m, d)
            t = (step + 1) * SIM_DT
            t_end = t

            # contacts checked every sim step (cheap; ncon is small)
            for ci in range(d.ncon):
                g1 = d.contact[ci].geom1
                g2 = d.contact[ci].geom2
                if floor_g in (g1, g2):
                    other = g2 if g1 == floor_g else g1
                    b = int(m.geom_bodyid[other])
                    # box_b exempt: the released box landing on the floor
                    # (pipeline_abc) is not a robot fall.
                    if (b != 0 and b != pillar_b and b != box_b
                            and b not in foot_bodies):
                        nonfoot_ground = True
                if pillar_g is not None and pillar_g in (g1, g2):
                    other = g2 if g1 == pillar_g else g1
                    b = int(m.geom_bodyid[other])
                    if b != 0 and b != pillar_b and not pillar_collision:
                        pillar_collision = True
                        pillar_collision_time = t

            if writer is not None and t >= next_frame_t:
                cam.lookat[:] = d.qpos[0:3]
                ctx["renderer"].update_scene(d, camera=cam)
                writer.append_data(ctx["renderer"].render())
                next_frame_t += 1.0 / VIDEO_FPS

            if (step + 1) % DECIMATION != 0:
                continue

            # ----------------- policy tick (50 Hz) -------------------------
            # divergence guard: weld-vs-PD blowups must not poison the run
            if (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.all(np.isfinite(d.qpos))):
                harness_error = "qacc_diverged"
                break

            if mission is not None:
                vx_c, vy_c, wz_c, h_c, phase = mission.command(
                    t, (float(d.qpos[0]), float(d.qpos[1])),
                    yaw_from_quat(d.qpos[3:7]))
                cyc = None
                if mission.release_pending:
                    release_box(m, d, box)
                    mission.release_pending = False
            elif pick_mission is not None:
                h_c, phase = pick_mission.command(t)
                vx_c = vy_c = wz_c = 0.0
                cyc = None
            else:
                vx_c, vy_c, wz_c, h_c, phase, cyc = command_at(
                    test, t, vx_target, direction,
                    target_height=target_height, ramp_speed=ramp_speed,
                    replay=rep, tape=tape)

            # psi0: upper-body PD targets replayed frame-by-frame at 50 Hz
            # (the 500 Hz PD above holds the latest target between ticks).
            if rep is not None:
                if test in ("squat_box_psi0", "squat_place_psi0"):
                    # replay starts at settle end, SINGLE play; afterwards
                    # the arms freeze (carry_pose for the squat cycles /
                    # last replay frame for the final stand).
                    k = int(math.floor((t - SETTLE_S) / POLICY_DT + 1e-9))
                    if k < 0:
                        upper_target = rep["pos15"][0]
                    elif k >= rep["n"]:
                        upper_target = (rep["carry15"]
                                        if test == "squat_box_psi0"
                                        else rep["pos15"][-1])
                    else:
                        upper_target = rep["pos15"][k]
                else:
                    # walk/circle psi0: real source looped from t=0.
                    k = ((step + 1) // DECIMATION) % rep["n"]
                    upper_target = rep["pos15"][k]

            # T9 low forward-reach arm swing: linear PD-target blend from
            # hanging to PICK_ARM_POSE over PICK_REACH_S after the settle
            # (the obs still subtracts the TRAINING defaults — zeros — not
            # these targets, same rule as the carry tests).
            if test == "squat_pick_ground":
                f_arm = min(max((t - SETTLE_S) / PICK_REACH_S, 0.0), 1.0)
                upper_target = ((1.0 - f_arm) * ZERO_ARM_POSE
                                + f_arm * PICK_ARM_POSE)

            # T9 grasp event (spec v5): while waiting at the bottom, the
            # FIRST tick with BOTH wrists < PICK_GRASP_DIST_M from the box
            # surface welds the box in place (same magnetic mechanism as
            # squat_box_psi0); PICK_BOTTOM_TIMEOUT_S without that ->
            # pick_failed, the mission rises empty-handed. grasp_dists is
            # captured at the grasp or timeout tick (spec).
            if (pick_mission is not None
                    and pick_mission.phase == "bottom_wait"):
                d_l = box_surface_dist(d, box, box["wrist_left"],
                                       half_extents=PICK_BOX_HALF)
                d_r = box_surface_dist(d, box, box["wrist_right"],
                                       half_extents=PICK_BOX_HALF)
                if d_l < PICK_GRASP_DIST_M and d_r < PICK_GRASP_DIST_M:
                    grasp_dists = (d_l, d_r)
                    pick_success = True
                    activate_welds_at_current_pose(m, d, box)
                    pick_mission.on_grasp(t)
                elif (t - pick_mission.phase_t0
                        >= PICK_BOTTOM_TIMEOUT_S):
                    grasp_dists = (d_l, d_r)
                    pick_success = False
                    pick_mission.on_timeout(t)

            # squat_box_psi0 grasp event: both wrists must be near the box
            # surface at grasp_close_t to weld it in place (else pick_failed
            # — the trial keeps running, success gated on pick_success).
            if (test == "squat_box_psi0" and not pick_checked
                    and t - SETTLE_S >= rep["grasp_close_t"]):
                pick_checked = True
                d_l = box_surface_dist(d, box, box["wrist_left"])
                d_r = box_surface_dist(d, box, box["wrist_right"])
                grasp_dists = (d_l, d_r)
                pick_success = bool(d_l < PSI0_GRASP_DIST_M
                                    and d_r < PSI0_GRASP_DIST_M)
                if pick_success:
                    activate_welds_at_current_pose(m, d, box)

            # squat_place_psi0 release event: at t_place = argmin(replay
            # height_cmd) drop both welds via the bit-2+body release
            # mechanism (box falls to the floor, never collides with the
            # robot) and remember the foot front edge along the current
            # heading — the box_land_dx ruler.
            if (test == "squat_place_psi0" and not place_released
                    and t - SETTLE_S >= rep["t_place"] - 1e-9):
                place_released = True
                place_release_t = t
                yaw_r = yaw_from_quat(d.qpos[3:7])
                hx, hy = math.cos(yaw_r), math.sin(yaw_r)
                front = max(float(d.xpos[b][0]) * hx
                            + float(d.xpos[b][1]) * hy
                            for b in foot_bodies) + FOOT_TOE_FORWARD_M
                place_frame = (hx, hy, front)
                release_box(m, d, box)

            # obs frame — exact mirror of mujoco_deploy_g1.py:59-93
            qj = d.qpos[7:7 + N_JOINTS]
            dqj = d.qvel[6:6 + N_JOINTS]
            quat = d.qpos[3:7].copy()              # wxyz
            omega = d.qvel[3:6]                    # body-frame ang vel
            grav = gravity_body_frame(quat)
            single = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
            single[0:3] = np.array([vx_c, vy_c, wz_c], dtype=np.float32) * CMD_SCALE
            single[3] = h_c                        # absolute height, scale 1.0
            single[4:7] = omega * ang_vel_scale
            single[7:10] = grav
            single[10:37] = (qj - OBS_DEFAULT_27) * DOF_POS_SCALE
            single[37:64] = dqj * DOF_VEL_SCALE
            single[64:76] = action                 # previous RAW policy output
            np.clip(single, -OBS_CLIP, OBS_CLIP, out=single)
            obs_hist.append(single)
            obs456 = np.concatenate(obs_hist).astype(np.float32)  # oldest->newest

            out = sess.run([out_name], {in_name: obs456[None, :]})[0][0]
            action = np.clip(out.astype(np.float32), -ACTION_CLIP, ACTION_CLIP)
            target_dof_pos = action.astype(np.float64) * ACTION_SCALE + LEG_DEFAULT

            # ----------------- state metrics -------------------------------
            base_z = float(d.qpos[2])
            yaw = yaw_from_quat(quat)
            v_fwd = float(d.qvel[0] * math.cos(yaw) + d.qvel[1] * math.sin(yaw))
            gxy = float(math.hypot(grav[0], grav[1]))
            tilt = float(math.acos(max(-1.0, min(1.0, -grav[2]))))
            rec["t"].append(t)
            rec["cmd_vx"].append(vx_c)
            rec["cmd_wz"].append(wz_c)
            rec["cmd_h"].append(h_c)
            rec["base_x"].append(float(d.qpos[0]))
            rec["base_y"].append(float(d.qpos[1]))
            rec["base_z"].append(base_z)
            rec["v_fwd"].append(v_fwd)
            rec["omega_z"].append(float(d.qvel[5]))
            rec["tilt"].append(tilt)
            rec["phase"].append(phase)
            rec["yaw"].append(yaw)

            # ----------------- T2 box tracking ------------------------------
            # pick-style tests (squat_box_psi0 / squat_pick_ground): only
            # once the box is welded (before the grasp it legitimately sits
            # on the table/floor, far from the torso). T9 holds the box at
            # arm's length LOW (torso distance legitimately ~0.6 m in the
            # deep squat), so its weld-integrity reference is the WRIST
            # midpoint (~0.15 m when held; same 0.6 m drop threshold).
            if box is not None and (test not in PICK_GATED_TESTS
                                    or bool(pick_success)):
                if test == "squat_pick_ground":
                    ref_pos = 0.5 * (d.xpos[box["wrist_left"]]
                                     + d.xpos[box["wrist_right"]])
                else:
                    ref_pos = d.xpos[ids["torso_body"]]
                box_dist = float(np.linalg.norm(
                    d.xpos[box["body"]] - ref_pos))
                rec["box_dist"].append(box_dist)
                if box_dist >= BOX_KEEP_DIST and box_drop_time is None:
                    box_drop_time = t
            if test == "squat_box_psi0":
                bz = float(d.xpos[box["body"]][2])
                box_z0 = bz if box_z0 is None else box_z0
                box_z_max = bz if box_z_max is None else max(box_z_max, bz)

            # ----------------- T3 lap tracking ------------------------------
            if test in ("circle_pillar", "circle_pillar_psi0") and t >= SETTLE_S:
                if start_xy is None:
                    start_xy = (float(d.qpos[0]), float(d.qpos[1]))
                    start_yaw = yaw
                    prev_pos_angle = math.atan2(d.qpos[1], d.qpos[0])
                pos_angle = math.atan2(d.qpos[1], d.qpos[0])
                accum_angle += wrap_angle(pos_angle - prev_pos_angle)
                prev_pos_angle = pos_angle
                if abs(accum_angle) >= 2.0 * math.pi * (len(laps) + 1):
                    laps.append({
                        "lap": len(laps) + 1,
                        "t": t,
                        "pos_err_m": float(math.hypot(
                            d.qpos[0] - start_xy[0], d.qpos[1] - start_xy[1])),
                        "yaw_err_rad": abs(wrap_angle(yaw - start_yaw)),
                    })

            # ----------------- fall detection (unified) --------------------
            reason = None
            if gxy > TILT_FALL_GXY:
                reason = "tilt"
            elif base_z < h_c - HEIGHT_FALL_MARGIN:
                reason = "low_height"
            elif nonfoot_ground:
                reason = "ground_contact"
            nonfoot_ground = False
            if reason is not None:
                fall = True
                fall_time, fall_phase, fall_reason = t, phase, reason
                fall_cycle = cyc
                fall_cmd_h, fall_base_z_v = h_c, base_z
                break

            if (test in ("circle_pillar", "circle_pillar_psi0")
                    and len(laps) >= CIRCLE_LAPS):
                break
            if mission is not None and mission.finished:
                break
            if pick_mission is not None and pick_mission.finished:
                break
    except Exception as e:  # noqa: BLE001 — record per-trial, don't kill batch
        harness_error = "exception: %s" % e
    finally:
        if writer is not None:
            writer.close()

    # ------------------ per-test metrics -----------------------------------
    ts = np.array(rec["t"]) if rec["t"] else np.zeros(0)
    seg = ts >= SETTLE_S
    metrics = {}

    def rmse(a, b):
        if a.size == 0:
            return None
        return float(np.sqrt(np.mean((a - b) ** 2)))

    if seg.any():
        metrics["max_tilt_rad"] = float(np.max(np.array(rec["tilt"])[seg]))
    else:
        metrics["max_tilt_rad"] = None

    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        metrics["cmd_vx"] = vx_target
        v = np.array(rec["v_fwd"])
        c = np.array(rec["cmd_vx"])
        metrics["vx_rmse"] = rmse(c[seg], v[seg]) if seg.any() else None
        win = ts >= (duration - WALK_LAST_WINDOW_S)
        metrics["mean_vx_last8s"] = float(np.mean(v[win])) if win.any() else None
        thr = (0.9 * vx_target
               if (test == "speed_sweep" or CUSTOM_VX_REL_THR) else 0.9)
        success = (not fall) and metrics["mean_vx_last8s"] is not None \
            and metrics["mean_vx_last8s"] >= thr
        if test == "walk_speed_psi0":
            metrics["box_kept"] = box_drop_time is None
            metrics["box_drop_time"] = box_drop_time
            metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                       if rec["box_dist"] else None)
            success = bool(success) and metrics["box_kept"]
    elif test == "squat_box_psi0":
        z = np.array(rec["base_z"])
        h = np.array(rec["cmd_h"])
        metrics["height_rmse"] = rmse(h[seg], z[seg]) if seg.any() else None
        cycles_t0 = SETTLE_S + rep["duration_s"]  # squat cycles start here
        if fall:
            cycles_done = int(max(0.0, fall_time - cycles_t0) // SQUAT_CYCLE_S)
        elif harness_error is not None:  # diverged mid-run: count observed
            cycles_done = int(max(0.0, t_end - cycles_t0) // SQUAT_CYCLE_S)
        else:
            cycles_done = SQUAT_CYCLES
        cycles_done = min(cycles_done, SQUAT_CYCLES)
        metrics["cycles_completed"] = cycles_done
        metrics["fall_cycle"] = fall_cycle
        metrics["pick_success"] = bool(pick_success)
        metrics["grasp_dist_left"] = grasp_dists[0] if grasp_dists else None
        metrics["grasp_dist_right"] = grasp_dists[1] if grasp_dists else None
        metrics["box_lift_height"] = ((box_z_max - box_z0)
                                      if box_z0 is not None else None)
        metrics["box_kept"] = bool(pick_success) and box_drop_time is None
        metrics["box_drop_time"] = box_drop_time
        metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                   if rec["box_dist"] else None)
        success = ((not fall) and bool(pick_success)
                   and cycles_done >= SQUAT_CYCLES and metrics["box_kept"])
    elif test == "squat_box":
        z = np.array(rec["base_z"])
        h = np.array(rec["cmd_h"])
        metrics["height_rmse"] = rmse(h[seg], z[seg]) if seg.any() else None
        if fall:
            cycles_done = int(max(0.0, (fall_time - SETTLE_S)) // SQUAT_CYCLE_S)
        else:
            cycles_done = SQUAT_CYCLES
        metrics["cycles_completed"] = min(cycles_done, SQUAT_CYCLES)
        metrics["fall_cycle"] = fall_cycle
        # box-carry v2 metrics: load goes through the arms now, so the box
        # can sag away from the torso if the arm PD cannot hold it.
        metrics["box_kept"] = box_drop_time is None
        metrics["box_drop_time"] = box_drop_time
        metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                   if rec["box_dist"] else None)
        success = (not fall) and cycles_done >= SQUAT_CYCLES
    elif test in ("circle_pillar", "circle_pillar_psi0"):
        metrics["direction"] = direction
        metrics["circle_vx"] = CIRCLE_VX
        metrics["circle_wz"] = CIRCLE_WZ
        if seg.any():
            r = np.hypot(np.array(rec["base_x"])[seg],
                         np.array(rec["base_y"])[seg])
            # nominal radius = vx/wz (1 m for the spec 0.4/0.4; follows
            # --custom-vx/--custom-wz overrides)
            radial = np.abs(r - CIRCLE_VX / CIRCLE_WZ)
            metrics["radial_err_mean"] = float(np.mean(radial))
            metrics["radial_err_max"] = float(np.max(radial))
        else:
            metrics["radial_err_mean"] = metrics["radial_err_max"] = None
        w = np.array(rec["omega_z"])
        cw = np.array(rec["cmd_wz"])
        metrics["wz_rmse"] = rmse(cw[seg], w[seg]) if seg.any() else None
        metrics["laps_completed"] = len(laps)
        for i in range(CIRCLE_LAPS):
            lp = laps[i] if i < len(laps) else None
            metrics["lap%d_pos_err_m" % (i + 1)] = lp["pos_err_m"] if lp else None
            metrics["lap%d_yaw_err_rad" % (i + 1)] = lp["yaw_err_rad"] if lp else None
        metrics["pillar_collision"] = bool(pillar_collision)
        metrics["pillar_collision_time"] = pillar_collision_time
        success = ((not fall) and (not pillar_collision)
                   and len(laps) >= CIRCLE_LAPS
                   and metrics["radial_err_mean"] is not None
                   and metrics["radial_err_mean"] <= 0.15)
        if test == "circle_pillar_psi0":
            metrics["box_kept"] = box_drop_time is None
            metrics["box_drop_time"] = box_drop_time
            metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                       if rec["box_dist"] else None)
            success = bool(success) and metrics["box_kept"]
    elif test == "squat_sweep":
        phases = np.array(rec["phase"]) if rec["phase"] else np.zeros(0)
        xs = np.array(rec["base_x"])
        ys = np.array(rec["base_y"])
        zs = np.array(rec["base_z"])
        hold = phases == "hold"
        achieved = float(zs[hold].mean()) if hold.any() else None
        if hold.any():
            # max XY drift during the bottom hold vs the first hold sample
            # = "didn't squat steadily"
            hx, hy = float(xs[hold][0]), float(ys[hold][0])
            drift_hold = float(np.hypot(xs[hold] - hx, ys[hold] - hy).max())
        else:
            drift_hold = None
        down = phases == "down"
        final = phases == "final"
        if down.any() and final.any() and not fall:
            # XY at trial end (after the rise) vs just before the squat
            px, py = float(xs[down][0]), float(ys[down][0])
            drift_total = math.hypot(float(xs[final][-1]) - px,
                                     float(ys[final][-1]) - py)
        else:
            drift_total = None
        metrics["sweep"] = sweep
        metrics["target_height"] = target_height
        metrics["ramp_speed"] = ramp_speed
        metrics["init_yaw_rad"] = round(yaw0, 4)
        metrics["achieved_depth"] = achieved
        metrics["depth_err"] = (achieved - target_height
                                if achieved is not None else None)
        metrics["root_drift_hold"] = drift_hold
        metrics["root_drift_total"] = drift_total
        metrics["box_kept"] = box_drop_time is None
        metrics["box_drop_time"] = box_drop_time
        metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                   if rec["box_dist"] else None)
        success = (not fall) and metrics["box_kept"]
    elif test == "squat_limit":
        # T7 (spec v4): descent-limit calibration. The raw 0.05 m/s ramp is
        # sent all the way to 0.10 m with NO [0.24, 0.74] clip; these
        # metrics locate where tracking saturates / the root drifts / the
        # policy falls. drift is measured vs the descent-start XY.
        xs = np.array(rec["base_x"])
        ys = np.array(rec["base_y"])
        zs = np.array(rec["base_z"])
        hs = np.array(rec["cmd_h"])
        tl = np.array(rec["tilt"])
        if seg.any():
            idx = np.where(seg)[0]
            x0, y0 = float(xs[idx[0]]), float(ys[idx[0]])
            drift = np.hypot(xs - x0, ys - y0)
            # raw height-tracking-drift curve, 50 Hz ticks -> 10 Hz
            metrics["descent_curve"] = [
                [round(float(hs[i]), 4), round(float(zs[i]), 4),
                 round(float(drift[i]), 4)]
                for i in idx[::SQUAT_LIMIT_CURVE_DECIM]]
            # tracking saturation: actual stuck ABOVE the descending cmd
            sat = idx[(zs[idx] - hs[idx]) > SQUAT_LIMIT_SAT_DEV_M]
            metrics["track_sat_h"] = float(hs[sat[0]]) if sat.size else None
            d5 = idx[drift[idx] > SQUAT_LIMIT_DRIFT5_M]
            metrics["drift5_h"] = float(hs[d5[0]]) if d5.size else None
            d20 = idx[drift[idx] > SQUAT_LIMIT_DRIFT20_M]
            metrics["drift20_h"] = float(hs[d20[0]]) if d20.size else None
            metrics["max_drift_xy"] = float(drift[idx].max())
            # depth_floor: min base_z while still STABLE (upright, and if
            # the trial fell, excluding the collapse window before it)
            stable = seg & (tl < SQUAT_LIMIT_STABLE_TILT)
            if fall:
                stable &= ts <= fall_time - SQUAT_LIMIT_FALL_EXCLUDE_S
            metrics["depth_floor"] = (float(zs[stable].min())
                                      if stable.any() else None)
        else:
            metrics["descent_curve"] = []
            metrics["track_sat_h"] = None
            metrics["drift5_h"] = None
            metrics["drift20_h"] = None
            metrics["max_drift_xy"] = None
            metrics["depth_floor"] = None
        metrics["fall_h_cmd"] = fall_cmd_h if fall else None
        metrics["fall_base_z"] = fall_base_z_v if fall else None
        metrics["box_kept"] = box_drop_time is None
        metrics["box_drop_time"] = box_drop_time
        metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                   if rec["box_dist"] else None)
        # spec v4: success = no fall (reached the 0.10 command and held);
        # box_kept recorded but not gating. Expected outcome for most
        # models is "no fall but saturated" — that boundary is the point.
        success = not fall
    elif test == "squat_place_psi0":
        # T8 (spec v4): lower-body error while the real_ep053 Psi0 stream
        # drives the upper body. Segments (descent/hold/rise) come from
        # the replay height_cmd profile, precomputed at npz load.
        phases = np.array(rec["phase"]) if rec["phase"] else np.zeros(0)
        xs = np.array(rec["base_x"])
        ys = np.array(rec["base_y"])
        zs = np.array(rec["base_z"])
        hs = np.array(rec["cmd_h"])
        for name in ("descent", "hold", "rise"):
            mask = phases == name
            metrics["height_rmse_%s" % name] = (
                rmse(hs[mask], zs[mask]) if mask.any() else None)
        # root XY drift during the reach-forward + place segment (vs the
        # replay-start XY) — forward CoM shift is the decoupling stressor
        reach = (phases == "descent") | (phases == "hold")
        if reach.any():
            rx, ry = float(xs[reach][0]), float(ys[reach][0])
            metrics["root_drift_place"] = float(
                np.hypot(xs[reach] - rx, ys[reach] - ry).max())
        else:
            metrics["root_drift_place"] = None
        metrics["release_t"] = place_release_t
        if box_drop_time is None:
            kept = True
        else:   # the box leaving the torso after the release is expected
            kept = (place_release_t is not None
                    and box_drop_time >= place_release_t - 0.05)
        metrics["box_kept_until_release"] = bool(kept)
        fb_valid = (not fall and harness_error is None and place_released
                    and place_frame is not None)
        if fb_valid:
            vbox = float(np.linalg.norm(
                d.qvel[box["vadr"]:box["vadr"] + 3]))
            zz = float(d.xmat[box["body"]].reshape(3, 3)[2, 2])
            btilt = math.acos(max(-1.0, min(1.0, zz)))
            bx, by, bz = (float(v) for v in d.xpos[box["body"]])
            hx, hy, front = place_frame
            land_dx = bx * hx + by * hy - front
            metrics["box_land_dx"] = land_dx
            metrics["box_speed_end"] = vbox
            metrics["box_tilt_end_rad"] = btilt
            metrics["box_z_end"] = bz
            # placed OK: at rest, upright, on the floor, IN FRONT of the
            # release-time foot front edge (land_dx > 0)
            box_place_ok = bool(vbox < BOX_PLACE_MAX_SPEED
                                and btilt < BOX_PLACE_UPRIGHT_TOL
                                and bz < PLACE_BOX_GROUND_Z_MAX
                                and land_dx > 0.0)
        else:
            metrics["box_land_dx"] = None
            metrics["box_speed_end"] = None
            metrics["box_tilt_end_rad"] = None
            metrics["box_z_end"] = None
            box_place_ok = False
        metrics["box_place_ok"] = box_place_ok
        endm = phases == "end_stand"
        metrics["stand_ok"] = bool(
            (not fall) and harness_error is None and endm.any()
            and float(np.array(rec["tilt"])[endm].max()) < STAND_OK_TILT_RAD)
        success = (not fall) and box_place_ok
    elif test == "squat_pick_ground":
        # T9 (spec v5): pick a 2 kg box off the floor. Whether the model
        # can squat deep enough for the wrists to reach the box IS the
        # measurement — min_root_z is the headline depth number.
        phases = np.array(rec["phase"]) if rec["phase"] else np.zeros(0)
        xs = np.array(rec["base_x"])
        ys = np.array(rec["base_y"])
        zs = np.array(rec["base_z"])
        metrics["pick_success"] = bool(pick_success)
        metrics["pick_timeout"] = bool(pick_mission.pick_failed)
        metrics["t_grasp"] = pick_mission.grasp_t
        metrics["grasp_dist_left"] = grasp_dists[0] if grasp_dists else None
        metrics["grasp_dist_right"] = grasp_dists[1] if grasp_dists else None
        metrics["min_root_z"] = (float(zs[seg].min()) if seg.any() else None)
        # root XY drift across the bottom wait + grasp hold (vs the first
        # bottom sample) — "didn't squat steadily" indicator
        bottom = (phases == "bottom_wait") | (phases == "grasp_hold")
        if bottom.any():
            bx0, by0 = float(xs[bottom][0]), float(ys[bottom][0])
            metrics["root_drift_bottom"] = float(
                np.hypot(xs[bottom] - bx0, ys[bottom] - by0).max())
        else:
            metrics["root_drift_bottom"] = None
        endm = phases == "stand"
        metrics["stand_ok"] = bool(
            (not fall) and harness_error is None and endm.any()
            and float(np.array(rec["tilt"])[endm].max()) < STAND_OK_TILT_RAD)
        # box_dist tracking starts at the grasp; recorded, not gating
        metrics["box_kept"] = bool(pick_success) and box_drop_time is None
        metrics["box_drop_time"] = box_drop_time
        metrics["box_dist_max"] = (float(np.max(rec["box_dist"]))
                                   if rec["box_dist"] else None)
        metrics["init_yaw_rad"] = round(yaw0, 4)
        success = ((not fall) and bool(pick_success)
                   and metrics["stand_ok"])
    elif test == "vln_follow":
        # T10 (spec v5): all errors in the TAPE-START frame — the robot
        # pose at the first follow tick is the tape origin, so settle
        # drift and the random initial yaw cancel out; the ref trajectory
        # is the ideal error-free integral of the same commands.
        metrics["tape_id"] = tape["id"]
        phases = np.array(rec["phase"]) if rec["phase"] else np.zeros(0)
        xs = np.array(rec["base_x"])
        ys = np.array(rec["base_y"])
        yaws = np.array(rec["yaw"]) if rec["yaw"] else np.zeros(0)
        follow = phases == "follow"
        if follow.any():
            i0 = int(np.where(follow)[0][0])
            x0, y0, yaw_s = float(xs[i0]), float(ys[i0]), float(yaws[i0])
            cs, sn = math.cos(yaw_s), math.sin(yaw_s)
            rel_x = cs * (xs - x0) + sn * (ys - y0)
            rel_y = -sn * (xs - x0) + cs * (ys - y0)
            rel_yaw = (yaws - yaw_s + math.pi) % (2.0 * math.pi) - math.pi
            metrics["final_pos_err"] = float(math.hypot(
                rel_x[-1] - tape["ref_xy"][-1, 0],
                rel_y[-1] - tape["ref_xy"][-1, 1]))
            metrics["final_yaw_err_rad"] = abs(wrap_angle(
                float(rel_yaw[-1]) - float(tape["ref_yaw"][-1])))
            # mean 1 Hz track error: nearest rec tick per ref sample (a
            # fallen/truncated trial only counts the samples it reached)
            errs = []
            for ri in range(tape["ref_t"].shape[0]):
                ta = SETTLE_S + float(tape["ref_t"][ri])
                j = int(np.searchsorted(ts, ta))
                j = min(max(j, 0), len(ts) - 1)
                if j > 0 and abs(ts[j - 1] - ta) < abs(ts[j] - ta):
                    j -= 1
                if abs(ts[j] - ta) <= VLN_TRACK_MATCH_TOL_S:
                    errs.append(math.hypot(
                        rel_x[j] - float(tape["ref_xy"][ri, 0]),
                        rel_y[j] - float(tape["ref_xy"][ri, 1])))
            metrics["mean_track_err"] = (float(np.mean(errs))
                                         if errs else None)
            metrics["track_err_n"] = len(errs)
            # small-command response: actual/commanded forward speed in
            # the |cmd vx| in [0.05, 0.15] band (signed ratio; 1.0 = ideal,
            # <1 under-tracks, <0 wrong direction)
            cv = np.array(rec["cmd_vx"])
            v = np.array(rec["v_fwd"])
            m_small = (follow & (np.abs(cv) >= VLN_SMALL_VX_LO)
                       & (np.abs(cv) <= VLN_SMALL_VX_HI))
            metrics["small_cmd_response"] = (
                float(np.mean(v[m_small] / cv[m_small]))
                if m_small.any() else None)
            # stop settle: residual XY displacement inside each full-stop
            # segment after a VLN_STOP_SKIP_S deceleration skip
            disps = []
            for t0s, t1s in tape["stops"]:
                w = ((ts >= SETTLE_S + t0s + VLN_STOP_SKIP_S)
                     & (ts <= SETTLE_S + t1s))
                wi = np.where(w)[0]
                if wi.size >= 2:
                    disps.append(math.hypot(
                        float(xs[wi[-1]]) - float(xs[wi[0]]),
                        float(ys[wi[-1]]) - float(ys[wi[0]])))
            metrics["stop_settle"] = (float(np.mean(disps))
                                      if disps else None)
            metrics["n_stops_measured"] = len(disps)
        else:
            metrics["final_pos_err"] = None
            metrics["final_yaw_err_rad"] = None
            metrics["mean_track_err"] = None
            metrics["track_err_n"] = 0
            metrics["small_cmd_response"] = None
            metrics["stop_settle"] = None
            metrics["n_stops_measured"] = 0
        success = ((not fall)
                   and metrics["final_pos_err"] is not None
                   and metrics["final_pos_err"] <= VLN_FINAL_POS_TOL
                   and metrics["final_yaw_err_rad"] is not None
                   and metrics["final_yaw_err_rad"] <= VLN_FINAL_YAW_TOL)
    elif test == "goto_ab":
        # T5: the project's direct measurement of whether small-command
        # calibration converges. Headline success = success_fine + upright.
        fill_nav_metrics(metrics, mission)
        metrics["init_yaw_rad"] = round(yaw0, 4)
        success = (not fall) and bool(metrics["success_fine"])
    elif test == "pipeline_abc":
        # T6: nav metrics + squat/place/rise outcome + placement quality.
        # success := success_coarse AND no fall AND box_place_ok (fine
        # convergence recorded but not gating, per spec).
        fill_nav_metrics(metrics, mission)
        squat_phases = ("squat_down", "squat_hold", "place_hold",
                        "squat_up", "end_stand")
        metrics["squat_fall"] = (fall_phase
                                 if fall and fall_phase in squat_phases
                                 else None)
        release_t = mission.release_t
        metrics["release_t"] = release_t
        if box_drop_time is None:
            kept = True
        else:  # after the release the box leaving the torso is expected
            kept = release_t is not None and box_drop_time >= release_t - 0.05
        metrics["box_kept_until_release"] = bool(kept)
        fb = None
        if mission.finished and not fall and harness_error is None:
            vbox = float(np.linalg.norm(
                d.qvel[box["vadr"]:box["vadr"] + 3]))
            zz = float(d.xmat[box["body"]].reshape(3, 3)[2, 2])
            fb = {
                "speed": vbox,
                "tilt": math.acos(max(-1.0, min(1.0, zz))),
                "x": float(d.xpos[box["body"]][0]),
                "y": float(d.xpos[box["body"]][1]),
            }
            fb["dist"] = math.hypot(fb["x"] - float(d.qpos[0]),
                                    fb["y"] - float(d.qpos[1]))
        box_place_ok = bool(
            fb is not None
            and fb["speed"] < BOX_PLACE_MAX_SPEED
            and fb["tilt"] < BOX_PLACE_UPRIGHT_TOL
            and fb["dist"] < BOX_PLACE_MAX_DIST)
        metrics["box_place_ok"] = box_place_ok
        if fb is not None:
            metrics["box_land_pos"] = [fb["x"] - PIPE_C_XY[0],
                                       fb["y"] - PIPE_C_XY[1]]
            metrics["box_land_dist_from_c"] = math.hypot(
                *metrics["box_land_pos"])
            metrics["box_speed_end"] = fb["speed"]
            metrics["box_tilt_end_rad"] = fb["tilt"]
            metrics["box_dist_to_robot_end"] = fb["dist"]
        else:
            metrics["box_land_pos"] = None
            metrics["box_land_dist_from_c"] = None
        metrics["total_t"] = round(t_end, 3)
        success = (bool(metrics["success_coarse"]) and (not fall)
                   and box_place_ok)
    else:
        raise ValueError(test)

    metrics["sim_time_s"] = t_end
    metrics["wall_time_s"] = round(time.time() - wall_t0, 2)

    if harness_error is not None:
        success = False

    result = {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial_idx,
        "seed": seed,
        "success": bool(success),
        "fall": bool(fall),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "fall_reason": fall_reason,
        "harness_error": harness_error,
        "metrics": metrics,
    }
    if test in BOX_TESTS:
        result["box_mode"] = BOX_MODE
    if test in PSI0_TESTS:
        result["upper_mode"] = "psi0_replay_%s" % rep["source"]
    return result


# ===========================================================================
# ManipArena v2 (BENCHMARK_V2_DESIGN §2-§4). Reference HOMIE adapter:
#   build_arena_model -> inject ms.furniture_xml into g1.xml
#   configure_arena_welds -> resolve free-body addresses + weld eq ids
#   HomieMissionIO(MissionIO) -> the ~13 callbacks manip_mission.Mission drives
#   run_arena_trial -> spawn at the spec pose, drive the mission, score it
# ===========================================================================
def reach_base_h(top_h):
    """HOMIE absolute base-height command to bring the wrists to a surface at
    height ``top_h`` (table top / store-pad floor). Linear "lower surface ->
    lower base" with a fixed reach offset, clipped to the trackable HOMIE
    domain [ARENA_H_MIN, ARENA_STAND_H]. The mission emits this as its squat
    target; the depth floor of the policy decides whether the low tables are
    reachable at all (the whole point of the H_pick sweep)."""
    return float(min(max(top_h + ARENA_REACH_OFFSET, ARENA_H_MIN),
                     ARENA_STAND_H))


def build_arena_model(repo, spec):
    """Compile g1.xml with the ManipArena furniture + free bodies injected.

    Mechanics (mirrors build_model's bit-2 path, BENCHMARK_V2_DESIGN §3):
      - furniture bodies appended AFTER all robot bodies (at FLOOR_ANCHOR) so
        the box/cube freejoints land at the qpos/qvel TAIL
        (FREE_BODY_QPOS_LAYOUT); the robot keeps qpos[0:34]/qvel[0:33] and the
        HOMIE obs slice (qpos[7:34]/qvel[6:33]) is never polluted.
      - the <equality> weld block inserted before </mujoco>.
      - ms.WRIST_L/WRIST_R placeholders replaced with the HOMIE wrist link
        names so the wrist<->box welds compile against real bodies.
      - runtime: floor planes (+ world body) get bit 2 OR-ed in so the box/cube
        collide with the ground (same bit-2 mechanism as BIT2_FLOOR_TESTS).
    """
    xml_path = os.path.join(repo, XML_REL)
    with open(xml_path, "r") as f:
        xml = f.read()
    mesh_abs = os.path.join(os.path.dirname(xml_path), "meshes")
    xml = patch_and_replace(xml, 'meshdir="meshes"', 'meshdir="%s"' % mesh_abs)

    frags = ms.furniture_xml(spec, collision_scheme="bit2")
    xml = patch_and_replace(xml, FLOOR_ANCHOR, FLOOR_ANCHOR + frags["bodies"])
    xml = patch_and_replace(xml, "</mujoco>", frags["equality"] + "</mujoco>")
    # Substitute the wrist placeholders AFTER injecting the weld block (the
    # placeholders only live inside that block).
    xml = xml.replace(ms.WRIST_L_PLACEHOLDER, ARENA_WRIST_L)
    xml = xml.replace(ms.WRIST_R_PLACEHOLDER, ARENA_WRIST_R)

    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = SIM_DT
    # bit-2 floor patch: box/cube (contype 2) must collide with the ground.
    for g in range(model.ngeom):
        if (model.geom_bodyid[g] == 0
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE):
            model.geom_contype[g] |= 2
            model.geom_conaffinity[g] |= 2
    if hasattr(model, "body_contype"):
        model.body_contype[0] |= 2
        model.body_conaffinity[0] |= 2
    return model


def configure_arena_welds(model):
    """Resolve the arena free-body addresses + weld equality ids. Unlike the
    carry-box welds, arena welds are NOT pre-anchored at a hug pose — they are
    re-anchored at the live wrist<->box pose at grasp time
    (activate_welds_at_current_pose-style), so here we only look up ids and
    confirm the tail layout matches FREE_BODY_QPOS_LAYOUT."""
    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    box_qadr, box_vadr = jq("carried_box_freejoint")
    cube_qadr, cube_vadr = jq("relay_cube_freejoint")
    lay = ms.FREE_BODY_QPOS_LAYOUT
    # robot end = 34/33 for HOMIE; box then cube at the tail (§FREE_BODY layout)
    assert box_qadr == 34 and cube_qadr == 41, \
        "arena free-body tail layout unexpected: box_qadr=%d cube_qadr=%d" \
        % (box_qadr, cube_qadr)
    assert (cube_qadr - box_qadr == lay["cube_qpos_offset_from_robot_end"]), \
        "cube qpos offset mismatch vs FREE_BODY_QPOS_LAYOUT"

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def eid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    return {
        "box_qadr": box_qadr, "box_vadr": box_vadr,
        "cube_qadr": cube_qadr, "cube_vadr": cube_vadr,
        "box_body": bid("carried_box"),
        "cube_body": bid("relay_cube"),
        "wrist_left": bid(ARENA_WRIST_L),
        "wrist_right": bid(ARENA_WRIST_R),
        "box_geom": gid("carried_box_geom"),
        "cube_geom": gid("relay_cube_geom"),
        "eq_box_left": eid("box_weld_left"),
        "eq_box_right": eid("box_weld_right"),
        "eq_cube_box": eid("cube_weld_box"),
        "eq_cube_right": eid("cube_weld_right"),
    }


def _arena_floor_bit2(m, body_id):
    """OR bit 2 into a free body + the floor planes + world body so the body
    collides with the ground (carries the §3 release semantics)."""
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


def _weld_at_pose(m, d, eq, body1, body2):
    """Activate weld ``eq`` anchored at the CURRENT relative body2-in-body1
    pose (no body moved). Identical math to activate_welds_at_current_pose,
    generalised to any body1/body2 pair (wrist<->box, box<->cube)."""
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


def _snap_box_upright(m, d, box, base_yaw):
    """ROOT CAUSE 2 fix. Before welding, move the carried box into a canonical
    UPRIGHT pose: position = midpoint of the two wrists (centred in the hands),
    orientation = pure yaw locked to the robot base (box z-axis == world z). This
    makes the two wrist<->box welds anchor at mutually-CONSISTENT relposes (the
    original divergence/topple came from anchoring them at a skewed live box
    pose) and guarantees the box rides upright so it lands level on release.
    Writes the box freejoint qpos + zeros its qvel, then mj_forward so the welds
    anchor against the corrected pose. Mutates d only."""
    bq = box["box_qadr"]
    bv = box["box_vadr"]
    mid = 0.5 * (d.xpos[box["wrist_left"]] + d.xpos[box["wrist_right"]])
    d.qpos[bq:bq + 3] = mid
    d.qpos[bq + 3:bq + 7] = [math.cos(base_yaw / 2.0), 0.0, 0.0,
                             math.sin(base_yaw / 2.0)]
    d.qvel[bv:bv + 6] = 0.0
    mujoco.mj_forward(m, d)


def _weld_box_both(m, d, box):
    """Activate BOTH wrist<->box welds, anchored at the CANONICAL UPRIGHT box
    pose (box centred between the wrists, z up — see _snap_box_upright). The
    original two-weld carry is stable (it carried the box fine across all table
    heights); the only defect was the SKEWED anchor pose that toppled the box on
    release, which the upright snap now fixes. Anchoring both welds at the
    symmetric upright pose makes their relposes mutually consistent (no fighting)
    and shares the 2 kg load between the two arms. solref/solimp left at the
    compliant XML defaults (0.05 1 / 0.8 0.95 ...) — overriding them
    destabilised the stand-up."""
    _weld_at_pose(m, d, box["eq_box_right"], box["wrist_right"],
                  box["box_body"])
    _weld_at_pose(m, d, box["eq_box_left"], box["wrist_left"],
                  box["box_body"])


class HomieMissionIO:
    """MissionIO (manip_mission.py Protocol) bound to a live HOMIE MjModel/
    MjData. The mission owns sequencing + per-stage metrics; this adapter owns
    the physics side effects. Per policy tick the rollout does:

        cmd = mission.step(t)          # mission reads THIS io, fires welds, ..
        io.set_command(*cmd)           # store (vx,vy,wz,height) for the policy
        <rollout builds obs from io.cmd, runs the onnx policy, steps sim>

    Heights are HOMIE-native ABSOLUTE base z; wz is the native yaw-rate channel
    (under-tracks ~x0.6-0.8, measured as-is). The upper body is driven by a
    real_ep053 Psi0 replay (set via replay_upper); the rollout reads
    io.upper_target each tick and runs the existing direct upper-body PD."""

    def __init__(self, m, d, box, ids, replay, h_stand=ARENA_STAND_H):
        self.m = m
        self.d = d
        self.box = box                  # configure_arena_welds() dict
        self.ids = ids                  # lookup_ids() dict (foot bodies, torso)
        self.replay = replay            # prepared real_ep053 dict (pos15 etc.)
        self.h_stand = h_stand
        # mutable slots the rollout reads:
        self.cmd = (0.0, 0.0, 0.0, h_stand)        # vx, vy, wz, height
        self.upper_target = (replay["pos15"][0] if replay is not None
                             else ZERO_ARM_POSE.copy())
        self.box_grasped = False
        self.cube_on_box = False
        self._fallen = False            # latched by the rollout each tick

    # --- rollout-facing helpers (NOT part of MissionIO) ------------------
    def set_command(self, vx, vy, wz, height):
        self.cmd = (float(vx), float(vy), float(wz), float(height))

    def set_fallen(self, flag):
        self._fallen = bool(flag)

    # --- MissionIO: read state -------------------------------------------
    def root_pose(self):
        return (float(self.d.qpos[0]), float(self.d.qpos[1]),
                yaw_from_quat(self.d.qpos[3:7]))

    def root_tilt(self):
        grav = gravity_body_frame(self.d.qpos[3:7])
        return float(math.hypot(grav[0], grav[1]))

    def wrist_box_dist(self):
        d_l = box_surface_dist(self.d, {"body": self.box["box_body"]},
                               self.box["wrist_left"],
                               half_extents=ARENA_BOX_HALF)
        d_r = box_surface_dist(self.d, {"body": self.box["box_body"]},
                               self.box["wrist_right"],
                               half_extents=ARENA_BOX_HALF)
        return (d_l, d_r)

    def right_wrist_cube_dist(self):
        return box_surface_dist(self.d, {"body": self.box["cube_body"]},
                                self.box["wrist_right"],
                                half_extents=ARENA_CUBE_HALF)

    def box_pose(self):
        b = self.box["box_body"]
        bx, by, bz = (float(v) for v in self.d.xpos[b])
        zz = float(self.d.xmat[b].reshape(3, 3)[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        speed = float(np.linalg.norm(
            self.d.qvel[self.box["box_vadr"]:self.box["box_vadr"] + 3]))
        return (bx, by, bz, tilt, speed)

    def cube_pose(self):
        c = self.box["cube_body"]
        cx, cy, cz = (float(v) for v in self.d.xpos[c])
        zz = float(self.d.xmat[c].reshape(3, 3)[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        speed = float(np.linalg.norm(
            self.d.qvel[self.box["cube_vadr"]:self.box["cube_vadr"] + 3]))
        return (cx, cy, cz, tilt, speed)

    def base_height(self):
        return float(self.d.qpos[2])

    def is_fallen(self):
        return self._fallen

    # --- MissionIO: act (lower body) -------------------------------------
    def send_leg_cmd(self, vx, vy, wz, height):
        self.set_command(vx, vy, wz, height)

    # --- MissionIO: act (upper body, Psi0 replay) ------------------------
    def replay_upper(self, segment, phase_t):
        """Drive the upper body from the real_ep053 stream for ``segment``
        ('reach_down' / 'carry' / 'place'). The harness owns the segment->frame
        mapping; the mission only names the segment + local time. No replay
        loaded -> hold the carry/zero pose (defensive; M-series always has
        one)."""
        rep = self.replay
        if rep is None:
            return
        seg = rep.get("arena_segments", {}).get(segment)
        if seg is None:
            self.upper_target = rep["pos15"][0]
            return
        k0, k1 = seg                    # [k0, k1) frame window for this segment
        n = max(k1 - k0, 1)
        k = k0 + int(min(max(phase_t / POLICY_DT, 0.0), n - 1))
        k = min(max(k, 0), rep["n"] - 1)
        self.upper_target = rep["pos15"][k]

    # --- MissionIO: act (weld / release / cube-transfer) -----------------
    def weld_box(self):
        """Magnetic grasp (ROOT CAUSE 2 fix). First SNAP the box to a canonical
        upright pose (z-axis up, yaw == base yaw, centred between the wrists) so
        it is never grasped skewed and lands upright on release. Then weld BOTH
        wrists to the box at that consistent symmetric pose (the original
        two-weld carry is stable; the upright snap removes the skew that toppled
        the box). Re-enable on a re-grasp too."""
        base_yaw = yaw_from_quat(self.d.qpos[3:7])
        _snap_box_upright(self.m, self.d, self.box, base_yaw)
        _weld_box_both(self.m, self.d, self.box)
        # While HELD the box must collide with NOTHING (contype/conaffinity 0):
        # it is snapped to the wrist midpoint, which during a deep reach can
        # overlap the pick TABLE (bits 1|2) — leaving the box on bit 2 smashed
        # it into the table and spiked qacc when standing up (H0.45 divergence).
        # Collision is re-enabled (bit 2) only at release.
        self.m.geom_contype[self.box["box_geom"]] = 0
        self.m.geom_conaffinity[self.box["box_geom"]] = 0
        if hasattr(self.m, "body_contype"):
            self.m.body_contype[self.box["box_body"]] = 0
            self.m.body_conaffinity[self.box["box_body"]] = 0
        self.box_grasped = True

    def release_box(self, surface_top_h=None, target_xy=None):
        """Set the box down + drop the welds + enable box ground collision
        (bit-2 scheme: box collides with the floor/furniture, never the bit-1
        robot).

        ROOT CAUSE 2 (landing upright + on target): the box is rigidly welded to
        the wrists, so by release time it has PITCHED with the arms through the
        squat AND sits wherever the 'place' Psi0 arm pose left it (often behind /
        beside the pelvis, not a clean forward offset). Modelling a deliberate
        set-down, we snap the box to a clean rest pose at release: orientation
        leveled upright (yaw == base yaw); xy at PELVIS + a fixed forward release
        offset along the heading (so the LANDING ERROR still reflects how
        accurately the robot positioned itself — a sloppy placement pose still
        misses the zone); z resting just above the surface so it lands flat at
        rest. The cube weld (if active) stays — the cube rides the box down."""
        bq = self.box["box_qadr"]
        bv = self.box["box_vadr"]
        base_yaw = yaw_from_quat(self.d.qpos[3:7])
        px, py = float(self.d.qpos[0]), float(self.d.qpos[1])
        # Set the box down AHEAD of the pelvis toward the placement target. The
        # placement legs are coarse_only (the fine CAL topples the carry stance),
        # so the robot stops ~0.30 m short; releasing only a fixed offset ahead
        # dropped the box on the near RIM of the zone, which toppled it. Instead
        # set it down where the pelvis would reach if it advanced the full
        # forward offset, but CLAMP the landing inside the target footprint
        # (radius) so a small coarse undershoot still lands the box cleanly in
        # the zone rather than on the rim. Larger nav errors still miss.
        if target_xy is not None:
            dx, dy = target_xy[0] - px, target_xy[1] - py
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dist, dy / dist
            reach = min(ARENA_RELEASE_FWD_M, dist)
            bx = px + reach * ux
            by = py + reach * uy
            # clamp toward the target so the box never lands on the rim
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
            # box centre = surface + half height + a hair, so it lands flat
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
        """Weld the cube to the box top at the live relative pose (cube rides
        the box thereafter). Also makes the cube collide via bit-2 so it never
        free-falls through a placed box later."""
        _weld_at_pose(self.m, self.d, self.box["eq_cube_box"],
                      self.box["box_body"], self.box["cube_body"])
        self.m.geom_contype[self.box["cube_geom"]] = ms.FREE_BODY_CT_CA
        self.m.geom_conaffinity[self.box["cube_geom"]] = ms.FREE_BODY_CT_CA
        self.cube_on_box = True


def prepare_arena_segments(rep):
    """Label the real_ep053 replay into reach_down / carry / place windows the
    M-series segments index into. Reuses the squat_place_psi0 segmentation
    (descent/hold/rise around argmin(height_cmd)): 'reach_down' = the descent +
    hold (lower toward the surface), 'place' = hold + rise (set down + back up),
    'carry' = the standing-tall frames (height near the max). Returns a NEW dict
    with arena_segments={name:(k0,k1)} added."""
    seg = prepare_place_segments(rep)          # adds place_k / seg_label
    n = seg["n"]
    k_low = int(np.argmin(seg["height"]))      # deepest reach frame
    labels = seg["seg_label"]                  # per-frame descent/hold/rise
    first_hold = labels.index("hold") if "hold" in labels else k_low
    # reach_down: the lowering motion (descent up to the bottom hold) — the
    #   bend/squat toward a surface.
    # place: the set-down + retract (bottom hold through the rise back up).
    # carry: a short standing-tall window (the rise tail) the robot holds while
    #   walking between legs. All windows clamped to valid [0, n).
    reach = (0, max(first_hold, 1))
    place = (first_hold, n)
    carry = (max(n - max(1, n // 10), 0), n)
    return dict(seg, arena_segments={
        "reach_down": reach, "place": place, "carry": carry},
        seg_bottom_k=k_low)


# ---------------------------------------------------------------------------
# Pillar-avoidance: insert a lateral skirt waypoint into a nav leg whose
# straight segment passes too close to the pillar (BENCHMARK_V2_DESIGN M1/M2
# "nav ... 避 pillar"). Implemented HARNESS-side: it rewrites the mission's
# NavLeg list before the Mission runs, so the shared mission engine stays
# physics-free and the math (nav_errors/coarse/cal) is untouched.
# ---------------------------------------------------------------------------
def _seg_point_dist(p0, p1, c):
    """Min distance from point c to the segment p0->p1 (all 2-tuples)."""
    ax, ay = p0
    bx, by = p1
    cx, cy = c
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(cx - ax, cy - ay)
    tparam = ((cx - ax) * dx + (cy - ay) * dy) / denom
    tparam = min(max(tparam, 0.0), 1.0)
    px, py = ax + tparam * dx, ay + tparam * dy
    return math.hypot(cx - px, cy - py)


def insert_pillar_detours(sequence, spawn_xy, pillar_xy,
                          clear_m=ARENA_PILLAR_CLEAR_M,
                          side_m=ARENA_PILLAR_SIDE_M):
    """Return a NEW primitive list where any NavLeg whose straight path from the
    previous waypoint passes within ``clear_m`` of the pillar is preceded by a
    skirt NavLeg offset to the side of the segment (away from the pillar). Pure
    geometry on the leg target poses; immutable (builds a fresh list)."""
    out = []
    prev_xy = tuple(spawn_xy)
    for prim in sequence:
        if isinstance(prim, mm.NavLeg):
            tgt_xy = (prim.target_pose[0], prim.target_pose[1])
            dist = _seg_point_dist(prev_xy, tgt_xy, pillar_xy)
            if dist < clear_m and not getattr(prim, "skip_detour", False):
                # midpoint, pushed perpendicular to the segment AWAY from the
                # pillar so the robot skirts it.
                mx = 0.5 * (prev_xy[0] + tgt_xy[0])
                my = 0.5 * (prev_xy[1] + tgt_xy[1])
                sx, sy = tgt_xy[0] - prev_xy[0], tgt_xy[1] - prev_xy[1]
                slen = math.hypot(sx, sy) or 1.0
                # left-normal of the heading:
                nx, ny = -sy / slen, sx / slen
                # choose the side that moves the midpoint AWAY from the pillar
                if ((mx + nx - pillar_xy[0]) ** 2
                        + (my + ny - pillar_xy[1]) ** 2) < \
                   ((mx - nx - pillar_xy[0]) ** 2
                        + (my - ny - pillar_xy[1]) ** 2):
                    nx, ny = -nx, -ny
                wp = (mx + nx * side_m, my + ny * side_m,
                      math.atan2(tgt_xy[1] - (my + ny * side_m),
                                 tgt_xy[0] - (mx + nx * side_m)))
                out.append(mm.NavLeg(wp, vx_max=prim.vx_max,
                                     mode="coarse_only"))
            out.append(prim)
            prev_xy = tgt_xy
        else:
            out.append(prim)
    return out


# ---------------------------------------------------------------------------
# Arena rollout (one variant trial). Reuses the standard obs/PD/policy/EGL/
# fall machinery; the locomotion command + upper-body target come from the
# Mission/HomieMissionIO instead of command_at.
# ---------------------------------------------------------------------------
def run_arena_trial(ctx, spec, trial_idx, seed, video_path=None):
    """Run M1/M2 on the ManipArena variant ``spec``. Deterministic given seed.
    Returns a result dict (mission block + scene-spec provenance)."""
    test = ctx["test"]
    m = ctx["model"]
    ids = ctx["ids"]
    box = ctx["box"]
    sess = ctx["sess"]
    in_name, out_name = ctx["in_name"], ctx["out_name"]
    ang_vel_scale = ctx["ang_vel_scale"]
    rep = ctx["upper"]

    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)

    sd = spec.to_dict()
    spawn = sd["spawn"]
    yaw0 = float(spawn[2])

    # --- re-disable any box/cube collision a previous trial's place enabled --
    for body, geom in ((box["box_body"], box["box_geom"]),
                       (box["cube_body"], box["cube_geom"])):
        m.geom_contype[geom] = ms.FREE_BODY_CT_CA
        m.geom_conaffinity[geom] = ms.FREE_BODY_CT_CA
        if hasattr(m, "body_contype"):
            m.body_contype[body] = ms.FREE_BODY_CT_CA
            m.body_conaffinity[body] = ms.FREE_BODY_CT_CA
    # floor keeps bit 2 (box/cube rest on tables/floor from t=0)

    # --- initial robot state (spawn at the spec pose, +0.02 rad joint noise) -
    d.qpos[:] = 0.0
    d.qpos[0] = spawn[0]
    d.qpos[1] = spawn[1]
    d.qpos[2] = INIT_BASE_Z
    d.qpos[3:7] = [math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)]
    d.qpos[7:19] = LEG_DEFAULT + rng.uniform(-JOINT_NOISE, JOINT_NOISE, 12)
    upper_target = rep["pos15"][0]
    d.qpos[19:34] = upper_target

    # --- free bodies at the spec ground-truth poses (box on T_pick, cube on
    #     T_relay), welds OFF (XML compiles inactive). ----------------------
    bq = box["box_qadr"]
    cq = box["cube_qadr"]
    d.qpos[bq:bq + 3] = sd["box"]["pos"]
    d.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[cq:cq + 3] = sd["cube"]["pos"]
    d.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:] = 0.0
    for eq in ("eq_box_left", "eq_box_right", "eq_cube_box", "eq_cube_right"):
        d.eq_active[box[eq]] = 0
    mujoco.mj_forward(m, d)

    # --- mission + io -----------------------------------------------------
    io = HomieMissionIO(m, d, box, ids, rep, h_stand=ARENA_STAND_H)
    io.upper_target = upper_target
    h_pick = reach_base_h(sd["T_pick"]["top_h"])
    h_relay = reach_base_h(sd["T_relay"]["top_h"])
    h_store = ARENA_STORE_PLACE_H                    # shallower, stable store squat
    h_cube = reach_base_h(sd["T_relay"]["top_h"])   # cube sits on the relay top
    if test == "arena_M1":
        seq = mm.build_m1(sd, ARENA_STAND_H, h_pick, h_store,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    else:
        seq = mm.build_m2(sd, ARENA_STAND_H, h_pick, h_relay, h_store, h_cube,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    seq = insert_pillar_detours(seq, (spawn[0], spawn[1]),
                                tuple(sd["pillar"]["pos"]))
    mission = mm.Mission(test, seq, io, ARENA_STAND_H)

    # --- policy state -----------------------------------------------------
    action = np.zeros(N_ACTIONS, dtype=np.float32)
    target_dof_pos = LEG_DEFAULT.copy()
    obs_hist = collections.deque(maxlen=HIST_LEN)
    for _ in range(HIST_LEN):
        obs_hist.append(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))

    # --- video ------------------------------------------------------------
    writer = None
    cam = None
    next_frame_t = 0.0
    if video_path is not None and ctx["renderer"] is not None:
        import imageio
        writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264",
                                    quality=8)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.elevation, cam.azimuth = 4.5, -25.0, 135.0

    rec = collections.defaultdict(list)
    fall = False
    fall_time = fall_phase = fall_reason = None
    harness_error = None

    floor_g = ids["floor_geom"]
    foot_bodies = ids["foot_bodies"]
    pillar_b = ids["pillar_body"]
    box_b = box["box_body"]
    cube_b = box["cube_body"]
    nonfoot_ground = False
    t_end = 0.0
    wall_t0 = time.time()
    # max wall budget: the mission has its own MISSION_TIMEOUT_S; cap sim steps
    # generously above it so a stuck trial still terminates.
    max_steps = int(round((mm.MISSION_TIMEOUT_S + 5.0) / SIM_DT))

    try:
        for step in range(max_steps):
            d.ctrl[0:12] = ((target_dof_pos - d.qpos[7:19]) * LEG_KP
                            - d.qvel[6:18] * LEG_KD)
            d.ctrl[12:27] = ((io.upper_target - d.qpos[19:34]) * UPPER_KP
                             - d.qvel[18:33] * UPPER_KD)
            mujoco.mj_step(m, d)
            t = (step + 1) * SIM_DT
            t_end = t

            for ci in range(d.ncon):
                g1 = d.contact[ci].geom1
                g2 = d.contact[ci].geom2
                if floor_g in (g1, g2):
                    other = g2 if g1 == floor_g else g1
                    b = int(m.geom_bodyid[other])
                    if (b != 0 and b != pillar_b and b != box_b
                            and b != cube_b and b not in foot_bodies):
                        nonfoot_ground = True

            if writer is not None and t >= next_frame_t:
                cam.lookat[:] = d.qpos[0:3]
                ctx["renderer"].update_scene(d, camera=cam)
                writer.append_data(ctx["renderer"].render())
                next_frame_t += 1.0 / VIDEO_FPS

            if (step + 1) % DECIMATION != 0:
                continue

            # divergence guard (weld-vs-PD blowups)
            if (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.all(np.isfinite(d.qpos))):
                harness_error = "qacc_diverged"
                break

            # unified fall flag for THIS tick (mission consumes it via io)
            grav = gravity_body_frame(d.qpos[3:7])
            gxy = float(math.hypot(grav[0], grav[1]))
            tilt = float(math.acos(max(-1.0, min(1.0, -grav[2]))))
            fall_now = (gxy > TILT_FALL_GXY) or nonfoot_ground
            io.set_fallen(fall_now)
            nonfoot_ground = False

            # mission tick: reads io (root/wrist/box poses, fall flag), fires
            # welds/release/transfer + replay_upper as side effects, returns the
            # leg command (vx, vy, wz, height) for THIS policy tick.
            vx_c, vy_c, wz_c, h_c = mission.step(t)
            io.set_command(vx_c, vy_c, wz_c, h_c)
            phase = mission.stages[-1].name if mission.stages else "init"

            # obs frame — identical to the standard rollout (qpos[7:34] slice is
            # robot-only; the box/cube tail never enters the obs). grav was
            # already computed above for the fall flag; reuse it here.
            qj = d.qpos[7:7 + N_JOINTS]
            dqj = d.qvel[6:6 + N_JOINTS]
            omega = d.qvel[3:6]
            single = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
            single[0:3] = np.array([vx_c, vy_c, wz_c], dtype=np.float32) * CMD_SCALE
            single[3] = h_c
            single[4:7] = omega * ang_vel_scale
            single[7:10] = grav
            single[10:37] = (qj - OBS_DEFAULT_27) * DOF_POS_SCALE
            single[37:64] = dqj * DOF_VEL_SCALE
            single[64:76] = action
            np.clip(single, -OBS_CLIP, OBS_CLIP, out=single)
            obs_hist.append(single)
            obs456 = np.concatenate(obs_hist).astype(np.float32)

            out = sess.run([out_name], {in_name: obs456[None, :]})[0][0]
            action = np.clip(out.astype(np.float32), -ACTION_CLIP, ACTION_CLIP)
            target_dof_pos = action.astype(np.float64) * ACTION_SCALE + LEG_DEFAULT

            # records
            rec["t"].append(t)
            rec["base_x"].append(float(d.qpos[0]))
            rec["base_y"].append(float(d.qpos[1]))
            rec["base_z"].append(float(d.qpos[2]))
            rec["tilt"].append(tilt)
            rec["phase"].append(phase)

            if fall_now:
                fall = True
                fall_time = t
                fall_phase = mission.fall_stage or phase
                fall_reason = "tilt" if gxy > TILT_FALL_GXY else "ground_contact"
                break
            if mission.finished:
                break
    except Exception as e:  # noqa: BLE001
        harness_error = "exception: %s" % e
    finally:
        if writer is not None:
            writer.close()

    # ------------------ metrics -------------------------------------------
    mres = mission.result()
    metrics = {
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "mission_status": mres["status"],
        "mission_success": mres["success"],
        "n_stages": mres["n_stages"],
        "fall_stage": mres["fall_stage"],
        "fall_t": mres["fall_t"],
        "stages": mres["stages"],
        "max_tilt_rad": (float(np.max(rec["tilt"])) if rec["tilt"] else None),
        "sim_time_s": round(t_end, 3),
        "wall_time_s": round(time.time() - wall_t0, 2),
    }
    # per-stage diagnostic flags (BENCHMARK_V2_DESIGN M2 §4 逐阶段 flag)
    metrics.update(_arena_stage_flags(mres["stages"]))

    # optional trajectory dump for nav/squat diagnostics (smoke debugging only)
    if os.environ.get("ARENA_TRAJ_DUMP") and rec["t"]:
        for i in range(0, len(rec["t"]), 25):  # ~0.5s stride at 50 Hz
            sys.stderr.write(
                "TRAJ t=%5.1f xy=(%6.2f,%6.2f) z=%.2f tilt=%.2f phase=%s\n"
                % (rec["t"][i], rec["base_x"][i], rec["base_y"][i],
                   rec["base_z"][i], rec["tilt"][i], rec["phase"][i]))

    success = bool(mres["success"] and not fall and harness_error is None)

    return {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial_idx,
        "seed": seed,
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "success": success,
        "fall": bool(fall),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "fall_reason": fall_reason,
        "harness_error": harness_error,
        "box_mode": "arena_wrist_weld",
        "upper_mode": "psi0_replay_%s" % rep["source"],
        "metrics": metrics,
    }


def _arena_stage_flags(stages):
    """Flatten the per-stage records into the M-series diagnostic flags
    (nav arrival errors, place_ok, cube_transfer_ok, regrasp_ok, ...). One pass
    over the executed stages; later same-kind stages overwrite earlier ones so
    the names track the M2 functional order (relay place then store place)."""
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
            if grasp_seen == 1:
                flags["grasp_ok"] = ok
            else:
                flags["regrasp_ok"] = ok
        elif kind == "place":
            place_seen += 1
            ok = bool(s.get("place_ok"))
            surface = s.get("name", "")
            if "relay" in surface:
                flags["place_ok_relay"] = ok
            elif "store" in surface:
                flags["place_ok_store"] = ok
            else:
                flags["place_ok_%d" % place_seen] = ok
        elif kind == "cube":
            flags["cube_transfer_ok"] = bool(s.get("cube_transfer_ok"))
    return flags


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(results):
    out = {
        "n_trials": len(results),
        "success_rate": float(np.mean([r["success"] for r in results]))
        if results else None,
        "fall_count": int(sum(r["fall"] for r in results)),
    }
    keys = set()
    for r in results:
        keys.update(r["metrics"].keys())
    stats = {}
    for k in sorted(keys):
        vals = [r["metrics"].get(k) for r in results]
        vals = [v for v in vals if v is not None]
        if not vals:
            stats[k] = None
            continue
        if all(isinstance(v, bool) for v in vals):
            stats[k] = {"rate": float(np.mean(vals)), "n": len(vals)}
        elif all(isinstance(v, (int, float)) for v in vals):
            arr = np.array(vals, dtype=np.float64)
            stats[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                        "min": float(arr.min()), "max": float(arr.max()),
                        "n": len(vals)}
        else:
            stats[k] = None
    out["metrics"] = stats
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_trial_plan(test, trials, n_tapes=None):
    """Return list of (trial_idx, kwargs) — seed == trial_idx (unified spec).
    n_tapes: vln_follow only (trial i plays tape i % n_tapes)."""
    plan = []
    if test in ("walk_speed", "squat_box", "walk_speed_psi0",
                "squat_box_psi0"):
        n = trials if trials is not None else 50
        for i in range(n):
            plan.append((i, {"vx_target": 1.0}))
    elif test in ("circle_pillar", "circle_pillar_psi0"):
        n = trials if trials is not None else 50
        n_ccw = (n + 1) // 2
        for i in range(n):
            plan.append((i, {"vx_target": CIRCLE_VX,
                             "direction": "ccw" if i < n_ccw else "cw"}))
    elif test == "speed_sweep":
        per = trials if trials is not None else 5
        i = 0
        for v in SWEEP_SPEEDS:
            for _ in range(per):
                plan.append((i, {"vx_target": v}))
                i += 1
    elif test == "squat_sweep":
        # --trials N = trials PER GRID POINT (default 5):
        # 8 depths @ 0.2 m/s + 4 ramp rates @ 0.45 m = 12 points
        per = trials if trials is not None else 5
        i = 0
        for h in SQUAT_SWEEP_HEIGHTS:
            for _ in range(per):
                plan.append((i, {"sweep": "depth", "target_height": h,
                                 "ramp_speed": SQUAT_SWEEP_BASE_RATE}))
                i += 1
        for v in SQUAT_SWEEP_RATES:
            for _ in range(per):
                plan.append((i, {"sweep": "speed",
                                 "target_height": SQUAT_SWEEP_FIXED_H,
                                 "ramp_speed": v}))
                i += 1
    elif test == "goto_ab":
        n = trials if trials is not None else 25
        for i in range(n):
            plan.append((i, {}))
    elif test == "pipeline_abc":
        n = trials if trials is not None else 15
        for i in range(n):
            plan.append((i, {}))
    elif test == "squat_limit":
        n = trials if trials is not None else 10
        for i in range(n):
            plan.append((i, {}))
    elif test == "squat_place_psi0":
        n = trials if trials is not None else 25
        for i in range(n):
            plan.append((i, {}))
    elif test == "squat_pick_ground":
        n = trials if trials is not None else 15
        for i in range(n):
            plan.append((i, {}))
    elif test == "vln_follow":
        n = trials if trials is not None else 10
        for i in range(n):
            plan.append((i, {"tape_id": i % n_tapes}))
    else:
        raise ValueError(test)
    return plan


def make_renderer(model):
    # TODO(verify-on-server): EGL offscreen rendering on the 4090 box
    # (MUJOCO_GL=egl). Falls back to no-video with a warning if it fails.
    try:
        return mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    except Exception as e:  # noqa: BLE001 — renderer failure must not kill run
        sys.stderr.write("WARNING: offscreen renderer unavailable (%s); "
                         "videos disabled.\n" % e)
        return None


def run_arena_main(args, repo, out_dir, video_dir):
    """ManipArena entry (arena_M1 / arena_M2). Builds the variant scene, drives
    the shared Mission, writes results.jsonl + summary.json + EGL videos. The
    upper body is a real_ep053 Psi0 replay (required, --upper-replay)."""
    if args.upper_replay is None:
        sys.stderr.write("ERROR: --upper-replay is required for %s "
                         "(real_ep053 homie27 npz, contract v1)\n" % args.test)
        sys.exit(2)
    if not (0 <= args.variant <= 49):
        sys.stderr.write("ERROR: --variant must be in [0, 49] (got %d)\n"
                         % args.variant)
        sys.exit(2)

    replay = load_upper_replay(args.upper_replay)
    if replay["embodiment"] not in ("homie27", "unknown"):
        raise SystemExit("upper-replay embodiment %r != homie27 — wrong npz?"
                         % replay["embodiment"])
    if not replay["source"].startswith("real"):
        sys.stderr.write("WARNING: %s normally uses a real_* replay (ep053), "
                         "got %r\n" % (args.test, replay["source"]))

    spec = ms.build_variant(args.variant)
    print("arena: variant_seed=%d H_pick=%.2f hash=%s"
          % (spec.variant_seed, spec.H_pick, spec.scene_spec_hash[:12]))

    model = build_arena_model(repo, spec)
    # robot 34/27 + box freejoint (7) + cube freejoint (7) = 48 qpos, 27 actuators
    assert model.nq == 48 and model.nu == 27, \
        "unexpected arena model dims nq=%d nu=%d" % (model.nq, model.nu)

    replay = prepare_upper_replay(model, replay)
    replay = prepare_arena_segments(replay)
    print("arena psi0: source=%s N=%d (%.2fs); segments reach_down=%s "
          "carry=%s place=%s"
          % (replay["source"], replay["n"], replay["duration_s"],
             replay["arena_segments"]["reach_down"],
             replay["arena_segments"]["carry"],
             replay["arena_segments"]["place"]))

    box_info = configure_arena_welds(model)

    onnx_path = os.path.join(repo, ONNX_REL)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    ctx = {
        "test": args.test,
        "model": model,
        "ids": lookup_ids(model),
        "box": box_info,
        "upper": replay,
        "sess": sess,
        "in_name": sess.get_inputs()[0].name,
        "out_name": sess.get_outputs()[0].name,
        "ang_vel_scale": float(args.ang_vel_scale),
        "renderer": make_renderer(model) if args.video != "none" else None,
    }

    n = args.trials if args.trials is not None else 1
    jsonl_path = os.path.join(out_dir, "results.jsonl")
    results = []
    ok_videos_saved = 0
    with open(jsonl_path, "w") as jf:
        for trial_idx in range(n):
            seed = trial_idx
            render_first_pass = (args.video == "all"
                                 and ctx["renderer"] is not None)
            tmp_video = (os.path.join(video_dir, "_tmp_t%02d.mp4" % trial_idx)
                         if render_first_pass else None)
            res = run_arena_trial(ctx, spec, trial_idx, seed,
                                  video_path=tmp_video)
            tag = "ok" if res["success"] else "fail"
            final_video = os.path.join(
                video_dir, "%s_%s_v%02d_t%02d_%s.mp4"
                % (FRAMEWORK, args.test, args.variant, trial_idx, tag))
            if render_first_pass and tmp_video and os.path.exists(tmp_video):
                os.replace(tmp_video, final_video)
            elif (args.video == "policy" and ctx["renderer"] is not None
                  and (not res["success"] or ok_videos_saved < MAX_OK_VIDEOS)):
                res2 = run_arena_trial(ctx, spec, trial_idx, seed,
                                       video_path=final_video)
                if res2["success"] != res["success"]:
                    sys.stderr.write(
                        "WARNING: arena trial %d video re-run diverged "
                        "(success %s->%s)\n"
                        % (trial_idx, res["success"], res2["success"]))
                if res["success"]:
                    ok_videos_saved += 1
            results.append(res)
            jf.write(json.dumps(to_jsonable(res)) + "\n")
            jf.flush()
            mtr = res["metrics"]
            print("[%s/%s] v%02d trial %02d -> %s%s%s  (status=%s, %ss sim)"
                  % (FRAMEWORK, args.test, args.variant, trial_idx,
                     "OK" if res["success"] else "FAIL",
                     "" if not res["fall"] else
                     " fall@%.2fs(%s,%s)" % (res["fall_time"],
                                             res["fall_phase"],
                                             res["fall_reason"]),
                     "" if not res["harness_error"] else
                     " HARNESS_ERROR(%s)" % res["harness_error"],
                     mtr["mission_status"], round(mtr["sim_time_s"], 1)))

    summary = {
        "framework": FRAMEWORK,
        "test": args.test,
        "repo": repo,
        "checkpoint": onnx_path,
        "ang_vel_scale": ctx["ang_vel_scale"],
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "scene_spec": spec.to_dict(),
        "box_mode": "arena_wrist_weld",
        "upper_mode": "psi0_replay_%s" % replay["source"],
        "upper_replay": replay["path"],
        "overall": aggregate(results),
    }
    summary["harness_error_count"] = sum(
        1 for r in results if r.get("harness_error"))
    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print("summary -> %s" % sum_path)
    print("results -> %s" % jsonl_path)


def main():
    ap = argparse.ArgumentParser(description="HOMIE G1 benchmark harness")
    ap.add_argument("--test", required=True,
                    choices=["walk_speed", "squat_box", "circle_pillar",
                             "speed_sweep", "walk_speed_psi0",
                             "squat_box_psi0", "circle_pillar_psi0",
                             "squat_sweep", "goto_ab", "pipeline_abc",
                             "squat_limit", "squat_place_psi0",
                             "squat_pick_ground", "vln_follow",
                             "arena_M1", "arena_M2"])
    ap.add_argument("--variant", type=int, default=0,
                    help="ManipArena (arena_M1/M2): variant_seed 0-49 "
                         "(BENCHMARK_V2_DESIGN §5; default 0). --trials N "
                         "re-runs the SAME variant N times (default 1).")
    ap.add_argument("--trials", type=int, default=None,
                    help="walk/squat/circle (+_psi0): total trials "
                         "(default 50); speed_sweep: trials PER SPEED "
                         "(default 5); squat_sweep: trials PER GRID POINT "
                         "(default 5); goto_ab: total (default 25); "
                         "pipeline_abc: total (default 15); squat_limit: "
                         "total (default 10; --video all recommended); "
                         "squat_place_psi0: total (default 25); "
                         "squat_pick_ground: total (default 15); "
                         "vln_follow: total (default 10, tape i%%n_tapes)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--upper-replay", default=None,
                    help="psi0 tests: upper_replay_homie27_*.npz (contract "
                         "v1; 4090:/sda/lizhe/g1bench/psi0_replay/out/)")
    ap.add_argument("--tapes", default=None,
                    help="vln_follow: tapes.json from make_vln_tapes.py "
                         "(REQUIRED for that test; one shared file across "
                         "all four harnesses)")
    ap.add_argument("--video", choices=["none", "policy", "all"],
                    default="policy",
                    help="policy = first %d successful + all failed trials"
                         % MAX_OK_VIDEOS)
    ap.add_argument("--repo", default=None, help="path to OpenHomie checkout")
    ap.add_argument("--ang-vel-scale", type=float, default=0.25,
                    help="obs ang-vel scale. 0.25 = MujocoDeploy g1.yaml "
                         "(default; A/B-measured better wz tracking); "
                         "0.5 = real-robot pairing (lcm_agent.py:72).")
    # ---- custom single-point render mode (experiment console /api/render):
    ap.add_argument("--custom-height", type=float, default=None,
                    help="squat_sweep: render --trials N at this absolute "
                         "base height [m] instead of the full grid")
    ap.add_argument("--custom-rate", type=float, default=None,
                    help="squat_sweep: height ramp speed [m/s] for the "
                         "custom point (default %.1f)" % SQUAT_SWEEP_BASE_RATE)
    ap.add_argument("--custom-vx", type=float, default=None,
                    help="walk_speed/speed_sweep: command this vx [m/s]; "
                         "circle_pillar: tangential speed (radius = vx/wz)")
    ap.add_argument("--custom-wz", type=float, default=None,
                    help="circle_pillar: yaw rate [rad/s]")
    args = ap.parse_args()

    global CIRCLE_VX, CIRCLE_WZ, CUSTOM_VX_REL_THR
    if args.test in ("circle_pillar", "circle_pillar_psi0"):
        if args.custom_vx is not None:
            CIRCLE_VX = float(args.custom_vx)
        if args.custom_wz is not None:
            CIRCLE_WZ = float(args.custom_wz)
    elif args.custom_vx is not None and args.test in ("walk_speed",
                                                      "speed_sweep"):
        CUSTOM_VX_REL_THR = True

    repo = find_repo(args.repo)
    out_dir = args.out_dir or os.path.join("homie_results", args.test)
    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    if args.video != "none":
        os.makedirs(video_dir, exist_ok=True)

    if args.test in ARENA_TESTS:
        run_arena_main(args, repo, out_dir, video_dir)
        return

    replay = None
    if args.test in PSI0_TESTS:
        if args.upper_replay is None:
            sys.stderr.write("ERROR: --upper-replay is required for %s "
                             "(upper_replay_homie27_*.npz, contract v1)\n"
                             % args.test)
            sys.exit(2)
        replay = load_upper_replay(args.upper_replay)
        if replay["embodiment"] not in ("homie27", "unknown"):
            raise SystemExit("upper-replay embodiment %r != homie27 — "
                             "wrong npz?" % replay["embodiment"])
        want_src = "sim" if args.test == "squat_box_psi0" else "real"
        if not replay["source"].startswith(want_src):
            sys.stderr.write("WARNING: %s normally uses a %s_* replay "
                             "source, got %r\n"
                             % (args.test, want_src, replay["source"]))
    elif args.upper_replay is not None:
        sys.stderr.write("WARNING: --upper-replay ignored (not a psi0 test)\n")

    tapes = None
    if args.test == "vln_follow":
        if args.tapes is None:
            sys.stderr.write("ERROR: --tapes is required for vln_follow "
                             "(tapes.json from make_vln_tapes.py)\n")
            sys.exit(2)
        tapes = load_vln_tapes(args.tapes)
        print("vln: %d tapes, %.1fs each, %d stop segments on tape 0"
              % (len(tapes), tapes[0]["duration_s"], len(tapes[0]["stops"])))
    elif args.tapes is not None:
        sys.stderr.write("WARNING: --tapes ignored (not vln_follow)\n")

    model = build_model(repo, args.test)
    # robot is 34/27; box tests add a free box body (+7 qpos, 0 actuators)
    expect_nq = 41 if args.test in BOX_TESTS else 34
    assert model.nq == expect_nq and model.nu == 27, \
        "unexpected model dims nq=%d nu=%d" % (model.nq, model.nu)
    if replay is not None:
        replay = prepare_upper_replay(model, replay)
        print("psi0: replay source=%s K=%d N=%d (%.2fs, grasp_close_t=%.2fs)"
              % (replay["source"], len(replay["names"]), replay["n"],
                 replay["duration_s"], replay["grasp_close_t"]))
        if args.test == "squat_place_psi0":
            replay = prepare_place_segments(replay)
            print("psi0 place: t_place=%.2fs (frame %d, h_min=%.3f m)"
                  % (replay["t_place"], replay["place_k"],
                     float(replay["height"][replay["place_k"]])))
    if args.test in BOX_TESTS:
        # weld anchor pose: replay frame 0 for psi0, the low-reach pose for
        # the T9 ground pick (its welds are re-anchored at the actual grasp
        # pose anyway), the HUG pose for the v2 carry tests
        if replay is not None:
            anchor_pose = replay["pos15"][0]
        elif args.test == "squat_pick_ground":
            anchor_pose = PICK_ARM_POSE
        else:
            anchor_pose = BOX_ARM_POSE
        box_info = configure_box_welds(model, upper_pose=anchor_pose)
    else:
        box_info = None

    onnx_path = os.path.join(repo, ONNX_REL)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1     # determinism + tiny MLP, no need for more
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    ctx = {
        "test": args.test,
        "model": model,
        "ids": lookup_ids(model),
        "box": box_info,
        "upper": replay,
        "sess": sess,
        "in_name": sess.get_inputs()[0].name,    # 'input'  [batch,456]
        "out_name": sess.get_outputs()[0].name,  # 'output' [batch,12]
        "ang_vel_scale": float(args.ang_vel_scale),
        "tapes": tapes,
        "renderer": make_renderer(model) if args.video != "none" else None,
    }

    plan = build_trial_plan(args.test, args.trials,
                            n_tapes=len(tapes) if tapes else None)
    # custom single-point overrides replace the standard grid/plan
    n_custom = args.trials if args.trials is not None else 1
    if args.test == "squat_sweep" and (args.custom_height is not None
                                       or args.custom_rate is not None):
        ch = (args.custom_height if args.custom_height is not None
              else SQUAT_SWEEP_FIXED_H)
        cr = (args.custom_rate if args.custom_rate is not None
              else SQUAT_SWEEP_BASE_RATE)
        plan = [(i, {"sweep": "custom", "target_height": ch,
                     "ramp_speed": cr}) for i in range(n_custom)]
        print("custom squat point: H=%.2f m @ %.2f m/s x%d" % (ch, cr, n_custom))
    elif (args.test in ("walk_speed", "speed_sweep")
          and args.custom_vx is not None):
        plan = [(i, {"vx_target": float(args.custom_vx)})
                for i in range(n_custom)]
        print("custom walk: vx=%.2f m/s x%d" % (args.custom_vx, n_custom))
    elif (args.test == "circle_pillar"
          and (args.custom_vx is not None or args.custom_wz is not None)):
        print("custom circle: vx=%.2f wz=%.2f (radius %.2f m) x%d"
              % (CIRCLE_VX, CIRCLE_WZ, CIRCLE_VX / CIRCLE_WZ, len(plan)))
    jsonl_path = os.path.join(out_dir, "results.jsonl")
    results = []
    ok_videos_saved = 0
    max_ok_videos = (PIPE_MAX_OK_VIDEOS if args.test == "pipeline_abc"
                     else MAX_OK_VIDEOS)

    with open(jsonl_path, "w") as jf:
        for trial_idx, kw in plan:
            seed = trial_idx
            render_first_pass = (args.video == "all"
                                 and ctx["renderer"] is not None)
            tmp_video = (os.path.join(video_dir, "_tmp_t%02d.mp4" % trial_idx)
                         if render_first_pass else None)
            res = run_trial(ctx, trial_idx, seed,
                            video_path=tmp_video, **kw)

            tag = "ok" if res["success"] else "fail"
            final_video = os.path.join(
                video_dir, "%s_%s_t%02d_%s.mp4"
                % (FRAMEWORK, args.test, trial_idx, tag))

            if render_first_pass and tmp_video and os.path.exists(tmp_video):
                os.replace(tmp_video, final_video)
            elif (args.video == "policy" and ctx["renderer"] is not None
                  and (not res["success"]
                       or ok_videos_saved < max_ok_videos)):
                # Deterministic re-run with rendering (physics unchanged).
                # TODO(verify-on-server): spot-check that re-run reproduces
                # the same outcome (warned below if it diverges).
                res2 = run_trial(ctx, trial_idx, seed,
                                 video_path=final_video, **kw)
                if (res2["success"] != res["success"]
                        or res2["fall_time"] != res["fall_time"]):
                    sys.stderr.write(
                        "WARNING: trial %d video re-run diverged "
                        "(success %s->%s, fall_time %s->%s)\n"
                        % (trial_idx, res["success"], res2["success"],
                           res["fall_time"], res2["fall_time"]))
                if res["success"]:
                    ok_videos_saved += 1

            results.append(res)
            jf.write(json.dumps(to_jsonable(res)) + "\n")
            jf.flush()
            print("[%s/%s] trial %02d seed %d -> %s%s%s  (%ss sim, %ss wall)"
                  % (FRAMEWORK, args.test, trial_idx, seed,
                     "OK" if res["success"] else "FAIL",
                     "" if not res["fall"] else
                     " fall@%.2fs(%s,%s)" % (res["fall_time"],
                                             res["fall_phase"],
                                             res["fall_reason"]),
                     "" if not res["harness_error"] else
                     " HARNESS_ERROR(%s)" % res["harness_error"],
                     round(res["metrics"]["sim_time_s"], 1),
                     res["metrics"]["wall_time_s"]))

    # ------------------ summary --------------------------------------------
    summary = {
        "framework": FRAMEWORK,
        "test": args.test,
        "repo": repo,
        "checkpoint": onnx_path,
        "ang_vel_scale": ctx["ang_vel_scale"],
        "overall": aggregate(results),
    }
    summary["harness_error_count"] = sum(
        1 for r in results if r.get("harness_error"))
    if args.test in BOX_TESTS:
        summary["box_mode"] = BOX_MODE
    if args.test in PSI0_TESTS:
        summary["upper_mode"] = "psi0_replay_%s" % replay["source"]
        summary["upper_replay"] = replay["path"]
        summary["upper_replay_grasp_close_t"] = replay["grasp_close_t"]
    if args.test in ("squat_box_psi0", "squat_pick_ground"):
        picks = [r["metrics"].get("pick_success") for r in results
                 if isinstance(r["metrics"].get("pick_success"), bool)]
        summary["pick_success_rate"] = (float(np.mean(picks))
                                        if picks else None)
    if args.test == "vln_follow":
        summary["tapes"] = os.path.abspath(args.tapes)
        summary["n_tapes"] = len(tapes)
        summary["by_tape"] = {
            str(tp["id"]): aggregate([r for r in results
                                      if r["metrics"].get("tape_id")
                                      == tp["id"]])
            for tp in tapes
            if any(r["metrics"].get("tape_id") == tp["id"] for r in results)
        }
    if args.test == "squat_sweep":
        # per-(height|speed) breakdown — directly answers "which heights
        # fall, how far does the root drift" (spec v3)
        def fall_phases_of(rs):
            phs = {}
            for r in rs:
                if r["fall"]:
                    ph = r.get("fall_phase") or "unknown"
                    phs[ph] = phs.get(ph, 0) + 1
            return phs

        by_height = {}
        for h in SQUAT_SWEEP_HEIGHTS:
            rs = [r for r in results
                  if r["metrics"].get("sweep") == "depth"
                  and r["metrics"].get("target_height") == h]
            if rs:
                agg = aggregate(rs)
                agg["fall_phases"] = fall_phases_of(rs)
                by_height["%.2f" % h] = agg
        by_speed = {}
        for v in SQUAT_SWEEP_RATES:
            rs = [r for r in results
                  if r["metrics"].get("sweep") == "speed"
                  and r["metrics"].get("ramp_speed") == v]
            if rs:
                agg = aggregate(rs)
                agg["fall_phases"] = fall_phases_of(rs)
                by_speed["%.1f" % v] = agg
        summary["by_height"] = by_height
        summary["by_speed"] = by_speed
    if args.test == "speed_sweep":
        by_speed = {}
        max_stable = None
        for v in SWEEP_SPEEDS:
            sub = [r for r in results if r["metrics"].get("cmd_vx") == v]
            agg = aggregate(sub)
            by_speed["%.1f" % v] = agg
            if agg["success_rate"] == 1.0:
                max_stable = v
        summary["by_speed"] = by_speed
        summary["max_stable_speed"] = max_stable
    if args.test in ("circle_pillar", "circle_pillar_psi0"):
        summary["by_direction"] = {
            dname: aggregate([r for r in results
                              if r["metrics"].get("direction") == dname])
            for dname in ("ccw", "cw")
        }

    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print("summary -> %s" % sum_path)
    print("results -> %s" % jsonl_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
