#!/usr/bin/env python3
"""AGILE (NVIDIA WBC-AGILE) G1 benchmark harness — 统一实验规范 v1.

Runs the pretrained Unitree G1 velocity+height student policy
(TorchScript LSTM, 4D command [vx, vy, wz, abs_pelvis_height]) through the
official sim2mujoco pipeline (agile.sim2mujoco.*) — no Isaac Lab / Isaac Sim.

Tests (unified spec v1):
  walk_speed    : vx ramp 0->1.0 m/s in 2 s, hold 10 s.   NOTE: AGILE trains
                  vx in [-0.5, 0.5]; 1.0 m/s is intentionally OOD — the
                  harness widens the CommandManager clamp so the policy
                  actually SEES 1.0. Local smoke runs: tracks ~0.84 m/s at
                  cmd 1.0 without falling (undershoot is the result).

UPSTREAM BUG WORKED AROUND (see BodyFrameGyroShim): AGILE's sim2mujoco
fallback treats free-joint qvel[3:6] as world-frame angular velocity; it is
body-frame. Without the shim, any trial with |initial yaw| >> 0 falls in <1 s.
  squat_box     : free-body 2 kg box (0.35x0.25x0.25 m) held between the two
                  palms via two SOFT weld constraints to left/right
                  wrist_yaw_link ("wrist_weld_v2") — the load is transmitted
                  through the arm joints (arm PD vs gravity; droop/oscillation
                  is real physics and recorded as-is). Arms in carry pose,
                  20 x (1.5 s ramp 0.72->0.45 m, 1 s hold, 1.5 s up, 1 s hold).
                  New metrics: box_kept (|box-torso| < 0.6 m all trial),
                  box_drop_time; qacc divergence is caught as harness_error.
  circle_pillar : r=0.15 m pillar at circle center, vx=0.4, wz=+/-0.4 rad/s,
                  2 laps; first half of trials CCW, second half CW.
  speed_sweep   : vx in {0.4, 0.6, 0.8, 1.0, 1.2}, trials/5 each.

psi0 replay variants (upper-body replay contract v1, --upper-replay NPZ from
4090:/sda/lizhe/g1bench/psi0_replay/out/upper_replay_agile29_*.npz; fields
t/upper_names/upper_pos/height_cmd/grasp_close_t/carry_pose/meta_json,
agile29: K=17 = arms14 + waist3, columns remapped by joint NAME). The upper
body (arms14 + waist3 — non-policy targets) is PD-positioned frame-by-frame
through the same JointCommand override path as the carry pose:
  walk_speed_psi0    : walk_speed profile + LOOPED real_ep053 upper stream
                       (from t=0, settle included); replay height_cmd ignored
                       (legs stay at stand height); cracker_box (0.411 kg)
                       wrist-welded from t=0. success = base + box_kept.
  squat_box_psi0     : BendPick table scene (psi0_replay/scene_info.json,
                       ep035; tabletop 0.40 m, cracker_box upright on it,
                       pelvis ~0.29 m from the near edge). 2 s settle ->
                       sim_ep035 replay with the legs following its
                       (pre-smoothed) height_cmd (clipped to AGILE's [0.4,
                       0.72] domain); at grasp_close_t both wrists must be
                       < 0.12 m from the box surface to activate the welds
                       (else pick_failed); afterwards the arms freeze at
                       carry_pose and the standard 20x5 s squat cycles run.
                       success := pick_success + 20/20 cycles + box_kept.
                       New metrics: pick_success, box_lift_height. Box
                       collides with table/floor (bit 2), never with the
                       robot (bit 1), per the unified psi0 spec.
  circle_pillar_psi0 : circle_pillar profile + looped real_ep053 upper
                       stream; cracker_box welded from t=0. success = base
                       + box_kept.
  JSONL results gain "upper_mode": "psi0_replay_<source>".

v3 tests (unified across the four harnesses):
  squat_sweep   : depth x ramp-speed grid while carrying the v2 box. Depth
                  scan H in {0.65..0.30} at 0.2 m/s + speed scan {0.1,0.2,
                  0.4,0.8} m/s at H=0.45; --trials = trials PER GRID POINT
                  (default 5). Per trial: 2 s settle -> ramp down to H ->
                  hold 3 s -> ramp up -> 1 s. Metrics: achieved_depth,
                  depth_err, root_drift_hold/total, box_kept. Summary groups
                  by target_height and by ramp_speed. NOTE: the manager's
                  height clamp [0.4,0.72] is widened so H=0.35/0.30 actually
                  reach the policy (OOD on purpose).
  goto_ab       : A=origin (yaw noise +/-0.3 rad) -> B=(3.0,1.0), theta_B=
                  +90deg. Two-stage shared nav controller: NAV (coarse P law,
                  exit at 0.30 m/15deg or 30 s) -> CAL (commands clamped to
                  |v|<=0.10 m/s, |wz|<=0.10 rad/s, converge 0.05 m/5deg held
                  1 s, timeout 15 s). Direct measurement of small-command
                  calibration. success := success_fine and no fall.
  pipeline_abc  : carry box (v2 welds) from A=origin -> C=(0.0,-2.5),
                  theta_C=-90deg (turn right then 2.5 m), NAV vx<=0.5 ->
                  CAL -> squat to 0.45 @0.2 m/s -> hold 0.5 s -> release
                  both welds + enable box collisions (box lands) -> hold 1 s
                  -> rise -> stand 1 s. success := success_coarse and no
                  fall in the place segment and box_place_ok. Videos: first
                  3 successes + all failures.

v4 tests (unified across the four harnesses):
  squat_limit   : descent-limit calibration while carrying the v2 2 kg box.
                  2 s settle (stand, box) -> height command ramps CONTINUOUSLY
                  down from 0.72 at 0.05 m/s to 0.10 m (far below the reachable
                  domain — the manager height clamp is widened to (0.05, 0.8)
                  so 0.10 is delivered verbatim; saturation is the RESULT) ->
                  hold 3 s at 0.10 -> end (no rise). Stops at fall. Raw
                  "height-tracking-drift" curve recorded at 10 Hz in
                  metrics["descent_curve"] = [[h_cmd, base_z, drift_xy], ...].
                  Scalars: fall_h_cmd/fall_base_z (None if no fall),
                  depth_floor (min stable base_z = physical squat limit),
                  track_sat_h (first |cmd-actual| > 5 cm), drift5_h/drift20_h
                  (cmd height when root XY drift first exceeds 5/20 cm),
                  max_drift_xy. success := no fall (reached 0.10 cmd + held).
                  Most models are expected to saturate without falling; the
                  ones that fall are exactly the boundary being calibrated.
                  Suggest --video all (every trial is a calibration sample).
  squat_place_psi0 : Psi0-coordinated reach-forward box placement (lower-body
                  error measurement). Requires --upper-replay real_ep053
                  (22.3 s, "two-handed lower-and-place", height 0.77->0.46->
                  0.75). t=0 carry the v2 2 kg box (wrist welds, stand) ->
                  2 s settle -> SINGLE (non-looped) playback of the ep053
                  upper stream (arms14+waist3) with the legs following its
                  height_cmd (clipped to AGILE's [0.4, 0.72] domain) -> at
                  t_place = argmin(height_cmd) both welds release and the box
                  gets bit-2 collisions (lands on the floor; same release
                  mechanism as pipeline_abc) -> playback continues through the
                  rise segment -> stand 1 s. Metrics: height_rmse_descent/
                  hold/rise, root_drift_place (root XY drift through the
                  reach+place window — the forward COM shift is the decoupling
                  stress test), max_tilt, box_place_ok + box_land_dx (box
                  landing point forward of the foot front edge at release;
                  positive = placed in front), stand_ok.
                  success := no fall AND box_place_ok. JSONL has upper_mode.

v5 tests (unified across the four harnesses):
  squat_pick_ground : floor-level box pick — root must go LOW. A v2-sized 2 kg
                  box stands on the floor 0.45 m in front of the robot
                  (bit-2 collisions: box vs floor only, never the robot).
                  2 s settle (arms policy-default) -> 1 s arm blend into the
                  FK-tuned "low forward reach" pose (see ARM_REACH_POSE) ->
                  height ramps 0.2 m/s down to the UNIFIED 0.25 m command
                  (manager clamp widened; each model saturates at its own
                  floor — AGILE ~0.50-0.53, R4) -> at the bottom the welds
                  fire ("magnetic grasp") once BOTH wrists are < 0.30 m from
                  the box surface; 5 s without a grasp -> pick_failed (still
                  rises) -> grasp hold 0.5 s -> ramp back up loaded -> stand
                  1 s. Metrics: pick_success, min_root_z, grasp_dist_l/r (at
                  grasp or timeout), root_drift (bottom+grasp XY), stand_ok.
                  success := pick AND no fall AND stand_ok. Offline FK puts
                  AGILE's wrist-to-box gap at ~0.26-0.29 m at saturation —
                  right AT the threshold; the honest outcome IS the result.
  vln_follow    : simulated VLN/navigation small-command stream following.
                  --tapes tapes.json (REQUIRED; from make_vln_tapes.py, the
                  SAME 10 tapes for all four models). Each tape: 30 s, ZOH
                  breakpoints [t,vx,vy,wz] updated at jittered 0.4-1.2 s
                  intervals, |vx|<=0.35 |vy|<=0.2 |wz|<=0.30, two 1-2 s
                  full stops, plus 1 Hz ideal ref trajectory [t,x,y,yaw].
                  2 s settle -> tape (trial i plays tape i%N) -> stand 1 s.
                  AGILE consumes wz natively as a yaw-rate command (no
                  target-yaw integration — contrast bench_amo). Metrics:
                  final_pos_err/final_yaw_err vs the ref end pose,
                  mean_track_err (1 Hz position error mean),
                  small_cmd_response (mean actual/commanded vx ratio over
                  |vx| in [0.05,0.15] segments), stop_settle (mean planar
                  speed in full-stop segments after a 0.5 s settle-in).
                  success := no fall AND final_pos_err<=0.30 m AND
                  final_yaw_err<=15 deg.

Interactive-render CLI (spec v5; console backend, AGILE + HOMIE only):
  --custom-height H --custom-rate R : squat_sweep grid -> single point [H]x[R]
  --custom-vx V                     : walk_speed target 1.0 -> V (pass >=0.9*V)
  --custom-vx V --custom-wz W       : circle_pillar 0.4/0.4 -> V/W, radius=V/W
  All optional; omitted -> behavior identical to spec v4.

Environment (headless Linux server, e.g. 4090 box):
  conda create -n agile_bench python=3.10 -y && conda activate agile_bench
  git clone https://github.com/nvidia-isaac/WBC-AGILE.git && cd WBC-AGILE && git lfs pull
  pip install -e . --no-deps
  pip install torch mujoco pyyaml numpy "imageio[ffmpeg]"
  git clone https://github.com/unitreerobotics/unitree_mujoco.git ~/unitree_mujoco
  # (pandas/matplotlib/seaborn/plotly/jinja2 NOT needed: harness stubs that import chain)

Examples:
  MUJOCO_GL=egl python bench_agile.py --test walk_speed --trials 50 \
      --agile-repo ~/WBC-AGILE \
      --mjcf ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
      --out-dir ~/bench_out/agile --video policy
  python bench_agile.py --test squat_box --trials 2 --video none ...   # smoke run
  MUJOCO_GL=egl python bench_agile.py --test squat_box_psi0 --trials 50 \
      --upper-replay /sda/lizhe/g1bench/psi0_replay/out/upper_replay_agile29_sim_ep035.npz \
      --out-dir ~/bench_out/agile_sbp --video policy
  MUJOCO_GL=egl python bench_agile.py --test walk_speed_psi0 --trials 50 \
      --upper-replay .../upper_replay_agile29_real_ep053.npz --out-dir ... --video policy
  python bench_agile.py --test squat_sweep --trials 5 ...   # 5 per grid point (8+4 points)
  python bench_agile.py --test goto_ab --trials 25 ...
  python bench_agile.py --test pipeline_abc --trials 15 ...
  MUJOCO_GL=egl python bench_agile.py --test squat_limit --trials 10 \
      --out-dir ... --video all   # every trial is a calibration sample
  MUJOCO_GL=egl python bench_agile.py --test squat_place_psi0 --trials 25 \
      --upper-replay .../upper_replay_agile29_real_ep053.npz --out-dir ... --video policy
  MUJOCO_GL=egl python bench_agile.py --test squat_pick_ground --trials 15 ...
  MUJOCO_GL=egl python bench_agile.py --test vln_follow --trials 10 \
      --tapes ../data/vln_tapes.json --out-dir ... --video policy
  python bench_agile.py --test squat_sweep --custom-height 0.50 --custom-rate 0.3 ...
  python bench_agile.py --test walk_speed --custom-vx 0.6 ...
  python bench_agile.py --test circle_pillar --custom-vx 0.5 --custom-wz 0.25 ...

Outputs: <out>/<label>_<test>_results.jsonl (one line per trial),
         <out>/<label>_<test>_summary.json,
         <out>/videos/<label>_<test>_tNN_{ok|fail}.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# MUJOCO_GL must be decided before `import mujoco` on headless servers.
# Harmless for non-video runs (no GL context is ever created without a Renderer).
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # TODO(verify-on-server): EGL works with driver 580

import numpy as np
import torch

import mujoco

# ManipArena v2 (BENCHMARK_V2_DESIGN.md). Shared scene generator + mission
# state machine; bench_homie.py is the reference adapter (§7.3) and this AGILE
# adapter mirrors it (only the robot embodiment / command channels differ).
# Imported from the same scripts/ dir so the four harnesses share one source
# of truth.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manip_scene as ms          # noqa: E402
import manip_mission as mm        # noqa: E402

# ---------------------------------------------------------------------------
# Constants (unified spec v1 + AGILE specifics)
# ---------------------------------------------------------------------------

FRAMEWORK = "AGILE"
TESTS = ("walk_speed", "squat_box", "circle_pillar", "speed_sweep",
         "walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
         "squat_sweep", "goto_ab", "pipeline_abc",
         "squat_limit", "squat_place_psi0",
         "squat_pick_ground", "vln_follow",
         "arena_M1", "arena_M2",
         "arena_L1", "arena_L2", "arena_L3", "arena_L4", "arena_L5")
# ManipArena M-series (mission state machine) + L-series (locomotion sweeps on
# the arena scene as a fixed background). See the arena adapter block below.
ARENA_M_TESTS = ("arena_M1", "arena_M2")
ARENA_L_TESTS = ("arena_L1", "arena_L2", "arena_L3", "arena_L4", "arena_L5")
ARENA_TESTS = ARENA_M_TESTS + ARENA_L_TESTS
PSI0_TESTS = ("walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0")
# All tests driven by an --upper-replay npz (psi0 trio + v4 squat_place_psi0).
REPLAY_TESTS = PSI0_TESTS + ("squat_place_psi0",)
CARRY_TESTS = ("squat_box", "squat_sweep", "pipeline_abc", "squat_limit")  # constant hug arms + 2 kg box
NAV_TESTS = ("goto_ab", "pipeline_abc")  # stateful two-stage navigation runner
# Tests with a free box + wrist welds in the scene. squat_place_psi0 carries
# the v2 2 kg box (NOT the psi0 cracker box) but its arms come from a replay;
# squat_pick_ground (v5) starts with the same 2 kg box ON THE FLOOR.
BOX_TESTS = CARRY_TESTS + PSI0_TESTS + ("squat_place_psi0", "squat_pick_ground")

CONTROL_DT = 0.02  # 50 Hz policy, enforced by exported YAML scene block (physics 200 Hz, decim 4)
SETTLE_S = 2.0  # static settle with zero commands before the test segment

STAND_HEIGHT = 0.72  # AGILE height command is ABSOLUTE pelvis height (m); stand default
SQUAT_HEIGHT = 0.45  # T2 squat target (within training domain [0.4, 0.72])

WALK_RAMP_S = 2.0
WALK_HOLD_S = 10.0
WALK_TARGET_VX = 1.0

# --custom-vx render mode (experiment console): walk success threshold
# becomes relative (0.9 * target) instead of the absolute 0.9 m/s spec gate.
# Set once in main() before any trial runs, read-only afterwards.
CUSTOM_VX_REL_THR = False
WALK_MEAN_WINDOW_S = 8.0
SWEEP_SPEEDS = (0.4, 0.6, 0.8, 1.0, 1.2)
WIDENED_VX_RANGE = (-2.0, 2.0)  # disables the CommandManager +/-0.5 clamp for T1/sweep (OOD on purpose)

SQUAT_RAMP_DOWN_S = 1.5
SQUAT_HOLD_DOWN_S = 1.0
SQUAT_RAMP_UP_S = 1.5
SQUAT_HOLD_UP_S = 1.0
SQUAT_CYCLE_S = SQUAT_RAMP_DOWN_S + SQUAT_HOLD_DOWN_S + SQUAT_RAMP_UP_S + SQUAT_HOLD_UP_S  # 5.0
SQUAT_N_CYCLES = 20

# T4 squat_sweep (spec v3): depth x ramp-speed grid, carrying the v2 box.
SQUAT_SWEEP_DEPTHS = (0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)  # abs base height targets
SQUAT_SWEEP_RATES = (0.1, 0.2, 0.4, 0.8)  # m/s ramp speeds at the fixed depth
SQUAT_SWEEP_FIXED_DEPTH = 0.45
SQUAT_SWEEP_FIXED_RATE = 0.2
SQUAT_SWEEP_HOLD_S = 3.0
SQUAT_SWEEP_END_S = 1.0
SQUAT_SWEEP_TRIALS_PER_POINT = 5
# The CommandManager clamps height to [0.4, 0.72]; widen it so the 0.35/0.30
# depth targets actually reach the policy (OOD on purpose — the sweep asks
# "which depths fall over").
SQUAT_SWEEP_HEIGHT_RANGE = (0.2, 0.8)

# T7 squat_limit (spec v4): continuous descent-limit calibration, v2 box carry.
SQUAT_LIMIT_RATE = 0.05  # m/s, continuous downward command ramp
SQUAT_LIMIT_FLOOR_H = 0.10  # command floor (far below the reachable domain — on purpose)
SQUAT_LIMIT_HOLD_S = 3.0  # hold at the 0.10 command, then end (no rise)
SQUAT_LIMIT_TRIALS = 10
# The CommandManager clamps height to [0.4, 0.72]; widen it past 0.10 so the
# command is delivered VERBATIM (spec v4: no clip — policy saturation is the
# measured result, not something to hide at the command layer).
SQUAT_LIMIT_HEIGHT_RANGE = (0.05, 0.8)
SQUAT_LIMIT_TRACK_SAT_M = 0.05  # |h_cmd - base_z| threshold for track_sat_h
SQUAT_LIMIT_DRIFT_MARKS_M = (0.05, 0.20)  # drift5_h / drift20_h thresholds
DESCENT_CURVE_EVERY_N_STEPS = 5  # 50 Hz / 5 = 10 Hz descent_curve sampling

# T8 squat_place_psi0 (spec v4): reach-forward place driven by real_ep053.
SQUAT_PLACE_TRIALS = 25
SQUAT_PLACE_STAND_S = 1.0  # post-replay stand segment
SQUAT_PLACE_HOLD_BAND_M = 0.03  # replay height within h_min+band => "hold" segment

# T9 squat_pick_ground (spec v5): floor-level pick — root must go LOW.
PICK_TRIALS = 15
PICK_BOX_AHEAD_M = 0.45  # box center this far in front of the pelvis (along yaw0)
PICK_BOX_HALF = (0.125, 0.175, 0.125)  # v2 2 kg box, standing upright on the floor
PICK_TARGET_H = 0.25  # UNIFIED squat command; each model saturates at its own floor
PICK_RAMP_RATE = 0.2  # m/s height ramp (down and up)
PICK_GRASP_DIST_M = 0.30  # BOTH wrists < this from the box surface -> magnetic grasp
PICK_BOTTOM_TIMEOUT_S = 5.0  # at the bottom without a grasp -> pick_failed (still rises)
PICK_HOLD_AFTER_GRASP_S = 0.5
PICK_STAND_S = 1.0
PICK_ARM_BLEND_S = 1.0  # settle-end -> reach-pose PD-target blend window
# The CommandManager clamps height to [0.4, 0.72]; widen it so the unified
# 0.25 command reaches the policy verbatim (OOD on purpose — saturation is
# the measured result, exactly like squat_sweep/squat_limit).
PICK_HEIGHT_RANGE = (0.2, 0.8)
# "Low forward reach" pose (spec v5: 肩pitch前倾+肘屈, wrists as low as
# possible straddling the box). FK-tuned OFFLINE against the G1 29dof chain
# (numpy FK over repos/AMO_5080/g1.xml, upright torso; G1 elbow neutral has
# the forearm HORIZONTAL, so the low reach needs elbow ~+1.4 to fold the
# forearm down while shoulder_pitch -0.45 leans the upper arm forward):
#   wrist_yaw origin ~ (x+0.20, y±0.17, base_z-0.024)
#   palm (+0.10 m local x) ~ (x+0.26, y±0.18, base_z-0.105)
# At AGILE's squat saturation (base ~0.50-0.53, R4) the wrist-to-box-surface
# gap computes to ~0.26-0.29 m — right AT the 0.30 m grasp threshold (any
# policy torso pitch-forward helps). Whether the grasp fires is the result.
ARM_REACH_POSE = {
    "left_shoulder_pitch_joint": -0.45,
    "right_shoulder_pitch_joint": -0.45,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.4,
    "right_elbow_joint": 1.4,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# T10 vln_follow (spec v5): irregular small-command tape following.
VLN_TRIALS = 10
VLN_END_STAND_S = 1.0  # post-tape stand segment
VLN_FINAL_POS_TOL_M = 0.30
VLN_FINAL_YAW_TOL_RAD = math.radians(15.0)
VLN_SMALL_VX_BAND = (0.05, 0.15)  # |vx| band for small_cmd_response
VLN_STOP_SKIP_S = 0.5  # settle-in skipped at the head of each full-stop segment
# Tape commands (|vx|<=0.35, |vy|<=0.2, |wz|<=0.30) are INSIDE AGILE's
# training domain, but the manager clamps are widened anyway (spec v5:
# 放宽到位) so any tape is delivered verbatim.
VLN_CMD_RANGES = {
    "linear_x_range": (-2.0, 2.0),
    "linear_y_range": (-1.0, 1.0),
    "angular_z_range": (-2.0, 2.0),
}

CIRCLE_VX = 0.4
CIRCLE_WZ = 0.4  # rad/s, radius = vx/wz = 1.0 m
CIRCLE_PERIOD_S = 2.0 * math.pi / CIRCLE_WZ  # ~15.708 s per lap
CIRCLE_N_LAPS = 2
CIRCLE_CENTER = (0.0, 1.0)  # robot spawns at origin; CCW yaw=0, CW yaw=pi -> same center
CIRCLE_RADIUS = 1.0
CIRCLE_RADIAL_TOL = 0.15

# Unified fall criteria
FALL_TILT_RAD = 0.9
FALL_HEIGHT_MARGIN = 0.2  # base_z < (current height target - 0.2 m)

# Trial init randomization
JOINT_NOISE_RAD = 0.02

# T2 carry pose (MJCF joint names; G1 "hold a box between the palms").
# FK-tuned against unitree g1_29dof.xml: palm points sit ~+/-0.184 m left/right
# of the box center (box half width 0.175 -> ~9 mm visual gap per side), box
# center ~0.25 m in front of and ~0.12 m below torso_link. Local fixed-base
# sanity sim (AGILE gains + armature, gravity, 10 s): stable, max droop ~0.09 m.
ARM_CARRY_POSE = {
    "left_shoulder_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

BOXDEMO_UPPER_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
)
BOXDEMO_UPPER_MODE = "box_demo_2_ik"
BOXDEMO_REPLAY_S = 8.0
BOXDEMO_GRASP_CLOSE_S = 5.0

# Video
VIDEO_W, VIDEO_H = 640, 480
VIDEO_EVERY_N_CTRL_STEPS = 2  # 50 Hz / 2 = 25 fps
VIDEO_FPS = 25
VIDEO_MAX_SUCCESS = 5  # per test, in --video policy mode

# ---------------------------------------------------------------------------
# ManipArena v2 constants (BENCHMARK_V2_DESIGN §2-§4). Mirrors the bench_homie.py
# ARENA_* block; only the height-command semantics (AGILE absolute pelvis
# height, training domain [0.4, 0.72], widened clamp (0.2, 0.8)) and the wrist
# link names differ. See the arena adapter block far below.
# ---------------------------------------------------------------------------
# AGILE wrist link names that ms.WRIST_L/WRIST_R placeholders are replaced with.
# (Literals here because WRIST_LINK_LEFT/RIGHT are defined later in the file; an
# assert below — after that block — pins them equal so the two never drift.)
ARENA_WRIST_L = "left_wrist_yaw_link"
ARENA_WRIST_R = "right_wrist_yaw_link"
# Box/cube half-extents from the SceneSpec (BOX_FULL/CUBE_FULL) so the
# wrist->box / wrist->cube surface distance uses the arena dims.
ARENA_BOX_HALF = tuple(v / 2.0 for v in ms.BOX_FULL)
ARENA_CUBE_HALF = tuple(v / 2.0 for v in ms.CUBE_FULL)
# Squat-height command mapping (AGILE absolute pelvis z). The mission only emits
# a target base height + rate; the harness maps surface height -> base command.
ARENA_STAND_H = STAND_HEIGHT      # 0.72 carry/stand
ARENA_H_MIN = 0.40                # AGILE training-domain floor (height clip lo)
# Widened manager height clamp so the deep reach commands (down to ARENA_H_MIN
# and the OOD ground reach) reach the policy verbatim — saturation at AGILE's
# ~0.49-0.53 depth floor is the measured result (low tables expected pick_failed).
ARENA_HEIGHT_RANGE = (0.2, 0.8)
# reach_base_h(top_h): base height command to bring the wrists to a surface at
# height top_h. Same linear "lower surface -> lower base" fit as bench_homie,
# clipped to AGILE's trackable domain.
ARENA_REACH_OFFSET = 0.10         # m, base above the surface for a tabletop reach
ARENA_STORE_FLOOR_H = ms.Z_STORE_RIM_H            # store-pad floor ~0.12 m
# ROOT CAUSE 1: the store placement squat is kept SHALLOW (a light bend).
# release_box() levels + lowers the box onto the pad regardless of squat depth,
# so a deep store squat under the 2 kg box (topple-prone) is unnecessary.
ARENA_STORE_PLACE_H = ARENA_STAND_H - 0.14        # ~0.58, light bend
ARENA_RELEASE_FWD_M = 0.35        # box set-down forward offset (matches
                                  # manip_mission.PLACE_STAND_FWD_M)
ARENA_PLACE_CLAMP_R = 0.12        # clamp box landing inside the zone footprint
ARENA_VX_CARRY = 0.4             # m/s coarse forward clip while carrying the box
ARENA_SQUAT_RATE = 0.2          # m/s base-height ramp
ARENA_CARRY_H = ARENA_STAND_H   # carry at full stand height (a lower carry
                                # stance hurt stability — same finding as HOMIE)
ARENA_PILLAR_CLEAR_M = 0.4      # min straight-line clearance to the pillar
ARENA_PILLAR_SIDE_M = 0.7       # lateral offset of the inserted skirt waypoint
# Extra squat depth for the M2 cube-touch leg vs the relay-table reach: the
# 0.08 m cube on the 0.55 m relay top needs the right wrist as low as possible.
# Commanding ARENA_CUBE_REACH_DROP below the table-reach saturates AGILE's depth
# floor (h0.45 cmd -> base ~0.61; deeper commands don't lower it further) =
# the smallest reachable cube gap. v3 smoke: 0.39 m at the shallow h0.65 ->
# 0.33 m at the saturated deep squat — right AT the shared CUBE_TOUCH_D 0.32 m
# gate (AGILE's depth limit; HOMIE squats deeper so its cube transfer clears).
ARENA_CUBE_REACH_DROP = 0.20
# AGILE box-carry hug pose for the M-series arena welds (the arena box is
# 0.30 wide; arms hug it between the palms). Same joints as ARM_CARRY_POSE but
# a touch wider roll for the wider arena box.
ARENA_HUG_POSE = dict(ARM_CARRY_POSE)
ARENA_RELEASE_BODY_QPOS = ms.FREE_BODY_CT_CA  # bit-2 release collision flag

# ---- squat_box mechanism v2 ("wrist_weld_v2") -----------------------------
# The box is a FREE body (own freejoint) "hugged" between the two palms by two
# SOFT weld constraints (one per wrist link). Loads go through the arm joints:
# the AGILE arm PD (shoulders kp 90/60/20, elbow kp 60, wrists kp 4!) must hold
# the 2 kg — droop / oscillation is real physics and is recorded as-is.
BOX_MODE = "wrist_weld_v2"
BOX_MASS_KG = 2.0
BOX_KEEP_DIST_M = 0.6  # box_kept criterion: |box_center - torso_link| < 0.6 m all trial
WRIST_LINK_LEFT = "left_wrist_yaw_link"  # unitree_mujoco g1_29dof.xml body names
WRIST_LINK_RIGHT = "right_wrist_yaw_link"
# Pin the arena wrist names (declared earlier) to the canonical carry-box names.
assert (ARENA_WRIST_L, ARENA_WRIST_R) == (WRIST_LINK_LEFT, WRIST_LINK_RIGHT)
PALM_FORWARD_OFFSET_M = 0.10  # palm center ~0.10 m ahead of wrist_yaw origin (+x; rubber hand)

# Scene injection snippets.
# Box: full size 0.25 (x, depth) x 0.35 (y, width across the hands) x 0.25 (z)
# m — geom size = half extents — 2 kg, FREE body, collisions OFF for everything
# (spec: no grasp contact simulation), translucent yellow. The XML pos is a
# placeholder: reset_trial() re-poses it between the palms every trial.
BOX_BODY_XML = (
    '<body name="bench_box" pos="0.25 0 0.9">'
    '<freejoint name="bench_box_joint"/>'
    f'<geom name="bench_box_geom" type="box" size="0.125 0.175 0.125" mass="{BOX_MASS_KG}" '
    'rgba="0.9 0.8 0.1 0.45" contype="0" conaffinity="0"/></body>'
)
# Soft solref so the welds do not fight the arm PD into numerical blow-up.
# relpose is rewritten per trial from carry-pose FK (see place_box_in_hands).
BOX_WELD_SOLREF = "0.02 1"
BOX_EQUALITY_XML = (
    "<equality>"
    f'<weld name="bench_box_weld_left" body1="{WRIST_LINK_LEFT}" body2="bench_box" solref="{BOX_WELD_SOLREF}"/>'
    f'<weld name="bench_box_weld_right" body1="{WRIST_LINK_RIGHT}" body2="bench_box" solref="{BOX_WELD_SOLREF}"/>'
    "</equality>"
)
# Pillar: r=0.15, h=1.2 (half-height 0.6), at circle center (0, 1, 0); collidable, static.
PILLAR_GEOM_XML = (
    '<geom name="bench_pillar" type="cylinder" size="0.15 0.6" '
    f'pos="{CIRCLE_CENTER[0]} {CIRCLE_CENTER[1]} 0.6" rgba="0.5 0.5 0.5 1"/>'
)
# T9 squat_pick_ground box: SAME body/geom/weld names as the v2 box but with
# bit-2 collisions from compile time (box vs floor only, never the robot —
# identical scheme to the psi0 cracker box; the floor planes get bit 2 OR-ed
# in at runtime). The XML pos is a placeholder: reset_trial() puts it
# PICK_BOX_AHEAD_M in front of the robot, resting on the floor.
PICK_BOX_XML = (
    '<body name="bench_box" pos="0.45 0 0.13">'
    '<freejoint name="bench_box_joint"/>'
    f'<geom name="bench_box_geom" type="box" '
    f'size="{PICK_BOX_HALF[0]} {PICK_BOX_HALF[1]} {PICK_BOX_HALF[2]}" '
    f'mass="{BOX_MASS_KG}" rgba="0.9 0.8 0.1 0.7" '
    'contype="2" conaffinity="2"/></body>'
)

# ---- psi0 upper-body replay (contract v1; psi0_replay/make_replay.py) ------
# npz fields: t (N,), upper_names (K,), upper_pos (N,K), height_cmd (N,),
# grasp_close_t (scalar), carry_pose (K,), meta_json. agile29: K=17 =
# arms14 + [waist_yaw, waist_roll, waist_pitch]; columns mapped by joint NAME.
PSI0_GRASP_DIST_M = 0.30  # both wrists closer than this to box surface ("magnetic
# grasp": the replayed AMO trajectory bends the torso deeper than leg-only
# policies can; smoke showed wrist gaps of 0.15-0.23 m at grasp time)
PSI0_HEIGHT_CLIP = (0.4, STAND_HEIGHT)  # AGILE height domain for the replayed height_cmd
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
# Collision masks (unified psi0 spec): robot keeps bit 1, box gets bit 2, the
# table gets bits 1|2 and the floor planes get bit 2 OR-ed in at runtime ->
# the box collides with table/floor but never with the robot.
PSI0_BOX_XML = (
    '<body name="bench_box" pos="0.3 0 0.9">'
    '<freejoint name="bench_box_joint"/>'
    f'<geom name="bench_box_geom" type="box" '
    f'size="{PSI0_BOX_HALF[0]} {PSI0_BOX_HALF[1]} {PSI0_BOX_HALF[2]}" '
    f'mass="{PSI0_BOX_MASS_KG}" rgba="0.85 0.3 0.15 0.8" '
    'contype="2" conaffinity="2"/></body>'
)
PSI0_TABLE_XML = (
    f'<geom name="bench_table" type="box" '
    f'size="{PSI0_TABLE_HALF[0]} {PSI0_TABLE_HALF[1]} {PSI0_TABLE_HALF[2]}" '
    f'pos="{PSI0_TABLE_CENTER[0]} {PSI0_TABLE_CENTER[1]} {PSI0_TABLE_CENTER[2]}" '
    'contype="3" conaffinity="3" rgba="0.55 0.42 0.26 1"/>'
)

DEFAULT_AGILE_REPO = os.environ.get(
    "AGILE_REPO", str(Path.home() / "Project/sim2real/cc/experiments/repos/WBC-AGILE")
)
DEFAULT_MJCF = os.environ.get(
    "UNITREE_MUJOCO_SCENE", str(Path.home() / "unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
)
REL_CHECKPOINT = "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt"
REL_CONFIG = "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml"


# ---------------------------------------------------------------------------
# AGILE imports (lazy: path setup + light stubs for the heavy plotting chain)
# ---------------------------------------------------------------------------


def import_agile(repo: Path) -> SimpleNamespace:
    """Import agile.sim2mujoco with stubs so pandas/matplotlib/... are not required.

    agile.sim2mujoco.__init__ imports command_scheduler (-> agile.algorithms.evaluation
    -> matplotlib/seaborn/plotly/jinja2) and data_logger (-> pandas). The harness uses
    neither, so pre-registering stub modules keeps the dependency set minimal.
    """
    if not (repo / "agile" / "sim2mujoco").is_dir():
        raise SystemExit(f"AGILE repo not found / wrong layout: {repo} (need <repo>/agile/sim2mujoco)")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    stub_specs = {
        "agile.sim2mujoco.command_scheduler": ("Sim2MuJoCoCommandScheduler", "RandomCommandScheduler"),
        "agile.sim2mujoco.data_logger": ("Sim2MuJoCoDataLogger",),
    }
    for mod_name, attr_names in stub_specs.items():
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            for attr in attr_names:
                setattr(mod, attr, type(attr, (), {"__doc__": "bench_agile stub (unused)"}))
            sys.modules[mod_name] = mod

    from agile.sim2mujoco.actions import ActionProcessor
    from agile.sim2mujoco.command_provider import VelocityCommandProvider, create_command_provider
    from agile.sim2mujoco.observations import ObservationProcessor
    from agile.sim2mujoco.policy import PolicyWrapper
    from agile.sim2mujoco.simulation import JointCommand, MuJocoSimulation
    from agile.sim2mujoco.utils import load_config

    return SimpleNamespace(
        ActionProcessor=ActionProcessor,
        create_command_provider=create_command_provider,
        VelocityCommandProvider=VelocityCommandProvider,
        ObservationProcessor=ObservationProcessor,
        PolicyWrapper=PolicyWrapper,
        JointCommand=JointCommand,
        MuJocoSimulation=MuJocoSimulation,
        load_config=load_config,
    )


# ---------------------------------------------------------------------------
# Scene variants (pure XML text injection; files written next to the original
# so relative meshdir/include/asset paths keep resolving)
# ---------------------------------------------------------------------------


def build_scene_variant(scene_path: Path, test: str) -> Path:
    """Return path to the (possibly modified) scene XML for the given test."""
    if test in ("walk_speed", "speed_sweep", "goto_ab", "vln_follow"):
        return scene_path
    if test not in TESTS:
        raise ValueError(f"Unknown test: {test}")

    # TODO(verify-on-server): assumes literal "<worldbody>" tag in scene_29dof.xml
    # (verified against unitreerobotics/unitree_mujoco@main on 2026-06-10).
    scene_text = scene_path.read_text()
    for token in ("<worldbody>", "</worldbody>", "</mujoco>"):
        if token not in scene_text:
            raise SystemExit(f"Cannot find {token} in {scene_path}")

    out_text = scene_text
    if test in ("circle_pillar", "circle_pillar_psi0"):
        out_text = out_text.replace("<worldbody>", "<worldbody>\n    " + PILLAR_GEOM_XML, 1)
    if test == "squat_box_psi0":
        out_text = out_text.replace("<worldbody>", "<worldbody>\n    " + PSI0_TABLE_XML, 1)

    # Free box + wrist welds. v2 (wrist_weld_v2): append the free box to the
    # SCENE worldbody (the robot include is untouched) + two weld constraints
    # to the wrist links. Because the scene <worldbody> merges AFTER the robot
    # include, the box freejoint compiles after the 29 robot hinges, so
    # AGILE's qpos[7:] / qvel[6:] joint slices stay index-stable (obs terms
    # only gather indices 0..28). Verified in build_box_rig(). psi0 tests use
    # the cracker-box variant (same body/geom/weld names, different size/mass/
    # collision bits).
    box_xml = None
    if test in CARRY_TESTS or test == "squat_place_psi0":
        box_xml = BOX_BODY_XML  # v2 2 kg box (squat_place_psi0 included)
    elif test in PSI0_TESTS:
        box_xml = PSI0_BOX_XML
    elif test == "squat_pick_ground":
        box_xml = PICK_BOX_XML  # v2-sized 2 kg box, bit-2 floor collisions
    if box_xml is not None:
        out_text = out_text.replace("</worldbody>", "    " + box_xml + "\n  </worldbody>", 1)
        out_text = out_text.replace("</mujoco>", "  " + BOX_EQUALITY_XML + "\n</mujoco>", 1)

    out_path = scene_path.with_name(scene_path.stem + f"__bench_{test}.xml")
    out_path.write_text(out_text)
    return out_path


# ---------------------------------------------------------------------------
# Geometry / kinematics helpers
# ---------------------------------------------------------------------------


def quat_yaw(q: np.ndarray) -> float:
    """Yaw from wxyz quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_tilt(q: np.ndarray) -> float:
    """Tilt angle = arccos(-projected_gravity_z) from wxyz quaternion.

    For unit gravity in base frame pg_z = -(1 - 2(x^2 + y^2)), so
    tilt = arccos(1 - 2(x^2 + y^2)).
    """
    _, x, y, _ = q
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def yaw_to_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------------------
# Unified two-stage navigation controller (spec v3) — SHARED MATH.
# This block is intentionally line-for-line identical across bench_agile.py /
# bench_homie.py / bench_falcon.py / bench_amo.py (transcribed from the spec;
# do not "improve" one copy without changing all four).
# ---------------------------------------------------------------------------
NAV_SWITCH_DIST_M = 0.5  # >0.5 m: head toward the goal point; <=0.5 m: head to goal yaw
NAV_KP_POS = 1.0
NAV_KP_YAW = 1.5
NAV_VX_RANGE = (0.0, 0.6)  # m/s coarse stage (pipeline_abc lowers the max to 0.5)
NAV_VY_MAX = 0.3
NAV_WZ_MAX = 0.6
NAV_EXIT_POS_M = 0.30
NAV_EXIT_YAW_RAD = math.radians(15.0)
NAV_TIMEOUT_S = 30.0
CAL_CMD_LIN_MAX = 0.10  # m/s   small-command calibration clamp
CAL_CMD_WZ_MAX = 0.10  # rad/s
CAL_POS_TOL_M = 0.05
CAL_YAW_TOL_RAD = math.radians(5.0)
CAL_HOLD_S = 1.0
CAL_TIMEOUT_S = 15.0


def nav_errors(x: float, y: float, yaw: float, goal_xy: tuple[float, float],
               goal_yaw: float) -> tuple[float, float, float, float]:
    """Body-frame position error + heading error with the two-zone heading
    target (far: bearing to the goal point; near: the goal orientation)."""
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    dist = math.hypot(dx, dy)
    ex = math.cos(yaw) * dx + math.sin(yaw) * dy
    ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
    heading_target = math.atan2(dy, dx) if dist > NAV_SWITCH_DIST_M else goal_yaw
    eyaw = wrap_angle(heading_target - yaw)
    return dist, ex, ey, eyaw


def nav_cmd_coarse(ex: float, ey: float, eyaw: float, vx_max: float) -> tuple[float, float, float]:
    vx = min(max(NAV_KP_POS * ex, NAV_VX_RANGE[0]), vx_max)
    vy = min(max(NAV_KP_POS * ey, -NAV_VY_MAX), NAV_VY_MAX)
    wz = min(max(NAV_KP_YAW * eyaw, -NAV_WZ_MAX), NAV_WZ_MAX)
    return vx, vy, wz


def nav_cmd_cal(ex: float, ey: float, eyaw: float) -> tuple[float, float, float]:
    """Small-command calibration: same P law, commands clamped to <=0.10.
    NO deadband compensation on purpose — the test measures the policy's
    native small-command effectiveness."""
    vx = min(max(NAV_KP_POS * ex, -CAL_CMD_LIN_MAX), CAL_CMD_LIN_MAX)
    vy = min(max(NAV_KP_POS * ey, -CAL_CMD_LIN_MAX), CAL_CMD_LIN_MAX)
    wz = min(max(NAV_KP_YAW * eyaw, -CAL_CMD_WZ_MAX), CAL_CMD_WZ_MAX)
    return vx, vy, wz
# ------------------- end shared navigation controller ----------------------


# T5 goto_ab
GOTO_AB_GOAL_XY = (3.0, 1.0)
GOTO_AB_GOAL_YAW = math.pi / 2.0
GOTO_AB_YAW0_NOISE = 0.3  # rad, initial yaw noise (seeded)

# T6 pipeline_abc
PIPE_GOAL_XY = (0.0, -2.5)
PIPE_GOAL_YAW = -math.pi / 2.0  # turn right ~90deg, then 2.5 m straight
PIPE_NAV_VX_MAX = 0.5  # conservative while carrying the box
PIPE_PLACE_HEIGHT = 0.45
PIPE_PLACE_RATE = 0.2  # m/s squat ramp speed
PIPE_HOLD_BEFORE_RELEASE_S = 0.5
PIPE_HOLD_AFTER_RELEASE_S = 1.0
PIPE_STAND_S = 1.0
BOX_PLACE_VEL_MAX = 0.05  # m/s: the placed box must be at rest
BOX_PLACE_UPRIGHT_RAD = math.radians(30.0)  # box z-axis vs world z
BOX_PLACE_NEAR_M = 0.8  # placed box must be within this distance of the robot
PIPE_VIDEO_MAX_SUCCESS = 3  # user spec: first 3 successes + all failures


@dataclass(frozen=True)
class GeomSets:
    floor: frozenset[int]
    foot: frozenset[int]
    nonfoot_robot: frozenset[int]
    robot: frozenset[int]
    pillar: int  # -1 if absent


def classify_geoms(model: mujoco.MjModel) -> GeomSets:
    """Classify geoms for fall / collision detection."""

    def body_name(bid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise SystemExit("Body 'pelvis' not found in MJCF — wrong scene file?")

    robot_bodies: set[int] = set()
    for b in range(model.nbody):
        cur = b
        while cur != 0:
            if cur == pelvis_id:
                robot_bodies.add(b)
                break
            cur = int(model.body_parentid[cur])

    foot_bodies = {b for b in robot_bodies if "ankle_roll" in body_name(b)}
    pillar_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bench_pillar")

    # NOTE: the free bench_box body (squat_box v2) is a worldbody child, not a
    # pelvis descendant, so its geom is naturally excluded here (contype=0 too).
    floor_g, robot_g, foot_g = set(), set(), set()
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in robot_bodies:
            robot_g.add(g)
            if b in foot_bodies:
                foot_g.add(g)
        elif b == 0 and g != pillar_gid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            floor_g.add(g)

    if not foot_g:
        raise SystemExit("No foot (ankle_roll) geoms found — fall detection would be wrong.")
    return GeomSets(
        floor=frozenset(floor_g),
        foot=frozenset(foot_g),
        nonfoot_robot=frozenset(robot_g - foot_g),
        robot=frozenset(robot_g),
        pillar=int(pillar_gid),
    )


def scan_contacts(data: mujoco.MjData, geoms: GeomSets) -> tuple[bool, bool]:
    """Return (nonfoot_ground_contact, pillar_contact)."""
    nonfoot_ground = False
    pillar_hit = False
    for ci in range(data.ncon):
        g1 = int(data.contact.geom1[ci])
        g2 = int(data.contact.geom2[ci])
        if (g1 in geoms.floor and g2 in geoms.nonfoot_robot) or (g2 in geoms.floor and g1 in geoms.nonfoot_robot):
            nonfoot_ground = True
        if geoms.pillar >= 0 and (
            (g1 == geoms.pillar and g2 in geoms.robot) or (g2 == geoms.pillar and g1 in geoms.robot)
        ):
            pillar_hit = True
    return nonfoot_ground, pillar_hit


# ---------------------------------------------------------------------------
# Command profiles  (t_cmd: seconds since end of settle; t_cmd < 0 -> settle)
# ---------------------------------------------------------------------------


def command_profile(test: str, t_cmd: float, vx_target: float, ccw: bool,
                    target_h: float | None = None, ramp_speed: float | None = None,
                    replay: SimpleNamespace | None = None) -> tuple[float, float, float, float, str]:
    """Return (vx, vy, wz, height, phase_label) at time t_cmd."""
    if t_cmd < 0.0:
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "settle"

    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        if t_cmd < WALK_RAMP_S:
            return vx_target * t_cmd / WALK_RAMP_S, 0.0, 0.0, STAND_HEIGHT, "ramp"
        return vx_target, 0.0, 0.0, STAND_HEIGHT, "hold"

    if test == "squat_box_psi0":
        # Replay segment: legs follow the (pre-smoothed) replayed height_cmd,
        # clipped to AGILE's height domain; then the standard squat cycles.
        if t_cmd < replay.duration_s:
            k = min(int(t_cmd / CONTROL_DT), replay.n - 1)
            h = min(max(float(replay.height[k]), PSI0_HEIGHT_CLIP[0]), PSI0_HEIGHT_CLIP[1])
            return 0.0, 0.0, 0.0, h, "replay"
        return command_profile("squat_box", t_cmd - replay.duration_s, vx_target, ccw)

    if test == "squat_place_psi0":
        # Single (non-looped) playback: legs follow the replayed height_cmd
        # (clipped to AGILE's domain); phases from the precomputed bottom
        # plateau (descent -> hold -> rise), then a 1 s stand segment.
        if t_cmd < replay.duration_s:
            k = min(int(t_cmd / CONTROL_DT), replay.n - 1)
            h = min(max(float(replay.height[k]), PSI0_HEIGHT_CLIP[0]), PSI0_HEIGHT_CLIP[1])
            if k < replay.hold_start_idx:
                phase = "descent"
            elif k <= replay.hold_end_idx:
                phase = "hold"
            else:
                phase = "rise"
            return 0.0, 0.0, 0.0, h, phase
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "stand_end"

    if test == "squat_limit":
        # Continuous 0.05 m/s descent to the 0.10 m command floor, hold 3 s,
        # END (no rise). The widened manager clamp delivers 0.10 verbatim.
        down_s = (STAND_HEIGHT - SQUAT_LIMIT_FLOOR_H) / SQUAT_LIMIT_RATE
        if t_cmd < down_s:
            return 0.0, 0.0, 0.0, STAND_HEIGHT - SQUAT_LIMIT_RATE * t_cmd, "descend"
        return 0.0, 0.0, 0.0, SQUAT_LIMIT_FLOOR_H, "hold_floor"

    if test == "squat_sweep":
        down_s = (STAND_HEIGHT - target_h) / ramp_speed
        if t_cmd < down_s:
            return 0.0, 0.0, 0.0, STAND_HEIGHT - ramp_speed * t_cmd, "down"
        if t_cmd < down_s + SQUAT_SWEEP_HOLD_S:
            return 0.0, 0.0, 0.0, target_h, "hold"
        if t_cmd < down_s + SQUAT_SWEEP_HOLD_S + down_s:
            tu = t_cmd - down_s - SQUAT_SWEEP_HOLD_S
            return 0.0, 0.0, 0.0, min(target_h + ramp_speed * tu, STAND_HEIGHT), "up"
        return 0.0, 0.0, 0.0, STAND_HEIGHT, "stand_end"

    if test == "squat_box":
        cycle = int(t_cmd // SQUAT_CYCLE_S)
        if cycle >= SQUAT_N_CYCLES:
            return 0.0, 0.0, 0.0, STAND_HEIGHT, "done"
        tc = t_cmd - cycle * SQUAT_CYCLE_S
        if tc < SQUAT_RAMP_DOWN_S:
            h = STAND_HEIGHT + (SQUAT_HEIGHT - STAND_HEIGHT) * tc / SQUAT_RAMP_DOWN_S
            phase = "ramp_down"
        elif tc < SQUAT_RAMP_DOWN_S + SQUAT_HOLD_DOWN_S:
            h, phase = SQUAT_HEIGHT, "hold_down"
        elif tc < SQUAT_RAMP_DOWN_S + SQUAT_HOLD_DOWN_S + SQUAT_RAMP_UP_S:
            tu = tc - SQUAT_RAMP_DOWN_S - SQUAT_HOLD_DOWN_S
            h = SQUAT_HEIGHT + (STAND_HEIGHT - SQUAT_HEIGHT) * tu / SQUAT_RAMP_UP_S
            phase = "ramp_up"
        else:
            h, phase = STAND_HEIGHT, "hold_up"
        return 0.0, 0.0, 0.0, h, f"{phase}_c{cycle:02d}"

    if test in ("circle_pillar", "circle_pillar_psi0"):
        wz = CIRCLE_WZ if ccw else -CIRCLE_WZ
        phase = "circle1" if t_cmd < CIRCLE_PERIOD_S else "circle2"
        return CIRCLE_VX, 0.0, wz, STAND_HEIGHT, phase

    raise ValueError(test)


def test_duration_s(test: str, target_h: float | None = None,
                    ramp_speed: float | None = None,
                    replay: SimpleNamespace | None = None) -> float:
    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        return WALK_RAMP_S + WALK_HOLD_S
    if test == "squat_box":
        return SQUAT_N_CYCLES * SQUAT_CYCLE_S
    if test == "squat_box_psi0":
        return replay.duration_s + SQUAT_N_CYCLES * SQUAT_CYCLE_S
    if test == "squat_place_psi0":
        return replay.duration_s + SQUAT_PLACE_STAND_S
    if test == "squat_limit":
        return (STAND_HEIGHT - SQUAT_LIMIT_FLOOR_H) / SQUAT_LIMIT_RATE + SQUAT_LIMIT_HOLD_S
    if test == "squat_sweep":
        down_s = (STAND_HEIGHT - target_h) / ramp_speed
        return down_s + SQUAT_SWEEP_HOLD_S + down_s + SQUAT_SWEEP_END_S
    if test in ("circle_pillar", "circle_pillar_psi0"):
        return CIRCLE_N_LAPS * CIRCLE_PERIOD_S
    raise ValueError(test)


# ---------------------------------------------------------------------------
# Rig: simulation + processors built once per test (model is test-specific)
# ---------------------------------------------------------------------------


class BodyFrameGyroShim:
    """Exact root angular velocity source for AGILE's sensor code path.

    MuJoCo free-joint qvel[3:6] is ALREADY the root angular velocity in the
    body-local frame (verified empirically vs mj_objectVelocity). AGILE's
    no-sensor fallback (agile/sim2mujoco/simulation.py:419-421) wrongly treats
    it as world-frame and rotates it again by the inverse root quaternion —
    a no-op at yaw=0 but it corrupts base_ang_vel at large yaw (random-yaw
    init / sustained turning), destabilizing the policy within ~0.5 s.
    The unitree G1 MJCF has no "angular-velocity" sensor, so that buggy
    fallback would always be active. Injecting this shim as the sensor makes
    the framework consume qvel[3:6] verbatim (the sensor branch uses values
    as root-frame directly), which is mathematically exact.
    """

    def __init__(self, mj_data: mujoco.MjData):
        self._mj_data = mj_data

    @property
    def data(self) -> np.ndarray:
        return self._mj_data.qvel[3:6]


@dataclass(frozen=True)
class BoxRig:
    """Model ids / addresses for the free box + wrist welds (squat_box v2)."""

    body_id: int
    geom_id: int
    qpos_adr: int  # 7 qpos numbers (free joint) start here
    dof_adr: int  # 6 dofs start here
    torso_body_id: int
    wrist_body_ids: tuple[int, int]  # (left, right)
    eq_ids: tuple[int, int]  # (left, right) weld equality ids


def build_box_rig(model: mujoco.MjModel) -> BoxRig:
    def need(objtype: mujoco.mjtObj, name: str) -> int:
        oid = mujoco.mj_name2id(model, objtype, name)
        if oid < 0:
            raise SystemExit(f"squat_box v2: '{name}' missing after scene injection")
        return int(oid)

    body_id = need(mujoco.mjtObj.mjOBJ_BODY, "bench_box")
    jnt_id = need(mujoco.mjtObj.mjOBJ_JOINT, "bench_box_joint")
    qpos_adr = int(model.jnt_qposadr[jnt_id])
    if qpos_adr != model.nq - 7:
        # AGILE slices joint qpos as qpos[7:] and gathers indices 0..28; the
        # box free joint must therefore be the LAST joint in the model.
        raise SystemExit(
            f"bench_box freejoint qpos_adr={qpos_adr} != nq-7={model.nq - 7}; "
            "injection order would corrupt AGILE's qpos[7:] joint slicing"
        )
    return BoxRig(
        body_id=body_id,
        geom_id=need(mujoco.mjtObj.mjOBJ_GEOM, "bench_box_geom"),
        qpos_adr=qpos_adr,
        dof_adr=int(model.jnt_dofadr[jnt_id]),
        torso_body_id=need(mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
        wrist_body_ids=(
            need(mujoco.mjtObj.mjOBJ_BODY, WRIST_LINK_LEFT),
            need(mujoco.mjtObj.mjOBJ_BODY, WRIST_LINK_RIGHT),
        ),
        eq_ids=(
            need(mujoco.mjtObj.mjOBJ_EQUALITY, "bench_box_weld_left"),
            need(mujoco.mjtObj.mjOBJ_EQUALITY, "bench_box_weld_right"),
        ),
    )


def place_box_in_hands(model: mujoco.MjModel, data: mujoco.MjData, box: BoxRig) -> None:
    """Pose the free box between the palms and re-anchor both welds.

    Call AFTER the arm joints are set to the carry pose and mj_forward has
    run. Rewrites the weld relpose (eq_data[3:10] = pos+quat of body2 in
    body1 frame) from the live FK: the XML default relpose would have been
    captured at qpos0 (arms hanging down), which is wrong for the carry pose.
    """
    palms = []
    offset = np.array([PALM_FORWARD_OFFSET_M, 0.0, 0.0])
    for wid in box.wrist_body_ids:
        rot = data.xmat[wid].reshape(3, 3)
        palms.append(data.xpos[wid] + rot @ offset)
    center = 0.5 * (palms[0] + palms[1])
    yaw = quat_yaw(np.asarray(data.qpos[3:7], dtype=np.float64))
    data.qpos[box.qpos_adr : box.qpos_adr + 3] = center
    data.qpos[box.qpos_adr + 3 : box.qpos_adr + 7] = yaw_to_quat(yaw)
    data.qvel[box.dof_adr : box.dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    for eq_id, wid in zip(box.eq_ids, box.wrist_body_ids):
        rot_w = data.xmat[wid].reshape(3, 3)
        relp = rot_w.T @ (data.xpos[box.body_id] - data.xpos[wid])
        neg = np.zeros(4)
        relq = np.zeros(4)
        mujoco.mju_negQuat(neg, data.xquat[wid])
        mujoco.mju_mulQuat(relq, neg, data.xquat[box.body_id])
        model.eq_data[eq_id, 3:6] = relp
        model.eq_data[eq_id, 6:10] = relq


def activate_welds_at_current_pose(model: mujoco.MjModel, data: mujoco.MjData, box: BoxRig) -> None:
    """Anchor both wrist welds at the CURRENT relative wrist->box pose and
    enable them. Unlike place_box_in_hands the box is NOT moved —
    squat_box_psi0 grasps the box where it stands on the table."""
    for eq_id, wid in zip(box.eq_ids, box.wrist_body_ids):
        rot_w = data.xmat[wid].reshape(3, 3)
        relp = rot_w.T @ (data.xpos[box.body_id] - data.xpos[wid])
        neg = np.zeros(4)
        relq = np.zeros(4)
        mujoco.mju_negQuat(neg, data.xquat[wid])
        mujoco.mju_mulQuat(relq, neg, data.xquat[box.body_id])
        model.eq_data[eq_id, 0:3] = 0.0  # anchor at the box origin
        model.eq_data[eq_id, 3:6] = relp
        model.eq_data[eq_id, 6:10] = relq
        model.eq_data[eq_id, 10] = 1.0  # torquescale
        data.eq_active[eq_id] = 1


def enable_floor_bit2(model: mujoco.MjModel) -> int:
    """OR bit 2 into every worldbody floor plane (geom AND the world-body
    compile-time aggregates) so a contype/conaffinity-2 box can collide with
    the floor. MuJoCo >= 3.2.4 culls BODY pairs via body_contype/
    body_conaffinity, so the geom-level writes alone would be invisible to
    broadphase. Shared by the psi0 tests, squat_pick_ground and release_box.
    Returns the number of plane geoms patched."""
    n_floor = 0
    for g in range(model.ngeom):
        if (model.geom_bodyid[g] == 0
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE):
            model.geom_contype[g] |= 2
            model.geom_conaffinity[g] |= 2
            n_floor += 1
    if hasattr(model, "body_contype"):
        model.body_contype[0] |= 2
        model.body_conaffinity[0] |= 2
    return n_floor


def release_box(model: mujoco.MjModel, data: mujoco.MjData, box: BoxRig) -> None:
    """Release the carried box: disable both wrist welds and flip the box to
    bit-2 collisions so it falls and lands on the floor.

    bit-2 scheme (floor OR bit 2, robot bit 1): the box collides with the
    floor but never with the robot — contype=1 caused depenetration buzz and
    box_place_ok always failed. The body-level aggregates must follow the
    geom bits (see enable_floor_bit2), otherwise the box would free-fall
    through the floor. Shared by pipeline_abc and squat_place_psi0 (spec v4)."""
    for eq_id in box.eq_ids:
        data.eq_active[eq_id] = 0
    model.geom_contype[box.geom_id] = 2
    model.geom_conaffinity[box.geom_id] = 2
    enable_floor_bit2(model)
    if hasattr(model, "body_contype"):
        model.body_contype[box.body_id] = 2
        model.body_conaffinity[box.body_id] = 2


def foot_front_proj_m(model: mujoco.MjModel, data: mujoco.MjData, geoms: GeomSets,
                      heading: tuple[float, float]) -> float:
    """Forward (heading-projected) coordinate of the foot FRONT EDGE:
    max over foot geoms of (geom center projection + geom x half-size).
    Approximation: uses geom_size[0] (box half-length / capsule radius) as the
    forward half-extent, good to ~1-2 cm on the G1 foot geoms."""
    hx, hy = heading
    return max(
        float(data.geom_xpos[g][0] * hx + data.geom_xpos[g][1] * hy + model.geom_size[g][0])
        for g in geoms.foot
    )


def box_surface_dist_m(data: mujoco.MjData, box: BoxRig, wrist_body_id: int,
                       half: tuple[float, float, float]) -> float:
    """Distance from a wrist body origin to the oriented box surface (0 inside)."""
    rel = data.xpos[wrist_body_id] - data.xpos[box.body_id]
    local = data.xmat[box.body_id].reshape(3, 3).T @ rel
    half_arr = np.asarray(half, dtype=np.float64)
    return float(np.linalg.norm(local - np.clip(local, -half_arr, half_arr)))


def load_upper_replay(path: Path) -> SimpleNamespace:
    """Load a contract-v1 upper-body replay npz (psi0_replay/make_replay.py).

    Returns a namespace with the raw arrays plus torch tensors (pos_t /
    carry_t, filled in build_rig once the device is known).
    """
    if not path.exists():
        raise SystemExit(f"upper-replay npz not found: {path}")
    z = np.load(path, allow_pickle=True)
    required = ("t", "upper_names", "upper_pos", "height_cmd",
                "grasp_close_t", "carry_pose", "meta_json")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise SystemExit(f"upper-replay npz missing fields {missing}: {path}")
    meta = json.loads(str(z["meta_json"]))
    t = np.asarray(z["t"], dtype=np.float64)
    pos = np.asarray(z["upper_pos"], dtype=np.float32)
    if len(t) < 2 or abs((t[1] - t[0]) - CONTROL_DT) > 1e-6:
        raise SystemExit(
            f"upper-replay dt {t[1] - t[0] if len(t) > 1 else 'n/a'} != "
            f"harness control dt {CONTROL_DT}: {path}"
        )
    names = [str(n) for n in z["upper_names"]]
    if pos.shape != (t.shape[0], len(names)):
        raise SystemExit(f"upper-replay shape mismatch {pos.shape}: {path}")
    if not np.isfinite(pos).all():
        raise SystemExit(f"upper-replay contains NaN/inf: {path}")
    height = np.asarray(z["height_cmd"], dtype=np.float64)
    if height.shape[0] != pos.shape[0]:
        raise SystemExit(f"upper-replay height_cmd length {height.shape[0]} != "
                         f"frames {pos.shape[0]}: {path}")
    # squat_place_psi0 (spec v4): t_place = argmin(height_cmd) = weld release
    # moment; the bottom plateau (height within h_min + band) splits the replay
    # into descent / hold / rise segments for the per-segment RMSE.
    place_idx = int(np.argmin(height))
    hold_idx = np.where(height <= height[place_idx] + SQUAT_PLACE_HOLD_BAND_M)[0]
    return SimpleNamespace(
        path=str(path),
        names=names,
        pos=pos,
        height=height,
        place_idx=place_idx,
        t_place_s=place_idx * CONTROL_DT,
        hold_start_idx=int(hold_idx[0]),
        hold_end_idx=int(hold_idx[-1]),
        grasp_close_t=float(z["grasp_close_t"]),
        carry_pose=np.asarray(z["carry_pose"], dtype=np.float32),
        n=int(pos.shape[0]),
        duration_s=float(pos.shape[0]) * CONTROL_DT,
        source=str(meta.get("source", "unknown")),
        embodiment=str(meta.get("embodiment", "unknown")),
        pos_t=None,
        carry_t=None,
    )


_BOXDEMO_ASSETS = None


def _load_box_demo_assets() -> dict:
    """Load the pure-kinematics slice from box_demo_2 without Unitree SDK deps."""
    global _BOXDEMO_ASSETS
    if _BOXDEMO_ASSETS is not None:
        return _BOXDEMO_ASSETS
    from enum import IntEnum

    here = Path(__file__).resolve()
    candidates = []
    env_dir = os.environ.get("BOX_DEMO_2_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "dual_arm_target_reach.py")
    candidates.extend([
        here.parent / "box_demo_2" / "dual_arm_target_reach.py",
        here.parents[3] / "box_demo_2" / "dual_arm_target_reach.py"
        if len(here.parents) > 3 else None,
        Path("/sda/lizhe/g1bench/box_demo_2/dual_arm_target_reach.py"),
        Path.home() / "Project/sim2real/box_demo_2/dual_arm_target_reach.py",
    ])
    src_path = next((p for p in candidates if p is not None and p.exists()), None)
    if src_path is None:
        tried = ", ".join(str(p) for p in candidates if p is not None)
        raise SystemExit(f"box_demo_2 dual_arm_target_reach.py not found; tried: {tried}")
    text = src_path.read_text()
    start = text.index("class G1Joint")
    end = text.index("class _LegacyStageUnused")
    ns = {
        "__name__": "_box_demo_2_kinematics",
        "math": math,
        "np": np,
        "IntEnum": IntEnum,
    }
    exec(text[start:end], ns)  # noqa: S102 - local trusted project source
    _BOXDEMO_ASSETS = {
        "path": str(src_path),
        "ArmKinematics": ns["ArmKinematics"],
        "ZERO_Q": np.asarray(ns["ZERO_Q"], dtype=np.float32),
        "RL_LOWER_HANDOFF_Q": np.asarray(ns["RL_LOWER_HANDOFF_Q"], dtype=np.float32),
    }
    return _BOXDEMO_ASSETS


def _boxdemo_pose_row(pose: dict[str, float]) -> np.ndarray:
    return np.asarray([pose.get(n, 0.0) for n in BOXDEMO_UPPER_NAMES], dtype=np.float32)


def _ease01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _lerp_row(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    t = _ease01(alpha)
    return (1.0 - t) * a + t * b


def _solve_boxdemo_pair(ik, left_target, right_target, left_q0=None, right_q0=None,
                        restarts: int = 2, max_iter: int = 300) -> tuple[np.ndarray, float, float]:
    left_q, left_err = ik.inverse_kinematics(
        left_target, left=True, q0=left_q0, num_restarts=restarts,
        max_iter=max_iter, tol=1e-4)
    right_q, right_err = ik.inverse_kinematics(
        right_target, left=False, q0=right_q0, num_restarts=restarts,
        max_iter=max_iter, tol=1e-4)
    row = np.asarray(left_q + right_q + [0.0, 0.0, 0.0], dtype=np.float32)
    return row, float(left_err), float(right_err)


def make_boxdemo_scripted_upper(test: str) -> SimpleNamespace:
    """Replay-like upper object for legacy R2/R3 tests.

    R6 uses BoxDemoUpperProvider online against live scene state. Legacy
    tests only have the replay contract, so we synthesize a deterministic
    box_demo-style sequence from the same IK and handoff pose.
    """
    assets = _load_box_demo_assets()
    ik = assets["ArmKinematics"]()
    carry = assets["RL_LOWER_HANDOFF_Q"].copy()
    reach, _, _ = _solve_boxdemo_pair(
        ik, [0.34, 0.17, -0.08], [0.34, -0.17, -0.08], restarts=4)
    place, _, _ = _solve_boxdemo_pair(
        ik, [0.38, 0.15, -0.06], [0.38, -0.15, -0.06],
        left_q0=reach[:7], right_q0=reach[7:14], restarts=1)
    n = int(round(BOXDEMO_REPLAY_S / CONTROL_DT))
    pos = np.zeros((n, len(BOXDEMO_UPPER_NAMES)), dtype=np.float32)
    height = np.full((n,), STAND_HEIGHT, dtype=np.float64)
    for k in range(n):
        t = k * CONTROL_DT
        if test == "squat_place_psi0":
            if t < 3.5:
                row = _lerp_row(carry, place, t / 3.5)
            elif t < 5.0:
                row = place
            elif t < 7.0:
                row = _lerp_row(place, carry, (t - 5.0) / 2.0)
            else:
                row = carry
        else:
            if t < 3.5:
                row = _lerp_row(carry, reach, t / 3.5)
            elif t < 5.5:
                row = reach
            elif t < 7.0:
                row = _lerp_row(reach, carry, (t - 5.5) / 1.5)
            else:
                row = carry
        pos[k] = row
        if t < 3.5:
            height[k] = STAND_HEIGHT + (SQUAT_HEIGHT - STAND_HEIGHT) * _ease01(t / 3.5)
        elif t < 5.5:
            height[k] = SQUAT_HEIGHT
        elif t < 7.0:
            height[k] = SQUAT_HEIGHT + (STAND_HEIGHT - SQUAT_HEIGHT) * _ease01((t - 5.5) / 1.5)
    hold = np.where(height <= float(np.min(height)) + SQUAT_PLACE_HOLD_BAND_M)[0]
    hold_start = int(hold[0]) if hold.size else int(3.5 / CONTROL_DT)
    hold_end = int(hold[-1]) if hold.size else int(5.5 / CONTROL_DT)
    return SimpleNamespace(
        path=assets["path"],
        names=list(BOXDEMO_UPPER_NAMES),
        pos=pos,
        height=height,
        place_idx=int(np.argmin(height)),
        t_place_s=float(np.argmin(height)) * CONTROL_DT,
        hold_start_idx=hold_start,
        hold_end_idx=hold_end,
        grasp_close_t=BOXDEMO_GRASP_CLOSE_S,
        carry_pose=carry,
        n=n,
        duration_s=float(n) * CONTROL_DT,
        source=BOXDEMO_UPPER_MODE,
        embodiment="agile29",
        pos_t=None,
        carry_t=None,
        is_boxdemo=True,
    )


class BoxDemoUpperProvider:
    """Online box_demo_2 upper-body IK provider for ManipArena M-series."""

    names = list(BOXDEMO_UPPER_NAMES)
    source = BOXDEMO_UPPER_MODE
    embodiment = "agile29"
    is_boxdemo = True

    def __init__(self):
        assets = _load_box_demo_assets()
        self.path = assets["path"]
        self.ik = assets["ArmKinematics"]()
        self.zero_q = assets["ZERO_Q"].copy()
        self.carry_pose = assets["RL_LOWER_HANDOFF_Q"].copy()
        self.pos = np.asarray([self.carry_pose], dtype=np.float32)
        self.height = np.asarray([STAND_HEIGHT], dtype=np.float64)
        self.n = 1
        self.duration_s = CONTROL_DT
        self.grasp_close_t = BOXDEMO_GRASP_CLOSE_S
        self.t_place_s = 0.0
        self.hold_start_idx = 0
        self.hold_end_idx = 0
        self._cache: dict[tuple, np.ndarray] = {}
        self._left_q0 = None
        self._right_q0 = None
        self.arena_qpos_adrs: list[int] = []
        self.arena_segments = {
            "reach_down": (0, 1),
            "carry": (0, 1),
            "place": (0, 1),
        }

    @staticmethod
    def _body_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
        forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        left = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
        return forward, left

    @staticmethod
    def _torso_target(io, world: np.ndarray, side: str) -> list[float]:
        rx, ry, yaw = io.root_pose()
        rz = io.base_height()
        dx, dy = float(world[0] - rx), float(world[1] - ry)
        x = math.cos(yaw) * dx + math.sin(yaw) * dy
        y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        z = float(world[2] - rz)
        x = float(np.clip(x, 0.12, 0.55))
        if side == "left":
            y = float(np.clip(y, 0.06, 0.42))
        else:
            y = float(np.clip(y, -0.42, -0.06))
        z = float(np.clip(z, -0.16, 0.52))
        return [x, y, z]

    def _box_targets(self, io, target_cube: bool = False) -> tuple[list[float], list[float]]:
        rx, ry, yaw = io.root_pose()
        forward, left_axis = self._body_axes(yaw)
        bx, by, bz, *_ = io.box_pose()
        box_center = np.asarray([bx, by, bz], dtype=np.float64)
        box_left = box_center + left_axis * (ARENA_BOX_HALF[1] + 0.04) + forward * 0.03
        box_right = box_center - left_axis * (ARENA_BOX_HALF[1] + 0.04) + forward * 0.03
        if target_cube:
            cx, cy, cz, *_ = io.cube_pose()
            cube_center = np.asarray([cx, cy, cz], dtype=np.float64)
            cube_right = cube_center - left_axis * (ARENA_CUBE_HALF[1] + 0.04) + forward * 0.03
            return (
                self._torso_target(io, box_left, "left"),
                self._torso_target(io, cube_right, "right"),
            )
        return (
            self._torso_target(io, box_left, "left"),
            self._torso_target(io, box_right, "right"),
        )

    def _place_targets(self, io) -> tuple[list[float], list[float]]:
        _rx, _ry, yaw = io.root_pose()
        forward, left_axis = self._body_axes(yaw)
        root = np.asarray([io.d.qpos[0], io.d.qpos[1], io.base_height()], dtype=np.float64)
        center = root + forward * 0.36
        center[2] = io.base_height() - 0.08
        return (
            self._torso_target(io, center + left_axis * 0.15, "left"),
            self._torso_target(io, center - left_axis * 0.15, "right"),
        )

    def _solve(self, left_target: list[float], right_target: list[float]) -> np.ndarray:
        key = tuple(round(float(v) / 0.05) for v in (*left_target, *right_target))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        row, _le, _re = _solve_boxdemo_pair(
            self.ik, left_target, right_target,
            left_q0=self._left_q0, right_q0=self._right_q0,
            restarts=0 if self._left_q0 is not None else 1,
            max_iter=80 if self._left_q0 is not None else 160)
        self._left_q0 = row[:7].copy()
        self._right_q0 = row[7:14].copy()
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = row
        return row

    def row(self, segment: str, phase_t: float, io=None) -> np.ndarray:
        if segment == "carry" or io is None:
            return self.carry_pose
        if segment == "place":
            target = self._solve(*self._place_targets(io))
            return _lerp_row(self.carry_pose, target, phase_t / 3.0)
        # During M2 cube transfer the box is already grasped but the cube is not.
        target_cube = bool(getattr(io, "box_grasped", False)
                           and not getattr(io, "cube_on_box", False))
        target = self._solve(*self._box_targets(io, target_cube=target_cube))
        return _lerp_row(self.carry_pose, target, phase_t / 3.5)


def prepare_boxdemo_arena_provider(model: mujoco.MjModel,
                                   provider: BoxDemoUpperProvider) -> BoxDemoUpperProvider:
    qpos_adrs = []
    for name in provider.names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"arena box_demo upper joint missing from model: {name}")
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
    provider.arena_qpos_adrs = qpos_adrs
    provider.pos = np.asarray([provider.carry_pose], dtype=np.float32)
    provider.n = 1
    return provider


@dataclass
class Rig:
    A: SimpleNamespace
    config: dict
    sim: object
    obs_processor: object
    act_processor: object
    mgr: object
    checkpoint: Path
    device: torch.device
    test: str
    arm_indices: torch.Tensor | None  # MJCF-order joint indices for carry pose
    arm_targets: torch.Tensor | None
    box: BoxRig | None  # free box (v2 / psi0 cracker), else None
    upper: SimpleNamespace | None  # psi0 upper-body replay, else None
    upper_indices: torch.Tensor | None  # MJCF-order indices for the replay columns
    upper_mode: str | None = None


def build_rig(A: SimpleNamespace, config: dict, mjcf_path: Path, checkpoint: Path, device: torch.device,
              test: str, pd_scale: float, replay: SimpleNamespace | None = None,
              upper_mode: str = "psi0") -> Rig:
    if pd_scale != 1.0:
        # Benchmark default is 1.0 (fidelity); knob kept for diagnosis only.
        robot_cfg = config["articulations"]["robot"]
        robot_cfg["default_joint_stiffness"] = [k * pd_scale for k in robot_cfg["default_joint_stiffness"]]
        robot_cfg["default_joint_damping"] = [k * pd_scale for k in robot_cfg["default_joint_damping"]]

    sim = A.MuJocoSimulation(config, device, enable_viewer=False, mjcf_path=mjcf_path)
    if abs(sim.dt - CONTROL_DT) > 1e-9:
        raise SystemExit(f"Unexpected control dt {sim.dt} (want {CONTROL_DT}) — wrong YAML?")
    # Fix upstream base_ang_vel frame bug (see BodyFrameGyroShim docstring).
    sim._root_angular_velocity_sensor = BodyFrameGyroShim(sim.mj_data)
    print("  Patched: base_ang_vel sourced from qvel[3:6] (body frame) via gyro shim")

    obs_processor = A.ObservationProcessor(config, sim.joint_names, device)
    provider = A.create_command_provider(config, device, motion_tracker=obs_processor.motion_tracker)
    if not isinstance(provider, A.VelocityCommandProvider) or provider.command_dim != 4:
        raise SystemExit("Expected a 4D velocity+height command provider — wrong policy YAML?")
    mgr = provider.manager
    sim.command_manager = mgr
    obs_processor.command_manager = mgr
    act_processor = A.ActionProcessor(config, sim.joint_names, device)

    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0", "goto_ab", "pipeline_abc"):
        # Deliberately widen the +/-0.5 m/s clamp: walk tests command 1.0 m/s
        # (OOD on purpose) and the v3 nav controller commands up to 0.6 m/s.
        mgr.linear_x_range = WIDENED_VX_RANGE
    if test == "squat_sweep":
        # Let the OOD 0.35/0.30 depth targets actually reach the policy.
        mgr.height_range = SQUAT_SWEEP_HEIGHT_RANGE
        print(f"  Patched: manager height_range widened to {SQUAT_SWEEP_HEIGHT_RANGE}")
    if test == "squat_limit":
        # Spec v4: the 0.72 -> 0.10 ramp must reach the policy UNCLIPPED
        # (saturation is the measured result). See NOTES_AGILE.md.
        mgr.height_range = SQUAT_LIMIT_HEIGHT_RANGE
        print(f"  Patched: manager height_range widened to {SQUAT_LIMIT_HEIGHT_RANGE} "
              "(squat_limit: 0.10 command delivered verbatim)")
    if test == "squat_pick_ground":
        # Spec v5: the unified 0.25 squat command must reach the policy
        # verbatim (AGILE saturates at ~0.50-0.53 — that IS the result).
        mgr.height_range = PICK_HEIGHT_RANGE
        print(f"  Patched: manager height_range widened to {PICK_HEIGHT_RANGE} "
              "(squat_pick_ground: 0.25 command delivered verbatim)")
    if test == "vln_follow":
        # Spec v5: tape commands pass through the manager unclipped.
        for attr, rng in VLN_CMD_RANGES.items():
            setattr(mgr, attr, rng)
        print(f"  Patched: manager clamps widened for vln_follow: {VLN_CMD_RANGES}")
    if test in ARENA_TESTS:
        # Arena: nav legs command up to 0.6 m/s + the reach/descent commands run
        # down to ARENA_H_MIN (and L5 to 0.10) — widen both clamps so the
        # mission/L-series commands reach the policy verbatim.
        mgr.linear_x_range = WIDENED_VX_RANGE
        mgr.height_range = ARENA_HEIGHT_RANGE
        if test in ("arena_L4",):
            for attr, rng in VLN_CMD_RANGES.items():
                setattr(mgr, attr, rng)
        print(f"  Patched (arena): linear_x_range={WIDENED_VX_RANGE} "
              f"height_range={ARENA_HEIGHT_RANGE}")

    arm_indices = arm_targets = box = None
    upper = replay
    upper_indices = None
    # Arena tests carry TWO free bodies (box + cube) at the qpos tail; the
    # ManipArena welds/addresses are resolved separately via configure_arena_welds
    # in run_arena_main (NOT build_box_rig, which assumes a single freejoint).
    # Here we only need (1) the get_state truncation to the 29 robot joints and
    # (2) the agile29 upper-replay indices for the M-series.
    if test in ARENA_TESTS:
        n_rj = len(sim.joint_names)
        _orig_get_state = sim.get_state

        def _robot_only_get_state(_orig=_orig_get_state, _n=n_rj):
            s = _orig()
            s.joint_pos = s.joint_pos[:_n]
            s.joint_vel = s.joint_vel[:_n]
            s.joint_effort = s.joint_effort[:_n]
            return s

        sim.get_state = _robot_only_get_state
        print(f"  Patched (arena): get_state truncated to first {n_rj} robot "
              "joints (box+cube freejoints excluded)")
        n_floor = enable_floor_bit2(sim.mj_model)
        print(f"  Patched (arena): floor bit-2 collision enabled on {n_floor} plane geom(s)")
        if test in ARENA_M_TESTS:
            if upper is None:
                raise SystemExit(f"--upper-replay is required for {test} with --upper-mode psi0 "
                                 "(or use --upper-mode box_demo)")
            missing = [n for n in upper.names if n not in sim.joint_names]
            if missing:
                raise SystemExit(f"arena upper joints missing from MJCF: {missing}")
            upper_indices = torch.tensor([sim.joint_names.index(n) for n in upper.names],
                                         dtype=torch.long, device=device)
            label = "box_demo upper" if getattr(upper, "is_boxdemo", False) else "upper-replay"
            print(f"  arena {label}: source={upper.source} K={len(upper.names)} "
                  f"N={upper.n} ({upper.duration_s:.2f}s)")
        return Rig(A, config, sim, obs_processor, act_processor, mgr, checkpoint, device,
                   test, arm_indices, arm_targets, box, upper, upper_indices,
                   BOXDEMO_UPPER_MODE if getattr(upper, "is_boxdemo", False) else None)
    if test in CARRY_TESTS:
        if upper_mode == "box_demo":
            carry_names = list(BOXDEMO_UPPER_NAMES)
            carry_targets = _load_box_demo_assets()["RL_LOWER_HANDOFF_Q"]
        else:
            carry_names = list(ARM_CARRY_POSE)
            carry_targets = np.asarray(list(ARM_CARRY_POSE.values()), dtype=np.float32)
        missing = [j for j in carry_names if j not in sim.joint_names]
        if missing:
            raise SystemExit(f"Carry-pose joints missing from MJCF: {missing}")
        arm_indices = torch.tensor([sim.joint_names.index(j) for j in carry_names],
                                   dtype=torch.long, device=device)
        arm_targets = torch.tensor(carry_targets, dtype=torch.float32, device=device)
    if test in BOX_TESTS:
        box = build_box_rig(sim.mj_model)
        # The box freejoint appends 7 qpos / 6 qvel AFTER the robot's 29 hinges
        # (guaranteed last by build_box_rig's qpos_adr assert), but upstream
        # MuJocoSimulation.get_state() slices qpos[7:]/qvel[6:]/qfrc_actuator[6:]
        # for the whole model -> 36/35/35-dim tensors that break the 29-joint
        # ObservationProcessor. Truncate to the robot's joints only.
        n_rj = len(sim.joint_names)
        _orig_get_state = sim.get_state

        def _robot_only_get_state(_orig=_orig_get_state, _n=n_rj):
            s = _orig()
            s.joint_pos = s.joint_pos[:_n]
            s.joint_vel = s.joint_vel[:_n]
            s.joint_effort = s.joint_effort[:_n]
            return s

        sim.get_state = _robot_only_get_state
        print(f"  Patched: get_state truncated to first {n_rj} robot joints (box freejoint excluded)")

    if test in REPLAY_TESTS:
        if upper is None:
            raise SystemExit(f"--upper-replay is required for {test} (contract v1 npz)")
        missing = [n for n in upper.names if n not in sim.joint_names]
        if missing:
            raise SystemExit(f"upper-replay joints missing from MJCF: {missing}")
        upper_indices = torch.tensor([sim.joint_names.index(n) for n in upper.names],
                                     dtype=torch.long)
        upper.pos_t = torch.tensor(upper.pos, dtype=torch.float32, device=device)
        upper.carry_t = torch.tensor(upper.carry_pose, dtype=torch.float32, device=device)
        # Unified psi0 spec: box collides with table/floor (bit 2), never with
        # the robot (bit 1). The floor planes ship with contype/conaffinity 1;
        # OR in bit 2 at runtime (model fields are writable). The body-level
        # world aggregates follow inside enable_floor_bit2 (MuJoCo >= 3.2.4
        # broadphase culling).
        n_floor = enable_floor_bit2(sim.mj_model)
        print(f"  psi0: floor bit-2 collision enabled on {n_floor} plane geom(s); "
              f"replay source={upper.source} K={len(upper.names)} N={upper.n} "
              f"({upper.duration_s:.2f}s, grasp_close_t={upper.grasp_close_t:.2f}s)")

    if test == "squat_pick_ground":
        # The pick box compiles with bit-2 collisions (PICK_BOX_XML); the
        # floor planes need bit 2 OR-ed in so the box rests on the ground
        # from t=0 (same scheme as the psi0 cracker box).
        n_floor = enable_floor_bit2(sim.mj_model)
        print(f"  squat_pick_ground: floor bit-2 collision enabled on {n_floor} plane geom(s)")

    effective_boxdemo = (
        upper_mode == "box_demo" and (test in CARRY_TESTS or test in REPLAY_TESTS)
    )
    return Rig(A, config, sim, obs_processor, act_processor, mgr, checkpoint, device,
               test, arm_indices, arm_targets, box, upper, upper_indices,
               BOXDEMO_UPPER_MODE if effective_boxdemo else None)


def fresh_policy(rig: Rig) -> tuple[object, int]:
    """Load the TorchScript policy fresh and zero recurrent state buffers.

    The exported student LSTM keeps hidden/cell state in internal buffers and
    the wrapper's reset() is a no-op (yaml has no top-level `policy:` section
    -> MLPPolicyWrapper). Reloading + zeroing guarantees no cross-trial leak
    and clears any state baked in at export time.
    """
    policy = rig.A.PolicyWrapper.from_config(rig.checkpoint, rig.config, rig.device)
    n_zeroed = 0
    try:
        for name, buf in policy.model.named_buffers():
            if "hidden" in name or "cell" in name:
                buf.zero_()
                n_zeroed += 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARN: could not enumerate policy buffers: {exc}")
    # TODO(verify-on-server): expect n_zeroed == 2 (hidden_state, cell_state) on
    # first trial log; if 0, inspect the TorchScript export for buffer names.
    return policy, n_zeroed


def reset_trial(rig: Rig, seed: int, yaw0: float) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rig.sim.reset()
    model = rig.sim.mj_model
    data = rig.sim.mj_data
    data.qpos[3:7] = yaw_to_quat(yaw0)
    # Joint noise only on the robot hinge block (excludes the box freejoint,
    # which sits at the end of qpos in squat_box v2).
    robot_qpos_end = rig.box.qpos_adr if rig.box is not None else len(data.qpos)
    noise = rng.uniform(-JOINT_NOISE_RAD, JOINT_NOISE_RAD, size=robot_qpos_end - 7)
    data.qpos[7:robot_qpos_end] = data.qpos[7:robot_qpos_end] + noise
    data.qvel[:] = 0.0

    if rig.test in ("pipeline_abc", "squat_place_psi0") and rig.box is not None:
        # The previous trial may have released the box (collision bits are
        # flipped on the MODEL, which persists across trials) — restore the
        # collision-free carry defaults (geom AND body bits — broadphase
        # filters on the body level). eq_active is restored by mj_resetData.
        model.geom_contype[rig.box.geom_id] = 0
        model.geom_conaffinity[rig.box.geom_id] = 0
        if hasattr(model, "body_contype"):
            model.body_contype[rig.box.body_id] = 0
            model.body_conaffinity[rig.box.body_id] = 0

    if rig.test == "squat_pick_ground":
        # v5: the 2 kg box stands on the floor PICK_BOX_AHEAD_M in front of
        # the robot (along its initial heading), yaw-aligned; the welds stay
        # OFF until the magnetic grasp at the squat bottom.
        adr = rig.box.qpos_adr
        data.qpos[adr] = PICK_BOX_AHEAD_M * math.cos(yaw0)
        data.qpos[adr + 1] = PICK_BOX_AHEAD_M * math.sin(yaw0)
        data.qpos[adr + 2] = PICK_BOX_HALF[2] + 0.0005
        data.qpos[adr + 3:adr + 7] = yaw_to_quat(yaw0)
        data.qvel[rig.box.dof_adr:rig.box.dof_adr + 6] = 0.0
        for eq_id in rig.box.eq_ids:
            data.eq_active[eq_id] = 0

    if rig.arm_indices is not None:
        # Carry tests: start already in the carry pose (exact, overrides noise
        # on the 14 arm joints) so the welds anchor a clean "hugging" pose.
        for joint_idx, target in zip(rig.arm_indices.detach().cpu().tolist(),
                                     rig.arm_targets.detach().cpu().tolist()):
            joint = rig.sim.joint_names[int(joint_idx)]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            data.qpos[model.jnt_qposadr[jid]] = target
        mujoco.mj_forward(model, data)
        for eq_id in rig.box.eq_ids:
            data.eq_active[eq_id] = 1
        place_box_in_hands(model, data, rig.box)

    if rig.upper is not None:
        # psi0: the 17 replayed joints start exactly at replay frame 0 (no
        # noise) so the per-frame PD targets do not snap at t=0.
        for name, val in zip(rig.upper.names, rig.upper.pos[0]):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = float(val)
        mujoco.mj_forward(model, data)
        if rig.test == "squat_box_psi0":
            # Box upright on the table; welds OFF until grasp_close_t.
            adr = rig.box.qpos_adr
            data.qpos[adr:adr + 2] = PSI0_BOX_START_XY
            data.qpos[adr + 2] = PSI0_BOX_START_Z
            quat = np.asarray(PSI0_BOX_START_QUAT, dtype=np.float64)
            data.qpos[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
            data.qvel[rig.box.dof_adr:rig.box.dof_adr + 6] = 0.0
            for eq_id in rig.box.eq_ids:
                data.eq_active[eq_id] = 0
        else:
            # walk/circle psi0: cracker box wrist-welded from t=0;
            # squat_place_psi0: v2 2 kg box wrist-welded from t=0 (arms at
            # replay frame 0 — the welds anchor to that pose, not carry pose).
            for eq_id in rig.box.eq_ids:
                data.eq_active[eq_id] = 1
            place_box_in_hands(model, data, rig.box)

    mujoco.mj_forward(model, data)
    rig.obs_processor.reset()
    # Explicit: provider default height is 0.74 (!= 0.72 stand height).
    rig.mgr.set_command(0.0, 0.0, 0.0, STAND_HEIGHT)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class TrialVideo:
    """Streams frames to a temp mp4; keeps or deletes after the trial verdict."""

    def __init__(self, renderer, videos_dir: Path, test: str, trial: int,
                 label: str = "agile"):
        import imageio  # lazy: only needed when recording

        self.renderer = renderer
        self.videos_dir = videos_dir
        self.test = test
        self.trial = trial
        self.label = label
        self.tmp_path = videos_dir / f".tmp_{label}_{test}_t{trial:02d}.mp4"
        self.writer = imageio.get_writer(
            str(self.tmp_path), fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=None
        )
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 3.0
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0

    def capture(self, data: mujoco.MjData) -> None:
        self.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]
        self.renderer.update_scene(data, camera=self.cam)
        self.writer.append_data(self.renderer.render())

    def finish(self, keep: bool, success: bool) -> str | None:
        self.writer.close()
        if not keep:
            self.tmp_path.unlink(missing_ok=True)
            return None
        final = self.videos_dir / f"{self.label}_{self.test}_t{self.trial:02d}_{'ok' if success else 'fail'}.mp4"
        self.tmp_path.replace(final)
        return str(final)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trial(rig: Rig, geoms: GeomSets, test: str, trial: int, seed: int,
              vx_target: float, ccw: bool, video: TrialVideo | None,
              target_h: float | None = None, ramp_speed: float | None = None,
              sweep_kind: str | None = None) -> dict:
    yaw0 = 0.0
    if test in ("circle_pillar", "circle_pillar_psi0"):
        yaw0 = 0.0 if ccw else math.pi  # tangent heading; same circle center (0, 1)
    elif test == "squat_box_psi0":
        yaw0 = 0.0  # BendPick table scene alignment (scene_info: identity heading)
    else:
        yaw0 = float(np.random.default_rng(seed).uniform(-math.pi, math.pi))

    reset_trial(rig, seed, yaw0)
    policy, n_zeroed = fresh_policy(rig)
    if trial == 0:
        print(f"  [policy] zeroed {n_zeroed} recurrent state buffer(s)")

    sim = rig.sim
    dt = sim.dt
    cmd_s = test_duration_s(test, target_h=target_h, ramp_speed=ramp_speed, replay=rig.upper)
    n_steps = int(round((SETTLE_S + cmd_s) / dt))

    # Sample buffers (per control step, measured AFTER stepping physics)
    ts, fwd_vels, vx_cmds, base_zs, h_cmds, tilts, xs, ys, yaws = ([] for _ in range(9))
    fell = False
    fall_time = None
    fall_phase = None
    collided = False
    collision_time = None
    closures: dict[int, dict] = {}
    start_pose = None  # (x, y, yaw) at first command-phase sample
    box_drop_time = None  # squat_box v2: first time |box - torso| >= BOX_KEEP_DIST_M
    harness_error = None  # numerical divergence guard (qacc explosion / NaN)
    # squat_box_psi0 grasp state
    pick_checked = False
    pick_success = None
    grasp_dists = None
    box_z0 = box_z_max = None
    # squat_sweep buffers
    hold_zs: list[float] = []
    hold_xy: list[tuple[float, float]] = []
    last_xy = None
    # v4 (squat_limit / squat_place_psi0) buffers
    phases: list[str] = []  # command phase per recorded sample
    drift_xys: list[float] = []  # root XY drift from cmd-phase start, per sample
    descent_curve: list[list[float]] = []  # squat_limit: 10 Hz [h_cmd, base_z, drift_xy]
    box_released = False  # squat_place_psi0: welds released at t_place
    release_ref = None  # (foot_front_proj, heading) captured at release

    wall0 = time.time()
    for i in range(n_steps):
        t_cmd = i * dt - SETTLE_S
        vx_c, vy_c, wz_c, h_c, phase = command_profile(
            test, t_cmd, vx_target, ccw,
            target_h=target_h, ramp_speed=ramp_speed, replay=rig.upper)
        rig.mgr.set_command(vx_c, vy_c, wz_c, h_c)

        # --- official sim2mujoco_eval loop order ---
        sim_state = sim.get_state()
        obs = rig.obs_processor.compute(sim_state)
        with torch.no_grad():
            actions = policy(obs)
        rig.obs_processor.set_last_action(actions)
        joint_cmd = rig.act_processor.process(actions)

        if rig.upper is not None:
            # psi0: upper body (arms14 + waist3) PD targets replayed
            # frame-by-frame through the same JointCommand override path.
            rep = rig.upper
            if test == "squat_box_psi0":
                # replay starts at settle end; afterwards arms freeze at carry_pose
                k = int(math.floor(t_cmd / CONTROL_DT + 1e-9))
                if k < 0:
                    row = rep.pos_t[0]
                elif k >= rep.n:
                    row = rep.carry_t
                else:
                    row = rep.pos_t[k]
            elif test == "squat_place_psi0":
                # SINGLE playback from settle end (spec v4: no loop); frame 0
                # during settle, last frame held through the 1 s stand segment.
                k = int(math.floor(t_cmd / CONTROL_DT + 1e-9))
                row = rep.pos_t[min(max(k, 0), rep.n - 1)]
            else:
                row = rep.pos_t[i % rep.n]  # real_ep053 looped from t=0
            position = joint_cmd.position.clone()
            position[rig.upper_indices] = row
            joint_cmd = rig.A.JointCommand(position=position, kp=joint_cmd.kp, kd=joint_cmd.kd)
        elif rig.arm_indices is not None:
            # Carry pose: constant PD targets for the arm joints (policy only
            # controls legs; qpos starts AT the carry pose so no blend needed).
            position = joint_cmd.position.clone()
            position[rig.arm_indices] = rig.arm_targets
            joint_cmd = rig.A.JointCommand(position=position, kp=joint_cmd.kp, kd=joint_cmd.kd)

        for _ in range(sim.decimation):
            sim.step(joint_cmd)
        # --------------------------------------------

        data = sim.mj_data
        # Divergence guard (spec: record harness_error if the welds blow up).
        if (
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
            or not np.isfinite(data.qpos).all()
        ):
            harness_error = f"divergence (qacc explosion / NaN qpos) at t={(i + 1) * dt:.2f}s"
            break

        t_post = (i + 1) * dt - SETTLE_S
        if test == "squat_box_psi0" and not pick_checked and t_post >= rig.upper.grasp_close_t:
            # Grasp event: both wrists must be near the box surface to weld.
            pick_checked = True
            d_l = box_surface_dist_m(data, rig.box, rig.box.wrist_body_ids[0], PSI0_BOX_HALF)
            d_r = box_surface_dist_m(data, rig.box, rig.box.wrist_body_ids[1], PSI0_BOX_HALF)
            grasp_dists = (d_l, d_r)
            pick_success = d_l < PSI0_GRASP_DIST_M and d_r < PSI0_GRASP_DIST_M
            if pick_success:
                activate_welds_at_current_pose(rig.sim.mj_model, data, rig.box)

        if test == "squat_place_psi0" and not box_released and t_post >= rig.upper.t_place_s:
            # t_place = argmin(replay height_cmd): release both welds and let
            # the box land (shared bit-2 release mechanism, see release_box).
            # The foot front edge + heading at the release moment anchor the
            # box_land_dx measurement (positive = placed in front of the feet).
            yaw_rel = quat_yaw(np.asarray(data.qpos[3:7], dtype=np.float64))
            heading = (math.cos(yaw_rel), math.sin(yaw_rel))
            release_ref = (foot_front_proj_m(rig.sim.mj_model, data, geoms, heading), heading)
            release_box(rig.sim.mj_model, data, rig.box)
            box_released = True

        # box_kept tracking: for squat_box_psi0 only once the box is welded
        # (before the grasp it legitimately sits on the table, far from torso);
        # for squat_place_psi0 only until the intentional release.
        track_box = (rig.box is not None and not box_released
                     and (test != "squat_box_psi0" or bool(pick_success)))
        if track_box and box_drop_time is None:
            box_dist = float(
                np.linalg.norm(data.xpos[rig.box.body_id] - data.xpos[rig.box.torso_body_id])
            )
            if box_dist >= BOX_KEEP_DIST_M:
                box_drop_time = (i + 1) * dt  # trial clock (settle included), like fall_time

        if test == "squat_box_psi0":
            bz = float(data.xpos[rig.box.body_id][2])
            box_z0 = bz if box_z0 is None else box_z0
            box_z_max = bz if box_z_max is None else max(box_z_max, bz)

        t_meas = (i + 1) * dt - SETTLE_S
        quat = np.array(data.qpos[3:7], dtype=np.float64)
        yaw = quat_yaw(quat)
        tilt = quat_tilt(quat)
        base_z = float(data.qpos[2])
        vw = np.array(data.qvel[:3], dtype=np.float64)
        fwd_vel = float(vw[0] * math.cos(yaw) + vw[1] * math.sin(yaw))

        if t_meas > 0.0:
            ts.append(t_meas)
            fwd_vels.append(fwd_vel)
            vx_cmds.append(vx_c)
            base_zs.append(base_z)
            h_cmds.append(h_c)
            tilts.append(tilt)
            xs.append(float(data.qpos[0]))
            ys.append(float(data.qpos[1]))
            yaws.append(yaw)
            if start_pose is None:
                start_pose = (float(data.qpos[0]), float(data.qpos[1]), yaw)
            last_xy = (float(data.qpos[0]), float(data.qpos[1]))
            if test in ("squat_limit", "squat_place_psi0"):
                drift = math.hypot(last_xy[0] - start_pose[0], last_xy[1] - start_pose[1])
                drift_xys.append(drift)
                phases.append(phase)
                if test == "squat_limit" and (len(ts) - 1) % DESCENT_CURVE_EVERY_N_STEPS == 0:
                    # 10 Hz raw height-tracking-drift curve (spec v4)
                    descent_curve.append([round(h_c, 4), round(base_z, 4), round(drift, 4)])
            if test == "squat_sweep" and phase == "hold":
                hold_zs.append(base_z)
                hold_xy.append(last_xy)
            if test in ("circle_pillar", "circle_pillar_psi0"):
                for k in (1, 2):
                    if k not in closures and t_meas >= k * CIRCLE_PERIOD_S:
                        closures[k] = {
                            "pos_err": math.hypot(data.qpos[0] - start_pose[0], data.qpos[1] - start_pose[1]),
                            "yaw_err": abs(wrap_angle(yaw - start_pose[2])),
                        }

        nonfoot_ground, pillar_hit = scan_contacts(data, geoms)
        if pillar_hit and not collided:
            collided = True
            collision_time = (i + 1) * dt  # trial clock (settle included)

        is_fall = tilt > FALL_TILT_RAD or base_z < (h_c - FALL_HEIGHT_MARGIN) or nonfoot_ground

        if video is not None and (is_fall or i % VIDEO_EVERY_N_CTRL_STEPS == 0):
            video.capture(data)

        if is_fall:
            fell = True
            fall_time = (i + 1) * dt
            fall_phase = phase
            break

    wall_s = time.time() - wall0

    # ---------------- metrics ----------------
    t_arr = np.asarray(ts)
    metrics: dict = {"max_tilt": float(np.max(tilts)) if tilts else None}

    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        win = (t_arr >= cmd_s - WALK_MEAN_WINDOW_S) & (t_arr <= cmd_s)
        fwd = np.asarray(fwd_vels)
        mean_vx = float(np.mean(fwd[win])) if win.any() else None
        rmse = float(np.sqrt(np.mean((fwd - np.asarray(vx_cmds)) ** 2))) if len(t_arr) else None
        metrics.update({"vx_target": vx_target, "mean_vx_last8s": mean_vx, "vx_rmse": rmse})
        threshold = (0.9 * vx_target
                     if (test == "speed_sweep" or CUSTOM_VX_REL_THR) else 0.9)
        success = (not fell) and mean_vx is not None and mean_vx >= threshold
        if test == "walk_speed_psi0":
            box_kept = box_drop_time is None
            metrics.update({"box_kept": box_kept, "box_drop_time": box_drop_time})
            success = success and box_kept
    elif test == "squat_box":
        if fell:
            cycles = max(0, int((fall_time - SETTLE_S) // SQUAT_CYCLE_S)) if fall_time > SETTLE_S else 0
        elif harness_error is not None:  # diverged mid-run: count what was observed
            cycles = max(0, int(t_arr[-1] // SQUAT_CYCLE_S)) if len(t_arr) else 0
        else:
            cycles = SQUAT_N_CYCLES
        h_rmse = (
            float(np.sqrt(np.mean((np.asarray(base_zs) - np.asarray(h_cmds)) ** 2))) if len(t_arr) else None
        )
        box_kept = box_drop_time is None
        metrics.update(
            {
                "cycles_completed": cycles,
                "height_rmse": h_rmse,
                "box_kept": box_kept,
                "box_drop_time": box_drop_time,
            }
        )
        success = (not fell) and cycles >= SQUAT_N_CYCLES and box_kept
    elif test == "squat_box_psi0":
        rep = rig.upper
        cycles_t0 = SETTLE_S + rep.duration_s  # trial clock when the squat cycles start
        if fell:
            cycles = int(max(0.0, fall_time - cycles_t0) // SQUAT_CYCLE_S)
        elif harness_error is not None:  # diverged mid-run: count what was observed
            done_t = (float(t_arr[-1]) + SETTLE_S) if len(t_arr) else 0.0
            cycles = int(max(0.0, done_t - cycles_t0) // SQUAT_CYCLE_S)
        else:
            cycles = SQUAT_N_CYCLES
        cycles = min(cycles, SQUAT_N_CYCLES)
        h_rmse = (
            float(np.sqrt(np.mean((np.asarray(base_zs) - np.asarray(h_cmds)) ** 2))) if len(t_arr) else None
        )
        box_kept = bool(pick_success) and box_drop_time is None
        metrics.update(
            {
                "cycles_completed": cycles,
                "height_rmse": h_rmse,
                "pick_success": bool(pick_success),
                "grasp_dist_left": grasp_dists[0] if grasp_dists else None,
                "grasp_dist_right": grasp_dists[1] if grasp_dists else None,
                "box_lift_height": (box_z_max - box_z0) if box_z0 is not None else None,
                "box_kept": box_kept,
                "box_drop_time": box_drop_time,
            }
        )
        success = (not fell) and bool(pick_success) and cycles >= SQUAT_N_CYCLES and box_kept
    elif test == "squat_sweep":
        achieved = float(np.mean(hold_zs)) if hold_zs else None
        drift_hold = None
        if hold_xy:
            ax, ay = hold_xy[0]  # squat-bottom anchor
            drift_hold = float(max(math.hypot(px - ax, py - ay) for px, py in hold_xy))
        drift_total = None
        if (not fell) and harness_error is None and start_pose is not None and last_xy is not None:
            drift_total = float(math.hypot(last_xy[0] - start_pose[0], last_xy[1] - start_pose[1]))
        box_kept = box_drop_time is None
        metrics.update(
            {
                "sweep_kind": sweep_kind,
                "target_height": target_h,
                "ramp_speed": ramp_speed,
                "achieved_depth": achieved,
                "depth_err": (achieved - target_h) if achieved is not None else None,
                "root_drift_hold": drift_hold,
                "root_drift_total": drift_total,
                "box_kept": box_kept,
                "box_drop_time": box_drop_time,
            }
        )
        success = (not fell) and box_kept
    elif test == "squat_limit":
        # depth_floor = min base_z over STABLE samples (the fall step itself is
        # excluded; samples just before fall detection are kept — approximation).
        stable_zs = base_zs[:-1] if fell else base_zs
        track_sat_h = None  # first |cmd - actual| > 5 cm => tracking saturation
        for hc, bz in zip(h_cmds, base_zs):
            if abs(hc - bz) > SQUAT_LIMIT_TRACK_SAT_M:
                track_sat_h = float(hc)
                break
        drift5_h = drift20_h = None  # cmd height at first 5 cm / 20 cm XY drift
        for hc, d in zip(h_cmds, drift_xys):
            if drift5_h is None and d > SQUAT_LIMIT_DRIFT_MARKS_M[0]:
                drift5_h = float(hc)
            if d > SQUAT_LIMIT_DRIFT_MARKS_M[1]:
                drift20_h = float(hc)
                break
        metrics.update(
            {
                "descent_curve": descent_curve,  # raw 10 Hz [h_cmd, base_z, drift_xy]
                "fall_h_cmd": float(h_cmds[-1]) if fell and h_cmds else None,
                "fall_base_z": float(base_zs[-1]) if fell and base_zs else None,
                "depth_floor": float(min(stable_zs)) if stable_zs else None,
                "track_sat_h": track_sat_h,
                "drift5_h": drift5_h,
                "drift20_h": drift20_h,
                "max_drift_xy": float(max(drift_xys)) if drift_xys else None,
                "box_kept": box_drop_time is None,
                "box_drop_time": box_drop_time,
            }
        )
        # success := no fall (reached the 0.10 command and held 3 s). Most
        # models are expected to saturate WITHOUT falling; the fall boundary
        # is exactly what this test calibrates.
        success = not fell
    elif test == "squat_place_psi0":

        def _seg_rmse(seg: str) -> float | None:
            errs = [bz - hc for bz, hc, ph in zip(base_zs, h_cmds, phases) if ph == seg]
            return float(np.sqrt(np.mean(np.square(errs)))) if errs else None

        # root drift through the reach-forward + place window (descent+hold):
        # the forward COM shift is the upper/lower decoupling stress test.
        place_drifts = [d for d, ph in zip(drift_xys, phases) if ph in ("descent", "hold")]
        completed = (not fell) and harness_error is None
        box_place_ok = False
        box_land_dx = None
        if box_released and completed:
            d_end = sim.mj_data
            v = float(np.linalg.norm(d_end.qvel[rig.box.dof_adr:rig.box.dof_adr + 3]))
            upright_cos = float(d_end.xmat[rig.box.body_id].reshape(3, 3)[2, 2])
            bx = float(d_end.xpos[rig.box.body_id][0])
            by = float(d_end.xpos[rig.box.body_id][1])
            near = math.hypot(bx - float(d_end.qpos[0]),
                              by - float(d_end.qpos[1])) < BOX_PLACE_NEAR_M
            box_place_ok = bool(
                v < BOX_PLACE_VEL_MAX
                and upright_cos > math.cos(BOX_PLACE_UPRIGHT_RAD)
                and near
            )
            # Forward distance of the landed box vs the foot front edge at the
            # release moment (positive = placed in front of the robot).
            front, heading = release_ref
            box_land_dx = float(bx * heading[0] + by * heading[1] - front)
        metrics.update(
            {
                "height_rmse_descent": _seg_rmse("descent"),
                "height_rmse_hold": _seg_rmse("hold"),
                "height_rmse_rise": _seg_rmse("rise"),
                "root_drift_place": float(max(place_drifts)) if place_drifts else None,
                "box_released": box_released,
                "box_place_ok": box_place_ok,
                "box_land_dx": box_land_dx,
                "box_kept_carry": box_drop_time is None,
                "box_drop_time": box_drop_time,
                "stand_ok": completed,  # rose with the replay + stood the final 1 s
            }
        )
        success = completed and box_place_ok
    elif test in ("circle_pillar", "circle_pillar_psi0"):
        cx, cy = CIRCLE_CENTER
        radial = np.abs(np.hypot(np.asarray(xs) - cx, np.asarray(ys) - cy) - CIRCLE_RADIUS) if len(t_arr) else None
        metrics.update(
            {
                "direction": "ccw" if ccw else "cw",
                "circle_vx": CIRCLE_VX,
                "circle_wz": CIRCLE_WZ,
                "radial_err_mean": float(np.mean(radial)) if radial is not None else None,
                "radial_err_max": float(np.max(radial)) if radial is not None else None,
                "closure_pos_err_lap1": closures.get(1, {}).get("pos_err"),
                "closure_yaw_err_lap1": closures.get(1, {}).get("yaw_err"),
                "closure_pos_err_lap2": closures.get(2, {}).get("pos_err"),
                "closure_yaw_err_lap2": closures.get(2, {}).get("yaw_err"),
                "pillar_collision": collided,
                "collision_time": collision_time,
            }
        )
        success = (
            (not fell)
            and (not collided)
            and metrics["radial_err_mean"] is not None
            and metrics["radial_err_mean"] <= CIRCLE_RADIAL_TOL
        )
        if test == "circle_pillar_psi0":
            box_kept = box_drop_time is None
            metrics.update({"box_kept": box_kept, "box_drop_time": box_drop_time})
            success = success and box_kept
    else:
        raise ValueError(test)

    metrics["wall_time_s"] = round(wall_s, 2)
    success = bool(success) and harness_error is None

    result = {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial,
        "seed": seed,
        "success": success,
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "harness_error": harness_error,
        "metrics": metrics,
        "video": None,  # filled by caller
    }
    if test in BOX_TESTS:
        result["box_mode"] = BOX_MODE
    if test in REPLAY_TESTS and rig.upper is not None:
        result["upper_mode"] = (
            BOXDEMO_UPPER_MODE if getattr(rig.upper, "is_boxdemo", False)
            else f"psi0_replay_{rig.upper.source}"
        )
    elif rig.upper_mode:
        result["upper_mode"] = rig.upper_mode
    return result


# ---------------------------------------------------------------------------
# Navigation trial runner (T5 goto_ab / T6 pipeline_abc, spec v3)
# ---------------------------------------------------------------------------


def run_nav_trial(rig: Rig, geoms: GeomSets, test: str, trial: int, seed: int,
                  video: TrialVideo | None) -> dict:
    """Two-stage navigation state machine: NAV (coarse) -> CAL (small-command
    calibration); pipeline_abc adds carry-box transport + squat-place-rise."""
    rng = np.random.default_rng(seed)
    if test == "goto_ab":
        goal_xy, goal_yaw = GOTO_AB_GOAL_XY, GOTO_AB_GOAL_YAW
        vx_max = NAV_VX_RANGE[1]
        yaw0 = float(rng.uniform(-GOTO_AB_YAW0_NOISE, GOTO_AB_YAW0_NOISE))
    else:
        goal_xy, goal_yaw = PIPE_GOAL_XY, PIPE_GOAL_YAW
        vx_max = PIPE_NAV_VX_MAX
        yaw0 = 0.0  # spec: A = origin, heading 0

    reset_trial(rig, seed, yaw0)
    policy, n_zeroed = fresh_policy(rig)
    if trial == 0:
        print(f"  [policy] zeroed {n_zeroed} recurrent state buffer(s)")

    sim = rig.sim
    dt = sim.dt
    place_down_s = (STAND_HEIGHT - PIPE_PLACE_HEIGHT) / PIPE_PLACE_RATE
    place_total_s = (place_down_s + PIPE_HOLD_BEFORE_RELEASE_S + PIPE_HOLD_AFTER_RELEASE_S
                     + place_down_s + PIPE_STAND_S)
    max_s = (SETTLE_S + NAV_TIMEOUT_S + CAL_TIMEOUT_S
             + (place_total_s if test == "pipeline_abc" else 0.0) + 1.0)
    n_steps = int(round(max_s / dt))

    stage = "settle"
    stage_t0 = 0.0
    cal_in_tol_t0 = None
    tilts: list[float] = []
    fell = False
    fall_time = fall_phase = None
    harness_error = None
    box_drop_time = None
    box_released = False
    t_end = 0.0
    m: dict = {
        "err_pos_nav": None, "err_yaw_nav": None, "t_nav": None,
        "err_pos_cal": None, "err_yaw_cal": None, "t_cal": None,
        "success_coarse": False, "success_fine": False,
    }

    wall0 = time.time()
    for i in range(n_steps):
        t = i * dt
        data = sim.mj_data
        x, y = float(data.qpos[0]), float(data.qpos[1])
        yaw = quat_yaw(np.asarray(data.qpos[3:7], dtype=np.float64))
        dist, ex, ey, eyaw = nav_errors(x, y, yaw, goal_xy, goal_yaw)
        yaw_err = abs(wrap_angle(goal_yaw - yaw))

        # ---- stage transitions (evaluated on the pre-step state) ----
        if stage == "settle" and t >= SETTLE_S:
            stage, stage_t0 = "nav", t
        if stage == "nav":
            t_in = t - stage_t0
            if (dist <= NAV_EXIT_POS_M and abs(eyaw) <= NAV_EXIT_YAW_RAD) or t_in >= NAV_TIMEOUT_S:
                m["err_pos_nav"], m["err_yaw_nav"] = dist, yaw_err
                m["t_nav"] = round(t_in, 2)
                m["success_coarse"] = bool(dist <= NAV_EXIT_POS_M and yaw_err <= NAV_EXIT_YAW_RAD)
                stage, stage_t0, cal_in_tol_t0 = "cal", t, None
        if stage == "cal":
            t_in = t - stage_t0
            in_tol = dist <= CAL_POS_TOL_M and abs(eyaw) <= CAL_YAW_TOL_RAD
            if in_tol and cal_in_tol_t0 is None:
                cal_in_tol_t0 = t
            elif not in_tol:
                cal_in_tol_t0 = None
            converged = cal_in_tol_t0 is not None and (t - cal_in_tol_t0) >= CAL_HOLD_S
            if converged or t_in >= CAL_TIMEOUT_S:
                m["err_pos_cal"], m["err_yaw_cal"] = dist, yaw_err
                m["t_cal"] = round(t_in, 2)
                m["success_fine"] = bool(converged)
                if test == "goto_ab":
                    t_end = t
                    break
                stage, stage_t0 = "place_down", t
        if stage == "place_down" and (t - stage_t0) >= place_down_s:
            stage, stage_t0 = "place_hold", t
        if stage == "place_hold" and (t - stage_t0) >= PIPE_HOLD_BEFORE_RELEASE_S:
            if not box_released:
                # Release: disable both welds + bit-2 collisions so the box
                # falls and lands (shared mechanism, see release_box).
                release_box(sim.mj_model, data, rig.box)
                box_released = True
            stage, stage_t0 = "place_wait", t
        if stage == "place_wait" and (t - stage_t0) >= PIPE_HOLD_AFTER_RELEASE_S:
            stage, stage_t0 = "place_up", t
        if stage == "place_up" and (t - stage_t0) >= place_down_s:
            stage, stage_t0 = "place_stand", t
        if stage == "place_stand" and (t - stage_t0) >= PIPE_STAND_S:
            t_end = t
            break

        # ---- command for this stage ----
        if stage == "nav":
            vx_c, vy_c, wz_c = nav_cmd_coarse(ex, ey, eyaw, vx_max)
            h_c = STAND_HEIGHT
        elif stage == "cal":
            vx_c, vy_c, wz_c = nav_cmd_cal(ex, ey, eyaw)
            h_c = STAND_HEIGHT
        elif stage == "place_down":
            vx_c = vy_c = wz_c = 0.0
            h_c = max(STAND_HEIGHT - PIPE_PLACE_RATE * (t - stage_t0), PIPE_PLACE_HEIGHT)
        elif stage in ("place_hold", "place_wait"):
            vx_c = vy_c = wz_c = 0.0
            h_c = PIPE_PLACE_HEIGHT
        elif stage == "place_up":
            vx_c = vy_c = wz_c = 0.0
            h_c = min(PIPE_PLACE_HEIGHT + PIPE_PLACE_RATE * (t - stage_t0), STAND_HEIGHT)
        else:  # settle / place_stand
            vx_c = vy_c = wz_c = 0.0
            h_c = STAND_HEIGHT
        rig.mgr.set_command(vx_c, vy_c, wz_c, h_c)

        # --- official sim2mujoco_eval loop order ---
        sim_state = sim.get_state()
        obs = rig.obs_processor.compute(sim_state)
        with torch.no_grad():
            actions = policy(obs)
        rig.obs_processor.set_last_action(actions)
        joint_cmd = rig.act_processor.process(actions)
        if rig.arm_indices is not None:  # pipeline_abc: carry pose all trial
            position = joint_cmd.position.clone()
            position[rig.arm_indices] = rig.arm_targets
            joint_cmd = rig.A.JointCommand(position=position, kp=joint_cmd.kp, kd=joint_cmd.kd)
        for _ in range(sim.decimation):
            sim.step(joint_cmd)
        t_end = (i + 1) * dt

        if (
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
            or not np.isfinite(data.qpos).all()
        ):
            harness_error = f"divergence (qacc explosion / NaN qpos) at t={t_end:.2f}s"
            break

        if rig.box is not None and not box_released and box_drop_time is None:
            box_dist = float(
                np.linalg.norm(data.xpos[rig.box.body_id] - data.xpos[rig.box.torso_body_id])
            )
            if box_dist >= BOX_KEEP_DIST_M:
                box_drop_time = t_end

        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        tilt = quat_tilt(quat)
        tilts.append(tilt)
        base_z = float(data.qpos[2])
        nonfoot_ground, _ = scan_contacts(data, geoms)
        is_fall = tilt > FALL_TILT_RAD or base_z < (h_c - FALL_HEIGHT_MARGIN) or nonfoot_ground

        if video is not None and (is_fall or i % VIDEO_EVERY_N_CTRL_STEPS == 0):
            video.capture(data)

        if is_fall:
            fell = True
            fall_time = t_end
            fall_phase = stage
            break

    wall_s = time.time() - wall0

    metrics = dict(m)
    metrics["max_tilt"] = float(np.max(tilts)) if tilts else None
    if metrics["err_pos_nav"] is not None and metrics["err_pos_cal"] is not None:
        metrics["err_pos_improve"] = metrics["err_pos_nav"] - metrics["err_pos_cal"]
        metrics["err_yaw_improve"] = metrics["err_yaw_nav"] - metrics["err_yaw_cal"]
    else:
        metrics["err_pos_improve"] = metrics["err_yaw_improve"] = None
    metrics["total_time_s"] = round(t_end, 2)
    metrics["wall_time_s"] = round(wall_s, 2)

    if test == "goto_ab":
        # Headline: can small commands converge? (coarse rate reported separately)
        success = bool(metrics["success_fine"]) and not fell
    else:
        data = sim.mj_data
        box_place_ok = False
        box_land_pos = None
        metrics["box_land_err"] = None
        if box_released and not fell and harness_error is None:
            v = float(np.linalg.norm(data.qvel[rig.box.dof_adr:rig.box.dof_adr + 3]))
            upright_cos = float(data.xmat[rig.box.body_id].reshape(3, 3)[2, 2])
            bx = float(data.xpos[rig.box.body_id][0])
            by = float(data.xpos[rig.box.body_id][1])
            near = math.hypot(bx - float(data.qpos[0]), by - float(data.qpos[1])) < BOX_PLACE_NEAR_M
            box_place_ok = bool(
                v < BOX_PLACE_VEL_MAX
                and upright_cos > math.cos(BOX_PLACE_UPRIGHT_RAD)
                and near
            )
            box_land_pos = [round(bx - PIPE_GOAL_XY[0], 4), round(by - PIPE_GOAL_XY[1], 4)]
            metrics["box_land_err"] = float(math.hypot(box_land_pos[0], box_land_pos[1]))
        metrics["squat_fall"] = fall_phase if (fell and str(fall_phase).startswith("place")) else None
        metrics["box_place_ok"] = box_place_ok
        metrics["box_land_pos"] = box_land_pos
        metrics["box_kept_carry"] = box_drop_time is None
        metrics["box_drop_time"] = box_drop_time
        # Spec: success := success_coarse AND no fall through squat-place-rise
        # AND box_place_ok (fine convergence recorded, not gated).
        success = bool(metrics["success_coarse"]) and not fell and box_place_ok

    success = bool(success) and harness_error is None

    result = {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial,
        "seed": seed,
        "success": success,
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "harness_error": harness_error,
        "metrics": metrics,
        "video": None,  # filled by caller
    }
    if test == "pipeline_abc":
        result["box_mode"] = BOX_MODE
    if rig.upper_mode:
        result["upper_mode"] = rig.upper_mode
    return result


# ---------------------------------------------------------------------------
# T9 squat_pick_ground trial runner (spec v5) — event-driven grasp, so it is
# a stateful stage machine like run_nav_trial (not a pure command profile).
# ---------------------------------------------------------------------------


def run_pick_trial(rig: Rig, geoms: GeomSets, test: str, trial: int, seed: int,
                   video: TrialVideo | None) -> dict:
    """settle -> arm blend into the reach pose -> squat to the unified 0.25
    command -> magnetic grasp at the bottom (both wrists < 0.30 m from the
    box surface; 5 s timeout -> pick_failed) -> loaded rise -> stand 1 s."""
    yaw0 = float(np.random.default_rng(seed).uniform(-math.pi, math.pi))
    reset_trial(rig, seed, yaw0)
    policy, n_zeroed = fresh_policy(rig)
    if trial == 0:
        print(f"  [policy] zeroed {n_zeroed} recurrent state buffer(s)")

    sim = rig.sim
    dt = sim.dt
    model = sim.mj_model
    reach_indices = torch.tensor([sim.joint_names.index(j) for j in ARM_REACH_POSE],
                                 dtype=torch.long)
    reach_targets = torch.tensor(list(ARM_REACH_POSE.values()),
                                 dtype=torch.float32, device=rig.device)
    arm_qpos_adrs = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
        for j in ARM_REACH_POSE
    ]

    ramp_s = (STAND_HEIGHT - PICK_TARGET_H) / PICK_RAMP_RATE
    max_s = (SETTLE_S + PICK_ARM_BLEND_S + ramp_s + PICK_BOTTOM_TIMEOUT_S
             + PICK_HOLD_AFTER_GRASP_S + ramp_s + PICK_STAND_S + 1.0)
    n_steps = int(round(max_s / dt))

    stage = "settle"
    stage_t0 = 0.0
    blend_from: torch.Tensor | None = None  # arm qpos captured at reach entry
    grasped = False
    grasp_dists: tuple[float, float] | None = None  # at grasp or at timeout
    bottom_anchor: tuple[float, float] | None = None  # XY at bottom entry
    root_drift = None  # max XY drift through bottom + grasp_hold
    min_root_z = None  # min base_z after settle
    stand_done = False
    tilts: list[float] = []
    fell = False
    fall_time = fall_phase = None
    harness_error = None
    t_end = 0.0

    wall0 = time.time()
    for i in range(n_steps):
        t = i * dt
        data = sim.mj_data

        # ---- stage transitions (evaluated on the pre-step state) ----
        if stage == "settle" and t >= SETTLE_S:
            stage, stage_t0 = "reach", t
            blend_from = torch.tensor([float(data.qpos[a]) for a in arm_qpos_adrs],
                                      dtype=torch.float32, device=rig.device)
        if stage == "reach" and (t - stage_t0) >= PICK_ARM_BLEND_S:
            stage, stage_t0 = "descend", t
        if stage == "descend" and (t - stage_t0) >= ramp_s:
            stage, stage_t0 = "bottom", t
            bottom_anchor = (float(data.qpos[0]), float(data.qpos[1]))
        if stage == "bottom":
            d_l = box_surface_dist_m(data, rig.box, rig.box.wrist_body_ids[0], PICK_BOX_HALF)
            d_r = box_surface_dist_m(data, rig.box, rig.box.wrist_body_ids[1], PICK_BOX_HALF)
            if d_l < PICK_GRASP_DIST_M and d_r < PICK_GRASP_DIST_M:
                grasp_dists = (d_l, d_r)
                grasped = True
                activate_welds_at_current_pose(model, data, rig.box)
                stage, stage_t0 = "grasp_hold", t
            elif (t - stage_t0) >= PICK_BOTTOM_TIMEOUT_S:
                grasp_dists = (d_l, d_r)  # at-timeout distances (spec v5)
                stage, stage_t0 = "rise", t  # pick_failed: still rises
        if stage == "grasp_hold" and (t - stage_t0) >= PICK_HOLD_AFTER_GRASP_S:
            stage, stage_t0 = "rise", t
        if stage == "rise" and (t - stage_t0) >= ramp_s:
            stage, stage_t0 = "stand", t
        if stage == "stand" and (t - stage_t0) >= PICK_STAND_S:
            stand_done = True
            t_end = t
            break

        # ---- height command for this stage (vx=vy=wz=0 all trial) ----
        if stage == "descend":
            h_c = max(STAND_HEIGHT - PICK_RAMP_RATE * (t - stage_t0), PICK_TARGET_H)
        elif stage in ("bottom", "grasp_hold"):
            h_c = PICK_TARGET_H
        elif stage == "rise":
            h_c = min(PICK_TARGET_H + PICK_RAMP_RATE * (t - stage_t0), STAND_HEIGHT)
        else:  # settle / reach / stand
            h_c = STAND_HEIGHT
        rig.mgr.set_command(0.0, 0.0, 0.0, h_c)

        # --- official sim2mujoco_eval loop order ---
        sim_state = sim.get_state()
        obs = rig.obs_processor.compute(sim_state)
        with torch.no_grad():
            actions = policy(obs)
        rig.obs_processor.set_last_action(actions)
        joint_cmd = rig.act_processor.process(actions)
        if stage != "settle":
            # Arms: linear PD-target blend from the settle-end pose into the
            # reach pose during "reach", then held there for the whole trial
            # (the policy keeps the legs; same override path as carry tests).
            if stage == "reach":
                alpha = min((t - stage_t0) / PICK_ARM_BLEND_S, 1.0)
                arm_row = blend_from + (reach_targets - blend_from) * alpha
            else:
                arm_row = reach_targets
            position = joint_cmd.position.clone()
            position[reach_indices] = arm_row
            joint_cmd = rig.A.JointCommand(position=position, kp=joint_cmd.kp, kd=joint_cmd.kd)
        for _ in range(sim.decimation):
            sim.step(joint_cmd)
        t_end = (i + 1) * dt

        if (
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
            or not np.isfinite(data.qpos).all()
        ):
            harness_error = f"divergence (qacc explosion / NaN qpos) at t={t_end:.2f}s"
            break

        base_z = float(data.qpos[2])
        if t_end > SETTLE_S:
            min_root_z = base_z if min_root_z is None else min(min_root_z, base_z)
        if bottom_anchor is not None and stage in ("bottom", "grasp_hold"):
            drift = math.hypot(float(data.qpos[0]) - bottom_anchor[0],
                               float(data.qpos[1]) - bottom_anchor[1])
            root_drift = drift if root_drift is None else max(root_drift, drift)

        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        tilt = quat_tilt(quat)
        tilts.append(tilt)
        nonfoot_ground, _ = scan_contacts(data, geoms)
        is_fall = tilt > FALL_TILT_RAD or base_z < (h_c - FALL_HEIGHT_MARGIN) or nonfoot_ground

        if video is not None and (is_fall or i % VIDEO_EVERY_N_CTRL_STEPS == 0):
            video.capture(data)

        if is_fall:
            fell = True
            fall_time = t_end
            fall_phase = stage  # "rise" = the loaded stand-up fall (spec v5)
            break

    wall_s = time.time() - wall0

    stand_ok = stand_done and not fell and harness_error is None
    metrics = {
        "pick_success": grasped,
        "grasp_dist_left": grasp_dists[0] if grasp_dists else None,
        "grasp_dist_right": grasp_dists[1] if grasp_dists else None,
        "min_root_z": min_root_z,
        "root_drift": root_drift,
        "stand_ok": stand_ok,
        "max_tilt": float(np.max(tilts)) if tilts else None,
        "wall_time_s": round(wall_s, 2),
    }
    success = grasped and (not fell) and stand_ok and harness_error is None

    return {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial,
        "seed": seed,
        "success": bool(success),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "harness_error": harness_error,
        "metrics": metrics,
        "video": None,  # filled by caller
        "box_mode": BOX_MODE,
    }


# ---------------------------------------------------------------------------
# T10 vln_follow (spec v5): shared tape loader + trial runner
# ---------------------------------------------------------------------------


def load_vln_tapes(path: Path) -> list[SimpleNamespace]:
    """Load and validate a make_vln_tapes.py tapes.json (spec v5).

    Format: {"tapes": [{"id", "dt", "cmds": [[t,vx,vy,wz],...] (ZOH
    breakpoints), "ref_xy_yaw": [[t,x,y,yaw],...] (1 Hz ideal integral)}]}.
    The SAME file is fed to all four harnesses for fairness.
    """
    if not path.exists():
        raise SystemExit(f"vln tapes json not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"vln tapes json is not valid JSON: {path} ({exc})")
    raw_tapes = payload.get("tapes") if isinstance(payload, dict) else None
    if not isinstance(raw_tapes, list) or not raw_tapes:
        raise SystemExit(f"vln tapes json must contain a non-empty 'tapes' list: {path}")

    tapes = []
    for k, raw in enumerate(raw_tapes):
        cmds = np.asarray(raw.get("cmds", []), dtype=np.float64)
        ref = np.asarray(raw.get("ref_xy_yaw", []), dtype=np.float64)
        if cmds.ndim != 2 or cmds.shape[1] != 4 or cmds.shape[0] == 0:
            raise SystemExit(f"tape {k}: cmds must be a non-empty list of [t,vx,vy,wz]: {path}")
        if ref.ndim != 2 or ref.shape[1] != 4 or ref.shape[0] == 0:
            raise SystemExit(f"tape {k}: ref_xy_yaw must be a non-empty list of [t,x,y,yaw]: {path}")
        if not (np.isfinite(cmds).all() and np.isfinite(ref).all()):
            raise SystemExit(f"tape {k}: NaN/inf in cmds/ref_xy_yaw: {path}")
        if np.any(np.diff(cmds[:, 0]) < 0.0) or np.any(np.diff(ref[:, 0]) < 0.0):
            raise SystemExit(f"tape {k}: cmds/ref timestamps must be sorted: {path}")
        dt_tape = float(raw.get("dt", CONTROL_DT))
        if abs(dt_tape - CONTROL_DT) > 1e-9:
            print(f"WARNING: tape {k} dt={dt_tape} != harness control dt {CONTROL_DT} "
                  "(cmds are ZOH breakpoints, so this only matters for provenance)")
        tapes.append(SimpleNamespace(
            id=int(raw.get("id", k)),
            cmds=cmds,
            ref=ref,
            duration_s=float(max(cmds[-1, 0], ref[-1, 0])),
        ))
    return tapes


def vln_zoh_cmd(cmds: np.ndarray, t: float) -> tuple[float, float, float]:
    """Zero-order-hold lookup: the last breakpoint with time <= t (zeros
    before the first breakpoint)."""
    k = int(np.searchsorted(cmds[:, 0], t + 1e-9, side="right")) - 1
    if k < 0:
        return 0.0, 0.0, 0.0
    return float(cmds[k, 1]), float(cmds[k, 2]), float(cmds[k, 3])


def vln_stop_segments(cmds: np.ndarray, duration_s: float) -> list[tuple[float, float]]:
    """Merged [t0, t1) intervals where the tape commands a FULL stop
    (vx = vy = wz = 0); t1 of the last breakpoint row is the tape end."""
    segments: list[tuple[float, float]] = []
    for k in range(cmds.shape[0]):
        is_stop = bool(np.all(np.abs(cmds[k, 1:4]) < 1e-9))
        if not is_stop:
            continue
        t0 = float(cmds[k, 0])
        t1 = float(cmds[k + 1, 0]) if k + 1 < cmds.shape[0] else duration_s
        if segments and abs(segments[-1][1] - t0) < 1e-9:
            segments[-1] = (segments[-1][0], t1)  # merge contiguous stop rows
        else:
            segments.append((t0, t1))
    return segments


def run_vln_trial(rig: Rig, geoms: GeomSets, test: str, trial: int, seed: int,
                  video: TrialVideo | None, tape: SimpleNamespace) -> dict:
    """2 s settle -> ZOH playback of one VLN command tape -> stand 1 s.
    The robot starts at the origin with yaw 0 (the ref trajectory is the
    ideal error-free integral from that pose). AGILE consumes wz natively
    as a yaw-rate command (no target-yaw integration — contrast bench_amo)."""
    reset_trial(rig, seed, 0.0)
    policy, n_zeroed = fresh_policy(rig)
    if trial == 0:
        print(f"  [policy] zeroed {n_zeroed} recurrent state buffer(s)")

    sim = rig.sim
    dt = sim.dt
    tape_s = tape.duration_s
    n_steps = int(round((SETTLE_S + tape_s + VLN_END_STAND_S) / dt))

    ts: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    fwd_vels: list[float] = []
    planar_speeds: list[float] = []
    vx_cmds: list[float] = []
    tilts: list[float] = []
    fell = False
    fall_time = fall_phase = None
    harness_error = None

    wall0 = time.time()
    for i in range(n_steps):
        t_cmd = i * dt - SETTLE_S
        if t_cmd < 0.0:
            vx_c, vy_c, wz_c = 0.0, 0.0, 0.0
            phase = "settle"
        elif t_cmd < tape_s:
            vx_c, vy_c, wz_c = vln_zoh_cmd(tape.cmds, t_cmd)
            phase = "tape"
        else:
            vx_c, vy_c, wz_c = 0.0, 0.0, 0.0
            phase = "stand_end"
        rig.mgr.set_command(vx_c, vy_c, wz_c, STAND_HEIGHT)

        # --- official sim2mujoco_eval loop order ---
        sim_state = sim.get_state()
        obs = rig.obs_processor.compute(sim_state)
        with torch.no_grad():
            actions = policy(obs)
        rig.obs_processor.set_last_action(actions)
        joint_cmd = rig.act_processor.process(actions)
        for _ in range(sim.decimation):
            sim.step(joint_cmd)

        data = sim.mj_data
        if (
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
            or not np.isfinite(data.qpos).all()
        ):
            harness_error = f"divergence (qacc explosion / NaN qpos) at t={(i + 1) * dt:.2f}s"
            break

        t_meas = (i + 1) * dt - SETTLE_S
        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        yaw = quat_yaw(quat)
        tilt = quat_tilt(quat)
        base_z = float(data.qpos[2])
        vw = np.asarray(data.qvel[:3], dtype=np.float64)

        if t_meas > 0.0:
            ts.append(t_meas)
            xs.append(float(data.qpos[0]))
            ys.append(float(data.qpos[1]))
            yaws.append(yaw)
            fwd_vels.append(float(vw[0] * math.cos(yaw) + vw[1] * math.sin(yaw)))
            planar_speeds.append(float(math.hypot(vw[0], vw[1])))
            vx_cmds.append(vx_c)
        tilts.append(tilt)

        nonfoot_ground, _ = scan_contacts(data, geoms)
        is_fall = (tilt > FALL_TILT_RAD
                   or base_z < (STAND_HEIGHT - FALL_HEIGHT_MARGIN)
                   or nonfoot_ground)

        if video is not None and (is_fall or i % VIDEO_EVERY_N_CTRL_STEPS == 0):
            video.capture(data)

        if is_fall:
            fell = True
            fall_time = (i + 1) * dt
            fall_phase = phase
            break

    wall_s = time.time() - wall0

    # ---------------- metrics ----------------
    final_pos_err = final_yaw_err = None
    if ts:
        rx, ry, ryaw = float(tape.ref[-1, 1]), float(tape.ref[-1, 2]), float(tape.ref[-1, 3])
        final_pos_err = float(math.hypot(xs[-1] - rx, ys[-1] - ry))
        final_yaw_err = float(abs(wrap_angle(yaws[-1] - ryaw)))

    # mean_track_err: position error vs the 1 Hz ref samples actually reached.
    track_errs = []
    for row in tape.ref:
        t_ref = float(row[0])
        if t_ref <= 0.0:
            continue  # both start at the origin by construction
        j = int(round(t_ref / dt)) - 1
        if j >= len(ts):
            break  # fell / diverged before this ref sample
        track_errs.append(math.hypot(xs[j] - float(row[1]), ys[j] - float(row[2])))
    mean_track_err = float(np.mean(track_errs)) if track_errs else None

    # small_cmd_response: actual/commanded vx ratio over |vx| in the band.
    lo, hi = VLN_SMALL_VX_BAND
    ratios = [fv / vc for fv, vc in zip(fwd_vels, vx_cmds) if lo <= abs(vc) <= hi]
    small_cmd_response = float(np.mean(ratios)) if ratios else None

    # stop_settle: mean planar speed inside full-stop tape segments, after a
    # 0.5 s settle-in at the head of each segment (deceleration excluded).
    stop_speeds = []
    for t0, t1 in vln_stop_segments(tape.cmds, tape_s):
        for j, t in enumerate(ts):
            if t0 + VLN_STOP_SKIP_S <= t < t1:
                stop_speeds.append(planar_speeds[j])
    stop_settle = float(np.mean(stop_speeds)) if stop_speeds else None

    metrics = {
        "tape_id": tape.id,
        "final_pos_err": final_pos_err,
        "final_yaw_err": final_yaw_err,
        "mean_track_err": mean_track_err,
        "small_cmd_response": small_cmd_response,
        "stop_settle": stop_settle,
        "max_tilt": float(np.max(tilts)) if tilts else None,
        "wall_time_s": round(wall_s, 2),
    }
    success = (
        (not fell)
        and harness_error is None
        and final_pos_err is not None
        and final_pos_err <= VLN_FINAL_POS_TOL_M
        and final_yaw_err <= VLN_FINAL_YAW_TOL_RAD
    )

    return {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial,
        "seed": seed,
        "success": bool(success),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "harness_error": harness_error,
        "metrics": metrics,
        "video": None,  # filled by caller
        "cmd_mode": "vln_tape_zoh_native_wz",
    }


# ===========================================================================
# ManipArena v2 (BENCHMARK_V2_DESIGN §2-§4). AGILE adapter — mirrors the
# reference HOMIE adapter (bench_homie.py):
#   build_arena_scene      -> inject ms.furniture_xml into scene_29dof.xml
#   configure_arena_welds  -> resolve free-body addresses + weld eq ids
#   AgileMissionIO         -> the MissionIO callbacks manip_mission.Mission drives
#   run_arena_trial        -> spawn at the spec pose, drive the Mission, score it
#   run_arena_l_trial      -> L-series locomotion sweeps on the arena scene
#   run_arena_main         -> entry (arena_M1/M2/L1-L5)
# Physics fixes carried over verbatim (a-e from the task brief): canonical
# upright weld snap, contype=0 while grasped, level box on release, carry vy
# damp + vx<=0.4, shallow store squat.
# ===========================================================================
def reach_base_h(top_h: float) -> float:
    """AGILE absolute pelvis-height command to bring the wrists to a surface at
    height ``top_h``. Linear "lower surface -> lower base" with a fixed reach
    offset, clipped to the trackable AGILE domain [ARENA_H_MIN, ARENA_STAND_H].
    The depth floor of the policy (~0.49-0.53) decides whether the low tables
    are reachable (the point of the H_pick sweep — low tables expected to fail)."""
    return float(min(max(top_h + ARENA_REACH_OFFSET, ARENA_H_MIN), ARENA_STAND_H))


def build_arena_scene(scene_path: Path, spec: "ms.SceneSpec") -> Path:
    """Inject the ManipArena furniture + free bodies into scene_29dof.xml and
    write a sibling file (so the relative include/meshdir paths still resolve).

    Mechanics (mirrors build_scene_variant + bench_homie.build_arena_model):
      - furniture bodies appended before </worldbody> (after the robot include),
        so the box/cube freejoints land at the qpos/qvel TAIL; AGILE keeps
        qpos[0:36]/qvel[0:35] and the 29-joint obs slice is never polluted.
      - the <equality> weld block inserted before </mujoco>.
      - ms.WRIST_L/WRIST_R placeholders replaced with the AGILE wrist links.
      - runtime: floor planes (+ world body) get bit 2 OR-ed in so the box/cube
        collide with the ground (same bit-2 mechanism as the psi0 tests)."""
    text = scene_path.read_text()
    for token in ("<worldbody>", "</worldbody>", "</mujoco>"):
        if token not in text:
            raise SystemExit(f"Cannot find {token} in {scene_path}")
    frags = ms.furniture_xml(spec, collision_scheme="bit2")
    text = text.replace("</worldbody>", "    " + frags["bodies"] + "\n  </worldbody>", 1)
    text = text.replace("</mujoco>", "  " + frags["equality"] + "\n</mujoco>", 1)
    # Substitute wrist placeholders AFTER injecting the weld block (they only
    # live inside that block).
    text = text.replace(ms.WRIST_L_PLACEHOLDER, ARENA_WRIST_L)
    text = text.replace(ms.WRIST_R_PLACEHOLDER, ARENA_WRIST_R)
    out_path = scene_path.with_name(scene_path.stem + f"__arena_v{spec.variant_seed}.xml")
    out_path.write_text(text)
    return out_path


@dataclass(frozen=True)
class ArenaRig:
    """Model ids / addresses for the arena free bodies + 4 welds."""
    box_qadr: int
    box_vadr: int
    cube_qadr: int
    cube_vadr: int
    box_body: int
    cube_body: int
    torso_body: int
    wrist_left: int
    wrist_right: int
    box_geom: int
    cube_geom: int
    eq_box_left: int
    eq_box_right: int
    eq_cube_box: int
    eq_cube_right: int


def configure_arena_welds(model: mujoco.MjModel) -> ArenaRig:
    """Resolve the arena free-body addresses + weld equality ids. Confirms the
    qpos tail layout matches FREE_BODY_QPOS_LAYOUT (box then cube after the
    robot's 36 qpos). The welds are re-anchored at the live wrist<->box pose at
    grasp time (activate_welds_at_current_pose-style), not pre-anchored."""
    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"arena joint missing: {name}")
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    box_qadr, box_vadr = jq("carried_box_freejoint")
    cube_qadr, cube_vadr = jq("relay_cube_freejoint")
    lay = ms.FREE_BODY_QPOS_LAYOUT
    # robot end = 36/35 for AGILE 29dof; box then cube at the tail.
    if box_qadr != 36 or cube_qadr != 43:
        raise SystemExit(
            f"arena free-body tail layout unexpected: box_qadr={box_qadr} "
            f"cube_qadr={cube_qadr} (want 36/43)"
        )
    if cube_qadr - box_qadr != lay["cube_qpos_offset_from_robot_end"]:
        raise SystemExit("cube qpos offset mismatch vs FREE_BODY_QPOS_LAYOUT")

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def eid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    return ArenaRig(
        box_qadr=box_qadr, box_vadr=box_vadr,
        cube_qadr=cube_qadr, cube_vadr=cube_vadr,
        box_body=bid("carried_box"), cube_body=bid("relay_cube"),
        torso_body=bid("torso_link"),
        wrist_left=bid(ARENA_WRIST_L), wrist_right=bid(ARENA_WRIST_R),
        box_geom=gid("carried_box_geom"), cube_geom=gid("relay_cube_geom"),
        eq_box_left=eid("box_weld_left"), eq_box_right=eid("box_weld_right"),
        eq_cube_box=eid("cube_weld_box"), eq_cube_right=eid("cube_weld_right"),
    )


def _arena_floor_bit2(model: mujoco.MjModel, body_id: int) -> None:
    """OR bit 2 into a free body + the floor planes + world body so the body
    collides with the ground (§3 release semantics; same as enable_floor_bit2)."""
    if hasattr(model, "body_contype"):
        model.body_contype[body_id] |= 2
        model.body_conaffinity[body_id] |= 2
        model.body_contype[0] |= 2
        model.body_conaffinity[0] |= 2
    for g in range(model.ngeom):
        if (model.geom_bodyid[g] == 0
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE):
            model.geom_contype[g] |= 2
            model.geom_conaffinity[g] |= 2


def _arena_weld_at_pose(model, data, eq, body1, body2) -> None:
    """Activate weld ``eq`` anchored at the CURRENT relative body2-in-body1 pose
    (no body moved). Same math as activate_welds_at_current_pose, generalised to
    any body1/body2 pair (wrist<->box, box<->cube)."""
    rot1 = data.xmat[body1].reshape(3, 3)
    relp = rot1.T @ (data.xpos[body2] - data.xpos[body1])
    q1inv = np.zeros(4)
    relq = np.zeros(4)
    mujoco.mju_negQuat(q1inv, data.xquat[body1])
    mujoco.mju_mulQuat(relq, q1inv, data.xquat[body2])
    model.eq_data[eq, 0:3] = 0.0
    model.eq_data[eq, 3:6] = relp
    model.eq_data[eq, 6:10] = relq
    model.eq_data[eq, 10] = 1.0
    data.eq_active[eq] = 1


def _arena_snap_box_upright(model, data, box: ArenaRig, base_yaw: float) -> None:
    """Physics fix (a)+(b): before welding, move the box into a canonical UPRIGHT
    pose — position = midpoint of the two wrists, orientation = pure yaw locked
    to the base (box z == world z). Makes the two wrist<->box welds anchor at
    mutually-consistent relposes (no fighting) and guarantees the box rides
    upright so it lands level on release. Writes the box freejoint qpos + zeros
    qvel, then mj_forward so the welds anchor against the corrected pose."""
    bq, bv = box.box_qadr, box.box_vadr
    mid = 0.5 * (data.xpos[box.wrist_left] + data.xpos[box.wrist_right])
    data.qpos[bq:bq + 3] = mid
    data.qpos[bq + 3:bq + 7] = [math.cos(base_yaw / 2.0), 0.0, 0.0,
                                math.sin(base_yaw / 2.0)]
    data.qvel[bv:bv + 6] = 0.0
    mujoco.mj_forward(model, data)


def _arena_weld_box_both(model, data, box: ArenaRig) -> None:
    """Activate BOTH wrist<->box welds, anchored at the canonical upright box
    pose (see _arena_snap_box_upright). Shares the 2 kg load between the arms;
    solref/solimp left at the compliant XML defaults."""
    _arena_weld_at_pose(model, data, box.eq_box_right, box.wrist_right, box.box_body)
    _arena_weld_at_pose(model, data, box.eq_box_left, box.wrist_left, box.box_body)


def _arena_box_surface_dist(data, body_id: int, wrist_body_id: int,
                            half: tuple) -> float:
    """Distance [m] from a wrist body origin to the oriented free-body surface
    (0 if inside)."""
    rel = data.xpos[wrist_body_id] - data.xpos[body_id]
    local = data.xmat[body_id].reshape(3, 3).T @ rel
    half_arr = np.asarray(half, dtype=np.float64)
    return float(np.linalg.norm(local - np.clip(local, -half_arr, half_arr)))


def prepare_arena_replay(model: mujoco.MjModel, replay: SimpleNamespace) -> SimpleNamespace:
    """Map the agile29 replay columns by joint NAME onto MJCF joint indices and
    label the frames into reach_down / carry / place windows the M-series
    segments index into (reuses the squat_place segmentation). Returns a NEW
    namespace with arena_indices (LongTensor of model joint qpos addresses for
    the replay columns), pos_np (N,K), and arena_segments={name:(k0,k1)}."""
    qpos_adrs = []
    for name in replay.names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"arena replay joint missing from model: {name}")
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
    # bottom plateau (already computed at load time: place_idx / hold_*).
    h = replay.height
    n = replay.n
    k_low = int(np.argmin(h))
    # descent/hold/rise labels around the bottom plateau (height <= h_min + band)
    hmin = float(h[k_low])
    first = replay.hold_start_idx
    last = replay.hold_end_idx
    reach = (0, max(first, 1))           # the lowering motion toward the surface
    place = (first, n)                   # the set-down + retract (hold -> rise)
    carry = (max(n - max(1, n // 10), 0), n)  # standing-tall tail (walk window)
    return SimpleNamespace(
        **vars(replay),
        arena_qpos_adrs=qpos_adrs,
        arena_segments={"reach_down": reach, "place": place, "carry": carry},
        seg_bottom_k=k_low,
    )


class AgileMissionIO:
    """MissionIO (manip_mission.py Protocol) bound to a live AGILE MjModel/
    MjData. The Mission owns sequencing + per-stage metrics; this adapter owns
    the physics side effects. Per policy tick the arena rollout does:

        cmd = mission.step(t)          # mission reads THIS io, fires welds, ..
        io.set_command(*cmd)           # store (vx,vy,wz,height) for the policy
        <rollout sets mgr command, overrides upper joints, runs the LSTM policy>

    Heights are AGILE-native ABSOLUTE pelvis height; wz is the native yaw-rate
    channel. The upper body is driven by an agile29 real_ep053 Psi0 replay (set
    via replay_upper); the rollout reads io.upper_row each tick and overrides
    the 17 replayed joints in the JointCommand position."""

    def __init__(self, model, data, box: ArenaRig, replay, h_stand=ARENA_STAND_H):
        self.m = model
        self.d = data
        self.box = box
        self.replay = replay
        self.h_stand = h_stand
        self.cmd = (0.0, 0.0, 0.0, h_stand)        # vx, vy, wz, height
        # 17-dim replay row (MJCF replay-column order) the rollout writes:
        self.upper_row = (replay.pos[0].copy() if replay is not None else None)
        self.box_grasped = False
        self.cube_on_box = False
        self._fallen = False

    # --- rollout-facing helpers (NOT part of MissionIO) ------------------
    def set_command(self, vx, vy, wz, height):
        self.cmd = (float(vx), float(vy), float(wz), float(height))

    def set_fallen(self, flag):
        self._fallen = bool(flag)

    # --- MissionIO: read state -------------------------------------------
    def root_pose(self):
        return (float(self.d.qpos[0]), float(self.d.qpos[1]),
                quat_yaw(np.asarray(self.d.qpos[3:7], dtype=np.float64)))

    def root_tilt(self):
        # unified fall tilt scalar = gravity-xy magnitude in the base frame =
        # sin(tilt); quat_tilt returns the tilt angle.
        return float(math.sin(quat_tilt(np.asarray(self.d.qpos[3:7], dtype=np.float64))))

    def wrist_box_dist(self):
        d_l = _arena_box_surface_dist(self.d, self.box.box_body,
                                      self.box.wrist_left, ARENA_BOX_HALF)
        d_r = _arena_box_surface_dist(self.d, self.box.box_body,
                                      self.box.wrist_right, ARENA_BOX_HALF)
        return (d_l, d_r)

    def right_wrist_cube_dist(self):
        return _arena_box_surface_dist(self.d, self.box.cube_body,
                                       self.box.wrist_right, ARENA_CUBE_HALF)

    def box_pose(self):
        b = self.box.box_body
        bx, by, bz = (float(v) for v in self.d.xpos[b])
        zz = float(self.d.xmat[b].reshape(3, 3)[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        speed = float(np.linalg.norm(
            self.d.qvel[self.box.box_vadr:self.box.box_vadr + 3]))
        return (bx, by, bz, tilt, speed)

    def cube_pose(self):
        c = self.box.cube_body
        cx, cy, cz = (float(v) for v in self.d.xpos[c])
        zz = float(self.d.xmat[c].reshape(3, 3)[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        speed = float(np.linalg.norm(
            self.d.qvel[self.box.cube_vadr:self.box.cube_vadr + 3]))
        return (cx, cy, cz, tilt, speed)

    def base_height(self):
        return float(self.d.qpos[2])

    def is_fallen(self):
        return self._fallen

    # --- MissionIO: act (lower body) -------------------------------------
    def send_leg_cmd(self, vx, vy, wz, height):
        # height is AGILE absolute pelvis z; the manager clamp is widened to
        # ARENA_HEIGHT_RANGE (0.2, 0.8) so the mission target reaches the policy.
        self.set_command(vx, vy, wz, height)

    # --- MissionIO: act (upper body, Psi0 replay) ------------------------
    def replay_upper(self, segment, phase_t):
        """Drive upper body for ``segment`` ('reach_down' / 'carry' / 'place').

        Psi0 uses a frame replay. box_demo_2 uses online IK against the live
        arena state but keeps the same MissionIO contract.
        """
        rep = self.replay
        if rep is None:
            return
        if getattr(rep, "is_boxdemo", False):
            self.upper_row = rep.row(segment, phase_t, io=self)
            return
        seg = rep.arena_segments.get(segment)
        if seg is None:
            self.upper_row = rep.pos[0]
            return
        k0, k1 = seg
        n = max(k1 - k0, 1)
        k = k0 + int(min(max(phase_t / CONTROL_DT, 0.0), n - 1))
        k = min(max(k, 0), rep.n - 1)
        self.upper_row = rep.pos[k]

    # --- MissionIO: act (weld / release / cube-transfer) -----------------
    def weld_box(self):
        """Magnetic grasp (physics fix a+b): SNAP the box to a canonical upright
        pose then weld BOTH wrists at that consistent symmetric pose, and turn
        the box collision OFF while held (it overlaps the pick table during a
        deep reach — leaving it collidable smashes it into the table and spikes
        qacc on stand-up). Collision (bit 2) is re-enabled only at release."""
        base_yaw = quat_yaw(np.asarray(self.d.qpos[3:7], dtype=np.float64))
        _arena_snap_box_upright(self.m, self.d, self.box, base_yaw)
        _arena_weld_box_both(self.m, self.d, self.box)
        self.m.geom_contype[self.box.box_geom] = 0
        self.m.geom_conaffinity[self.box.box_geom] = 0
        if hasattr(self.m, "body_contype"):
            self.m.body_contype[self.box.box_body] = 0
            self.m.body_conaffinity[self.box.box_body] = 0
        self.box_grasped = True

    def release_box(self, surface_top_h=None, target_xy=None):
        """Physics fix (c): set the box down LEVEL upright (yaw == base yaw) at
        PELVIS + a forward release offset toward the placement target (clamped
        inside the zone footprint so a small coarse-nav undershoot still lands
        clean), z resting just above the surface so it lands flat. Drop both
        welds + enable bit-2 ground collision. The cube weld (if active) stays."""
        bq, bv = self.box.box_qadr, self.box.box_vadr
        base_yaw = quat_yaw(np.asarray(self.d.qpos[3:7], dtype=np.float64))
        px, py = float(self.d.qpos[0]), float(self.d.qpos[1])
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
        self.d.eq_active[self.box.eq_box_left] = 0
        self.d.eq_active[self.box.eq_box_right] = 0
        self.m.geom_contype[self.box.box_geom] = ms.FREE_BODY_CT_CA
        self.m.geom_conaffinity[self.box.box_geom] = ms.FREE_BODY_CT_CA
        _arena_floor_bit2(self.m, self.box.box_body)
        mujoco.mj_forward(self.m, self.d)
        self.box_grasped = False

    def transfer_cube(self):
        """Weld the cube to the box top at the live relative pose (cube rides the
        box thereafter). Also makes the cube collide via bit-2 so it never
        free-falls through a placed box later."""
        _arena_weld_at_pose(self.m, self.d, self.box.eq_cube_box,
                            self.box.box_body, self.box.cube_body)
        self.m.geom_contype[self.box.cube_geom] = ms.FREE_BODY_CT_CA
        self.m.geom_conaffinity[self.box.cube_geom] = ms.FREE_BODY_CT_CA
        self.cube_on_box = True


# ---------------------------------------------------------------------------
# Pillar-avoidance: insert a lateral skirt waypoint into a nav leg whose
# straight segment passes too close to the pillar (mirrors bench_homie).
# Implemented HARNESS-side: rewrites the mission's NavLeg list before the
# Mission runs, so the shared mission engine stays physics-free.
# ---------------------------------------------------------------------------
def _seg_point_dist(p0, p1, c) -> float:
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
                          clear_m=ARENA_PILLAR_CLEAR_M, side_m=ARENA_PILLAR_SIDE_M):
    """Return a NEW primitive list where any NavLeg whose straight path passes
    within ``clear_m`` of the pillar is preceded by a skirt NavLeg offset to the
    side of the segment (away from the pillar). Pure geometry; immutable."""
    out = []
    prev_xy = tuple(spawn_xy)
    for prim in sequence:
        if isinstance(prim, mm.NavLeg):
            tgt_xy = (prim.target_pose[0], prim.target_pose[1])
            dist = _seg_point_dist(prev_xy, tgt_xy, pillar_xy)
            if dist < clear_m and not getattr(prim, "skip_detour", False):
                mx = 0.5 * (prev_xy[0] + tgt_xy[0])
                my = 0.5 * (prev_xy[1] + tgt_xy[1])
                sx, sy = tgt_xy[0] - prev_xy[0], tgt_xy[1] - prev_xy[1]
                slen = math.hypot(sx, sy) or 1.0
                nx, ny = -sy / slen, sx / slen
                if ((mx + nx - pillar_xy[0]) ** 2 + (my + ny - pillar_xy[1]) ** 2) < \
                   ((mx - nx - pillar_xy[0]) ** 2 + (my - ny - pillar_xy[1]) ** 2):
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


def _arena_stage_flags(stages: list) -> dict:
    """Flatten per-stage records into the M-series diagnostic flags (nav arrival
    errors, place_ok, cube_transfer_ok, regrasp_ok). Mirrors bench_homie."""
    flags: dict = {}
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
            surface = s.get("name", "")
            if "relay" in surface:
                flags["place_ok_relay"] = ok
            elif "store" in surface:
                flags["place_ok_store"] = ok
            else:
                flags[f"place_ok_{place_seen}"] = ok
        elif kind == "cube":
            flags["cube_transfer_ok"] = bool(s.get("cube_transfer_ok"))
    return flags


def _arena_apply_upper(rig: Rig, joint_cmd, upper_row, upper_indices):
    """Override the replayed upper-body joints in a JointCommand position.
    ``upper_row`` is a numpy (K,) row in the replay-column order; upper_indices
    are the MJCF joint indices for those columns."""
    if upper_row is None:
        return joint_cmd
    position = joint_cmd.position.clone()
    row_t = torch.as_tensor(upper_row, dtype=position.dtype, device=position.device)
    position[upper_indices] = row_t
    return rig.A.JointCommand(position=position, kp=joint_cmd.kp, kd=joint_cmd.kd)


def run_arena_trial(rig: Rig, geoms: GeomSets, box: ArenaRig, spec: "ms.SceneSpec",
                    test: str, trial: int, seed: int,
                    video: TrialVideo | None) -> dict:
    """Run M1/M2 on the ManipArena variant ``spec``. Reuses the AGILE obs/policy/
    action pipeline; the locomotion command + upper-body target come from the
    Mission/AgileMissionIO instead of command_profile. Deterministic given seed."""
    sim = rig.sim
    model = sim.mj_model
    data = sim.mj_data
    dt = sim.dt
    rep = rig.upper
    upper_indices = rig.upper_indices

    rng = np.random.default_rng(seed)
    sd = spec.to_dict()
    spawn = sd["spawn"]
    yaw0 = float(spawn[2])

    # --- reset + spawn at the spec pose --------------------------------------
    sim.reset()
    torch.manual_seed(seed)
    # re-disable any box/cube collision a previous trial's place enabled
    for body, geom in ((box.box_body, box.box_geom), (box.cube_body, box.cube_geom)):
        model.geom_contype[geom] = ms.FREE_BODY_CT_CA
        model.geom_conaffinity[geom] = ms.FREE_BODY_CT_CA
        if hasattr(model, "body_contype"):
            model.body_contype[body] = ms.FREE_BODY_CT_CA
            model.body_conaffinity[body] = ms.FREE_BODY_CT_CA
    # robot root + legs (+0.02 rad joint noise on the hinge block only)
    data.qpos[0] = spawn[0]
    data.qpos[1] = spawn[1]
    data.qpos[3:7] = yaw_to_quat(yaw0)
    noise = rng.uniform(-JOINT_NOISE_RAD, JOINT_NOISE_RAD, size=box.box_qadr - 7)
    data.qpos[7:box.box_qadr] = data.qpos[7:box.box_qadr] + noise
    # upper body to replay frame 0
    for adr, val in zip(rep.arena_qpos_adrs, rep.pos[0]):
        data.qpos[adr] = float(val)
    # free bodies at the spec ground-truth poses (box on T_pick, cube on T_relay)
    data.qpos[box.box_qadr:box.box_qadr + 3] = sd["box"]["pos"]
    data.qpos[box.box_qadr + 3:box.box_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[box.cube_qadr:box.cube_qadr + 3] = sd["cube"]["pos"]
    data.qpos[box.cube_qadr + 3:box.cube_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    for eq in (box.eq_box_left, box.eq_box_right, box.eq_cube_box, box.eq_cube_right):
        data.eq_active[eq] = 0
    mujoco.mj_forward(model, data)
    rig.obs_processor.reset()
    rig.mgr.set_command(0.0, 0.0, 0.0, ARENA_STAND_H)

    policy, n_zeroed = fresh_policy(rig)
    if trial == 0:
        print(f"  [policy] zeroed {n_zeroed} recurrent state buffer(s)")

    # --- mission + io --------------------------------------------------------
    io = AgileMissionIO(model, data, box, rep, h_stand=ARENA_STAND_H)
    io.upper_row = rep.pos[0]
    h_pick = reach_base_h(sd["T_pick"]["top_h"])
    h_relay = reach_base_h(sd["T_relay"]["top_h"])
    h_store = ARENA_STORE_PLACE_H
    # Cube touch needs the right wrist to reach an 0.08 m cube on the 0.55 m
    # relay top — a DEEPER bend than the relay-table reach (the smoke run got
    # the wrist to 0.39 m at the table-reach h0.65, just past CUBE_TOUCH_D 0.32).
    # Squat ARENA_CUBE_REACH_DROP m lower so the reach-pose wrist drops onto the
    # small cube, clipped to the trackable domain.
    h_cube = max(reach_base_h(sd["T_relay"]["top_h"]) - ARENA_CUBE_REACH_DROP,
                 ARENA_H_MIN)
    if test == "arena_M1":
        seq = mm.build_m1(sd, ARENA_STAND_H, h_pick, h_store,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    else:
        seq = mm.build_m2(sd, ARENA_STAND_H, h_pick, h_relay, h_store, h_cube,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    seq = insert_pillar_detours(seq, (spawn[0], spawn[1]), tuple(sd["pillar"]["pos"]))
    mission = mm.Mission(test, seq, io, ARENA_STAND_H)

    rec = {"t": [], "x": [], "y": [], "z": [], "tilt": [], "phase": []}
    fell = False
    fall_time = fall_phase = fall_reason = None
    harness_error = None
    t_end = 0.0
    wall0 = time.time()
    max_steps = int(round((mm.MISSION_TIMEOUT_S + 5.0) / dt))

    try:
        for step in range(max_steps):
            t = (step + 1) * dt   # mission clock (no separate settle; the
                                  # mission's first nav leg is the start)
            # unified fall flag for THIS tick (before the mission consumes it)
            quat = np.asarray(data.qpos[3:7], dtype=np.float64)
            tilt = quat_tilt(quat)
            base_z = float(data.qpos[2])
            nonfoot_ground, _ = scan_contacts(data, geoms)
            fall_now = tilt > FALL_TILT_RAD or nonfoot_ground
            io.set_fallen(fall_now)

            # mission tick (reads io, fires welds/release/transfer + replay_upper,
            # returns the leg command for THIS policy tick)
            vx_c, vy_c, wz_c, h_c = mission.step(t)
            io.set_command(vx_c, vy_c, wz_c, h_c)
            phase = mission.stages[-1].name if mission.stages else "init"
            rig.mgr.set_command(vx_c, vy_c, wz_c, h_c)

            # --- official sim2mujoco loop order ---
            sim_state = sim.get_state()
            obs = rig.obs_processor.compute(sim_state)
            with torch.no_grad():
                actions = policy(obs)
            rig.obs_processor.set_last_action(actions)
            joint_cmd = rig.act_processor.process(actions)
            joint_cmd = _arena_apply_upper(rig, joint_cmd, io.upper_row, upper_indices)
            for _ in range(sim.decimation):
                sim.step(joint_cmd)
            t_end = t

            if (data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.isfinite(data.qpos).all()):
                harness_error = f"divergence (qacc explosion / NaN qpos) at t={t:.2f}s"
                break

            rec["t"].append(t)
            rec["x"].append(float(data.qpos[0]))
            rec["y"].append(float(data.qpos[1]))
            rec["z"].append(base_z)
            rec["tilt"].append(tilt)
            rec["phase"].append(phase)

            if video is not None and (fall_now or step % VIDEO_EVERY_N_CTRL_STEPS == 0):
                video.capture(data)

            if fall_now:
                fell = True
                fall_time = t
                fall_phase = mission.fall_stage or phase
                fall_reason = "tilt" if tilt > FALL_TILT_RAD else "ground_contact"
                break
            if mission.finished:
                break
    except Exception as exc:  # noqa: BLE001
        harness_error = f"exception: {exc}"

    wall_s = time.time() - wall0
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
        "wall_time_s": round(wall_s, 2),
    }
    metrics.update(_arena_stage_flags(mres["stages"]))

    if os.environ.get("ARENA_TRAJ_DUMP") and rec["t"]:
        for i in range(0, len(rec["t"]), 25):
            sys.stderr.write(
                "TRAJ t=%5.1f xy=(%6.2f,%6.2f) z=%.2f tilt=%.2f phase=%s\n"
                % (rec["t"][i], rec["x"][i], rec["y"][i], rec["z"][i],
                   rec["tilt"][i], rec["phase"][i]))

    success = bool(mres["success"] and not fell and harness_error is None)
    return {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial,
        "seed": seed,
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "success": success,
        "fall": bool(fell),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "fall_reason": fall_reason,
        "harness_error": harness_error,
        "box_mode": "arena_wrist_weld",
        "upper_mode": (
            BOXDEMO_UPPER_MODE if getattr(rep, "is_boxdemo", False)
            else f"psi0_replay_{rep.source}"
        ),
        "metrics": metrics,
        "video": None,  # filled by caller
    }


# ---------------------------------------------------------------------------
# Arena L-series (locomotion sweeps on the ManipArena scene as a fixed
# background). The furniture + free bodies are static obstacles (box/cube rest
# on the tables); the robot runs a plain locomotion profile referencing the
# SceneSpec coordinates. No mission / no grasp. Reuses scan_contacts (pillar/
# furniture collisions) + the shared nav math.
#   L1 walk_sweep   : forward vx sweep along +X lane (arena furniture as scenery)
#   L2 circle       : circle around the scene pillar (radius from spec)
#   L3 goto         : two-stage nav to P_pick_front (calibration vs the scene)
#   L4 vln          : tape following on the arena scene (--tapes)
#   L5 squat_limit  : continuous descent limit while CARRYING the 2 kg box
# ---------------------------------------------------------------------------
ARENA_L1_VX = 0.4               # m/s forward walk for L1 (single hold speed)
# L1 walk-sweep: the pick table sits ~1.4 m ahead on the +X lane, so a long
# forward hold rams it. Hold just long enough to reach steady-state tracking and
# measure the CLEAR stretch right after the ramp (before the table); reaching
# the table early (no fall) is fine — L1 measures walk tracking, not distance.
ARENA_L1_HOLD_S = 4.0
ARENA_L1_MEAS_WIN = (0.5, 3.0)  # [s into hold] window measured for mean_vx
ARENA_L2_VX = 0.4
ARENA_L2_WZ = 0.4
ARENA_L2_LAPS = 2
ARENA_L5_RATE = 0.05            # m/s continuous descent (matches squat_limit)
ARENA_L5_FLOOR_H = 0.10
ARENA_L5_HOLD_S = 3.0


def run_arena_l_trial(rig: Rig, geoms: GeomSets, box: ArenaRig | None,
                      spec: "ms.SceneSpec", test: str, trial: int, seed: int,
                      video: TrialVideo | None, tape: SimpleNamespace | None) -> dict:
    """L-series locomotion rollout on the arena scene. The robot spawns at the
    spec spawn pose; furniture/free-bodies are static scenery. L5 carries the
    2 kg box (welds anchored at the hug pose, like squat_box)."""
    sim = rig.sim
    model = sim.mj_model
    data = sim.mj_data
    dt = sim.dt
    sd = spec.to_dict()
    spawn = sd["spawn"]
    rng = np.random.default_rng(seed)

    # heading: L2 circle tangent; L3/L4 spec yaw; L1/L5 spec yaw
    if test == "arena_L2":
        yaw0 = float(spawn[2])
    else:
        yaw0 = float(spawn[2])

    sim.reset()
    torch.manual_seed(seed)
    data.qpos[0] = spawn[0]
    data.qpos[1] = spawn[1]
    data.qpos[3:7] = yaw_to_quat(yaw0)
    robot_end = box.box_qadr if box is not None else len(data.qpos)
    noise = rng.uniform(-JOINT_NOISE_RAD, JOINT_NOISE_RAD, size=robot_end - 7)
    data.qpos[7:robot_end] = data.qpos[7:robot_end] + noise
    # free bodies (if present) rest on their tables; box welded for L5 carry
    if box is not None:
        data.qpos[box.box_qadr:box.box_qadr + 3] = sd["box"]["pos"]
        data.qpos[box.box_qadr + 3:box.box_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[box.cube_qadr:box.cube_qadr + 3] = sd["cube"]["pos"]
        data.qpos[box.cube_qadr + 3:box.cube_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        for eq in (box.eq_box_left, box.eq_box_right, box.eq_cube_box, box.eq_cube_right):
            data.eq_active[eq] = 0
    else:
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # L5: hug pose + weld the box into the hands (carry from t=0)
    arm_indices = arm_targets = None
    if test == "arena_L5":
        for joint, target in ARENA_HUG_POSE.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            data.qpos[model.jnt_qposadr[jid]] = target
        mujoco.mj_forward(model, data)
        base_yaw = quat_yaw(np.asarray(data.qpos[3:7], dtype=np.float64))
        _arena_snap_box_upright(model, data, box, base_yaw)
        _arena_weld_box_both(model, data, box)
        model.geom_contype[box.box_geom] = 0
        model.geom_conaffinity[box.box_geom] = 0
        if hasattr(model, "body_contype"):
            model.body_contype[box.box_body] = 0
            model.body_conaffinity[box.box_body] = 0
        mujoco.mj_forward(model, data)
        arm_indices = torch.tensor([sim.joint_names.index(j) for j in ARENA_HUG_POSE],
                                   dtype=torch.long)
        arm_targets = torch.tensor(list(ARENA_HUG_POSE.values()),
                                   dtype=torch.float32, device=rig.device)

    rig.obs_processor.reset()
    rig.mgr.set_command(0.0, 0.0, 0.0, ARENA_STAND_H)
    policy, n_zeroed = fresh_policy(rig)
    if trial == 0:
        print(f"  [policy] zeroed {n_zeroed} recurrent state buffer(s)")

    # geometry references from the spec
    pillar_xy = tuple(sd["pillar"]["pos"])
    pillar_r = float(sd["pillar"]["r"])
    # L2 traces a circle IN PLACE from the spawn (vx/wz radius), same as the
    # standalone circle_pillar test; the scene pillar is a fixed obstacle the
    # robot's circle is offset to avoid (pillar collisions are scored). The
    # circle center = spawn + R along the left-normal of the spawn heading
    # (vx>0,wz>0 turns left), and radial error is measured against THAT center.
    circle_r = ARENA_L2_VX / ARENA_L2_WZ   # 1.0 m
    circle_period = 2.0 * math.pi / ARENA_L2_WZ
    circle_center = (spawn[0] - circle_r * math.sin(yaw0),
                     spawn[1] + circle_r * math.cos(yaw0))
    goal_xy = tuple(sd["targets"]["P_pick_front"][:2])
    goal_yaw = float(sd["targets"]["P_pick_front"][2])

    # durations
    if test == "arena_L1":
        cmd_s = WALK_RAMP_S + ARENA_L1_HOLD_S
    elif test == "arena_L2":
        cmd_s = ARENA_L2_LAPS * circle_period
    elif test == "arena_L3":
        cmd_s = NAV_TIMEOUT_S + CAL_TIMEOUT_S
    elif test == "arena_L4":
        cmd_s = tape.duration_s + VLN_END_STAND_S
    else:  # arena_L5
        cmd_s = (ARENA_STAND_H - ARENA_L5_FLOOR_H) / ARENA_L5_RATE + ARENA_L5_HOLD_S
    n_steps = int(round((SETTLE_S + cmd_s) / dt))

    ts, xs, ys, yaws, fwd_vels, vx_cmds, tilts, base_zs, h_cmds = ([] for _ in range(9))
    drift_xys: list[float] = []
    descent_curve: list[list[float]] = []
    start_pose = None
    fell = False
    fall_time = fall_phase = None
    harness_error = None
    collided = False
    collision_time = None
    box_drop_time = None
    closures: dict[int, dict] = {}
    # L3 nav state machine
    nav_stage = "settle"
    nav_t0 = 0.0
    cal_in_tol_t0 = None
    m_nav = {"err_pos_nav": None, "err_yaw_nav": None, "err_pos_cal": None,
             "err_yaw_cal": None, "success_coarse": False, "success_fine": False}

    wall0 = time.time()
    for i in range(n_steps):
        t_cmd = i * dt - SETTLE_S
        data = sim.mj_data
        x, y = float(data.qpos[0]), float(data.qpos[1])
        yaw = quat_yaw(np.asarray(data.qpos[3:7], dtype=np.float64))

        # --- command for this tick ---
        h_c = ARENA_STAND_H
        phase = "settle"
        if t_cmd < 0.0:
            vx_c = vy_c = wz_c = 0.0
        elif test == "arena_L1":
            if t_cmd < WALK_RAMP_S:
                vx_c, phase = ARENA_L1_VX * t_cmd / WALK_RAMP_S, "ramp"
            else:
                vx_c, phase = ARENA_L1_VX, "hold"
            vy_c = wz_c = 0.0
        elif test == "arena_L2":
            vx_c, vy_c, wz_c = ARENA_L2_VX, 0.0, ARENA_L2_WZ
            phase = "circle1" if t_cmd < circle_period else "circle2"
        elif test == "arena_L3":
            dist, ex, ey, eyaw = nav_errors(x, y, yaw, goal_xy, goal_yaw)
            yaw_err = abs(wrap_angle(goal_yaw - yaw))
            if nav_stage == "settle":
                nav_stage, nav_t0 = "nav", t_cmd
            if nav_stage == "nav":
                t_in = t_cmd - nav_t0
                if (dist <= NAV_EXIT_POS_M and abs(eyaw) <= NAV_EXIT_YAW_RAD) or t_in >= NAV_TIMEOUT_S:
                    m_nav["err_pos_nav"], m_nav["err_yaw_nav"] = dist, yaw_err
                    m_nav["success_coarse"] = bool(dist <= NAV_EXIT_POS_M and yaw_err <= NAV_EXIT_YAW_RAD)
                    nav_stage, nav_t0, cal_in_tol_t0 = "cal", t_cmd, None
            if nav_stage == "cal":
                t_in = t_cmd - nav_t0
                in_tol = dist <= CAL_POS_TOL_M and abs(eyaw) <= CAL_YAW_TOL_RAD
                if in_tol and cal_in_tol_t0 is None:
                    cal_in_tol_t0 = t_cmd
                elif not in_tol:
                    cal_in_tol_t0 = None
                converged = cal_in_tol_t0 is not None and (t_cmd - cal_in_tol_t0) >= CAL_HOLD_S
                if converged or t_in >= CAL_TIMEOUT_S:
                    m_nav["err_pos_cal"], m_nav["err_yaw_cal"] = dist, yaw_err
                    m_nav["success_fine"] = bool(converged)
                    nav_stage = "done"
            if nav_stage == "nav":
                vx_c, vy_c, wz_c = nav_cmd_coarse(ex, ey, eyaw, NAV_VX_RANGE[1])
                phase = "nav"
            elif nav_stage == "cal":
                vx_c, vy_c, wz_c = nav_cmd_cal(ex, ey, eyaw)
                phase = "cal"
            else:
                vx_c = vy_c = wz_c = 0.0
                phase = "done"
        elif test == "arena_L4":
            if t_cmd < tape.duration_s:
                vx_c, vy_c, wz_c = vln_zoh_cmd(tape.cmds, t_cmd)
                phase = "tape"
            else:
                vx_c = vy_c = wz_c = 0.0
                phase = "end_stand"
        else:  # arena_L5: continuous descent while carrying the box
            vx_c = vy_c = wz_c = 0.0
            down_s = (ARENA_STAND_H - ARENA_L5_FLOOR_H) / ARENA_L5_RATE
            if t_cmd < down_s:
                h_c, phase = ARENA_STAND_H - ARENA_L5_RATE * t_cmd, "descend"
            else:
                h_c, phase = ARENA_L5_FLOOR_H, "hold_floor"
        rig.mgr.set_command(vx_c, vy_c, wz_c, h_c)

        # --- official sim2mujoco loop order ---
        sim_state = sim.get_state()
        obs = rig.obs_processor.compute(sim_state)
        with torch.no_grad():
            actions = policy(obs)
        rig.obs_processor.set_last_action(actions)
        joint_cmd = rig.act_processor.process(actions)
        if arm_indices is not None:  # L5 carry pose
            position = joint_cmd.position.clone()
            position[arm_indices] = arm_targets
            joint_cmd = rig.A.JointCommand(position=position, kp=joint_cmd.kp, kd=joint_cmd.kd)
        for _ in range(sim.decimation):
            sim.step(joint_cmd)

        data = sim.mj_data
        if (data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                or not np.isfinite(data.qpos).all()):
            harness_error = f"divergence (qacc explosion / NaN qpos) at t={(i + 1) * dt:.2f}s"
            break

        t_meas = (i + 1) * dt - SETTLE_S
        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        yaw = quat_yaw(quat)
        tilt = quat_tilt(quat)
        base_z = float(data.qpos[2])
        vw = np.asarray(data.qvel[:3], dtype=np.float64)
        fwd_vel = float(vw[0] * math.cos(yaw) + vw[1] * math.sin(yaw))
        tilts.append(tilt)

        if t_meas > 0.0:
            ts.append(t_meas)
            xs.append(x := float(data.qpos[0]))
            ys.append(y := float(data.qpos[1]))
            yaws.append(yaw)
            fwd_vels.append(fwd_vel)
            vx_cmds.append(vx_c)
            base_zs.append(base_z)
            h_cmds.append(h_c)
            if start_pose is None:
                start_pose = (x, y, yaw)
            drift = math.hypot(x - start_pose[0], y - start_pose[1])
            drift_xys.append(drift)
            if test == "arena_L5" and (len(ts) - 1) % DESCENT_CURVE_EVERY_N_STEPS == 0:
                descent_curve.append([round(h_c, 4), round(base_z, 4), round(drift, 4)])
            if test == "arena_L2":
                for k in (1, 2):
                    if k not in closures and t_meas >= k * circle_period:
                        closures[k] = {
                            "pos_err": math.hypot(x - start_pose[0], y - start_pose[1]),
                            "yaw_err": abs(wrap_angle(yaw - start_pose[2])),
                        }

        # box drop tracking (L5)
        if box is not None and test == "arena_L5" and box_drop_time is None:
            bd = float(np.linalg.norm(data.xpos[box.box_body] - data.xpos[box.torso_body]))
            if bd >= BOX_KEEP_DIST_M:
                box_drop_time = (i + 1) * dt

        nonfoot_ground, pillar_hit = scan_contacts(data, geoms)
        if pillar_hit and not collided:
            collided = True
            collision_time = (i + 1) * dt
        is_fall = tilt > FALL_TILT_RAD or base_z < (h_c - FALL_HEIGHT_MARGIN) or nonfoot_ground

        if video is not None and (is_fall or i % VIDEO_EVERY_N_CTRL_STEPS == 0):
            video.capture(data)

        if is_fall:
            fell = True
            fall_time = (i + 1) * dt
            fall_phase = phase
            break
        if test == "arena_L3" and nav_stage == "done":
            break

    wall_s = time.time() - wall0
    t_arr = np.asarray(ts)
    metrics: dict = {
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "max_tilt": float(np.max(tilts)) if tilts else None,
        "wall_time_s": round(wall_s, 2),
        "pillar_collision": collided,
        "collision_time": collision_time,
    }

    if test == "arena_L1":
        # measure the clear-lane window right after the ramp (before the robot
        # reaches the pick table); furniture collisions are scored separately.
        w0 = WALK_RAMP_S + ARENA_L1_MEAS_WIN[0]
        w1 = WALK_RAMP_S + ARENA_L1_MEAS_WIN[1]
        win = (t_arr >= w0) & (t_arr <= w1) if len(t_arr) else None
        fwd = np.asarray(fwd_vels)
        mean_vx = float(np.mean(fwd[win])) if win is not None and win.any() else None
        metrics.update({"vx_target": ARENA_L1_VX, "mean_vx_track": mean_vx})
        success = (not fell) and mean_vx is not None and mean_vx >= 0.9 * ARENA_L1_VX
    elif test == "arena_L2":
        radial = (np.abs(np.hypot(np.asarray(xs) - circle_center[0],
                                  np.asarray(ys) - circle_center[1]) - circle_r)
                  if len(t_arr) else None)
        metrics.update({
            "circle_radius": circle_r,
            "radial_err_mean": float(np.mean(radial)) if radial is not None else None,
            "radial_err_max": float(np.max(radial)) if radial is not None else None,
            "closure_pos_err_lap2": closures.get(2, {}).get("pos_err"),
        })
        success = ((not fell) and (not collided)
                   and metrics["radial_err_mean"] is not None
                   and metrics["radial_err_mean"] <= 0.30)
    elif test == "arena_L3":
        metrics.update(m_nav)
        success = bool(m_nav["success_fine"]) and not fell
    elif test == "arena_L4":
        final_pos_err = final_yaw_err = None
        if ts:
            rx, ry, ryaw = float(tape.ref[-1, 1]), float(tape.ref[-1, 2]), float(tape.ref[-1, 3])
            final_pos_err = float(math.hypot(xs[-1] - rx, ys[-1] - ry))
            final_yaw_err = float(abs(wrap_angle(yaws[-1] - ryaw)))
        metrics.update({"tape_id": tape.id, "final_pos_err": final_pos_err,
                        "final_yaw_err": final_yaw_err})
        success = ((not fell) and final_pos_err is not None
                   and final_pos_err <= VLN_FINAL_POS_TOL_M
                   and final_yaw_err <= VLN_FINAL_YAW_TOL_RAD)
    else:  # arena_L5
        stable_zs = base_zs[:-1] if fell else base_zs
        track_sat_h = None
        for hc, bz in zip(h_cmds, base_zs):
            if abs(hc - bz) > SQUAT_LIMIT_TRACK_SAT_M:
                track_sat_h = float(hc)
                break
        metrics.update({
            "descent_curve": descent_curve,
            "fall_h_cmd": float(h_cmds[-1]) if fell and h_cmds else None,
            "fall_base_z": float(base_zs[-1]) if fell and base_zs else None,
            "depth_floor": float(min(stable_zs)) if stable_zs else None,
            "track_sat_h": track_sat_h,
            "max_drift_xy": float(max(drift_xys)) if drift_xys else None,
            "box_kept": box_drop_time is None,
            "box_drop_time": box_drop_time,
        })
        success = (not fell) and box_drop_time is None
    success = bool(success) and harness_error is None

    return {
        "framework": FRAMEWORK,
        "test": test,
        "trial": trial,
        "seed": seed,
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "success": success,
        "fall": bool(fell),
        "fall_time": fall_time,
        "fall_phase": fall_phase,
        "harness_error": harness_error,
        "metrics": metrics,
        "video": None,  # filled by caller
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_NON_AGG_KEYS = {"vx_target", "direction", "pillar_collision", "wall_time_s",
                 "target_height", "ramp_speed", "sweep_kind", "box_land_pos",
                 "squat_fall", "descent_curve", "tape_id", "stages",
                 "scene_spec_hash", "fall_stage", "mission_status"}


def _group_squat_sweep(results: list[dict], key: str) -> dict:
    """T4 summary groups: success/fall rates + achieved_depth/root_drift means
    per target_height or per ramp_speed (spec: answers "which depths fall
    over and how much does the root drift")."""

    def mean_of(grp: list[dict], mk: str):
        xs = [r["metrics"].get(mk) for r in grp]
        xs = [float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return float(np.mean(xs)) if xs else None

    out = {}
    vals = sorted({r["metrics"].get(key) for r in results
                   if r.get("metrics", {}).get(key) is not None})
    for v in vals:
        grp = [r for r in results if r.get("metrics", {}).get(key) == v]
        out[f"{v:.2f}"] = {
            "n": len(grp),
            "success_rate": sum(1 for r in grp if r.get("success")) / len(grp),
            "fall_rate": sum(1 for r in grp if r.get("fall_time") is not None) / len(grp),
            "achieved_depth_mean": mean_of(grp, "achieved_depth"),
            "depth_err_mean": mean_of(grp, "depth_err"),
            "root_drift_hold_mean": mean_of(grp, "root_drift_hold"),
            "root_drift_total_mean": mean_of(grp, "root_drift_total"),
            "box_kept_rate": (
                float(np.mean([r["metrics"]["box_kept"] for r in grp
                               if isinstance(r["metrics"].get("box_kept"), bool)]))
                if any(isinstance(r["metrics"].get("box_kept"), bool) for r in grp) else None
            ),
        }
    return out


def summarize(results: list[dict], test: str, provenance: dict) -> dict:
    n = len(results)
    ok = [r for r in results if r.get("success")]
    falls = [r for r in results if r.get("fall_time") is not None]
    errors = [r for r in results if "error" in r]

    agg: dict = {}
    keys = {k for r in results for k in r.get("metrics", {})} - _NON_AGG_KEYS
    for key in sorted(keys):
        raw = [r.get("metrics", {}).get(key) for r in results]
        bools = [v for v in raw if isinstance(v, bool)]
        vals = [float(v) for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        elif bools:
            agg[key] = {"rate": float(np.mean(bools)), "n": len(bools)}

    summary = {
        "framework": FRAMEWORK,
        "test": test,
        "n_trials": n,
        "success_rate": (len(ok) / n) if n else None,
        "n_falls": len(falls),
        "n_errors": len(errors),
        "n_harness_errors": sum(1 for r in results if r.get("harness_error")),
        "metrics": agg,
        "provenance": provenance,
    }

    if test in BOX_TESTS:
        summary["box_mode"] = BOX_MODE
        drop_key = ("box_kept_carry" if test in ("pipeline_abc", "squat_place_psi0")
                    else "box_kept")
        summary["n_box_drops"] = sum(
            1 for r in results if r.get("metrics", {}).get(drop_key) is False
        )

    modes = sorted({r.get("upper_mode") for r in results if r.get("upper_mode")})
    if modes:
        summary["upper_mode"] = modes[0]

    if test in ("squat_box_psi0", "squat_pick_ground"):
        picks = [r["metrics"].get("pick_success") for r in results
                 if isinstance(r.get("metrics", {}).get("pick_success"), bool)]
        summary["pick_success_rate"] = float(np.mean(picks)) if picks else None

    if test == "vln_follow":
        modes = sorted({r.get("cmd_mode") for r in results if r.get("cmd_mode")})
        summary["cmd_mode"] = modes[0] if modes else None

    if test == "squat_sweep":
        summary["by_target_height"] = _group_squat_sweep(results, "target_height")
        summary["by_ramp_speed"] = _group_squat_sweep(results, "ramp_speed")

    if test in NAV_TESTS:
        for label, key in (("coarse_rate", "success_coarse"), ("fine_rate", "success_fine")):
            vals = [r["metrics"].get(key) for r in results
                    if isinstance(r.get("metrics", {}).get(key), bool)]
            summary[label] = float(np.mean(vals)) if vals else None

    if test in ("pipeline_abc", "squat_place_psi0"):
        places = [r["metrics"].get("box_place_ok") for r in results
                  if isinstance(r.get("metrics", {}).get("box_place_ok"), bool)]
        summary["box_place_rate"] = float(np.mean(places)) if places else None

    if test == "speed_sweep":
        per_speed = {}
        for speed in SWEEP_SPEEDS:
            grp = [r for r in results if r.get("metrics", {}).get("vx_target") == speed]
            if not grp:
                continue
            mvx = [r["metrics"]["mean_vx_last8s"] for r in grp if r["metrics"].get("mean_vx_last8s") is not None]
            per_speed[str(speed)] = {
                "n": len(grp),
                "success_rate": sum(1 for r in grp if r["success"]) / len(grp),
                "mean_vx": float(np.mean(mvx)) if mvx else None,
                "falls": sum(1 for r in grp if r.get("fall_time") is not None),
            }
        stable = [s for s in SWEEP_SPEEDS if per_speed.get(str(s), {}).get("success_rate", 0) >= 0.8]
        summary["per_speed"] = per_speed
        summary["max_stable_speed"] = max(stable) if stable else None

    if test in ("circle_pillar", "circle_pillar_psi0"):
        summary["n_pillar_collisions"] = sum(1 for r in results if r.get("metrics", {}).get("pillar_collision"))

    return summary


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _arena_aggregate(results: list[dict]) -> dict:
    """Scalar/bool aggregation for arena results (mirrors bench_homie.aggregate;
    nested dicts/lists like stages are skipped)."""
    out: dict = {
        "n_trials": len(results),
        "success_rate": (float(np.mean([r["success"] for r in results]))
                         if results else None),
        "fall_count": int(sum(bool(r.get("fall")) for r in results)),
        "harness_error_count": sum(1 for r in results if r.get("harness_error")),
    }
    keys: set = set()
    for r in results:
        keys.update(r.get("metrics", {}).keys())
    stats: dict = {}
    for k in sorted(keys):
        vals = [r["metrics"].get(k) for r in results]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            stats[k] = {"rate": float(np.mean(vals)), "n": len(vals)}
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            arr = np.asarray(vals, dtype=np.float64)
            stats[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                        "min": float(arr.min()), "max": float(arr.max()), "n": len(vals)}
    out["metrics"] = stats
    return out


def run_arena_main(args: argparse.Namespace, repo: Path, out_dir: Path,
                   videos_dir: Path) -> int:
    """ManipArena entry (arena_M1/M2 + arena_L1-L5). Builds the variant scene,
    drives the shared Mission (M-series) or a locomotion sweep (L-series), and
    writes results.jsonl + summary.json + EGL videos. M-series + L5 require the
    box; M-series also requires --upper-replay (agile29 real_ep053)."""
    test = args.test
    if not (0 <= args.variant <= 49):
        raise SystemExit(f"--variant must be in [0, 49] (got {args.variant})")

    replay = None
    if test in ARENA_M_TESTS:
        if args.upper_mode == "box_demo":
            replay = BoxDemoUpperProvider()
            print(f"arena upper: {BOXDEMO_UPPER_MODE} from {replay.path}")
        elif args.upper_replay is None:
            raise SystemExit(f"--upper-replay is required for {test} "
                             "(agile29 real_ep053 npz, contract v1) "
                             "or use --upper-mode box_demo")
        else:
            replay = load_upper_replay(args.upper_replay.expanduser().resolve())
            if replay.embodiment not in ("agile29", "unknown"):
                raise SystemExit(f"upper-replay embodiment '{replay.embodiment}' "
                                 "!= agile29 — wrong npz?")
            if not replay.source.startswith("real"):
                print(f"WARNING: {test} normally uses a real_* replay (ep053), "
                      f"got '{replay.source}'")

    tape_list = None
    if test == "arena_L4":
        if args.tapes is None:
            raise SystemExit("--tapes is required for arena_L4 (vln tapes.json)")
        tape_list = load_vln_tapes(args.tapes.expanduser().resolve())
        print(f"arena_L4 tapes: {len(tape_list)} tape(s) (trial i plays i % N)")

    spec = ms.build_variant(args.variant)
    print(f"arena: variant_seed={spec.variant_seed} H_pick={spec.H_pick:.2f} "
          f"hash={spec.scene_spec_hash[:12]}")

    scene_path = args.mjcf.expanduser().resolve()
    checkpoint = (args.checkpoint or repo / REL_CHECKPOINT).expanduser().resolve()
    config_path = (args.config or repo / REL_CONFIG).expanduser().resolve()
    for path, label in ((checkpoint, "checkpoint"), (config_path, "config"),
                        (scene_path, "mjcf scene")):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    A = import_agile(repo)
    device = torch.device(args.device)
    config = A.load_config(config_path)
    variant_scene = build_arena_scene(scene_path, spec)
    print(f"arena scene: {variant_scene}")
    rig = build_rig(A, config, variant_scene, checkpoint, device, test,
                    args.pd_scale, replay=replay, upper_mode=args.upper_mode)
    # robot 36/35 + box freejoint (7/6) + cube freejoint (7/6) = 50 qpos, 29 actuators
    if rig.sim.mj_model.nq != 50 or rig.sim.mj_model.nu != 29:
        raise SystemExit(f"unexpected arena model dims nq={rig.sim.mj_model.nq} "
                         f"nu={rig.sim.mj_model.nu} (want 50/29)")
    box = configure_arena_welds(rig.sim.mj_model)
    if test in ARENA_M_TESTS:
        if getattr(rig.upper, "is_boxdemo", False):
            rig.upper = prepare_boxdemo_arena_provider(rig.sim.mj_model, rig.upper)
            print(f"arena box_demo: source={rig.upper.source} online_ik=1")
        else:
            rig.upper = prepare_arena_replay(rig.sim.mj_model, rig.upper)
            seg = rig.upper.arena_segments
            print(f"arena psi0: source={rig.upper.source} N={rig.upper.n} "
                  f"({rig.upper.duration_s:.2f}s); segments reach_down={seg['reach_down']} "
                  f"carry={seg['carry']} place={seg['place']}")
    geoms = classify_geoms(rig.sim.mj_model)

    renderer = None
    if args.video != "none":
        renderer = mujoco.Renderer(rig.sim.mj_model, height=VIDEO_H, width=VIDEO_W)

    n = args.trials if args.trials is not None else 1
    jsonl_path = out_dir / f"{args.label}_{test}_results.jsonl"
    summary_path = out_dir / f"{args.label}_{test}_summary.json"
    results: list[dict] = []
    ok_videos = 0

    with open(jsonl_path, "w") as jf:
        for trial in range(n):
            seed = trial + args.seed_offset
            video = None
            if renderer is not None:
                video = TrialVideo(renderer, videos_dir, f"{test}_v{args.variant:02d}",
                                   trial, label=args.label)
            print(f"[{FRAMEWORK}/{test}] v{args.variant:02d} trial {trial:02d} running...",
                  flush=True)
            try:
                if test in ARENA_M_TESTS:
                    res = run_arena_trial(rig, geoms, box, spec, test, trial, seed, video)
                else:
                    tape = tape_list[trial % len(tape_list)] if tape_list else None
                    lbox = box if test == "arena_L5" else None
                    res = run_arena_l_trial(rig, geoms, lbox, spec, test, trial, seed,
                                            video, tape)
            except Exception as exc:  # keep the batch alive
                res = {
                    "framework": FRAMEWORK, "test": test, "trial": trial, "seed": seed,
                    "variant_seed": spec.variant_seed, "H_pick": spec.H_pick,
                    "scene_spec_hash": spec.scene_spec_hash,
                    "success": False, "fall": False, "fall_time": None,
                    "fall_phase": "error", "harness_error": f"{type(exc).__name__}: {exc}",
                    "metrics": {}, "video": None,
                }
                print(f"  ERROR: {exc}")

            if video is not None:
                success = bool(res["success"])
                keep = args.video == "all" or (not success) or ok_videos < VIDEO_MAX_SUCCESS
                saved = video.finish(keep=keep, success=success)
                if saved and success and args.video == "policy":
                    ok_videos += 1
                res["video"] = saved

            results.append(res)
            jf.write(json.dumps(res) + "\n")
            jf.flush()
            mtr = res.get("metrics", {})
            tag = "OK" if res["success"] else "FAIL"
            extra = ""
            if res.get("fall"):
                extra = (f" fall@{res['fall_time']:.2f}s({res.get('fall_phase')},"
                         f"{res.get('fall_reason')})")
            if res.get("harness_error"):
                extra += f" HARNESS_ERROR({res['harness_error']})"
            print(f"[{FRAMEWORK}/{test}] v{args.variant:02d} trial {trial:02d} -> "
                  f"{tag}{extra} (status={mtr.get('mission_status', 'l-series')}, "
                  f"{mtr.get('sim_time_s', mtr.get('wall_time_s', '?'))}s)", flush=True)

    provenance = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "mjcf_scene": str(variant_scene),
        "label": args.label,
        "agile_commit": git_commit(repo),
        "device": str(device),
        "control_dt": CONTROL_DT,
        "stand_height": STAND_HEIGHT,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "scene_spec": spec.to_dict(),
        "box_mode": "arena_wrist_weld",
        "upper_replay": (rig.upper.path if test in ARENA_M_TESTS
                         and not getattr(rig.upper, "is_boxdemo", False) else None),
        "upper_mode": (
            BOXDEMO_UPPER_MODE if test in ARENA_M_TESTS
            and getattr(rig.upper, "is_boxdemo", False)
            else (f"psi0_replay_{rig.upper.source}" if test in ARENA_M_TESTS else None)
        ),
        "vln_tapes": (str(args.tapes) if tape_list is not None else None),
    }
    summary = {
        "framework": FRAMEWORK,
        "test": test,
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "overall": _arena_aggregate(results),
        "provenance": provenance,
    }
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"results -> {jsonl_path}")

    if renderer is not None:
        renderer.close()
    rig.sim.close()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AGILE G1 benchmark harness (unified spec v1)")
    p.add_argument("--test", required=True, choices=TESTS)
    p.add_argument("--trials", type=int, default=None,
                   help="default: 50 (speed_sweep 25; goto_ab 25; pipeline_abc 15; "
                        "squat_limit 10; squat_place_psi0 25; squat_pick_ground 15; "
                        "vln_follow 10); squat_sweep: trials PER GRID POINT (default 5)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--variant", type=int, default=0,
                   help="ManipArena (arena_M1/M2/L1-L5): variant_seed 0-49 "
                        "(BENCHMARK_V2_DESIGN §5; default 0). --trials N re-runs "
                        "the SAME variant N times (default 1).")
    p.add_argument("--upper-replay", type=Path, default=None,
                   help="psi0 / arena_M tests: upper_replay_agile29_*.npz "
                        "(contract v1; 4090:/sda/lizhe/g1bench/psi0_replay/out/)")
    p.add_argument("--upper-mode", choices=("psi0", "box_demo"), default="psi0",
                   help="Upper-body source for replay/arena/carry tests. "
                        "box_demo uses box_demo_2 IK + handoff pose and does "
                        "not require --upper-replay.")
    p.add_argument("--tapes", type=Path, default=None,
                   help="vln_follow / arena_L4 (REQUIRED): tapes.json from "
                        "make_vln_tapes.py (the SAME 10 tapes for all harnesses)")
    p.add_argument("--video", choices=("none", "policy", "all"), default="policy")
    p.add_argument("--agile-repo", type=Path, default=Path(DEFAULT_AGILE_REPO))
    p.add_argument("--mjcf", type=Path, default=Path(DEFAULT_MJCF),
                   help="unitree_mujoco scene_29dof.xml (MUST be the scene file with floor)")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: student .pt inside the repo")
    p.add_argument("--config", type=Path, default=None, help="default: student .yaml inside the repo")
    p.add_argument("--device", default="cpu", help="cpu (default; 2MB LSTM) or cuda")
    p.add_argument("--pd-scale", type=float, default=1.0, help="keep 1.0 for the benchmark (fidelity)")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--label", default="agile",
                   help="Output filename prefix/model label (default: agile; "
                        "use agile-boxdemo for console-separated box_demo runs)")
    # ---- custom single-point render mode (experiment console /api/render):
    p.add_argument("--custom-height", type=float, default=None,
                   help="squat_sweep: render --trials N at this absolute "
                        "pelvis height [m] instead of the full grid")
    p.add_argument("--custom-rate", type=float, default=None,
                   help="squat_sweep: height ramp speed [m/s] for the custom "
                        "point (default %.1f)" % SQUAT_SWEEP_FIXED_RATE)
    p.add_argument("--custom-vx", type=float, default=None,
                   help="walk_speed/speed_sweep: command this vx [m/s]; "
                        "circle_pillar: tangential speed (radius = vx/wz)")
    p.add_argument("--custom-wz", type=float, default=None,
                   help="circle_pillar: yaw rate [rad/s]")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    test = args.test

    # ManipArena tests get a dedicated entry (build the variant scene, drive the
    # Mission / L-series sweep). Routed first so it does not touch the
    # build_scene_variant / command_profile machinery below.
    if test in ARENA_TESTS:
        repo = args.agile_repo.expanduser().resolve()
        out_dir = args.out_dir.expanduser().resolve()
        videos_dir = out_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.video != "none":
            videos_dir.mkdir(parents=True, exist_ok=True)
        return run_arena_main(args, repo, out_dir, videos_dir)

    # custom single-point render mode (experiment console /api/render)
    global CIRCLE_VX, CIRCLE_WZ, CIRCLE_PERIOD_S, CIRCLE_CENTER, CIRCLE_RADIUS
    global PILLAR_GEOM_XML, CUSTOM_VX_REL_THR
    if test in ("circle_pillar", "circle_pillar_psi0"):
        if args.custom_vx is not None:
            if args.custom_vx <= 0:
                raise SystemExit("--custom-vx must be > 0 for circle_pillar")
            CIRCLE_VX = float(args.custom_vx)
        if args.custom_wz is not None:
            if args.custom_wz <= 0:
                raise SystemExit("--custom-wz must be > 0 (sign comes from CCW/CW split)")
            CIRCLE_WZ = float(args.custom_wz)
            CIRCLE_PERIOD_S = 2.0 * math.pi / CIRCLE_WZ
        if args.custom_vx is not None or args.custom_wz is not None:
            # Spec v5: radius = vx/wz. The circle center, the radial-error
            # reference AND the pillar position all follow the new radius
            # (PILLAR_GEOM_XML must be rebound BEFORE build_scene_variant).
            CIRCLE_RADIUS = CIRCLE_VX / CIRCLE_WZ
            CIRCLE_CENTER = (0.0, CIRCLE_RADIUS)
            PILLAR_GEOM_XML = (
                '<geom name="bench_pillar" type="cylinder" size="0.15 0.6" '
                f'pos="{CIRCLE_CENTER[0]} {CIRCLE_CENTER[1]} 0.6" rgba="0.5 0.5 0.5 1"/>'
            )
            print(f"custom circle: vx={CIRCLE_VX:.2f} wz={CIRCLE_WZ:.2f} "
                  f"(radius {CIRCLE_RADIUS:.2f} m, period {CIRCLE_PERIOD_S:.2f} s)")
    elif args.custom_vx is not None and test in ("walk_speed", "speed_sweep"):
        CUSTOM_VX_REL_THR = True

    sweep_plan = None
    if test == "squat_sweep" and (args.custom_height is not None
                                  or args.custom_rate is not None):
        ch = (args.custom_height if args.custom_height is not None
              else SQUAT_SWEEP_FIXED_DEPTH)
        cr = (args.custom_rate if args.custom_rate is not None
              else SQUAT_SWEEP_FIXED_RATE)
        n_custom = args.trials if args.trials is not None else 1
        sweep_plan = [{"target_h": ch, "ramp_speed": cr,
                       "sweep_kind": "custom"} for _ in range(n_custom)]
        trials = len(sweep_plan)
        print(f"custom squat point: H={ch:.2f} m @ {cr:.2f} m/s x{n_custom}")
    elif test == "squat_sweep":
        per_point = args.trials if args.trials is not None else SQUAT_SWEEP_TRIALS_PER_POINT
        sweep_plan = (
            [{"target_h": h, "ramp_speed": SQUAT_SWEEP_FIXED_RATE, "sweep_kind": "depth"}
             for h in SQUAT_SWEEP_DEPTHS for _ in range(per_point)]
            + [{"target_h": SQUAT_SWEEP_FIXED_DEPTH, "ramp_speed": v, "sweep_kind": "speed"}
               for v in SQUAT_SWEEP_RATES for _ in range(per_point)]
        )
        trials = len(sweep_plan)
    elif args.trials is not None:
        trials = args.trials
    else:
        trials = {"speed_sweep": 25, "goto_ab": 25, "pipeline_abc": 15,
                  "squat_limit": SQUAT_LIMIT_TRIALS,
                  "squat_place_psi0": SQUAT_PLACE_TRIALS,
                  "squat_pick_ground": PICK_TRIALS,
                  "vln_follow": VLN_TRIALS}.get(test, 50)

    tapes = None
    if test == "vln_follow":
        if args.tapes is None:
            raise SystemExit("--tapes is required for vln_follow "
                             "(tapes.json from make_vln_tapes.py)")
        tapes = load_vln_tapes(args.tapes.expanduser().resolve())
        print(f"vln tapes: {len(tapes)} tape(s) from {args.tapes} "
              f"(trial i plays tape i % {len(tapes)})")
    elif args.tapes is not None:
        print("WARNING: --tapes ignored (only used by vln_follow)")

    replay = None
    if test in REPLAY_TESTS:
        if args.upper_mode == "box_demo":
            replay = make_boxdemo_scripted_upper(test)
            print(f"upper: {BOXDEMO_UPPER_MODE} scripted sequence from {replay.path}")
        elif args.upper_replay is None:
            raise SystemExit(
                f"--upper-replay is required for {test} "
                "(upper_replay_agile29_*.npz, contract v1) "
                "or use --upper-mode box_demo"
            )
        else:
            replay = load_upper_replay(args.upper_replay.expanduser().resolve())
            if replay.embodiment not in ("agile29", "unknown"):
                raise SystemExit(
                    f"upper-replay embodiment '{replay.embodiment}' != agile29 — wrong npz?"
                )
            want_src = "sim" if test == "squat_box_psi0" else "real"
            if not replay.source.startswith(want_src):
                print(f"WARNING: {test} normally uses a {want_src}_* replay source, "
                      f"got '{replay.source}'")
    elif args.upper_replay is not None:
        print("WARNING: --upper-replay ignored (not a psi0 test)")
    elif args.upper_mode == "box_demo" and test not in CARRY_TESTS:
        print(f"WARNING: --upper-mode box_demo has no effect for {test}")

    repo = args.agile_repo.expanduser().resolve()
    checkpoint = (args.checkpoint or repo / REL_CHECKPOINT).expanduser().resolve()
    config_path = (args.config or repo / REL_CONFIG).expanduser().resolve()
    scene_path = args.mjcf.expanduser().resolve()
    for path, label in ((checkpoint, "checkpoint"), (config_path, "config"), (scene_path, "mjcf scene")):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path} (did you run `git lfs pull` / clone unitree_mujoco?)")
    if checkpoint.suffix == ".onnx":
        raise SystemExit("Do not use .onnx with sim2mujoco (recurrent ONNX has 3 inputs); use the .pt")

    out_dir = args.out_dir.expanduser().resolve()
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.video != "none":
        videos_dir.mkdir(parents=True, exist_ok=True)

    A = import_agile(repo)
    device = torch.device(args.device)
    config = A.load_config(config_path)
    variant = build_scene_variant(scene_path, test)
    print(f"Scene: {variant}")
    rig = build_rig(A, config, variant, checkpoint, device, test, args.pd_scale,
                    replay=replay, upper_mode=args.upper_mode)
    geoms = classify_geoms(rig.sim.mj_model)
    if test in ("circle_pillar", "circle_pillar_psi0") and geoms.pillar < 0:
        raise SystemExit("Pillar geom missing after scene injection — check build_scene_variant().")

    # Custom render values may exceed the manager's default clamps (vx +/-0.5,
    # wz +/-1.0, height [0.4, 0.72]) — widen them so the custom command is
    # delivered verbatim (same philosophy as the OOD walk/squat tests).
    if args.custom_vx is not None and test in ("walk_speed", "speed_sweep",
                                               "circle_pillar", "circle_pillar_psi0"):
        vx_lim = max(2.0, abs(args.custom_vx) + 0.5)
        rig.mgr.linear_x_range = (-vx_lim, vx_lim)
        print(f"  Patched: linear_x_range -> {rig.mgr.linear_x_range} (--custom-vx)")
    if args.custom_wz is not None and test in ("circle_pillar", "circle_pillar_psi0"):
        wz_lim = max(1.0, abs(args.custom_wz) + 0.3)
        rig.mgr.angular_z_range = (-wz_lim, wz_lim)
        print(f"  Patched: angular_z_range -> {rig.mgr.angular_z_range} (--custom-wz)")
    if args.custom_height is not None and test == "squat_sweep":
        h_lo = min(SQUAT_SWEEP_HEIGHT_RANGE[0], args.custom_height - 0.05)
        h_hi = max(SQUAT_SWEEP_HEIGHT_RANGE[1], args.custom_height + 0.05)
        rig.mgr.height_range = (h_lo, h_hi)
        print(f"  Patched: height_range -> {rig.mgr.height_range} (--custom-height)")

    renderer = None
    if args.video != "none":
        # TODO(verify-on-server): needs MUJOCO_GL=egl + working EGL; offscreen
        # framebuffer default (640x480) matches VIDEO_W/H.
        renderer = mujoco.Renderer(rig.sim.mj_model, height=VIDEO_H, width=VIDEO_W)

    jsonl_path = out_dir / f"{args.label}_{test}_results.jsonl"
    summary_path = out_dir / f"{args.label}_{test}_summary.json"

    per_speed = max(1, trials // len(SWEEP_SPEEDS)) if test == "speed_sweep" else None
    n_ccw = (trials + 1) // 2
    # pipeline_abc spec: record the first 3 successes + all failures by default.
    video_max_ok = PIPE_VIDEO_MAX_SUCCESS if test == "pipeline_abc" else VIDEO_MAX_SUCCESS

    results: list[dict] = []
    success_videos_kept = 0

    with open(jsonl_path, "w") as jf:
        for trial in range(trials):
            seed = trial + args.seed_offset
            vx_target = WALK_TARGET_VX
            if args.custom_vx is not None and test in ("walk_speed",
                                                       "speed_sweep"):
                vx_target = float(args.custom_vx)
            elif test == "speed_sweep":
                vx_target = SWEEP_SPEEDS[min(trial // per_speed, len(SWEEP_SPEEDS) - 1)]
            ccw = trial < n_ccw  # circle_pillar(_psi0) only
            sweep_kw = sweep_plan[trial] if sweep_plan is not None else {}

            video = None
            if renderer is not None:
                # NOTE: in `policy` mode every trial is rendered (streamed to a temp
                # file) because failures are only known post-hoc and the spec wants
                # video for ALL failed trials. Use --video none for max speed.
                video = TrialVideo(renderer, videos_dir, test, trial, label=args.label)

            label = f"[{test} t{trial:02d} seed={seed}"
            if test == "speed_sweep":
                label += f" vx={vx_target}"
            if sweep_kw:
                label += f" H={sweep_kw['target_h']} v={sweep_kw['ramp_speed']}"
            if tapes is not None:
                label += f" tape={tapes[trial % len(tapes)].id}"
            label += "]"
            print(f"{label} running...", flush=True)
            try:
                if test in NAV_TESTS:
                    result = run_nav_trial(rig, geoms, test, trial, seed, video)
                elif test == "squat_pick_ground":
                    result = run_pick_trial(rig, geoms, test, trial, seed, video)
                elif test == "vln_follow":
                    result = run_vln_trial(rig, geoms, test, trial, seed, video,
                                           tapes[trial % len(tapes)])
                else:
                    result = run_trial(rig, geoms, test, trial, seed, vx_target, ccw, video,
                                       **sweep_kw)
            except Exception as exc:  # keep the batch alive; count as error
                result = {
                    "framework": FRAMEWORK, "test": test, "trial": trial, "seed": seed,
                    "success": False, "fall_time": None, "fall_phase": "error",
                    "harness_error": f"{type(exc).__name__}: {exc}",
                    "metrics": {}, "video": None, "error": f"{type(exc).__name__}: {exc}",
                }
                if test in BOX_TESTS:
                    result["box_mode"] = BOX_MODE
                if test in REPLAY_TESTS and replay is not None:
                    result["upper_mode"] = (
                        BOXDEMO_UPPER_MODE if getattr(replay, "is_boxdemo", False)
                        else f"psi0_replay_{replay.source}"
                    )
                elif rig.upper_mode:
                    result["upper_mode"] = rig.upper_mode
                print(f"{label} ERROR: {exc}")

            if video is not None:
                success = bool(result["success"])
                keep = args.video == "all" or (not success) or success_videos_kept < video_max_ok
                saved = video.finish(keep=keep, success=success)
                if saved and success and args.video == "policy":
                    success_videos_kept += 1
                result["video"] = saved

            results.append(result)
            jf.write(json.dumps(result) + "\n")
            jf.flush()
            status = "OK " if result["success"] else "FAIL"
            print(f"{label} {status} fall={result['fall_time']} metrics={result['metrics']}", flush=True)

    provenance = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "mjcf_scene": str(variant),
        "label": args.label,
        "agile_commit": git_commit(repo),
        "pd_scale": args.pd_scale,
        "device": str(device),
        "seed_offset": args.seed_offset,
        "control_dt": CONTROL_DT,
        "settle_s": SETTLE_S,
        "stand_height": STAND_HEIGHT,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "note_ood": "AGILE training vx domain is [-0.5, 0.5] m/s; vx>0.5 commands are OOD by design",
        "upper_replay": replay.path if replay is not None else None,
        "upper_mode": (BOXDEMO_UPPER_MODE if getattr(replay, "is_boxdemo", False)
                       else (f"psi0_replay_{replay.source}" if replay is not None
                             else rig.upper_mode)),
        "upper_replay_source": (None if getattr(replay, "is_boxdemo", False)
                                else (replay.source if replay is not None else None)),
        "upper_replay_grasp_close_t": replay.grasp_close_t if replay is not None else None,
        "upper_replay_t_place_s": replay.t_place_s if replay is not None else None,
        "nav_spec": "v3 shared two-stage controller" if test in NAV_TESTS else None,
        "vln_tapes": str(args.tapes) if tapes is not None else None,
        "vln_n_tapes": len(tapes) if tapes is not None else None,
        "vln_wz_semantics": ("native yaw-rate command (no target-yaw integration)"
                             if test == "vln_follow" else None),
        "pick_reach_pose": (dict(ARM_REACH_POSE) if test == "squat_pick_ground" else None),
        "custom_flags": ({k: getattr(args, k)
                          for k in ("custom_height", "custom_rate", "custom_vx", "custom_wz")
                          if getattr(args, k) is not None} or None),
    }
    summary = summarize(results, test, provenance)
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))

    if renderer is not None:
        renderer.close()
    rig.sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
