#!/usr/bin/env python3
"""
FALCON (self-trained) G1 benchmark harness — unified experiment spec v1
(+ box-carry mechanism v2: free box welded to both wrists).

Runs the self-trained sim2real/models/falcon/g1_29dof_trained.onnx policy
(575-d obs history -> 29 whole-body actions, 50 Hz) in MuJoCo, fully headless.
The obs/action pipeline REUSES the deployment class LocoManipPolicy (same
shim approach as sim2real/eval/auto_eval.py: DDS replaced by a MuJoCo
state/command shim, keyboard listener disabled). No DDS, single process.

Key FALCON specifics (read from sim2real/rl_policy/* + config):
  - stand_command flag: 1 = stepping/walk mode, 0 = stance mode (squat uses 0).
  - base_height_command: absolute base height (config DESIRED_BASE_HEIGHT 0.75).
  - obs actor_obs (575) = 5-frame history of 115-d frames; frame is the
    ALPHABETICALLY sorted concat of {actions(29), base_ang_vel(3),
    command_ang_vel(1), command_base_height(1), command_lin_vel(2),
    command_stand(1), command_waist_dofs(3), dof_pos(29), dof_vel(29),
    projected_gravity(3), ref_upper_dof_pos(14)} — all handled by the
    deployment class itself, we only feed robot_state_data + commands.
  - upper body: residual_upper_body_action=True; ref_upper_dof_pos (14, abs
    joint targets) enters the obs AND is added as residual to the action.
    squat_box sets it to a box-hug pose; other tests keep arm defaults.

Runtime environment (4090-lab):
    conda env fcreal (/hhd2/ljk/miniconda3/envs/fcreal), repo /hhd2/ljk/FALCON
    cd /hhd2/ljk/FALCON/sim2real
    MUJOCO_GL=egl /hhd2/ljk/miniconda3/envs/fcreal/bin/python \
        /path/to/bench_falcon.py --test walk_speed --trials 50 \
        --out-dir /path/to/results/falcon/walk_speed --video policy
    # psi0: --upper-replay /hhd2/ljk/g1bench/psi0_replay/\
    #       upper_replay_falcon29_{sim_ep035|real_ep053}.npz
    #       (squat_box_psi0 -> sim_ep035, walk/circle _psi0 -> real_ep053)

Tests (unified spec v1):
    walk_speed    50 trials, settle 2s -> vx ramp 2s -> 1.0 m/s hold 10s,
                  success: no fall and mean_vx_last8s >= 0.9
    squat_box     50 trials, 20 x 5s cycles 0.75 <-> 0.45 m, 2 kg free box
                  welded to both wrists (box_mode=wrist_weld_v2),
                  success: no fall and 20 cycles (box_kept reported separately)
    circle_pillar 50 trials (25 ccw + 25 cw), vx=0.4 wz=+/-0.4 around r=1 m
                  pillar, success: 2 laps, no fall/collision, radial err <=0.15
    speed_sweep   {0.4,0.6,0.8,1.0,1.2} x 5 trials, success thr 0.9*v

Tests (spec v3 additions, all psi0-free):
    squat_sweep   depth x speed grid with the v2 carried box (weld from t=0,
                  2 s settle at stand height): 8 depths {0.65..0.30} at
                  0.2 m/s + 4 ramp speeds {0.1,0.2,0.4,0.8} at 0.45 m,
                  --trials N per grid point (default 5). Per trial:
                  settle 2 s -> ramp down to H -> hold 3 s -> ramp up ->
                  1 s. Metrics: achieved_depth (hold-mean base_z),
                  depth_err, root_drift_hold, root_drift_total, box_kept.
                  Summary grouped by target_height and by ramp_speed.
    goto_ab       A(origin, yaw noise +-0.3 rad by seed) -> B(3.0, 1.0),
                  theta_B=+90 deg, no box. Shared two-phase nav controller:
                  NAV coarse P-law (vx=clip(1.0*ex,0,0.6),
                  vy=clip(1.0*ey,+-0.3), wz=clip(1.5*eyaw,+-0.6); heading
                  goal = bearing to B while >0.5 m, else theta_B; exit
                  0.30 m & 15 deg or 30 s timeout) then CAL small-command
                  calibration (same P-law, cmds clipped to |0.10|, NO
                  deadband compensation; fine = 0.05 m & 5 deg held 1 s,
                  timeout 15 s). FALCON adaptation: stand=1 during NAV/CAL
                  (stepping), stand=0 once CAL ends. trial success =
                  success_fine & no fall (success_coarse reported too).
    pipeline_abc  full carry pipeline: box welded at t=0 (standing hug),
                  A(origin, yaw 0) -> C(0.0, -2.5), theta_C=-90 deg (right
                  turn then 2.5 m straight; FALCON's weak cw side). NAV
                  (vx_max 0.5, conservative with box) -> CAL -> squat to
                  0.45 @0.2 m/s -> bottom hold 0.5 s -> release both welds
                  AND enable box collision (geom contype/conaffinity
                  runtime-writable) -> hold 1 s -> rise -> stand 1 s.
                  box_place_ok = box still (|v|<0.05), upright (<30 deg)
                  and within 0.8 m of the robot at the end. success =
                  success_coarse & no fall & box_place_ok (fine convergence
                  reported, not gated). Videos: first 3 OK + all failed.

Tests (spec v4 additions):
    squat_limit   T7 depth-limit calibration with the 2 kg carried box
                  (wrist_weld_v2, welded from t=0): 2 s settle at stand
                  height -> height command descends CONTINUOUSLY at
                  0.05 m/s down to 0.10 m (far below the feasible domain,
                  deliberately NOT clipped — verified the FALCON command
                  layer has no clip: base_height_command enters the obs
                  verbatim, only disabled joystick handlers write it) ->
                  hold 3 s at 0.10 -> end (no rise). Fall ends the trial.
                  metrics: descent_curve [[h_cmd, base_z, drift_xy], ...]
                  @10 Hz, fall_h_cmd/fall_base_z, depth_floor (min stable
                  base_z, tilt <= 0.3 rad), track_sat_h (|cmd-actual| >
                  5 cm first time), drift5_h/drift20_h, max_drift_xy.
                  success = no fall (most models saturate, some fall —
                  that boundary is exactly what is being calibrated).
                  --trials N (default 10); --video all recommended.
    squat_place_psi0
                  T8 Psi0-coordinated reach-forward box place. Requires
                  --upper-replay real_ep053 (22.3 s, arms +-1.3 rad,
                  height 0.77->0.46->0.75). 2 kg box welded from t=0
                  (wrist_weld_v2, standing carry) -> 2 s settle -> single
                  playback of the ep053 arm+waist stream while
                  base_height_command follows its height_cmd; at t_place
                  = argmin(height_cmd) both welds release and box
                  collision enables (bit-2 + body mirror) so the box
                  lands; playback continues through the rise; 1 s final
                  stand (arms frozen at the last replay frame).
                  metrics: height_rmse_descent/hold/rise,
                  root_drift_place (descent+hold root XY drift),
                  box_place_ok, box_land_dx (box landing point forward of
                  the foot front edge at release, along release heading),
                  end_stand_ok. success = no fall AND box_place_ok.
                  --trials N (default 25). stand=0 throughout.

Tests (spec v5 additions, all psi0-free, no custom flags for FALCON):
    squat_pick_ground
                  T9 ground pick: 2 kg bench box (0.35x0.25x0.25) standing
                  UPRIGHT on the floor (long axis vertical, top at 0.35 m,
                  bit-2 ground-collision scheme: box contype/conaffinity 2,
                  floor ORs in bit 2, robot keeps bit 1 -> box collides
                  with the floor, never the robot) 0.45 m in front of the
                  robot; robot starts empty-handed, arms at the DEFAULT
                  hanging pose. 2 s settle -> 1 s arm blend to the low
                  forward-reach cradle pose PICK_ARM_POSE (sh_pitch -0.6,
                  sh_roll +/-0.2, elbow 0.8 rad — wrists low + on both
                  sides of the box) -> height cmd ramps 0.2 m/s down to
                  0.25 m (UNIFIED command, each model saturates at its own
                  depth limit; R4 calibration FALCON depth_floor ~0.213) ->
                  bottom hold; when BOTH wrist-to-box-surface dists
                  < 0.30 m the magnetic grasp fires (anchor both welds in
                  place, same psi0 pick mechanism) -> hold 0.5 s -> ramp
                  back to stand height with the 2 kg load -> stand 1 s.
                  5 s at the bottom without a grasp -> pick_failed, still
                  rises. stand=0 (stance mode) throughout. metrics:
                  pick_success, min_root_z, grasp_dist_l/r (at grasp or
                  timeout), root_drift (bottom+grasp XY), box_lift_height,
                  stand_ok. success = pick AND no fall AND stand_ok.
                  --trials N (default 15).
    vln_follow    T10 VLN small-command stream following. Requires
                  --tapes tapes.json (make_vln_tapes.py output, SHARED
                  verbatim across all four harnesses): per tape 30 s of
                  zero-order-hold [t,vx,vy,wz] breakpoints (update jitter
                  0.4-1.2 s, |vx|<=0.35 |vy|<=0.2 |wz|<=0.30, two 1-2 s
                  full stops, several small-command stretches) plus the
                  ideal error-free integral ref_xy_yaw at 1 Hz. Trial i
                  plays tape i mod n_tapes. 2 s settle -> tape (ZOH) -> 1 s
                  stand. FALCON adaptation (mode semantics): stand=1
                  (stepping) while following, stand=0 (stance) during the
                  full-stop segments and the final stand; wz is consumed
                  natively (no target-yaw integration — that note is for
                  AMO). Robot poses are measured in the tape-start frame
                  (pose at settle end) since the ref integrates from the
                  identity pose. metrics: final_pos_err/final_yaw_err vs
                  the ref end pose, mean_track_err (1 Hz position error
                  mean), small_cmd_response (mean v_fwd/cmd_vx over the
                  |vx| in [0.05,0.15] ticks), stop_settle (mean residual
                  XY displacement per full-stop segment), fall. success =
                  no fall AND final_pos_err <= 0.30 m AND final_yaw_err
                  <= 15 deg. --trials N (default 10).

psi0 replay variants (upper-body replay contract v1, --upper-replay NPZ from
4090:/hhd2/ljk/g1bench/psi0_replay/upper_replay_falcon29_*.npz; fields
t/upper_names/upper_pos/height_cmd/grasp_close_t/carry_pose/meta_json,
dt=0.02):
    walk_speed_psi0    walk_speed profile + upper body LOOPS the real_ep053
                       arm+waist stream (height_cmd ignored, lower body stays
                       at stand height); cracker_box (0.411 kg) welded to both
                       wrists from t=0 (v2 dual-weld mechanism). success =
                       walk_speed criteria AND box_kept.
    squat_box_psi0     table-top pick + carry-squat. Scene: teatable (top
                       0.40 m) + cracker box (0.411 kg) standing on it, pelvis
                       ~0.29 m from the near table edge. 2 s settle -> replay
                       sim_ep035 upper stream while base_height_command
                       follows its smoothed height_cmd VERBATIM (0.45 is
                       likely outside FALCON's verified ~0.55 domain — sent
                       anyway, recorded as-is); at grasp_close_t welds
                       activate IFF both wrists are < 0.12 m from the box
                       surface (else pick_failed); afterwards arms freeze at
                       carry_pose and the standard 20 x 5 s squat cycles run.
                       success = pick_success AND 20/20 no-fall AND box_kept.
                       New metrics: pick_success, box_lift_height. Collision
                       masks (unified psi0 spec): box bit 2, table bits 1|2,
                       floor ORs in bit 2, robot keeps bit 1 -> the box
                       collides with table/floor but never with the robot.
    circle_pillar_psi0 circle_pillar profile + looped real_ep053 upper
                       stream; cracker_box welded from t=0. success =
                       circle_pillar criteria AND box_kept.
    Replay mapping (verified on 4090 FALCON checkout): the falcon29 npz has
    K=17 columns (14 arm joints in dof_names_upper_body order + 3 waist);
    arm14 -> policy.ref_upper_dof_pos (residual upper path, obs + action),
    waist3 -> policy.waist_dofs_command [yaw, roll, pitch].
    stand flag: replay segment + squat cycles use stand=0 (stance mode).
    JSONL results gain "upper_mode": "psi0_replay_<source>".

Unified fall detection: tilt > 0.9 rad OR base_z < height_cmd - 0.2 OR
non-foot ground contact. Outputs: results.jsonl + summary.json + videos
(policy: first 5 ok + all failed, 640x480@30).
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

# Must be set before importing mujoco for headless EGL rendering (Linux).
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

# ManipArena v2 (BENCHMARK_V2_DESIGN.md). Shared scene generator + mission
# state machine; bench_homie.py is the reference adapter (§7.3) and this file
# mirrors it in FALCON idioms (deployment-policy pipeline + stand flag +
# ref_upper_dof_pos / waist_dofs_command upper path). Imported from the same
# scripts/ dir so the four harnesses share one source of truth.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manip_scene as ms          # noqa: E402
import manip_mission as mm        # noqa: E402

# ---------------------------------------------------------------------------
# Constants — FALCON pipeline (sources: sim2real/config/g1/g1_29dof_falcon.yaml,
# sim2real/eval/auto_eval.py, sim2real/rl_policy/loco_manip/loco_manip.py)
# ---------------------------------------------------------------------------
FRAMEWORK = "falcon"
# --label overrides the framework name written into JSONL / summary / video
# filenames / captions (default 'falcon'). Set once in main() before any
# trial; read-only afterwards. ljk-falcon runs the same 575->29 architecture
# with the v4 weight-only ONNX and pass --label ljk-falcon to tag its output.
LABEL = "falcon"
CONFIG_REL = "sim2real/config/g1/g1_29dof_falcon.yaml"
ONNX_REL = "sim2real/models/falcon/g1_29dof_trained.onnx"

N_JOINTS = 29
RL_RATE = 50              # Hz, deployment rate (auto_eval.py)
POLICY_ACTION_SCALE = 0.25

SQUAT_HEIGHT = 0.45       # unified spec target. NOTE: FALCON training domain
                          # verified down to ~0.55 only; 0.45 may be OUT OF
                          # DOMAIN — sent anyway and recorded as-is.
INIT_BASE_Z = 0.76        # spawn slightly above stand height, settle handles it
JOINT_NOISE = 0.02        # unified spec: +/-0.02 rad init noise

# Fall detection (unified spec)
TILT_FALL_GXY = math.sin(0.9)   # |projected gravity xy| > sin(0.9 rad)
HEIGHT_FALL_MARGIN = 0.2        # base_z < height_cmd - 0.2
# Floor contact is made by the 8 dummy foot-corner sphere bodies (the
# ankle_roll mesh geoms are contype=0 visual-only in this MJCF).
FOOT_BODY_NAMES = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "dummy_lf_1", "dummy_lf_2", "dummy_lf_3", "dummy_lf_4",
    "dummy_rf_1", "dummy_rf_2", "dummy_rf_3", "dummy_rf_4",
)

# Test timing (unified spec)
SETTLE_S = 2.0
WALK_RAMP_S = 2.0
WALK_HOLD_S = 10.0
WALK_LAST_WINDOW_S = 8.0
SQUAT_CYCLES = 20
SQUAT_CYCLE_S = 5.0       # 1.5 down + 1.0 hold + 1.5 up + 1.0 hold
CIRCLE_VX = 0.4
CIRCLE_WZ = 0.4
CIRCLE_RAMP_S = 1.0
CIRCLE_LAPS = 2
CIRCLE_MARGIN_S = 8.0
SWEEP_SPEEDS = (0.4, 0.6, 0.8, 1.0, 1.2)

# --- v3 T4 squat_sweep grid (spec v3) --------------------------------------
SQUAT_SWEEP_HEIGHTS = (0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)
SQUAT_SWEEP_RAMP = 0.2            # m/s, fixed ramp rate for the depth scan
SQUAT_SWEEP_SPEEDS = (0.1, 0.2, 0.4, 0.8)   # m/s ramp rates, speed scan
SQUAT_SWEEP_DEPTH = 0.45          # fixed depth for the speed scan
SQUAT_SWEEP_HOLD_S = 3.0
SQUAT_SWEEP_END_S = 1.0
SQUAT_SWEEP_TRIALS_PER_POINT = 5

# --- v3 shared two-phase navigation controller (spec v3, T5/T6) ------------
# The MATH below is the unified spec, identical across all four harnesses.
NAV_SWITCH_DIST_M = 0.5           # >0.5 m: head toward goal; <=: final yaw
NAV_KP_LIN = 1.0
NAV_KP_YAW = 1.5
NAV_VX_MAX = 0.6                  # NAV vx clip upper bound (lower bound 0)
NAV_VY_MAX = 0.3
NAV_WZ_MAX = 0.6
NAV_POS_TOL_M = 0.30
NAV_YAW_TOL_RAD = math.radians(15.0)
NAV_TIMEOUT_S = 30.0
CAL_LIN_MAX = 0.10                # CAL: |vx|,|vy| <= 0.10 m/s
CAL_WZ_MAX = 0.10                 # CAL: |wz| <= 0.10 rad/s
CAL_POS_TOL_M = 0.05
CAL_YAW_TOL_RAD = math.radians(5.0)
CAL_HOLD_S = 1.0                  # fine tolerance must hold this long
CAL_TIMEOUT_S = 15.0
NAV_END_STAND_S = 1.0             # post-CAL stand (stand=0) before trial end

# T5 goto_ab geometry
GOTO_B_XY = (3.0, 1.0)
GOTO_B_YAW = math.pi / 2.0        # +90 deg
GOTO_YAW_NOISE = 0.3              # +-0.3 rad initial yaw noise (seeded)
GOTO_TRIALS_DEFAULT = 25

# T6 pipeline_abc geometry / squat-place profile
PIPE_C_XY = (0.0, -2.5)
PIPE_C_YAW = -math.pi / 2.0       # -90 deg (right turn — FALCON cw weak side)
PIPE_NAV_VX_MAX = 0.5             # conservative while carrying the box
PIPE_SQUAT_HEIGHT = 0.45
PIPE_SQUAT_SPEED = 0.2            # m/s ramp
PIPE_BOTTOM_HOLD_S = 0.5          # bottom hold before releasing the box
PIPE_PLACE_HOLD_S = 1.0           # hold after release (box lands)
PIPE_END_STAND_S = 1.0
PIPE_TRIALS_DEFAULT = 15
PIPE_SQUAT_PHASES = ("squat_down", "squat_hold", "place_hold", "squat_up",
                     "end_stand")
BOX_PLACE_MAX_SPEED = 0.05        # m/s, box considered at rest
BOX_PLACE_UPRIGHT_RAD = math.radians(30.0)
BOX_PLACE_MAX_DIST_M = 0.8        # box-to-robot distance at trial end

# --- v4 T7 squat_limit (depth-limit calibration, carried 2 kg box) ---------
SQUAT_LIMIT_RATE = 0.05           # m/s, continuous height-command descent
SQUAT_LIMIT_FLOOR_H = 0.10        # final command height — deliberately far
                                  # below the feasible domain, NOT clipped
                                  # (FALCON command layer has no clip; the
                                  # obs takes base_height_command verbatim)
SQUAT_LIMIT_HOLD_S = 3.0          # hold at 0.10 after the ramp (no rise)
SQUAT_LIMIT_TRIALS_DEFAULT = 10
DESCENT_CURVE_DT_S = 0.1          # descent_curve downsample (10 Hz)
TRACK_SAT_ERR_M = 0.05            # |cmd - actual| threshold -> track_sat_h
DRIFT5_M, DRIFT20_M = 0.05, 0.20  # drift thresholds -> drift5_h / drift20_h
STABLE_TILT_RAD = 0.3             # "stable" gate for depth_floor / end stand

# --- v4 T8 squat_place_psi0 (Psi0 reach-forward place, real_ep053) ----------
PLACE_HOLD_EPS_M = 0.02           # height_cmd <= min + eps -> "hold" segment
PLACE_END_STAND_S = 1.0           # post-replay final stand
PLACE_TRIALS_DEFAULT = 25

# --- v5 T9 squat_pick_ground (ground pick: deep squat + magnetic grasp) -----
PICK_BOX_DIST_M = 0.45            # box center this far in front of the robot
PICK_SQUAT_HEIGHT = 0.25          # unified height command floor (unclipped;
                                  # each model saturates at its own limit)
PICK_RAMP_SPEED = 0.2             # m/s height ramp down AND back up
PICK_ARM_BLEND_S = 1.0            # settle -> cradle-pose arm blend duration
PICK_BOTTOM_TIMEOUT_S = 5.0       # grasp window at the bottom -> pick_failed
PICK_GRASP_HOLD_S = 0.5           # hold after the grasp before rising
PICK_END_STAND_S = 1.0
PICK_GRASP_DIST_M = 0.30          # both wrists closer than this to the box
PICK_TRIALS_DEFAULT = 15
# Low forward-reach cradle pose, 14-d in dof_names_upper_body order
# [L sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw, R mirror]:
# shoulder pitch -0.6 rad (forward swing), roll +/-0.2 (hands at the box
# sides), elbow 0.8 — wrists as low and forward as the FALCON arm reaches
# without folding into the legs at the squat bottom (NOTES_FALCON.md v5).
PICK_ARM_POSE = np.array([-0.6,  0.2, 0.0, 0.8, 0.0, 0.0, 0.0,
                          -0.6, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
                         dtype=np.float64)
# GROUND_BOX_XML (same 2 kg bench box, bit-2 ground-collision scheme) is
# defined below BOX_XML, next to the other scene XML snippets.

# --- v5 T10 vln_follow (VLN small-command stream following) -----------------
VLN_TRIALS_DEFAULT = 10
VLN_END_STAND_S = 1.0
VLN_FINAL_POS_TOL_M = 0.30
VLN_FINAL_YAW_TOL_RAD = math.radians(15.0)
VLN_SMALL_VX_LO = 0.05            # small_cmd_response band |vx| in [lo, hi]
VLN_SMALL_VX_HI = 0.15

BOX_TESTS = ("squat_box", "squat_sweep", "pipeline_abc", "squat_limit")
NAV_TESTS = ("goto_ab", "pipeline_abc")
PSI0_TESTS = ("walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
              "squat_place_psi0")
# psi0 tests that carry the small YCB cracker box (squat_place_psi0 instead
# carries the 2 kg bench box, v2 mechanism, and releases it at t_place).
PSI0_CRACKER_TESTS = ("walk_speed_psi0", "squat_box_psi0",
                      "circle_pillar_psi0")
PSI0_WELD_T0_TESTS = ("walk_speed_psi0", "circle_pillar_psi0",
                      "squat_place_psi0")
# every test whose model carries the bench_box body + the two wrist welds
# (v5: squat_pick_ground grasps the ground box with the same dual welds)
WELDED_BOX_TESTS = BOX_TESTS + PSI0_TESTS + ("squat_pick_ground",)

# --- ManipArena v2 (BENCHMARK_V2_DESIGN §2-§4): unified-scene M-series ------
# FALCON mirror of bench_homie.py's arena adapter. The shared scene/mission
# code is identical; what differs here is the FALCON command interface
# (deployment policy + stand flag + ref_upper_dof_pos/waist_dofs_command upper
# path + absolute base_height_command, vs HOMIE's raw onnx + direct upper PD).
ARENA_TESTS = ("arena_M1", "arena_M2")
# FALCON arm-end body names that ms.WRIST_L/WRIST_R placeholders sed-replace
# to (the wrist<->box welds anchor on the rubber hands, like every other
# FALCON box test which welds left/right_rubber_hand).
ARENA_WRIST_L = "left_rubber_hand"
ARENA_WRIST_R = "right_rubber_hand"
# Box/cube half-extents come from the SceneSpec (BOX_FULL/CUBE_FULL); used by
# box_surface_dist so the grasp gate measures the ARENA box dims.
ARENA_BOX_HALF = tuple(v / 2.0 for v in ms.BOX_FULL)
ARENA_CUBE_HALF = tuple(v / 2.0 for v in ms.CUBE_FULL)
# Squat-height command mapping (FALCON ABSOLUTE base_height_command, same
# semantics as HOMIE: lower surface -> lower base, fixed reach offset, clipped
# to the FALCON-trackable domain). FALCON's verified squat domain is shallower
# than HOMIE's (R4 depth_floor ~0.213, but tracking saturates well above that);
# the H_pick sweep deliberately probes where the low tables become unreachable.
ARENA_STAND_H = 0.75               # FALCON DESIRED_BASE_HEIGHT stand/carry
ARENA_H_MIN = 0.30                 # FALCON squat command floor (shallower than
                                   # HOMIE's 0.24: FALCON tracks 0.45 poorly
                                   # and 0.30 is near its usable squat command)
ARENA_REACH_OFFSET = 0.10          # base above the surface for a tabletop reach
ARENA_STORE_FLOOR_H = ms.Z_STORE_RIM_H           # store-pad floor ~0.12 m
# Store placement squat kept SHALLOW (release_box levels + lowers the box onto
# the pad regardless of squat depth, so a deep store squat is unnecessary and
# topple-prone under the forward-slung 2 kg box). Mirrors HOMIE ARENA_STORE_PLACE_H.
ARENA_STORE_PLACE_H = 0.58
# Forward offset (m, heading) the box is set down ahead of the pelvis at
# release. Matched to manip_mission.PLACE_STAND_FWD_M (0.35) so a robot
# standing that far in front of the zone, facing it, drops the box on centre.
ARENA_RELEASE_FWD_M = 0.35
# Max radius (m) the set-down box may land from the placement target centre.
ARENA_PLACE_CLAMP_R = 0.12
ARENA_VX_CARRY = 0.4               # m/s coarse forward clip while carrying
                                   # (FALCON持箱 vx<=0.4 — see physics note d)
ARENA_SQUAT_RATE = 0.2            # m/s base-height ramp
ARENA_CARRY_H = ARENA_STAND_H     # carry at full stand height (lower carry
                                  # stance hurt stability under the 2 kg load)
# Pillar avoidance: if a nav leg's straight segment passes within this of the
# pillar centre, insert one lateral side-offset waypoint to skirt it.
ARENA_PILLAR_CLEAR_M = 0.4
ARENA_PILLAR_SIDE_M = 0.7

# Video
VIDEO_W, VIDEO_H = 640, 480
VIDEO_FPS = 30
MAX_OK_VIDEOS = 5
PIPE_MAX_OK_VIDEOS = 3            # T6: first 3 OK + all failed (spec v3)

# --- Box-carry v2: free body held between the wrists via two soft welds ----
BOX_MODE = "wrist_weld_v2"
BOX_SIZE = (0.175, 0.125, 0.125)   # half extents: 0.35(y across hands) x 0.25 x 0.25
BOX_MASS = 2.0
BOX_KEEP_DIST = 0.6                # box_kept: |box - torso| < 0.6 m throughout
WELD_SOLREF = "0.02 1"             # soft weld so it does not fight the arm PD

# Box-hug arm pose, 14-d in dof_names_upper_body order:
# [L sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw, R mirror].
# Targets wrists ~0.35-0.42 m apart in front of the chest; the soft welds
# absorb the residual gap. Sag/oscillation under the 2 kg load is REAL
# physics (load goes through the arm PD) and is recorded as-is.
HUG_ARM_POSE = np.array([-0.4,  0.15, 0.0, 1.2, 0.0, 0.0, 0.0,
                         -0.4, -0.15, 0.0, 1.2, 0.0, 0.0, 0.0],
                        dtype=np.float64)

BOX_XML = (
    '<body name="bench_box" pos="0.3 0 0.9">'
    '<freejoint name="bench_box_joint"/>'
    '<geom name="bench_box_geom" type="box" size="%.3f %.3f %.3f" '
    'mass="%.1f" rgba="1 0.85 0.1 0.45" contype="0" conaffinity="0"/>'
    "</body>" % (BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], BOX_MASS)
)
WELD_XML = (
    "<equality>"
    '<weld name="weld_box_lh" body1="left_rubber_hand" body2="bench_box" '
    'anchor="0 0 0" solref="%s" active="false"/>'
    '<weld name="weld_box_rh" body1="right_rubber_hand" body2="bench_box" '
    'anchor="0 0 0" solref="%s" active="false"/>'
    "</equality>" % (WELD_SOLREF, WELD_SOLREF)
)
PILLAR_XML = (
    '<body name="pillar" pos="0 0 0">'
    '<geom name="pillar_geom" type="cylinder" size="0.15 0.6" pos="0 0 0.6" '
    'rgba="0.5 0.5 0.5 1"/>'
    "</body>"
)
# v5 T9: same 2 kg bench box but compiled with bit 2 (ground-collision
# scheme): collides with the floor (which ORs in bit 2 at startup), never
# with the robot (bit 1). Spawn pose is overwritten by place_box_on_ground().
GROUND_BOX_XML = (
    '<body name="bench_box" pos="0.45 0 0.2">'
    '<freejoint name="bench_box_joint"/>'
    '<geom name="bench_box_geom" type="box" size="%.3f %.3f %.3f" '
    'mass="%.1f" rgba="1 0.85 0.1 0.9" contype="2" conaffinity="2"/>'
    "</body>" % (BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2], BOX_MASS)
)

# --- psi0 upper-body replay (contract v1; psi0_replay/make_replay.py) -------
# npz fields: t (N,), upper_names (17,), upper_pos (N,17), height_cmd (N,),
# grasp_close_t (scalar), carry_pose (17,), meta_json. falcon29 columns =
# arm14 (dof_names_upper_body order, verified vs g1_29dof_falcon.yaml) +
# waist3 [yaw, roll, pitch]; the mapping below is POSITIONAL, so the names
# are checked verbatim against PSI0_UPPER_NAMES at load time.
PSI0_REPLAY_FIELDS = ("t", "upper_names", "upper_pos", "height_cmd",
                      "grasp_close_t", "carry_pose", "meta_json")
PSI0_UPPER_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
)
PSI0_GRASP_DIST_M = 0.30   # both wrists closer than this to the box surface
PSI0_BOX_HALF = (0.036, 0.082, 0.1065)   # cracker_box (YCB 003) half extents
PSI0_BOX_MASS_KG = 0.411
# BendPick scene (psi0_replay/scene_info.json, ep035), shifted +0.6185 m in x
# so the robot pelvis spawns at the origin. Pelvis ends up ~0.29 m from the
# table's near edge (x=0.2935), matching the source scene.
PSI0_TABLE_HALF = (0.625, 0.395, 0.05)
PSI0_TABLE_CENTER = (0.9185, 0.0, 0.35)  # top face at z=0.40
PSI0_BOX_START_XY = (0.2985, -0.0488)    # ep035 target pose, shifted
PSI0_BOX_START_Z = 0.40 + PSI0_BOX_HALF[2] + 0.0005  # upright on the tabletop
PSI0_BOX_START_QUAT = (0.997384, 0.0, 0.0, -0.072316)  # ep035 yaw ~ -0.145 rad
# Collision masks (unified psi0 spec): robot keeps bit 1, box gets bit 2, the
# table gets bits 1|2 and the floor gets bit 2 OR-ed in at runtime -> the box
# collides with table/floor but never with the robot. Same bench_box_* names
# as BOX_XML so lookup_ids()/weld helpers work unchanged.
PSI0_BOX_XML = (
    '<body name="bench_box" pos="0.3 0 0.9">'
    '<freejoint name="bench_box_joint"/>'
    '<geom name="bench_box_geom" type="box" size="%.4f %.4f %.4f" '
    'mass="%.3f" rgba="0.85 0.3 0.15 0.8" contype="2" conaffinity="2"/>'
    "</body>" % (PSI0_BOX_HALF + (PSI0_BOX_MASS_KG,))
)
PSI0_TABLE_XML = (
    '<geom name="bench_table" type="box" size="%.3f %.3f %.3f" '
    'pos="%.4f %.1f %.2f" contype="3" conaffinity="3" '
    'rgba="0.55 0.42 0.26 1"/>' % (PSI0_TABLE_HALF + PSI0_TABLE_CENTER)
)


# ---------------------------------------------------------------------------
# Math / misc helpers
# ---------------------------------------------------------------------------
def yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def gravity_body_frame(quat):
    """projected gravity = R(q)^T * (0,0,-1), quat wxyz."""
    w, x, y, z = quat
    # third column of R^T applied to -e_z == -(third row of R)
    return np.array([
        -(2.0 * (x * z - w * y)),
        -(2.0 * (y * z + w * x)),
        -(1.0 - 2.0 * (x * x + y * y)),
    ])


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
# Repo / scene building
# ---------------------------------------------------------------------------
def find_repo(arg_repo):
    candidates = [
        arg_repo,
        os.environ.get("FALCON_REPO"),
        "/hhd2/ljk/FALCON",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FALCON"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, CONFIG_REL)):
            return os.path.abspath(c)
    sys.stderr.write(
        "ERROR: FALCON repo not found. Pass --repo /path/to/FALCON "
        "(must contain %s)\n" % CONFIG_REL
    )
    sys.exit(2)


def patch_and_replace(xml, anchor, replacement):
    if xml.count(anchor) != 1:
        raise RuntimeError("XML anchor not unique/found: %r" % anchor)
    return xml.replace(anchor, replacement)


def build_model(repo, config, test):
    """Load ROBOT_SCENE; for box/pillar/psi0 tests inject extras via a temp
    scene file written NEXT TO the original (keeps the <include> + mesh paths
    valid), removed right after model compilation. The box body is appended
    LAST in the worldbody so its freejoint compiles after the 29 robot hinges
    (asserted in lookup_ids)."""
    scene_path = os.path.normpath(
        os.path.join(repo, "sim2real", config["ROBOT_SCENE"]))
    worldbody_extras = []
    if test in ("circle_pillar", "circle_pillar_psi0"):
        worldbody_extras.append(PILLAR_XML)
    if test == "squat_box_psi0":
        worldbody_extras.append(PSI0_TABLE_XML)
    if test in BOX_TESTS or test == "squat_place_psi0":
        worldbody_extras.append(BOX_XML)      # 2 kg bench box
    elif test == "squat_pick_ground":
        worldbody_extras.append(GROUND_BOX_XML)  # 2 kg box, bit-2 vs floor
    elif test in PSI0_CRACKER_TESTS:
        worldbody_extras.append(PSI0_BOX_XML)  # 0.411 kg cracker box
    if not worldbody_extras:
        model = mujoco.MjModel.from_xml_path(scene_path)
    else:
        with open(scene_path, "r") as f:
            xml = f.read()
        xml = patch_and_replace(xml, "</worldbody>",
                                "".join(worldbody_extras) + "</worldbody>")
        if test in WELDED_BOX_TESTS:
            xml = patch_and_replace(xml, "</mujoco>", WELD_XML + "</mujoco>")
        tmp_path = os.path.join(os.path.dirname(scene_path),
                                "_bench_falcon_%s.xml" % test)
        with open(tmp_path, "w") as f:
            f.write(xml)
        try:
            model = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    model.opt.timestep = float(config["SIMULATE_DT"])
    return model


def lookup_ids(model, test):
    ids = {}
    ids["floor_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ids["foot_bodies"] = set()
    for n in FOOT_BODY_NAMES:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if b >= 0:
            ids["foot_bodies"].add(b)
    for nm in ("pelvis", "torso_link", "left_rubber_hand", "right_rubber_hand"):
        ids[nm] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if ids[nm] < 0:
            raise RuntimeError("body not found in model: %s" % nm)
    pg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_geom")
    ids["pillar_geom"] = pg if pg >= 0 else None
    pb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pillar")
    ids["pillar_body"] = pb if pb >= 0 else None
    if test in WELDED_BOX_TESTS:
        ids["box_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            "bench_box")
        ids["box_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                            "bench_box_geom")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                "bench_box_joint")
        ids["box_qposadr"] = int(model.jnt_qposadr[jid])
        ids["box_dofadr"] = int(model.jnt_dofadr[jid])
        ids["weld_lh"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                           "weld_box_lh")
        ids["weld_rh"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                           "weld_box_rh")
        # robot free joint must come first: robot qpos[0:36], box after
        assert ids["box_qposadr"] == 7 + N_JOINTS, \
            "unexpected box qpos addr %d" % ids["box_qposadr"]
    else:
        ids["box_body"] = None
    return ids


# ---------------------------------------------------------------------------
# Policy shim (mirror of sim2real/eval/auto_eval.py — no DDS, no keyboard)
# ---------------------------------------------------------------------------
class MujocoStateShim:
    def __init__(self, num_dof):
        n = num_dof
        # q(3+4+n) + dq(3+3+n) + tau(3+3+n) + ddq(3+3+n)
        self.size = (3 + 4 + n) + 3 * (3 + 3 + n)
        self.robot_state_data = np.zeros((1, self.size), dtype=np.float64)


class MujocoCmdShim:
    def __init__(self, num_dof):
        self.cmd_q = np.zeros(num_dof)
        self.no_action = 0
        self.kp_level = 1.0
        self.kd_level = 1.0

    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None):
        self.cmd_q = np.array(cmd_q, dtype=np.float64)


def make_policy(repo, config, model_path):
    """Build the DDS-free deployment policy (lazy import after sys.path)."""
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "sim2real"))
    from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy

    class EvalPolicy(LocoManipPolicy):
        def _init_sdk_components(self):
            pass  # no DDS

        def _init_communication_components(self):
            self.state_processor = MujocoStateShim(self.robot.NUM_JOINTS)
            self.command_sender = MujocoCmdShim(self.robot.NUM_JOINTS)

        def _init_input_device(self):
            self.use_joystick = False  # no keyboard listener thread

    cfg = dict(config)
    cfg["use_upper_body_controller"] = False  # arms via ref_upper_dof_pos only
    policy = EvalPolicy(config=cfg, model_path=model_path,
                        rl_rate=RL_RATE, policy_action_scale=POLICY_ACTION_SCALE)
    policy.use_policy_action = True
    policy.get_ready_state = False
    return policy


def reset_policy(policy, stand_height, upper_ref):
    policy.last_policy_action = np.zeros((1, policy.num_dofs))
    for k in policy.obs_buf_dict:
        policy.obs_buf_dict[k][:] = 0.0
    policy.lin_vel_command = np.array([[0.0, 0.0]])
    policy.ang_vel_command = np.array([[0.0]])
    policy.stand_command = np.array([[0]])
    policy.base_height_command = np.array([[stand_height]])
    policy.waist_dofs_command = np.zeros((1, 3))
    policy.ref_upper_dof_pos = upper_ref.reshape(1, -1).copy()
    policy.use_policy_action = True
    policy.get_ready_state = False


def push_state(policy, d, n):
    rs = policy.state_processor.robot_state_data
    rs[0, 0:3] = d.qpos[0:3]
    rs[0, 3:7] = d.qpos[3:7]
    rs[0, 7:7 + n] = d.qpos[7:7 + n]
    off = 3 + 4 + n
    rs[0, off + 0:off + 3] = d.qvel[0:3]           # base lin vel (unused by obs)
    rs[0, off + 3:off + 6] = d.qvel[3:6]           # base ang vel, body frame
    rs[0, off + 6:off + 6 + n] = d.qvel[6:6 + n]   # joint vel


# ---------------------------------------------------------------------------
# Command profiles. Return (vx, vy, wz, height_abs, stand_flag, phase, cycle).
# FALCON: stand=1 -> stepping/walk mode, stand=0 -> stance (squat) mode.
# ---------------------------------------------------------------------------
def command_at(test, t, stand_height, vx_target, direction,
               target_height=None, ramp_speed=None, replay=None):
    if t < SETTLE_S:
        return 0.0, 0.0, 0.0, stand_height, 0, "settle", None
    tt = t - SETTLE_S
    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        if tt < WALK_RAMP_S:
            return (vx_target * tt / WALK_RAMP_S, 0.0, 0.0, stand_height, 1,
                    "ramp", None)
        return vx_target, 0.0, 0.0, stand_height, 1, "hold", None
    if test == "squat_box_psi0":
        # Replay segment: base_height_command follows the (pre-smoothed)
        # replayed height_cmd VERBATIM — 0.45 is likely outside FALCON's
        # verified ~0.55 height domain, sent anyway and recorded as-is.
        # Then the standard 20 x 5 s squat cycles; stand=0 throughout
        # (stance mode for both the replay segment and the cycles).
        if tt < replay.duration_s:
            kk = min(int(tt * RL_RATE + 1e-9), replay.n - 1)
            return 0.0, 0.0, 0.0, float(replay.height[kk]), 0, "replay", None
        return command_at("squat_box", t - replay.duration_s, stand_height,
                          vx_target, direction)
    if test == "squat_box":
        cyc = min(int(tt // SQUAT_CYCLE_S), SQUAT_CYCLES - 1)
        tc = tt - cyc * SQUAT_CYCLE_S
        if tc < 1.5:
            h = stand_height - (stand_height - SQUAT_HEIGHT) * tc / 1.5
            ph = "down"
        elif tc < 2.5:
            h, ph = SQUAT_HEIGHT, "hold_low"
        elif tc < 4.0:
            h = SQUAT_HEIGHT + (stand_height - SQUAT_HEIGHT) * (tc - 2.5) / 1.5
            ph = "up"
        else:
            h, ph = stand_height, "hold_high"
        return 0.0, 0.0, 0.0, h, 0, ph, cyc
    if test == "squat_limit":
        # T7: continuous 0.05 m/s descent from stand height down to 0.10 m
        # (deliberately unclipped, far below the feasible domain), then hold
        # 3 s. No rise. stand=0 (stance mode) throughout.
        t_ramp = (stand_height - SQUAT_LIMIT_FLOOR_H) / SQUAT_LIMIT_RATE
        if tt < t_ramp:
            return (0.0, 0.0, 0.0, stand_height - SQUAT_LIMIT_RATE * tt, 0,
                    "down", None)
        return 0.0, 0.0, 0.0, SQUAT_LIMIT_FLOOR_H, 0, "hold", None
    if test == "squat_place_psi0":
        # T8: base_height_command follows the ep053 height_cmd verbatim
        # (single playback); after the replay ends hold the final replayed
        # height for the 1 s stand. stand=0 (stance mode) throughout.
        if tt < replay.duration_s:
            kk = min(int(tt * RL_RATE + 1e-9), replay.n - 1)
            if kk < replay.k_hold0:
                ph = "descent"
            elif kk <= replay.k_hold1:
                ph = "hold"
            else:
                ph = "rise"
            return 0.0, 0.0, 0.0, float(replay.height[kk]), 0, ph, None
        return 0.0, 0.0, 0.0, float(replay.height[-1]), 0, "stand", None
    if test == "squat_sweep":
        # settle(2s, box welded from t=0) -> ramp down to target_height at
        # ramp_speed -> hold 3s -> ramp up -> 1s stand. stand flag 0 (stance).
        t_ramp = (stand_height - target_height) / ramp_speed
        if tt < t_ramp:
            return (0.0, 0.0, 0.0, stand_height - ramp_speed * tt, 0,
                    "down", None)
        if tt < t_ramp + SQUAT_SWEEP_HOLD_S:
            return 0.0, 0.0, 0.0, target_height, 0, "hold", None
        if tt < 2.0 * t_ramp + SQUAT_SWEEP_HOLD_S:
            h = target_height + ramp_speed * (tt - t_ramp - SQUAT_SWEEP_HOLD_S)
            return 0.0, 0.0, 0.0, h, 0, "up", None
        return 0.0, 0.0, 0.0, stand_height, 0, "end", None
    if test in ("circle_pillar", "circle_pillar_psi0"):
        sign = 1.0 if direction == "ccw" else -1.0
        f = min(tt / CIRCLE_RAMP_S, 1.0)
        ph = "ramp" if tt < CIRCLE_RAMP_S else "circle"
        return CIRCLE_VX * f, 0.0, sign * CIRCLE_WZ * f, stand_height, 1, ph, None
    raise ValueError(test)


def trial_duration(test, stand_height=0.75, target_height=None,
                   ramp_speed=None, replay=None, tape=None):
    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        return SETTLE_S + WALK_RAMP_S + WALK_HOLD_S
    if test == "squat_box":
        return SETTLE_S + SQUAT_CYCLES * SQUAT_CYCLE_S
    if test == "squat_box_psi0":
        return SETTLE_S + replay.duration_s + SQUAT_CYCLES * SQUAT_CYCLE_S
    if test in ("circle_pillar", "circle_pillar_psi0"):
        return (SETTLE_S + CIRCLE_RAMP_S
                + CIRCLE_LAPS * 2.0 * math.pi / CIRCLE_WZ + CIRCLE_MARGIN_S)
    if test == "squat_limit":
        t_ramp = (stand_height - SQUAT_LIMIT_FLOOR_H) / SQUAT_LIMIT_RATE
        return SETTLE_S + t_ramp + SQUAT_LIMIT_HOLD_S
    if test == "squat_place_psi0":
        return SETTLE_S + replay.duration_s + PLACE_END_STAND_S
    if test == "squat_pick_ground":
        t_ramp = (stand_height - PICK_SQUAT_HEIGHT) / PICK_RAMP_SPEED
        return (SETTLE_S + PICK_ARM_BLEND_S + 2.0 * t_ramp
                + PICK_BOTTOM_TIMEOUT_S + PICK_GRASP_HOLD_S
                + PICK_END_STAND_S + 0.5)  # margin; loop breaks on finished
    if test == "vln_follow":
        return SETTLE_S + tape.duration_s + VLN_END_STAND_S
    if test == "squat_sweep":
        t_ramp = (stand_height - target_height) / ramp_speed
        return SETTLE_S + 2.0 * t_ramp + SQUAT_SWEEP_HOLD_S + SQUAT_SWEEP_END_S
    if test == "goto_ab":
        return (SETTLE_S + NAV_TIMEOUT_S + CAL_TIMEOUT_S + NAV_END_STAND_S
                + 1.0)  # margin; loop breaks early when mission finishes
    if test == "pipeline_abc":
        t_ramp = (stand_height - PIPE_SQUAT_HEIGHT) / PIPE_SQUAT_SPEED
        return (SETTLE_S + NAV_TIMEOUT_S + CAL_TIMEOUT_S + 2.0 * t_ramp
                + PIPE_BOTTOM_HOLD_S + PIPE_PLACE_HOLD_S + PIPE_END_STAND_S
                + 1.0)
    raise ValueError(test)


# ---------------------------------------------------------------------------
# Box helpers (squat_box / squat_sweep / pipeline_abc / psi0 tests)
# ---------------------------------------------------------------------------
def anchor_welds_at_current_pose(m, d, ids):
    """Anchor both wrist welds at the CURRENT hand->box relative poses and
    activate them (eq_data relpose = pose of box in hand frame; anchor 0 =
    box origin). The box is NOT moved — squat_box_psi0 grasps it where it
    stands on the table."""
    box = ids["box_body"]
    for eq_id, hand in ((ids["weld_lh"], ids["left_rubber_hand"]),
                        (ids["weld_rh"], ids["right_rubber_hand"])):
        R1 = d.xmat[hand].reshape(3, 3)
        rel_pos = R1.T @ (d.xpos[box] - d.xpos[hand])
        rel_quat = quat_mul(quat_conj(d.xquat[hand]), d.xquat[box])
        rel_quat = rel_quat / np.linalg.norm(rel_quat)
        m.eq_data[eq_id, 0:3] = 0.0        # anchor: box origin
        m.eq_data[eq_id, 3:6] = rel_pos    # relpose pos
        m.eq_data[eq_id, 6:10] = rel_quat  # relpose quat
        m.eq_data[eq_id, 10] = 1.0         # torquescale
        d.eq_active[eq_id] = 1


def place_box_and_weld(m, d, ids, yaw0):
    """Place the free box at the wrist midpoint, then weld it to both hands
    with the CURRENT relative poses."""
    lh, rh = ids["left_rubber_hand"], ids["right_rubber_hand"]
    box_adr = ids["box_qposadr"]
    mid = 0.5 * (d.xpos[lh] + d.xpos[rh])
    qyaw = np.array([math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)])
    d.qpos[box_adr:box_adr + 3] = mid
    d.qpos[box_adr + 3:box_adr + 7] = qyaw
    d.qvel[ids["box_dofadr"]:ids["box_dofadr"] + 6] = 0.0
    mujoco.mj_forward(m, d)
    anchor_welds_at_current_pose(m, d, ids)


def place_box_on_table(m, d, ids):
    """squat_box_psi0: cracker box upright on the tabletop at the ep035 pose;
    both welds stay INACTIVE until the grasp event at grasp_close_t."""
    adr = ids["box_qposadr"]
    d.qpos[adr:adr + 2] = PSI0_BOX_START_XY
    d.qpos[adr + 2] = PSI0_BOX_START_Z
    quat = np.asarray(PSI0_BOX_START_QUAT, dtype=np.float64)
    d.qpos[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
    d.qvel[ids["box_dofadr"]:ids["box_dofadr"] + 6] = 0.0
    mujoco.mj_forward(m, d)


def box_surface_dist(d, ids, hand_body, half=PSI0_BOX_HALF):
    """Distance from a hand body origin to the oriented box surface
    (0 if inside the box). half = box half extents in its LOCAL frame
    (default: cracker box; squat_pick_ground passes BOX_SIZE)."""
    box_b = ids["box_body"]
    local = d.xmat[box_b].reshape(3, 3).T @ (d.xpos[hand_body] - d.xpos[box_b])
    half = np.asarray(half, dtype=np.float64)
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def place_box_on_ground(m, d, ids, base_xy, yaw0):
    """squat_pick_ground (v5 T9): bench box standing UPRIGHT on the floor
    (long 0.35 axis vertical -> local x up via a -90 deg pitch about y; top
    face at 0.35 m) PICK_BOX_DIST_M in front of the robot, yawed with it so
    the 0.25 x 0.25 footprint faces the hands. Welds stay INACTIVE until
    the magnetic grasp at the squat bottom."""
    adr = ids["box_qposadr"]
    d.qpos[adr + 0] = base_xy[0] + PICK_BOX_DIST_M * math.cos(yaw0)
    d.qpos[adr + 1] = base_xy[1] + PICK_BOX_DIST_M * math.sin(yaw0)
    d.qpos[adr + 2] = BOX_SIZE[0] + 0.0005   # x-half = the vertical one
    q_pitch = np.array([math.cos(-math.pi / 4.0), 0.0,
                        math.sin(-math.pi / 4.0), 0.0])  # local x -> world z
    q_yaw = np.array([math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)])
    q = quat_mul(q_yaw, q_pitch)
    d.qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
    d.qvel[ids["box_dofadr"]:ids["box_dofadr"] + 6] = 0.0
    mujoco.mj_forward(m, d)


def load_vln_tapes(path):
    """Load + validate the shared tapes.json (make_vln_tapes.py, v5 T10):
    {"tapes":[{"id","dt","cmds":[[t,vx,vy,wz],...],
               "ref_xy_yaw":[[t,x,y,yaw],...]}]} — cmds are zero-order-hold
    breakpoints, ref is the ideal error-free integral sampled at 1 Hz. The
    SAME file is fed verbatim to all four harnesses (fairness)."""
    if not os.path.isfile(path):
        raise SystemExit("vln tapes json not found: %s" % path)
    with open(path) as f:
        try:
            raw = json.load(f)
        except ValueError as e:
            raise SystemExit("vln tapes json unparseable (%s): %s" % (e, path))
    tapes_raw = raw.get("tapes") if isinstance(raw, dict) else None
    if not isinstance(tapes_raw, list) or not tapes_raw:
        raise SystemExit("tapes json has no non-empty 'tapes' list: %s" % path)
    tapes = []
    for k, tp in enumerate(tapes_raw):
        try:
            cmds = np.asarray(tp["cmds"], dtype=np.float64)
            ref = np.asarray(tp["ref_xy_yaw"], dtype=np.float64)
            dt = float(tp["dt"])
            tid = int(tp.get("id", k))
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit("tape %d malformed (%r): %s" % (k, e, path))
        if cmds.ndim != 2 or cmds.shape[1] != 4 or cmds.shape[0] < 1 \
                or not np.isfinite(cmds).all():
            raise SystemExit("tape %d cmds must be finite [[t,vx,vy,wz],...]"
                             ": %s" % (k, path))
        if ref.ndim != 2 or ref.shape[1] != 4 or ref.shape[0] < 2 \
                or not np.isfinite(ref).all():
            raise SystemExit("tape %d ref_xy_yaw must be finite "
                             "[[t,x,y,yaw],...]: %s" % (k, path))
        if (np.any(np.diff(cmds[:, 0]) <= 0.0)
                or np.any(np.diff(ref[:, 0]) <= 0.0)):
            raise SystemExit("tape %d timestamps not strictly increasing: %s"
                             % (k, path))
        tapes.append(SimpleNamespace(
            id=tid, dt=dt,
            cmd_t=cmds[:, 0].copy(), cmd_v=cmds[:, 1:4].copy(),
            ref_t=ref[:, 0].copy(), ref_xy=ref[:, 1:3].copy(),
            ref_yaw=ref[:, 3].copy(),
            duration_s=float(ref[-1, 0])))
    return tapes


def load_upper_replay(path):
    """Load + validate a contract-v1 upper-body replay npz (psi0_replay/
    make_replay.py). falcon29: K=17 = arm14 (dof_names_upper_body order) +
    waist3 [yaw, roll, pitch]. Column mapping is positional -> names are
    checked verbatim against PSI0_UPPER_NAMES."""
    if not os.path.isfile(path):
        raise SystemExit("upper-replay npz not found: %s" % path)
    z = np.load(path, allow_pickle=True)
    missing = [k for k in PSI0_REPLAY_FIELDS if k not in z.files]
    if missing:
        raise SystemExit("upper-replay npz missing fields %s: %s"
                         % (missing, path))
    t = np.asarray(z["t"], dtype=np.float64)
    pos = np.asarray(z["upper_pos"], dtype=np.float64)
    names = [str(nm) for nm in z["upper_names"]]
    if len(t) < 2 or abs((t[1] - t[0]) - 1.0 / RL_RATE) > 1e-6:
        raise SystemExit(
            "upper-replay dt %r != harness control dt %r: %s"
            % (t[1] - t[0] if len(t) > 1 else None, 1.0 / RL_RATE, path))
    if names != list(PSI0_UPPER_NAMES):
        raise SystemExit(
            "upper-replay joint names/order mismatch (want 14 arm joints in "
            "dof_names_upper_body order + waist [yaw, roll, pitch]): %s" % path)
    if pos.shape != (t.shape[0], len(names)):
        raise SystemExit("upper-replay shape mismatch %s: %s"
                         % (pos.shape, path))
    carry = np.asarray(z["carry_pose"], dtype=np.float64)
    if carry.shape != (len(names),):
        raise SystemExit("upper-replay carry_pose shape %s: %s"
                         % (carry.shape, path))
    height = np.asarray(z["height_cmd"], dtype=np.float64)
    if height.shape != (t.shape[0],):
        raise SystemExit("upper-replay height_cmd shape %s: %s"
                         % (height.shape, path))
    if not (np.isfinite(pos).all() and np.isfinite(carry).all()
            and np.isfinite(height).all()):
        raise SystemExit("upper-replay contains NaN/inf: %s" % path)
    meta = json.loads(str(z["meta_json"]))
    # squat_place_psi0 (v4 T8): the weld release fires at t_place =
    # argmin(height_cmd); descent/hold/rise segmentation = contiguous span
    # where height_cmd <= min + PLACE_HOLD_EPS_M (computed once at load,
    # harmless for the other replay sources).
    k_place = int(np.argmin(height))
    hold_ks = np.flatnonzero(height <= float(height.min()) + PLACE_HOLD_EPS_M)
    return SimpleNamespace(
        path=os.path.abspath(path),
        names=names,
        pos=pos,
        height=height,
        grasp_close_t=float(z["grasp_close_t"]),
        carry_pose=carry,
        n=int(pos.shape[0]),
        duration_s=float(pos.shape[0]) / RL_RATE,
        source=str(meta.get("source", "unknown")),
        embodiment=str(meta.get("embodiment", "unknown")),
        k_place=k_place,
        t_place_s=k_place / RL_RATE,
        k_hold0=int(hold_ks[0]),
        k_hold1=int(hold_ks[-1]),
    )


def set_box_collision(m, ids, enabled):
    """pipeline_abc: box geom contype/conaffinity are runtime-writable model
    fields. Disabled (0/0) while carried, enabled (1/1) on release so the
    box lands on the floor (and may touch the robot — real physics)."""
    v = 2 if enabled else 0
    m.geom_contype[ids["box_geom"]] = v
    m.geom_conaffinity[ids["box_geom"]] = v
    # MuJoCo >= 3.2.4 broadphase culls BODY pairs via body_contype/
    # body_conaffinity (compile-time OR of each body's geom bits). The box
    # compiles 0/0 and the world body 1/1, so geom-level writes alone are
    # invisible to broadphase and the box free-falls through the floor.
    # Mirror the geom bits on the body level.
    if hasattr(m, "body_contype"):
        bid = int(m.geom_bodyid[ids["box_geom"]])
        m.body_contype[bid] = v
        m.body_conaffinity[bid] = v
    if enabled:
        # bit-2 scheme: floor OR-ed with bit 2; robot stays bit 1 -> the box
        # falls to the floor and never jams between the palms.
        fg = ids.get("floor_geom")
        if fg is not None:
            m.geom_contype[fg] |= 2
            m.geom_conaffinity[fg] |= 2
            if hasattr(m, "body_contype"):
                fb = int(m.geom_bodyid[fg])
                m.body_contype[fb] |= 2
                m.body_conaffinity[fb] |= 2


def release_box(m, d, ids):
    """pipeline_abc bottom-of-squat release: deactivate both wrist welds and
    let the box collide with everything."""
    d.eq_active[ids["weld_lh"]] = 0
    d.eq_active[ids["weld_rh"]] = 0
    set_box_collision(m, ids, True)


# ---------------------------------------------------------------------------
# Shared two-phase navigation controller (spec v3 — math identical across
# the four harnesses; copied from the spec, do not improvise).
# ---------------------------------------------------------------------------
def nav_errors(base_xy, yaw, goal_xy, goal_yaw):
    """e = goal - base, rotated into the body frame. Heading goal = bearing
    to the goal while dist > 0.5 m, else the final goal yaw. Also returns
    the error vs the FINAL goal yaw (used by all exit/success criteria)."""
    dx = goal_xy[0] - base_xy[0]
    dy = goal_xy[1] - base_xy[1]
    dist = math.hypot(dx, dy)
    ex = math.cos(yaw) * dx + math.sin(yaw) * dy
    ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
    heading_goal = math.atan2(dy, dx) if dist > NAV_SWITCH_DIST_M else goal_yaw
    eyaw = wrap_angle(heading_goal - yaw)
    eyaw_final = wrap_angle(goal_yaw - yaw)
    return ex, ey, eyaw, dist, eyaw_final


def nav_cmd_coarse(ex, ey, eyaw, vx_max):
    """NAV phase P-law: vx=clip(1.0*ex, 0, vx_max), vy=clip(1.0*ey, +-0.3),
    wz=clip(1.5*eyaw, +-0.6)."""
    vx = min(max(NAV_KP_LIN * ex, 0.0), vx_max)
    vy = min(max(NAV_KP_LIN * ey, -NAV_VY_MAX), NAV_VY_MAX)
    wz = min(max(NAV_KP_YAW * eyaw, -NAV_WZ_MAX), NAV_WZ_MAX)
    return vx, vy, wz


def nav_cmd_cal(ex, ey, eyaw):
    """CAL phase: SAME P-law, commands clipped to |vx|,|vy|<=0.10 m/s and
    |wz|<=0.10 rad/s. Deliberately NO deadband compensation — this test
    measures native small-command effectiveness."""
    vx = min(max(NAV_KP_LIN * ex, -CAL_LIN_MAX), CAL_LIN_MAX)
    vy = min(max(NAV_KP_LIN * ey, -CAL_LIN_MAX), CAL_LIN_MAX)
    wz = min(max(NAV_KP_YAW * eyaw, -CAL_WZ_MAX), CAL_WZ_MAX)
    return vx, vy, wz


class NavMission:
    """Phase machine for goto_ab / pipeline_abc.

    Sequencing + FALCON stand-flag adaptation live here; ALL command math is
    in nav_errors/nav_cmd_coarse/nav_cmd_cal (shared, spec-identical).
    FALCON: stand=1 (stepping) for the whole NAV+CAL, stand=0 afterwards
    (end stand / squat-place), settle uses stand=0 like the other tests.
    """

    def __init__(self, test, stand_height):
        self.test = test
        self.stand_height = stand_height
        if test == "goto_ab":
            self.goal_xy, self.goal_yaw = GOTO_B_XY, GOTO_B_YAW
            self.vx_max = NAV_VX_MAX
        else:
            self.goal_xy, self.goal_yaw = PIPE_C_XY, PIPE_C_YAW
            self.vx_max = PIPE_NAV_VX_MAX
        self.squat_t_ramp = (stand_height - PIPE_SQUAT_HEIGHT) / PIPE_SQUAT_SPEED
        self.phase = "settle"
        self.phase_t0 = 0.0
        self.cal_hold_t0 = None
        self.fine_converged = False
        self.nav_end = None      # {err_pos_m, err_yaw_rad, t_nav_s, timeout}
        self.cal_end = None      # {err_pos_m, err_yaw_rad, t_cal_s}
        self.release_pending = False
        self.finished = False

    def _to(self, phase, t):
        self.phase = phase
        self.phase_t0 = t

    def command(self, t, base_xy, yaw):
        """-> (vx, vy, wz, height_abs, stand_flag, phase). Falls through
        phase transitions so the new phase issues its command immediately."""
        ex, ey, eyaw, dist, eyaw_final = nav_errors(
            base_xy, yaw, self.goal_xy, self.goal_yaw)
        if self.phase == "settle":
            if t < SETTLE_S:
                return 0.0, 0.0, 0.0, self.stand_height, 0, "settle"
            self._to("nav", t)
        if self.phase == "nav":
            el = t - self.phase_t0
            reached = (dist <= NAV_POS_TOL_M
                       and abs(eyaw_final) <= NAV_YAW_TOL_RAD)
            if reached or el >= NAV_TIMEOUT_S:
                self.nav_end = {"err_pos_m": dist,
                                "err_yaw_rad": abs(eyaw_final),
                                "t_nav_s": el, "timeout": not reached}
                self._to("cal", t)
            else:
                vx, vy, wz = nav_cmd_coarse(ex, ey, eyaw, self.vx_max)
                return vx, vy, wz, self.stand_height, 1, "nav"
        if self.phase == "cal":
            el = t - self.phase_t0
            within = (dist <= CAL_POS_TOL_M
                      and abs(eyaw_final) <= CAL_YAW_TOL_RAD)
            if within:
                if self.cal_hold_t0 is None:
                    self.cal_hold_t0 = t
                elif t - self.cal_hold_t0 >= CAL_HOLD_S:
                    self.fine_converged = True
            else:
                self.cal_hold_t0 = None
            if self.fine_converged or el >= CAL_TIMEOUT_S:
                self.cal_end = {"err_pos_m": dist,
                                "err_yaw_rad": abs(eyaw_final),
                                "t_cal_s": el}
                self._to("end_stand" if self.test == "goto_ab"
                         else "squat_down", t)
            else:
                vx, vy, wz = nav_cmd_cal(ex, ey, eyaw)
                return vx, vy, wz, self.stand_height, 1, "cal"
        if self.phase == "squat_down":
            el = t - self.phase_t0
            if el < self.squat_t_ramp:
                h = self.stand_height - PIPE_SQUAT_SPEED * el
                return 0.0, 0.0, 0.0, h, 0, "squat_down"
            self._to("squat_hold", t)
        if self.phase == "squat_hold":
            el = t - self.phase_t0
            if el < PIPE_BOTTOM_HOLD_S:
                return 0.0, 0.0, 0.0, PIPE_SQUAT_HEIGHT, 0, "squat_hold"
            self.release_pending = True   # executed by the rollout loop
            self._to("place_hold", t)
        if self.phase == "place_hold":
            el = t - self.phase_t0
            if el < PIPE_PLACE_HOLD_S:
                return 0.0, 0.0, 0.0, PIPE_SQUAT_HEIGHT, 0, "place_hold"
            self._to("squat_up", t)
        if self.phase == "squat_up":
            el = t - self.phase_t0
            if el < self.squat_t_ramp:
                h = PIPE_SQUAT_HEIGHT + PIPE_SQUAT_SPEED * el
                return 0.0, 0.0, 0.0, h, 0, "squat_up"
            self._to("end_stand", t)
        # end_stand (both tests): stand flag back to 0, hold stand height.
        end_s = (NAV_END_STAND_S if self.test == "goto_ab"
                 else PIPE_END_STAND_S)
        if t - self.phase_t0 >= end_s:
            self.finished = True
        return 0.0, 0.0, 0.0, self.stand_height, 0, "end_stand"


def fill_nav_metrics(metrics, mission):
    """Common T5/T6 navigation metrics; returns (success_coarse,
    success_fine). Coarse = NAV-end errors within 0.30 m / 15 deg (False if
    the trial died before NAV ended); fine = CAL convergence held 1 s."""
    ne, ce = mission.nav_end, mission.cal_end
    metrics["nav_err_pos_m"] = ne["err_pos_m"] if ne else None
    metrics["nav_err_yaw_rad"] = ne["err_yaw_rad"] if ne else None
    metrics["nav_timeout"] = bool(ne["timeout"]) if ne else None
    metrics["t_nav_s"] = ne["t_nav_s"] if ne else None
    metrics["cal_err_pos_m"] = ce["err_pos_m"] if ce else None
    metrics["cal_err_yaw_rad"] = ce["err_yaw_rad"] if ce else None
    metrics["t_cal_s"] = ce["t_cal_s"] if ce else None
    if ne and ce:
        metrics["cal_improve_pos_m"] = ne["err_pos_m"] - ce["err_pos_m"]
        metrics["cal_improve_yaw_rad"] = ne["err_yaw_rad"] - ce["err_yaw_rad"]
    else:
        metrics["cal_improve_pos_m"] = None
        metrics["cal_improve_yaw_rad"] = None
    success_coarse = bool(ne is not None
                          and ne["err_pos_m"] <= NAV_POS_TOL_M
                          and ne["err_yaw_rad"] <= NAV_YAW_TOL_RAD)
    success_fine = bool(mission.fine_converged)
    metrics["success_coarse"] = success_coarse
    metrics["success_fine"] = success_fine
    return success_coarse, success_fine


# ---------------------------------------------------------------------------
# v5 T9 squat_pick_ground phase machine
# ---------------------------------------------------------------------------
class PickGroundMission:
    """Phase machine for squat_pick_ground (v5 T9): settle -> arm_reach
    (1 s blend to PICK_ARM_POSE) -> down (0.2 m/s to the 0.25 m command) ->
    bottom (magnetic grasp when both wrists < 0.30 m from the box surface;
    5 s timeout -> pick_failed) -> grasp_hold 0.5 s -> rise -> stand 1 s.
    Pure stance test: vx=vy=wz=0 and stand=0 throughout (FALCON stance
    mode). The weld anchoring itself is executed by the rollout loop via
    grasp_pending (same pattern as NavMission.release_pending)."""

    def __init__(self, stand_height):
        self.stand_height = stand_height
        self.t_ramp = (stand_height - PICK_SQUAT_HEIGHT) / PICK_RAMP_SPEED
        self.phase = "settle"
        self.phase_t0 = 0.0
        self.grasped = False
        self.grasp_pending = False
        self.pick_failed = False
        self.grasp_dists = None   # (l, r) at the grasp or timeout moment
        self.grasp_time = None
        self.finished = False

    def _to(self, phase, t):
        self.phase = phase
        self.phase_t0 = t

    def arm_frac(self, t):
        """0 -> 1 blend factor default arms -> PICK_ARM_POSE."""
        if t < SETTLE_S:
            return 0.0
        return min(1.0, (t - SETTLE_S) / PICK_ARM_BLEND_S)

    def command(self, t, wrist_dists):
        """-> (height_abs, phase). Falls through phase transitions so the
        new phase issues its command immediately."""
        if self.phase == "settle":
            if t < SETTLE_S:
                return self.stand_height, "settle"
            self._to("arm_reach", t)
        if self.phase == "arm_reach":
            if t - self.phase_t0 < PICK_ARM_BLEND_S:
                return self.stand_height, "arm_reach"
            self._to("down", t)
        if self.phase == "down":
            el = t - self.phase_t0
            if el < self.t_ramp:
                return self.stand_height - PICK_RAMP_SPEED * el, "down"
            self._to("bottom", t)
        if self.phase == "bottom":
            el = t - self.phase_t0
            if (not self.grasped and wrist_dists is not None
                    and max(wrist_dists) < PICK_GRASP_DIST_M):
                self.grasped = True
                self.grasp_pending = True   # weld anchored by rollout loop
                self.grasp_dists = wrist_dists
                self.grasp_time = t
                self._to("grasp_hold", t)
            elif el >= PICK_BOTTOM_TIMEOUT_S:
                self.pick_failed = True     # rises anyway (spec v5)
                self.grasp_dists = wrist_dists
                self._to("rise", t)
            else:
                return PICK_SQUAT_HEIGHT, "bottom"
        if self.phase == "grasp_hold":
            if t - self.phase_t0 < PICK_GRASP_HOLD_S:
                return PICK_SQUAT_HEIGHT, "grasp_hold"
            self._to("rise", t)
        if self.phase == "rise":
            el = t - self.phase_t0
            if el < self.t_ramp:
                return PICK_SQUAT_HEIGHT + PICK_RAMP_SPEED * el, "rise"
            self._to("stand", t)
        if t - self.phase_t0 >= PICK_END_STAND_S:
            self.finished = True
        return self.stand_height, "stand"


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------
def run_trial(ctx, trial_idx, seed, vx_target=1.0, direction="ccw",
              target_height=None, ramp_speed=None, sweep_kind=None,
              tape_idx=None, video_path=None):
    """One trial. Deterministic given seed (rendering does not affect physics,
    so failed/selected trials can be re-run with video)."""
    test = ctx["test"]
    m = ctx["model"]
    ids = ctx["ids"]
    policy = ctx["policy"]
    replay = ctx.get("replay")
    tape = ctx["tapes"][tape_idx] if test == "vln_follow" else None
    n = N_JOINTS
    kp, kd, effort = ctx["kp"], ctx["kd"], ctx["effort"]
    default = ctx["default"]
    stand_height = ctx["stand_height"]
    decimation = ctx["decimation"]
    sim_dt = ctx["sim_dt"]
    ctrl_dt = sim_dt * decimation

    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)

    # --- initial state ------------------------------------------------------
    if test in ("circle_pillar", "circle_pillar_psi0"):
        yaw0 = math.pi / 2.0 if direction == "ccw" else -math.pi / 2.0
        base_xy = (1.0, 0.0)
    elif test == "squat_box_psi0":
        yaw0 = 0.0   # BendPick table scene alignment (identity heading)
        base_xy = (0.0, 0.0)
    elif test == "goto_ab":
        # A = origin, heading 0 + seeded yaw noise +-0.3 rad (spec v3)
        yaw0 = float(rng.uniform(-GOTO_YAW_NOISE, GOTO_YAW_NOISE))
        base_xy = (0.0, 0.0)
    elif test == "pipeline_abc":
        yaw0 = 0.0                  # A = origin, heading 0 (spec v3)
        base_xy = (0.0, 0.0)
    elif test == "vln_follow":
        # the shared tape ref integrates from the identity pose; errors are
        # measured in the tape-start frame anyway, yaw0=0 keeps it simple
        yaw0 = 0.0
        base_xy = (0.0, 0.0)
    else:
        yaw0 = float(rng.uniform(-math.pi, math.pi))
        base_xy = (0.0, 0.0)

    init_joints = default.copy()
    upper_ref = default[15:15 + 14].copy()
    if test in BOX_TESTS:
        init_joints[15:15 + 14] = HUG_ARM_POSE
        upper_ref = HUG_ARM_POSE.copy()
    elif test in PSI0_TESTS:
        # waist (qpos 12:15, [yaw, roll, pitch]) + arms (qpos 15:29) start
        # exactly at replay frame 0 so the replayed refs do not snap at t=0.
        init_joints[12:15] = replay.pos[0, 14:17]
        init_joints[15:15 + 14] = replay.pos[0, 0:14]
        upper_ref = replay.pos[0, 0:14].copy()

    d.qpos[:] = 0.0
    d.qpos[0], d.qpos[1], d.qpos[2] = base_xy[0], base_xy[1], INIT_BASE_Z
    d.qpos[3:7] = [math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)]
    d.qpos[7:7 + n] = init_joints + rng.uniform(-JOINT_NOISE, JOINT_NOISE, n)
    if test in PSI0_TESTS:
        d.qpos[7 + 12:7 + 29] = init_joints[12:29]  # replayed 17 joints exact
    d.qvel[:] = 0.0
    if test in ("pipeline_abc", "squat_place_psi0"):
        # model-level fields persist across trials: re-disable box collision
        # (a previous trial's release set contype/conaffinity bits).
        set_box_collision(m, ids, False)
    mujoco.mj_forward(m, d)
    if test in BOX_TESTS + PSI0_WELD_T0_TESTS:
        # v2 mechanism verbatim: box at the wrist midpoint, welded from t=0;
        # the 2 s settle then serves as the standing carry settle (spec v3).
        place_box_and_weld(m, d, ids, yaw0)
    elif test == "squat_box_psi0":
        place_box_on_table(m, d, ids)   # welds OFF until grasp_close_t
    elif test == "squat_pick_ground":
        # v5 T9: box upright on the floor in front; welds OFF until the
        # magnetic grasp at the squat bottom.
        place_box_on_ground(m, d, ids, base_xy, yaw0)

    mission = NavMission(test, stand_height) if test in NAV_TESTS else None
    pick = (PickGroundMission(stand_height)
            if test == "squat_pick_ground" else None)

    reset_policy(policy, stand_height, upper_ref)
    policy.command_sender.cmd_q = init_joints.copy()

    # --- video ----------------------------------------------------------
    writer = None
    cam = None
    next_frame_t = 0.0
    if video_path is not None and ctx["renderer"] is not None:
        import cv2
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 VIDEO_FPS, (VIDEO_W, VIDEO_H))
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.elevation, cam.azimuth = 3.5, -20.0, 135.0

    # --- records ----------------------------------------------------------
    rec = {k: [] for k in ("t", "cmd_vx", "cmd_wz", "cmd_h", "base_x",
                           "base_y", "base_z", "v_fwd", "omega_z", "tilt",
                           "yaw", "box_dist", "phase")}
    fall = False
    fall_time = fall_phase = fall_reason = fall_cycle = None
    harness_error = None
    pillar_collision = False
    pillar_collision_time = None
    box_kept = True
    box_drop_time = None
    box_released = False
    box_carry_dist_max = None
    release_frame = None     # squat_place_psi0: heading + foot front edge
    # squat_box_psi0 grasp state
    pick_checked = False
    pick_success = None
    grasp_dists = None
    box_z0 = box_z_max = None
    laps = []
    accum_angle = 0.0
    prev_pos_angle = None
    start_xy = start_yaw = None

    duration = trial_duration(test, stand_height=stand_height,
                              target_height=target_height,
                              ramp_speed=ramp_speed, replay=replay,
                              tape=tape)
    n_ticks = int(round(duration / ctrl_dt))
    t = 0.0
    wall_t0 = time.time()
    floor_g = ids["floor_geom"]
    pillar_g = ids["pillar_geom"]
    pillar_b = ids["pillar_body"]
    box_b = ids["box_body"]
    foot_bodies = ids["foot_bodies"]
    torso_b = ids["torso_link"]
    nonfoot_ground = False

    try:
        for k in range(n_ticks):
            t = k * ctrl_dt
            if mission is not None:
                base_xy_now = (float(d.qpos[0]), float(d.qpos[1]))
                yaw_now = yaw_from_quat(d.qpos[3:7])
                vx_c, vy_c, wz_c, h_c, stand_c, phase = mission.command(
                    t, base_xy_now, yaw_now)
                cyc = None
                if mission.release_pending:
                    release_box(m, d, ids)
                    mission.release_pending = False
                    box_released = True
            elif pick is not None:
                # v5 T9: stance test — only the height command moves; the
                # magnetic grasp anchors both welds at the current pose.
                d_l = box_surface_dist(d, ids, ids["left_rubber_hand"],
                                       half=BOX_SIZE)
                d_r = box_surface_dist(d, ids, ids["right_rubber_hand"],
                                       half=BOX_SIZE)
                h_c, phase = pick.command(t, (d_l, d_r))
                vx_c = vy_c = wz_c = 0.0
                stand_c = 0          # FALCON stance mode throughout (T9)
                cyc = None
                if pick.grasp_pending:
                    anchor_welds_at_current_pose(m, d, ids)
                    pick.grasp_pending = False
            elif tape is not None:
                # v5 T10: zero-order hold of the tape breakpoints. FALCON
                # adaptation (mode semantics, see NOTES): stand=1 while
                # following, stand=0 during full-stop segments + end stand.
                tt = t - SETTLE_S
                cyc = None
                h_c = stand_height
                if tt < 0.0:
                    vx_c = vy_c = wz_c = 0.0
                    stand_c, phase = 0, "settle"
                elif tt < tape.duration_s:
                    j = int(np.searchsorted(tape.cmd_t, tt + 1e-9)) - 1
                    if j >= 0:
                        vx_c = float(tape.cmd_v[j, 0])
                        vy_c = float(tape.cmd_v[j, 1])
                        wz_c = float(tape.cmd_v[j, 2])
                    else:
                        vx_c = vy_c = wz_c = 0.0
                    is_stop = (vx_c == 0.0 and vy_c == 0.0 and wz_c == 0.0)
                    stand_c = 0 if is_stop else 1
                    phase = "stop" if is_stop else "follow"
                else:
                    vx_c = vy_c = wz_c = 0.0
                    stand_c, phase = 0, "stand"
            else:
                vx_c, vy_c, wz_c, h_c, stand_c, phase, cyc = command_at(
                    test, t, stand_height, vx_target, direction,
                    target_height=target_height, ramp_speed=ramp_speed,
                    replay=replay)

            # command injection into the deployment policy
            policy.lin_vel_command = np.array([[vx_c, vy_c]])
            policy.ang_vel_command = np.array([[wz_c]])
            policy.stand_command = np.array([[stand_c]])
            policy.base_height_command = np.array([[h_c]])

            if replay is not None:
                # psi0 upper-body replay: arm14 -> ref_upper_dof_pos (enters
                # the obs AND the residual action path), waist3 ->
                # waist_dofs_command [yaw, roll, pitch].
                if test == "squat_box_psi0":
                    # replay starts at settle end; afterwards the arms freeze
                    # at carry_pose for the squat cycles.
                    kk = int(math.floor((t - SETTLE_S) / ctrl_dt + 1e-9))
                    if kk < 0:
                        row = replay.pos[0]
                    elif kk >= replay.n:
                        row = replay.carry_pose
                    else:
                        row = replay.pos[kk]
                elif test == "squat_place_psi0":
                    # single playback starting at settle end; after the
                    # rise segment the arms freeze at the LAST replayed
                    # frame (stood-up pose — the box has been placed).
                    kk = int(math.floor((t - SETTLE_S) / ctrl_dt + 1e-9))
                    if kk < 0:
                        row = replay.pos[0]
                    elif kk >= replay.n:
                        row = replay.pos[replay.n - 1]
                    else:
                        row = replay.pos[kk]
                else:
                    row = replay.pos[k % replay.n]  # looped from t=0
                policy.ref_upper_dof_pos = row[0:14].reshape(1, 14).copy()
                policy.waist_dofs_command = row[14:17].reshape(1, 3).copy()

            if pick is not None:
                # v5 T9: blend the arms from the default hang to the low
                # forward-reach cradle pose over arm_reach, frozen after.
                fr = pick.arm_frac(t)
                policy.ref_upper_dof_pos = (
                    upper_ref + fr * (PICK_ARM_POSE - upper_ref)
                ).reshape(1, 14)

            # 50 Hz policy tick via the deployment pipeline
            push_state(policy, d, n)
            policy.policy_action()
            cmd_q = policy.command_sender.cmd_q

            # PD at sim rate (auto_eval.py loop), decimation sub-steps
            for _ in range(decimation):
                q = d.qpos[7:7 + n]
                dq = d.qvel[6:6 + n]
                tau = np.clip(kp * (cmd_q - q) - kd * dq, -effort, effort)
                d.ctrl[0:6] = 0.0
                d.ctrl[6:6 + n] = tau
                mujoco.mj_step(m, d)

                for ci in range(d.ncon):
                    g1 = d.contact[ci].geom1
                    g2 = d.contact[ci].geom2
                    if floor_g in (g1, g2):
                        other = g2 if g1 == floor_g else g1
                        b = int(m.geom_bodyid[other])
                        # box_b exempt: the released box landing on the
                        # floor (pipeline_abc) is not a robot fall.
                        if (b != 0 and b != pillar_b and b != box_b
                                and b not in foot_bodies):
                            nonfoot_ground = True
                    if pillar_g is not None and pillar_g in (g1, g2):
                        other = g2 if g1 == pillar_g else g1
                        b = int(m.geom_bodyid[other])
                        if b != 0 and b != pillar_b and not pillar_collision:
                            pillar_collision = True
                            pillar_collision_time = t

            t = (k + 1) * ctrl_dt

            # numeric divergence guard (weld vs PD can blow up qacc)
            if (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.isfinite(d.qpos).all()):
                harness_error = "numeric_divergence (bad qacc / nonfinite qpos)"
                break

            # squat_box_psi0 grasp event: weld the box in place IFF both
            # wrists are close enough to its surface, else pick_failed.
            if (test == "squat_box_psi0" and not pick_checked
                    and t - SETTLE_S >= replay.grasp_close_t):
                pick_checked = True
                d_l = box_surface_dist(d, ids, ids["left_rubber_hand"])
                d_r = box_surface_dist(d, ids, ids["right_rubber_hand"])
                grasp_dists = (d_l, d_r)
                pick_success = bool(d_l < PSI0_GRASP_DIST_M
                                    and d_r < PSI0_GRASP_DIST_M)
                if pick_success:
                    anchor_welds_at_current_pose(m, d, ids)

            # squat_place_psi0 place event (v4 T8): at t_place =
            # argmin(height_cmd) release both welds + enable bit-2 box
            # collision (set_box_collision mirrors geom bits onto the body
            # level) so the box lands on the floor. Record the heading and
            # the foot FRONT EDGE projection for box_land_dx (the 8 dummy
            # foot-corner bodies include the front corners).
            if (test == "squat_place_psi0" and not box_released
                    and t - SETTLE_S >= replay.t_place_s):
                release_box(m, d, ids)
                box_released = True
                yaw_rel = yaw_from_quat(d.qpos[3:7])
                head_rel = (math.cos(yaw_rel), math.sin(yaw_rel))
                s_edge = max(
                    float(d.xpos[b][0]) * head_rel[0]
                    + float(d.xpos[b][1]) * head_rel[1]
                    for b in foot_bodies)
                release_frame = {"heading": head_rel, "s_edge": s_edge,
                                 "t": t}

            if writer is not None and t >= next_frame_t:
                import cv2
                cam.lookat[:] = d.qpos[0:3]
                ctx["renderer"].update_scene(d, camera=cam)
                writer.write(cv2.cvtColor(ctx["renderer"].render(),
                                          cv2.COLOR_RGB2BGR))
                next_frame_t += 1.0 / VIDEO_FPS

            # ----------------- state metrics -------------------------------
            quat = d.qpos[3:7].copy()
            grav = gravity_body_frame(quat)
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
            rec["yaw"].append(yaw)
            rec["phase"].append(phase)

            if box_b is not None:
                bd = float(np.linalg.norm(d.xpos[box_b] - d.xpos[torso_b]))
                rec["box_dist"].append(bd)
                if test in ("squat_box_psi0", "squat_pick_ground"):
                    bz = float(d.xpos[box_b][2])
                    box_z0 = bz if box_z0 is None else box_z0
                    box_z_max = bz if box_z_max is None else max(box_z_max, bz)
                # carry phase only: pipeline_abc releases the box at place;
                # squat_box_psi0 / squat_pick_ground carry only after a
                # successful pick (before the grasp the box legitimately
                # sits on the table/floor, beyond BOX_KEEP_DIST).
                carrying = not box_released
                if test == "squat_box_psi0":
                    carrying = carrying and bool(pick_success)
                elif pick is not None:
                    carrying = carrying and pick.grasped
                if carrying:
                    box_carry_dist_max = (bd if box_carry_dist_max is None
                                          else max(box_carry_dist_max, bd))
                    if bd >= BOX_KEEP_DIST and box_kept:
                        box_kept = False
                        box_drop_time = t

            # ----------------- lap tracking (circle) -----------------------
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
                break

            if (test in ("circle_pillar", "circle_pillar_psi0")
                    and len(laps) >= CIRCLE_LAPS):
                break
            if mission is not None and mission.finished:
                break
            if pick is not None and pick.finished:
                break
    except Exception as e:  # noqa: BLE001 — record, never kill the batch
        harness_error = "exception: %r" % (e,)
    finally:
        if writer is not None:
            writer.release()

    # ------------------ per-test metrics -----------------------------------
    ts = np.array(rec["t"]) if rec["t"] else np.zeros(0)
    seg = ts >= SETTLE_S
    metrics = {}

    def rmse(a, b):
        if a.size == 0:
            return None
        return float(np.sqrt(np.mean((a - b) ** 2)))

    metrics["max_tilt_rad"] = (float(np.max(np.array(rec["tilt"])[seg]))
                               if seg.any() else None)

    if test in ("walk_speed", "speed_sweep", "walk_speed_psi0"):
        metrics["cmd_vx"] = vx_target
        v = np.array(rec["v_fwd"])
        c = np.array(rec["cmd_vx"])
        metrics["vx_rmse"] = rmse(c[seg], v[seg]) if seg.any() else None
        win = ts >= (duration - WALK_LAST_WINDOW_S)
        metrics["mean_vx_last8s"] = float(np.mean(v[win])) if win.any() else None
        thr = 0.9 * vx_target if test == "speed_sweep" else 0.9
        success = (not fall) and harness_error is None \
            and metrics["mean_vx_last8s"] is not None \
            and metrics["mean_vx_last8s"] >= thr
        if test == "walk_speed_psi0":
            metrics["box_kept"] = bool(box_kept)
            metrics["box_drop_time"] = box_drop_time
            metrics["box_torso_dist_max"] = box_carry_dist_max
            success = bool(success) and box_kept
    elif test == "squat_box_psi0":
        z = np.array(rec["base_z"])
        h = np.array(rec["cmd_h"])
        metrics["height_rmse"] = rmse(h[seg], z[seg]) if seg.any() else None
        metrics["min_base_z"] = float(np.min(z[seg])) if seg.any() else None
        cycles_t0 = SETTLE_S + replay.duration_s  # trial clock, cycles start
        end_t = fall_time if fall else (ts[-1] if ts.size else 0.0)
        if fall or harness_error is not None:
            cycles_done = int(max(0.0, end_t - cycles_t0) // SQUAT_CYCLE_S)
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
        kept = bool(pick_success) and box_drop_time is None
        metrics["box_kept"] = kept
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = box_carry_dist_max
        success = ((not fall) and harness_error is None
                   and bool(pick_success) and cycles_done >= SQUAT_CYCLES
                   and kept)
    elif test == "squat_box":
        z = np.array(rec["base_z"])
        h = np.array(rec["cmd_h"])
        metrics["height_rmse"] = rmse(h[seg], z[seg]) if seg.any() else None
        metrics["min_base_z"] = float(np.min(z[seg])) if seg.any() else None
        end_t = fall_time if fall else (ts[-1] if ts.size else 0.0)
        if fall or harness_error is not None:
            cycles_done = int(max(0.0, end_t - SETTLE_S) // SQUAT_CYCLE_S)
        else:
            cycles_done = SQUAT_CYCLES
        metrics["cycles_completed"] = min(cycles_done, SQUAT_CYCLES)
        metrics["fall_cycle"] = fall_cycle
        metrics["box_kept"] = bool(box_kept)
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = (float(np.max(rec["box_dist"]))
                                         if rec["box_dist"] else None)
        success = (not fall) and harness_error is None \
            and cycles_done >= SQUAT_CYCLES
        metrics["success_with_box"] = bool(success and box_kept)
    elif test in ("circle_pillar", "circle_pillar_psi0"):
        metrics["direction"] = direction
        if seg.any():
            r = np.hypot(np.array(rec["base_x"])[seg],
                         np.array(rec["base_y"])[seg])
            radial = np.abs(r - 1.0)
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
            metrics["lap%d_yaw_err_rad" % (i + 1)] = (lp["yaw_err_rad"]
                                                      if lp else None)
        metrics["pillar_collision"] = bool(pillar_collision)
        metrics["pillar_collision_time"] = pillar_collision_time
        success = ((not fall) and harness_error is None
                   and (not pillar_collision)
                   and len(laps) >= CIRCLE_LAPS
                   and metrics["radial_err_mean"] is not None
                   and metrics["radial_err_mean"] <= 0.15)
        if test == "circle_pillar_psi0":
            metrics["box_kept"] = bool(box_kept)
            metrics["box_drop_time"] = box_drop_time
            metrics["box_torso_dist_max"] = box_carry_dist_max
            success = bool(success) and box_kept
    elif test == "squat_sweep":
        metrics["sweep_kind"] = sweep_kind
        metrics["target_height"] = target_height
        metrics["ramp_speed"] = ramp_speed
        hold_idx = [i for i, p in enumerate(rec["phase"]) if p == "hold"]
        if hold_idx:
            metrics["achieved_depth"] = float(np.mean(
                [rec["base_z"][i] for i in hold_idx]))
            metrics["depth_err"] = (metrics["achieved_depth"]
                                    - target_height)  # signed, + = too high
            hx0 = rec["base_x"][hold_idx[0]]
            hy0 = rec["base_y"][hold_idx[0]]
            metrics["root_drift_hold"] = float(max(
                math.hypot(rec["base_x"][i] - hx0, rec["base_y"][i] - hy0)
                for i in hold_idx))
        else:
            metrics["achieved_depth"] = None
            metrics["depth_err"] = None
            metrics["root_drift_hold"] = None
        down_idx = [i for i, p in enumerate(rec["phase"]) if p == "down"]
        trial_complete = ((not fall) and harness_error is None
                          and rec["phase"] and rec["phase"][-1] == "end")
        if trial_complete and down_idx:
            # base XY at end of trial vs just before the squat started
            metrics["root_drift_total"] = float(math.hypot(
                rec["base_x"][-1] - rec["base_x"][down_idx[0]],
                rec["base_y"][-1] - rec["base_y"][down_idx[0]]))
        else:
            metrics["root_drift_total"] = None
        metrics["box_kept"] = bool(box_kept)
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = box_carry_dist_max
        success = (not fall) and harness_error is None
        metrics["success_with_box"] = bool(success and box_kept)
    elif test == "squat_limit":
        # v4 T7: full-rate post-settle arrays; drift reference = root XY at
        # the first post-settle tick (descent start).
        z = np.array(rec["base_z"])
        h = np.array(rec["cmd_h"])
        si = np.flatnonzero(seg)
        if si.size:
            x0 = float(rec["base_x"][si[0]])
            y0 = float(rec["base_y"][si[0]])
            drift = np.hypot(np.array(rec["base_x"])[si] - x0,
                             np.array(rec["base_y"])[si] - y0)
            zs, hs = z[si], h[si]
            tls = np.array(rec["tilt"])[si]
            tss = ts[si]
            # raw height-tracking-drift curve, 10 Hz downsample
            step = max(1, int(round(DESCENT_CURVE_DT_S * RL_RATE)))
            metrics["descent_curve"] = [
                [round(float(hs[i]), 4), round(float(zs[i]), 4),
                 round(float(drift[i]), 4)]
                for i in range(0, int(si.size), step)]
            # depth_floor: min base_z while STABLE (tilt <= 0.3 rad, before
            # the fall) = the method's physical squat-depth limit.
            stable = tls <= STABLE_TILT_RAD
            if fall:
                stable &= tss < fall_time
            metrics["depth_floor"] = (float(zs[stable].min())
                                      if stable.any() else None)
            sat = np.flatnonzero(np.abs(zs - hs) > TRACK_SAT_ERR_M)
            metrics["track_sat_h"] = (float(hs[sat[0]]) if sat.size else None)
            d5 = np.flatnonzero(drift > DRIFT5_M)
            d20 = np.flatnonzero(drift > DRIFT20_M)
            metrics["drift5_h"] = float(hs[d5[0]]) if d5.size else None
            metrics["drift20_h"] = float(hs[d20[0]]) if d20.size else None
            metrics["max_drift_xy"] = float(drift.max())
        else:
            metrics["descent_curve"] = []
            metrics["depth_floor"] = metrics["track_sat_h"] = None
            metrics["drift5_h"] = metrics["drift20_h"] = None
            metrics["max_drift_xy"] = None
        # loop breaks right after recording the fall tick -> last sample
        metrics["fall_h_cmd"] = (float(rec["cmd_h"][-1])
                                 if fall and rec["cmd_h"] else None)
        metrics["fall_base_z"] = (float(rec["base_z"][-1])
                                  if fall and rec["base_z"] else None)
        metrics["box_kept"] = bool(box_kept)
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = box_carry_dist_max
        # success = no fall = reached the 0.10 command and held it
        # (saturated tracking is expected, NOT a failure).
        success = (not fall) and harness_error is None
        metrics["success_with_box"] = bool(success and box_kept)
    elif test == "squat_place_psi0":
        # v4 T8: lower-body error during the Psi0-coordinated place.
        z = np.array(rec["base_z"])
        h = np.array(rec["cmd_h"])
        for ph_name in ("descent", "hold", "rise"):
            idx = [i for i, p in enumerate(rec["phase"]) if p == ph_name]
            metrics["height_rmse_%s" % ph_name] = (
                rmse(h[idx], z[idx]) if idx else None)
        # root XY drift over reach-forward + place (descent+hold), relative
        # to the root position when the replay starts — the forward CoM
        # shift is exactly the decoupling stress being measured.
        dp = [i for i, p in enumerate(rec["phase"])
              if p in ("descent", "hold")]
        if dp:
            px0, py0 = rec["base_x"][dp[0]], rec["base_y"][dp[0]]
            metrics["root_drift_place"] = float(max(
                math.hypot(rec["base_x"][i] - px0, rec["base_y"][i] - py0)
                for i in dp))
        else:
            metrics["root_drift_place"] = None
        metrics["t_place_s"] = replay.t_place_s
        metrics["box_kept"] = bool(box_kept)        # carry segment only
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = box_carry_dist_max
        metrics["box_released"] = bool(box_released)
        box_place_ok = False
        metrics["box_land_dx"] = None
        metrics["box_speed_end"] = metrics["box_upright_cos"] = None
        metrics["box_robot_dist_end"] = None
        if box_released and harness_error is None \
                and np.isfinite(d.qpos).all():
            adr = ids["box_dofadr"]
            box_speed = float(np.linalg.norm(d.qvel[adr:adr + 3]))
            upright_cos = float(d.xmat[box_b].reshape(3, 3)[2, 2])
            box_robot_dist = float(math.hypot(
                d.xpos[box_b][0] - d.qpos[0], d.xpos[box_b][1] - d.qpos[1]))
            metrics["box_speed_end"] = box_speed
            metrics["box_upright_cos"] = upright_cos
            metrics["box_robot_dist_end"] = box_robot_dist
            if release_frame is not None:
                hx, hy = release_frame["heading"]
                metrics["box_land_dx"] = float(
                    d.xpos[box_b][0] * hx + d.xpos[box_b][1] * hy
                    - release_frame["s_edge"])  # + = in front of the feet
            # T8 placement gate (spec v4): box at rest, upright, landed IN
            # FRONT of the foot front edge (box_land_dx > 0). Unlike
            # pipeline_abc there is NO robot-distance-at-end criterion —
            # FALCON drifts AFTER placing (smoke: ~1.0-1.1 m at trial end)
            # and that drift is already measured by root_drift_place /
            # box_robot_dist_end, not conflated into placement success.
            box_place_ok = ((not fall)
                            and box_speed < BOX_PLACE_MAX_SPEED
                            and upright_cos > math.cos(BOX_PLACE_UPRIGHT_RAD)
                            and metrics["box_land_dx"] is not None
                            and metrics["box_land_dx"] > 0.0)
        metrics["box_place_ok"] = bool(box_place_ok)
        # stood up and stable after the place? (final 1 s stand segment)
        st = [i for i, p in enumerate(rec["phase"]) if p == "stand"]
        end_stand_ok = bool(
            (not fall) and harness_error is None and st
            and float(np.mean([rec["tilt"][i] for i in st]))
            <= STABLE_TILT_RAD
            and abs(float(np.mean([rec["base_z"][i] for i in st]))
                    - float(rec["cmd_h"][st[-1]])) < 0.1)
        metrics["end_stand_ok"] = end_stand_ok
        success = ((not fall) and harness_error is None and box_place_ok)
    elif test == "squat_pick_ground":
        # v5 T9: pick depends on squatting low enough for the wrists to
        # reach the ground box — depth saturation directly decides it.
        z = np.array(rec["base_z"])
        metrics["pick_success"] = bool(pick.grasped)
        metrics["pick_failed_timeout"] = bool(pick.pick_failed)
        metrics["grasp_dist_left"] = (pick.grasp_dists[0]
                                      if pick.grasp_dists else None)
        metrics["grasp_dist_right"] = (pick.grasp_dists[1]
                                       if pick.grasp_dists else None)
        metrics["grasp_time_s"] = pick.grasp_time
        metrics["min_root_z"] = float(np.min(z[seg])) if seg.any() else None
        bg = [i for i, p in enumerate(rec["phase"])
              if p in ("bottom", "grasp_hold")]
        if bg:
            bx0, by0 = rec["base_x"][bg[0]], rec["base_y"][bg[0]]
            metrics["root_drift"] = float(max(
                math.hypot(rec["base_x"][i] - bx0, rec["base_y"][i] - by0)
                for i in bg))
        else:
            metrics["root_drift"] = None
        metrics["box_lift_height"] = ((box_z_max - box_z0)
                                      if box_z0 is not None else None)
        metrics["box_kept"] = bool(box_kept)        # carry segment only
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = box_carry_dist_max
        # stood back up with the 2 kg load? (final 1 s stand segment)
        st = [i for i, p in enumerate(rec["phase"]) if p == "stand"]
        stand_ok = bool(
            (not fall) and harness_error is None and st
            and float(np.mean([rec["tilt"][i] for i in st]))
            <= STABLE_TILT_RAD
            and abs(float(np.mean([rec["base_z"][i] for i in st]))
                    - stand_height) < 0.1)
        metrics["stand_ok"] = stand_ok
        success = ((not fall) and harness_error is None
                   and bool(pick.grasped) and stand_ok)
    elif test == "vln_follow":
        # v5 T10: all poses in the tape-start frame (robot pose at settle
        # end) — the shared ref integrates from the identity pose.
        metrics["tape_id"] = tape.id
        xs = np.array(rec["base_x"])
        ys = np.array(rec["base_y"])
        yws = np.array(rec["yaw"])
        si = np.flatnonzero(seg)
        if si.size:
            i0 = int(si[0])
            x0, y0, yw0 = float(xs[i0]), float(ys[i0]), float(yws[i0])
            c0, s0 = math.cos(yw0), math.sin(yw0)
            lx = c0 * (xs - x0) + s0 * (ys - y0)
            ly = -s0 * (xs - x0) + c0 * (ys - y0)
            # final pose vs the ref end pose: last tick at/before tape end
            # (if the trial fell earlier this is the last live tick — the
            # fall already gates success).
            t_end = SETTLE_S + tape.duration_s
            ie = min(int(np.searchsorted(ts, t_end + 1e-9)) - 1, ts.size - 1)
            ie = max(ie, i0)
            metrics["final_pos_err"] = float(math.hypot(
                lx[ie] - tape.ref_xy[-1, 0], ly[ie] - tape.ref_xy[-1, 1]))
            metrics["final_yaw_err"] = abs(wrap_angle(
                wrap_angle(float(yws[ie]) - yw0) - float(tape.ref_yaw[-1])))
            errs = []   # 1 Hz tracking error against the ref samples
            for rt, rxy in zip(tape.ref_t, tape.ref_xy):
                if SETTLE_S + rt > float(ts[-1]) + 1e-9:
                    break   # trial ended early (fall) — no ghost samples
                ti = int(np.searchsorted(ts, SETTLE_S + rt + 1e-9)) - 1
                if ti < i0 or ti >= ts.size:
                    continue
                errs.append(math.hypot(lx[ti] - rxy[0], ly[ti] - rxy[1]))
            metrics["mean_track_err"] = (float(np.mean(errs))
                                         if errs else None)
            metrics["track_err_n"] = len(errs)
        else:
            metrics["final_pos_err"] = metrics["final_yaw_err"] = None
            metrics["mean_track_err"] = None
            metrics["track_err_n"] = 0
        # small-command response: actual fwd speed / commanded vx over the
        # |vx| in [0.05, 0.15] follow ticks (signed ratio, 1.0 = perfect)
        cvx = np.array(rec["cmd_vx"])
        vfd = np.array(rec["v_fwd"])
        ph = np.array(rec["phase"], dtype=object)
        sm = ((ph == "follow") & (np.abs(cvx) >= VLN_SMALL_VX_LO)
              & (np.abs(cvx) <= VLN_SMALL_VX_HI))
        metrics["small_cmd_response"] = (float(np.mean(vfd[sm] / cvx[sm]))
                                         if sm.any() else None)
        metrics["small_cmd_ticks"] = int(sm.sum())
        # stop_settle: residual XY displacement inside each full-stop run
        stop_disps = []
        run0 = None
        phases = rec["phase"]
        for i in range(len(phases) + 1):
            in_stop = i < len(phases) and phases[i] == "stop"
            if in_stop and run0 is None:
                run0 = i
            elif not in_stop and run0 is not None:
                j = i - 1
                stop_disps.append(math.hypot(
                    rec["base_x"][j] - rec["base_x"][run0],
                    rec["base_y"][j] - rec["base_y"][run0]))
                run0 = None
        metrics["stop_settle"] = (float(np.mean(stop_disps))
                                  if stop_disps else None)
        metrics["n_stop_segments"] = len(stop_disps)
        success = ((not fall) and harness_error is None
                   and metrics["final_pos_err"] is not None
                   and metrics["final_pos_err"] <= VLN_FINAL_POS_TOL_M
                   and metrics["final_yaw_err"] <= VLN_FINAL_YAW_TOL_RAD)
    elif test == "goto_ab":
        sc, sf = fill_nav_metrics(metrics, mission)
        # This test directly measures small-command calibration convergence:
        # trial success = fine convergence, no fall, no harness error.
        # success_coarse is reported alongside in the metrics.
        success = sf and (not fall) and harness_error is None
    elif test == "pipeline_abc":
        sc, sf = fill_nav_metrics(metrics, mission)
        metrics["squat_fall"] = (fall_phase
                                 if fall and fall_phase in PIPE_SQUAT_PHASES
                                 else None)
        metrics["box_kept"] = bool(box_kept)      # carry segment only
        metrics["box_drop_time"] = box_drop_time
        metrics["box_torso_dist_max"] = box_carry_dist_max
        metrics["box_released"] = bool(box_released)
        box_place_ok = False
        metrics["box_land_dx"] = metrics["box_land_dy"] = None
        metrics["box_land_dist"] = metrics["box_speed_end"] = None
        metrics["box_upright_cos"] = metrics["box_robot_dist_end"] = None
        if box_released and harness_error is None \
                and np.isfinite(d.qpos).all():
            adr = ids["box_dofadr"]
            box_speed = float(np.linalg.norm(d.qvel[adr:adr + 3]))
            upright_cos = float(d.xmat[box_b].reshape(3, 3)[2, 2])
            box_robot_dist = float(math.hypot(
                d.xpos[box_b][0] - d.qpos[0], d.xpos[box_b][1] - d.qpos[1]))
            metrics["box_speed_end"] = box_speed
            metrics["box_upright_cos"] = upright_cos
            metrics["box_robot_dist_end"] = box_robot_dist
            metrics["box_land_dx"] = float(d.xpos[box_b][0] - PIPE_C_XY[0])
            metrics["box_land_dy"] = float(d.xpos[box_b][1] - PIPE_C_XY[1])
            metrics["box_land_dist"] = float(math.hypot(
                metrics["box_land_dx"], metrics["box_land_dy"]))
            box_place_ok = ((not fall)
                            and box_speed < BOX_PLACE_MAX_SPEED
                            and upright_cos > math.cos(BOX_PLACE_UPRIGHT_RAD)
                            and box_robot_dist < BOX_PLACE_MAX_DIST_M)
        metrics["box_place_ok"] = bool(box_place_ok)
        metrics["t_total_s"] = float(ts[-1]) if ts.size else 0.0
        # spec v3: coarse nav + whole squat-place-rise without fall +
        # box_place_ok; fine convergence reported but NOT gated.
        success = (sc and (not fall) and harness_error is None
                   and box_place_ok)
    else:
        raise ValueError(test)

    metrics["sim_time_s"] = float(ts[-1]) if ts.size else 0.0
    metrics["wall_time_s"] = round(time.time() - wall_t0, 2)

    result = {
        "framework": LABEL,
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
    if test in WELDED_BOX_TESTS:
        result["box_mode"] = BOX_MODE
    if test in PSI0_TESTS:
        result["upper_mode"] = "psi0_replay_%s" % replay.source
    return result


# ===========================================================================
# ManipArena v2 adapter (BENCHMARK_V2_DESIGN §3-§4, §7.3). FALCON mirror of
# bench_homie.py's arena section. The shared scene (manip_scene) + mission
# state machine (manip_mission) are identical to HOMIE; this adapter binds
# them to FALCON's deployment-policy pipeline (stand flag + ref_upper_dof_pos/
# waist_dofs_command upper path + absolute base_height_command). All physics-
# repair lessons from the carry tests are reused (a-e):
#   (a) _snap_box_upright before welding (canonical relative-to-torso upright);
#   (b) box contype/conaffinity 0 while held (deep reach overlaps the table);
#   (c) release levels the box to base yaw + drops it on the placement target;
#   (d) carry vx<=0.4 + vy damp (manip_mission CARRY_VY_DAMP) + turn-before-walk;
#   (e) store placement squat kept shallow (ARENA_STORE_PLACE_H).
# ===========================================================================
def reach_base_h(top_h):
    """FALCON absolute base_height_command to bring the rubber hands to a
    surface at height ``top_h`` (table top / store-pad floor). Linear "lower
    surface -> lower base" with a fixed reach offset, clipped to the trackable
    FALCON domain [ARENA_H_MIN, ARENA_STAND_H]. The mission emits this as its
    squat target; the depth floor of the policy decides whether the low tables
    are reachable at all (the point of the H_pick sweep)."""
    return float(min(max(top_h + ARENA_REACH_OFFSET, ARENA_H_MIN),
                     ARENA_STAND_H))


def build_arena_model(repo, config, spec):
    """Compile the FALCON freebase scene with the ManipArena furniture + free
    bodies injected (mirror of build_model's temp-file path; the box/cube
    freejoints land at the qpos/qvel TAIL per FREE_BODY_QPOS_LAYOUT, so the
    robot keeps qpos[0:36]/qvel[0:35] and the obs slice is never polluted).

    Like build_model, the patched XML is written NEXT TO the original scene so
    the <include> + relative meshdir resolve, then removed after compilation.
    ms.WRIST_L/WRIST_R placeholders are replaced with the FALCON rubber-hand
    body names; the floor plane (+ world body) get bit 2 OR-ed in so the
    box/cube collide with the ground (bit-2 scheme)."""
    scene_path = os.path.normpath(
        os.path.join(repo, "sim2real", config["ROBOT_SCENE"]))
    with open(scene_path, "r") as f:
        xml = f.read()
    frags = ms.furniture_xml(spec, collision_scheme="bit2")
    xml = patch_and_replace(xml, "</worldbody>",
                            frags["bodies"] + "</worldbody>")
    xml = patch_and_replace(xml, "</mujoco>", frags["equality"] + "</mujoco>")
    # Substitute the wrist placeholders AFTER injecting the weld block (they
    # only live inside that block).
    xml = xml.replace(ms.WRIST_L_PLACEHOLDER, ARENA_WRIST_L)
    xml = xml.replace(ms.WRIST_R_PLACEHOLDER, ARENA_WRIST_R)

    tmp_path = os.path.join(os.path.dirname(scene_path),
                            "_bench_falcon_arena.xml")
    with open(tmp_path, "w") as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    model.opt.timestep = float(config["SIMULATE_DT"])
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
    """Resolve arena free-body addresses + weld equality ids and confirm the
    tail layout matches FREE_BODY_QPOS_LAYOUT. Arena welds are re-anchored at
    the live wrist<->box pose at grasp time (not pre-anchored at a hug pose)."""
    def jq(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    box_qadr, box_vadr = jq("carried_box_freejoint")
    cube_qadr, cube_vadr = jq("relay_cube_freejoint")
    lay = ms.FREE_BODY_QPOS_LAYOUT
    # robot end = 36/35 for FALCON (free 7 + 29 hinges); box then cube tail.
    assert box_qadr == 7 + N_JOINTS and cube_qadr == 7 + N_JOINTS + 7, \
        "arena free-body tail layout unexpected: box_qadr=%d cube_qadr=%d" \
        % (box_qadr, cube_qadr)
    assert (cube_qadr - box_qadr == lay["cube_qpos_offset_from_robot_end"]
            - lay["box_qpos_offset_from_robot_end"]), \
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
    collides with the ground (§3 release semantics)."""
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
    pose (no body moved). Same math as anchor_welds_at_current_pose,
    generalised to any body1/body2 pair (wrist<->box, box<->cube)."""
    rot1 = d.xmat[body1].reshape(3, 3)
    relp = rot1.T @ (d.xpos[body2] - d.xpos[body1])
    relq = quat_mul(quat_conj(d.xquat[body1]), d.xquat[body2])
    relq = relq / (np.linalg.norm(relq) or 1.0)
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = relp
    m.eq_data[eq, 6:10] = relq
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def _snap_box_upright(m, d, box, base_yaw):
    """Physics-repair (a): before welding, move the carried box into a
    canonical UPRIGHT pose — position = midpoint of the two rubber hands,
    orientation = pure yaw locked to the base (box z-axis == world z) — so the
    two wrist<->box welds anchor at mutually-CONSISTENT relposes and the box
    rides upright and lands level on release. Mutates d only."""
    bq = box["box_qadr"]
    bv = box["box_vadr"]
    mid = 0.5 * (d.xpos[box["wrist_left"]] + d.xpos[box["wrist_right"]])
    d.qpos[bq:bq + 3] = mid
    d.qpos[bq + 3:bq + 7] = [math.cos(base_yaw / 2.0), 0.0, 0.0,
                             math.sin(base_yaw / 2.0)]
    d.qvel[bv:bv + 6] = 0.0
    mujoco.mj_forward(m, d)


def _weld_box_both(m, d, box):
    """Activate BOTH wrist<->box welds at the canonical upright box pose (see
    _snap_box_upright): shares the 2 kg load across both arms with consistent
    relposes. solref/solimp left at the compliant XML defaults."""
    _weld_at_pose(m, d, box["eq_box_right"], box["wrist_right"],
                  box["box_body"])
    _weld_at_pose(m, d, box["eq_box_left"], box["wrist_left"],
                  box["box_body"])


class FalconMissionIO:
    """MissionIO (manip_mission.py Protocol) bound to a live FALCON MjModel/
    MjData + the deployment policy command slots. The mission owns sequencing
    + per-stage metrics; this adapter owns the physics side effects.

    Per policy tick the rollout does:
        cmd = mission.step(t)        # mission reads THIS io, fires welds, ..
        io.set_command(*cmd)         # store (vx,vy,wz,height)
        <rollout writes policy.lin_vel/ang_vel/stand/base_height_command,
         feeds ref_upper_dof_pos/waist_dofs_command from io.upper_target,
         runs policy.policy_action(), PD-steps the sim>

    Heights are FALCON-native ABSOLUTE base height (base_height_command); wz is
    the native yaw-rate channel; the stand flag is set by the rollout from the
    active primitive (stepping during nav, stance during squat/grasp/stand).
    The upper body replays the real_ep053 falcon29 stream: arm14 ->
    ref_upper_dof_pos, waist3 -> waist_dofs_command."""

    def __init__(self, m, d, box, ids, replay, h_stand=ARENA_STAND_H):
        self.m = m
        self.d = d
        self.box = box                  # configure_arena_welds() dict
        self.ids = ids                  # lookup_ids() dict (foot bodies, torso)
        self.replay = replay            # prepared real_ep053 namespace
        self.h_stand = h_stand
        # mutable slots the rollout reads:
        self.cmd = (0.0, 0.0, 0.0, h_stand)        # vx, vy, wz, height
        # upper_target = 17-d row [arm14 | waist3]
        self.upper_target = (replay.pos[0].copy() if replay is not None
                             else np.zeros(17))
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
        d_l = box_surface_dist(self.d, {"box_body": self.box["box_body"]},
                               self.box["wrist_left"], half=ARENA_BOX_HALF)
        d_r = box_surface_dist(self.d, {"box_body": self.box["box_body"]},
                               self.box["wrist_right"], half=ARENA_BOX_HALF)
        return (d_l, d_r)

    def right_wrist_cube_dist(self):
        return box_surface_dist(self.d, {"box_body": self.box["cube_body"]},
                                self.box["wrist_right"], half=ARENA_CUBE_HALF)

    def box_pose(self):
        b = self.box["box_body"]
        bx, by, bz = (float(v) for v in self.d.xpos[b])
        zz = float(self.d.xmat[b].reshape(3, 3)[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        speed = float(np.linalg.norm(
            self.d.qvel[self.box["box_vadr"]:self.box["box_vadr"] + 3]))
        return (bx, by, bz, tilt, speed)

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
        ('reach_down' / 'carry' / 'place'). Sets the 17-d upper_target row the
        rollout splits into ref_upper_dof_pos (arm14) + waist_dofs_command
        (waist3). No replay -> hold frame 0 (defensive; M-series always has
        one)."""
        rep = self.replay
        if rep is None:
            return
        seg = rep.arena_segments.get(segment)
        if seg is None:
            self.upper_target = rep.pos[0]
            return
        k0, k1 = seg                    # [k0, k1) frame window for this segment
        nseg = max(k1 - k0, 1)
        k = k0 + int(min(max(phase_t * RL_RATE, 0.0), nseg - 1))
        k = min(max(k, 0), rep.n - 1)
        self.upper_target = rep.pos[k]

    # --- MissionIO: act (weld / release / cube-transfer) -----------------
    def weld_box(self):
        """Magnetic grasp (physics-repair a+b). SNAP the box to a canonical
        upright pose (z up, yaw == base yaw, centred between the hands), weld
        BOTH hands at that consistent symmetric pose, then disable box
        collision (contype/conaffinity 0) while held: a deep reach snaps the
        box to the hand midpoint, which can overlap the pick table (bits 1|2)
        and spike qacc on stand-up. Collision is re-enabled (bit 2) at release."""
        base_yaw = yaw_from_quat(self.d.qpos[3:7])
        _snap_box_upright(self.m, self.d, self.box, base_yaw)
        _weld_box_both(self.m, self.d, self.box)
        self.m.geom_contype[self.box["box_geom"]] = 0
        self.m.geom_conaffinity[self.box["box_geom"]] = 0
        if hasattr(self.m, "body_contype"):
            self.m.body_contype[self.box["box_body"]] = 0
            self.m.body_conaffinity[self.box["box_body"]] = 0
        self.box_grasped = True

    def release_box(self, surface_top_h=None, target_xy=None):
        """Physics-repair (c): set the box down + drop the welds + enable box
        ground collision (bit-2 scheme: box collides with floor/furniture,
        never the bit-1 robot). The box is snapped to a clean rest pose at
        release — leveled upright (yaw == base yaw), xy ahead of the pelvis
        toward the placement target (so the landing error still reflects the
        nav accuracy, clamped inside the target footprint), z just above the
        surface so it lands flat. The cube weld (if active) stays."""
        bq = self.box["box_qadr"]
        bv = self.box["box_vadr"]
        base_yaw = yaw_from_quat(self.d.qpos[3:7])
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
                kk = (rem - ARENA_PLACE_CLAMP_R) / rem
                bx += (target_xy[0] - bx) * kk
                by += (target_xy[1] - by) * kk
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
        """Weld the cube to the box top at the live relative pose (cube rides
        the box thereafter). Also makes the cube collide via bit-2 so it never
        free-falls through a placed box later."""
        _weld_at_pose(self.m, self.d, self.box["eq_cube_box"],
                      self.box["box_body"], self.box["cube_body"])
        self.m.geom_contype[self.box["cube_geom"]] = ms.FREE_BODY_CT_CA
        self.m.geom_conaffinity[self.box["cube_geom"]] = ms.FREE_BODY_CT_CA
        self.cube_on_box = True


def prepare_arena_segments(rep):
    """Label the real_ep053 falcon29 replay into reach_down / carry / place
    frame windows the M-series segments index into. Reuses the
    squat_place_psi0 segmentation already computed at load (k_hold0/k_hold1
    around argmin(height_cmd)): 'reach_down' = descent up to the bottom hold,
    'place' = bottom hold through the rise, 'carry' = a short standing-tall tail.
    Mutates the namespace in place (adds .arena_segments) and returns it."""
    n = rep.n
    first_hold = int(rep.k_hold0)
    reach = (0, max(first_hold, 1))
    place = (first_hold, n)
    carry = (max(n - max(1, n // 10), 0), n)
    rep.arena_segments = {"reach_down": reach, "place": place, "carry": carry}
    rep.seg_bottom_k = int(np.argmin(rep.height))
    return rep


# ---------------------------------------------------------------------------
# Pillar-avoidance: insert a lateral skirt waypoint into a nav leg whose
# straight segment passes too close to the pillar (harness-side rewrite of the
# mission's NavLeg list; the shared mission engine stays physics-free).
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
    skirt NavLeg offset to the side away from the pillar. Pure geometry on the
    leg target poses; immutable (builds a fresh list)."""
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
                nx, ny = -sy / slen, sx / slen   # left-normal of the heading
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


# Mission primitive kinds that walk (stand=1, stepping); everything else is a
# stance primitive (stand=0). FALCON: stand=1 = stepping/walk, 0 = stance.
_ARENA_STEPPING_KINDS = ("nav",)


def _arena_stand_flag(mission):
    """FALCON stand flag for the CURRENT mission stage: 1 (stepping) while a
    NavLeg is active, 0 (stance) during squat/grasp/stand/place/cube. Read
    after mission.step() so it reflects the stage that issued this tick's
    command. Defaults to 0 (stance) before any stage / once finished."""
    if mission.finished or not mission.stages:
        return 0
    return 1 if mission.stages[-1].kind in _ARENA_STEPPING_KINDS else 0


def run_arena_trial(ctx, spec, trial_idx, seed, video_path=None):
    """Run M1/M2 on the ManipArena variant ``spec`` through the FALCON
    deployment-policy pipeline. Deterministic given seed. Returns a result
    dict (mission block + scene-spec provenance)."""
    test = ctx["test"]
    m = ctx["model"]
    ids = ctx["ids"]
    box = ctx["box"]
    policy = ctx["policy"]
    rep = ctx["upper"]
    n = N_JOINTS
    kp, kd, effort = ctx["kp"], ctx["kd"], ctx["effort"]
    default = ctx["default"]
    decimation = ctx["decimation"]
    sim_dt = ctx["sim_dt"]
    ctrl_dt = sim_dt * decimation

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
    init_joints = default.copy()
    # arms (qpos 15:29) + waist (qpos 12:15, [yaw, roll, pitch]) start exactly
    # at replay frame 0 so the replayed refs do not snap at t=0.
    init_joints[12:15] = rep.pos[0, 14:17]
    init_joints[15:15 + 14] = rep.pos[0, 0:14]
    upper_ref0 = rep.pos[0].copy()

    d.qpos[:] = 0.0
    d.qpos[0], d.qpos[1], d.qpos[2] = spawn[0], spawn[1], INIT_BASE_Z
    d.qpos[3:7] = [math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)]
    d.qpos[7:7 + n] = init_joints + rng.uniform(-JOINT_NOISE, JOINT_NOISE, n)
    d.qpos[7 + 12:7 + 29] = init_joints[12:29]      # replayed 17 joints exact

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
    io = FalconMissionIO(m, d, box, ids, rep, h_stand=ARENA_STAND_H)
    io.upper_target = upper_ref0
    h_pick = reach_base_h(sd["T_pick"]["top_h"])
    h_relay = reach_base_h(sd["T_relay"]["top_h"])
    h_store = ARENA_STORE_PLACE_H                    # shallow, stable store squat
    h_cube = reach_base_h(sd["T_relay"]["top_h"])    # cube sits on the relay top
    if test == "arena_M1":
        seq = mm.build_m1(sd, ARENA_STAND_H, h_pick, h_store,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    else:
        seq = mm.build_m2(sd, ARENA_STAND_H, h_pick, h_relay, h_store, h_cube,
                          vx_carry=ARENA_VX_CARRY, h_carry=ARENA_CARRY_H)
    seq = insert_pillar_detours(seq, (spawn[0], spawn[1]),
                                tuple(sd["pillar"]["pos"]))
    mission = mm.Mission(test, seq, io, ARENA_STAND_H)

    reset_policy(policy, ARENA_STAND_H, upper_ref0[0:14])
    policy.command_sender.cmd_q = init_joints.copy()

    # --- video ------------------------------------------------------------
    writer = None
    cam = None
    next_frame_t = 0.0
    if video_path is not None and ctx["renderer"] is not None:
        import cv2
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 VIDEO_FPS, (VIDEO_W, VIDEO_H))
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.elevation, cam.azimuth = 4.5, -25.0, 135.0

    rec = {k: [] for k in ("t", "base_x", "base_y", "base_z", "tilt", "phase")}
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
    max_ticks = int(round((mm.MISSION_TIMEOUT_S + 5.0) / ctrl_dt))

    try:
        for step in range(max_ticks):
            t = step * ctrl_dt

            # unified fall flag for THIS tick (mission consumes it via io)
            grav = gravity_body_frame(d.qpos[3:7])
            gxy = float(math.hypot(grav[0], grav[1]))
            tilt = float(math.acos(max(-1.0, min(1.0, -grav[2]))))
            fall_now = (gxy > TILT_FALL_GXY) or nonfoot_ground
            io.set_fallen(fall_now)
            nonfoot_ground = False

            # mission tick: reads io, fires welds/release/transfer +
            # replay_upper as side effects, returns the leg command.
            vx_c, vy_c, wz_c, h_c = mission.step(t)
            io.set_command(vx_c, vy_c, wz_c, h_c)
            stand_c = _arena_stand_flag(mission)
            phase = mission.stages[-1].name if mission.stages else "init"

            # command injection into the deployment policy
            policy.lin_vel_command = np.array([[vx_c, vy_c]])
            policy.ang_vel_command = np.array([[wz_c]])
            policy.stand_command = np.array([[stand_c]])
            policy.base_height_command = np.array([[h_c]])
            # upper-body replay: arm14 -> ref_upper_dof_pos (obs + residual),
            # waist3 -> waist_dofs_command [yaw, roll, pitch].
            row = io.upper_target
            policy.ref_upper_dof_pos = row[0:14].reshape(1, 14).copy()
            policy.waist_dofs_command = row[14:17].reshape(1, 3).copy()

            # 50 Hz policy tick via the deployment pipeline
            push_state(policy, d, n)
            policy.policy_action()
            cmd_q = policy.command_sender.cmd_q

            # PD at sim rate (auto_eval.py loop), decimation sub-steps
            for _ in range(decimation):
                q = d.qpos[7:7 + n]
                dq = d.qvel[6:6 + n]
                tau = np.clip(kp * (cmd_q - q) - kd * dq, -effort, effort)
                d.ctrl[0:6] = 0.0
                d.ctrl[6:6 + n] = tau
                mujoco.mj_step(m, d)

                for ci in range(d.ncon):
                    g1 = d.contact[ci].geom1
                    g2 = d.contact[ci].geom2
                    if floor_g in (g1, g2):
                        other = g2 if g1 == floor_g else g1
                        b = int(m.geom_bodyid[other])
                        if (b != 0 and b != pillar_b and b != box_b
                                and b != cube_b and b not in foot_bodies):
                            nonfoot_ground = True

            t = (step + 1) * ctrl_dt
            t_end = t

            # divergence guard (weld-vs-PD blowups)
            if (d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
                    or not np.isfinite(d.qpos).all()):
                harness_error = "qacc_diverged"
                break

            if writer is not None and t >= next_frame_t:
                import cv2
                cam.lookat[:] = d.qpos[0:3]
                ctx["renderer"].update_scene(d, camera=cam)
                writer.write(cv2.cvtColor(ctx["renderer"].render(),
                                          cv2.COLOR_RGB2BGR))
                next_frame_t += 1.0 / VIDEO_FPS

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
        harness_error = "exception: %r" % (e,)
    finally:
        if writer is not None:
            writer.release()

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
    metrics.update(_arena_stage_flags(mres["stages"]))

    if os.environ.get("ARENA_TRAJ_DUMP") and rec["t"]:
        for i in range(0, len(rec["t"]), 25):  # ~0.5s stride at 50 Hz
            sys.stderr.write(
                "TRAJ t=%5.1f xy=(%6.2f,%6.2f) z=%.2f tilt=%.2f phase=%s\n"
                % (rec["t"][i], rec["base_x"][i], rec["base_y"][i],
                   rec["base_z"][i], rec["tilt"][i], rec["phase"][i]))

    success = bool(mres["success"] and not fall and harness_error is None)

    return {
        "framework": LABEL,
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
        "upper_mode": "psi0_replay_%s" % rep.source,
        "metrics": metrics,
    }


def _arena_stage_flags(stages):
    """Flatten the per-stage records into the M-series diagnostic flags (nav
    arrival errors, place_ok, cube_transfer_ok, regrasp_ok, ...). One pass;
    later same-kind stages overwrite earlier ones so the names track the M2
    functional order (relay place then store place)."""
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


def run_arena_main(args, repo, config, out_dir, video_dir):
    """ManipArena entry (arena_M1 / arena_M2). Builds the variant scene, drives
    the shared Mission through the FALCON pipeline, writes results.jsonl +
    summary.json + videos. The upper body is a real_ep053 falcon29 Psi0 replay
    (required, --upper-replay)."""
    if args.upper_replay is None:
        sys.stderr.write("ERROR: --upper-replay is required for %s "
                         "(real_ep053 falcon29 npz, contract v1)\n" % args.test)
        sys.exit(2)
    if not (0 <= args.variant <= 49):
        sys.stderr.write("ERROR: --variant must be in [0, 49] (got %d)\n"
                         % args.variant)
        sys.exit(2)

    replay = load_upper_replay(os.path.expanduser(args.upper_replay))
    if replay.embodiment not in ("falcon29", "unknown"):
        raise SystemExit("upper-replay embodiment %r != falcon29 — wrong npz?"
                         % replay.embodiment)
    if not replay.source.startswith("real"):
        sys.stderr.write("WARNING: %s normally uses a real_* replay (ep053), "
                         "got %r\n" % (args.test, replay.source))

    spec = ms.build_variant(args.variant)
    print("arena: variant_seed=%d H_pick=%.2f hash=%s"
          % (spec.variant_seed, spec.H_pick, spec.scene_spec_hash[:12]))

    model = build_arena_model(repo, config, spec)
    # robot 36/35 + box freejoint (7) + cube freejoint (7) = 50 qpos / nu = 35
    assert model.nq == 7 + N_JOINTS + 14 and model.nu == 6 + N_JOINTS, \
        "unexpected arena model dims nq=%d nu=%d" % (model.nq, model.nu)
    model.opt.timestep = float(config["SIMULATE_DT"])

    replay = prepare_arena_segments(replay)
    print("arena psi0: source=%s N=%d (%.2fs); segments reach_down=%s "
          "carry=%s place=%s"
          % (replay.source, replay.n, replay.duration_s,
             replay.arena_segments["reach_down"],
             replay.arena_segments["carry"],
             replay.arena_segments["place"]))

    box_info = configure_arena_welds(model)

    sim_dt = float(config["SIMULATE_DT"])
    decimation = max(1, int(round((1.0 / RL_RATE) / sim_dt)))
    ctx = {
        "test": args.test,
        "model": model,
        "ids": lookup_ids(model, args.test),
        "box": box_info,
        "upper": replay,
        "policy": make_policy(repo, config, args.model_path
                              or os.path.join(repo, ONNX_REL)),
        "kp": np.array(config["MOTOR_KP"], dtype=np.float64),
        "kd": np.array(config["MOTOR_KD"], dtype=np.float64),
        "effort": np.array(config["motor_effort_limit_list"], dtype=np.float64),
        "default": np.array(config["DEFAULT_MOTOR_ANGLES"], dtype=np.float64),
        "sim_dt": sim_dt,
        "decimation": decimation,
        "renderer": make_renderer(model) if args.video != "none" else None,
    }

    n = args.trials if args.trials is not None else 1
    onnx_path = args.model_path or os.path.join(repo, ONNX_REL)
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
                % (LABEL, args.test, args.variant, trial_idx, tag))
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
                  % (LABEL, args.test, args.variant, trial_idx,
                     "OK" if res["success"] else "FAIL",
                     "" if not res["fall"] else
                     " fall@%.2fs(%s,%s)" % (res["fall_time"],
                                             res["fall_phase"],
                                             res["fall_reason"]),
                     "" if not res["harness_error"] else
                     " HARNESS_ERROR(%s)" % res["harness_error"],
                     mtr["mission_status"], round(mtr["sim_time_s"], 1)))

    summary = {
        "framework": LABEL,
        "test": args.test,
        "repo": repo,
        "checkpoint": onnx_path,
        "stand_height": ARENA_STAND_H,
        "variant_seed": spec.variant_seed,
        "H_pick": spec.H_pick,
        "scene_spec_hash": spec.scene_spec_hash,
        "scene_spec": spec.to_dict(),
        "box_mode": "arena_wrist_weld",
        "upper_mode": "psi0_replay_%s" % replay.source,
        "upper_replay": replay.path,
        "overall": aggregate(results),
    }
    summary["harness_error_count"] = sum(
        1 for r in results if r.get("harness_error"))
    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print("summary -> %s" % sum_path)
    print("results -> %s" % jsonl_path)


# ---------------------------------------------------------------------------
# Aggregation (identical shape to bench_homie.py)
# ---------------------------------------------------------------------------
def aggregate(results):
    out = {
        "n_trials": len(results),
        "success_rate": float(np.mean([r["success"] for r in results]))
        if results else None,
        "fall_count": int(sum(r["fall"] for r in results)),
        "harness_error_count": int(sum(1 for r in results
                                       if r.get("harness_error"))),
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
    """(trial_idx, kwargs) list — seed == trial_idx (unified spec)."""
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
        per = trials if trials is not None else SQUAT_SWEEP_TRIALS_PER_POINT
        i = 0
        for hgt in SQUAT_SWEEP_HEIGHTS:        # depth scan @ fixed 0.2 m/s
            for _ in range(per):
                plan.append((i, {"target_height": hgt,
                                 "ramp_speed": SQUAT_SWEEP_RAMP,
                                 "sweep_kind": "depth"}))
                i += 1
        for spd in SQUAT_SWEEP_SPEEDS:         # speed scan @ fixed 0.45 m
            for _ in range(per):
                plan.append((i, {"target_height": SQUAT_SWEEP_DEPTH,
                                 "ramp_speed": spd,
                                 "sweep_kind": "speed"}))
                i += 1
    elif test == "squat_limit":
        n = trials if trials is not None else SQUAT_LIMIT_TRIALS_DEFAULT
        for i in range(n):
            plan.append((i, {}))
    elif test == "squat_place_psi0":
        n = trials if trials is not None else PLACE_TRIALS_DEFAULT
        for i in range(n):
            plan.append((i, {}))
    elif test == "squat_pick_ground":
        n = trials if trials is not None else PICK_TRIALS_DEFAULT
        for i in range(n):
            plan.append((i, {}))
    elif test == "vln_follow":
        n = trials if trials is not None else VLN_TRIALS_DEFAULT
        for i in range(n):
            plan.append((i, {"tape_idx": i % n_tapes}))  # tape i%n (spec v5)
    elif test == "goto_ab":
        n = trials if trials is not None else GOTO_TRIALS_DEFAULT
        for i in range(n):
            plan.append((i, {}))
    elif test == "pipeline_abc":
        n = trials if trials is not None else PIPE_TRIALS_DEFAULT
        for i in range(n):
            plan.append((i, {}))
    else:
        raise ValueError(test)
    return plan


def make_renderer(model):
    try:
        return mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    except Exception as e:  # noqa: BLE001 — renderer failure must not kill run
        sys.stderr.write("WARNING: offscreen renderer unavailable (%s); "
                         "videos disabled.\n" % e)
        return None


def main():
    ap = argparse.ArgumentParser(description="FALCON G1 benchmark harness")
    ap.add_argument("--test", required=True,
                    choices=["walk_speed", "squat_box", "circle_pillar",
                             "speed_sweep", "squat_sweep", "goto_ab",
                             "pipeline_abc", "squat_limit",
                             "squat_pick_ground", "vln_follow",
                             "walk_speed_psi0", "squat_box_psi0",
                             "circle_pillar_psi0", "squat_place_psi0",
                             "arena_M1", "arena_M2"])
    ap.add_argument("--variant", type=int, default=0,
                    help="ManipArena (arena_M1/M2): variant_seed 0-49 "
                         "(BENCHMARK_V2_DESIGN §5; default 0). --trials N "
                         "re-runs the SAME variant N times (default 1).")
    ap.add_argument("--label", default="falcon",
                    help="framework name written into JSONL/summary/video "
                         "filenames + captions (default 'falcon'). Use "
                         "--label ljk-falcon with --model-path .../g1_29dof_v4"
                         ".onnx so the ljk-falcon run is tagged distinctly.")
    ap.add_argument("--trials", type=int, default=None,
                    help="walk/squat/circle (+_psi0): total trials "
                         "(default 50); "
                         "speed_sweep: trials PER SPEED (default 5); "
                         "squat_sweep: trials PER GRID POINT (default 5); "
                         "goto_ab: total (default 25); "
                         "pipeline_abc: total (default 15); "
                         "squat_limit: total (default 10, --video all "
                         "recommended — every trial is a calibration "
                         "sample); squat_place_psi0: total (default 25); "
                         "squat_pick_ground: total (default 15); "
                         "vln_follow: total (default 10, tape = i %% n_tapes)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--upper-replay", default=None,
                    help="psi0 tests: upper_replay_falcon29_*.npz (contract "
                         "v1; 4090:/hhd2/ljk/g1bench/psi0_replay/)")
    ap.add_argument("--tapes", default=None,
                    help="vln_follow (REQUIRED there): tapes.json from "
                         "make_vln_tapes.py — the same file is fed to all "
                         "four harnesses")
    ap.add_argument("--video", choices=["none", "policy", "all"],
                    default="policy",
                    help="policy = first %d successful (+ first %d for "
                         "pipeline_abc) + all failed trials"
                         % (MAX_OK_VIDEOS, PIPE_MAX_OK_VIDEOS))
    ap.add_argument("--repo", default=None, help="path to FALCON checkout")
    ap.add_argument("--model-path", default=None,
                    help="ONNX checkpoint (default <repo>/%s)" % ONNX_REL)
    args = ap.parse_args()

    global LABEL
    LABEL = args.label

    # ManipArena M-series: separate entry (shared scene + mission engine).
    if args.test in ARENA_TESTS:
        repo = find_repo(args.repo)
        import yaml
        with open(os.path.join(repo, CONFIG_REL)) as f:
            config = yaml.safe_load(f)
        out_dir = args.out_dir or os.path.join("falcon_results", args.test)
        video_dir = os.path.join(out_dir, "videos")
        os.makedirs(out_dir, exist_ok=True)
        if args.video != "none":
            os.makedirs(video_dir, exist_ok=True)
        run_arena_main(args, repo, config, out_dir, video_dir)
        sys.exit(0)

    replay = None
    if args.test in PSI0_TESTS:
        if args.upper_replay is None:
            sys.stderr.write(
                "ERROR: --upper-replay is required for %s "
                "(upper_replay_falcon29_*.npz, contract v1)\n" % args.test)
            sys.exit(2)
        replay = load_upper_replay(os.path.expanduser(args.upper_replay))
        if replay.embodiment not in ("falcon29", "unknown"):
            sys.stderr.write(
                "ERROR: upper-replay embodiment %r != falcon29 — wrong npz?\n"
                % replay.embodiment)
            sys.exit(2)
        want_src = "sim" if args.test == "squat_box_psi0" else "real"
        if not replay.source.startswith(want_src):
            print("WARNING: %s normally uses a %s_* replay source, got %r"
                  % (args.test, want_src, replay.source))
        print("psi0 replay: %s source=%s K=%d N=%d (%.2fs, grasp_close_t=%.2fs)"
              % (replay.path, replay.source, len(replay.names), replay.n,
                 replay.duration_s, replay.grasp_close_t))
    elif args.upper_replay is not None:
        print("WARNING: --upper-replay ignored (not a psi0 test)")

    tapes = None
    if args.test == "vln_follow":
        if args.tapes is None:
            sys.stderr.write(
                "ERROR: --tapes is required for vln_follow "
                "(tapes.json from make_vln_tapes.py)\n")
            sys.exit(2)
        tapes = load_vln_tapes(os.path.expanduser(args.tapes))
        print("vln tapes: %s n=%d durations %s s"
              % (os.path.abspath(os.path.expanduser(args.tapes)), len(tapes),
                 sorted({round(tp.duration_s, 1) for tp in tapes})))
    elif args.tapes is not None:
        print("WARNING: --tapes ignored (not vln_follow)")

    repo = find_repo(args.repo)
    import yaml
    with open(os.path.join(repo, CONFIG_REL)) as f:
        config = yaml.safe_load(f)
    onnx_path = args.model_path or os.path.join(repo, ONNX_REL)

    out_dir = args.out_dir or os.path.join("falcon_results", args.test)
    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    if args.video != "none":
        os.makedirs(video_dir, exist_ok=True)

    model = build_model(repo, config, args.test)
    assert model.nu == 6 + N_JOINTS, "unexpected nu=%d" % model.nu

    sim_dt = float(config["SIMULATE_DT"])
    decimation = max(1, int(round((1.0 / RL_RATE) / sim_dt)))
    ctx = {
        "test": args.test,
        "model": model,
        "ids": lookup_ids(model, args.test),
        "policy": make_policy(repo, config, onnx_path),
        "kp": np.array(config["MOTOR_KP"], dtype=np.float64),
        "kd": np.array(config["MOTOR_KD"], dtype=np.float64),
        "effort": np.array(config["motor_effort_limit_list"], dtype=np.float64),
        "default": np.array(config["DEFAULT_MOTOR_ANGLES"], dtype=np.float64),
        "stand_height": float(config.get("DESIRED_BASE_HEIGHT", 0.75)),
        "sim_dt": sim_dt,
        "decimation": decimation,
        "renderer": make_renderer(model) if args.video != "none" else None,
        "replay": replay,
        "tapes": tapes,
    }
    if args.test in PSI0_TESTS or args.test == "squat_pick_ground":
        # Unified bit-2 scheme (psi0 spec, reused by v5 squat_pick_ground):
        # box (bit 2) collides with table/floor, never the robot (bit 1).
        # The floor ships with contype/conaffinity 1 — OR in bit 2 (model
        # fields are runtime-writable).
        fg = ctx["ids"]["floor_geom"]
        model.geom_contype[fg] |= 2
        model.geom_conaffinity[fg] |= 2
        # MuJoCo >= 3.2.4 also culls body pairs via the compile-time
        # body_contype/body_conaffinity aggregates; OR bit 2 into the
        # floor's body so the box (body bits 2/2 from its geom) can land.
        if hasattr(model, "body_contype"):
            fb = int(model.geom_bodyid[fg])
            model.body_contype[fb] |= 2
            model.body_conaffinity[fb] |= 2
        print("floor bit-2 collision enabled (geom id %d)" % fg)

    plan = build_trial_plan(args.test, args.trials,
                            n_tapes=len(tapes) if tapes else None)
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
            res = run_trial(ctx, trial_idx, seed, video_path=tmp_video, **kw)

            tag = "ok" if res["success"] else "fail"
            final_video = os.path.join(
                video_dir, "%s_%s_t%02d_%s.mp4"
                % (LABEL, args.test, trial_idx, tag))

            if render_first_pass and tmp_video and os.path.exists(tmp_video):
                os.replace(tmp_video, final_video)
            elif (args.video == "policy" and ctx["renderer"] is not None
                  and (not res["success"] or ok_videos_saved < max_ok_videos)):
                # Deterministic re-run with rendering (physics unchanged).
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
            extra = ""
            if res["fall"]:
                extra = " fall@%.2fs(%s,%s)" % (res["fall_time"],
                                                res["fall_phase"],
                                                res["fall_reason"])
            if res.get("harness_error"):
                extra += " HARNESS_ERROR(%s)" % res["harness_error"]
            print("[%s/%s] trial %02d seed %d -> %s%s  (%ss sim, %ss wall)"
                  % (LABEL, args.test, trial_idx, seed,
                     "OK" if res["success"] else "FAIL", extra,
                     round(res["metrics"]["sim_time_s"], 1),
                     res["metrics"]["wall_time_s"]))

    # ------------------ summary --------------------------------------------
    summary = {
        "framework": LABEL,
        "test": args.test,
        "repo": repo,
        "checkpoint": onnx_path,
        "stand_height": ctx["stand_height"],
        "squat_height": (SQUAT_HEIGHT
                         if args.test in ("squat_box", "squat_box_psi0")
                         else PIPE_SQUAT_HEIGHT
                         if args.test == "pipeline_abc" else None),
        "overall": aggregate(results),
    }
    if args.test in WELDED_BOX_TESTS:
        summary["box_mode"] = BOX_MODE
        # squat_place_psi0 / squat_pick_ground use the 2 kg bench box,
        # the cracker tests the 0.411 kg YCB box
        summary["box_mass_kg"] = (PSI0_BOX_MASS_KG
                                  if args.test in PSI0_CRACKER_TESTS
                                  else BOX_MASS)
    if args.test in PSI0_TESTS:
        summary["upper_mode"] = "psi0_replay_%s" % replay.source
        summary["upper_replay"] = replay.path
        summary["upper_replay_grasp_close_t"] = replay.grasp_close_t
    if args.test == "squat_place_psi0":
        summary["upper_replay_t_place_s"] = replay.t_place_s
    if args.test in ("squat_box_psi0", "squat_pick_ground"):
        picks = [r["metrics"].get("pick_success") for r in results
                 if isinstance(r["metrics"].get("pick_success"), bool)]
        summary["pick_success_rate"] = (float(np.mean(picks))
                                        if picks else None)
    if args.test == "vln_follow":
        summary["tapes_path"] = os.path.abspath(
            os.path.expanduser(args.tapes))
        summary["n_tapes"] = len(tapes)
        summary["by_tape"] = {
            str(tp.id): aggregate([r for r in results
                                   if r["metrics"].get("tape_id") == tp.id])
            for tp in tapes
        }
    if args.test == "squat_sweep":
        # Directly answers "which heights fall over, how big is root drift":
        # success rate / achieved_depth / root_drift stats per group.
        summary["by_target_height"] = {
            "%.2f" % h: aggregate(
                [r for r in results
                 if r["metrics"].get("sweep_kind") == "depth"
                 and r["metrics"].get("target_height") == h])
            for h in SQUAT_SWEEP_HEIGHTS
        }
        summary["by_ramp_speed"] = {
            "%.1f" % s: aggregate(
                [r for r in results
                 if r["metrics"].get("sweep_kind") == "speed"
                 and r["metrics"].get("ramp_speed") == s])
            for s in SQUAT_SWEEP_SPEEDS
        }
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
