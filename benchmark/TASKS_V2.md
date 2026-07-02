# TASKS_V2 — machine-readable task table (ManipArena, L/M series)

> Companion to `BENCHMARK_V2_DESIGN.md` (§4 task redefinition). This file
> turns each L1–L5 / M1 / M2 into a precise, implementation-ready spec:
> SceneSpec fields consumed, command profile, stage machine, per-stage
> metrics, success criterion, JSONL schema, default trials. The stage
> machines for M1/M2 are realised by `scripts/manip_mission.py`
> (`build_m1` / `build_m2`); L1–L5 reuse the existing per-test command
> profiles in the harnesses, retargeted to SceneSpec coordinates.
>
> Per-task parameters (trials, box/Psi0 usage, variant count) are mirrored
> in machine-readable form in `scripts/task_registry.json`.

---

## 0. Conventions

- **variant_seed** 0–49 selects every jitter + the H_pick tier
  (`seed mod 4 -> {0.30, 0.45, 0.60, 0.75} m`). All four models run the
  same 50 seeds (BENCHMARK_V2_DESIGN §5).
- **Pose** = `(x, y, yaw)`, world frame, metres / radians. Heights are the
  harness's own base-height command semantics (HOMIE absolute base z,
  stand=0.74; AMO/AGILE/FALCON relative — each harness's adapter maps the
  mission's `h_*` targets into its own command).
- **Nav tolerances** (shared, spec-identical): coarse 0.30 m & 15°, CAL
  fine 0.05 m & 5° held 1.0 s. Math in `nav_errors / nav_cmd_coarse /
  nav_cmd_cal`.
- **Grasp / place / cube thresholds**: `d_grasp=0.30 m`, `cube_touch_d=0.15 m`,
  box place_ok = upright (<30°) ∧ at rest (<0.05 m/s) ∧ within 0.20 m of
  target footprint.
- **Fall** (unified): tilt gravity-xy > 0.9, or non-foot ground contact;
  record stage + time.

---

## 1. L-series — locomotion primitives (no box, no Psi0)

These are the R1–R5 walk/circle/goto/vln/squat_limit profiles moved into the
arena and retargeted to SceneSpec. They keep the existing harness command
profiles (`command_at`, `NavMission`, squat-limit ramp); only coordinates
change to reference the SceneSpec. No box, no upper-body replay.

### L1 — walk (speed sweep)

| key | value |
|---|---|
| scene fields | `lane` (start at `spawn`, +X 4 m straight) |
| cmd profile | vx sweep ∈ {0.4, 0.6, 0.8, 1.0, 1.2} m/s, ramp `WALK_RAMP_S` then hold, 10 s; vy=wz=0; height=stand |
| stage machine | `settle → ramp → hold` (existing `command_at("walk_speed")`) |
| per-stage metrics | `mean_v_fwd`, `v_track_ratio = mean_v_fwd / vx_target`, `max_tilt`, `max_drift_y` |
| success | no fall ∧ `v_track_ratio ≥ 0.9` |
| trials | 5 speed bins × variants (arena is fixed background) |

### L2 — circle (around pillar)

| key | value |
|---|---|
| scene fields | `pillar{pos,r,h}` (circle radius 1 m about pillar) |
| cmd profile | vx=0.4, wz=±0.4 rad/s, 2 laps; height=stand |
| stage machine | `settle → circle` (existing `command_at("circle_pillar")`, lap accounting) |
| per-stage metrics | per-lap `pos_err_m`, `yaw_err_rad`, `radial_err = |dist_to_pillar − 1.0|`, `max_tilt` |
| success | no fall ∧ no pillar contact ∧ radial_err ≤ 0.15 m |
| trials | 1 per variant (×2 directions optional) |

### L3 — goto (precise nav)

| key | value |
|---|---|
| scene fields | `spawn`, `targets.P_pick_front` |
| cmd profile | two-phase: coarse NAV → CAL (≤0.10 cmd) to 5 cm/5° |
| stage machine | `settle → nav(coarse) → cal(fine) → end_stand` (existing `NavMission`; `nav_errors/nav_cmd_coarse/nav_cmd_cal`) |
| per-stage metrics | `coarse_err_pos/yaw`, `coarse_reached`, `cal_err_pos/yaw`, `cal_converged`, `cal_t` |
| success | CAL converged (5 cm & 5° held 1 s) ∧ no fall |
| trials | 1 per variant |

### L4 — vln (tape-following traversal)

| key | value |
|---|---|
| scene fields | full arena (avoid `pillar`); shared VLN tape (`make_vln_tapes.py`) |
| cmd profile | zero-order-hold replay of tape breakpoints through native vx/vy/wz |
| stage machine | `settle → follow → end_stand` (existing `command_at("vln_follow")`) |
| per-stage metrics | `path_track_err` (vs tape-implied path), small-cmd response, `max_tilt`, `min_pillar_clearance` |
| success | no fall ∧ tape completed ∧ no pillar contact |
| trials | 1 per variant per tape |

### L5 — squat_limit (descent-limit calibration, in-place, carrying box)

| key | value |
|---|---|
| scene fields | none (in-place at `spawn`); carries the 2 kg box welded from t=0 (hug pose) |
| cmd profile | continuous 0.05 m/s descent from stand to 0.10 m, then hold 3 s; **not** clipped to training domain |
| stage machine | `settle → descend → bottom_hold` (existing `command_at("squat_limit")`) |
| per-stage metrics | `descent_curve` [[h_cmd, base_z, drift_xy] @10 Hz], `depth_floor`, `track_sat_h`, `drift5_h`, `drift20_h`, `fall_h_cmd`, `fall_base_z`, `box_kept` |
| success | no fall (reached & held 0.10) — calibration task, box_kept recorded not gating |
| trials | 1 per variant (every trial is a calibration sample) |

> L5 carries the box (the v2 wrist-weld load) but does **not** use Psi0
> replay (hug arms held by a fixed carry pose) — so it is "box=yes, Psi0=no".

---

## 2. M-series — manipulation tasks (Psi0 upper body, full scene)

Both M-tasks are driven by `manip_mission.Mission` over the primitive
sequence from `build_m1` / `build_m2`. Lower body = tested policy (consumes
`send_leg_cmd`); upper body = Psi0 replay segments (`reach_down / carry /
place`); box / cube via weld callbacks.

**Pillar avoidance** (BENCHMARK_V2_DESIGN §4 M1 step 1 "避 pillar", M2 step 2
"绕去 P_relay_front"): primary nav legs are threaded with `pillar{pos,r}` from
the SceneSpec. At leg start a leg whose straight (live root pose)→target
segment passes within `PILLAR_AVOID_CLEARANCE_M` (0.40 m) of the pillar centre
inserts ONE lateral detour waypoint that bows the path away from the pillar
(offset `pillar.r + clearance + margin`), coarse-routes there first, then
proceeds to the target. This is a **runtime** insertion — it does **not**
change the builder signatures (pillar comes from the already-present `scene`
arg) or the `n_primitives` count (8 / 20), and the cross-harness
`nav_errors / nav_cmd_coarse / nav_cmd_cal` P-law is unchanged. Pre-planned
skirt waypoints opt out via `NavLeg.skip_detour`.

### M1 — table_pick_place (BENCHMARK_V2_DESIGN §4 M1)

**SceneSpec fields used**: `spawn`, `T_pick{pos,top_h,size}`, `H_pick`,
`box{pos,size,mass}`, `Z_store{pos,size,rim_h}`, `pillar`,
`targets.P_pick_front`, `targets.P_store`, `grasp.box_grasp_h`,
`grasp.d_grasp`.

**Command profile**: nav legs use the two-phase controller; squat legs ramp
to the harness's `h_pick` (reaches box on T_pick at H_pick) / `h_store`
(reaches Z_store floor) at 0.20 m/s; carry nav uses `vx_carry=0.5`.

**Stage machine** (5 functional stages → 8 primitives, `build_m1`):

| # | primitive | stage name | upper seg | metric hook |
|---|---|---|---|---|
| 1 | NavLeg(P_pick_front, cal) | nav#0 | — | coarse/cal err pos+yaw, cal_converged |
| 2 | SquatTo(h_pick) | squat#1 | reach_down | h_dev, h_achieved_min, root_drift_m, max_tilt |
| 3 | Grasp(d_grasp) | grasp#2 | reach_down | wrist_dist_at_event, grasp_ok |
| 4 | StandTo(h_stand) | stand#3 | carry | max_tilt |
| 5 | NavLeg(P_store, vx_carry, cal) | nav#4 | carry | coarse/cal err, cal_converged |
| 6 | SquatTo(h_store) | squat#5 | place | h_dev, root_drift_m, max_tilt |
| 7 | PlaceOn(store) | place#6 | place | place_ok, box_land_pos, box_land_err_m, box_tilt_end_rad, box_speed_end |
| 8 | StandTo(h_stand) | stand#7 | carry | max_tilt |

**Success**: `status == "done"` ∧ all stages ok ∧ box upright/at-rest within
Z_store tolerance ∧ no fall. (Engine `Mission.result()["success"]`.)
**Cross-50 report**: success rate stratified by `H_pick` tier (answers "how
low must the table be before the box can't be picked up").
**Trials**: 50 (1 per variant).

### M2 — relay_pick_place (BENCHMARK_V2_DESIGN §4 M2) ← hardest

**SceneSpec fields used**: all of M1 plus `T_relay{pos,top_h,size}`,
`cube{pos,size,mass}`, `targets.P_relay_front`, `targets.P_cube_touch`,
`grasp.cube_touch_d`.

**Stage machine** (7 functional stages → 20 primitives, `build_m2`):

| # | primitive | stage name | upper seg | metric hook |
|---|---|---|---|---|
| 1 | NavLeg(P_pick_front, cal) | nav#0 | — | nav err, cal_converged |
| 2 | SquatTo(h_pick) | squat#1 | reach_down | h_dev, root_drift_m, max_tilt |
| 3 | Grasp(d_grasp) | grasp#2 | reach_down | grasp_ok, wrist_dist_at_event |
| 4 | StandTo(h_stand) | stand#3 | carry | max_tilt |
| 5 | NavLeg(P_relay_front, vx_carry, cal) | nav#4 | carry | nav err, cal_converged |
| 6 | SquatTo(h_relay) | squat#5 | place | h_dev, max_tilt |
| 7 | PlaceOn(relay) | place#6 | place | place_ok, box_land_err_m |
| 8 | StandTo(h_stand) | stand#7 | carry | max_tilt |
| 9 | NavLeg(P_cube_touch, vx_carry, cal) | nav#8 | carry | nav err (precise — must reach to touch cube) |
| 10 | SquatTo(h_cube) | squat#9 | reach_down | h_dev, max_tilt |
| 11 | CubeTransfer(cube_touch_d) | cube#10 | reach_down | cube_transfer_ok, right_wrist_cube_dist |
| 12 | StandTo(h_stand) | stand#11 | carry | max_tilt |
| 13 | NavLeg(P_relay_front, vx_carry, cal) | nav#12 | carry | nav err |
| 14 | SquatTo(h_relay) | squat#13 | reach_down | h_dev, max_tilt |
| 15 | Grasp(d_grasp) | grasp#14 | reach_down | grasp_ok (re-grasp box+cube) |
| 16 | StandTo(h_stand) | stand#15 | carry | max_tilt |
| 17 | NavLeg(P_store, vx_carry, cal) | nav#16 | carry | nav err, cal_converged |
| 18 | SquatTo(h_store) | squat#17 | place | h_dev, max_tilt |
| 19 | PlaceOn(store) | place#18 | place | place_ok, box_land_err_m, box_tilt_end_rad |
| 20 | StandTo(h_stand) | stand#19 | carry | max_tilt |

**Success**: box (+ cube) finally in Z_store ∧ all 20 stages ok ∧ no fall.
**Per-stage diagnostic flags** (the long-chain dimension, §4 M2): nav arrival
errors, place_ok (relay), cube_transfer_ok, re-grasp ok, final place_ok,
per-stage fall stamp.
**Trials**: 50 (1 per variant).

> Mapping note: §4 M2 "step 6 re-grasp box" is realised as
> nav→relay-front → squat → Grasp (primitives 13–15): the box rests on
> T_relay after step 3, so re-acquisition requires returning to the relay
> table and welding again. If a harness implements step 5 cube-transfer
> *without* first placing the box (box stays welded through the cube touch),
> drop primitives 6–8 + 13–16 and use a single `Grasp` after CubeTransfer —
> `build_m2` documents the place-then-regrasp interpretation, which matches
> the §4 text "在 T_relay 放下箱 … 重新抱起箱".

---

## 3. JSONL schema (one line per trial)

Every trial writes one JSON object. Top level mirrors the existing harness
result (`framework / test / trial / seed / success / fall / metrics`) plus
the v2 arena fields:

```jsonc
{
  "framework": "homie|amo|agile|falcon",
  "task": "arena_L1|...|arena_M1|arena_M2",
  "variant_seed": 0,                  // 0..49
  "H_pick": 0.45,                     // m, the controlled variable
  "scene_spec_hash": "sha1:...",      // reproduce the scene exactly
  "scene_spec": { /* full SceneSpec, BENCHMARK_V2_DESIGN §2 */ },
  "trial": 0,
  "success": true,
  "fall": false,
  "fall_time": null,                  // sim s, or float
  "fall_stage": null,                 // stage name, M-series
  "fall_reason": null,                // "tilt"|"low_height"|"ground_contact"
  "status": "done",                   // done|fall|timeout|fail (M-series)
  "box_mode": "wrist_weld_v2",        // present if box used
  "upper_mode": "psi0_replay_real_ep053", // present if Psi0 used
  "metrics": {
    "total_t": 21.9,
    "sim_time_s": 21.9,
    "wall_time_s": 4.2,
    // M-series only: mission + per-stage block
    "mission": "arena_M2",
    "n_stages": 20,
    "stages": [
      { "name": "nav#0->1.05,0.00", "kind": "nav", "t_start": 2.0,
        "t_end": 6.1, "ok": true, "coarse_err_pos": 0.21,
        "coarse_err_yaw": 0.08, "coarse_reached": true,
        "cal_err_pos": 0.04, "cal_err_yaw": 0.03, "cal_converged": true },
      { "name": "squat#1:h0.45", "kind": "squat", "ok": true,
        "h_target": 0.45, "h_achieved_min": 0.47, "h_dev": 0.02,
        "root_drift_m": 0.03, "max_tilt": 0.12 },
      { "name": "grasp#2", "kind": "grasp", "ok": true,
        "wrist_dist_at_event": [0.10, 0.11], "grasp_ok": true }
      // ... one entry per stage
    ]
  }
}
```

Stage-record field set is the `StageRecord.to_dict()` projection in
`manip_mission.py` (only non-null fields emitted). L-series trials reuse the
existing flat metric keys (`mean_v_fwd`, `descent_curve`, `laps`, …) under
`metrics` and omit the `stages[]` block.

---

## 4. Defaults summary

| task | trials/model | box | Psi0 | variants | stage engine |
|---|---|---|---|---|---|
| arena_L1 walk | 5 (speed bins) × variants | no | no | 50 | harness `command_at` |
| arena_L2 circle | 1 × variants | no | no | 50 | harness `command_at` |
| arena_L3 goto | 1 × variants | no | no | 50 | harness `NavMission` |
| arena_L4 vln | 1 × variants × tapes | no | no | 50 | harness `command_at` |
| arena_L5 squat_limit | 1 × variants | yes (hug) | no | 50 | harness `command_at` |
| arena_M1 | 50 (1/variant) | yes | yes | 50 | `manip_mission.build_m1` |
| arena_M2 | 50 (1/variant) | yes | yes | 50 | `manip_mission.build_m2` |
