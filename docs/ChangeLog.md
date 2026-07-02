# ChangeLog

## 2026-06-17 box_demo_2 AGILE real-robot lower-body pipeline

Added a hardware-facing AGILE lower-body route for `box_demo_2` without changing
the camera/VLM/SAM3/IK/arm_sdk upper-body manipulation flow.

### What Changed

- `box_demo_2/agile_lowcmd_pipeline.py`
  - Loads WBC-AGILE `velocity_height_g1` policy/config.
  - Reads `/tmp/robojudo_ext_cmd.json`.
  - Feeds AGILE `[vx, vy, wz, absolute_height]`.
  - Publishes `rt/lowcmd_rl`, not final `rt/lowcmd`.
  - Holds upper motors `12..28` at measured positions so upper-body motion still
    comes from `rt/arm_sdk`.
  - Uses Unitree IMU gyroscope as body-frame angular velocity; no extra
    world/body rotation is applied.

- `box_demo_2/agile_keyboard_control.py`
  - Adds a passive IPC keyboard controller.
  - Keys: `w/s/a/d` translation, `q/e` yaw, `z/x` height down/up,
    `c` pick-height preset, `r` stand height, `space` stop, `l/f` lower/full.
  - Height is clamped to `[0.40, 0.72]m`; default step is `0.02m`.

- `box_demo_2/start_agile_box.sh`
  - New tmux launcher for merge + AGILE pipeline + keyboard + existing
    `box_demo_main.py`.
  - The old `box_demo_2/start.sh` still launches the original RoboJuDo
    `run_pipeline.py`; use `start_agile_box.sh` for AGILE.

- `box_demo_2/robot_move.py`
  - Extends the existing IPC command with optional `height`.
  - Existing velocity/fsm fields remain backward-compatible.

- `box_demo_2/AGILE_DEPLOY.md`
  - Documents architecture, key bindings, height-control strategy, launch
    command, and safety notes.

### Verification

- Local syntax checks:
  - `python3 -m py_compile` for the modified/new Python files.
  - `bash -n` for `start_agile_box.sh`.
- Entry-point checks:
  - `agile_keyboard_control.py --help`.
  - `agile_lowcmd_pipeline.py --help` after lazy-loading ML dependencies.
- 4090 AGILE environment smoke:
  - Loaded WBC-AGILE policy/config.
  - Built 29-joint observation/action stack.
  - Verified `obs_dim=80`, `action_dim=12`, `MLPPolicyWrapper`, and two recurrent
    buffers reset.

This smoke does not exercise DDS or real motors; the real-robot bring-up still
needs to start with `--dry-run`, then publish `rt/lowcmd_rl` with the robot on
support or with a human ready to cut power.

## 2026-06-16 AGILE + box_demo_2 upper-body integration

### Summary

Added an `AGILE-BOXDEMO` benchmark path that keeps AGILE as the lower-body
locomotion policy and replaces the Psi0/constant upper-body source with the
`box_demo_2` upper-body logic.

The integration is intentionally scoped to the benchmark harness:

- Lower body remains AGILE `velocity_height` policy.
- R2/R3 carry-style tests use the `box_demo_2` RL-lower handoff/carry upper pose.
- R3 `*_psi0` compatibility tests use a replay-like scripted sequence generated
  from `box_demo_2` `ArmKinematics`, so no `--upper-replay` npz is required.
- R6 `arena_M1/M2` uses an online `BoxDemoUpperProvider` that computes IK from
  live arena box/cube state through the existing `MissionIO.replay_upper()`
  contract.

### Changed Files

- `experiments/scripts/bench_agile.py`
  - Added `--upper-mode {psi0,box_demo}`.
  - Added `--label`, used for JSON/video filename prefixes such as
    `agile-boxdemo_*`.
  - Added a lightweight loader for the pure-kinematics section of
    `box_demo_2/dual_arm_target_reach.py`; this avoids importing Unitree DDS
    dependencies in benchmark environments.
  - Added scripted box_demo replay-like upper streams for legacy R3 replay tests.
  - Added online `BoxDemoUpperProvider` for ManipArena M-series.
  - Tagged JSONL rows/summaries with `upper_mode=box_demo_2_ik` only when the
    upper override is actually active.

- `experiments/scripts/caption_videos.py`
  - Caption text now distinguishes `box_demo_2 IK upper` from `Psi0 traj replay`.

- `experiments/console/make_manifest.py`
  - Added `agile-boxdemo` to the scanned model list.
  - Added `AGILE-BOXDEMO` harness notes.
  - Classified `agile-boxdemo` legacy rows as R2-family rows.

### Remote Smoke Test

Ran on `4090:/sda/lizhe/g1bench` with:

```bash
/sda/lizhe/miniforge3/envs/agile/bin/python bench_agile.py \
  --agile-repo /sda/lizhe/g1bench/WBC-AGILE \
  --mjcf /sda/lizhe/g1bench/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --device cpu \
  --upper-mode box_demo \
  --label agile-boxdemo
```

Smoke output root:

```text
/sda/lizhe/g1bench/boxdemo_smoke_20260616
```

Coverage:

- R2 smoke: `walk_speed`, `squat_box`, `circle_pillar`, `speed_sweep`, 1 trial each.
- R3 smoke: `walk_speed_psi0`, `squat_box_psi0`, `circle_pillar_psi0`,
  `goto_ab`, `pipeline_abc`, plus `squat_sweep` custom single point.
- R6 smoke: `arena_M1` and `arena_M2`, variant `v00`, 1 trial each.

Key results:

- R2 `squat_box`: success, 20/20 cycles, box kept, no fall.
- R3 `squat_box_psi0`: success, box_demo IK grasp succeeded, box kept, no fall.
- R3 `pipeline_abc`: success, box placed, box kept, no fall.
- R6 `arena_M1 v00`: mission done, grasp/place done, no fall.
- R6 `arena_M2 v00`: chain stages all completed (`grasp`, `relay`, `cube`,
  `regrasp`, `store` all true), no fall. Raw top-level success is false because
  one intermediate nav-cal stage missed the strict 5cm/5deg hold gate; console
  reports the chain completion count.

Notes:

- This was a 1-trial smoke test, not the full 50-variant/50-trial benchmark.
- The first smoke pass used `--video none` for fast validation.

### Video Pass

After video inspection was requested, reran the same R2/R3/R6 smoke coverage with
`--video policy`.

Video output root:

```text
/sda/lizhe/g1bench/boxdemo_video_20260616
```

Captioned videos:

```text
/sda/lizhe/g1bench/results/videos_r2plus/agile-boxdemo
/sda/lizhe/g1bench/site/videos/agile-boxdemo
/sda/lizhe/g1bench/site/videos/arena/agile-boxdemo
```

Video-pass status:

- 12 captioned AGILE-BOXDEMO videos were generated: 4 R2, 6 R3, 2 R6.
- No falls were observed in the 1-trial video rows.
- R2 `walk_speed` and `speed_sweep`, plus R3 `walk_speed_psi0`, still fail
  their velocity criteria, but they do not fall.
- R6 `arena_M2 v00` still has raw top-level `success=false` because one strict
  nav-cal hold gate misses; the chain stages complete (`grasp`, `relay`,
  `cube`, `regrasp`, `store`) and there is no fall.
- `bash build_site.sh` passed with `site video check: 101/101 present, 0 missing`.

### Console Sync

Synced smoke JSONL into:

```text
/sda/lizhe/g1bench/results/agile-boxdemo
/sda/lizhe/g1bench/results/final/agile-boxdemo
```

Merged R6 into:

```text
/sda/lizhe/g1bench/console/arena_agg.json
```

Regenerated and built the console site:

```bash
cd /sda/lizhe/g1bench/console
/sda/lizhe/miniforge3/envs/agile/bin/python merge_arena_model.py \
  agile-boxdemo /sda/lizhe/g1bench/boxdemo_smoke_20260616/arena
/sda/lizhe/miniforge3/envs/agile/bin/python make_manifest.py --site
bash build_site.sh
```

Verified `manifest.json` contains 12 `AGILE-BOXDEMO` experiment rows:

- R2: 4 rows.
- R3: 6 rows.
- R6-Arena: 2 rows.

The built site manifest is at:

```text
/sda/lizhe/g1bench/site/manifest.json
```
