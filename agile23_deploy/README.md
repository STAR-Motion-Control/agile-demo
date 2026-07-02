# AGILE 23-DoF Sim2Real Deploy (box_demo_2 / RoboJuDo wire line)

Run the AGILE velocity-height **recurrent LSTM student** (12-DoF leg action, squat+walk) on
the real 23-DoF basic G1, conforming to the box_demo_2 / RoboJuDo wire line, with keyboard
teleop. RoboJuDo's `start.sh` stays a **hot fallback** — this does NOT run inside RoboJuDo's
closed wheel; it publishes `rt/lowcmd_rl` and lets `merge_lowcmd_arm_sdk.py` remain the sole
`rt/lowcmd` publisher.

## Files
| file | role |
|---|---|
| `agile23_lowcmd_pipeline.py` | main runtime: `rt/lowstate` → AGILE `ObservationProcessor`→LSTM student→`ActionProcessor` (verbatim wrap) → `rt/lowcmd_rl` (drive legs 0-11, hold 12-28 at measured q, slot29=0). Safety guards: NaN/tilt/clamp+rate-limit/overspeed/staleness/obs-dim/self-damp/wireless-estop. |
| `agile23_keyboard_control.py` | keyboard teleop, **single writer** of `/tmp/robojudo_ext_cmd.json` (units="agile"), heartbeat so the command never goes stale. I/J/K/L+arrows, U/O yaw, 9/0 height, H stop, 1/2/3/4 FSM. |
| `start_agile23_box.sh` | tmux launcher (merger / pipeline / keyboard [/ box_demo]). |
| `AGILE23_BOXDEMO_DEPLOY_DESIGN.md` | full design spec (architecture, obs/action contract, joint map, safety, GO/NO-GO). |
| `AGILE23_DEPLOY_RESEARCH_SLICES.md` | raw source-grounded research (obs/action/policy/keyboard/jointmap). |

## Prerequisites (on the robot)
- `conda activate robojudo` (has `torch`, `unitree_sdk2py`, cyclonedds).
- AGILE repo (`WBC-AGILE`) on disk → `--agile-repo`.
- The **23-DoF student** artifacts (produced after the distillation finishes + export):
  - `--checkpoint` the 23-DoF student `_checkpoint.pt` (cleanest LSTM state),
  - `--config` the **re-exported 23-DoF IODescriptor YAML** (`export_IODescriptors.py
    --task Velocity-Height-G1-23dof-Distillation-Recurrent-v0`). **NOT** the 29-DoF YAML.

## Bring-up (safety-first, see DESIGN §8)
0. Keep RoboJuDo `start.sh` available as fallback.
1. `./start_agile23_box.sh --dry-run` — pipeline computes obs/policy/action, publishes
   nothing, no merger. Confirm 50 Hz holds and `obs_dim` matches.
2. **HARD GO/NO-GO — obs byte-compare:** dump the on-robot 68-vector vs `sim2mujoco_eval`
   (fed the 23-DoF YAML) for the same static pose; all six terms must match incl. units.
3. Trip-test the safety boundary (tilt abort, kill pipeline → merger damps, wireless e-stop).
4. Suspended → ground, conservative `PD_SCALE=0.3`, zero-command balance → small `vx` →
   height steps → turning. Validate distances **side-by-side vs RoboJuDo's known-good walk**.

## ⚠️ Must verify on the real robot
1. **`rt/lowstate.motor_state[]` layout** — full 29 slots (6 absent joints present-but-inert)
   vs compacted 23. The `MOTOR_BY_JOINT` scheme assumes 29 slots. #1 risk.
2. **23-DoF descriptor `joint_names`** order — read from the exported YAML, never assumed.
   `MOTOR_BY_JOINT` keys are base names; `strip_joint()` handles a `_joint` suffix.
3. **`mode_pr`** (hardcoded 0) and **IMU convention** (quaternion wxyz, gyro body-frame).
4. **height clamp** — pipeline/keyboard default to (0.20, 0.74) so the deep-squat target is
   reachable; stock AGILE `command_manager.height_range` is (0.40,0.72) and is overridden.
5. **`merge_lowcmd_arm_sdk.py`** expects 35-slot HG LowCmd and `slot29==0` → arm_sdk owns 12-28.

> Code is written against the source-verified contract but is **untested** (the robot was
> offline and the 23-DoF student is still distilling). Treat the first hardware run as a
> bring-up, not a deploy.
