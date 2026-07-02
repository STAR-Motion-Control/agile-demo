# G1 Sim2Real Control Demo

This repository packages the local Unitree G1 sim2real control code for AGILE
23-DoF velocity-height, Decoupled-WBC / GR00T-WBC, and the `box_demo_2`
application pipeline.

## Layout

- `WBC-AGILE/` - upstream NVIDIA WBC-AGILE source plus local 23-DoF
  velocity-height task changes.
- `agile23_deploy/` - 23-DoF sim2sim, RoboJuDo integration, lowcmd pipeline,
  keyboard control, and deployment notes.
- `benchmark/` - local benchmark harness scripts, task definitions, console
  manifest utilities, and AGILE-DEEPSQUAT result jsonl files.
- `decoupled_wbc/` - Decoupled-WBC / GR00T-WBC source code, configs, robot
  XML/URDF files, teleop/navigation loops, and tests without large binaries.
- `decoupled_wbc_sim2mujoco/` - compact sim2mujoco runner copy used by the
  benchmark/deployment lane, without ONNX weights or mesh assets.
- `box_demo_2/` - original detection + SAM3 + IK + arm/locomotion arbitration
  box demo code.
- `box_demo_groot/` - GROOT-WBC/box-demo integration, precision tests, and
  remote/manual mover tooling.
- `docs/` - local sim2real decision notes.

## Key Training Tasks

The 23-DoF velocity-height tasks are registered in:

- `WBC-AGILE/agile/rl_env/tasks/locomotion_height/g1/__init__.py`
- `WBC-AGILE/agile/rl_env/tasks/locomotion_height/g1/velocity_height_env_cfg.py`

Task ids:

```bash
Velocity-Height-G1-23dof-v0
Velocity-Height-G1-23dof-Distillation-Recurrent-v0
```

Typical training entry:

```bash
cd WBC-AGILE
python scripts/train.py --task Velocity-Height-G1-23dof-Distillation-Recurrent-v0
```

Export the 23-DoF IO descriptor before deployment:

```bash
cd WBC-AGILE
python scripts/export_IODescriptors.py --task Velocity-Height-G1-23dof-Distillation-Recurrent-v0
```

## Testing And Deployment

Local AGILE 23-DoF deployment/testing code lives in `agile23_deploy/`.
Decoupled-WBC and box-demo integration code lives in `decoupled_wbc/`,
`decoupled_wbc_sim2mujoco/`, `box_demo_2/`, and `box_demo_groot/`.

Useful entry points:

- `agile23_deploy/agile23_sim2sim_video.py`
- `agile23_deploy/agile23_lowcmd_pipeline.py`
- `agile23_deploy/agile23_keyboard_control.py`
- `agile23_deploy/robojudo_integration/`
- `decoupled_wbc/control/main/teleop/run_navigation_policy_loop.py`
- `decoupled_wbc/control/main/teleop/run_g1_control_loop.py`
- `decoupled_wbc/scripts/deploy_g1.py`
- `decoupled_wbc_sim2mujoco/scripts/run_mujoco_gear_wbc.py`
- `box_demo_2/box_demo_main.py`
- `box_demo_2/merge_lowcmd_arm_sdk.py`
- `box_demo_groot/groot_wbc_boxdemo_adapter.py`
- `box_demo_groot/test_small_moves.py`

Benchmark scripts and result aggregation live in `benchmark/scripts/` and
`benchmark/results/`.

## Collaboration Workflow

Use `main` as the shared stable branch. Navigation/interface work should happen
on feature branches and merge back after review, for example:

```bash
git checkout -b nav-interface/<name>
```

Keep large generated artifacts out of git. The repository `.gitignore` excludes
weights, videos, SDK binaries, replay datasets, and bulky sim assets.

## Artifact Notes

The WBC-AGILE pretrained velocity-height artifacts are kept under
`WBC-AGILE/agile/data/policy/velocity_height_g1/` in the source checkout, but
weights and videos are intentionally not committed here. Copy or export the
needed `.pt`, `.onnx`, and video artifacts separately when running deployment or
benchmark reproduction.

Known external artifact/config locations used by the current AGILE 23-DoF work:

| Purpose | External path / command | Notes |
|---|---|---|
| Generic WBC-AGILE velocity-height policy files | `/Users/lizhe/Project/sim2real/cc/experiments/repos/WBC-AGILE/agile/data/policy/velocity_height_g1/` | Source checkout contains `unitree_g1_velocity_height_*` weights/YAML; weights are omitted here. |
| 23-DoF IO descriptor | `python WBC-AGILE/scripts/export_IODescriptors.py --task Velocity-Height-G1-23dof-Distillation-Recurrent-v0` | Expected 23-DoF student obs dim is 68. Store generated descriptor next to the checkpoint used for deployment. |
| 23-DoF sim2sim example artifacts | `/sdb/lizhe/g1_deepsquat/student_23dof_1400_export/policy.pt`, `/sdb/lizhe/g1_deepsquat/student_23dof_1400_export/<IO_descriptors>.yaml`, `/sdb/lizhe/g1_deepsquat/g1_23dof_desc/scene_23dof.xml` | Paths are documented in `agile23_deploy/agile23_sim2sim_video.py`; do not commit these binaries. |
| RoboJuDo real-robot model target | `~/23dof_deployment/RoboJuDo621/assets/models/g1/agile/velheight_23dof_recurrent.pt` | The integration expects the `model_2999`/true-23DoF recurrent student there. |
| Decoupled-WBC sim2mujoco ONNX policies | `/Users/lizhe/Project/sim2real/GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-{Walk,Balance}.onnx` | Omitted from git; copy them back to the same relative path when running sim2mujoco. |
| Decoupled-WBC G1 mesh assets | `/Users/lizhe/Project/sim2real/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/meshes/` and `/Users/lizhe/Project/sim2real/GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/meshes/` | Omitted as bulky binary assets; XML/URDF configs remain committed. |
| Decoupled-WBC RoboCasa model assets | `/Users/lizhe/Project/sim2real/GR00T-WholeBodyControl/decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/` | Omitted because these are large texture/mesh assets, not source code. |
| Decoupled-WBC device SDK binaries | `/Users/lizhe/Project/sim2real/GR00T-WholeBodyControl/decoupled_wbc/control/teleop/device/` | Manus `.so`, Pico `.deb`, and object files are omitted; source/header files remain committed. |
| Decoupled-WBC test replay data | `/Users/lizhe/Project/sim2real/GR00T-WholeBodyControl/decoupled_wbc/tests/replay_data/` | Omitted generated data (`.pkl`, `.parquet`, `.npy`). |
| box_demo_2 local captures and archive | `/Users/lizhe/Project/sim2real/box_demo_2/img/`, `/Users/lizhe/Project/sim2real/box_demo_2-caokong.zip` | Omitted local camera captures/archive; runnable demo code and configs remain committed. |

For true 23-DoF deployment, use a checkpoint and IO descriptor whose observation
dimension matches the 23-DoF contract:

```text
commands(4) + base_ang_vel(3) + projected_gravity(3)
+ joint_pos_rel(23) + joint_vel_rel(23) + last_action(12) = 68
```

Do not deploy the 29-DoF IO descriptor as a 23-DoF policy without an explicit
compatibility shim.
