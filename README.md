# AGILE G1 23-DoF Velocity-Height

This repository packages the local AGILE 23-DoF velocity-height training and
testing work for Unitree G1.

## Layout

- `WBC-AGILE/` - upstream NVIDIA WBC-AGILE source plus local 23-DoF
  velocity-height task changes.
- `agile23_deploy/` - 23-DoF sim2sim, RoboJuDo integration, lowcmd pipeline,
  keyboard control, and deployment notes.
- `benchmark/` - local benchmark harness scripts, task definitions, console
  manifest utilities, and AGILE-DEEPSQUAT result jsonl files.
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

Useful entry points:

- `agile23_deploy/agile23_sim2sim_video.py`
- `agile23_deploy/agile23_lowcmd_pipeline.py`
- `agile23_deploy/agile23_keyboard_control.py`
- `agile23_deploy/robojudo_integration/`

Benchmark scripts and result aggregation live in `benchmark/scripts/` and
`benchmark/results/`.

## Artifact Notes

The WBC-AGILE pretrained velocity-height artifacts are kept under
`WBC-AGILE/agile/data/policy/velocity_height_g1/` in the source checkout, but
weights and videos are intentionally not committed here. Copy or export the
needed `.pt`, `.onnx`, and video artifacts separately when running deployment or
benchmark reproduction.
For true 23-DoF deployment, use a checkpoint and IO descriptor whose observation
dimension matches the 23-DoF contract:

```text
commands(4) + base_ang_vel(3) + projected_gravity(3)
+ joint_pos_rel(23) + joint_vel_rel(23) + last_action(12) = 68
```

Do not deploy the 29-DoF IO descriptor as a 23-DoF policy without an explicit
compatibility shim.
