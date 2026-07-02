# Path A — AGILE 23-DoF velocity-height student → RoboJuDo621 (real 23-DoF G1)

Integrate **our** distilled 23-DoF basic-G1 velocity-height LSTM student (`model_2999`) into the
classmate's **RoboJuDo621** framework on the real robot, reusing RoboJuDo's `rt/lowcmd` wire +
keyboard + the existing `AgileVelHeightRecurrentPolicy` pattern.

## Robot / access
- Jump host: `ssh 5090-jump` (zhengye@183.238.119.251:44125, key id_rsa_zheli).
- Robot: `ssh unitree-23dof` (unitree@192.168.4.102, ProxyJump 5090-jump; password omitted).
  `unitree-g1-nx`, Jetson Orin NX aarch64, JetPack R36.4.3, conda env **`robojudo`** (py3.11),
  torch 2.12.1 **CPU-only** (driver 12060 < cu130 → CPU; fine). DDS iface **`enP8p1s0`** (192.168.123.164).
- RoboJuDo621 publishes **`rt/lowcmd` directly** (+ `rt/arm_sdk`); **no merger**. Arms held in the
  SAME `rt/lowcmd` as legs (`arm_sdk_motor_idx=None`, per the [ih] 23-DoF real findings).

## What this adds (files on the robot, under `~/23dof_deployment/RoboJuDo621/`)
| path | kind |
|---|---|
| `robojudo/policy/agile_velheight_23dof_policy.py` | NEW — `AgileVelHeight23DoFRecurrentPolicy` (subclass of [ih]'s; obs **68**, drops the 24 frozen-hand zero-pad) |
| `robojudo/config/g1/policy/g1_agile_velheight_23dof_cfg.py` | NEW — `G1AgileVH23BodyDoF` (23 obs joints + defaults, from our IO descriptor) + `G1AgileVelHeight23DoFPolicyCfg` (reuses `G1AgileVHLegsDoF` 12-leg action verbatim) |
| `robojudo/policy/__init__.py` | EDIT — register the policy (backup `.bak_lz`) |
| `robojudo/config/g1/g1_cfg.py` | EDIT — `g1_lz_velheight_23dof` / `_real` / `_real_keyboard` pipelines (backup `.bak_lz`) |
| `assets/models/g1/agile/velheight_23dof_recurrent.pt` | NEW — our `model_2999` student JIT |

Local source-of-truth copies live here in `cc/agile23_deploy/robojudo_integration/`.

## Why a new policy class (vs [ih]'s)
[ih]'s `AgileVelHeightRecurrentPolicy` targets the **29-body + 24 frozen-hand** variant → obs **128**.
Our basic 23-DoF G1 has no hands → obs **68** = commands(4)+ang_vel(3)+gravity(3)+joint_pos_rel(23)
+joint_vel_rel(23)·0.1+last_action(12). The subclass only overrides `get_observation` to drop the
hand zero-pad. The 12-leg action (scale + gains 40.18/99.10/28.50) is **identical** → reused verbatim.

## obs joint order (from our exported IO descriptor — do NOT hand-derive)
23 = the 29-DoF interleaved order minus waist_roll/pitch & both wrist_pitch/yaw:
`L/R_hip_pitch, waist_yaw, L/R_hip_roll, L/R_shoulder_pitch, L/R_hip_yaw, L/R_shoulder_roll,
L/R_knee, L/R_shoulder_yaw, L/R_ankle_pitch, L/R_elbow, L/R_ankle_roll, L/R_wrist_roll`.
default_pos: hip_pitch −0.1, knee +0.3, ankle_pitch −0.2, rest 0.

## Validation status
- ✅ **Static (on robot, robojudo env):** `g1_lz_velheight_23dof` config resolves; model loads;
  **obs(68)→action(12)**; LSTM hidden state carries (2nd-call Δ=0.71); obs_dof=23, action_dof=12,
  policy registered. → integration is structurally correct.
- ✅ **Walking** already proven for this student in AGILE's own sim2mujoco (see `agile23_walk_2999.mp4`).
- ⚠️ **RoboJuDo MuJoCo sim2sim on the robot: NOT runnable** — RoboJuDo's patched `mujoco_viewer`
  submodule (`third_party/mujoco_viewer` + `patches/mujoco_viewer.patch`) isn't set up on this
  Jetson (it's the real-deploy target, not a sim box). The stock pip `mujoco-python-viewer 0.1.4`
  lacks RoboJuDo's `diable_key_callbacks` kwarg. For a RoboJuDo MuJoCo run, do it on a sim-capable
  machine, or set up the patched viewer submodule.

## Run
- (sim box) `SDL_AUDIODRIVER=dummy python scripts/run_pipeline.py -c g1_lz_velheight_23dof`
- **Real robot (keyboard):** `python scripts/run_pipeline.py -c g1_lz_velheight_23dof_real_keyboard`
  - keys: `w/s` vx, `a/d` vy, `q/e` wz, **`r` taller / `f` squat**, `o`/`esc` shutdown.
  - 6 s cosine prepare-ramp to default pose; `do_safety_check=True`. Start suspended, conservative.
  - height range 0.25–0.72 m (our deep-squat student); default stand 0.72.

## Env gaps summary (this robot)
- RoboJuDo Path A (this): **no extra pip deps needed** (RoboJuDo reimplements obs in numpy).
- Our standalone `agile23_lowcmd_pipeline.py` path (alternative): would need `matplotlib`+`pandas`
  and a `rt/lowcmd_rl`→`rt/lowcmd` change (no merger here) — Path A avoids both.
- Note: I pip-installed `mujoco-python-viewer 0.1.4` while probing the sim path; it's unused by the
  real deploy. Uninstall if it shadows a future patched-submodule sim setup.
