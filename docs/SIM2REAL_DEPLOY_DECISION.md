# Sim2Real Deployment — Decision Memo & Method Analysis

> Status: **Decision locked.** Build ONE unified Python sim2real runtime + per-method adapters. Do NOT fork sonic's C++ as the host. Sonic stays on its native stack. First hardware bring-up = **HOMIE**.
>
> Date: 2026-06-15 · Scope: G1 locomotion policies (AGILE, FALCON, AMO, HOMIE, GR00T-sonic) moving from sim2sim → sim2real.
> Source: code-grounded reverse-engineering of each method's deploy path + the existing `cc/experiments/scripts/bench_*.py` harness (7-agent survey, real source reads).

---

## 0. Bottom line up front

- **5 methods, only 3 ship a real-robot deploy path**: GR00T-sonic (C++/TensorRT), FALCON (Python/onnxruntime), HOMIE (C++ low-level + Python ONNX). **AMO and AGILE are sim-only** (no `unitree_sdk2`/DDS code in either tree).
- They **align** on the deploy *runtime* (G1, 50 Hz position control, `q = default + scale·action`, decimated PD, same physical state, converging command vocab) but **diverge** on the *obs assembly* (5 mutually-incompatible byte layouts, each with method-specific pre-stages/quirks).
- ⇒ **A single obs-builder is wrong; a single runtime is right.** Separate the universal runtime (~90% of deploy code: transport + RT loop + safety SM + tensor I/O) from the bespoke adapter (~10%: obs build + action map + command source). You write five obs-builders no matter what — the choice is whether they sit behind one battle-tested runtime or five copy-pasted ones.
- The existing **sim2sim bench already proves the pattern**: FALCON/AMO/AGILE bench harnesses wrap the upstream deploy class unchanged and only swap a MuJoCo state/command shim for the transport. That seam (FALCON's `MujocoStateShim`/`MujocoCmdShim`) is promoted to a real `RobotBackend`; add a `unitree_sdk2_python` impl next to MuJoCo.

---

## 1. Per-method deploy analysis

Each method on a shared rubric: existence/maturity · target HW + SDK · policy format + device · obs vector · action path · control loop · state machine + safety · command interface · entry point · reusability · gaps.

### 1.1 GR00T-sonic ("sonic") — the reference, C++/TensorRT

- **Has own deploy:** ✅ **yes, hardware-tested.** Production-grade standalone C++20 stack `gear_sonic_deploy/` (~20k LOC) — own CMake build, `deploy.sh` launcher, JetPack6/Orin flashing guide, tmux all-in-one launcher (`gear_sonic/scripts/launch_inference.py`). **Same binary** runs MuJoCo sim (DDS loopback, `--disable-crc-check`) and real G1. Many real-robot teleop GIFs in `media/`. Most mature path of the five.
- **Target HW + SDK:** Unitree G1 (23/29 DoF, +Dex3/BrainCo hands). Onboard = Jetson Orin NX, or x86+GPU. **Vendored `unitree_sdk2` (C++) over CycloneDDS.** Topics: `rt/lowcmd`, `rt/lowstate`, `rt/secondary_imu` (`robot_parameters.hpp:27-29`). CRC32 on every LowCmd/LowState. `ChannelFactory::Instance()->Init(0, iface)`. LowState callback @ 500 Hz.
- **Policy format + device:** ONNX → auto-converted to **TensorRT** engines cached on disk per-GPU. **Two nets:** encoder (obs→64-d token) + decoder/policy (obs→29 action). Separate locomotion planner ONNX. **GPU mandatory, `device_id=0` hardcoded**, CUDA-graph capture for sub-ms latency. No CPU fallback.
- **Obs (994-d):** YAML-driven (`policy/release/observation_config.yaml`) name→{dim,lambda} registry, concatenated in YAML order. Release order: `token_state[64]`, `his_base_angular_velocity_10frame[30]`, `his_body_joint_positions_10frame[290]`, `his_body_joint_velocities_10frame[290]`, `his_last_actions_10frame[290]`, `his_gravity_dir_10frame[30]`. 10-frame history (0.2 s). All in IsaacLab joint order; joint pos has **default-angle subtracted**; **no obs scaling** in C++. ⚠️ The YAML's "436" comment is **stale** (describes the 4-frame variant) — real dim is 994.
- **Action (29):** `q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] · action_scale[i]`. `tau_ff=0, dq_target=0`, fixed per-motor kp/kd. `action_scale = 0.25·effort_limit/stiffness`. default = bent-knee crouch. PD from motor armature (ω=10Hz·2π, ζ=2.0); ankles & waist roll/pitch get 2× gains. No runtime tuning.
- **Control loop:** **4 RT threads** (`CreateRecurrentThreadEx`): Input 100 Hz, Control/policy 50 Hz, Planner 10 Hz, CmdWriter 500 Hz. Policy 50 Hz republished at 500 Hz → 10:1 decimation; kHz PD on motor controllers. SCHED_FIFO, pinned CPU 0.
- **State machine + safety:** `INIT (3s ramp to default) → WAIT_FOR_CONTROL (start key) → CONTROL`. E-stop ('O'/'`'/gamepad) → `CreateDampingCommand` kp=0, kd=8. Stale LowState >500 ms abort; joint over-speed >35 rad/s abort; motor temp hysteresis + TTS alerts; error-code monitoring; teleop-dropout debouncer.
- **Command interface:** pluggable `InputInterface` (keyboard/gamepad/zmq/ros2/manager). **Two paths:** (1) locomotion via planner ONNX + `MovementState{mode, dir, speed, height}` (27 LocomotionModes); (2) whole-body via streamed reference motion + VR-3point targets (PICO VR → GMR retarget over ZMQ). Upper body is injected as a **tracking target** the single net follows — not a separate arm IK controller.
- **Entry:** `bash deploy.sh --cp policy/release/model_step_NNNNNN --obs-config ... --input-type manager [sim|real|<iface>]`. Binary `g1_deploy_onnx_ref`.
- **Reusability:** loop/safety/DDS skeleton is **reusable boilerplate**; the obs registry + encoder-token + 3 encoder-modes + GMR/VR teleop are **bespoke**. Hardcodes G1 29-DoF, 3 joint orderings, armature constants, crouch stance.
- **Gaps:** CUDA/TensorRT/Linux-gated (cannot build on Mac); ONNX re-export needs Isaac Lab on a separate server; models are Git-LFS / NVIDIA Open Model License; `.trt` engines are GPU-arch-specific. **Keep sonic on its own stack.**

### 1.2 AMO — MuJoCo-only, no real transport

- **Has own deploy:** ❌ **MuJoCo sim2sim teleop demo only.** `grep` for `unitree_sdk2`/DDS/serial/LowCmd/LowState → **zero hits**. State from simulated IMU sensors; actions → MuJoCo `<motor>` torque. Upstream README **discourages** real-hardware deploy (links a sim2real crash video). Local `AMO/deploy/` is the user's own headless/GPU-preflight scaffolding (`run_amo.sh`, `smoke_test.py`), not a hardware bridge. Smoke test passes (stands ~0.74 m).
- **Target HW:** Unitree G1 only (`robot_type=="g1"`), 23 DoF (15 policy lower+waist, 8 open-loop arms) — but only ever a MuJoCo MjModel.
- **Policy + device:** two TorchScript `.pt` — main `amo_jit.pt` + adapter `adapter_jit.pt` (+ `adapter_norm_stats.pt`). **`cuda:0` baked into the traced graph** → `map_location="cpu"` can't remap; **healthy GPU mandatory** (`cpu_test.py` demonstrates failure). On 5080 needs torch≥2.7/cu128.
- **Obs (1043 + extra 2325):** `get_observation()` builds `obs = cat(obs_prop[93], obs_demo[17], obs_priv=zeros[3], obs_hist[930=10×93])` + separate `extra_hist[2325=25×93]`. `obs_prop` = `[ang_vel·0.25(3), roll/pitch(2), sin/cos dyaw(2), (q-default)(23), dq·0.05(23) (some idx zeroed), last_action(23), gait sin/cos(2), adapter_output(15)]`. Two zero-init history deques. **adapter runs first** (normalized teacher-proxy: desired torso pose → 15-d reference).
- **Action (15 + 8):** num_actions=15. raw clip ±40, `scaled = raw·0.25`, `pd_target = cat(scaled, zeros(8)) + default`. **Arms (8) bypass policy** — open-loop blended PD (random poses on `T` toggle). Per-joint kp/kd/torque-limit tables. Torque computed in Python: `tau = (pd_target-q)·kp - dq·kd`, clipped, → `data.ctrl`.
- **Control loop:** single-threaded **non-RT** for-loop. physics 500 Hz, decimation 10 → policy 50 Hz. Inner PD-torque recomputed every physics step.
- **State machine + safety:** **none** (sim demo). No zero-torque/damping/ramp startup, no e-stop to robot, no fall-abort. Only sim-side clips (action ±40, torque limits, arm ±0.4, arm_blend ramp).
- **Command interface:** GLFW keyboard → 8-elem `viewer.commands`. **`commands[1]=ABSOLUTE target heading (not wz)`; `commands[3]=height DELTA` on 0.75 base.** Arms not via policy; torso rpy + height feed the adapter. Bench injects by writing `env.viewer.commands` + `env.prev_arm_action/arm_action/arm_blend` directly.
- **Entry:** `python play_amo.py` (or `bash run_amo.sh`). Bench reuses `play_amo.HumanoidEnv` verbatim, stubbing the viewer.
- **Reusability:** **bespoke & tightly coupled.** Decimated PD-torque skeleton + keyboard pattern are generic; everything between `get_observation` and `pd_target` is AMO-specific. Loop is **inverted** (`env.run()` owns it; bench injects via `viewer.render` hook).
- **Gaps:** no real-robot path (entire LowState→obs / action→LowCmd to write from scratch); cuda:0 baked in; no safety SM; arms are random not real teleop. **Most refactoring of any method.**

### 1.3 HOMIE / OpenHomie — real deploy, legs-only, first bring-up

- **Has own deploy:** ✅ **yes, hardware-tested** (paper + YouTube G1 teleop). Ships `deploy.onnx` (456→12) + C++ low-level + Python inference. **Partially self-contained:** checkpoint path hardcoded to `/home/unitree/deploy/deploy.onnx`, some arm paths commented out, a fall-estop kwarg bug, `np.float` usage. Exoskeleton/glove/pedal firmware form-gated (`HomieHardware` is a README). Clean MuJoCo-only replica in `MujocoDeploy/`.
- **Target HW + SDK:** Unitree G1 29-DoF + Dex-3 hands. Onboard Jetson Orin. **Two-layer transport:** LOW = `unitree_sdk2` C++ (`g1_control.cpp`, DDS, CRC); BRIDGE = **LCM over UDP** between C++ and Python (channels `pd_plustau_targets`, `state_estimator_data`, `rc_command`, `pedal_command`, `arm_action`). Hands via separate `hand_control` binary.
- **Policy + device:** ONNX via onnxruntime (`deploy_policy.py`). MujocoDeploy uses TorchScript. Single MLP `[1,456]→[1,12]`. **`cuda:0` hardcoded** (only to wrap the ORT output as a torch tensor; ORT itself runs CPU). Jetson works; CPU-only box would crash as-is.
- **Obs (456 = 6×76):** HARDWARE order (`lcm_agent.py:71-77`): `[0:4] cmd(vx,vy,wz,height)·[2,2,0.25,1]`, `[4:7] ang_vel·0.5`, `[7:10] projected_gravity`, `[10:37] (q-default) all 27 joints`, `[37:64] dq·0.05`, `[64:76] last action (12 legs)`. ⚠️ **`ang_vel_scale` real value is 0.25, not the repo's 0.5** (A/B-tested — `PROTOCOL.md:102`, `NOTES_HOMIE.md:65`). Obs clip ±100 present in training, omitted in shipped deploy.
- **Action (12 legs):** `target = action·0.25 + default[:12]`. clip ±100. default legs `[-0.1,0,0,0.3,-0.2,0]×2`. **Per-joint kp/kd in C++** (`g1_control.cpp`): legs Kp `[150,150,150,300,40,40]×2`, Kd `[2,2,2,4,2,2]×2`. (MuJoCo yaml uses softer legs — slight sim/real mismatch.) Torque-limit code present but **commented out** — pos targets sent, PD on robot side.
- **Control loop:** policy 50 Hz (Python, busy-wait); low-level PD + DDS 500 Hz (C++, 3 RT threads). **Async LCM** decouples the two loops.
- **State machine + safety:** (1) operator kills G1 built-in controller via remote; (2) C++ 3 s ramp all 29 motors to q=0; (3) Python `calibrate()` waits R2, interpolates to default at 20 Hz, waits 2nd R2 to start. **Bad-orientation e-stop**: `|roll|>1.6 or |pitch|>1.6` → `calibrate(wait=False, low=True)` — ⚠️ **kwargs not declared → would crash** (latent bug). R2 re-triggers recalibration.
- **Command interface:** 4-D `[vx, vy, wz, ABSOLUTE height]` (stand 0.74, deep squat ~0.45). On real HW the command enters via the exoskeleton **pedal** (LCM `pedal_command`); joystick path commented out. **Upper body driven OUTSIDE the locomotion policy** — exoskeleton arms → `arm_action` LCM channel → C++ `q_target[15:29]`; in the open slice arms are stubbed to zero. The policy reacts to arm motion only through measured-joint obs (the loco-manip decoupling). ⇒ **locomotion deploy is separable from the exoskeleton.**
- **Entry:** 3 terminals on the G1/Orin: `./hand_control`; `./g1_control eth0`; `python g1_gym_deploy/scripts/deploy_policy.py`; then R2 to stand/calibrate, R2 to start. Sim: `python MujocoDeploy/mujoco_deploy_g1.py`.
- **Reusability:** C++ `g1_control.cpp` (DDS↔LCM bridge, 500 Hz PD, CRC, zero-pose ramp, gamepad) is a clean **method-agnostic G1 low-level skeleton** (walk-these-ways lineage) — reusable by any policy emitting a 29-vec q_des on `pd_plustau_targets`. Python side is bespoke to the 76-d obs / 12-action layout + LCM contract.
- **Gaps:** hardcoded checkpoint path; `cuda:0`; the fall-estop crash bug; `np.float` (NumPy≥1.24 breaks); form-gated exoskeleton/pedal (the command source on real HW — only commented-out joystick fallback otherwise). **Small, code-local fixes ⇒ first bring-up.** License CC BY-NC-SA (no commercial).

### 1.4 AGILE / WBC-AGILE — sim2mujoco only, but clean exported interface

- **Has own deploy:** ❌ **sim2sim (MuJoCo) only** (`agile/sim2mujoco/` + `scripts/sim2mujoco_eval.py`). **No real-robot code in-repo** (zero `unitree_sdk2`/`booster_robotics_sdk`/DDS/C++/e-stop). README sim2real GIFs come from an **unreleased internal/Booster deployer**. FAQ: "AGILE is mainly designed for policy training." But it **ships a clean deploy artifact**: exported TorchScript/ONNX policy + **IODescriptor YAML**.
- **Target HW:** trained/validated on G1 29-DoF + Booster T1; only MuJoCo MJCF consumed.
- **Policy + device:** TorchScript `.pt` (normalizer baked in) primary; ONNX exported too; raw RSL-RL checkpoint fallback. Recurrent G1 student is an LSTM with hidden/cell as internal buffers. **CPU or GPU, `--device` selectable, NOT hardcoded cuda:0.** Policy is tiny (~2 MB).
- **Obs (80, LSTM no history):** data-driven from IODescriptor YAML; order: `generated_commands(4)=[vx,vy,wz,abs_pelvis_height]`, `base_ang_vel(3)`, `projected_gravity(3)`, `joint_pos_rel(29)`, `joint_vel_rel(29, scale 0.1)`, `last_action(12, clip ±10)`. Per-term raw→noise→clip→scale→history. ⚠️ **Upstream bug:** the no-sensor `base_ang_vel` fallback double-rotates body-frame `qvel[3:6]` (`simulation.py:418-421`) → falls at large initial yaw. **A real deploy must carry the `BodyFrameGyroShim`.**
- **Action (12 legs):** legs only; upper 17 joints held at default. `target_q = action·scale + offset` (per-joint from YAML ~0.35-0.55). raw clip ±6 before scale/offset. PD from `default_joint_stiffness/damping` YAML. `--pd-scale` flag scales kp/kd uniformly.
- **Control loop:** single-threaded blocking Python loop. physics 200 Hz, decimation 4 → policy 50 Hz. PD by MuJoCo each substep; no separate low-level thread.
- **State machine + safety:** **essentially none** of a real-robot FSM. `reset()` snaps to default pose + zeros ctrl. Only viewer pause/step + push-force test. Implicit limits: action clip ±6, last_action clip ±10, MuJoCo joint limits, `--pd-scale`. Fall detection added by the external bench, not in-repo.
- **Command interface:** `CommandManager`, 3 modes: keyboard teleop / deterministic YAML schedule / random. 4-D `[vx,vy,wz,abs_height]`, native wz, clamps vx,vy∈[-0.5,0.5], wz∈[-1,1], height∈[0.4,0.72]. **No live upper-body injection** (whole-body via offline motion files only). Bench overrides arm targets after the act processor for carry poses.
- **Entry:** `python scripts/sim2mujoco_eval.py --checkpoint ...student.pt --config ...yaml --mjcf scene_29dof.xml`. Re-export needs Isaac (`scripts/eval.py`, `export_IODescriptors.py`). **No real-robot entry point.**
- **Reusability:** the sim2mujoco **loop+obs+action skeleton is notably reusable & policy-agnostic** — `ObservationProcessor`/`ActionProcessor` are generic factories driven entirely by the IODescriptor YAML (term→fn dispatch, joint-name remap, history, per-term noise/clip/scale); `PolicyWrapper` auto-handles TorchScript/ONNX/checkpoint + MLP/LSTM/GRU. **The IODescriptor YAML is the clean unified interface a harness should target.**
- **Gaps:** no robot transport/safety FSM/state-estimation glue/live upper-body; Isaac-gated re-export (but the already-exported `.pt/.onnx/.yaml` run on CPU+torch+mujoco); the gyro double-rotation bug; need `unitree_mujoco` for the G1 MJCF.

### 1.5 FALCON — real deploy, repo NOT local (GitHub-sourced)

> ⚠️ Repo is **not on this machine** — only eval `.jsonl` + the user's `bench_falcon.py`. Source claims are from GitHub `LeCAR-Lab/FALCON` main; line numbers approximate. Public ONNX obs width may differ from the user's self-trained 575-d checkpoint — **re-confirm against the loaded model before trusting per-frame width.**

- **Has own deploy:** ✅ **yes, hardware-tested.** README §"FALCON Deploy": "seamless sim2sim and sim2real deployment scripts supporting both `unitree_sdk2_python` and `booster_robotics_sdk`." Real G1 + Booster T1 (payload transport, cart-pulling, door opening). `sim2real/` is a full self-contained on-robot stack (config/, models/, rl_policy/, sim_env/, utils/{arm_ik,comm,sdk2py_bridge}).
- **Target HW + SDK:** G1 29-DoF + T1 29-DoF. **Two backends by config:** G1 → `unitree_sdk2_python` (DDS); T1 → forked `booster_robotics_sdk` (C++ + py bindings). Transport via `utils/sdk2py_bridge` + `utils/comm`. Low-level command is **position (cmd_q) + kp/kd**, on-board PD.
- **Policy + device:** ONNX via onnxruntime. `InferenceSession(model_path)` with **no `providers=`** → **CPU by default** (deliberately onboard-friendly), no cuda hardcode. Public `models/falcon/g1_29dof.onnx`; user variant `g1_29dof_trained.onnx` `[1,575]→[1,29]`.
- **Obs (575 = 5×115):** ⚠️ flat vector built by `parse_current_obs_dict` using **`sorted()` = ALPHABETICAL key order**, NOT config-list order. Single-frame keys: `{actions(29), base_ang_vel(3), command_ang_vel(1), command_base_height(1), command_lin_vel(2), command_stand(1), command_waist_dofs(3), dof_pos(29), dof_vel(29), projected_gravity(3), ref_upper_dof_pos(14)}`. Scales: ang_vel 0.25, dof_vel 0.05, base_height 2.0, others 1.0. Phase fields exist in dec_loco but NOT in the falcon actor_obs.
- **Action (29):** clip ±100, `scaled = raw·0.25`; if `residual_upper_body_action` (True): the 14 upper DoFs get `(ref_upper_dof_pos - default_upper)` added as **residual**; `q_target = scaled + default`, clip to motor limits. Per-joint MOTOR_KP/KD/effort tables. On HW host sends cmd_q + kp/kd; bench reconstructs PD: `tau = clip(kp·(cmd_q-q) - kd·dq, ±effort)`.
- **Control loop:** policy 50 Hz via `RateLimiter`. sim 200 Hz, decimation 4 (sim only; real PD on motor driver ~1 kHz). Single-threaded blocking; keyboard listener daemon thread; joystick via SDK DDS callback. No separate async low-level host thread.
- **State machine + safety:** 3 modes in `base_policy.py`: (1) DEFAULT zero-torque/hold; (2) INIT — interp current→default over ~500 steps; (3) POLICY. Triggers: kbd `i`/`]`/`o`, joystick `A+X`/`start`/`B+Y`. Stop → damping/zero-torque. **No dedicated fall-protection beyond stop→damping**; relies on operator + checklist (start small gains, feet down, sim2sim first). Only auto guards: action clip ±100, motor pos/effort limits.
- **Command interface:** keyboard + joystick → policy fields: `lin_vel_command` (WASD ±0.1), `ang_vel_command` (Q/E), `stand_command` toggle (`=`/R2 — **0=stance/squat, 1=stepping/walk**, easy to invert), base height (`1`/`2`), waist (`,`/`.`). **Upper body:** `ref_upper_dof_pos` (14 abs arm targets) drives arms — enters obs AND adds as residual action; source = `utils/arm_ik` Pinocchio IK from EE waypoints (`use_upper_body_controller=True`). IK can take EE wrench (`EE_efrc_L/R`) but **the shipped loop leaves it 0** — force-adaptation is **learned in training, not fed at deploy**. Bench bypasses IK, writes `ref_upper_dof_pos` directly.
- **Entry:** `python rl_policy/loco_manip/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml --model_path=models/falcon/g1_29dof.onnx` (swap config/model for T1); then operator `i`→init, `]`→start.
- **Reusability:** the `BasePolicy` run-loop + 3-state machine + RateLimiter + `sdk2py_bridge` command_sender + kbd/joystick handlers are a clean **policy-agnostic skeleton** — exactly why `bench_falcon.py` subclasses `LocoManipPolicy` and only overrides `_init_sdk_components`/`_init_communication_components`/`_init_input_device` to swap DDS for a MuJoCo shim. obs layout hard-coupled to FALCON's command set.
- **Gaps:** repo not local; force-adaptive arm is a trained property (EE force zeroed at deploy); public ONNX obs width (115/164) may differ from the 575-d self-trained; no software fall-recovery; onnxruntime provider unspecified (CPU/CUDA numeric drift).

---

## 2. The existing sim2sim bench harness (what a unified layer builds on)

`cc/experiments/scripts/bench_{agile,amo,falcon,homie}.py` + `task_registry.json` + `NOTES_*.md` + `PROTOCOL.md`.

- **No shared abstraction.** Four separate single-file scripts (~3.4-4.7k lines each), parallel **by convention, not by code** (enforced by `PROTOCOL.md` + NOTES, not inheritance). Shared = identical helper signatures (`nav_errors`, `gravity_body_frame`, …), identical Mission/Schedule classes (re-declared per file), one **JSONL record envelope** (`{framework, test, trial, seed, success, fall_time, fall_phase, metrics{...}, video, ...}`), and unified constants (`SETTLE_S=2.0`, `SQUAT_SWEEP_HEIGHTS`, 50 Hz) — but each redefines them locally.
- **The per-tick loop differs structurally in all four** (exactly what a sim2real layer must abstract):

  | File | Policy load | obs built by | action→ctrl | Loop driver |
  |---|---|---|---|---|
  | **HOMIE** | onnxruntime | **harness inline** | harness PD inline | harness `for step` |
  | **AGILE** | upstream import | **upstream `obs_processor.compute()`** | upstream `act_processor.process()` | harness loop, upstream `sim.step` |
  | **FALCON** | upstream `LocoManipPolicy` subclass | **upstream deploy class** (`policy.policy_action()`) | harness reads `cmd_q`, applies PD | harness loop |
  | **AMO** | `torch.jit.load` | **upstream `HumanoidEnv` verbatim** | upstream inside `env.run()` | **inverted** — `env.run()` owns loop; harness injects via `viewer.render` hook |

  ⇒ two patterns coexist: HOMIE re-implements the full pipeline; AGILE/FALCON/AMO wrap the upstream deploy class + only inject commands and read back state/action.
- **`task_registry.json` is task-side only** (mission/scene/tolerances: `nav_tol_cal {pos 0.05, yaw 5°}`, grasp dists, fall tilt). It has **NO** per-method obs spec / action scale / default pose / gains — those are **hardcoded constants scattered across the four bench files**. ⇒ there is no per-method deploy config today; extracting one is the highest-leverage refactor.
- **Cleanest seam = the per-framework state/command shim, NOT the obs builder.** For AGILE/FALCON/AMO the obs+action+PD pipeline is upstream deploy code we must not touch (that's the sim2real fidelity). The harness only ever touches three thin surfaces: **state in** (`push_state` / `sim.get_state` / `env.data` / `qpos`), **command in** (field writes / `set_command` / `viewer.commands`), **action out** (`cmd_q` / `JointCommand` / ctrl / `target_dof_pos`). The MuJoCo backend is **already abstracted in 3 of 4** (FALCON `Mujoco{State,Cmd}Shim`, AMO `_make_mujoco_proxy`, AGILE `MuJocoSimulation`) — those shims are the natural swap point for `unitree_sdk2`.
- **Mission/Schedule layer is backend-agnostic** (pure command-vs-time/pose functions) — the most reusable existing asset; becomes the default `command_source`.

---

## 3. Commonality matrix

| Axis | GR00T-sonic | FALCON | AMO | HOMIE | AGILE |
|---|---|---|---|---|---|
| Real-robot code ships? | ✅ HW C++ | ✅ HW py | ❌ sim | ✅ HW mixed | ❌ sim |
| SDK | `unitree_sdk2` C++ | `unitree_sdk2_python` / Booster | NONE | `unitree_sdk2` C++ + LCM | NONE |
| Language | C++20 | Python | Python | Mixed | Python |
| Policy | ONNX→TRT (enc+dec) | ONNX (ort) | TorchScript (adapter+policy) | ONNX (ort) | TorchScript / ONNX |
| Device | **CUDA req, cuda:0** | **CPU** ✅ | **cuda:0 baked** ✗ | cuda:0 (MLP→CPU-able) | CPU/GPU selectable ✅ |
| Obs shape | 994 (10-frame + token) | 575 (5×115, **alphabetical**) | 1043 + 2325 (2 deques) | 456 (6×76) | 80 (LSTM, no hist) |
| Action / PD | 29; remap·scale; fixed PD | 29; +ref_upper residual; PD on driver | 15 (+8 open-loop); PD in py | 12 legs; PD in C++ | 12 legs; PD by MuJoCo |
| Control rate | 50/500 Hz, 4 RT threads | 50 Hz, RateLimiter | 50/500 Hz, non-RT | 50 py / 500 C++, async LCM | 50/200 Hz, single loop |
| Safety SM | ✅ full | ✅ 3-state | ❌ none | ✅ (latent estop bug) | ❌ none |
| Command | planner ONNX + VR/GMR | kbd/joy + Pinocchio IK | kbd 8-vec; arms open-loop; heading≠wz | exo PEDAL; arms via LCM | kbd/YAML; no live upper-body |

**Align (the unifiable core):** all G1, 50 Hz position-control, `q = default + scale·action`, decimated PD, same physical state (IMU ang_vel + gravity + joint pos/vel), converging command vocab `(vx, vy, wz/heading, height, [stand], [upper-targets])`, ONNX/TorchScript single-shot inference.

**Diverge (the un-unifiable core):** obs **assembly** (994 token-based / 575 alphabetical-sort / 1043+2325 dual-deque / 456 6-frame / 80 LSTM — mutually incompatible byte layouts with method-specific pre-stages: sonic encoder token, AMO adapter, AGILE gyro-shim+LSTM buffers, FALCON alphabetical sort). Action DoF split (29 vs 15+8 vs 12). Upper-body path (planner+VR vs IK-residual vs open-loop random vs pedal+LCM vs none). Language.

---

## 4. Three archetypes

- **A — Velocity+Height, legs-only:** HOMIE, AGILE. 12 leg actions, 4-D `[vx,vy,wz,abs-height]`, upper body held/external. Compact obs. Lowest risk.
- **B — Whole-body, upper-body retarget:** FALCON, GR00T-sonic. 29 whole-body actions; upper body injected into both obs and action as a tracking target; command source is a rich external stream (IK / VR-GMR).
- **C — Hybrid policy-stage + open-loop arms:** AMO. 15 policy DoF (2-stage adapter→policy), 8 open-loop arms; command[1]=absolute heading, height=delta. Its own archetype.

**Why a single obs-builder is hard but a single runtime is easy:** the obs-builder is genuinely bespoke (byte layout + term order + scales + *extra pipeline stages* differ in kind — encoder forward pass, adapter forward pass, alphabetical sort, LSTM buffers + gyro fix). You can't write one function for all five; you'd write five behind a switch — which **is** an adapter. But the runtime is identical in shape for all five: read LowState @500 Hz → at 50 Hz hand raw state to a method adapter → 1-2 forward passes → `q=default+scale·act` with per-method gains → write LowCmd (q+kp/kd, tau_ff=0) → wrap in INIT-ramp→WAIT→CONTROL with kd=8 damping e-stop. Transport + RT loop + safety + tensor I/O ≈ 90% of deploy code, method-agnostic. Bespoke ≈ 10%.

---

## 5. Decision: UNIFIED Python runtime + per-method adapters

### Why not fork sonic's C++ as the host (despite it being the most mature)
1. Would require **reimplementing four obs-builders in C++** — throwing away the validated Python obs builders; every port is a new chance for a silent obs bug (obs bugs don't crash, they make the robot fall).
2. **Inference backends don't match** — sonic TRT/CUDA-only, FALCON CPU onnxruntime, AMO cuda TorchScript, AGILE CPU LSTM, HOMIE ort MLP. Bolting onnxruntime + libtorch + TRT into one C++ binary is heavy/brittle; the sonic stack can't even build on the Mac.
3. **FALCON repo isn't local; AMO/AGILE have no transport at all** — the C++ host saves nothing on the methods that need the most work.

### Why not per-method deploy
You'd maintain 4-5 safety state machines / DDS bring-ups / e-stops (HOMIE's shipped one already has a latent crash bug). Safety is exactly what to write **once**, validate hard, reuse. Five e-stops = five times the risk on the one thing protecting the robot.

### Decisive factor
The sim2sim bench **already proves the unified pattern in Python** — FALCON/AMO/AGILE wrap the upstream deploy class unchanged and only swap a MuJoCo state/command shim. That is a unified runtime in all but name; the seam is already cut. Promote it to a real `RobotBackend`, keep the upstream Python obs-builders **verbatim** (max fidelity), add a `unitree_sdk2_python` backend.

### Hybrid weighting
**Unified Python runtime for FALCON / HOMIE / AGILE / AMO; sonic stays on its native C++/TRT stack** (known-good, CUDA/TRT-locked anyway), joining the unified runtime later only as an optional `SonicAdapter` if desired.

---

## 6. Recommended architecture

```
┌─────────────────────────────────────────────────────────────┐
│  UnifiedRunner  (write ONCE, method-agnostic)                │
│  • RT loop: 500 Hz cmd-writer / 50 Hz policy tick            │
│  • Safety SM: INIT-ramp → WAIT(start) → CONTROL              │
│    e-stop = kd8 damping; stale-state >500ms abort;           │
│    over-speed / tilt abort; temp monitor                     │
│  • Policy I/O: run 1-2 inference sessions, time the tick     │
└───────────────┬──────────────────────────┬──────────────────┘
                │ RobotBackend protocol     │ MethodAdapter protocol
                │ read_state()->StateBundle │ build_obs(state)->tensor
                │ write_cmd(q_tgt,kp,kd)     │ map_action(act)->(q_tgt,kp,kd)
                │ step()                     │ command_source()->Command
        ┌───────┴────────┐          ┌────────┴──────────────────────┐
        │ MuJoCoBackend  │          │ Homie / Falcon / Agile / Amo  │
        │ (= bench shim) │          │   / Sonic Adapter             │
        │ Sdk2Backend    │          │  ← each WRAPS the upstream     │
        │ (unitree_sdk2_ │          │    Python obs-builder verbatim│
        │  python, DDS)  │          │    + a method_registry.json   │
        └────────────────┘          └───────────────────────────────┘
```

- **`RobotBackend` protocol** — `read_state() -> StateBundle(q, dq, quat, ang_vel, gravity, ...)`, `write_cmd(q_target, kp, kd, tau_ff=0)`, `step()`. MuJoCo impl = today's bench shim (FALCON `MujocoStateShim`/`MujocoCmdShim` already this interface). `Sdk2Backend` = `LowState_`→read / `LowCmd_`→write over CycloneDDS; drops in with zero change to upstream deploy classes.
- **`MethodAdapter`** — carries the per-method config currently scattered as Python constants: obs dims/order/scales, action scale/DoF map/default pose, per-joint kp/kd, command-channel semantics (`wz` vs absolute-heading, height absolute vs delta, `stand` polarity), and **upstream-fix flags** (carry AGILE's `BodyFrameGyroShim` to hardware). Each adapter **wraps** the upstream obs-builder, never reimplements it.
- **`method_registry.json`** — promote `task_registry.json`'s pattern: one entry per method `{obs_dim, action_dim, action_scale, default_pose, kp[], kd[], joint_order, command_map{height:abs|delta, yaw:wz|heading, stand_polarity}, fixes:[gyro_shim], device}`.
- **Reuse the bench Mission/Schedule layer** as the default `command_source` (already backend-agnostic). Live teleop (pedal/VR/IK/keyboard) is an alternative source per archetype.
- **What to fork:** FALCON's `bench_falcon.py` shim pattern (`MujocoStateShim`/`MujocoCmdShim` + DDS-disabled subclass, lines 686-728) as the host skeleton — the cleanest existing `backend ⟷ deploy-class` boundary. Borrow sonic's safety-SM **logic** (INIT-ramp→WAIT→CONTROL, kd8 e-stop, stale/over-speed/temp guards), reimplemented in Python.

---

## 7. Risk & sequencing

**First hardware bring-up: HOMIE.** ① real HW-tested `unitree_sdk2` DDS deploy + complete safety sequence already exists; ② Archetype A (legs-only 12-DoF) = simplest, lowest-consequence; ③ obs-builder fully self-contained + already byte-validated in the bench; ④ blockers are small/local (the 5 fixes + 0.25 calibration); exoskeleton not needed — feed the 4-D command from keyboard/Mission layer (locomotion is separable from the exoskeleton).

**Order:** HOMIE → FALCON (Archetype B, CPU, cleanest shim, but clone repo + re-confirm obs width) → AGILE (Archetype A, ship gyro shim + IODescriptor already exported) → **AMO last** (most refactoring).

**Validate on the real G1, in order:**
1. Transport + safety SM on a hung/blocked robot — INIT-ramp, WAIT, kd8 e-stop, stale-state abort — **before any policy runs.**
2. Single obs round-trip — dump on-robot obs vs sim for the same pose, byte-match. (Silent falls hide here.)
3. Standing/zero-command hold → small vx → height changes → turning.

**Hardware-gated (can't validate off-robot):** DDS transport, real IMU/gravity vs sim sensors, contact/grasp physics, motor temp/effort limits, on-board PD vs host-reconstructed PD.

**Biggest sim2real-gap risks per archetype:**
- **A (HOMIE/AGILE):** AGILE `qvel[3:6]` double-rotation bug — gyro shim **must** ship to HW or it falls <0.5 s at non-zero yaw; AGILE IODescriptor export Isaac-gated (use already-exported `.pt/.yaml`, verify match); HOMIE `ang_vel_scale=0.25` calibration + C++/MuJoCo gain mismatch.
- **B (FALCON/sonic):** FALCON repo not local — clone first; public ONNX obs width may ≠ self-trained 575-d → confirm against loaded model, not docstring; alphabetical obs-sort trap; sonic stale "436" YAML comment (real 994); upper-body IK/VR is the hardest half + partly form-gated → deploy locomotion-only first, arms held.
- **C (AMO):** `cuda:0` baked into the TorchScript graph (needs healthy NVIDIA GPU/Orin; CPU impossible without re-tracing); **no real-robot transport at all** (write LowState→obs / action→LowCmd from scratch); `env.run()` loop inverted (must invert back to runner-driven); no hardware safety SM; arms open-loop random, not real teleop.

---

## 8. Key file references

- Method deploy entry points & seams:
  - sonic: `GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`, `.../include/policy_parameters.hpp`, `policy/release/observation_config.yaml`, `deploy.sh`
  - AMO: `AMO/upstream/play_amo.py`, `AMO/deploy/{run_amo.sh,smoke_test.py}`
  - HOMIE: `cc/experiments/repos/OpenHomie/HomieDeploy/g1_gym_deploy/{scripts/deploy_policy.py,envs/lcm_agent.py,utils/deployment_runner.py}`, `.../unitree_sdk2/g1_control.cpp`, `MujocoDeploy/mujoco_deploy_g1.py`
  - AGILE: `cc/experiments/repos/WBC-AGILE/scripts/sim2mujoco_eval.py`, `agile/sim2mujoco/{simulation,observations,actions,policy}.py`, `scripts/export_IODescriptors.py`
  - FALCON: `github.com/LeCAR-Lab/FALCON/sim2real/{rl_policy/base_policy.py,rl_policy/loco_manip/loco_manip.py,config/g1/g1_29dof_falcon.yaml,utils/{arm_ik,sdk2py_bridge}}`
- Harness seams to fork/reuse:
  - fork target (cleanest backend boundary): `cc/experiments/scripts/bench_falcon.py:686-728`
  - first adapter (inline obs): `cc/experiments/scripts/bench_homie.py:1709-1729`
  - gyro shim to port to HW: `cc/experiments/scripts/bench_agile.py:950,1215`
  - inverted-loop risk: `cc/experiments/scripts/bench_amo.py:904,2132`
  - config to promote: `cc/experiments/scripts/task_registry.json` → sibling `method_registry.json`
  - calibrations & known bugs: `cc/experiments/PROTOCOL.md`, `cc/experiments/scripts/NOTES_{HOMIE,AGILE,AMO,FALCON}.md`

---

## 9. OSS controller-framework comparison

Five open-source frameworks reverse-engineered (real source reads) for "取其精华弃其糟粕".

| Framework | Repo | License | Robot/SDK | Sim↔real seam | Hardware-tested |
|---|---|---|---|---|---|
| **unitree_rl_gym** | `unitreerobotics/unitree_rl_gym` | BSD-3 ✅ | G1/H1/Go2, `unitree_sdk2py` DDS | ❌ **absent** (forked scripts, obs typed twice) | ✅ |
| **RoboJuDo** | `HansZ8/RoboJuDo` | ⚠️ README NC vs LICENSE BY conflict | G1/H1/gr1t1, `unitree_sdk2py` + C++ pybind | ✅ **`Environment` ABC, `step(pd_target)`** | ✅ (10+ policies) |
| **booster_gym** | `BoosterRobotics/booster_gym` | Apache-2.0 ✅ | Booster T1, `booster_robotics_sdk_python` | ❌ weak (fixed in successor `booster_deploy`) | ✅ |
| **humanoid-gym** | `roboterax/humanoid-gym` | BSD-3 ✅ | XBot 12-DoF (NOT G1) | ❌ sim2sim only, real path proprietary | vendor only |
| **unitree_mujoco** | `unitreerobotics/unitree_mujoco` + `unitree_sdk2_python` | BSD-3 ✅ | all Unitree, CycloneDDS | ✅ **DDS-loopback mirror** (same wire, `(1,'lo')` vs `(0,nic)`) | ✅ |

**Headline findings:**
- **RoboJuDo is ~our target design already** — copy the `Environment` ABC seam, the name-based `DoFAdapter` joint remap, the 3-phase `prepare()` ramp/blend startup, and the COMMANDS bus + multi-policy `PolicyManager`. ⚠️ But license is ambiguous (README badge says NC, LICENSE file says BY) → borrow *interface shape*, not code verbatim, until clarified.
- **unitree_rl_gym + unitree_mujoco** = the BSD-clean transport recipe to steal wholesale: `ChannelFactoryInitialize(domain,iface)`, `rt/lowcmd`/`rt/lowstate` **`unitree_hg`** msgs for G1, mandatory `cmd.crc = CRC().Crc(cmd)`, the `leg/arm_waist_joint2motor_idx` remap, PD-on-motor `(q,dq,kp,kd,tau)` contract, and the timerfd-based `RecurrentThread` for drift-free pacing.
- **The seam every framework lacks is our whole opportunity.** 4 of 5 fork sim/real into separate scripts that re-type the obs assembly → guaranteed drift. Our `RobotBackend` runs obs+safety **once** over either backend.

**精华 (borrow):** `Environment`/`BaseController` ABC seam (RoboJuDo, booster_deploy) · DDS wiring + CRC + joint2motor remap (unitree_rl_gym, unitree_mujoco) · name-based DoF adapter (RoboJuDo) · 3-phase ramp-from-measured-q startup (all) · dual-clock 50/500 Hz loop + EMA action filter (booster_gym) · timerfd `RecurrentThread` pacing (unitree_sdk2_python) · MuJoCo-as-mandatory-gate + TorchScript I/O boundary (humanoid-gym) · `(domain,iface)` as the only sim/real diff (unitree_mujoco).

**糟粕 (avoid):** naive `time.sleep(dt−elapsed)` pacing (all examples) · copy-paste sim/real obs duplication (unitree_rl_gym, booster, humanoid-gym) · **commented-out/absent hardware joint+torque clamps** (RoboJuDo, booster) · safety = "human holds the remote", damping sent once on exit (all) · index-addressed obs DSL & magic slices (booster, humanoid-gym) · two parallel config schemas (booster, humanoid-gym) · global DDS/CRC singletons leaking into method code · `__getattr__` magic delegation (RoboJuDo) · `cuda:0` baked into traced graph (AMO/HOMIE/sonic) · inverted loop ownership (AMO, humanoid-gym).

---

## 10. Our framework — design & build plan

### 10.1 The big decision — MuJoCoBackend: in-process (default) + DDS-mirror (optional CI)
`MujocoBackend` is a **plain in-process MuJoCo loop** (fast, deterministic, Mac-friendly, **promotes the already-working bench shims** `MujocoStateShim`/`MujocoCmdShim`/`BodyFrameGyroShim` 1:1). The `RobotBackend` ABC's single `step(pd_target)` contract (from RoboJuDo) already gives ~90% of the anti-drift benefit, so we don't pay the DDS-mirror's cost on the everyday path. **DDS-mirror** (point `Sdk2Backend` at a running `unitree_mujoco` on `(domain=1, iface='lo')`) is kept as an **integration-test flag** that exercises the real DDS+CRC+IDL+`LowCmdHG` wire path with **zero hardware** — invaluable, but narrow, so it's CI not default.

### 10.2 `Sdk2Backend` G1 checklist (must get right)
G1 = **`unitree_hg`** IDL (29 joints in a 35-slot `LowCmdHG`/`LowStateHG`). Per write: `cmd.crc = CRC().Crc(cmd); pub.Write(cmd)`. Set top-level **`mode_pr = 0` (PR mode)** + per-motor `mode=0x01`; **echo `mode_machine` from the incoming `LowState_`** (firmware handshake — don't hardcode). Per-motor cmd = `(q=target, dq=0, kp, kd, tau=0)`; firmware runs PD. Read `motor_state[].q/.dq/.tau_est`, `imu_state.quaternion/gyroscope/rpy`, `wireless_remote` (40-byte button decode for FSM), `tick` (freshness watchdog). **Weak motors** (wrists/some arm) get softer gains + lower torque limits in the clamp table. Pre-write: clamp `q_target` to position limits + torque limits + max-delta rate-limit, else emit damping.

### 10.3 Module layout (`g1_runner/`)
```
g1_runner/
  cli.py                    # entry: --method --backend mujoco|sdk2 --domain --iface
  registry.py + method_registry.json   # per-method config (single source of truth), validated dataclass
  backends/  base.py(RobotBackend ABC) state.py(RobotState) mujoco_backend.py sdk2_backend.py sdk2_msgs.py sdk2_crc.py
  runtime/   runner.py(UnifiedRunner) rate.py(timerfd pacing) writer.py(500Hz+EMA) command_bus.py rt_sched.py
  safety/    state_machine.py(ZERO_TORQUE→RAMP→HOLD→POLICY→DAMPING) guards.py(tilt/stale/clamp) ramp.py
  adapters/  base.py(MethodAdapter ABC) homie.py agile.py falcon.py amo.py  _upstream/
  utils/     rotation.py remote.py logging.py
  tests/     test_registry / test_mujoco_backend / test_safety_sm / test_sdk2_msgs / test_homie_adapter
```
Domain seams map 1:1: `backends/`=RobotBackend, `runtime/`=UnifiedRunner, `safety/`=SafetyStateMachine, `adapters/`=MethodAdapter, `registry.py`=config.

### 10.4 `method_registry.json` schema (per method)
`{ policy_path, policy_format(onnx|torchscript), device, num_obs, num_actions, obs_history_len, obs_clip, joint_names[], leg_joint2motor_idx[], arm_waist_joint2motor_idx[], default_angles[], kp[], kd[], torque_limit[], position_limit_lower/upper[], weak_motor_idx[], obs_scales{ang_vel,dof_pos,dof_vel,projected_gravity,command[]}, action_scale, action_clip, policy_dt, control_dt, decimation, action_filter_beta, max_delta_per_step, command_source(keyboard|gamepad|tape|pedal), command_dim, command_is_absolute_height, command_defaults[], msg_type(hg|go), mode_pr, topic_lowcmd, topic_lowstate, mujoco_model_path, tilt_limit_rad, state_timeout_ms, joint_overspeed_rad_s, ramp_to_default_s, damping_kd }`. Loaded via dataclass with **fail-fast validation** (all per-joint arrays == `len(joint_names)`; `num_obs` cross-checked vs the policy's actual input dim at first inference — RoboJuDo's silent-misalignment trap).

### 10.5 Design rule — obs stays wrapped, not abstracted
The prior memo proved **5 mutually-incompatible obs byte-layouts** (encoder token / alphabetical sort / dual-deque / 6-frame / LSTM). So we **do NOT** build a declarative obs-term DSL (booster/humanoid-gym's footgun, over-corrected). Instead each `MethodAdapter.build_obs` **wraps the upstream obs assembly verbatim** (the bench already did this for all 4), and the registry carries only *scales/dims/joint-order/history-len*. Frame quirks (AGILE `BodyFrameGyroShim`, HOMIE `ang_vel_scale=0.25`) are baked into the **method adapter**, never normalized away.

### 10.6 Build order (5 steps)
1. **Bare transport + state roundtrip** (no policy/safety): `RobotBackend` ABC + `RobotState` + `Sdk2Backend` + CRC. Prove `(1,'lo')` against `unitree_mujoco`: read `LowStateHG`, hold-at-measured-q `LowCmdHG` w/ CRC @500 Hz via `RecurrentThread`.
2. **Safety SM bare** (still no policy): `ZERO_TORQUE→(start)→RAMP_TO_DEFAULT(from measured q,2s)→HOLD→(A)→POLICY-stub→(tilt/stale/select)→DAMPING(held)`. Trip-test tilt + kill-publisher.
3. **MujocoBackend (in-process) + UnifiedRunner skeleton**: promote bench shims; dual-clock loop; `--backend mujoco|sdk2` selects, runner code unchanged ⇒ seam proven.
4. **registry.py + method_registry.json + MethodAdapter ABC**: dataclass loader + validation + onnx/torchscript policy load + a hold-default adapter.
5. **HOMIE adapter (first real policy, legs-only)**: wrap OpenHomie obs verbatim (456=6×76, `ang_vel_scale=0.25`), run in MuJoCo vs `bench_homie` golden, then DDS-mirror — ready for real-hardware `--backend sdk2 --domain 0 --iface <nic>`.

### 10.7 Assets to promote (not rewrite)
`bench_falcon.py:686` `MujocoStateShim`/`MujocoCmdShim` → `MujocoBackend`; `bench_agile.py:950` `BodyFrameGyroShim` → `adapters/agile.py`; per-method obs assemblies in `bench_{homie,agile,falcon,amo}.py` → wrapped verbatim in `adapters/*.py`.

> Full per-framework analysis + borrow/avoid tables: `cc/experiments/` workflow output (framework-survey). This section is the implementation blueprint; code lives in `g1_runner/`.
