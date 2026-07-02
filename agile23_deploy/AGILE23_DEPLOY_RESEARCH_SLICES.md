# AGILE 23-DoF 部署调研 — 分项原始记录


---

# obs — AGILE velocity-height recurrent student observation vector contract (23-DoF G1 deploy)

## The student obs vector (deploy contract)

The deploy artifact is the **recurrent LSTM student**, whose obs group is `StudentVelocityPolicyCfg` (velocity_height_env_cfg.py:109-136). At runtime, sim2mujoco's `ObservationProcessor` does NOT read this Python class — it reads the **exported IODescriptor YAML** (`agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml`), parsing `config["observations"]["policy"]` as an ordered list (observations.py:303-307). **The YAML's `policy` list order is the ground-truth obs order, and it differs from the Python class order** (YAML puts `generated_commands` FIRST; the Python ObsGroup lists it first too via `velocity_height_commands`, so they agree).

### Exact ordered obs terms (from the checked-in 29-DoF YAML; this is what the loaded student consumes)
Order, name, raw dim, scale, clip, noise (train-only), units:
1. `generated_commands` — dim **4** = `[vx, vy, wz, height]`, scale=null, clip=null. Computed by `_compute_generated_commands` → `command_manager.get_command()` → tensor `[linear_x, linear_y, angular_z, height]` (commands.py get_command). vx,vy in m/s; wz in rad/s; height in m **absolute**. NOTE: for a velocity-only (3-dim) policy this term auto-truncates to `[vx,vy,wz]`, but this student expects 4 (shape:[4]).
2. `base_ang_vel` — dim **3** = gyro `[wx,wy,wz]` in root frame, rad/s, scale=null, noise Unoise(-0.2,0.2). Computed `_compute_base_ang_vel` = `sim_state.root_ang_vel` (raw IMU gyro).
3. `projected_gravity` — dim **3**, scale=null, noise Unoise(-0.01,0.01). Computed `_compute_projected_gravity` = `quat_rotate_inverse(root_quat, [0,0,-1])` (UNIT gravity, not 9.81; YAML `units: m/s^2` is cosmetic). This is the gravity direction in body frame.
4. `joint_pos_rel` — dim **29** (29-DoF) / will be **23** (23-DoF), scale=null, noise Unoise(-0.01,0.01). Computed `_compute_joint_pos_rel` = `joint_pos[joint_indices] - joint_offsets`. `joint_offsets` = the YAML `joint_pos_offsets` = `articulations.robot.default_joint_pos`. rad.
5. `joint_vel_rel` — dim **29** / **23**, **scale=0.1**, noise Unoise(-1.5,1.5). Computed = `(joint_vel - joint_vel_offsets)*0.1`; `joint_vel_offsets` all 0.0. rad/s pre-scale.
6. `last_action` — dim **12**, clip=(-10,10), scale=null. Computed `_compute_last_action` = previous raw policy action (the 12-dim leg action), zeros on first step.

### Total obs dim
- **29-DoF**: 4 + 3 + 3 + 29 + 29 + 12 = **80** ✓ (matches the "80 for 29-DoF" in the brief).
- **23-DoF**: 4 + 3 + 3 + 23 + 23 + 12 = **68**. (joint_pos_rel/joint_vel_rel shrink from 29→23; commands/gyro/grav/last_action are fixed.) **NOTE: student does NOT include `base_lin_vel`** — that term is privileged (teacher/critic only, env_cfg.py:52, 83). The student blindly estimates linear velocity via the LSTM. Do not feed lin_vel.

### joint_pos_rel offset (default_joint_pos) — the squat-relevant non-zero entries
In the YAML's 29-joint order: hip_pitch=-0.10 (both), knee=+0.30 (both), ankle_pitch=-0.20 (both); ALL other joints (waist, arms, hip_roll/yaw, ankle_roll) = 0.0. (env_cfg init_state unitree_g1.py:116-124.) For 23-DoF the same leg offsets apply; the absent 6 joints (waist_roll/pitch, wrist_pitch/yaw ×2) simply drop out of the list.

### CRITICAL: the checked-in YAML is 29-DoF, not 23-DoF
`joint_pos_rel`/`joint_vel_rel` use `SceneEntityCfg("robot")` with NO joint filter → io_descriptors.py:118-121 returns the FULL articulation joint list in USD/MJCF DOF order. The committed YAML lists **29 joints** in this exact interleaved order (NOT grouped legs-first): `left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, right_hip_roll, waist_roll, left_hip_yaw, right_hip_yaw, waist_pitch, left_knee, right_knee, left_shoulder_pitch, right_shoulder_pitch, left_ankle_pitch, right_ankle_pitch, left_shoulder_roll, right_shoulder_roll, left_ankle_roll, right_ankle_roll, left_shoulder_yaw, right_shoulder_yaw, left_elbow, right_elbow, left_wrist_roll, right_wrist_roll, left_wrist_pitch, right_wrist_pitch, left_wrist_yaw, right_wrist_yaw`. **For the 23-DoF deploy you MUST re-export a 23-DoF IODescriptor** from task `Velocity-Height-G1-23dof-Distillation-Recurrent-v0` (registered, g1/__init__.py:74-78, cfg `G1VelocityHeight23dofRecurrentStudentEnvCfg`) via `scripts/export_IODescriptors.py`. The 23-DoF articulation order will be the same interleaved sequence minus the 6 absent joints, but **the exact DOF order must be read from the re-exported YAML / verified on the real robot's rt/lowstate motor index map** — do not hand-derive it.

### How the IODescriptor maps sensors→obs (auto-adapt)
`ObservationProcessor.__init__` (observations.py:282-355): builds `self.terms` from the YAML `observations.policy` list. For joint terms it stores `joint_indices = [full_joint_names.index(jn) for jn in term joint_names]` where `full_joint_names` = `articulations.robot.joint_names` (the sim's joint list). It validates `joint_indices` re-maps back to the same names (raises on mismatch, observations.py:339-348) and validates each term's `output_dim()` equals YAML `shape[0]×history` (observations.py:325-336). So if you supply a 23-DoF YAML + a 23-DoF `joint_names`, the processor auto-adapts: 23-dim joint terms, indices into the 23-joint state. **The mapping is name-based**, so the real robot's rt/lowstate must be assembled into a vector whose index order equals `articulations.robot.joint_names`.

### History / LSTM
The recurrent student YAML sets `history_length: 0` on every term (no obs history concatenation) — temporal memory lives entirely in the LSTM, not in obs stacking. (The separate `G1VelocityHeightHistoryStudentEnvCfg` with history_length=5 is a DIFFERENT artifact — do not use it.) The LSTM hidden/cell buffers are zeroed once at construction (`_CheckpointInferenceModel.__init__` registers `hidden_state`/`cell_state` zeros, policy.py; `reset()`/`reset_hidden()` re-zeros) and updated in-graph each forward (policy.py forward: `self.hidden_state[:]=h`, `self.cell_state[:]=c`). Per the brief: zero ONCE before the loop, never re-zero in-loop.

### Processing order per term (observations.py:178-205)
`raw → noise(train only, off at deploy: enable_corruption=False in eval, env_cfg.py:861/903) → clip → scale → (history, here none)`. At deploy noise_scale=0, so noise is skipped. So effectively: raw → clip(last_action only) → scale(joint_vel_rel ×0.1).

### Action side (for completeness; separate slice)
Policy output = 12-dim leg action (left/right ×: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll). ActionProcessor: `joint_pos = action*scale + offset`, clip first to (-6,6). scale=[0.5475,0.5475,0.3507,0.3507,0.5475,0.5475,0.3507,0.3507,0.4386,0.4386,0.4386,0.4386], offset=[-0.10,-0.10,0,0,0,0,0.30,0.30,-0.20,-0.20,0,0] (= G1_ACTION_SCALE_LOWER = 0.25*effort_limit/stiffness; offset = leg default_joint_pos). Action joint order is leg-grouped, DIFFERENT from obs joint order — verify motor index map on robot.

## key_facts
- Deploy artifact = recurrent LSTM student; obs group StudentVelocityPolicyCfg (velocity_height_env_cfg.py:109-136); runtime reads exported IODescriptor YAML not the Python class.
- Student obs order (from YAML observations.policy list): [generated_commands(4), base_ang_vel(3), projected_gravity(3), joint_pos_rel(N), joint_vel_rel(N), last_action(12)].
- Total obs dim: 29-DoF = 80; 23-DoF = 68 (joint_pos_rel & joint_vel_rel shrink 29→23; other 4 terms fixed at 4+3+3+12=22).
- STUDENT HAS NO base_lin_vel — lin_vel is privileged (teacher/critic only). LSTM estimates it. Never feed lin_vel.
- generated_commands = command_manager.get_command() = [linear_x(m/s), linear_y(m/s), angular_z(rad/s), height(m absolute)]; this 4D is consumed whole (no truncation for this student).
- projected_gravity = quat_rotate_inverse(root_quat, [0,0,-1]) — UNIT gravity direction (magnitude 1, NOT 9.81); YAML units:m/s^2 is cosmetic.
- base_ang_vel = raw root/IMU gyro [wx,wy,wz] rad/s, no scale.
- joint_pos_rel = q[idx] - default_joint_pos; offsets non-zero only for legs: hip_pitch=-0.10, knee=+0.30, ankle_pitch=-0.20 (both sides); everything else 0.
- joint_vel_rel: scale=0.1 (the ONLY scaled obs term); vel offsets all 0.
- last_action = previous 12-dim leg action, clip(-10,10), zeros on first step.
- joint_pos_rel/joint_vel_rel/articulation joint_names use SceneEntityCfg('robot') with no filter → FULL articulation DOF order (interleaved, NOT legs-first): left_hip_pitch,right_hip_pitch,waist_yaw,left_hip_roll,right_hip_roll,waist_roll,left_hip_yaw,right_hip_yaw,waist_pitch,left_knee,right_knee,left_shoulder_pitch,right_shoulder_pitch,left_ankle_pitch,right_ankle_pitch,... (29-DoF).
- CRITICAL: checked-in YAML (unitree_g1_velocity_height_recurrent_student.yaml) is 29-DoF. Deploy needs a re-exported 23-DoF YAML from task Velocity-Height-G1-23dof-Distillation-Recurrent-v0.
- history_length=0 on all terms for this recurrent student — no obs stacking; memory is purely LSTM. (Do NOT confuse with G1VelocityHeightHistoryStudentEnvCfg history_length=5.)
- LSTM hidden/cell zeroed once at model construction & on reset(); updated in-graph each forward (policy.py). Never re-zero in loop.
- At deploy enable_corruption=False / noise_scale=0 → all noise skipped. Effective per-term pipeline: raw → clip(last_action) → scale(joint_vel_rel*0.1).
- IODescriptor maps sensors→obs by NAME: joint_indices = [articulations.robot.joint_names.index(jn) ...]; processor validates dims & name round-trip, raising on mismatch (observations.py:325-348).
- controller_freq=50Hz, physics_freq=200Hz, decimation=10 in env (env_cfg.py:817-823); but exported YAML 'scene' says physics_dt=0.005, dt=0.02, decimation=4 (50Hz control, 200Hz physics) — verify which the eval loop uses.
- Action: 12-dim leg policy output → joint_pos = clip(a,-6,6)*scale + offset; scale~[0.548×4_hip_pitch/yaw_pattern...], offset=leg defaults. Action joint order is leg-grouped, DIFFERENT from obs order.

## open_questions
- The checked-in IODescriptor YAML is 29-DoF (joint_pos_rel shape=29, 29-joint articulation). A 23-DoF YAML must be RE-EXPORTED via scripts/export_IODescriptors.py --task Velocity-Height-G1-23dof-Distillation-Recurrent-v0. Must confirm the recurrent_student .pt/.onnx weights match a 23-DoF input (68) vs 29-DoF (80) — the committed weights may be 29-DoF-trained. VERIFY input dim of the actual deploy checkpoint.
- Exact 23-DoF articulation DOF order (the joint_names list that defines obs joint indexing) must be read from the re-exported YAML AND cross-checked against the real G1 rt/lowstate motor index map. Do NOT hand-derive by deleting 6 joints — USD DOF ordering is authoritative.
- The real robot's rt/lowstate must be assembled into a vector indexed in EXACTLY articulations.robot.joint_names order before feeding joint_pos_rel/joint_vel_rel. The motor→joint-name mapping on the physical 23-DoF G1 (and which 23 of the 29 motor slots are populated) must be verified on hardware.
- projected_gravity uses unit gravity [0,0,-1] (magnitude 1). Confirm the real IMU/orientation source yields a quaternion in the same [w,x,y,z] convention as quat_rotate_inverse expects (utils.py) — sign/convention mismatch silently corrupts this term.
- base_ang_vel is raw gyro with NO scale and (at deploy) NO noise — confirm the robot gyro units are rad/s and frame is body/root frame matching sim.
- generated_commands height is ABSOLUTE pelvis height in meters, clamped by CommandManager to height_range=(0.4,0.72) in code (commands.py) but the env training range is (0.20, 0.72) (env_cfg.py:263). The robojudo_ext_cmd height must be mapped to absolute m; verify whether the deploy command provider uses the (0.4,0.72) clamp or the wider training range — a clamp at 0.4 would block the deep-squat 0.20 target.
- Scene timing: env cfg says 50Hz control / 200Hz physics (decimation=10), but exported YAML scene block says dt=0.02, physics_dt=0.005, decimation=4. Confirm the real control loop runs at 50Hz (dt=0.02) to match training.
---

# actpolicy

## Action mapping (actions.py)

Two-level design: `ActionProcessor` owns N `ActionTerm`s. For the velocity_height G1 student there is exactly ONE term, `joint_position_action`, with **action_dim = 12** (lower body legs only), NOT 29 and NOT 23. The action vector the LSTM emits is 12-D regardless of robot DoF count.

**Per-joint pipeline (`ActionTerm.process`, actions.py:64-86):**
1. Clip raw action first: per-joint `[-6.0, 6.0]` (YAML `actions[0].clip`, 12 pairs). Clip happens BEFORE scale/offset.
2. `joint_positions = actions * self.scale + self.offset`.
   - `scale` is a per-joint tensor (YAML flattens nested `[[...]]` → 12 values): hips_pitch/yaw = 0.5475464, hips_roll = 0.3506615, knees = 0.3506615... actually per YAML order the 12 scales are: [0.5475, 0.5475, 0.3507, 0.3507, 0.5475, 0.5475, 0.3507, 0.3507, 0.4386, 0.4386, 0.4386, 0.4386].
   - `offset` (12) = default leg pose: [-0.1, -0.1, 0,0,0,0, 0.3,0.3, -0.2,-0.2, 0,0]. NOTE: the offset, not a separate `default_joint_pos` add, is what re-centers the leg targets. There is NO additional `default_joint_pos` addition for action joints — the offset already encodes it.
3. **There is NO clip to `joint_pos_limits` in the action path.** `default_joint_pos_limits` exists in YAML (29 pairs) but ActionProcessor never reads it. Only the `[-6,6]` raw-action clip applies. (A real-robot deploy should add its own joint-limit safety clamp.)

**ActionProcessor.process (actions.py:175-196):** starts `joint_positions = zeros_like(default_joint_pos)` (length = num MJCF joints, e.g. 29), writes the 12 processed values into `term.joint_indices` (the indices of the 12 action joints within the full joint order), and returns `JointCommand(position, kp, kd)` with **kp/kd for ALL joints** (full length, e.g. 29), not just 12.

**Action joint order (the 12, YAML actions[0].joint_names, actions.py maps via full_joint_names.index):** left_hip_pitch, right_hip_pitch, left_hip_roll, right_hip_roll, left_hip_yaw, right_hip_yaw, left_knee, right_knee, left_ankle_pitch, right_ankle_pitch, left_ankle_roll, right_ankle_roll. CRITICAL: this leg order differs from the IsaacLab articulation order (which interleaves waist). `joint_indices` resolves this against `sim.joint_names` (MJCF order). For real-robot deploy you must map these 12 names to the DDS motor indices yourself.

**kp/kd/default source (actions.py:137-152):** read from `config["articulations"]["robot"]` keys `default_joint_stiffness` (kp), `default_joint_damping` (kd), `default_joint_pos`, each parsed by `_parse_joint_values` which remaps from YAML/articulation `joint_names` order (29) into the MJCF/sim `joint_names` order. So kp/kd are defined for all 29 joints. Leg kp values (articulation order): hip_pitch 40.179, hip_roll 99.098, hip_yaw 40.179, knee 99.098, ankle_pitch 28.501, ankle_roll 60.0; waist_yaw 300, waist_roll 300, waist_pitch 300; arms 4–90. Leg kd: hip_pitch 2.5579, hip_roll 6.3088, hip_yaw 2.5579, knee 6.3088, ankle_pitch 1.8144, ankle_roll 1.0.

For the 23-DoF target: the 12-D action stays identical (legs unchanged). For the held waist_yaw + 10 arms you supply kp/kd/q from these same articulation entries (waist_yaw 300/5.0, shoulders/elbows/wrists per arm rows), holding position at measured q so the arm_sdk overlay survives.

## PolicyWrapper + LSTM (policy.py)

`PolicyWrapper.from_config` (policy.py:~250): `.onnx` → `ONNXPolicyWrapper`; else tries `torch.jit.load` (TorchScript) → `MLPPolicyWrapper`/`RNNPolicyWrapper`; on RuntimeError → `CheckpointPolicyWrapper.from_checkpoint`. **For this student the YAML has NO top-level `policy:` block** (the `policy:` at YAML line 2 is the observation group `observations.policy`, not a config). So `config.get("policy", {})` → `{}`, activation defaults to **"elu"**.

The shipped `.pt` (`unitree_g1_velocity_height_recurrent_student.pt`, 2.07MB) is the JIT/TorchScript export; the `_checkpoint.pt` (6.65MB) is the raw RSL-RL training checkpoint that goes through `CheckpointPolicyWrapper`. `.onnx` also present.

**LSTM state — `_CheckpointInferenceModel` (policy.py:~160-215):** RNN built as `nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, batch_first=False)`. Two **registered buffers**: `hidden_state` and `cell_state`, each shape `(num_layers, hidden_dim)`, initialized to zeros at construction. `_infer_architecture` decides lstm vs gru by gate size (`4*hidden` → lstm).

**Forward (policy.py:~190-210, THE in-graph copy_ trap):**
```
x = normalizer(x)                        # (obs_dim,)
x = x.unsqueeze(0).unsqueeze(0)          # (seq=1, batch=1, input)
hidden = (hidden_state.unsqueeze(1), cell_state.unsqueeze(1))  # (L,1,H)
x, (h, c) = self.rnn(x, hidden)
self.hidden_state[:] = h.squeeze(1)      # in-graph copy_ — carries state across ticks
self.cell_state[:]  = c.squeeze(1)
x = x.squeeze(0).squeeze(0)
out = self.actor(x)                      # ELU MLP
if noise_std_type == "pred": out = out[:num_actions]   # take mean only
```
The state is carried via `self.hidden_state[:] = ...` / `self.cell_state[:] = ...` (slice-assign = `copy_`). It is created+zeroed ONCE in `__init__`. `reset_hidden()` (policy.py:~217) zeros both buffers. `CheckpointPolicyWrapper.reset()` calls `model.reset_hidden()`.

**Stateful call signature the deploy must drive each tick:** `actions = policy(obs)` where `obs` is a 1-D `torch.Tensor` (shape `(obs_dim,)`), returns 1-D `(12,)` tensor. NO hidden state passed in/out by the caller — it lives inside the model buffers. Contrast `RNNPolicyWrapper` (TorchScript path, policy.py:~300): there `self.hidden` IS caller-managed (`output, self.hidden = self.model(obs.unsqueeze(0), self.hidden)`, default `hidden_shape=[2,1,128]`), and `.reset()` re-zeros `self.hidden`. ONNX path (`ONNXPolicyWrapper`) is stateless single-input — NOT usable for the recurrent student as-is.

**Silent-drift trap for deploy:** call `policy.reset()` ONCE before the loop (after model build), NEVER inside the loop. Driving the `.pt`/checkpoint wrapper means you must NOT re-instantiate or re-zero `hidden_state`/`cell_state` per tick. If you re-export to ONNX you must thread h/c manually (the in-graph buffer trick won't survive ONNX statelessness).

## simulation.py (SimState↔JointCommand, gyro fallback)

`SimState` (sim.py:28-49): `joint_pos, joint_vel (num_joints,), root_pos(3), root_quat[w,x,y,z](4), root_lin_vel(3, ROOT frame), root_ang_vel(3, ROOT frame), joint_effort, anchor_body_*`. `JointCommand` (52-61): `position, kp, kd` all full-length.

Stateful API the deploy replaces: `sim.get_state() -> SimState` (sim.py:371) and `sim.step(JointCommand)` (447). For real robot: synthesize SimState from rt/lowstate (joint q/dq from motors in the policy's joint order; root_ang_vel from IMU gyro; projected_gravity from IMU quat; root_lin_vel typically unobservable on hardware).

**"No-sensor gyro fallback" region (sim.py:394-421):** when MuJoCo named velocity sensors `linear-velocity`/`angular-velocity` are absent, it falls back to computing from `qvel` and transforms world→root via `quat_rotate_inverse`. On the real robot the gyro already reports body-frame angular velocity, so this fallback math is sim-only — the deploy feeds the IMU gyro directly as `root_ang_vel`. Verify the obs sign/frame conventions match (projected_gravity & base_ang_vel are the IMU-derived obs terms).

**PD application (sim.py:447-495):** position-actuator mode writes `mj_data.ctrl = position[_joint_to_ctrl_indices]` and pushes kp/kd into `actuator_gainprm[:,0]=kp`, `actuator_biasprm[:,1]=-kp`, `biasprm[:,2]=-kd`. Torque mode: `tau = kp*(pos_target - q) + kd*(0 - dq)`. The real robot's DDS lowcmd carries q/kp/kd/(tau=0) per motor, so torque-mode formula = the on-robot PD law. `scene`: physics_dt=0.005, control dt=0.02 (50 Hz), decimation=4 — policy ticks at **50 Hz**.

## key_facts
- Action dim = 12 (legs only) for the velocity_height student — independent of 23 vs 29 DoF. Only ONE action term: joint_position_action.
- Action pipeline: clip raw to [-6,6] FIRST, then joint_positions = action*scale + offset. No joint_pos_limits clamp anywhere in the action path — deploy must add its own safety clamp.
- Per-joint scale (12): [0.5475,0.5475,0.3507,0.3507,0.5475,0.5475,0.3507,0.3507,0.4386,0.4386,0.4386,0.4386]; offset (12)=[-0.1,-0.1,0,0,0,0,0.3,0.3,-0.2,-0.2,0,0]. offset already encodes default leg pose; no separate default_joint_pos add for action joints.
- 12 action joint order: L/R hip_pitch, L/R hip_roll, L/R hip_yaw, L/R knee, L/R ankle_pitch, L/R ankle_roll. Differs from articulation order (which interleaves waist) — joint_indices remaps via name lookup.
- kp = articulations.robot.default_joint_stiffness, kd = default_joint_damping, both 29-long, remapped YAML->MJCF order by _parse_joint_values. Leg kp: hip_pitch40.18 hip_roll99.10 hip_yaw40.18 knee99.10 ankle_pitch28.50 ankle_roll60. Leg kd: 2.558/6.309/2.558/6.309/1.814/1.0.
- JointCommand.kp/kd/position are FULL length (29), not 12. Non-action joints get position 0 unless held — deploy holds arm/waist at measured q.
- Deploy artifact = recurrent LSTM student. Shipped .pt (2.07MB)=TorchScript export, _checkpoint.pt(6.65MB)=raw RSL-RL ckpt (uses CheckpointPolicyWrapper), .onnx also present.
- YAML has NO top-level policy: block (the line-2 policy: is observations.policy). So config.get('policy',{}) -> {}, activation defaults to 'elu', and from_config auto-detects RNN via _infer_architecture.
- LSTM state = registered buffers hidden_state & cell_state, shape (num_layers, hidden_dim), zeroed ONCE in __init__. Carried across ticks via in-graph copy_: self.hidden_state[:]=h.squeeze(1); self.cell_state[:]=c.squeeze(1).
- Per-tick stateful call: actions = policy(obs); obs is 1-D tensor (obs_dim,), returns 1-D (12,). No h/c passed by caller for CheckpointPolicyWrapper — state lives in model buffers. Call policy.reset() ONCE before the loop, NEVER inside (silent-drift trap).
- noise_std_type=='pred' -> actor output sliced out[:num_actions] (mean only). Output noise/std logits discarded.
- Control rate: physics_dt=0.005, decimation=4, control dt=0.02 => policy ticks at 50 Hz.
- Real-robot mapping: torque-mode PD (sim.py:477) tau=kp*(target-q)+kd*(-dq) == on-robot DDS lowcmd PD law. Gyro fallback (sim.py:394-421) is sim-only; feed IMU gyro directly as root_ang_vel.

## open_questions
- LSTM exact dims (num_layers, hidden_dim, rnn_input_dim) could not be confirmed at runtime — torch not importable in the plain ssh shell and onnx module missing. Architecture is auto-inferred at load by _infer_architecture from the checkpoint state dict (memory_a.rnn.* gate size). Must dump shapes inside the project's conda/uv env (where torch lives) before coding the buffer init. The RNNPolicyWrapper default hidden_shape=[2,1,128] HINTS 2 layers x 128 hidden but that default is only used for the TorchScript+explicit-policy-config path, NOT this checkpoint, so do NOT assume it.
- Whether to deploy the .pt (TorchScript, JIT — but then which wrapper? it has no top-level policy block so from_config would still go MLP/RNN path and likely mishandle state) vs the _checkpoint.pt (CheckpointInferenceModel, in-graph buffers, cleanest stateful semantics). Recommend the _checkpoint.pt path for deploy because its LSTM state handling is explicit and zeroed-once. Verify the TorchScript .pt's exported state-carry semantics before choosing it.
- rnn_input_dim vs total obs dim: the LSTM input may be the raw obs or a pre-MLP embedding (memory_a). Confirm obs vector length feeding policy(obs) matches rnn_input_dim (normalizer applied first). The obs builder (observations.py) is out of this slice's scope but must agree.
- No 23-DoF MJCF/config exists in WBC-AGILE (find for *23dof* and scene*.xml returned nothing under repo; examples use scene_29dof.xml). The 12-D leg action and leg kp/kd are reusable verbatim, but the held waist_yaw(1)+arms(10) slots must be filled from the 29-long articulation kp/kd/default arrays — verify the real 23-DoF motor index map matches these names on the physical robot.
- joint_pos_limits are NOT enforced in the action path; on hardware add an explicit per-joint position clamp (limits available in YAML default_joint_pos_limits, 29 pairs) before publishing rt/lowcmd_rl.
- root_lin_vel is part of SimState but is generally unobservable on the real robot; confirm the student obs terms (per the student YAML: generated_commands[4], base_ang_vel[3], projected_gravity[3], joint_pos_rel[29], joint_vel_rel[29 scale 0.1], last_action[12]) do NOT include base_lin_vel — the listed obs do not, so this is fine, but verify obs assembly order/scale on robot.
---

# keyboard

## AGILE keyboard teleop + command schema (verbatim, source-grounded)

### Command fields and the command vector
The command vector is produced by `CommandManager` in `agile/sim2mujoco/commands.py`. There are exactly 4 scalar fields, internally named `linear_x`, `linear_y`, `angular_z`, `height`. The CLI/user-facing aliases are `vx`, `vy`, `wz`, `height` (mapping in `command_scheduler.py:30-35` `FIELD_SPECS`).

- `get_command()` (commands.py:166-176) returns a `torch.Tensor` shape **(4,)** = `[linear_x, linear_y, angular_z, height]` → i.e. `[vx, vy, wz, height]`.
- `get_navigation_command()` (commands.py:178-188) returns shape **(3,)** = `[vx, vy, wz]` (height dropped).

The provider factory (`command_provider.py`) decides 3D vs 4D from the policy's observation term name:
- obs term named `navigation_command`/`locomotion_command` → 3D velocity-only, default height **0.72**.
- obs term named `velocity_and_height_command`/`generated_commands` → 4D velocity+height, default height **0.74** (command_provider.py:171-185).
- `command_names` are `["vx","vy","wz"]` (3D) or `["vx","vy","wz","height"]` (4D) (command_provider.py:78-83).

There is **no gait field**. Gait/contact is not a command dimension anywhere in this stack.

### Units, ranges, step sizes (commands.py)
Docstring claims (commands.py:27-31) and limit attributes (commands.py:73-77) — NOTE A DISCREPANCY:
- `linear_x` (vx, forward, m/s): range **(-0.5, 0.5)**, step `vel_step = 0.1` m/s.
- `linear_y` (vy, lateral, m/s): range **(-0.5, 0.5)**, step `vel_step = 0.1` m/s.
- `angular_z` (wz, yaw, rad/s): range **(-1.0, 1.0)**, step `ang_step = 0.2` rad/s.
- `height` (m, ABSOLUTE base height): step `height_step = 0.05` m. **The runtime `height_range` attribute is `(0.4, 0.72)`** (commands.py:77) — but the class docstring says `[0.3, 0.8]` and the viewer help text prints `[0.3, 0.8]`. The code-enforced clamp is the attribute, i.e. **height is clamped to [0.4, 0.72] m at runtime**, not [0.3, 0.8]. This must be reconciled for deploy; see open_questions.

Height is commanded as an **absolute target height in meters** (not a delta from a nominal). `update_height(delta)` increments the absolute value then clamps (commands.py:111-114). `set_command(...)` sets absolute values; `height=None` leaves height unchanged (commands.py:116-138). `stop()` resets all four to defaults (commands.py:149-156).

### Keyboard key map (GLFW, in `agile/sim2mujoco/simulation.py` `_key_callback`, lines 502-575)
Each key press calls the corresponding `update_*` (single increment of the step) then `print_status()`. Keyboard input requires clicking the MuJoCo viewer window first.

Velocity/height (only active when `command_manager is not None`):
- **UP arrow OR `I`** → `update_linear_x(+vel_step)` → vx += 0.1 (simulation.py:537-539)
- **DOWN arrow OR `K`** → `update_linear_x(-vel_step)` → vx -= 0.1 (540-542)
- **LEFT arrow OR `J`** → `update_linear_y(+vel_step)` → vy += 0.1 (LEFT strafe) (545-547)
- **RIGHT arrow OR `L`** → `update_linear_y(-vel_step)` → vy -= 0.1 (RIGHT strafe) (548-550). Note: +linear_y = robot's left (+y body axis); right strafe is negative.
- **`U`** → `update_angular_z(+ang_step)` → wz += 0.2 (turn LEFT) (553-555)
- **`O`** → `update_angular_z(-ang_step)` → wz -= 0.2 (turn RIGHT) (556-558)
- **PAGE_UP OR `9`** → `update_height(+height_step)` → height += 0.05 (561-563)
- **PAGE_DOWN OR `0`** → `update_height(-height_step)` → height -= 0.05 (564-566)
- **`H`** → `command_manager.stop()` (reset all to defaults; "Home") (569-570)
- **`P`** → `print_status()` only (573-574)

Always-active (NOT command fields — sim-only, MUST be dropped/disabled on the real robot):
- **SPACE** → toggle pause; **N** → single-step while paused (512-518)
- **F/B/G/V** → apply a transient 100 N push disturbance to the body (forward/back/left/right), 0.15 s expiry (521-529). These are MuJoCo perturbations — do not map them on hardware.

So the canonical I/J/K/L scheme is: I=forward, K=back, J=left-strafe, L=right-strafe; U/O = yaw; 9/0 (or PgUp/PgDn) = height; H = stop. Arrow keys mirror I/J/K/L.

### Command update rate
Keyboard events are event-driven (per keypress via GLFW callback), each press = one step increment. The policy reads commands every control step. Control rate (`simulation.py:133-136`): `dt = physics_dt * decimation`, default `decimation = 4`. Eval loop (`sim2mujoco_eval.py:296-305`) prints "Control frequency" = `1/control_dt`. With the typical G1 MuJoCo scene (`physics_dt = 0.005` → 200 Hz physics, decimation 4) this is **50 Hz control**, but the exact value comes from `scene_config["physics_dt"]` and `decimation` in the loaded YAML — verify against the AGILE 23-DoF scene config. The scheduler variants (`Sim2MuJoCoCommandScheduler`, `RandomCommandScheduler`) call `.update(control_dt)` once per control step (sim2mujoco_eval.py:343), but for keyboard teleop there is no fixed cadence — the command tensor simply holds its last value between keypresses.

### Replication plan for `agile23_keyboard_control.py` (writes /tmp/robojudo_ext_cmd.json, single IPC writer)
- Maintain four floats (vx, vy, wz, height_abs) with the same step sizes (0.1, 0.1, 0.2, 0.05) and the same clamps. Use the **runtime** ranges from the attributes: vx,vy ∈ [-0.5,0.5]; wz ∈ [-1.0,1.0]; height ∈ [0.4,0.72] (NOT the help-text [0.3,0.8]).
- Initialize height to the deploy default (0.72 for 3D-nav configs, 0.74 for 4D vel+height). The AGILE 23-DoF student is velocity+height (4D) per the decision memo, so default 0.74 is the likely start, but confirm which obs term the deployed policy uses.
- Map keys: I/J/K/L (+arrows) for vx/vy, U/O for wz, 9/0 (+PgUp/PgDn) for height, H for STOP-to-defaults. Drop SPACE/N/F/B/G/V (sim-only).
- Emit absolute height in meters to the JSON `height` field; emit velocities to `velocity.{forward,lateral,yaw}` = (vx, vy, wz). Set `units` consistently (m/s, rad/s, m). Provide FSM (`RL_FULL`/`RL_LOWER`/`DAMP`) and `estop` per the box_demo_2 wire contract — those are NOT part of AGILE's keyboard scheme and must be added (e.g. dedicated keys), since AGILE's H is "reset commands", not an FSM/estop.

### Sign conventions to preserve (load-bearing)
- +vx = forward (+x body), +vy = robot's LEFT (+y body), +wz = turn left (CCW). Right strafe and turn-right are negative. These match the AGILE training frame; mismatching them on the real robot inverts steering.

## key_facts
- Command vector = 4 scalars: internal [linear_x, linear_y, angular_z, height] = user [vx, vy, wz, height]; no gait dimension exists.
- get_command() returns shape (4,) [vx,vy,wz,height]; get_navigation_command() returns shape (3,) [vx,vy,wz] (commands.py:166-188).
- Step sizes: vel_step=0.1 m/s, ang_step=0.2 rad/s, height_step=0.05 m (commands.py:69-72).
- RUNTIME clamps (the attributes actually used): vx,vy in [-0.5,0.5]; wz in [-1.0,1.0]; height in [0.4,0.72] (commands.py:73-77).
- DISCREPANCY: class docstring and viewer help text say height range [0.3,0.8], but code clamps to [0.4,0.72]. Code wins at runtime.
- Height is ABSOLUTE base height in meters (update_height increments then clamps absolute value).
- Default height = 0.72 for 3D nav configs, 0.74 for 4D velocity+height configs (command_provider.py:171,182).
- Keyboard (simulation.py:537-574): I/UP=+vx, K/DOWN=-vx, J/LEFT=+vy(left strafe), L/RIGHT=-vy(right strafe), U=+wz(turn left), O=-wz(turn right), 9/PgUp=+height, 0/PgDn=-height, H=stop-to-defaults, P=print.
- Each keypress = one step increment via GLFW callback then print_status(); event-driven, not a fixed cadence. Command holds last value between presses.
- Sim-only keys to DROP on hardware: SPACE(pause), N(step), F/B/G/V(100 N push disturbance) (simulation.py:512-529).
- Sign frame: +vx=forward(+x body), +vy=robot LEFT(+y body), +wz=CCW/turn-left. Right strafe/turn-right are negative.
- Control rate = physics_dt * decimation; decimation default 4 (simulation.py:133-136); exact Hz comes from the scene YAML (likely 50 Hz with physics_dt=0.005).
- stop() (key H) resets all four commands to defaults — this is NOT an FSM/estop; box_demo_2 FSM(RL_FULL/RL_LOWER/DAMP) and estop must be added separately.
- Scheduler classes call .update(control_dt) per control step; FIELD_SPECS maps vx/vy/wz/height -> (linear_x/linear_y/angular_z/height, *_range) (command_scheduler.py:30-35).

## open_questions
- Height clamp conflict: code attribute height_range=(0.4,0.72) vs docstring/help-text [0.3,0.8]. Which range did the deployed 23-DoF student train with? For the deep-squat fine-tune (target 0.20 m per memory) the [0.4,0.72] clamp would block reaching 0.20 m — the keyboard tool's height clamp MUST be set to match the actually-trained range, not blindly copied from commands.py. Verify against the exact policy YAML/training config used for the 23-DoF artifact.
- Which obs command term does the deployed 23-DoF student use (navigation_command 3D vs velocity_and_height_command 4D)? This sets command_dim, whether height is a command at all, and the default height (0.72 vs 0.74). Must read the deployed policy's config.
- Exact control frequency: depends on scene_config physics_dt and decimation in the AGILE 23-DoF MuJoCo scene YAML. Confirm the actual Hz (assumed ~50 Hz) so the IPC writer / runtime loop cadence matches; keyboard itself is event-driven so this only matters for how often the runtime samples the JSON.
- AGILE's keyboard scheme has NO FSM or estop concept (H = reset commands only). The box_demo_2 wire contract requires fsm (RL_FULL/RL_LOWER/DAMP) and estop fields; these need newly-assigned keys in agile23_keyboard_control.py and are not derivable from AGILE source — confirm desired key bindings with the operator.
- Sign/frame convention (+vy = robot left, +wz = CCW) is taken from AGILE sim; must be verified on the physical G1 that the rt/lowcmd_rl path preserves the same body-frame sign, otherwise strafe/turn invert on hardware.
---

# boxline

## The box_demo_2 / RoboJuDo wire line (GR00T adapter as template)

This is the EXACT skeleton the AGILE-23dof adapter must mirror. Three files: the adapter (policy→`rt/lowcmd_rl`), the mover (manipulation→IPC file), the launcher (tmux panes + merger).

### 1. `/tmp/robojudo_ext_cmd.json` schema (read by `read_external_command`, adapter L81-144)
Constant `CMD_FILE = "/tmp/robojudo_ext_cmd.json"` (adapter L29, mover L37). Fields:
- `fsm` (str): one of `RL_FULL` / `RL_LOWER` / `DAMP` / `LIMP`. Default `"RL_FULL"` when absent or null (`str(raw.get("fsm") or "RL_FULL")`, L103).
- `velocity` (obj): `{forward, lateral, yaw}` → `vx, vy, wz` (L115-118). forward=vx (m/s), lateral=vy (m/s), yaw=wz (rad/s).
- `units` (str): if `!= "agile"` the adapter treats the three velocities as NORMALIZED and multiplies by caps `vx*=fwd_max; vy*=lat_max; wz*=yaw_max` (L122-126). If `== "agile"` they are PHYSICAL m/s,rad/s and are NOT rescaled, only clamped (L127-129). GrootMover ALWAYS writes `"units": "agile"` (mover L144).
- `height` (float): ABSOLUTE pelvis height in meters, clamped to [min_height,max_height]=[0.40,0.80] (adapter defaults L32-33; mover clamps to [0.40,0.72], L41-42,L148). Falls back to `last_height` if absent.
- `timestamp` (float, epoch secs): staleness = `now - timestamp`; `fresh = age <= stale_s` with `stale_s` default 0.40 (L101-102, L317).
- `estop` (bool) OR `fsm=="DAMP"` → returns `DAMP`, zero velocity, `estop=True` (L109-110).
- `limp` (bool) OR `fsm=="LIMP"` → returns `LIMP`, zero velocity, `estop=True` (L106-107).
- `duration` (float, optional): written by mover for open-loop timed moves (mover L149-150); the ADAPTER does not read it.
- `source` (str, informational, mover writes `"groot_mover"`).

**Stale > 0.4s behavior**: when `not fresh` (or `fsm=="RL_LOWER"`), velocities are forced to zero but `fsm` and `height` are preserved (L112-113). So a stale command = stand-in-place at last height, NOT a fall. estop/limp/damp branches are checked BEFORE the freshness gate, so safety states always win. Missing file → `RL_FULL` zero-velocity (L95-96). Parse error → same, with WARN print (L97-99).

**Dead-band guard** (L131-135): a velocity norm in `(0, 0.05)` (`WALK_THRESHOLD_NOMOVE=0.05`, L39) is snapped to exact zero, because GR00T auto-switches Balance↔Walk at `||[vx,vy,wz]||<0.05` (g1_gear_wbc_policy.py:223 per comment). This is GR00T-specific; AGILE may not need it.

### 2. DDS topics
- **Subscribe**: `rt/lowstate` — NOT subscribed in the adapter directly. The adapter reads low_state via the GR00T env: `env.body().body_state_processor.robot_low_state` (`_latest_low_state`, L299-303). The AGILE adapter, lacking GR00T's env, must subscribe `rt/lowstate` itself.
- **Publish**: `rt/lowcmd_rl` ONLY (`--publish-topic` default `"rt/lowcmd_rl"`, L315; `LowCmdRlPublisher` topic L170). It uses HG IDL `unitree_hg_msg_dds__LowCmd_` / `LowCmd_` (L172-173).
- **Sole `rt/lowcmd` publisher**: `merge_lowcmd_arm_sdk.py` (launcher pane1, L139), which merges `rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd` (launcher L5, L137; adapter docstring L8-9). The RL process NEVER publishes `rt/lowcmd`. The merger file itself is NOT in this slice — confirm its merge rule (which slots arm_sdk owns) on the merger source.

### 3. 50Hz loop (adapter `main`, L436-515)
`dt = 1/hz`, hz default 50.0 (L316,L415). Per tick: optional sim step → `read_external_command` → `height_cmd = approach(height_cmd, cmd.height, height_rate*dt)` slew (L452, rate default 0.20 m/s L324) → `env.observe()` (skips tick on exception, L454-462) → fetch low_state → branch on fsm → `time.sleep(max(0, dt-elapsed))` (L514-515). Note the loop runs the policy every tick (no LSTM here, but the AGILE LSTM hidden/cell must be zeroed ONCE before this loop, never inside).

### 4. Holding arm/waist motor slots (`publish_body_targets`, L214-251)
Motor-count constants: `TOTAL_HG_MOTORS=35`, `BODY_MOTORS=29`, `ARM_SDK_ENABLE_SLOT=29` (L41-43; duplicated L45-47). Loops all 35 HG slots:
- slots `0..28` (i<BODY_MOTORS): driven by policy via `motor2joint`/`joint2motor` remap. `joint_index = motor2joint[i]`; if `==-1` use `default_motor_angles[motor_index]`, else `body_q[joint_index]`. Sets `out.q`, `dq=0, tau=0, kp=robot_kp[motor_index], kd=robot_kd[motor_index]` (L224-236).
- slot 29 (`ARM_SDK_ENABLE_SLOT`): forced `q=0.0` — this is the arm_sdk ENABLE flag; keeping it 0 means "RL stream does NOT claim arm ownership" so the merger lets arm_sdk overlay win (L220-221 comment, L244).
- slots `30..34` and slot 29: `kp=0, kd=0, dq=0, tau=0`. The non-enable extra slots (`i!=29`) are HELD at MEASURED q: `q = low_state.motor_state[i].q` (L238-244). With kp=kd=0 these are passive/zero-torque holds (the arm_sdk overlay supplies actual arm stiffness via the merger).
- Mode per motor: `_mode_for_motor` = `0x01` if weak motor else `0x0A` for body; non-body slots forced `0x01` (L195-196, L207, L223).

### 5. FSM states → legs/arms
- `RL_FULL`: full policy drives legs (and arms if the policy owns them); normal walking + height. Velocities pass through (when fresh).
- `RL_LOWER`: legs keep balancing via policy but velocities forced to ZERO (handled identically to stale: L112). Mover uses this as "handoff_to_arms" — legs balance, arms freed to arm_sdk (mover L292-298). Standing-only.
- `DAMP`: `publish_damping(low_state, kd_value)` (L253-272) — every slot held at measured q, `kp=0`, `kd=damping_kd` (default 8.0, L334), tau=0. Pure joint damping (soft stop). slot29 q=0.
- `LIMP` (extra, beyond the three named): `publish_limp` (L274-286) sets `q=PosStopF`, `dq=VelStopF`, all gains 0 → release all stiffness. slot29 q=0.

### 6. CRC + LowCmd construction
`from unitree_sdk2py.utils.crc import CRC` (L174). Init (`_init_lowcmd`, L198-213): `level_flag=0xFF`, `gpio=0`, `mode_pr`, `mode_machine` from `wbc_config["UNITREE_LEGGED_CONST"]` (L191-192). Every publish sets `low_cmd.mode_pr`/`mode_machine`, fills 35 motor_cmd slots, then `low_cmd.crc = self.crc.Crc(self.low_cmd)` and `self.pub.Write(self.low_cmd)` (L250-251, L271-272, L285-286). `PosStopF`/`VelStopF` from `UNITREE_LEGGED_CONST` (L189-190).

### 7. Per-motor kp/kd/q/dq/tau source
kp/kd come from `wbc_config["MOTOR_KP"]` / `["MOTOR_KD"]` indexed by motor_index (L183-184, L235-236). Remap tables `JOINT2MOTOR`/`MOTOR2JOINT` and `DEFAULT_MOTOR_ANGLES`, `WeakMotorJointIndex` all from wbc_config (L185-188). The AGILE adapter must supply its OWN 23-DoF (12+1+10) kp/kd/default/remap tables — these GR00T tables are 29-DoF and not reusable.

### Launcher (`start_groot_wbc_box.sh`)
4 tmux panes: (1) `merge_lowcmd_arm_sdk.py --iface $IFACE` (sole rt/lowcmd publisher), (2) the adapter, (3) optional `agile_keyboard_control.py` IPC teleop, (4) `box_demo_main.py ... --locomotion groot`. `--dry-run` skips the merger (so nothing reaches rt/lowcmd). Default IFACE `enP8p1s0`. Caps fwd/lat/yaw=0.50/0.30/0.60, height-rate 0.20.

## key_facts
- CMD_FILE = /tmp/robojudo_ext_cmd.json (adapter L29, mover L37)
- JSON fields: fsm, velocity{forward,lateral,yaw}, units, height(absolute m), timestamp, estop, limp, duration(mover-only), source
- units=='agile' => velocities are PHYSICAL m/s,rad/s, NOT rescaled, only clamped (L122-129); else NORMALIZED, multiplied by fwd/lat/yaw_max
- GrootMover ALWAYS writes units='agile' with physical units (mover L144)
- stale_s default 0.40s (L317); fresh=age<=stale_s; stale OR RL_LOWER => velocities forced 0, fsm/height preserved (L112-113); NOT a fall
- Safety branches (limp/estop/damp) checked BEFORE freshness gate so they always win (L106-110)
- FSM: RL_FULL(walk+height), RL_LOWER(legs balance v=0, arms->arm_sdk), DAMP(kp=0,kd=8 hold@measured q), LIMP(PosStopF/VelStopF, gains 0)
- Subscribe rt/lowstate; Publish rt/lowcmd_rl ONLY (HG IDL LowCmd_). GR00T gets low_state via env.body().body_state_processor.robot_low_state (L299-303); AGILE adapter must subscribe rt/lowstate itself
- merge_lowcmd_arm_sdk.py is the SOLE rt/lowcmd publisher: rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd (launcher L139)
- Motor constants: TOTAL_HG_MOTORS=35, BODY_MOTORS=29, ARM_SDK_ENABLE_SLOT=29 (L41-43)
- slot 29 ALWAYS q=0.0 (arm_sdk enable flag) so merger gives arm_sdk ownership (L244)
- slots 30..34 held at MEASURED q = low_state.motor_state[i].q with kp=kd=0 (passive hold) (L238-244)
- Body slots 0..28: kp=MOTOR_KP[m], kd=MOTOR_KD[m], q via motor2joint/joint2motor remap (joint==-1 => DEFAULT_MOTOR_ANGLES), dq=0,tau=0 (L224-236)
- mode_for_motor: 0x01 if weak else 0x0A for body; non-body forced 0x01
- CRC: low_cmd.crc = CRC().Crc(low_cmd) then pub.Write every publish; level_flag=0xFF, gpio=0, mode_pr/mode_machine from UNITREE_LEGGED_CONST
- 50Hz loop (default hz=50): read cmd -> approach height slew(rate 0.20 m/s) -> env.observe() -> branch fsm -> sleep(dt-elapsed)
- GR00T-specific dead-band: norm in (0,0.05) snapped to 0 (WALK_THRESHOLD_NOMOVE) due to Balance<->Walk switch
- Caps default fwd=0.50 lat=0.30 yaw=0.60; height [0.40,0.80] adapter / [0.40,0.72] mover
- GR00T tables (MOTOR_KP/KD, JOINT2MOTOR, DEFAULT_MOTOR_ANGLES, WeakMotorJointIndex) are 29-DoF from wbc_config; AGILE-23dof must supply its own

## open_questions
- merge_lowcmd_arm_sdk.py source is NOT in this slice. Must confirm its exact merge rule on-robot: which motor slots arm_sdk owns vs which it copies from rt/lowcmd_rl, how it reads slot29 as the arm_sdk-enable flag, and whether it expects HG 35-slot or 29-slot LowCmd. The whole hold-at-measured-q strategy depends on the merger honoring slot29==0 to keep arm ownership.
- The GR00T adapter never directly subscribes rt/lowstate — it borrows GR00T's env state processor. The AGILE-23dof adapter has no such env and MUST add its own rt/lowstate ChannelSubscriber; verify the HG LowState_ IDL and that motor_state[i].q ordering matches the 35-slot motor index space used for holding slots 13..34.
- Whether the 23-DoF 'basic' G1 still exposes 35 HG motor_cmd slots (with arms at 15..34) or fewer — slot29 (ARM_SDK_ENABLE_SLOT) and the hold-at-measured-q loop assume the 29-body / 35-total HG layout. Confirm the basic-G1 motor index map and that arm_sdk enable is still slot 29 on the physical robot.
- The dead-band snap (norm<0.05 -> 0) and WALK_THRESHOLD are GR00T-policy-specific (Balance<->Walk). AGILE's student likely has a different/no gait-switch threshold; do not copy this guard blindly — verify AGILE's velocity-to-step behavior at small commands.
- GR00T tables MOTOR_KP/MOTOR_KD/JOINT2MOTOR/MOTOR2JOINT/DEFAULT_MOTOR_ANGLES come from wbc_config YAML and are 29-DoF. The AGILE adapter must source equivalent 23-DoF (12 leg + 1 waist_yaw + 10 arm) kp/kd/default/order tables from AGILE's sim2mujoco config — these are NOT defined in this slice.
- height units: adapter clamps [0.40,0.80], mover clamps [0.40,0.72], STAND_HEIGHT differs (adapter DEFAULT_BASE_HEIGHT=0.74 vs mover 0.72). Confirm the AGILE student's height-command convention (absolute pelvis height in m, and its valid range / deep-squat 0.20m target) before wiring.
---

# jointmap

## The contract, end to end

There are THREE distinct joint orderings in play; conflating them is the silent-failure trap.

### (1) The MuJoCo XML actuator order (`g1_23dof_rev_1_0.xml`)
`/sdb/lizhe/g1_deepsquat/g1_23dof_desc/g1_23dof_rev_1_0.xml` lines 194-218 list the 23 `<motor>` entries IN THIS ORDER (this is the MJCF/`mj_model` joint enumeration order that sim2mujoco's `simulation.py` discovers at runtime via `mj_id2name`):
1 left_hip_pitch, 2 left_hip_roll, 3 left_hip_yaw, 4 left_knee, 5 left_ankle_pitch, 6 left_ankle_roll, 7 right_hip_pitch, 8 right_hip_roll, 9 right_hip_yaw, 10 right_knee, 11 right_ankle_pitch, 12 right_ankle_roll, 13 waist_yaw, 14 left_shoulder_pitch, 15 left_shoulder_roll, 16 left_shoulder_yaw, 17 left_elbow, 18 left_wrist_roll, 19 right_shoulder_pitch, 20 right_shoulder_roll, 21 right_shoulder_yaw, 22 right_elbow, 23 right_wrist_roll.
This confirms 23dof = 12 legs + waist_yaw + 10 arms (shoulder p/r/y, elbow, wrist_roll per side). NO waist_roll/pitch, NO wrist_pitch/yaw. This is a block order (all-left-leg, all-right-leg, waist_yaw, left-arm, right-arm) — NOT the policy's order.

### (2) The POLICY joint order (what the LSTM actually sees / emits)
This is set by the deploy descriptor YAML's `articulations.robot.joint_names` and the per-obs-term `joint_names`. For the CURRENTLY EXPORTED student (`unitree_g1_velocity_height_recurrent_student.yaml`, a 29-DoF artifact) the order is IsaacLab's internal depth-first interleaved order:
`left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, right_hip_roll, waist_roll, left_hip_yaw, right_hip_yaw, waist_pitch, left_knee, right_knee, left_shoulder_pitch, right_shoulder_pitch, left_ankle_pitch, right_ankle_pitch, left_shoulder_roll, right_shoulder_roll, left_ankle_roll, right_ankle_roll, left_shoulder_yaw, right_shoulder_yaw, left_elbow, right_elbow, left_wrist_roll, right_wrist_roll, left_wrist_pitch, right_wrist_pitch, left_wrist_yaw, right_wrist_yaw` (29 entries).
- `joint_pos_rel` / `joint_vel_rel` obs are each shape 29, in this order.
- `last_action` obs is shape 12 (lower body only).
- The ACTION head is shape 12, `actions.joint_names` (descriptor order): `left_hip_pitch, right_hip_pitch, left_hip_roll, right_hip_roll, left_hip_yaw, right_hip_yaw, left_knee, right_knee, left_ankle_pitch, right_ankle_pitch, left_ankle_roll, right_ankle_roll`. Per-joint `scale`/`offset`/`clip(±6)` are baked here. Action drives ONLY the 12 leg joints — waist_yaw is NOT actioned even though it is in the obs.

CRITICAL: a 23-DoF student descriptor does NOT yet exist (only the 29-DoF `*_recurrent_student.yaml/.pt/.onnx` are in `agile/data/policy/velocity_height_g1/`). A 23dof training run exists (`logs/.../2026-06-16_10-56-53_velocity_height_g1_lower/`, env.yaml usd_path=`/hdd0/.../g1_23dof.usd`) but its action uses regex `.*_hip_.*_joint,.*_knee_joint,.*_ankle_.*_joint` with `preserve_order: false` — so the resolved policy order is determined by the 23dof USD's traversal, and the obs `joint_pos_rel` will be 23 (not 29). The exact 23-entry interleaved order MUST be read from the exported 23dof descriptor once it is produced; do not assume it equals the 29-order minus 6.

### (3) The Unitree SDK 29-slot LowCmd motor array (the wire)
Canonical hg motor index map, authoritative copy in `box_demo_2/agile_lowcmd_pipeline.py:45-75` (`MOTOR_BY_JOINT`):
0-5 left leg (hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll); 6-11 right leg (same order); 12 waist_yaw; 13 waist_roll; 14 waist_pitch; 15-21 left arm (shoulder_pitch,roll,yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw); 22-28 right arm (same order). `NUM_MOTORS=35`; slot 29 is the arm_sdk enable slot.
The 23-DoF basic G1 OMITS exactly 6 SDK slots: 13 (waist_roll), 14 (waist_pitch), 20 (left_wrist_pitch), 21 (left_wrist_yaw), 27 (right_wrist_pitch), 28 (right_wrist_yaw). **On the real 23-DoF robot, whether `motor_state[]` is still indexed 0-28 with those 6 slots present-but-dead, or compacted to 23, MUST be verified against the physical robot's `rt/lowstate` — this is the single highest-risk unknown.** The existing pipeline assumes the full 29-slot layout (it reads `msg.motor_state[mi]` for mi up to 28 and holds 12-28).

## The permutation that publishes rt/lowcmd_rl
The mapping is done BY JOINT NAME, never by raw index, in `agile_lowcmd_pipeline.py`:
- Obs build: `motor_to_policy_arrays` (lines 235-247) loops over `joint_names` (the descriptor order) and reads `msg.motor_state[MOTOR_BY_JOINT[name]].q/.dq/.tau_est`. So lowstate(SDK-slot) -> policy order is `policy[i] = motor_state[MOTOR_BY_JOINT[joint_names[i]]]`.
- Action publish: `build_lowcmd` (lines 419-451). Init: every slot 0..34 stamped with measured `q`, kp=kd=0 (passive). Then for each policy joint, `mi=MOTOR_BY_JOINT[name]`; if `mi<=11` (legs) stamp policy target+policy kp/kd; if `12<=mi<=28` HOLD at measured `q` with `upper_hold_kp/kd` (default 40/1). Slot 29 (arm_sdk enable) stamped 0/0/0. So the AGILE pipeline DRIVES only motor slots 0-11 (legs). It HOLDS 12-28 — including waist_yaw (slot 12) — and the separate arm_sdk overlay merged by `merge_lowcmd_arm_sdk.py` overrides 12-28 during grasp.

NOTE the asymmetry vs the policy: the descriptor has waist_yaw in the OBS (slot 12 read into obs) but the build_lowcmd HOLDS slot 12 rather than driving it (the action head is legs-only anyway, so there is no waist_yaw target to publish — correct). PD gains: kp/kd come from descriptor `default_joint_stiffness`/`default_joint_damping` reordered to descriptor order by `ActionProcessor._parse_joint_values`; for legs these are e.g. hip_pitch kp=40.18, hip_roll kp=99.10, knee kp=99.10, ankle kp=28.50 (see recurrent_student.yaml). Default leg pose offsets: hip_pitch −0.10, knee +0.30, ankle_pitch −0.20.

## LSTM zeroing (drift trap)
`zero_policy_recurrent_state(policy)` (lines 266-275) zeroes any buffer whose name contains "hidden" or "cell", called ONCE before the loop (line 591). Must never be re-called in-loop.

## key_facts
- 23dof = 12 legs + waist_yaw + 10 arms (shoulder p/r/y, elbow, wrist_roll per side). XML actuator block order: L-leg(6), R-leg(6), waist_yaw, L-arm(5), R-arm(5). Source: g1_23dof_rev_1_0.xml:194-218.
- Three orders differ: (1) MJCF block order, (2) policy interleaved order from descriptor YAML, (3) SDK 29-slot wire order. All mapping in the live pipeline is by JOINT NAME, never raw index.
- SDK 29-slot map (authoritative: agile_lowcmd_pipeline.py:45-75): 0-5 L-leg, 6-11 R-leg, 12 waist_yaw, 13 waist_roll, 14 waist_pitch, 15-21 L-arm(sh p/r/y,elbow,wrist roll/pitch/yaw), 22-28 R-arm. Slot 29 = arm_sdk enable. NUM_MOTORS=35.
- 23-DoF basic G1 omits exactly these 6 SDK slots: 13 waist_roll, 14 waist_pitch, 20 L-wrist_pitch, 21 L-wrist_yaw, 27 R-wrist_pitch, 28 R-wrist_yaw.
- AGILE lower policy DRIVES SDK slots 0-11 (legs) only. It HOLDS 12-28 at measured q with upper_hold_kp/kd (default 40/1). waist_yaw (slot 12) is read into obs but HELD, not driven (action head is legs-only). build_lowcmd: agile_lowcmd_pipeline.py:419-451.
- Obs->policy permutation: policy[i] = lowstate.motor_state[MOTOR_BY_JOINT[joint_names[i]]] (motor_to_policy_arrays:235-247). joint_names = descriptor articulations.robot.joint_names.
- Currently exported deploy artifact is 29-DoF (unitree_g1_velocity_height_recurrent_student.yaml/.pt/.onnx): obs joint_pos_rel/joint_vel_rel shape=29 interleaved order; action shape=12 (legs); last_action shape=12.
- The 29-DoF policy interleaved obs order has waist_roll(idx5), waist_pitch(idx8), and wrist_pitch/yaw at obs indices 25-28 — joints ABSENT on 23dof hardware. A 23dof student must re-export a 23-length obs vector.
- No 23-DoF deploy descriptor exists yet. A 23dof training run exists (logs/.../2026-06-16_10-56-53_velocity_height_g1_lower, usd=g1_23dof.usd) but action uses regex with preserve_order:false, so the resolved 23-joint order comes from the 23dof USD traversal and must be read from the exported descriptor, not assumed = 29-order-minus-6.
- Leg PD gains/defaults (recurrent_student.yaml, descriptor order): hip_pitch kp40.18/kd2.56, hip_roll kp99.10/kd6.31, hip_yaw kp40.18, knee kp99.10/kd6.31, ankle kp28.50/kd1.81. Default offsets: hip_pitch -0.10, knee +0.30, ankle_pitch -0.20.
- LSTM hidden/cell buffers zeroed once pre-loop via zero_policy_recurrent_state (named_buffers containing 'hidden'/'cell'), agile_lowcmd_pipeline.py:266-275,591. Never re-zero in loop.
- sim2mujoco maps YAML joint order <-> MJCF order purely by name (ActionProcessor._parse_joint_values; ObservationTerm joint_indices=joint_names.index(jn)); raises ValueError on any name/order mismatch.

## open_questions
- MUST VERIFY ON ROBOT: does the physical 23-DoF basic G1 rt/lowstate expose motor_state[] in the full 29-slot layout (with slots 13,14,20,21,27,28 present-but-inert) or compacted to 23 entries? The entire MOTOR_BY_JOINT index scheme and build_lowcmd hold-loop assume the full 29-slot hg layout. If the firmware compacts, every index >12 is wrong.
- The exact policy joint order for a TRUE 23-DoF student is unknown because no 23dof deploy descriptor has been exported. It is resolved by the 23dof USD traversal (preserve_order:false). The 23-entry interleaved obs order MUST be read from the exported descriptor's articulations.robot.joint_names, NOT assumed to equal the 29-order with the 6 absent joints deleted.
- The currently-wired pipeline (agile_lowcmd_pipeline.py) deploys the 29-DoF student, whose obs vector (29) and last_action (12) include joints absent on 23dof hardware. Running the 29dof artifact on 23dof hardware would feed garbage/held values for waist_roll/pitch and wrist_pitch/yaw obs slots. Confirm whether the plan is to (a) re-export a 23dof student, or (b) keep the 29dof policy and synthesize the 6 missing obs entries (e.g. zeros) — these are functionally different and must be decided.
- Verify IMU convention on the real robot: pipeline assumes imu_state.quaternion is wxyz (fed to AGILE quat_rotate_inverse) and imu_state.gyroscope is already body-frame (used directly as root_ang_vel). Confirm against actual G1 firmware (lowstate_to_sim_state:250-263).
- waist_yaw (SDK slot 12) is read into the policy obs but HELD (not driven) by build_lowcmd, and is the boundary between legs(driven) and upper(held/arm_sdk). Confirm the arm_sdk overlay (merge_lowcmd_arm_sdk.py) intends to own slot 12 too, or whether a future RL_LOWER mode should drive waist_yaw from a 23dof policy that includes it in its action head.
- default_joint_pos_limits in the 29dof descriptor lists wrist limits as ±1.6144 (not the XML's ±1.972 for wrist_roll) — confirm which limit set the 23dof student's leg-only safety clamp (_pose_limits at agile_lowcmd_pipeline.py:390-398) should use; only the 12 leg entries matter for publishing but the indexing reads from the descriptor order.