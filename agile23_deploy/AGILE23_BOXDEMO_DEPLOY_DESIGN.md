# AGILE 23-DoF × box_demo_2/RoboJuDo Sim2Real 部署设计

> 部署目标：把 AGILE velocity-height **recurrent LSTM student**（蹲起+行走，仅腿部 action）跑在真机 23-DoF basic G1 上，复用 box_demo_2/RoboJuDo 的整套连线（`rt/lowcmd_rl` + `merge_lowcmd_arm_sdk.py`），键盘 teleop。

---

## 1. 架构与数据流

```
agile23_keyboard_control.py ──写──> /tmp/robojudo_ext_cmd.json (单写者)
                                          │ read (50Hz, stale_s=0.40)
agile23_lowcmd_pipeline.py ──sub── rt/lowstate (HG LowState_)
   obs组装 → LSTM policy → 12腿action → build_lowcmd(35槽)
                                          │ pub
                                     rt/lowcmd_rl ──┐
box_demo_main (抱箱) ─────────────> rt/arm_sdk ─────┤
                                          merge_lowcmd_arm_sdk.py (唯一 rt/lowcmd 发布者)
                                          │ pub
                                     rt/lowcmd → 机器人
```

- **唯一的 `rt/lowcmd` 发布者**是 merger（pane1）；RL 进程**只**发 `rt/lowcmd_rl`，绝不直接发 `rt/lowcmd`。
- **GR00T adapter 借 env 拿 low_state；AGILE 没有 env，必须自己 `ChannelSubscriber("rt/lowstate")`**（jointmap `lowstate_to_sim_state`，参考 box_demo_groot adapter L299-303 的缺口）。
- **50Hz 主循环**（physics_dt=0.005, decimation=4 → control dt=0.02）：`read_cmd → height 斜率逼近(0.20 m/s) → 组 obs → policy(obs) → fsm 分支 → build_lowcmd → CRC → Write → sleep(dt-elapsed)`。LSTM 隐状态在循环**前**清零一次，**循环内绝不再清**。
- 启动器 `start_agile23_box.sh`：tmux 4 pane = merger / pipeline / keyboard / box_demo_main；默认 IFACE `enP8p1s0`；`--dry-run` 跳过 merger。

---

## 2. obs 向量契约（23-DoF 完整有序布局）

运行期读的是**re-export 的 23-DoF IODescriptor YAML**（不是 Python 类，不是已 check-in 的 29-DoF YAML）。obs 顺序 = YAML `observations.policy` 列表序：

| # | term | dim | scale | clip | 单位 | 计算 |
|---|------|-----|-------|------|------|------|
| 1 | generated_commands | 4 | — | — | m/s,m/s,rad/s,**m(绝对)** | `[vx,vy,wz,height]` 整体喂入，不截断 |
| 2 | base_ang_vel | 3 | — | — | rad/s, body frame | IMU 陀螺原始 `[wx,wy,wz]` |
| 3 | projected_gravity | 3 | — | — | **单位向量(模1)** | `quat_rotate_inverse(quat_wxyz,[0,0,-1])` |
| 4 | joint_pos_rel | **23** | — | — | rad | `q[idx]-default_joint_pos` |
| 5 | joint_vel_rel | **23** | **0.1** | — | rad/s(预scale) | `(dq-0)*0.1` |
| 6 | last_action | 12 | — | (-10,10) | — | 上一拍12维腿action，首拍0 |

**总维 = 4+3+3+23+23+12 = 68**（29-DoF 是 80）。

- **学生没有 `base_lin_vel`**（特权项，teacher/critic 专用）。LSTM 自行估计线速度。**永不喂 lin_vel**。
- `joint_pos_rel` 的 default offset 非零项只有腿：hip_pitch=-0.10、knee=+0.30、ankle_pitch=-0.20（左右各一），其余全 0。
- 部署期 `enable_corruption=False / noise=0`，逐项 pipeline 退化为：`raw → clip(仅 last_action) → scale(仅 joint_vel_rel ×0.1)`。
- joint_pos_rel/joint_vel_rel 的 23-joint 顺序 = **re-export YAML 的 `articulations.robot.joint_names`（USD 遍历序，interleaved）**，**不可**用 29 序删 6 个手推。映射全程**按关节名**，processor 做 name round-trip 校验（不匹配抛 ValueError）。

---

## 3. action→关节目标 + LSTM 状态处理（verbatim AGILE）

**action = 12 维腿（与 23/29 无关）**，唯一 term `joint_position_action`。

逐关节 pipeline（actions.py:64）：
1. `a = clip(raw, -6, 6)`（先 clip）
2. `joint_target = a * scale + offset`
   - `scale(12) = [0.5475,0.5475,0.3507,0.3507,0.5475,0.5475,0.3507,0.3507,0.4386,0.4386,0.4386,0.4386]`
   - `offset(12) = [-0.1,-0.1,0,0,0,0,0.3,0.3,-0.2,-0.2,0,0]`（已含 default 腿姿，**不再另加 default_joint_pos**）
3. **action 路径无 joint_pos_limits clamp**——真机部署必须**自加**腿位安全 clamp（用 descriptor `default_joint_pos_limits` 的 12 个腿项）。

action 关节序（leg-grouped，**异于** obs 序）：L/R hip_pitch, L/R hip_roll, L/R hip_yaw, L/R knee, L/R ankle_pitch, L/R ankle_roll。

腿 PD：hip_pitch kp40.18/kd2.56，hip_roll kp99.10/kd6.31，hip_yaw kp40.18/kd2.56，knee kp99.10/kd6.31，ankle_pitch kp28.50/kd1.81，ankle_roll kp60/kd1.0。

**LSTM（`_CheckpointInferenceModel`）**：用 `_checkpoint.pt`（6.65MB，in-graph buffer 语义最干净；优于 TorchScript .pt 与无状态 .onnx）。`hidden_state`/`cell_state` 是 registered buffer，`__init__` 清零一次；forward 内 `self.hidden_state[:]=h.squeeze(1)` slice-assign 携带跨拍。调用签名：`actions = policy(obs)`，obs 1-D `(68,)` → 1-D `(12,)`，**caller 不传 h/c**。`zero_policy_recurrent_state(policy)`（清零名含 hidden/cell 的 buffer）在循环前调用一次。**漂移陷阱**：循环内绝不 reset / 重建。⚠️ **必须在装 torch 的 env 内 dump 真实 LSTM (num_layers, hidden_dim, rnn_input_dim)**——`RNNPolicyWrapper` 默认 `[2,1,128]` 仅 TorchScript 路径用，不可假设。

---

## 4. 23-DoF 关节 ↔ 29-slot SDK 电机映射

SDK 29-slot 序（`MOTOR_BY_JOINT`，权威，`agile_lowcmd_pipeline.py:45-75`），23-DoF basic G1 **omit 槽 13,14,20,21,27,28**。NUM_MOTORS=35，slot29=arm_sdk enable。

| SDK slot | 关节 | 23-DoF 在? | pipeline 行为 |
|---|---|---|---|
| 0-5 | 左腿 hip_p/r/y,knee,ank_p/r | ✓ | **驱动**(policy target+policy kp/kd) |
| 6-11 | 右腿 同序 | ✓ | **驱动** |
| 12 | waist_yaw | ✓ | **持有**@measured q, upper_hold_kp/kd(40/1)；读入obs但不action |
| 13,14 | waist_roll/pitch | ✗ | 持有(若硬件保留死槽) |
| 15-19 | 左 shoulder_p/r/y,elbow,wrist_roll | ✓ | **持有**(arm_sdk overlay 接管) |
| 20,21 | 左 wrist_pitch/yaw | ✗ | 持有 |
| 22-26 | 右 shoulder_p/r/y,elbow,wrist_roll | ✓ | **持有** |
| 27,28 | 右 wrist_pitch/yaw | ✗ | 持有 |
| 29 | arm_sdk enable | — | **强制 q=0.0**（保 arm_sdk 拥有权，merger 让 overlay 胜） |
| 30-34 | 额外 HG 槽 | — | 持有@measured q，kp=kd=0 |

- obs 组装：`policy[i] = motor_state[MOTOR_BY_JOINT[joint_names[i]]]`。
- build_lowcmd：legs(mi≤11) 发 policy 目标；12≤mi≤28 持有；slot29=0。body 槽 dq=0,tau=0；mode `0x0A`（弱电机 `0x01`）。

⚠️ **最高风险未知**：真机 23-DoF `rt/lowstate.motor_state[]` 是 **29 槽含死槽（13/14/20/21/27/28 present-but-inert）** 还是**压缩成 23 项**？整套 index 方案依赖 29 槽布局；若压缩，所有 >12 的索引全错。**必须先在真机验。**

---

## 5. 键盘操控方案（单写者 → IPC）

`agile23_keyboard_control.py` 维护 4 个 float，单写 `/tmp/robojudo_ext_cmd.json`：

| 按键 | 动作 | step | clamp(runtime) |
|---|---|---|---|
| I/↑ ; K/↓ | vx ±0.1 (前/后) | 0.1 | [-0.5,0.5] |
| J/← ; L/→ | vy ±0.1 (左/右 strafe) | 0.1 | [-0.5,0.5] |
| U ; O | wz ±0.2 (左转/右转) | 0.2 | [-1.0,1.0] |
| 9/PgUp ; 0/PgDn | height ±0.05 (绝对m) | 0.05 | **见下** |
| H | stop→默认(四项复位) | — | — |
| **新增** F1/F2/F3 | fsm=RL_FULL/RL_LOWER/DAMP | — | — |
| **新增** ESC/空格长按 | estop=true | — | — |

- **符号**：+vx 前进(+x body)，+vy 机器人**左**(+y)，+wz **CCW 左转**。右/右转为负。真机须验 `rt/lowcmd_rl` 保号。
- **丢弃 sim-only 键**：SPACE/N(暂停步进)、F/B/G/V(100N 推扰)。
- **height clamp 争议**：commands.py 属性 `(0.4,0.72)`，docstring/help `(0.3,0.8)`，env 训练域 `(0.20,0.72)`。⚠️ 蹲到 0.20m 的目标会被 0.4 下限**卡死**——**必须按 23-DoF student 实际训练域设 clamp**，不可照抄 0.4。
- JSON 写法：`units:"agile"`（物理单位，不重缩放，仅 clamp）；`velocity:{forward:vx,lateral:vy,yaw:wz}`；`height` 绝对米；`fsm`；`estop`；`timestamp`（epoch）；`source:"agile23_keyboard"`。
- 单写者：**只有 keyboard 写该文件**（抱箱场景用 mover 时二选一，不可双写）。

---

## 6. 文件清单

| 文件 | 职责 |
|---|---|
| `agile23_lowcmd_pipeline.py` | 主控。自订 `rt/lowstate` sub + `rt/lowcmd_rl` pub(HG LowCmd_)；`lowstate_to_sim_state`(quat wxyz, gyro body-frame)；23-DoF obs 组装(§2)；加载 `_checkpoint.pt` + LSTM 清零一次；50Hz 循环；`build_lowcmd`(35槽, §4) + CRC；fsm 分支；安全 guard(§7)。**自带 23-DoF kp/kd/default/remap 表**（GR00T 的 29-DoF 表不可复用）。 |
| `agile23_keyboard_control.py` | 单写者 teleop(§5)，无 DDS，仅写 IPC json。 |
| `start_agile23_box.sh` | tmux 4 pane(merger/pipeline/keyboard/box_demo_main)；IFACE、caps、`--dry-run`。 |
| `agile23_safety.py`(可并入 pipeline) | NaN/tilt/clamp/ratelimit/overspeed/staleness/obs-dim/self-damp 守卫(§7)。 |

> **re-export 前置（非代码）**：`scripts/export_IODescriptors.py --task Velocity-Height-G1-23dof-Distillation-Recurrent-v0` 产出 23-DoF YAML；并 **dump checkpoint input dim 确认 = 68**（committed 权重可能是 29-DoF/80）。

---

## 7. 安全 guard

- **NaN/Inf**：obs 或 action 含 NaN → 立刻进 DAMP（kp=0,kd=8 持 measured q），不发腿目标。
- **tilt**：`projected_gravity.z > -cos(35°)`（≈躯干倾斜>35°）→ DAMP。
- **clamp + ratelimit**：腿 target 先 clip 到 descriptor 腿位限，再做逐拍变化率限幅（防单拍跳变）。
- **overspeed**：发布前校验腿 |Δq|/dt 与 |dq|，超阈进 DAMP。
- **mode_pr / mode_machine**：每次发布从 `UNITREE_LEGGED_CONST` 写入；`level_flag=0xFF, gpio=0`；漏写会被固件拒。
- **staleness**：`age = now - timestamp > 0.40s` → 速度强制 0，fsm/height 保留（**站立不是摔倒**）。estop/limp/damp 分支**先于** freshness gate 判定，安全态恒胜。缺文件/解析失败 → `RL_FULL` 零速。
- **obs-dim 校验**：启动时断言组装 obs 长度 == YAML 期望(68)，processor name round-trip 通过，否则拒绝起跑。
- **self-damp / wireless-estop**：键盘 estop 键 + 无线手柄 estop → DAMP；`LIMP`(PosStopF/VelStopF, 全增益0) 作为软释放保留。
- **height 斜率**：`height_cmd` 以 0.20 m/s 逼近目标，禁止阶跃蹲。

---

## 8. 必须在真机核实的项（GO/NO-GO）

1. **[最高风险] rt/lowstate motor_state 布局**：29 槽含死槽 vs 压缩 23。决定全部 index。
2. **23-DoF policy 关节序**：从 re-export descriptor 的 `joint_names` 读，**禁止**假设 = 29 序删 6。
3. **deploy checkpoint input dim**：68(23-DoF) vs 80(29-DoF)。committed 权重可能是 29-DoF 训练——若是，须先训/导 23-DoF student，否则维度对不上直接 NO-GO。
4. **LSTM 形状**：在 torch env 内 dump `(num_layers, hidden_dim, rnn_input_dim)`，确认 rnn_input_dim 与 normalizer 后 obs 长度一致。
5. **IMU 约定**：quat 是 wxyz？gyro 已是 body-frame？符号/帧错会**静默**毁掉 projected_gravity 与 base_ang_vel。
6. **height 命令域**：deploy command provider 用 (0.4,0.72) 还是训练域 (0.20,0.72)；deep-squat 目标须确认未被卡。
7. **控制频率**：确认真机环 50Hz(dt=0.02) 与训练一致（env cfg 说 decimation=10/50Hz，YAML scene 说 decimation=4/50Hz——结论同为 50Hz，但须确认 eval loop 取哪个）。
8. **符号一致性**：+vy=左、+wz=CCW 在 `rt/lowcmd_rl` → merger → 电机后是否保持，否则 strafe/转向反向。
9. **merger 规则**：`merge_lowcmd_arm_sdk.py` 是否按 slot29==0 让 arm_sdk 拥有 12-28、是否期望 35 槽 HG LowCmd——hold-at-measured-q 策略完全依赖此。

**GO/NO-GO obs 字节对比**：起跑前，用同一组真机 lowstate，分别在 (a) sim2mujoco eval（喂 23-DoF YAML）与 (b) `agile23_lowcmd_pipeline` 组的 68 维 obs，**逐字节 diff**；要求 generated_commands/base_ang_vel/projected_gravity/joint_pos_rel/joint_vel_rel/last_action 六段全等（scale/offset/顺序/单位）。任一段不等 = NO-GO。
