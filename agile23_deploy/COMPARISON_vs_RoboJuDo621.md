# 对比：同学的 RoboJuDo621 (`ih/agile-velocity-sim2sim`) vs 我们的 `agile23_deploy`

> 源：`github.com/Leoliyanmin/RoboJuDo621` 分支 `ih/agile-velocity-sim2sim`（最后 commit「docs: 补充23dof部署键盘控制说明」）。
> 整理日期 2026-06-30。

## 一句话结论
**两者部署的是不同的 AGILE 策略，架构路线也不同。** 不是"同一件事的两种写法"——权重 / obs / action / PD 增益全不一样，**不可互换**。

- **同学**：AGILE **velocity-tracking、history-MLP** 策略（`Velocity-G1-History-23DOF-Wrist20-v0`），整合进 **RoboJuDo 框架**（纯 config + `DoFAdapter`），在 **RoboJuDo 自带 MuJoCo 里 sim2sim** 验证。
- **我们**：AGILE **velocity-HEIGHT、recurrent-LSTM** 深蹲微调 student，**独立脚本** verbatim 包 AGILE `sim2mujoco`，走 box_demo_2 / RoboJuDo **线协议**（只发 `rt/lowcmd_rl`），瞄准**真机 sim2real**。

## 对比表

| 维度 | 同学 RoboJuDo621 | 我们 agile23_deploy |
|---|---|---|
| **策略族** | velocity tracking（vx,vy,wz，**无高度**） | velocity + **height**（vx,vy,wz,height，能蹲） |
| **网络** | MLP + **5 帧 history**（obs 240） | **recurrent LSTM**，无 history（obs **68**） |
| **训练特色** | **wrist-load 课程**（0/5/10/20 N 腕负载） | **deep-squat 微调**（base_height→0.20，model_3750 蒸馏） |
| **控制关节** | **13** = 12 腿 + `waist_yaw` | **12** = 仅腿（`waist_yaw` 持有不驱） |
| **obs 组成** | 5×[ang_vel·0.2, grav, cmd, dofpos−def, dofvel·**0.05**, last_act]（48/帧） | [cmd(4), ang_vel(3), grav(3), jpos_rel(23), jvel_rel(23)·**0.1**, last_act(12)] |
| **action** | 13, `clip(±10)·0.5 + default` | 12, `clip(±6)·scale + offset` |
| **PD 增益** | hip 100 / knee 200 / ankle 20 / waist_yaw 300 | hip 40 / knee 99 / ankle 28（**不同任务→不同增益**） |
| **架构** | RoboJuDo 原生：config 类 + 复用 `UnitreeWoGaitPolicy` + `DoFAdapter` | 独立脚本：`import agile.sim2mujoco.*` verbatim wrap |
| **关节↔电机映射** | `DoFAdapter` 按名 slice/expand（env 23 ↔ policy 13） | `MOTOR_BY_JOINT` 按名映射 Unitree 29-slot，驱腿 0-11 持 12-28 |
| **运行环境** | RoboJuDo MuJoCo（`run_pipeline.py -c g1_agile_velocity_23dof`），**sim2sim** | DDS：sub `rt/lowstate` → pub `rt/lowcmd_rl` → `merge_lowcmd_arm_sdk.py`，**目标真机** |
| **键盘** | RoboJuDo handler：w/s=vx, a/d=vy, q/e=wz（edge-trigger ×1.5，作者注明 jumpy） | 自写 `agile23_keyboard_control.py`：I/J/K/L+方向键, U/O, 9/0；heartbeat + **单写 IPC** |
| **臂处理** | `DoFAdapter` expand 13→23，臂保持 env default | `rt/lowcmd_rl` 持 12-28 measured q，slot29=0，留 `rt/arm_sdk` overlay |
| **机器人/场景** | `g1_23dof_rev_1_0.xml`，200 Hz 物理 / 50 Hz 控制 | **同** `g1_23dof_rev_1_0.xml`，**同** 50 Hz |
| **验证** | headless（agile_env + xvfb）：vx0.4 走 ~1.6 m，站 0.719 m，抬脚 ~5 cm | AGILE sim2mujoco + EGL 出视频：前进 1.58 m，4 向不摔 |

## 互相印证的相同点
- **同一个 23-DoF MJCF**（`g1_23dof_rev_1_0.xml`）、同 50 Hz/200 Hz —— 验证了我们的 scene/控制率选择。
- **都"按关节名"映射**（他们 `DoFAdapter`，我们 `MOTOR_BY_JOINT`），都以导出的 **IODescriptor `joint_names`** 为 ground truth、不手推。
- 都确认 **23dof = 12 腿 + waist_yaw + 10 臂**（肩 3 + 肘 1 + 腕 roll）。

## 关键差异 & 对我们的启示
1. **不同策略，各有用途**：他们是「纯速度跟踪 + 负载鲁棒」，适合稳定行走/抗负重；我们是「速度+高度」，能边走边蹲（box_demo 抱箱蹲取的诉求）。
2. **控制 13 vs 12 是任务差异、不是 bug**：velocity-history task 把 `waist_yaw` 放进 action（13），velocity-height task 没有（12）。**我们已从导出 IODescriptor 核实 action = 12 腿**，故我们 pipeline「持有 slot 12（waist_yaw）」是正确的。
3. **PD 增益绝不可跨用**：两任务训练增益不同（他们 hip100/knee200/ankle20，我们 hip40/knee99/ankle28）。各自从自己的 IODescriptor 读 kp/kd —— 我们 pipeline 正是这么做的。
4. **架构哲学不同**：他们融入 RoboJuDo（config 化、可多策略统一管理、复用框架键盘/sim）；我们独立脚本（可审计、不进闭源 `robojudo` wheel、严格走 box_demo_2 线 + 保 arm_sdk overlay + 自带 10 项安全 guard）。**这正是我们 decision memo 选「独立脚本而非 RoboJuDo 宿主」的体现**——他们的是 sim2sim 框架内验证，我们的瞄准真机且要可审计。
5. **`DoFAdapter` 值得借鉴**：本质同样是「按名 slice/expand」，但更通用（任意 src/tar joint set，带 template expand）。我们的 `MOTOR_BY_JOINT` 直接面向 29-slot 线协议；可吸收其通用 `fit/expand` 思路。
6. **wrist-load 课程**：他们训练加了腕负载鲁棒性；我们没有。若我们的 23dof 要真机抱箱行走，可能需要类似负载/扰动课程。
7. **键盘**：他们复用 RoboJuDo handler（有 edge-trigger×1.5 的 jumpy，作者自己标注）；我们自写、带 heartbeat、单写 IPC，更可控、且天然契合 box_demo_2 的 IPC FSM。
8. **sim2sim vs sim2real 阶段**：他们的分支明确是 **sim2sim**（RoboJuDo MuJoCo 内验证，真机前一步）；我们已写到真机 `rt/lowcmd_rl` + 安全 guard，但同样**未上机**。两边都还差真机这一步。

## 建议
- **可借鉴**：`DoFAdapter` 的通用 slice/expand；wrist-load（如做负载行走）；他们的 RoboJuDo-MuJoCo + xvfb 验证可与我们的 AGILE-sim2mujoco + EGL 在**同一 MJCF** 上交叉验证。
- **务必注意**：两套**不可混用权重/config/增益/obs 维度**（68 vs 240、12 vs 13、LSTM vs history-MLP）。
- **若要"统一"**：可把我们的 velocity-height-LSTM 也做成一个 RoboJuDo config（像他们那样），或反过来把他们的 velocity-history 接入我们的 box_demo_2 线——但这是后续工程，当前两条线服务不同目标。
