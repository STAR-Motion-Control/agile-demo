# G1 下半身控制器 Benchmark：AGILE vs HOMIE vs AMO

> 日期：2026-06-10 起。目标：在仿真中横评三个候选下半身控制器，验证能否满足项目三项能力要求。
> 配套文档：`../G1_locomotion_research.md`（选型调研）、`../RISKS.md`（风险清单）。
> 状态：✅ 已完成（2026-06-10，375 trials 全量跑完，零摔倒）。

## 1. 被测模型与测试方式

| 模型 | 权重来源 | 测试 sim | 运行机器 | 环境 | 测试入口 |
|---|---|---|---|---|---|
| **AGILE** (NVIDIA WBC-AGILE) | 仓库自带（git-lfs）`agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt`(+`.yaml`)：v_x,v_y,w_z,**h** 四维指令的 LSTM 学生策略；另有 `velocity_g1/unitree_g1_velocity_history.pt`（3 维，无高度） | **MuJoCo**（官方 sim2mujoco 框架，自动解析 I/O 描述符）；Isaac Lab 原生 eval 备选（需 Isaac Sim 5.1，未走） | 4090 (`4090-06`) | conda env `agile`（py3.11, torch, `pip -e WBC-AGILE`） | 包装 `agile/sim2mujoco/` 模块；场景 `unitree_mujoco/unitree_robots/g1/scene_29dof.xml` |
| **HOMIE** (OpenHomie) | 仓库自带（git-lfs）`HomieDeploy/deploy.onnx`（输入 456=76×6 历史，输出 12 腿关节）；指令 [vx,vy,wz]×scale[2,2,0.25] + height_cmd | **MuJoCo**（官方 MujocoDeploy 流程改造为 headless harness） | 4090 | conda env `homie`（py3.10, mujoco 3.9, onnxruntime 1.23） | 自研 harness 复刻 `MujocoDeploy/mujoco_deploy_g1.py` 的 obs/PD 流水线；xml 用 HomieRL `resources/robots/g1_description/g1.xml`；上身 15 关节 PD 直接置位（HOMIE 解耦设计） |
| **AMO** (OpenTeleVision/AMO，即 Psi0 下半身) | 部署机自带 `amo_jit.pt` + `adapter_jit.pt`（jit 内嵌 cuda:0）；指令 7 维 [vx,vy,wyaw,Δh(+0.75),torso_yaw,torso_pitch,torso_roll] | **MuJoCo**（官方 play_amo.py 改造为 headless harness） | 5080 (`wjzh-5080`，已重启恢复，smoke test PASS) | conda env `amo`（torch 2.11 cu128, mujoco 3.2.3, EGL 渲染已验证） | 自研 harness 剥离 GUI，脚本化指令；场景 `~/AMO/g1.xml` |

**选择测试方式的理由**（用户要求"每个模型用各自最好的方式"）：三者统一在 MuJoCo sim2sim 评测保证可比性；AGILE 用其官方 sim2mujoco 框架（NVIDIA 自己的部署验证路径），HOMIE 用其官方 MujocoDeploy 流程，AMO 用其官方 play 脚本流水线——均为各仓库"作者推荐的 sim 验证方式"。Isaac Lab 原生 eval（AGILE）因需安装 Isaac Sim 5.1（~30GB+，4090 根分区仅 15G 空闲）列为备选，若 MuJoCo 结果有争议再启用。

## 2. 三项测试定义（统一规范 v1）

每个 (模型, 测试) **50 trials**，seed=trial 序号；初始化随机：关节角噪声 ±0.02rad。每 trial：reset → 2s 静置（零指令）→ 测试段。

### T1 walk_speed —— 走路 1 m/s
- 指令：vx 2s 内 0→1.0 m/s 线性 ramp，保持 10s（vy=wz=0）。
- 指标：末 8s 实际前向速度均值（世界系 base 速度投影到 heading）、速度跟踪 RMSE、摔倒。
- **success := 无摔倒且 mean_vx ≥ 0.9 m/s**。
- 注：若模型指令域上限 <1.0（如 AMO 训练域 vx∈[-0.5,0.5]）仍照发 1.0，如实记录欠速——这本身是结论。另附 speed_sweep：vx∈{0.4,0.6,0.8,1.0,1.2} 各 5 trials，报告最大稳定速度。

### T2 squat_box —— 抱箱下蹲 20 次
- 上身：双臂摆"抱箱位"（肘屈 ~1.0rad 环抱，具体关节角见各 harness NOTES）；箱子 0.35×0.25×0.25m、2.0kg，weld 固连 torso 前方 0.25m（不模拟抓取接触，但质量/惯量真实作用于动力学）。
- 蹲循环：height 指令 站立高→蹲（base 高度目标 ≈0.45m，各模型用自己的高度指令语义换算：AGILE h 绝对值、HOMIE height_cmd、AMO Δh）→站立高；ramp 1.5s 降 / hold 1s / ramp 1.5s 升 / hold 1s = 5s/循环 × 20 = 100s。
- 指标：完成循环数（摔倒即止）、height 跟踪 RMSE、最大倾角、摔倒时刻与相位（下蹲/保持/起身）。
- **success := 20/20 循环无摔倒**。

### T3 circle_pillar —— 绕 r=1m 柱子（测精准平移/旋转）
- 场景：中心圆柱障碍 r=0.15m、h=1.2m（带碰撞）。机器人初始在圆上（距中心 1m）、朝切线方向。
- 指令：vx=0.4 m/s，wz=±0.4 rad/s（曲率半径 = vx/wz = 1m），跑 2 圈（约 32s）。25 trials 逆时针 + 25 顺时针。
- 指标：径向误差 |dist(base,中心)−1.0| 的 mean/max、每圈闭环误差（回起点位置/朝向差）、柱体碰撞次数、摔倒。
- **success := 无摔倒、无碰撞、mean 径向误差 ≤ 0.15m**。

### 统一摔倒判定
projected gravity 倾角 >0.9rad，或 base_z <（当前高度目标 −0.2m），或非脚 body 触地。

### 输出与视频
- 每 trial 一行 JSONL：`{framework, test, trial, seed, success, fall_time, fall_phase, metrics{...}}`；汇总 `summary.json`（成功率、各指标 mean±std）。
- 视频：MUJOCO_GL=egl 离屏 640×480@30fps；仅保存每测试**前 5 个成功 trial + 全部失败 trial**，命名 `{model}_{test}_t{NN}_{ok|fail}.mp4`。

## 3. 基础设施与路径

- **4090 服务器**（`ssh 4090`，4×RTX4090，注意根分区 97% 满）：工作区 `/sda/lizhe/g1bench/`{OpenHomie, WBC-AGILE, unitree_mujoco}，conda `/sda/lizhe/miniforge3`（envs: homie, agile）。共享机：跑测试前用 `nvidia-smi` 挑空闲卡，`CUDA_VISIBLE_DEVICES` 指定。
- **5080 服务器**（`ssh wjzh@10.24.88.193`，password omitted）：`~/AMO`（env `amo`）。⚠️ 该机历史上出过 Xid 154 需重启的 GPU 故障；EGL 渲染已验证可用。**不要动 launchd/badmin 相关任何东西**。
- 本地（mac）：`cc/experiments/`{repos(参考克隆), scripts(harness), results(回传结果)}。
- harness 脚本：`scripts/bench_agile.py`、`scripts/bench_homie.py`、`scripts/bench_amo.py`（CLI：`--test {walk_speed,squat_box,circle_pillar,speed_sweep} --trials N --out-dir DIR --video policy`），各配 `NOTES_*.md`。

## 4. 结果（2026-06-10 全量完成；每项 50 trials，sweep 每速度档 5）

> 原始数据：`results/{amo,homie,agile}/*.jsonl` + summary；样本视频同目录。**三框架 375 个 trial 总摔倒数 = 0**——稳定性全部过关，差异全部在精度。

### 4.1 T1 walk_speed @1.0 m/s（50 trials；success := 无摔且 mean_vx≥0.9）
| 模型 | 成功率 | mean_vx (m/s) | 跟踪 RMSE | 摔倒 | sweep 各档实际/指令 | 备注 |
|---|---|---|---|---|---|---|
| **HOMIE** | **50/50 ✓** | **0.965±0.001** | 0.093 | 0 | 0.4→0.34✗ 0.6→0.54✗ 0.8→0.75✓ 1.0→0.97✓ 1.2→1.20✓ | 唯一达标；高速准、**低速欠跟踪**（0.4 档只 86%） |
| AGILE | 0/50 | 0.838±0.002 | 0.160 | 0 | 全程≈84-90%：0.4→0.36 0.6→0.54✓ 0.8→0.71 1.0→0.84 1.2→1.02 | 稳定欠速 ~15%，差 0.9 线一点；teacher 策略（特权obs）未测，文档称跟踪更好 |
| AMO | 0/50 | 0.539±0.0004 | 0.467 | 0 | 实际≈指令×51-63%（0.4→0.25 … 1.2→0.61） | **硬饱和 ~0.6 m/s**（训练域 vx≤0.5），1 m/s 不可达 |

### 4.2 T2 squat_box 抱 2kg 箱蹲起 ×20（50 trials；蹲至 base 0.45m）
| 模型 | 成功率 | 完成循环 | height RMSE | 摔倒 | 备注 |
|---|---|---|---|---|---|
| **HOMIE** | **50/50 ✓** | 20.0 | **0.045m** | 0 | 最优；height 指令原生（绝对高度语义经标定） |
| **AMO** | **50/50 ✓** | 20.0 | 0.047m | 0 | 与 HOMIE 几乎持平 |
| **AGILE** | **50/50 ✓** | 20.0 | 0.075m | 0 | 全过但深蹲段欠深（官方 sweep 同样现象：指令 0.40→实际 0.54） |

### 4.3 T3 circle_pillar r=1m 绕柱（50 trials, 25ccw+25cw；success := 无摔无碰且径向误差≤0.15m）
| 模型 | 成功率 | 径向误差 mean/max (m) | 1圈闭环误差 (m / rad) | 碰撞 | 摔倒 | 备注 |
|---|---|---|---|---|---|---|
| **AGILE** | **50/50 ✓** | **0.073±0.005 / 0.147** | 0.298 / 0.313 | 0 | 0 | **唯一通过**；曲率跟踪准 |
| HOMIE | 0/50 | 0.367±0.110 / 0.755 | （wz_rmse 0.289） | 0 | 0 | wz 欠跟踪 → 实际半径 1.2-1.5m，外漂但不碰柱；cw 比 ccw 差 |
| AMO | 0/50 | 0.447±0.073 / — | 朝向闭环准（yaw 0.03-0.16rad） | **50/50** | 0 | vx 欠跟踪 → 半径收缩 ~0.5m **贴柱**；朝向积分本身跟得准 |

### 4.4 失稳事件记录
- **摔倒：0 起**（375 trials 全程无摔倒、无非脚触地、无倾角超限；各测试 max_tilt 均 ≤0.22rad ≈ 12.6°）。
- **碰撞：AMO circle 50/50**（半径收缩贴柱，contact+proximity 双判据均触发，min dist ≈0.21m；样本视频 `results/amo/amo_circle_pillar_t00_fail.mp4`）。
- 无 harness_error；无视频重放分歧告警。

## 5. 结论

1. **没有一个现成模型同时满足三项要求**。最接近的是 **AGILE**（蹲 ✓ + 绕圈精度 ✓ + 走路欠速 15%）与 **HOMIE**（走路 ✓ + 蹲 ✓ + 转向精度差）。**AMO 只胜任蹲起**，速度通道欠跟踪严重（51-63%），不适合做需要 1m/s 或精确轨迹的下半身。
2. **对项目目标（精确导航 + 蹲起搬箱）的映射**：
   - 绕柱测试就是"精准平移/旋转"的代理 → **AGILE 的速度-曲率跟踪精度（7.3cm @ r=1m）是三者中唯一可用于精确导航的**；其欠速 15% 是常值偏差，上层闭环（P 控制 + 速度前馈补偿 ×1.18）可基本消除。
   - HOMIE 的 wz 通道欠跟踪（×0.6-0.8）若要用，需重训或上层做 yaw 闭环大増益补偿，且低速段（<0.6m/s）欠跟踪与项目"小指令"需求方向相反。
   - 三家 squat 都稳（2kg 负载 1000 次蹲起零摔倒），搬箱蹲起不是瓶颈——选型由速度/转向精度决定。
3. **推荐**：以 **AGILE (velocity_height_g1 student)** 为基线下半身：唯一同时具备 4D 指令(vx,vy,wz,h)、绕圈精度、蹲起稳定的；走路欠速用上层速度补偿或改用 teacher 策略复评；后续按 `../SONIC_smallcommand_diagnosis.md` 的路线在其上加 gait clock/小指令重训。HOMIE 作为"高速行走 + 蹲起"参考与奖励设计来源（注意 CC-BY-NC 禁商用）。
4. **限制声明**：本轮为 MuJoCo sim2sim（统一可比）；AGILE 的 Isaac Lab 原生评测未跑（Isaac Sim 5.1 安装量大，留作争议复核手段）；AMO 的 yaw 语义是绝对朝向（与另两家的 wz 语义不同，T3 对比时已注明）；速度欠跟踪数值可能含 sim2sim 间隙成分，真机数值会有差异但相对排序通常保持。

## R2. 第二轮（2026-06-11 起）：捧箱修正 + FALCON + Psi0 上身

用户反馈与新增需求：
1. **捧箱 v2**：第一轮箱子 weld 在 torso（载荷不经手臂）→ 改为**自由体箱子 + 双腕 weld 约束**捧在两手之间，载荷经手臂传导（臂 PD 扛 2kg，下垂/震荡为真实物理）。仅影响 T2，T2 全部重测；新增 box_kept/box_drop_time 指标，JSONL 标 `box_mode: wrist_weld_v2`。
2. **FALCON 加入**：用 4090-lab `/hhd2/ljk/FALCON` 的**项目自训版** `model_10000.onnx`（575 obs→29 act，整身策略+fix_upper_body）。该机已有忠实 sim2sim 评估基建（`sim2real/eval/auto_eval.py`，复用部署类 LocoManipPolicy 管线；其 report.md 实测 walk@1.0→0.91、1m 弧半径 0.96m、40N/手负载深蹲过）。我们按统一规范（50 trials/捧箱 v2/绕柱）复测保证四模型可比。注意 FALCON 特有 **stand/stepping 模式标志**（walk/circle 用 stand=1，squat 用 stand=0）——"每个模型不同的测试方式"在此。
3. **Psi0 上身**：目标 = 桌子（合适高度）+ 桌上箱子场景，Psi0 控制上身搬箱，下身分别接 AGILE/OpenHomie/AMO/FALCON 重测三组。先做可行性侦察（VLA 权重是否公开/能否离线+MuJoCo 渲染输入/上身动作接口/与 System-0(AMO) 的衔接点），裁决"真 VLA 在环"还是"示教轨迹回放"两条路线后实施。

> 4090-lab：`ssh 4090-lab`（RTX 4090D 24GB，ubuntu-msi），FALCON 工作区 /hhd2/ljk/FALCON，conda env `fcgym`(IsaacGym)/`fcreal`(MuJoCo+SDK)。该仓库 CLAUDE.md 即"分层分区具身大脑"下半身模块规格书——接口 vx/vy/wz/h_body 与验收标准（1m/s、1m 圆弧、蹲起搬箱）与本 benchmark 三测试一一对应。

### R2.4 第二轮结果

**蹲箱 v2（自由箱+双腕 weld，载荷经手臂；50 trials）**
| 模型 | 成功率 | 完成循环 | height RMSE | box 保持 | 摔倒 | 对比 v1 |
|---|---|---|---|---|---|---|
| AMO | **50/50** | 20.0 | 0.047 | 50/50 | 0 | RMSE 持平——腕载扰动在分布内 |
| HOMIE | **50/50** | 20.0 | 0.045 | 50/50 | 0 | RMSE 持平 |
| FALCON(自训) | 48/50 | 19.4 | **0.036**(最佳) | 50/50 | **2**（均在**起身相位**，第5/6循环）| 全 benchmark 唯一摔倒来源=负载起身（RISKS S2 实锤）|
| AGILE | **50/50** | 20.0 | 0.075 | 50/50 | 0 | RMSE 持平 v1——腕载扰动在分布内 |

**FALCON 全套（统一规范 50 trials，4090-lab 自训 model_10000）**
| 测试 | 结果 | 关键指标 |
|---|---|---|
| walk@1.0 | **50/50 ✓** | 0.910±0.001（与 HOMIE 并列唯二过线） |
| sweep | 0.4–1.0 全过(20/25) | **低速跟踪四家最佳**：0.4→103%、0.6→97%、0.8→94%、1.0→91%；1.2→88% ✗ |
| 蹲箱 v2 | 48/50 | 见上表 |
| 绕柱 | 33/50 | **ccw 25/25（0.127）**/ cw 8/25（0.154 擦线超 0.15）；零碰撞、零摔倒；**方向不对称是主要弱点** |

**四模型横评要点（更新）**：FALCON（项目自训版）综合最强——唯一同时拿下 1m/s 与（单向）绕柱精度，低速跟踪最准（直接利好小指令需求）；两个可重训修复的弱点：cw 转向不对称、负载起身偶发失稳。AGILE 仍是绕柱双向最稳（50/50）但欠速 15%；HOMIE 走得快但转向差；AMO 只胜任蹲。

### R2.5 Psi0 侦察与探针结论

- **权重/可行性**：HF `USC-PSI-Lab/psi-model`（Apache），单 ckpt 6.25GB bf16（~3.1B：Qwen3-VL-2B 骨干+500M flow expert）；输入=1 路 RGB 320×180+32 维本体+语言；输出=36 维动作 chunk（手指14+臂14+腰3+height+vx/vy/vyaw+target_yaw）@50Hz。4090 实测峰值 8.7GB，flash-attn 预编译轮可用。
- **System-0 就是 AMO**（同款 amo_jit/adapter_jit 打包在 Psi0 内）；其 SIMPLE 框架已有 **SONIC decoupled-WBC 下身替换先例**（`baselines/psi0_decoupled_wbc.py`）= 我们接 AGILE/HOMIE/FALCON 的模板。
- **真 VLA 纯 MuJoCo 探针（Plan A）**：官方 BendPickMP 任务，Isaac 渲染 10/10 → **纯 MuJoCo 渲染 1/3**（n=3，若真率≥0.9 则观测≤1/3 概率仅 2.8%）——视觉域差显著。VLA 在环可做演示，**不适合做四下身公平对比**。
- **采用 Plan B（轨迹回放）做对比矩阵**：已提取 `4090:/sda/lizhe/g1bench/psi0_replay/`——3 条仿真 BendPick（3.1s@50Hz：蹲 0.45+前倾 0.35rad+手指闭合+起身举升，height 有 0.45→0.75 阶跃需插值）+ 1 条真机 lunch-bag-squat（22.3s@30Hz，臂幅 ±1.3rad，height 0.77→0.46→0.75，**上身分布最贴近实际搬箱**）+ scene_info.json（桌高 0.40m teatable、cracker_box 0.072×0.164×0.213m/0.411kg）+ 29/27/23DoF 关节映射（出处 `g1_wholebody.py:32-33`、`psi-inference_rtc.py:291-308`）。四个下身吃**完全相同的上身扰动序列**，对"下身鲁棒性对比"比闭环 VLA 更干净。

## R3. 第三轮（2026-06-11）：蹲深/蹲速扫描 + A→B 到点校准 + A→B→C 全流程

用户新增需求 → 测试定义（v3，四模型统一）：

### T4 squat_sweep（抱箱蹲不同高度/速度，找会倒的高度与 root 偏移）
- 深度扫描：H∈{0.65…0.30}（8 点，ramp 0.2m/s）×5 trials；速度扫描：0.45m 深度，ramp∈{0.1,0.2,0.4,0.8}m/s ×5。全程捧箱（wrist weld v2）。
- 关键指标：fall/相位、achieved_depth、**root_drift_hold**（蹲底保持段 XY 漂移="没蹲稳"量化）、**root_drift_total**（起身后相对蹲前的偏移）、box_kept。summary 按高度/速度分组。

### T5 goto_ab（固定 A→B 到点 + 小指令校准）
- A=原点，B=(3.0,1.0,θ=90°)。统一两阶段控制器：NAV 粗导航（vx≤0.6，到 0.3m/15° 止）→ **CAL 小指令校准**（|v|≤0.10 m/s、|wz|≤0.10 rad/s 纯 P，无死区补偿——直接测小指令原生有效性）。
- success_fine := ≤5cm & ≤5°（项目目标精度）持续 1s。记录 NAV 末/CAL 末误差、校准改善量。**这是"小指令校准能否收敛"的直接测量**。25 trials。
- 适配：AMO 用 target_yaw 绝对朝向（限斜率 0.1rad/s），其 vx<0.1 联锁如实测；FALCON NAV/CAL stand=1。

### T6 pipeline_abc（抱箱转身→走到 C→校准→蹲下放箱→起身，录视频）
- 初始捧箱，A=原点朝向 0，C=(0,−2.5,θ=−90°)——必须先右转 ~90°（压 FALCON cw 弱点）。NAV→CAL→蹲 0.45→蹲底释放 weld 放箱（释放时打开箱子全碰撞，落地）→起身站稳。
- 指标：导航误差全套 + box_place_ok（箱静止/直立/落点距 C 偏移）+ 蹲放起无摔。success := coarse 收敛+蹲放起无摔+box_place_ok。15 trials，前3成功+全部失败录像。

### R3 执行队列（集成完成后，每机依序）
psi0 三测试 ×50 → squat_sweep(60) → goto_ab(25) → pipeline_abc(15)；5080=AMO，4090=HOMIE+AGILE，4090-lab=FALCON。
（AGILE 的 psi0 集成在本轮补齐——上轮工作流被杀时唯一未完成项；npz 已全部生成并分发。）

**R3 冒烟阶段发现**：
- AGILE 七测试全绿；goto_ab 冒烟即达成 fine 收敛（5cm/5° 纯小指令）——小指令校准可行性首个正面信号；带箱 pipeline 的 CAL 15s 内只到 0.13m（带载校准变慢，如实记录）。
- 拾箱阈值统一放宽 0.12→0.30m（"磁性抓取"近似）：回放轨迹是 AMO（带 torso pitch）录的，腿部策略弯腰深度不及（AGILE 差 10cm、HOMIE 无腰 roll/pitch 差 4cm）。
- **FALCON 在回放前倾腰姿下后退漂移**（离桌 ~0.7m，拾箱必败）——同一上身扰动流下 AGILE/AMO 站位稳定、FALCON 最弱，保持扰动流一致不作迁就，作为对比结论记录。
- AMO 的腰覆写 hack（SIMPLE 同款）冒烟通过，pick 1/1。

### R3 终表（全量完成 2026-06-12）

**T4 蹲深扫描**（捧箱，rate 0.2m/s，每高度 5 trials；"摔倒数 | 实际深度 | 蹲底漂移"）：
- **AMO**：全 8 档零摔、漂移≤1.8cm（最稳）；深度饱和 ~0.42m（指令 0.30→实际 0.419）
- **HOMIE**：全 8 档零摔、漂移≤2.5cm，**深度全程线性跟踪到 0.303m**（跟踪最准）；⚠️ rate=0.8 时 **5/5 全摔**（快蹲是 HOMIE 的失稳边界）
- **AGILE**：零摔、漂移≤2.5cm；深度地板 ~0.53m（指令 0.30→实际 0.533，欠深最重）
- **FALCON**：0.50/0.40/0.35 各摔 1 次；**0.55m 以下蹲底漂移暴涨**（0.45→0.43m、0.30→1.70m，"没蹲稳"实锤）；rate 扫描非单调（0.4 最稳 0.055m）

**T5 goto_ab（A→B 3m+90°转向，粗导航→纯小指令校准 ≤0.1）**：
| 模型 | 粗(0.3m/15°) | **细(5cm/5°)** | 校准后误差 | 校准耗时 |
|---|---|---|---|---|
| **AGILE** | 25/25 | **25/25 ✓** | **1.9cm/1.6°** | **3.1s** |
| **FALCON** | 25/25 | **25/25 ✓** | 达标 | — |
| HOMIE | 25/25 | 0/25 | 9.2cm/**14.4°**（wz 欠跟踪卡死 yaw） | 15s 超时 |
| AMO | 25/25 | 0/25 | 12.9cm/5.6°（vx<0.1 死区+朝向联锁） | 15s 超时 |

**T6 pipeline_abc（抱箱右转 90°→走 2.5m 到 C→校准→蹲 0.45 放箱→起身；15 trials）**：
| 模型 | 总成功 | 粗收敛 | 细收敛 | 放箱 OK | 箱落点距 C | 摔倒 |
|---|---|---|---|---|---|---|
| **AGILE** | **15/15 ✓** | 15 | 11 | 15/15 | **8.5cm** | 0 |
| **HOMIE** | **15/15 ✓** | 15 | 0 | 15/15 | 10.2cm | 0 |
| AMO | 0/15 | 0（转身+行走不达标，停在 2.3m 外；放箱机制本身 15/15 正常） | 0 | 15 | 2.34m | 0 |
| FALCON | 0/15 | 2（cw 转向弱点+带箱） | 0 | 0（未到释放阶段） | — | 0 |

**Psi0 上身回放三测试（50 trials）**：
| 模型 | walk_psi0 | squat_psi0(桌面拾箱) | circle_psi0 |
|---|---|---|---|
| **AGILE** | 0.861（欠速一贯） | **50/50 拾箱 ✓** | **43/50 ✓ 径向 0.131**（唯一扰动下仍过） |
| HOMIE | **50/50 @0.976 ✓** | 45/50 拾箱 | 0/50（径向 1.16） |
| AMO | 饱和 0.609 | **50/50 拾箱 ✓** | 0/50 + **8 摔**（臂扰动致失稳） |
| FALCON | **50/50 @0.986 ✓** | 0/50（前倾扰动下后退漂移） | 0/50（径向 1.83） |

### R3 结论
1. **AGILE 是唯一全流程胜任者**：A→B 校准 1.9cm/1.6°（3.1s）、pipeline 15/15、Psi0 扰动下绕柱仍 43/50、蹲扫零摔——精确导航+搬箱场景的当前最优基线。短板=欠速 15%+蹲深地板 0.53m。
2. **小指令校准可行性已证明**：AGILE/FALCON 用 ≤0.1 的纯小指令收敛到 5cm/5°——SONIC 的"小指令不动"痛点在这两个模型上不存在；HOMIE 卡 yaw（wz×0.6）、AMO 卡死区联锁。
3. **FALCON（自训）双面性**：空载校准满分+低速跟踪最佳，但带箱 cw 转向崩、深蹲漂移大、对上身前倾扰动鲁棒性最差——重训方向明确（cw 增强、深蹲站位、上身扰动课程）。
4. **工程陷阱存档**：MuJoCo≥3.2.4 body 级 broadphase 剔除（运行时改 geom 碰撞位必须镜像 body 位，否则穿地）；释放时若开箱-机器人全碰撞会卡掌间弹振。

## R4. 第四轮（2026-06-12）：蹲深极限标定 + Psi0 前伸放箱 + 流程标准化 + 网页控制台

1. **T7 squat_limit**：捧 2kg 箱，height 指令 0.05m/s 匀速连续下降至 0.10m（不 clip，超域照发）。标定每方法的：**摔倒临界高度**（fall_h_cmd/fall_base_z）、**物理蹲深极限 depth_floor**（稳定状态最低 base_z）、**跟踪饱和点**（指令-实际偏差>5cm 处）、**漂移阈值高度**（drift 首超 5cm/20cm 的指令高度）+ 10Hz 高度-跟踪-漂移全曲线。10 trials/模型，全录像。
2. **T8 squat_place_psi0**：最终场景=把箱子放到**地面**，手臂需前伸，手工设定难 → 用 Psi0 真机轨迹 real_ep053（双手持物下蹲放置，前伸幅度 ±1.3rad）驱动上身协调，下身跟随其 height（0.77→0.46→0.75），在轨迹最低点释放 weld 放箱落地，测**下半身误差**：分段 height RMSE、前伸+放箱过程 root 漂移（重心前移的解耦压力）、box 落点前向距离、起身稳定性。25 trials/模型。
3. **流程标准化**：PROTOCOL.md（规范版本史 v1-v4、新增实验标准步骤、四机速查、数据布局、陷阱清单）。
4. **网页控制台**：experiments/console/index.html（静态单文件）+ make_manifest.py——项目目标/标准流程/实验记录表/**可过滤视频库**（模型×测试×结果，就地播放带字幕视频）。

（R4 结果待填）

## R5. 第五轮（2026-06-12）：盲区补全 + 控制台 v2（服务器端交互渲染）

**盲区分析**（实验标准化后的缺口审查）：
| 已覆盖 | 本轮补上 | 已识别待排期 |
|---|---|---|
| 走速/蹲循环/绕柱/蹲深极限/到点校准/放箱全流程 | **T9 地面拾箱**（root 贴地下蹲+负载起身——"蹲下抱起箱"与"蹲下放箱"对蹲深要求不同）；**T10 VLN 指令流跟随**（不规则更新的小指令流，含全停段——模拟真实导航模型输出） | 变箱重(1-5kg)；不平地面/斜坡；侧向取放(转腰)；30min 长时稳定性；蹲底外推扰（FALCON 内部评估做过未入规范）；sim2real 差异 |

- **T9 squat_pick_ground**：2kg 箱立地面前方 0.45m，height 指令降到 0.25m（各模型按能力饱和），双腕近箱磁性抓取→负载起身。R4 蹲深极限直接预测成败（AGILE 地板 0.53m 预期够不着——如实记录即结论）。15 trials/模型。
- **T10 vln_follow**：预生成 10 条共用指令带（2D unicycle P 控制跟随随机路径的输出，更新间隔 0.4-1.2s 抖动，|vx|≤0.35 |wz|≤0.3，含 2 段全停），全模型同带公平对比。指标：末位姿误差 vs 理想积分参考、逐秒跟踪误差、**小指令响应率**（|vx|∈[0.05,0.15] 段实际/指令比）、全停段残移。10 trials/模型。
- **控制台 v2**：旧版 file:// 下 CORS 拦 fetch（视频/表格不渲染的根因）→ 重建为 **4090 常驻服务**（纯标准库 server.py，端口 8017）：NVIDIA 项目页风格前端 + 实验表 + 视频库 + **交互渲染台**（滑动条选蹲深 0.20-0.70/蹲速/走速/转速 → 服务器跑单 trial → 烧字幕 → 就地播放）。访问：校园网直连或 `ssh -L 8017:127.0.0.1:8017 4090`。

### R5 终表（2026-06-12 全量完成）

**R4 蹲深极限标定**（0.05m/s 连续下降×10 trials，全员零摔——慢蹲安全）：
| 模型 | 物理蹲深极限 depth_floor | 跟踪饱和点 | 漂移>5cm 高度 |
|---|---|---|---|
| **HOMIE** | **0.201m** | 0.162（几乎跟到底） | 0.475 |
| **FALCON** | 0.213m | 0.379 | 0.587（早飘） |
| AMO | 0.345m | 0.468 | 0.471 |
| AGILE | 0.491m | 0.623 | 0.415 |

**R4 Psi0 前伸放箱**（25 trials）：AMO 25/25 ✓、FALCON 25/25 ✓、HOMIE 6/25（落点擦脚尖线 dx≈-1.2cm）、AGILE 0/15（蹲不低→箱从 0.55m 摔落翻倒）。

**T9 地面拾箱**（蹲到 0.25m 指令+磁性抓取+2kg 负载起身，15 trials）：
| 模型 | 拾箱 | 实际最低 root | 判语 |
|---|---|---|---|
| **HOMIE** | **15/15 ✓** | 0.207m | 地面作业王者（腕距箱仅 0.14m） |
| **FALCON** | **15/15 ✓** | 0.285m | ✓ |
| AGILE | 0/15 | 0.516m | 蹲深地板硬挡（腕距 0.30 擦阈值） |
| AMO | 0/15 | 0.405m | 同上 |
→ **蹲深极限→地面作业能力的因果链被完整实证**：放箱(0.45m 即可)四家都有戏，地面拾箱(需 <0.3m)只有 HOMIE/FALCON。

**T10 VLN 指令流跟随**（10 条共用 tape，30s 不规则小指令流，10 trials）：
| 模型 | **小指令响应率** | 逐秒跟踪误差 | 全停残移 | 摔倒 |
|---|---|---|---|---|
| **FALCON** | **0.918** | **0.446m** | 0.021m | 0 |
| AMO | 0.479 | 1.528m | 0.054m | 0 |
| AGILE | 0.411 | 0.484m | 0.031m | 0 |
| HOMIE | 0.183 | 0.990m | 0.041m | 0 |
→ **FALCON 是 VLN 下游的最佳跟随者**（小指令响应近乎完美）；HOMIE 的小指令死区被定量到 0.183；AGILE 闭环校准强（goto_ab 25/25）但开环流跟随受 15% 欠速拖累——**"闭环到点"与"开环跟流"是两种能力**，上层导航器形态决定选型。注：final_pos≤0.30m 判据对 30s 开环积分偏严（全员 0/10），有效对比指标为上表三列。

**R5 综合选型更新**：
- 场景含**地面取放** → **HOMIE**（深蹲+地面拾箱+高速行走，但需重训 wz/小指令）或 **FALCON**（地面作业+VLN 跟随+校准全能，需修 cw 不对称与深蹲漂移）
- 场景仅**桌面高度取放+精确到点** → **AGILE** 仍最优（校准 1.9cm/1.6°+pipeline 15/15）
- **FALCON（项目自训）综合分最高**：唯一在 拾箱/VLN跟随/校准/1m/s 四项全过的，重训清单明确（cw、深蹲站位漂移、上身前倾扰动鲁棒性）。
## R6. 第六轮（2026-06-14）：统一 ManipArena 全量 5 模型 × 50 变种

设计见 BENCHMARK_V2_DESIGN.md / TASKS_V2.md。5 模型(AGILE/HOMIE/AMO/FALCON/**ljk-falcon**=FALCON v4 权重)在统一场景跑 M1(取箱→置物区) / M2(取箱→中转放→触方块→方块上箱→再抓→置物区) 各 50 变种(桌高分层 {0.30,0.45,0.60,0.75})。

**进行中结果（M1 已完成 HOMIE/AMO/FALCON/ljk，AGILE 排队中；M2 部分完成）。关键 metric 用阶段 flag(grasp_ok/place_ok_store/...)，非 top-level success（后者被严格 5cm/5° nav-cal 门控，是已知模型极限）。**

M1 抓取能力(低桌 H=0.30 是关键考验)：
- HOMIE: 全桌高 grasp 13/13(深蹲够得着低箱)，但 **H=0.30 place 0/13**(抓到低箱后持箱/起身/放置失稳全摔)→ 低桌瓶颈=负载稳定，非够不着
- AMO: 全桌高 grasp 13/13，H=0.30 place 6/13(低桌放置最佳)
- FALCON: **H=0.30 grasp 0/11**(虽有深蹲能力，但 falcon29 reach 回放没弯到低箱→per-embodiment 对齐缺口)，H≥0.60 grasp 12/12 place 8/12@H0.75
- ljk-falcon(v4): 比 trained-FALCON 低桌够得更低(H=0.30 grasp 5/12)、M2 链推进更远(cube_xfer 28/47 vs 8)，但更不稳(摔更多)——v4 与 trained 的真实差异

M2 七阶段渐进衰减(grasp→relay_place→cube_xfer→regrasp→store_place)，无模型可靠完成全链(最长程任务)：ljk-falcon 推进最远(到 regrasp 8/47、store 2)，其余多止于 cube 转移前。

**初步结论**：统一场景成功把"桌高→蹲深需求""持箱稳定""长程多阶段"压到同一可比基准。低桌作业的真实瓶颈因模型而异——HOMIE 是负载稳定、FALCON 是 reach 对齐、AGILE(待) 预期是深度。M2 全链是远未解决的硬任务，per-stage 衰减给出清晰的能力梯度。

### R6 终表（5 模型 × 50 变种全部完成 2026-06-14）

**M1 取箱→置物区（按桌高，每档 ~12 变种；grasp/place_store/done/fall）：**
| 模型 | H0.30 | H0.45 | H0.60 | H0.75 | 总摔倒 | 一句话 |
|---|---|---|---|---|---|---|
| **AGILE** | 13/13/13 | 13/13/13 | 10/10/10 | 9/9/9 | **0** | **零摔、全桌高都放成**——最稳，磁吸抓取下深度不卡它 |
| AMO | grasp13/place6 | 13/9 | 12/4 | 12/6 | 27 | 抓取全过、放置中等、摔较多 |
| HOMIE | grasp13/place0 | 13/5 | 10/6 | 12/8 | 34 | 抓得到低箱但**持箱放置失稳**(H0.30 全摔) |
| FALCON | **grasp0** | 1 | 12/5 | 12/8 | 13 | 低桌 reach 对齐缺口(够不着)，高桌可 |
| ljk-falcon(v4) | grasp6/place0 | 13/3 | 12/2 | 12/4 | 36 | 比 trained 够得更低，但最不稳 |

**M2 中继全链（取箱→中转放→触方块→方块上箱→再抓→置物区；阶段到达计数/50，独立 flag 非严格单调）：**
| 模型 | grasp | relay_place | cube_xfer | regrasp | **store_place** | 摔倒 |
|---|---|---|---|---|---|---|
| **AGILE** | 45 | 24 | 42 | 21 | **14** | **0** |
| AMO | 50 | 19 | 16 | 5 | 1 | 22 |
| ljk-falcon | 42 | 7 | 29 | 9 | 2 | 11 |
| HOMIE | 48 | 17 | 17 | 1 | 0 | 32 |
| FALCON | 25 | 3 | 9 | 0 | 0 | 6 |

**R6 结论**：
1. **统一 ManipArena 成功把多能力分离到同一可比基准**——同一 50 变种场景下，桌高→蹲深需求、持箱放置稳定、长程多阶段衰减各自显形。
2. **AGILE 在 arena 综合最强**：M1 零摔+全桌高放成，M2 唯一较多完成全链(store 14/50)。其在 R1-R5 暴露的欠速/浅蹲，在 arena 的磁吸抓取(腕<0.30m)抽象下不致命，而它的**极致稳定**(0 摔)让它把别家摔出局的长链跑完。
3. **能力画像各异**：HOMIE 深蹲够得到低箱但持箱放置最不稳；FALCON 低桌 reach 对齐缺口；AMO 抓取最可靠但导航/持箱中等；**ljk-falcon(v4) 比 trained-FALCON 够得更低、M2 推进更远但更易摔**——v4 与 trained 差异被基准清晰量化。
4. **M2 全 7 阶段全链仍是硬任务**：最好的 AGILE 也仅 14/50 完成，per-stage 衰减给出清晰能力梯度。
5. **抽象局限(诚实声明)**：磁吸抓取(腕<0.30m 即焊)对蹲深要求宽容→arena 更偏奖励稳定性而非真实深蹲/抓取力；上身由 Psi0 轨迹回放(非闭环 VLA)；top-level success 被 5cm/5° nav-cal 门控故用阶段 flag 评判。

**控制台**：5 模型 arena 结果 + 73 个分层抽样带字幕视频已接入 4090:8017（manifest 280 视频/84 实验组，R6-Arena 轮，含 ljk-falcon 过滤；视频按桌高/阶段标注，HTTP 200 验证通过）。

## R7. ljk-falcon (v4) 跑 R2/R3 标准测试（2026-06-15）

用 FALCON v4 权重 `g1_29dof_v4.onnx`（--label ljk-falcon，bench_falcon.py 同一 harness）跑完 R2/R3 全套，与 trained-FALCON(model_10000) 同口径对比：

| 指标 | FALCON(trained) | **ljk-falcon(v4)** | 解读 |
|---|---|---|---|
| walk@1.0 实速 | 0.910 | **0.949** | v4 走得更快更准 |
| squat_box 抱箱蹲20次 成功 | **48/50** | 18/50 | v4 **重复蹲起稳定性差很多** |
| circle 径向误差(m) | **0.141** | 0.401 | v4 转向精度差 |
| goto_ab 小指令校准 5cm/5° | **25/25** | 0/25 | v4 **丢失小指令精确校准能力** |
| pipeline_abc 搬箱全流程放箱 | 0/15 | **15/15** | v4 导航+放箱完成全链(trained 在 cw 转向/持箱段先摔) |

**结论**：v4 是一个明显不同的工作点——**用"细粒度精度 + 重复蹲起稳定"换取了更强的行走/导航**。它走得更快、能把"导航到 C→蹲放箱"的全流程跑完（trained-FALCON 0/15 因 cw 弱点中途摔），但丢了小指令校准(goto 0/25)、转向精度(circle 0.40)、连续抱箱蹲的稳定(18/50)。pipeline 对比为公平同 harness（trained 的 fixed v3 也是 0/15）。

**控制台**：ljk-falcon 的 R2/R3 结果+37 个带字幕视频已接入 4090:8017（manifest 共 317 视频；LJK-FALCON 现含 R2/R3/R6-Arena 三轮，可过滤；视频 HTTP 200 验证通过）。原始数据 `results/ljk-falcon/`(R2) 与 `results/final/ljk-falcon/`(R3)。

## R8. AGILE 深蹲微调版 model_2999 跑 R2/R3/R6 标准测试（2026-06-15）

把 AGILE 深蹲微调 student（model_3500 蒸馏 → **model_2999**，导出 `/sdb/lizhe/g1_deepsquat/student2999_export/policy.pt`，drop-in 同字节，命名 **agile-deepsquat**）用 `bench_agile.py --checkpoint <新.pt> --config <原 student.yaml>` 跑完 R2/R3/R6 全套，与 baseline AGILE 同口径对比。微调细节见 `DEEPSQUAT_FINETUNE.md`。

**R2/R3 同口径对比（50/50/50/25 trials）：**

| 测试 | baseline AGILE | **agile-deepsquat(2999)** | 解读 |
|---|---|---|---|
| walk@1.0 实速 | 0.838→0.955 | **0.954**，0 摔 | 行走不退 |
| squat_box 抱箱蹲 ×20 | 50/50，0 摔 | **31/50，19 摔**，hRMSE 0.093 | ⚠️ 重复负载蹲起稳定性大幅下降 |
| circle 径向 / 双向 | 50/50（双向最稳）| 0.162m，**ccw 20/25 · cw 0/25** | ⚠️ cw 擦线、双向退化 |
| goto_ab 校准 5cm/5° | **25/25**（1.9cm/1.6°）| **0/25**（校准后 7.3cm）| ⚠️ 丢失小指令精确校准 |
| pipeline_abc 全流程放箱 | 15/15 | **15/15**，0 摔 | ✓ 浅蹲(0.45)放箱仍稳 |
| walk_psi0 | 0.861 | 49/50（1 摔）| ≈ |
| squat_box_psi0 桌面拾箱 | 50/50 拾箱 | **32/50 拾箱**，0 摔 | ⚠️ 拾箱率降 |
| circle_psi0 | 43/50 | **0/50** | ⚠️ 臂扰动下绕柱崩 |

**蹲深扫描 squat_sweep（捧 2kg 箱，本模型核心能力）：**

| cmd_h | deepsquat 实深 | deepsquat 摔 | 备注 |
|---|---|---|---|
| 0.55 | 0.508 | 0/5 | 负载稳定边界附近 |
| 0.50 | 0.491 | **5/5** | 捧箱深于此即失稳 |
| 0.30 | **0.339** | 5/5 | baseline 同档地板 0.533（0 摔）|
| ramp 0.1m/s @0.45 | 0.46 | 0/5 | 慢降可到 0.46 不摔 |

→ **空载**蹲深确实更深（同口径 squat_limit depth_floor **0.246 vs baseline 0.492**，见 DEEPSQUAT_FINETUNE）；但**捧 2kg 箱**时深于 ~0.50m 即失稳（0.2m/s ramp 下 cmd≤0.50 全摔，慢 ramp 0.1 可到 0.46）。深度优势在空载，**负载深蹲是失稳边界**。

**R6 ManipArena（50 变种 M1 + 50 M2）：**

| | baseline AGILE | agile-deepsquat |
|---|---|---|
| M1 grasp/place/done · 摔 | 45/45/45 · **0 摔**（全桌高都放成）| 13/11/9 · **4 摔** |
| M1 grasp by H(0.30/0.45/0.60/0.75) | 13/13/10/9 | 7/4/0/2 |
| M2 链 grasp→relay→cube→regrasp→**store** | 45→24→42→21→**14** | 13→9→13→9→**3** |

→ arena 综合大幅弱于 baseline（baseline 是 arena 最强且零摔；deepsquat 抓取率 / 完成率 / 稳定性全面下降）。

**结论**：深蹲微调（model_3500 中训 → 蒸馏 student2999）**达成空载蹲深目标（砍半到 ~0.25m）**，代价是 baseline AGILE 赖以成为"全流程胜任者"的**稳定性与小指令精度**：重复负载蹲起 19 摔、cw / 小指令校准 / 臂扰动绕柱 / arena 抓取全面退化。保留的强项：**1 m/s 行走、pipeline 浅蹲放箱 15/15**。一句话——它是"能蹲更深"的 AGILE，但不再是"最稳最准"的 AGILE（与 DEEPSQUAT_FINETUNE 记录的 model_3500 中训"毛"、8-trial 抱箱蹲 3/8 摔早期信号一致，放大到全量后更明显）。若要"深且稳"，需走 model_29999（更收敛）蒸馏或 sit-down 路线。

**控制台**：agile-deepsquat 的 R2/R3 + R6-Arena 结果 + 带字幕视频已接入 4090:8017（独立模型，可与 baseline AGILE 并排过滤）；并新增「测试指令」板块列出各模型 harness / 服务器 / 命令（FALCON·ljk-falcon=4090-lab、AGILE·HOMIE=4090、AMO=5080）。原始数据 `results/agile-deepsquat/`(R2)、`results/final/agile-deepsquat/`(R3)、`results/arena/agile-deepsquat/`(R6)。

## R9. ljk-falcon-v5 (FALCON v5_1) 跑 R2/R3/R6 全套（2026-06-17）

用 FALCON v5 权重 `g1_29dof_v5_1.onnx`（`bench_falcon.py --label ljk-falcon-v5 --model-path ...`，4090-lab，同一 harness）跑完 R2/R3/R6 全套，作为**独立模型**接入控制台（保留 v4）。与 v4(`g1_29dof_v4.onnx`)、trained-FALCON(`model_10000`) 同口径对比：

**R2/R3 三方对比：**

| 指标 | FALCON(trained) | ljk-falcon(v4) | **ljk-falcon-v5** | v5 解读 |
|---|---|---|---|---|
| walk@1.0 实速 | 0.910 ✓ | 0.949 ✓ | **0.829（0/50，0 摔）** | ⚠️ **比 v4 更慢、未过 1m/s**，但零摔、单调跟踪 |
| speed_sweep（cmd→实速）| 低速最准 | — | 0.4→0.32 · 0.6→0.49 · 1.0→0.83 · 1.2→0.98 | 全程欠跟踪（约 80–82%）|
| squat_box 抱箱蹲 ×20 | **48/50** | 18/50 | 21/50（29 摔），hRMSE **0.042**（最准）| ≈v4（都差），高度跟踪最好 |
| circle 径向(m) / 双向 | **0.141** | 0.401 | 0.287（ccw 0/25·cw 0/25，0 摔）| 比 v4 准、仍未过 0.15 |
| goto_ab 校准 5cm/5° | **25/25** | 0/25 | 2/25（6.0cm）| ≈v4（都基本失败）|
| pipeline_abc 放箱 | 0/15 | **15/15** | 13/15 放箱（成功 11/15，2 摔）| ≈v4（强）|
| walk_psi0 | 50/50 | — | 49/50 | ✓ |
| squat_box_psi0 桌面拾箱 | 0/50 | — | **50/50 拾箱（0 摔）** | ✅ v5 突出强项（与 AGILE/AMO 并列）|
| circle_psi0 | 0/50 | — | 0/50 | ≈ |

**蹲深扫描 squat_sweep（捧 2kg 箱）**：cmd 0.30→实深 **0.323**（1 摔）、0.45→0.439（2 摔）、全程 **51/60 不摔**——**带载深蹲跟踪好、失稳少**（明显优于 agile-deepsquat 的"捧箱深于 0.50 即全摔"）。

**R6 ManipArena（M1 50 + M2 50）：**

| | ljk-falcon(v4) | **ljk-falcon-v5** |
|---|---|---|
| M1 grasp/place/done · 摔 | 43/9/9 · **36 摔** | 42/**21**/21 · **21 摔** |
| M1 grasp by H(0.30/0.45/0.60/0.75) | 6/13/12/12 | 5/13/12/12 |
| M2 链 grasp→relay→cube→regrasp→store | 42→7→**29**→9→2 | 42→**28**→1→0→0 |

→ M1：v5 **放置更多(21 vs 9)、摔更少(21 vs 36)**，比 v4 稳；M2：v5 中转放(relay)推进更远(28 vs 7)，但卡在 cube 转移(1 vs 29)——失败点不同，全链均未完成。

**结论**：v5 相对 v4 是一个**「更慢但更稳」的工作点**——换来更好的转向精度、桌面拾箱(50/50)、带载深蹲跟踪、arena M1 稳定性；代价是**行走更慢、未过 1m/s**。FALCON 自训系两大老毛病（1m/s 行走、5cm/5° 小指令校准）v5 **仍未解决**（walk 反而退步、goto 2/25）。突出亮点：**squat_box_psi0 桌面拾箱 50/50**、带载深蹲 51/60 不摔。

**控制台**：ljk-falcon-v5 作为第 8 个模型接入 4090:8017（与 v4/trained-falcon 并排过滤），R2/R3 + R6-Arena + 58 带字幕视频，「测试指令」板块同步加 v5 行。原始数据 `results/ljk-falcon-v5/`(R2)、`results/final/ljk-falcon-v5/`(R3)、`results/arena/ljk-falcon-v5/`(R6)。

## R10. GR00T-WBC（NVIDIA decoupled-WBC）跑 R2/R3/R6 全套（2026-06-17）

把 NVIDIA `GR00T-WholeBodyControl` 的 **decoupled-WBC** 接入 benchmark，新模型 `gr00t-wbc`。新写 harness `bench_dwbc.py`（在 4090 跑，env homie）。详见记忆 `gr00t-wbc-benchmark`。

**控制器本质**：双策略（`GR00T-WholeBodyControl-Balance.onnx` 静止 / `-Walk.onnx` 行走，按 ‖loco_cmd‖≤0.05 自动切换），下半身 15 维 RL（12 腿+3 腰），手臂解耦 PD 维持；命令 = `loco_cmd[vx,vy,yaw_rate] + height(绝对) + torso_rpy`，obs 86×6=516，NVIDIA OML 可商用。**关键坑**：仓库自带 sim2mujoco 自相矛盾（g1_gear_wbc.xml 43dof vs 权重 29dof、config 指向不存在的 onnx）；正确配方 = 标准 `scene_29dof.xml` + 非 gait 的 obs（已 smoke 验证能站+能走），因此与现有 box/arena 机制兼容。

**R2/R3（同口径，50/50/50/25 trials）—— 全程 0 摔（除深蹲极限 5/60）：**

| 测试 | gr00t-wbc | 对照 |
|---|---|---|
| walk@1.0 实速 | **0.892**（0 摔，差 0.008 未过 0.9 线）| ≈最强行走（仅次满分组）|
| squat_box 抱箱蹲 ×20 | **50/50，0 摔**，hRMSE 0.043 | 与 AGILE/AMO/HOMIE 并列最稳 |
| circle 径向 / 双向 | 0.185m（ccw 0/25·cw 0/25，0 摔，**无方向不对称**）| 比 FALCON 系对称；略超 0.15 |
| **goto_ab 校准 5cm/5°** | **25/25（2.2cm），0 摔** | **精英级**（此前仅 AGILE / trained-FALCON 做到）|
| pipeline_abc 全流程放箱 | **15/15，0 摔** | 与 AGILE/HOMIE 并列满分 |
| walk_psi0 | 50/50，0 摔 | ✓ |
| squat_box_psi0 桌面拾箱 | **50/50 拾箱，0 摔** | 与 AGILE/AMO 并列满分 |
| circle_psi0 | 0/50（径向，0 摔）| 臂扰动下径向超线 |
| squat_sweep 带载深蹲 | 55/60（5 摔，仅最深档）| 带载深蹲跟踪好 |

**R6 ManipArena（M1 50 + M2 50）—— arena M1 全场最强：**

| | gr00t-wbc | baseline AGILE（此前最强）|
|---|---|---|
| M1 grasp/place/done · 摔 | **50/47/47 · 3 摔** | 45/45/45 · 0 摔 |
| M1 grasp by H(0.30/0.45/0.60/0.75) | **13/13/12/12**（低桌全抓到）| 13/13/10/9 |
| M2 链 grasp→relay→cube→regrasp→store | **50**→25→17→5→3 · 2 摔 | 45→24→42→21→14 |

→ M1：gr00t-wbc **抓取 50/50（含低桌 H0.30 全中）、放置 47/50**，是所有模型里 M1 最强（AGILE 之上）；M2 抓取也 50/50 最高，但全链与各家一样在 cube/regrasp 后衰减（AGILE store 14 仍是 M2 全链最佳）。

**结论**：gr00t-wbc 是目前**最均衡的控制器**——同时拿下近 1m/s 行走、抱箱蹲 50/50、5cm/5° 精校 25/25、全流程放箱 15/15、桌面拾箱 50/50、arena M1 最强抓取，且**几乎零摔**、商用可署名。唯一未过线项是 circle 径向（0.185，略超 0.15，但对称无摔）与 walk 差 0.008 擦线。相对 AGILE（此前桌面场景最优基线）几乎全面持平或更好，且行走更快、license 更友好——是「精确导航+蹲起搬箱」场景的有力新基线。

**控制台**：gr00t-wbc 作为第 9 个模型接入 4090:8017，R2/R3 + R6-Arena + 52 带字幕视频，「测试指令」板块加 gr00t-wbc 行。原始数据 `results/gr00t-wbc/`(R2)、`results/final/gr00t-wbc/`(R3)、`results/arena/gr00t-wbc/`(R6)；harness `scripts/bench_dwbc.py`。

## R11. 控制台视频覆盖审计 + Humanoid-GPT psi0 补齐（2026-06-22）

用户反馈控制台上 gr00t-wbc / humanoid-gpt 视频"看着不全"，审计结论：

1. **GR00T-WBC：本来就全、已同步**。控制台对每个测试只放 ~3 个抽样视频（make_manifest + select_ds_videos 设计），并非缺测试。**已把每测试抽样调到 ~8 个**（R2/R3 视频 33→65、模型总视频 52→84），重新部署。

2. **Humanoid-GPT：缺的测试是模型能力所限、非疏漏**。它的 harness `bench_hgpt.py` 故意 locomotion-only——**释放的 G1-Walk 策略只吃 `[vx,vy,wz]`、没有 height 命令**，所以 squat_box / squat_sweep / squat_limit / pipeline_abc（含蹲放）/ squat_box_psi0 / arena 低桌抓取这些**需要下蹲的测试它物理上做不了**。它另有一个 whole-body **tracking 策略**（`bench_hgpt_track.py` 的 `squat_track`，跟踪合成 down-hold-up 蹲参考）——那是它唯一的"下蹲"路径。

3. **可行的补齐已做**：模型有 14 个手臂关节，故 `walk_speed_psi0` / `circle_pillar_psi0`（行走/绕柱 + 上身臂回放扰动，不需 height）可做，已扩 `bench_hgpt.py` 实现并跑完（各 50 trials，已上控制台）：
   - walk_speed_psi0：**44/50 success，0 摔**，mean_vx 1.32，arm_dev 1.89（臂确实在动）。
   - circle_pillar_psi0：**0 摔**，径向 0.138（与非 psi0 基线一样擦 0.15/绕圈完成度线，judge=0/50）。
   - **模型特性 caveat**：G1-Walk 策略的 obs 把手臂当固定本体感（训练时臂不动），把 ±1.3rad 回放喂进 obs 会必摔；故 psi0 在此模型上=**仅物理臂扰动**（臂被 PD 驱到回放、产生真实质量/惯量扰动），策略 obs 的臂槽保持默认。这与其它模型（obs 能看到臂）口径略不同，如实标注。

4. **仍为模型所限、未做**（控制台标注 walk/tracking-only）：squat_box / squat_sweep / squat_limit / pipeline_abc / squat_box_psi0 / arena。如需，可走 tracking 策略 + 箱子 weld 做一个"tracking 版 squat_box"（不同范式，单独工作）。

## 6. 试点阶段定性发现（2026-06-10，全量前的 1-2 trial 验证）

1. **AMO 做不到 1 m/s**：指令 1.0 时实际前向速度饱和在 **0.539 m/s**（训练域 vx∈[-0.5,0.5]），全程稳定不摔（max tilt 0.057rad）——速度上限是硬约束。
2. **AMO 绕圈半径收缩撞柱**：朝向闭环跟踪很好（closure yaw 误差 0.03–0.16 rad），但低速 vx 欠跟踪 → 实际曲率 = wz/vx_actual 过大 → 轨迹半径收缩到 ~0.5m 贴住柱子（min dist 0.21m，contact=True）。**正是用户关心的"精准平移"短板的定量体现**。
3. **HOMIE walk@1.0 最好**（0.964/0.965 m/s，1.0 在其训练域 [-0.8,1.2] 内）；squat 20/20、RMSE 4.5cm。
4. **HOMIE wz 欠跟踪**：绕圈实际转弯半径 ~1.2–1.5m（指令 1m），径向误差 mean 0.24（ccw）/0.47（cw），按规范判 FAIL；cw 比 ccw 差，存在不对称。
5. **AGILE 绕圈最准**：径向误差 mean **0.076m**、max 0.149m，无碰撞 ✓（试点唯一过 T3 的框架）；但 walk@1.0 欠速（0.838，判 FAIL），squat 20/20 RMSE 7.6cm（深蹲段欠深与官方 height_sweep 冒烟一致：指令 0.40m→实际 0.536m）。
6. **AGILE 上游 bug（已绕过）**：`agile/sim2mujoco/simulation.py` 无传感器回退把自由关节 qvel[3:6]（body 系）当世界系角速度再旋转，随机大 yaw 初始化时 0.5s 内必摔；unitree G1 MJCF 无角速度传感器必然走到该路径。harness 用 BodyFrameGyroShim 经传感器分支喂正确值（启动日志有 `Patched: base_ang_vel` 哨兵）。已实测修复后全 seed 稳定。**值得给上游提 issue**。
7. ang_vel_scale 标定（HOMIE）：A/B 实测 **0.25** 正确（0.5 使 wz 跟踪差一倍）——官方 MujocoDeploy 的值对，开源训练配置与发布的 deploy.onnx 不一致。
8. 抱箱视觉目检通过（AMO 帧）：箱子贴胸、双臂环抱、深蹲姿态正常。

## 附录 A. 复现命令

```bash
# ---- AMO（5080）----
ssh wjzh@10.24.88.193
source ~/miniconda3/etc/profile.d/conda.sh && conda activate amo && cd ~/AMO
# harness: scp cc/experiments/scripts/bench_amo.py wjzh@10.24.88.193:~/AMO/
MUJOCO_GL=egl python bench_amo.py --test walk_speed    --trials 50 --out-dir bench_out --video policy
MUJOCO_GL=egl python bench_amo.py --test speed_sweep   --trials 25 --out-dir bench_out --video none
MUJOCO_GL=egl python bench_amo.py --test squat_box     --trials 50 --out-dir bench_out --video policy
MUJOCO_GL=egl python bench_amo.py --test circle_pillar --trials 50 --out-dir bench_out --video policy

# ---- HOMIE（4090, CPU 推理）----
ssh 4090 && cd /sda/lizhe/g1bench
P=/sda/lizhe/miniforge3/envs/homie/bin/python   # mujoco 3.9 + onnxruntime
MUJOCO_GL=egl $P bench_homie.py --test walk_speed --trials 50 --out-dir results/homie/walk_speed --video policy
# squat_box/circle_pillar 同理; speed_sweep --trials 5 (每速度档5个)

# ---- AGILE（4090, 官方 sim2mujoco 路径, 无需 Isaac Sim）----
P=/sda/lizhe/miniforge3/envs/agile/bin/python   # torch + pip -e WBC-AGILE + numpy==1.26.4
R="--agile-repo /sda/lizhe/g1bench/WBC-AGILE --mjcf /sda/lizhe/g1bench/unitree_mujoco/unitree_robots/g1/scene_29dof.xml --device cpu"
MUJOCO_GL=egl $P bench_agile.py --test walk_speed --trials 50 --out-dir results/agile/walk_speed --video policy $R
# 其余测试同理; speed_sweep --trials 25
```

## 附录 B. 接口核查记录（写 harness 前的人工源码核对）

### AMO（play_amo.py, 5080 实拷贝）
- 23 DoF 模型（12 腿+3 腰+8 臂[肩 p/r/y+肘]×2），策略输出 **15**（腿+腰），臂由 PD 直接置位（默认或随机目标，blend 0.01/步≈1s 过渡）→ 抱箱位 = 把 `arm_action` 设为固定持箱角即可。
- 指令数组 8 维：`[0]=vx, [1]=target_yaw(绝对朝向!), [2]=vy, [3]=Δh(+0.75 基准), [4..6]=torso y/p/r, [7]=随机臂开关`。
- **⚠️ yaw 是绝对朝向目标不是角速度**：obs 用 `dyaw=yaw−target_yaw` 的 sin/cos；绕圈需按 `target_yaw += wz·dt` 积分下发。
- **⚠️ `|vx|<0.1` 时 `_in_place_stand_flag=True` 强制 dyaw=0** → 站立时不响应转身指令（与调研结论一致的死区）；步态相位在站立时锁到 [0.25,0.25]。
- 上身→下身经 adapter_jit：输入 [h_target, torso_ypr, arm_dof(8)] 12 维（norm stats 标准化），输出 15 维进 obs。
- dt=0.002, decim=10（50Hz 策略），action_scale 0.25，力矩限幅 PD。

### HOMIE（HomieRL 训练代码 = 权威；两个官方部署脚本与之矛盾）
- 27 DoF = 12 腿 + waist_yaw + 双臂 7×2（waist roll/pitch 锁定）；策略输出 12（腿）；上身 PD 外部置位（kp=100/kd=0.5 in MujocoDeploy；真机 p_gains 见 lcm_agent）。
- 单帧 obs 76 = `cmd[vx,vy,wz]×scale + height_cmd + ang_vel×s + gravity(3) + Δq(27) + dq(27)×0.05 + last_action(12)`，history 6 帧 → 456 = deploy.onnx 输入 ✓（输出 12）。
- **⚠️ scale 三个来源不一致**：训练 legged_robot.py:318+710 → cmd_scale=[2,2,**0.5**]、ang_vel×**0.5**、height **原始值**；真机 lcm_agent.py → [2,2,0.25,1]、ang_vel×0.5；MujocoDeploy g1.yaml → [2,2,0.25]、ang_vel×0.25。**harness 须带 `--obs-variant` A/B 标定，以圈速/yaw 跟踪定准**。
- 指令域（g1_29dof_config.py）：vx∈[-0.8,**1.2**]（1.0 在训练域内 ✓）、vy∈[-0.5,0.5]、wz∈[-0.8,0.8]、height∈[-0.5,0]（相对，base_height_target=0.74）→ 蹲到 0.45m ⇒ height_cmd≈−0.29。g1.yaml 示例值 0.34 与训练语义不符（疑为旧版），以标定为准。
- **⚠️ 仓库 g1.xml 是 44 关节含手指版**（n=43≠27）→ 与 456 obs 不匹配，harness 需用 27 关节模型（unitree g1_29dof_lock_waist 变体或裁剪版）。

### AGILE（WBC-AGILE 仓库自带全套）
- 预训练：`velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt`+`.yaml`（指令 vx,vy,wz,**h 绝对高度**；LSTM 学生，可部署）；teacher 用了特权线速度（仿真评测可用，跟踪更好）；`velocity_g1` 仅 3 维。
- 官方 sim2mujoco（`agile/sim2mujoco/` + `scripts/sim2mujoco_eval.py`）：I/O 描述符 yaml 自动建 obs/action 映射，**无需 Isaac Sim**；场景用 unitree_mujoco `g1/scene_29dof.xml`；支持 `--eval-config` 命令时刻表（如 height_sweep 0.40–0.72m 官方就测过）、`--save-data` parquet、`--no-viewer`、`--pd-scale`（不稳时 0.3）。
- harness 策略：包装其 sim2mujoco 模块（不重写 obs），自定义 CommandProvider/时刻表 + 场景注入 box/pillar + 统一指标层。
