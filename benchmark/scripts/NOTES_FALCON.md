# FALCON (自训版) benchmark harness 笔记

harness: `bench_falcon.py` (统一规范 v1 + 捧箱机制 v2)。
对照同目录 `bench_homie.py` / `bench_agile.py` / `bench_amo.py`。

## 环境 / 启动

- 目标机: `ssh 4090-lab` (免密)。代码 `/hhd2/ljk/FALCON`,
  conda env `fcreal` (`/hhd2/ljk/miniconda3/envs/fcreal`, py3.10,
  mujoco 3.9.0, onnxruntime 1.23.2, numpy 2.2.6, cv2, pinocchio,
  unitree_sdk2py 已装 — 后两者只是 import 依赖, harness 不用 DDS)。
- checkpoint: `sim2real/models/falcon/g1_29dof_trained.onnx`
  (= release_4090/model_10000.onnx; 输入 `actor_obs` [1,575] → 输出
  `action` [1,29])。
- 部署 (scp 后):

```bash
scp bench_falcon.py 4090-lab:/hhd2/ljk/FALCON/sim2real/eval/   # 或任意路径
ssh 4090-lab
PY=/hhd2/ljk/miniconda3/envs/fcreal/bin/python
for T in walk_speed squat_box circle_pillar; do
  MUJOCO_GL=egl $PY bench_falcon.py --test $T --trials 50 \
      --out-dir <RESULTS>/falcon/$T --video policy
done
MUJOCO_GL=egl $PY bench_falcon.py --test speed_sweep --trials 5 \
    --out-dir <RESULTS>/falcon/speed_sweep --video policy
# 冒烟: --test walk_speed --trials 2 --video none
```

- 脚本放哪都行: `--repo` 默认探测 `/hhd2/ljk/FALCON` (或环境变量
  `FALCON_REPO`)。输出目录完全由 `--out-dir` 控制, 不硬编码。
- 速度参考 (4090, 2026-06-11 实测): walk 14s sim ≈ 0.34s wall,
  squat 102s sim ≈ 2.5s wall, circle ≈ 0.9s wall。50 trials 全套分钟级。

## 接口要点 (读自部署代码)

- 复用部署类 `sim2real/rl_policy/loco_manip/loco_manip.py::LocoManipPolicy`
  (继承 DecLocomotionPolicy → BasePolicy), 学 `sim2real/eval/auto_eval.py`
  的单进程 shim: 覆写 `_init_sdk_components`(去 DDS) /
  `_init_communication_components`(MuJoCo state/cmd shim) /
  `_init_input_device`(去键盘线程); `use_upper_body_controller=False`
  (避免 IK 路径, 上身经 `ref_upper_dof_pos` 控制)。
- 指令注入 (每 tick 直接写策略字段):
  `lin_vel_command[[vx,vy]]`, `ang_vel_command[[wz]]`,
  `stand_command[[0/1]]`, `base_height_command[[h_abs]]`,
  `waist_dofs_command[[0,0,0]]`。
  **stand=1 = 迈步/走, stand=0 = 站立/蹲** (walk/circle 用 1, settle 与
  squat 用 0; 与 auto_eval.py 场景一致)。
- obs 575 = 5 帧历史 × 115; 单帧为 **字典序拼接** {actions 29,
  base_ang_vel 3, command_ang_vel 1, command_base_height 1,
  command_lin_vel 2, command_stand 1, command_waist_dofs 3, dof_pos 29,
  dof_vel 29, projected_gravity 3, ref_upper_dof_pos 14}; 全由部署类自己
  构造, harness 只往 `state_processor.robot_state_data` 填
  q(3+4+29)+dq(3+3+29) (角速度 = MuJoCo 自由关节 qvel[3:6], 体坐标系)。
  config 里的 phase_time 不在 actor_obs 列表中 (其 wall-clock 实现不影响)。
- 动作: 29 维全身, scale 0.25, q_target = scaled + DEFAULT_MOTOR_ANGLES;
  `residual_upper_body_action=True` → 上身 14 维动作再加
  (ref_upper_dof_pos − default_upper) 残差。PD 增益/力矩限幅取
  `config/g1/g1_29dof_falcon.yaml` 的 MOTOR_KP/KD/effort_limit, 200 Hz
  (sim_dt 0.005, decimation 4 → 50 Hz 策略), ctrl[0:6]=0 (freebase 6 个
  基座 motor), ctrl[6:35]=tau。
- MJCF: `humanoidverse/data/robots/g1/scene_g1_29dof_freebase.xml`
  (include `g1_29dof_old_freebase.xml`)。手 link = `left_rubber_hand` /
  `right_rubber_hand` (= config left/right_hand_link_name; 注意该 body 无
  inertial、零质量, weld 仍有效)。脚部地面接触只经 8 个
  `dummy_lf_1..4`/`dummy_rf_1..4` 小球 geom (ankle_roll mesh 是
  contype=0 视觉件) — 摔倒"非脚触地"判定的脚集合含 ankle_roll + 8 dummy。
- 指令域 (config + 既往报告): 站高 DESIRED_BASE_HEIGHT=0.75;
  walk@1.0 实测 mean_vx≈0.91 (本 harness 冒烟复现 0.909);
  arc r≈0.96; 高度域实证到 0.55, **0.45 蹲目标可能超域 — 照发并如实记录**
  (冒烟实测能蹲到 base_z≈0.43, height_rmse≈0.032)。

## 捧箱机制 v2 (squat_box, box_mode="wrist_weld_v2")

- 自由体黄色半透明箱 0.35×0.25×0.25 m / 2.0 kg (freejoint), contype=0
  不与任何物体碰撞; 注入临时 scene XML (写在原 scene 同目录,
  `_bench_falcon_squat_box.xml`, 编译完即删, 保证 include/mesh 相对路径)。
- 两条 equality weld (box↔left_rubber_hand, box↔right_rubber_hand),
  solref="0.02 1" 偏软避免与臂 PD 打架; XML 里 active=false, 每 trial
  把箱子摆到双手中点(姿态取基座 yaw)后, 运行时计算 relpose
  (手系下箱位姿) 写入 `m.eq_data`, 再 `d.eq_active=1`。
- 抱箱臂姿 `HUG_ARM_POSE` (14 维, L/R: sh_pitch −0.4, sh_roll ±0.15,
  elbow 1.2): 初始关节直接置位 + `ref_upper_dof_pos` 设为同姿
  (进 obs + 残差), 双手贴箱两侧 (视频帧已核验)。载荷经手臂 PD 传导,
  下垂/震荡是真实物理, 如实记录。
- 新指标: `box_kept` (全程 |箱−torso| < 0.6 m), `box_drop_time`,
  `box_torso_dist_max`, `success_with_box`; success 本身仍 = 不摔 + 20
  循环完成 (与 HOMIE 可比), box_kept 单列。
- 兜底: 每 tick 查 `mjWARN_BADQACC` / 非有限 qpos → trial 记
  `harness_error="numeric_divergence..."`, 不中断批次; 任意异常同样记
  harness_error。

## 与统一规范的差异 / 备注

- sim_dt 0.005 / decimation 4 (FALCON 部署原生), 非 HOMIE 的 0.002/10;
  策略频率同为 50 Hz。
- settle 阶段 stand=0 (FALCON 的站立模式), walk/circle 自 t=2s 切 stand=1。
- 上身: walk/circle/sweep 用默认臂姿 (ref=default, 残差=0, 但 29 维策略
  仍主动控臂); squat_box 用 HUG 姿。
- 视频 policy 策略 = 前 5 个成功 + 全部失败 trial 确定性重跑渲染
  (cv2 mp4v 640×480@30, EGL 离屏)。

## 冒烟证据 (2026-06-11, 4090-lab, /tmp 下, 不污染仓库)

- `/tmp/falcon_smoke/{walk,squat,circle,walk_vid,squat_vid}`:
  walk t0 OK (mean_vx_last8s 0.909); squat t0 OK (20 cycles, box_kept,
  box_torso_dist_max 0.237, min_base_z 0.426, 无数值发散);
  circle ccw OK (radial_mean 0.123), cw FAIL (radial_mean 0.161 >
  0.15 — 真实行为, 非 harness 问题); 视频/抽帧已人工核验箱在双手之间。

## 残留风险

1. **0.45 m 蹲深可能超训练域**: 冒烟单 trial 能跟到 ~0.43, 但 50 trials
   随机初始化下可能出现低高度跌倒判定 (base_z < h_cmd−0.2) 或姿态发散 —
   属于要测的真实结论, 非 bug。
2. **cw 圆弧径向误差临界** (0.16 vs 阈 0.15): circle 成功率可能明显
   方向不对称 (arc r≈0.96 偏内), summary 里 by_direction 已分开统计。
3. weld relpose 在每 trial 写 `m.eq_data` (model 级字段): 串行跑没问题,
   不要并行复用同一 model 实例。
4. `left_rubber_hand` 零质量 body: weld 力经腕链传导正常 (已实测),
   但若未来换 MJCF (手 link 改名/加 inertial) 需重核 `FOOT_BODY_NAMES`
   与手 link 名。
5. onnxruntime 默认 provider (可能走 CUDA): 同机重跑确定性已经
   视频重跑校验 (无 divergence 警告), 跨机/跨版本数值可能轻微漂移。
6. 与 HOMIE 不同, FALCON 是 29 维全身策略: walk 时臂也在动, 摔倒判定的
   "非脚触地" 用 dummy 脚球集合, 若策略出现手撑地恢复会被判 fall
   (统一规范如此)。

## v4 新增测试 (2026-06-12: T7 squat_limit + T8 squat_place_psi0)

### T7 squat_limit (蹲深下降极限标定, 捧 2kg 箱 wrist-weld v2)

- `--test squat_limit --trials N` (默认 10, 建议 `--video all` — 每 trial
  都是有效标定样本)。流程: 2s 静置(站高捧箱, weld t=0) → height 指令
  0.05 m/s 匀速连续降到 0.10 m → 保持 3s → 结束(不起身), 摔倒即止。
  stand=0 全程。
- **clip 核验**: FALCON 命令层无 height clip — `base_height_command`
  原样进 obs (loco_manip.py:37), 唯一写它的是已禁用的键盘 handler
  (loco_manip.py:255-264 / dec_loco.py:109, 均在 joystick 回调内);
  harness 每 tick 直写策略字段 → 0.10 指令原样下发, 无需放开任何 clip。
- 指标: `descent_curve` ([[h_cmd, base_z, drift_xy],...] 10Hz, 漂移参考
  = 下降起始 tick 的 root XY), `fall_h_cmd`/`fall_base_z` (未摔=None),
  `depth_floor` (稳定态最小 base_z, 稳定 = tilt≤0.3 rad 且在摔倒前),
  `track_sat_h` (|指令−实际|首超 5cm 的指令高度), `drift5_h`/`drift20_h`,
  `max_drift_xy`; success = 无摔 (饱和不算失败), `success_with_box` 单列。
- 冒烟 (4090, 2 trials, /tmp/falcon_v4_smoke): 均 OK 不摔;
  track_sat_h 0.34/0.42, depth_floor 0.203/0.233, drift20_h 0.39/0.71,
  max_drift_xy 最大 2.14 m (低指令下原地漂移大, 真实行为) — FALCON 属
  "不摔但饱和"型。18s sim ≈ 0.6s wall。

### T8 squat_place_psi0 (Psi0 前伸协调放箱)

- `--test squat_place_psi0 --trials N` (默认 25), 必须
  `--upper-replay .../upper_replay_falcon29_real_ep053.npz` (22.24s,
  t_place=argmin(height_cmd)=18.04s, hold 段 16.72–18.50s
  [height≤min+0.02])。
- 流程: t=0 捧 **2kg 箱** (BOX_XML, 非 cracker; wrist-weld v2, 臂姿 =
  replay 帧 0) → 2s 静置 → 单次回放 ep053 臂+腰 (复用 psi0 通路:
  arm14→ref_upper_dof_pos, waist3→waist_dofs_command), 下身
  base_height_command 跟随其 height_cmd verbatim → t_place 释放双 weld +
  开 bit-2+body 箱碰撞 (复用 release_box/set_box_collision, 每 trial
  开头重置 model 级碰撞位) → 回放至结束(起身) → 站稳 1s, 臂冻结在
  replay 末帧。stand=0 全程。
- 指标: `height_rmse_descent/hold/rise` (按 replay 高度分段),
  `root_drift_place` (descent+hold 内相对回放起始 root XY 的最大漂移),
  `box_land_dx` (箱落点沿释放时刻朝向、相对足前缘[8 dummy 脚球投影最大值]
  的前向距离, 正=放在身前), `box_place_ok` (静止+直立+`box_land_dx>0`;
  **不**含 pipeline 的"终点距机器人<0.8m"项 — FALCON 放箱后起身阶段
  会漂移 ~1.0–1.1m 远离箱子, 该漂移由 root_drift_place/
  box_robot_dist_end 单独记录, 不混入放置判定), `end_stand_ok`,
  box_kept(携带段)。success = 无摔 ∧ box_place_ok; JSONL/summary 带
  upper_mode, summary 加 upper_replay_t_place_s, box_mass_kg=2.0。
- 冒烟 (2 trials): 均 OK; rmse d/h/r ≈0.045/0.021/0.024,
  root_drift_place ≈0.54/0.56 m (前伸重心前移的解耦压力, 真实行为),
  box_land_dx +0.14/+0.10 m, 箱静止直立, end_stand_ok。25.2s sim ≈
  0.6s wall。

### v4 零回归证据 (4090, /tmp/falcon_v4_smoke, bit-exact)

- walk_speed t0 mean_vx_last8s = 0.9089745736275969 (= results2 基线逐位)
- squat_box t0 min_base_z = 0.4257492699425572, 20 cycles (= 基线逐位)
- pipeline_abc t0 box_land_dist = 3.061228599270657 (= final/pipeline_v3
  基线逐位; FAIL 为真实行为)
- squat_box_psi0 t0 pick_failed + 20 cycles (与 final 基线同模式)
- squat_limit --video all 正常出 mp4 (4.0 MB)。

### v4 残留风险

1. T7 的 `depth_floor` 取 "tilt≤0.3 rad 稳定态" 最小 base_z — 若策略
   出现大幅前倾但不摔的深蹲姿态, 该阈值会把真实蹲深样本滤掉
   (FALCON 冒烟 max_tilt≈0.17 不受影响, 跨框架对比时注意)。
2. T7 低指令段 root 漂移可达 ~2m: 摄像机跟随 pelvis, 视频可用;
   但 drift5_h 可能在下降早期即触发 (trial1 在 0.73 就 >5cm),
   解读时配合 descent_curve 看全程。
3. T8 hold 段分段阈 `min+0.02m` 是经验值; ep053 上 hold=1.78s 连续段,
   换别的 replay 源需复核分段是否合理。
4. T8 `box_place_ok` 判定与 pipeline_abc 不同 (无 0.8m 距离项),
   跨测试比较 place 成功率时注意口径差异。

## v5 新增测试 (2026-06-12: T9 squat_pick_ground + T10 vln_follow)

实现追加在同一 `bench_falcon.py`; 既有 12 个测试代码路径零改动 (仅
`BOX_TESTS + PSI0_TESTS` 四处合并为超集常量 `WELDED_BOX_TESTS`、
`box_surface_dist` 增加默认参数 `half`、rec 增加 `yaw` 记录——均不影响
旧测试行为)。FALCON **不加** `--custom-height/--custom-rate/--custom-vx/
--custom-wz` 交互渲染 flags (规范 v5: 只加 HOMIE/AGILE 两份)。

### T9 squat_pick_ground (地面拾箱: 贴地蹲 + 抱起站立)

- `--test squat_pick_ground --trials N` (默认 15), 无需 replay/tapes。
- 场景: 2kg 0.35×0.25×0.25 箱 **立放** 地面 (长轴 0.35 竖直, 顶面
  0.35m; 经 -90° pitch 绕 y 使局部 x 轴朝上, 再随机器人 yaw 转,
  0.25×0.25 底面正对手), 箱心在机器人正前方 0.45m。碰撞 = bit-2 方案
  (箱 geom contype/conaffinity=2 编译态, 启动时地面 OR bit2 + body 级
  镜像, 同 psi0 通路): 箱碰地、永不碰机器人。
- 流程 (PickGroundMission 相位机, 同 NavMission 模式): 2s 静置(臂下垂
  默认位) → 1s 线性混合到低位前伸捧位 **PICK_ARM_POSE: 肩pitch -0.6,
  肩roll ±0.2, 肘 0.8 rad** (其余 0; ref_upper_dof_pos 残差通路下发)
  → height 指令 0.2 m/s ramp 降到 **0.25m** (统一指令不削顶, FALCON
  按 R4 标定 ~0.213 深度自行饱和) → 蹲底 hold, 双腕(rubber_hand 体)
  距箱面均 <0.30m 时磁性抓取 (原位激活双 weld, 同 psi0 拾箱机制,
  无吸附跳变) → hold 0.5s → 0.2 m/s 起身(带 2kg 载) → 站 1s。
  蹲底 5s 未达成 → pick_failed, 仍起身。stand=0 (stance) 全程。
- 指标: pick_success, pick_failed_timeout, grasp_dist_left/right
  (抓取或超时时刻), grasp_time_s, min_root_z (settle 后实际最低 root),
  root_drift (bottom+grasp_hold 段相对段首的最大 XY), box_lift_height,
  box_kept(抓取后), stand_ok (末段 1s: 平均 tilt≤0.3 rad 且
  |平均 base_z−站高|<0.1)。success = pick ∧ 无摔 ∧ stand_ok。
  summary 加 pick_success_rate (复用 squat_box_psi0 通路)。
- 预期: 成败由蹲深极限决定 (R4: FALCON 0.213 → 应可够到; AGILE ~0.53
  预期够不着, 如实记录即是结论)。

### T10 vln_follow (VLN 小指令流跟随)

- `--test vln_follow --trials N` (默认 10) `--tapes tapes.json` (必填,
  make_vln_tapes.py 预生成, **四个 harness 共用同一文件保证公平**);
  trial i 用 tape i % n_tapes, seed=trial。
- tape: 30s, [t,vx,vy,wz] breakpoint 零阶保持 (更新间隔 0.4~1.2s,
  |vx|≤0.35 |vy|≤0.2 |wz|≤0.30, 含 2 段全停 + 小指令段),
  ref_xy_yaw = 理想无误差积分 (1Hz)。载入时校验结构/有限值/时间递增。
- **FALCON stand 适配 (模式语义, 本 harness 注明)**: 跟随段 stand=1
  (stepping), tape 内全停段 (vx=vy=wz=0 的 breakpoint) 切 stand=0
  (stance), 末段 1s 站立 stand=0。wz 原生下发 (无 AMO 的
  wz→target_yaw 积分语义差)。
- 位姿统一在 **tape 起始帧** (settle 结束 tick 的 root XY+yaw) 下度量
  — ref 从单位位姿积分, 起始重零零化消除静置漂移。
- 指标: final_pos_err/final_yaw_err (tape 结束 tick 机器人位姿 vs ref
  末样本; 提前摔则取最后存活 tick, 反正 success 已被 fall 否决),
  mean_track_err (逐秒 ref 样本位置误差均值; 摔后 ghost 样本跳过),
  track_err_n, small_cmd_response (跟随段 |cmd_vx|∈[0.05,0.15] tick 上
  v_fwd/cmd_vx 比值均值, 1.0=完美), small_cmd_ticks, stop_settle
  (每个全停连续段段首→段尾 XY 位移的均值), n_stop_segments, fall。
  success = 无摔 ∧ final_pos_err≤0.30m ∧ final_yaw_err≤15°。
  summary 加 tapes_path/n_tapes/by_tape (按 tape id 聚合)。

### v5 本机验证 (无 4090 冒烟, 待远端补)

- `py_compile` 通过 (本机 py3.14)。
- mujoco stub 下单测: PickGroundMission 50Hz 全相位序列 (抓取路径
  settle→arm_reach→down→bottom→grasp_hold→rise→stand, 边界均 ≤1 tick
  偏差; 超时路径 10.5s rise, 总时长 < trial_duration 上限 15s);
  load_vln_tapes 合法/非法样例; T10 指标块 **逐字提取执行**: 完美跟随
  → final_pos_err≈0、small_cmd_response=1.0、stop_settle=0、success;
  60% 增益跟随 → response=0.600、final_pos_err>0.3、FAIL; 静置摔 →
  全 None+FAIL; 中途摔 → ghost ref 样本被截断。地面箱四元数: 局部 x
  → 世界 +z, 底面随 yaw 旋转 (数值验证)。
- 旧测试零回归: 新分支全部 `elif` 追加; pick/tape 均 None 时调度、
  rec、metrics 路径与 v4 逐字相同 (待 4090 跑 walk_speed t0 位级复核)。

### v5 残留风险

1. **PICK_ARM_POSE 未经物理冒烟**: -0.6/±0.2/0.8 是几何估计, 腕在蹲底
   能否进 0.30m 半径未在真机 mujoco 验证 — 若 4090 冒烟 grasp_dist
   始终 >0.30, 优先调肩pitch(-0.8~-0.5)/肘(0.5~1.0), 角度改动需同步
   本节与 docstring。
2. T9 磁性抓取阈值 0.30m 比 psi0 的 0.12m 宽松: weld 原位激活意味着
   箱可能"悬空"挂在离手 ~0.2m 处随臂走 (规范如此, 视频里会看到);
   起身段 2kg 经软 weld 拉起, 若 qacc 发散会被 numeric guard 截获记
   harness_error。
3. T9 立放朝向取"长轴竖直"解释 ("立放" 0.35 竖直, 顶 0.35m); 若其他
   harness 实现成平放 (顶 0.25m), 跨模型 grasp 几何不可比 — 统一前
   先对齐此口径。
4. T10 全停段判定基于 **指令逐字段全零**: tape 若给 1e-6 级残余速度
   则不会切 stand=0; make_vln_tapes.py 需保证停段精确写 0。
5. T10 摔后 final_pos_err 取最后存活 tick vs ref 末位姿, 数值偏大属
   预期 (success 已由 fall 否决), 聚合时建议按 fall 过滤再看均值。
6. stand 切换瞬间 (停段↔跟随) FALCON 模式切换可能带来小幅过渡抖动,
   会计入 stop_settle/track_err — 这是其模式语义的真实代价, 不修。
