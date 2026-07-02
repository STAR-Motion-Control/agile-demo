# AGILE (NVIDIA WBC-AGILE) benchmark — 运行笔记

Harness: `bench_agile.py`（单文件, 统一实验规范 v1, MuJoCo sim2sim 路径, 零 Isaac 依赖）。
Policy: `unitree_g1_velocity_height_recurrent_student.pt`（TorchScript LSTM 2MB, 4D 命令
[vx, vy, wz, 绝对 pelvis 高度], normalizer 已内置）。

## 运行机器

- **首选: 4090 服务器（`ssh 4090`, 磁盘放 `/sda`）** — sm_89, 任意 torch cu12x 都行,
  总磁盘需求 < 5GB（AGILE repo + lfs 权重 + unitree_mujoco + conda env）。
- 备选: 5080（10.24.88.193）— 必须 torch>=2.7/cu128（Blackwell sm_120），**新建独立
  conda env, 不要动已就绪的 `amo` env**。
- CPU 就够（`--device cpu`，默认）：策略是 2MB LSTM；GPU 只对视频 EGL 渲染有帮助。

## 环境准备（在服务器上）

```bash
conda create -n agile_bench python=3.10 -y && conda activate agile_bench

# AGILE 仓库 + 权重（git-lfs!）
cd /sda/$USER   # 4090 上放大盘
git clone https://github.com/nvidia-isaac/WBC-AGILE.git
cd WBC-AGILE && git lfs pull        # 没有 lfs: conda install -c conda-forge git-lfs; git lfs install
# 验证权重是真文件不是 lfs 指针（应为 ~2.1MB zip）:
python -c "import zipfile; zipfile.ZipFile('agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt')"

pip install -e . --no-deps
pip install torch mujoco pyyaml numpy "imageio[ffmpeg]"
# 注: harness 在 sys.modules 里 stub 掉了 command_scheduler/data_logger 的导入链,
# 所以 pandas/matplotlib/seaborn/plotly/jinja2 都不需要。若还想跑官方
# scripts/sim2mujoco_eval.py 做交叉验证, 再补: pip install pandas matplotlib seaborn plotly jinja2

# 场景 MJCF（不在 AGILE 仓库内, 官方指定来源）
git clone https://github.com/unitreerobotics/unitree_mujoco.git ~/unitree_mujoco
ls ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml   # 必须用带地面的 scene 文件
```

把 `bench_agile.py` scp 到服务器任意位置即可（自包含, 通过 `--agile-repo` 指仓库）。

## 启动命令

```bash
export AGILE_REPO=/sda/$USER/WBC-AGILE
export UNITREE_MUJOCO_SCENE=~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
OUT=/sda/$USER/bench_out/agile

# 先冒烟（~1 分钟, 看 obs dim=80 / zeroed 2 recurrent buffers / 不摔即正常站立）
MUJOCO_GL=egl python bench_agile.py --test walk_speed --trials 2 --out-dir $OUT/smoke --video none

# 正式四个测试（可放 4 个 tmux 窗口并行, 互不共享状态）
MUJOCO_GL=egl python bench_agile.py --test walk_speed    --trials 50 --out-dir $OUT --video policy
MUJOCO_GL=egl python bench_agile.py --test squat_box     --trials 50 --out-dir $OUT --video policy
MUJOCO_GL=egl python bench_agile.py --test circle_pillar --trials 50 --out-dir $OUT --video policy
MUJOCO_GL=egl python bench_agile.py --test speed_sweep   --trials 25 --out-dir $OUT --video policy
```

输出: `$OUT/agile_<test>_results.jsonl`、`agile_<test>_summary.json`、`videos/agile_<test>_tNN_{ok|fail}.mp4`。
退出码恒为 0（单 trial 异常被捕获并记为 error 行, 不中断批次）。

预计耗时: mac M 系 CPU 冒烟实测 14s trial ≈ 0.5s 墙钟（无视频）、squat 102s trial ≈ 3s,
即仿真 ~30-35× 实时; 服务器 x86 单核可能慢 2-5×, 但全套 4 测试×50 trial 含视频也应在
**1 小时以内**。瓶颈是视频渲染（policy 模式所有 trial 都渲染, 见下"关键实现决策"#8）。
仍嫌慢: squat_box 拆两进程 `--trials 25 --seed-offset 0 / 25`（--out-dir 分开, 事后合并 JSONL）。

## 已在本机(macOS, CPU)完成的冒烟验证 (2026-06-10)

用临时 venv (mujoco 3.9.0 + torch 2.1.1) + 在线 clone 的 unitree_mujoco 实跑过:

- speed_sweep 5 trial (0.4-1.2 m/s 各一): 全程无摔倒, mean_vx = 0.36/0.54/0.71/0.84/1.01
  → 超出训练域后跟踪逐渐欠速但不失稳; max_stable_speed(按 0.9×target 判) = 0.6。
- walk_speed: 1.0 m/s 实测 ~0.837 m/s, 不摔但 <0.9 → FAIL(欠速), 这就是预期记录方式。
- squat_box 2 trial: 20/20 循环, height RMSE ≈ 0.075 m, 无摔倒（箱子+抱臂位生效）。
- circle_pillar CCW+CW 各 1: 径向误差 mean ≈ 0.07 m, 无碰柱无摔倒。
- 视频: 640×480@25fps mp4 正常产出（mac 上无需 EGL; 服务器用 MUJOCO_GL=egl）。
- 性能: mac CPU 上 14s trial ≈ 0.5s 墙钟(无视频)/6-7s(带视频); squat 102s trial ≈ 3s。
  服务器全套 50 trial×4 测试预计 **远低于** 早先 1-3 小时估计, 带视频也在几十分钟级。

## ⚠️ 发现并绕过的 AGILE 上游 bug（重要）

`agile/sim2mujoco/simulation.py:419-421` 的无传感器回退把自由关节 `qvel[3:6]` 当
**世界系**角速度再旋转到 base 系 — 实际上 MuJoCo 自由关节 `qvel[3:6]` 本来就是
**body 系**（已用 `mj_objectVelocity` 实证）。unitree G1 MJCF 没有名为
`angular-velocity` 的传感器, 所以这条错误回退路径必然生效: yaw≈0 时双重旋转近似
恒等(官方 demo 从 yaw=0 起步所以没暴露), 但随机初始 yaw 大时 base_ang_vel x/y 分量
被搞反 → 正反馈 → **0.5s 内必摔**（冒烟时 5 个 seed 摔 4 个, 全是大 yaw）。
harness 的 `BodyFrameGyroShim` 把 `qvel[3:6]` 经由"传感器分支"(原样使用, 数学上精确)
喂给框架, 修复后同一批 seed 全部稳定。**解读 AGILE 官方评测结果时也要记住这一点**;
该 bug 同样影响官方 sim2mujoco 的转向场景。

## 关键实现决策（对结果解读重要）

1. **T1 的 1.0 m/s 是明确 OOD**: AGILE 两个 G1 任务训练 vx 域都是 [-0.5, 0.5] m/s。
   CommandManager 默认硬 clamp ±0.5, harness 对 walk_speed/speed_sweep 把
   `mgr.linear_x_range` 改宽到 ±2.0, 保证策略真的看到 1.0/1.2。冒烟实测 1.0 指令
   下实际 ~0.84 m/s（欠速但不失稳）, **如实记录, 这本身就是 benchmark 结果**,
   T1 按 mean_vx≥0.9 判则大概率 success_rate=0。
2. **height 语义**: 绝对 pelvis 高度。站立 0.72, T2 蹲直接下发 0.45（在训练域
   [0.4,0.72] 内）。T2 全程 vx=vy=wz=0, 与训练语义一致（蹲只在零速度时采样）。
3. **LSTM 状态**: 导出 .pt 把 hidden/cell 存成内部 buffer, 框架的 reset() 是空操作。
   harness 每 trial 重新 `torch.jit.load` + 把名字含 hidden/cell 的 buffer 清零;
   第一个 trial 会打印 `zeroed N recurrent state buffer(s)`, **N 应为 2**。
4. **上身/抱箱**: 部署侧上身 17 关节 PD 目标本来恒为 0（默认位）。T2 在
   `act_processor.process` 之后覆写臂关节目标为抱箱位（肩 pitch 0.3, 肩 roll ±0.2,
   肘 1.0, 腕 0）, 开局 0.5s 线性过渡; 箱子(2kg, 0.35×0.25×0.25) 注入 XML 固连
   torso_link 前方 0.25m, contype=0 不参与碰撞。腕部 kp 只有 4, 形变大属正常。
5. **场景注入**: 纯 XML 文本注入, 变体文件写在原 scene 同目录（相对 meshdir/include
   才能继续解析）: `scene_29dof__bench_circle.xml` / `__bench_box.xml`（及
   `g1_29dof__bench_box.xml`）。会在 unitree_mujoco 目录里留下这几个生成文件。
6. **柱子几何**: 圆心 (0,1), 机器人统一从原点出发 — CCW yaw=0/wz=+0.4, CW yaw=π/wz=−0.4,
   两方向共用同一场景。
7. **摔倒判定（统一规范）**: 倾角>0.9rad（由 base quat 算 projected gravity）或
   base_z < 当前高度指令−0.2m 或 非脚(ankle_roll 之外) geom 触地。比 AGILE 训练侧
   的终止条件宽。柱碰撞只记录不终止, 摔倒才终止。
8. **视频 policy 模式渲染所有 trial**（流式写临时 mp4, 结束后按"前5成功+全部失败"
   决定保留/删除）。这是规范里"不录的 trial 跳过渲染"与"所有失败 trial 必须有视频"
   的取舍 — 失败只有事后才知道。嫌慢用 `--video none`。
9. dt 全部来自导出 YAML（physics 0.005 / decimation 4 → 50Hz）, 覆盖 MJCF 自带
   timestep; sim2mujoco 初始 root 高度硬编码 0.76、frictionloss=0.1/damping=0/
   per-joint armature 覆写 — 这些是官方 sim2sim 校准, 保留不动。`--pd-scale` 保持
   1.0（官方稳定性补丁建议 0.3, 但 benchmark 求保真; 不稳如实记录）。

## v4 新增测试 (squat_limit / squat_place_psi0)

- **squat_limit 放开了 CommandManager 的 height clip**: 管理器默认 clamp height 到
  [0.4, 0.72]（squat_sweep 已放宽到 (0.2, 0.8)）; squat_limit 进一步放宽到
  **(0.05, 0.8)**, 使 0.72→0.10 的 0.05 m/s 连续下降指令**如实下发、全程不 clip**
  （规范 v4: 策略饱和是测量结果, 不在命令层掩盖）。启动日志须有
  `Patched: manager height_range widened to (0.05, 0.8)`。
  本机冒烟(2026-06-12, 同 6-10 venv): 不摔、depth_floor≈0.49 m（饱和不再跟随）,
  track_sat_h≈0.64, drift5_h≈0.42, 曲线 154 点@10Hz 全部入 JSONL。
  建议 `--video all`（每个 trial 都是有效标定样本）。
- **squat_place_psi0 用 v2 2kg 箱（非 psi0 cracker 箱）**, 上身走既有 psi0 回放通路
  但**单次播放不 loop**; t_place=argmin(height_cmd)（加载 npz 时算, 进 provenance
  `upper_replay_t_place_s`）时刻调 `release_box()`（与 pipeline_abc 共用的
  bit-2+body 位释放, 已抽成共享函数）; box_land_dx 以释放瞬间足前缘投影为基准
  （geom 中心+x 半长近似, 误差 ~1-2 cm）。分段 RMSE 的 hold 段 = replay 高度进入
  h_min+3cm 带的区间, descent/rise 为带前/带后。
  本机用合成 ep053 形状 npz 冒烟: 释放→落箱→起身→站稳全链路 OK,
  box_land_dx 前向为正已验证（真 real_ep053 npz 在 4090:/sda/lizhe/g1bench/）。

## v5 新增 (squat_pick_ground / vln_follow / 交互渲染 flags)

- **T9 squat_pick_ground 前伸捧位角度（规范要求记录在 NOTES）**: ARM_REACH_POSE =
  shoulder_pitch **-0.45**(前倾), shoulder_roll **±0.10**, shoulder_yaw 0,
  elbow **+1.40**(屈), 三个 wrist 全 0。来源: 对 G1 29dof 链做离线 numpy FK
  （repos/AMO_5080/g1.xml, 躯干竖直假设）网格搜索, 目标=腕尽量低且左右
  跨在 0.35 m 箱宽两侧。注意 **G1 肘关节零位的前臂是水平前指的**（不是下垂!）,
  所以"低位前伸"靠 elbow≈+1.4 把前臂折向下, 而非伸直手臂。该位姿下:
  wrist_yaw 原点 ≈ (pelvis 前 0.20, ±0.17, pelvis 高 -0.024),
  palm(+0.10m 局部 x) ≈ (0.26, ±0.18, -0.105)。
- **AGILE 预期结论（如实跑）**: 蹲深饱和 ~0.50-0.53 (R4/squat_limit 标定),
  上述位姿的腕-箱面距离离线计算 ≈ **0.26-0.29 m, 恰好贴着 0.30 m 磁吸阈值**
  ——策略蹲底时若躯干前倾会再近一点。所以 AGILE 的 pick_success 是
  边缘情形, 跑出来什么就是什么（这正是 T9 要标定的结论）。失败模式预期为
  pick_failed 而非摔倒; 起身段空手/带 2kg 都在训练域内。
- T9 机制: 箱=v2 同尺寸 2 kg (0.25×0.35×0.25), **compile 时即 bit-2 碰撞**
  (PICK_BOX_XML contype=2)+地面 OR bit2(与 psi0 同方案, 已抽成
  `enable_floor_bit2()` 共享), 立放于初始朝向正前 0.45 m; 高度钳放宽到
  (0.2, 0.8) 使 0.25 指令如实下发（日志须有对应 `Patched:` 行）;
  磁吸=`activate_welds_at_current_pose`(与 squat_box_psi0 同机制, 箱不瞬移,
  原位 weld——抓住后箱可能悬在掌下方一段距离, 这是"磁吸"语义的一部分)。
  手臂在 settle 后用 1 s 从实测 qpos 线性混合到捧位（避免 PD 目标跳变）。
- **T10 vln_follow**: `--tapes` 必填（data/vln_tapes.json, 4 个 harness 共用
  同一份保证公平; 本机已验证 10 条 tape: 30 s、各含 2 段 1-2 s 全停、
  小 vx 段充足）。AGILE 的 wz **原生就是 yaw-rate 指令**, 无 AMO 的
  target_yaw 积分语义差（provenance.vln_wz_semantics 已注明）。
  CommandManager 三个钳全部放宽 (vx ±2 / vy ±1 / wz ±2)——tape 本身在训练域内,
  放宽只为"如实下发"承诺。stop_settle 跳过每段全停头 0.5 s（扣除减速过程）;
  small_cmd_response = |vx|∈[0.05,0.15] 段内 实际前向速度/指令 之比的均值;
  mean_track_err 按 1 Hz ref 采样点取最近 50 Hz 样本。
  success := 不摔 ∧ final_pos_err≤0.30 m ∧ final_yaw_err≤15°。
- **交互渲染 flags（控制台后端）**: `--custom-height/--custom-rate` 把
  squat_sweep 网格换成单点(sweep_kind="custom", `--trials`=该点次数);
  `--custom-vx` 换 walk_speed/speed_sweep 目标速度且判据变 ≥0.9×V
  (CUSTOM_VX_REL_THR); `--custom-vx`+`--custom-wz` 换 circle_pillar 的
  0.4/0.4, **半径=V/W, 圆心/柱子位置/radial_err 基准全部跟随新半径**
  (PILLAR_GEOM_XML 在建场景前重绑)。自定义值超出管理器默认钳时
  (vx>0.5 / wz>1.0 / height<0.4) 会在 build 后补放宽并打印 `Patched:` 行。
  全部 flag 不给时路径与 v4 完全一致（零回归）。
- v5 本机验证: py_compile 通过; vln tape 纯函数(ZOH/全停段/track-err 索引)
  对真实 tapes.json 全量断言通过; T9/T10 完整 rollout 需服务器 venv
  (本机已无 mujoco 环境, 见下方 TODO)。

## 已知风险点 / TODO(verify-on-server) 汇总

- [ ] **v5 squat_pick_ground 冒烟**: 跑 2 trial 看 (a) 启动日志有
      height_range (0.2, 0.8) 与 floor bit-2 两行 Patched/enabled;
      (b) settle 段箱静置地面不穿地; (c) reach 段手臂 1 s 内到位不打摆;
      (d) grasp_dist_l/r 记录值 ≈ 0.25-0.31（离线 FK 预测), pick 成败如实;
      (e) pick_failed 路径也会起身收尾。
- [ ] **v5 vln_follow 冒烟**: 跑 2 trial 看 final_pos_err 量级（无摔时
      预期 <1 m）, stop_settle ≈ 0.01-0.05 m/s, small_cmd_response 0.5-1.1。

- [ ] **EGL 渲染**: `MUJOCO_GL=egl` 在服务器 GPU/driver 上首次跑通需验证;
      失败可退 `MUJOCO_GL=osmesa`（慢, 需 osmesa 系统包）或 `--video none`。
      offscreen framebuffer 用 MuJoCo 默认 640×480, 恰好等于输出分辨率。
      （mac 冒烟时无需 EGL 即出片, 服务器路径未实测。）
- [ ] **recurrent buffer 清零数 == 2**: 看首 trial 日志
      `zeroed 2 recurrent state buffer(s)`。mac 冒烟已确认为 2; 若服务器上为 0,
      去查 TorchScript 导出的 buffer 命名（exporter.py:98-131）, 否则 LSTM 状态泄漏。
- [ ] **场景注入锚点**: 假定 scene_29dof.xml 有字面 `<worldbody>` 和
      `<include file="g1_29dof.xml"/>`, g1_29dof.xml 有 `<body name="torso_link" ...>`
      （2026-06-10 已对 unitree_mujoco@main 在线核对并实跑通过, 但上游可能改）。
      注入失败会直接 SystemExit, 不会静默错。
- [ ] **gyro shim 必须生效**: 启动日志须有
      `Patched: base_ang_vel sourced from qvel[3:6]`。若上游 AGILE 更新了
      simulation.py 的传感器命名/回退逻辑, 重新核对 shim 是否还被走到
      （MJCF 增加了名为 angular-velocity 的传感器时, shim 仍会覆盖, 语义不变）。
- [ ] **`linear-velocity` 传感器缺失 Warning 属预期**: base_lin_vel 不在学生观测里,
      回退只影响日志; harness 自己的前向速度指标直接读 qvel[:3]（已实证为世界系）。
- [ ] **mujoco 版本**: 框架要求 mujoco>=3.3.x; 冒烟用 3.9.0 通过。
      `data.contact.geom1` 向量化访问、`mujoco.Renderer` 需 3.x。pip 装最新即可。
- [ ] **torch 版本**: 冒烟用 2.1.1 能加载该 TorchScript; 服务器装新版 torch 没问题
      (TorchScript 向后兼容)。5080 必须 >=2.7/cu128。
- [ ] **T2 蹲到 0.45 + base_z 摔倒线 0.25**: ramp 阶段阈值跟随指令瞬时值, 若策略
      高度跟踪滞后 >0.2m 会被判摔倒 — 统一规范定义, 三框架一致, 不调。冒烟实测
      height RMSE ~0.075m, 未触发。
- [ ] **walk 阶段 RMSE 含 ramp 段**: vx_rmse 在整个 12s 指令段上算（ramp+hold）,
      mean_vx 只看最后 8s。与其他两个框架的 harness 保持同一定义。
- [ ] **首次 import 打印一堆框架日志**（joint mapping/警告）属正常; obs dim 80。

## 结果文件 schema

JSONL 每行: `{framework:"AGILE", test, trial, seed, success, fall_time(含2s静置的
trial 时钟,秒), fall_phase, metrics{...}, video}`;
- walk/sweep metrics: vx_target, mean_vx_last8s, vx_rmse, max_tilt
- squat_box: cycles_completed, height_rmse, max_tilt
- circle: direction, radial_err_mean/max, closure_pos/yaw_err_lap1/2,
  pillar_collision, collision_time, max_tilt

summary.json: success_rate / n_falls / 各指标 mean±std / provenance（commit、版本、
路径）; speed_sweep 另有 per_speed 表 + max_stable_speed（success_rate≥0.8 的最大档）。
