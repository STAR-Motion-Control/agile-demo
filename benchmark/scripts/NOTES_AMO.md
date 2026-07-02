# AMO Benchmark Harness 运行笔记 (bench_amo.py)

统一实验规范 v1 下的 AMO (G1 23-DoF, 下身 RL + AMO adapter, 双臂直接 PD) benchmark。
harness 完整复用官方 `play_amo.HumanoidEnv`(obs/action 流水线与源码一字不差), 只替换
GUI viewer (sys.modules 注入假 glfw/mujoco_viewer, 不 import 任何真 GUI 库) 和场景 XML。

## 运行机器

- **5080 服务器**: `ssh wjzh@10.24.88.193` (RTX 5080, sm_120)
- AMO 已部署在 `~/AMO/` (amo_jit.pt / adapter_jit.pt / adapter_norm_stats.pt / g1.xml /
  play_amo.py / smoke_test.py 均就位, smoke_test 此前 PASS)
- **GPU 必须健康**: JIT trace 硬编码 cuda:0, map_location 救不了。Xid-154 故障后必须
  `sudo reboot` (参考 `~/AMO/run_amo.sh` 的 pre-flight)。harness 启动时自查, CUDA 不可用
  退出码 2。
- 不需要 VNC/DISPLAY: 全程 headless, 录像走 `MUJOCO_GL=egl` 离屏渲染。

## 环境准备命令

```bash
ssh wjzh@10.24.88.193
source ~/miniconda3/etc/profile.d/conda.sh && conda activate amo   # 已有, 勿重建
pip install "imageio[ffmpeg]"        # 唯一新增依赖(写 mp4); --video none 时可跳过
# 从 Mac 上传 harness:
#   scp /Users/lizhe/Project/sim2real/cc/experiments/scripts/bench_amo.py \
#       wjzh@10.24.88.193:~/AMO/
# 快速验环境(可选): cd ~/AMO && python smoke_test.py   # 应 PASS
```

## 启动命令

```bash
cd ~/AMO    # cwd 必须是 ~/AMO: 模型/ckpt/g1.xml 全是相对路径裸文件名
# 建议放 nohup/tmux 里跑(T2 全程 ~105s 仿真 x50 trial, 最久):
MUJOCO_GL=egl python bench_amo.py --test walk_speed    --trials 50 --out-dir bench_out --video policy
MUJOCO_GL=egl python bench_amo.py --test speed_sweep   --trials 25 --out-dir bench_out --video none
MUJOCO_GL=egl python bench_amo.py --test squat_box     --trials 50 --out-dir bench_out --video policy
MUJOCO_GL=egl python bench_amo.py --test circle_pillar --trials 50 --out-dir bench_out --video policy
```

- 输出: `bench_out/amo_<test>.jsonl` (逐 trial, 即写即 flush, 中断也保留已完成部分),
  `bench_out/summary.json` (按 test 合并更新), `bench_out/amo_<test>_tNN_{ok|fail}.mp4`。
- 录像策略: `policy` = 每个 test 前 5 个成功 trial + 所有失败 trial。仿真期间只存 qpos
  快照(零渲染开销), trial 结束后只对需要保存的 trial 做状态回放渲染 —— 满足
  "不录的 trial 完全跳过渲染"。25fps(50Hz 控制步每 2 步取 1 帧), 640x480。
- 先跑一个小冒烟: `--test walk_speed --trials 2 --video all` 确认全链路+视频 OK 再上量。
- 单 trial 仿真时长: T1/sweep 14s, T2 105s, T3 ~33.9s; 500Hz 子步 + 50Hz GPU 推理,
  预计实时比 0.5~2x, T2 x50 全程可能 2~4 小时。

## 实现要点(与源码对齐)

- 指令注入: 写 `env.viewer.commands` (8 维)。**[1] 是绝对目标航向角不是 wz, [2] 才是 vy**;
  T3 用 `yaw_cmd = dir*0.4*(t-2s)` 解析积分(恒定 wz 下精确), T1/T2 全程钉住初始 yaw。
- T3 保持 vx=0.4 ≥ 0.1, 避开 `_in_place_stand_flag` 把 dyaw 强制清零的原地站立联锁。
- height 是 delta: 蹲 0.45m ⇒ commands[3] = −0.30; 摔倒判据的"当前高度目标"= 0.75+cmd[3]。
- 抱箱臂: 不开 commands[7], 直接写 `env.prev_arm_action/arm_action/arm_blend`,
  靠 play_amo.py:317-319 的 blend 通路 2s 过渡(toggle_arm 恒 False 不会被覆盖);
  时序 = 2s 静置 → 3s 摆臂 → 20x5s 蹲循环 (总 105s)。
- 箱子: 2.0kg 0.35x0.25x0.25m, torso_link 子 body 纯质量惯量, contype=0 不碰撞(规范:
  不模拟抓取接触)。柱子: cylinder r=0.15 h=1.2 静态 geom, condim=3。两者均为 g1.xml
  文本锚点注入 → 临时 xml 写进 cwd 再 from_xml_path(meshdir 相对 xml 文件解析)。
- 每控制步 hook = 替换 `env.viewer.render`(官方每控制步调一次), 在里面做指令调度/
  指标采集/摔倒检测/选帧; 提前终止用异常打断 `env.run()`。
- seed=trial_index, `np.random.seed(seed)` 全局(关节噪声 ±0.02rad、随机初始 yaw,
  circle 测试 yaw 固定 0); xml 传感器噪声未启用(option 无 sensornoise flag), 仿真确定。

## 已知风险点 / TODO(verify-on-server) 汇总

1. **抱箱臂角度与箱子位置未目检** — `HUG_ARM_POSE = L[-0.2,0.2,0.1,1.0] R[-0.2,-0.2,-0.1,1.0]`,
   箱子 pos="0.25 0 0.18"(torso_link 系)。上线前跑
   `--test squat_box --trials 1 --video all` 看一帧: 臂应环抱箱子且不穿箱过深。
2. **碰撞覆盖不全(框架硬限制)** — 机器人带碰撞 geom 只有 pelvis_contour + 8 个脚部小球;
   臂/膝/躯干是纯视觉 mesh。柱子碰撞检测会漏掉手臂擦碰; 已加径向距离补充判据
   (base 到柱心 < 0.35m 计 proximity collision), JSONL 里 contact/proximity 两个标志分开
   记录, success 用两者之 OR。"非脚触地"摔倒判据同样只覆盖 pelvis_contour, summary 注明。
3. **T1 的 vx=1.0 / sweep 的 1.2 超出训练域 [-0.5,0.5]** — 按规范不 clip 直接下发,
   预期跟踪率低/可能摔, 这本身是结果, 不是 bug。
4. **T2 起身段的高度摔倒判据可能误报** — 目标 0.45→0.75 以 0.2m/s 上升, 若机器人滞后
   >0.2m 会被判 base_height 摔倒(统一规范如此)。fall_reason 单独记录, 事后可甄别。
5. **临时 xml 的 mesh 路径解析** — 修改后的 xml 写到 cwd(~/AMO) 临时文件再 from_xml_path,
   meshdir="meshes" 相对 xml 所在目录 = ~/AMO/meshes, 应当 OK; 若报 mesh 找不到,
   首个 trial 即失败, 一眼可见。
6. **EGL 可用性** — 服务器没验证过 MUJOCO_GL=egl(此前都是 VNC GLX)。若 Renderer 报错,
   退路 `MUJOCO_GL=osmesa`(需 libosmesa)或 `--video none` 先出数据。
   渲染缓冲: g1.xml 无 offwidth/offheight, MuJoCo 默认 640x480, 恰好匹配。
7. **imageio[ffmpeg] 未装** — 仅影响视频, harness 会打 WARN 跳过视频继续跑数据。
8. **指令生效有 1 控制步(20ms)延迟** — hook 在 get_observation 之后运行, 第 k 步写的指令
   第 k+1 步被读。恒定 20ms 平移, 对 14~105s 的指标无影响, 但三框架对比时注意一致性。
9. **每 trial 重建 env** — adapter/norm_stats 每 trial 重新加载(干净状态, 防跨 trial 泄漏),
   单次 <1s; MjModel 按场景变体缓存只读一次 mesh。
10. **视频 25fps 而非 30fps** — 50Hz 控制步取偶数分频(每 2 步 1 帧), 规范允许
    "按 dt 换算"; 三框架统一用 25fps 即可对比。
11. **harness_error 兜底** — 单 trial 抛异常不会中断整批, JSONL 记
    `fall_reason=harness_error` + repr(exc), success=False。
12. **T3 闭环误差参考点** — 以测试段第一个控制步的实际 base 位姿为起点(非理论圆),
    指令 yaw 积分跨过 2π/4π 时刻采样; 机器人若严重偏航, 闭环误差和径向误差都会变大,
    属于预期测量行为。

## v4 新增测试 (T7 squat_limit / T8 squat_place_psi0)

```bash
MUJOCO_GL=egl python bench_amo.py --test squat_limit --trials 10 --out-dir bench_out --video all
MUJOCO_GL=egl python bench_amo.py --test squat_place_psi0 --trials 25 --out-dir bench_out --video policy \
    --upper-replay ~/AMO/psi0_replay/upper_replay_amo23_real_ep053.npz
```

- **T7 squat_limit**: 捧 2kg v2 箱, 2s 静置 → height 指令 0.75→0.10m 以 0.05m/s 匀速连续
  下降 → 0.10 指令保持 3s, 不起身。**commands[3] 下发到 −0.65 不 clip**: 经核查
  play_amo 命令层本来就没有 clip(commands[3] 原样进 adapter 输入和 obs_demo,
  play_amo.py:248/282), harness 在 squat_limit 通路上也不加任何 clip —— 即"放开 clip"
  实为"确认无 clip 可放", 数据按规范原样下发。10Hz `descent_curve=[[h_cmd, base_z,
  drift_xy],...]`; depth_floor 用稳定态过滤(tilt<0.5rad 且 dz/dt>−0.25m/s, 排除塌落段;
  指令下降速率仅 0.05m/s, 下沉快 5 倍即视为塌落)。success=无摔; 预期多数 trial
  "不摔但饱和"(track_sat_h 记录饱和点指令高度)。
- **T8 squat_place_psi0**: 2kg v2 箱 t=0 焊在 ep053 第 0 帧上身姿态的双手间(不是
  cracker 箱), 复用 psi0 回放通路(squat 模式单次播放 + 腰 3 关节 pd_target 覆写 hack),
  下身 commands[3] 跟随 ep053 height_cmd(沿用 psi0 高度 clip [0.45,0.75]; ep053 顶端
  0.77 被截到 0.75, 谷底 0.46 在域内, clip 实际几乎不起作用)。t_place=argmin(height_cmd)
  (npz 加载时算好)时刻用 pipeline_abc 的 bit-2+body 位释放机制放箱。box_land_dx = 落箱点
  相对释放时刻"脚前缘"(可碰撞 ankle_roll 球心航向投影最大值+球半径)的前向距离, 正值=
  放在身前。box_place_ok 比 T6 多一条 `box_z_end < 0.30m`(落地判据, 防止焊接释放静默
  失败时箱仍挂胸前被误判成功)。success = 无摔 ∧ box_place_ok。
- T7/T8 风险(verify-on-server): ① T8 的 ep053 第 0 帧双手间距未目检, 0.35m 宽箱可能与
  手视觉穿插(焊接不受影响, 只影响观感) —— 先跑 `--trials 1 --video all` 看一帧;
  ② T7 指令远超训练域, weld-vs-PD 数值发散会被记为 harness_error(非摔倒), 关注 JSONL
  中 harness_error 占比; ③ T8 真实 ep053 的 height_cmd 若非平滑 V 形(多谷), hold 段按
  "min+0.02m 包络窗"取首末样本, 分段 RMSE 含义需结合曲线核对。

## v5 新增测试 (T9 squat_pick_ground / T10 vln_follow)

```bash
MUJOCO_GL=egl python bench_amo.py --test squat_pick_ground --trials 15 --out-dir bench_out --video policy
MUJOCO_GL=egl python bench_amo.py --test vln_follow --trials 10 --out-dir bench_out --video policy \
    --tapes ~/AMO/vln_tapes.json     # make_vln_tapes.py 预生成, 四 harness 共用同一文件
```

- **T9 squat_pick_ground**: 2kg 箱(0.35×0.25×0.25, 同 v2 carry 箱尺寸/质量)立放地面、
  机器人正前方 0.45m(随 trial 随机朝向旋转), 碰撞 bit2 方案(地面 conaffinity=15 含 bit2,
  机器人 bit1 → 箱与地碰、与机器人永不碰; 编译期位即正确, 无需运行时改位)。空手站立
  (keyframe 下垂臂)2s 静置 → 双臂 blend 到**低位前伸捧位 PICK_ARM_POSE =
  [shoulder_pitch −0.9, shoulder_roll ±0.15, shoulder_yaw 0, elbow 0.5] (L/R 对称,
  负 pitch=前摆, 同 HUG 姿约定; 目标: 腕尽量低且分列箱两侧 ±0.175m 外)** → height 指令
  0.75→0.25m @0.2m/s (commands[3]=−0.50 **不 clip 原样下发**, AMO 按 R4 标定蹲深极限
  ~0.345m 饱和) → 蹲底 hold: 双腕距箱面均 <0.30m 即磁性抓取(welds 原位锚定+激活,
  squat_box_psi0 机制, 箱不动) → 抓到再 hold 0.5s → 带 2kg 负载起身 → 站 1s;
  蹲底 5s 内未抓到 → pick_failed 仍起身。success = pick ∧ 无摔 ∧ stand_ok。
- **T10 vln_follow**: tapes.json 零阶保持下发(trial i 用 tape i%n); **AMO 无 wz 通道,
  wz 积分进绝对航向 commands[1]**(同 T3/T5 实现; 航向跟踪 vs 角速度跟踪的语义差按惯例
  注明不绕过), vy→commands[2]。tape 坐标系锚定在 tape 起始时刻机器人实际位姿(消除静置
  漂移偏置), 航向积分也从实际 yaw 起步。末位姿误差取最后一个 tape 采样(排除终段站立)。
  success = 无摔 ∧ final_pos_err≤0.30m ∧ final_yaw_err≤15°。
- T9/T10 风险(verify-on-server): ① PICK_ARM_POSE 是 FK 直觉值未在服务器目检 —— 先跑
  `--test squat_pick_ground --trials 1 --video all` 看蹲底帧, 确认腕低于箱顶且分列两侧,
  必要时只调这一个常量; ② AMO 蹲深饱和 ~0.345m, 腕能否够到 0.30m 抓取窗取决于前伸幅度,
  pick 失败率本身就是结论(AGILE 预期整体够不着, 同理如实记录); ③ T9 起身段 base_height
  摔倒判据与 T2/T4 同款(滞后指令 >0.2m 判摔), 负载起身失败会记为 fall_phase=rise, 这是
  要测的信号; ④ T10 小指令段 |vx|∈[0.05,0.1) 落在 play_amo 原地站立联锁死区内
  (vln_yaw_locked_frac 记录暴露度), small_cmd_response 偏低是测量结果不是 bug;
  ⑤ stop_settle 用 tape 全停段半开窗 [s0,s1) 测站立残移。

## 与其他框架对齐时注意

- AMO height = 0.75 + delta (HOMIE 是 [-0.5,0] 相对量, AGILE 是绝对 pelvis 高度),
  本 harness 已换算为绝对高度做 RMSE/摔倒判据。
- AMO 没有 wz 指令通道, T3 的角速度是用绝对航向角积分模拟的 —— 与有原生 wz 的框架
  在控制语义上略有差别(航向跟踪 vs 角速度跟踪), 报告时注明。
- 双臂不经策略(脚本直接 PD), T2 的"上身抱箱"对 AMO 而言是开环臂目标 + adapter 补偿,
  与上身也在策略内的框架不是同一机制, 对比时注明。
