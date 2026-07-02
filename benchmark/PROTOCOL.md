# G1 Benchmark 标准实验流程（PROTOCOL）

> 适用范围：`cc/experiments/` 下的 G1 下半身控制器横评（AGILE / HOMIE / AMO / FALCON，+Psi0 上身）。
> 数据与结论记录在 `experiment.md`；本文档只管「怎么跑一轮新实验」的规范本身。
> 控制台：`console/index.html`（由 `console/make_manifest.py` 自动刷新）。

---

## 1. 规范版本史

| 版本 | 轮次 | 日期 | 内容 |
|---|---|---|---|
| **v1** | R1 | 2026-06-10 | 三测试统一规范：T1 walk_speed（vx 0→1.0 m/s ramp 2s + hold 10s，success := 无摔且 mean_vx≥0.9）、T2 squat_box（箱 2kg **weld@torso**，蹲至 base 0.45m ×20 循环）、T3 circle_pillar（vx=0.4 wz=±0.4 r=1m ×2 圈，25ccw+25cw）。每 (模型,测试) 50 trials，seed=trial，初始关节噪声 ±0.02rad，reset→2s 静置→测试段。统一摔倒判定：倾角 >0.9rad / base_z < 目标−0.2m / 非脚 body 触地。另附 speed_sweep（vx∈{0.4,0.6,0.8,1.0,1.2}×5）。 |
| **v2** | R2 | 2026-06-11 | **捧箱修正**：箱改为自由体 + **双腕 weld**（载荷经手臂 PD 传导，下垂/震荡为真实物理），T2 全部重测；JSONL 标 `box_mode: wrist_weld_v2`，新增 `box_kept`/`box_drop_time`。**FALCON（项目自训 model_10000.onnx）加入**，按统一规范补齐全套。Psi0 可行性侦察（裁决：VLA 在环只做演示，对比矩阵走轨迹回放 Plan B）。 |
| **v3** | R3 | 2026-06-11~12 | 新增 **T4 squat_sweep**（深扫 H∈{0.65…0.30}×5 @0.2m/s + 速扫 0.45m × rate∈{0.1,0.2,0.4,0.8}×5，全程捧箱；关键指标 achieved_depth / root_drift_hold / root_drift_total）、**T5 goto_ab**（A→B=(3,1,+90°)，NAV 粗导航 ≤0.6 → CAL 纯小指令 ≤0.1 纯 P 无死区补偿；success_fine := 5cm/5° 持续 1s；25 trials）、**T6 pipeline_abc**（捧箱右转 90°→走 2.5m 到 C=(0,−2.5,−90°)→校准→蹲 0.45 放箱→起身；15 trials）+ **Psi0 上身回放三测试**（walk/squat/circle_psi0，四下身吃同一上身扰动流；拾箱「磁性抓取」阈值 0.30m）。 |
| **v4** | R4(规划) | — | **squat_limit** 蹲深极限：0.05 m/s 缓降高度指令直至摔倒或深度饱和，记录 fall@cmd 高度 / floor 饱和高度 / drift 超 5cm 时的指令高度。**squat_place_psi0** 前伸放箱：Psi0 real_ep053 前伸轨迹回放放箱，记录 height_rmse / drift_place / land_dx。caption_videos.py 已预置两个测试的字幕分支。 |

---

## 2. 新增一个实验的标准步骤

1. **写 spec**：在 `experiment.md` 新开小节，定下测试定义（指令时序/场景物体）、指标、success 判据、trials 数、录像策略。四模型统一规范，模型特异适配（如 FALCON stand 标志、AMO target_yaw）单独注明。
2. **harness 实现**：给 `scripts/bench_{agile,homie,amo,falcon}.py` 各加 `--test <name>` 分支（复用现有指令调度 hook / 摔倒判定 / JSONL writer），本地 `python3 -m py_compile scripts/bench_*.py` 过编译。
3. **冒烟 1-2 trial**：scp 到对应机器，`--trials 1~2 --video all`（或 `none` 先出数据），目检视频帧（臂姿/箱位/场景物体），确认 JSONL 字段齐全、无 harness_error。
4. **全量 50**（goto 25 / pipeline 15 / sweep 60 按 spec），`--video policy`（= 前 5 个成功 + 全部失败 trial），JSONL+视频回传 `results/`（R3 之后入 `results/final/<model>/`）。
5. **烧字幕**：`python3 scripts/caption_videos.py --pairs VIDEODIR:JSONL --out results/videos_r2plus/<model>/`（新测试先在 `caption_lines()` 加分支；默认每组 ok×2+fail×2，`--all` 全烧）。
6. **刷新控制台**：`python3 console/make_manifest.py`（自动扫描 JSONL+视频生成 manifest.json，输出统计与断链检查），浏览器从 `cc/experiments/` 起 `python3 -m http.server` 访问 `/console/` 验证。
7. **记录**：结果表 + 结论写回 `experiment.md`（含失稳事件、与上轮对比、工程陷阱新增项）。

---

## 3. 四 harness 机器 / 环境 / 启动速查

> 完整细节见 `experiment.md` 附录A 与 `scripts/NOTES_{AGILE,HOMIE,AMO,FALCON}.md`。harness 全部单文件、headless、`--out-dir` 控制输出、即写即 flush。

| 模型 | 机器 | 工作区 | 环境 | 推理 | 备注 |
|---|---|---|---|---|---|
| **AGILE** | `ssh 4090` | `/sda/lizhe/g1bench/WBC-AGILE` + `unitree_mujoco` | `/sda/lizhe/miniforge3/envs/agile/bin/python`（torch、`pip -e WBC-AGILE`、numpy==1.26.4） | CPU 够（2MB LSTM） | 权重走 git-lfs；场景必须 `g1/scene_29dof.xml` |
| **HOMIE** | `ssh 4090` | `/sda/lizhe/g1bench/OpenHomie` | env `homie`（py3.10, mujoco 3.9, onnxruntime） | CPU | `deploy.onnx` 456→12；xml 用 27 关节版（仓库 g1.xml 是 44 关节版，不匹配） |
| **AMO** | `ssh wjzh@10.24.88.193`（5080；password omitted） | `~/AMO`（**cwd 必须是 ~/AMO**，模型相对路径） | conda `amo`（torch 2.11 cu128, mujoco 3.2.3） | **GPU 必须**（jit 硬编码 cuda:0） | Xid 154 后须 `sudo reboot`；**勿动该机 badmin/launchd 任何东西** |
| **FALCON** | `ssh 4090-lab` | `/hhd2/ljk/FALCON` | conda `fcreal`（py3.10, mujoco 3.9, ort 1.23） | CPU/ORT | ckpt `sim2real/models/falcon/g1_29dof_trained.onnx`（575→29）；`--repo` 或 `FALCON_REPO` 指仓库 |

启动命令模板（各机通用）：

```bash
# AMO（5080）
cd ~/AMO && MUJOCO_GL=egl python bench_amo.py --test <T> --trials 50 --out-dir bench_out --video policy

# HOMIE / AGILE（4090）
P=/sda/lizhe/miniforge3/envs/homie/bin/python
MUJOCO_GL=egl $P bench_homie.py --test <T> --trials 50 --out-dir results/homie/<T> --video policy
P=/sda/lizhe/miniforge3/envs/agile/bin/python
R="--agile-repo /sda/lizhe/g1bench/WBC-AGILE --mjcf /sda/lizhe/g1bench/unitree_mujoco/unitree_robots/g1/scene_29dof.xml --device cpu"
MUJOCO_GL=egl $P bench_agile.py --test <T> --trials 50 --out-dir results/agile/<T> --video policy $R

# FALCON（4090-lab）
PY=/hhd2/ljk/miniconda3/envs/fcreal/bin/python
MUJOCO_GL=egl $PY bench_falcon.py --test <T> --trials 50 --out-dir <RESULTS>/falcon/<T> --video policy
```

trials 约定：常规测试 50；speed_sweep 25（5 档×5）；squat_sweep 60（8 深×5 + 4 速×5）；goto_ab 25；pipeline_abc 15。

---

## 4. 数据布局约定

```
cc/experiments/
├── experiment.md                 # 实验记录（spec + 结果 + 结论）
├── PROTOCOL.md                   # 本文档
├── scripts/
│   ├── bench_{agile,homie,amo,falcon}.py   # 四 harness（单文件）
│   ├── NOTES_{AGILE,HOMIE,AMO,FALCON}.md   # 各 harness 运行笔记
│   └── caption_videos.py         # 视频烧字幕
├── results/
│   ├── <model>/*.jsonl           # R1（v1 三测试）；squat_box_v2.jsonl / falcon 全套 = R2
│   ├── <model>/summary.json      # 按 test 聚合（成功率、指标 mean±std、notes）
│   ├── final/<model>/<test>.jsonl  # R3：squat_sweep / goto_ab / pipeline_abc / *_psi0
│   └── videos_r2plus/<model>/*_cap.mp4   # R2+ 烧字幕视频（控制台视频库数据源）
└── console/
    ├── index.html                # 静态控制台（读同目录 manifest.json）
    ├── make_manifest.py          # 扫描 results/ 生成 manifest.json
    └── manifest.json             # 生成物，勿手编
```

- **JSONL 每 trial 一行**：`{framework, test, trial, seed, success, fall_time, fall_phase, harness_error, metrics{...}, video, box_mode?, upper_mode?}`。metrics 命名存在模型间差异（如 `height_rmse` vs `height_rmse_m`、`err_pos_cal` vs `cal_err_pos_m`），下游消费端（make_manifest / caption）一律做 fallback 链。
- **视频命名**：`{model}_{test}_t{NN}_{ok|fail}.mp4`（harness 产出，`--video policy` = 每测试前 5 个成功 + 全部失败）→ 烧字幕后 `..._cap.mp4` 入 `videos_r2plus/<model>/`。
- 视频规格：640×480 EGL 离屏，30fps（AMO 25fps，规范允许）；caption 默认每组 ok×2+fail×2，`--all` 全量。
- 控制台视频相对路径：`console/` 相对 `../results/videos_r2plus/<model>/<file>`，因此静态服务必须从 `cc/experiments/`（或更上层）起。

---

## 5. 已知陷阱清单

1. **MuJoCo ≥3.2.4 body 级 broadphase**：运行时改 geom 碰撞位（contype/conaffinity）必须**镜像更新 body 位**（body_contype 等），否则 broadphase 直接剔除 body 对，自由箱穿地（`bench_amo.py:1290` 有注释与处理）。
2. **EGL 离屏渲染**：所有渲染走 `MUJOCO_GL=egl`；5080 EGL 已验证可用。失败退路 `osmesa` 或 `--video none` 先出数据。无 DISPLAY/VNC 依赖。
3. **磁性抓取阈值 0.30m**：psi0 桌面拾箱判定 = 双手到箱面距离 ≤0.30m（从 0.12 放宽：回放轨迹是 AMO（带 torso pitch）录的，腿部策略弯腰深度不及——AGILE 差 10cm、HOMIE 无腰 pitch 差 4cm）。改阈值会改写四模型对比结论，动之前先读 experiment.md R3。
4. **AMO 帧 hook 内省**：harness 把 `env.viewer.render` 替换为每控制步 hook（指令调度/指标采集/摔倒检测/选帧），sys.modules 注入假 glfw/mujoco_viewer 实现 headless；**腰覆写 hack** = 策略跑完后、PD 力矩前覆写 pd_target（SIMPLE `g1_wholebody.py:271` 同款先例）。指令生效有 1 控制步（20ms）延迟。
5. **AMO 指令语义**：`commands[1]` 是**绝对目标航向角**不是 wz（绕圈需 `target_yaw += wz·dt` 积分下发）；`|vx|<0.1` 触发原地站立联锁强制 dyaw=0（小指令校准卡死的根因）；height 是 Δ（基准 0.75m，蹲 0.45 ⇒ cmd=−0.30）。
6. **FALCON stand 标志语义易反**：`stand=1 = 迈步/走`，`stand=0 = 站立/蹲`。walk/circle/NAV/CAL 用 1，settle 与 squat 用 0。
7. **AGILE 上游角速度 bug（已绕过）**：`agile/sim2mujoco/simulation.py` 无传感器回退把 body 系 qvel[3:6] 当世界系再旋转，大 yaw 初始化 0.5s 内必摔；harness 用 BodyFrameGyroShim 走传感器分支（启动日志须见 `Patched: base_ang_vel` 哨兵）。
8. **HOMIE 标定**：`ang_vel_scale=0.25` 才对（A/B 实测；训练 repo 的 0.5 与发布 onnx 不一致）；仓库 `g1.xml` 是 44 关节含手指版，与 456 obs 不匹配，必须 27 关节模型。
9. **放箱释放顺序**：蹲底释放 weld 时若直接打开箱↔机器人全碰撞，会卡掌间弹振；先关 weld、只开箱↔地碰撞落箱。
10. **FALCON weld relpose 写 `m.eq_data`（model 级）**：串行 OK，**不可并行复用同一 model 实例**；`left/right_rubber_hand` 是零质量 body，换 MJCF 需重核手 link 名与脚集合。
11. **碰撞覆盖不全（AMO）**：机器人带碰撞 geom 仅 pelvis_contour + 8 脚球，柱碰撞需 proximity 补充判据（base 到柱心 <0.35m），contact/proximity 双标志分开记录。
12. **机器卫生**：4090 根分区 97% 满——一切放 `/sda`；5080 出过 Xid 154（需 reboot 恢复 GPU），且该机**严禁动 badmin/launchd/模拟器相关任何东西**；4090 是共享机，跑前 `nvidia-smi` 挑卡。
