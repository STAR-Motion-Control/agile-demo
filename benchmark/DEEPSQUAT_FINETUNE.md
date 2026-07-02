# AGILE 深蹲微调 — 实验记录

目标：把 AGILE G1 `velocity-height` 下半身策略的蹲深从 ~0.49m 压到 **0.20m**（极限蹲）。
速度不重要，要深度。详见记忆 `agile-deepsquat-finetune`。

## 改动（cfg，8 处，`velocity_height_env_cfg.py`）
命令下限 0.4→0.20；终止 `illegal_ground_contact min_height` 0.5→0.25(+去膝监控)、`bad_orientation` 40°→55°、`knee_distance` 0.2→0.08；奖励 `track_base_height std` 0.2→0.35、`flat_orientation` −10→−2、`ankle_pos_limits` −10→−2。

## 训练
- 机器：4090 `isaac_sonic_env` docker(IsaacLab 2.2.1/Sim 5.0)，GPU0，num_envs=4096，seed=42
- task `Velocity-Height-G1-v0`，从头训（仓库无可续训 teacher），max_iter 30000(~50h)，每 250 iter 存 ckpt
- 路径：`/sdb/lizhe/g1_deepsquat/WBC-AGILE/logs/rsl_rl/velocity_height_g1_lower/2026-06-13_06-37-50_*/`

---

## 蹲深对照

### Baseline：原始 AGILE student（MuJoCo `bench_agile.py --test squat_limit`, 8 trials）
- **depth_floor = 0.492 m**（std 0.0018，零摔，track_sat_h 0.622，抱箱保持 100%）
- 复现了 2026-06-12 R4/R5 的 0.491；server: `/sda/lizhe/g1bench/deepsquat_cmp/baseline/`

### 当前 teacher checkpoint（model_3500 = iter ~3500/30000，~12% 进度；Isaac 高度扫描，32 env，每档 hold 4s）
| cmd_h | achieved_median | gap | resets |
|------:|----------------:|----:|-------:|
| 0.60 | 0.559 | −0.041 | 0 |
| 0.50 | 0.516 | +0.016 | 0 |
| 0.45 | 0.493 | +0.043 | 0 |
| 0.40 | 0.471 | +0.071 | 0 |
| 0.35 | 0.409 | +0.059 | 0 |
| 0.30 | 0.325 | +0.025 | 1 |
| 0.25 | **0.261** | +0.011 | 0 |
| 0.20 | 0.478 | +0.278 | 34（跌倒重置→回站立） |

- **depth_floor_tracked (≤5cm) = 0.25m，最深稳定达成 ~0.26m** → 比 baseline 0.49 深了 ~23cm（仅 12% 训练量）
- 0.20m 暂站不住（34 次跌倒）；预期随训练改善
- server: `/sdb/lizhe/g1_deepsquat/depth_eval_model_3500.json`，脚本 `eval_squat_depth.py`

> 注意口径差异：baseline 是 MuJoCo 连续下降扫描的 student；当前是 Isaac 定高保持扫描的 teacher。方向可比（最低稳定 pelvis 高），绝对值待蒸馏出 student 后用同一 `squat_limit` 复测。

## 最终结果（训练 2026-06-15 跑满 30000 iter，自行结束）
最终指标：mean reward 83.7、error_height 0.070、time_out 91%（不摔跑满）、终止率<1% —— 平均行为很稳、很会跟高度。

### 蹲深 vs 训练步数（Isaac 定高扫描，64 env，每档 hold 4s；"摔"=全部 reset）
| iter | cmd0.30 达成 | cmd0.25 达成 | cmd0.20 | 最深稳定 |
|-----:|---:|---:|---|---:|
| 3500  | 0.325 | **0.261**(gap+0.01) | 摔(34) | **~0.26** |
| 8000  | 0.421 | 0.378 | 摔(64) | ~0.38 |
| 15000 | 0.418 | 0.350 | 摔(63) | ~0.35 |
| 22000 | 0.457 | 0.432 | 摔(66) | ~0.43 |
| 29999 | 0.392 | 0.326 | 摔(64) | ~0.33 |
| baseline student | — | — | — | 0.49 |

JSON: `/sdb/lizhe/g1_deepsquat/depth_eval_model_*.json`

### 结论
1. **目标 0.20m 未达成**：所有 checkpoint 在 cmd 0.20 全部跌倒，没有一个能站住 0.20m。
2. **蹲深 iter~3500 最深(0.26m，跟住0.25)，之后越训越浅**。根因：随训练推进 `action_rate` 课程加重到 −1.0 + 熵退火 → 策略求稳求平滑；高度跟踪奖励(权重1.0)压不过 −200 摔倒惩罚，深蹲到极限不划算 → 收敛到"蹲一半保平安"。
3. 仍优于 baseline：最深稳定 0.26–0.33m vs 0.49m。

## 下一步（待定，见与用户讨论）
- **路线A 改奖励重训**(冲 0.20–0.25)：track_base_height 权重 1.0→4~6；削/去 `increase_action_rate_regularization` 课程；`bias_height_randomization=True` 多采深目标；考虑命令下限渐降课程；可从 model_29999 或 model_3500 续训(已有可续训 ckpt)省时。
- **路线B 蒸馏现有最佳**：取 model_3500(最深~0.26) 或 model_29999(最稳~0.33) → 改 `rsl_rl_ppo_cfg.py:90` teacher_path → 蒸馏 student → 导出 → `bench_agile squat_limit` 同口径复测，先拿可部署的"比baseline深"版。
- 注：0.20m 可能超出该奖励形式下 G1 稳定站立蹲的极限；最深已验证稳定档≈0.26m。

## ✅ 部署版结果：新 student vs baseline（同口径 bench_agile squat_limit, MuJoCo, 8 trials）
| 模型 | depth_floor | 零摔 | track_sat_h |
|---|---|---|---|
| baseline 原始 student | 0.492m | 8/8 | 0.622 |
| **新 student（model_3500 蒸馏）** | **0.246m** | **8/8** | 0.72 |

**蹲深 0.492→0.246m（砍半，深 24.6cm），零摔**。与 teacher Isaac 测的 ~0.26m 吻合（蒸馏忠实，behavior loss 0.0096）。代价：track_sat_h 0.72>0.622（下降跟踪精度略松，model_3500 中训特性）。

### 全面复测（同口径 8 trials）—— 有取舍
| 测试 | baseline | 新 student(3500) | 结论 |
|---|---|---|---|
| squat_limit | depth_floor 0.492 | **0.246** 零摔 | ✅ 砍半 |
| walk_speed | ~0.84 m/s(欠速) | **0.955 m/s** 零摔 | ✅ 不退反进 |
| squat_box(2kg) | ~零摔(稳) | success 0.625 **3/8摔**, hRMSE 0.093 | ⚠️ 退化 |

**权衡明确**：model_3500 最深(0.25)但中训"略毛"→带载(抱箱)反复蹲起鲁棒性不足(3/8摔);空载深蹲+走路都好。若带载蹲起是关键用途,候选改用 model_29999(蹲深~0.33但完全收敛、预期带载稳),或走 sit-down 路线拿"深且稳"。
- 蒸馏 student ckpt: `logs/rsl_rl/velocity_height_g1_lower_distillation/2026-06-15_03-09-30_*/model_2999.pt`
- 导出 student（drop-in, 2072958B 同原版）: `/sdb/lizhe/g1_deepsquat/student2999_export/policy.pt`(+onnx)
- bench: `--checkpoint <新.pt> --config <原student.yaml>`(obs/动作结构未改,yaml复用); 结果 `/sda/lizhe/g1bench/deepsquat_cmp/student2999/`
- 导出脚本 `/sdb/lizhe/g1_deepsquat/export_student_jit.py`(用蒸馏任务/cfg, exporter 自动取 .student 子网络)

## 全程蹲深曲线（含早期 checkpoint，2026-06-15 评估）
| iter | cmd0.25 达成 | 备注 |
|-----:|---:|---|
| 500/1000/1500 | 0.71/0.68/0.64 | 几乎不蹲（harness 期）|
| 2000/2500 | 0.48/0.33 | 逐渐变深 |
| 3000/3250 | 0.278/0.267 | 接近最深 |
| **3500** | **0.261** | **最深，且 0.20 跌倒最少** |
| 8000/15000/22000/29999 | 0.378/0.350/0.432/0.326 | 回退（越训越保守）|

**最佳下蹲 checkpoint = model_3500（~0.26m）**。3500 之前更浅（曲线是"渐深→3500见底→回退"）。
→ 已导出 jit `/hdd0/lizhe/g1_deepsquat/teacher_model_3500_jit.pt`，蒸馏成 student（task `Velocity-Height-G1-Distillation-Recurrent-v0`，teacher_path 指向它）。注意 3500 是中训 ckpt，蒸出的 student 深但可能更"毛"。

## 探索：sit-down/stand-up 任务（`HeightTracking-G1-v0`, stand_up/g1/height_tracking_env_cfg.py）
这是独立任务（不是 velocity-height）：用**跌倒状态数据集**训练从地面起身，命令 torso 高度 `(-0.5, 0.9)`。
**为何 sit-down 会"直接躺平"**：
1. `CommandsCfg.height.flat_ratio=0.2` —— 20% 指令显式命令"躺平"（最低档=躺地）。
2. 奖励 `torso_upright`(upright_orientation_after_standing) 只在高度 **>0.4m** 启用（`standing_height_threshold=0.4`）→ 低于 0.4 无竖直约束 → 折叠/躺。
3. 命令下限 -0.5 = 躯干贴地 = 躺。
**"坐而不躺"改法（可行）**：① `flat_ratio=0`，命令下限抬到坐姿高度(~0.2-0.3)；② `torso_upright` 的 `standing_height_threshold` 0.4→~0.15（坐姿仍强制躯干竖直）；③ 保留 `forward_pitch`(-10) / `severely_tilted` / `torso_slam` 等。
**亮点对照**：该任务高度跟踪是**多尺度、总权重 28**（rough4+medium8+fine16），远强于 velocity-height 的 1.0 —— 这印证了 velocity-height 蹲不深正是跟踪奖励太弱。若要稳坐到 0.2-0.3，这个任务（或把它的多尺度跟踪搬过来）是更优基座。注：repo 未发 HeightTracking-G1 的 checkpoint，需从头训。

### ✅ sit-not-lie 训练 + 评估（2026-06-15，model_4999，5000 iter 测试）
改 4 处（patch=/tmp/sitnotlie.patch）：命令下限 -0.5→0.55、flat_ratio 0.2→0、severely_tilted 135°→75°、torso_upright 门控 0.4→0.2。
**评估（num_envs=16，cmd_sit=0.55）**：pelvis 坐到中位 **0.313m**（最低 0.258）、躯干倾角中位 **28.4°**（最大 50.5）、**tilt>60° 比例 = 0.00（完全不躺）**、measured 跟住 0.524（差 2.6cm）。**"坐而不躺"成立**（仅 5000 iter）。
**视频**：`cc/experiments/results/deepsquat/sit_demo.mp4`（站→深坐→站×2，640×480/25fps/17.8s）。
渲染注意：Isaac headless RTX 在该训练容器死锁 → 录 qpos 轨迹（sit_record_qpos.py）→ host MuJoCo+EGL 回放渲染（mujoco_replay_render.py）。
**下一步**：在此基础上加速度跟踪（能走 + 能稳坐）；想坐更低→降命令下限（逼近失稳代价）。
