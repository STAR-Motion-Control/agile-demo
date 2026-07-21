# 三项测试指标 — decoupled-WBC vs AGILE-29dof

三个指标(真机验收标准):
1. **最快速度** 达到 1 m/s
2. **抱箱下蹲** 20 次,平稳 ≥18 次
3. **自然绕障**(半径按输入 0.4/0.6/0.8/1.0 m,固定朝向半圆绕行 180°)

流程:**先在 5080 MuJoCo 仿真找到达标方法 + 统计成功率 + 录视频,再上真机**。
代码在 `box_demo_2/three_tests/`(本地源 `cc/box_demo_groot/three_tests/`)。

---

## 0. AGILE-29dof 部署(已有,decoupled-WBC 同架构)

AGILE 29dof 的真机部署**已经存在**且与 decoupled-WBC 完全同构 —— 两边读同一个
IPC 命令文件、发同一个 `rt/lowcmd_rl`、共用同一个 merger 和键盘:

| | decoupled-WBC | AGILE-29dof |
|---|---|---|
| adapter | `groot_wbc_boxdemo_adapter.py` | `agile_lowcmd_pipeline.py` |
| launcher | `start_groot_wbc_manual.sh` | `start_agile_manual.sh` |
| 命令入口 | `/tmp/robojudo_ext_cmd.json`(units=agile 物理量) | 同一个文件、同一 schema |
| 输出 | `rt/lowcmd_rl`(50Hz) → merger → `rt/lowcmd` | 同 |
| 站立高度 | 0.74 | 0.72 |
| 高度范围 | 0.40–0.80(walk 需 ≥0.72) | 0.40–0.72(训练范围) |
| RL 关节 | 15(12腿+3腰),臂解耦 PD | 12(腿),腰+臂 pipeline PD 保持 |
| 政策文件 | Balance/Walk onnx | velocity_height recurrent student .pt |
| 运行 env | `.venv_wbc` | **hdmi**(2026-07-02 修复过 typing_extensions) |

**因此三个测试脚本天然双控制器通用**:测试只写 IPC 文件,哪个 adapter 在跑就测哪个。

AGILE 底座启动(替代 GR00T 底座, 二选一, 别同时跑):
```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_agile_manual.sh --iface enp130s0    # merger + agile pipeline + 键盘
```

---

## 1. 仿真测试(5080)

统一入口 `three_tests/run_sim_test.py`,按控制器选解释器:

```bash
cd ~/agile_boxdeploy/box_demo_2/three_tests
DW=~/GR00T-WholeBodyControl/.venv_wbc/bin/python      # dwbc
AG=~/miniconda3/envs/hdmi/bin/python                  # agile29

# 测1 速度: 0.5(2s)->v2 扫 1.0/1.2/1.4(找达到真实 1m/s 需要的命令值)
MUJOCO_GL=egl $DW run_sim_test.py --controller dwbc    --test speed
MUJOCO_GL=egl $AG run_sim_test.py --controller agile29 --test speed

# 测2 下蹲: 无箱 20 次(h_low 扫 0.55/0.50/0.45)
MUJOCO_GL=egl $DW run_sim_test.py --controller dwbc    --test squat
# 测2 抱箱下蹲: benchmark 箱(0.25x0.35x0.25m 2kg)递到怀里+抱姿, 腰前倾扫 0/0.15/0.3 rad
MUJOCO_GL=egl $DW run_sim_test.py --controller dwbc    --test squat_box

# 测3 绕障: 半径扫 0.4/0.6/0.8/1.0, 固定朝向半圆, 带/不带 yaw-hold 对比
MUJOCO_GL=egl $DW run_sim_test.py --controller dwbc    --test circle
```

输出到 `~/three_tests_results/`:每配置一行指标 + `<test>_<controller>.json` + mp4 视频。

**仿真机制要点**:
- 两个 backend 共用 `scene_29dof.xml`(+`scene_inject.py` 注入箱子/柱子; 柱子=benchmark 规格 r0.15 h1.2)。
- **抱箱** = 真机 arm_sdk 覆盖的仿真等价:腰+臂 PD 顶到抱姿(kp40/kd2, 腰 60/2),
  抱姿 = box_demo_2 原 carry 姿态 `[0.29,0.22,0,0.98,...]` + 肩前倾 delta 0.15(前向下伸展防膝盖顶箱)。
  箱子是**自由体**(非 weld)→ 膝盖顶箱、滑落都是真实接触物理, 统计里有 `knee_contact_reps`。
- **腰部前倾**(重心控制)= 蹲深进度线性耦合的 waist_pitch 目标(0→pitch_max), 站起同步恢复;
  仿真扫角度选最优, 真机把选出的角度加进抱姿的 waist_pitch。
- **绕障** = 固定朝向半圆(不转身): 障碍在正前 r 米, vx/vy 合成切向速度绕 180°,
  终点在障碍后方 2r 处朝向不变。可选 `--yaw-hold-gain`(仿真里用真值 yaw 反馈; 真机对应今后接 IMU yaw-hold)。
- 判定: 速度=1s 窗口均值≥1.0 且不摔; 下蹲=到深度+回站立+不摔(+箱不丢)算稳, ≥18/20 PASS;
  绕障=不碰柱+间隙>5cm+终点误差<0.6m+终向偏差<20°。

---

## 2. 真机测试脚本(controller 常驻, 脚本发指令)

真机流程:先起底座(GR00T 或 AGILE 二选一)+ merger,**停掉键盘 pane**(单写者),
然后跑测试脚本(在 5080 上, 写 IPC 文件)。所有脚本都有:
`space`=暂停(速度归零) `r`=继续 `s`=回站立 `o`=急停 DAMP `q`=退出。

```bash
cd ~/agile_boxdeploy/box_demo_2/three_tests
# 测1 加速(0->0.5, 2s 后过渡到 1.0; 防撞墙用 space):
#    ⚠️ adapter 默认 fwd-max 0.50 挡 1m/s! 底座启动时加 --fwd-max 1.3
python3 test_speed_ramp.py --v2 1.0 --hold2-s 4 --stand-height 0.74   # AGILE 用 0.72
# 测2 无箱下蹲 ×20(蹲下缓一下站起缓一下):
python3 test_squat_cycles.py --mode plain --reps 20 --h-low 0.50 --controller dwbc
# 测2 抱箱下蹲(先递箱+arm_sdk 抱稳(box_demo 抓取或手动), 确认后回车开始):
python3 test_squat_cycles.py --mode box --reps 20 --h-low <sim选出的值> --controller dwbc
# 测3 绕障(障碍放正前 r 米, 完成后用键盘继续直行):
python3 test_circle_obstacle.py --radius 0.6 --speed 0.20 --side left
```

注意:
- **测1 场地**:1m/s × 4s ≈ 5m+ 直线, 脚本先打印预计距离, 确认场地再跑; `space` 随时刹停。
- **测2 抱箱**:脚本只发高度循环, 抱姿由 arm_sdk 进程保持(box_demo 抓取后状态), 手臂不动 → 箱不丢。
  腰前倾角用 sim 选出的值, 加到抱姿 waist_pitch(dual_arm/WaistRotator 侧)。
- **测3**:开环无里程计, 真机绕完可能有朝向漂移, 先 `q/e` 修朝向再 `w` 直行。
  峰值横移=--speed, 别超 adapter lat-max(GR00T 0.30)。

---

## 3. 结果(仿真, 2026-07-02)

| 测试 | dwbc | agile29 |
|------|------|---------|
| **①速度 ≥1m/s** | ✅ **cmd 1.2 → 实测 1.077 m/s**(跟踪~90%, tilt 12°, 0摔, 1.5s 内停稳) | ✅ **cmd 1.2 → 实测 1.096 m/s**(tilt 13.5°, 0摔;首测钉 0.5 是 sim 端 UI clamp, 已解) |
| **②下蹲 20 次(无箱)** | ✅ h_low 0.45/0.50/0.55 全 **20/20**(高度跟踪好) | ✅ h_low 0.50/0.55 → **19/20**;0.45→8/20(其物理下限 pelvis~0.49;**高度静差**: cmd0.50 实到 0.58) |
| **②抱箱下蹲(weld)** | ✅ **h_low 0.55 和 0.50 全 20/20**(深度 21~27cm, 膝碰 0, pitch 0/0.15 均可) | ✅ **h_low 0.55 + 腰前倾 0.15rad → 20/20**(有膝-箱轻触但不失稳);**pitch=0 或 0.3 → 膝顶箱失稳(2/20)** — "下蹲同时缓慢前倾"设计被精确验证, **0.15 rad(≈8.6°)是甜点**;0.50 到不了(带箱只降 10cm) |
| **③绕障半圆** | ✅ r 0.6/0.8/1.0: **cmd_gain=1.4 全 PASS**(端点误差~0.25m, 朝向≤2°;横向跟踪仅~35% 故须 gain) | ✅ r 0.6/0.8/1.0 **无补偿直接 PASS**(横向跟踪好, 端点误差 0.34-0.53m) |
| ③ r=0.4(临界) | ⚠️ 路径可达(端点误差 0.16m)但**摆臂起步瞬间蹭柱**(柱距前脸仅 0.25m);**收臂(arm_sdk tuck)后无碰撞**✓, 但收臂改变横向步态、gain 需在收臂下重标(遗留) | 同左(端点 0.03-0.06m 更准, 同样手蹭柱) |

**关键标定值(真机用)**:
- **速度**: 两边都**命令 1.2 m/s** 得真实 ~1.08;底座要 `--fwd-max 1.3`(AGILE pipeline 的
  mgr range 挂在 --fwd-max 旗标上, 不用改代码)。脚本 `test_speed_ramp.py --v2 1.2`。
- **抱箱蹲**: dwbc 蹲深 0.50/0.55 都行(0.50 实际沉到 ~0.47, 深度 27cm);**AGILE 用 0.55**
  (0.50 带箱只降 10cm)。**AGILE 必须配腰前倾 0.15 rad**(加到抱姿 waist_pitch;dwbc 加不加
  都行, 建议也加保余量)。抱姿(sim 调出, 供真机 arm_sdk 参考): 肩 pitch **-1.2**(前伸!
  非 box_demo carry 的 +0.29)、roll 0.15、肘 0.7 — 箱重心在膝前上方。
- **绕障**: dwbc `--cmd-gain 1.4`, agile29 1.0;r=0.4 用 0.6 的圈绕(或收臂, 待重标)。
- **0 摔**: 两控制器全部测试无一摔倒。视频/JSON: 5080 `~/three_tests_results/`。

**真机⇄仿真已知差距**: sim 是乐观下界(瞬时命令/理想摩擦/真值);真机速度跟踪可能更低
(命令再加 10-20%)、绕障开环有航向漂移(完成后键盘修向)、抱箱 weld=理想抓握
(真机靠 box_demo 力控抓稳后再蹲, 对应 --mode box 的"确认抱到"流程)。

---

## 4. 增补(2026-07-03): 箱子尺寸参数化 + 绕障模式B

**换箱子**: `--box-size 深,宽,高`(米) → `hug_for_box()` 自动推导抱姿(宽→肩roll夹持
±0.45m/rad; 深→肩pitch前伸±0.17m/rad + 递箱位x; 高→递箱位z)。sim 验证: 小箱
0.2×0.25×0.2 与默认箱**零调参 20/20**(双控制器); 大箱 0.3×0.45×0.3 近臂展极限,
用 tune 修正值 `--hug-roll 0.21 --box-x 0.29 --box-z 0.91` 稳抱。新尺寸先
`tune_box_hold.py --box-size ...` 验证。真机: 同一映射输出 = arm_sdk 抱姿目标。

**真机换箱接口(2026-07-03): `hold_box_hug.py` — 主参数只有箱宽**。与 sim 共用
`box_hug.py` 里同一份 `hug_for_box()` 公式, 通过 rt/arm_sdk 持续保持抱姿
(发布约定/kp/kd 照抄 dual_arm_target_reach.py 真机验证值; 腰 250/5, 肩肘 120–150/10):

```bash
# G1 本体, 底座(dwbc 或 agile)+merger 已在跑、机器人站稳后:
cd ~/zihou/box_demo_1/three_tests
python hold_box_hug.py --box-width 0.30                    # ① 只给宽, ramp 6s 进抱姿
python hold_box_hug.py --box-width 0.35 --waist-pitch 0.15 # AGILE 抱蹲必给腰前倾
python hold_box_hug.py --box-width 0.45 --roll 0.21        # 大箱用 sim tune 修正值
# ② 人工递箱(箱底放掌台, 站立时约 0.80m), 按 [ / ] 微调夹紧, , / . 微调腰前倾
# ③ 另开终端: python test_squat_cycles.py --mode box --controller dwbc|agile29 ...
# ④ 测完取走箱子 → 按 q 两次释放(先回中性0位再淡出 arm_sdk 权重, 交还底座)
```

深/高可选(`--box-depth/--box-height`, 缺省基准箱 0.25/0.25 — 抱姿本身只由宽决定,
深只在换更深的箱时给)。推导超线性区会打印 `clamped` 警告 → 先回 sim 用
`tune_box_hold.py --box-size` 验证。宽的有效线性区约 0.26–0.62m(对应 roll
0.05–0.45 限幅); 基准 0.35 → roll 0.15, 每 +10cm 宽 ≈ roll +0.11。

**绕障模式B(朝向跟随, wz 参与)**: 转90° → 面向行进方向绕弧(vx+wz) → 按半径
超越(弧上180°) → 转回原向。`test_circle_obstacle.py --mode heading`。
- **dwbc 全 PASS**(r 0.6/0.8/1.0, 端点 0.14–0.30m, 朝向 1–4°): 标定
  `--turn-rate 0.30 --turn-comp 1.25 --lead-out 0.25 --fwd-gain 1.4`。
  三个关键补偿: lead-out 外扩(给弧内侧手臂让空间, 治蹭柱)、turn-comp(补两次
  原地转的开环欠转 ~22%, 治末端朝向偏 37–45°→1–4°)、fwd-gain(vx 欠跟踪)。
- **AGILE 模式B 会摔**(原地转+复合弧超出其稳定域, 慢转也不行) → AGILE 绕障用
  已达标的模式A(固定朝向)。

---

## 5. 增补(2026-07-06): dwbc 后退稳定性 + 自动前倾补偿

**问题**: 真机 dwbc 后退时重心太靠后欲摔。**sim 复现**(`--test backward`, 1.5m,
v 0.2–0.5): 骨盆全程系统性后仰 **−6~−7°**(峰值 −10°, 速度越快越深), sim 平地
无扰动没摔但没有余量 — 真机有扰动就危险。

**前倾注入三种方式全试过(round8/9/10)**:
| 方式 | 结果 |
|---|---|
| rpy 命令通道(obs command[4:7]) | ❌ 策略基本不跟(pitch 仅变 0.1°) |
| 纯腰pitch动作偏置 | ❌ **更糟**: 策略从关节观测发现偏置→反补偿, 骨盆更仰(−8~−10°)+位移超冲33% |
| **动作偏置+观测补偿(obscomp)** | ✅ 骨盆后仰 −6.3→−4.5°, tilt 11.5→10.7, 停稳振荡 5.5→3.3°, 躯干转前倾 +3~+7°, 0摔(代价欠距5-10%) |

obscomp = 腰pitch 动作目标 `+= lean`, 同时喂策略的 obs 腰pitch `-= lean`(策略
看到的=它自己命令的, 无感不反抗)。**甜点增益 0.3 rad/(m/s)**。

**真机接口**(`groot_wbc_boxdemo_adapter.py`, 与 sim `--pitch-mode waist_obscomp`
同构): `--back-lean-gain 0.3 [--back-lean-max 0.15] [--back-lean-rate 0.30]`,
默认关; 只在 vx<0 生效, lean 随 |vx| 成比例、渐入渐出, DAMP/LIMP/急停自动归零。
真机首测建议 0.2 起步。sim 复跑: `run_round8.sh`(基线+rpy) `run_round10.sh`
(obscomp), 结果 5080 `~/three_tests_results/backward_*`。

**行走高度地板(round11, 真机现场"不习惯的高度"假设验证)**: h 0.60–0.78 扫掠
(后退 v0.3/0.4 + 横移 v0.25):
- **低位 = 步态实质崩坏**: h0.60 后退只走 60% 距离; 横移 h≤0.66 几乎不动
  (1m 命令只挪 2–5cm), h0.70 也仅 37% — 真机上表现为原地乱蹭/晃, 即现场看到的
  "很不稳";
- **h0.74(站高)是后退/横移最优点**; **h0.78 并不更稳**(后仰峰 −10.5° 最深,
  停稳振荡更大) — "站更高退更直"不成立;
- 低位入口: 键盘 `c`(pick 0.36)/`z` 降高 / nav HTTP 带 height 命令。groot_mover
  早有 WALK_MIN_HEIGHT=0.72 地板, 但键盘/nav 链路没有 →
- **修复**: adapter 加 `--walk-height-floor 0.72`(默认开, 0=关): 速度非零且命令
  高度低于地板 → 拦速度打印 [GATE]; 高度从低位恢复(slew)期间也拦, 恢复过 0.72
  自动放行 = "先恢复高度再走"的 warmup。sim 复跑: `run_round11.sh`。

**vx 不对称硬限(2026-07-06 晚, 真机)**: 后退 0.4 时机器人保不住高度、越走越低
直至摔(前倾补偿救不了这个模式) → adapter 加 `BACK_VX_MAX=0.20`(后退恒 ≤0.2,
**常量, 有意不开放 CLI**)+ `FWD_VX_HARD=1.00`(--fwd-max 超 1.0 压回并告警;
复跑 cmd1.2 速度指标需临时改此常量)。键盘两启动器统一
`--vx 0.40 --vy 0.25 --wz 0.40`(s 后退发 -0.4 被截 -0.2, 单键双向不用分)。

## 6. 增补(2026-07-08): taptap 停止回正踏步(测试版)

**动机**: 导航模式走完仍漂; 宇树官方运控每段动作结束后会原地踏步恢复标准静止
站姿(双脚固定间距+正向), 段间清零开环漂移。**sim 验证(round13/13b)**:
fwd/back/lat/turn × {无回正, vx方波, vx渐减, wz原地小转} — **wz 模式最优**:
脚间距 err 1.9→0.4cm, 前后错位 11→7.6cm(fwd), 踏步自身平移漂移仅 1–2cm,
代价 ~2° yaw 漂移(修复后净激励为零)。vx 系激励无明显优势。

**实现**: adapter `--taptap` 旗标(默认关) — 显式停止(fresh 零命令)+防抖 0.35s
后注入原地踏步窗: wz ±0.15rad/s 方波(偶数个半周期+相邻回正起始方向轮换 →
激励积分净零), 高度自动回站高 0.74, 完成后交还 Balance; 新命令到达立即中断。
入口: `start_g1_onboard_taptap.sh`(=dwbc 版+taptap) /
`start_g1_onboard_nav_taptap.sh`(=nav 版+taptap)。
可调: `--taptap-settle-s 1.2 --taptap-cmd 0.15 --taptap-mode wz|vx|vx_taper
--taptap-period-s 0.4 --taptap-debounce-s 0.35 --taptap-min-motion-s 0.4`。

**对抗审查修复(4 个确认缺陷)**: ①蹲位被 GATE 拦下的"假运动"误触发回正并强制
站起 → 运动判定改用 GATE 后命令+开窗要求高度在地板上; ②奇数半周期净漂移
(+3.4°/次) → 窗长取偶数半周期+方向轮换; ③命令黑箱(mover 崩溃 stale)/RL_LOWER
被当"用户停止"而自主动作 → 只认 fresh 零命令且 fsm==RL_FULL; ④单帧毛刺覆盖
待回正记录 → 毛刺不覆盖, 被打断的回正下次停止补做。

**默认值变更(2026-07-06 晚, 真机确认后退稳后)**: `--back-lean-gain` 默认
**0.0 → 0.3**(两启动器/nav 自动继承, 关掉给 0)。**横移速度扫掠(round12,
h0.74, vy 0.15–0.35)**: 稳定性完全无差别(roll 振荡 ~9°, tilt ~9.5°, 0摔),
距离跟踪恒 ~58% — **横移不挑速度只挑高度**(地板已保证); 键盘默认 vy:
标准版 0.25(round12 区间中点), nav 版留师弟的 0.30(也在区间内)。
sim 复跑: `run_round12.sh`。
