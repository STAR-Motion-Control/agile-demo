# G1 Benchmark v2 — 统一仿真场景 "ManipArena" + 任务重定义

> 2026-06-13。目标：把 R1–R5 所有零散测试 + 新搬箱流程，统一到**一个可复现的仿真场景**上；场景里物体/桌子位置做有限随机（50 个变种 seed 0–49，全模型共用）；任务重新定义为干净的分层体系。
> 取代：R1–R5 的 per-test 临时场景注入。旧 harness 测试保留可跑，新体系是 `bench_arena_*` 系列。

---

## 1. 设计动机（为什么要统一场景）

现状问题：每个测试各自注入场景（circle 一个柱子、squat 一个箱、pipeline 一组 A/B/C 点、psi0 一张桌），坐标/箱属性/碰撞方案各 harness 各测试都不同 → 难复现、跨测试不可比、新增任务要重写场景。

v2 方案：**一个 ManipArena 场景生成器**（`manip_scene.py`）按 `variant_seed` 产出 (a) 家具 XML 片段（注入各 harness 的机器人 XML）+ (b) `SceneSpec` 结构体（所有地标真值位姿 + 派生导航目标 + 抓取/weld 定义）。所有任务都是这个场景上的"任务脚本（mission）"，只用到场景的不同子集。50 个 seed 全模型共用 → 严格可比。

---

## 2. ManipArena 场景布局

俯视坐标系，机器人初始在原点、朝 +X。6m×6m 平地房间。地标语义固定，位姿按变种 jitter：

| 地标 | 符号 | 标称位姿 | 变种 jitter | 说明 |
|---|---|---|---|---|
| 机器人初始 | spawn | (0,0), yaw 0 | ±0.10m, ±0.10rad | |
| **取货桌** | T_pick | (1.6, 0), 桌面高 **H_pick** | 位置±0.15m, yaw±10° | 桌面 0.6×0.4m。H_pick 是核心受控变量 |
| **箱子**（2kg） | box | T_pick 桌面中心 | 桌上随机±0.05m | 0.30×0.22×0.22m，需 Psi0 上身弯腰/下蹲抱起 |
| **中转桌** | T_relay | (2.4, 1.8), 桌面高 0.55 | 位置±0.2m | 桌面 0.5×0.4m。放 M2 的小方块 |
| **小方块** | cube | T_relay 桌面 | ±0.05m | 0.08m³, 0.3kg。M2 的第二物体 |
| **置物区** | Z_store | (3.2, −1.2) | ±0.3m | 0.6×0.6m 浅框/标记台，rim 0.12m，放箱目标 |
| **柱状障碍** | pillar | (1.5, −0.8) | ±0.3m | r=0.15 h=1.2，导航/绕圈共用 |
| 开阔走廊 | lane | +X 方向 4m 直道 | — | 走速测试用 |

**H_pick 分层**：variant seed mod 4 → H_pick ∈ {0.30, 0.45, 0.60, 0.75}m（≈12–13 个变种/高度）。这样"取箱要弯多低"成为可统计的自变量：0.75=站着略弯，0.30=深蹲近地。

**派生导航目标**（由家具真值算出，每变种自洽，全模型一致）：
- `P_pick_front`：T_pick 边缘前 0.55m，朝桌。
- `P_relay_front`：T_relay 前 0.55m，朝桌。
- `P_cube_touch`：使右腕能触到 cube 的**固定目标位姿**（cube 位姿 + 机器人臂展反算）。
- `P_store`：Z_store 边缘，朝置物区。

`SceneSpec` 字段（JSON 可序列化，写进每条 JSONL 便于复现）：
```
{variant_seed, H_pick, spawn[x,y,yaw],
 T_pick{pos,top_h,size}, box{pos,size,mass},
 T_relay{...}, cube{...}, Z_store{pos,size,rim_h},
 pillar{pos,r,h},
 targets{P_pick_front, P_relay_front, P_cube_touch, P_store}[x,y,yaw],
 grasp{box_grasp_h, d_grasp=0.30, cube_touch_d=0.15}}
```

---

## 3. 抓取/放置抽象（无真实抓取，统一约定）

沿用 R3–R5 已验证机制（bit-2 碰撞 + body 位镜像 + 运行时 weld）：
- **磁吸抓箱**：双腕到 box 表面距离均 < `d_grasp`(0.30m) 时，原位激活 双腕↔box weld。弯腰深度由"够得着"自然涌现——桌越低越要蹲深。
- **放箱到面**：释放 weld + 开启 box 碰撞；box 须在目标面（桌/置物区）静止、直立(<30°)、落点在容差内。
- **小方块转移**（M2）：机器人到 `P_cube_touch`、腕到 cube < `cube_touch_d`(0.15m) 时，激活 cube↔box 顶面 weld（小方块"放到箱上"），此后 cube 随箱移动。
- **上身动作**：抱箱/弯腰/放置的上身关节由 **Psi0 真机轨迹回放**驱动（real_ep053 持物蹲放；按阶段选 reach-down / carry / place 片段），下身=被测模型。

---

## 4. 任务重定义（全部在 ManipArena 上）

分两层：**L 系（locomotion 原语，无箱，验下半身能力）** 与 **M 系（manipulation 任务，Psi0 上身，全场景）**。

### L 系 — locomotion 原语（在 arena 内执行）
| ID | 名称 | 用到的场景子集 | 指令 | success |
|---|---|---|---|---|
| **L1** walk | 开阔走廊 | vx sweep 0.4–1.2，10s | 无摔 ∧ 实速≥0.9×指令 |
| **L2** circle | 绕 pillar | vx=0.4 wz=±0.4 r=1m, 2圈 | 无摔无碰 ∧ 径向err≤0.15m |
| **L3** goto | spawn→P_pick_front | 粗导航→小指令校准(≤0.1) | 5cm/5° 收敛 |
| **L4** vln | 穿越 arena（避柱） | 共用 tape 指令流 | 无摔 ∧ 小指令响应/跟踪 |
| **L5** squat_limit | 原地（持箱） | 0.05m/s 连降到极限 | 标定 depth_floor/drift/fall_h |

（L 系 = 把 R1–R5 的 walk/circle/goto/vln/squat_limit 搬进统一场景；坐标改为引用 SceneSpec。）

### M 系 — manipulation 任务（核心）

**M1 — table_pick_place（取箱→放置物区）**
分阶段状态机：
1. nav spawn→`P_pick_front`（避 pillar）
2. 按 H_pick 弯腰/下蹲（Psi0 reach-down 上身），磁吸抓箱
3. 持箱起身
4. nav→`P_store`（绕 pillar）
5. 蹲到置物区高度，放箱
- **success**：box 直立静止于 Z_store 容差内 ∧ 全程无摔。
- **跨 50 变种**：按 H_pick 分层报告成功率（直接答"桌多低就抱不起"）。

**M2 — relay_pick_place（取箱→中转放箱→取方块放箱上→再抱→置物区）** ← 新任务
1. nav→`P_pick_front`，抓箱（同 M1 step1-3）
2. nav 朝置物区方向出发，中途绕去 `P_relay_front`
3. 在 T_relay **放下箱**（放到中转桌面）
4. nav→`P_cube_touch`（固定位姿，须精确到位才够得着）
5. 腕触 cube → **cube 转移到箱顶**（weld）
6. **重新抱起箱**（箱+方块）
7. nav→`P_store`，放箱（箱+方块入置物区）
- **success**：箱+方块最终在 Z_store ∧ 7 阶段全过 ∧ 无摔。
- **逐阶段 flag**：nav 到位误差、放箱 ok、cube 转移 ok、再抓 ok、终放 ok、各阶段摔倒——长程任务的诊断维度。

M2 把"精确导航(到 cube-touch 位)+多次变高蹲放+变载稳定+模式切换"压成一条长链，是最严苛的综合测试。

---

## 5. 变种与可比性

- `variant_seed` 0–49：决定全部 jitter + H_pick 分层。**所有模型跑同一组 50 seed**（家具真值逐位一致；仅机器人 URDF 因模型而异）。
- L 系：在 arena 内跑各自 sweep（如 L1 速度档），arena 家具作为固定背景/障碍。
- M 系：每模型 50 变种各 1 trial（M1/M2 各 50）。
- JSONL 每行带 `variant_seed`/`H_pick`/`scene_spec_hash` → 任意结果可重建场景复跑。

---

## 6. 指标与成功判据（统一）

- **导航**：每段到位 pos/yaw 误差；精确段 5cm/5°。
- **下蹲**：达成最低 root、目标-实际偏差、root 漂移（"没蹲稳"）。
- **抓放**：grasp_success、place_ok（直立+静止+落点容差）、cube_transfer_ok。
- **负载稳定**：持箱/持箱+方块各段 max_tilt、drift。
- **摔倒**：统一判据（tilt>0.9 / 非脚触地），记阶段+时刻。
- **任务**：完成率、逐阶段成功率、总时长、终态物体位姿误差。

---

## 7. 工程落地计划（分步、每步可验证）

1. **`manip_scene.py`**（共享场景生成器）+ `validate_scene.py`（渲染若干变种 PNG 目检：俯视正交 + 3/4 视角，覆盖 4 个 H_pick）。→ 本步先确认场景"长得对"。
2. **`manip_mission.py`**（共享 mission 状态机：nav-leg / squat / grasp / place / cube-transfer 原语，吃 SceneSpec；复用 R3–R5 的 nav 控制器+磁吸+释放）。
3. **接入 4 harness**：新增 `--test arena_M1 / arena_M2 / arena_L1..L5 --variant N`；各 harness 注入家具片段到自己机器人 XML，上身走各自 Psi0 回放通路。
4. **试点**：单变种各任务 1 trial 目检（重点 M2 全链）。
5. **全量**：M1/M2 各 50 变种 ×4 模型；L 系 sweep。
6. **可视化+控制台**：烧字幕（含 H_pick/阶段）；manifest 加 arena 任务 + 变种过滤；控制台交互台增"选 H_pick/变种渲染"。

> 本轮先交付 §7.1–§7.2（场景生成器 + 渲染验证 + mission 骨架），确认场景与任务定义无误后再铺开 §7.3+ 的全量。
