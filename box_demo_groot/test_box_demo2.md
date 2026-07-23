# test_box_demo2 — GR00T-WBC 底座真机测试指令

box_demo_2 的运控底座 = **GR00T-WBC**(decoupled-WBC)。本文件汇总两类测试:
1. **手动键盘驱底座**(`start_groot_wbc_manual.sh`)— 验证 GR00T-WBC 迈步 + **调线速度/角速度**。
2. **完整抱箱 pipeline**(box_demo 跑机器人,运控经 HTTP 转 5080)— 详见 `HARDWARE_TEST_GROOT_REMOTE.md`。

全程机器人吊架吊着 / 急停在手。所有底座进程跑在 **5080**(`192.168.123.222`,网口 `enp130s0`),经 DDS 下发到机器人(`enx2c16dbaa7742 = 192.168.123.164`)。

> 上机前先确认 DDS 链路活着:`ip -br addr show enp130s0`(应有 192.168.123.222/UP)、`ping -c2 192.168.123.164`(0% 丢包)。链路断 → merger/adapter 会 `create domain error` 崩(不是代码 bug)。
>
> ⚠️ **启动顺序**:完整 pipeline(§3)要起 SAM3 时,**必须先起 SAM3 等加载完、再起底座**,否则机器人乱跳(详见 §3,待排查根因)。纯键盘调速(§1)不涉及 SAM3,无此问题。

---

## 1. 手动键盘底座测试(含调速)

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_manual.sh --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble \
  --key-vx 0.20 --key-vy 0.12 --key-wz 0.18
```

起 3 个 tmux pane(`g1-groot-wbc-manual`):
- pane1 `merge_lowcmd_arm_sdk.py` — `rt/lowcmd_rl` + `rt/arm_sdk` → `rt/lowcmd`(唯一写 `rt/lowcmd` 者)
- pane2 `groot_wbc_boxdemo_adapter.py` — GR00T-WBC → `rt/lowcmd_rl`(50Hz)
- pane3 `agile_keyboard_control.py` — 键盘 → IPC `/tmp/robojudo_ext_cmd.json`

**操作前把焦点放在 pane3**(`Ctrl+B → 方向键` 切 pane)。键位:

| 键 | 动作 | 键 | 动作 |
|----|------|----|------|
| `w` / `s` | 前进 / 后退(±vx) | `c` | 蹲到 pick-height 0.36m |
| `a` / `d` | 左移 / 右移(±vy) | `r` | 回站立 0.74m |
| `q` / `e` | 左转 / 右转(±wz) | `space` | 停(速度归零) |
| `z` / `x` | 高度 −/+ 2cm | `f` / `l` | 切 FSM RL_FULL / RL_LOWER |
| `h` | 平滑切到双臂自然下垂；再次按下平滑交还 policy | | |
| | | `o` | 急停(DAMP,仅阻尼) |

`h` 会先把底座速度归零，并且只在 `rt/lowcmd_rl` 与最终 `rt/lowcmd`
均新鲜、没有其他活跃 `rt/arm_sdk` 发布者时接管。完整 box_demo 操控场景会
禁用此键，避免和抓取控制器双写手臂。

---

## 2. 调线速度 / 角速度(本次重点)

每次按键下发的速度**幅值**由这三个参数决定(单位 m/s、m/s、rad/s):

| 参数 | 脚本默认 | 本次用 | 控制 | adapter 上限 |
|------|---------|--------|------|-------------|
| `--key-vx` | 0.12 | **0.20** | `w`/`s` 前后线速度 | `--fwd-max` 0.30 |
| `--key-vy` | 0.08 | **0.12** | `a`/`d` 横移线速度 | `--lat-max` 0.15 |
| `--key-wz` | 0.10 | **0.18** | `q`/`e` 偏航角速度 | `--yaw-max` 0.30 |

其它相关旋钮:
- `--key-timeout 0.25` — 松键 0.25s 后速度过期归零(防失控)。
- `--height-step 0.02` — `z`/`x` 每次 ±2cm;`--height-rate 0.10` — adapter 内部高度变化速率上限。
- adapter 截断:`--fwd-max / --lat-max / --yaw-max`(默认 0.30 / 0.15 / 0.30)。

**调速规则(重要):**
1. **下限**:速度幅值必须 **> 0.05**。GR00T-WBC 在 `‖[vx,vy,wz]‖ < 0.05` 时进 Balance(原地平衡、**不迈步**),小于该值机器人只会原地晃不会动。别把 key 速度设到 0.05 以下。
2. **上限**:key 速度必须 ≤ 对应 adapter clamp(vx≤fwd-max、vy≤lat-max、wz≤yaw-max),否则被 adapter 截断,设了也白设。要更快需**同时**调大 clamp,例如:
   ```bash
   bash start_groot_wbc_manual.sh --iface enp130s0 \
     --groot-repo ~/GR00T-WholeBodyControl --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble \
     --key-vx 0.30 --key-wz 0.25  --fwd-max 0.35 --yaw-max 0.30
   ```
3. 也可用环境变量替代旗标:`KEY_VX=0.20 KEY_VY=0.12 KEY_WZ=0.18 bash start_groot_wbc_manual.sh ...`。
4. 改完**重起整个脚本**生效(参数在启动时传给 pane3);跑起来后只能靠按键,不能热改幅值。

> 经验起点:平稳走 `--key-vx 0.12~0.20`;想快一点的转向 `--key-wz 0.18~0.25`。先小后大,吊架兜底。

---

## 3. 完整抱箱 pipeline(box_demo 跑机器人)

底座(本文件 §1 的 merger + adapter)照常跑在 5080,但键盘 pane 换成 **HTTP 桥**,box_demo 跑机器人。完整步骤见 `HARDWARE_TEST_GROOT_REMOTE.md`,核心三件套。

> ⚠️ **启动顺序:先 SAM3,再底座(已知问题,待排查根因)**
> 必须**先起 SAM3 并等它加载完**,**再起** `start_groot_wbc_manual.sh`。反过来(底座已在控、再起 SAM3)会在 SAM3 上 GPU 加载模型时**让机器人乱跳** —— 疑似 SAM3 抢 GPU 卡顿 stall 了 adapter 50Hz 控制环、输出跳变。先按此顺序测,根因之后查。
> 三个服务各占**一个终端、前台跑(都去掉 `&`)**,方便看日志/状态、避免忘关。

```bash
# 5080 终端1:① ★SAM3 服务(必须最先起,前台跑;另开终端验证:curl -s 127.0.0.1:5300/docs >/dev/null && echo SAM3_OK)
cd ~/sam3 && ~/sam3/.venv/bin/uvicorn scripts.sam_server:app --host 0.0.0.0 --port 5300

# 5080 终端2:② 底座(merger+adapter;等 SAM3 加载完再起。起来后 Ctrl-C 掉 pane3 键盘,只留 merger+adapter)
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_manual.sh --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble

# 5080 终端3:③ HTTP IPC 桥(收机器人 HTTP /cmd → 写 units:agile IPC 给 adapter;前台跑,自检 curl 127.0.0.1:5001/status)
~/miniconda3/envs/hdmi/bin/python ~/agile_boxdeploy/box_demo_2/agile_http_ipc_server.py \
  --backend legacy-ipc --host 0.0.0.0 --port 5001
```

```bash
# 机器人(192.168.123.164 = alias unitree-g1b):box_demo,先 --no-vlm 验链路
ssh unitree@192.168.123.164
conda activate robojudo
cd /home/unitree/zihou/box_demo_2
UNITREE_SUDO_PASS=123 python box_demo_main.py --locomotion remote --no-vlm \
  --ipc-url http://192.168.123.222:5001 --host 192.168.123.222 --port 5300 --iface enx2c16dbaa7742
```

> §1 的 `--key-vx/--key-vy/--key-wz` **不影响** pipeline 阶段的走位 —— pipeline 的小步对齐速度由 `groot_mover.py` 的 `FWD_CRUISE/LAT_CRUISE/YAW_CRUISE` + floor + warmup 决定(SAM3 给 cm/度,GrootMover 换算成速度×时长)。

---

## 4. 抱箱小距离调整测试(新前进方法)

box_demo 靠近箱子时会输出 **cm 级的小前进/横移微调**。GR00T-WBC 步态从站立起步有 ~0.6–0.8s 加速斜坡 + 残余站立摆动,**旧方法**小命令会踏步不前甚至**净后退**;**新方法**(`groot_mover.py`,默认 `warmup_time=0.6` 先起步态 + `min_duration=1.5` + `min_distance=0.08` + `v_floor=0.12`)在 5080 MuJoCo sim2sim 端到端验证做到**跨所有摆动相位净前进 +3.9~+18cm、绝不后退**(OLD 最差 −7.6cm)。细节见 `FORWARD_STEP_TUNING.md`。真机验证:

**前置**:§3 的 5080 底座(终端②merger+adapter)+ HTTP 桥(终端③ :5001)要起着。**隔离测试(A)不需要 SAM3/相机/箱子。**

### A. 隔离小步测试(box-free,先单独验运控)

机器人上跑,复现 box_demo 会发的一串小前进/横移/转,逐条按 Enter 看每条是否净前进:

```bash
ssh unitree@192.168.123.164          # 机器人 DDS IP(=alias unitree-g1b)
conda activate robojudo
cd /home/unitree/zihou/box_demo_2
python test_small_moves.py --ipc-url http://192.168.123.222:5001
```

序列:前进 4/6/8/10cm → 横移 ±5cm → 转 ±10° → 后退 6cm(逐条按 Enter;`--auto` 免按)。
**看**:每条前进/横移后 base **净前进/横移**(warmup 期先小步把步态起来再走),**绝不后退**;后退 −6cm 仍能后退。

**A/B 对比新旧(同一台直接切)**:
```bash
python test_small_moves.py --warmup-time 0     # 关 warmup(≈旧行为)→ 小步可能踏步/后退
python test_small_moves.py --warmup-time 0.6   # 新默认 → 应始终净前进
```
其它可调:`--min-duration 1.5` / `--min-distance 0.08` / `--dist-gain 1.7`(开环补偿,单次更接近目标)。

### B. 完整 box_demo(真出小距离调整)

跑完整 pipeline(需 §3 三件套含 SAM3),box_demo 感知箱子后自动输出小距离对齐,底座用新方法执行:

```bash
ssh unitree@192.168.123.164
conda activate robojudo
cd /home/unitree/zihou/box_demo_2
UNITREE_SUDO_PASS=123 python box_demo_main.py --locomotion remote --no-vlm \
  --ipc-url http://192.168.123.222:5001 --host 192.168.123.222 --port 5300 --iface enx2c16dbaa7742
# box_demo 打印每步 "forward Xcm (v=..., ...s)";想对比关 warmup 加 --warmup-time 0
```

**看**:每次对齐小步都净前进(plan 里 warmup 期 + 主 move),**绝不后退**。
**蹲不能走**:箱子在地上要蹲抓时,底座 <0.72m 会**告警且不迈步**(GR00T-WBC 蹲下不走)——
先站立高度走到位→停→再蹲抓,别边蹲边走;要"蹲前自动抬高再走"加 `--auto-raise-walk`。

> **部署提醒**:上面命令要求机器人已装**新版** `groot_mover.py`/`remote_mover.py`/`box_demo_main.py`/`test_small_moves.py`。
> DDS 链路当前离线(5080 enp130s0 或机器人 enx2c16dbaa7742 未通)。链路恢复后从 5080 一条命令推:
> `ssh 5080-laptop 'scp ~/agile_boxdeploy/box_demo_2/{groot_mover,remote_mover,box_demo_main,test_small_moves}.py unitree@192.168.123.164:/home/unitree/zihou/box_demo_2/'`

---

## 急停 / 回退

- **软急停**:键盘 pane 按 `o`(DAMP);或任意机器 `curl http://192.168.123.222:5001/damp`(pipeline 阶段)。
- **硬急停**:无线遥控 `select`。
- **链路/进程崩**:`create domain error` = DDS 链路断,先恢复 enp130s0 + ping 通,再重起脚本。
- **回退 AGILE 底座**:`start_agile_box.sh`(热备)。
