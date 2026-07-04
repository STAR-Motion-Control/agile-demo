# G1 本体部署 — dwbc + agile29 双控制器上机

> **2026-07-03 更新:换机 unitree-001**(002 返场维修: MCU↔NX 断连)。
> 001 已完成同套部署且**双控制器 dry-run 冒烟通过**(内网健康: MCU ping 通, lowstate 正常收)。
> `ssh unitree-001`(Mac/5080 都免密, wifi 10.33.12.89)。与 002 的差异:
> - conda env = **robojudo_zihou2**(launcher 自动挑; ⚠️ **numpy 必须钉 1.26.4** —
>   pip 装 pin 时被拉到 2.x 会弄坏 onnxruntime(_ARRAY_API not found), 已降回)
> - **001 无 /opt/ros** → dwbc 用 `~/zihou/rclpy_stub`(functional fake: G1Env 的
>   ROSManager 只服务图像发布旁路, 控制链全走 sdk2py DDS; launcher 自动挂)
> - adapter 打了两个可移植性补丁(已存档回 cc/box_demo_groot/): ① wbc_config["INTERFACE"]
>   跟 --interface(yaml 默认 lo); ② 删 main 里的显式 ChannelFactoryInitialize
>   (捆绑版 sdk2py 不幂等, 交给 G1Env 唯一 init)
> - box_demo 用相机时注意: 001 的 D435 序列号 ≠ 默认 254322075278(002 的), 要 --camera-serial

**背景(2026-07-02)**:测试中网线松动机器人摔倒(人没事),5080↔机器人**有线口坏了**
→ 两套控制器全部搬到 G1 本体(Jetson Orin NX, `unitree-g1-nx`)上跑,不再依赖 5080。

## 拓扑(新)

```
G1 本体 (Jetson NX, ssh unitree-002 = wifi 10.33.2.42)
┌────────────────────────────────────────────────┐
│ merger + adapter(dwbc 或 agile) + 键盘/测试脚本 │
│           全在 conda env robojudo 里             │
│ DDS iface = enP8p1s0 (内网 192.168.123.164      │
│             → MCU 192.168.123.161, 实测 ping 通) │
└────────────────────────────────────────────────┘
5080 只剩可选角色: SAM3 服务(box_demo 用, 走 wifi HTTP)
```

## 环境盘点结论(2026-07-02 实测)

| 项 | 状态 |
|---|---|
| 磁盘 | 111G 空闲 ✓ |
| 内网 DDS | enP8p1s0 UP, MCU 123.161 ping 通 ✓(坏的只是对 5080 的外链) |
| `robojudo` env | py3.10.20 aarch64, **torch 2.11+cu130 / onnxruntime 1.23.2 / mujoco 3.9 / yaml / unitree_sdk2py 全有** ✓ |
| 缺(dwbc 用) | `pin`(pinocchio) + loop-rate-limiters + typer + termcolor + loguru → pip 装 |
| `~/GR00T-WholeBodyControl` | **已在机器人上**, Balance/Walk onnx ✓;但 `.venv_teleop/.venv_sim` 是从 x86(zhengye@5090)整个 rsync 来的 — **x86 的 .so 在 aarch64 上全段错误, 不可用, 别碰** |
| WBC-AGILE | 机器人上没有 → 从 5080 经 Mac 中转推到 `~/zihou/WBC-AGILE`(16M 去 .git) |
| `~/zihou/box_demo_2` | merger / groot adapter / agile pipeline / 键盘都已在 ✓;缺 three_tests/ |

## 部署清单(机器人回线后执行)

```bash
# 1) 代码(Mac 上推)
rsync -a <scratchpad>/wbc-agile-relay/ unitree-002:zihou/WBC-AGILE/
scp cc/box_demo_groot/three_tests/{schedules,ipc_ctl,test_speed_ramp,test_squat_cycles,test_circle_obstacle}.py \
    cc/box_demo_groot/three_tests/THREE_TESTS.md unitree-002:zihou/box_demo_2/three_tests/
scp cc/box_demo_groot/start_g1_onboard.sh unitree-002:zihou/box_demo_2/

# 2) 依赖(机器人上, 放后台防 wifi 断)
conda activate robojudo && pip install pin loop-rate-limiters typer termcolor loguru

# 3) 验证 import 链
#   agile: agile_lowcmd_pipeline --help(拉起 agile.sim2mujoco 栈)
#   dwbc:  groot_wbc_boxdemo_adapter import(拉起 decoupled_wbc 栈, 缺啥补啥)

# 4) 干跑冒烟(不发 rt/lowcmd_rl, 安全)
bash start_g1_onboard.sh dwbc --dry-run
bash start_g1_onboard.sh agile --dry-run
```

## ⚠️ 当前物理链路状态(2026-07-02 晚)

- **NX 自带口 `enP8p1s0`:物理层坏/抖动**(一会 UP/1000Mb 一会整个 DOWN,重启丢 IPv4)—— 摔倒损伤。
- **现场已插 USB 网卡 `enx2c16dbaa7742`**(已配 192.168.123.164、UP/LOWER_UP),
  但**从它 ping 不通 MCU(192.168.123.161)** → 它的网线可能没接到机器人内部交换/MCU 网段。
- **rt/lowcmd 必须经 123.x 内网到 MCU** — 这条物理链路不通,两套控制器都动不了真机。

**现场检查清单**:
1. USB 网卡的网线另一头插在哪?要插到**机器人内部网段的口**(原 5080 线拔下来的那个外部
   RJ45 如果通内部交换机, 插它;或直接插 MCU 侧口)。
2. 插对后验证:`ping 192.168.123.161` 通 → 启动器把 IFACE 换成 USB 卡:
   `IFACE=enx2c16dbaa7742 bash start_g1_onboard.sh dwbc`
3. 若 enP8p1s0 修好了(换线/换水晶头), 默认 IFACE=enP8p1s0 直接用;启动器会自动补回
   重启丢失的 192.168.123.164/24。

## 正式运行

```bash
ssh unitree-002 && cd ~/zihou/box_demo_2
bash start_g1_onboard.sh dwbc            # 或 agile;测速度加 --fwd-max 1.3
# 指标测试: Ctrl-C 掉键盘 pane 后
cd three_tests && python3 test_speed_ramp.py --v2 1.2      # 等(见 THREE_TESTS.md)
```

注意:
- 键盘 pane 与测试脚本**单写者互斥**(同一 IPC 文件)。
- 摔倒后首次上电:先 DAMP/吊架确认关节正常再进 RL_FULL。
- wifi(10.33.2.42)不稳会断 ssh — tmux 里跑的进程不受影响;pip 用 nohup。
- box_demo(抱箱)在此拓扑下:SAM3 仍在 5080(:5300), 机器人经 wifi 访问
  `http://10.24.88.193:5300`(改 --host;有线 123.222 已不可用);运控直接本机 IPC
  (`--locomotion groot`),不再需要 HTTP 桥/RemoteMover。
