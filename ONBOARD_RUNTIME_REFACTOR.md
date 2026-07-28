# G1 板载运行时重构：证据、架构与 A/B 验证

更新日期：2026-07-28（Asia/Shanghai）

## 0. 文档状态与证据边界

本文描述“操控 + 导航 + 运控”同时运行时的 CPU/时序问题、当前候选重构以及后续真机 A/B
验证方法。它不是启动授权，也不是已经通过真机验收的部署说明。

- 当前工作分支为 `refactor/onboard-runtime`。候选代码保持在独立目录，与
  `baseline/live-demo-20260728` 分离；部署前必须记录最终 commit、dirty status 和关键文件哈希。
- **本地受管环境无法连接 5080：**对 `10.24.88.193:22` 的只读 SSH 尝试由当前
  网络沙箱以 `Operation not permitted` 拒绝。这不能证明 `g001` 本身可达或不可达，
  只能说本轮未重新读取其时间、进程、tmux、DDS、IPC、代码哈希和机器人状态。
- **当前候选重构尚未在 `g001` 上启动，也没有做任何真机动作验证。**
- 本轮只做了本机静态检查和离线测试，未启动、重启或操作任何真机控制脚本。
- `SESSION_HANDOFF.md` 中的 PID、CPU、频率、DDS、网络、姿态和所有权数据都是
  **2026-07-28 的历史快照**，不是当前状态。
- 5080 上的 ONNX Runtime 测试是隔离的离线微基准，不是 Jetson 板载系统基准；其绝对数值
  **不得直接外推到 Jetson**。

后续任何远程判断都必须在具备只读访问能力时重新采样。进程存在不代表 MCU 正常在控，
最终判断必须同时包含 DDS 新鲜度、最终控制流和现场机器人状态。

## 1. 当前结论

历史联调中，机器人不稳定的首要根因是板载 Jetson 的系统级 CPU 饱和，造成 GR00T
`rt/lowcmd_rl` 周期失约和动作陈旧；不是等待阶段的操控直接锁腰，也不能由最终
`rt/lowcmd` 高频重复旧目标来补偿。

负载由多个结构性来源叠加：

1. 多个 CycloneDDS participant 重复接收同一高频状态流，原生 `recvMC` 线程即使在 Python
   主线程等待时仍持续消耗 CPU。
2. 导航相机在同一进程内把 RealSense 图像发布到 ROS，再由定位和局部规划重复订阅、同步、
   解码和复制；低频 VPR 与已禁用的 NavDP 也承担了持续 RGB-D 路径成本。
3. 操控入口在收到抓取请求前就初始化 mover、DDS writer 和 `rt/lowstate` subscriber，
   所谓“等待”并不冷。
4. 导航、键盘和操控通过 HTTP/文件各自写入同一个 JSON 命令，既有线程/文件开销，也没有
   可验证的 source、lease、priority 和唯一所有者。
5. ONNX Runtime 两个 session 的默认线程池和 spinning 会放大调度争用。
6. 实时 JPEG、Matplotlib/GIF、回放落盘等诊断路径在演示主链上持续运行。

因此，重构目标不是给某个循环加 `sleep`，而是减少常驻参与者和数据副本、建立唯一写者与租约，
把操控真正变成冷入口，并用控制周期健康度阻止过载时继续输出非零运动。

## 2. 根因证据

### 2.1 Jetson 历史负载快照

> **以下全部为 2026-07-28 11:13 左右的历史值，不代表 `g001` 当前状态。**

| 项目 | 历史观察值 |
| --- | --- |
| CPU | 8 核在线，约 1.50 GHz，25W 模式 |
| load average | 15.64 / 15.79 / 13.94 |
| run queue | 约 11..19 |
| CPU idle | 约 7..8% |
| swap used | 约 1.2 GiB |
| 温度 | 当时正常 |
| GPU | 当时不是主要瓶颈 |

历史进程 CPU 快照：

| 进程 | 历史 CPU |
| --- | ---: |
| GR00T adapter | 127% |
| adapter inference child | 56% |
| `head_camera_ros2_node` | 87% |
| `robot_status_node` | 82% |
| `body_state_relay` | 72% |
| `box_demo_main` | 69% |
| merger | 65% |
| navigation `run_ros` | 55% |

当时的 6 秒 DDS 周期历史采样：

| 指标 | 历史观察值 |
| --- | ---: |
| `rt/lowcmd_rl` | 38.24 Hz，目标接近 50 Hz |
| median dt | 23.55 ms |
| p95 dt | 48.72 ms |
| p99 dt | 73.27 ms |
| max dt | 97.8 ms |
| `>30 ms` gaps | 72 |
| `>40 ms` gaps | 28 |
| `>60 ms` gaps | 5 |
| `rt/arm_sdk` | 约 49.3 Hz |
| 最终 `rt/lowcmd` | 约 270 Hz |

`rt/lowcmd` 的 270 Hz 只是 merger 重发最近一帧 RL 目标。历史采样中
`rt/lowcmd_rl` 最长接近 100 ms 没有新动作，高 kp 会持续保持旧目标，足以破坏步态并让
taptap 的物理回正失败。

### 2.2 DDS 重复接收的历史证据

> **以下同样是历史值。**

`box_demo_main.py` 等待抓取时，Python 主线程约 0.6% CPU，但 CycloneDDS `recvMC`
线程约 66% CPU。对应 UDP 7401 的 Recv-Q 约 325..356 KiB，`enP8p1s0` 约
4200 packet/s、3.3 MB/s RX。

同类历史 `recvMC` CPU：

| participant | 历史 CPU |
| --- | ---: |
| `robot_status_node` | 约 74% |
| `body_state_relay` | 约 63% |
| `box_demo` | 约 66% |
| adapter child | 约 36% |

静态代码路径与该现象一致：旧 `box_demo_main.py` 在 `wait_grasp()` 前创建 DDS participant，
调用 `mover.initialize()`，并由 `ArmNaturalHangKeeper._ensure_dds()` 创建
`rt/arm_sdk` writer 和 `rt/lowstate` subscriber。改变等待循环 sleep 或只把 subscriber
队列从 10 改成 1，不能消除 participant 对多播数据的接收与反序列化。

### 2.3 导航图像路径的静态证据

旧导航直连 RealSense 时，在一个进程内完成：

```text
RealSense 640x480 RGB-D
  -> RGB/Depth ROS topics
  -> Localization 的 RGB+Depth 同步、解码和缓存
  -> ActionPlanner 的 RGB+Depth 同步、解码和缓存
```

代码分析表明：

- VPR 约每 6 秒才调用一次且只使用 RGB，但旧 Localization 每帧转换 RGB 和 depth。
- 历史实际配置中 NavDP 曾关闭，但旧 ActionPlanner 仍持续订阅和转换 RGB-D。
- 直接相机模式同时承担 ROS message 构造、发布、两套时间同步、四次图像转换/缓存。
- 旧 executor 未显式限制工作线程；实时 JPEG 预览、Matplotlib PNG、GIF 和 replay
  进一步制造线程、编码与落盘开销。

按 640x480@30 的数组尺寸估算，单路原始 RGB 约 27.6 MB/s，`float32` depth 约
36.9 MB/s；两组 RGB-D 消费者的逻辑 payload 可达到约 129 MB/s。该数值是
**代码尺寸估算，不是 Jetson 总线实测**，用途仅是说明重复复制的量级。

历史 handoff 还记录过 `head_camera_ros2_node.py --fps 15 --capture-hz 15
--publish-hz 15` 约 87% CPU；这也是历史进程快照，必须在新版本上重新 profile。

### 2.4 腰部与 taptap 的边界

历史等待/导航阶段的 DDS 样本中：

- `arm_sdk` weight = 1.0，`dq = 1.0`，owner marker = `0x4B48`，表示 H arms-only。
- arm waist q/kp/kd 为 0。
- 最终腰与 RL 腰最大 q 差约 0.00014 rad。

这些都是历史值，只支持“当时等待阶段没有直接锁腰”。真正抓取时可以合法进入 full-body
`arm_sdk`，必须重新记录 owner、模式与最终腰误差，不能外推等待阶段结论。

历史日志也表明 taptap 状态机进入过 `verifying`、`retry_started`、failed/blocked；
它不是被关闭，而是在控制周期抖动、状态采样延迟和后续指令取消下执行退化。重构不得通过禁用
taptap 掩盖 deadline miss。

### 2.5 5080 ORT 离线微基准

在 5080 上，用两个默认 ONNX Runtime session 做过隔离小基准：

| 配置 | 5080 离线观察 |
| --- | --- |
| ORT 默认 session 设置 | 新增约 47 个线程，进程约消耗 9.3 个 CPU 核 |
| `intra=1`、`inter=1`、sequential、关闭 spinning | 空载趋近约 1 个新增线程，CPU 可忽略 |

这个对比支持一个有限但重要的判断：**默认 ORT 线程池和 spinning 是结构性调度争用源，应该显式
约束。**

它不支持以下推断：

- 不能据此预测 Jetson 上会减少多少 CPU。
- 不能据此预测 Jetson 的单次推理延迟或 50 Hz 控制余量。
- 不能证明 DDS、相机和操控同时运行时仍能满足 deadline。
- 不能证明新配置不会影响策略输出或真机步态。

原因是该测试运行在 5080，模型、输入和板载 Jetson 不同，而且没有包含 Jetson 的 DDS 回调、
相机、merger、导航、操控和系统调度竞争。部署后必须在同一 Jetson、同一模型、同一阶段重新
比较线程数、CPU、推理延迟和 `rt/lowcmd_rl` 周期。

## 3. 旧运行拓扑

```text
navigation run_ros
  -> localhost HTTP bridge
  -> /tmp/robojudo_ext_cmd.json

keyboard ------------------------\
box demo GrootMover --------------+-> 同一个 JSON，多个 last-write-wins 写者
HTTP bridge ---------------------/

/tmp/robojudo_ext_cmd.json
  -> GR00T adapter + 两个默认 ORT sessions
  -> rt/lowcmd_rl

keyboard arm hang / box arm keeper / dual-arm controller
  -> 各自 DDS participant
  -> rt/arm_sdk

rt/lowcmd_rl + rt/arm_sdk
  -> merger
  -> rt/lowcmd
  -> G1 MCU

RealSense -> raw ROS RGB-D
  -> Localization subscriber
  -> ActionPlanner subscriber

box_demo_main 启动
  -> 提前初始化 mover、DDS、lowstate subscriber
  -> 再等待抓取请求
```

旧拓扑的核心问题是：命令、状态和图像都存在重复所有者或重复接收者；控制语义靠人工约定，
不能由运行时证明。

## 4. 候选新运行拓扑

```text
navigation / DDS-free keyboard / manipulation GrootMover
  -> Unix datagram command + source + sequence + lease
  -> MotionCommandBroker
       - priority/lease 仲裁
       - DAMP/LIMP latch
       - adapter + merger composite health gate
       - per-launch runtime ID，拒绝旧心跳文件
       - 20 Hz bounded output
       - producer 进程实例独立 sequence 空间
       - wire timestamp + monotonic age，积压旧帧不会重新变 fresh
  -> /tmp/groot_adapter_command.sock  [latest-only]
  -> /tmp/groot_merger_command.sock   [latest-only]

/tmp/groot_adapter_command.sock
  -> GR00T adapter                    [rt/lowcmd_rl 唯一写者]
       - torch threads = 1
       - BLAS/OpenMP/OpenCV native threads = 1
       - ORT intra/inter = 1/1
       - sequential, spinning off
       - control-loop health reporter
       - p99 + 单次 60 ms gap 双门槛
  -> rt/lowcmd_rl

/tmp/groot_merger_command.sock
  -> merger 腰运动窗口仲裁 + consumer health heartbeat

DDS-free keyboard H
  -> /tmp/groot_arm_control.sock
  -> merger 内置 ArmHangPlanner

抓取子进程（仅任务期间，无 DDS participant）
  -> /tmp/groot_arm_runtime.sock
       - source/token/sequence/deadline/<=250 ms lease
       - latest-only upper-body state/policy snapshot
  -> merger 内置 ManipulationArmBroker
       - 唯一读取 rt/lowstate 与 rt/lowcmd_rl
       - 服务端 policy 对齐、gain/weight fade 和 fail-closed 释放
       - 统一 mode/gain/CRC

rt/lowcmd_rl + optional external rt/arm_sdk + integrated H
  -> merger                           [rt/lowcmd 唯一写者]
       - bus DAMP/LIMP 合成为最终全身安全帧并绕过全部 overlay
       - 命令面失联、health 失败或 tick gap 超限时直接合成 DAMP
       - worker exception/stall、SIGTERM/SIGHUP 退出均 best-effort 发布最终 DAMP
       - RL stale 默认 0.12 s 后停止重发
  -> rt/lowcmd
  -> G1 MCU

cold manipulation HTTP ingress
  -> idle: 无 DDS、相机、IK、mover
  -> accepted request: 至多一个隔离子进程

RealSense / external ROS image
  -> navigation CameraFrameHub        [latest-only]
       -> VPR 触发时才取 RGB
       -> 保留 depth 采集，NavDP 请求时才转 float32 米制
       -> raw ROS publish 保留给进程外兼容消费者
```

导航候选默认还包括：

- 保留原相机 640x480@30 FPS、RGB-D 采集和 ROS 话题；内部 VPR/NavDP
  直接读取 FrameHub，不再通过 ROS 回环解码。
- ROS executor 显式限制为 2 线程。
- RealSense `uint16` depth 到米制 `float32` 的转换延迟到 NavDP 请求时。
- 实时 JPEG frame pipeline、GIF、VPR 图片落盘和 replay 默认关闭并延迟导入。
- 操控视觉落盘默认关闭；夹持检测保留，Matplotlib/PNG 仅在显式诊断选项下加载。

重要边界：

- `CameraFrameHub` 当前只消除了导航进程内部的图像回环和双订阅；它还不是导航与 box demo
  共用的系统级 camera broker。导航与操控当前配置使用不同相机序列号，不能在未核实物理
  设备和数据需求前强行合并。
- 候选抓取路径默认使用 arm runtime，不创建操控 DDS participant。原 `rt/arm_sdk` 路径由
  `--legacy-arm-dds` 显式保留用于 A/B。candidate arm runtime 与 legacy overlay 是互斥模式：
  runtime 开启时 legacy 帧只用于冲突观测，永不延迟接管；legacy A/B 必须显式关闭 runtime socket。
- motion bus 的 JSON 命令文件在 B 中为 `null`，adapter 和 merger 不再执行 50/20 Hz 文件打开、
  解析或原子 rename。`/tmp/robojudo_ext_cmd.json` 只在显式 legacy 回退时使用。
- `robot_status_node`、`body_state_relay` 等非候选运行时进程的去重尚未完成；它们的代码与
  当前状态必须等 `g001` 可达后重新读取。候选启动前应把额外高频 participant 纳入审计。
- 当前新拓扑只通过本机离线测试，不能写成已在 Jetson 降载成功。
- motion bus、arm runtime 和导航 launcher guard 都是本地同一 Unix 账号内的运行时约束，
  不是针对同账号恶意进程的密码学认证。可写 socket/status/环境变量的同账号进程仍在
  信任边界内；现场部署必须使用受控账号、固定路径和最小权限。

## 5. 单一所有者约束

| 资源 | 必须的唯一所有者 | 候选实现 | 违反时的处理 |
| --- | --- | --- | --- |
| base command lease | 同一时刻一个 active source | motion bus 按 safety > operator > manipulation > navigation 仲裁 | 拒绝旧 sequence；lease 超时回退；冲突记录状态 |
| base command stream | 一个 `MotionCommandBroker` | broker 直发 adapter/merger 两个 latest-only UDS；B 不写 command JSON | endpoint 用 `flock` 独占；第二 broker/consumer 失败且不能删除现有 socket |
| `rt/lowcmd_rl` | GR00T adapter | merger 和其他工具只订阅 | launcher 用已知进程名做启动前拒绝；尚未做 DDS discovery 级别的运行时强制 |
| `rt/lowcmd` | merger | adapter 只发 `rt/lowcmd_rl` | launcher 启动前拒绝已知 writer；第二 publisher 的运行时发现仍是未完成项 |
| integrated H arms-only | merger 内 `ArmHangPlanner` | keyboard 只发 UDS 请求，不创建 DDS | external `arm_sdk` 活跃时立即释放 integrated H |
| manipulation upper body | merger 内 `ManipulationArmBroker` | box 子进程通过实例化 source、UDS token/lease 持有所有权；merger 统一生成最终命令 | stale state/policy、旧 sequence/token、NaN、并发 owner 均 fail-closed；client 等服务端交权完成才返回 |
| legacy `rt/arm_sdk` | 仅 A/B 回退时一个 producer | `--legacy-arm-dds` 显式选择 | 与 arm runtime 冲突即拒绝新 owner 并记录状态 |
| manipulation job | 一个活动子进程 | 冷 HTTP 入口对并发返回 BUSY | 取消/超时按进程组回收；清理失败不接新任务 |
| navigation camera frame | 一个 `RGBDClient` ingress | `CameraFrameHub` latest-only | 禁止 Localization/ActionPlanner 自建重复订阅 |

motion bus lease 不是运动授权。它只解决已授权运行时内部的所有权；没有现场人员明确许可时，
任何 source 都不得发送真机运动命令。

## 6. 真机安全锁

### 6.1 不可绕过的规则

1. 真实 G1 必须有人在环，现场必须有人可立即急停和支撑机器人。
2. 未获得用户当次明确允许，不得启动、重启或向真机发送任何控制命令。
3. 本机离线测试、静态检查、`--print-config` 不构成真机启动许可。
4. 未完成当次远程重读时，不得沿用 handoff 的 PID、进程、DDS、哈希或姿态快照。
5. 疑似右膝问题未完成机械/电气检查前，不得通过行走测试验证硬件。

### 6.2 启动器锁

候选 `box_demo_groot/start_g1_onboard_runtime.sh`：

- 默认只检查配置，缺少 `--human-approved-control-start` 时拒绝启动。
- `--preflight-only` 必须在不创建控制进程的情况下退出。
- 检查 DDS 网卡地址、MCU 可达性、依赖、已有控制进程和已有 tmux session。
- 检测到任何旧 runtime/control 进程时拒绝重叠启动。
- 默认限制 torch/BLAS/ORT 线程；不允许 ORT 线程数为 0 的自动模式。
- base broker、adapter stream、merger stream、arm runtime 和 integrated H endpoint 都有独占锁；
  重复实例不能通过 unlink 抢走正在运行的 socket。
- merger 同时托管 arm runtime；操控子进程不再创建 DDS participant 或计算最终 CRC。
- 操控入口默认关闭视觉图片落盘和夹持 Matplotlib 绘图，诊断时才显式开启。
- broker 身份、PID、runtime ID、路径和 socket 验证成功后才创建 merger；merger
  heartbeat/endpoints 初始化后才创建 adapter；三者连续健康达到稳定窗口后才开放键盘和
  操控入口。任一阶段失败只回收本次新建的 tmux session；损坏 safety journal 会在任何
  DDS 控制组件启动前阻断。

候选 `nav_uat_overlay/start_nav_refactored.sh`：

- 要求 motion bus socket 存在。
- 要求 bus status 在 2 s 窗口内、adapter 和 merger health 都为 healthy、两者携带
  当次 runtime ID，且没有 latched DAMP/LIMP。
- 要求 `repo_root`、broker input socket 与两个 output socket 都属于预期 B 目录，并要求
  `command_file` 为 `null`；旧版或 A/B 混合 runtime 会被拒绝。
- 默认把 OpenMP、MKL、OpenBLAS、NumExpr、VECLIB 和 OpenCV native threads 统一限制为 1。
- 缺少 `--human-approved-control-start` 时拒绝启动导航。
- 检测已有 `run_ros.py` 时拒绝重复启动；`run_ros.py` 自身还要求 launcher 授权环境变量
  并全程持有非阻塞 `flock`，直接 `python run_ros.py` 会在导入 ROS/相机前退出。

操控冷入口自身默认 `allow_execute=false`。候选统一启动器只有在整个 runtime 已通过显式现场授权后
才给本地冷入口启用执行能力；不得把允许执行的入口暴露到非本机网络或无人值守启动项。

### 6.3 运行时运动门

motion bus 要求 adapter 控制周期和 merger 最终消费者的心跳都新鲜、healthy 且 runtime ID
匹配，才允许非零运动。DAMP/LIMP 是锁存安全状态，
不能因 lease 自然过期而自动消失。锁存建立时会清空全部普通 lease；解除锁存后保持零速
`RL_FULL`。在安全 epoch 中出现的 source 保持 blocked，清锁后也不接受其周期命令；
该进程必须在清锁后先显式 `release`，或由现场人员停止并重启产生新的实例 source。clear 后
尚未 release 的 blocked source 以 `cleared` tombstone 持久化；broker 在这段窗口重启也不会绕过隔离。

adapter 在策略 observation/inference 之前处理 DAMP/LIMP；DAMP 缺少 lowstate 或发布失败时
降级为不依赖状态的 LIMP。merger 不依赖 adapter 及时产生安全 RL 帧：它会根据 motion
stream 的确切 DAMP/LIMP 直接覆盖缓存的刚性 RL 帧，并把 command-plane/health/timing 故障
转为最终 DAMP。

现场确认原因已排除后，操作工具要求显式双重意图：

```bash
python -m onboard_runtime.runtime_ctl clear-safety \
  --human-approved-safety-clear
```

这个 CLI flag 是防误操的双重意图提示，**不是 broker 可验证的人员身份授权**；
broker 的信任边界仍是能写本地 Unix socket 的同机账号/进程。该命令只解锁，随后仍为零速
`RL_FULL`，且被隔离的 source 还要先 release/restart。Codex 不得自行执行清锁或重新武装。

handoff 给出的建议门槛是：

```text
rt/lowcmd_rl >= 45 Hz
p99 control dt < 35..40 ms
不持续出现 >60 ms 空窗
CPU idle >= 20%
DDS 关键 socket 无持续积压
```

这些是待现场确认的验收门槛，不是当前实测结果。任何一项不满足时应保持零速、保留数据并停止
推进测试；不得在 deadline 已失效时依赖 taptap 继续补救。

## 7. A/B 部署方法

### 7.1 版本隔离

定义两个完全独立的目录或 git worktree：

- **Variant A，legacy baseline**：以 `baseline/live-demo-20260728`（历史快照 commit
  `49f2631`）为候选起点，实际部署前重新核对现场当前可用版本和关键文件哈希。
- **Variant B，refactor candidate**：`refactor/onboard-runtime` 的最终审阅 commit。部署时只允许
  使用本文对应的 clean commit；最终 commit ID 由交付记录填写，不能部署临时 dirty worktree。

不得在真机目录上来回 checkout dirty 分支。A/B 各自保留：

- commit、dirty status 和关键文件 SHA-256。
- ONNX 模型、配置、地图、conda environment export。
- 启动参数、环境变量、Jetson power mode、CPU/GPU 频率。
- 对应的日志和测量输出目录。

Variant A 与 B 不得同时运行。切换版本必须：

1. 由现场人员明确停止当前 variant。
2. 确认所有控制进程、tmux、Unix socket、HTTP bridge 和 DDS publisher 已退出。
3. 确认机器人处于受支撑的安全零速状态。
4. 再从另一个独立目录启动下一 variant；禁止热切换控制发布者。

旧入口保留用于 A：

- `box_demo_groot/start_g1_onboard_nav_taptap.sh`
- 现有 `box_demo_main.py` / legacy HTTP 与文件 IPC

候选新入口用于 B：

- `box_demo_groot/start_g1_onboard_runtime_taptap.sh`
- `nav_uat_overlay/start_nav_refactored.sh`

这些路径只是版本映射。只有现场授权操作者可以使用真机启动参数，Codex 不得自行执行。

### 7.2 保持变量一致

A/B 比较必须固定：

- 同一台 `g001`、同一电池区间和电源模式。
- 相同 CPU/GPU 频率策略、风扇和温度起点。
- 同一 DDS 网卡、域、模型文件、stand height、速度上限和 taptap 配置。
- 相同导航地图、VPR/NavDP 开关、相机分辨率与帧率。
- 相同手臂姿态、负载、地面、支撑和操作员。
- 相同阶段持续时间与命令序列。

建议 A-B-A 或 B-A-B 交错运行并至少重复 3 次，避免温度、电池和现场顺序偏差。每次切换前留出
零速稳定和温度记录窗口。

## 8. A/B 测量清单

### 8.1 每个 variant 的阶段

先做零动作资源测试，满足门槛后才逐级增加功能：

| 阶段 | 负载 | 是否允许运动 |
| --- | --- | --- |
| S0 | merger + adapter，零命令 | 否 |
| S1 | 导航 + 运控，先零命令 | 门槛通过后仅小速度 |
| S2 | 操控冷入口 + 导航 + 运控 | 与 S1 相同；冷入口不应初始化 DDS/相机/IK |
| S3 | 相机/VPR/NavDP/SAM3 对齐 + 运控 | 底盘先保持零速 |
| S4 | 底盘零速 + 双臂抓取 | 只在 owner/腰交权确认后 |
| S5 | 抓取释放、交权完成、恢复导航 | 先零速验证，再小速度 |

S0-S5 全部通过后，才进入项目验收动作（1 m/s 行走、搬箱深蹲、1 m 半径圆弧）。验收动作不是
本次 CPU 重构的首轮测试。

### 8.2 系统资源

每阶段至少记录稳定窗口和状态转换窗口：

- 每进程和每线程 CPU，特别是 `recvMC`、adapter inference、相机、导航、merger。
- load average、run queue、CPU idle、每核利用率和频率。
- 内存、RSS/PSS、swap in/out。
- Jetson power mode、功耗、温度、降频状态和 GPU 利用率。
- 网卡 packet/s、bytes/s、CycloneDDS UDP Recv-Q/Send-Q。
- 进程数、线程数、DDS participant 数。
- ORT session 数、线程数、inference mean/p95/p99/max。

### 8.3 控制时序

- `rt/lowcmd_rl` 实际 Hz、median/p95/p99/max dt。
- `>30 ms`、`>40 ms`、`>60 ms` gap 次数。
- adapter compute p95/p99、overrun ratio、last success age 和 health reason。
- `rt/arm_sdk` 与最终 `rt/lowcmd` Hz，但不得把高频重发当作新鲜 RL 动作。
- lowstate age、policy age、merger RL age、arm lease age、最终发布空窗。
- motion bus output Hz、active source、lease age/expiry、rejected/expired count。

### 8.4 所有权与动作质量

- Variant A 的 `/tmp/robojudo_ext_cmd.json` writer PID 必须唯一；Variant B 必须没有 command-file
  热路径，status 中 `command_file=null`，两个 stream socket 都存在且只有一个 owner。
- `rt/lowcmd_rl`、`rt/lowcmd` publisher 必须分别唯一。
- active base source、priority、sequence、lease 和 safety latch。
- arm runtime boot id、owner/token 生命周期、applied sequence、lease expiry、release state；
  legacy A/B 时另记 arms-only/full-body、weight、marker。
- 最终腰与 RL 腰 q/kp/kd/tau 误差及交权连续性。
- taptap 状态、触发、取消、retry、blocked 及对应控制 gap。
- 足宽、stagger、相对足 yaw、base height、roll/pitch。
- 机器人步态、抖动、滑步、膝关节跟踪误差和人工观察视频时间戳。

### 8.5 重构验收条件

至少满足：

1. Variant B 的 `导航 + 操控冷等待` 相比 B 的 `仅导航`，新增 CPU 目标小于 5%。
2. B 不再出现 base command JSON 热路径；三个 producer 只写 motion bus，broker 只发两条 UDS。
3. B 的 keyboard H 不创建额外 DDS participant。
4. B 的 box 操控默认路径不创建 DDS participant；merger 是状态、policy、gain 和 CRC 的唯一所有者。
5. B 的 VPR/NavDP 不再各自持续订阅同一进程发布的 raw RGB-D。
6. B 的 ORT session 使用显式线程上限、sequential 和 spinning off，并在 Jetson 上重新测量。
7. B 在 S0-S5 均满足控制周期门槛，且没有新的 owner 冲突或交权跳变。
8. 相同阶段下，B 的 CPU idle、run queue、`rt/lowcmd_rl` p99 和大 gap 数显著优于 A。
9. 任何性能收益都不能以禁用 taptap、降低安全检查或跳过腰/手臂交权为代价。

当前尚不能声称这些条件已满足；本轮无法通过受管网络连接 5080/g001，
且没有做真机验证。

## 9. 数据记录模板

每次运行建议保存一份元数据：

```text
variant:
commit:
dirty:
key_file_hashes:
model_hashes:
date/time:
operator:
robot:
battery_start/end:
power_mode:
cpu/gpu_frequency_policy:
temperature_start/end:
dds_iface/domain:
launcher/config:
stage:
command_sequence:
duration:
human_approval_reference:
stop_reason:
```

结果文件统一使用单调时钟和墙钟，保证 CPU、DDS、owner、视频和机器人姿态可以对齐。不得只保存
平均 CPU；本问题的核心是 deadline tail，必须保留原始周期样本和 p95/p99/max。

## 10. 当前未完成项

- 在具备只读远程访问能力后重新读取 5080/g001 状态、代码哈希和真机进程拓扑。
- 实现 DDS discovery 或等价的运行时 publisher 身份/数量监测；当前 `pgrep` 只是启动前
  best-effort 防重，不能证明 `rt/lowcmd_rl` / `rt/lowcmd` 全程唯一写者。
- 读取 `robot_status_node`、`body_state_relay` 的实际代码与消费者后，决定合并、降频或替换；
  未确认契约前不能直接删除。
- 分别核实导航与 box demo 的物理相机、帧需求和生命周期；只有确认共享同一数据源时才建立
  系统级 camera broker。
- 在真机上验证 arm runtime 的 state/policy freshness、lease expiry、异常退出保持和
  server-side policy 对齐；同时保留一次 legacy `rt/arm_sdk` 对照。
- 在 Jetson 上重做 ORT 默认/受限配置、相机和完整 S0-S5 负载基准。
- 完成右膝机械、电气与状态检查后，才讨论地面行走验证。
- 如需把同账号进程排除在信任边界外，需为 motion bus/consumer 引入可验证 peer
  credential、固定 broker 身份或独立服务账号。当前 source/token/runtime ID 主要防误用、陈旧实例
  和竞争，不能抵抗有意伪造的同账号进程。

## 11. 离线验证记录

本轮没有启动 DDS、ROS、相机或真机控制。已完成的纯离线结果：

- 候选离线矩阵合计 `243 passed`：`onboard_runtime` 62、`box_demo_groot` 117、导航核心 40、
  `box_demo_2` 离线单元 20、WBC 线程约束 4（另含 5 个 subtests）。覆盖启动屏障、backlog
  timestamp、clear/restart/release tombstone、最终 DAMP/LIMP、单次 60 ms gap、worker/signal
  退出、stop/pause/preempt 竞态、arm owner/lease、冷操控生命周期和相机退避。
- 另有 13 项在当前受管沙箱统一停在 `AF_UNIX/TCP socket.bind(...): Operation not permitted`
  （onboard 9、merger 1、导航 2、box HTTP 1）；逐项堆栈均停在本地 bind 或其直接后果，
  没有业务断言回归。两支 `box_demo_2/test` 真机手臂脚本因本机无 `unitree_sdk2py` 不纳入离线
  pytest 收集；导航 visualization 全集因本机无 ROS `builtin_interfaces` 未运行。
- Python `py_compile`、三个 shell 启动器 `bash -n`、`git diff --check` 均通过。

这些结果只能证明协议和离线生命周期，不证明 Jetson CPU、DDS deadline 或步态已经改善；最终
结论必须来自同一台 `g001` 上的 A/B S0-S5 数据。
