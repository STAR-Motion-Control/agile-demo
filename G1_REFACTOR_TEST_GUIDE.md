# G1-001 新版运行时测试指引

本文用于测试独立部署在以下目录的新版代码：

```text
/home/unitree/releases/agile-demo-refactor
```

新版部署标签为 `g001-runtime-refactor-v1`。目录名不使用 Git 提交哈希；Git 提交号仅作为内部完整性记录。

本文中的控制启动、停止、运动、导航和操控命令都只能由现场人员在机器人旁、完成安全检查并对当次操作明确授权后执行。代码同步、Git 校验和只读状态检查不代表已经获得控制授权。

## 1. 不可突破的安全边界

1. 机器人必须使用吊架或可靠支撑，急停操作员必须能立即触达急停。
2. 机器人周围清场，测试人员不得位于机器人可能跌倒、迈步或手臂扫过的范围内。
3. 旧版和新版不得同时运行。任何时刻只能有一个 `rt/lowcmd` 最终发布者。
4. 未获得当次明确授权，不得启动、重启、停止或向真机控制链发送命令。
5. `space` 和 `/nav/stop_cmd` 只是零速度请求，不是硬件急停；异常时使用现场急停或键盘 `o` 的 DAMP 锁存。
6. DAMP/LIMP 锁存后先停止测试并查明原因。不得删除 `/tmp/groot_motion_safety_latch.json`，不得为了继续测试而绕过安全门。
7. 出现步态不稳定、异常关节声、支撑松动、通信中断、健康门失败或 CPU 持续饱和时，立即终止当前阶段。

## 2. 版本和目录关系

新版只使用：

```text
/home/unitree/releases/agile-demo-refactor
```

以下现场旧目录必须保持原样，禁止覆盖、移动、修改或在其中切换分支：

```text
/home/unitree/zihou/box_demo_1
/home/unitree/zihou/box_demo_2
/home/unitree/zihou/GR00T-WholeBodyControl
/home/unitree/workspace/nav_uat
```

新版关键入口：

```text
运控：box_demo_groot/start_g1_onboard_runtime_taptap.sh
导航：nav_uat_overlay/start_nav_refactored.sh --config config_g001
操控：box_demo_2/box_agent_tools_server.py
```

`config_g001` 是 G1-001 首轮兼容配置。它保留现场外置相机话题、5 FPS 配置、原导航速度、开环行为、现场地图和回放功能，只把旧 HTTP/JSON 运动命令链替换为新版 motion bus。

## 3. 同步后只读验收

这一节不启动任何控制脚本，可以在没有运动授权时执行。

```bash
cd /home/unitree/releases/agile-demo-refactor
pwd
git status --short --branch
git describe --tags --exact-match
git log -1 --oneline
```

预期结果：

- `pwd` 为 `/home/unitree/releases/agile-demo-refactor`。
- 当前分支为 `runtime-refactor-v1`。
- 当前标签为 `g001-runtime-refactor-v1`。
- `git status --short` 没有源码修改。ONNX 模型受 `.gitignore` 管理，不应形成源码 dirty 状态。

确认新目录不是旧目录的软链接：

```bash
readlink -f /home/unitree/releases/agile-demo-refactor
readlink -f /home/unitree/zihou/box_demo_1
readlink -f /home/unitree/workspace/nav_uat
```

三个结果必须不同。

确认 G1-001 现场地图和模型完整：

```bash
sha256sum nav_uat_overlay/src/map/sustech_demo_g001/config.yaml
sha256sum nav_uat_overlay/src/map/sustech_demo_g001/label.json
sha256sum nav_uat_overlay/src/map/sustech_demo_g001/map.png
sha256sum decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx
sha256sum decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx
```

预期 SHA-256：

```text
config.yaml  7b698cded33f02f9693cac1f5682b322aafd0e12f7b9220f6bcac7c5af19a6e9
label.json   079b7a84ec6cbcf7f78006ccd20fcf7cfb3619ec00070d57aaff4e47d7fa2f34
map.png      720c33d10b6901b82106ce7f1d8a6d3cf71a86c169b158ddc476f7a03b18c68d
Balance ONNX f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18
Walk ONNX    7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407
```

最后确认旧版仍存在、当前进程没有因同步而变化：

```bash
date
tmux ls
pgrep -af 'start_g1_onboard|motion_bus|merge_lowcmd|groot_wbc|run_ros.py|box_demo_main.py|box_agent_tools_server.py'
ps -eo pid,ppid,stat,pcpu,pmem,nlwp,cmd --sort=-pcpu | head -30
```

只记录输出，不停止任何进程。

2026-07-28 19:15 的只读快照仅供识别现场状态：当时只有旧版 `/home/unitree/workspace/nav_uat/src/run_ros.py` 在运行，来自 VS Code 终端，使用 conda `nav`、ROS 2 Humble 和 `enP8p1s0`；约 103% CPU、51 个线程。5001 无监听，GR00T adapter、merger、motion bus 和操控入口均未运行。这个快照会过期，测试前必须重新执行上面的命令。

Unitree 系统 tmux 中的 `robot_status`、`vision_node`、`zenoh_bridge`、音频和其他系统服务不属于旧版 demo 控制 session，不能因为切换 demo 而停止。

## 4. 建立测试记录

在新版目录创建一次测试的独立记录目录：

```bash
cd /home/unitree/releases/agile-demo-refactor
export TEST_RUN_ID="$(date +%Y%m%d_%H%M%S)"
export TEST_RESULT_DIR="$PWD/test_results/$TEST_RUN_ID"
mkdir -p "$TEST_RESULT_DIR"
date -Ins | tee "$TEST_RESULT_DIR/start_time.txt"
git status --short --branch | tee "$TEST_RESULT_DIR/git_status.txt"
git log -1 --format=fuller | tee "$TEST_RESULT_DIR/git_commit.txt"
uname -a | tee "$TEST_RESULT_DIR/uname.txt"
```

记录电池、Jetson power mode、频率策略、风扇、温度、机器人负载和地面条件。A/B 对比必须保持这些条件一致。

## 5. 记录现场旧状态和可比基线

只有在机器人已安全支撑且旧版处于零速度时采集。不要为了采样重启旧版。若现场只有导航、没有 adapter/merger/操控，这一轮只能记作“导航单进程基线”，不能与新版三模块联合负载直接比较。

```bash
ps -eLo pid,tid,psr,pcpu,stat,comm,args --sort=-pcpu | head -80 \
  | tee "$TEST_RESULT_DIR/legacy_threads.txt"
vmstat 1 30 | tee "$TEST_RESULT_DIR/legacy_vmstat.txt"
timeout 30 tegrastats --interval 1000 \
  | tee "$TEST_RESULT_DIR/legacy_tegrastats.txt"
```

如果安装了 `pidstat`：

```bash
pidstat -durh -p ALL 1 30 | tee "$TEST_RESULT_DIR/legacy_pidstat.txt"
```

记录旧版当前启动命令、父进程和 tmux pane，不做任何改动：

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}' \
  | tee "$TEST_RESULT_DIR/legacy_tmux.txt"
pgrep -af 'start_g1_onboard|merge_lowcmd|groot_wbc|run_ros.py|head_camera_ros2_node|robot_status_node' \
  | tee "$TEST_RESULT_DIR/legacy_processes.txt"
```

如果 `run_ros.py` 存在，记录它是否来自 tmux、VS Code 或其他终端：

```bash
NAV_PID="$(pgrep -n -f '[r]un_ros.py')"
ps -p "$NAV_PID" -o pid,ppid,lstart,etime,stat,pcpu,pmem,nlwp,psr,args \
  | tee "$TEST_RESULT_DIR/legacy_nav_process.txt"
readlink -f "/proc/$NAV_PID/exe" | tee "$TEST_RESULT_DIR/legacy_nav_exe.txt"
readlink -f "/proc/$NAV_PID/cwd" | tee "$TEST_RESULT_DIR/legacy_nav_cwd.txt"
```

CPU 基线采样期间不要运行 `ros2 node list`、`ros2 topic hz` 或 `ros2 topic echo`；这些命令会创建额外 DDS participant/subscriber，并非严格的被动观测。

## 6. 停止旧版：必须重新获得当次授权

本节会影响真机控制状态。现场人员必须先明确批准“停止旧版并准备切换新版”。

1. 确认吊架、急停、清场和第二名观察员就位。
2. 重新运行第 3 节的进程检查，识别当时真正运行的旧组件和父终端，不能照搬历史快照。
3. 若只有 VS Code 终端中的旧 `run_ros.py`，回到它的原终端按 `Ctrl+C`，等待 ROS shutdown；不要停止 Unitree 系统 tmux 服务。
4. 若现场重新出现完整旧运控，先进入其键盘 pane 按 `space` 并确认零速度，再逐个在导航、操控、键盘、adapter、merger pane 中用 `Ctrl+C` 正常退出。
5. 只有相关控制进程已经退出、旧 demo pane 仅剩 shell 时，才允许清理对应旧 demo tmux session。

```bash
tmux ls
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
```

正常退出后检查：

```bash
pgrep -af 'merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|agile_lowcmd_pipeline.py|run_g1_control_loop|agile_http_ipc_server.py|agile_keyboard_control.py|box_demo_main.py|run_ros.py'
ss -lxnp | grep -E 'groot_motion|groot_adapter|groot_merger|groot_arm' || true
```

第一条应无输出，且不能存在旧版最终控制发布者或旧 `run_ros.py`。不要手工删除 socket 或 safety journal。

若旧版发生异常、无法正常归零或退出，使用现场急停并停止测试，不进入新版启动步骤。

## 7. 新版静态配置与 preflight

以下两条命令都不会创建控制进程，但只能在旧控制链完全退出后继续到 preflight。

```bash
cd /home/unitree/releases/agile-demo-refactor
CONDA_ENV=g1_deploy IFACE=enP8p1s0 SESSION=g1-runtime-refactor \
  bash box_demo_groot/start_g1_onboard_runtime_taptap.sh dwbc \
  --no-manip-ingress --no-attach --print-config
```

核对输出：

```text
refactor_root=/home/unitree/releases/agile-demo-refactor
groot_repo=/home/unitree/releases/agile-demo-refactor
iface/domain=enP8p1s0/0
motion bus output=20 Hz
native/torch/ORT threads=1
ORT execution mode=sequential
ORT spinning=false
health p99 threshold=40 ms
health max gap threshold=60 ms
RL stale threshold=0.12 s
manip_ingress=0
```

执行不启动控制的完整前置检查：

```bash
CONDA_ENV=g1_deploy IFACE=enP8p1s0 SESSION=g1-runtime-refactor \
  bash box_demo_groot/start_g1_onboard_runtime_taptap.sh dwbc \
  --no-manip-ingress --no-attach --preflight-only
```

必须看到 `preflight passed; nothing started`。若检测到旧进程、已有 tmux、网卡地址错误、MCU 不可达或模型缺失，停止并解决原因。新版 preflight 不会自动改网卡，也不会杀旧进程。

## 8. 启动新版运控：必须重新获得当次授权

现场人员明确批准本次新版控制启动后执行：

```bash
cd /home/unitree/releases/agile-demo-refactor
CONDA_ENV=g1_deploy IFACE=enP8p1s0 SESSION=g1-runtime-refactor \
  bash box_demo_groot/start_g1_onboard_runtime_taptap.sh dwbc \
  --no-manip-ingress --no-attach \
  --human-approved-control-start
```

该命令启动一个独立 tmux session：motion bus、唯一 merger、受限线程的 GR00T adapter 和无 DDS 键盘。操控入口在首轮明确关闭。

```bash
tmux list-windows -t g1-runtime-refactor
tmux list-panes -t g1-runtime-refactor:runtime \
  -F '#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
tmux attach -t g1-runtime-refactor:runtime
```

启动后先保持机器人零速度 60 秒，不发送运动命令。

## 9. S0 运控健康检查

在另一终端进入新版环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1_deploy
cd /home/unitree/releases/agile-demo-refactor
python -m onboard_runtime.runtime_ctl status
python -m json.tool /tmp/groot_adapter_health.json
python -m json.tool /tmp/groot_merger_health.json
python -m json.tool /tmp/groot_arm_runtime_status.json
python -m json.tool /tmp/groot_arm_control_status.json
```

必须同时满足：

- bus `healthy=true`、`motion_blocked=false`、`safety_latched=false`、`command_file=null`。
- adapter 和 merger 均为 `healthy=true`。
- adapter、merger 的 `runtime_id` 与 bus `health_runtime_id` 相同。
- adapter `cycle_p99_ms <= 40`，最大周期不超过 60 ms，成功时间持续刷新。
- merger `reason=healthy`、`motion_stream_fresh=true`、`motion_health_ok=true`、`tick_gap_ms <= 60`。
- 零动作观察期间没有 DAMP/LIMP 锁存、owner 冲突或陈旧命令告警。

采集 S0：

```bash
python -m onboard_runtime.runtime_ctl status > "$TEST_RESULT_DIR/s0_bus.json"
cp /tmp/groot_adapter_health.json "$TEST_RESULT_DIR/s0_adapter_health.json"
cp /tmp/groot_merger_health.json "$TEST_RESULT_DIR/s0_merger_health.json"
vmstat 1 60 | tee "$TEST_RESULT_DIR/s0_vmstat.txt"
timeout 60 tegrastats --interval 1000 | tee "$TEST_RESULT_DIR/s0_tegrastats.txt"
```

CPU idle 持续低于 20%、p99 超过 40 ms 或出现超过 60 ms 的控制空窗时，不进入运动测试。

## 10. 键盘小动作回归

本节每组动作都需要现场人员确认后逐项执行，不得一次性自动发送按键序列。

新版键盘 pane 的键位：

```text
w/s       前进/后退
a/d       左移/右移
q/e       左转/右转
z/x       降低/升高基座
c         深蹲高度 0.36 m
r         回到站立高度
space     速度归零并保持平衡
f         RL_FULL
l         RL_LOWER
h         双臂自然下垂
o         DAMP 锁存，仅异常或急停使用
Ctrl+C    键盘退出并释放命令 lease
```

测试顺序：

1. 按 `space`，确认速度为零。
2. 短按一次 `w`，立即按 `space`，检查前向响应和停止。
3. 依次短测 `s`、`a`、`d`，每次动作后都按 `space` 并等待站稳。
4. 依次短测 `q`、`e`，每次只做小角度响应并回零。
5. 在吊架承载下按 `c` 到 0.36 m，只做一次深蹲，再按 `r` 回 0.76 m。
6. 每一步后读取 adapter/merger health；任何一步不稳即停止后续测试。

首轮不做 1 m/s、20 次深蹲或 1 m 半径圆弧验收。这些动作只能在小动作和联合负载门槛全部通过后另行批准。

## 11. 导航启动前检查

导航继续使用现场外置相机进程，`config_g001` 不抢占 RealSense。先确认相机、里程计和 LiDAR 安全话题存在：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
ros2 topic list | grep -E '/camera/captured_(image|depth)|/dog_odom|/safety/lidar_state'
ros2 topic echo --once /safety/lidar_state
```

这些 ROS 检查会临时创建 DDS participant/subscriber，因此只在功能测试阶段执行，不与 CPU 基线采样同时进行。

如果图像、深度、里程计或 LiDAR 状态缺失/陈旧，不启动导航，不通过手工发布 `clear` 绕过安全监控。

基础 runtime 健康后，先执行不启动导航的 preflight：

```bash
cd /home/unitree/releases/agile-demo-refactor
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --native-threads 1 \
  --preflight-only
```

必须看到 navigation preflight passed。该检查会拒绝旧版/新版混合路径、陈旧 health、错误 runtime ID、legacy command file 和重复 `run_ros.py`。

## 12. 启动导航：必须重新获得当次授权

现场人员明确批准本次导航控制启动后，在独立终端前台执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd /home/unitree/releases/agile-demo-refactor
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --native-threads 1 \
  --human-approved-control-start
```

不要直接运行 `python run_ros.py`，也不要手工设置内部授权环境变量。

在第三个已加载 ROS 环境的终端检查：

```bash
ros2 topic echo --once /nav/status
ros2 service call /nav/get_pose std_srvs/srv/Trigger '{}'
```

`/nav/status` 应为 idle，`/nav/get_pose` 应返回 `[x, y, theta]`。定位不可用时不要发送运动命令。

## 13. 导航功能和速度回归

原现场速度必须保持：前进 0.40 m/s、后退 0.20 m/s、横移 0.20 m/s、转向 0.40 rad/s。首轮只发小距离/小角度命令，验证行为，不追求这些速度上限。

每个命令执行前由现场人员确认空间和吊架状态；执行后读取 `/nav/status` 并等待完全站稳。

1. 0.10 m 前进：

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
ros2 topic echo --once /nav/status
```

2. 0.10 m 左移，再做 0.10 m 右移：

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.10, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: -0.10, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

3. 左转 10 度，再右转 10 度：

```bash
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.174532925}}'
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.174532925}}'
```

4. 验证停止接口：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
ros2 topic echo --once /nav/status
```

5. 现场语义点测试。先通过可视化地图确认机器人到目标的路径安全，再一次只测试一个目标：

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: '抓取点'}"
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
```

下一轮单独测试：

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: '放置点'}"
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
```

不要连续发送两个目标。目标完成、失败或停止后都要核对 pose、状态、路径、实际用时和 adapter/merger health。与旧版相同路线相比，中位完成时间建议不增加超过 5%，p95 不增加超过 10%，导航成功数不得下降。

## 14. S1/S2 导航联合负载采集

S1 为导航空闲，S2 为小距离导航运动。每个阶段采集 60 秒：

```bash
ps -eLo pid,tid,psr,pcpu,stat,comm,args --sort=-pcpu | head -100 \
  | tee "$TEST_RESULT_DIR/nav_threads.txt"
vmstat 1 60 | tee "$TEST_RESULT_DIR/nav_vmstat.txt"
timeout 60 tegrastats --interval 1000 | tee "$TEST_RESULT_DIR/nav_tegrastats.txt"
cp /tmp/groot_adapter_health.json "$TEST_RESULT_DIR/nav_adapter_health.json"
cp /tmp/groot_merger_health.json "$TEST_RESULT_DIR/nav_merger_health.json"
python -m onboard_runtime.runtime_ctl status > "$TEST_RESULT_DIR/nav_bus.json"
```

重点记录 `head_camera_ros2_node.py`、`robot_status_node.py`、`run_ros.py`、adapter、merger 的 CPU 和线程数。相机/状态节点仍高负载时要单独记录，它们不属于首轮兼容配置中可以直接改变的现场接口。

## 15. 操控冷入口测试

先保持导航 idle、底座零速度。新版 runtime 是用 `--no-manip-ingress` 启动的，因此 5055 端口应空闲：

```bash
ss -ltnp | grep ':5055' || true
```

在新终端启动不允许执行动作的冷入口。该进程本身不初始化 DDS、相机、IK 或 mover：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1_deploy
cd /home/unitree/releases/agile-demo-refactor/box_demo_2
python box_agent_tools_server.py \
  --host 127.0.0.1 \
  --port 5055 \
  --locomotion groot \
  --iface enP8p1s0 \
  --no-vision-log
```

在另一终端验证：

```bash
curl -fsS http://127.0.0.1:5055/health | python -m json.tool
curl -fsS http://127.0.0.1:5055/status | python -m json.tool
curl -i -X POST http://127.0.0.1:5055/tools/manipulate_object \
  -H 'Content-Type: application/json' \
  -d '{"action":"grasp","item_text":"test box"}'
```

最后一条必须返回 `403 EXECUTION_DISABLED`，且不得出现子进程。冷入口 idle CPU 应接近 0%，线程数不应持续增长。

```bash
pgrep -af 'box_agent_tools_server.py|box_demo_main.py'
ps -eLo pid,tid,pcpu,stat,comm,args --sort=-pcpu | head -80 \
  | tee "$TEST_RESULT_DIR/cold_manip_threads.txt"
```

## 16. 真实操控测试：需要单独授权

只有导航、运控、小动作和冷入口全部通过后，才允许申请真实抓取授权。

1. 在冷入口终端按 `Ctrl+C`，确认 5055 已释放。
2. 现场人员重新确认吊架、手臂范围、目标物和急停。
3. 明确批准本次操控执行后，才可用 `--allow-execute` 重启入口：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1_deploy
cd /home/unitree/releases/agile-demo-refactor/box_demo_2
python box_agent_tools_server.py \
  --host 127.0.0.1 \
  --port 5055 \
  --allow-execute \
  --locomotion groot \
  --iface enP8p1s0 \
  --no-vision-log
```

4. 另一个终端发送一次明确的抓取请求：

```bash
curl -fsS -X POST http://127.0.0.1:5055/tools/manipulate_object \
  -H 'Content-Type: application/json' \
  -d '{"action":"grasp","item_text":"<现场目标物描述>"}' \
  | tee "$TEST_RESULT_DIR/grasp_request.json"
```

5. 从响应取得 `request_id`，轮询：

```bash
curl -fsS http://127.0.0.1:5055/requests/<request-id> | python -m json.tool
```

6. 需要取消时：

```bash
curl -fsS -X POST http://127.0.0.1:5055/requests/<request-id>/cancel \
  -H 'Content-Type: application/json' \
  -d '{}'
```

轮询到 `cancelled`、`failed`、`timed_out` 或 `succeeded` 后才能继续。一次只允许一个操控请求。

## 17. S3/S4 全链路负载测试

按以下阶段逐级测试，前一阶段不通过不得进入后一阶段：

| 阶段 | 运行内容 | 允许动作 |
| --- | --- | --- |
| S0 | motion bus + merger + adapter + idle keyboard | 零速度 |
| S1 | S0 + 导航 idle + 外置相机 | 零速度 |
| S2 | S1 + 小距离导航 | 仅 0.1 m / 10 度 |
| S3 | S2 + 冷操控入口 | 冷入口不允许执行 |
| S4 | 导航 idle + 运控 + 单次真实抓取 | 单独授权 |

每阶段至少保存：

```bash
vmstat 1 60
pidstat -durh -p ALL 1 60
timeout 60 tegrastats --interval 1000
ps -eLo pid,tid,psr,pcpu,stat,comm,args --sort=-pcpu
python -m onboard_runtime.runtime_ctl status
python -m json.tool /tmp/groot_adapter_health.json
python -m json.tool /tmp/groot_merger_health.json
```

验收门槛：

- CPU idle 稳定窗口不低于 20%。
- `rt/lowcmd_rl` 实际频率不低于 45 Hz。
- adapter 控制周期 p99 不超过 40 ms。
- 不持续出现超过 60 ms 的控制空窗。
- merger motion stream 持续 fresh，RL age 不超过 0.12 s。
- 冷操控入口 idle CPU 不超过 5%，不创建 DDS/相机/IK 子进程。
- 相同导航任务成功数不下降，原导航速度配置不变。
- 无 owner 冲突、无陈旧高刚度动作、无步态不稳定。

首轮性能结论必须来自同一真机、相同条件下的 A/B 数据。5080 仿真只能证明功能兼容和线程约束，不能代替 Jetson 真机 CPU 与步态结论。

## 18. 异常处理

出现任一异常：

1. 导航终端发送 `/nav/stop_cmd`。
2. 键盘 pane 按 `space` 请求零速度。
3. 如果仍不安全，按 `o` 发送 DAMP 锁存或使用现场硬件急停。
4. 停止当前阶段，保存 bus/health/tegrastats/进程信息。
5. 不清锁、不重启、不切回旧版，直到原因被现场人员确认。

正常测试流程不执行以下命令。只有查明锁存原因、机器人受支撑且现场人员再次明确授权解锁时才允许：

```bash
python -m onboard_runtime.runtime_ctl clear-safety \
  --human-approved-safety-clear
```

清锁后仍是零速度 RL_FULL；被隔离的命令 source 必须 release 或重启，不能立即恢复运动。

## 19. 有序停止新版

停止本身会改变真机控制状态，必须获得当次明确授权并由现场人员执行。

1. 发送导航停止并等待 `/nav/status` 进入 stopped/idle：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
ros2 topic echo --once /nav/status
```

2. 在导航前台按 `Ctrl+C`，等待 `run_ros.py` 完成 stop 和 ROS shutdown。
3. 查询操控 `/status`。若存在 active request，先调用 cancel 并轮询到终态，再在操控终端按 `Ctrl+C`。
4. 在 runtime 键盘 pane 按 `space`，确认零速度，再按 `Ctrl+C` 释放 keyboard lease。
5. 读取 bus status 中的 `health_runtime_id`，把它填入下方 `<runtime-id>`。逐个核对 PID 和命令行，按 adapter、merger、motion bus 的顺序正常停止；每一步确认前一进程退出后再继续。

```bash
ps -fp "$(cat /tmp/groot_runtime_<runtime-id>_adapter.pid)"
kill -TERM "$(cat /tmp/groot_runtime_<runtime-id>_adapter.pid)"

ps -fp "$(cat /tmp/groot_runtime_<runtime-id>_merger.pid)"
kill -TERM "$(cat /tmp/groot_runtime_<runtime-id>_merger.pid)"

ps -fp "$(cat /tmp/groot_runtime_<runtime-id>_motion_bus.pid)"
kill -TERM "$(cat /tmp/groot_runtime_<runtime-id>_motion_bus.pid)"
```

6. 确认全部控制进程已经退出后，才清理空 tmux session：

```bash
tmux kill-session -t g1-runtime-refactor
```

7. 最终核对：

```bash
pgrep -af 'motion_bus|merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|run_ros.py|box_demo_main.py|box_agent_tools_server.py'
ss -lxnp | grep -E 'groot_motion|groot_adapter|groot_merger|groot_arm' || true
```

不要把 `tmux kill-session` 当作首个停止动作，不要手工删除 socket、health 文件或 safety journal。

## 20. 回滚旧版

只有确认新版全部退出、机器人受支撑且现场人员明确批准重新启动旧版后才能回滚。不得在新版仍有任何控制发布者时启动旧版。

先使用第 5 节记录的现场原命令。2026-07-28 19:15 的实际旧状态只有 VS Code 终端里的 legacy navigation，恢复它时使用原 `nav` 环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd /home/unitree/workspace/nav_uat/src
python3 run_ros.py
```

这条命令也会启动导航进程，仍需现场人员当次明确批准。若回滚目标是旧版完整运控，2026-07-28 已知入口为：

```bash
cd /home/unitree/zihou/box_demo_1
bash start_g1_onboard_nav.sh \
  --taptap \
  --taptap-recovery adaptive
```

回滚前必须重新读取当前文件、进程、tmux、DDS、IPC 和配置状态，不得仅依赖本文的历史记录。旧导航和旧运控也不得彼此错误重叠或在 5001 缺少后端时误判为联合运行成功。

## 21. 结果记录模板

每轮填写：

```text
日期/操作员：
部署标签：g001-runtime-refactor-v1
新目录：/home/unitree/releases/agile-demo-refactor
测试阶段：S0 / S1 / S2 / S3 / S4
电池/功率模式/温度：
现场负载和地面：
CPU idle median/min：
总线程数：
adapter cycle p50/p95/p99/max：
>40 ms / >60 ms gap 次数：
merger tick gap max：
rt/lowcmd_rl Hz：
相机 CPU/线程：
robot_status CPU/线程：
run_ros CPU/线程：
冷操控 idle CPU/线程/子进程数：
导航命令/目标：
导航成功与用时：
操控请求和结果：
是否出现步态不稳定：
是否触发 DAMP/LIMP：
与旧版差异：
结论：通过 / 不通过 / 需要复测
日志目录：
```
