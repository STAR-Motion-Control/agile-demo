# G1-001 新版快速测试手册

新版目录：`/home/unitree/releases/agile-demo-refactor`

首次联调只按本页第 0～9 步顺序执行；任一“必须”条件不满足就停止。这份简版只做最小安全验证：运控、导航、冷操控和 CPU。真实抓取、文本目标、深蹲、DAMP 解锁和完整回滚见 [详细手册](G1_REFACTOR_TEST_GUIDE.md)。

## 0. 安全门

- 必须有吊架或可靠支撑、现场急停操作员，并完成清场。
- 右膝机械/电气复查未由现场人员确认通过前，只能做到第 4 步的零速 health 检查，不得用行走测试验证右膝。
- 旧版和新版绝对不能同时运行。
- 不要整页复制执行，必须逐步确认。
- 停止旧版、启动新版、发送动作和停止新版都需要现场人员当次明确授权。
- `space` 和 `/nav/stop_cmd` 不是硬件急停；异常时使用键盘 `o` 的 DAMP 或现场急停。
- health 失败、CPU 饱和或步态异常时立即停止，不得绕过安全门继续测试。

准备 3 个终端：

```text
A：运控和键盘
B：导航
C：状态检查和测试命令
```

## 1. 检查版本和旧进程（只读）

终端 C：

```bash
ssh unitree@10.33.12.89
cd /home/unitree/releases/agile-demo-refactor
git status --short --branch
git describe --tags --exact-match
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus|[s]tart_g1_onboard'
```

必须确认：目录正确、标签为 `g001-runtime-refactor-v1.1`、源码 clean，并记清楚最后一条命令列出的现场进程。不要直接杀进程。

## 2. 停止旧版（需要授权）

获得切换授权后：

- 只有旧 `run_ros.py`：回到启动它的原终端按 `Ctrl+C`。
- 有旧运控：先在旧键盘按 `space`，确认零速度，再依次正常退出导航、操控、键盘、adapter、merger。
- 不要停止 Unitree 系统的 `robot_status`、`vision_node`、`zenoh_bridge` 等服务。

退出后重新检查：

```bash
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus|[s]tart_g1_onboard'
```

相关旧 demo 进程必须全部消失，否则不启动新版。

## 3. Preflight 和启动运控

终端 A 先执行不会启动控制的检查：

```bash
ssh unitree@10.33.12.89
cd /home/unitree/releases/agile-demo-refactor
CONDA_ENV=g1_deploy IFACE=enP8p1s0 SESSION=g1-runtime-refactor \
  bash box_demo_groot/start_g1_onboard_runtime_taptap.sh dwbc \
  --no-manip-ingress --no-attach --preflight-only
```

必须看到：

```text
preflight passed; nothing started
```

获得新版运控启动授权后：

```bash
CONDA_ENV=g1_deploy IFACE=enP8p1s0 SESSION=g1-runtime-refactor \
  bash box_demo_groot/start_g1_onboard_runtime_taptap.sh dwbc \
  --no-manip-ingress --no-attach \
  --human-approved-control-start
tmux attach -t g1-runtime-refactor:runtime
```

先保持零速度 60 秒，不按运动键。

## 4. 检查运控健康

终端 C：

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate g1_deploy
cd /home/unitree/releases/agile-demo-refactor
python -m onboard_runtime.runtime_ctl status
python -m json.tool /tmp/groot_adapter_health.json
python -m json.tool /tmp/groot_merger_health.json
```

必须同时满足：

```text
bus:     healthy=true, motion_blocked=false, safety_latched=false
adapter: healthy=true, cycle_p99_ms <= 40, cycle_max_ms <= 60
merger:  healthy=true, motion_stream_fresh=true, motion_health_ok=true
```

任一项不满足：不运动、不启动导航，直接执行第 9 步。

## 5. 键盘最小测试（逐项授权）

只有右膝复查已经通过，才进入本节。

终端 A 的 keyboard pane：

1. 按 `space`，确认零速度。
2. 短按一次 `w`，立即按 `space`，等待站稳。
3. 重新执行第 4 步 health 检查。

首轮不测试深蹲、1 m/s 或圆弧。出现步态不稳或异常关节声时立即停止。

## 6. Preflight 和启动导航

终端 B 先加载环境并检查：

```bash
ssh unitree@10.33.12.89
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source /home/unitree/unitree_ros2/install/setup.bash
cd /home/unitree/releases/agile-demo-refactor
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --native-threads 1 \
  --preflight-only
```

必须看到 `navigation preflight passed`。

获得导航启动授权后，在终端 B 前台执行：

```bash
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --native-threads 1 \
  --human-approved-control-start
```

不要直接执行 `python run_ros.py`。

## 7. 导航最小测试（逐项授权）

只有右膝复查已经通过，才进入本节。

终端 C 加载 ROS 环境：

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source /home/unitree/unitree_ros2/install/setup.bash
```

先检查状态和定位：

```bash
ros2 topic echo --once /nav/status
ros2 service call /nav/get_pose std_srvs/srv/Trigger '{}'
```

只有状态正常且 pose 可用时，才逐项执行：

```bash
# 前进 0.10 m
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
ros2 topic echo --once /nav/status
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'

# 左转 10 度
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.174532925}}'
ros2 topic echo --once /nav/status
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
```

每项后重新执行第 4 步 health 检查。本简版不测试文本目标。

## 8. 冷操控和 CPU

保持导航 idle、底座零速度。新开终端，启动禁止执行动作的冷入口：

```bash
ssh unitree@10.33.12.89
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate g1_deploy
cd /home/unitree/releases/agile-demo-refactor/box_demo_2
python box_agent_tools_server.py \
  --host 127.0.0.1 --port 5055 \
  --locomotion groot --iface enP8p1s0 \
  --no-vision-log
```

终端 C 验证冷入口不会执行：

```bash
curl -i -X POST http://127.0.0.1:5055/tools/manipulate_object \
  -H 'Content-Type: application/json' \
  -d '{"action":"grasp","item_text":"test box"}'
pgrep -af '[b]ox_demo_main.py'
```

必须返回 `403 EXECUTION_DISABLED`，且最后一条无输出。本简版不使用 `--allow-execute`，不做真实抓取。

观察 CPU：

```bash
vmstat 1
```

另开终端：

```bash
tegrastats --interval 1000
```

观察 60 秒后按 `Ctrl+C`。最低通过标准：

```text
CPU idle >= 20%
adapter cycle_p99_ms <= 40
没有持续 > 60 ms 控制空窗
冷操控入口 idle CPU < 5%
机器人步态无明显不稳定
```

只有和旧版相同条件对比后，才能判断是否真正改善。

## 9. 正常停止新版（需要授权）

按顺序执行：

1. 终端 C 发送 `/nav/stop_cmd`。
2. 终端 B 按 `Ctrl+C`，等待导航退出。
3. 冷操控终端按 `Ctrl+C`。
4. 终端 A 的 keyboard pane 按 `space`，确认零速度，再按 `Ctrl+C`。
5. 在 tmux 中依次进入 adapter、merger、motion bus pane，各按一次 `Ctrl+C`；前一个退出后再停下一个。
6. 所有进程退出后才执行：

```bash
tmux kill-session -t g1-runtime-refactor
```

最后确认：

```bash
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus'
```

应无新版相关输出。不要手工删除 socket、health 文件或 safety journal。

## 紧急情况

1. 有立即危险时，现场人员直接使用硬件急停；可控时按 `o` 触发 DAMP。
2. 只有确认软件仍响应且没有立即危险时，才补发 `/nav/stop_cmd` 并按 `space`。
3. 停止测试，不清锁、不重启、不回滚，先查明原因。

旧版回滚和 DAMP 解锁只按 [详细手册](G1_REFACTOR_TEST_GUIDE.md) 操作。
