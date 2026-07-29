# G1-001 新版快速测试手册

新版目录：`/home/unitree/releases/agile-demo-refactor`

适用标签：`g001-runtime-refactor-v1.3.1`

本标签已把导航 profile 同步为旧真机 launcher 的默认参数。5080 已验证旧、新 profile 的命令规划和 MuJoCo 路线一致；这不等于真机位移、Jetson CPU 或三模块联合负载已经通过。

## 0. 环境和安全

```text
运控：launcher 自动激活原 Conda，现场不手工激活或填写 CONDA_ENV。
导航：使用 nav 环境。
操控：使用 robojudo_zihou2 环境。
```

- 必须有人在环、可靠支撑、急停操作员和清场。
- 未经当次明确授权，不得停止、启动或重启真机控制进程。
- 旧版和新版不得同时运行。
- 右膝机械/电气复查未通过前，只允许零速检查。
- 有立即危险时使用现场硬件急停；可控时按 `o` 触发 DAMP。
- 不要整页复制执行，必须逐步确认。

## 1. 版本和进程（只读）

```bash
ssh unitree@10.33.12.89
cd /home/unitree/releases/agile-demo-refactor
git status --short --branch
git describe --tags --exact-match
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus|[s]tart_g1_onboard'
```

必须确认标签为 `g001-runtime-refactor-v1.3.1`、源码 clean，并记下现场进程。不要直接杀进程。

## 2. 运控 preflight 和启动

只有相关旧 demo 进程已经由现场人员授权并在原终端正常退出后，才执行：

```bash
cd /home/unitree/releases/agile-demo-refactor
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --preflight-only
```

必须看到 `preflight passed; nothing started`。

获得本次新版运控启动授权后：

```bash
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --human-approved-control-start
```

launcher 会自动激活 Conda 并进入默认 tmux session `g1-onboard-runtime`。先保持零速度 60 秒，不按运动键。

## 3. 运控 health

另一终端执行：

```bash
cd /home/unitree/releases/agile-demo-refactor
/usr/bin/python3 -m onboard_runtime.runtime_ctl status
/usr/bin/python3 -m json.tool /tmp/groot_adapter_health.json
/usr/bin/python3 -m json.tool /tmp/groot_merger_health.json
```

必须同时满足：

```text
bus:     healthy=true, motion_blocked=false, safety_latched=false
adapter: healthy=true, cycle_p99_ms <= 40, cycle_max_ms <= 60
merger:  healthy=true, motion_stream_fresh=true, motion_health_ok=true
```

任一项不满足：不启动导航、不运动。

## 4. 导航 profile 和 preflight

新终端加载导航环境：

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source /home/unitree/unitree_ros2/install/setup.bash
cd /home/unitree/releases/agile-demo-refactor
```

先校验 profile：

```bash
python onboard_runtime/nav_profile.py validate \
  --input /tmp/groot_nav_motion_profile.json \
  --expected-socket /tmp/groot_motion_bus.sock
```

必须看到 `G1-001 navigation profile matches preserved legacy defaults`。该检查会确认旧版的 `precise`、关闭 warmup、速度下限、站高和速度上限；不一致时禁止启动导航。

它校验的是旧真机实际默认值：站高/行走下限 `0.76/0.72 m`，warmup `false/0.0 s/0.15 m/s`，前进/后退/横移/偏航上限 `0.50/0.20/0.30/0.60`，横移巡航 `0.20`，线速度/角速度下限 `0.12/0.10`。不要在现场手工改这些参数。

检查导航输入：

```bash
ros2 topic list | grep -E '/camera/captured_(image|depth)|/dog_odom|/safety/lidar_state'
ros2 topic echo --once /safety/lidar_state
```

图像、深度或里程计缺失时停止。LiDAR 回复只接受 `data: clear`；`blocked`、`stale`、`unknown`、无消息或其他值都不启动导航。

执行导航 preflight：

```bash
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --preflight-only
```

必须看到 profile 校验成功和 `navigation preflight passed`。

## 5. 启动导航和最小动作

获得本次导航启动授权后，在导航终端前台执行：

```bash
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --human-approved-control-start
```

不要直接执行 `python run_ros.py`。

另一导航环境终端先检查：

```bash
ros2 topic echo --once /nav/status
ros2 service call /nav/get_pose std_srvs/srv/Trigger '{}'
```

`/nav/status` 必须同时显示任务 idle、`lidar_safety.state=clear` 和 `lidar_safety.paused=false`，pose 必须可用。不能用启动前的 LiDAR echo 代替这项启动后检查。

保持导航 idle 60 秒，重新执行第 3 节的三个 health 命令，再采样 CPU：

```bash
vmstat 1 60
```

必须仍满足第 3 节 health；忽略 `vmstat` 首个累计行后，后续 `id` 不低于 20%。只有这个 S1 负载门通过、右膝复查通过且现场重新授权，才做下面动作。

前进 `0.10 m`：

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

等待站稳，检查 `/nav/status` 和第 3 节 health，然后发送零速请求：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
```

只有前一步完全正常并再次获得授权，才左转 `10°`：

```bash
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.174532925}}'
```

再次等待站稳、检查 status 和 health，并发送 `/nav/stop_cmd`。

首轮不测试后退、横移、右转、文本目标、连续路线、1 m/s、深蹲或圆弧。

## 6. 冷操控 403

保持导航 idle、底座零速度。操控入口使用 `robojudo_zihou2`：

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate robojudo_zihou2
cd /home/unitree/releases/agile-demo-refactor/box_demo_2
python box_agent_tools_server.py
```

不要添加 `--allow-execute`。另一终端验证：

```bash
curl -i -X POST http://127.0.0.1:5055/tools/manipulate_object \
  -H 'Content-Type: application/json' \
  -d '{"action":"grasp","item_text":"test box"}'
pgrep -af '[b]ox_demo_main.py'
```

必须返回 `403 EXECUTION_DISABLED`，且 `box_demo_main.py` 无输出。该步骤不能证明真机三模块联跑已经通过。

## 7. 正常停止（需要授权）

获得停止授权后按顺序执行：

1. 发布 `/nav/stop_cmd`，确认导航回到 idle。
2. 在导航启动终端按 `Ctrl+C`，确认 `run_ros.py` 退出。
3. 冷操控终端按 `Ctrl+C`。
4. keyboard pane 按 `space`，确认零速度，再按 `Ctrl+C`。
5. 依次停止 adapter、merger、motion bus；确认前一个退出后再停下一个。
6. 全部退出后执行 `tmux kill-session -t g1-onboard-runtime`。

最后确认：

```bash
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus'
```

## 紧急情况

1. 有立即危险时直接使用现场硬件急停；可控时按 `o` 触发 DAMP。
2. `/nav/stop_cmd` 和 `space` 都不是硬件急停。
3. 停止测试，不清锁、不重启、不回滚，先查明原因。
