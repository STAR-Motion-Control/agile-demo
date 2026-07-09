# G001 本体导航 + GR00T 运控联调指引

本文件适用于新机器人 **宇树 G001**：

- SSH：`unitree@10.33.12.89`
- 密码：`123`
- Wi-Fi：`G1-SUSTech-8021x`
- Wi-Fi 网卡：`wlx94ba06f8171b`
- 运控/操控部署目录：`/home/unitree/zihou/box_demo_1`
- 导航目录：`/home/unitree/workspace/nav_uat`

## 0. 安全原则

真机测试必须有人在环，吊架/急停/遥控器准备好。

Codex 只做代码部署、静态检查和日志读取；不要让 Codex 直接启动、重启非 dry-run 真机控制脚本。

## 1. 当前拓扑

导航、运控、操控都在机器人本体运行，不再经过 5080：

```text
nav_uat
  -> http://127.0.0.1:5001
  -> agile_http_ipc_server.py
  -> /tmp/robojudo_ext_cmd.json
  -> groot_wbc_boxdemo_adapter.py
  -> rt/lowcmd_rl
  -> merge_lowcmd_arm_sdk.py
  -> rt/lowcmd
```

导航联调时不要启动 `box_demo_main.py`。`start_g1_onboard_nav.sh` 会启动一个
与操控+运控一致的直接 IPC 键盘 pane，用于手动验底座和人工安全介入。
手动键盘和 ROS 导航命令都写 `/tmp/robojudo_ext_cmd.json`，必须人工互斥。

操控+抱箱测试仍使用 `start_g1_onboard.sh dwbc` 或操控团队指定脚本；导航联调使用
`start_g1_onboard_nav.sh`。

## 2. 上机前检查

```bash
ssh unitree@10.33.12.89
cd ~/zihou/box_demo_1

ip -br addr show enP8p1s0
ping -c2 192.168.123.161
pgrep -af "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|agile_lowcmd_pipeline.py|agile_keyboard_control.py|box_demo_main.py"
```

预期：

- `enP8p1s0` 有 `192.168.123.164/24`。
- 能 ping 通 MCU `192.168.123.161`。
- 没有旧的底层控制、键盘或 box_demo 进程。

如果 `enP8p1s0` 没有 IPv4，启动脚本会尝试补 `192.168.123.164/24`。如果 MCU ping 不通，
只能做 dry-run，不能做真机运动。

## 3. 非真机 dry-run

dry-run 不启动 merger，不发布最终 `rt/lowcmd`：

```bash
cd ~/zihou/box_demo_1
bash start_g1_onboard_nav.sh --dry-run --no-attach
tmux attach -t g1-onboard-nav
```

检查：

- pane2 的 GR00T adapter 能 import 并进入 dry-run。
- pane3 的 HTTP bridge 显示 `http://127.0.0.1:5001`。
- 新终端执行：

```bash
curl -s http://127.0.0.1:5001/status
curl -s "http://127.0.0.1:5001/stop"
```

结束 dry-run：

```bash
tmux kill-session -t g1-onboard-nav
```

## 4. 启动导航运控底座

现场确认安全后，由现场人员执行：

```bash
cd ~/zihou/box_demo_1
bash start_g1_onboard_nav.sh
```

默认 profile 是 `precise`，导航离散动作会保留 `min_duration/min_distance` 可靠小步逻辑。若要让导航指令使用和键盘更接近的巡航速度，启动时改为：

```bash
cd ~/zihou/box_demo_1
bash start_g1_onboard_nav.sh --nav-motion-profile keyboard
```

两种 profile 都保留线性运动前的 warm-up。`keyboard` 只关闭 `min_duration/min_distance` 对导航距离/角度的改写，使 `/nav/forward_cmd`、`/nav/rotate_cmd` 和 `/planned_action` 在主运动段使用 `0.40/0.20/0.25/0.40` 这组速度。

这个脚本只启动：

- merger
- GR00T adapter
- localhost HTTP IPC bridge
- direct IPC keyboard

不会启动 box demo。

直接 IPC 键盘在 tmux pane4，键位和 `start_g1_onboard.sh dwbc` 的操控+运控键盘一致：

- `w/s`：前进/后退，键盘写入 `+0.40/-0.40 m/s`，adapter 会把后退夹到安全上限 `0.20 m/s`。
- `a/d`：左/右横移，导航启动器传参为 `0.25 m/s`。
- `q/e`：左/右转向，导航启动器传参为 `0.40 rad/s`。
- `z/x`：高度下降/上升。
- `space`：速度归零，GR00T 继续保持平衡。
- `o`：DAMP 阻尼急停；adapter 以当前关节位置为目标，对全身命令 `dq=0`、`kp=0`、`kd=damping`。
- `Ctrl+C`：退出键盘 pane，并写零速度。

切换要求：

- 手动键盘验底座时，不要发 `/nav/relative_cmd` 或 `/nav/text_nav`。
- 正式导航时，不要按 `w/s/a/d/q/e/z/x/c/r`；现场只保留 `space` 和 `o` 作为人工安全入口。

ROS 导航信号和 HTTP bridge 使用同一组速度上限：前进 `0.40 m/s`、后退 `0.20 m/s`、横移 `0.25 m/s`、转向 `0.40 rad/s`，统一站高 `0.74 m`。实际执行 profile 由 `start_g1_onboard_nav.sh --nav-motion-profile` 决定：

- `precise`：默认。保留 `min_duration=1.0`、`min_distance=0.08`、`v_floor=0.10`，适合可靠小步，但主运动速度可能低于键盘速度。
- `keyboard`：使用键盘同样巡航速度，关闭 `min_duration/min_distance`，适合对比键盘和导航姿态。

两种模式都保留 warm-up。profile 会写入 `/tmp/groot_nav_motion_profile.json`，`nav_uat` 的 `motion_backend.py` 会在启动时读取。

## 5. 启动导航 ROS bridge

另开终端：

```bash
ssh unitree@10.33.12.89
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd ~/workspace/nav_uat/src
python run_ros.py
```

确认 `~/workspace/nav_uat/src/config.yaml` 中：

```yaml
motion_backend:
  type: groot_http_discrete
  ipc_url: http://127.0.0.1:5001
  box_demo_module_path: /home/unitree/zihou/box_demo_1
  motion_profile: precise
  profile_file: /tmp/groot_nav_motion_profile.json
  stand_height: 0.74
  fwd_cruise: 0.40
  back_cruise: 0.20
  lat_cruise: 0.25
  yaw_cruise: 0.40
  min_duration: 1.0
  min_distance: 0.08
  v_floor: 0.10
  w_floor: 0.10
```

同时确认 RGBD 由导航进程自启动并发布：

```yaml
rgbd_server:
  launch_rgbd_server: true
  publish_topic:
    rgb: /externel_front_image
    depth: /externel_front_depth
  subscrib_topic:
    rgb: /externel_front_image
    depth: /externel_front_depth
```

## 6. 小步联调顺序

先做 stop：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

再做小角度旋转：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.0, \"direction_deg\": 10}'}"
```

再做小距离前进：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.1, \"direction_deg\": 0}'}"
```

最后做组合动作：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.3, \"direction_deg\": 20}'}"
```

观察：

- `g1-onboard-nav` pane2 adapter 日志。
- `g1-onboard-nav` pane3 HTTP bridge `/status`。
- 机器人实际步态、是否稳定、stop 是否及时。

## 7. 完整导航入口

小步测试稳定后，再发送文本导航：

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: '去指定目标点'}"
```

可视化页面：

```text
http://10.33.12.89:8008/viz
```

## 8. 回退

停止导航任务：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

现场键盘安全入口：

```text
tmux pane4: space 停止导航速度；o 策略阻尼急停。
```

停止底座 session：

```bash
tmux kill-session -t g1-onboard-nav
```

如需回到操控+抱箱，不要复用导航 session；先停掉 `g1-onboard-nav`，再由现场人员启动操控对应脚本。
