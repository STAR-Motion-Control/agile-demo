# GR00T-WBC + nav_uat 导航合并测试指引

当前 G001 测试使用 **本体运行拓扑**：导航、HTTP IPC bridge、GR00T adapter、merger
全部运行在机器人本体，不再经过 5080。

详细步骤见同目录：

```text
NAV_ONBOARD_TEST_GUIDE.md
```

## 核心启动顺序

1. 确认没有旧控制进程：

```bash
pgrep -af "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|agile_lowcmd_pipeline.py|agile_keyboard_control.py|box_demo_main.py"
```

2. 在机器人本体启动导航运控底座：

```bash
ssh unitree@10.33.12.89
cd ~/zihou/box_demo_1
bash start_g1_onboard_nav.sh
```

启动后 tmux 中会有直接 IPC 键盘 pane，键位和 `start_g1_onboard.sh dwbc` 一致：

- `w/s`：前进/后退，键盘写入 `+0.40/-0.40 m/s`，adapter 会把后退夹到安全上限 `0.20 m/s`。
- `a/d`：左/右横移，导航启动器传参为 `0.25 m/s`。
- `q/e`：左/右转向，导航启动器传参为 `0.40 rad/s`。
- `space`：导航速度归零，保持 GR00T 平衡。
- `o`：DAMP 策略急停，全身关节目标速度 `dq=0`，阻尼保持。
- `Ctrl+C`：退出键盘 pane，并写零速度。

手动键盘和 ROS 导航命令都写 `/tmp/robojudo_ext_cmd.json`，必须人工互斥。
正式导航时不要按运动键，只保留 `space` 和 `o` 作为人工安全入口。
ROS 导航信号和 HTTP bridge 的默认值与键盘/adapter 对齐：前进 `0.40 m/s`、后退 `0.20 m/s`、横移 `0.25 m/s`、转向 `0.40 rad/s`，默认站高为 `0.74 m`。负距离后退会按 `0.20 m/s` 计算持续时间。

3. 在另一个终端启动导航：

```bash
ssh unitree@10.33.12.89
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd ~/workspace/nav_uat/src
python run_ros.py
```

## 必须使用的导航 backend 配置

`~/workspace/nav_uat/src/config.yaml`：

```yaml
motion_backend:
  type: groot_http_discrete
  ipc_url: http://127.0.0.1:5001
  box_demo_module_path: /home/unitree/zihou/box_demo_1
  stand_height: 0.74
  fwd_cruise: 0.40
  back_cruise: 0.20
  lat_cruise: 0.25
  yaw_cruise: 0.40
```

RGBD 也必须由导航进程自启动：

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

## 最小测试

先 stop：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

再测距离 + 方向：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.1, \"direction_deg\": 10}'}"
```

安全稳定后再接 `/nav/text_nav` 完整导航。
