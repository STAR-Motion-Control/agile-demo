# G001 导航 + GR00T 运控联调

## 1. 确认 backend

`/home/unitree/workspace/nav_uat/src/config.yaml` 必须是：

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

这表示导航通过机器人本体 localhost HTTP bridge 写入
`/tmp/robojudo_ext_cmd.json`，不再经过 5080。
默认站高和巡航速度与导航启动器 pane4 的键盘控制、HTTP bridge、adapter 保持一致：前进 `0.40 m/s`、后退 `0.20 m/s`、横移 `0.25 m/s`、转向 `0.40 rad/s`、站高 `0.74 m`。键盘 `s` 会写入 `-0.40 m/s`，但 adapter 后退安全上限为 `0.20 m/s`；导航和 HTTP 负距离后退会直接按 `0.20 m/s` 计算持续时间。

RGBD 配置必须让导航自启动 RealSense，并订阅它自己发布的图像：

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

## 2. 启动本体运控服务

现场确认安全后，在机器人本体执行：

```bash
cd /home/unitree/zihou/box_demo_1
bash start_g1_onboard_nav.sh
```

该脚本启动 merger、GR00T adapter、HTTP IPC bridge 和直接 IPC 键盘，不启动 box demo。

dry-run：

```bash
bash start_g1_onboard_nav.sh --dry-run --no-attach
curl -s http://127.0.0.1:5001/status
tmux kill-session -t g1-onboard-nav
```

## 3. 启动导航

另开终端：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd /home/unitree/workspace/nav_uat/src
python run_ros.py
```

## 4. 小步测试

停止：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

旋转 10 度：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.0, \"direction_deg\": 10}'}"
```

前进 0.1 m：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.1, \"direction_deg\": 0}'}"
```

组合动作：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.3, \"direction_deg\": 20}'}"
```

## 5. 完整导航

小步测试稳定后：

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: '去指定目标点'}"
```

可视化：

```text
http://10.33.12.89:8008/viz
```
