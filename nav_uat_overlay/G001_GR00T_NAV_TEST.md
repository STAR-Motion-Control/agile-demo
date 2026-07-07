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
  lat_cruise: 0.25
  yaw_cruise: 0.40
```

这表示导航通过机器人本体 localhost HTTP bridge 写入
`/tmp/robojudo_ext_cmd.json`，不再经过 5080。
默认站高和巡航速度与导航启动器 pane4 的键盘控制保持一致。

## 2. 启动本体运控服务

现场确认安全后，在机器人本体执行：

```bash
cd /home/unitree/zihou/box_demo_1
bash start_g1_onboard_nav.sh
```

该脚本启动 merger、GR00T adapter、HTTP IPC bridge，不启动键盘和 box demo。

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
