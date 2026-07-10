# G001 导航 + GR00T 运控测试指引

## 1. 当前状态

- 标准入口 `start_g1_onboard_nav.sh` 保留原控制行为。
- 优化入口 `start_g1_onboard_nav_taptap.sh` 只增加经过 5080 MuJoCo A/B 验证的保护：
  - 横移上限 `0.20 m/s`；
  - 转向上限 `0.40 rad/s`；
  - 连续导航线速度斜率 `0.80 m/s^2`；
  - 连续导航角速度斜率 `1.20 rad/s^2`；
  - 旧版停止后固定方波踏步关闭。
- 显式零速、`stop` 和策略急停不经过斜率限制，立即写入 IPC。
- 5080 仿真已通过；本版本尚未完成真机验收，不能写成“必定安全”。

仿真最坏值对比：停后双脚姿态误差 `17.9 cm -> 9.5 cm`，最小双脚间距
`15.2 cm -> 18.4 cm`，最大步周期 CV `0.121 -> 0.115`。六组动作及左右镜像均未摔倒。

## 2. 配置确认

G001 当前 GR00T 配置是：

```text
/home/unitree/workspace/nav_uat/src/config_bk.yaml
```

其中应包含：

```yaml
motion_backend:
  type: groot_http_discrete
  ipc_url: http://127.0.0.1:5001
  box_demo_module_path: /home/unitree/zihou/box_demo_1
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
  warmup_time: 0.6
  warmup_speed: 0.15
```

导航启动时必须确认实际加载的是 `config_bk.yaml`，不是宇树自带运控使用的
`config.yaml`。RGB 相机 topic 应以真机当前 `config_bk.yaml` 为准，不要为运控测试改相机配置。

## 3. 两种入口如何切换

先停止旧 session，确认没有残留 IPC 写者：

```bash
tmux kill-session -t g1-onboard-nav 2>/dev/null || true
pgrep -af 'groot_wbc_boxdemo_adapter|agile_http_ipc_server|agile_keyboard_control|merge_lowcmd_arm_sdk'
```

标准对照组：

```bash
cd /home/unitree/zihou/box_demo_1
bash start_g1_onboard_nav.sh --nav-motion-profile keyboard
```

优化实验组：

```bash
cd /home/unitree/zihou/box_demo_1
bash start_g1_onboard_nav_taptap.sh --nav-motion-profile keyboard
```

启动后另开终端确认标志。标准入口应为 `false`，优化入口应为 `true`：

```bash
python3 -m json.tool /tmp/groot_nav_motion_profile.json
```

该 JSON 文件用于跨终端传递优化标志，因此导航可以在另一个 shell 中启动。

## 4. 真机安全要求

必须有人扶持或吊架保护，急停人员不得离开。首次测试清空周围障碍，并从小指令开始。

pane4 键盘和 ROS 导航都会写 `/tmp/robojudo_ext_cmd.json`，二者必须人工互斥：

- `space`：全身目标速度归零并保持平衡；
- `o`：策略 `DAMP` 急停；
- `Ctrl+C`：只退出键盘进程，不等价于 `DAMP`；
- 导航运行时不要按 `w/s/a/d/q/e`；键盘测试时不要发布导航 topic。

`DAMP/LIMP` 只用于明确急停或退出，不作为普通动作结束方式。

## 5. 启动导航

另开终端，使用项目原有导航环境和启动方式，并确认参数指向 `config_bk.yaml`：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd /home/unitree/workspace/nav_uat/src
python run_ros.py
```

启动后先确认关键 topic：

```bash
ros2 topic list | grep -E 'planned_action|nav/.*_cmd|externel_front_image'
curl -s http://127.0.0.1:5001/status
```

若 RGB topic 缺失、HTTP 不通或 adapter 有 traceback，停止测试，不发布运动命令。

## 6. 分级真机测试

每一级都先跑标准入口，再彻底停止 session，切换优化入口重复。不要热切换。

停止：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
```

`10 deg` 转向：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.0, \"direction_deg\": 10}'}"
```

`0.10 m` 前进：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.10, \"direction_deg\": 0}'}"
```

通过后依次测试：

1. 左右各 `10 deg`，每次动作后观察双脚是否平行、间距是否缩小。
2. 左右横移各 `0.10 m`，连续重复 4 次。
3. `30 deg + 0.50 m`，左右镜像各重复 4 组。
4. 最后才发布 `/nav/text_nav`，验证完整 `/planned_action` 序列。

每组记录：最小脚间距、停后脚前后错位、机器人高度、路径终点偏差、停止响应和 adapter 日志。

## 7. 5080 仿真复现

5080 快照位于：

```text
/home/wjzh/agile-demo/g001_onboard
```

使用 Python 虚拟环境，不是 Conda：

```bash
cd /home/wjzh/agile-demo/g001_onboard
/home/wjzh/GR00T-WholeBodyControl/.venv_wbc/bin/python \
  experiments/slew_sequence_ab.py
/home/wjzh/GR00T-WholeBodyControl/.venv_wbc/bin/python \
  -m unittest discover -s nav_uat/tests -v
```

结果写入 `sim_results/slew_sequence_ab.json`。仿真通过只表示可以进入低速、有人在环的真机测试，不能替代真机验收。
