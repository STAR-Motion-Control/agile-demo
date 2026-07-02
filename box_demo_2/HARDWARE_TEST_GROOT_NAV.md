# GR00T-WBC + nav_uat 导航合并测试指引

本文用于导航功能接入 GR00T-WBC 下半身运控后的联调。推荐拓扑是：导航 `nav_uat` 运行在宇树机器人本体上，GR00T-WBC adapter、lowcmd 合并器和 HTTP IPC 桥运行在 5080 服务器上。

安全要求：任何真机运行都必须有人在环，机器人必须吊挂或处于可人工保护状态；本文命令只描述启动和测试顺序，不代表可以无人值守启动真机。

## 1. 系统拓扑

```text
宇树机器人本体 (/home/unitree/workspace/nav_uat)
  run_ros.py
  motion_backend.type = groot_http_discrete
  通过 RemoteMover 兼容接口发送 HTTP 命令
        |
        v
5080: http://192.168.123.222:5001
  agile_http_ipc_server.py --backend legacy-ipc
        |
        v
  /tmp/robojudo_ext_cmd.json
        |
        v
  groot_wbc_boxdemo_adapter.py -> rt/lowcmd_rl
  merge_lowcmd_arm_sdk.py      -> rt/lowcmd
```

`start_groot_wbc_nav.sh` 只启动导航需要的下半身服务：merger、GR00T adapter、HTTP IPC bridge。它不会启动键盘控制，也不会启动 `box_demo_main.py`。导航测试期间，`agile_http_ipc_server.py` 应该是 `/tmp/robojudo_ext_cmd.json` 的唯一写入者。

## 2. 5080 启动下半身服务

先做 dry-run，确认 tmux 窗口、参数和 HTTP bridge 能正常启动。dry-run 不启动最终 `rt/lowcmd` 合并发布。

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_nav.sh \
  --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble \
  --dry-run
```

确认无误后，在有人保护真机的情况下启动真实下半身服务：

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_nav.sh \
  --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble
```

启动后应看到 3 个 tmux pane：

- `merge_lowcmd_arm_sdk.py`：合并 `rt/lowcmd_rl` 和 `rt/arm_sdk`，发布最终 `rt/lowcmd`。
- `groot_wbc_boxdemo_adapter.py`：读取 `/tmp/robojudo_ext_cmd.json`，调用 GR00T-WBC，发布 `rt/lowcmd_rl`。
- `agile_http_ipc_server.py`：接收机器人导航侧 HTTP 命令，写入 `/tmp/robojudo_ext_cmd.json`。

## 3. 机器人侧启动导航

确认 `/home/unitree/workspace/nav_uat/src/config.yaml` 中使用 GR00T HTTP backend：

```yaml
motion_backend:
  type: groot_http_discrete
  ipc_url: http://192.168.123.222:5001
  box_demo_module_path: /home/unitree/zihou/box_demo_2
```

然后在机器人本体启动 ROS 导航桥：

```bash
ssh unitree-002
conda activate nav
cd /home/unitree/workspace/nav_uat/src
python run_ros.py
```

## 4. 上机前链路检查

从机器人侧检查 5080 HTTP bridge 是否可达：

```bash
curl -s http://192.168.123.222:5001/status
```

期望返回 JSON，且 `backend` 为 `legacy-ipc`，`cmd_file` 指向 `/tmp/robojudo_ext_cmd.json`。

如果不可达，先检查：

- 5080 的 `agile_http_ipc_server.py` 是否已启动。
- 机器人到 `192.168.123.222:5001` 的网络是否连通。
- 5080 防火墙或端口占用情况。

## 5. 最小运动测试

先测试停止命令，确认导航侧能安全写零速度：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

再测试小角度旋转，正值表示向左/逆时针：

```bash
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist \
  "{angular: {z: 0.17}}"
```

再测试小距离前进，单位为米：

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}}"
```

## 6. 距离 + 方向测试接口

新增 `/nav/relative_cmd` 用于第一版导航合并联调。它先按 `direction_deg` 旋转，再按 `distance_m` 前进。

JSON 格式：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.1, \"direction_deg\": 10}'}"
```

简写格式：

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String "{data: '0.1 10'}"
```

第一轮建议只测：

- `direction_deg = 10`，`distance_m = 0.1`
- `direction_deg = 30`，`distance_m = 0.5`

观察 5080 上 adapter 日志中的 `cmd=(vx, vy, wz, h)` 是否和命令方向一致。

## 7. 完整导航测试

只有在第 4-6 节都通过后，再测试 `/nav/text_nav`：

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: '去指定目标点'}"
```

测试时重点观察：

- `/nav/status` 是否进入 `processing`、`completed` 或明确失败状态。
- 5080 HTTP bridge 是否持续收到导航命令。
- GR00T adapter 是否按预期切换 `vx/wz`。
- 真机是否能稳定完成小步旋转和前进。

## 8. 回退方案

如果需要退回原来的 Unitree 手柄控制路径，把机器人侧 `config.yaml` 改为：

```yaml
motion_backend:
  type: wireless_controller
```

然后重启 `run_ros.py`。这会恢复原来的 `/wirelesscontroller` 发布逻辑，不再通过 5080 HTTP bridge 控制 GR00T-WBC。
