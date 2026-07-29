# G1-001 新版运行时测试指引

首次联调请优先使用 [快速测试手册](G1_REFACTOR_QUICK_TEST.md)；本手册保留完整验收、诊断和回滚细节。

本文用于测试独立部署在以下目录的新版代码：

```text
/home/unitree/releases/agile-demo-refactor
```

新版部署标签为 `g001-runtime-refactor-v1.3.2`。目录名不使用 Git 提交哈希；Git 提交号仅作为内部完整性记录。

当前标签已将导航 profile 的行为参数同步为旧真机 launcher 的实际默认值。5080 已验证旧、新 launcher profile 的命令规划与 MuJoCo 路线等价；这只是导航逻辑和速度契约验证，不代表真机位移、Jetson CPU 或三模块联合负载已经通过。

本文中的控制启动、停止、运动、导航和操控命令都只能由现场人员在机器人旁、完成安全检查并对当次操作明确授权后执行。代码同步、Git 校验和只读状态检查不代表已经获得控制授权。

## 1. 不可突破的安全边界

1. 机器人必须使用吊架或可靠支撑，急停操作员必须能立即触达急停。
2. 机器人周围清场，测试人员不得位于机器人可能跌倒、迈步或手臂扫过的范围内。
3. 旧版和新版不得同时运行。任何时刻只能有一个 `rt/lowcmd` 最终发布者。
4. 未获得当次明确授权，不得启动、重启、停止或向真机控制链发送命令。
5. `space` 和 `/nav/stop_cmd` 只是零速度请求，不是硬件急停；异常时使用现场急停或键盘 `o` 的 DAMP 锁存。
6. DAMP/LIMP 锁存后先停止测试并查明原因。不得删除 `/tmp/groot_motion_safety_latch.json`，不得为了继续测试而绕过安全门。
7. 出现步态不稳定、异常关节声、支撑松动、通信中断、健康门失败或 CPU 持续饱和时，立即终止当前阶段。
8. 右膝机械/电气复查未通过前，只允许零速度检查，不得用行走验证右膝。

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

环境边界：

```text
运控：由 launcher 自动激活 Conda，现场不手工激活或填写 CONDA_ENV。
导航：使用 nav 环境。
操控：使用 robojudo_zihou2 环境。
```

Conda 自动激活只是 launcher 的运行机制，不表示运控业务绑定到某个环境。本文沿用该机制，不修改、不覆盖，也不额外探测环境。

`config_g001` 保留了现场外置相机、5 FPS、开环模式和现场地图副本，并把旧 HTTP/JSON 命令链替换为 motion bus。新 launcher 会显式写入旧版 profile 的 14 个行为字段，`config_g001` 启动前会强制校验，这 14 个字段不再回落到 YAML。其他导航参数仍按 `config_g001`/`config_bk` 的原有 YAML 读取。

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
- 当前标签为 `g001-runtime-refactor-v1.3.2`。
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
pgrep -af 'start_g1_onboard|motion_bus|merge_lowcmd|groot_wbc|agile_runtime_keyboard.py|run_ros.py|box_demo_main.py|box_agent_tools_server.py'
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

## 6. 停止旧版：需要当次授权

只有现场人员明确批准切换后，才在旧进程原终端正常退出：先停止导航命令，旧运控按 `space` 请求零速度，再依次退出导航、操控、键盘、adapter 和 merger。不要停止 Unitree 系统服务，不要直接杀 PID。相关旧 demo 进程未全部退出时，不启动新版。

## 7. 新版静态配置与 preflight

`--print-config` 不会创建控制进程，可以用于静态查看：

```bash
cd /home/unitree/releases/agile-demo-refactor
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --print-config
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

确认相关旧控制进程已经正常退出后执行 preflight：

```bash
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --preflight-only
```

必须看到 `preflight passed; nothing started`。若检测到旧进程、已有 tmux、网卡地址错误、MCU 不可达或必需源码/依赖路径缺失，停止测试。新版 preflight 不会自动改网卡，也不会杀旧进程；它不检查 ONNX 内容，也不证明导航相机、地图或定位已经可用。

这里不填写 `IFACE`、`dwbc`、线程数或 session 名：它们已有 G1-001 默认值。`--no-manip-ingress` 和 `--preflight-only` 是本步骤必须保留的安全参数。

## 8. 启动新版运控：需要当次授权

现场人员明确批准本次新版运控启动后执行。不要手工激活 Conda，也不要填写 `CONDA_ENV`：

```bash
cd /home/unitree/releases/agile-demo-refactor
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --human-approved-control-start
```

launcher 自动激活 Conda 并进入默认 tmux session `g1-onboard-runtime`。先保持零速度 60 秒，不按运动键。

## 9. S0 运控 health

另一终端不需要激活 Conda：

```bash
cd /home/unitree/releases/agile-demo-refactor
/usr/bin/python3 -m onboard_runtime.runtime_ctl status
/usr/bin/python3 -m json.tool /tmp/groot_adapter_health.json
/usr/bin/python3 -m json.tool /tmp/groot_merger_health.json
```

必须同时满足：

- bus `healthy=true`、`motion_blocked=false`、`safety_latched=false`、`command_file=null`。
- adapter 和 merger 均为 `healthy=true`。
- adapter、merger 的 `runtime_id` 与 bus `health_runtime_id` 相同。
- adapter `cycle_p99_ms <= 40`，最大周期不超过 60 ms，成功时间持续刷新。
- merger `reason=healthy`、`motion_stream_fresh=true`、`motion_health_ok=true`、`tick_gap_ms <= 60`。
- 零动作观察期间没有 DAMP/LIMP 锁存、owner 冲突或陈旧命令告警。
- CPU idle 不低于 20%，控制周期 p99 不超过 40 ms，不持续出现超过 60 ms 的控制空窗。

## 10. 键盘最小测试

只有右膝机械/电气复查通过且现场人员逐项授权后，才允许在 keyboard pane 按 `space`、短按一次 `w` 并立即再按 `space`。等待站稳后重新执行第 9 节 health。当前不测试后退、横移、转向、深蹲、连续行走、1 m/s 或圆弧。

## 11. 导航 profile 与 preflight

这一节必须在新版 runtime 成功启动且第 9 节 health 通过后执行。runtime 启动时才会原子写入当次实际 profile。

在新终端加载真机原有导航环境：

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source /home/unitree/unitree_ros2/install/setup.bash
cd /home/unitree/releases/agile-demo-refactor
```

先校验 runtime 刚生成的 profile：

```bash
python onboard_runtime/nav_profile.py validate \
  --input /tmp/groot_nav_motion_profile.json \
  --expected-socket /tmp/groot_motion_bus.sock
```

必须看到 `G1-001 navigation profile matches preserved legacy defaults`。校验项是从旧真机 `start_g1_onboard_nav.sh` 实际默认值提取的：

```text
motion_profile=precise
stand_height=0.76
walk_min_height=0.72
warmup_enabled=false
warmup_time=0.0
warmup_speed=0.15
fwd_max=0.50
back_max=0.20
lat_max=0.30
yaw_max=0.60
lat_cruise=0.20
v_floor=0.12
w_floor=0.10
waist_to_rl_on_motion=true
```

任一项不一致都不启动导航；不手工修改 JSON、YAML 或环境变量绕过检查。

再检查现场导航输入：

```bash
ros2 topic list | grep -E '/camera/captured_(image|depth)|/dog_odom|/safety/lidar_state'
ros2 topic echo --once /safety/lidar_state
```

图像、深度或里程计缺失/陈旧时停止。LiDAR 回复只接受 `data: clear`；`blocked`、`stale`、`unknown`、无消息或其他值都停止，不手工发布安全状态绕过监控。上述 ROS 检查会临时创建 DDS participant/subscriber，不要与 CPU 基线采样同时进行。

执行导航 preflight：

```bash
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --preflight-only
```

必须同时看到 profile 校验成功和 `navigation preflight passed`。这里不填 `--native-threads`，因为 launcher 的 G1-001 默认值已经是 1。

## 12. 导航 idle 与最小动作

现场人员明确批准本次导航启动后，在导航终端前台执行：

```bash
bash nav_uat_overlay/start_nav_refactored.sh \
  --config config_g001 \
  --human-approved-control-start
```

不要直接执行 `python run_ros.py`，也不要手工设置内部授权环境变量。在另一个已加载同样导航环境的终端检查：

```bash
ros2 topic echo --once /nav/status
ros2 service call /nav/get_pose std_srvs/srv/Trigger '{}'
```

`/nav/status` 必须同时显示任务 idle、`lidar_safety.state=clear` 和 `lidar_safety.paused=false`，pose 必须可用。导航进程启动时 LiDAR 初值是 `unknown`，因此不能用启动前的 topic echo 代替这项启动后检查。

保持导航 idle 至少 60 秒，然后在检查终端重新读取 S1 负载下的 runtime health 并采样 CPU：

```bash
/usr/bin/python3 -m onboard_runtime.runtime_ctl status
/usr/bin/python3 -m json.tool /tmp/groot_adapter_health.json
/usr/bin/python3 -m json.tool /tmp/groot_merger_health.json
vmstat 1 60
```

必须仍满足第 9 节的所有 health 门；忽略 `vmstat` 首个累计行后，后续 `id` 不低于 20%。任一项失败就停在 S1，不进入 S2。右膝机械/电气复查未通过时也只能停在 S1 idle。

申请第一个动作授权前，在 CPU 采样结束后最后再读一次新鲜状态：

```bash
ros2 topic echo --once /safety/lidar_state
ros2 topic echo --once /nav/status
ros2 service call /nav/get_pose std_srvs/srv/Trigger '{}'
```

必须再次看到 `data: clear`、任务 idle、`lidar_safety.state=clear`、`lidar_safety.paused=false` 和可用 pose；否则不进入 S2。

只有现场对本次动作再次授权，才发送 `0.10 m` 前进：

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

等待站稳，重新检查 `/nav/status`、pose 和第 9 节 health，然后发送零速请求：

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty '{}'
```

只有前一步完全正常且现场对本次转向再次授权，才发送 `10°` 左转：

```bash
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.174532925}}'
```

再次等待站稳，检查 status、pose 和 health，并发送 `/nav/stop_cmd`。首轮不测试后退、横移、右转、文本目标、连续路线、1 m/s、深蹲或圆弧。

旧版的导航巡航速度契约仍为：前进 `0.40 m/s`、后退 `0.20 m/s`、横移 `0.20 m/s`、转向 `0.40 rad/s`。这些是代码回归值，不是首轮真机的执行许可。

## 13. 当前允许的性能采集

当前允许采集 S0 运控零速、S1 导航 idle、逐项授权后的 S2 最小动作和 C0 冷操控入口数据。这些阶段数据可用于真机 A/B，但不得宣称“导航 + 真实操控 + 运控”已经完成真机验证。

## 14. 操控冷入口测试

保持导航 idle 和底座零速度。新版 runtime 使用 `--no-manip-ingress` 启动，因此 5055 端口应空闲；若已有监听者，停止本节，不抢占端口：

```bash
ss -ltnp | grep ':5055' || true
```

在新终端启动不允许执行动作的冷入口。该进程本身不初始化 DDS、相机、IK 或 mover：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robojudo_zihou2
cd /home/unitree/releases/agile-demo-refactor/box_demo_2
python box_agent_tools_server.py
```

`127.0.0.1:5055` 和禁止执行都是程序默认值。冷测试不要添加 `--allow-execute`；`--locomotion`、`--iface` 和 `--no-vision-log` 只影响未来动作子进程，冷测试不需要填写。

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
pgrep -af '[b]ox_agent_tools_server.py|[b]ox_demo_main.py'
ps -eLo pid,tid,pcpu,stat,comm,args --sort=-pcpu | head -80 \
  | tee "$TEST_RESULT_DIR/cold_manip_threads.txt"
```

## 15. 真实操控和全链路阻断

当前标签不提供 `--allow-execute` 的现场执行步骤，也不允许真实抓取。真实操控和完整三模块联跑仍需另行测试指引和当次授权。

## 16. 当前测试阶段

当前标签只允许以下状态：

| 阶段 | 运行内容 | 当前状态 |
| --- | --- | --- |
| R0 | 现有现场进程只读检查 | 允许，不改变进程状态 |
| P0 | 静态配置和 preflight | 旧控制进程退出后允许 |
| S0 | 新版 motion bus + merger + adapter | 允许零速度；右膝通过后允许单次键盘小动作 |
| S1 | S0 + 导航 idle | profile/preflight 通过且获得启动授权后允许；需 idle 60 秒并重新通过 LiDAR、health 和 CPU 门 |
| S2 | S1 + 导航运动 | 只有 S1 负载门通过后，才允许逐项授权的 `0.10 m` 前进和 `10°` 左转 |
| C0 | S1 idle + 冷操控入口 | 只允许验证 403，禁止操控执行 |
| S3/S4 | 真实操控或三模块联跑 | 阻断 |

各阶段可保存：

```bash
vmstat 1 60
pidstat -durh -p ALL 1 60
timeout 60 tegrastats --interval 1000
ps -eLo pid,tid,psr,pcpu,stat,comm,args --sort=-pcpu
```

当前可用门槛：

- CPU idle 稳定窗口不低于 20%。
- adapter 控制周期 p99 不超过 40 ms。
- 不持续出现超过 60 ms 的控制空窗。
- merger motion stream 持续 fresh，RL age 不超过 0.12 s。
- 冷操控入口 idle CPU 不超过 5%，不创建 DDS/相机/IK 子进程。

只有同一真机、相同条件下的 A/B 才能形成 Jetson CPU 与步态结论。S1/S2 可形成导航 + 运控阶段结果；C0 的操控仅是冷入口，不能称为真实三模块联跑。5080 仿真不能代替真机结论。

## 17. 异常处理

1. 有立即危险时，现场人员直接使用硬件急停；可控时按 `o` 触发 DAMP。
2. 只有确认软件仍响应且没有立即危险时，才按 `space` 请求零速度。
3. 停止测试，不清锁、不重启、不回滚，先查明原因。

## 18. 有序停止新版

停止本身需要当次明确授权。按以下顺序执行：

1. 若导航在运行，先发布 `/nav/stop_cmd`，确认 `/nav/status` 回到 idle。
2. 在导航启动的原终端按 `Ctrl+C`，确认 `run_ros.py` 退出。
3. 冷操控入口若在运行，在其原终端按 `Ctrl+C`。
4. keyboard pane 按 `space`，确认零速度，再按 `Ctrl+C`。
5. 在 tmux 中依次停止 adapter、merger、motion bus；确认前一个进程已经退出后再停下一个。
6. 全部控制进程退出后，才清理默认 session：

```bash
tmux kill-session -t g1-onboard-runtime
```

最后确认：

```bash
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[a]gile_runtime_keyboard.py|[b]ox_agent_tools_server.py|[b]ox_demo_main.py|[o]nboard_runtime.motion_bus'
```

应无新版相关输出。不要手工删除 socket、health 文件或 safety journal。

## 19. 回滚边界

只有确认新版全部退出、机器人受支撑且现场人员明确批准后，才允许恢复旧版。必须重新读取现场状态，并使用第 5 节记录的原启动命令；本文不提供固定的可复制回滚命令，避免猜测或混启旧拓扑。

## 20. 结果记录模板

每轮填写：

```text
日期/操作员：
部署标签：g001-runtime-refactor-v1.3.2
新目录：/home/unitree/releases/agile-demo-refactor
测试阶段：R0 / P0 / S0 / S1 / S2 / C0
导航 profile 校验：通过 / 不通过
导航 idle 60 秒后 LiDAR/S1 health/CPU：
导航动作、授权与执行时间：
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
导航动作后 status/pose/health：
冷操控 idle CPU/线程/子进程数：
操控请求和结果：
是否出现步态不稳定：
是否触发 DAMP/LIMP：
与旧版差异：
结论：通过 / 不通过 / 需要复测
日志目录：
```
