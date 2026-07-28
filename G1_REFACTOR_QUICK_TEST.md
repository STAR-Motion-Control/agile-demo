# G1-001 新版快速测试手册

新版目录：`/home/unitree/releases/agile-demo-refactor`

适用标签：`g001-runtime-refactor-v1.2`

当前允许测试：运控 preflight、运控零速 health、右膝复查通过后的单次键盘小动作，以及冷操控 403。导航 profile 尚未与旧版对齐，因此禁止启动导航和发送导航命令。

## 0. 环境和安全

```text
运控：由 launcher 自动激活 Conda，现场不手工激活或填写 CONDA_ENV。
导航：使用 nav 环境，但当前标签禁止启动。
操控：使用 robojudo_zihou2 环境。
```

运控 launcher 已默认使用 G1-001 的 `enP8p1s0`、`dwbc`、默认 tmux session 和单线程参数；现场命令不重复填写这些默认值。

- 必须有人在环、可靠支撑、急停操作员和清场。
- 未经当次明确授权，不得停止、启动或重启真机控制进程。
- 旧版和新版不得同时运行。
- 右膝机械/电气复查未通过前，只允许零速检查。
- 有立即危险时使用现场硬件急停；可控时按 `o` 触发 DAMP。

## 1. 版本和进程（只读）

```bash
ssh unitree@10.33.12.89
cd /home/unitree/releases/agile-demo-refactor
git status --short --branch
git describe --tags --exact-match
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus|[s]tart_g1_onboard'
```

必须确认：标签为 `g001-runtime-refactor-v1.2`、源码 clean，并记下现有进程。不要直接杀进程。

## 2. 运控 preflight 和启动

只有相关旧 demo 控制进程已经由现场人员授权并在原终端正常退出后，才执行：

```bash
cd /home/unitree/releases/agile-demo-refactor
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --preflight-only
```

必须看到 `preflight passed; nothing started`。`--no-manip-ingress` 和 `--preflight-only` 必须保留。

获得本次新版运控启动授权后：

```bash
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --no-manip-ingress \
  --human-approved-control-start
```

launcher 会自动激活 Conda 并进入默认 tmux session `g1-onboard-runtime`。先保持零速度 60 秒，不按运动键。SSH 断开后可重新连接：

```bash
tmux attach -t g1-onboard-runtime
```

## 3. 运控 health

另一终端不需要激活 Conda：

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

任一项不满足：不运动，获得停止授权后执行第 7 步。

## 4. 键盘最小测试（逐项授权）

只有右膝复查已经通过，才允许执行：

1. 在 keyboard pane 按 `space`，确认零速度。
2. 获得本次动作授权后短按一次 `w`，立即按 `space`。
3. 等待站稳，重新执行第 3 步 health 检查。

当前不测试后退、横移、转向、深蹲、连续行走、1 m/s 或圆弧。

## 5. 冷操控 403

保持底座零速度。操控入口使用 `robojudo_zihou2`：

```bash
ssh unitree@10.33.12.89
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate robojudo_zihou2
cd /home/unitree/releases/agile-demo-refactor/box_demo_2
python box_agent_tools_server.py
```

不要添加 `--allow-execute`。默认地址为 `127.0.0.1:5055`，不会初始化 DDS、相机、IK 或 mover。

另一终端验证：

```bash
curl -i -X POST http://127.0.0.1:5055/tools/manipulate_object \
  -H 'Content-Type: application/json' \
  -d '{"action":"grasp","item_text":"test box"}'
pgrep -af '[b]ox_demo_main.py'
```

必须返回 `403 EXECUTION_DISABLED`，且 `box_demo_main.py` 无输出。冷入口 idle CPU 应低于 5%。

当前不能据此宣称“导航 + 操控 + 运控”联跑已经通过。

## 6. 导航阻断项

当前禁止启动 `start_nav_refactored.sh`，也禁止发布任何导航 topic。原因：

```text
旧版：warmup_time=0.0, v_floor=0.12
当前新版：warmup_time=0.6, v_floor=0.10
```

解除阻断必须满足：新版 profile 与旧版一致、5080 A/B 覆盖 `config_g001` 和两个 launcher 的真实 profile、发布新代码标签并更新本文。

## 7. 正常停止新版（需要授权）

获得停止授权后：

1. 冷操控终端按 `Ctrl+C`。
2. keyboard pane 按 `space`，确认零速度，再按 `Ctrl+C`。
3. 在 tmux 中依次停止 adapter、merger、motion bus；确认前一个进程退出后再停下一个。
4. 全部进程退出后执行：

```bash
tmux kill-session -t g1-onboard-runtime
```

最后确认：

```bash
pgrep -af '[r]un_ros.py|[g]root_wbc_boxdemo_adapter.py|[m]erge_lowcmd_arm_sdk.py|[b]ox_demo_main.py|[b]ox_agent_tools_server.py|[o]nboard_runtime.motion_bus'
```

应无新版相关输出。不要手工删除 socket、health 文件或 safety journal。

## 紧急情况

1. 有立即危险时，现场人员直接使用硬件急停；可控时按 `o` 触发 DAMP。
2. 只有确认软件仍响应且没有立即危险时，才按 `space` 请求零速度。
3. 停止测试，不清锁、不重启、不回滚，先查明原因。
