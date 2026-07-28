# Box Agent 冷入口

`box_agent_tools_server.py` 是操控服务的冷等待入口。服务进程只使用
Python 标准库，不导入 DDS、相机、感知、IK 或运控模块。收到明确允许的执行
请求后，它才在独立子进程中运行现有 `box_demo_main.py`；旋转命令使用同一
子进程机制，并只在旋转子进程内导入 mover。

## 安全边界

- 默认不允许执行。未传入 `--allow-execute` 时，运动请求返回
  `403 EXECUTION_DISABLED`，不会创建子进程。
- 不要把 `--allow-execute` 写入无人值守的默认启动项。启用该参数意味着请求
  可以触发真实机器人动作，必须满足项目 `AGENTS.md` 的人在环要求。
- 服务一次只允许一个运动子进程。并发请求返回 `409 BUSY`。
- 子进程使用独立进程组，标准输入关闭，命令使用参数数组执行，不经过 shell。
- 服务退出时会停止并回收活动子进程，再关闭 HTTP 服务。

仅启动冷等待服务的示例（执行保持禁用）：

```bash
python3 box_agent_tools_server.py --host 127.0.0.1 --port 5055
```

## 请求生命周期

正常状态流转：

```text
accepted -> running -> succeeded | failed
                    -> cancelling -> cancelled
                    -> timing_out -> timed_out
```

默认单任务超时为 900 秒，可用 `--job-timeout-seconds` 调整。取消和超时先向
整个子进程组发送 `SIGINT`，给现有入口执行 Python 清理逻辑的机会；超过宽限
时间后依次升级到 `SIGTERM` 和 `SIGKILL`。宽限时间由
`--terminate-grace-seconds` 控制。

如果进程在最终升级后仍无法回收，请求保持为 `cleanup_failed`，服务保持占用，
不会接受下一条运动任务。

## HTTP 接口

- `GET /health`
- `GET /status`
- `GET /requests/<request_id>`
- `POST /requests/<request_id>/cancel`
- `POST /tools/manipulate_object`
- `POST /tools/query_holding`
- `POST /tools/patrol_rotate`

抓取请求体：

```json
{"action": "grasp", "item_text": "red box"}
```

旋转请求体：

```json
{"angle_deg": 15, "speed_deg_s": 20}
```

`POST /requests/<request_id>/cancel` 可以使用空 JSON 对象。取消受理返回 202；
调用方应继续读取 `GET /requests/<request_id>`，直到状态进入终态。

## 离线验证

以下测试只运行临时的无害 Python 子进程，不导入或启动真机控制代码：

```bash
python3 box_demo_2/test/test_box_agent_tools_server.py -v
```

测试覆盖冷导入、默认禁用、正常回收、并发拒绝、主动取消、超时回收、服务退出
清理，以及旋转命令的子进程隔离。
