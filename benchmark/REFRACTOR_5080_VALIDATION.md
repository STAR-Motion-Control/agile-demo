# 5080 重构候选离线验证

验证日期：2026-07-28

分支：`refactor/onboard-runtime`

代码基线：`49f2631`

导航完整导入参考：`eb1b6e9`

重构主提交：`9aea48a`

## 安全与环境

- 所有操作都在 5080 的 `/home/wjzh/agile-demo-validation` 隔离目录中完成。
- 未修改 5080 上现有的 `/home/wjzh/agile-demo`。
- 未连接真机 DDS，未启动 ROS 导航、相机、adapter、merger 或任何真机控制脚本。
- MuJoCo 使用 5080 本地的 GR00T-WBC Balance/Walk ONNX 模型。
- 联合烟雾验证只启动了绑定 `127.0.0.1` 且不带
  `--allow-execute` 的操控冷服务，所有动作请求都必须被拒绝。

## 导航兼容性修正

首轮审计发现重构候选把 RealSense 从基线的 `640x480@30` 改成了
15 FPS，并默认关闭 RGB/depth ROS 话题。这会改变感知延迟和进程外
接口，不符合本轮“不影响原有功能和速度”的约束。最终候选恢复：

```yaml
rgbd_server:
  fps: 30
  publish_ros_topics: true
  capture_depth: true
```

进程内的 Localization 和 NavDP 仍共享 latest-only `CameraFrameHub`，不再
各自订阅本进程发布的 RGB-D 话题。因此保留了原始相机速率、depth
数据和对外话题，同时消除两组内部 message-filter/CvBridge 回环。

## 结果

### 代码与导航契约

`benchmark/scripts/offline_nav_ab.py` 直接从 Git object 读取三个版本，
只使用虚拟时钟和假 transport：

- 30 组运动求解、18 组离散动作、744 个命令帧与基线一致。
- 前进/后退/横移/偏航巡航速度保持 `0.40/0.20/0.20/0.40`。
- 两份配置共 36 个运动速度键、86 个闭环控制键不变。
- 98 个导航核心方法和 2 个调度文件未改，3 组路径转换固定样例一致。
- stop/cancel 后候选产生的过期非零 transport 写入为 0。
- 5080 上候选 backend p95 为 `1.771 us`，基线假 transport p95 为
  `1.899 us`；100 点路径转换 p95 为 `25.696 us`。

### MuJoCo 物理 A/B

`benchmark/scripts/sim_nav_route.py` 调用生产 `GrootMover` 依次执行：

```text
前进 0.25 m -> 左转 0.35 rad -> 左移 0.15 m
-> 后退 0.12 m -> 右转 0.20 rad -> 右移 0.10 m
```

导航参考与重构候选的 765 个 50 Hz 控制 tick、每段计划和最终位姿
JSON 逐字节一致，SHA256 均为：

```text
51a7625102d0a197ac28fe2a4cebdc57b4ba5d20e1648d58c2838ce2a32b7b33
```

两者都未摔倒，路线中最低骨盆高度为 `0.723429663756 m`。原有
`sim2sim_groot_mover.py` 的前进相位扫描归一化输出也逐字节一致。

### 操控冷等待 + 导航 + 运控联合烟雾

操控冷服务与上述 MuJoCo 路线同时运行 4.53 s：

- 冷服务保持 1 个线程，窗口内 CPU 增量为 0 个 Linux clock tick。
- RSS 前后均为 `21936 KiB`。
- 测试抓取请求返回 `403 EXECUTION_DISABLED`，没有创建操控子进程。
- 并行时 MuJoCo 路线的 JSON 与单独运行仍逐字节一致。

### 回归测试

- 重构主提交的分组回归：256 passed。
- 恢复导航兼容默认后：导航核心 42 passed；定向 mover/adapter/
  backend/cancel 集成集 67 passed。
- 导航全量测试的重构候选为 142 passed / 7 failed，导航导入参考
  为 105 passed / 7 failed；7 个失败测试名称完全相同，未引入新失败。

## 结论边界

这些结果支持“重构后导航运动逻辑、物理命令速度和冷操控等待
可在 5080 离线环境工作”。不支持以下结论：

- 未在 5080 上运行真实 RealSense，所以 30 FPS 是配置/API 契约验证，
  不是相机吞吐实测。
- 未联通外部 VPR/LLM/NavDP 服务，不是真实地图闭环路线验收。
- 5080 x86 的 CPU 数据不能外推 Jetson 收益。
- 未做 Jetson 或 G1 真机 A/B，不能声称已改善真机步态。

下一阶段仍需要人在环明确授权，在同一台 Jetson、同一地图/路线、同一
外设与温度条件下对比 CPU、run queue、控制周期 p95/p99/max、deadline miss
和步态。

## 2026-07-29 G1-001 旧版导航参数对齐回归

验证提交：`206e05e`

这次回归修正了上述旧报告的一个覆盖缺口。旧报告的 744 帧 A/B 使用了共享的固定 mover 参数，且没有执行真机保留 launcher 的 profile heredoc，因此不能证明 `config_g001` 的实际默认行为不变。该缺口曾经使新版回退到 `warmup_time=0.6` 和 `v_floor=0.10`，而真机旧版实际为 `0.0` 和 `0.12`。

新版不修改 `config_g001.yaml`、`config_bk.yaml` 或导航速度算法。runtime 现在显式写入从旧真机 `start_g1_onboard_nav.sh` 提取的 14 个行为字段，导航使用 `config_g001` 时必须先通过 profile 强校验。新架构所需的 schema、source、motion-bus backend 和 socket 元数据仍保留。

### 独立 profile 和命令 A/B

旧侧从保留 launcher 的 Python heredoc 生成 profile，新侧从 runtime profile builder 生成另一份 profile；两侧分别与 `config_g001` 组合成有效 mover 参数，不再共享手写常量。结果：

- 30 组运动求解和 22 组命令 trace 全部一致，共比较 721 个命令帧。
- `config`、`config_bk`、`config_g001` 共 54 个速度键通过契约比较。
- `0.10 m` 前进在旧、新两侧均为 `(0.12 m/s, 1.0 s, 0.12 m, precise)`，没有 warmup 帧。
- 两份 profile 的 14 个行为字段逐项相等。

独立验证目录中的导航/runtime 核心回归为 `165 passed in 2.67 s`；其中定向 profile 与 `config_g001` 测试为 `48 passed`。

### MuJoCo 实际 profile A/B

两份独立 JSON profile 分别传入生产 `GrootMover` 的 MuJoCo 路线测试。两次运行都产生 645 个控制 tick，没有摔倒，最低骨盆高度均为 `0.744301195007 m`。计划、有效参数、控制帧和位姿 JSON 逐字节一致，SHA-256 均为：

```text
24f6be53f99b193b4b81f7d4b3dd7dc26850b238fcb5810c3bf38302cdc7d125
```

与不允许执行的操控冷服务并行时，操控请求正确返回 `403 EXECUTION_DISABLED`，没有启动 `box_demo_main.py`；并行 MuJoCo 输出的 SHA-256 仍与单独运行一致。

### 性能烟雾值

在 5080 隔离环境和假 transport 下：

- 旧侧 backend p95：`1.920 us`。
- 新侧 backend p95：`1.617 us`。
- motion-bus encode p95：`3.606 us`。
- frame hub p95：`1.524 us`。
- 100 点路径转换 p95：`24.717 us`。

这些数据只用于检查重构没有引入明显的离线开销回归，不是 Jetson 收益数据。

### 结论边界

本轮可以证明新版导航 profile、`config_g001` 有效参数、运动规划、命令帧和 MuJoCo 路线与旧版一致。本轮仍未运行 G1 真机控制、真实 RealSense/定位/路径闭环，也未测量 Jetson CPU 或真机步态；因此不能声称真机“导航 + 操控 + 运控”负载问题已经解决。
