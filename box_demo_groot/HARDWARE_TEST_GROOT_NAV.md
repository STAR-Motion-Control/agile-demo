# GR00T-WBC + nav_uat 真机测试入口

当前 G001 使用本体运行拓扑。完整且唯一的参数、启动、安全和分级测试说明见：

```text
/home/unitree/zihou/box_demo_1/NAV_ONBOARD_TEST_GUIDE.md
```

不要沿用旧版 5080 控制拓扑，也不要从其他历史文档复制速度或回正参数。

## 当前入口

- 普通对照：`bash start_g1_onboard_nav.sh`
- 自适应回正：`bash start_g1_onboard_nav_taptap.sh`
- 导航配置：`/home/unitree/workspace/nav_uat/src/config_bk.yaml`
- 宇树自带运控对照配置：`config.yaml`
- 运行时站高：`0.76 m`
- warm-up：默认关闭，两种入口使用同一开关

两种入口的速度、高度、profile 和 mover 停止保持参数一致；taptap 入口只额外启用基于双足几何的
自适应回正。禁止通过旧的 `GROOT_TAPTAP_ADAPTIVE` 环境变量或固定 `2.2 s` 等待判断当前模式。
导航回正边界是完整 `/planned_action` 批次，不是批次内单个 `rotate/forward` 原语。

## 安全边界

必须有人在环、吊架和急停就绪。键盘与 ROS 导航都会写
`/tmp/robojudo_ext_cmd.json`，必须人工互斥。`space` 是速度归零并保持平衡，`o` 是 DAMP 急停。

Codex 不得代替现场人员启动真机控制脚本。
