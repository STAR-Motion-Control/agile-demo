# G001 导航 + GR00T 运控测试入口

本文件是导航仓库内的快速入口。完整且唯一的测试步骤见：

```text
/home/unitree/zihou/box_demo_1/NAV_ONBOARD_TEST_GUIDE.md
```

导航目录也保留同版镜像：

```text
/home/unitree/workspace/nav_uat/NAV_ONBOARD_TEST_GUIDE.md
```

## 当前配置

- `run_ros.py` 的 Hydra `config_name` 是 `config_bk`，GR00T 测试读取 `src/config_bk.yaml`。
- `src/config.yaml` 是宇树自带运控对照配置，不要把两者混用。
- 普通入口：`/home/unitree/zihou/box_demo_1/start_g1_onboard_nav.sh`
- 自适应回正入口：`start_g1_onboard_nav_taptap.sh`
- 运行时站高：`0.76 m`
- warm-up：默认关闭
- taptap 固定参考：足间距 `0.24 m`、有符号前后脚差 `0.08 m`

普通和 taptap 入口的速度、高度、profile、warm-up 与 `0.4 s` mover 停止保持完全一致。
taptap 只在初始站姿或动作停止后的双足几何超限时回正；状态通过
`/tmp/groot_taptap_status.json` 和 HTTP `/status` 传递，导航距离计时在回正期间暂停。
单个 `rotate/forward` 或短 HTTP hold 结束只 defer，完整 `/planned_action` 批次成功结束才调用
`finish_segment` 检查回正。

真机测试必须有人在环。键盘和 ROS 导航命令必须人工互斥，先完成主指引中的 stop、`10 deg`、
`0.10 m` 分级测试，再接 `/planned_action` 或 `/nav/text_nav`。
