# G001 导航 + GR00T 运控测试入口

本文件只保留入口，防止旧参数与主指引冲突。完整测试流程见：

```text
/home/unitree/zihou/box_demo_1/NAV_ONBOARD_TEST_GUIDE.md
```

当前普通入口为 `start_g1_onboard_nav.sh`，自适应回正入口为
`start_g1_onboard_nav_taptap.sh`。导航实际加载
`/home/unitree/workspace/nav_uat/src/config_bk.yaml`；`config.yaml` 是宇树自带运控对照配置。

运行时站高为 `0.76 m`，warm-up 默认关闭。普通与 taptap 的速度、高度、profile 和 mover
停止保持参数一致；taptap 只额外启用双足几何检测与条件回正，不再使用固定 `2.2 s` 等待。
导航只在整个 `/planned_action` 批次完成后检查回正，批次内部原语停止使用 `defer_recovery`。

真机测试必须有人在环，并严格执行主指引中的分级测试和键盘/ROS 单写者要求。
