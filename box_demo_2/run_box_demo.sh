#!/bin/bash
# 手动开 box_demo_main.py 的封装 —— 先净化 ROS 污染, 再跑 box_demo.
#
# 为什么要它: ~/.bashrc 无条件 source ROS(fishros + unitree_ros2), 把 ROS 版
# cyclonedds 的 libddsc(/opt/ros/humble/.../libddsc.so, 开了 iceoryx) 塞进
# LD_LIBRARY_PATH。box_demo 用 conda 里 pip 的 cyclonedds 发 rt/arm_sdk 时会
# 错误加载到 ROS 的 libddsc → dds_write.c:318 iox 断言 → core dumped。
# 这里 unset 掉 ROS 的 LD_LIBRARY_PATH/PYTHONPATH 等, 与 start_g1_onboard.sh
# 各 pane 的净化一致。不改 box_demo_main.py, 只是干净地启动它。
#
# 用法(参数原样透传给 box_demo_main.py; 默认不带 --locomotion, 记得自己传):
#   bash ~/zihou/box_demo_2/run_box_demo.sh --locomotion groot  --iface enP8p1s0 --host <SAM3_IP> --port 5300 --point-cloud-stage sor
#   bash ~/zihou/box_demo_2/run_box_demo.sh --locomotion remote --ipc-url http://127.0.0.1:5001 --iface enP8p1s0 --host <SAM3_IP> --port 5300

unset LD_LIBRARY_PATH PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY

CONDA_ENV="${BOX_CONDA_ENV:-robojudo_zihou2}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export UNITREE_DDS_INTERFACE="${UNITREE_DDS_INTERFACE:-enP8p1s0}"

cd "$HOME/zihou/box_demo_2"
echo "[run_box_demo] env=$CONDA_ENV  python=$(which python)  args: $*"
exec python box_demo_main.py "$@"
