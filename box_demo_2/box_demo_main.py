#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_demo_2 主入口 — 货架前搜索 + 抓取循环

流程：
  1. 预走 30cm 到货架前
  2. 正前方搜索箱子，VLM 检测 + SAM3 确认
  3. 反复抓取直到没有箱子
  4. 采集图像 → SAM3 预测抓取点 → 坐标转换 → IK 检查 → 移动对齐 → 双臂抓取

用法:
    python box_demo_main.py --vlm-endpoint <url> --iface eth0
                            [--no-confirm]
"""

import argparse
import csv
import ctypes
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────
# 本地模块导入（同目录）
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import fcntl
import termios


def _flush_stdin():
    """清空 stdin 缓冲区，避免残留按键（如行走期间误触 Enter）被后续 input() 吃掉。"""
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        while sys.stdin.read(1):
            pass
    except (TypeError, OSError):
        pass
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, fl)

# 须先于 unitree_sdk2py / cyclonedds：避免 LD_LIBRARY_PATH 先加载错误 libddsc（与 RoboJuDo run_pipeline 一致）
_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from capture_and_predict import CameraSession, predict_from_arrays
from dual_arm_target_reach import DualArmController, ARM_JOINTS, ArmKinematics
from vlm_guide import VLMGuide
from robot_move import RobotMover

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

GOAL_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal.csv")


# ─────────────────────────────────────────────────────────────────
# 相机外参（来自 URDF d435_joint，固定不变）
# ─────────────────────────────────────────────────────────────────

# d435_joint origin: xyz (m), 相对于 torso_link
_T_TRANS = np.array([0.0576235, 0.01753, 0.42987])

# d435_joint origin: rpy = [0, pitch, 0]，纯 Y 轴旋转
_PITCH = 0.8307767239493009   # rad ≈ 47.6°
_cp, _sp = math.cos(_PITCH), math.sin(_PITCH)

# R_torso_d435_body = Ry(pitch)
_R_TORSO_BODY = np.array([
    [ _cp,  0,  _sp],
    [ 0,    1,  0  ],
    [-_sp,  0,  _cp],
])

# 光学坐标系 → d435 body 坐标系的固定旋转
#   optical: Z=深度/前, X=右, Y=下
#   body:    X=深度/前, Y=左, Z=上
# 对应关系: X_body = Z_optical, Y_body = -X_optical, Z_body = -Y_optical
_R_BODY_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
])

# 总旋转矩阵：从光学坐标系直接到 torso_link
_R_TORSO_OPT = _R_TORSO_BODY @ _R_BODY_OPTICAL


def cam_to_torso(p_cam: list) -> np.ndarray:
    """将相机光学坐标系下的三维点转换到 torso_link 坐标系。"""
    p = np.asarray(p_cam, dtype=float)
    return _R_TORSO_OPT @ p + _T_TRANS


def save_goal_to_csv(left_torso, right_torso, target_q, left_fk, right_fk, need_header: list) -> None:
    """将目标值追加写入 goal.csv（转置格式：第一列为变量名，后续列为各次数据）"""
    names = [
        "timestamp",
        "left_target_x", "left_target_y", "left_target_z",
        "right_target_x", "right_target_y", "right_target_z",
        "left_fk_x", "left_fk_y", "left_fk_z",
        "right_fk_x", "right_fk_y", "right_fk_z",
    ] + [f"{joint.name}_q" for joint in ARM_JOINTS]
    new_col = [
        datetime.now().isoformat(),
        float(left_torso[0]), float(left_torso[1]), float(left_torso[2]),
        float(right_torso[0]), float(right_torso[1]), float(right_torso[2]),
        float(left_fk[0]), float(left_fk[1]), float(left_fk[2]),
        float(right_fk[0]), float(right_fk[1]), float(right_fk[2]),
    ] + [float(q) for q in target_q]
    if need_header[0]:
        rows = [[names[i], new_col[i]] for i in range(len(names))]
        need_header[0] = False
    else:
        with open(GOAL_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) == len(names) and len(rows[0]) > 0:
            for i in range(len(names)):
                rows[i].append(str(new_col[i]))
        else:
            rows = [[names[i], new_col[i]] for i in range(len(names))]
    with open(GOAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


# ─────────────────────────────────────────────────────────────────
# WaistRotator: 上半身旋转控制（通过 rt/arm_sdk，下半身由 RL_LOWER 接管）
# ─────────────────────────────────────────────────────────────────

class WaistRotator:
    """Controls upper-body yaw rotation via rt/arm_sdk while RL_LOWER handles legs."""

    KP_WAIST = 20.0
    KD_WAIST = 1.5
    KP_ARM = 40.0
    KD_ARM = 2.0
    ROTATE_DURATION = 3.5    # seconds to hold rotation command
    ROTATE_INTERVAL = 0.05   # seconds between frames during rotation
    KEEPALIVE_INTERVAL = 0.05  # seconds between keepalive frames (20Hz)
    SETTLE_TOLERANCE = 0.02  # rad (~1.1°), convergence check for waist position
    SETTLE_TIMEOUT   = 1.0  # max seconds to wait for waist to settle

    def __init__(self):
        self._pub = None          # ChannelPublisher("rt/arm_sdk")
        self._sub = None          # ChannelSubscriber("rt/lowstate")
        self._low_state = None    # latest LowState_ snapshot
        self._arm_q = None        # arm joint positions captured at init
        self._ready = threading.Event()
        self.yaw = 0.0            # accumulated yaw angle (rad)
        self._hold = False
        self._hold_target = 0.0
        self._thread = None

    def initialize(self):
        """Set up DDS publisher/subscriber. Idempotent."""
        if self._pub is not None:
            return
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        self._ready.wait(timeout=1.0)
        print("  腰部旋转器已就绪 (RL_LOWER + rt/arm_sdk)")

    def rotate(self, angle_rad: float):
        """Rotate upper body by angle_rad (positive=left). Blocks, then starts keepalive."""
        if abs(angle_rad) < 0.04:
            return
        start_yaw = self.yaw
        target = start_yaw + angle_rad
        direction = "左" if angle_rad > 0 else "右"
        print(f"  腰部旋转: 往{direction}转 {abs(math.degrees(angle_rad)):.0f}° "
              f"(累计{math.degrees(target):.0f}°)")
        self.initialize()
        # cosine interpolation: zero velocity at start and end → no overshoot
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= self.ROTATE_DURATION:
                break
            progress = 0.5 * (1 - math.cos(math.pi * elapsed / self.ROTATE_DURATION))
            self._publish_once(start_yaw + angle_rad * progress)
            time.sleep(self.ROTATE_INTERVAL)
        # final frame at exact target
        self._publish_once(target)
        # start background keepalive to hold position
        self._start_hold(target)
        self.yaw = target
        self._wait_settle(target)

    def stop_hold(self):
        """Stop background keepalive thread."""
        self._hold = False

    def sync_yaw(self):
        """Re-sync self.yaw from actual robot waist state after external control."""
        self.initialize()
        time.sleep(0.1)
        if self._low_state is not None:
            self.yaw = float(self._low_state.motor_state[12].q)
        else:
            self.yaw = 0.0

    # ── private ──────────────────────────────────────────────────

    def _on_state(self, msg):
        self._low_state = msg
        if self._arm_q is None:
            self._arm_q = [float(msg.motor_state[i].q) for i in range(12, 29)]
        if not self._ready.is_set():
            self._ready.set()

    def _wait_settle(self, target_yaw):
        """Wait until actual waist yaw converges to target within tolerance."""
        t0 = time.time()
        while time.time() - t0 < self.SETTLE_TIMEOUT:
            if self._low_state is not None:
                actual = float(self._low_state.motor_state[12].q)
                if abs(actual - target_yaw) < self.SETTLE_TOLERANCE:
                    return
            time.sleep(0.02)

    def _publish_once(self, target_yaw: float):
        """Publish one arm_sdk frame: waist at target, arms hold position."""
        if self._pub is None or self._arm_q is None:
            return
        cmd = unitree_hg_msg_dds__LowCmd_()
        if self._low_state is not None:
            cmd.mode_machine = self._low_state.mode_machine
            cmd.mode_pr = self._low_state.mode_pr
        # arm_sdk enable flag
        en = cmd.motor_cmd[29]
        en.mode = 1; en.q = 1.0; en.kp = 0.0; en.kd = 0.0; en.tau = 0.0
        # waist joints (12=yaw, 13=roll, 14=pitch)
        for mi, tgt in zip((12, 13, 14), (target_yaw, 0.0, 0.0)):
            mc = cmd.motor_cmd[mi]
            mc.mode = 1; mc.tau = 0.0; mc.dq = 0.0
            mc.q = float(tgt); mc.kp = self.KP_WAIST; mc.kd = self.KD_WAIST
        # arm joints (15-28): hold current position
        for mi in range(15, 29):
            mc = cmd.motor_cmd[mi]
            mc.mode = 1; mc.tau = 0.0; mc.dq = 0.0
            mc.q = self._arm_q[mi - 12]; mc.kp = self.KP_ARM; mc.kd = self.KD_ARM
        self._pub.Write(cmd)

    def _start_hold(self, target_yaw: float):
        """Start or update background keepalive thread."""
        self._hold_target = target_yaw
        if self._thread is not None and self._thread.is_alive():
            # thread already running: just update target, no gap
            return
        self._hold = True
        self._thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._thread.start()

    def _keepalive_loop(self):
        """Background loop: publish arm_sdk to prevent waist drift-back."""
        while self._hold:
            self._publish_once(self._hold_target)
            time.sleep(self.KEEPALIVE_INTERVAL)


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="box_demo: 采集→预测→坐标转换→双臂抓取")
    parser.add_argument("--prompt",     default="box",             help="SAM3 提示词，默认 box")
    parser.add_argument("--host",       default="192.168.112.198", help="SAM3 服务端 IP")
    parser.add_argument("--port",       default=5300, type=int,    help="SAM3 服务端端口")
    parser.add_argument("--iface",      default=None,              help="DDS 网络接口，如 eth0")
    parser.add_argument("--no-confirm", action="store_true",       help="跳过执行前确认提示")
    parser.add_argument(
        "--end-behavior",
        choices=("handoff_rl_lower", "release"),
        default="handoff_rl_lower",
        help="结束行为：对齐到 RL_LOWER 后释放，或直接 release",
    )
    # ── VLM 引导参数 ─────────────────────────────────────────
    parser.add_argument("--vlm-endpoint",  default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="VLM API 地址 (OpenAI 兼容，如 Qwen)")
    parser.add_argument("--vlm-api-key",   default=os.environ.get("QWEN_API_KEY", ""),
                        help="VLM API Key（默认从环境变量 QWEN_API_KEY 读取）")
    parser.add_argument("--vlm-model",     default="qwen-vl-max",
                        help="VLM 模型名称")
    parser.add_argument("--vlm-max-iter",  default=5, type=int,
                        help="VLM 引导最大循环次数")
    parser.add_argument("--vlm-velocity",  default=0.7, type=float,
                        help="行走速度 m/s (默认 0.7，匹配键盘 w 的速度)")
    parser.add_argument("--walk-scale",    default=1.0, type=float,
                        help="行走距离补偿系数 (>1 走更远，补偿 RL 欠追踪)")
    parser.add_argument("--vlm-min-move",  default=200, type=float,
                        help="最小移动时长 (ms)，低于此值跳过")
    args = parser.parse_args()

    print("=" * 65)
    print("  box_demo: 视觉感知 + 双臂抓取")
    print("=" * 65)

    if not args.vlm_endpoint:
        parser.error("--vlm-endpoint 是必须的 (默认千问: https://dashscope.aliyuncs.com/compatible-mode/v1)")

    waist = WaistRotator()

    # ── 辅助函数 ────────────────────────────────────────────────
    def _walk(dist_x: float, dist_y: float):
        """控制机器人移动（前后 + 横向，转向交给腰部）。"""
        if abs(dist_x) < 0.03 and abs(dist_y) < 0.03:
            return
        waist.stop_hold()  # 走路切 RL_FULL，停掉 arm_sdk 腰保持
        moved = False
        if abs(dist_x) >= 0.03:
            scaled = dist_x * args.walk_scale
            tag = f" (补偿×{args.walk_scale})" if args.walk_scale != 1.0 else ""
            direction = "后退" if dist_x < 0 else "直走"
            print(f"  {direction}: {abs(dist_x)*100:.0f}cm{tag}")
            mover.move_forward(scaled)
            moved = True
        if abs(dist_y) >= 0.03:
            side = "左移" if dist_y > 0 else "右移"
            print(f"  {side}: {abs(dist_y)*100:.0f}cm")
            mover.move_left(dist_y)
            moved = True
        if moved:
            time.sleep(1.0)  # 等 RL 策略执行完毕、机器人稳定后再进行下一步

    # ── 快速 IK 参数 ─────────────────────────────────────────────
    _ik_fast = ArmKinematics()
    IK_ERR_LIMIT = 0.040   # 收紧至 40mm，确保快检通过后完整IK必定成功
    IK_FAST_RESTARTS = 3   # 3 重起点平衡速度与精度
    IK_FAST_MAX_ITER = 250 # 250 次已足够收敛
    IDEAL_REACH_X = 0.40   # 理想的抓取距离（torso 系 X 轴，手臂半伸展、重心稳定）

    def _compute_move_distance(target_left, target_right, box_center_y):
        """根据几何位置计算机器人需要移动的距离（纯几何估算）。"""
        move_x = target_left[0] - IDEAL_REACH_X
        move_y = box_center_y
        print(f"  几何估算: 前移 {move_x*100:+.0f}cm, 横移 {move_y*100:+.0f}cm (目标距={IDEAL_REACH_X*100:.0f}cm)")
        return move_x, move_y

    def _ik_error_at(target, left):
        q, err = _ik_fast.inverse_kinematics(
            target.tolist(), left=left, num_restarts=IK_FAST_RESTARTS, max_iter=IK_FAST_MAX_ITER)
        return q, err

    def _check_workspace_ik(name, target, left):
        q, err = _ik_error_at(target, left)
        if err <= IK_ERR_LIMIT:
            print(f"  ✓  {name}手在工作空间内 (IK误差={err*1000:.1f}mm)")
            return True, err, q
        else:
            print(f"  ✗  {name}手超出工作空间 (IK误差={err*1000:.1f}mm > {IK_ERR_LIMIT*1000:.0f}mm)")
            return False, err, q

    # ── VLM / DDS 初始化（VLM 现在必须） ──────────────────────────
    print("\n[VLM Guide] 启动 VLM 视觉引导模式")
    print("  初始化 DDS 通信 (用于 RL Pipeline 行走控制)...")
    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)
    mover = RobotMover(velocity=args.vlm_velocity)
    mover.initialize()
    time.sleep(0.5)
    vlm = VLMGuide(args.vlm_endpoint, args.vlm_api_key, args.vlm_model)

    # ── 图像保存目录 ─────────────────────────────────────────────
    img_dir = os.path.join(os.path.dirname(__file__), "img")
    os.makedirs(img_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # 主循环：预走 → 正前方反复抓取
    # ═══════════════════════════════════════════════════════════════
    max_attempts = args.vlm_max_iter
    _MOVE_DECAY_FACTOR = 0.6
    _MOVE_DECAY_FLOOR = 0.6
    _MOVE_MAX_X = 0.50
    _MOVE_MAX_Y = 0.30

    with CameraSession() as camera:

        # ── 预走：先向货架方向前进 30cm ──
        # print("\n  预走: 向货架前进 30cm...")
        # _walk(0.30, 0.0)
        # time.sleep(0.5)

        # ── 腰部回正 ──
        if abs(waist.yaw) > 0.01:
            waist.rotate(-waist.yaw)
            waist.stop_hold()
            time.sleep(0.3)

        # ═══════════════════════════════════════════════════════════════
        # 搜索+抓取：仅正前方
        # ═══════════════════════════════════════════════════════════════

        while True:  # 反复检查+抓取

            # ── VLM 检测 + SAM3 确认 ──────────────────────────
            print(f"\n{'=' * 55}")
            print(f"  [正前方] 拍照检测箱子...")
            print(f"{'=' * 55}")

            color, depth = camera.capture()
            cv2.imwrite(os.path.join(img_dir, "one.png"), color)
            print(f"  已保存拍照图像 → img/one.png")
            vlm_result = vlm.query(color, depth)

            if not vlm_result.get("box_visible"):
                print(f"  [正前方] 未发现箱子，结束抓取")
                if not args.no_confirm:
                    _flush_stdin(); input("  按 Enter 结束...")
                break

            # ── VLM 侧移对齐：箱子被裁切时侧移直到完整显示 ──
            while vlm_result.get("box_in_frame", "complete") != "complete":
                box_frame = vlm_result.get("box_in_frame", "complete")
                if box_frame == "left_cut":
                    step = 0.05 * 1.2
                    print(f"  [VLM] 箱子左侧被裁切，左移 {abs(step)*100:.0f}cm 让箱子完整显示...")
                    waist.stop_hold()
                    mover.move_left(step)
                    time.sleep(1.5)
                elif box_frame == "right_cut":
                    step = -0.05 * 1.2
                    print(f"  [VLM] 箱子右侧被裁切，右移 {abs(step)*100:.0f}cm 让箱子完整显示...")
                    waist.stop_hold()
                    mover.move_left(step)
                    time.sleep(1.5)
                elif box_frame == "both_cut":
                    print(f"  [VLM] 箱子两侧都被裁切（可能太近），后退 5cm...")
                    waist.stop_hold()
                    mover.move_forward(-0.05)
                    time.sleep(1.5)
                elif box_frame == "top_cut":
                    print(f"  [VLM] 箱子在图片上方被裁切（太近），前移 4cm...")
                    _walk(0.04, 0.0)
                else:
                    break
                color, depth = camera.capture()
                vlm_result = vlm.query(color, depth)
                if not vlm_result.get("box_visible"):
                    print(f"  [VLM] 移动后未发现箱子，结束抓取")
                    break

            if not vlm_result.get("box_visible"):
                if not args.no_confirm:
                    _flush_stdin(); input("  按 Enter 结束...")
                break

            # VLM 发现完整箱子 → SAM3 预测（循环直到找到完整两个面）
            while True:
                sam3_check = predict_from_arrays(color, depth, args.prompt, args.host, args.port)
                if sam3_check is not None:
                    break
                # SAM3 预测失败：根据深度判断是太远还是太近
                h, w = depth.shape
                center_depth = float(np.median(depth[h//3:2*h//3, w//3:2*w//3])) / 1000.0
                NEAR_THRESHOLD = 0.45
                if center_depth <= 0 or center_depth < NEAR_THRESHOLD:
                    print(f"  [正前方] SAM3 只找到一个面（深度 {center_depth*100:.0f}cm，太近或无深度数据），后退 5cm...")
                    _walk(-0.05, 0.0)
                else:
                    print(f"  [正前方] SAM3 只找到一个面（深度 {center_depth*100:.0f}cm，太远），前移 3cm...")
                    _walk(0.03, 0.0)
                color, depth = camera.capture()

            found_angle = waist.yaw
            print(f"  ✓ 在正前方发现箱子! 相对正前方 {math.degrees(found_angle):+.0f}°")

            # ── 腰回正 + 机身转向箱子 ──────────────────────────
            # 已注释：假设箱子只出现在正前方，不需要转腰/转向
            # print(f"\n  腰回正，机身转向箱子方向 ({math.degrees(found_angle):+.0f}°)...")
            # if abs(waist.yaw) > 0.01:
            #     waist.rotate(-waist.yaw)
            # waist.stop_hold()
            # time.sleep(0.15)
            #
            # if abs(found_angle) > math.radians(30):
            #     mover.initialize()
            #     time.sleep(0.3)
            #     mover.rotate(found_angle)
            #     time.sleep(0.8)

            # ── 侧向对齐：已跳过（RL 策略侧移不可靠，靠 y 偏移量补偿）──

            # ── 抓取循环 ──────────────────────────────────────
            _move_decay = 1.0
            _prev_ik_err = float("inf")
            _yaw_aligned = False

            for attempt in range(1, max_attempts + 1):
                print(f"\n{'─' * 55}")
                print(f"  第 {attempt}/{max_attempts} 次尝试")
                print(f"{'─' * 55}")

                # ── Step 1+2: 采集 + SAM3 预测 ──────────────────────
                print("\n[Step 1+2] 采集图像 → SAM3 预测抓取点")
                color, depth = camera.capture()

                cv2.imwrite(os.path.join(img_dir, f"attempt{attempt}_color.png"), color)
                cv2.imwrite(os.path.join(img_dir, f"attempt{attempt}_depth.png"), depth)

                result = predict_from_arrays(color, depth, args.prompt, args.host, args.port)
                if result is None:
                    # SAM3 预测失败：根据深度判断是太远还是太近
                    # 取图像中心区域的平均深度（mm → m）
                    h, w = depth.shape
                    center_depth = float(np.median(depth[h//3:2*h//3, w//3:2*w//3])) / 1000.0
                    NEAR_THRESHOLD = 0.45  # 小于 45cm 认为太近
                    if center_depth > 0 and center_depth < NEAR_THRESHOLD:
                        print(f"\n[错误] SAM3 预测失败（中心深度 {center_depth*100:.0f}cm < {NEAR_THRESHOLD*100:.0f}cm，太近），后退 5cm...")
                        _walk(-0.05, 0.0)
                    else:
                        print(f"\n[错误] SAM3 预测失败（中心深度 {center_depth*100:.0f}cm，可能太远只看到一个面），前移 3cm...")
                        _walk(0.03, 0.0)
                    continue

                # ── 箱子边缘检测：已注释（假设箱子只在正前方，不需要转腰重拍） ─
                # CAMERA_HFOV_HALF = math.radians(34.5)   # D435 彩色半 FOV ≈ 34.5°
                # EDGE_THRESHOLD   = 0.55
                # MAX_RECAPTURE    = 2
                # _total_waist_for_recapture = 0.0
                #
                # for _rc in range(MAX_RECAPTURE):
                #     center_cam = result["center"]
                #     angle_h = math.atan2(center_cam[0], center_cam[2])
                #     ratio = abs(angle_h) / CAMERA_HFOV_HALF
                #
                #     if ratio < EDGE_THRESHOLD:
                #         break
                #
                #     print(f"  箱子中心偏离 {math.degrees(angle_h):+.1f}° (FOV 比率 {ratio:.0%})，转腰重拍...")
                #     waist.rotate(-angle_h)
                #     _total_waist_for_recapture += angle_h
                #
                #     color, depth = camera.capture()
                #     cv2.imwrite(os.path.join(img_dir, f"attempt{attempt}_rc{_rc}_color.png"), color)
                #     cv2.imwrite(os.path.join(img_dir, f"attempt{attempt}_rc{_rc}_depth.png"), depth)
                #     result = predict_from_arrays(color, depth, args.prompt, args.host, args.port)
                #     if result is None:
                #         print(f"  重拍后 SAM3 预测失败，跳出重拍循环")z
                #         break
                #
                # if abs(_total_waist_for_recapture) > 0.01:
                #     print(f"  转腰回正 {-math.degrees(_total_waist_for_recapture):+.1f}°")
                #     waist.rotate(_total_waist_for_recapture)

                # ── Step 3: 坐标转换 ────────────────────────────────
                if result is None:
                    h, w = depth.shape
                    center_depth = float(np.median(depth[h//3:2*h//3, w//3:2*w//3])) / 1000.0
                    NEAR_THRESHOLD = 0.45
                    if center_depth <= 0 or center_depth < NEAR_THRESHOLD:
                        print(f"\n[错误] SAM3 预测失败（中心深度 {center_depth*100:.0f}cm，太近或无深度数据），后退 5cm...")
                        _walk(-0.05, 0.0)
                    else:
                        print(f"\n[错误] SAM3 预测失败（中心深度 {center_depth*100:.0f}cm，太远），前移 3cm...")
                        _walk(0.03, 0.0)
                    continue

                print("\n[Step 3] 坐标转换：相机系 → torso 系")
                left_torso  = cam_to_torso(result["grasp_left"])
                right_torso = cam_to_torso(result["grasp_right"])
                # SAM3 给出手腕目标；沿 -X 补偿手长，得到 IK 用的掌心目标
                left_torso[0]  -= 0.16;  left_torso[1]  += 0.05;  left_torso[2]  += 0.09
                right_torso[0] -= 0.16;  right_torso[1] += 0.05;  right_torso[2] += 0.09

                box_center_cam = np.array(result["center"], dtype=float)
                box_center_torso = cam_to_torso(box_center_cam.tolist())

                print(f"  torso系 left  = [{left_torso[0]:.4f}, {left_torso[1]:.4f}, {left_torso[2]:.4f}]")
                print(f"  torso系 right = [{right_torso[0]:.4f}, {right_torso[1]:.4f}, {right_torso[2]:.4f}]")
                print(f"  箱子中心 (torso): x={box_center_torso[0]:.3f}  y={box_center_torso[1]:.3f}  z={box_center_torso[2]:.3f}")

                # ── Step 3.05: 检查是否站太近 ──────────────────────
                extent = result.get("extent", [0, 0, 0])
                min_extent = min(abs(e) for e in extent)
                max_extent = max(abs(e) for e in extent)
                box_x = float(box_center_torso[0])
                if min_extent < 0.05 and max_extent > 0.10 and box_x < 0.35:
                    step_back = max(0.3 - box_x, 0.15)
                    print(f"\n[太近] 箱子最小维度 {min_extent*100:.1f}cm，距机器人 {box_x*100:.0f}cm → 后退 {step_back*100:.0f}cm")
                    waist.stop_hold()
                    mover.move_forward(-step_back)
                    time.sleep(1.5)  # 等后退完成并稳定
                    continue

                # ── Step 3.1: 箱子对齐 ────────────────────────────
                # 偏移较大时才侧移（RL侧移不精确，小偏移让IK处理不对称抓取）
                need_center = abs(float(box_center_torso[1])) > 0.15  # 横向偏移 > 15cm 才侧移

                if need_center and not _yaw_aligned:
                    _yaw_aligned = True
                    lateral = float(box_center_torso[1])
                    step = lateral * 1.2  # 多移 20% 补偿 RL 欠追踪
                    side = "左" if lateral > 0 else "右"
                    print(f"\n[侧移] 箱子中心偏{side} {abs(lateral)*100:.0f}cm，跨步 {abs(step)*100:.0f}cm")
                    waist.stop_hold()
                    mover.move_left(step)
                    time.sleep(1.5)  # 等侧移完成并稳定
                    continue

                # ── Step 3.4: 距离前移检查（抓取点太远则先走近） ────────
                MAX_REACH_X = 0.45  # 抓取点前向距离上限（超过则先走近）
                avg_reach_x = (float(left_torso[0]) + float(right_torso[0])) / 2.0
                if avg_reach_x > MAX_REACH_X:
                    move_forward = (avg_reach_x - IDEAL_REACH_X) * 0.7
                    print(f"\n[距离前移] 抓取点平均 {avg_reach_x*100:.0f}cm > {MAX_REACH_X*100:.0f}cm，"
                          f"前移 {move_forward*100:.0f}cm (目标距 {IDEAL_REACH_X*100:.0f}cm)")
                    _walk(move_forward, 0.0)
                    continue

                # ── Step 3.5: IK 工作空间检查 ────────────────────────
                print("\n[Step 3.5] IK 工作空间检查")
                left_ok, left_err, left_q_fast = _check_workspace_ik("左", left_torso, left=True)
                right_ok, right_err, right_q_fast = _check_workspace_ik("右", right_torso, left=False)

                if not left_ok or not right_ok:
                    print("\n[结果] 抓取点超出工作空间，计算机器人需要移动的距离...")
                    move_x, move_y = _compute_move_distance(
                        left_torso, right_torso, float(box_center_torso[1]))

                    cur_err = left_err if left_err > right_err else right_err
                    if cur_err < _prev_ik_err * 0.90:
                        _move_decay = min(1.0, _move_decay * 1.5)
                        print(f"  IK误差缩小 ({_prev_ik_err*100:.0f}→{cur_err*100:.0f}cm), 衰减恢复至×{_move_decay:.1f}")
                    else:
                        _move_decay = max(_MOVE_DECAY_FLOOR, _move_decay * _MOVE_DECAY_FACTOR)
                    _prev_ik_err = cur_err

                    if abs(move_x) > 0.01 or abs(move_y) > 0.01:
                        safe_x = max(-_MOVE_MAX_X, min(_MOVE_MAX_X, move_x * _move_decay))
                        safe_y = max(-_MOVE_MAX_Y, min(_MOVE_MAX_Y, move_y * _move_decay))
                        x_desc = f"{'前' if safe_x > 0 else '后'}移 {abs(safe_x)*100:.0f}cm" if abs(safe_x) > 0.01 else ""
                        y_desc = f"往{'左' if safe_y > 0 else '右'}移 {abs(safe_y)*100:.0f}cm" if abs(safe_y) > 0.01 else ""
                        desc = "，".join(filter(None, [x_desc, y_desc]))
                        decay_tag = f" [衰减×{_move_decay:.1f}]" if _move_decay < 1.0 else ""
                        print(f"\n  >>> 需要: {desc}{decay_tag}")

                        _walk(safe_x, safe_y)
                        continue
                    else:
                        print("\n[错误] 无法找到可达位置，跳过本次尝试。")
                        break

                # ── Step 4: 重置衰减参数 ──────────────────────────────
                _move_decay = 1.0
                _prev_ik_err = float("inf")

                # ── Step 5: 完整 IK 求解 ────────────────────────────
                print("\n[Step 5] 求解逆运动学并初始化控制器")
                ik_start = time.time()
                try:
                    controller = DualArmController(
                        left_target=left_torso.tolist(),
                        right_target=right_torso.tolist(),
                        end_behavior=args.end_behavior,
                        left_q0=left_q_fast,
                        right_q0=right_q_fast,
                    )
                except ValueError as e:
                    ik_elapsed = time.time() - ik_start
                    print(f"\n[IK 失败] {e} (耗时 {ik_elapsed:.1f}s)")
                    print("完整 IK 无法收敛，目标点可能在工作空间边缘。")
                    if attempt < max_attempts:
                        print("自动前移一小步后重新采集...")
                        _walk(0.10, 0.0)
                        continue
                    else:
                        print(f"\n已达最大尝试次数 ({max_attempts})，结束抓取...")
                        break

                ik_elapsed = time.time() - ik_start
                print(f"  IK 求解完成 (耗时 {ik_elapsed:.1f}s)")

                print(f"\n  左手目标 (torso): x={left_torso[0]:+.3f}  y={left_torso[1]:+.3f}  z={left_torso[2]:+.3f} m")
                print(f"  右手目标 (torso): x={right_torso[0]:+.3f}  y={right_torso[1]:+.3f}  z={right_torso[2]:+.3f} m")

                print("\n  目标关节角 (ARM_JOINTS 顺序：左臂7 + 右臂7 + 腰3)：")
                for i, joint in enumerate(ARM_JOINTS):
                    q = controller.target_q[i]
                    deg = math.degrees(q)
                    print(f"    {joint.name:20s}  q={q:+.4f} rad ({deg:+.2f}°)")

                csv_need_header = [not os.path.exists(GOAL_CSV_PATH)]
                save_goal_to_csv(
                    left_torso, right_torso, controller.target_q,
                    controller.left_fk, controller.right_fk,
                    csv_need_header,
                )
                print(f"\n  已写入目标值 → {GOAL_CSV_PATH}")

                if not args.no_confirm:
                    print("\n⚠  执行前请确认：")
                    print("   1. 机器人周围无障碍物，手臂可以自由活动")
                    print("   2. 抓取目标点位置正确（见上方坐标）")
                    _flush_stdin(); input("按 Enter 开始执行，Ctrl+C 取消...\n")

                # ── Step 6: 漂移补偿 + 交权 ─────────────────────────
                print("[Step 6] 停掉腰部 keepalive，切换到 RL_LOWER，交权给手臂...")
                waist.stop_hold()
                time.sleep(0.15)
                mover.shutdown()
                time.sleep(0.5)

                print("[Step 7] 开始执行双臂运动")
                try:
                    controller.run()
                except KeyboardInterrupt:
                    print("\n用户中断。")
                    mover.shutdown()
                    sys.exit(0)

                print("\n抓取完成! 检查同一位置是否还有箱子...")

                # 跳出 for attempt 循环 → 回到 while True 再次 VLM 检查
                break

            # ── 抓取后重置状态 ──────────────────────────────────────
            waist.sync_yaw()
            waist._arm_q = None
            mover.initialize()
            time.sleep(0.5)

            # 重置腰部回正，准备下一次拍照
            if abs(waist.yaw) > 0.01:
                waist.rotate(-waist.yaw)
            waist.stop_hold()
            time.sleep(0.3)

    # ── 抓取结束 ──────────────────────────────────────────
    print("\n抓取结束。")
    if abs(waist.yaw) > 0.01:
        waist.rotate(-waist.yaw)
    mover.release()


if __name__ == "__main__":
    main()
