#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared command-schedule generators for the three benchmark tests.

Pure stdlib — imported by BOTH the MuJoCo sim tests (5080) and the real-robot
IPC test scripts, so sim and real drive the exact same command profiles.

All profiles return (total_time_s, sample_fn) where sample_fn(t) -> dict with
keys vx, vy, wz (m/s, rad/s), height (m), pitch (rad, waist lean; 0 = upright).
"""
from __future__ import annotations

import math


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------------------------------------------ test 1
def speed_profile(v1: float = 0.5, hold1_s: float = 2.0, v2: float = 1.0,
                  ramp_s: float = 1.5, hold2_s: float = 4.0,
                  settle_s: float = 2.0, soft_start_s: float = 0.5,
                  height: float = 0.72):
    """加速测试: settle -> soft-start到v1 -> 保持hold1 -> ramp到v2 -> 保持hold2 -> 停.

    直接给 1m/s 会弹射起步, 所以 0.5 起步 2s 后过渡到 1.0 (用户规格).
    """
    t0 = settle_s                       # stand still
    t1 = t0 + soft_start_s              # 0 -> v1
    t2 = t1 + hold1_s                   # hold v1
    t3 = t2 + ramp_s                    # v1 -> v2
    t4 = t3 + hold2_s                   # hold v2
    total = t4 + 1.0                    # trailing zero-velocity stop window

    def f(t: float) -> dict:
        if t < t0:
            vx = 0.0
        elif t < t1:
            vx = v1 * (t - t0) / max(1e-6, soft_start_s)
        elif t < t2:
            vx = v1
        elif t < t3:
            vx = v1 + (v2 - v1) * (t - t2) / max(1e-6, ramp_s)
        elif t < t4:
            vx = v2
        else:
            vx = 0.0
        return dict(vx=vx, vy=0.0, wz=0.0, height=height, pitch=0.0)

    return total, f


# ------------------------------------------------------------------ test 2
def squat_profile(n_reps: int = 20, h_top: float = 0.72, h_low: float = 0.50,
                  descend_s: float = 2.0, dwell_low_s: float = 1.5,
                  ascend_s: float = 2.0, dwell_top_s: float = 1.5,
                  settle_s: float = 2.0, pitch_max: float = 0.0):
    """下蹲测试: n_reps 个 (下蹲-停-站起-停) 循环, 速度恒 0.

    pitch_max>0 时腰部随下蹲深度成比例缓慢前倾(控制重心), 站起时同步恢复 —
    倾角与高度进度线性耦合, 在最低点达到 pitch_max, 回到顶端归零.
    """
    cycle = descend_s + dwell_low_s + ascend_s + dwell_top_s
    total = settle_s + n_reps * cycle + 1.0

    def f(t: float) -> dict:
        if t < settle_s or t >= settle_s + n_reps * cycle:
            return dict(vx=0.0, vy=0.0, wz=0.0, height=h_top, pitch=0.0)
        tc = (t - settle_s) % cycle
        if tc < descend_s:                       # going down
            frac = tc / max(1e-6, descend_s)
        elif tc < descend_s + dwell_low_s:       # hold bottom
            frac = 1.0
        elif tc < descend_s + dwell_low_s + ascend_s:  # going up
            frac = 1.0 - (tc - descend_s - dwell_low_s) / max(1e-6, ascend_s)
        else:                                    # hold top
            frac = 0.0
        h = h_top + (h_low - h_top) * frac
        return dict(vx=0.0, vy=0.0, wz=0.0, height=h, pitch=pitch_max * frac)

    def rep_of(t: float) -> int:
        """Which rep (0-based) t falls in; -1 outside the cycling window."""
        if t < settle_s or t >= settle_s + n_reps * cycle:
            return -1
        return int((t - settle_s) // cycle)

    f.rep_of = rep_of          # type: ignore[attr-defined]
    f.cycle_s = cycle          # type: ignore[attr-defined]
    f.n_reps = n_reps          # type: ignore[attr-defined]
    return total, f


# ------------------------------------------------------------ backward walk
def backward_profile(speed: float = 0.3, distance: float = 1.5,
                     lean_gain: float = 0.0, settle_s: float = 2.0,
                     soft_start_s: float = 0.8, height: float = 0.72,
                     axis: str = "back"):
    """后退测试: 给定距离 distance, vx=-speed 后退, 走完停下.

    lean_gain>0 时同步下发前倾补偿 pitch = lean_gain * |vx|(随软起动比例
    渐入渐出) — 对应真机 dwbc 后退重心靠后要摔的问题: 上半身微前倾把
    CoM 拉回支撑面中心。真机侧同一映射由 adapter --back-lean-gain 实现
    (经 target_upper_body_pose 腰pitch -> 下身策略 torso rpy 通道, 与
    sim 的 rpy_cmd/command[4:7] 同源)。
    """
    walk_T = distance / max(1e-6, speed) + soft_start_s * 0.5
    t0 = settle_s
    t1 = t0 + walk_T
    total = t1 + 2.5                    # 停稳观察窗

    def f(t: float) -> dict:
        if t < t0 or t >= t1:
            return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)
        scale = _clamp((t - t0) / max(1e-6, soft_start_s), 0.0, 1.0)
        # 尾段 0.5s 缓出, 避免急停摆动
        tail = _clamp((t1 - t) / 0.5, 0.0, 1.0)
        s = scale * tail
        if axis == "lat":                        # 横移(左) — 高度扫掠共用
            return dict(vx=0.0, vy=speed * s, wz=0.0, height=height, pitch=0.0)
        return dict(vx=-speed * s, vy=0.0, wz=0.0, height=height,
                    pitch=lean_gain * speed * s)

    f.walk_T = walk_T                  # type: ignore[attr-defined]
    f.walk_window = (t0, t1)           # type: ignore[attr-defined]
    return total, f


# ------------------------------------------------------------ taptap 回正
def taptap_profile(motion: str = "fwd", speed: float = 0.3, amount: float = 1.0,
                   taptap_s: float = 1.2, taptap_cmd: float = 0.08,
                   taptap_period_s: float = 0.4, settle_s: float = 2.0,
                   soft_start_s: float = 0.6, height: float = 0.72,
                   taptap_mode: str = "vx"):
    """停止回正测试: 运动一段 -> 停 -> (可选)原地踏步回正窗口 -> 静置观察.

    参考宇树官方运控: 每段动作结束后原地踏步恢复标准静止站姿(双脚固定间距、
    朝向正), 把开环漂移在段间清零。踏步实现 = vx 以 ±taptap_cmd 方波交替
    (周期 taptap_period_s, 净位移≈0), 幅值刚超 Walk 阈值(0.05)让步态不落进
    Balance 冻结; 期间 height 固定站高。taptap_s=0 为无回正对照组。

    motion: fwd|back|lat|turn; amount = 距离 m (turn 为角度 rad, speed=wz)。
    """
    move_T = amount / max(1e-6, speed) + (0.0 if motion == "turn" else soft_start_s * 0.5)
    t0 = settle_s
    t1 = t0 + move_T                    # 运动结束
    t2 = t1 + taptap_s                  # 踏步回正结束
    total = t2 + 3.0                    # 静置观察窗

    def f(t: float) -> dict:
        if t0 <= t < t1:                # 运动段(缓入 + 尾段0.4s缓出)
            scale = _clamp((t - t0) / max(1e-6, soft_start_s), 0.0, 1.0)
            tail = _clamp((t1 - t) / 0.4, 0.0, 1.0)
            s = scale * tail
            v = dict(fwd=(speed * s, 0.0, 0.0), back=(-speed * s, 0.0, 0.0),
                     lat=(0.0, speed * s, 0.0), turn=(0.0, 0.0, speed * s))[motion]
            return dict(vx=v[0], vy=v[1], wz=v[2], height=height, pitch=0.0)
        if t1 <= t < t2:                # 踏步回正窗口
            phase = int((t - t1) / max(1e-6, taptap_period_s)) % 2
            amp = taptap_cmd
            if taptap_mode == "vx_taper":   # 幅值渐减(1.0->0.7, 保持超阈值)
                amp = taptap_cmd * (1.0 - 0.3 * (t - t1) / max(1e-6, taptap_s))
            tap = amp if phase == 0 else -amp
            if taptap_mode == "wz":         # 原地小转踏步(净角度≈0, 平移小)
                return dict(vx=0.0, vy=0.0, wz=tap, height=height, pitch=0.0)
            return dict(vx=tap, vy=0.0, wz=0.0, height=height, pitch=0.0)
        return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)

    f.phases = (t0, t1, t2)            # type: ignore[attr-defined]
    return total, f


# ------------------------------------------------------------------ test 3
def circle_profile(radius: float = 0.6, speed: float = 0.20, side: str = "left",
                   settle_s: float = 2.0, soft_start_s: float = 0.8,
                   height: float = 0.72, bulge: float | None = None,
                   cmd_gain: float = 1.0, lead_strafe_s: float = 0.0):
    """绕障测试: 固定朝向的半(椭)圆绕行 (不转身 — 避免 benchmark 绕圈朝向变化过大).

    障碍在正前方 `radius` 米处; 机器人保持初始朝向, 用 vx/vy 合成绕到障碍正后方
    (行进 180°), 终点在障碍前方 2*radius 处, 朝向不变 — 之后键盘直行.

    几何(半椭圆, bulge=radius 时退化为圆): 前向半轴 a=radius, 侧向半轴 b=bulge.
    p(θ) = (a(1-cosθ), ±b sinθ), θ:0->π. 朝向固定 => 机体系速度=世界系速度
    = (a sinθ, ±b cosθ)·ω. 小半径时用 bulge>radius 保证侧向净空(机器人半宽
    ~0.25 + 柱半径 0.15 => b 至少 ~0.55 才有余量).

    cmd_gain: 命令速度增益, 补偿真实/仿真的速度欠跟踪(实走≈命令×55-70%) —
    路径尺寸与速度成正比, 欠跟踪会把圈走小、蹭到障碍; gain>1 把命令圈放大.
    """
    sgn = +1.0 if side == "left" else -1.0
    a = radius
    b = radius if bulge is None else max(bulge, radius)
    omega = speed / max(1e-6, max(a, b))        # peak |v| == speed
    arc_T = math.pi / omega
    total = settle_s + lead_strafe_s + arc_T + 1.5

    def f(t: float) -> dict:
        if t < settle_s or t >= settle_s + lead_strafe_s + arc_T:
            return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)
        tt = t - settle_s
        scale = _clamp(tt / max(1e-6, soft_start_s), 0.0, 1.0) * cmd_gain
        if tt < lead_strafe_s:
            # 小半径先"侧撤": 横移 + 轻微后撤 — 障碍 0.4m 时起步摆臂就能扫到柱子
            # (实测 right_rubber_hand 蹭柱), 后撤半步给摆臂让出前向空间
            return dict(vx=-0.3 * speed * scale, vy=sgn * speed * scale, wz=0.0,
                        height=height, pitch=0.0)
        theta = omega * (tt - lead_strafe_s)
        vx = a * omega * math.sin(theta) * scale
        vy = sgn * b * omega * math.cos(theta) * scale
        return dict(vx=vx, vy=vy, wz=0.0, height=height, pitch=0.0)

    f.arc_T = arc_T            # type: ignore[attr-defined]
    f.omega = omega            # type: ignore[attr-defined]
    f.end_point = (2.0 * radius, 0.0)   # type: ignore[attr-defined]
    f.bulge = b                # type: ignore[attr-defined]
    return total, f


def heading_circle_profile(radius: float = 0.6, speed: float = 0.25,
                           side: str = "left", turn_rate: float = 0.35,
                           yaw_gain: float = 1.0, fwd_gain: float = 1.0,
                           lead_out_m: float = 0.0, turn_comp: float = 1.0,
                           settle_s: float = 2.0, pause_s: float = 1.0,
                           height: float = 0.72):
    """绕障模式B: 朝向跟随(用 wz) — 面向行进方向绕过障碍, 超越后转回原朝向.

    障碍在正前方 `radius` 米处。三段:
      ① 原地转 90°(side=left 左转) — 面向弧线切向;
      ② 沿半径 radius 的半圆弧走: vx=speed 前进 + wz=∓speed/radius 持续转向
         (朝向始终沿切向), 弧上行进 180° = "按半径算超越"(到达障碍正后方,
          出发点前方 2*radius 处);
      ③ 原地再转 90°(同向) — 朝向回到原前进方向, 之后键盘直行.
    段间各停 pause_s 稳一下。

    yaw_gain: 补偿 wz 欠跟踪(dwbc yaw 实测 65-89% 效率 → ~1.2);
    fwd_gain: 补偿 vx 欠跟踪(dwbc ~0.7 → ~1.4)。开环, sim 标定后真机微调.
    """
    sgn = +1.0 if side == "left" else -1.0
    r_arc = radius + lead_out_m              # 外扩后按更大的弧绕(给手臂让空间)
    omega_arc = speed / max(1e-6, r_arc)
    # turn_comp: 原地转90°的开环欠转补偿(dwbc yaw 效率~78% -> ~1.25)
    turn_T = (math.pi / 2.0) / max(1e-6, turn_rate) * turn_comp
    lead_T = lead_out_m / max(1e-6, speed) if lead_out_m > 0 else 0.0
    arc_T = math.pi / omega_arc
    t1 = settle_s                    # ① turn 90
    t2 = t1 + turn_T + pause_s       # pause
    t2b = t2 + lead_T                # ②a 直行外扩(离开障碍连线)
    t3 = t2b + arc_T                 # ②b arc
    t4 = t3 + pause_s                # pause
    t5 = t4 + turn_T                 # ③ turn 90 back
    total = t5 + 1.5

    def f(t: float) -> dict:
        if t < t1:
            return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)
        if t < t1 + turn_T:                       # ① 左/右转 90°
            return dict(vx=0.0, vy=0.0, wz=sgn * turn_rate * yaw_gain,
                        height=height, pitch=0.0)
        if t < t2:
            return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)
        if t < t2b:                               # ②a 直行外扩 lead_out_m
            return dict(vx=speed * fwd_gain, vy=0.0, wz=0.0,
                        height=height, pitch=0.0)
        if t < t3:                                # ②b 弧线: 前进 + 反向持续转
            return dict(vx=speed * fwd_gain, vy=0.0,
                        wz=-sgn * omega_arc * yaw_gain, height=height, pitch=0.0)
        if t < t4:
            return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)
        if t < t5:                                # ③ 同向再转 90° 回原朝向
            return dict(vx=0.0, vy=0.0, wz=sgn * turn_rate * yaw_gain,
                        height=height, pitch=0.0)
        return dict(vx=0.0, vy=0.0, wz=0.0, height=height, pitch=0.0)

    f.arc_T = arc_T                  # type: ignore[attr-defined]
    f.turn_T = turn_T                # type: ignore[attr-defined]
    f.end_point = (2.0 * radius, 0.0)  # type: ignore[attr-defined]
    return total, f
