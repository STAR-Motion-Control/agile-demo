#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""箱子尺寸 -> 抱姿 的共享参数化映射 (sim 与真机同一份公式).

纯 stdlib — 被 run_sim_test.py / tune_box_hold.py (5080 sim) 和
hold_box_hug.py (真机 rt/arm_sdk 保持) 共同 import, 保证 sim 里验证的
尺寸->抱姿映射与真机下发的完全一致.

参考点 = 已验证的 benchmark 箱(0.25深 x 0.35宽 x 0.25高) @ sp=-1.2
roll=0.15 elbow=0.7, 递箱位 spawn(0.30, 0.92)。臂几何来自 armprobe 实测:
  肩 roll  -> 手掌横向:  ~0.45 m/rad (roll 越大手越开)
  肩 pitch -> 手掌前伸:  ~0.17 m/rad (sp 越负手越前)
宽度决定 roll(手掌压住箱面, 半宽-2cm 预压); 深度决定 sp+递箱x(箱背面
离开胸口, 手在箱子中段); 高度决定递箱z(箱底放在手掌台面 ~0.795m)。

主接口按【宽】: hug_for_box(width) — 深/高缺省用基准箱值, 对抱姿本身
只有宽起决定作用(深只在换更深的箱时需要), 与用户"主要是箱子的宽"一致.
"""
from __future__ import annotations

# 29 关节索引: 腰 12-14, 左臂 15-21, 右臂 22-28 (与 G1 hg LowCmd 一致)
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 12, 13, 14
L_ARM_BASE, R_ARM_BASE = 15, 22

REF = dict(width=0.35, depth=0.25, height=0.25,
           sp=-1.2, roll=0.15, elbow=0.7, x=0.30, z=0.92)
ROLL_TO_Y = 0.45    # m/rad
SP_TO_X = 0.17      # m/rad (对 -sp)

ROLL_MIN, ROLL_MAX = 0.05, 0.45      # 夹持横滚限幅 (0.45 ≈ 臂展上限)
SP_MIN, SP_MAX = -1.5, -0.6          # 前伸俯仰限幅
PALM_SHELF_Z = REF["z"] - REF["height"] / 2.0   # 手掌台面高 ~0.795m (站立时)


def hug_pose(sp: float = -0.9, roll: float = 0.22, elbow: float = 0.98,
             wrist_roll: float = 0.20) -> dict:
    """抱箱姿 -> {关节索引: 角度}. G1 负肩pitch 才前伸 (armprobe 实测)."""
    L = [sp, roll, 0.0, elbow, wrist_roll, 0.03, -0.03]
    R = [sp, -roll, 0.0, elbow, -wrist_roll, 0.03, 0.03]
    pose = {WAIST_YAW: 0.0, WAIST_ROLL: 0.0, WAIST_PITCH: 0.0}
    pose.update({L_ARM_BASE + i: q for i, q in enumerate(L)})
    pose.update({R_ARM_BASE + i: q for i, q in enumerate(R)})
    return pose


def hug_for_box(width: float, depth: float | None = None,
                height: float | None = None, elbow: float | None = None):
    """(箱宽[, 深, 高]) -> (pose dict, 递箱x, 递箱z, info)。线性外推 + 限幅.

    深/高不给时用基准箱值 — 抱姿只由宽决定. info 里 clamped=True 表示
    宽/深超出臂几何线性区, 抱姿被限幅 — 新尺寸先 tune_box_hold.py 验证.
    """
    depth = REF["depth"] if depth is None else depth
    height = REF["height"] if height is None else height
    roll_raw = REF["roll"] + (width - REF["width"]) / 2.0 / ROLL_TO_Y
    roll = max(ROLL_MIN, min(ROLL_MAX, roll_raw))
    sp_raw = REF["sp"] - (depth - REF["depth"]) / SP_TO_X
    sp = max(SP_MIN, min(SP_MAX, sp_raw))
    spawn_x = REF["x"] + (depth - REF["depth"]) / 2.0
    spawn_z = PALM_SHELF_Z + height / 2.0        # 箱底贴掌面
    pose = hug_pose(sp, roll, elbow if elbow is not None else REF["elbow"])
    info = dict(sp=round(sp, 2), roll=round(roll, 2),
                clamped=(roll != roll_raw or sp != sp_raw))
    return pose, spawn_x, spawn_z, info
