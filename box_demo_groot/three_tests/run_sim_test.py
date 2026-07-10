#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified sim harness for the three benchmark tests (speed / squat / circle).

Runs on the 5080, headless EGL. Pick interpreter by controller:
  dwbc    -> ~/GR00T-WholeBodyControl/.venv_wbc/bin/python
  agile29 -> ~/miniconda3/envs/hdmi/bin/python

Examples:
  run_sim_test.py --controller dwbc    --test speed  --v2-list 1.0,1.2,1.4
  run_sim_test.py --controller agile29 --test squat  --h-low-list 0.55,0.50,0.45
  run_sim_test.py --controller dwbc    --test squat_box --pitch-list 0,0.15,0.3
  run_sim_test.py --controller dwbc    --test circle --radius-list 0.4,0.6,0.8,1.0

Outputs per case: metrics JSON + (unless --no-video) an mp4, under --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_inject
from schedules import (backward_profile, circle_profile,
                       heading_circle_profile, speed_profile, squat_profile,
                       taptap_profile)

# 箱子尺寸->抱姿映射抽到 box_hug.py (sim 与真机 hold_box_hug.py 共用同一份
# 公式); hug_pose/hug_for_box/REF 从这里 re-export, tune_box_hold 等不受影响.
from box_hug import REF, hug_for_box, hug_pose  # noqa: F401  (re-export)


def make_backend(controller: str, scene_xml: str):
    if controller == "dwbc":
        from sim_dwbc_backend import DwbcBackend
        return DwbcBackend(scene_xml)
    if controller == "agile29":
        from sim_agile_backend import AgileBackend
        return AgileBackend(scene_xml)
    raise SystemExit(f"unknown controller {controller}")


def run_profile(b, total, f, *, video_frames=None, fps=25, yaw_hold_gain=0.0,
                extra_state_cb=None):
    """Drive sample_fn f through the backend, collecting states (+frames)."""
    states = []
    n = int(total / b.ctrl_dt)
    render_every = max(1, int(round((1.0 / fps) / b.ctrl_dt)))
    for k in range(n):
        t = k * b.ctrl_dt
        cmd = f(t)
        if yaw_hold_gain > 0.0:
            yaw = b.state()["yaw"]
            cmd = dict(cmd)
            cmd["wz"] = cmd.get("wz", 0.0) + max(-0.15, min(0.15, -yaw_hold_gain * yaw))
        b.step(cmd)
        st = b.state()
        st["profile_t"] = t
        st["cmd"] = {k2: float(v) for k2, v in cmd.items()}
        if extra_state_cb:
            extra_state_cb(t, st)
        states.append(st)
        if video_frames is not None and (k % render_every) == 0:
            video_frames.append(b.render())
    return states


def save_video(frames, path: Path, fps=25):
    if not frames:
        return
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"  video: {path}")


def horizontal_speed(st) -> float:
    v = st["base_vel_world"]
    return float(math.hypot(float(v[0]), float(v[1])))


def fell(st) -> bool:
    return st["pelvis_z"] < 0.35 or st["tilt_deg"] > 50.0


# ===================================================================== tests
def test_speed(args, out: Path):
    scene = scene_inject.make_scene(args.scene, tag=f"speed_{args.controller}")
    b = make_backend(args.controller, scene)
    results = []
    for v2 in [float(x) for x in args.v2_list.split(",")]:
        b.reset(settle_s=2.5)
        total, f = speed_profile(v1=args.v1, hold1_s=2.0, v2=v2, ramp_s=args.ramp_s,
                                 hold2_s=args.hold2_s, height=b.stand_height)
        frames = [] if args.video else None
        states = run_profile(b, total, f, video_frames=frames, fps=args.fps)
        # peak 1s-window mean horizontal speed
        w = int(1.0 / b.ctrl_dt)
        sp = [horizontal_speed(s) for s in states]
        win = [sum(sp[i:i+w]) / w for i in range(0, max(1, len(sp) - w))]
        peak = max(win) if win else 0.0
        anyfall = any(fell(s) for s in states)
        tilt_max = max(s["tilt_deg"] for s in states)
        # stop check: displacement in the trailing 1s of zero command
        stop_disp = float(abs(states[-1]["base_pos"][0] - states[-int(1.0/b.ctrl_dt)]["base_pos"][0]))
        ok = (peak >= args.target_speed) and not anyfall
        r = dict(v2_cmd=v2, peak_1s_speed=round(peak, 3), tilt_max=round(tilt_max, 1),
                 fell=anyfall, stop_disp_m=round(stop_disp, 3), success=ok)
        results.append(r)
        print(f"[speed/{args.controller}] cmd v2={v2:.2f} -> peak(1s)={peak:.3f} m/s "
              f"tilt_max={tilt_max:.1f} fell={anyfall} stop_disp={stop_disp:.2f} "
              f"{'PASS' if ok else 'fail'}")
        if frames:
            save_video(frames, out / f"speed_{args.controller}_v{v2:.1f}.mp4", args.fps)
    return results


def _squat_common(args, out: Path, with_box: bool):
    tag = ("squat_box" if with_box else "squat") + f"_{args.controller}"
    # 箱子尺寸: --box-size 深,宽,高(米, 完整边长) -> 场景 + 抱姿 + 递箱位
    dims = [float(v) for v in args.box_size.split(",")]
    assert len(dims) == 3, "--box-size 深,宽,高"
    depth, width, height = dims
    box_half = (depth / 2, width / 2, height / 2)
    d_pose, d_x, d_z, d_info = hug_for_box(width, depth, height)
    if args.hug_sp is None and args.hug_roll is None and args.hug_elbow is None:
        hug = d_pose                                   # 全部由箱子尺寸推导
    else:                                              # 手动覆盖(缺省项用推导值)
        hug = hug_pose(args.hug_sp if args.hug_sp is not None else d_info["sp"],
                       args.hug_roll if args.hug_roll is not None else d_info["roll"],
                       args.hug_elbow if args.hug_elbow is not None else REF["elbow"])
    box_x = args.box_x if args.box_x is not None else d_x
    box_z = args.box_z if args.box_z is not None else d_z
    if with_box:
        print(f"[{tag}] box 深x宽x高={depth}x{width}x{height}m mass={args.box_mass}kg "
              f"-> hug {d_info if args.hug_sp is None else '(manual)'} "
              f"spawn=({box_x:.2f},{box_z:.2f})")
    scene = scene_inject.make_scene(
        args.scene, box=((0.35, 0.93) if with_box else None),
        box_half=box_half, box_mass=args.box_mass, tag=tag)
    b = make_backend(args.controller, scene)
    results = []
    h_lows = [float(x) for x in args.h_low_list.split(",")]
    pitches = [float(x) for x in args.pitch_list.split(",")] if with_box else [0.0]
    for h_low in h_lows:
        for pitch in pitches:
            b.reset(settle_s=2.5)
            if with_box:
                b.set_upper_override(hug)
                for _ in range(int(2.5 / b.ctrl_dt)):     # arms into pose + re-settle
                    b.step(dict(vx=0, vy=0, wz=0, height=b.stand_height, pitch=0))
                b.spawn_box(x=box_x, z=box_z)             # 递箱子
                if args.box_weld:
                    # 用户流程 "确认抱到后再开始" 的 sim 等价: 递到怀里即视为抱稳
                    # (真机是柔顺力控抓握; sim 刚性 PD 挤压过脆) — benchmark 同款 weld
                    for _ in range(int(0.5 / b.ctrl_dt)):
                        b.step(dict(vx=0, vy=0, wz=0, height=b.stand_height, pitch=0))
                    if not b.weld_box():
                        raise SystemExit("scene lacks test_box_weld equality")
                held = True
                for _ in range(int(args.hold_check_s / b.ctrl_dt)):  # hold check
                    b.step(dict(vx=0, vy=0, wz=0, height=b.stand_height, pitch=0))
                    if b.state()["box_torso_dist"] > 0.55:
                        held = False
                        break
                if not held:
                    print(f"[squat_box/{args.controller}] h_low={h_low} pitch={pitch}: "
                          f"box NOT held at spawn -> skip (tune --box-x/--box-z/--hug-*)")
                    results.append(dict(h_low=h_low, pitch=pitch, held_at_spawn=False,
                                        stable=0, n=args.reps, success=False))
                    continue
            total, f = squat_profile(n_reps=args.reps, h_top=b.stand_height, h_low=h_low,
                                     descend_s=args.descend_s, dwell_low_s=args.dwell_s,
                                     ascend_s=args.ascend_s, dwell_top_s=args.dwell_s,
                                     pitch_max=pitch)
            frames = [] if args.video else None
            rep_log = {}

            def cb(t, st, f=f, rep_log=rep_log):
                r = f.rep_of(t)
                if r < 0:
                    return
                e = rep_log.setdefault(r, dict(min_z=9.9, max_z=0.0, tilt=0.0,
                                               fell=False, box_lost=False, knee=False))
                e["min_z"] = min(e["min_z"], st["pelvis_z"])
                e["max_z"] = max(e["max_z"], st["pelvis_z"])
                e["tilt"] = max(e["tilt"], st["tilt_deg"])
                e["fell"] |= fell(st)
                if with_box:
                    # 深蹲时箱子合法地跟着躯干降到 z~0.33, 掉落判据用躯干相对距离
                    # + 真正落地 (箱半高 0.125 -> 地上 ~0.13)
                    e["box_lost"] |= (st["box_torso_dist"] > 0.60 or st["box_pos"][2] < 0.20)
                    e["knee"] |= st.get("knee_box_contact", False)

            run_profile(b, total, f, video_frames=frames, fps=args.fps, extra_state_cb=cb)
            stable = 0
            depths = []
            for r in range(args.reps):
                e = rep_log.get(r)
                if not e:
                    continue
                depths.append(b.stand_height - e["min_z"])
                # reach: 至少完成命令深度的 60% (高度跟踪有静差 — AGILE cmd0.50
                # 实到 ~0.58; 用户指标核心是"稳", 深度按比例给分)
                cmd_depth = b.stand_height - h_low
                reach = (b.stand_height - e["min_z"]) >= max(0.08, 0.6 * cmd_depth)
                recover = e["max_z"] >= b.stand_height - 0.08
                bad = e["fell"] or (with_box and e["box_lost"]) or e["tilt"] > 35.0
                if reach and recover and not bad:
                    stable += 1
            ok = stable >= args.pass_reps
            knee_hits = sum(1 for e in rep_log.values() if e.get("knee"))
            mean_depth = sum(depths) / len(depths) if depths else 0.0
            r = dict(h_low=h_low, pitch=pitch, stable=stable, n=args.reps,
                     mean_depth_cm=round(mean_depth * 100, 1),
                     knee_contact_reps=knee_hits, success=ok)
            if with_box:
                r["held_at_spawn"] = True
            results.append(r)
            print(f"[{tag}] h_low={h_low:.2f} pitch={pitch:.2f} -> "
                  f"stable {stable}/{args.reps} depth~{mean_depth*100:.0f}cm "
                  f"knee_hits={knee_hits} {'PASS' if ok else 'fail'}")
            if frames:
                save_video(frames, out / f"{tag}_h{h_low:.2f}_p{pitch:.2f}.mp4", args.fps)
    return results


def _signed_pitch(quat) -> float:
    """带符号躯干 pitch (rad, +=前倾). MuJoCo quat 序 (w,x,y,z), ZYX 约定."""
    w, x, y, z = (float(v) for v in quat)
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def _signed_roll(quat) -> float:
    """带符号躯干 roll (rad, +=向左倾). ZYX 约定."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def test_backward(args, out: Path):
    """后退稳定性: 给定距离多速度后退; --lean-gain-list 0 = 原版基线."""
    scene = scene_inject.make_scene(args.scene, tag=f"backward_{args.controller}")
    b = make_backend(args.controller, scene)
    if hasattr(b, "pitch_mode"):
        b.pitch_mode = args.pitch_mode
    results = []
    heights = ([float(x) for x in args.height_list.split(",")]
               if args.height_list else [None])
    for v in [float(x) for x in args.back_speed_list.split(",")]:
      for gain in [float(x) for x in args.lean_gain_list.split(",")]:
        for h in heights:
            hh = b.stand_height if h is None else h
            b.reset(settle_s=2.5)
            total, f = backward_profile(speed=v, distance=args.back_dist,
                                        lean_gain=gain, height=hh,
                                        axis=args.axis)
            frames = [] if args.video else None
            states = run_profile(b, total, f, video_frames=frames, fps=args.fps)
            walk = [s for s in states
                    if s["cmd"]["vx"] < -0.01 or s["cmd"]["vy"] > 0.01]
            # torso 绝对俯仰 = 骨盆 pitch + 腰pitch关节角 (真正决定 CoM 前后)
            torso = [_signed_pitch(s["base_quat"]) + s.get("waist_pitch_q", 0.0)
                     for s in walk] or [0.0]
            # 位移: back 取 x(后退为负), lat 取 y(左移为正)
            ax = 1 if args.axis == "lat" else 0
            disp_x = float(states[-1]["base_pos"][ax] - states[0]["base_pos"][ax])
            pitches = [_signed_pitch(s["base_quat"]) for s in walk] or [0.0]
            tilt_max = max(s["tilt_deg"] for s in states)
            anyfall = any(fell(s) for s in states)
            # 停稳窗(最后 1.5s)的俯仰振荡幅度 = 稳定性代理
            w = int(1.5 / b.ctrl_dt)
            endp = [_signed_pitch(s["base_quat"]) for s in states[-w:]]
            end_osc = math.degrees(max(endp) - min(endp))
            ok = (not anyfall) and (abs(disp_x) >= 0.8 * args.back_dist)
            rolls = [_signed_roll(s["base_quat"]) for s in walk] or [0.0]
            r = dict(speed=v, lean_gain=gain, mode=args.pitch_mode,
                     axis=args.axis, height=round(hh, 2),
                     roll_mean_deg=round(math.degrees(sum(rolls) / len(rolls)), 2),
                     roll_osc_deg=round(math.degrees(max(rolls) - min(rolls)), 2),
                     disp_x=round(disp_x, 3),
                     pitch_mean_deg=round(math.degrees(sum(pitches) / len(pitches)), 2),
                     pitch_min_deg=round(math.degrees(min(pitches)), 2),
                     torso_pitch_mean_deg=round(math.degrees(sum(torso) / len(torso)), 2),
                     tilt_max=round(tilt_max, 1), end_pitch_osc_deg=round(end_osc, 2),
                     fell=anyfall, success=ok)
            results.append(r)
            print(f"[backward/{args.controller}] {args.axis} v={v:.2f} h={hh:.2f} "
                  f"gain={gain:.2f} -> disp={disp_x:+.2f}m "
                  f"pitch(mean/min)={r['pitch_mean_deg']:+.1f}/"
                  f"{r['pitch_min_deg']:+.1f}deg tilt_max={tilt_max:.1f} "
                  f"end_osc={end_osc:.1f}deg fell={anyfall} {'PASS' if ok else 'fail'}")
            if frames:
                save_video(frames, out / (f"backward_{args.controller}_{args.axis}"
                                          f"_v{v:.1f}_h{hh:.2f}_g{gain:.2f}.mp4"),
                           args.fps)
    return results


def _stance(st):
    """双脚相对(yaw系): dx=前后错位 m(标准站姿≈0), dy=横向间距 m."""
    fl, fr = st.get("foot_l"), st.get("foot_r")
    if fl is None or fr is None:
        return None
    yaw = st["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = float(fl[0] - fr[0]), float(fl[1] - fr[1])
    return ex * c + ey * s, -ex * s + ey * c


def _mean_std_cv(values):
    values = [float(value) for value in values]
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return mean, std, std / abs(mean) if abs(mean) > 1e-9 else 0.0


def _foot_yaw(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_error(a: float, b: float) -> float:
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


def _gait_metrics(states, start_t: float, end_t: float) -> dict:
    """Measure touchdown timing and stance geometry during one motion segment."""
    window = [
        state
        for state in states
        if start_t <= state.get("profile_t", state["t"]) <= end_t
    ]
    if not window or "foot_l_contact" not in window[0]:
        return {
            "touchdown_count": 0,
            "alternation_errors": 0,
            "step_interval_mean_s": 0.0,
            "step_interval_cv": 0.0,
            "stride_interval_cv": 0.0,
            "stride_length_cv": 0.0,
            "double_support_width_min_m": 0.0,
            "double_support_width_cv": 0.0,
        }

    contact_key = {"L": "foot_l_contact", "R": "foot_r_contact"}
    position_key = {"L": "foot_l", "R": "foot_r"}
    swing_ticks = {"L": 0, "R": 0}
    armed = {"L": False, "R": False}
    events = []
    for state in window:
        for side in ("L", "R"):
            if not bool(state[contact_key[side]]):
                swing_ticks[side] += 1
                if swing_ticks[side] >= 3:
                    armed[side] = True
                continue
            if armed[side]:
                events.append((float(state["t"]), side, state[position_key[side]][:2].copy()))
            armed[side] = False
            swing_ticks[side] = 0

    step_intervals = [events[i][0] - events[i - 1][0] for i in range(1, len(events))]
    alternation_errors = sum(events[i][1] == events[i - 1][1] for i in range(1, len(events)))
    stride_intervals = []
    stride_lengths = []
    previous = {}
    for timestamp, side, position in events:
        if side in previous:
            prev_t, prev_pos = previous[side]
            stride_intervals.append(timestamp - prev_t)
            stride_lengths.append(math.hypot(float(position[0] - prev_pos[0]),
                                             float(position[1] - prev_pos[1])))
        previous[side] = (timestamp, position)

    widths = []
    for state in window:
        if state["foot_l_contact"] and state["foot_r_contact"]:
            stance = _stance(state)
            if stance is not None:
                widths.append(abs(float(stance[1])))

    step_mean, _, step_cv = _mean_std_cv(step_intervals)
    _, _, stride_interval_cv = _mean_std_cv(stride_intervals)
    _, _, stride_length_cv = _mean_std_cv(stride_lengths)
    width_mean, _, width_cv = _mean_std_cv(widths)
    return {
        "touchdown_count": len(events),
        "alternation_errors": int(alternation_errors),
        "step_interval_mean_s": round(step_mean, 3),
        "step_interval_cv": round(step_cv, 3),
        "stride_interval_cv": round(stride_interval_cv, 3),
        "stride_length_cv": round(stride_length_cv, 3),
        "double_support_width_mean_m": round(width_mean, 3),
        "double_support_width_min_m": round(min(widths), 3) if widths else 0.0,
        "double_support_width_cv": round(width_cv, 3),
    }


def test_taptap(args, out: Path):
    """停止回正: 运动->停 之后有无踏步窗对站姿/高度恢复的影响(对照实验)."""
    scene = scene_inject.make_scene(args.scene, tag=f"taptap_{args.controller}")
    b = make_backend(args.controller, scene)
    MOVES = dict(fwd=(0.30, 1.0), back=(0.20, 0.8), lat=(0.25, 0.5),
                 turn=(0.40, math.pi / 2))
    results = []
    cmds = [float(x) for x in args.taptap_cmd_list.split(",")]
    for m in args.taptap_motion_list.split(","):
        v, amount = MOVES[m]
        for ts in [float(x) for x in args.taptap_s_list.split(",")]:
            for tc in cmds:
                if ts == 0.0 and tc != cmds[0]:
                    continue                      # 对照组与踏步幅值无关, 跑一次
                b.reset(settle_s=2.5)
                total, f = taptap_profile(motion=m, speed=v, amount=amount,
                                          taptap_s=ts, taptap_cmd=tc,
                                          height=b.stand_height,
                                          taptap_mode=args.taptap_mode)
                frames = [] if args.video else None
                states = run_profile(b, total, f, video_frames=frames, fps=args.fps)
                t0, t1, t2 = f.phases
                k = lambda t: min(len(states) - 1, int(t / b.ctrl_dt))
                ref = states[max(0, k(t0) - 15):k(t0)]          # 运动前 0.3s
                endw = states[-int(0.8 / b.ctrl_dt):]           # 末尾 0.8s
                sref = [_stance(s) for s in ref if _stance(s)]
                send = [_stance(s) for s in endw if _stance(s)]
                dx0 = sum(x for x, _ in sref) / max(1, len(sref))
                dy0 = sum(y for _, y in sref) / max(1, len(sref))
                dxf = sum(x for x, _ in send) / max(1, len(send))
                dyf = sum(y for _, y in send) / max(1, len(send))
                z0 = sum(s["pelvis_z"] for s in ref) / max(1, len(ref))
                zf = sum(s["pelvis_z"] for s in endw) / max(1, len(endw))
                p1, p2 = states[k(t1)]["base_pos"], states[k(t2)]["base_pos"]
                drift = math.hypot(float(p2[0] - p1[0]), float(p2[1] - p1[1]))
                yaw_dr = math.degrees(abs(float(states[k(t2)]["yaw"])
                                          - float(states[k(t1)]["yaw"])))
                gait = _gait_metrics(states, t0, t1)
                final_yaw_diff = 0.0
                if "foot_l_quat" in endw[-1]:
                    final_yaw_diff = math.degrees(abs(_angle_error(
                        _foot_yaw(endw[-1]["foot_l_quat"]),
                        _foot_yaw(endw[-1]["foot_r_quat"]),
                    )))
                anyfall = any(fell(s) for s in states)
                r = dict(motion=m, taptap_s=ts, taptap_cmd=tc,
                         taptap_mode=args.taptap_mode,
                         stance_dx_err=round(abs(dxf - dx0), 3),
                         stance_dy_err=round(abs(dyf - dy0), 3),
                         height_err=round(abs(zf - z0), 3),
                         taptap_drift_m=round(drift, 3),
                         taptap_yaw_deg=round(yaw_dr, 1),
                         final_foot_yaw_diff_deg=round(final_yaw_diff, 1),
                         fell=anyfall,
                         success=not anyfall)
                r.update(gait)
                results.append(r)
                print(f"[taptap/{args.controller}] {m} ts={ts:.1f} cmd={tc:.2f} -> "
                      f"站姿err 前后{r['stance_dx_err']*100:.1f}cm/"
                      f"间距{r['stance_dy_err']*100:.1f}cm 高度err{r['height_err']*100:.1f}cm "
                      f"踏步漂移{drift*100:.1f}cm/{yaw_dr:.1f}deg "
                      f"touch={gait['touchdown_count']} stepCV={gait['step_interval_cv']:.2f} "
                      f"strideCV={gait['stride_length_cv']:.2f} "
                      f"minW={gait['double_support_width_min_m']*100:.1f}cm fell={anyfall}")
                if frames:
                    save_video(frames, out / f"taptap_{args.controller}_{m}_ts{ts:.1f}_c{tc:.2f}.mp4",
                               args.fps)
    return results


def test_squat(args, out: Path):
    return _squat_common(args, out, with_box=False)


def test_squat_box(args, out: Path):
    return _squat_common(args, out, with_box=True)


def test_circle(args, out: Path):
    results = []
    gains = [float(x) for x in args.cmd_gain_list.split(",")]
    for radius in [float(x) for x in args.radius_list.split(",")]:
        scene = scene_inject.make_scene(args.scene, pillar=(radius, 0.0),
                                        tag=f"circle_{args.controller}_r{radius:.1f}")
        b = make_backend(args.controller, scene)
        bulge = max(radius, args.bulge_min)
        for gain in gains:
            b.reset(settle_s=2.5)
            total, f = circle_profile(radius=radius, speed=args.circle_speed,
                                      side=args.side, height=b.stand_height,
                                      bulge=bulge, cmd_gain=gain,
                                      lead_strafe_s=args.lead_strafe_s)
            frames = [] if args.video else None
            states = run_profile(b, total, f, video_frames=frames, fps=args.fps,
                                 yaw_hold_gain=args.yaw_hold_gain)
            anyfall = any(fell(s) for s in states)
            hit = any(s.get("pillar_contact", False) for s in states)
            min_clear = min(s.get("pillar_clearance", 9.9) for s in states)
            end = states[-1]
            ex, ey = f.end_point
            end_err = float(math.hypot(float(end["base_pos"][0]) - ex,
                                       float(end["base_pos"][1]) - ey))
            end_yaw = abs(math.degrees(end["yaw"]))
            ok = (not anyfall) and (not hit) and min_clear > 0.05 \
                and end_err < args.end_tol and end_yaw < 20.0
            r = dict(radius=radius, bulge=bulge, cmd_gain=gain,
                     yaw_hold=args.yaw_hold_gain, min_clearance=round(min_clear, 3),
                     pillar_contact=hit, end_pos_err=round(end_err, 3),
                     end_yaw_deg=round(end_yaw, 1), fell=anyfall, success=ok)
            results.append(r)
            print(f"[circle/{args.controller}] r={radius:.1f} b={bulge:.2f} "
                  f"gain={gain:.1f} -> clear_min={min_clear:.2f} hit={hit} "
                  f"end_err={end_err:.2f}m end_yaw={end_yaw:.0f}deg fell={anyfall} "
                  f"{'PASS' if ok else 'fail'}")
            if frames:
                save_video(frames, out / f"circle_{args.controller}_r{radius:.1f}"
                           f"_g{gain:.1f}.mp4", args.fps)
    return results


def test_circle_heading(args, out: Path):
    """绕障模式B: 朝向跟随(wz 参与) — 转90° -> 弧线(vx+wz) -> 超越后转回."""
    results = []
    combos = [tuple(float(v) for v in c.split(":"))
              for c in args.gain_combo_list.split(",")]
    for radius in [float(x) for x in args.radius_list.split(",")]:
        scene = scene_inject.make_scene(
            args.scene, pillar=(radius, 0.0),
            tag=f"circleH_{args.controller}_r{radius:.1f}")
        b = make_backend(args.controller, scene)
        for yaw_gain, fwd_gain in combos:
            b.reset(settle_s=2.5)
            total, f = heading_circle_profile(radius=radius, speed=args.circle_speed,
                                              side=args.side,
                                              turn_rate=args.turn_rate,
                                              yaw_gain=yaw_gain, fwd_gain=fwd_gain,
                                              lead_out_m=args.lead_out_m,
                                              turn_comp=args.turn_comp,
                                              height=b.stand_height)
            frames = [] if args.video else None
            states = run_profile(b, total, f, video_frames=frames, fps=args.fps)
            anyfall = any(fell(s) for s in states)
            hit = any(s.get("pillar_contact", False) for s in states)
            min_clear = min(s.get("pillar_clearance", 9.9) for s in states)
            end = states[-1]
            ex, ey = f.end_point
            end_err = float(math.hypot(float(end["base_pos"][0]) - ex,
                                       float(end["base_pos"][1]) - ey))
            end_yaw = abs(math.degrees(end["yaw"]))
            ok = (not anyfall) and (not hit) and min_clear > 0.05 \
                and end_err < args.end_tol_heading and end_yaw < 25.0
            r = dict(radius=radius, yaw_gain=yaw_gain, fwd_gain=fwd_gain,
                     min_clearance=round(min_clear, 3), pillar_contact=hit,
                     end_pos_err=round(end_err, 3), end_yaw_deg=round(end_yaw, 1),
                     fell=anyfall, success=ok)
            results.append(r)
            print(f"[circleH/{args.controller}] r={radius:.1f} yg={yaw_gain:.1f} "
                  f"fg={fwd_gain:.1f} -> clear_min={min_clear:.2f} hit={hit} "
                  f"end_err={end_err:.2f}m end_yaw={end_yaw:.0f}deg fell={anyfall} "
                  f"{'PASS' if ok else 'fail'}")
            if frames:
                save_video(frames, out / f"circleH_{args.controller}_r{radius:.1f}"
                           f"_yg{yaw_gain:.1f}_fg{fwd_gain:.1f}.mp4", args.fps)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", required=True, choices=("dwbc", "agile29"))
    ap.add_argument("--test", required=True,
                    choices=("speed", "squat", "squat_box", "circle",
                             "circle_heading", "backward", "taptap"))
    ap.add_argument("--taptap-motion-list", default="fwd,back,lat,turn")
    ap.add_argument("--taptap-s-list", default="0,1.2",
                    help="taptap: 踏步窗时长列表 s (0=无回正对照)")
    ap.add_argument("--taptap-cmd-list", default="0.08",
                    help="taptap: 踏步激励幅值列表 m/s (刚超 Walk 阈值 0.05)")
    ap.add_argument("--taptap-mode", default="vx", choices=("vx", "vx_taper", "wz"),
                    help="taptap: 激励方式 (vx方波 | 幅值渐减 | wz原地小转)")
    ap.add_argument("--back-speed-list", default="0.2,0.3,0.4,0.5",
                    help="backward: 后退速度列表 m/s")
    ap.add_argument("--back-dist", type=float, default=1.5,
                    help="backward: 后退距离 m")
    ap.add_argument("--lean-gain-list", default="0",
                    help="backward: 前倾补偿增益列表 rad/(m/s), 0=原版基线")
    ap.add_argument("--axis", default="back", choices=("back", "lat"),
                    help="backward: 测试轴 (back=后退, lat=左横移)")
    ap.add_argument("--height-list", default="",
                    help="backward: 行走高度列表 m (空=站立高度; 高度扫掠用)")
    ap.add_argument("--pitch-mode", default="rpy",
                    choices=("rpy", "waist", "waist_obscomp"),
                    help="backward: 前倾注入方式 (waist_obscomp=偏置腰pitch目标"
                         "+观测补偿, 真机 --back-lean-gain 同构, 推荐)")
    ap.add_argument("--scene", default=os.path.expanduser(
        "~/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/scene_29dof.xml"))
    ap.add_argument("--out", default=os.path.expanduser("~/three_tests_results"))
    ap.add_argument("--video", action="store_true", default=True)
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--fps", type=int, default=25)
    # speed
    ap.add_argument("--v1", type=float, default=0.5)
    ap.add_argument("--ramp-s", type=float, default=1.5)
    ap.add_argument("--hold2-s", type=float, default=4.0)
    ap.add_argument("--v2-list", default="1.0,1.2,1.4")
    ap.add_argument("--target-speed", type=float, default=1.0)
    # squat
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--pass-reps", type=int, default=18)
    ap.add_argument("--h-low-list", default="0.55,0.50,0.45")
    ap.add_argument("--pitch-list", default="0,0.15,0.3")
    ap.add_argument("--descend-s", type=float, default=2.0)
    ap.add_argument("--ascend-s", type=float, default=2.0)
    ap.add_argument("--dwell-s", type=float, default=1.5)
    ap.add_argument("--box-size", default="0.25,0.35,0.25",
                    help="箱子完整尺寸 深,宽,高(米)。抱姿/递箱位默认由此推导(hug_for_box)")
    ap.add_argument("--box-mass", type=float, default=2.0)
    ap.add_argument("--hug-sp", type=float, default=None,
                    help="覆盖肩pitch(负=前伸)。缺省=由 --box-size 推导")
    ap.add_argument("--hug-roll", type=float, default=None,
                    help="覆盖肩roll(夹持开度)。缺省=由 --box-size 推导")
    ap.add_argument("--hug-elbow", type=float, default=None)
    ap.add_argument("--box-x", type=float, default=None,
                    help="覆盖递箱位x。缺省=由 --box-size 推导")
    ap.add_argument("--box-z", type=float, default=None,
                    help="覆盖递箱位z。缺省=由 --box-size 推导")
    ap.add_argument("--hold-check-s", type=float, default=3.0)
    ap.add_argument("--box-weld", action="store_true", default=True,
                    help="递箱后 weld 到躯干(=真机'确认抱到'; benchmark 同款)")
    ap.add_argument("--no-box-weld", dest="box_weld", action="store_false",
                    help="用自由箱+摩擦挤压 (sim 里对姿态极敏感)")
    # circle
    ap.add_argument("--radius-list", default="0.4,0.6,0.8,1.0")
    ap.add_argument("--circle-speed", type=float, default=0.20)
    ap.add_argument("--side", default="left", choices=("left", "right"))
    ap.add_argument("--yaw-hold-gain", type=float, default=0.8)
    ap.add_argument("--end-tol", type=float, default=0.60)
    ap.add_argument("--cmd-gain-list", default="1.0,1.4,1.8",
                    help="velocity-command gain sweep (undertracking compensation)")
    ap.add_argument("--bulge-min", type=float, default=0.55,
                    help="min lateral semi-axis (robot half-width + pillar r + margin)")
    ap.add_argument("--lead-strafe-s", type=float, default=0.0,
                    help="小半径先纯横移这么久再入弧 (r=0.4 推荐 1.0-1.5)")
    # circle_heading (模式B: 朝向跟随)
    ap.add_argument("--turn-rate", type=float, default=0.35,
                    help="原地转90°的角速度 rad/s")
    ap.add_argument("--gain-combo-list", default="1.0:1.0,1.2:1.4,1.3:1.4",
                    help="yaw_gain:fwd_gain 组合扫描 (补偿 wz/vx 欠跟踪)")
    ap.add_argument("--turn-comp", type=float, default=1.0,
                    help="原地转90°时长补偿(开环欠转; dwbc ~1.25)")
    ap.add_argument("--lead-out-m", type=float, default=0.25,
                    help="模式B: 转90°后先直行外扩(米)再入弧 — 给内侧手臂让空间")
    ap.add_argument("--end-tol-heading", type=float, default=0.90,
                    help="模式B端点容差(m) — 开环转向误差会累积, 比模式A松")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    fn = dict(speed=test_speed, squat=test_squat, squat_box=test_squat_box,
              circle=test_circle, circle_heading=test_circle_heading,
              backward=test_backward, taptap=test_taptap)[args.test]
    results = fn(args, out)
    j = out / f"{args.test}_{args.controller}.json"
    j.write_text(json.dumps(dict(test=args.test, controller=args.controller,
                                 results=results), indent=2))
    print(f"\nwrote {j}")
    npass = sum(1 for r in results if r.get("success"))
    print(f"summary: {npass}/{len(results)} configs PASS")


if __name__ == "__main__":
    main()
