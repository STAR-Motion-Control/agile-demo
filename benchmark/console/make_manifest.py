#!/usr/bin/env python3
"""Generate manifest.json for the G1 benchmark experiment console.

Scans (relative to cc/experiments/):
  results/<model>/*.jsonl              -> R1 (spec v1) / R2 (squat_box v2, FALCON joined R2)
  results/final/<model>/*.jsonl        -> R3 (squat_sweep / goto_ab / pipeline_abc / *_psi0)
  results/videos_r2plus/<model>/*_cap.mp4 -> video library (R2+ captioned videos)

Output: console/manifest.json (video paths relative to console/: ../results/...)
Run:    python3 make_manifest.py
        python3 make_manifest.py --site   # site mode: rel_path videos/<model>/<file>,
                                          # protocol_doc PROTOCOL.md (for server.py root)
"""
import argparse
import datetime
import glob
import json
import os
import re

CONSOLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(CONSOLE_DIR)
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FINAL_DIR = os.path.join(RESULTS_DIR, "final")
VIDEOS_DIR = os.path.join(RESULTS_DIR, "videos_r2plus")
OUT_PATH = os.path.join(CONSOLE_DIR, "manifest.json")

# ManipArena (R6) data: a single pre-aggregated JSON produced by
# arena_aggregate.py across the 3 machines (homie/agile on 4090, falcon/
# ljk-falcon on 4090-lab, amo on 5080) and merged here. Captioned videos live
# under site/videos/arena/<model>/.  Path overridable via ARENA_AGG env.
ARENA_AGG_PATH = os.environ.get(
    "ARENA_AGG", os.path.join(CONSOLE_DIR, "arena_agg.json"))
ARENA_TESTS = ("arena_M1", "arena_M2")
ARENA_H_ORDER = ("0.30", "0.45", "0.60", "0.75")
ARENA_VIDEO_RE = re.compile(
    r"^(?P<model>.+)_(?P<test>arena_M\d+)_v(?P<variant>\d+)_t\d+_(?P<outcome>ok|fail)_cap\.mp4$")

MODELS = ("agile", "agile-boxdemo", "agile-deepsquat", "amo", "falcon", "gr00t-wbc", "homie", "humanoid-gpt", "ljk-falcon", "ljk-falcon-v5")
VIDEO_RE = re.compile(r"^([a-z0-9-]+)_(.+)_t(\d+)_(ok|fail)_cap\.mp4$")
R3_TESTS = ("squat_sweep", "goto_ab", "pipeline_abc")

TEST_ORDER = (
    "walk_speed", "speed_sweep", "squat_box", "squat_box_v2", "circle_pillar",
    "squat_sweep", "goto_ab", "pipeline_abc",
    "vln_follow", "squat_track",
    "walk_speed_psi0", "squat_box_psi0", "circle_pillar_psi0",
)

TEST_LABELS = {
    "walk_speed": "T1 走路 @1.0 m/s",
    "speed_sweep": "T1b 速度扫描",
    "squat_box": "T2 抱箱蹲起 ×20（v1 箱 weld@torso）",
    "squat_box_v2": "T2 捧箱蹲起 ×20（v2 双腕 weld）",
    "circle_pillar": "T3 绕柱 r=1m",
    "squat_sweep": "T4 蹲深/蹲速扫描（捧箱）",
    "goto_ab": "T5 A→B 到点 + 小指令校准",
    "pipeline_abc": "T6 全流程搬运放箱 A→C",
    "vln_follow": "T10 VLN 小指令流跟随",
    "squat_track": "T-HGPT 下蹲 motion tracking",
    "walk_speed_psi0": "T1 +Psi0 上身回放",
    "squat_box_psi0": "T2 +Psi0 桌面拾箱",
    "circle_pillar_psi0": "T3 +Psi0 上身回放",
}

TEST_SETTINGS = {
    "walk_speed": "vx 0→1.0 m/s（2s ramp + 10s hold）；success := 无摔且 mean_vx≥0.9",
    "speed_sweep": "vx∈{0.4,0.6,0.8,1.0,1.2} ×5 trials/档；报告各档实际速度",
    "squat_box": "蹲至 base 0.45m，1.5s降/1s持/1.5s升 ×20；箱 2kg weld@torso",
    "squat_box_v2": "蹲至 0.45m ×20；自由箱 2kg 双腕 weld（载荷经手臂）",
    "circle_pillar": "vx=0.4, wz=±0.4（r=1m）×2 圈；25 ccw + 25 cw；柱 r=0.15m",
    "squat_sweep": "深扫 H∈{0.65…0.30}×5 @0.2m/s + 速扫 0.45m × rate∈{0.1,0.2,0.4,0.8}×5；捧箱 v2",
    "goto_ab": "A→B=(3,1,+90°)；NAV(≤0.6) → CAL 纯小指令(≤0.1)；fine := 5cm/5° 持续1s",
    "pipeline_abc": "捧箱右转90°→走2.5m到 C=(0,−2.5,−90°)→校准→蹲0.45 放箱→起身",
    "vln_follow": "10 条共享 VLN-style 指令带；|vx|≤0.35 |vy|≤0.2 |wz|≤0.30，含小指令段和全停段",
    "squat_track": "Humanoid-GPT tracking policy 跟踪 synthetic qpos 下蹲（0.60/0.52/0.45m），非速度/高度 command policy",
    "walk_speed_psi0": "同 T1 + Psi0 真机轨迹回放上身扰动（real_ep053）",
    "squat_box_psi0": "桌面(0.40m)拾箱（磁性抓取阈 0.30m）+ 蹲循环；Psi0 sim_ep035 回放",
    "circle_pillar_psi0": "同 T3 + Psi0 上身回放扰动（real_ep053）；捧箱",
}

GOALS = [
    "为「精确导航 + 蹲起搬箱」场景在 MuJoCo sim2sim 横评 4 个 G1 下半身控制器：AGILE / HOMIE / AMO / FALCON(自训)，每个模型用其作者推荐的 sim 验证方式。",
    "三项能力硬指标：1 m/s 行走、抱/捧 2kg 箱 20 次蹲起不摔、r=1m 绕柱精准平移旋转（径向误差 ≤0.15m）。",
    "项目目标精度 5cm/5°：用 ≤0.1 m/s 纯小指令校准直接测「小指令能不能动」（SONIC 痛点验证）。",
    "Psi0(VLA) 上身在环：同一上身扰动轨迹回放下对比四下身鲁棒性；全流程 A→B→C 搬运放箱端到端验证。",
    "产出可复现 harness（4 个 bench_*.py）+ 每 trial JSONL + 视频证据 + 本控制台，可随新一轮实验持续刷新。",
]

PROTOCOL_SUMMARY = [
    "v1（R1, 2026-06-10）：三测试 walk_speed / squat_box(箱 weld@torso) / circle_pillar，每组合 50 trials，seed=trial，统一摔倒判定（倾角>0.9rad / base_z 低于目标−0.2m / 非脚触地）。",
    "v2（R2, 2026-06-11）：捧箱修正为自由箱+双腕 weld（载荷经手臂传导），T2 全部重测；FALCON(自训 model_10000) 加入；新增 box_kept / box_drop_time 指标。",
    "v3（R3, 2026-06-11~12）：新增 squat_sweep 蹲深/蹲速扫描、goto_ab 小指令校准、pipeline_abc 全流程放箱；Psi0 上身轨迹回放三测试（Plan B，四下身吃同一扰动流）。",
    "v4（规划中）：squat_limit 蹲深极限（0.05 m/s 缓降至摔/饱和）+ squat_place_psi0 前伸放箱（ep053）。",
    "新增实验标准步骤：写 spec → 4 个 harness 实现 + py_compile → 冒烟 1-2 trial → 全量 50 → caption_videos.py 烧字幕 → make_manifest.py 刷新控制台 → experiment.md 记录。",
]

ROUNDS = {
    "R1": "第一轮 · 统一规范 v1 三测试（2026-06-10，375 trials 零摔倒）",
    "R2": "第二轮 · 捧箱 v2 + FALCON 加入（2026-06-11）",
    "R3": "第三轮 · 蹲扫 / 到点校准 / 全流程 + Psi0 上身回放（2026-06-11~12）",
    "R6-Arena": "统一场景 ManipArena · M1 桌面拾放(4 桌高) / M2 中继搬运链 · "
                "5 模型 × 50 变种（含 ljk-falcon）",
}

# ---------------------------------------------------------------- harnesses / commands
# Each model uses its own bench_<model>.py harness on its own server (作者推荐的
# sim 验证方式)，共享同一份测试规范（spec v1/v3 + ManipArena）。下表给出机器 / 解释器
# / 工作目录 / 脚本 / 权重，命令模板用 $PY bench_<m>.py 占位（$PY = 对应解释器）。
HARNESSES = [
    {"model": "AGILE", "server": "4090（ssh 4090，外网）",
     "env": "miniforge3/envs/agile/bin/python", "workdir": "/sda/lizhe/g1bench",
     "script": "bench_agile.py",
     "weights": "仓库自带 velocity_height student（.pt/.yaml）；--checkpoint/--config 可替换"},
    {"model": "AGILE-DEEPSQUAT", "server": "4090（ssh 4090）",
     "env": "miniforge3/envs/agile/bin/python", "workdir": "/sda/lizhe/g1bench",
     "script": "bench_agile.py",
     "weights": "深蹲微调 student model_2999：--checkpoint /sdb/lizhe/g1_deepsquat/"
                "student2999_export/policy.pt --config <原 student.yaml>"},
    {"model": "AGILE-BOXDEMO", "server": "4090（ssh 4090）",
     "env": "miniforge3/envs/agile/bin/python", "workdir": "/sda/lizhe/g1bench",
     "script": "bench_agile.py --upper-mode box_demo --label agile-boxdemo",
     "weights": "AGILE velocity_height student + box_demo_2 ArmKinematics/hand-off upper body"},
    {"model": "HOMIE", "server": "4090（ssh 4090）",
     "env": "venv_homie（py3.10，纯 CPU）", "workdir": "/sda/g1_bench",
     "script": "bench_homie.py", "weights": "OpenHomie/HomieDeploy/deploy.onnx（456→12）"},
    {"model": "Humanoid-GPT", "server": "4090（ssh 4090）",
     "env": "miniforge3/envs/homie/bin/python（复用 numpy+mujoco+onnxruntime）",
     "workdir": "/sda/lizhe/repos/Humanoid-GPT + /sda/lizhe/g1bench",
     "script": "bench_hgpt.py --label humanoid-gpt --hgpt-repo /sda/lizhe/repos/Humanoid-GPT",
     "weights": "released G1-Walk ONNX；输入 [vx,vy,wz]，无 height/squat command，cmd norm <=0.2 不启动 gait"},
    {"model": "AMO", "server": "5080（ssh wjzh@10.24.88.193）",
     "env": "conda amo（JIT 硬编码 cuda:0，GPU 必需）", "workdir": "~/AMO",
     "script": "bench_amo.py", "weights": "amo_jit.pt + adapter_jit.pt（G1 23-DoF）"},
    {"model": "FALCON", "server": "4090-lab（ssh 4090-lab，校内 MSI）",
     "env": "conda fcreal（/hhd2/ljk/miniconda3/envs/fcreal）",
     "workdir": "/hhd2/ljk/FALCON/sim2real", "script": "bench_falcon.py --label falcon",
     "weights": "项目自训 model_10000.onnx（575→29，整身+fix_upper_body）"},
    {"model": "ljk-falcon", "server": "4090-lab（ssh 4090-lab）",
     "env": "conda fcreal", "workdir": "/hhd2/ljk/FALCON/sim2real",
     "script": "bench_falcon.py --label ljk-falcon --model-path "
               ".../models/falcon/g1_29dof_v4.onnx",
     "weights": "FALCON v4 权重 g1_29dof_v4.onnx（仅权重，整套部署管线复用）"},
    {"model": "ljk-falcon-v5", "server": "4090-lab（ssh 4090-lab）",
     "env": "conda fcreal", "workdir": "/hhd2/ljk/FALCON/sim2real",
     "script": "bench_falcon.py --label ljk-falcon-v5 --model-path "
               ".../models/falcon/g1_29dof_v5_1.onnx",
     "weights": "FALCON v5 权重 g1_29dof_v5_1.onnx（仅权重，整套部署管线复用）"},
    {"model": "gr00t-wbc", "server": "4090（ssh 4090）",
     "env": "miniforge3/envs/homie/bin/python（mujoco+onnxruntime，纯 CPU）",
     "workdir": "/sda/lizhe/g1bench", "script": "bench_dwbc.py --label gr00t-wbc",
     "weights": "NVIDIA GR00T-WholeBodyControl decoupled-WBC 双策略 "
                "GR00T-WholeBodyControl-{Balance,Walk}.onnx（516→15，scene_29dof）"},
]

_PRE = "MUJOCO_GL=egl $PY bench_<m>.py"
_OUT = "--out-dir <DIR> --video policy"
COMMAND_GROUPS = [
    {"round": "R2", "title": "捧箱 v2 + 行走 / 绕柱 / 速度扫描", "items": [
        {"test": "walk_speed", "label": "T1 走路 @1.0 m/s",
         "cmd": "%s --test walk_speed --trials 50 %s" % (_PRE, _OUT)},
        {"test": "squat_box", "label": "T2 捧箱蹲起 ×20（默认 v2 双腕 weld，2kg）",
         "cmd": "%s --test squat_box --trials 50 %s" % (_PRE, _OUT)},
        {"test": "circle_pillar", "label": "T3 绕柱 r=1m（25 ccw + 25 cw）",
         "cmd": "%s --test circle_pillar --trials 50 %s" % (_PRE, _OUT)},
        {"test": "speed_sweep", "label": "T1b 速度扫描（每档 5 trials，5 档→25）",
         "cmd": "%s --test speed_sweep --trials 5 %s" % (_PRE, _OUT)},
    ]},
    {"round": "R3", "title": "蹲扫 / 到点校准 / 全流程 + Psi0 上身回放", "items": [
        {"test": "walk_speed_psi0", "label": "T1 +Psi0 上身回放（real_ep053）",
         "cmd": "%s --test walk_speed_psi0 --trials 50 --upper-replay <real_ep053.npz> %s" % (_PRE, _OUT)},
        {"test": "squat_box_psi0", "label": "T2 +Psi0 桌面拾箱（sim_ep035）",
         "cmd": "%s --test squat_box_psi0 --trials 50 --upper-replay <sim_ep035.npz> %s" % (_PRE, _OUT)},
        {"test": "circle_pillar_psi0", "label": "T3 +Psi0 上身回放（real_ep053）",
         "cmd": "%s --test circle_pillar_psi0 --trials 50 --upper-replay <real_ep053.npz> %s" % (_PRE, _OUT)},
        {"test": "squat_sweep", "label": "T4 蹲深/蹲速扫描（捧箱；每格 5，8深+4速=12格→60）",
         "cmd": "%s --test squat_sweep --trials 5 %s" % (_PRE, _OUT)},
        {"test": "goto_ab", "label": "T5 A→B 到点 + 小指令校准（5cm/5°）",
         "cmd": "%s --test goto_ab --trials 25 %s" % (_PRE, _OUT)},
        {"test": "pipeline_abc", "label": "T6 全流程搬运放箱 A→C",
         "cmd": "%s --test pipeline_abc --trials 15 %s" % (_PRE, _OUT)},
    ]},
    {"round": "R6-Arena", "title": "统一 ManipArena · 50 变种（每变种单独一次）", "items": [
        {"test": "arena_M1", "label": "M1 桌面拾放（4 桌高分层）",
         "cmd": "for v in $(seq 0 49); do $PY bench_<m>.py --test arena_M1 "
                "--variant $v --trials 1 --upper-replay <real_ep053.npz> "
                "--out-dir <DIR>/arena_M1/v$v --video policy; done"},
        {"test": "arena_M2", "label": "M2 中继搬运链 grasp→relay→cube→regrasp→store",
         "cmd": "for v in $(seq 0 49); do $PY bench_<m>.py --test arena_M2 "
                "--variant $v --trials 1 --upper-replay <real_ep053.npz> "
                "--out-dir <DIR>/arena_M2/v$v --video policy; done"},
    ]},
]
COMMAND_NOTES = [
    "$PY 与 bench_<m>.py 按上表替换：AGILE/HOMIE 在 4090、AMO 在 5080、FALCON/ljk-falcon 在 4090-lab。",
    "上身回放 npz 按本体命名：AGILE=upper_replay_agile29_* / FALCON=falcon29_* / AMO=amo23_*"
    "（squat_box_psi0→sim_ep035，walk/circle/arena→real_ep053）。",
    "FALCON/ljk-falcon：walk/circle/nav 用 stand=1（stepping）、squat 用 stand=0，harness 内部自动切换。",
    "AGILE-DEEPSQUAT（本轮新模型）= AGILE 命令加 --checkpoint .../student2999_export/policy.pt "
    "--config <原 student.yaml>（obs/动作契约不变，yaml 复用）。",
    "AGILE-BOXDEMO = AGILE 命令加 --upper-mode box_demo --label agile-boxdemo；R6 arena_M1/M2 不需要 "
    "--upper-replay，R2/R3 carry/psi0 兼容路径会写 upper_mode=box_demo_2_ik。",
    "默认 trial 数即规范数（walk/squat/circle/psi0=50、sweep=5/格点、goto=25、pipeline=15、arena=1/变种）。",
]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def gm(metrics, *keys):
    """First non-None metric value among fallback keys."""
    for k in keys:
        v = metrics.get(k)
        if v is not None:
            return v
    return None


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def f(v, p=3):
    return ("%." + str(p) + "f") % v if isinstance(v, (int, float)) else "—"


def is_fall(row):
    return bool(row.get("fall_time") is not None or row.get("fall") is True)


def canonical_test(jsonl_path, model, rows):
    base = os.path.basename(jsonl_path)[:-len(".jsonl")]
    if base.startswith(model + "_"):
        base = base[len(model) + 1:]
    has_v2_box = any(r.get("box_mode") == "wrist_weld_v2" for r in rows[:3])
    if base == "squat_box" and has_v2_box:
        return "squat_box_v2"
    return base


def legacy_round(model, test):
    if test == "squat_box_v2" or model in ("falcon", "ljk-falcon", "ljk-falcon-v5", "agile-deepsquat", "agile-boxdemo", "gr00t-wbc"):
        return "R2"
    return "R1"


# ---------------------------------------------------------------- summaries

def family(test):
    if test.startswith("walk_speed"):
        return "walk"
    if test.startswith("speed_sweep"):
        return "sweep"
    if test.startswith("squat_box"):
        return "squat"
    if test.startswith("circle"):
        return "circle"
    return test  # squat_sweep / goto_ab / pipeline_abc


def km_walk(ms):
    cmd = gm(ms[0], "vx_target", "cmd_vx", "target_vx") if ms else None
    return "实际 vx=%s m/s（指令 %s）· RMSE %s" % (
        f(mean([gm(m, "mean_vx_last8s") for m in ms])),
        f(cmd, 1),
        f(mean([gm(m, "vx_rmse", "vx_rmse_test_seg") for m in ms])))


def km_sweep(ms):
    by_cmd = {}
    for m in ms:
        cmd = gm(m, "vx_target", "cmd_vx", "target_vx")
        if cmd is not None:
            by_cmd.setdefault(round(cmd, 1), []).append(gm(m, "mean_vx_last8s"))
    return " · ".join("%s→%s" % (f(c, 1), f(mean(v), 2))
                      for c, v in sorted(by_cmd.items()))


def km_squat(rows, ms, n):
    bits = ["循环 %s/20" % f(mean([gm(m, "cycles_completed") for m in ms]), 1),
            "h_RMSE %sm" % f(mean([gm(m, "height_rmse", "height_rmse_m") for m in ms]))]
    kept = [gm(m, "box_kept") for m in ms if gm(m, "box_kept") is not None]
    if kept:
        bits.append("box_kept %d/%d" % (sum(bool(k) for k in kept), len(kept)))
    picks = [gm(m, "pick_success") for m in ms if gm(m, "pick_success") is not None]
    if picks:
        bits.append("拾箱 %d/%d" % (sum(bool(p) for p in picks), len(picks)))
    return " · ".join(bits)


def km_circle(rows, ms, n):
    cols = sum(bool(gm(m, "pillar_collision", "collision")) for m in ms)
    per_dir = {}
    for r, m in zip(rows, ms):
        d = gm(m, "direction")
        if d:
            ok, tot = per_dir.get(d, (0, 0))
            per_dir[d] = (ok + bool(r.get("success")), tot + 1)
    dir_bits = ["%s %d/%d" % (d, ok, tot) for d, (ok, tot) in sorted(per_dir.items())]
    return " · ".join(["径向 %sm (max %sm)" % (
        f(mean([gm(m, "radial_err_mean", "radial_err_mean_m") for m in ms])),
        f(mean([gm(m, "radial_err_max", "radial_err_max_m") for m in ms]))),
        "碰撞 %d/%d" % (cols, n)] + dir_bits)


def km_squat_sweep(rows, ms):
    depth = [(r, m) for r, m in zip(rows, ms) if gm(m, "sweep_kind", "sweep") == "depth"]
    rate = [(r, m) for r, m in zip(rows, ms) if gm(m, "sweep_kind", "sweep") != "depth"]
    deepest = [gm(m, "achieved_depth") for _, m in depth
               if gm(m, "target_height") == 0.30]
    drifts = [gm(m, "root_drift_hold") for m in ms]
    drifts = [d for d in drifts if d is not None]
    return " · ".join([
        "深扫摔 %d/%d" % (sum(is_fall(r) for r, _ in depth), len(depth)),
        "速扫摔 %d/%d" % (sum(is_fall(r) for r, _ in rate), len(rate)),
        "0.30m 档实深 %sm" % f(mean(deepest), 2),
        "drift_hold max %sm" % f(max(drifts) if drifts else None)])


def km_goto(rows, ms, n):
    fine = sum(bool(gm(m, "success_fine")) for m in ms)
    pos = mean([gm(m, "err_pos_cal", "cal_err_pos_m") for m in ms])
    yaw = mean([gm(m, "err_yaw_cal", "cal_err_yaw_rad") for m in ms])
    tc = mean([gm(m, "t_cal", "t_cal_s") for m in ms])
    return "细(5cm/5°) %d/%d · 校准后 %scm/%s° · 校准 %ss" % (
        fine, n, f(pos * 100 if pos else None, 1),
        f(yaw * 57.296 if yaw else None, 1), f(tc, 1))


def km_pipeline(rows, ms, n):
    place = sum(bool(gm(m, "box_place_ok")) for m in ms)
    land = mean([gm(m, "box_land_dist_from_c", "box_land_err", "box_land_dist")
                 for m in ms])
    fine = sum(bool(gm(m, "success_fine")) for m in ms)
    return "放箱OK %d/%d · 落点距C %sm · 细收敛 %d/%d" % (place, n, f(land, 2), fine, n)


def km_vln(rows, ms, n):
    pos = mean([gm(m, "final_pos_err") for m in ms])
    yaw = mean([gm(m, "final_yaw_err") for m in ms])
    small = mean([gm(m, "small_cmd_response") for m in ms])
    stop = mean([gm(m, "stop_settle") for m in ms])
    return "末误差 %sm/%s° · 小指令响应 %s · 停止残速 %sm/s" % (
        f(pos, 2), f(yaw * 57.296 if yaw else None, 1), f(small, 2), f(stop, 3))


def km_squat_track(rows, ms, n):
    succ = sum(bool(r.get("success")) for r in rows)
    z = mean([gm(m, "min_base_z") for m in ms])
    rz = mean([gm(m, "root_z_rmse") for m in ms])
    drift = mean([gm(m, "root_xy_drift_max") for m in ms])
    return "track %d/%d · min_z %sm · z_RMSE %sm · xy漂移 %sm" % (
        succ, n, f(z, 2), f(rz, 3), f(drift, 2))


def key_metrics(test, rows):
    ms = [r.get("metrics", {}) for r in rows]
    n = len(rows)
    fam = family(test)
    if fam == "walk":
        return km_walk(ms)
    if fam == "sweep":
        return km_sweep(ms)
    if fam == "squat":
        return km_squat(rows, ms, n)
    if fam == "circle":
        return km_circle(rows, ms, n)
    if fam == "squat_sweep":
        return km_squat_sweep(rows, ms)
    if fam == "goto_ab":
        return km_goto(rows, ms, n)
    if fam == "pipeline_abc":
        return km_pipeline(rows, ms, n)
    if fam == "vln_follow":
        return km_vln(rows, ms, n)
    if fam == "squat_track":
        return km_squat_track(rows, ms, n)
    return ""


def settings_summary(test, rows):
    base = TEST_SETTINGS.get(test, "")
    upper = next((r.get("upper_mode") for r in rows if r.get("upper_mode")), None)
    if upper:
        return "%s · 上身=%s" % (base, upper)
    return base


def summarize(round_id, model, test, rows):
    n = len(rows)
    n_success = sum(bool(r.get("success")) for r in rows)
    return {
        "round": round_id,
        "model": model.upper(),
        "test": test,
        "label": TEST_LABELS.get(test, test),
        "n_trials": n,
        "n_success": n_success,
        "success_rate": round(n_success / n, 4) if n else 0,
        "n_falls": sum(is_fall(r) for r in rows),
        "key_metrics": key_metrics(test, rows),
        "settings": settings_summary(test, rows),
    }


# ---------------------------------------------------------------- videos

def video_round(test):
    if test in R3_TESTS or test.endswith("_psi0") or test in ("vln_follow", "squat_track"):
        return "R3"
    return "R2"


def fall_note(row):
    if row.get("fall_time") is None:
        return None
    return "摔倒@%s t=%ss" % (row.get("fall_phase") or "?", f(row["fall_time"], 1))


def subtitle_for(test, row):
    if row is None:
        return ""
    m = row.get("metrics", {})
    fam = family(test)
    bits = []
    if fam == "squat_sweep":
        bits = ["H=%sm @%sm/s" % (f(gm(m, "target_height"), 2), f(gm(m, "ramp_speed"), 1)),
                "实深 %sm" % f(gm(m, "achieved_depth"), 2),
                "蹲底漂移 %sm" % f(gm(m, "root_drift_hold"))]
    elif fam == "squat":
        bits = ["蹲 0.45m ×20", "h_rmse=%sm" % f(gm(m, "height_rmse", "height_rmse_m"))]
        if gm(m, "pick_success") is not None:
            bits.append("拾箱=%s" % ("Y" if gm(m, "pick_success") else "N"))
    elif fam == "circle":
        bits = ["%s vx=0.4 wz=±0.4" % (gm(m, "direction") or "?"),
                "径向 %sm" % f(gm(m, "radial_err_mean", "radial_err_mean_m"))]
        if gm(m, "pillar_collision", "collision"):
            bits.append("撞柱")
    elif fam == "walk" or fam == "sweep":
        bits = ["指令 %s → 实际 %s m/s" % (
            f(gm(m, "vx_target", "cmd_vx", "target_vx"), 1),
            f(gm(m, "mean_vx_last8s"), 2))]
    elif fam == "goto_ab":
        pos = gm(m, "err_pos_cal", "cal_err_pos_m")
        yaw = gm(m, "err_yaw_cal", "cal_err_yaw_rad")
        bits = ["校准后 %scm/%s°" % (f(pos * 100 if pos else None, 1),
                                     f(yaw * 57.296 if yaw else None, 1)),
                "fine=%s" % ("Y" if gm(m, "success_fine") else "N")]
    elif fam == "vln_follow":
        yaw = gm(m, "final_yaw_err")
        bits = ["final=%sm/%s°" % (f(gm(m, "final_pos_err"), 2),
                                   f(yaw * 57.296 if yaw else None, 1)),
                "small_resp=%s" % f(gm(m, "small_cmd_response"), 2),
                "stop=%sm/s" % f(gm(m, "stop_settle"), 3)]
    elif fam == "squat_track":
        bits = ["%s target=%sm" % (gm(m, "variant") or "variant",
                                   f(gm(m, "target_min_base_z"), 2)),
                "min_z=%sm" % f(gm(m, "min_base_z"), 2),
                "z_rmse=%sm" % f(gm(m, "root_z_rmse"), 3),
                "xy=%sm" % f(gm(m, "root_xy_drift_max"), 2)]
    elif fam == "pipeline_abc":
        bits = ["cal=%sm" % f(gm(m, "err_pos_cal", "cal_err_pos_m"), 2),
                "放箱=%s" % ("OK" if gm(m, "box_place_ok") else "✗"),
                "落点 %sm" % f(gm(m, "box_land_dist_from_c", "box_land_err",
                                  "box_land_dist"), 2)]
    upper = row.get("upper_mode")
    if upper:
        bits.append(upper.replace("psi0_replay_", ""))
    fn = fall_note(row)
    if fn:
        bits.append(fn)
    return " · ".join(b for b in bits if b)


# ---------------------------------------------------------------- main scan

def scan_experiments():
    """Returns (experiments list, rows_index {(model,test): {trial: row}})."""
    experiments = []
    rows_index = {}

    def ingest(round_id, model, test, rows):
        experiments.append(summarize(round_id, model, test, rows))
        idx = {}
        for r in rows:
            try:
                idx[int(r.get("trial"))] = r
            except (TypeError, ValueError):
                pass
        rows_index[(model, test)] = idx

    for model in MODELS:
        legacy = sorted(glob.glob(os.path.join(RESULTS_DIR, model, "*.jsonl")))
        for path in legacy:
            rows = load_jsonl(path)
            if not rows:
                continue
            test = canonical_test(path, model, rows)
            ingest(legacy_round(model, test), model, test, rows)
        final = sorted(glob.glob(os.path.join(FINAL_DIR, model, "*.jsonl")))
        for path in final:
            rows = load_jsonl(path)
            if not rows:
                continue
            test = canonical_test(path, model, rows)
            ingest("R3", model, test, rows)

    order = {t: i for i, t in enumerate(TEST_ORDER)}
    experiments.sort(key=lambda e: (e["round"], order.get(e["test"], 99), e["model"]))
    return experiments, rows_index


def scan_videos(rows_index, site_mode=False):
    videos, missing_rows = [], 0
    for model in MODELS:
        vdir = os.path.join(VIDEOS_DIR, model)
        if not os.path.isdir(vdir):
            continue
        for path in sorted(glob.glob(os.path.join(vdir, "*_cap.mp4"))):
            name = os.path.basename(path)
            mm = VIDEO_RE.match(name)
            if not mm or mm.group(1) != model:
                continue
            test, trial, outcome = mm.group(2), int(mm.group(3)), mm.group(4)
            row = (rows_index.get((model, test + "_v2"), {}).get(trial)
                   or rows_index.get((model, test), {}).get(trial))
            if row is None:
                missing_rows += 1
            rel_path = ("videos/%s/%s" % (model, name) if site_mode
                        else "../results/videos_r2plus/%s/%s" % (model, name))
            videos.append({
                "file": name,
                "rel_path": rel_path,
                "model": model.upper(),
                "test": test,
                "label": TEST_LABELS.get(test, TEST_LABELS.get(test + "_v2", test)),
                "trial": trial,
                "outcome": "success" if outcome == "ok" else "fail",
                "round": video_round(test),
                "subtitle": subtitle_for(test, row),
            })
    return videos, missing_rows


# ---------------------------------------------------------------- arena (R6)

def load_arena_agg():
    """Load merged arena aggregate JSON, or None if absent (keeps R1-R5 intact)."""
    if not os.path.exists(ARENA_AGG_PATH):
        return None
    try:
        with open(ARENA_AGG_PATH) as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None


def _exp_row(model, test, label, n, n_success, n_falls, key_metrics, settings):
    return {
        "round": "R6-Arena",
        "model": model.upper(),
        "test": test,
        "label": label,
        "n_trials": n,
        "n_success": n_success,
        "success_rate": round(n_success / n, 4) if n else 0,
        "n_falls": n_falls,
        "key_metrics": key_metrics,
        "settings": settings,
    }


def arena_experiments(agg):
    """One M1 row per H_pick (4 rows) + one M2 chain row, per model."""
    rows = []
    for model in sorted(agg.get("models", {})):
        mo = agg["models"][model]
        m1 = mo.get("arena_M1")
        if m1:
            by_h = m1.get("by_h", {})
            for hk in sorted(by_h, key=lambda x: float(x) if _isnum(x) else 99):
                b = by_h[hk]
                n = b["n"]
                # "success" column = mission done (task complete), per console note
                rows.append(_exp_row(
                    model, "arena_M1",
                    "M1 桌面拾放 · 桌高 %sm" % hk, n,
                    b["done"], b["fall"],
                    "拾取 %d/%d · 放回 %d/%d · 完成 %d/%d · 摔 %d/%d" % (
                        b["grasp"], n, b["place"], n, b["done"], n, b["fall"], n),
                    "导航到桌→蹲拾箱(H=%sm)→起身→原位放回；5cm/5° nav-cal 门控" % hk))
        m2 = mo.get("arena_M2")
        if m2:
            c = m2.get("chain", {})
            n = m2["n"]
            rows.append(_exp_row(
                model, "arena_M2",
                "M2 中继搬运链 grasp→relay→cube→regrasp→store", n,
                c.get("store", 0), m2["fall"],
                "抓取 %d/%d → 中继放 %d/%d → 立方转移 %d/%d → 重抓 %d/%d → 入库放 %d/%d · 摔 %d/%d" % (
                    c.get("grasp", 0), n, c.get("relay", 0), n,
                    c.get("cube", 0), n, c.get("regrasp", 0), n,
                    c.get("store", 0), n, m2["fall"], n),
                "多段中继搬运：桌面抓箱→中继台放→立方块转移→重新抓取→送入库位放置"))
    return rows


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def arena_subtitle(test, row):
    if not row:
        return ""
    h = row.get("H_pick")
    hs = ("%.2fm" % h) if isinstance(h, (int, float)) else "?"
    bits = []
    if test == "arena_M1":
        bits = ["桌高 %s" % hs,
                "拾=%s" % _yn(row.get("grasp_ok")),
                "放回=%s" % _yn(row.get("place_ok_store")),
                "任务=%s" % (row.get("mission_status") or "?")]
    else:  # arena_M2 -> deepest reached stage
        chain = [("store", row.get("place_ok_store")),
                 ("regrasp", row.get("regrasp_ok")),
                 ("cube_xfer", row.get("cube_transfer_ok")),
                 ("relay", row.get("place_ok_relay")),
                 ("grasp", row.get("grasp_ok"))]
        reached = next((nm for nm, v in chain if v), "none")
        bits = ["到达阶段 %s" % reached, "H=%s" % hs]
    if row.get("fall_time") is not None:
        bits.append("摔倒@%s" % (row.get("fall_phase") or "?"))
    return " · ".join(bits)


def _yn(v):
    return "Y" if v else ("N" if v is False else "—")


def arena_videos(agg, site_mode=True):
    """Build video items from agg['video_files'] (model/<file>) + agg['rows']."""
    out, missing = [], 0
    rows = agg.get("rows", {})
    for rel in agg.get("video_files", []):
        rel = rel.strip()
        if not rel:
            continue
        model_dir, _, name = rel.partition("/")
        mm = ARENA_VIDEO_RE.match(name)
        if not mm:
            continue
        model = mm.group("model")
        test = mm.group("test")
        variant = int(mm.group("variant"))
        outcome = mm.group("outcome")
        row = rows.get("%s:%s:%s" % (model, test, variant))
        if row is None:
            missing += 1
        rel_path = ("videos/arena/%s/%s" % (model_dir, name) if site_mode
                    else "../results/videos_arena/%s/%s" % (model_dir, name))
        out.append({
            "file": name,
            "rel_path": rel_path,
            "model": model.upper(),
            "test": test,
            "label": "M1 桌面拾放" if test == "arena_M1" else "M2 中继搬运链",
            "trial": variant,
            "outcome": "success" if outcome == "ok" else "fail",
            "round": "R6-Arena",
            "subtitle": arena_subtitle(test, row),
        })
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", action="store_true",
                    help="site mode: rel_path=videos/<model>/<file>, "
                         "protocol_doc=PROTOCOL.md (deploy via build_site.sh)")
    args = ap.parse_args()

    experiments, rows_index = scan_experiments()
    videos, missing_rows = scan_videos(rows_index, site_mode=args.site)

    arena_agg = load_arena_agg()
    n_arena_exp = n_arena_vid = arena_missing = 0
    if arena_agg:
        a_exp = arena_experiments(arena_agg)
        a_vid, arena_missing = arena_videos(arena_agg, site_mode=args.site)
        experiments += a_exp
        videos += a_vid
        n_arena_exp, n_arena_vid = len(a_exp), len(a_vid)
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "title": "G1 下半身控制器 Benchmark 实验控制台",
        "goals": GOALS,
        "protocol_summary": PROTOCOL_SUMMARY,
        "protocol_doc": "PROTOCOL.md" if args.site else "../PROTOCOL.md",
        "rounds": ROUNDS,
        "harnesses": HARNESSES,
        "commands": COMMAND_GROUPS,
        "command_notes": COMMAND_NOTES,
        "experiments": experiments,
        "videos": videos,
    }
    with open(OUT_PATH, "w") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)

    n_trials = sum(e["n_trials"] for e in experiments)
    n_falls = sum(e["n_falls"] for e in experiments)
    print("manifest.json written -> %s" % OUT_PATH)
    print("experiment groups: %d  (R1 %d / R2 %d / R3 %d / R6-Arena %d)" % (
        len(experiments),
        sum(e["round"] == "R1" for e in experiments),
        sum(e["round"] == "R2" for e in experiments),
        sum(e["round"] == "R3" for e in experiments),
        n_arena_exp))
    print("total trials: %d   total falls: %d" % (n_trials, n_falls))
    print("videos: %d  (success %d / fail %d)  rows-not-found: %d" % (
        len(videos),
        sum(v["outcome"] == "success" for v in videos),
        sum(v["outcome"] == "fail" for v in videos), missing_rows))
    if arena_agg is not None:
        print("arena: %d exp rows, %d videos, %d subtitle-rows-not-found "
              "(arena videos live on 4090 site/videos/arena/)" % (
                  n_arena_exp, n_arena_vid, arena_missing))
    # presence check skips arena (videos only exist on the 4090 site, not locally)
    nonarena = [v for v in videos if v["round"] != "R6-Arena"]
    if args.site:
        # site mode: check against the local videos_r2plus mirror instead
        bad = [v for v in nonarena if not os.path.exists(
            os.path.join(VIDEOS_DIR, v["model"].lower(), v["file"]))]
    else:
        bad = [v for v in nonarena
               if not os.path.exists(os.path.join(CONSOLE_DIR, v["rel_path"]))]
    print("video rel-path check (non-arena): %d broken" % len(bad))
    for v in bad[:5]:
        print("  broken:", v["rel_path"])


if __name__ == "__main__":
    main()
