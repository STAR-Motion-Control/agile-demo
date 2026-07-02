#!/usr/bin/env python3
"""Humanoid-GPT tracking-policy squat probe for the local G1 benchmark.

This is intentionally separate from bench_hgpt.py.  bench_hgpt.py evaluates
the released G1-Walk velocity policy.  This probe evaluates the released
Humanoid-GPT whole-body tracking policy on synthetic down-hold-up squat
references, which is the closest released path for squat-like behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imageio
import mujoco
import numpy as np
from jax import tree_util as jtu


FREQ = 50
CTRL_DT = 1.0 / FREQ
VIDEO_FPS = 25
VIDEO_EVERY = 2
VIDEO_W, VIDEO_H = 640, 480
FALL_TILT_RAD = 0.9
FALL_BASE_Z = 0.35


@dataclass(frozen=True)
class SquatVariant:
    name: str
    target_z: float
    hip_pitch: float
    knee: float
    ankle_pitch: float
    notes: str


VARIANTS = (
    SquatVariant(
        "conservative_060",
        0.60,
        -1.00,
        1.57,
        0.40,
        "feet stay within about 8cm of nominal FK footprint",
    ),
    SquatVariant(
        "low_052",
        0.52,
        -0.18,
        1.57,
        0.40,
        "near the kinematic lower bound before large foot/root displacement",
    ),
    SquatVariant(
        "deep_045",
        0.45,
        -1.25,
        0.55,
        -0.05,
        "benchmark-depth probe; requires a far-forward-foot synthetic pose",
    ),
)


def install_hgpt_imports(repo: Path):
    if not (repo / "tracking" / "infer_utils.py").exists():
        raise SystemExit(f"Humanoid-GPT repo not found or wrong layout: {repo}")
    sys.path.insert(0, str(repo))
    from tracking import constants as consts  # noqa: PLC0415
    from tracking.convert_qpos2kpt import qpos2kpt  # noqa: PLC0415
    from tracking.infer_utils import (  # noqa: PLC0415
        G1TrackInferFn,
        G1TrackMjSim,
        apply_ema_qpos,
        g1_infer_env_config,
    )
    from tracking.policy import Args as PolicyArgs, get_policy_onnx  # noqa: PLC0415

    return consts, qpos2kpt, G1TrackInferFn, G1TrackMjSim, apply_ema_qpos, g1_infer_env_config, PolicyArgs, get_policy_onnx


def smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def quat_tilt(q: np.ndarray) -> float:
    w, x, y, z = q
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, float(up_z))))


def make_qpos_reference(default_qpos: np.ndarray, variant: SquatVariant) -> tuple[np.ndarray, dict]:
    start = default_qpos.astype(np.float32).copy()
    goal = start.copy()
    goal[2] = variant.target_z
    for off in (0, 6):
        goal[7 + off + 0] = variant.hip_pitch
        goal[7 + off + 3] = variant.knee
        goal[7 + off + 4] = variant.ankle_pitch

    n_down = int(1.5 * FREQ)
    n_hold = int(1.0 * FREQ)
    n_up = int(1.5 * FREQ)
    a_down = smoothstep(np.linspace(0.0, 1.0, n_down, endpoint=False, dtype=np.float32))
    a_up = smoothstep(np.linspace(0.0, 1.0, n_up, endpoint=True, dtype=np.float32))
    frames = []
    for a in a_down:
        frames.append(start * (1.0 - a) + goal * a)
    frames.extend([goal.copy() for _ in range(n_hold)])
    for a in a_up:
        frames.append(goal * (1.0 - a) + start * a)
    qpos = np.asarray(frames, dtype=np.float32)
    meta = {
        "n_down": n_down,
        "n_hold": n_hold,
        "n_up": n_up,
        "hold_start": n_down,
        "hold_end": n_down + n_hold,
    }
    return qpos, meta


class Recorder:
    def __init__(self, model, enabled: bool):
        self.enabled = enabled
        self.frames = []
        self.renderer = None
        self.camera = None
        if enabled:
            self.renderer = mujoco.Renderer(model, width=VIDEO_W, height=VIDEO_H)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 0
            self.camera.azimuth = 90
            self.camera.elevation = -20
            self.camera.distance = 2.2

    def frame(self, data, i: int):
        if not self.enabled or i % VIDEO_EVERY:
            return
        self.renderer.update_scene(data, self.camera)
        self.frames.append(self.renderer.render())

    def save(self, path: Path):
        if not self.enabled or not self.frames:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=VIDEO_FPS)
        return str(path)


def row_for(test, trial, variant, success, metrics, video_path, error=None):
    return {
        "framework": "humanoid-gpt",
        "test": test,
        "trial": trial,
        "seed": trial,
        "success": bool(success),
        "fall_time": metrics.get("fall_time"),
        "fall_phase": metrics.get("fall_phase"),
        "harness_error": error,
        "metrics": metrics,
        "video": video_path,
        "notes": (
            "Humanoid-GPT pns_wo_priv216 tracking policy on synthetic squat qpos; "
            "not a velocity/height command interface."
        ),
        "variant": variant.name,
    }


def run_variant(args, imports, variant: SquatVariant, trial: int):
    (
        consts,
        qpos2kpt,
        G1TrackInferFn,
        G1TrackMjSim,
        apply_ema_qpos,
        g1_infer_env_config,
        PolicyArgs,
        get_policy_onnx,
    ) = imports
    env_cfg = g1_infer_env_config(ctrl_dt=CTRL_DT)
    convert_model = mujoco.MjModel.from_xml_path(str(consts.TRACK_XML))
    qpos_ref, meta = make_qpos_reference(consts.DEFAULT_QPOS, variant)
    qpos_ref = apply_ema_qpos(qpos_ref)
    ref_traj = qpos2kpt(
        convert_model,
        qpos_src=qpos_ref,
        freq_src=FREQ,
        freq_tgt=FREQ,
        interp_sec=0.0,
        end_default_sec=0.0,
        debug=False,
        foot_contact_est=False,
        height_clip_mode=None,
        video_path=None,
    )

    init_qpos = qpos_ref[0].copy()
    init_qpos[:2] = 0.0
    mj_sim = G1TrackMjSim(init_qpos=init_qpos, headless=True, ctrl_dt=CTRL_DT)
    policy_args = PolicyArgs(load_path=args.onnx_track, device=args.device)
    policy = get_policy_onnx(policy_args)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, policy, privileged=args.privileged)
    state = mj_sim.reset(mj_sim.init_state())
    rec = Recorder(mj_sim.mj_model, args.video != "none")

    actual_qpos = []
    root_pos_err = []
    joint_err = []
    tilts = []
    fall_time = None
    fall_phase = None

    for i in range(len(qpos_ref)):
        ref_curr = jtu.tree_map(lambda x: x[i][None], ref_traj)
        j = min(i + 1, len(qpos_ref) - 1)
        ref_next = jtu.tree_map(lambda x: x[j][None], ref_traj)
        action = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        state = mj_sim.step(state, action)
        rec.frame(state.mj_data, i)

        q = state.mj_data.qpos.copy()
        actual_qpos.append(q)
        root_pos_err.append(float(np.linalg.norm(q[:3] - qpos_ref[i, :3])))
        joint_err.append(float(np.mean(np.abs(q[7:] - qpos_ref[i, 7:]))))
        tilt = quat_tilt(q[3:7])
        tilts.append(tilt)
        if fall_time is None and (tilt > FALL_TILT_RAD or q[2] < FALL_BASE_Z):
            fall_time = i * CTRL_DT
            if i < meta["hold_start"]:
                fall_phase = "down"
            elif i < meta["hold_end"]:
                fall_phase = "hold"
            else:
                fall_phase = "up"
            break

    aq = np.asarray(actual_qpos)
    hold = aq[meta["hold_start"] : min(meta["hold_end"], len(aq))]
    start_xy = aq[0, :2] if len(aq) else np.zeros(2)
    root_xy_drift = (
        float(np.max(np.linalg.norm(aq[:, :2] - start_xy, axis=1))) if len(aq) else None
    )
    video_path = None
    tag = "ok"
    metrics = {
        "variant": variant.name,
        "variant_notes": variant.notes,
        "target_min_base_z": variant.target_z,
        "completion": float(len(aq) / len(qpos_ref)) if len(qpos_ref) else 0.0,
        "min_base_z": float(np.min(aq[:, 2])) if len(aq) else None,
        "hold_base_z_mean": float(np.mean(hold[:, 2])) if len(hold) else None,
        "root_z_rmse": float(np.sqrt(np.mean((aq[:, 2] - qpos_ref[: len(aq), 2]) ** 2))) if len(aq) else None,
        "root_pos_err_mean": float(np.mean(root_pos_err)) if root_pos_err else None,
        "joint_pos_mae": float(np.mean(joint_err)) if joint_err else None,
        "root_xy_drift_max": root_xy_drift,
        "max_tilt": float(np.max(tilts)) if tilts else None,
        "fall_time": fall_time,
        "fall_phase": fall_phase,
    }
    success = (
        fall_time is None
        and metrics["completion"] >= 0.999
        and metrics["min_base_z"] is not None
        and metrics["min_base_z"] <= variant.target_z + 0.08
        and metrics["root_xy_drift_max"] is not None
        and metrics["root_xy_drift_max"] <= 0.30
        and metrics["root_pos_err_mean"] is not None
        and metrics["root_pos_err_mean"] <= 0.25
    )
    tag = "ok" if success else "fail"
    if args.video != "none":
        video_path = rec.save(
            Path(args.out_dir)
            / "videos"
            / f"{args.label}_{args.test}_t{trial:02d}_{tag}.mp4"
        )
    return row_for(args.test, trial, variant, success, metrics, video_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="squat_track")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="humanoid-gpt")
    ap.add_argument("--hgpt-repo", default="/sda/lizhe/repos/Humanoid-GPT")
    ap.add_argument("--onnx-track", default="/sda/lizhe/repos/Humanoid-GPT/storage/ckpts/pns_wo_priv216.onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--video", choices=("none", "policy", "all"), default="none")
    args = ap.parse_args()

    if args.test != "squat_track":
        raise SystemExit("bench_hgpt_track.py currently supports only --test squat_track")

    imports = install_hgpt_imports(Path(args.hgpt_repo).resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.label}_{args.test}.jsonl"
    rows = []
    with open(jsonl_path, "w") as fp:
        for trial, variant in enumerate(VARIANTS):
            try:
                row = run_variant(args, imports, variant, trial)
            except Exception as exc:
                row = row_for(args.test, trial, variant, False, {}, None, error=repr(exc))
            fp.write(json.dumps(row) + "\n")
            fp.flush()
            rows.append(row)

    summary = {
        "framework": "humanoid-gpt",
        "test": args.test,
        "n_trials": len(rows),
        "n_success": sum(bool(r.get("success")) for r in rows),
        "variants": [r.get("metrics", {}).get("variant") for r in rows],
    }
    with open(out_dir / f"{args.label}_{args.test}_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[ok] wrote {jsonl_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
