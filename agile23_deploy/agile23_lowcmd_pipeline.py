#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGILE 23-DoF velocity-height student -> real Unitree G1, via the box_demo_2 / RoboJuDo wire line.

  /tmp/robojudo_ext_cmd.json  (keyboard or box_demo mover, SINGLE writer)
        |  read @ 50 Hz, stale_s=0.40
  rt/lowstate  --sub-->  build SimState  -->  AGILE ObservationProcessor
        -->  recurrent LSTM student  -->  12-DoF leg action  -->  ActionProcessor
        -->  build LowCmd (35 HG slots: drive legs 0-11, HOLD 12-28 at measured q, slot29=0)
        --pub-->  rt/lowcmd_rl
  rt/lowcmd_rl + rt/arm_sdk  -->  merge_lowcmd_arm_sdk.py (SOLE rt/lowcmd publisher)  -->  rt/lowcmd

This is the AGILE analogue of groot_wbc_boxdemo_adapter.py. It wraps AGILE's own
sim2mujoco ObservationProcessor / PolicyWrapper / ActionProcessor / CommandManager
VERBATIM (so obs/action/LSTM math is identical to sim2mujoco_eval), and only swaps the
MuJoCo state source for the robot's DDS rt/lowstate, and the MuJoCo step for an
rt/lowcmd_rl publish that drives ONLY the legs and holds the upper body for the arm_sdk
overlay.

Deploy artifact = the distilled recurrent LSTM STUDENT for the 23-DoF basic G1
(12 legs + waist_yaw + 10 arms). Pass --config the re-exported 23-DoF IODescriptor YAML
and --checkpoint the 23-DoF student .pt (the _checkpoint.pt path keeps the cleanest LSTM
state semantics). Run AGILE on PYTHONPATH (the same repo used to train/export).

⚠️ MUST be verified on the physical robot before trusting locomotion — see HARDWARE
GO/NO-GO notes in AGILE23_BOXDEMO_DEPLOY_DESIGN.md, especially:
  (1) whether rt/lowstate.motor_state[] is the full 29-slot HG layout (with the 6 absent
      joints present-but-inert) or compacted to 23 -> the entire MOTOR_BY_JOINT scheme
      assumes the 29-slot layout;
  (2) the 23-DoF descriptor joint_names order (read from the exported YAML, never assumed);
  (3) the obs byte-compare against sim2mujoco for the same static pose (HARD GO/NO-GO).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

CMD_FILE = "/tmp/robojudo_ext_cmd.json"

# ----- Unitree G1 HG 29-slot motor index map (authoritative wire order) -----------------
# slot -> joint base-name. The 23-DoF basic G1 OMITS slots 13,14 (waist roll/pitch) and
# 20,21,27,28 (each wrist pitch/yaw); on the real robot those slots are expected to be
# present-but-inert (VERIFY). The deploy never indexes by raw policy index -- it maps by
# joint NAME: motor slot for descriptor joint "name" = MOTOR_BY_JOINT[strip_joint(name)].
MOTOR_BY_JOINT: dict[str, int] = {
    "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
    "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
    "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
    "left_elbow": 18, "left_wrist_roll": 19, "left_wrist_pitch": 20, "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
    "right_elbow": 25, "right_wrist_roll": 26, "right_wrist_pitch": 27, "right_wrist_yaw": 28,
}

LEG_SLOTS = set(range(0, 12))          # AGILE lower policy drives only these
TOTAL_HG_MOTORS = 35
BODY_MOTORS = 29
ARM_SDK_ENABLE_SLOT = 29               # must stay q=0 so the merger lets arm_sdk own 12-28
WEAK_MOTOR_SLOTS = {4, 5, 10, 11}      # ankles use the weak-motor mode (0x01) on G1

# FSM states recognized on the box_demo_2 wire (same as the GR00T adapter).
#   RL_FULL  : track [vx,vy,wz,height] from IPC
#   RL_LOWER : legs balance in place (zero velocity), arms owned by arm_sdk during grasp
#   DAMP     : passive damping (kp=0, kd=damping_kd) holding measured q
#   LIMP     : PosStop/VelStop, all gains 0 (soft release)


def strip_joint(name: str) -> str:
    return name[:-6] if name.endswith("_joint") else name


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def approach(cur: float, tgt: float, max_delta: float) -> float:
    if tgt > cur:
        return min(tgt, cur + max_delta)
    return max(tgt, cur - max_delta)


def projected_gravity_z(qw: float, qx: float, qy: float, qz: float) -> float:
    """z-component of gravity ([0,0,-1] world) expressed in the body frame.
    Upright -> -1.0; tilt angle from vertical = acos(-gz). Used by the tilt guard,
    independent of AGILE's own projected_gravity obs term (defense in depth)."""
    return -(1.0 - 2.0 * (qx * qx + qy * qy))


def _load_ddsc() -> None:
    home = Path.home()
    for cand in (
        os.environ.get("CYCLONEDDS_HOME"),
        str(home / "cyclonedds-0.10-install"),
        str(home / "cyclonedds" / "install"),
    ):
        if not cand:
            continue
        so = Path(cand) / "lib" / "libddsc.so.0"
        if so.is_file():
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            return


# --------------------------------------------------------------------------- #
# IPC command (identical schema to groot_wbc_boxdemo_adapter.read_external_command)
# --------------------------------------------------------------------------- #
class ExternalCommand:
    __slots__ = ("fsm", "vx", "vy", "wz", "height", "fresh", "estop")

    def __init__(self, fsm, vx, vy, wz, height, fresh, estop=False):
        self.fsm, self.vx, self.vy, self.wz = fsm, vx, vy, wz
        self.height, self.fresh, self.estop = height, fresh, estop


def read_external_command(cmd_file: Path, last_height: float, stale_s: float,
                          fwd_max: float, lat_max: float, yaw_max: float,
                          min_height: float, max_height: float) -> ExternalCommand:
    now = time.time()
    try:
        with cmd_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return ExternalCommand("RL_FULL", 0.0, 0.0, 0.0, last_height, False)
    except Exception as exc:
        print(f"[WARN] cannot read {cmd_file}: {exc}")
        return ExternalCommand("RL_FULL", 0.0, 0.0, 0.0, last_height, False)

    age = now - float(raw.get("timestamp", 0.0))
    fresh = age <= stale_s
    fsm = str(raw.get("fsm") or "RL_FULL")
    height = clamp(float(raw.get("height", last_height)), min_height, max_height)

    # Safety FSMs win regardless of freshness.
    if bool(raw.get("limp")) or fsm == "LIMP":
        return ExternalCommand("LIMP", 0.0, 0.0, 0.0, height, fresh, estop=True)
    if bool(raw.get("estop")) or fsm == "DAMP":
        return ExternalCommand("DAMP", 0.0, 0.0, 0.0, height, fresh, estop=True)

    # Grasp state, or a stale command -> hold height, zero velocity (standing, not falling).
    if not fresh or fsm == "RL_LOWER":
        return ExternalCommand(fsm if fresh else "RL_FULL", 0.0, 0.0, 0.0, height, fresh)

    vel = raw.get("velocity") or {}
    vx = float(vel.get("forward", 0.0))
    vy = float(vel.get("lateral", 0.0))
    wz = float(vel.get("yaw", 0.0))
    # agile_keyboard writes physical units (units="agile"); a normalized writer (RobotMover)
    # gets rescaled by the per-axis caps.
    if raw.get("units") != "agile":
        vx *= fwd_max
        vy *= lat_max
        wz *= yaw_max
    vx = clamp(vx, -fwd_max, fwd_max)
    vy = clamp(vy, -lat_max, lat_max)
    wz = clamp(wz, -yaw_max, yaw_max)
    return ExternalCommand(fsm, vx, vy, wz, height, fresh)


# --------------------------------------------------------------------------- #
# AGILE sim2mujoco stack (verbatim wrap)
# --------------------------------------------------------------------------- #
def import_agile_stack(repo: Path):
    repo = repo.expanduser().resolve()
    if not (repo / "agile").is_dir():
        raise SystemExit(f"AGILE repo (WBC-AGILE) not found: {repo}")
    for p in (str(repo), str(repo / "agile" / "algorithms" / "rsl_rl")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from agile.sim2mujoco.actions import ActionProcessor
    from agile.sim2mujoco.command_provider import VelocityCommandProvider, create_command_provider
    from agile.sim2mujoco.observations import ObservationProcessor
    from agile.sim2mujoco.policy import PolicyWrapper
    from agile.sim2mujoco.simulation import JointCommand, SimState
    from agile.sim2mujoco.utils import load_config
    return {
        "ActionProcessor": ActionProcessor,
        "VelocityCommandProvider": VelocityCommandProvider,
        "create_command_provider": create_command_provider,
        "ObservationProcessor": ObservationProcessor,
        "PolicyWrapper": PolicyWrapper,
        "JointCommand": JointCommand,
        "SimState": SimState,
        "load_config": load_config,
    }


# --------------------------------------------------------------------------- #
# DDS state source + lowcmd publisher
# --------------------------------------------------------------------------- #
class LowStateSource:
    """Latest rt/lowstate (HG) held from a DDS subscriber callback."""

    def __init__(self, topic: str = "rt/lowstate"):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self._latest = None
        self.sub = ChannelSubscriber(topic, LowState_)
        self.sub.Init(self._on_msg, 10)

    def _on_msg(self, msg) -> None:
        self._latest = msg

    def latest(self):
        return self._latest

    def wait(self, timeout_s: float = 5.0):
        t0 = time.time()
        while self._latest is None:
            if time.time() - t0 > timeout_s:
                raise TimeoutError("no rt/lowstate received; check --interface / DDS domain")
            time.sleep(0.01)
        return self._latest


class LowCmdRlPublisher:
    """Publish AGILE leg targets to rt/lowcmd_rl; hold 12-28 for the arm_sdk overlay."""

    def __init__(self, mode_pr: int, mode_machine: int, topic: str = "rt/lowcmd_rl"):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        # PosStopF/VelStopF live in unitree's constants; fall back to the documented values.
        try:
            from unitree_sdk2py.utils.thread import PosStopF, VelStopF  # type: ignore
            self.pos_stop, self.vel_stop = float(PosStopF), float(VelStopF)
        except Exception:
            self.pos_stop, self.vel_stop = 2.146e9, 16000.0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.pub = ChannelPublisher(topic, LowCmd_)
        self.pub.Init()
        self.crc = CRC()
        self.mode_pr = int(mode_pr)
        self.mode_machine = int(mode_machine)
        self._init_lowcmd()

    def _mode_for(self, slot: int) -> int:
        return 0x01 if (slot in WEAK_MOTOR_SLOTS or slot >= BODY_MOTORS) else 0x0A

    def _init_lowcmd(self) -> None:
        if hasattr(self.low_cmd, "level_flag"):
            self.low_cmd.level_flag = 0xFF
        if hasattr(self.low_cmd, "gpio"):
            self.low_cmd.gpio = 0
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for(i)
            m.q = self.pos_stop
            m.dq = self.vel_stop
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def _measured_q(self, low_state, slot: int) -> float:
        try:
            return float(low_state.motor_state[slot].q)
        except Exception:
            return 0.0

    def publish_legs(self, leg_targets: dict[int, tuple[float, float, float]],
                     low_state, hold_kp: float, hold_kd: float) -> None:
        """leg_targets: {motor_slot -> (q, kp, kd)} for slots 0..11.
        12..28 held at measured q with (hold_kp, hold_kd); slot29=0; 30..34 passive."""
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for(i)
            m.dq = 0.0
            m.tau = 0.0
            if i in leg_targets:
                q, kp, kd = leg_targets[i]
                m.q, m.kp, m.kd = float(q), float(kp), float(kd)
            elif i == ARM_SDK_ENABLE_SLOT:
                m.q, m.kp, m.kd = 0.0, 0.0, 0.0
            elif i < BODY_MOTORS:                       # 12..28 -> hold for arm_sdk overlay
                m.q, m.kp, m.kd = self._measured_q(low_state, i), float(hold_kp), float(hold_kd)
            else:                                       # 30..34 -> passive at measured q
                m.q, m.kp, m.kd = self._measured_q(low_state, i), 0.0, 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def publish_damping(self, low_state, kd_value: float) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for(i)
            m.q = 0.0 if i == ARM_SDK_ENABLE_SLOT else self._measured_q(low_state, i)
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0 if i == ARM_SDK_ENABLE_SLOT else float(kd_value)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def publish_limp(self) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for(i)
            m.q = 0.0 if i == ARM_SDK_ENABLE_SLOT else self.pos_stop
            m.dq = self.vel_stop
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AGILE 23-DoF rt/lowcmd_rl pipeline for box_demo_2 / RoboJuDo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--agile-repo", type=Path,
                   default=Path(os.environ.get("AGILE_REPO", "~/WBC-AGILE")))
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="23-DoF student .pt (prefer the _checkpoint.pt for clean LSTM state)")
    p.add_argument("--config", type=Path, required=True,
                   help="re-exported 23-DoF IODescriptor YAML (NOT the 29-DoF one)")
    p.add_argument("--interface", "--iface", dest="interface",
                   default=os.environ.get("UNITREE_DDS_INTERFACE", "enP8p1s0"))
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu (default; ~0.15ms LSTM) or cuda")
    p.add_argument("--cmd-file", type=Path, default=Path(CMD_FILE))
    p.add_argument("--publish-topic", default="rt/lowcmd_rl")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--cmd-stale-s", type=float, default=0.40)
    p.add_argument("--fwd-max", type=float, default=0.50)
    p.add_argument("--lat-max", type=float, default=0.30)
    p.add_argument("--yaw-max", type=float, default=0.60)
    # height domain: the deep-squat 23-DoF student trains down to ~0.20m, so DO NOT use the
    # stock command_manager (0.40,0.72) clamp. Override its height_range with these.
    p.add_argument("--min-height", type=float, default=0.20)
    p.add_argument("--max-height", type=float, default=0.74)
    p.add_argument("--stand-height", type=float, default=0.72)
    p.add_argument("--height-rate", type=float, default=0.20, help="height slew rate (m/s)")
    p.add_argument("--hold-kp", type=float, default=40.0, help="kp holding motors 12-28")
    p.add_argument("--hold-kd", type=float, default=1.0, help="kd holding motors 12-28")
    p.add_argument("--mode-pr", type=int, default=0, help="LowCmd mode_pr (VERIFY on robot)")
    p.add_argument("--mode-machine", type=int, default=0)
    p.add_argument("--allow-zero-mode-machine", action="store_true",
                   help="do not abort if rt/lowstate reports mode_machine==0")
    # safety guards
    p.add_argument("--tilt-limit-deg", type=float, default=35.0)
    p.add_argument("--max-delta-rad", type=float, default=0.6, help="per-tick leg target slew cap")
    p.add_argument("--overspeed-rad-s", type=float, default=30.0)
    p.add_argument("--state-timeout-s", type=float, default=0.5, help="rt/lowstate staleness -> DAMP")
    p.add_argument("--no-remote-estop", action="store_true", help="ignore wireless_remote estop bit")
    p.add_argument("--damping-kd", type=float, default=8.0)
    p.add_argument("--pd-scale", type=float, default=1.0, help="scale leg kp/kd (0.3-0.5 if unstable)")
    p.add_argument("--dry-run", action="store_true", help="compute everything, publish nothing")
    p.add_argument("--print-every-s", type=float, default=1.0)
    return p


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = build_arg_parser().parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _load_ddsc()

    import numpy as np
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)

    A = import_agile_stack(args.agile_repo)
    config = A["load_config"](args.config)

    # joint_names = the descriptor's articulation order (deploy has no MuJoCo to ask).
    try:
        joint_names = list(config["articulations"]["robot"]["joint_names"])
    except Exception as exc:
        raise SystemExit(f"config missing articulations.robot.joint_names: {exc}")
    n_joints = len(joint_names)

    # optional PD scale (mirrors sim2mujoco_eval --pd-scale).
    if args.pd_scale != 1.0:
        rc = config["articulations"]["robot"]
        rc["default_joint_stiffness"] = [k * args.pd_scale for k in rc["default_joint_stiffness"]]
        rc["default_joint_damping"] = [k * args.pd_scale for k in rc["default_joint_damping"]]

    policy = A["PolicyWrapper"].from_config(args.checkpoint, config, device)
    obs_processor = A["ObservationProcessor"](config, joint_names, device)
    command_provider = A["create_command_provider"](config, device,
                                                     motion_tracker=obs_processor.motion_tracker)
    if not isinstance(command_provider, A["VelocityCommandProvider"]):
        raise SystemExit("expected a velocity command provider (velocity-height policy)")
    command_manager = command_provider.manager
    obs_processor.command_manager = command_manager
    # widen the height clamp for the deep-squat student (stock is (0.40,0.72)).
    if hasattr(command_manager, "height_range"):
        command_manager.height_range = (float(args.min_height), float(args.max_height))
    act_processor = A["ActionProcessor"](config, joint_names, device)

    obs_dim = int(obs_processor.total_obs_dim)
    print("=" * 72)
    print("AGILE 23-DoF rt/lowcmd_rl pipeline (box_demo_2 / RoboJuDo wire line)")
    print(f"  repo:       {args.agile_repo.expanduser().resolve()}")
    print(f"  config:     {args.config}  (joints={n_joints}, obs_dim={obs_dim})")
    print(f"  checkpoint: {args.checkpoint}  (policy={type(policy).__name__})")
    print(f"  publish:    {'DRY-RUN' if args.dry_run else args.publish_topic} @ {args.hz:.0f}Hz, iface={args.interface}")
    print(f"  height:     {args.min_height:.2f}..{args.max_height:.2f}m  rate={args.height_rate:.2f}m/s")
    print(f"  guards:     tilt>{args.tilt_limit_deg:.0f}deg, |dq|>{args.overspeed_rad_s:.0f}, "
          f"d<={args.max_delta_rad:.2f}rad/tick, state_to={args.state_timeout_s:.2f}s")
    print("  merger:     keep merge_lowcmd_arm_sdk.py running (sole rt/lowcmd publisher)")
    print("=" * 72)

    # map each descriptor joint -> its leg motor slot (None for non-leg/held joints).
    slot_for_joint: list[int | None] = []
    for jn in joint_names:
        slot = MOTOR_BY_JOINT.get(strip_joint(jn))
        if slot is None:
            raise SystemExit(f"joint '{jn}' not in MOTOR_BY_JOINT; check descriptor names")
        slot_for_joint.append(slot if slot in LEG_SLOTS else None)
    n_leg_driven = sum(1 for s in slot_for_joint if s is not None)
    if n_leg_driven != 12:
        print(f"[WARN] expected 12 driven leg joints, got {n_leg_driven} — VERIFY descriptor")

    state_src = LowStateSource("rt/lowstate")
    print("waiting for rt/lowstate ...")
    first = state_src.wait(timeout_s=10.0)
    sm = int(getattr(first, "mode_machine", 0) or 0)
    if sm == 0 and not args.allow_zero_mode_machine:
        raise SystemExit("rt/lowstate mode_machine==0 (motors not enabled?); "
                         "pass --allow-zero-mode-machine to override")
    mode_machine = args.mode_machine or sm
    publisher = None if args.dry_run else LowCmdRlPublisher(args.mode_pr, mode_machine, args.publish_topic)

    obs_processor.reset()
    policy.reset()                      # LSTM hidden/cell zeroed ONCE — never again in-loop

    tilt_cos = math.cos(math.radians(args.tilt_limit_deg))
    dt = 1.0 / float(args.hz)
    height_cmd = clamp(float(args.stand_height), args.min_height, args.max_height)
    prev_leg_q: dict[int, float] = {}
    last_print = 0.0
    damp_latched = False

    def build_simstate(low_state):
        q = np.zeros(n_joints, dtype=np.float32)
        dq = np.zeros(n_joints, dtype=np.float32)
        for i, jn in enumerate(joint_names):
            slot = MOTOR_BY_JOINT[strip_joint(jn)]
            ms = low_state.motor_state[slot]
            q[i] = float(ms.q)
            dq[i] = float(ms.dq)
        imu = low_state.imu_state
        quat = np.asarray(imu.quaternion, dtype=np.float32)        # [w,x,y,z] (VERIFY order)
        gyro = np.asarray(imu.gyroscope, dtype=np.float32)         # body-frame rad/s
        z = torch.zeros(3, dtype=torch.float32, device=device)
        return A["SimState"](
            joint_pos=torch.as_tensor(q, device=device),
            joint_vel=torch.as_tensor(dq, device=device),
            root_pos=z,
            root_quat=torch.as_tensor(quat, device=device),
            root_lin_vel=z,                                        # unobservable; student ignores
            root_ang_vel=torch.as_tensor(gyro, device=device),
        ), quat, dq

    def remote_estop(low_state) -> bool:
        if args.no_remote_estop:
            return False
        try:
            wr = low_state.wireless_remote
            return bool(wr[2] & 0x08)            # SELECT bit (byte2 bit3); VERIFY mapping
        except Exception:
            return False

    last_state_t = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            low_state = state_src.latest()

            # --- state staleness / availability -> DAMP ---
            if low_state is None:
                if not args.dry_run and publisher is not None:
                    publisher.publish_damping(first, args.damping_kd)
                time.sleep(dt)
                continue
            last_state_t = time.monotonic()

            cmd = read_external_command(args.cmd_file, height_cmd, args.cmd_stale_s,
                                        args.fwd_max, args.lat_max, args.yaw_max,
                                        args.min_height, args.max_height)
            height_cmd = approach(height_cmd, cmd.height, float(args.height_rate) * dt)

            estop = cmd.estop or remote_estop(low_state) or damp_latched
            if cmd.fsm == "LIMP":
                if publisher:
                    publisher.publish_limp()
                _maybe_print(args, cmd, height_cmd, "LIMP")
                _pace(t0, dt); continue
            if estop or cmd.fsm == "DAMP":
                if publisher:
                    publisher.publish_damping(low_state, args.damping_kd)
                _maybe_print(args, cmd, height_cmd, "DAMP")
                _pace(t0, dt); continue

            # --- RL_FULL / RL_LOWER: run the policy ---
            command_manager.set_command(cmd.vx, cmd.vy, cmd.wz, height_cmd)
            sim_state, quat, dq = build_simstate(low_state)

            # tilt guard (independent of obs term) — #1 fall risk
            gz = projected_gravity_z(*[float(x) for x in quat])
            if not math.isfinite(gz) or (-gz) < tilt_cos:
                damp_latched = True
                if publisher:
                    publisher.publish_damping(low_state, args.damping_kd)
                print(f"[ABORT] tilt guard: -gz={-gz:.3f} < cos(limit)={tilt_cos:.3f} -> DAMP latched")
                _pace(t0, dt); continue

            obs = obs_processor.compute(sim_state, noise_scale=0.0)
            if obs_dim and obs.numel() != obs_dim:
                raise SystemExit(f"obs dim {obs.numel()} != expected {obs_dim}")
            if not bool(torch.isfinite(obs).all()):
                damp_latched = True
                if publisher:
                    publisher.publish_damping(low_state, args.damping_kd)
                print("[ABORT] non-finite obs -> DAMP latched")
                _pace(t0, dt); continue

            with torch.no_grad():
                actions = policy(obs)
            obs_processor.set_last_action(actions)
            joint_cmd = act_processor.process(actions)

            pos = joint_cmd.position.detach().cpu().numpy()
            kp = joint_cmd.kp.detach().cpu().numpy()
            kd = joint_cmd.kd.detach().cpu().numpy()
            if not np.isfinite(pos).all():
                damp_latched = True
                if publisher:
                    publisher.publish_damping(low_state, args.damping_kd)
                print("[ABORT] non-finite leg target -> DAMP latched")
                _pace(t0, dt); continue

            # overspeed abort (measured leg dq)
            leg_idx = [i for i, s in enumerate(slot_for_joint) if s is not None]
            if any(abs(float(dq[i])) > args.overspeed_rad_s for i in leg_idx):
                damp_latched = True
                if publisher:
                    publisher.publish_damping(low_state, args.damping_kd)
                print("[ABORT] leg overspeed -> DAMP latched")
                _pace(t0, dt); continue

            # build per-slot leg targets with per-tick rate limiting
            leg_targets: dict[int, tuple[float, float, float]] = {}
            for i, slot in enumerate(slot_for_joint):
                if slot is None:
                    continue
                q_t = float(pos[i])
                prev = prev_leg_q.get(slot, float(low_state.motor_state[slot].q))
                q_t = clamp(q_t, prev - args.max_delta_rad, prev + args.max_delta_rad)
                prev_leg_q[slot] = q_t
                leg_targets[slot] = (q_t, float(kp[i]), float(kd[i]))

            if publisher:
                publisher.publish_legs(leg_targets, low_state, args.hold_kp, args.hold_kd)
            _maybe_print(args, cmd, height_cmd, cmd.fsm, extra=f"obs={obs.numel()}")
            _pace(t0, dt)
    except KeyboardInterrupt:
        print("\nstopping; sending damping ...")
        if publisher:
            for _ in range(int(args.hz)):
                ls = state_src.latest() or first
                publisher.publish_damping(ls, args.damping_kd)
                time.sleep(dt)


def _pace(t0: float, dt: float) -> None:
    time.sleep(max(0.0, dt - (time.monotonic() - t0)))


def _maybe_print(args, cmd, height_cmd, state, extra: str = "") -> None:
    now = time.monotonic()
    if now - _maybe_print.last >= float(args.print_every_s):
        print(f"[AGILE23] fsm={state:8s} fresh={int(cmd.fresh)} "
              f"cmd=({cmd.vx:+.2f},{cmd.vy:+.2f},{cmd.wz:+.2f},h={height_cmd:.2f}) {extra}")
        _maybe_print.last = now


_maybe_print.last = 0.0


if __name__ == "__main__":
    main()
