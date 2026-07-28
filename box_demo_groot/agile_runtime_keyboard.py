#!/usr/bin/env python3
"""DDS-free operator keyboard for the refactored onboard runtime."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onboard_runtime.motion_bus import MotionBusClient, MotionBusError

from arm_control_ipc import ArmControlClient


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-bus-socket", default="/tmp/groot_motion_bus.sock")
    parser.add_argument("--arm-control-socket", default="/tmp/groot_arm_control.sock")
    parser.add_argument(
        "--arm-control-status-file",
        default="/tmp/groot_arm_control_status.json",
    )
    parser.add_argument("--vx", type=float, default=0.40)
    parser.add_argument("--vy", type=float, default=0.20)
    parser.add_argument("--wz", type=float, default=0.40)
    parser.add_argument("--stand-height", type=float, default=0.76)
    parser.add_argument("--pick-height", type=float, default=0.36)
    parser.add_argument("--height-step", type=float, default=0.02)
    parser.add_argument("--min-height", type=float, default=0.30)
    parser.add_argument("--max-height", type=float, default=0.80)
    parser.add_argument("--refresh-hz", type=float, default=20.0)
    parser.add_argument("--key-timeout", type=float, default=0.25)
    args = parser.parse_args()

    if not args.min_height <= args.stand_height <= args.max_height:
        parser.error("--stand-height must be inside --min-height..--max-height")

    motion = MotionBusClient("operator.keyboard", args.motion_bus_socket)
    arms = ArmControlClient(
        args.arm_control_socket,
        args.arm_control_status_file,
    )
    height = clamp(args.stand_height, args.min_height, args.max_height)
    fsm = "RL_FULL"
    vx = vy = wz = 0.0
    last_motion_key = 0.0
    dirty = True
    period = 1.0 / args.refresh_hz

    print("=" * 68)
    print("Runtime keyboard -> lease-based motion bus (no DDS participant)")
    print("w/s/a/d/q/e move, z/x height, c pick, r stand, space stop")
    print("f RL_FULL, l RL_LOWER, h natural arm hang, o DAMP, Ctrl+C exit")
    print("=" * 68)

    old_terminal = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], period)
            if ready:
                key = sys.stdin.read(1)
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "w":
                    vx, vy, wz = args.vx, 0.0, 0.0
                    last_motion_key = time.monotonic()
                elif key == "s":
                    vx, vy, wz = -args.vx, 0.0, 0.0
                    last_motion_key = time.monotonic()
                elif key == "a":
                    vx, vy, wz = 0.0, args.vy, 0.0
                    last_motion_key = time.monotonic()
                elif key == "d":
                    vx, vy, wz = 0.0, -args.vy, 0.0
                    last_motion_key = time.monotonic()
                elif key == "q":
                    vx, vy, wz = 0.0, 0.0, args.wz
                    last_motion_key = time.monotonic()
                elif key == "e":
                    vx, vy, wz = 0.0, 0.0, -args.wz
                    last_motion_key = time.monotonic()
                elif key == "z":
                    height = clamp(
                        height - args.height_step, args.min_height, args.max_height
                    )
                    dirty = True
                elif key == "x":
                    height = clamp(
                        height + args.height_step, args.min_height, args.max_height
                    )
                    dirty = True
                elif key == "c":
                    height = clamp(
                        args.pick_height, args.min_height, args.max_height
                    )
                    dirty = True
                elif key == "r":
                    height = args.stand_height
                    dirty = True
                elif key == " ":
                    vx = vy = wz = 0.0
                    dirty = True
                elif key == "f":
                    fsm = "RL_FULL"
                    dirty = True
                elif key == "l":
                    fsm = "RL_LOWER"
                    vx = vy = wz = 0.0
                    dirty = True
                elif key == "h":
                    vx = vy = wz = 0.0
                    last_motion_key = 0.0
                    dirty = True
                    ok, message = arms.toggle()
                    print(f"\n[ARM_HANG]{'' if ok else '[WARN]'} {message}")
                elif key == "o":
                    vx = vy = wz = 0.0
                    last_motion_key = 0.0
                    motion.publish(
                        fsm="DAMP",
                        height=height,
                        lease_s=5.0,
                        estop=True,
                    )
                    arms.emergency_release()
                    print("\n[DAMP] safety request sent")
                    continue

            active = time.monotonic() - last_motion_key <= args.key_timeout
            if not active and (vx or vy or wz):
                vx = vy = wz = 0.0
                dirty = True
            if active or dirty:
                motion.publish(
                    fsm=fsm,
                    forward=vx,
                    lateral=vy,
                    yaw=wz,
                    height=height,
                    lease_s=max(0.10, args.key_timeout + 0.10),
                    allow_recovery=fsm == "RL_FULL",
                )
                dirty = False
            print(
                f"\rmode={fsm:8s} vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} "
                f"height={height:.2f}m arm={arms.state:19s}",
                end="",
                flush=True,
            )
    except (KeyboardInterrupt, MotionBusError) as exc:
        print(f"\nexit: {exc or 'operator request'}")
        try:
            motion.publish(fsm="RL_FULL", height=height, lease_s=0.3)
        except MotionBusError:
            pass
    finally:
        arms.close(release=True)
        motion.close(release=True)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)


if __name__ == "__main__":
    main()
