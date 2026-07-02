#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyboard teleop for the AGILE 23-DoF pipeline (SINGLE writer of /tmp/robojudo_ext_cmd.json).

Mirrors AGILE's own sim2mujoco keyboard scheme (commands.py / simulation.py:_key_callback),
emitting PHYSICAL units with units="agile" so agile23_lowcmd_pipeline.py uses them directly:

  I / ↑    vx += 0.10   (forward, +x body)        K / ↓    vx -= 0.10  (back)
  J / ←    vy += 0.10   (LEFT strafe, +y body)    L / →    vy -= 0.10  (right strafe)
  U        wz += 0.20   (turn LEFT, CCW)          O        wz -= 0.20  (turn right)
  9 / PgUp height += 0.05 (absolute m)            0 / PgDn height -= 0.05
  H        stop -> vx=vy=wz=0 (height kept)
  1 RL_FULL   2 RL_LOWER   3/space DAMP(estop)    4 LIMP    5 clear estop -> RL_FULL
  P print     Q / ESC quit (sends DAMP)

A heartbeat thread re-writes the JSON at --hz with a fresh timestamp so the command never
goes stale (the pipeline forces zero velocity when age > cmd-stale-s). DO NOT run this at the
same time as box_demo's mover — the IPC file must have exactly one writer.

Sign convention (load-bearing, must match training): +vx forward, +vy robot-left, +wz CCW.
Height is the ABSOLUTE base-height target in meters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import termios
import threading
import time
import tty
from pathlib import Path

CMD_FILE = "/tmp/robojudo_ext_cmd.json"

VEL_STEP = 0.10
ANG_STEP = 0.20
HEIGHT_STEP = 0.05


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TeleopState:
    def __init__(self, args):
        self.lock = threading.Lock()
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.height = float(args.stand_height)
        self.fsm = "RL_FULL"
        self.estop = False
        self.a = args
        self.running = True

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "fsm": self.fsm,
                "velocity": {"forward": self.vx, "lateral": self.vy, "yaw": self.wz},
                "height": self.height,
                "units": "agile",
                "estop": self.estop,
                "timestamp": time.time(),
                "source": "agile23_keyboard",
            }

    def apply(self, key: str) -> None:
        a = self.a
        with self.lock:
            if key in ("i", "I", "UP"):
                self.vx = clamp(self.vx + VEL_STEP, -a.fwd_max, a.fwd_max)
            elif key in ("k", "K", "DOWN"):
                self.vx = clamp(self.vx - VEL_STEP, -a.fwd_max, a.fwd_max)
            elif key in ("j", "J", "LEFT"):
                self.vy = clamp(self.vy + VEL_STEP, -a.lat_max, a.lat_max)
            elif key in ("l", "L", "RIGHT"):
                self.vy = clamp(self.vy - VEL_STEP, -a.lat_max, a.lat_max)
            elif key in ("u", "U"):
                self.wz = clamp(self.wz + ANG_STEP, -a.yaw_max, a.yaw_max)
            elif key in ("o", "O"):
                self.wz = clamp(self.wz - ANG_STEP, -a.yaw_max, a.yaw_max)
            elif key in ("9", "PGUP"):
                self.height = clamp(self.height + HEIGHT_STEP, a.min_height, a.max_height)
            elif key in ("0", "PGDN"):
                self.height = clamp(self.height - HEIGHT_STEP, a.min_height, a.max_height)
            elif key in ("h", "H"):
                self.vx = self.vy = self.wz = 0.0
            elif key == "1":
                self.fsm, self.estop = "RL_FULL", False
            elif key == "2":
                self.fsm = "RL_LOWER"
            elif key in ("3", " "):
                self.fsm, self.estop = "DAMP", True
                self.vx = self.vy = self.wz = 0.0
            elif key == "4":
                self.fsm, self.estop = "LIMP", True
            elif key == "5":
                self.fsm, self.estop = "RL_FULL", False
            else:
                return
        self._print()

    def _print(self) -> None:
        print(f"\rfsm={self.fsm:8s} estop={int(self.estop)} "
              f"vx={self.vx:+.2f} vy={self.vy:+.2f} wz={self.wz:+.2f} h={self.height:.2f}    ",
              end="", flush=True)


def writer_loop(state: TeleopState, cmd_file: Path, hz: float) -> None:
    dt = 1.0 / hz
    tmp = cmd_file.with_suffix(".json.tmp")
    while state.running:
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(state.snapshot(), f)
            os.replace(tmp, cmd_file)        # atomic publish (single writer)
        except Exception as exc:
            print(f"\n[WARN] write {cmd_file}: {exc}")
        time.sleep(dt)


def read_key(stream) -> str:
    """Return a logical key name; decode arrow/page escape sequences."""
    ch = stream.read(1)
    if ch != "\x1b":
        return ch
    seq = stream.read(2)
    mapping = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT", "[5": "PGUP", "[6": "PGDN"}
    if seq in mapping:
        if seq in ("[5", "[6"):
            stream.read(1)                    # consume trailing '~'
        return mapping[seq]
    return "ESC"


def main() -> None:
    p = argparse.ArgumentParser(description="AGILE 23-DoF keyboard teleop (IPC single writer)")
    p.add_argument("--cmd-file", type=Path, default=Path(CMD_FILE))
    p.add_argument("--hz", type=float, default=20.0, help="heartbeat write rate")
    p.add_argument("--fwd-max", type=float, default=0.50)
    p.add_argument("--lat-max", type=float, default=0.30)
    p.add_argument("--yaw-max", type=float, default=0.60)
    p.add_argument("--min-height", type=float, default=0.20)
    p.add_argument("--max-height", type=float, default=0.74)
    p.add_argument("--stand-height", type=float, default=0.72)
    args = p.parse_args()

    state = TeleopState(args)
    wt = threading.Thread(target=writer_loop, args=(state, args.cmd_file, args.hz), daemon=True)
    wt.start()

    print(__doc__)
    print(f"writing {args.cmd_file} @ {args.hz:.0f}Hz. Press keys (Q/ESC to quit).")
    state._print()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = read_key(sys.stdin)
            if key in ("q", "Q", "ESC", "\x03"):
                break
            if key in ("p", "P"):
                state._print()
                continue
            state.apply(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # leave the robot safe: latch DAMP for a moment, then stop the heartbeat.
        with state.lock:
            state.fsm, state.estop = "DAMP", True
            state.vx = state.vy = state.wz = 0.0
        time.sleep(0.3)
        state.running = False
        print("\nstopped (DAMP latched).")


if __name__ == "__main__":
    main()
