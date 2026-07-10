#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared IPC command streamer + keyboard control for the real-robot tests.

Streams a schedules.py profile into /tmp/robojudo_ext_cmd.json (units="agile",
physical m/s — the SAME schema both the GR00T adapter and agile_lowcmd_pipeline
read), with pause / resume / stand / e-stop keys. Controller-agnostic: works
with whichever base adapter is running. SINGLE WRITER: stop the keyboard pane
before running a test script.

Keys (raw stdin, no Enter):
  space  pause  (velocity -> 0, height frozen; schedule clock stops)
  r      resume (schedule clock continues from the pause point)
  s      abort -> stand (zero velocity, stand height)
  o      E-STOP: DAMP (kd-only) and exit
  q      quit  (zero velocity, keep balancing)
"""
from __future__ import annotations

import json
import os
import select
import sys
import termios
import time
import tty

CMD_FILE = "/tmp/robojudo_ext_cmd.json"
WRITE_HZ = 20.0


def write_cmd(fsm: str, vx: float, vy: float, wz: float, height: float,
              estop: bool = False, cmd_file: str = CMD_FILE) -> None:
    cmd = {
        "fsm": fsm,
        "velocity": {"forward": float(vx), "lateral": float(vy), "yaw": float(wz)},
        "height": float(height),
        "units": "agile",
        "timestamp": time.time(),
        "source": "three_tests",
    }
    if estop:
        cmd["estop"] = True
    tmp = cmd_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, cmd_file)


class RawKeys:
    """Non-blocking single-key reader."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def stream_profile(total: float, sample_fn, *, stand_height: float,
                   label: str = "test", on_done: str = "stand",
                   status_every_s: float = 1.0) -> str:
    """Stream sample_fn through the IPC file with pause/resume keys.

    Returns how it ended: "done" | "stand" | "quit" | "estop".
    The schedule clock only advances while NOT paused, so resume continues
    exactly where the motion left off.
    """
    print(f"[{label}] total {total:.1f}s.  keys: space=暂停  r=继续  s=回站立  "
          f"o=急停DAMP  q=退出", flush=True)
    period = 1.0 / WRITE_HZ
    t_sched = 0.0
    paused = False
    last_status = 0.0
    frozen_height = stand_height
    with RawKeys() as keys:
        while t_sched < total:
            k = keys.get()
            if k == " ":
                paused = True
                print(f"[{label}] 暂停 @ {t_sched:.1f}s (速度归零)", flush=True)
            elif k == "r" and paused:
                paused = False
                print(f"[{label}] 继续", flush=True)
            elif k == "s":
                print(f"[{label}] 中止 -> 回站立", flush=True)
                write_cmd("RL_FULL", 0, 0, 0, stand_height)
                return "stand"
            elif k == "o":
                print(f"[{label}] 急停 DAMP!", flush=True)
                write_cmd("DAMP", 0, 0, 0, frozen_height, estop=True)
                return "estop"
            elif k == "q":
                print(f"[{label}] 退出 (速度归零)", flush=True)
                write_cmd("RL_FULL", 0, 0, 0, frozen_height)
                return "quit"

            if paused:
                write_cmd("RL_FULL", 0, 0, 0, frozen_height)
            else:
                c = sample_fn(t_sched)
                frozen_height = c["height"]
                write_cmd("RL_FULL", c["vx"], c["vy"], c["wz"], c["height"])
                t_sched += period
                if t_sched - last_status >= status_every_s:
                    print(f"[{label}] t={t_sched:5.1f}/{total:.1f}s  "
                          f"vx={c['vx']:+.2f} vy={c['vy']:+.2f} wz={c['wz']:+.2f} "
                          f"h={c['height']:.2f}", flush=True)
                    last_status = t_sched
            time.sleep(period)
    if on_done == "stand":
        write_cmd("RL_FULL", 0, 0, 0, stand_height)
    print(f"[{label}] 完成", flush=True)
    return "done"
