#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim2sim test of the ACTUAL groot_mover.py through the validated policy sim.

Unlike sweep_step_limit.py (which issues raw vx/T), this drives the REAL edited
`groot_mover.GrootMover` code path — _snap_min_distance -> solve_linear ->
_guard_walk_height -> _execute_move(settle-before -> warm-up -> move -> settle) —
with the shipped defaults, against the same MuJoCo Balance/Walk policy sim. It is
the end-to-end sim2sim check of the fix before real-robot deployment: does calling
`mover.move_forward_cm(4)` actually net forward (never backward) across sway phases?

Run on the 5080: MUJOCO_GL=egl ~/GR00T-WholeBodyControl/.venv_wbc/bin/python sim2sim_groot_mover.py
(needs sweep_step_limit.py + groot_mover.py in the same dir.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_step_limit as S   # validated Sim + policy (obs/PD/onnx)
import groot_mover as gm       # the REAL, edited mover under test

CTRL_DT = S.SIM_DT * S.DECIM   # 0.02 s per control tick (50 Hz)
STAND = 0.74


class SimMover(gm.GrootMover):
    """GrootMover whose _hold/_settle STEP THE SIM instead of writing IPC + sleeping.
    Everything else (planning, phase order, defaults, guards) is the real code."""

    def __init__(self, sim, **kw):
        stand_height = kw.pop("stand_height", STAND)
        super().__init__(verbose=False, stand_height=stand_height, **kw)
        self._sim = sim

    def _hold(self, forward, lateral, yaw, duration, cancel_event=None):
        if duration <= 0:
            return True
        for _ in range(int(round(duration / CTRL_DT))):
            if cancel_event is not None and cancel_event.is_set():
                return False
            self._sim.step_once([forward, lateral, yaw], self._height)
        return True

    def _settle(self, allow_recovery=None, cancel_event=None):
        del allow_recovery
        for _ in range(int(round(self.stop_hold_s / CTRL_DT))):
            if cancel_event is not None and cancel_event.is_set():
                return False
            self._sim.step_once([0.0, 0.0, 0.0], self._height)
        return True

    def set_height(self, height):
        self._height = float(gm._clamp(height, gm.MIN_HEIGHT, gm.MAX_HEIGHT))

    def initialize(self):
        pass  # no IPC in sim


def run_case(sim, kw, D_cm, pre_settle_s, post_s=2.0):
    """Settle to a sway phase, call the REAL move_forward_cm, return net dx (cm)."""
    sim.reset_stand()
    for _ in range(int(pre_settle_s / CTRL_DT)):
        sim.step_once([0.0, 0.0, 0.0], STAND)
    x0 = sim.base()[0]
    mv = SimMover(sim, **kw)
    mv._height = STAND
    mv.move_forward_cm(D_cm)               # <-- exercises the real groot_mover code
    for _ in range(int(post_s / CTRL_DT)):
        sim.step_once([0.0, 0.0, 0.0], STAND)
    return (sim.base()[0] - x0) * 100.0


# configs: OLD = pre-fix defaults (the bug); NEW = shipped defaults; variants
OLD = dict(min_duration=0.6, min_distance=0.0, v_floor=0.08,
           settle_before_s=0.0, warmup_time=0.0, fwd_cruise=0.12)
NEW = dict()                               # module defaults (the fix)
NEW_WARM = dict(warmup_time=0.4)           # + gait warm-up
NEW_SETTLE1 = dict(settle_before_s=1.0)    # + longer settle
NEW_GAIN = dict(dist_gain=1.7)             # + open-loop compensation


def sweep(sim, kw, tag):
    print(f"\n== {tag} ==")
    worst = 999.0
    for D in [4, 6, 8, 10, 15]:
        row = [run_case(sim, kw, D, ps) for ps in [1.0, 1.5, 2.0, 2.5, 3.5]]
        worst = min(worst, min(row))
        print(f"  req D={D:2d}cm  net by sway-phase: " +
              " ".join(f"{v:+5.1f}" for v in row) +
              f"   [min {min(row):+5.1f}  mean {sum(row)/len(row):+5.1f}  #neg {sum(1 for v in row if v < 0)}]")
    print(f"  --> worst-case net across ALL D & phases: {worst:+.1f}cm "
          f"({'FAIL: can still go backward' if worst < 0 else 'OK: never backward'})")


def settled_check(sim, kw, tag, D=8):
    """Realistic pipeline: robot stands >=2.5s (perception pause) before moving."""
    row = [run_case(sim, kw, D, ps) for ps in [2.5, 3.0, 3.5, 4.0, 5.0]]
    print(f"  {tag:42s} net@settled(2.5-5s): " + " ".join(f"{v:+5.1f}" for v in row) +
          f"   [min {min(row):+.1f}  #neg {sum(1 for v in row if v < 0)}]")


def warm_grid(sim, D=8):
    """Find a warm-up config with strictly-positive worst case across sway phases."""
    print("\n== WARM-UP GRID (worst-case net across sway phases 1.0-3.5s, D=8) ==")
    for wt in [0.4, 0.6, 0.8]:
        for ws in [0.15, 0.20]:
            for sb in [0.0, 0.5]:
                kw = dict(warmup_time=wt, warmup_speed=ws, settle_before_s=sb)
                row = [run_case(sim, kw, D, ps) for ps in [1.0, 1.5, 2.0, 2.5, 3.5]]
                flag = "OK" if min(row) > 0 else "neg"
                print(f"  warmup={wt}s@{ws} settle={sb}: " +
                      " ".join(f"{v:+5.1f}" for v in row) +
                      f"   [min {min(row):+5.1f} {flag}]")


if __name__ == "__main__":
    sim = S.Sim()
    print(f"model nq={sim.m.nq} nu={sim.m.nu}; groot_mover defaults: "
          f"min_dur={gm.MIN_DURATION} v_floor={gm.V_FLOOR} min_dist={gm.MIN_DISTANCE} "
          f"settle={gm.SETTLE_BEFORE_S} cruise={gm.FWD_CRUISE} walk_h>={gm.WALK_MIN_HEIGHT}",
          file=sys.stderr)
    sweep(sim, OLD, "OLD defaults (min_dur0.6 floor0.08 no-settle) -- the reported bug")
    sweep(sim, NEW, "NEW shipped defaults (settle0.5 min_dur1.5 floor0.12 min_dist0.08 cruise0.15)")
    sweep(sim, NEW_WARM, "NEW + warmup_time=0.4")

    print("\n== SETTLED REGIME (the realistic pipeline: >=2.5s stand before move) ==")
    settled_check(sim, NEW, "NEW")
    settled_check(sim, NEW_WARM, "NEW + warmup0.4")
    settled_check(sim, dict(warmup_time=0.6, settle_before_s=0.0), "warmup0.6 no-settle")

    warm_grid(sim)
