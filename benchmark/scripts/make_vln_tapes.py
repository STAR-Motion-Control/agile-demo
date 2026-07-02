#!/usr/bin/env python3
"""Generate VLN-style command tapes for the T10 vln_follow benchmark.

Produces tapes.json with 10 tapes (seeds 0-9). Each tape mimics the command
stream of a VLN/navigation model following a random smooth path:

  - a 2D unicycle (x, y, yaw; body-frame vx, vy, wz) is simulated at dt=0.02
  - a P controller chases a random waypoint sequence
  - controller output is re-sampled (zero-order hold) at random intervals
    of 0.4-1.2 s, clipped to |vx|<=0.35, |vy|<=0.2, |wz|<=0.30
  - per-waypoint speed caps create multiple small-command segments
    (|vx| < 0.15 / |wz| < 0.1)
  - 2 full-stop segments of 1-2 s are inserted
  - duration 30 s per tape

The ideal (error-free) integrated trajectory is stored as ref_xy_yaw at 1 Hz.

Output schema:
  {"tapes": [{"id": 0, "dt": 0.02,
              "cmds": [[t, vx, vy, wz], ...],        # ZOH breakpoints
              "ref_xy_yaw": [[t, x, y, yaw], ...]}]} # 1 Hz reference

Usage:
  python3 make_vln_tapes.py [--out PATH] [--n-tapes 10] [--duration 30.0]
"""

import argparse
import json
import math
import random
from pathlib import Path

# --- tape spec constants ---------------------------------------------------
DT = 0.02                      # sim step [s]
DURATION_S = 30.0              # tape length [s]
N_TAPES = 10                   # seeds 0..9
VX_MAX = 0.35                  # |vx| limit [m/s]
VY_MAX = 0.20                  # |vy| limit [m/s]
WZ_MAX = 0.30                  # |wz| limit [rad/s]
UPDATE_MIN_S = 0.4             # command update interval bounds [s]
UPDATE_MAX_S = 1.2
STOP_DUR_MIN_S = 1.0           # full-stop segment duration bounds [s]
STOP_DUR_MAX_S = 2.0
STOP1_START_RANGE = (6.0, 11.0)   # windows keep the two stops disjoint
STOP2_START_RANGE = (16.0, 23.0)
REF_HZ = 1.0                   # reference trajectory sample rate

# --- waypoint path constants -----------------------------------------------
N_WAYPOINTS = 16
WP_SPACING_RANGE = (1.0, 2.0)     # [m] between consecutive waypoints
WP_TURN_MAX_RAD = math.radians(60.0)
WP_REACH_DIST_M = 0.25
# per-waypoint forward speed cap [m/s]: legs alternate slow/fast regimes so
# every tape is guaranteed several small-command (|vx|<0.15) segments
VX_CAP_SLOW_RANGE = (0.08, 0.14)
VX_CAP_FAST_RANGE = (0.18, VX_MAX)

# --- P controller gains ------------------------------------------------------
KP_LIN = 0.8
KP_LAT = 0.8
KP_YAW = 1.2

SMALL_VX_BAND = (0.05, 0.15)   # band used for the small-command statistic


def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def make_waypoints(rng):
    """Random smooth polyline: bounded heading change per segment."""
    pts = [(0.0, 0.0)]
    heading = rng.uniform(-math.pi, math.pi)
    x, y = 0.0, 0.0
    for _ in range(N_WAYPOINTS):
        heading = wrap_angle(heading + rng.uniform(-WP_TURN_MAX_RAD,
                                                   WP_TURN_MAX_RAD))
        dist = rng.uniform(*WP_SPACING_RANGE)
        x, y = x + dist * math.cos(heading), y + dist * math.sin(heading)
        pts.append((x, y))
    return pts


def make_stop_windows(rng):
    """Two disjoint full-stop windows of 1-2 s each."""
    return tuple(
        (start, start + rng.uniform(STOP_DUR_MIN_S, STOP_DUR_MAX_S))
        for start in (rng.uniform(*STOP1_START_RANGE),
                      rng.uniform(*STOP2_START_RANGE))
    )


def in_stop(t, stop_windows):
    return any(s <= t < e for s, e in stop_windows)


def sample_vx_cap(rng, wp_idx):
    """Slow regime on odd legs, fast on even — guarantees small-cmd segments."""
    cap_range = VX_CAP_SLOW_RANGE if wp_idx % 2 else VX_CAP_FAST_RANGE
    return rng.uniform(*cap_range)


def p_control(state, waypoint, vx_cap):
    """P controller toward waypoint; returns clipped (vx, vy, wz)."""
    x, y, yaw = state
    dx, dy = waypoint[0] - x, waypoint[1] - y
    # body-frame position error
    ex = math.cos(yaw) * dx + math.sin(yaw) * dy
    ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
    yaw_err = wrap_angle(math.atan2(dy, dx) - yaw)
    # turn-then-go: forward speed fades when badly misaligned
    align = max(0.0, math.cos(yaw_err))
    vx = clip(KP_LIN * ex * align, 0.0, vx_cap)
    vy = clip(KP_LAT * ey, -VY_MAX, VY_MAX)
    wz = clip(KP_YAW * yaw_err, -WZ_MAX, WZ_MAX)
    return (round(vx, 4), round(vy, 4), round(wz, 4))


def integrate_step(state, cmd):
    """One dt of unicycle kinematics with body-frame velocities."""
    x, y, yaw = state
    vx, vy, wz = cmd
    return (x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT,
            y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT,
            wrap_angle(yaw + wz * DT))


def generate_tape(tape_id, duration_s):
    """Simulate one tape; returns (tape_dict, stats_dict)."""
    rng = random.Random(tape_id)
    waypoints = make_waypoints(rng)
    stop_windows = make_stop_windows(rng)

    n_steps = int(round(duration_s / DT))
    ref_stride = int(round(1.0 / (REF_HZ * DT)))

    state = (0.0, 0.0, 0.0)
    wp_idx, vx_cap = 1, sample_vx_cap(rng, 1)
    cmd = (0.0, 0.0, 0.0)
    next_update_t = 0.0
    was_stopped = False

    cmds, ref = [], []
    small_steps = stop_steps = 0
    path_len = total_turn = 0.0

    for i in range(n_steps):
        t = round(i * DT, 2)

        # advance waypoint when reached (skip-ahead safe)
        while (wp_idx < len(waypoints)
               and math.dist(state[:2], waypoints[wp_idx]) < WP_REACH_DIST_M):
            wp_idx += 1
            vx_cap = sample_vx_cap(rng, wp_idx)

        stopped = in_stop(t, stop_windows)
        if stopped and not was_stopped:
            cmd = (0.0, 0.0, 0.0)
            cmds.append([t, *cmd])
        elif not stopped and (was_stopped or t >= next_update_t):
            cmd = (p_control(state, waypoints[wp_idx], vx_cap)
                   if wp_idx < len(waypoints) else (0.0, 0.0, 0.0))
            cmds.append([t, *cmd])
            next_update_t = t + rng.uniform(UPDATE_MIN_S, UPDATE_MAX_S)
        was_stopped = stopped

        if i % ref_stride == 0:
            ref.append([t, round(state[0], 4), round(state[1], 4),
                        round(state[2], 4)])

        new_state = integrate_step(state, cmd)
        path_len += math.dist(state[:2], new_state[:2])
        total_turn += abs(cmd[2]) * DT
        small_steps += int(SMALL_VX_BAND[0] <= abs(cmd[0]) <= SMALL_VX_BAND[1])
        stop_steps += int(stopped)
        state = new_state

    ref.append([round(duration_s, 2), round(state[0], 4),
                round(state[1], 4), round(state[2], 4)])

    tape = {"id": tape_id, "dt": DT, "cmds": cmds, "ref_xy_yaw": ref}
    stats = {
        "id": tape_id,
        "n_cmds": len(cmds),
        "small_vx_frac": small_steps / n_steps,
        "stop_time_s": stop_steps * DT,
        "path_len_m": path_len,
        "net_disp_m": math.dist((0.0, 0.0), state[:2]),
        "total_turn_deg": math.degrees(total_turn),
        "final_yaw_deg": math.degrees(state[2]),
    }
    return tape, stats


def validate_tape(tape, duration_s):
    """Hard assertions on the spec: limits, stops, ZOH ordering."""
    cmds, ref = tape["cmds"], tape["ref_xy_yaw"]
    assert cmds[0][0] == 0.0, "first command must start at t=0"
    assert all(c0[0] < c1[0] for c0, c1 in zip(cmds, cmds[1:])), \
        "command breakpoints must be strictly increasing"
    for _, vx, vy, wz in cmds:
        assert abs(vx) <= VX_MAX + 1e-9 and abs(vy) <= VY_MAX + 1e-9 \
            and abs(wz) <= WZ_MAX + 1e-9, "command exceeds limits"
    n_stops = sum(
        1 for prev, cur in zip(cmds, cmds[1:])
        if cur[1:] == [0.0, 0.0, 0.0] and prev[1:] != [0.0, 0.0, 0.0]
        and cur[0] < duration_s - 2.0
    )
    assert n_stops >= 2, f"expected >=2 full-stop segments, got {n_stops}"
    assert len(ref) == int(duration_s * REF_HZ) + 1, "ref must be 1 Hz + endpoint"


MIN_SMALL_VX_FRAC = 0.10  # every tape must spend >=10% time in the small band


def validate_stats(stats):
    assert stats["small_vx_frac"] >= MIN_SMALL_VX_FRAC, (
        f"tape {stats['id']}: small-cmd fraction "
        f"{stats['small_vx_frac']:.2f} < {MIN_SMALL_VX_FRAC}")


def print_stats(all_stats):
    header = (f"{'id':>2} {'n_cmds':>6} {'small_vx%':>9} {'stop_s':>6} "
              f"{'path_m':>7} {'net_m':>6} {'turn_deg':>8} {'yaw_deg':>8}")
    print(header)
    print("-" * len(header))
    for s in all_stats:
        print(f"{s['id']:>2} {s['n_cmds']:>6} {s['small_vx_frac']*100:>8.1f}% "
              f"{s['stop_time_s']:>6.2f} {s['path_len_m']:>7.2f} "
              f"{s['net_disp_m']:>6.2f} {s['total_turn_deg']:>8.1f} "
              f"{s['final_yaw_deg']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = (Path(__file__).resolve().parent.parent
                   / "data" / "vln_tapes.json")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output json path (default: %(default)s)")
    parser.add_argument("--n-tapes", type=int, default=N_TAPES,
                        help="number of tapes, seeds 0..n-1 (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=DURATION_S,
                        help="tape duration in seconds (default: %(default)s)")
    args = parser.parse_args()

    tapes, all_stats = [], []
    for tape_id in range(args.n_tapes):
        tape, stats = generate_tape(tape_id, args.duration)
        validate_tape(tape, args.duration)
        validate_stats(stats)
        tapes.append(tape)
        all_stats.append(stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"tapes": tapes}, separators=(",", ":"))
                        + "\n")

    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, "
          f"{len(tapes)} tapes, {args.duration:.0f}s each)")
    print_stats(all_stats)


if __name__ == "__main__":
    main()
