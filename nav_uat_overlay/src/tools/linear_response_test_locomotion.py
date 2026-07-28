#!/usr/bin/env python3
"""Measure distance-command scale for the discrete locomotion motion backend."""

import argparse
import csv
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from motion_backend import GrootHttpDiscreteBackend, config_get


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOOLS_DIR.parent / "config.yaml"


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class PositionSample:
    t: float
    source: str
    x: float
    y: float
    yaw: float


class LinearLocomotionResponseTester(Node):
    def __init__(self, args):
        super().__init__("linear_response_tester_locomotion")
        self.args = args
        self.samples = []
        self.latest_by_source = {}
        self._lock = threading.Lock()
        self._spin_stop = threading.Event()
        self._spin_thread = None

        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Odometry, args.odom_topic, self._odom_callback, qos)
        self.create_subscription(PoseStamped, args.pose_topic, self._pose_callback, qos)

    def _record(self, source, position, orientation):
        sample = PositionSample(
            time.monotonic(),
            source,
            float(position.x),
            float(position.y),
            yaw_from_quaternion(orientation),
        )
        with self._lock:
            self.samples.append(sample)
            self.latest_by_source[source] = sample

    def _odom_callback(self, msg):
        self._record("odom", msg.pose.pose.position, msg.pose.pose.orientation)

    def _pose_callback(self, msg):
        self._record("pose", msg.pose.position, msg.pose.orientation)

    def start_spinner(self):
        if self._spin_thread is not None:
            return

        def spin_loop():
            while rclpy.ok() and not self._spin_stop.is_set():
                rclpy.spin_once(self, timeout_sec=0.05)

        self._spin_thread = threading.Thread(target=spin_loop, daemon=True)
        self._spin_thread.start()

    def stop_spinner(self):
        self._spin_stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

    def wait_for_source(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.latest_sample() is not None:
                return True
            time.sleep(0.05)
        return False

    def latest_sample(self):
        with self._lock:
            return self.latest_by_source.get(self.args.source)

    def wait_for_sample_after(self, earliest_time, timeout=None):
        timeout = self.args.wait_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        latest = None
        while rclpy.ok() and time.monotonic() < deadline:
            latest = self.latest_sample()
            if latest is not None and latest.t >= earliest_time:
                return latest
            time.sleep(0.02)
        return latest

    def samples_snapshot(self):
        with self._lock:
            return list(self.samples)

    @staticmethod
    def projected_position(sample, origin):
        dx = sample.x - origin.x
        dy = sample.y - origin.y
        return dx * math.cos(origin.yaw) + dy * math.sin(origin.yaw)

    def run_trial(self, trial_index, motion, target_distance, backend):
        commanded_distance = target_distance * self.args.command_scale
        stop_backend(backend)
        time.sleep(self.args.pre_settle)

        start_sample = self.wait_for_sample_after(time.monotonic() - self.args.max_sample_age)
        if start_sample is None:
            return {
                "trial": trial_index,
                "source": self.args.source,
                "motion": motion,
                "target_distance": target_distance,
                "commanded_distance": commanded_distance,
                "command_scale": self.args.command_scale,
                "error": "no_start_sample",
            }

        command_start = time.monotonic()
        success, message, code = backend.forward(commanded_distance)
        command_end = time.monotonic()

        if self.args.post_settle > 0.0:
            time.sleep(self.args.post_settle)
        end_sample = self.wait_for_sample_after(command_end)
        trial_end = time.monotonic()

        base = {
            "trial": trial_index,
            "source": self.args.source,
            "motion": motion,
            "target_distance": target_distance,
            "commanded_distance": commanded_distance,
            "command_scale": self.args.command_scale,
            "backend_success": success,
            "backend_code": code,
            "backend_message": message,
            "duration": command_end - command_start,
        }
        if end_sample is None:
            return {**base, "error": "no_end_sample"}

        actual_distance = self.projected_position(end_sample, start_sample)
        response_ratio = (
            None if abs(commanded_distance) < 1e-9 else actual_distance / commanded_distance
        )
        measured_scale = (
            None if abs(actual_distance) < self.args.min_actual else commanded_distance / actual_distance
        )
        samples_used = [
            sample for sample in self.samples_snapshot()
            if command_start <= sample.t <= trial_end and sample.source == self.args.source
        ]
        summary = {
            **base,
            "actual_distance": actual_distance,
            "response_ratio": response_ratio,
            "measured_scale": measured_scale,
            "abs_measured_scale": None if measured_scale is None else abs(measured_scale),
            "start_x": start_sample.x,
            "start_y": start_sample.y,
            "start_yaw": start_sample.yaw,
            "end_x": end_sample.x,
            "end_y": end_sample.y,
            "end_yaw": end_sample.yaw,
            "sample_count": len(samples_used),
        }
        if not success:
            summary["error"] = "backend_failed"
        return summary


def load_config(path):
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(path)
    except ImportError:
        try:
            import yaml

            with path.open() as stream:
                return yaml.safe_load(stream) or {}
        except ImportError as exc:
            raise RuntimeError("Install omegaconf or pyyaml to load --config.") from exc


def optional_backend_keys():
    return (
        "ipc_url",
        "box_demo_module_path",
        "http_timeout",
        "initialize_before_motion",
        "fwd_cruise",
        "back_cruise",
        "lat_cruise",
        "yaw_cruise",
        "fwd_max",
        "back_max",
        "lat_max",
        "yaw_max",
        "min_duration",
        "min_distance",
        "warmup_time",
        "warmup_speed",
        "settle_before_s",
        "stop_hold_s",
        "stand_height",
        "walk_min_height",
        "auto_raise_for_walk",
        "dist_gain",
        "verbose",
    )


def build_backend_config(args):
    config = load_config(args.config)
    loaded_backend_cfg = config_get(config, "motion_backend", {})
    backend_cfg = {"type": args.backend_type}
    for key in optional_backend_keys():
        value = getattr(args, key)
        if value is None:
            value = config_get(loaded_backend_cfg, key, None)
        if value is not None:
            backend_cfg[key] = value
    return {"motion_backend": backend_cfg}


def stop_backend(backend):
    try:
        backend.stop()
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send discrete forward/backward distances through motion_backend.py and measure odom scale."
    )
    parser.add_argument("--motion", choices=("forward", "backward", "both"), default="both")
    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        default=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        help="Positive distance magnitudes in meters to test.",
    )
    parser.add_argument("--command-scale", type=float, default=1.0)
    parser.add_argument("--source", choices=("odom", "pose"), default="odom")
    parser.add_argument("--odom-topic", default="/dog_odom")
    parser.add_argument("--pose-topic", default="/global_position")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend-type", default="groot_http_discrete")
    parser.add_argument("--ipc-url", default=None)
    parser.add_argument("--box-demo-module-path", default=None)
    parser.add_argument("--http-timeout", type=float, default=None)
    parser.add_argument("--initialize-before-motion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fwd-cruise", type=float, default=None)
    parser.add_argument("--back-cruise", type=float, default=None)
    parser.add_argument("--lat-cruise", type=float, default=None)
    parser.add_argument("--yaw-cruise", type=float, default=None)
    parser.add_argument("--fwd-max", type=float, default=None)
    parser.add_argument("--back-max", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--yaw-max", type=float, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--min-distance", type=float, default=None)
    parser.add_argument("--warmup-time", type=float, default=None)
    parser.add_argument("--warmup-speed", type=float, default=None)
    parser.add_argument("--settle-before-s", type=float, default=None)
    parser.add_argument("--stop-hold-s", type=float, default=None)
    parser.add_argument("--stand-height", type=float, default=None)
    parser.add_argument("--walk-min-height", type=float, default=None)
    parser.add_argument("--auto-raise-for-walk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dist-gain", type=float, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pre-settle", type=float, default=0.5)
    parser.add_argument("--post-settle", type=float, default=1.0)
    parser.add_argument("--arming-delay", type=float, default=3.0)
    parser.add_argument("--wait-timeout", type=float, default=5.0)
    parser.add_argument("--max-sample-age", type=float, default=0.5)
    parser.add_argument("--min-actual", type=float, default=1e-4)
    parser.add_argument(
        "--csv",
        default=f"/tmp/linear_response_locomotion_{int(time.time())}.csv",
        help="CSV path for raw position samples and per-trial summary rows.",
    )
    args = parser.parse_args()
    if args.command_scale <= 0.0:
        parser.error("--command-scale must be > 0")
    if args.repeats <= 0:
        parser.error("--repeats must be > 0")
    if args.pre_settle < 0.0 or args.post_settle < 0.0 or args.arming_delay < 0.0:
        parser.error("--pre-settle, --post-settle and --arming-delay must be >= 0")
    if not args.distances or any(distance <= 0.0 for distance in args.distances):
        parser.error("--distances must contain positive values")
    return args


def trials_from_args(args):
    motions = ("forward", "backward") if args.motion == "both" else (args.motion,)
    trials = []
    for motion in motions:
        sign = 1.0 if motion == "forward" else -1.0
        for distance in args.distances:
            trials.append((motion, sign * abs(distance)))
    return trials


def value_text(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(summary):
    if "error" in summary and summary["error"] in ("no_start_sample", "no_end_sample"):
        print(
            f"trial={summary['trial']} source={summary['source']} motion={summary['motion']} "
            f"target={summary['target_distance']:.3f}m sent={summary['commanded_distance']:.3f}m "
            f"error={summary['error']}"
        )
        return
    print(
        f"trial={summary['trial']} source={summary['source']} motion={summary['motion']} "
        f"target={summary['target_distance']:.3f}m sent={summary['commanded_distance']:.3f}m "
        f"actual={summary['actual_distance']:.3f}m scale={value_text(summary['measured_scale'], 3)} "
        f"abs_scale={value_text(summary['abs_measured_scale'], 3)} "
        f"response_ratio={value_text(summary['response_ratio'], 3)} "
        f"duration={summary['duration']:.3f}s backend_success={summary['backend_success']} "
        f"samples={summary['sample_count']}"
    )
    if summary.get("error") == "backend_failed":
        print(f"  backend_error={summary['backend_message']}")


def write_csv(path, summaries, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["section", "trial", "source", "t", "x", "y", "yaw", "key", "value"])
        for sample in samples:
            writer.writerow([
                "sample", "", sample.source, f"{sample.t:.6f}", f"{sample.x:.9f}",
                f"{sample.y:.9f}", f"{sample.yaw:.9f}", "", "",
            ])
        for summary in summaries:
            for key, value in summary.items():
                writer.writerow(["summary", summary["trial"], summary["source"], "", "", "", "", key, value])


def main():
    args = parse_args()
    rclpy.init()
    node = LinearLocomotionResponseTester(args)
    backend = None
    summaries = []
    try:
        node.start_spinner()
        topic = args.odom_topic if args.source == "odom" else args.pose_topic
        print(f"Waiting for {args.source} samples ({topic})...")
        if not node.wait_for_source():
            raise RuntimeError(f"No {args.source} samples received within {args.wait_timeout:.1f}s")

        backend = GrootHttpDiscreteBackend(build_backend_config(args), log=node.get_logger())
        if not backend.enabled:
            raise RuntimeError(f"Motion backend is disabled: backend_type={args.backend_type!r}")

        print(
            "Linear locomotion response test will call motion_backend.forward(distance). "
            f"command_scale={args.command_scale:.2f}. Keep the robot clear. "
            f"Starting in {args.arming_delay:.1f}s."
        )
        time.sleep(args.arming_delay)

        trial_index = 0
        for _ in range(args.repeats):
            for motion, target_distance in trials_from_args(args):
                trial_index += 1
                summary = node.run_trial(trial_index, motion, target_distance, backend)
                summaries.append(summary)
                print_summary(summary)
    finally:
        if backend is not None:
            stop_backend(backend)
            backend.shutdown()
        samples = node.samples_snapshot()
        write_csv(args.csv, summaries, samples)
        print(f"Wrote CSV: {args.csv}")
        node.stop_spinner()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
