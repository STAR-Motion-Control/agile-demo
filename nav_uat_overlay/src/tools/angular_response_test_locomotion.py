#!/usr/bin/env python3
"""Measure angle-command scale for the discrete locomotion motion backend."""

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


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class YawSample:
    t: float
    source: str
    yaw: float
    yaw_unwrapped: float


class AngularLocomotionResponseTester(Node):
    def __init__(self, args):
        super().__init__("angular_response_tester_locomotion")
        self.args = args
        self.samples = []
        self._last_yaw_by_source = {}
        self._unwrapped_yaw_by_source = {}
        self._latest_by_source = {}
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

    def _record_yaw(self, source, yaw):
        now = time.monotonic()
        with self._lock:
            previous = self._last_yaw_by_source.get(source)
            if previous is None:
                unwrapped = yaw
            else:
                unwrapped = self._unwrapped_yaw_by_source[source] + normalize_angle(yaw - previous)
            self._last_yaw_by_source[source] = yaw
            self._unwrapped_yaw_by_source[source] = unwrapped
            sample = YawSample(now, source, yaw, unwrapped)
            self.samples.append(sample)
            self._latest_by_source[source] = sample

    def _odom_callback(self, msg):
        self._record_yaw("odom", yaw_from_quaternion(msg.pose.pose.orientation))

    def _pose_callback(self, msg):
        self._record_yaw("pose", yaw_from_quaternion(msg.pose.orientation))

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

    def latest_sample(self):
        with self._lock:
            return self._latest_by_source.get(self.args.source)

    def wait_for_source(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.latest_sample() is not None:
                return True
            time.sleep(0.05)
        return False

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

    def run_trial(self, trial_index, target_angle, backend):
        commanded_angle = target_angle * self.args.command_scale
        stop_backend(backend)
        time.sleep(self.args.pre_settle)

        start_sample = self.wait_for_sample_after(time.monotonic() - self.args.max_sample_age)
        if start_sample is None:
            return {
                "trial": trial_index,
                "source": self.args.source,
                "target_angle": target_angle,
                "target_angle_deg": math.degrees(target_angle),
                "commanded_angle": commanded_angle,
                "commanded_angle_deg": math.degrees(commanded_angle),
                "command_scale": self.args.command_scale,
                "error": "no_start_sample",
            }

        command_start = time.monotonic()
        success, message, code = backend.rotate(commanded_angle)
        command_end = time.monotonic()

        if self.args.post_settle > 0.0:
            time.sleep(self.args.post_settle)
        end_sample = self.wait_for_sample_after(command_end)
        trial_end = time.monotonic()

        base = {
            "trial": trial_index,
            "source": self.args.source,
            "target_angle": target_angle,
            "target_angle_deg": math.degrees(target_angle),
            "commanded_angle": commanded_angle,
            "commanded_angle_deg": math.degrees(commanded_angle),
            "command_scale": self.args.command_scale,
            "backend_success": success,
            "backend_code": code,
            "backend_message": message,
            "duration": command_end - command_start,
        }
        if end_sample is None:
            return {**base, "error": "no_end_sample"}

        actual_angle = end_sample.yaw_unwrapped - start_sample.yaw_unwrapped
        response_ratio = None if abs(commanded_angle) < 1e-9 else actual_angle / commanded_angle
        measured_scale = None if abs(actual_angle) < self.args.min_actual else commanded_angle / actual_angle
        samples_used = [
            sample for sample in self.samples_snapshot()
            if command_start <= sample.t <= trial_end and sample.source == self.args.source
        ]
        summary = {
            **base,
            "actual_angle": actual_angle,
            "actual_angle_deg": math.degrees(actual_angle),
            "response_ratio": response_ratio,
            "measured_scale": measured_scale,
            "abs_measured_scale": None if measured_scale is None else abs(measured_scale),
            "start_yaw": start_sample.yaw,
            "start_yaw_unwrapped": start_sample.yaw_unwrapped,
            "end_yaw": end_sample.yaw,
            "end_yaw_unwrapped": end_sample.yaw_unwrapped,
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
        description="Send discrete rotate angles through motion_backend.py and measure odom scale."
    )
    parser.add_argument("--direction", choices=("positive", "negative", "both"), default="both")
    parser.add_argument(
        "--angles-deg",
        nargs="+",
        type=float,
        default=(15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0),
        help="Positive angle magnitudes in degrees to test.",
    )
    parser.add_argument(
        "--angles-rad",
        nargs="+",
        type=float,
        default=None,
        help="Positive angle magnitudes in radians. Overrides --angles-deg when set.",
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
    parser.add_argument("--pre-settle", type=float, default=2)
    parser.add_argument("--post-settle", type=float, default=2)
    parser.add_argument("--arming-delay", type=float, default=3.0)
    parser.add_argument("--wait-timeout", type=float, default=5.0)
    parser.add_argument("--max-sample-age", type=float, default=0.5)
    parser.add_argument("--min-actual", type=float, default=1e-4)
    parser.add_argument(
        "--csv",
        default=f"/tmp/angular_response_locomotion_{int(time.time())}.csv",
        help="CSV path for raw yaw samples and per-trial summary rows.",
    )
    args = parser.parse_args()
    if args.command_scale <= 0.0:
        parser.error("--command-scale must be > 0")
    if args.repeats <= 0:
        parser.error("--repeats must be > 0")
    if args.pre_settle < 0.0 or args.post_settle < 0.0 or args.arming_delay < 0.0:
        parser.error("--pre-settle, --post-settle and --arming-delay must be >= 0")
    angle_values = args.angles_rad if args.angles_rad is not None else args.angles_deg
    if not angle_values or any(angle <= 0.0 for angle in angle_values):
        parser.error("--angles-deg/--angles-rad must contain positive values")
    return args


def angles_from_args(args):
    if args.angles_rad is not None:
        magnitudes = [abs(angle) for angle in args.angles_rad]
    else:
        magnitudes = [math.radians(abs(angle)) for angle in args.angles_deg]
    if args.direction == "positive":
        signs = (1.0,)
    elif args.direction == "negative":
        signs = (-1.0,)
    else:
        signs = (1.0, -1.0)
    return [sign * magnitude for sign in signs for magnitude in magnitudes]


def value_text(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(summary):
    if "error" in summary and summary["error"] in ("no_start_sample", "no_end_sample"):
        print(
            f"trial={summary['trial']} source={summary['source']} "
            f"target={summary['target_angle']:.3f}rad sent={summary['commanded_angle']:.3f}rad "
            f"error={summary['error']}"
        )
        return
    print(
        f"trial={summary['trial']} source={summary['source']} "
        f"target={summary['target_angle']:.3f}rad({summary['target_angle_deg']:.1f}deg) "
        f"sent={summary['commanded_angle']:.3f}rad({summary['commanded_angle_deg']:.1f}deg) "
        f"actual={summary['actual_angle']:.3f}rad({summary['actual_angle_deg']:.1f}deg) "
        f"scale={value_text(summary['measured_scale'], 3)} "
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
        writer.writerow(["section", "trial", "source", "t", "yaw", "yaw_unwrapped", "key", "value"])
        for sample in samples:
            writer.writerow([
                "sample", "", sample.source, f"{sample.t:.6f}",
                f"{sample.yaw:.9f}", f"{sample.yaw_unwrapped:.9f}", "", "",
            ])
        for summary in summaries:
            for key, value in summary.items():
                writer.writerow(["summary", summary["trial"], summary["source"], "", "", "", key, value])


def main():
    args = parse_args()
    rclpy.init()
    node = AngularLocomotionResponseTester(args)
    backend = None
    summaries = []
    try:
        node.start_spinner()
        topic = args.odom_topic if args.source == "odom" else args.pose_topic
        print(f"Waiting for {args.source} samples ({topic})...")
        if not node.wait_for_source():
            raise RuntimeError(f"No {args.source} samples received within {args.wait_timeout:.1f}s")

        backend = GrootHttpDiscreteBackend(build_backend_config(args), log=None)
        if not backend.enabled:
            raise RuntimeError(f"Motion backend is disabled: backend_type={args.backend_type!r}")

        print(
            "Angular locomotion response test will call motion_backend.rotate(angle). "
            f"command_scale={args.command_scale:.2f}. Keep the robot clear. "
            f"Starting in {args.arming_delay:.1f}s."
        )
        time.sleep(args.arming_delay)

        trial_index = 0
        for _ in range(args.repeats):
            for target_angle in angles_from_args(args):
                trial_index += 1
                summary = node.run_trial(trial_index, target_angle, backend)
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
