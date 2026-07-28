#!/usr/bin/env python3
"""Measure robot angular velocity response using odometry or global pose.

This script publishes WirelessController commands using the same sign convention
as ActionExecutorClient.rotate(): positive yaw command publishes negative rx.
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg._wireless_controller import WirelessController


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def linear_slope(samples):
    if len(samples) < 2:
        return None
    t0 = samples[0][0]
    xs = [t - t0 for t, _ in samples]
    ys = [yaw for _, yaw in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


@dataclass
class YawSample:
    t: float
    source: str
    yaw: float
    yaw_unwrapped: float


class AngularResponseTester(Node):
    def __init__(self, args):
        super().__init__("angular_response_tester")
        self.args = args
        self.samples = []
        self._last_yaw_by_source = {}
        self._unwrapped_yaw_by_source = {}

        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Odometry, args.odom_topic, self._odom_callback, qos)
        self.create_subscription(PoseStamped, args.pose_topic, self._pose_callback, qos)
        self.publisher = self.create_publisher(WirelessController, args.command_topic, 10)

    def _record_yaw(self, source, yaw):
        now = time.monotonic()
        previous = self._last_yaw_by_source.get(source)
        if previous is None:
            unwrapped = yaw
        else:
            unwrapped = self._unwrapped_yaw_by_source[source] + normalize_angle(yaw - previous)
        self._last_yaw_by_source[source] = yaw
        self._unwrapped_yaw_by_source[source] = unwrapped
        self.samples.append(YawSample(now, source, yaw, unwrapped))

    def _odom_callback(self, msg):
        self._record_yaw("odom", yaw_from_quaternion(msg.pose.pose.orientation))

    def _pose_callback(self, msg):
        self._record_yaw("pose", yaw_from_quaternion(msg.pose.orientation))

    def source_ready(self):
        return self.args.source in self._unwrapped_yaw_by_source

    def wait_for_source(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.source_ready():
                return True
        return False

    @staticmethod
    def zero_msg():
        msg = WirelessController()
        msg.lx = 0.0
        msg.ly = 0.0
        msg.rx = 0.0
        msg.ry = 0.0
        msg.keys = 0
        return msg

    def command_msg(self, commanded_yaw_rate):
        msg = self.zero_msg()
        msg.rx = -float(commanded_yaw_rate)
        return msg

    def publish_stop(self, repeat=None):
        repeat = self.args.stop_repeat if repeat is None else repeat
        for _ in range(max(1, repeat)):
            self.publisher.publish(self.zero_msg())
            rclpy.spin_once(self, timeout_sec=self.args.stop_interval)

    def samples_between(self, start, end, source=None):
        source = self.args.source if source is None else source
        return [s for s in self.samples if start <= s.t <= end and s.source == source]

    def run_trial(self, trial_index, target_yaw_rate):
        commanded_yaw_rate = target_yaw_rate * self.args.command_scale
        self.publish_stop()
        settle_start = time.monotonic()
        while rclpy.ok() and time.monotonic() - settle_start < self.args.pre_settle:
            self.publisher.publish(self.zero_msg())
            rclpy.spin_once(self, timeout_sec=0.02)

        command_start = time.monotonic()
        command_end = command_start + self.args.duration
        period = 1.0 / self.args.publish_hz
        while rclpy.ok() and time.monotonic() < command_end:
            self.publisher.publish(self.command_msg(commanded_yaw_rate))
            rclpy.spin_once(self, timeout_sec=period)

        stop_time = time.monotonic()
        self.publish_stop()
        settle_end = stop_time + self.args.post_settle
        while rclpy.ok() and time.monotonic() < settle_end:
            self.publisher.publish(self.zero_msg())
            rclpy.spin_once(self, timeout_sec=0.02)

        trial_end = time.monotonic()
        return self.summarize_trial(
            trial_index,
            target_yaw_rate,
            commanded_yaw_rate,
            command_start,
            stop_time,
            trial_end,
        )

    def summarize_trial(
        self,
        trial_index,
        target_yaw_rate,
        commanded_yaw_rate,
        command_start,
        stop_time,
        trial_end,
    ):
        command_samples = self.samples_between(command_start, stop_time)
        settle_samples = self.samples_between(stop_time, trial_end)
        if len(command_samples) < 2:
            return {
                "trial": trial_index,
                "source": self.args.source,
                "target_yaw_rate": target_yaw_rate,
                "commanded_yaw_rate": commanded_yaw_rate,
                "command_scale": self.args.command_scale,
                "raw_rx": -commanded_yaw_rate,
                "error": "not_enough_samples",
            }

        start_yaw = command_samples[0].yaw_unwrapped
        stop_yaw = command_samples[-1].yaw_unwrapped
        actual_angle = stop_yaw - start_yaw

        trim_start = command_start + self.args.fit_ignore_start
        trim_end = stop_time - self.args.fit_ignore_end
        fit_samples = [
            (s.t, s.yaw_unwrapped)
            for s in command_samples
            if trim_start <= s.t <= trim_end
        ]
        if len(fit_samples) < 2:
            fit_samples = [(s.t, s.yaw_unwrapped) for s in command_samples]
        steady_rate = linear_slope(fit_samples)

        startup_delay = None
        threshold = max(0.01, abs(commanded_yaw_rate) * self.args.startup_threshold_ratio)
        rate_window = max(2, int(self.args.startup_window * self.args.sample_rate_hint))
        yaw_pairs = [(s.t, s.yaw_unwrapped) for s in command_samples]
        for index in range(0, max(0, len(yaw_pairs) - rate_window)):
            window = yaw_pairs[index : index + rate_window]
            slope = linear_slope(window)
            if slope is not None and abs(slope) >= threshold:
                startup_delay = window[0][0] - command_start
                break

        residual_angle = 0.0
        residual_rate = None
        if len(settle_samples) >= 2:
            residual_angle = settle_samples[-1].yaw_unwrapped - settle_samples[0].yaw_unwrapped
            residual_rate = linear_slope([(s.t, s.yaw_unwrapped) for s in settle_samples])

        return {
            "trial": trial_index,
            "source": self.args.source,
            "target_yaw_rate": target_yaw_rate,
            "commanded_yaw_rate": commanded_yaw_rate,
            "command_scale": self.args.command_scale,
            "raw_rx": -commanded_yaw_rate,
            "duration": stop_time - command_start,
            "actual_angle": actual_angle,
            "mean_rate": actual_angle / max(1e-6, stop_time - command_start),
            "steady_rate": steady_rate,
            "target_response_ratio": None if steady_rate is None or abs(target_yaw_rate) < 1e-9 else steady_rate / target_yaw_rate,
            "command_response_ratio": None if steady_rate is None or abs(commanded_yaw_rate) < 1e-9 else steady_rate / commanded_yaw_rate,
            "startup_delay": startup_delay,
            "residual_angle": residual_angle,
            "residual_rate": residual_rate,
            "sample_count": len(command_samples),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish angular commands and measure actual yaw response from odom/pose."
    )
    parser.add_argument("--source", choices=("odom", "pose"), default="odom")
    parser.add_argument("--odom-topic", default="/dog_odom")
    parser.add_argument("--pose-topic", default="/global_position")
    parser.add_argument("--command-topic", default="/wirelesscontroller")
    parser.add_argument("--yaw-rate", type=float, default=0.3, help="Target yaw rate in rad/s.")
    parser.add_argument(
        "--command-scale",
        type=float,
        default=1.0,
        help="Multiply the target yaw-rate command before publishing. Example: 1.7 sends 0.51 for a 0.30 target.",
    )
    parser.add_argument("--direction", choices=("positive", "negative", "both"), default="both")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--pre-settle", type=float, default=0.5)
    parser.add_argument("--post-settle", type=float, default=3.5)
    parser.add_argument("--arming-delay", type=float, default=3.0)
    parser.add_argument("--wait-timeout", type=float, default=5.0)
    parser.add_argument("--stop-repeat", type=int, default=10)
    parser.add_argument("--stop-interval", type=float, default=0.02)
    parser.add_argument("--fit-ignore-start", type=float, default=0.8)
    parser.add_argument("--fit-ignore-end", type=float, default=0.15)
    parser.add_argument("--startup-threshold-ratio", type=float, default=0.2)
    parser.add_argument("--startup-window", type=float, default=0.15)
    parser.add_argument("--sample-rate-hint", type=float, default=50.0)
    parser.add_argument(
        "--csv",
        default=f"/tmp/angular_response_{int(time.time())}.csv",
        help="CSV path for raw yaw samples and per-trial summary rows.",
    )
    args = parser.parse_args()
    if args.command_scale <= 0.0:
        parser.error("--command-scale must be > 0")
    return args


def directions_from_args(args):
    rate = abs(args.yaw_rate)
    if args.direction == "positive":
        return [rate]
    if args.direction == "negative":
        return [-rate]
    return [rate, -rate]


def print_summary(summary):
    if "error" in summary:
        print(
            "trial={trial} source={source} target={target_yaw_rate:.3f} "
            "sent={commanded_yaw_rate:.3f} scale={command_scale:.2f} "
            "raw_rx={raw_rx:.3f} error={error}".format(
                **summary
            )
        )
        return
    steady_rate = summary["steady_rate"]
    target_ratio = summary["target_response_ratio"]
    command_ratio = summary["command_response_ratio"]
    residual_rate = summary["residual_rate"]
    startup_delay = summary["startup_delay"]
    print(
        "trial={trial} source={source} target={target_yaw_rate:.3f}rad/s "
        "sent={commanded_yaw_rate:.3f}rad/s scale={command_scale:.2f} raw_rx={raw_rx:.3f} "
        "angle={actual_angle:.3f}rad mean={mean_rate:.3f}rad/s steady={steady}rad/s "
        "target_ratio={target_ratio} cmd_ratio={command_ratio} startup={startup}s "
        "residual={residual_angle:.3f}rad "
        "residual_rate={residual_rate_text}".format(
            steady="n/a" if steady_rate is None else f"{steady_rate:.3f}",
            target_ratio="n/a" if target_ratio is None else f"{target_ratio:.2f}",
            command_ratio="n/a" if command_ratio is None else f"{command_ratio:.2f}",
            startup="n/a" if startup_delay is None else f"{startup_delay:.3f}",
            residual_rate_text="n/a" if residual_rate is None else f"{residual_rate:.3f}",
            **summary,
        )
    )


def write_csv(path, summaries, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["section", "trial", "source", "t", "yaw", "yaw_unwrapped", "key", "value"])
        for sample in samples:
            writer.writerow(["sample", "", sample.source, f"{sample.t:.6f}", f"{sample.yaw:.9f}", f"{sample.yaw_unwrapped:.9f}", "", ""])
        for summary in summaries:
            for key, value in summary.items():
                writer.writerow(["summary", summary.get("trial"), summary.get("source"), "", "", "", key, value])


def main():
    args = parse_args()
    rclpy.init()
    node = AngularResponseTester(args)
    summaries = []
    try:
        print(
            f"Waiting for {args.source} samples "
            f"({args.odom_topic if args.source == 'odom' else args.pose_topic})..."
        )
        if not node.wait_for_source():
            raise RuntimeError(f"No {args.source} samples received within {args.wait_timeout:.1f}s")

        print(
            "Angular response test will publish to "
            f"{args.command_topic} with command_scale={args.command_scale:.2f}. "
            f"Keep the robot clear. Starting in {args.arming_delay:.1f}s."
        )
        end_delay = time.monotonic() + args.arming_delay
        while rclpy.ok() and time.monotonic() < end_delay:
            node.publish_stop(repeat=1)

        trial_index = 0
        for _ in range(max(1, args.repeats)):
            for target_yaw_rate in directions_from_args(args):
                trial_index += 1
                summary = node.run_trial(trial_index, target_yaw_rate)
                summaries.append(summary)
                print_summary(summary)
    finally:
        node.publish_stop()
        write_csv(args.csv, summaries, node.samples)
        print(f"Wrote CSV: {args.csv}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
