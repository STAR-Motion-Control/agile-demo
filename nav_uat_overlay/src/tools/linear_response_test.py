#!/usr/bin/env python3
"""Measure forward/backward/lateral velocity response from odometry or pose.

The WirelessController convention matches ActionExecutorClient:
positive ly moves forward and negative lx moves left.
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


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def linear_slope(samples):
    if len(samples) < 2:
        return None
    t0 = samples[0][0]
    xs = [t - t0 for t, _ in samples]
    ys = [value for _, value in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 1e-12:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


@dataclass
class PositionSample:
    t: float
    source: str
    x: float
    y: float
    yaw: float


class LinearResponseTester(Node):
    def __init__(self, args):
        super().__init__("linear_response_tester")
        self.args = args
        self.samples = []
        self.latest_by_source = {}
        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Odometry, args.odom_topic, self._odom_callback, qos)
        self.create_subscription(PoseStamped, args.pose_topic, self._pose_callback, qos)
        self.publisher = self.create_publisher(WirelessController, args.command_topic, 10)

    def _record(self, source, position, orientation):
        sample = PositionSample(
            time.monotonic(),
            source,
            float(position.x),
            float(position.y),
            yaw_from_quaternion(orientation),
        )
        self.samples.append(sample)
        self.latest_by_source[source] = sample

    def _odom_callback(self, msg):
        self._record("odom", msg.pose.pose.position, msg.pose.pose.orientation)

    def _pose_callback(self, msg):
        self._record("pose", msg.pose.position, msg.pose.orientation)

    def wait_for_source(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.args.source in self.latest_by_source:
                return True
        return False

    @staticmethod
    def zero_msg():
        msg = WirelessController()
        msg.lx = msg.ly = msg.rx = msg.ry = 0.0
        msg.keys = 0
        return msg

    @staticmethod
    def command_msg(motion, command):
        msg = LinearResponseTester.zero_msg()
        if motion in ("forward", "backward"):
            msg.ly = float(command)
        else:
            # ActionExecutorClient.shift(): positive/left distance uses negative lx.
            msg.lx = -float(command)
        return msg

    def publish_stop(self, repeat=None):
        repeat = self.args.stop_repeat if repeat is None else repeat
        for _ in range(max(1, repeat)):
            self.publisher.publish(self.zero_msg())
            rclpy.spin_once(self, timeout_sec=self.args.stop_interval)

    def samples_between(self, start, end):
        return [
            sample for sample in self.samples
            if start <= sample.t <= end and sample.source == self.args.source
        ]

    @staticmethod
    def projected_position(sample, origin, lateral):
        dx = sample.x - origin.x
        dy = sample.y - origin.y
        if lateral:
            return -dx * math.sin(origin.yaw) + dy * math.cos(origin.yaw)
        return dx * math.cos(origin.yaw) + dy * math.sin(origin.yaw)

    def run_trial(self, trial_index, motion, target_velocity):
        commanded_velocity = target_velocity * self.args.command_scale
        self.publish_stop()
        settle_start = time.monotonic()
        while rclpy.ok() and time.monotonic() - settle_start < self.args.pre_settle:
            self.publisher.publish(self.zero_msg())
            rclpy.spin_once(self, timeout_sec=0.02)

        command_start = time.monotonic()
        command_end = command_start + self.args.duration
        period = 1.0 / self.args.publish_hz
        while rclpy.ok() and time.monotonic() < command_end:
            self.publisher.publish(self.command_msg(motion, commanded_velocity))
            rclpy.spin_once(self, timeout_sec=period)

        stop_time = time.monotonic()
        self.publish_stop()
        settle_end = stop_time + self.args.post_settle
        while rclpy.ok() and time.monotonic() < settle_end:
            self.publisher.publish(self.zero_msg())
            rclpy.spin_once(self, timeout_sec=0.02)

        return self.summarize_trial(
            trial_index, motion, target_velocity, commanded_velocity,
            command_start, stop_time, time.monotonic(),
        )

    def summarize_trial(
        self, trial_index, motion, target_velocity, commanded_velocity,
        command_start, stop_time, trial_end,
    ):
        command_samples = self.samples_between(command_start, stop_time)
        settle_samples = self.samples_between(stop_time, trial_end)
        raw_lx = -commanded_velocity if motion in ("left", "right") else 0.0
        raw_ly = commanded_velocity if motion in ("forward", "backward") else 0.0
        base = {
            "trial": trial_index,
            "source": self.args.source,
            "motion": motion,
            "target_velocity": target_velocity,
            "commanded_velocity": commanded_velocity,
            "command_scale": self.args.command_scale,
            "raw_lx": raw_lx,
            "raw_ly": raw_ly,
        }
        if len(command_samples) < 2:
            return {**base, "error": "not_enough_samples"}

        origin = command_samples[0]
        lateral = motion in ("left", "right")
        projected = [
            (sample.t, self.projected_position(sample, origin, lateral))
            for sample in command_samples
        ]
        displacement = projected[-1][1] - projected[0][1]
        trim_start = command_start + self.args.fit_ignore_start
        trim_end = stop_time - self.args.fit_ignore_end
        fit_samples = [(t, value) for t, value in projected if trim_start <= t <= trim_end]
        if len(fit_samples) < 2:
            fit_samples = projected
        steady_velocity = linear_slope(fit_samples)

        startup_delay = None
        threshold = max(0.01, abs(commanded_velocity) * self.args.startup_threshold_ratio)
        window_size = max(2, int(self.args.startup_window * self.args.sample_rate_hint))
        for index in range(max(0, len(projected) - window_size)):
            window = projected[index:index + window_size]
            slope = linear_slope(window)
            if slope is not None and slope * target_velocity > 0 and abs(slope) >= threshold:
                startup_delay = window[0][0] - command_start
                break

        residual_displacement = 0.0
        residual_velocity = None
        if len(settle_samples) >= 2:
            settle_projected = [
                (sample.t, self.projected_position(sample, origin, lateral))
                for sample in settle_samples
            ]
            residual_displacement = settle_projected[-1][1] - settle_projected[0][1]
            residual_velocity = linear_slope(settle_projected)

        duration = stop_time - command_start
        return {
            **base,
            "duration": duration,
            "displacement": displacement,
            "mean_velocity": displacement / max(1e-6, duration),
            "steady_velocity": steady_velocity,
            "target_response_ratio": (
                None if steady_velocity is None or abs(target_velocity) < 1e-9
                else steady_velocity / target_velocity
            ),
            "command_response_ratio": (
                None if steady_velocity is None or abs(commanded_velocity) < 1e-9
                else steady_velocity / commanded_velocity
            ),
            "startup_delay": startup_delay,
            "residual_displacement": residual_displacement,
            "residual_velocity": residual_velocity,
            "sample_count": len(command_samples),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish linear WirelessController commands and measure odometry response."
    )
    parser.add_argument("--motion", choices=("forward", "backward", "left", "right", "all"), default="all")
    parser.add_argument("--source", choices=("odom", "pose"), default="odom")
    parser.add_argument("--odom-topic", default="/dog_odom")
    parser.add_argument("--pose-topic", default="/global_position")
    parser.add_argument("--command-topic", default="/wirelesscontroller")
    parser.add_argument("--speed", type=float, default=0.3, help="Target speed magnitude in m/s.")
    parser.add_argument("--command-scale", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--pre-settle", type=float, default=0.5)
    parser.add_argument("--post-settle", type=float, default=1.0)
    parser.add_argument("--arming-delay", type=float, default=3.0)
    parser.add_argument("--wait-timeout", type=float, default=5.0)
    parser.add_argument("--stop-repeat", type=int, default=10)
    parser.add_argument("--stop-interval", type=float, default=0.02)
    parser.add_argument("--fit-ignore-start", type=float, default=0.25)
    parser.add_argument("--fit-ignore-end", type=float, default=0.15)
    parser.add_argument("--startup-threshold-ratio", type=float, default=0.2)
    parser.add_argument("--startup-window", type=float, default=0.15)
    parser.add_argument("--sample-rate-hint", type=float, default=50.0)
    parser.add_argument("--csv", default=f"/tmp/linear_response_{int(time.time())}.csv")
    args = parser.parse_args()
    if args.speed <= 0 or args.command_scale <= 0 or args.publish_hz <= 0:
        parser.error("--speed, --command-scale and --publish-hz must be > 0")
    return args


def trials_from_args(args):
    speed = abs(args.speed)
    motions = ("forward", "backward", "left", "right") if args.motion == "all" else (args.motion,)
    return [(motion, speed if motion in ("forward", "left") else -speed) for motion in motions]


def value_text(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def unit_value_text(value, unit, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def print_summary(summary):
    if "error" in summary:
        print(
            f"trial={summary['trial']} motion={summary['motion']} "
            f"target={summary['target_velocity']:.3f}m/s error={summary['error']}"
        )
        return
    print(
        f"trial={summary['trial']} source={summary['source']} motion={summary['motion']} "
        f"target={summary['target_velocity']:.3f}m/s sent={summary['commanded_velocity']:.3f}m/s "
        f"scale={summary['command_scale']:.2f} raw_lx={summary['raw_lx']:.3f} raw_ly={summary['raw_ly']:.3f} "
        f"distance={summary['displacement']:.3f}m mean={summary['mean_velocity']:.3f}m/s "
        f"steady={value_text(summary['steady_velocity'])}m/s "
        f"target_ratio={value_text(summary['target_response_ratio'], 2)} "
        f"cmd_ratio={value_text(summary['command_response_ratio'], 2)} "
        f"startup={unit_value_text(summary['startup_delay'], 's')} "
        f"residual={summary['residual_displacement']:.3f}m "
        f"residual_rate={unit_value_text(summary['residual_velocity'], 'm/s')}"
    )


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
    node = LinearResponseTester(args)
    summaries = []
    try:
        topic = args.odom_topic if args.source == "odom" else args.pose_topic
        print(f"Waiting for {args.source} samples ({topic})...")
        if not node.wait_for_source():
            raise RuntimeError(f"No {args.source} samples received within {args.wait_timeout:.1f}s")
        print(
            f"Linear response test will publish to {args.command_topic} with "
            f"command_scale={args.command_scale:.2f}. Keep the robot clear. "
            f"Starting in {args.arming_delay:.1f}s."
        )
        deadline = time.monotonic() + args.arming_delay
        while rclpy.ok() and time.monotonic() < deadline:
            node.publish_stop(repeat=1)
        trial_index = 0
        for _ in range(max(1, args.repeats)):
            for motion, target_velocity in trials_from_args(args):
                trial_index += 1
                summary = node.run_trial(trial_index, motion, target_velocity)
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
