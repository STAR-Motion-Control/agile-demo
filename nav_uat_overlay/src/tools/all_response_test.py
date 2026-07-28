#!/usr/bin/env python3
"""Run configured forward/backward/left/right/yaw response tests.

This is a safe orchestration entry point around linear_response_test.py and
angular_response_test.py. Directional command scales and command limits are
loaded from config.yaml before any command is published.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOOLS_DIR.parent / "config.yaml"


def positive_float(value, name):
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def config_value(config, name):
    try:
        return positive_float(config.motion_control.closed_loop[name], name)
    except Exception as exc:
        raise ValueError(f"Missing or invalid motion_control.closed_loop.{name}") from exc


def effective_scale(target_magnitude, configured_scale, command_limit):
    requested_command = target_magnitude * configured_scale
    sent_command = min(requested_command, command_limit)
    return sent_command / target_magnitude, requested_command, sent_command


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read config.yaml and test forward/backward/lateral/angular response."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--linear-speed", type=float, default=0.3, help="Target linear speed in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.3, help="Target yaw rate in rad/s.")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fit-ignore-start", type=float, default=0.8)
    parser.add_argument("--linear-post-settle", type=float, default=3.0)
    parser.add_argument("--angular-post-settle", type=float, default=3.0)
    parser.add_argument("--source", choices=("odom", "pose"), default="odom")
    parser.add_argument("--odom-topic", default="/dog_odom")
    parser.add_argument("--pose-topic", default="/global_position")
    parser.add_argument("--command-topic", default="/wirelesscontroller")
    parser.add_argument("--dry-run", action="store_true", help="Print configured commands without moving the robot.")
    parser.add_argument(
        "--motions",
        nargs="+",
        choices=("forward", "backward", "left", "right", "angular"),
        default=("forward", "backward", "left", "right", "angular"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"/tmp/all_response_{int(time.time())}"),
    )
    args = parser.parse_args()
    if args.linear_speed <= 0.0 or args.yaw_rate <= 0.0:
        parser.error("--linear-speed and --yaw-rate must be > 0")
    if args.duration <= 0.0 or args.repeats <= 0:
        parser.error("--duration and --repeats must be > 0")
    return args


def run_command(command):
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Response test failed with exit code {completed.returncode}")


def common_args(args, csv_path):
    return [
        "--source", args.source,
        "--odom-topic", args.odom_topic,
        "--pose-topic", args.pose_topic,
        "--command-topic", args.command_topic,
        "--duration", str(args.duration),
        "--repeats", str(args.repeats),
        "--post-settle", str(args.linear_post_settle),
        "--fit-ignore-start", str(args.fit_ignore_start),
        "--csv", str(csv_path),
    ]


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)

    scales = {
        "forward": config_value(config, "forward_command_scale"),
        "backward": config_value(config, "backward_command_scale"),
        "left": config_value(config, "left_command_scale"),
        "right": config_value(config, "right_command_scale"),
        "angular_positive": config_value(config, "positive_angular_command_scale"),
        "angular_negative": config_value(config, "negative_angular_command_scale"),
    }
    limits = {
        "forward": config_value(config, "max_forward_speed"),
        "backward": config_value(config, "max_forward_speed"),
        "left": config_value(config, "max_lateral_command"),
        "right": config_value(config, "max_lateral_command"),
        "angular": config_value(config, "max_angular_speed"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded config: {config_path}")
    print(f"Output directory: {args.output_dir}")
    print("The robot will move in every selected direction. Keep the full area clear.")

    for motion in args.motions:
        if motion == "angular":
            for direction, scale_key in (
                ("positive", "angular_positive"),
                ("negative", "angular_negative"),
            ):
                scale, requested, sent = effective_scale(
                    args.yaw_rate, scales[scale_key], limits["angular"]
                )
                limited = requested > sent + 1e-9
                print(
                    f"\n[angular/{direction}] configured_scale={scales[scale_key]:.3f} "
                    f"limit={limits['angular']:.3f} requested_raw={requested:.3f} "
                    f"sent_raw={sent:.3f}{' (limited)' if limited else ''}",
                    flush=True,
                )
                csv_path = args.output_dir / f"angular_{direction}.csv"
                command = [
                    sys.executable,
                    str(TOOLS_DIR / "angular_response_test.py"),
                    "--yaw-rate", str(args.yaw_rate),
                    "--command-scale", str(scale),
                    "--direction", direction,
                    "--source", args.source,
                    "--odom-topic", args.odom_topic,
                    "--pose-topic", args.pose_topic,
                    "--command-topic", args.command_topic,
                    "--duration", str(args.duration),
                    "--repeats", str(args.repeats),
                    "--post-settle", str(args.angular_post_settle),
                    "--fit-ignore-start", str(args.fit_ignore_start),
                    "--csv", str(csv_path),
                ]
                if not args.dry_run:
                    run_command(command)
            continue

        target = args.linear_speed
        scale, requested, sent = effective_scale(target, scales[motion], limits[motion])
        limited = requested > sent + 1e-9
        print(
            f"\n[{motion}] configured_scale={scales[motion]:.3f} "
            f"limit={limits[motion]:.3f} requested_raw={requested:.3f} "
            f"sent_raw={sent:.3f}{' (limited)' if limited else ''}",
            flush=True,
        )

        csv_path = args.output_dir / f"{motion}.csv"
        command = [
            sys.executable,
            str(TOOLS_DIR / "linear_response_test.py"),
            "--motion", motion,
            "--speed", str(args.linear_speed),
            "--command-scale", str(scale),
            *common_args(args, csv_path),
        ]
        if not args.dry_run:
            run_command(command)

    if args.dry_run:
        print("\nDry run completed; no commands were published.")
    else:
        print(f"\nAll selected response tests completed. CSV directory: {args.output_dir}")


if __name__ == "__main__":
    main()
