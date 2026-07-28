#!/usr/bin/env python3
"""Manual smoke test for GR00T continuous velocity control.

This script sends short GR00T HTTP velocity holds through
Node.motion_backend.GrootHttpDiscreteBackend.publish_velocity(). It is intended
for checking whether forward and yaw can be commanded together before running
the full navigation stack.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import time
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TOOLS_DIR.parent
DEFAULT_CONFIG = SRC_ROOT / "config.yaml"

def load_node_motion_backend():
    module_path = SRC_ROOT / "Node" / "motion_backend.py"
    spec = importlib.util.spec_from_file_location("nav_node_motion_backend", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load motion backend module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_motion_backend_module = load_node_motion_backend()
GrootHttpDiscreteBackend = _motion_backend_module.GrootHttpDiscreteBackend
config_get = _motion_backend_module.config_get


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
        "stand_height",
        "fwd_cruise",
        "back_cruise",
        "lat_cruise",
        "yaw_cruise",
        "fwd_max",
        "back_max",
        "lat_max",
        "yaw_max",
        "v_floor",
        "w_floor",
        "min_duration",
        "min_distance",
        "warmup_time",
        "warmup_speed",
        "settle_before_s",
        "stop_hold_s",
        "walk_min_height",
        "auto_raise_for_walk",
        "dist_gain",
        "verbose",
        "continuous_velocity_enable",
        "continuous_command_interval",
        "continuous_hold_duration",
        "continuous_command_epsilon",
        "continuous_fsm",
    )


def build_backend_config(args):
    config = load_config(args.config)
    loaded_backend_cfg = config_get(config, "motion_backend", {})
    backend_cfg = {"type": args.backend_type}
    for key in optional_backend_keys():
        cli_name = key.replace("_", "-")
        value = getattr(args, key, None)
        if value is None:
            value = config_get(loaded_backend_cfg, key, None)
        if value is not None:
            backend_cfg[key] = value
        elif args.print_missing_config:
            print(f"missing config/CLI value: --{cli_name}")
    return {"motion_backend": backend_cfg}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Send a continuous GR00T velocity command through the nav motion backend. "
            "Use --dry-run to inspect the resolved command without moving."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend-type", default="groot_http_discrete")
    parser.add_argument("--forward", type=float, default=0.12, help="Forward velocity vx in m/s.")
    parser.add_argument("--lateral", type=float, default=0.0, help="Lateral velocity vy in m/s.")
    parser.add_argument("--yaw", type=float, default=0.15, help="Yaw velocity wz in rad/s.")
    parser.add_argument("--duration", type=float, default=2.0, help="Command duration in seconds.")
    parser.add_argument("--rate", type=float, default=10.0, help="publish_velocity call rate in Hz.")
    parser.add_argument("--pre-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for real motion. Confirms the robot is clear to move.",
    )
    parser.add_argument("--ipc-url", default=None)
    parser.add_argument("--box-demo-module-path", default=None)
    parser.add_argument("--http-timeout", type=float, default=None)
    parser.add_argument("--initialize-before-motion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stand-height", type=float, default=None)
    parser.add_argument("--fwd-cruise", type=float, default=None)
    parser.add_argument("--back-cruise", type=float, default=None)
    parser.add_argument("--lat-cruise", type=float, default=None)
    parser.add_argument("--yaw-cruise", type=float, default=None)
    parser.add_argument("--fwd-max", type=float, default=None)
    parser.add_argument("--back-max", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--yaw-max", type=float, default=None)
    parser.add_argument("--v-floor", type=float, default=None)
    parser.add_argument("--w-floor", type=float, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--min-distance", type=float, default=None)
    parser.add_argument("--warmup-time", type=float, default=None)
    parser.add_argument("--warmup-speed", type=float, default=None)
    parser.add_argument("--settle-before-s", type=float, default=None)
    parser.add_argument("--stop-hold-s", type=float, default=None)
    parser.add_argument("--walk-min-height", type=float, default=None)
    parser.add_argument("--auto-raise-for-walk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dist-gain", type=float, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continuous-velocity-enable", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continuous-command-interval", type=float, default=None)
    parser.add_argument("--continuous-hold-duration", type=float, default=None)
    parser.add_argument("--continuous-command-epsilon", type=float, default=None)
    parser.add_argument("--continuous-fsm", default=None)
    parser.add_argument("--print-missing-config", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be > 0")
    if args.rate <= 0.0:
        parser.error("--rate must be > 0")
    if not args.dry_run and not args.yes:
        parser.error("real motion requires --yes")
    return args


def print_backend_config(cfg):
    backend_cfg = cfg["motion_backend"]
    print("resolved motion_backend:")
    for key in sorted(backend_cfg):
        print(f"  {key}: {backend_cfg[key]}")


def run_dry_run(args, backend_cfg):
    print_backend_config(backend_cfg)
    print(
        "dry-run command: "
        f"forward={args.forward:+.3f} m/s lateral={args.lateral:+.3f} m/s "
        f"yaw={args.yaw:+.3f} rad/s duration={args.duration:.2f}s rate={args.rate:.1f}Hz"
    )
    print("dry-run only; no HTTP command was sent.")


def stop_backend(backend, label):
    success, message, code = backend.stop()
    print(f"{label}: success={success} code={code} message={message}")
    return success


def run_motion(args, backend_cfg):
    backend = GrootHttpDiscreteBackend(backend_cfg, log=logging.getLogger("continuous_test"))
    if not backend.enabled:
        raise RuntimeError(
            f"backend is not enabled; resolved type={backend.backend_type!r}, "
            "expected 'groot_http_discrete'"
        )
    if not backend.supports_continuous_velocity():
        raise RuntimeError("backend does not support continuous velocity")

    print_backend_config(backend_cfg)
    print(
        "sending continuous command: "
        f"forward={args.forward:+.3f} m/s lateral={args.lateral:+.3f} m/s "
        f"yaw={args.yaw:+.3f} rad/s duration={args.duration:.2f}s rate={args.rate:.1f}Hz"
    )

    if args.pre_stop:
        stop_backend(backend, "pre-stop")
        time.sleep(0.2)

    period = 1.0 / args.rate
    deadline = time.monotonic() + args.duration
    sent = 0
    failures = 0
    last_message = None

    try:
        while time.monotonic() < deadline:
            success, message, code = backend.publish_velocity(
                forward=args.forward,
                lateral=args.lateral,
                yaw=args.yaw,
            )
            sent += 1
            last_message = message
            print(
                f"tick={sent:03d} success={success} code={code} message={message}",
                flush=True,
            )
            if not success:
                failures += 1
            time.sleep(period)
    except KeyboardInterrupt:
        print("interrupted; stopping backend")
    finally:
        if args.post_stop:
            stop_backend(backend, "post-stop")

    print(
        f"summary: calls={sent} failures={failures} last_message={last_message!r} "
        f"duration={args.duration:.2f}s"
    )
    return 1 if failures else 0


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backend_cfg = build_backend_config(args)
    if args.dry_run:
        run_dry_run(args, backend_cfg)
        return 0
    return run_motion(args, backend_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
