#!/usr/bin/env python3
"""Build and validate the G1-001 legacy-compatible navigation profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


PROFILE_SCHEMA_VERSION = 3
PROFILE_SOURCE = "start_g1_onboard_runtime.sh"

# These values are the defaults written by the preserved G1-001 launcher
# box_demo_groot/start_g1_onboard_nav.sh. Keep this contract explicit so YAML
# fallback values cannot silently change the deployed navigation behavior.
G001_LEGACY_NAV_DEFAULTS: dict[str, Any] = {
    "motion_profile": "precise",
    "stand_height": 0.76,
    "walk_min_height": 0.72,
    "warmup_enabled": False,
    "warmup_time": 0.0,
    "warmup_speed": 0.15,
    "fwd_max": 0.50,
    "back_max": 0.20,
    "lat_max": 0.30,
    "yaw_max": 0.60,
    "lat_cruise": 0.20,
    "v_floor": 0.12,
    "w_floor": 0.10,
    "waist_to_rl_on_motion": True,
}


class NavProfileError(RuntimeError):
    pass


def _finite_float(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise NavProfileError(f"{name} must be finite")
    return parsed


def build_runtime_profile(
    *,
    motion_bus_socket: str,
    stand_height: float,
    walk_min_height: float,
    fwd_max: float,
    lat_max: float,
    yaw_max: float,
    lat_cruise: float,
    updated_at: float | None = None,
) -> dict[str, Any]:
    """Return the motion-bus profile with legacy G1-001 motion semantics."""
    if not motion_bus_socket:
        raise NavProfileError("motion_bus_socket must not be empty")

    stand_height = _finite_float("stand_height", stand_height)
    walk_min_height = _finite_float("walk_min_height", walk_min_height)
    fwd_max = _finite_float("fwd_max", fwd_max)
    lat_max = _finite_float("lat_max", lat_max)
    yaw_max = _finite_float("yaw_max", yaw_max)
    lat_cruise = _finite_float("lat_cruise", lat_cruise)
    timestamp = (
        time.time() if updated_at is None else _finite_float("updated_at", updated_at)
    )

    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "source": PROFILE_SOURCE,
        "motion_backend": "groot_motion_bus",
        "motion_bus_socket": motion_bus_socket,
        **G001_LEGACY_NAV_DEFAULTS,
        "stand_height": stand_height,
        "walk_min_height": walk_min_height,
        "fwd_max": fwd_max,
        "back_max": min(0.20, fwd_max),
        "lat_max": lat_max,
        "yaw_max": yaw_max,
        "lat_cruise": lat_cruise,
        "updated_at": timestamp,
    }
    return profile


def write_profile(path: str | Path, profile: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".groot_runtime.", suffix=".json", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(profile), stream, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_profile(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NavProfileError(f"cannot read navigation profile {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NavProfileError("navigation profile must be a JSON object")
    return payload


def validate_g001_legacy_profile(
    profile: Mapping[str, Any], *, expected_socket: str | None = None
) -> None:
    expected_metadata = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "source": PROFILE_SOURCE,
        "motion_backend": "groot_motion_bus",
    }
    expected = {**expected_metadata, **G001_LEGACY_NAV_DEFAULTS}
    if expected_socket is not None:
        expected["motion_bus_socket"] = expected_socket

    mismatches = []
    for key, wanted in expected.items():
        if key not in profile:
            mismatches.append(f"{key}=<missing> (expected {wanted!r})")
        elif isinstance(wanted, bool) and profile[key] is not wanted:
            mismatches.append(f"{key}={profile[key]!r} (expected {wanted!r})")
        elif not isinstance(wanted, bool) and profile[key] != wanted:
            mismatches.append(f"{key}={profile[key]!r} (expected {wanted!r})")

    updated_at = profile.get("updated_at")
    if not isinstance(updated_at, (int, float)) or not math.isfinite(float(updated_at)):
        mismatches.append("updated_at must be a finite number")
    if mismatches:
        raise NavProfileError("G1-001 navigation profile mismatch: " + "; ".join(mismatches))


def _print_contract() -> None:
    for key, value in G001_LEGACY_NAV_DEFAULTS.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        print(f"nav_{key}={rendered}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write")
    write.add_argument("--output", required=True, type=Path)
    write.add_argument("--motion-bus-socket", required=True)
    write.add_argument("--stand-height", required=True, type=float)
    write.add_argument("--walk-min-height", required=True, type=float)
    write.add_argument("--fwd-max", required=True, type=float)
    write.add_argument("--lat-max", required=True, type=float)
    write.add_argument("--yaw-max", required=True, type=float)
    write.add_argument("--lat-cruise", required=True, type=float)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--expected-socket")

    subparsers.add_parser("show-contract")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "show-contract":
        _print_contract()
        return 0
    if args.command == "write":
        profile = build_runtime_profile(
            motion_bus_socket=args.motion_bus_socket,
            stand_height=args.stand_height,
            walk_min_height=args.walk_min_height,
            fwd_max=args.fwd_max,
            lat_max=args.lat_max,
            yaw_max=args.yaw_max,
            lat_cruise=args.lat_cruise,
        )
        write_profile(args.output, profile)
        return 0

    profile = read_profile(args.input)
    validate_g001_legacy_profile(profile, expected_socket=args.expected_socket)
    print("G1-001 navigation profile matches preserved legacy defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
