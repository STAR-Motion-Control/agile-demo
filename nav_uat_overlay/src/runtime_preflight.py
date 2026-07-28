"""Validate that navigation is attaching to the expected stream-only B runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import stat
import time
from collections.abc import Mapping, Sequence
from typing import Any


class RuntimePreflightError(RuntimeError):
    pass


def _identity_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimePreflightError(f"motion bus status has invalid {field}")
    return os.path.realpath(os.path.abspath(os.path.expanduser(value)))


def _require_unix_socket(path: str, field: str) -> None:
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError as exc:
        raise RuntimePreflightError(f"{field} is missing: {path}") from exc
    if not stat.S_ISSOCK(mode):
        raise RuntimePreflightError(f"{field} is not a Unix socket: {path}")


def _require_live_motion_bus(path: str) -> None:
    """Reject a stale socket inode without acquiring a motion lease."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        probe.sendto(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "release",
                    "source": "preflight.probe",
                },
                separators=(",", ":"),
            ).encode(),
            path,
        )
    except OSError as exc:
        raise RuntimePreflightError(
            f"motion bus input_socket has no live owner: {path}: {exc}"
        ) from exc
    finally:
        probe.close()


def _require_broker_process(broker_pid: int, expected_repo: str) -> None:
    try:
        os.kill(broker_pid, 0)
    except OSError as exc:
        raise RuntimePreflightError(
            f"motion bus broker pid is not alive: {broker_pid}"
        ) from exc

    proc_root = f"/proc/{broker_pid}"
    if not os.path.isdir(proc_root):
        return
    try:
        with open(os.path.join(proc_root, "cmdline"), "rb") as stream:
            command_line = (
                stream.read()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        process_cwd = os.path.realpath(os.readlink(os.path.join(proc_root, "cwd")))
    except OSError as exc:
        raise RuntimePreflightError(
            f"cannot inspect motion bus broker pid {broker_pid}: {exc}"
        ) from exc
    if "onboard_runtime.motion_bus" not in command_line:
        raise RuntimePreflightError(
            f"broker pid {broker_pid} is not onboard_runtime.motion_bus"
        )
    if process_cwd != expected_repo:
        raise RuntimePreflightError(
            f"broker pid {broker_pid} cwd mismatch: {process_cwd} != {expected_repo}"
        )


def validate_motion_bus_status(
    status: Mapping[str, Any],
    *,
    expected_repo_root: str,
    expected_input_socket: str,
    required_output_sockets: Sequence[str],
    required_health_files: Sequence[str] = (),
    max_age_s: float = 2.0,
    now: float | None = None,
    check_socket_files: bool = True,
    check_process_identity: bool = True,
) -> None:
    if not isinstance(status, Mapping):
        raise RuntimePreflightError("motion bus status must be a JSON object")
    try:
        schema_version = int(status.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimePreflightError("unsupported motion bus status schema") from exc
    if schema_version != 1:
        raise RuntimePreflightError("unsupported motion bus status schema")

    maximum_age = float(max_age_s)
    if not math.isfinite(maximum_age) or maximum_age <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    try:
        timestamp = float(status.get("timestamp"))
    except (TypeError, ValueError) as exc:
        raise RuntimePreflightError("motion bus status timestamp is invalid") from exc
    if not math.isfinite(timestamp):
        raise RuntimePreflightError("motion bus status timestamp is invalid")
    current = time.time() if now is None else float(now)
    if not math.isfinite(current):
        raise ValueError("now must be finite")
    age = current - timestamp
    if age > maximum_age:
        raise RuntimePreflightError(f"motion bus status is stale ({age:.2f}s)")
    if age < -maximum_age:
        raise RuntimePreflightError(
            f"motion bus status timestamp is in the future ({-age:.2f}s)"
        )

    if not status.get("healthy", False):
        raise RuntimePreflightError(
            f"base runtime is unhealthy: {status.get('health_reason')}"
        )
    if status.get("motion_blocked", False):
        raise RuntimePreflightError("base runtime currently blocks motion")
    if status.get("safety_latched", False):
        raise RuntimePreflightError(
            "base runtime has a latched DAMP/LIMP safety command"
        )
    if not isinstance(status.get("broker_id"), str) or not status["broker_id"]:
        raise RuntimePreflightError("motion bus status has no broker identity")
    try:
        broker_pid = int(status.get("broker_pid"))
    except (TypeError, ValueError) as exc:
        raise RuntimePreflightError("motion bus status has invalid broker pid") from exc
    if broker_pid <= 0:
        raise RuntimePreflightError("motion bus status has invalid broker pid")

    expected_repo = _identity_path(expected_repo_root, "expected repo_root")
    actual_repo = _identity_path(status.get("repo_root"), "repo_root")
    if actual_repo != expected_repo:
        raise RuntimePreflightError(
            f"motion bus repo_root mismatch: {actual_repo} != {expected_repo}"
        )

    expected_input = _identity_path(expected_input_socket, "expected input_socket")
    actual_input = _identity_path(status.get("input_socket"), "input_socket")
    if actual_input != expected_input:
        raise RuntimePreflightError(
            f"motion bus input_socket mismatch: {actual_input} != {expected_input}"
        )

    if "command_file" not in status or status["command_file"] is not None:
        raise RuntimePreflightError(
            "motion bus is not the stream-only B runtime: command_file must be null"
        )

    raw_outputs = status.get("output_sockets")
    if not isinstance(raw_outputs, list):
        raise RuntimePreflightError("motion bus status has invalid output_sockets")
    actual_outputs = {
        _identity_path(path, "output_sockets entry") for path in raw_outputs
    }
    required_outputs = {
        _identity_path(path, "required output socket")
        for path in required_output_sockets
    }
    missing_outputs = sorted(required_outputs - actual_outputs)
    if missing_outputs:
        raise RuntimePreflightError(
            "motion bus is missing required output sockets: "
            + ", ".join(missing_outputs)
        )

    raw_health_files = status.get("health_files")
    if not isinstance(raw_health_files, list):
        raise RuntimePreflightError("motion bus status has invalid health_files")
    actual_health_files = {
        _identity_path(path, "health_files entry") for path in raw_health_files
    }
    required_health = {
        _identity_path(path, "required health file")
        for path in required_health_files
    }
    missing_health = sorted(required_health - actual_health_files)
    if missing_health:
        raise RuntimePreflightError(
            "motion bus is missing required health gates: "
            + ", ".join(missing_health)
        )
    health_runtime_id = status.get("health_runtime_id")
    if not isinstance(health_runtime_id, str) or len(health_runtime_id) < 8:
        raise RuntimePreflightError(
            "motion bus has no per-launch health runtime identity"
        )

    if check_socket_files:
        _require_unix_socket(expected_input, "motion bus input_socket")
        _require_live_motion_bus(expected_input)
        for output in sorted(required_outputs):
            _require_unix_socket(output, "motion bus output_socket")
    if check_process_identity:
        _require_broker_process(broker_pid, expected_repo)


def load_and_validate(
    status_file: str,
    **kwargs: Any,
) -> None:
    try:
        with open(status_file, "r", encoding="utf-8") as stream:
            status = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePreflightError(
            f"cannot read motion bus status {status_file}: {exc}"
        ) from exc
    validate_motion_bus_status(status, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--input-socket", required=True)
    parser.add_argument("--required-output-socket", action="append", default=[])
    parser.add_argument("--required-health-file", action="append", default=[])
    parser.add_argument("--max-age-s", type=float, default=2.0)
    args = parser.parse_args()
    try:
        load_and_validate(
            args.status_file,
            expected_repo_root=args.repo_root,
            expected_input_socket=args.input_socket,
            required_output_sockets=args.required_output_socket,
            required_health_files=args.required_health_file,
            max_age_s=args.max_age_s,
        )
    except RuntimePreflightError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
