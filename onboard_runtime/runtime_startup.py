"""Strict, offline-testable startup barriers for the onboard runtime launcher."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar


class StartupError(RuntimeError):
    """A readiness condition is absent, stale, or belongs to another runtime."""


_T = TypeVar("_T")


def _canonical_path(value: str | os.PathLike[str]) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def _schema_one(payload: Mapping[str, Any], label: str) -> None:
    value = payload.get("schema_version")
    if not isinstance(value, int) or isinstance(value, bool) or value != 1:
        raise StartupError(f"{label} schema_version must be 1")


def _finite_timestamp(payload: Mapping[str, Any], label: str) -> float:
    value = payload.get("timestamp")
    if isinstance(value, bool):
        raise StartupError(f"{label} timestamp is invalid")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise StartupError(f"{label} timestamp is invalid") from exc
    if not math.isfinite(timestamp):
        raise StartupError(f"{label} timestamp is invalid")
    return timestamp


def _validate_freshness(
    timestamp: float,
    *,
    label: str,
    started_after: float,
    max_age_s: float,
    now: float,
) -> None:
    if not all(math.isfinite(value) for value in (started_after, max_age_s, now)):
        raise StartupError(f"{label} freshness bounds must be finite")
    if max_age_s <= 0.0:
        raise StartupError(f"{label} max age must be positive")
    if timestamp < started_after:
        raise StartupError(f"{label} predates this startup")
    age = now - timestamp
    if age < -1.0:
        raise StartupError(f"{label} timestamp is in the future")
    if age > max_age_s:
        raise StartupError(f"{label} is stale ({age:.2f}s old)")


def _normalized_paths(values: Sequence[Any], label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise StartupError(f"{label} must be a list of paths")
    if any(not isinstance(value, str) or not value for value in values):
        raise StartupError(f"{label} contains an invalid path")
    return tuple(sorted(_canonical_path(value) for value in values))


def _required_path(payload: Mapping[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise StartupError(f"{label} {field} is missing")
    return value


def read_json_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    try:
        with candidate.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise StartupError(f"cannot read {label} {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StartupError(f"{label} must contain a JSON object")
    return payload


def validate_broker_status(
    payload: Mapping[str, Any],
    *,
    runtime_id: str,
    repo_root: str,
    input_socket: str,
    output_sockets: Sequence[str],
    health_files: Sequence[str],
    started_after: float,
    max_age_s: float,
    now: float | None = None,
    expected_pid: int | None = None,
    require_healthy: bool = False,
) -> None:
    """Validate that a broker status belongs exactly to this launch."""

    label = "motion bus status"
    _schema_one(payload, label)
    timestamp = _finite_timestamp(payload, label)
    _validate_freshness(
        timestamp,
        label=label,
        started_after=float(started_after),
        max_age_s=float(max_age_s),
        now=time.time() if now is None else float(now),
    )
    if payload.get("health_runtime_id") != runtime_id:
        raise StartupError("motion bus status runtime_id does not match this launch")
    if _canonical_path(_required_path(payload, "repo_root", label)) != _canonical_path(
        repo_root
    ):
        raise StartupError("motion bus status repo_root does not match this checkout")
    if _canonical_path(_required_path(payload, "input_socket", label)) != (
        _canonical_path(input_socket)
    ):
        raise StartupError("motion bus status input_socket does not match")
    if _normalized_paths(payload.get("output_sockets", ()), "output_sockets") != (
        _normalized_paths(output_sockets, "expected output_sockets")
    ):
        raise StartupError("motion bus status output_sockets do not match")
    if _normalized_paths(payload.get("health_files", ()), "health_files") != (
        _normalized_paths(health_files, "expected health_files")
    ):
        raise StartupError("motion bus status health_files do not match")
    if payload.get("command_file") is not None:
        raise StartupError("motion bus hot-path command_file must be disabled")

    broker_pid = payload.get("broker_pid")
    if not isinstance(broker_pid, int) or isinstance(broker_pid, bool) or broker_pid <= 0:
        raise StartupError("motion bus status broker_pid is invalid")
    if expected_pid is not None and broker_pid != expected_pid:
        raise StartupError("motion bus status broker_pid does not match its pane")
    if not isinstance(payload.get("broker_id"), str) or not payload["broker_id"]:
        raise StartupError("motion bus status broker_id is missing")
    healthy = payload.get("healthy")
    if not isinstance(healthy, bool):
        raise StartupError("motion bus status healthy flag is invalid")
    if require_healthy and not healthy:
        reason = str(payload.get("health_reason") or "unhealthy")
        raise StartupError(f"motion bus is not healthy: {reason}")


def validate_health_status(
    payload: Mapping[str, Any],
    *,
    runtime_id: str,
    started_after: float,
    max_age_s: float,
    now: float | None = None,
    component: str | None = None,
    require_healthy: bool = False,
) -> None:
    """Validate one component heartbeat without assuming it is healthy yet."""

    label = f"{component or 'component'} health"
    _schema_one(payload, label)
    timestamp = _finite_timestamp(payload, label)
    _validate_freshness(
        timestamp,
        label=label,
        started_after=float(started_after),
        max_age_s=float(max_age_s),
        now=time.time() if now is None else float(now),
    )
    if payload.get("runtime_id") != runtime_id:
        raise StartupError(f"{label} runtime_id does not match this launch")
    if component is not None and payload.get("component") != component:
        raise StartupError(f"{label} component does not match")
    healthy = payload.get("healthy")
    if not isinstance(healthy, bool):
        raise StartupError(f"{label} healthy flag is invalid")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason:
        raise StartupError(f"{label} reason is missing")
    if require_healthy and not healthy:
        raise StartupError(f"{label} is not healthy: {reason}")


def validate_pid_file(
    path: str | os.PathLike[str], *, expected_fragment: str = ""
) -> int:
    """Require a live PID and, on Linux, its expected command line."""

    candidate = Path(path).expanduser()
    try:
        raw = candidate.read_text(encoding="ascii").strip()
        pid = int(raw)
    except (OSError, UnicodeError, ValueError) as exc:
        raise StartupError(f"cannot read valid PID from {candidate}") from exc
    if pid <= 0:
        raise StartupError(f"invalid PID in {candidate}")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise StartupError(f"process {pid} from {candidate} is not alive") from exc
    except PermissionError:
        pass
    except OSError as exc:
        raise StartupError(f"cannot verify process {pid}: {exc}") from exc

    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if expected_fragment and proc_cmdline.exists():
        try:
            command_line = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError as exc:
            raise StartupError(f"cannot inspect process {pid}: {exc}") from exc
        if expected_fragment not in command_line:
            raise StartupError(
                f"process {pid} does not match expected command {expected_fragment!r}"
            )
    return pid


def require_unix_socket(path: str | os.PathLike[str]) -> None:
    candidate = Path(path).expanduser()
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise StartupError(f"required socket is absent: {candidate}") from exc
    if not stat.S_ISSOCK(mode):
        raise StartupError(f"required endpoint is not a Unix socket: {candidate}")


def probe_broker_socket(path: str | os.PathLike[str]) -> None:
    """Send a no-op release datagram to prove the bound broker is reachable."""

    candidate = str(Path(path).expanduser())
    payload = json.dumps(
        {"schema_version": 1, "type": "release", "source": "startup.probe"},
        separators=(",", ":"),
    ).encode("ascii")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        probe.setblocking(False)
        probe.sendto(payload, candidate)
    except OSError as exc:
        raise StartupError(f"motion bus socket is not reachable: {candidate}: {exc}") from exc
    finally:
        probe.close()


def wait_until(
    check: Callable[[], _T],
    *,
    timeout_s: float,
    stable_s: float = 0.0,
    poll_s: float = 0.10,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> _T:
    """Poll a readiness check and optionally require a continuous stable window."""

    timeout_s = float(timeout_s)
    stable_s = float(stable_s)
    poll_s = float(poll_s)
    if not all(math.isfinite(value) for value in (timeout_s, stable_s, poll_s)):
        raise StartupError("wait bounds must be finite")
    if timeout_s < 0.0 or stable_s < 0.0 or poll_s <= 0.0:
        raise StartupError("timeout/stable must be nonnegative and poll must be positive")
    deadline = clock() + timeout_s
    stable_since: float | None = None
    last_error = "readiness condition was not met"
    while True:
        try:
            result = check()
        except StartupError as exc:
            stable_since = None
            last_error = str(exc)
        else:
            now = clock()
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_s:
                return result

        now = clock()
        if now >= deadline:
            raise StartupError(f"timed out after {timeout_s:.1f}s: {last_error}")
        sleeper(min(poll_s, max(0.0, deadline - now)))


def _check_broker(args: argparse.Namespace, *, require_healthy: bool) -> None:
    pid = validate_pid_file(args.pid_file, expected_fragment="onboard_runtime.motion_bus")
    payload = read_json_object(args.status_file, "motion bus status")
    validate_broker_status(
        payload,
        runtime_id=args.runtime_id,
        repo_root=args.repo_root,
        input_socket=args.input_socket,
        output_sockets=args.output_socket,
        health_files=args.health_file,
        started_after=args.started_after,
        max_age_s=args.max_age,
        expected_pid=pid,
        require_healthy=require_healthy,
    )
    require_unix_socket(args.input_socket)


def _check_component(
    *,
    health_file: str,
    runtime_id: str,
    component: str | None,
    pid_file: str,
    process_fragment: str,
    required_sockets: Sequence[str],
    started_after: float,
    max_age: float,
    require_healthy: bool,
) -> None:
    validate_pid_file(pid_file, expected_fragment=process_fragment)
    payload = read_json_object(health_file, f"{component or 'component'} health")
    validate_health_status(
        payload,
        runtime_id=runtime_id,
        component=component,
        started_after=started_after,
        max_age_s=max_age,
        require_healthy=require_healthy,
    )
    for required_socket in required_sockets:
        require_unix_socket(required_socket)


def _add_broker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--input-socket", required=True)
    parser.add_argument("--output-socket", action="append", default=[], required=True)
    parser.add_argument("--health-file", action="append", default=[], required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--started-after", type=float, required=True)
    parser.add_argument("--max-age", type=float, default=2.5)
    parser.add_argument("--timeout", type=float, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    broker = commands.add_parser("wait-broker")
    _add_broker_arguments(broker)

    component = commands.add_parser("wait-component")
    component.add_argument("--health-file", required=True)
    component.add_argument("--runtime-id", required=True)
    component.add_argument("--component", required=True)
    component.add_argument("--pid-file", required=True)
    component.add_argument("--process-fragment", required=True)
    component.add_argument("--required-socket", action="append", default=[])
    component.add_argument("--started-after", type=float, required=True)
    component.add_argument("--max-age", type=float, default=2.5)
    component.add_argument("--timeout", type=float, required=True)

    composite = commands.add_parser("wait-composite")
    _add_broker_arguments(composite)
    composite.add_argument("--merger-health-file", required=True)
    composite.add_argument("--merger-pid-file", required=True)
    composite.add_argument("--merger-process-fragment", required=True)
    composite.add_argument("--merger-socket", action="append", default=[])
    composite.add_argument("--adapter-health-file", required=True)
    composite.add_argument("--adapter-pid-file", required=True)
    composite.add_argument("--adapter-process-fragment", required=True)
    composite.add_argument("--adapter-socket", action="append", default=[])
    composite.add_argument("--stable-s", type=float, default=1.0)
    return parser


def _run_command(args: argparse.Namespace) -> None:
    if args.command == "wait-broker":
        def check_broker_ready() -> None:
            _check_broker(args, require_healthy=False)
            probe_broker_socket(args.input_socket)

        wait_until(
            check_broker_ready,
            timeout_s=args.timeout,
        )
        print("[startup] broker ready and matched to this runtime")
        return

    if args.command == "wait-component":
        wait_until(
            lambda: _check_component(
                health_file=args.health_file,
                runtime_id=args.runtime_id,
                component=args.component,
                pid_file=args.pid_file,
                process_fragment=args.process_fragment,
                required_sockets=args.required_socket,
                started_after=args.started_after,
                max_age=args.max_age,
                require_healthy=False,
            ),
            timeout_s=args.timeout,
        )
        print(f"[startup] {args.component} initialized")
        return

    if args.command == "wait-composite":
        def check_composite() -> None:
            _check_broker(args, require_healthy=True)
            _check_component(
                health_file=args.merger_health_file,
                runtime_id=args.runtime_id,
                component="lowcmd_merger",
                pid_file=args.merger_pid_file,
                process_fragment=args.merger_process_fragment,
                required_sockets=args.merger_socket,
                started_after=args.started_after,
                max_age=args.max_age,
                require_healthy=True,
            )
            _check_component(
                health_file=args.adapter_health_file,
                runtime_id=args.runtime_id,
                component=None,
                pid_file=args.adapter_pid_file,
                process_fragment=args.adapter_process_fragment,
                required_sockets=args.adapter_socket,
                started_after=args.started_after,
                max_age=args.max_age,
                require_healthy=True,
            )

        wait_until(
            check_composite,
            timeout_s=args.timeout,
            stable_s=args.stable_s,
        )
        print("[startup] broker, merger, and adapter are continuously healthy")
        return

    raise StartupError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _run_command(args)
    except StartupError as exc:
        print(f"[startup] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
