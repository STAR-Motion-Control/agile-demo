#!/usr/bin/env python3
"""Fixed-rate, lease-based arbiter for GR00T base commands.

Producers send small Unix datagrams. One broker selects a single owner and
publishes latest-only local streams at a bounded rate. A legacy JSON output is
optional for explicit A/B compatibility. This removes per-request HTTP threads
and prevents navigation, manipulation, and keyboard control from racing as
independent writers.

This module has no DDS or robot SDK dependency and is safe to exercise offline.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import re
import secrets
import select
import signal
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SOCKET = "/tmp/groot_motion_bus.sock"
DEFAULT_COMMAND_FILE = "/tmp/robojudo_ext_cmd.json"
DEFAULT_STATUS_FILE = "/tmp/groot_motion_bus_status.json"
DEFAULT_SAFETY_FILE = "/tmp/groot_motion_safety_latch.json"
DEFAULT_PRIORITIES = {
    "safety": 1000,
    "operator": 900,
    "manipulation": 600,
    "navigation": 500,
}
_SOURCE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,48}$")
_ALLOWED_FSM = frozenset(("RL_FULL", "RL_LOWER", "DAMP", "LIMP"))
_SAFETY_FSM = frozenset(("DAMP", "LIMP"))


class MotionBusError(RuntimeError):
    """Raised when a command cannot be sent or validated."""


@dataclass(frozen=True)
class _Lease:
    source: str
    priority: int
    sequence: int
    received_at: float
    expires_at: float
    command: dict[str, Any]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
            stream.flush()
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MotionBusError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise MotionBusError(f"{name} must be finite")
    return result


def _normalize_command(raw: Any, default_height: float) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MotionBusError("command must be an object")

    fsm = str(raw.get("fsm", "RL_FULL")).strip().upper()
    if fsm not in _ALLOWED_FSM:
        raise MotionBusError(f"unsupported fsm: {fsm!r}")

    velocity = raw.get("velocity") or {}
    if not isinstance(velocity, Mapping):
        raise MotionBusError("command.velocity must be an object")
    forward = _finite(velocity.get("forward", 0.0), "velocity.forward")
    lateral = _finite(velocity.get("lateral", 0.0), "velocity.lateral")
    yaw = _finite(velocity.get("yaw", 0.0), "velocity.yaw")
    height = _finite(raw.get("height", default_height), "height")

    # The adapter performs the deployment-specific limits.  These broad bounds
    # reject corrupt payloads without silently changing a tuned command.
    if abs(forward) > 2.0 or abs(lateral) > 1.0 or abs(yaw) > 2.0:
        raise MotionBusError("velocity exceeds motion-bus sanity bounds")
    if not 0.25 <= height <= 0.90:
        raise MotionBusError("height exceeds motion-bus sanity bounds")

    command = {
        "fsm": fsm,
        "velocity": {
            "forward": forward,
            "lateral": lateral,
            "yaw": yaw,
        },
        "height": height,
        "units": "agile",
        "allow_recovery": bool(raw.get("allow_recovery", False)),
        "defer_recovery": bool(raw.get("defer_recovery", False)),
    }
    if fsm == "DAMP" or bool(raw.get("estop", False)):
        command["fsm"] = "DAMP"
        command["estop"] = True
    elif fsm == "LIMP" or bool(raw.get("limp", False)):
        command["fsm"] = "LIMP"
        command["limp"] = True
    return command


class _HealthGate:
    def __init__(
        self,
        paths: str | Sequence[str] | None,
        stale_s: float,
        required: bool,
        runtime_id: str | None = None,
    ):
        if isinstance(paths, str):
            raw_paths = [paths] if paths else []
        else:
            raw_paths = list(paths or ())
        self.paths = tuple(Path(path).expanduser() for path in raw_paths if path)
        self.stale_s = max(0.1, float(stale_s))
        self.required = bool(required)
        self.runtime_id = str(runtime_id or "")
        self._next_read = 0.0
        self._healthy = not self.required
        self._reason = "not_required" if not self.required else "health_missing"

    def check(self, now_mono: float, now_wall: float) -> tuple[bool, str]:
        if not self.required:
            return True, "not_required"
        if now_mono < self._next_read:
            return self._healthy, self._reason
        self._next_read = now_mono + 0.10
        if not self.paths:
            self._healthy, self._reason = False, "health_missing"
            return self._healthy, self._reason

        failures = []
        for path in self.paths:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                updated_at = float(payload.get("timestamp", 0.0))
                age = now_wall - updated_at
                if not math.isfinite(age) or age < -self.stale_s:
                    failures.append(f"{path.name}:timestamp_invalid")
                elif age > self.stale_s:
                    failures.append(f"{path.name}:stale:{age:.3f}s")
                elif not bool(payload.get("healthy", False)):
                    reason = str(payload.get("reason") or "unhealthy")
                    failures.append(f"{path.name}:{reason}")
                elif self.runtime_id and payload.get("runtime_id") != self.runtime_id:
                    failures.append(f"{path.name}:runtime_id_mismatch")
            except FileNotFoundError:
                failures.append(f"{path.name}:missing")
            except Exception as exc:
                failures.append(f"{path.name}:invalid:{type(exc).__name__}")
        self._healthy = not failures
        self._reason = "healthy" if self._healthy else ",".join(failures)
        return self._healthy, self._reason


class MotionCommandBroker:
    """Lease arbiter with one bounded-rate compatibility output."""

    def __init__(
        self,
        *,
        socket_path: str = DEFAULT_SOCKET,
        command_file: str | None = DEFAULT_COMMAND_FILE,
        status_file: str = DEFAULT_STATUS_FILE,
        safety_file: str = DEFAULT_SAFETY_FILE,
        output_sockets: list[str] | tuple[str, ...] | None = None,
        output_hz: float = 20.0,
        default_height: float = 0.76,
        max_lease_s: float = 5.0,
        priorities: Mapping[str, int] | None = None,
        health_file: str | Sequence[str] | None = None,
        health_stale_s: float = 0.5,
        require_health: bool = False,
        health_runtime_id: str | None = None,
    ):
        if output_hz <= 0.0 or output_hz > 100.0:
            raise ValueError("output_hz must be in (0, 100]")
        self.socket_path = Path(socket_path).expanduser()
        self.command_file = (
            Path(command_file).expanduser() if command_file not in (None, "") else None
        )
        self.status_file = Path(status_file).expanduser()
        self.safety_file = Path(safety_file).expanduser()
        self.output_sockets = tuple(
            str(Path(path).expanduser()) for path in (output_sockets or ()) if path
        )
        self.period_s = 1.0 / float(output_hz)
        self.default_height = float(default_height)
        self.max_lease_s = max(0.1, float(max_lease_s))
        self.priorities = dict(DEFAULT_PRIORITIES)
        if priorities:
            self.priorities.update({str(k): int(v) for k, v in priorities.items()})
        self.health = _HealthGate(
            health_file,
            health_stale_s,
            require_health,
            runtime_id=health_runtime_id,
        )
        self._leases: dict[str, _Lease] = {}
        self._latched_safety: _Lease | None = None
        self._blocked_sources: set[str] = set()
        self._last_sequence: dict[str, int] = {}
        self._last_height = self.default_height
        self._last_safe_height = self.default_height
        self._socket: socket.socket | None = None
        self._stream_socket: socket.socket | None = None
        self._lock_stream = None
        self.broker_id = secrets.token_hex(16)
        self._broker_sequence = 0
        self._last_status_key: tuple[Any, ...] | None = None
        self._last_status_write = 0.0
        self._metrics = {
            "accepted": 0,
            "blocked": 0,
            "released": 0,
            "rejected": 0,
            "expired": 0,
            "output_writes": 0,
            "stream_sends": 0,
            "stream_drops": 0,
            "file_writes": 0,
        }
        self._restore_existing_safety()

    def _restore_existing_safety(self) -> None:
        try:
            with self.safety_file.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except FileNotFoundError:
            self._restore_legacy_command_safety()
            return
        except Exception as exc:
            raise MotionBusError(
                f"cannot read persisted safety state {self.safety_file}: {exc}"
            ) from exc

        try:
            if not isinstance(raw, Mapping):
                raise MotionBusError("persisted safety state must be an object")
            schema_version = raw.get("schema_version")
            if (
                not isinstance(schema_version, int)
                or isinstance(schema_version, bool)
                or schema_version not in (1, 2)
            ):
                raise MotionBusError(
                    "persisted safety schema_version must be 1 or 2"
                )
            state = "latched" if schema_version == 1 else raw.get("state")
            if state not in {"latched", "cleared"}:
                raise MotionBusError("persisted safety state is invalid")
            required = {"schema_version", "timestamp"}
            if state == "latched":
                required.update({"source", "sequence", "fsm", "velocity", "height"})
            if schema_version == 2:
                required.update({"state", "blocked_sources"})
            missing = sorted(required.difference(raw))
            if missing:
                raise MotionBusError(
                    f"persisted safety state is missing: {', '.join(missing)}"
                )
            timestamp = raw.get("timestamp")
            if isinstance(timestamp, bool):
                raise MotionBusError("persisted safety timestamp must be numeric")
            _finite(timestamp, "persisted safety timestamp")
            blocked_raw = raw.get("blocked_sources")
            if blocked_raw is None and state == "latched":
                blocked_raw = [raw.get("source")]
            if not isinstance(blocked_raw, list) or any(
                not isinstance(item, str) or not _SOURCE_RE.fullmatch(item)
                for item in blocked_raw
            ):
                raise MotionBusError("persisted blocked_sources is invalid")
            blocked_sources = set(blocked_raw)
            if state == "cleared":
                if not blocked_sources:
                    raise MotionBusError(
                        "persisted cleared safety epoch has no blocked sources"
                    )
                self._blocked_sources.update(blocked_sources)
                return

            source = raw.get("source")
            if not isinstance(source, str) or not _SOURCE_RE.fullmatch(source):
                raise MotionBusError("persisted safety source is invalid")
            sequence = raw.get("sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise MotionBusError("persisted safety sequence must be an integer")
            velocity = raw.get("velocity")
            if not isinstance(velocity, Mapping) or not {
                "forward",
                "lateral",
                "yaw",
            }.issubset(velocity):
                raise MotionBusError("persisted safety velocity is incomplete")
            blocked_sources.add(source)
            command = _normalize_command(raw, self.default_height)
            if command["fsm"] not in _SAFETY_FSM:
                raise MotionBusError("persisted safety state must be DAMP or LIMP")
        except Exception as exc:
            if isinstance(exc, MotionBusError):
                detail = str(exc)
            else:
                detail = f"{type(exc).__name__}: {exc}"
            raise MotionBusError(
                f"invalid persisted safety state {self.safety_file}: {detail}"
            ) from exc
        self._blocked_sources.update(blocked_sources)
        self._last_sequence[source] = sequence
        self._latched_safety = self._restored_safety_lease(command)

    def _restore_legacy_command_safety(self) -> None:
        if self.command_file is None:
            return
        try:
            with self.command_file.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
            command = _normalize_command(raw, self.default_height)
            if command["fsm"] in _SAFETY_FSM:
                self._latched_safety = self._restored_safety_lease(command)
        except Exception:
            # The legacy output is not a safety journal and may be mid-update.
            return

    def _restored_safety_lease(self, command: dict[str, Any]) -> _Lease:
        return _Lease(
            source="safety.restored",
            priority=self.priorities["safety"],
            sequence=-1,
            received_at=time.monotonic(),
            expires_at=math.inf,
            command=command,
        )

    def _persist_safety(self, lease: _Lease) -> None:
        payload = dict(lease.command)
        payload.update(
            {
                "schema_version": 2,
                "state": "latched",
                "timestamp": time.time(),
                "source": lease.source,
                "sequence": lease.sequence,
                "blocked_sources": sorted(self._blocked_sources),
            }
        )
        _atomic_write_json(self.safety_file, payload)

    def _persist_cleared_epoch(self, blocked_sources: set[str]) -> None:
        if not blocked_sources:
            self._clear_persisted_safety()
            return
        _atomic_write_json(
            self.safety_file,
            {
                "schema_version": 2,
                "state": "cleared",
                "timestamp": time.time(),
                "blocked_sources": sorted(blocked_sources),
            },
        )

    def _clear_persisted_safety(self) -> None:
        try:
            self.safety_file.unlink()
        except FileNotFoundError:
            pass

    def _priority_for(self, source: str) -> int:
        if source in self.priorities:
            return self.priorities[source]
        prefix = source.split(".", 1)[0]
        return self.priorities.get(prefix, 0)

    def handle_payload(self, payload: Any, received_at: float | None = None) -> None:
        now = time.monotonic() if received_at is None else float(received_at)
        if not isinstance(payload, Mapping):
            raise MotionBusError("payload must be an object")
        kind = str(payload.get("type", "command")).strip().lower()
        source = str(payload.get("source", "")).strip()
        if not _SOURCE_RE.fullmatch(source):
            raise MotionBusError("source must match [A-Za-z0-9_.-]{1,48}")

        if kind == "release":
            lease_removed = source in self._leases
            blocked_removed = (
                self._latched_safety is None and source in self._blocked_sources
            )
            if blocked_removed:
                remaining_sources = self._blocked_sources.difference({source})
                self._persist_cleared_epoch(remaining_sources)
                self._blocked_sources = remaining_sources
            if lease_removed:
                self._leases.pop(source, None)
            if lease_removed or blocked_removed:
                self._metrics["released"] += 1
            return
        if kind == "clear_safety":
            try:
                sequence = int(payload.get("sequence"))
            except (TypeError, ValueError) as exc:
                raise MotionBusError("sequence must be an integer") from exc
            if not source.startswith("operator"):
                raise MotionBusError("only an operator source may clear latched safety")
            if sequence <= self._last_sequence.get(source, -1):
                raise MotionBusError("sequence is not newer than the last accepted command")
            self._persist_cleared_epoch(self._blocked_sources)
            self._last_sequence[source] = sequence
            self._latched_safety = None
            self._metrics["accepted"] += 1
            return
        if kind != "command":
            raise MotionBusError(f"unsupported payload type: {kind!r}")

        if self._latched_safety is not None and source not in self._blocked_sources:
            self._blocked_sources.add(source)
            self._persist_safety(self._latched_safety)

        try:
            sequence = int(payload.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise MotionBusError("sequence must be an integer") from exc
        if sequence <= self._last_sequence.get(source, -1):
            raise MotionBusError("sequence is not newer than the last accepted command")
        lease_s = _finite(payload.get("lease_s", 0.35), "lease_s")
        if not 0.05 <= lease_s <= self.max_lease_s:
            raise MotionBusError(
                f"lease_s must be in [0.05, {self.max_lease_s:.2f}]"
            )
        command = _normalize_command(payload.get("command"), self._last_height)
        lease = _Lease(
            source=source,
            priority=self._priority_for(source),
            sequence=sequence,
            received_at=now,
            expires_at=now + lease_s,
            command=command,
        )
        if command["fsm"] in _SAFETY_FSM:
            self._blocked_sources.update(self._last_sequence)
            self._blocked_sources.update(self._leases)
            self._blocked_sources.add(source)
            self._leases.clear()
            self._latched_safety = _Lease(
                source=lease.source,
                priority=max(self.priorities["safety"], lease.priority),
                sequence=lease.sequence,
                received_at=lease.received_at,
                expires_at=math.inf,
                command=lease.command,
            )
            self._persist_safety(self._latched_safety)
        else:
            if self._latched_safety is not None:
                self._metrics["blocked"] += 1
                raise MotionBusError("ordinary commands are blocked by latched safety")
            if source in self._blocked_sources:
                self._metrics["blocked"] += 1
                raise MotionBusError(
                    "source is blocked until it releases after safety is cleared"
                )
            self._leases[source] = lease
        self._last_height = float(command["height"])
        self._last_sequence[source] = sequence
        self._metrics["accepted"] += 1

    def handle_datagram(self, data: bytes, received_at: float | None = None) -> bool:
        try:
            if len(data) > 8192:
                raise MotionBusError("datagram exceeds 8192 bytes")
            self.handle_payload(json.loads(data.decode("utf-8")), received_at)
            return True
        except Exception:
            self._metrics["rejected"] += 1
            return False

    def _select(self, now_mono: float) -> _Lease | None:
        expired = [
            source for source, lease in self._leases.items()
            if lease.expires_at <= now_mono
        ]
        for source in expired:
            del self._leases[source]
        self._metrics["expired"] += len(expired)
        if self._latched_safety is not None:
            return self._latched_safety
        if not self._leases:
            return None
        return max(
            self._leases.values(),
            key=lambda lease: (lease.priority, lease.received_at, lease.sequence),
        )

    def tick(
        self,
        now_mono: float | None = None,
        now_wall: float | None = None,
    ) -> dict[str, Any]:
        mono = time.monotonic() if now_mono is None else float(now_mono)
        wall = time.time() if now_wall is None else float(now_wall)
        selected = self._select(mono)
        healthy, health_reason = self.health.check(mono, wall)
        blocked = False

        if selected is None:
            output = {
                "fsm": "RL_FULL",
                "velocity": {"forward": 0.0, "lateral": 0.0, "yaw": 0.0},
                "height": self._last_safe_height,
                "units": "agile",
                "allow_recovery": False,
            }
            source, priority, sequence = "motion_bus.idle", -1, -1
        else:
            output = dict(selected.command)
            output["velocity"] = dict(selected.command["velocity"])
            source, priority, sequence = (
                selected.source,
                selected.priority,
                selected.sequence,
            )
            if not healthy and output["fsm"] not in _SAFETY_FSM:
                output["velocity"] = {
                    "forward": 0.0,
                    "lateral": 0.0,
                    "yaw": 0.0,
                }
                output["allow_recovery"] = False
                output["height"] = self._last_safe_height
                blocked = True
            elif healthy and output["fsm"] not in _SAFETY_FSM:
                self._last_safe_height = float(output["height"])

        output.update(
            {
                "timestamp": wall,
                "motion_bus_source": source,
                "motion_bus_sequence": sequence,
                "motion_bus_health_ok": healthy,
                "motion_bus_health_reason": health_reason,
            }
        )
        self._broker_sequence += 1
        output.update(
            {
                "schema_version": 1,
                "stream_type": "motion_command",
                "broker_id": self.broker_id,
                "broker_pid": os.getpid(),
                "broker_sequence": self._broker_sequence,
            }
        )
        self._publish_stream(output)
        if self.command_file is not None:
            _atomic_write_json(self.command_file, output)
            self._metrics["file_writes"] += 1
        self._metrics["output_writes"] += 1

        status = {
            "schema_version": 1,
            "timestamp": wall,
            "healthy": healthy,
            "health_reason": health_reason,
            "motion_blocked": blocked,
            "selected_source": source,
            "selected_priority": priority,
            "selected_sequence": sequence,
            "active_sources": sorted(self._leases),
            "blocked_sources": sorted(self._blocked_sources),
            "safety_latched": self._latched_safety is not None,
            "broker_id": self.broker_id,
            "broker_pid": os.getpid(),
            "repo_root": str(Path(__file__).resolve().parents[1]),
            "input_socket": str(self.socket_path),
            "output_sockets": list(self.output_sockets),
            "command_file": (
                None if self.command_file is None else str(self.command_file)
            ),
            "health_files": [str(path) for path in self.health.paths],
            "health_runtime_id": self.health.runtime_id or None,
            "metrics": dict(self._metrics),
        }
        status_key = (
            source,
            healthy,
            health_reason,
            blocked,
            tuple(status["active_sources"]),
            tuple(status["blocked_sources"]),
            status["safety_latched"],
        )
        if status_key != self._last_status_key or mono - self._last_status_write >= 1.0:
            _atomic_write_json(self.status_file, status)
            self._last_status_key = status_key
            self._last_status_write = mono
        return status

    def _publish_stream(self, output: Mapping[str, Any]) -> None:
        if not self.output_sockets:
            return
        if self._stream_socket is None:
            self._stream_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._stream_socket.setblocking(False)
        data = json.dumps(output, separators=(",", ":"), allow_nan=False).encode()
        for destination in self.output_sockets:
            try:
                self._stream_socket.sendto(data, destination)
                self._metrics["stream_sends"] += 1
            except OSError as exc:
                if exc.errno not in {
                    errno.EAGAIN,
                    errno.EWOULDBLOCK,
                    errno.ENOENT,
                    errno.ECONNREFUSED,
                    errno.ENOBUFS,
                }:
                    raise
                self._metrics["stream_drops"] += 1

    def _bind(self) -> socket.socket:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.socket_path.with_name(f"{self.socket_path.name}.lock")
        lock_stream = lock_path.open("a+")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise MotionBusError(
                f"motion bus is already active: {self.socket_path}"
            ) from exc
        try:
            if self.socket_path.exists():
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                probe.setblocking(False)
                try:
                    probe.sendto(b'{"type":"probe"}', str(self.socket_path))
                except OSError as exc:
                    if exc.errno not in {
                        errno.ENOENT,
                        errno.ECONNREFUSED,
                        errno.ECONNRESET,
                    }:
                        raise
                else:
                    raise MotionBusError(
                        f"another motion bus owns {self.socket_path}"
                    )
                finally:
                    probe.close()
                self.socket_path.unlink(missing_ok=True)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o660)
            sock.setblocking(False)
        except BaseException:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
            raise
        self._lock_stream = lock_stream
        self._socket = sock
        return sock

    def close(self) -> None:
        owned = self._lock_stream is not None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._stream_socket is not None:
            self._stream_socket.close()
            self._stream_socket = None
        if owned:
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def run(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or threading.Event()
        sock = self._bind()
        next_tick = time.monotonic()
        try:
            while not stop.is_set():
                now = time.monotonic()
                timeout = max(0.0, min(self.period_s, next_tick - now))
                readable, _, _ = select.select([sock], [], [], timeout)
                if readable:
                    while True:
                        try:
                            data = sock.recv(8192)
                        except BlockingIOError:
                            break
                        self.handle_datagram(data)
                now = time.monotonic()
                if now >= next_tick:
                    self.tick(now_mono=now)
                    missed = max(1, int((now - next_tick) / self.period_s) + 1)
                    next_tick += missed * self.period_s
        finally:
            self.close()


class MotionBusClient:
    """Nonblocking producer for one named motion source."""

    def __init__(
        self,
        source: str,
        socket_path: str = DEFAULT_SOCKET,
        *,
        instance_scoped: bool = True,
    ):
        if not _SOURCE_RE.fullmatch(source):
            raise ValueError("invalid motion source")
        self.logical_source = source
        if instance_scoped:
            suffix = f".p{os.getpid()}.{secrets.token_hex(4)}"
            if len(source) + len(suffix) > 48:
                raise ValueError("motion source is too long for an instance suffix")
            source = f"{source}{suffix}"
        self.source = source
        self.socket_path = str(Path(socket_path).expanduser())
        self._sequence = 0
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._lock = threading.Lock()

    def _send(self, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        try:
            self._socket.sendto(data, self.socket_path)
        except OSError as exc:
            raise MotionBusError(
                f"motion bus unavailable at {self.socket_path}: {exc}"
            ) from exc

    def publish(
        self,
        *,
        fsm: str = "RL_FULL",
        forward: float = 0.0,
        lateral: float = 0.0,
        yaw: float = 0.0,
        height: float = 0.76,
        lease_s: float = 0.35,
        allow_recovery: bool = False,
        defer_recovery: bool = False,
        estop: bool = False,
        limp: bool = False,
    ) -> int:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self._send(
                {
                    "schema_version": 1,
                    "type": "command",
                    "source": self.source,
                    "sequence": sequence,
                    "lease_s": lease_s,
                    "command": {
                        "fsm": fsm,
                        "velocity": {
                            "forward": forward,
                            "lateral": lateral,
                            "yaw": yaw,
                        },
                        "height": height,
                        "units": "agile",
                        "allow_recovery": allow_recovery,
                        "defer_recovery": defer_recovery,
                        "estop": estop,
                        "limp": limp,
                    },
                }
            )
            return sequence

    def release(self) -> None:
        self._send({"schema_version": 1, "type": "release", "source": self.source})

    def clear_safety(self) -> int:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self._send(
                {
                    "schema_version": 1,
                    "type": "clear_safety",
                    "source": self.source,
                    "sequence": sequence,
                }
            )
            return sequence

    def close(self, release: bool = True) -> None:
        if release:
            try:
                self.release()
            except MotionBusError:
                pass
        self._socket.close()

    def __enter__(self) -> "MotionBusClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _parse_priorities(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        source, separator, priority = value.partition("=")
        if not separator or not _SOURCE_RE.fullmatch(source):
            raise argparse.ArgumentTypeError(
                f"invalid priority {value!r}; expected source=integer"
            )
        parsed[source] = int(priority)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--command-file", default=DEFAULT_COMMAND_FILE)
    parser.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    parser.add_argument("--safety-file", default=DEFAULT_SAFETY_FILE)
    parser.add_argument("--output-socket", action="append", default=[])
    parser.add_argument("--output-hz", type=float, default=20.0)
    parser.add_argument("--default-height", type=float, default=0.76)
    parser.add_argument("--max-lease-s", type=float, default=5.0)
    parser.add_argument("--priority", action="append", default=[])
    parser.add_argument("--health-file", action="append", default=[])
    parser.add_argument("--health-stale-s", type=float, default=0.5)
    parser.add_argument("--health-runtime-id", default="")
    parser.add_argument("--require-health", action="store_true")
    args = parser.parse_args()

    broker = MotionCommandBroker(
        socket_path=args.socket,
        command_file=args.command_file,
        status_file=args.status_file,
        safety_file=args.safety_file,
        output_sockets=args.output_socket,
        output_hz=args.output_hz,
        default_height=args.default_height,
        max_lease_s=args.max_lease_s,
        priorities=_parse_priorities(args.priority),
        health_file=args.health_file,
        health_stale_s=args.health_stale_s,
        require_health=args.require_health,
        health_runtime_id=args.health_runtime_id or None,
    )
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    broker.run(stop)


if __name__ == "__main__":
    main()
