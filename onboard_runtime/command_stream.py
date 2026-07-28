"""Latest-only Unix datagram stream from the motion broker to local consumers."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any


DEFAULT_ADAPTER_COMMAND_SOCKET = "/tmp/groot_adapter_command.sock"
DEFAULT_MERGER_COMMAND_SOCKET = "/tmp/groot_merger_command.sock"


class CommandStreamError(RuntimeError):
    pass


class LatestCommandReceiver:
    """Owns one consumer socket and retains only the newest valid command."""

    def __init__(
        self,
        socket_path: str,
        *,
        max_datagram: int = 8192,
        max_future_skew_s: float = 1.0,
    ):
        self.socket_path = Path(socket_path).expanduser()
        self.max_datagram = int(max_datagram)
        self.max_future_skew_s = float(max_future_skew_s)
        if (
            not math.isfinite(self.max_future_skew_s)
            or self.max_future_skew_s < 0.0
        ):
            raise ValueError("max_future_skew_s must be finite and non-negative")
        self._socket: socket.socket | None = None
        self._lock_stream = None
        self._latest: dict[str, Any] | None = None
        self._received_at = 0.0
        self._wire_age_at_receive = math.inf
        self._broker_id = ""
        self._broker_sequence = -1
        self.metrics = {"accepted": 0, "rejected": 0}

    def bind(self) -> None:
        if self._socket is not None:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.socket_path.with_name(f"{self.socket_path.name}.lock")
        lock_stream = lock_path.open("a+")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise CommandStreamError(
                f"command receiver is already active: {self.socket_path}"
            ) from exc
        try:
            if self.socket_path.exists():
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                probe.setblocking(False)
                try:
                    probe.sendto(b'{}', str(self.socket_path))
                except OSError as exc:
                    if exc.errno not in {
                        errno.ENOENT,
                        errno.ECONNREFUSED,
                        errno.ECONNRESET,
                    }:
                        raise
                else:
                    raise CommandStreamError(
                        f"another process owns command receiver: {self.socket_path}"
                    )
                finally:
                    probe.close()
                self.socket_path.unlink(missing_ok=True)
        except BaseException:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
            raise
        try:
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

    def drain(
        self,
        *,
        max_datagrams: int = 64,
        now: float | None = None,
        now_wall: float | None = None,
    ) -> int:
        sock = self._socket
        if sock is None:
            raise CommandStreamError("command receiver is not bound")
        received_at = time.monotonic() if now is None else float(now)
        wall = time.time() if now_wall is None else float(now_wall)
        if not math.isfinite(received_at) or not math.isfinite(wall):
            raise ValueError("drain timestamps must be finite")
        handled = 0
        while handled < max_datagrams:
            try:
                data = sock.recv(self.max_datagram)
            except BlockingIOError:
                break
            except OSError:
                break
            handled += 1
            try:
                raw = json.loads(data.decode("utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("command must be an object")
                if int(raw.get("schema_version", 0)) != 1:
                    raise ValueError("unsupported schema")
                if raw.get("stream_type") != "motion_command":
                    raise ValueError("unsupported stream type")
                broker_id = str(raw.get("broker_id", ""))
                sequence = int(raw.get("broker_sequence", -1))
                if not broker_id or sequence < 0:
                    raise ValueError("missing broker identity")
                if broker_id == self._broker_id and sequence <= self._broker_sequence:
                    raise ValueError("stale broker sequence")
                timestamp_raw = raw.get("timestamp")
                if isinstance(timestamp_raw, bool):
                    raise ValueError("timestamp must be numeric")
                timestamp = float(timestamp_raw)
                if not math.isfinite(timestamp):
                    raise ValueError("timestamp must be finite")
                wire_age = wall - timestamp
                if wire_age < -self.max_future_skew_s:
                    raise ValueError("timestamp exceeds allowed future skew")
                self._latest = raw
                self._received_at = received_at
                self._wire_age_at_receive = max(0.0, wire_age)
                self._broker_id = broker_id
                self._broker_sequence = sequence
                self.metrics["accepted"] += 1
            except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                self.metrics["rejected"] += 1
        return handled

    def latest(
        self,
        *,
        now: float | None = None,
        stale_s: float = 0.40,
    ) -> tuple[dict[str, Any] | None, bool]:
        current = time.monotonic() if now is None else float(now)
        stale_limit = float(stale_s)
        if not math.isfinite(current) or not math.isfinite(stale_limit):
            raise ValueError("latest timestamps must be finite")
        raw = self._latest
        local_age = max(0.0, current - self._received_at)
        effective_age = local_age + self._wire_age_at_receive
        fresh = raw is not None and effective_age <= stale_limit
        return raw, bool(fresh)

    def close(self) -> None:
        owned = self._lock_stream is not None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if owned:
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def __enter__(self) -> "LatestCommandReceiver":
        self.bind()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
