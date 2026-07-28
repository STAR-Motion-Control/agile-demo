#!/usr/bin/env python3
"""DDS-free client for the arm-hang state machine hosted by the merger."""

from __future__ import annotations

import json
import os
import select
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_SOCKET = "/tmp/groot_arm_control.sock"
DEFAULT_STATUS_FILE = "/tmp/groot_arm_control_status.json"


class ArmControlClient:
    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET,
        status_file: str = DEFAULT_STATUS_FILE,
        source: str = "operator.keyboard",
        ack_timeout_s: float = 0.25,
    ):
        self.socket_path = socket_path
        self.status_file = Path(status_file)
        self.source = f"{source}.p{os.getpid()}.{uuid.uuid4().hex[:8]}"
        self.ack_timeout_s = max(0.05, float(ack_timeout_s))
        self._sequence = 0
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._client_dir: Path | None = None
        self._client_path: Path | None = None

    def _ensure_bound(self) -> None:
        if self._client_path is not None:
            return
        directory = Path(tempfile.mkdtemp(prefix="groot-arm-client-", dir="/tmp"))
        path = directory / "client.sock"
        try:
            self._socket.bind(str(path))
            os.chmod(path, 0o600)
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            directory.rmdir()
            raise
        self._client_dir = directory
        self._client_path = path

    def request(self, command: str) -> tuple[bool, str]:
        self._sequence += 1
        request_id = uuid.uuid4().hex
        payload = {
            "schema_version": 1,
            "request_id": request_id,
            "source": self.source,
            "sequence": self._sequence,
            "command": command,
            "timestamp": time.time(),
        }
        try:
            self._ensure_bound()
            self._socket.sendto(
                json.dumps(payload, separators=(",", ":")).encode(),
                self.socket_path,
            )
        except OSError as exc:
            return False, f"arm control unavailable: {exc}"

        deadline = time.monotonic() + self.ack_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False, f"arm control ACK timeout for {command}"
            readable, _, _ = select.select([self._socket], [], [], remaining)
            if not readable:
                return False, f"arm control ACK timeout for {command}"
            try:
                raw = self._socket.recv(4096)
                response: Any = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(response, dict):
                continue
            try:
                response_sequence = int(response.get("sequence", -1))
            except (TypeError, ValueError):
                continue
            if (
                response.get("request_id") != request_id
                or response.get("source") != self.source
                or response_sequence != self._sequence
            ):
                continue
            return bool(response.get("accepted", False)), str(
                response.get("message") or "arm control request rejected"
            )

    def toggle(self) -> tuple[bool, str]:
        return self.request("toggle")

    def release(self) -> tuple[bool, str]:
        return self.request("release")

    def emergency_release(self) -> tuple[bool, str]:
        return self.request("emergency_release")

    @property
    def state(self) -> str:
        try:
            payload: Any = json.loads(self.status_file.read_text(encoding="utf-8"))
            age = max(0.0, time.time() - float(payload.get("timestamp", 0.0)))
            if age > 1.0:
                return "status_stale"
            return str(payload.get("state", "unknown"))
        except Exception:
            return "unavailable"

    def close(self, release: bool = True) -> None:
        if release:
            self.release()
        self._socket.close()
        if self._client_path is not None:
            try:
                self._client_path.unlink()
            except FileNotFoundError:
                pass
            self._client_path = None
        if self._client_dir is not None:
            try:
                self._client_dir.rmdir()
            except OSError:
                pass
            self._client_dir = None
