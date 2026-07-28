"""DDS-free manipulation client for the arm runtime hosted by the merger."""

from __future__ import annotations

import json
import os
import select
import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from onboard_runtime.arm_protocol import (
    DEFAULT_ARM_RUNTIME_SOCKET,
    PROFILE_MANIP,
    SCHEMA_VERSION,
    SOURCE_RE,
    ArmFrame,
    ArmProtocolError,
    UpperBodyState,
)


class ArmRuntimeError(RuntimeError):
    def __init__(self, message: str, code: str = "ARM_RUNTIME_ERROR"):
        super().__init__(message)
        self.code = code


class ArmRuntimeClient:
    """One process-wide arm lease, state cache, and heartbeat."""

    is_ipc = True

    def __init__(
        self,
        socket_path: str = DEFAULT_ARM_RUNTIME_SOCKET,
        source: str | None = None,
        *,
        state_poll_hz: float = 50.0,
        lease_s: float = 0.25,
        heartbeat_s: float = 0.08,
        request_timeout_s: float = 0.10,
        instance_scoped: bool = True,
    ):
        logical_source = source or "manipulation.box_demo"
        if not SOURCE_RE.fullmatch(logical_source):
            raise ValueError("invalid arm runtime source")
        self.logical_source = logical_source
        if instance_scoped:
            suffix = f".p{os.getpid()}.{uuid.uuid4().hex[:8]}"
            if len(logical_source) + len(suffix) > 64:
                raise ValueError(
                    "arm runtime source is too long for an instance suffix"
                )
            logical_source = f"{logical_source}{suffix}"
        self.source = logical_source
        if state_poll_hz <= 0.0 or state_poll_hz > 100.0:
            raise ValueError("state_poll_hz must be in (0, 100]")
        if not 0.05 <= lease_s <= 0.25:
            raise ValueError("lease_s must be in [0.05, 0.25]")
        if not 0.02 <= heartbeat_s < lease_s:
            raise ValueError("heartbeat_s must be smaller than lease_s")
        self.socket_path = str(Path(socket_path).expanduser())
        self.state_poll_s = 1.0 / float(state_poll_hz)
        self.lease_s = float(lease_s)
        self.heartbeat_s = float(heartbeat_s)
        self.request_timeout_s = max(0.02, float(request_timeout_s))
        self._socket: socket.socket | None = None
        self._client_dir: str | None = None
        self._client_path: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._send_lock = threading.RLock()
        self._owner_lock = threading.Lock()
        self._state_condition = threading.Condition(threading.RLock())
        self._sequence = 0
        self._boot_id = ""
        self._token: str | None = None
        self._last_frame: ArmFrame | None = None
        self._last_state: UpperBodyState | None = None
        self._last_state_received_at = 0.0
        self._last_heartbeat = 0.0
        self._last_ack: dict[str, Any] | None = None
        self._last_error: ArmRuntimeError | None = None
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}

    @property
    def connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def token(self) -> str | None:
        return self._token

    @property
    def last_error(self) -> ArmRuntimeError | None:
        return self._last_error

    def connect(self) -> None:
        if self.connected:
            return
        client_dir = tempfile.mkdtemp(prefix=f"gar-{os.getpid()}-", dir="/tmp")
        client_path = os.path.join(client_dir, "client.sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.bind(client_path)
            os.chmod(client_path, 0o600)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            try:
                os.rmdir(client_dir)
            except OSError:
                pass
            raise
        self._client_dir = client_dir
        self._client_path = client_path
        self._socket = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._io_loop,
            name="arm_runtime_ipc",
            daemon=True,
        )
        self._thread.start()

    def _next_sequence(self) -> int:
        with self._send_lock:
            self._sequence += 1
            return self._sequence

    def _send_payload(self, payload: dict[str, Any]) -> None:
        sock = self._socket
        if sock is None:
            raise ArmRuntimeError("arm runtime client is not connected", "NOT_CONNECTED")
        data = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        try:
            with self._send_lock:
                sock.sendto(data, self.socket_path)
        except OSError as exc:
            raise ArmRuntimeError(
                f"arm runtime unavailable at {self.socket_path}: {exc}",
                "UNAVAILABLE",
            ) from exc

    def _envelope(
        self,
        op: str,
        *,
        request_id: str | None = None,
        require_boot: bool = True,
        **fields: Any,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "op": op,
            "source": self.source,
            "sequence": self._next_sequence(),
            "request_id": request_id or uuid.uuid4().hex,
            **fields,
        }
        if require_boot:
            payload["boot_id"] = self._boot_id
        return payload

    def _request(
        self,
        op: str,
        *,
        timeout_s: float | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        self.connect()
        request_id = uuid.uuid4().hex
        event = threading.Event()
        slot: dict[str, Any] = {}
        with self._state_condition:
            self._pending[request_id] = (event, slot)
        try:
            with self._send_lock:
                self._send_payload(
                    self._envelope(op, request_id=request_id, **fields)
                )
            timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
            if not event.wait(timeout):
                raise ArmRuntimeError(
                    f"{op} acknowledgement timed out",
                    "ACK_TIMEOUT",
                )
            response = slot.get("response")
            if not isinstance(response, dict):
                raise ArmRuntimeError("invalid arm runtime acknowledgement")
            if not bool(response.get("ok", False)):
                raise ArmRuntimeError(
                    str(response.get("message") or f"{op} rejected"),
                    str(response.get("error_code") or "REJECTED"),
                )
            return response
        finally:
            with self._state_condition:
                self._pending.pop(request_id, None)

    def _send_async(self, op: str, **fields: Any) -> None:
        with self._send_lock:
            self._send_payload(self._envelope(op, **fields))

    def _io_loop(self) -> None:
        next_snapshot = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_snapshot:
                try:
                    with self._send_lock:
                        self._send_payload(
                            self._envelope("snapshot", require_boot=False)
                        )
                except ArmRuntimeError as exc:
                    self._last_error = exc
                next_snapshot = now + self.state_poll_s
            token = self._token
            if (
                token is not None
                and now - self._last_heartbeat >= self.heartbeat_s
            ):
                try:
                    self._send_async(
                        "heartbeat",
                        token=token,
                        lease_s=self.lease_s,
                        deadline_mono_ns=int((now + 0.20) * 1_000_000_000),
                    )
                    self._last_heartbeat = now
                except ArmRuntimeError as exc:
                    self._last_error = exc

            sock = self._socket
            if sock is None:
                break
            timeout = max(0.0, min(0.02, next_snapshot - time.monotonic()))
            try:
                readable, _, _ = select.select([sock], [], [], timeout)
            except (OSError, ValueError):
                break
            if not readable:
                continue
            while True:
                try:
                    data = sock.recv(16384)
                except BlockingIOError:
                    break
                except OSError:
                    return
                self._handle_response(data)

    def _handle_response(self, data: bytes) -> None:
        try:
            raw = json.loads(data.decode("utf-8"))
            if not isinstance(raw, dict):
                return
            boot_id = str(raw.get("boot_id", ""))
            if boot_id:
                if self._boot_id and boot_id != self._boot_id:
                    self._token = None
                    self._last_error = ArmRuntimeError(
                        "arm runtime restarted; old lease invalidated",
                        "BOOT_CHANGED",
                    )
                self._boot_id = boot_id
            if raw.get("op") == "snapshot_ack" and raw.get("state") is not None:
                state = UpperBodyState.from_snapshot(raw)
                with self._state_condition:
                    self._last_state = state
                    self._last_state_received_at = time.monotonic()
                    self._last_error = None
                    self._state_condition.notify_all()
            elif raw.get("op") == "ack":
                self._last_ack = raw
                if not raw.get("ok", False):
                    error = ArmRuntimeError(
                        str(raw.get("message") or "arm command rejected"),
                        str(raw.get("error_code") or "REJECTED"),
                    )
                    self._last_error = error
                    if error.code in {
                        "BOOT_CHANGED",
                        "LEASE_EXPIRED",
                        "LEASE_LOST",
                        "BAD_TOKEN",
                        "SAFETY_ACTIVE",
                    }:
                        self._token = None
                else:
                    self._last_error = None
            request_id = str(raw.get("request_id", ""))
            with self._state_condition:
                pending = self._pending.get(request_id)
                if pending is not None:
                    event, slot = pending
                    slot["response"] = raw
                    event.set()
        except (ArmProtocolError, ValueError, TypeError, json.JSONDecodeError):
            return

    def latest_state(self, max_age_s: float = 0.10) -> UpperBodyState | None:
        with self._state_condition:
            state = self._last_state
            received_at = self._last_state_received_at
        if state is None:
            return None
        local_age = max(0.0, time.monotonic() - received_at)
        total_age = state.state_age_ms / 1000.0 + local_age
        if total_age > max(0.0, float(max_age_s)):
            return None
        return state

    def wait_state(
        self,
        timeout_s: float = 2.0,
        max_age_s: float = 0.10,
    ) -> UpperBodyState:
        self.connect()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._state_condition:
            while True:
                state = self.latest_state(max_age_s=max_age_s)
                if state is not None:
                    return state
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    error = self._last_error
                    if error is not None:
                        raise error
                    raise ArmRuntimeError(
                        "fresh robot state was not received",
                        "STATE_TIMEOUT",
                    )
                self._state_condition.wait(min(remaining, 0.05))

    def latest_policy(
        self,
        max_age_s: float = 0.10,
    ) -> tuple[float, ...] | None:
        with self._state_condition:
            state = self._last_state
            received_at = self._last_state_received_at
        if state is None or state.policy_q is None or state.policy_age_ms is None:
            return None
        total_age = state.policy_age_ms / 1000.0 + max(
            0.0, time.monotonic() - received_at
        )
        if total_age > max(0.0, float(max_age_s)):
            return None
        return state.policy_q

    def publish(
        self,
        q: Any,
        *,
        weight: float = 1.0,
        profile: str = PROFILE_MANIP,
    ) -> int:
        frame = ArmFrame.from_wire(
            {
                "q": q,
                "weight": weight,
                "profile": profile,
                "scope": "upper_body",
            }
        )
        self.connect()
        with self._owner_lock:
            if not self._boot_id:
                self.wait_state()
            if self._token is None:
                response = self._request(
                    "acquire",
                    frame=frame.to_wire(),
                    lease_s=self.lease_s,
                )
                self._token = str(response["token"])
                self._last_frame = frame
                self._last_heartbeat = time.monotonic()
                return int(response.get("runtime", {}).get("applied_sequence", -1))
            token = self._token
            self._last_frame = frame
            sequence = self._sequence + 1
            self._send_async(
                "update",
                token=token,
                frame=frame.to_wire(),
                lease_s=self.lease_s,
                deadline_mono_ns=int((time.monotonic() + 0.10) * 1_000_000_000),
            )
            return sequence

    def release_to_policy(
        self,
        *,
        duration_s: float = 2.5,
        timeout_s: float = 0.20,
        completion_timeout_s: float | None = None,
    ) -> bool:
        with self._owner_lock:
            token = self._token
            if token is None:
                return False
            response = self._request(
                "release_to_policy",
                timeout_s=timeout_s,
                token=token,
                duration_s=duration_s,
            )
            self._token = None
            self._last_frame = None
            ok = bool(response.get("ok"))
        if ok:
            self._wait_runtime_idle(
                timeout_s=(
                    max(0.5, float(duration_s) + 0.75)
                    if completion_timeout_s is None
                    else completion_timeout_s
                )
            )
        return ok

    def abort(
        self,
        *,
        duration_s: float = 0.50,
        timeout_s: float = 0.20,
        completion_timeout_s: float | None = None,
    ) -> bool:
        with self._owner_lock:
            token = self._token
            if token is None:
                return False
            response = self._request(
                "abort",
                timeout_s=timeout_s,
                token=token,
                duration_s=duration_s,
            )
            self._token = None
            self._last_frame = None
            ok = bool(response.get("ok"))
        if ok:
            self._wait_runtime_idle(
                timeout_s=(
                    max(0.5, float(duration_s) + 0.75)
                    if completion_timeout_s is None
                    else completion_timeout_s
                )
            )
        return ok

    def _wait_runtime_idle(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        with self._state_condition:
            while True:
                state = self._last_state
                if state is not None and state.runtime_state == "idle":
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ArmRuntimeError(
                        "server-side policy handoff did not complete",
                        "RELEASE_TIMEOUT",
                    )
                self._state_condition.wait(min(remaining, 0.05))

    def close(self, release: bool = True) -> None:
        if release and self._token is not None:
            try:
                self.abort()
            except ArmRuntimeError:
                # The server lease expires within 250 ms and performs the same
                # policy-aligned handoff when a fresh policy is available.
                pass
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        sock = self._socket
        self._socket = None
        if sock is not None:
            sock.close()
        if self._client_path:
            try:
                os.unlink(self._client_path)
            except FileNotFoundError:
                pass
        if self._client_dir:
            try:
                os.rmdir(self._client_dir)
            except OSError:
                pass
        self._client_path = None
        self._client_dir = None

    def __enter__(self) -> "ArmRuntimeClient":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
