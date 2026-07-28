"""Lease-based manipulation arm broker hosted by the final lowcmd merger."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import secrets
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .arm_protocol import (
    ARM_JOINT_COUNT,
    DEFAULT_ARM_RUNTIME_SOCKET,
    DEFAULT_ARM_RUNTIME_STATUS,
    SCHEMA_VERSION,
    SOURCE_RE,
    TOKEN_RE,
    ArmFrame,
    ArmProtocolError,
    finite_float,
    finite_vector,
)


@dataclass(frozen=True)
class _RobotState:
    sequence: int
    received_at: float
    mode_machine: int
    mode_pr: int
    q: tuple[float, ...]
    tau_est: tuple[float, ...]


@dataclass(frozen=True)
class _PolicyState:
    sequence: int
    received_at: float
    q: tuple[float, ...]


@dataclass
class _Owner:
    source: str
    token: str
    last_sequence: int
    expires_at: float
    last_publish_at: float
    frame: ArmFrame


@dataclass
class _Release:
    source: str
    request_sequence: int
    frame: ArmFrame
    duration_s: float
    hold_s: float
    progress_s: float
    last_tick: float
    last_q: tuple[float, ...]
    last_weight: float
    reason: str


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class ManipulationArmBroker:
    """Owns one upper-body lease and never initializes DDS itself."""

    def __init__(
        self,
        socket_path: str = DEFAULT_ARM_RUNTIME_SOCKET,
        status_file: str = DEFAULT_ARM_RUNTIME_STATUS,
        *,
        max_lease_s: float = 0.25,
        state_stale_s: float = 0.10,
        policy_stale_s: float = 0.10,
        auto_release_s: float = 0.50,
        release_hold_s: float = 0.10,
    ):
        if not 0.05 <= float(max_lease_s) <= 1.0:
            raise ValueError("max_lease_s must be in [0.05, 1.0]")
        self.socket_path = Path(socket_path)
        self.status_file = Path(status_file)
        self.max_lease_s = float(max_lease_s)
        self.state_stale_s = max(0.02, float(state_stale_s))
        self.policy_stale_s = max(0.02, float(policy_stale_s))
        self.auto_release_s = max(0.05, float(auto_release_s))
        self.release_hold_s = max(0.0, float(release_hold_s))
        self.boot_id = uuid.uuid4().hex
        self._socket: socket.socket | None = None
        self._lock_stream = None
        self._state: _RobotState | None = None
        self._policy: _PolicyState | None = None
        self._owner: _Owner | None = None
        self._release: _Release | None = None
        self._orphan: ArmFrame | None = None
        self._last_sequences: dict[str, int] = {}
        self._state_sequence = 0
        self._policy_sequence = 0
        self._applied_sequence = -1
        self._external_conflict = False
        self._last_error = ""
        self._last_status_write = 0.0
        self._last_status_key: tuple[Any, ...] | None = None
        self._metrics = {
            "accepted": 0,
            "rejected": 0,
            "snapshots": 0,
            "lease_expirations": 0,
            "releases": 0,
            "safety_stops": 0,
        }

    def activate_safety_stop(self) -> None:
        """Invalidate every manipulation owner while base safety is active."""
        if self._owner is not None or self._release is not None or self._orphan is not None:
            self._metrics["safety_stops"] += 1
        self._owner = None
        self._release = None
        self._orphan = None
        self._last_error = "SAFETY_ACTIVE:arm ownership invalidated"

    def update_robot_state(
        self,
        *,
        mode_machine: int,
        mode_pr: int,
        q: Any,
        tau_est: Any,
        received_at: float | None = None,
    ) -> None:
        self._state_sequence += 1
        self._state = _RobotState(
            sequence=self._state_sequence,
            received_at=time.monotonic() if received_at is None else float(received_at),
            mode_machine=int(mode_machine),
            mode_pr=int(mode_pr),
            q=finite_vector(q, "robot_state.q"),
            tau_est=finite_vector(
                tau_est,
                "robot_state.tau_est",
                absolute_limit=1000.0,
            ),
        )

    def update_policy(
        self,
        q: Any,
        *,
        received_at: float | None = None,
    ) -> None:
        self._policy_sequence += 1
        self._policy = _PolicyState(
            sequence=self._policy_sequence,
            received_at=time.monotonic() if received_at is None else float(received_at),
            q=finite_vector(q, "policy.q"),
        )

    def _fresh_state(self, now: float) -> _RobotState:
        state = self._state
        if state is None:
            raise ArmProtocolError("robot state is unavailable", "STATE_MISSING")
        if now - state.received_at > self.state_stale_s:
            raise ArmProtocolError("robot state is stale", "STATE_STALE")
        return state

    def _fresh_policy(self, now: float) -> _PolicyState:
        policy = self._policy
        if policy is None:
            raise ArmProtocolError("policy command is unavailable", "POLICY_MISSING")
        if now - policy.received_at > self.policy_stale_s:
            raise ArmProtocolError("policy command is stale", "POLICY_STALE")
        return policy

    @staticmethod
    def _source(raw: Mapping[str, Any]) -> str:
        source = str(raw.get("source", "")).strip()
        if not SOURCE_RE.fullmatch(source):
            raise ArmProtocolError("invalid source")
        return source

    def _sequence(self, raw: Mapping[str, Any], source: str) -> int:
        try:
            sequence = int(raw.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise ArmProtocolError("sequence must be an integer") from exc
        if sequence <= self._last_sequences.get(source, -1):
            raise ArmProtocolError("sequence is stale", "STALE_SEQUENCE")
        self._last_sequences[source] = sequence
        return sequence

    def _require_boot(self, raw: Mapping[str, Any]) -> None:
        if str(raw.get("boot_id", "")) != self.boot_id:
            raise ArmProtocolError("server boot id changed", "BOOT_CHANGED")

    def _lease_s(self, raw: Mapping[str, Any]) -> float:
        lease_s = finite_float(raw.get("lease_s", self.max_lease_s), "lease_s")
        if not 0.05 <= lease_s <= self.max_lease_s:
            raise ArmProtocolError(
                f"lease_s must be in [0.05, {self.max_lease_s:.3f}]"
            )
        return lease_s

    def _deadline(self, raw: Mapping[str, Any], now: float) -> None:
        try:
            deadline_ns = int(raw.get("deadline_mono_ns"))
        except (TypeError, ValueError) as exc:
            raise ArmProtocolError("deadline_mono_ns must be an integer") from exc
        now_ns = int(now * 1_000_000_000)
        if deadline_ns <= now_ns:
            raise ArmProtocolError("command deadline has expired", "DEADLINE_EXPIRED")
        if deadline_ns - now_ns > 1_000_000_000:
            raise ArmProtocolError("command deadline is too far in the future")

    def _require_owner(
        self,
        raw: Mapping[str, Any],
        source: str,
        now: float,
    ) -> _Owner:
        owner = self._owner
        token = str(raw.get("token", ""))
        if not TOKEN_RE.fullmatch(token):
            raise ArmProtocolError("invalid token", "BAD_TOKEN")
        if owner is None or owner.source != source or owner.token != token:
            raise ArmProtocolError("lease token is not current", "LEASE_LOST")
        if owner.expires_at <= now:
            raise ArmProtocolError("arm lease expired", "LEASE_EXPIRED")
        return owner

    def _begin_release(
        self,
        *,
        source: str,
        sequence: int,
        frame: ArmFrame,
        duration_s: float,
        now: float,
        reason: str,
    ) -> None:
        self._release = _Release(
            source=source,
            request_sequence=sequence,
            frame=frame,
            duration_s=max(0.05, float(duration_s)),
            hold_s=self.release_hold_s,
            progress_s=0.0,
            last_tick=now,
            last_q=frame.q,
            last_weight=1.0,
            reason=reason,
        )
        self._owner = None
        self._orphan = None
        self._metrics["releases"] += 1

    def _runtime_wire(self, now: float) -> dict[str, Any]:
        owner = self._owner
        release = self._release
        if owner is not None:
            state = "owned"
            source = owner.source
            applied = owner.last_sequence
            publish_age = max(0.0, now - owner.last_publish_at) * 1000.0
        elif release is not None:
            state = "releasing"
            source = release.source
            applied = release.request_sequence
            publish_age = None
        elif self._orphan is not None:
            state = "holding_orphan"
            source = None
            applied = self._applied_sequence
            publish_age = None
        else:
            state = "idle"
            source = None
            applied = self._applied_sequence
            publish_age = None
        return {
            "state": state,
            "owner": source,
            "applied_sequence": applied,
            "last_publish_age_ms": publish_age,
            "external_conflict": self._external_conflict,
            "last_error": self._last_error,
        }

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        current = time.monotonic() if now is None else float(now)
        state = self._state
        policy = self._policy
        state_wire = None
        if state is not None:
            state_wire = {
                "sequence": state.sequence,
                "received_mono_ns": int(state.received_at * 1_000_000_000),
                "age_ms": max(0.0, current - state.received_at) * 1000.0,
                "mode_machine": state.mode_machine,
                "mode_pr": state.mode_pr,
                "q": list(state.q),
                "tau_est": list(state.tau_est),
            }
        policy_wire = None
        if policy is not None:
            policy_wire = {
                "sequence": policy.sequence,
                "received_mono_ns": int(policy.received_at * 1_000_000_000),
                "age_ms": max(0.0, current - policy.received_at) * 1000.0,
                "q": list(policy.q),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "op": "snapshot_ack",
            "ok": state_wire is not None,
            "boot_id": self.boot_id,
            "state": state_wire,
            "policy": policy_wire,
            "runtime": self._runtime_wire(current),
        }

    def handle_request(
        self,
        raw: Any,
        *,
        now: float | None = None,
        external_active: bool = False,
        safety_active: bool = False,
    ) -> dict[str, Any]:
        current = time.monotonic() if now is None else float(now)
        request_id = ""
        response: dict[str, Any]
        try:
            if not isinstance(raw, Mapping):
                raise ArmProtocolError("request must be an object")
            request_id = str(raw.get("request_id", ""))
            if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
                raise ArmProtocolError("unsupported schema_version")
            op = str(raw.get("op", "")).strip().lower()
            source = self._source(raw)
            sequence = self._sequence(raw, source)

            if op == "snapshot":
                response = self.snapshot(current)
                response["request_id"] = request_id
                response["sequence"] = sequence
                self._metrics["snapshots"] += 1
                return response

            if safety_active:
                self.activate_safety_stop()
                raise ArmProtocolError(
                    "base DAMP/LIMP safety is active",
                    "SAFETY_ACTIVE",
                )

            self._require_boot(raw)
            self._fresh_state(current)

            if op == "acquire":
                self._fresh_policy(current)
                if external_active:
                    raise ArmProtocolError(
                        "legacy rt/arm_sdk owner is active",
                        "LEGACY_OWNER_ACTIVE",
                    )
                if self._release is not None or self._orphan is not None:
                    raise ArmProtocolError(
                        "previous arm owner is still releasing",
                        "RELEASE_IN_PROGRESS",
                    )
                if self._owner is not None and self._owner.expires_at > current:
                    raise ArmProtocolError(
                        f"arm runtime is owned by {self._owner.source}",
                        "OWNER_BUSY",
                    )
                frame = ArmFrame.from_wire(raw.get("frame"))
                lease_s = self._lease_s(raw)
                token = secrets.token_hex(16)
                self._owner = _Owner(
                    source=source,
                    token=token,
                    last_sequence=sequence,
                    expires_at=current + lease_s,
                    last_publish_at=current,
                    frame=frame,
                )
                self._applied_sequence = sequence
                response = {
                    "ok": True,
                    "token": token,
                    "lease_s": lease_s,
                    "message": "arm lease acquired",
                }
            elif op == "update":
                self._deadline(raw, current)
                owner = self._require_owner(raw, source, current)
                frame = ArmFrame.from_wire(raw.get("frame"))
                lease_s = self._lease_s(raw)
                owner.last_sequence = sequence
                owner.expires_at = current + lease_s
                owner.last_publish_at = current
                owner.frame = frame
                self._applied_sequence = sequence
                response = {"ok": True, "message": "arm frame accepted"}
            elif op == "heartbeat":
                self._deadline(raw, current)
                owner = self._require_owner(raw, source, current)
                lease_s = self._lease_s(raw)
                owner.last_sequence = sequence
                owner.expires_at = current + lease_s
                response = {"ok": True, "message": "lease renewed"}
            elif op in ("release_to_policy", "abort"):
                owner = self._require_owner(raw, source, current)
                self._fresh_policy(current)
                requested = finite_float(
                    raw.get(
                        "duration_s",
                        self.auto_release_s if op == "abort" else 2.5,
                    ),
                    "duration_s",
                )
                if not 0.05 <= requested <= 5.0:
                    raise ArmProtocolError("duration_s must be in [0.05, 5.0]")
                self._begin_release(
                    source=source,
                    sequence=sequence,
                    frame=owner.frame,
                    duration_s=requested,
                    now=current,
                    reason=op,
                )
                response = {
                    "ok": True,
                    "message": "server-side policy handoff started",
                    "duration_s": requested,
                }
            else:
                raise ArmProtocolError(f"unsupported op: {op!r}")
            self._metrics["accepted"] += 1
            self._last_error = ""
        except ArmProtocolError as exc:
            self._metrics["rejected"] += 1
            self._last_error = f"{exc.code}:{exc}"
            response = {
                "ok": False,
                "error_code": exc.code,
                "message": str(exc),
            }
        except Exception as exc:
            self._metrics["rejected"] += 1
            self._last_error = f"INTERNAL_ERROR:{type(exc).__name__}"
            response = {
                "ok": False,
                "error_code": "INTERNAL_ERROR",
                "message": f"request failed: {type(exc).__name__}",
            }
        response.update(
            {
                "schema_version": SCHEMA_VERSION,
                "op": "ack",
                "request_id": request_id,
                "boot_id": self.boot_id,
                "runtime": self._runtime_wire(current),
            }
        )
        return response

    def current_frame(
        self,
        *,
        now: float | None = None,
        external_active: bool = False,
    ) -> ArmFrame | None:
        current = time.monotonic() if now is None else float(now)
        self._external_conflict = bool(
            external_active
        )
        owner = self._owner
        if owner is not None:
            if owner.expires_at > current:
                self._applied_sequence = owner.last_sequence
                return owner.frame
            self._owner = None
            self._orphan = owner.frame
            self._metrics["lease_expirations"] += 1
            self._last_error = "LEASE_EXPIRED:holding last frame until fresh policy"

        if self._orphan is not None and self._release is None:
            try:
                self._fresh_policy(current)
            except ArmProtocolError:
                return self._orphan
            self._begin_release(
                source="expired_owner",
                sequence=self._applied_sequence,
                frame=self._orphan,
                duration_s=self.auto_release_s,
                now=current,
                reason="lease_expired",
            )

        release = self._release
        if release is None:
            return None
        try:
            policy = self._fresh_policy(current)
        except ArmProtocolError as exc:
            # Reassert full ownership while the handoff target is unavailable.
            # Restart alignment from the exact held q when policy recovers.
            release.frame = ArmFrame(
                q=release.last_q,
                weight=1.0,
                profile=release.frame.profile,
                scope=release.frame.scope,
            )
            release.progress_s = 0.0
            release.last_weight = 1.0
            release.last_tick = current
            self._last_error = f"{exc.code}:release paused"
            return ArmFrame(
                q=release.last_q,
                weight=release.last_weight,
                profile=release.frame.profile,
                scope=release.frame.scope,
            )

        dt = min(0.05, max(0.0, current - release.last_tick))
        release.last_tick = current
        previous_progress = min(release.progress_s, release.duration_s)
        release.progress_s += dt
        alignment_progress = min(release.progress_s, release.duration_s)
        previous_ratio = 0.5 * (
            1.0 - math.cos(math.pi * previous_progress / release.duration_s)
        )
        next_ratio = 0.5 * (
            1.0 - math.cos(math.pi * alignment_progress / release.duration_s)
        )
        if next_ratio > previous_ratio:
            remaining_ratio = max(1e-12, 1.0 - previous_ratio)
            incremental = min(1.0, (next_ratio - previous_ratio) / remaining_ratio)
            release.last_q = tuple(
                start * (1.0 - incremental) + target * incremental
                for start, target in zip(release.last_q, policy.q)
            )

        if release.progress_s <= release.duration_s:
            release.last_weight = 1.0
        else:
            # q is aligned before ownership/gains fade. Continue tracking the
            # fresh policy during the short fade so the final switch is bumpless.
            release.last_q = policy.q
            if release.hold_s <= 0.0:
                release.last_weight = 0.0
            else:
                fade_ratio = min(
                    1.0,
                    (release.progress_s - release.duration_s) / release.hold_s,
                )
                release.last_weight = 0.5 * (
                    1.0 + math.cos(math.pi * fade_ratio)
                )
        if (
            release.progress_s >= release.duration_s + release.hold_s
            or release.last_weight <= 0.0
        ):
            self._release = None
            self._last_error = ""
            return None
        return ArmFrame(
            q=release.last_q,
            weight=release.last_weight,
            profile=release.frame.profile,
            scope=release.frame.scope,
        )

    def bind(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.socket_path.with_name(f"{self.socket_path.name}.lock")
        lock_stream = lock_path.open("a+")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise ArmProtocolError(
                f"arm runtime is already active: {self.socket_path}",
                "OWNER_ACTIVE",
            ) from exc
        try:
            if self.socket_path.exists():
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                probe.setblocking(False)
                try:
                    probe.sendto(b'{"op":"probe"}', str(self.socket_path))
                except OSError as exc:
                    if exc.errno not in {
                        errno.ENOENT,
                        errno.ECONNREFUSED,
                        errno.ECONNRESET,
                    }:
                        raise
                else:
                    raise ArmProtocolError(
                        f"another arm runtime owns {self.socket_path}",
                        "OWNER_ACTIVE",
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

    def drain(
        self,
        *,
        now: float | None = None,
        external_active: bool = False,
        safety_active: bool = False,
        max_datagrams: int = 64,
    ) -> int:
        sock = self._socket
        if sock is None:
            return 0
        handled = 0
        while handled < max_datagrams:
            try:
                data, address = sock.recvfrom(16384)
            except BlockingIOError:
                break
            except OSError:
                break
            handled += 1
            try:
                if len(data) > 16384:
                    raise ArmProtocolError("datagram is too large")
                raw = json.loads(data.decode("utf-8"))
                response = self.handle_request(
                    raw,
                    now=now,
                    external_active=external_active,
                    safety_active=safety_active,
                )
            except Exception as exc:
                response = {
                    "schema_version": SCHEMA_VERSION,
                    "op": "ack",
                    "ok": False,
                    "boot_id": self.boot_id,
                    "error_code": "INVALID_DATAGRAM",
                    "message": f"invalid datagram: {type(exc).__name__}",
                }
            if address:
                try:
                    sock.sendto(
                        json.dumps(
                            response,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode(),
                        address,
                    )
                except OSError:
                    pass
        return handled

    def write_status(self, now: float | None = None, now_wall: float | None = None) -> None:
        current = time.monotonic() if now is None else float(now)
        runtime = self._runtime_wire(current)
        key = (
            runtime["state"],
            runtime["owner"],
            runtime["external_conflict"],
            runtime["last_error"],
        )
        # Called from the 500 Hz merger loop. Applied sequence changes at the
        # manipulation control rate, so it must not trigger filesystem writes.
        # State/error transitions stay immediate; steady state is capped at 2 Hz.
        if key == self._last_status_key and current - self._last_status_write < 0.5:
            return
        state_age = (
            None
            if self._state is None
            else max(0.0, current - self._state.received_at) * 1000.0
        )
        policy_age = (
            None
            if self._policy is None
            else max(0.0, current - self._policy.received_at) * 1000.0
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": time.time() if now_wall is None else float(now_wall),
            "boot_id": self.boot_id,
            **runtime,
            "state_age_ms": state_age,
            "policy_age_ms": policy_age,
            "metrics": dict(self._metrics),
        }
        _atomic_write_json(self.status_file, payload)
        self._last_status_key = key
        self._last_status_write = current

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
