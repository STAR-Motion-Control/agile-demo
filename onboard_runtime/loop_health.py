"""Low-overhead control-loop health reporter used by the motion gate."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any


def _percentile(values: deque[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return float(ordered[index])


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
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


class ControlLoopHealthReporter:
    """Tracks timing in memory and writes at most ``report_hz`` status files."""

    def __init__(
        self,
        path: str,
        *,
        expected_period_s: float,
        p99_max_s: float = 0.040,
        max_gap_s: float = 0.060,
        recent_success_s: float = 0.20,
        min_samples: int = 10,
        window_samples: int = 200,
        report_hz: float = 2.0,
        runtime_id: str | None = None,
    ):
        self.path = Path(path).expanduser()
        self.expected_period_s = max(1e-4, float(expected_period_s))
        self.p99_max_s = max(self.expected_period_s, float(p99_max_s))
        self.max_gap_s = max(self.expected_period_s, float(max_gap_s))
        self.recent_success_s = max(self.expected_period_s, float(recent_success_s))
        self.min_samples = max(2, int(min_samples))
        self._cycle_periods: deque[float] = deque(maxlen=max(self.min_samples, window_samples))
        self._compute_times: deque[float] = deque(maxlen=max(self.min_samples, window_samples))
        self._last_cycle_start: float | None = None
        self._last_success: float | None = None
        self._next_report = 0.0
        self._report_period = 1.0 / max(0.2, float(report_hz))
        self._success_count = 0
        self._failure_count = 0
        self.runtime_id = str(
            runtime_id
            if runtime_id is not None
            else os.environ.get("GROOT_RUNTIME_ID", "")
        )

    def record(
        self,
        *,
        cycle_started_at: float,
        compute_s: float,
        success: bool,
        now_wall: float | None = None,
    ) -> dict[str, Any] | None:
        now_mono = float(cycle_started_at) + max(0.0, float(compute_s))
        if self._last_cycle_start is not None:
            interval = max(0.0, float(cycle_started_at) - self._last_cycle_start)
            self._cycle_periods.append(interval)
        self._last_cycle_start = float(cycle_started_at)
        self._compute_times.append(max(0.0, float(compute_s)))
        if success:
            self._last_success = now_mono
            self._success_count += 1
        else:
            self._failure_count += 1
        if now_mono < self._next_report:
            return None
        self._next_report = now_mono + self._report_period
        payload = self.snapshot(now_mono, time.time() if now_wall is None else now_wall)
        _atomic_write(self.path, payload)
        return payload

    def snapshot(self, now_mono: float, now_wall: float) -> dict[str, Any]:
        sample_count = len(self._cycle_periods)
        cycle_p99 = _percentile(self._cycle_periods, 0.99)
        cycle_max = max(self._cycle_periods, default=0.0)
        compute_p99 = _percentile(self._compute_times, 0.99)
        last_success_age = (
            float("inf")
            if self._last_success is None
            else max(0.0, now_mono - self._last_success)
        )
        overruns = sum(
            value > self.expected_period_s * 1.5 for value in self._cycle_periods
        )
        overrun_ratio = overruns / sample_count if sample_count else 1.0

        reasons = []
        if sample_count < self.min_samples:
            reasons.append("warming_up")
        if last_success_age > self.recent_success_s:
            reasons.append("observation_stale")
        if cycle_p99 > self.p99_max_s:
            reasons.append("cycle_p99_high")
        if cycle_max > self.max_gap_s:
            reasons.append("cycle_gap_high")
        if overrun_ratio > 0.20:
            reasons.append("overrun_ratio_high")
        healthy = not reasons
        payload = {
            "schema_version": 1,
            "timestamp": float(now_wall),
            "healthy": healthy,
            "reason": "healthy" if healthy else ",".join(reasons),
            "sample_count": sample_count,
            "cycle_p99_ms": cycle_p99 * 1000.0,
            "cycle_max_ms": cycle_max * 1000.0,
            "compute_p99_ms": compute_p99 * 1000.0,
            "overrun_ratio": overrun_ratio,
            "last_success_age_ms": (
                None if self._last_success is None else last_success_age * 1000.0
            ),
            "success_count": self._success_count,
            "failure_count": self._failure_count,
        }
        if self.runtime_id:
            payload["runtime_id"] = self.runtime_id
        return payload

    def mark_stopped(self, reason: str = "adapter_stopped") -> None:
        payload = {
            "schema_version": 1,
            "timestamp": time.time(),
            "healthy": False,
            "reason": reason,
            "sample_count": len(self._cycle_periods),
        }
        if self.runtime_id:
            payload["runtime_id"] = self.runtime_id
        _atomic_write(self.path, payload)


class RuntimeHealthHeartbeat:
    """Low-rate liveness heartbeat for a runtime consumer.

    A control loop may call :meth:`record` at hundreds of hertz; disk writes
    only happen on a health transition or at ``report_hz``. Write failures are
    deliberately visible so the owner cannot leave a stale healthy file.
    """

    def __init__(
        self,
        path: str,
        *,
        component: str,
        report_hz: float = 2.0,
        runtime_id: str | None = None,
    ):
        self.path = Path(path).expanduser()
        self.component = str(component)
        self._report_period = 1.0 / max(0.2, float(report_hz))
        self._next_report = 0.0
        self._last_state: tuple[bool, str] | None = None
        self._sequence = 0
        self.runtime_id = str(
            runtime_id
            if runtime_id is not None
            else os.environ.get("GROOT_RUNTIME_ID", "")
        )

    def record(
        self,
        *,
        healthy: bool,
        reason: str,
        now_mono: float | None = None,
        now_wall: float | None = None,
        **details: Any,
    ) -> dict[str, Any] | None:
        mono = time.monotonic() if now_mono is None else float(now_mono)
        state = (bool(healthy), str(reason))
        if mono < self._next_report and state == self._last_state:
            return None
        self._next_report = mono + self._report_period
        self._last_state = state
        self._sequence += 1
        payload = {
            "schema_version": 1,
            "timestamp": time.time() if now_wall is None else float(now_wall),
            "component": self.component,
            "healthy": state[0],
            "reason": state[1],
            "sequence": self._sequence,
            **details,
        }
        if self.runtime_id:
            payload["runtime_id"] = self.runtime_id
        _atomic_write(self.path, payload)
        return payload

    def mark_stopped(self, reason: str = "runtime_stopped") -> None:
        self._sequence += 1
        payload = {
            "schema_version": 1,
            "timestamp": time.time(),
            "component": self.component,
            "healthy": False,
            "reason": str(reason),
            "sequence": self._sequence,
        }
        if self.runtime_id:
            payload["runtime_id"] = self.runtime_id
        _atomic_write(self.path, payload)
