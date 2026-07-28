"""Small runtime helpers for the navigation camera ingress."""

from __future__ import annotations

import math
import time


class CaptureErrorBackoff:
    """Bound retry CPU and error logging while a camera remains unavailable."""

    def __init__(
        self,
        *,
        initial_s: float = 0.05,
        maximum_s: float = 1.0,
        log_interval_s: float = 5.0,
    ):
        initial, maximum, log_interval = (
            float(initial_s),
            float(maximum_s),
            float(log_interval_s),
        )
        if not all(
            math.isfinite(value) for value in (initial, maximum, log_interval)
        ):
            raise ValueError("camera backoff values must be finite")
        if initial <= 0.0 or maximum < initial or log_interval < 0.0:
            raise ValueError("invalid camera backoff bounds")
        self.initial_s = initial
        self.maximum_s = maximum
        self.log_interval_s = log_interval
        self.reset()

    def failure(self, now: float | None = None) -> tuple[float, bool]:
        current = time.monotonic() if now is None else float(now)
        delay = self._next_delay_s
        self._next_delay_s = min(self.maximum_s, delay * 2.0)
        should_log = current >= self._next_log_at
        if should_log:
            self._next_log_at = current + self.log_interval_s
        return delay, should_log

    def reset(self) -> None:
        self._next_delay_s = self.initial_s
        self._next_log_at = float("-inf")
