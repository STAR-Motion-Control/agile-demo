from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Any


@dataclass(frozen=True)
class CameraFrame:
    rgb: Any
    depth: Any
    sequence: int
    captured_at: float
    source: str
    depth_scale: float = 1.0

    @property
    def has_depth(self) -> bool:
        return self.depth is not None


class CameraFrameHub:
    """Thread-safe latest-only exchange for coherent camera frames."""

    def __init__(self, clock: Callable[[], float] = monotonic):
        self._clock = clock
        self._condition = Condition()
        self._latest: CameraFrame | None = None
        self._sequence = 0

    def publish(
        self,
        rgb: Any,
        depth: Any = None,
        *,
        captured_at: float | None = None,
        source: str = "unknown",
        depth_scale: float = 1.0,
    ) -> CameraFrame:
        if rgb is None:
            raise ValueError("rgb frame is required")
        timestamp = self._clock() if captured_at is None else float(captured_at)
        with self._condition:
            self._sequence += 1
            frame = CameraFrame(
                rgb=rgb,
                depth=depth,
                sequence=self._sequence,
                captured_at=timestamp,
                source=str(source),
                depth_scale=float(depth_scale),
            )
            self._latest = frame
            self._condition.notify_all()
            return frame

    def latest(
        self,
        *,
        require_depth: bool = False,
        max_age_s: float | None = None,
    ) -> CameraFrame | None:
        with self._condition:
            frame = self._latest
        if frame is None or (require_depth and not frame.has_depth):
            return None
        if max_age_s is not None and self._clock() - frame.captured_at > max(0.0, float(max_age_s)):
            return None
        return frame

    def wait_for_frame(
        self,
        *,
        after_sequence: int = 0,
        require_depth: bool = False,
        timeout: float | None = None,
    ) -> CameraFrame | None:
        deadline = None if timeout is None else monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                frame = self._latest
                if (
                    frame is not None
                    and frame.sequence > int(after_sequence)
                    and (not require_depth or frame.has_depth)
                ):
                    return frame
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
