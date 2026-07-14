#!/usr/bin/env python3
"""Adaptive stop-to-stance recovery for the explicit taptap launch path."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StanceMetrics:
    width: float
    # Signed left-minus-right fore/aft offset in the pelvis-yaw frame.
    stagger: float
    height_delta: float
    yaw_error: float = 0.0


class AdaptiveTapTapController:
    IDLE = "IDLE"
    SETTLING = "SETTLING"
    RECOVERING = "RECOVERING"

    def __init__(
        self,
        *,
        reference_width: float = 0.24,
        reference_stagger: float = 0.08,
        width_margin: float = 0.035,
        stagger_limit: float = 0.08,
        yaw_limit: float = 0.12,
        max_height_delta: float = 0.03,
        debounce_s: float = 0.35,
        confirm_s: float = 0.12,
        min_motion_s: float = 0.40,
        recovery_s: float = 1.60,
        recovery_speed: float = 0.08,
        phase_s: float = 0.40,
        calibration_s: float = 0.30,
        auto_calibrate: bool = False,
        startup_check: bool = True,
    ):
        self.reference_width = float(reference_width)
        self.reference_stagger = float(reference_stagger)
        self.width_margin = max(0.0, float(width_margin))
        self.stagger_limit = max(0.0, float(stagger_limit))
        self.yaw_limit = max(0.0, float(yaw_limit))
        self.max_height_delta = max(0.0, float(max_height_delta))
        self.debounce_s = max(0.0, float(debounce_s))
        self.confirm_s = max(0.0, float(confirm_s))
        self.min_motion_s = max(0.0, float(min_motion_s))
        self.recovery_s = max(0.0, float(recovery_s))
        self.recovery_speed = max(0.0, float(recovery_speed))
        self.phase_s = max(1e-3, float(phase_s))
        self.calibration_s = max(0.0, float(calibration_s))
        self.auto_calibrate = bool(auto_calibrate)

        self.state = self.IDLE
        self._motion_since = None
        self._settle_since = None
        self._recover_since = None
        self._samples: list[StanceMetrics] = []
        self._calibration_since = None
        self._calibration_widths: list[float] = []
        self._calibration_staggers: list[float] = []
        self._calibrated = False
        self._startup_checked = not bool(startup_check)
        self._checking_startup = False
        self._pending = None
        self._start_sign = 1.0

    @staticmethod
    def _moving(cmd) -> bool:
        return abs(cmd.vx) + abs(cmd.vy) + abs(cmd.wz) > 1e-6

    @staticmethod
    def _hard_stop(cmd) -> bool:
        return bool(cmd.estop or cmd.fsm != "RL_FULL")

    def _calibrate(self, now: float, cmd, stance: StanceMetrics | None) -> None:
        if (not self.auto_calibrate or self._calibrated
                or self.state != self.IDLE or self._moving(cmd)):
            return
        if (self._hard_stop(cmd) or not cmd.fresh or stance is None
                or stance.height_delta > self.max_height_delta
                or self.stance_bad(stance)):
            self._calibration_since = None
            self._calibration_widths.clear()
            self._calibration_staggers.clear()
            return
        if self._calibration_since is None:
            self._calibration_since = now
        self._calibration_widths.append(stance.width)
        self._calibration_staggers.append(stance.stagger)
        if now - self._calibration_since >= self.calibration_s:
            self.reference_width = statistics.median(self._calibration_widths)
            self.reference_stagger = statistics.median(self._calibration_staggers)
            self._calibrated = True

    def stance_bad(self, stance: StanceMetrics) -> bool:
        return (
            abs(stance.width - self.reference_width) > self.width_margin
            or abs(stance.stagger - self.reference_stagger) > self.stagger_limit
            or abs(stance.yaw_error) > self.yaw_limit
        )

    def _finish(self):
        pending = self._pending
        self.state = self.IDLE
        self._settle_since = None
        self._recover_since = None
        self._samples.clear()
        self._pending = None
        self._checking_startup = False
        return pending

    def update(self, now: float, cmd, stance: StanceMetrics | None):
        """Return ``(effective_cmd, recovery_active, event)``."""
        self._calibrate(now, cmd, stance)
        moving = self._moving(cmd)

        if self._hard_stop(cmd):
            was_active = self.state != self.IDLE
            self._finish()
            self._motion_since = None
            return cmd, False, "cancelled" if was_active else None

        if self.state == self.IDLE:
            if not cmd.fresh:
                self._motion_since = None
                return cmd, False, None
            if moving:
                if (cmd.allow_recovery and not self._startup_checked and stance is not None
                        and stance.height_delta <= self.max_height_delta):
                    self._startup_checked = True
                    if self.stance_bad(stance):
                        self.state = self.SETTLING
                        self._settle_since = now - self.debounce_s
                        self._samples = [stance]
                        self._pending = cmd
                        self._checking_startup = True
                        zero_cmd = replace(cmd, vx=0.0, vy=0.0, wz=0.0)
                        return zero_cmd, False, "checking_initial"
                if self._motion_since is None:
                    self._motion_since = now
                return cmd, False, None
            if self._motion_since is None:
                return cmd, False, None
            motion_duration = now - self._motion_since
            self._motion_since = None
            if not cmd.allow_recovery or motion_duration < self.min_motion_s:
                return cmd, False, None
            self.state = self.SETTLING
            self._settle_since = now
            self._samples.clear()
            return replace(cmd, vx=0.0, vy=0.0, wz=0.0), False, "checking"

        # A fresh explicit stop is a safety override. Stale input is allowed
        # only after a fresh allow_recovery transition has armed this bounded
        # internal sequence; this keeps standard and taptap mover timing equal.
        if cmd.fresh and not cmd.allow_recovery:
            self._finish()
            self._motion_since = None
            return cmd, False, "cancelled"

        if moving:
            self._pending = cmd

        zero_cmd = replace(cmd, vx=0.0, vy=0.0, wz=0.0)
        if self.state == self.SETTLING:
            elapsed = now - self._settle_since
            sample_start = max(0.0, self.debounce_s - self.confirm_s)
            if (elapsed >= sample_start and stance is not None
                    and stance.height_delta <= self.max_height_delta):
                self._samples.append(stance)
            if elapsed < self.debounce_s:
                return zero_cmd, False, None
            if not self._samples:
                pending = self._finish()
                return pending or zero_cmd, False, "no_stance"
            measured = StanceMetrics(
                width=statistics.median(x.width for x in self._samples),
                stagger=statistics.median(x.stagger for x in self._samples),
                height_delta=statistics.median(x.height_delta for x in self._samples),
                yaw_error=statistics.median(x.yaw_error for x in self._samples),
            )
            if not self.stance_bad(measured):
                pending = self._finish()
                return pending or zero_cmd, False, "healthy"
            self.state = self.RECOVERING
            self._recover_since = now
            # Step first toward the calibrated fore/aft relationship. This is
            # deterministic and avoids making an already bad stagger worse.
            self._start_sign = (
                1.0 if self.reference_stagger - measured.stagger >= 0.0 else -1.0
            )
            return zero_cmd, True, "started"

        elapsed = now - self._recover_since
        if elapsed >= self.recovery_s:
            pending = self._finish()
            return pending or zero_cmd, False, "completed"
        phase = int(elapsed / self.phase_s) % 2
        vx = self._start_sign * (self.recovery_speed if phase == 0 else -self.recovery_speed)
        return replace(zero_cmd, vx=vx), True, None
