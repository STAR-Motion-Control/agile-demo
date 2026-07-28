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
    BLOCKED = "BLOCKED"

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
        navigation_min_width: float = 0.18,
        navigation_release_width: float = 0.19,
        navigation_lateral_min_speed: float = 0.08,
        navigation_guard_confirm_s: float = 0.06,
        max_recovery_attempts: int = 2,
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
        self.navigation_min_width = max(0.0, float(navigation_min_width))
        self.navigation_release_width = max(
            self.navigation_min_width,
            float(navigation_release_width),
        )
        self.navigation_lateral_min_speed = max(
            0.0, float(navigation_lateral_min_speed)
        )
        self.navigation_guard_confirm_s = max(
            0.0, float(navigation_guard_confirm_s)
        )
        self.max_recovery_attempts = max(1, int(max_recovery_attempts))

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
        self._waiting_for_stance = False
        self._navigation_guard_active = False
        self._recovery_attempts = 0
        self._navigation_narrow_since = None

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

    @property
    def recovery_attempts(self) -> int:
        return self._recovery_attempts

    def _needs_recovery(self, stance: StanceMetrics) -> bool:
        if self._navigation_guard_active:
            # The in-navigation gate has one safety objective: prevent a
            # narrow double-support stance from being carried into the next
            # step.  Do not turn normal H-arm stagger/yaw variation into a
            # permanent navigation lock.
            return stance.width < self.navigation_release_width
        return self.stance_bad(stance)

    def _finish(self):
        pending = self._pending
        self.state = self.IDLE
        self._settle_since = None
        self._recover_since = None
        self._samples.clear()
        self._pending = None
        self._checking_startup = False
        self._waiting_for_stance = False
        self._navigation_guard_active = False
        self._recovery_attempts = 0
        self._navigation_narrow_since = None
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
                if not cmd.defer_recovery:
                    self._motion_since = None
                return cmd, False, None
            if moving:
                # Only meaningful lateral navigation is relevant to the
                # crossover problem.  Forward/turn commands must not be
                # interrupted merely because the normal standing width is
                # below the policy's canonical 0.24 m reference.
                narrow_lateral_support = (
                    cmd.defer_recovery
                    and abs(cmd.vy) >= self.navigation_lateral_min_speed
                    and stance is not None
                    and stance.height_delta <= self.max_height_delta
                    and stance.width < self.navigation_min_width
                )
                if narrow_lateral_support:
                    if self._navigation_narrow_since is None:
                        self._navigation_narrow_since = now
                    narrow_confirmed = (
                        now - self._navigation_narrow_since
                        >= self.navigation_guard_confirm_s
                    )
                else:
                    self._navigation_narrow_since = None
                    narrow_confirmed = False
                if narrow_confirmed:
                    self.state = self.SETTLING
                    self._settle_since = now
                    self._samples = [stance]
                    self._pending = cmd
                    self._navigation_guard_active = True
                    self._recovery_attempts = 0
                    self._navigation_narrow_since = None
                    self._waiting_for_stance = False
                    zero_cmd = replace(cmd, vx=0.0, vy=0.0, wz=0.0)
                    return zero_cmd, False, "navigation_guard"
                if ((cmd.allow_recovery or cmd.defer_recovery)
                        and not self._startup_checked and stance is not None
                        and stance.height_delta <= self.max_height_delta):
                    self._startup_checked = True
                    if self.stance_bad(stance):
                        self.state = self.SETTLING
                        self._settle_since = now - self.debounce_s
                        self._samples = [stance]
                        self._waiting_for_stance = False
                        self._pending = cmd
                        self._checking_startup = True
                        zero_cmd = replace(cmd, vx=0.0, vy=0.0, wz=0.0)
                        return zero_cmd, False, "checking_initial"
                if self._motion_since is None:
                    self._motion_since = now
                return cmd, False, None
            if self._motion_since is None:
                return cmd, False, None
            if cmd.defer_recovery:
                return cmd, False, None
            motion_duration = now - self._motion_since
            self._motion_since = None
            if not cmd.allow_recovery or motion_duration < self.min_motion_s:
                return cmd, False, None
            self.state = self.SETTLING
            self._settle_since = now
            self._samples.clear()
            self._waiting_for_stance = False
            return replace(cmd, vx=0.0, vy=0.0, wz=0.0), False, "checking"

        # A fresh explicit stop is a safety override. Stale input is allowed
        # only after a fresh allow_recovery transition has armed this bounded
        # internal sequence; this keeps standard and taptap mover timing equal.
        if cmd.fresh and not cmd.allow_recovery and not cmd.defer_recovery:
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
                # Both feet are not yet in a valid double-support sample
                # (typically height_delta is still above the swing threshold).
                # Fail closed: keep the requested motion pending and continue
                # sampling at zero speed instead of silently releasing it.
                first_wait = not self._waiting_for_stance
                self._waiting_for_stance = True
                return zero_cmd, False, "waiting_stance" if first_wait else None
            self._waiting_for_stance = False
            measured = StanceMetrics(
                width=statistics.median(x.width for x in self._samples),
                stagger=statistics.median(x.stagger for x in self._samples),
                height_delta=statistics.median(x.height_delta for x in self._samples),
                yaw_error=statistics.median(x.yaw_error for x in self._samples),
            )
            if not self._needs_recovery(measured):
                recovered = self._recovery_attempts > 0
                pending = self._finish()
                event = "completed" if recovered else "healthy"
                return pending or zero_cmd, False, event
            if self._recovery_attempts >= self.max_recovery_attempts:
                self.state = self.BLOCKED
                return zero_cmd, True, "blocked"
            self.state = self.RECOVERING
            self._recover_since = now
            self._recovery_attempts += 1
            # Step first toward the calibrated fore/aft relationship. This is
            # deterministic and avoids making an already bad stagger worse.
            self._start_sign = (
                1.0 if self.reference_stagger - measured.stagger >= 0.0 else -1.0
            )
            event = "started" if self._recovery_attempts == 1 else "retry_started"
            return zero_cmd, True, event

        if self.state == self.RECOVERING:
            elapsed = now - self._recover_since
            if elapsed >= self.recovery_s:
                if not self._navigation_guard_active:
                    # Preserve the established segment-end taptap semantics.
                    # Closed-loop retry/blocking is deliberately scoped to the
                    # navigation width guard so other stance metrics and
                    # manipulation flows are not changed by this fix.
                    pending = self._finish()
                    return pending or zero_cmd, False, "completed"
                # Never declare success from elapsed time alone.  Stop, obtain
                # a fresh double-support window and only then release pending
                # navigation motion.
                self.state = self.SETTLING
                self._settle_since = now
                self._samples.clear()
                self._waiting_for_stance = False
                return zero_cmd, True, "verifying"
            phase = int(elapsed / self.phase_s) % 2
            vx = self._start_sign * (
                self.recovery_speed if phase == 0 else -self.recovery_speed
            )
            return replace(zero_cmd, vx=vx), True, None

        # Two bounded recovery attempts failed.  Fail closed: keep navigation
        # motion pending at zero until an explicit stop/cancel resets the state.
        return zero_cmd, True, None
