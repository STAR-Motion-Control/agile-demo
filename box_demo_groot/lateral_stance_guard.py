#!/usr/bin/env python3
"""Command-side minimum foot-spacing guard for lateral locomotion."""

from __future__ import annotations

from dataclasses import replace


class LateralStanceGuard:
    """Stop accepting lateral commands before foot spacing reaches a hard floor.

    This cannot directly place a learned-policy foot.  It therefore uses a
    predictive margin and latches after intervention, preventing repeated
    lateral commands from driving an already narrow/crossed stance further.
    """

    def __init__(
        self,
        *,
        min_width: float = 0.15,
        guard_margin: float = 0.03,
        prediction_s: float = 0.0,
        release_margin: float = 0.05,
        rate_alpha: float = 0.35,
    ):
        self.min_width = max(0.0, float(min_width))
        self.guard_width = self.min_width + max(0.0, float(guard_margin))
        self.prediction_s = max(0.0, float(prediction_s))
        self.release_width = self.min_width + max(
            float(release_margin), float(guard_margin)
        )
        self.rate_alpha = max(0.0, min(1.0, float(rate_alpha)))
        self.blocked = False
        self._last_t = None
        self._last_width = None
        self._width_rate = 0.0

    @property
    def width_rate(self) -> float:
        return self._width_rate

    def _observe(self, now: float, stance) -> None:
        if stance is None:
            return
        width = float(stance.width)
        if self._last_t is not None and now > self._last_t:
            raw_rate = (width - self._last_width) / (now - self._last_t)
            raw_rate = max(-2.0, min(2.0, raw_rate))
            self._width_rate += self.rate_alpha * (raw_rate - self._width_rate)
        self._last_t = float(now)
        self._last_width = width

    def update(self, now: float, cmd, stance):
        """Return ``(effective_cmd, blocked, event)``.

        Once tripped, non-lateral commands remain available, but every lateral
        command is zeroed until an explicit zero-lateral interval observes a
        recovered spacing.  This avoids an automatic reversal whose swing foot
        cannot be selected through the velocity-only policy interface.
        """
        self._observe(float(now), stance)
        lateral = abs(float(cmd.vy)) > 1e-6
        width = None if stance is None else float(stance.width)

        if self.blocked:
            if not lateral and width is not None and width >= self.release_width:
                self.blocked = False
                return cmd, False, "released"
            if lateral:
                return replace(cmd, vy=0.0), True, None
            return cmd, True, None

        if not lateral or width is None:
            return cmd, False, None

        predicted_width = width + min(0.0, self._width_rate) * self.prediction_s
        if width <= self.guard_width or predicted_width <= self.guard_width:
            self.blocked = True
            return replace(cmd, vy=0.0), True, "blocked"
        return cmd, False, None
