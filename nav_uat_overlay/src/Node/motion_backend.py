import logging
import json
import os
import sys
import time
from typing import Any


logger = logging.getLogger(__name__)


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get") and callable(config.get):
        return config.get(key, default)
    return getattr(config, key, default)


def _profile_alias(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    if name in ("keyboard", "keyboard_like", "direct", "direct_speed"):
        return "keyboard"
    if name in ("precise", "min_step", "minstep", "reliable", "default"):
        return "precise"
    return name


def _read_profile_file(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Cannot read GR00T nav motion profile file %s: %s", path, exc)
        return None
    for key in ("motion_profile", "nav_motion_profile", "profile"):
        if key in payload:
            return str(payload[key])
    return None


def resolve_motion_profile(backend_cfg: Any) -> str:
    configured = config_get(backend_cfg, "motion_profile", "precise")
    profile_file = os.environ.get(
        "GROOT_NAV_MOTION_PROFILE_FILE",
        str(config_get(backend_cfg, "profile_file", "/tmp/groot_nav_motion_profile.json")),
    )
    raw_profile = (
        os.environ.get("GROOT_NAV_MOTION_PROFILE")
        or _read_profile_file(profile_file)
        or configured
    )
    profile = _profile_alias(raw_profile)
    if profile not in ("precise", "keyboard"):
        logger.warning("Unknown GR00T nav motion profile %r; falling back to precise", raw_profile)
        return "precise"
    return profile


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _read_taptap_marker(path: str | None) -> bool:
    if not path:
        return False
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, TypeError, ValueError):
        return False
    except Exception as exc:
        logger.warning("Cannot read GR00T nav taptap marker %s: %s", path, exc)
        return False
    return bool(payload.get("taptap_optimized", False))


def resolve_taptap_limits(backend_cfg: Any) -> dict[str, Any]:
    """Resolve gait guards enabled only by the taptap wrapper."""
    profile_file = os.environ.get(
        "GROOT_NAV_MOTION_PROFILE_FILE",
        str(config_get(backend_cfg, "profile_file", "/tmp/groot_nav_motion_profile.json")),
    )
    enabled = (
        _env_enabled("GROOT_NAV_TAPTAP_LIMITS", False)
        or _read_taptap_marker(profile_file)
    )
    values = {
        "enabled": enabled,
        "lat_cruise": float(config_get(backend_cfg, "lat_cruise", 0.25)),
        "lat_max": float(config_get(backend_cfg, "lat_max", 0.40)),
        "yaw_cruise": float(config_get(backend_cfg, "yaw_cruise", 0.40)),
        "yaw_max": float(config_get(backend_cfg, "yaw_max", 0.60)),
        "linear_slew_rate": 0.0,
        "yaw_slew_rate": 0.0,
    }
    if enabled:
        values["lat_cruise"] = min(values["lat_cruise"], 0.20)
        values["lat_max"] = min(values["lat_max"], 0.20)
        values["yaw_cruise"] = min(values["yaw_cruise"], 0.40)
        values["yaw_max"] = min(values["yaw_max"], 0.40)
        values["linear_slew_rate"] = max(
            0.0, float(config_get(backend_cfg, "taptap_linear_slew_rate", 0.80))
        )
        values["yaw_slew_rate"] = max(
            0.0, float(config_get(backend_cfg, "taptap_yaw_slew_rate", 1.20))
        )
    return values


class GrootHttpDiscreteBackend:
    """Distance/angle motion backend backed by box_demo RemoteMover.

    Discrete rotate/forward/shift calls remain the default interface. Closed-loop
    navigation can also use publish_velocity() to send short HTTP velocity holds
    for combined forward+yaw waypoint control. On G001 the bridge runs locally on
    127.0.0.1; older deployments used a remote 5080 bridge.
    """

    def __init__(self, cfg: Any, log=None):
        backend_cfg = config_get(cfg, "motion_backend", None)
        self.backend_type = str(config_get(backend_cfg, "type", "wireless_controller"))
        self.enabled = self.backend_type == "groot_http_discrete"
        self.log = log or logger
        self._mover = None
        self._initialized = False
        self._initialize_before_motion = bool(config_get(backend_cfg, "initialize_before_motion", True))
        taptap_limits = resolve_taptap_limits(backend_cfg)
        self._continuous_velocity_enabled = bool(config_get(backend_cfg, "continuous_velocity_enable", True))
        self._continuous_command_interval = max(
            0.02,
            float(config_get(backend_cfg, "continuous_command_interval", 0.10)),
        )
        self._continuous_hold_duration = max(
            self._continuous_command_interval * 1.5,
            float(config_get(backend_cfg, "continuous_hold_duration", 0.25)),
        )
        self._continuous_command_epsilon = max(
            0.0,
            float(config_get(backend_cfg, "continuous_command_epsilon", 0.01)),
        )
        self._continuous_fsm = str(config_get(backend_cfg, "continuous_fsm", "RL_FULL"))
        self._last_continuous_command = None
        self._last_continuous_send_time = 0.0
        self._continuous_applied_command = (0.0, 0.0, 0.0)
        self._continuous_applied_time = None
        self._continuous_linear_slew_rate = taptap_limits["linear_slew_rate"]
        self._continuous_yaw_slew_rate = taptap_limits["yaw_slew_rate"]

        if not self.enabled:
            return

        module_path = os.path.expanduser(
            str(config_get(backend_cfg, "box_demo_module_path", "/home/unitree/zihou/box_demo_2"))
        )
        ipc_url = str(config_get(backend_cfg, "ipc_url", "http://192.168.123.222:5001")).rstrip("/")
        timeout = float(config_get(backend_cfg, "http_timeout", 2.0))
        motion_profile = resolve_motion_profile(backend_cfg)
        warmup_time = float(config_get(backend_cfg, "warmup_time", 0.6))
        warmup_speed = float(config_get(backend_cfg, "warmup_speed", 0.15))
        if warmup_time <= 0.0:
            warmup_time = 0.6
        if warmup_speed <= 0.0:
            warmup_speed = 0.15
        if motion_profile == "keyboard":
            min_duration = 0.0
            min_distance = 0.0
        else:
            min_duration = float(config_get(backend_cfg, "min_duration", 1.0))
            min_distance = float(config_get(backend_cfg, "min_distance", 0.08))
        v_floor = float(config_get(backend_cfg, "v_floor", 0.10))
        w_floor = float(config_get(backend_cfg, "w_floor", 0.10))

        if module_path and module_path not in sys.path:
            sys.path.insert(0, module_path)

        try:
            from remote_mover import RemoteMover
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import RemoteMover from {module_path!r}; "
                "check motion_backend.box_demo_module_path"
            ) from exc

        class StrictRemoteMover(RemoteMover):
            def _get(self, path: str, **params):
                response = self._session.get(
                    f"{self.ipc_url}{path}",
                    params=params,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()

        mover_kwargs = {
            "fwd_cruise": float(config_get(backend_cfg, "fwd_cruise", 0.40)),
            "back_cruise": float(config_get(backend_cfg, "back_cruise", 0.20)),
            "lat_cruise": taptap_limits["lat_cruise"],
            "yaw_cruise": taptap_limits["yaw_cruise"],
            "fwd_max": float(config_get(backend_cfg, "fwd_max", 0.50)),
            "back_max": float(config_get(backend_cfg, "back_max", 0.20)),
            "lat_max": taptap_limits["lat_max"],
            "yaw_max": taptap_limits["yaw_max"],
            "v_floor": v_floor,
            "w_floor": w_floor,
            "min_duration": min_duration,
            "min_distance": min_distance,
            "warmup_time": warmup_time,
            "warmup_speed": warmup_speed,
            "settle_before_s": float(config_get(backend_cfg, "settle_before_s", 0.0)),
            "stop_hold_s": float(config_get(backend_cfg, "stop_hold_s", 0.4)),
            "stand_height": float(config_get(backend_cfg, "stand_height", 0.74)),
            "walk_min_height": float(config_get(backend_cfg, "walk_min_height", 0.72)),
            "auto_raise_for_walk": bool(config_get(backend_cfg, "auto_raise_for_walk", False)),
            "dist_gain": float(config_get(backend_cfg, "dist_gain", 1.0)),
            "verbose": bool(config_get(backend_cfg, "verbose", True)),
        }
        self._mover = StrictRemoteMover(ipc_url, timeout=timeout, **mover_kwargs)
        self.log.info(
            "Using GR00T HTTP discrete motion backend at %s (profile=%s, "
            "min_duration=%.2f, min_distance=%.2f, v_floor=%.2f, "
            "warmup=%.2fs@%.2fm/s, taptap_limits=%s, slew=%.2f/%.2f)",
            ipc_url,
            motion_profile,
            min_duration,
            min_distance,
            v_floor,
            warmup_time,
            warmup_speed,
            taptap_limits["enabled"],
            self._continuous_linear_slew_rate,
            self._continuous_yaw_slew_rate,
        )

    def _ensure_ready(self):
        if not self.enabled or self._mover is None:
            return
        if self._initialize_before_motion and not self._initialized:
            self._mover.initialize()
            self._initialized = True

    def _result(self, success: bool, message: str, code: int):
        return bool(success), str(message), int(code)

    def supports_continuous_velocity(self) -> bool:
        return bool(
            self.enabled
            and self._continuous_velocity_enabled
            and self._mover is not None
            and hasattr(self._mover, "_get")
        )

    @staticmethod
    def _clamp_axis(value: float, limit: float) -> float:
        limit = abs(float(limit))
        return max(-limit, min(limit, float(value)))

    @staticmethod
    def _slew_axis(value: float, target: float, rate: float, dt: float) -> float:
        if rate <= 0.0:
            return float(target)
        step = rate * max(0.0, dt)
        if value < target:
            return min(float(target), float(value) + step)
        return max(float(target), float(value) - step)

    def _reset_continuous_state(self):
        self._last_continuous_command = None
        self._last_continuous_send_time = 0.0
        self._continuous_applied_command = (0.0, 0.0, 0.0)
        self._continuous_applied_time = None

    def publish_velocity(self, forward: float, lateral: float = 0.0, yaw: float = 0.0):
        if not self.supports_continuous_velocity():
            return self._result(False, "GR00T continuous velocity is unavailable.", -1)
        try:
            self._ensure_ready()
            forward = self._clamp_axis(forward, getattr(self._mover, "fwd_max", 0.50))
            lateral = self._clamp_axis(lateral, getattr(self._mover, "lat_max", 0.30))
            yaw = self._clamp_axis(yaw, getattr(self._mover, "yaw_max", 0.60))
            if abs(forward) < 1e-6 and abs(lateral) < 1e-6 and abs(yaw) < 1e-6:
                return self.stop()

            now = time.monotonic()
            target_command = (forward, lateral, yaw)
            if self._continuous_applied_time is None:
                slew_dt = self._continuous_command_interval
            else:
                slew_dt = min(
                    self._continuous_hold_duration,
                    max(0.0, now - self._continuous_applied_time),
                )
            previous = self._continuous_applied_command
            command = (
                self._slew_axis(previous[0], target_command[0],
                                self._continuous_linear_slew_rate, slew_dt),
                self._slew_axis(previous[1], target_command[1],
                                self._continuous_linear_slew_rate, slew_dt),
                self._slew_axis(previous[2], target_command[2],
                                self._continuous_yaw_slew_rate, slew_dt),
            )
            if (
                self._last_continuous_command is not None
                and now - self._last_continuous_send_time < self._continuous_command_interval
                and all(
                    abs(command[index] - self._last_continuous_command[index])
                    <= self._continuous_command_epsilon
                    for index in range(3)
                )
            ):
                return self._result(True, "continuous velocity held.", 1)

            params = {
                "vx": command[0],
                "vy": command[1],
                "wz": command[2],
                "duration": self._continuous_hold_duration,
                "fsm": self._continuous_fsm,
            }
            height = getattr(self._mover, "_height", None)
            if height is not None:
                params["height"] = height
            self._mover._get("/cmd", **params)
            self._last_continuous_command = command
            self._last_continuous_send_time = now
            self._continuous_applied_command = command
            self._continuous_applied_time = now
            return self._result(True, "success.", 1)
        except Exception as exc:
            self.log.warning("GR00T continuous velocity failed: %s", exc)
            return self._result(False, f"GR00T continuous velocity failed: {exc}", -1)

    def forward(self, distance_m: float):
        if not self.enabled:
            return self._result(False, "GR00T backend is disabled.", -1)
        try:
            if abs(float(distance_m)) < 1e-6:
                return self._result(True, "zero forward distance skipped.", 1)
            self._ensure_ready()
            self._mover.move_forward(float(distance_m))
            return self._result(True, "success.", 1)
        except Exception as exc:
            self.log.warning("GR00T forward failed: %s", exc)
            return self._result(False, f"GR00T forward failed: {exc}", -1)

    def shift(self, distance_m: float):
        if not self.enabled:
            return self._result(False, "GR00T backend is disabled.", -1)
        try:
            if abs(float(distance_m)) < 1e-6:
                return self._result(True, "zero lateral distance skipped.", 1)
            self._ensure_ready()
            self._mover.move_left(float(distance_m))
            return self._result(True, "success.", 1)
        except Exception as exc:
            self.log.warning("GR00T lateral move failed: %s", exc)
            return self._result(False, f"GR00T lateral move failed: {exc}", -1)

    def rotate(self, angle_rad: float):
        if not self.enabled:
            return self._result(False, "GR00T backend is disabled.", -1)
        try:
            if abs(float(angle_rad)) < 1e-6:
                return self._result(True, "zero rotation skipped.", 1)
            self._ensure_ready()
            self._mover.rotate(float(angle_rad))
            return self._result(True, "success.", 1)
        except Exception as exc:
            self.log.warning("GR00T rotate failed: %s", exc)
            return self._result(False, f"GR00T rotate failed: {exc}", -1)

    def stop(self):
        if not self.enabled:
            return self._result(False, "GR00T backend is disabled.", -1)
        try:
            # A stop is a safety command: bypass slew limiting and clear its state.
            self._reset_continuous_state()
            if self._mover is not None:
                self._mover.stop()
            return self._result(True, "success.", 1)
        except Exception as exc:
            self.log.warning("GR00T stop failed: %s", exc)
            return self._result(False, f"GR00T stop failed: {exc}", -1)

    def shutdown(self):
        return self.stop()
