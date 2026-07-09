import logging
import json
import os
import sys
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


class GrootHttpDiscreteBackend:
    """Distance/angle motion backend backed by box_demo RemoteMover.

    This deliberately exposes only discrete moves. Navigation continues to
    plan in rotate(theta) + forward(distance); the GR00T adapter converts those
    requests to timed velocity holds through the HTTP IPC bridge. On G001 the
    bridge runs locally on 127.0.0.1; older deployments used a remote 5080 bridge.
    """

    def __init__(self, cfg: Any, log=None):
        backend_cfg = config_get(cfg, "motion_backend", None)
        self.backend_type = str(config_get(backend_cfg, "type", "wireless_controller"))
        self.enabled = self.backend_type == "groot_http_discrete"
        self.log = log or logger
        self._mover = None
        self._initialized = False
        self._initialize_before_motion = bool(config_get(backend_cfg, "initialize_before_motion", True))

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
            min_duration = float(config_get(backend_cfg, "min_duration", 1.5))
            min_distance = float(config_get(backend_cfg, "min_distance", 0.08))

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
            "lat_cruise": float(config_get(backend_cfg, "lat_cruise", 0.25)),
            "yaw_cruise": float(config_get(backend_cfg, "yaw_cruise", 0.40)),
            "fwd_max": float(config_get(backend_cfg, "fwd_max", 0.50)),
            "back_max": float(config_get(backend_cfg, "back_max", 0.20)),
            "lat_max": float(config_get(backend_cfg, "lat_max", 0.40)),
            "yaw_max": float(config_get(backend_cfg, "yaw_max", 0.60)),
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
            "min_duration=%.2f, min_distance=%.2f, warmup=%.2fs@%.2fm/s)",
            ipc_url,
            motion_profile,
            min_duration,
            min_distance,
            warmup_time,
            warmup_speed,
        )

    def _ensure_ready(self):
        if not self.enabled or self._mover is None:
            return
        if self._initialize_before_motion and not self._initialized:
            self._mover.initialize()
            self._initialized = True

    def _result(self, success: bool, message: str, code: int):
        return bool(success), str(message), int(code)

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
            if self._mover is not None:
                self._mover.stop()
            return self._result(True, "success.", 1)
        except Exception as exc:
            self.log.warning("GR00T stop failed: %s", exc)
            return self._result(False, f"GR00T stop failed: {exc}", -1)

    def shutdown(self):
        return self.stop()
