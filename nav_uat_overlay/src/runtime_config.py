from __future__ import annotations

from typing import Any


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get") and callable(config.get):
        return config.get(key, default)
    return getattr(config, key, default)


def model_planner_enabled(cfg: Any) -> bool:
    local_planner = config_get(cfg, "local_planner", None)
    return bool(config_get(local_planner, "enbale_model_planner", False))


def resolve_capture_depth(rgbd_cfg: Any, require_depth: bool = False) -> bool:
    if require_depth:
        return True
    configured = config_get(rgbd_cfg, "capture_depth", None)
    if configured is not None:
        return bool(configured)
    # Legacy configs predate both keys and always captured/published depth.
    return bool(config_get(rgbd_cfg, "publish_ros_topics", True))


def resolve_executor_threads(cfg: Any, default: int = 2) -> int:
    runtime = config_get(cfg, "runtime", None)
    value = int(config_get(runtime, "executor_threads", default))
    return max(1, min(value, 4))


def diagnostics_enabled(cfg: Any, key: str, default: bool = False) -> bool:
    runtime = config_get(cfg, "runtime", None)
    diagnostics = config_get(runtime, "diagnostics", None)
    return bool(config_get(diagnostics, key, default))


def resolve_frame_max_age(cfg: Any, section: str, default: float = 1.0) -> float | None:
    owner = config_get(cfg, section, None)
    value = config_get(owner, "frame_max_age", default)
    if value is None:
        return None
    return max(0.0, float(value))
