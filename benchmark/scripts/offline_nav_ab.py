#!/usr/bin/env python3
"""Offline navigation A/B contract and micro-benchmark.

This tool reads source files from Git objects and executes them only with fake
clocks and fake command transports.  It never imports ROS, DDS, a robot SDK, or
opens a network/Unix socket.  The default references are the preserved live-demo
baseline, the commit that imported the complete 5080 navigation tree without a
behavioral refactor, and the runtime-refactor candidate.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable


DEFAULT_BASELINE = "49f2631"
DEFAULT_NAV_REFERENCE = "eb1b6e9"
DEFAULT_CANDIDATE = "HEAD"
MOVER_PATH = "box_demo_groot/groot_mover.py"
MOTION_BACKEND_PATH = "nav_uat_overlay/src/Node/motion_backend.py"


class ContractError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


class GitSources:
    def __init__(self, repository: Path):
        self.repository = repository
        self._cache: dict[tuple[str, str], str] = {}

    def source(self, ref: str, path: str) -> str:
        key = (ref, path)
        if key not in self._cache:
            try:
                self._cache[key] = subprocess.check_output(
                    ["git", "show", f"{ref}:{path}"],
                    cwd=self.repository,
                    text=True,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.strip() if exc.stderr else str(exc)
                raise ContractError(f"cannot read {ref}:{path}: {detail}") from exc
        return self._cache[key]

    def commit(self, ref: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", f"{ref}^{{commit}}"],
                cwd=self.repository,
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if exc.stderr else str(exc)
            raise ContractError(f"cannot resolve Git ref {ref!r}: {detail}") from exc

    def parent(self, ref: str) -> str:
        return self.commit(f"{ref}^")


@contextlib.contextmanager
def deterministic_mover_environment():
    names = (
        "GROOT_BOX_RUNTIME_CONFIG",
        "GROOT_MOTION_BUS_SOCKET",
        "GROOT_STAND_HEIGHT",
        "GROOT_WARMUP_SPEED",
        "GROOT_WARMUP_TIME",
    )
    before = {name: os.environ.get(name) for name in names}
    os.environ["GROOT_BOX_RUNTIME_CONFIG"] = "/__offline_nav_ab_missing__.json"
    os.environ.pop("GROOT_MOTION_BUS_SOCKET", None)
    os.environ["GROOT_STAND_HEIGHT"] = "0.76"
    os.environ["GROOT_WARMUP_SPEED"] = "0.15"
    os.environ["GROOT_WARMUP_TIME"] = "0.0"
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_module(source: str, name: str, filename: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def load_git_module(sources: GitSources, ref: str, path: str, tag: str):
    digest = hashlib.sha256(f"{ref}:{path}:{tag}".encode()).hexdigest()[:12]
    name = f"_offline_nav_ab_{tag}_{digest}"
    with deterministic_mover_environment():
        return load_module(sources.source(ref, path), name, f"{ref}:{path}")


def assert_close(actual: Any, expected: Any, path: str = "value") -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        require(actual is expected, f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        require(
            math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11),
            f"{path}: {actual!r} != {expected!r}",
        )
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        require(set(actual) == set(expected), f"{path}: keys differ")
        for key in sorted(actual):
            assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f"{path}: lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, f"{path}[{index}]")
        return
    require(actual == expected, f"{path}: {actual!r} != {expected!r}")


class VirtualClock:
    def __init__(self):
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += max(0.0, float(duration))


class VirtualEvent:
    def __init__(self, clock: VirtualClock, initial: bool = False):
        self.clock = clock
        self.value = bool(initial)

    def is_set(self) -> bool:
        return self.value

    def set(self) -> None:
        self.value = True

    def clear(self) -> None:
        self.value = False

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None and not self.value:
            self.clock.sleep(timeout)
        return self.value


def plan_tuple(plan: Any) -> tuple[float, float, float, bool]:
    return (
        float(plan.speed),
        float(plan.duration),
        float(plan.expected),
        bool(plan.floored),
    )


def normalize_write(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    def positional(index: int, key: str, default: Any):
        return args[index] if len(args) > index else kwargs.get(key, default)

    return {
        "fsm": positional(1, "fsm", None),
        "forward": float(positional(2, "forward", 0.0)),
        "lateral": float(positional(3, "lateral", 0.0)),
        "yaw": float(positional(4, "yaw", 0.0)),
        "height": float(kwargs.get("height", 0.76)),
        "duration": (
            None if kwargs.get("duration") is None else float(kwargs["duration"])
        ),
        "allow_recovery": bool(kwargs.get("allow_recovery", False)),
        "defer_recovery": bool(kwargs.get("defer_recovery", False)),
        "estop": bool(kwargs.get("estop", False)),
        "limp": bool(kwargs.get("limp", False)),
    }


MOVER_KWARGS = {
    "stand_height": 0.74,
    "fwd_cruise": 0.40,
    "back_cruise": 0.20,
    "lat_cruise": 0.20,
    "yaw_cruise": 0.40,
    "fwd_max": 0.50,
    "back_max": 0.20,
    "lat_max": 0.40,
    "yaw_max": 0.60,
    "v_floor": 0.10,
    "w_floor": 0.10,
    "min_duration": 1.0,
    "min_distance": 0.08,
    "warmup_time": 0.6,
    "warmup_speed": 0.15,
    "settle_before_s": 0.0,
    "stop_hold_s": 0.4,
    "walk_min_height": 0.72,
    "auto_raise_for_walk": False,
    "dist_gain": 1.0,
    "recover_each_move": False,
    "refresh_hz": 20.0,
    "verbose": False,
}


def trace_move(module: Any, method: str, value: float):
    clock = VirtualClock()
    module.time = clock
    trace: list[dict[str, Any]] = []
    module.write_command = lambda *args, **kwargs: trace.append(
        normalize_write(args, kwargs)
    )
    mover = module.GrootMover(cmd_file="/__offline_no_write__.json", **MOVER_KWARGS)
    if hasattr(mover, "_motion_cancel"):
        mover._motion_cancel = VirtualEvent(clock)
    plan = getattr(mover, method)(value)
    return plan_tuple(plan), trace


def validate_motion_math_and_traces(
    sources: GitSources, baseline: str, candidate: str
) -> dict[str, Any]:
    baseline_module = load_git_module(sources, baseline, MOVER_PATH, "mover_base")
    candidate_module = load_git_module(sources, candidate, MOVER_PATH, "mover_candidate")

    distances = [
        -2.0, -1.0, -0.5, -0.2, -0.081, -0.08, -0.079, -0.03,
        -0.001, 0.0, 0.001, 0.03, 0.079, 0.08, 0.081, 0.2, 0.5, 1.0, 2.0,
    ]
    math_cases = 0
    for distance in distances:
        left = baseline_module.solve_linear(distance, 0.4, 0.1, 0.5, 1.0)
        right = candidate_module.solve_linear(distance, 0.4, 0.1, 0.5, 1.0)
        assert_close(plan_tuple(right), plan_tuple(left), f"solve_linear[{distance}]")
        math_cases += 1
    for angle in [
        -math.pi, -math.pi / 2, -0.25, -0.02, -0.001, 0.0,
        0.001, 0.02, 0.25, math.pi / 2, math.pi,
    ]:
        left = baseline_module.solve_yaw(angle, 0.4, 0.1, 0.6, 1.0)
        right = candidate_module.solve_yaw(angle, 0.4, 0.1, 0.6, 1.0)
        assert_close(plan_tuple(right), plan_tuple(left), f"solve_yaw[{angle}]")
        math_cases += 1

    move_cases = {
        "move_forward": [-0.25, -0.03, 0.0, 0.03, 0.08, 0.25, 1.0],
        "move_left": [-0.15, -0.02, 0.0, 0.02, 0.05, 0.15],
        "rotate": [-math.pi / 2, -0.02, 0.0, 0.02, math.pi / 2],
    }
    trace_cases = 0
    command_frames = 0
    for method, values in move_cases.items():
        for value in values:
            baseline_plan, baseline_trace = trace_move(baseline_module, method, value)
            candidate_plan, candidate_trace = trace_move(candidate_module, method, value)
            assert_close(candidate_plan, baseline_plan, f"{method}[{value}].plan")
            assert_close(candidate_trace, baseline_trace, f"{method}[{value}].trace")
            trace_cases += 1
            command_frames += len(candidate_trace)

    return {
        "math_cases": math_cases,
        "trace_cases": trace_cases,
        "candidate_command_frames_compared": command_frames,
        "physical_speed_contract": {
            "forward_cruise_mps": 0.40,
            "backward_cruise_mps": 0.20,
            "lateral_cruise_mps": 0.20,
            "yaw_cruise_radps": 0.40,
        },
    }


def class_method_asts(source: str, class_name: str) -> dict[str, str]:
    tree = ast.parse(source)
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    require(class_node is not None, f"class {class_name} is missing")
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


EXPECTED_METHOD_DELTAS = {
    "nav_uat_overlay/src/Node/action_planner.py": {
        "class": "ActionPlanner",
        "changed": {"__init__", "get_action_from_action_model"},
        "added": set(),
        "removed": {"rgbd_callback"},
    },
    "nav_uat_overlay/src/Node/path_planner.py": {
        "class": "PathPlanner",
        "changed": {"__init__", "plot", "save_fig"},
        "added": set(),
        "removed": set(),
    },
    "nav_uat_overlay/src/Node/action_executor.py": {
        "class": "ActionExecutorClient",
        "changed": {"_publish_motion", "forward", "rotate", "shift", "shutdown"},
        "added": {"_motion_cancel_token"},
        "removed": set(),
    },
    "nav_uat_overlay/src/Node/step_control.py": {
        "class": "StepControlClient",
        "changed": {"forward", "rotate", "shift"},
        "added": {"shutdown"},
        "removed": set(),
    },
    "nav_uat_overlay/src/Node/__init__.py": {
        "class": "NodeManager",
        "changed": {"__init__", "navigation", "shutdown"},
        "added": set(),
        "removed": set(),
    },
}


def extracted_action_planner(source: str, tag: str):
    tree = ast.parse(source)
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActionPlanner"
    )
    wanted = {"path_to_actions", "actions_to_local_path", "normalize_angle"}
    methods = [
        copy.deepcopy(node)
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    require({node.name for node in methods} == wanted, "path conversion methods missing")
    extracted = ast.ClassDef(
        name=f"ExtractedActionPlanner{tag}",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module_tree = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    namespace = {"math": math}
    exec(compile(module_tree, f"<action-planner-{tag}>", "exec"), namespace)
    return namespace[extracted.name]


def validate_path_contract(
    sources: GitSources, nav_reference: str, candidate: str
) -> dict[str, Any]:
    method_counts = 0
    for path, expected in EXPECTED_METHOD_DELTAS.items():
        before = class_method_asts(sources.source(nav_reference, path), expected["class"])
        after = class_method_asts(sources.source(candidate, path), expected["class"])
        common = set(before) & set(after)
        changed = {name for name in common if before[name] != after[name]}
        added = set(after) - set(before)
        removed = set(before) - set(after)
        require(changed == expected["changed"], f"unexpected changed methods in {path}: {sorted(changed)}")
        require(added == expected["added"], f"unexpected added methods in {path}: {sorted(added)}")
        require(removed == expected["removed"], f"unexpected removed methods in {path}: {sorted(removed)}")
        method_counts += len(common - changed)

    unchanged_files = (
        "nav_uat_overlay/src/navigation/controller.py",
        "nav_uat_overlay/src/navigation/task_service.py",
    )
    for path in unchanged_files:
        require(
            sources.source(nav_reference, path) == sources.source(candidate, path),
            f"navigation orchestration changed unexpectedly: {path}",
        )

    before_cls = extracted_action_planner(
        sources.source(nav_reference, "nav_uat_overlay/src/Node/action_planner.py"),
        "Baseline",
    )
    after_cls = extracted_action_planner(
        sources.source(candidate, "nav_uat_overlay/src/Node/action_planner.py"),
        "Candidate",
    )
    fixtures = [
        ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], None),
        ([(0.0, 0.0, 0.3), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], 1.2),
        ([(1.0, 2.0, -1.0), (0.5, 1.5, 0.0), (-0.5, 1.5, 0.0)], -2.7),
    ]
    behavior_cases = 0
    for path, goal_theta in fixtures:
        before_obj = before_cls()
        after_obj = after_cls()
        before_actions = before_obj.path_to_actions(path, goal_theta)
        after_actions = after_obj.path_to_actions(path, goal_theta)
        assert_close(after_actions, before_actions, "path_to_actions")
        before_obj.path_planner = types.SimpleNamespace(current_position=list(path[0]))
        after_obj.path_planner = types.SimpleNamespace(current_position=list(path[0]))
        assert_close(
            after_obj.actions_to_local_path(after_actions),
            before_obj.actions_to_local_path(before_actions),
            "actions_to_local_path",
        )
        behavior_cases += 1

    return {
        "unchanged_core_methods": method_counts,
        "unchanged_orchestration_files": len(unchanged_files),
        "path_behavior_fixtures": behavior_cases,
    }


_YAML_SCALAR = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):(?:\s+(.*?))?\s*$")


def yaml_scalar_paths(source: str) -> dict[str, str]:
    """Extract scalar mapping paths from this project's simple Hydra YAML."""
    stack: list[tuple[int, str]] = []
    output: dict[str, str] = {}
    for raw_line in source.splitlines():
        line = raw_line.split(" #", 1)[0].rstrip()
        match = _YAML_SCALAR.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        key = match.group(2)
        value = match.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([entry[1] for entry in stack] + [key])
        if value is None or value == "":
            stack.append((indent, key))
        else:
            output[path] = value.strip()
    return output


def validate_config_contract(
    sources: GitSources, nav_reference: str, candidate: str
) -> dict[str, Any]:
    speed_keys = (
        "motion_backend.stand_height",
        "motion_backend.fwd_cruise",
        "motion_backend.back_cruise",
        "motion_backend.lat_cruise",
        "motion_backend.yaw_cruise",
        "motion_backend.fwd_max",
        "motion_backend.back_max",
        "motion_backend.lat_max",
        "motion_backend.yaw_max",
        "motion_backend.min_duration",
        "motion_backend.min_distance",
        "motion_backend.v_floor",
        "motion_backend.w_floor",
        "motion_backend.warmup_time",
        "motion_backend.warmup_speed",
        "motion_backend.stop_hold_s",
        "motion_backend.walk_min_height",
        "motion_backend.dist_gain",
    )
    configs: dict[str, Any] = {}
    total_control_keys = 0
    for config_name in ("config", "config_bk"):
        path = f"nav_uat_overlay/src/{config_name}.yaml"
        baseline = yaml_scalar_paths(sources.source(nav_reference, path))
        current = yaml_scalar_paths(sources.source(candidate, path))
        for key in speed_keys:
            require(
                key in baseline and key in current,
                f"missing navigation speed key in {config_name}: {key}",
            )
            require(
                current[key] == baseline[key],
                f"navigation speed changed in {config_name}: {key}",
            )

        baseline_control = {
            key: value
            for key, value in baseline.items()
            if key.startswith("motion_control.")
        }
        current_control = {
            key: value
            for key, value in current.items()
            if key.startswith("motion_control.")
        }
        require(
            current_control == baseline_control,
            f"closed-loop navigation tuning changed in {config_name}",
        )
        require(
            current.get("local_planner.enbale_model_planner")
            == baseline.get("local_planner.enbale_model_planner"),
            f"local planner enablement changed in {config_name}",
        )
        if config_name == "config_bk":
            require(current.get("map_path") == baseline.get("map_path"), "map path changed")
        else:
            require(
                baseline.get("map_path") == "map/sustech_demon"
                and current.get("map_path") == "map/sustech_demo",
                "unexpected config.yaml map-path migration",
            )
        require(
            current.get("rgbd_server.fps") == "30",
            f"camera capture rate changed in {config_name}",
        )
        require(
            current.get("rgbd_server.publish_ros_topics") == "true",
            f"legacy raw ROS image topics are not enabled in {config_name}",
        )
        require(
            current.get("rgbd_server.capture_depth") == "true",
            f"legacy depth capture is not enabled in {config_name}",
        )
        total_control_keys += len(baseline_control)
        configs[config_name] = {
            "unchanged_speed_keys": len(speed_keys),
            "unchanged_closed_loop_keys": len(baseline_control),
            "map_path": current.get("map_path"),
            "model_planner_enabled": current.get(
                "local_planner.enbale_model_planner"
            ),
            "camera_fps": int(current.get("rgbd_server.fps", "30")),
            "raw_ros_topics_default": current.get(
                "rgbd_server.publish_ros_topics"
            ),
            "depth_capture_default": current.get("rgbd_server.capture_depth"),
        }

    return {
        "configs": configs,
        "unchanged_speed_keys_total": len(speed_keys) * len(configs),
        "unchanged_closed_loop_keys_total": total_control_keys,
        "baseline_camera_fps_hardcoded": 30,
    }


def validate_frame_hub_contract(sources: GitSources, candidate: str) -> dict[str, Any]:
    module = load_git_module(
        sources, candidate, "nav_uat_overlay/src/frame_hub.py", "frame_contract"
    )
    now = [100.0]
    hub = module.CameraFrameHub(clock=lambda: now[0])
    require(hub.latest() is None, "empty frame hub returned a frame")

    first_rgb = object()
    first_depth = object()
    first = hub.publish(
        first_rgb,
        first_depth,
        captured_at=now[0],
        source="offline-rgbd",
        depth_scale=0.001,
    )
    require(first.sequence == 1, "first frame sequence is not one")
    require(first.rgb is first_rgb and first.depth is first_depth, "RGB-D pair was split")
    require(hub.latest(require_depth=True) is first, "fresh RGB-D frame unavailable")

    now[0] += 1.01
    require(hub.latest(max_age_s=1.0) is None, "stale frame passed max-age gate")
    second_rgb = object()
    second = hub.publish(second_rgb, source="offline-rgb")
    require(second.sequence == 2, "frame sequence did not advance")
    require(hub.latest() is second, "latest-only hub did not replace the old frame")
    require(
        hub.latest(require_depth=True) is None,
        "depth request reused depth from an older RGB frame",
    )
    return {
        "frames_published": 2,
        "latest_sequence": second.sequence,
        "stale_frame_rejected": True,
        "cross_frame_depth_reuse": False,
    }


class SetToken:
    def is_set(self) -> bool:
        return True


def make_backend(module: Any, bus: bool, record: bool = True):
    trace: list[dict[str, Any]] = []

    class FakeMover:
        fwd_max = 0.50
        lat_max = 0.40
        yaw_max = 0.60
        _height = 0.74

        def _get(self, path: str, **params):
            if record:
                trace.append({"kind": path, **params})
            return {"ok": True}

        def publish_velocity(self, forward, lateral, yaw, **kwargs):
            if record:
                trace.append(
                    {
                        "kind": "/cmd",
                        "vx": float(forward),
                        "vy": float(lateral),
                        "wz": float(yaw),
                        "duration": 0.25,
                        "fsm": "RL_FULL",
                        "allow_recovery": int(bool(kwargs.get("allow_recovery"))),
                        "defer_recovery": int(bool(kwargs.get("defer_recovery"))),
                        "height": self._height,
                    }
                )

        def stop(self):
            if record:
                trace.append({"kind": "/stop", "height": self._height})

        def finish_segment(self):
            return None

    backend = object.__new__(module.GrootHttpDiscreteBackend)
    backend.backend_type = "groot_motion_bus" if bus else "groot_http_discrete"
    backend._bus_enabled = bool(bus)
    backend.enabled = True
    backend.log = types.SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    backend._mover = FakeMover()
    backend._initialized = True
    backend._initialize_before_motion = False
    backend._continuous_velocity_enabled = True
    backend._continuous_command_interval = 0.10
    backend._continuous_hold_duration = 0.25
    backend._continuous_command_epsilon = 0.01
    backend._continuous_fsm = "RL_FULL"
    backend._continuous_linear_slew_rate = 0.0
    backend._continuous_yaw_slew_rate = 0.0
    backend._reset_continuous_state()
    return backend, trace


def normalize_backend_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in trace:
        if item["kind"] == "/stop":
            normalized.append({"kind": "/stop", "height": float(item["height"])})
            continue
        normalized.append(
            {
                "kind": "/cmd",
                "vx": float(item["vx"]),
                "vy": float(item["vy"]),
                "wz": float(item["wz"]),
                "duration": float(item["duration"]),
                "fsm": str(item["fsm"]),
                "allow_recovery": int(item["allow_recovery"]),
                "defer_recovery": int(item["defer_recovery"]),
                "height": float(item["height"]),
            }
        )
    return normalized


def validate_continuous_mapping_and_cancel(
    sources: GitSources, baseline: str, candidate: str
) -> dict[str, Any]:
    baseline_module = load_git_module(
        sources, baseline, MOTION_BACKEND_PATH, "backend_base"
    )
    candidate_module = load_git_module(
        sources, candidate, MOTION_BACKEND_PATH, "backend_candidate"
    )
    before, before_trace = make_backend(baseline_module, bus=False)
    after, after_trace = make_backend(candidate_module, bus=True)

    base_clock = VirtualClock()
    candidate_clock = VirtualClock()
    baseline_module.time = base_clock
    candidate_module.time = candidate_clock
    calls = [
        (0.0, 0.40, 0.20, 0.40),
        (0.01, 0.40, 0.20, 0.40),
        (0.11, 0.80, -0.80, 0.90),
        (0.11, 0.0, 0.0, 0.0),
    ]
    return_codes = []
    for advance, forward, lateral, yaw in calls:
        base_clock.sleep(advance)
        candidate_clock.sleep(advance)
        left = before.publish_velocity(forward, lateral, yaw)
        right = after.publish_velocity(forward, lateral, yaw)
        require(left[0] == right[0] and left[2] == right[2], "backend result code changed")
        return_codes.append(int(right[2]))

    assert_close(
        normalize_backend_trace(after_trace),
        normalize_backend_trace(before_trace),
        "continuous command mapping",
    )
    count_before_cancel = len(after_trace)
    cancelled = after.publish_velocity(0.2, 0.0, 0.0, cancel_event=SetToken())
    require(cancelled[0] is False and cancelled[2] == -2, "preset cancel was not rejected")
    require(len(after_trace) == count_before_cancel, "cancelled command reached the transport")

    candidate_mover = load_git_module(
        sources, candidate, MOVER_PATH, "mover_cancel_candidate"
    )
    clock = VirtualClock()
    candidate_mover.time = clock
    writes: list[Any] = []
    candidate_mover.write_command = lambda *args, **kwargs: writes.append((args, kwargs))
    mover = candidate_mover.GrootMover(
        cmd_file="/__offline_no_write__.json", **MOVER_KWARGS
    )
    mover._motion_cancel = VirtualEvent(clock)
    try:
        mover.move_forward(0.2, cancel_event=SetToken())
    except candidate_mover.MotionCancelled:
        pass
    else:
        raise ContractError("cancelled discrete move did not raise MotionCancelled")
    require(not writes, "cancelled discrete move emitted a command")

    token = candidate_module.CombinedCancelToken(VirtualEvent(clock), SetToken())
    require(token.is_set(), "combined cancel token ignored a set source")
    return {
        "continuous_calls": len(calls),
        "transport_frames": len(after_trace),
        "return_codes": return_codes,
        "cancelled_transport_frames": 0,
    }


def percentile(samples: list[int], fraction: float) -> int:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return int(ordered[index])


def benchmark(operation: Callable[[int], None], iterations: int) -> dict[str, float]:
    warmup = min(1000, max(100, iterations // 10))
    for index in range(warmup):
        operation(index)
    samples: list[int] = []
    started = time.perf_counter_ns()
    for index in range(iterations):
        before = time.perf_counter_ns()
        operation(index)
        samples.append(time.perf_counter_ns() - before)
    total = time.perf_counter_ns() - started
    mean_ns = statistics.fmean(samples)
    return {
        "iterations": iterations,
        "p50_us": percentile(samples, 0.50) / 1000.0,
        "p95_us": percentile(samples, 0.95) / 1000.0,
        "mean_us": mean_ns / 1000.0,
        "throughput_per_s": iterations * 1e9 / total,
    }


def validate_microbenchmarks(
    sources: GitSources,
    baseline: str,
    candidate: str,
    iterations: int,
) -> dict[str, Any]:
    baseline_module = load_git_module(
        sources, baseline, MOTION_BACKEND_PATH, "perf_backend_base"
    )
    candidate_module = load_git_module(
        sources, candidate, MOTION_BACKEND_PATH, "perf_backend_candidate"
    )
    baseline_module.time = time
    candidate_module.time = time
    before, _ = make_backend(baseline_module, bus=False, record=False)
    after, _ = make_backend(candidate_module, bus=True, record=False)
    before_metric = benchmark(
        lambda index: before.publish_velocity(0.4 if index % 2 else -0.4, 0.2, 0.3),
        iterations,
    )
    after_metric = benchmark(
        lambda index: after.publish_velocity(0.4 if index % 2 else -0.4, 0.2, 0.3),
        iterations,
    )
    backend_limit_us = max(100.0, before_metric["p95_us"] * 3.0)
    require(
        after_metric["p95_us"] <= backend_limit_us,
        f"candidate backend p95 {after_metric['p95_us']:.1f}us exceeds {backend_limit_us:.1f}us",
    )
    require(
        after_metric["throughput_per_s"] >= 5000.0,
        "candidate backend pure-Python throughput is below 5000/s",
    )

    bus_module = load_git_module(
        sources, candidate, "onboard_runtime/motion_bus.py", "perf_motion_bus"
    )
    client = object.__new__(bus_module.MotionBusClient)
    client.source = "navigation.offline"
    client.logical_source = "navigation"
    client.socket_path = "/__offline_no_socket__.sock"
    client._sequence = 0
    client._lock = threading.Lock()
    encoded_size = [0]

    def encode_only(payload):
        encoded_size[0] = len(
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        )

    client._send = encode_only
    bus_metric = benchmark(
        lambda index: client.publish(
            forward=0.4 if index % 2 else -0.4,
            lateral=0.2,
            yaw=0.3,
            height=0.74,
            lease_s=0.25,
            defer_recovery=True,
        ),
        iterations,
    )
    require(bus_metric["p95_us"] <= 250.0, "motion-bus encode p95 exceeds 250us")
    require(bus_metric["throughput_per_s"] >= 4000.0, "motion-bus encode throughput below 4000/s")

    frame_module = load_git_module(
        sources, candidate, "nav_uat_overlay/src/frame_hub.py", "perf_frame_hub"
    )
    frame_clock = [1000.0]
    hub = frame_module.CameraFrameHub(clock=lambda: frame_clock[0])
    rgb = object()

    def frame_operation(index: int):
        frame_clock[0] += 1.0 / 30.0
        hub.publish(rgb, source="offline")
        require(hub.latest(max_age_s=0.1) is not None, f"latest frame lost at {index}")

    frame_metric = benchmark(frame_operation, iterations)
    require(frame_metric["p95_us"] <= 100.0, "latest-only frame hub p95 exceeds 100us")
    require(frame_metric["throughput_per_s"] >= 10000.0, "frame hub throughput below 10000/s")
    require(hub.latest().sequence == iterations + min(1000, max(100, iterations // 10)), "frame sequence is not latest-only")

    planner_cls = extracted_action_planner(
        sources.source(candidate, "nav_uat_overlay/src/Node/action_planner.py"),
        "Performance",
    )
    planner = planner_cls()
    path = [(index * 0.02, math.sin(index * 0.03), 0.0) for index in range(100)]
    path_metric = benchmark(
        lambda _index: planner.path_to_actions(path, goal_theta=0.7),
        max(1000, iterations // 4),
    )
    require(path_metric["p95_us"] <= 2000.0, "100-waypoint conversion p95 exceeds 2ms")

    return {
        "baseline_backend_fake_transport": before_metric,
        "candidate_backend_fake_transport": after_metric,
        "candidate_motion_bus_encode_no_socket": {
            **bus_metric,
            "payload_bytes": encoded_size[0],
        },
        "candidate_frame_hub_publish_latest": frame_metric,
        "candidate_path_to_actions_100_waypoints": path_metric,
        "thresholds": {
            "backend_p95_us": backend_limit_us,
            "backend_min_throughput_per_s": 5000.0,
            "motion_bus_encode_p95_us": 250.0,
            "motion_bus_encode_min_throughput_per_s": 4000.0,
            "frame_hub_p95_us": 100.0,
            "frame_hub_min_throughput_per_s": 10000.0,
            "path_100_waypoints_p95_us": 2000.0,
        },
    }


def run_case(report: dict[str, Any], name: str, function: Callable[[], Any]) -> None:
    try:
        details = function()
    except Exception as exc:
        report["checks"][name] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        report["failures"].append(name)
    else:
        report["checks"][name] = {"passed": True, "details": details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--nav-reference-ref", default=DEFAULT_NAV_REFERENCE)
    parser.add_argument("--candidate-ref", default=DEFAULT_CANDIDATE)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.iterations >= 1000, "--iterations must be at least 1000")
    repository = args.repo.resolve()
    sources = GitSources(repository)
    refs = {
        "baseline": sources.commit(args.baseline_ref),
        "nav_reference": sources.commit(args.nav_reference_ref),
        "candidate": sources.commit(args.candidate_ref),
    }
    require(
        sources.parent(args.nav_reference_ref) == refs["baseline"],
        "navigation reference must be the direct import child of the baseline",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "offline_only": True,
        "opened_sockets": False,
        "imported_ros_dds_robot_sdk": False,
        "refs": refs,
        "checks": {},
        "failures": [],
        "limitations": [
            "No ROS/DDS/camera/model/robot process is started.",
            "49f2631 lacks the full navigation tree; eb1b6e9 is its direct child that only imports the complete 5080 navigation staging.",
            "No physical camera is attached, so RealSense 640x480 RGB-D at 30 fps is verified as a configuration/API contract rather than measured hardware throughput.",
            "Micro-benchmarks exclude kernel scheduling, HTTP, Unix datagram, ROS, model inference, and hardware I/O.",
        ],
    }
    run_case(
        report,
        "motion_math_and_discrete_command_trace",
        lambda: validate_motion_math_and_traces(
            sources, args.baseline_ref, args.candidate_ref
        ),
    )
    run_case(
        report,
        "path_and_navigation_logic_contract",
        lambda: validate_path_contract(
            sources, args.nav_reference_ref, args.candidate_ref
        ),
    )
    run_case(
        report,
        "navigation_config_and_speed_contract",
        lambda: validate_config_contract(
            sources, args.nav_reference_ref, args.candidate_ref
        ),
    )
    run_case(
        report,
        "latest_only_camera_frame_contract",
        lambda: validate_frame_hub_contract(sources, args.candidate_ref),
    )
    run_case(
        report,
        "continuous_velocity_mapping_and_cancel",
        lambda: validate_continuous_mapping_and_cancel(
            sources, args.baseline_ref, args.candidate_ref
        ),
    )
    if not args.skip_performance:
        run_case(
            report,
            "pure_python_hot_path_performance",
            lambda: validate_microbenchmarks(
                sources,
                args.baseline_ref,
                args.candidate_ref,
                args.iterations,
            ),
        )
    report["passed"] = not report["failures"]
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
