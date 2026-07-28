import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
NODE_ROOT = SRC_ROOT / "Node"


def _stub_module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def action_executor_module(monkeypatch):
    class FakeNode:
        pass

    class FakeWirelessController:
        def __init__(self):
            self.lx = 0.0
            self.ly = 0.0
            self.rx = 0.0
            self.ry = 0.0
            self.keys = 0

    class FakeTrigger:
        pass

    class FakeVisualizationAdapter:
        def __init__(self, visualization=None):
            self.visualization = visualization

    rclpy = _stub_module(monkeypatch, "rclpy", ok=lambda: True)
    rclpy_node = _stub_module(monkeypatch, "rclpy.node", Node=FakeNode)
    rclpy.node = rclpy_node
    rclpy_qos = _stub_module(
        monkeypatch,
        "rclpy.qos",
        qos_profile_sensor_data=object(),
        QoSProfile=object,
        ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=object()),
        DurabilityPolicy=SimpleNamespace(VOLATILE=object()),
        HistoryPolicy=SimpleNamespace(KEEP_LAST=object()),
    )
    rclpy.qos = rclpy_qos

    geometry_msgs = _stub_module(monkeypatch, "geometry_msgs")
    geometry_msgs.msg = _stub_module(
        monkeypatch, "geometry_msgs.msg", PoseStamped=object
    )
    nav_msgs = _stub_module(monkeypatch, "nav_msgs")
    nav_msgs.msg = _stub_module(monkeypatch, "nav_msgs.msg", Path=object)
    std_srvs = _stub_module(monkeypatch, "std_srvs")
    std_srvs.srv = _stub_module(monkeypatch, "std_srvs.srv", Trigger=FakeTrigger)

    unitree_go = _stub_module(monkeypatch, "unitree_go")
    unitree_go.msg = _stub_module(monkeypatch, "unitree_go.msg")
    wireless_module = _stub_module(
        monkeypatch,
        "unitree_go.msg._wireless_controller",
        WirelessController=FakeWirelessController,
    )
    unitree_go.msg._wireless_controller = wireless_module

    visualization = _stub_module(monkeypatch, "visualization")
    visualization.adapters = _stub_module(
        monkeypatch,
        "visualization.adapters",
        ExecutorVisualizationAdapter=FakeVisualizationAdapter,
    )

    package_name = "_action_cancel_node"
    package = types.ModuleType(package_name)
    package.__path__ = [str(NODE_ROOT)]
    monkeypatch.setitem(sys.modules, package_name, package)
    module_name = f"{package_name}.action_executor"
    spec = importlib.util.spec_from_file_location(
        module_name, NODE_ROOT / "action_executor.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class BlockingDiscreteBackend:
    enabled = True

    def __init__(self):
        self.entered = threading.Event()
        self.continue_entry = threading.Event()
        self.cancel_seen = False
        self.commands = []
        self.stop_count = 0
        self.shutdown_count = 0

    def _move(self, value, *, cancel_event=None):
        self.entered.set()
        if not self.continue_entry.wait(timeout=1.0):
            return False, "entry timeout.", -1
        self.cancel_seen = bool(cancel_event is not None and cancel_event.is_set())
        if self.cancel_seen:
            return False, "cancelled.", -2
        self.commands.append(float(value))
        return True, "success.", 1

    forward = _move
    shift = _move
    rotate = _move

    def stop(self):
        self.stop_count += 1
        return True, "success.", 1

    def shutdown(self):
        self.shutdown_count += 1
        return True, "success.", 1


class BlockingContinuousBackend(BlockingDiscreteBackend):
    def supports_continuous_velocity(self):
        return True

    def publish_velocity(
        self,
        forward,
        lateral=0.0,
        yaw=0.0,
        *,
        cancel_event=None,
    ):
        del lateral, yaw
        return self._move(forward, cancel_event=cancel_event)


def _make_executor(module):
    executor = object.__new__(module.ActionExecutorClient)
    executor.lock = threading.Lock()
    executor._pause_flag = threading.Event()
    executor._stop_flag = threading.Event()
    executor._shutdown_flag = threading.Event()
    executor._finish_flag = threading.Event()
    executor._new_action_event = threading.Event()
    executor._preempt_action_event = threading.Event()
    executor._pause_reason = None
    executor._navigation_active = True
    executor.actions = []
    executor.action_plan_metadata = None
    executor.motion_control = module.ActionExecutorClient._load_motion_control_config(None)
    executor.motion_backend = BlockingDiscreteBackend()
    executor.thread = SimpleNamespace(is_alive=lambda: False)
    return executor


def _start_move(executor, method_name, value):
    results = []
    worker = threading.Thread(
        target=lambda: results.append(getattr(executor, method_name)(value))
    )
    worker.start()
    assert executor.motion_backend.entered.wait(timeout=0.5)
    return worker, results


def _finish_cancelled_move(executor, worker, results):
    executor.motion_backend.continue_entry.set()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results == [(False, "cancelled.", -2)]
    assert executor.motion_backend.cancel_seen is True
    assert executor.motion_backend.commands == []


def test_stop_after_precheck_but_before_discrete_entry_cancels_command(
    action_executor_module,
):
    executor = _make_executor(action_executor_module)
    worker, results = _start_move(executor, "forward", 0.2)

    try:
        executor.stop()
    finally:
        _finish_cancelled_move(executor, worker, results)


def test_pause_after_precheck_but_before_discrete_entry_cancels_command(
    action_executor_module,
):
    executor = _make_executor(action_executor_module)
    worker, results = _start_move(executor, "rotate", 0.4)

    try:
        executor.set_safety_pause(True, reason="lidar blocked")
    finally:
        _finish_cancelled_move(executor, worker, results)


def test_shutdown_after_precheck_cancels_command_and_releases_backend(
    action_executor_module,
):
    executor = _make_executor(action_executor_module)
    worker, results = _start_move(executor, "shift", 0.1)

    try:
        executor.shutdown(timeout_sec=0.1)
    finally:
        _finish_cancelled_move(executor, worker, results)

    assert executor.motion_backend.shutdown_count == 1


def test_pause_after_continuous_precheck_suppresses_stale_nonzero_write(
    action_executor_module,
):
    executor = _make_executor(action_executor_module)
    executor.motion_backend = BlockingContinuousBackend()
    message = SimpleNamespace(lx=0.0, ly=0.2, rx=0.0)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(executor._publish_motion(message))
    )
    worker.start()
    assert executor.motion_backend.entered.wait(timeout=0.5)

    try:
        executor.set_safety_pause(True, reason="lidar blocked")
    finally:
        executor.motion_backend.continue_entry.set()
        worker.join(timeout=0.5)

    assert not worker.is_alive()
    assert results == [False]
    assert executor.motion_backend.cancel_seen is True
    assert executor.motion_backend.commands == []
