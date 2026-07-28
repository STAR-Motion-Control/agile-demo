import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from camera_runtime import CaptureErrorBackoff


def _load_rgbd_module(monkeypatch):
    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: True
    callback_groups = ModuleType("rclpy.callback_groups")
    callback_groups.MutuallyExclusiveCallbackGroup = type(
        "MutuallyExclusiveCallbackGroup", (), {}
    )
    node = ModuleType("rclpy.node")
    node.Node = type("Node", (), {})
    qos = ModuleType("rclpy.qos")
    qos.qos_profile_sensor_data = object()
    rclpy.callback_groups = callback_groups
    rclpy.node = node
    rclpy.qos = qos
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.callback_groups", callback_groups)
    monkeypatch.setitem(sys.modules, "rclpy.node", node)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)

    module_path = SRC_ROOT / "Node" / "rgbd.py"
    spec = importlib.util.spec_from_file_location("_rgbd_runtime_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_capture_error_backoff_is_bounded_rate_limited_and_resettable():
    backoff = CaptureErrorBackoff(
        initial_s=0.05,
        maximum_s=0.20,
        log_interval_s=5.0,
    )

    assert backoff.failure(now=10.0) == (0.05, True)
    assert backoff.failure(now=11.0) == (0.10, False)
    assert backoff.failure(now=12.0) == (0.20, False)
    assert backoff.failure(now=15.0) == (0.20, True)

    backoff.reset()
    assert backoff.failure(now=15.1) == (0.05, True)


def test_external_ros_input_uses_capture_depth_even_without_navdp(monkeypatch):
    module = _load_rgbd_module(monkeypatch)
    subscriptions = []

    class FakeSubscriber:
        def __init__(self, _node, _message_type, topic, **_kwargs):
            self.topic = topic
            subscriptions.append(topic)

    class FakeSynchronizer:
        def __init__(self, inputs, **_kwargs):
            self.inputs = inputs
            self.callback = None

        def registerCallback(self, callback):
            self.callback = callback

    message_filters = ModuleType("message_filters")
    message_filters.Subscriber = FakeSubscriber
    message_filters.ApproximateTimeSynchronizer = FakeSynchronizer
    monkeypatch.setitem(sys.modules, "message_filters", message_filters)

    client = object.__new__(module.RGBDClient)
    client.capture_depth = True
    client.require_depth = False
    client._load_image_type = lambda: object
    client._load_cv_bridge = lambda: type("Bridge", (), {})
    client.create_subscription = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("RGB-only subscription path was used")
    )
    cfg = SimpleNamespace(
        subscrib_topic=SimpleNamespace(rgb="/rgb", depth="/depth")
    )

    client._initialize_ros_input(cfg)

    assert subscriptions == ["/rgb", "/depth"]
    assert client.timesynchronizer.callback == client.rgbd_callback


def test_capture_loop_resets_backoff_after_a_successful_frame(
    monkeypatch,
    caplog,
):
    module = _load_rgbd_module(monkeypatch)

    class FakeColorFrame:
        def get_data(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class FakeFrames:
        def get_color_frame(self):
            return FakeColorFrame()

    class FakePipeline:
        def __init__(self):
            self.responses = [
                RuntimeError("camera-1"),
                RuntimeError("camera-2"),
                FakeFrames(),
                RuntimeError("camera-3"),
            ]
            self.stopped = False

        def wait_for_frames(self, **_kwargs):
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def stop(self):
            self.stopped = True

    class FakeStopEvent:
        def __init__(self):
            self.delays = []
            self.stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, delay):
            self.delays.append(delay)
            if len(self.delays) == 3:
                self.stopped = True
            return self.stopped

    client = object.__new__(module.RGBDClient)
    client.pipeline = FakePipeline()
    client.capture_depth = False
    client.publish_ros_topics = False
    client.depth_scale = 1.0
    client._stop_event = FakeStopEvent()
    client._camera_error_backoff = CaptureErrorBackoff(
        initial_s=0.05,
        maximum_s=0.20,
        log_interval_s=60.0,
    )
    accepted = []
    client._accept_frame = lambda *args, **kwargs: accepted.append((args, kwargs))

    with caplog.at_level(logging.ERROR, logger=module.logger.name):
        client._capture_loop()

    assert client._stop_event.delays == [0.05, 0.10, 0.05]
    assert len(accepted) == 1
    assert len(caplog.records) == 2
    assert client.pipeline.stopped is True


def test_capture_loop_backs_off_when_framesets_are_incomplete(monkeypatch, caplog):
    module = _load_rgbd_module(monkeypatch)

    class IncompleteFrames:
        def get_color_frame(self):
            return None

    class FakePipeline:
        def __init__(self):
            self.stopped = False

        def wait_for_frames(self, **_kwargs):
            return IncompleteFrames()

        def stop(self):
            self.stopped = True

    class FakeStopEvent:
        def __init__(self):
            self.delays = []

        def is_set(self):
            return False

        def wait(self, delay):
            self.delays.append(delay)
            return len(self.delays) == 3

    client = object.__new__(module.RGBDClient)
    client.pipeline = FakePipeline()
    client.capture_depth = False
    client._stop_event = FakeStopEvent()
    client._camera_error_backoff = CaptureErrorBackoff(
        initial_s=0.05,
        maximum_s=0.20,
        log_interval_s=60.0,
    )

    with caplog.at_level(logging.ERROR, logger=module.logger.name):
        client._capture_loop()

    assert client._stop_event.delays == [0.05, 0.10, 0.20]
    assert len(caplog.records) == 1
    assert "incomplete frameset" in caplog.records[0].message
    assert client.pipeline.stopped is True
