import json
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onboard_runtime.motion_bus import MotionCommandBroker

MODULE_PATH = REPO_ROOT / "nav_uat_overlay" / "src" / "Node" / "motion_backend.py"
SPEC = importlib.util.spec_from_file_location("motion_backend_bus_under_test", MODULE_PATH)
motion_backend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_backend
SPEC.loader.exec_module(motion_backend)
GrootHttpDiscreteBackend = motion_backend.GrootHttpDiscreteBackend


def test_two_navigation_backends_share_one_ordered_bus_producer(tmp_path, monkeypatch):
    socket_path = f"/tmp/groot-nav-bus-test-{os.getpid()}-{time.time_ns()}.sock"
    broker = MotionCommandBroker(
        socket_path=socket_path,
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
        output_hz=100.0,
    )
    stop = threading.Event()
    thread = threading.Thread(target=broker.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 1.0
    while not os.path.exists(socket_path) and time.monotonic() < deadline:
        time.sleep(0.005)

    monkeypatch.delenv("GROOT_MOTION_BUS_SOCKET", raising=False)
    cfg = {
        "motion_backend": {
            "type": "groot_motion_bus",
            "motion_bus_socket": socket_path,
            "runtime_module_path": str(REPO_ROOT),
            "box_demo_module_path": str(REPO_ROOT / "box_demo_groot"),
            "profile_file": str(tmp_path / "missing-profile.json"),
            "initialize_before_motion": False,
            "continuous_command_interval": 0.02,
            "continuous_hold_duration": 0.25,
            "warmup_time": 0.0,
            "min_duration": 0.0,
            "min_distance": 0.0,
            "verbose": False,
        }
    }
    first = GrootHttpDiscreteBackend(cfg)
    second = GrootHttpDiscreteBackend(cfg)
    assert first.supports_continuous_velocity()
    assert second.supports_continuous_velocity()

    assert first.publish_velocity(0.20)[0]
    assert second.publish_velocity(0.10, lateral=0.05)[0]
    deadline = time.monotonic() + 1.0
    output = {}
    while time.monotonic() < deadline:
        try:
            output = json.loads((tmp_path / "command.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if output.get("motion_bus_sequence", 0) >= 2:
            break
        time.sleep(0.01)

    stop.set()
    thread.join(timeout=1.0)
    assert output["motion_bus_source"].startswith("navigation.p")
    assert output["motion_bus_sequence"] >= 2
    assert output["velocity"]["forward"] == 0.10
    assert output["velocity"]["lateral"] == 0.05
