import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

from onboard_runtime.motion_bus import MotionCommandBroker
from onboard_runtime.mover_transport import MotionBusCommandSink


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_manipulation_mover():
    path = REPO_ROOT / "box_demo_2" / "groot_mover.py"
    spec = importlib.util.spec_from_file_location("_manip_groot_mover_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_environment_routes_manipulation_mover_to_bus(tmp_path, monkeypatch):
    socket_path = f"/tmp/groot-manip-bus-test-{os.getpid()}-{time.time_ns()}.sock"
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

    monkeypatch.setenv("GROOT_MOTION_BUS_SOCKET", socket_path)
    monkeypatch.setenv("GROOT_MOTION_SOURCE", "manipulation.test")
    module = _load_manipulation_mover()
    mover = module.GrootMover(verbose=False)
    mover.publish_velocity(0.15, lateral=-0.05)

    output = {}
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            output = json.loads((tmp_path / "command.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if str(output.get("motion_bus_source", "")).startswith(
            "manipulation.test.p"
        ):
            break
        time.sleep(0.01)

    mover.release()
    stop.set()
    thread.join(timeout=1.0)
    assert output["motion_bus_source"].startswith("manipulation.test.p")
    assert output["velocity"]["forward"] == 0.15
    assert output["velocity"]["lateral"] == -0.05


def test_rl_lower_hold_refreshes_lease_until_next_command(tmp_path):
    socket_path = f"/tmp/groot-hold-bus-test-{os.getpid()}-{time.time_ns()}.sock"
    command_path = tmp_path / "command.json"
    broker = MotionCommandBroker(
        socket_path=socket_path,
        command_file=str(command_path),
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

    sink = MotionBusCommandSink(
        source="manipulation.hold",
        socket_path=socket_path,
        lease_s=0.15,
    )
    try:
        sink.hold(fsm="RL_LOWER", height=0.76)
        # This exceeds the original one-shot lease by more than 3x.
        time.sleep(0.50)
        output = json.loads(command_path.read_text())
        assert output["fsm"] == "RL_LOWER"
        assert output["motion_bus_source"].startswith("manipulation.hold")

        sink(
            "",
            "RL_FULL",
            forward=0.12,
            height=0.76,
        )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            output = json.loads(command_path.read_text())
            if output["velocity"]["forward"] == 0.12:
                break
            time.sleep(0.01)
        assert output["fsm"] == "RL_FULL"
        assert output["velocity"]["forward"] == 0.12
    finally:
        sink.release()
        stop.set()
        thread.join(timeout=1.0)
