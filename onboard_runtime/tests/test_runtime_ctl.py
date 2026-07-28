import json
import os
import threading
import time

import pytest

from onboard_runtime.motion_bus import MotionBusClient, MotionCommandBroker
from onboard_runtime.runtime_ctl import clear_latched_safety


def _wait_for(predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_clear_safety_requires_explicit_human_approval(tmp_path):
    with pytest.raises(PermissionError):
        clear_latched_safety(
            socket_path=str(tmp_path / "missing.sock"),
            status_file=str(tmp_path / "missing.json"),
        )


def test_clear_safety_round_trip_against_offline_broker(tmp_path):
    # macOS AF_UNIX paths are limited to roughly 104 bytes.
    socket_path = f"/tmp/groot-runtime-ctl-{os.getpid()}-{time.time_ns()}.sock"
    status_path = tmp_path / "status.json"
    broker = MotionCommandBroker(
        socket_path=socket_path,
        command_file=str(tmp_path / "command.json"),
        status_file=str(status_path),
        safety_file=str(tmp_path / "safety.json"),
        output_hz=100.0,
    )
    stop = threading.Event()
    thread = threading.Thread(target=broker.run, args=(stop,), daemon=True)
    thread.start()
    assert _wait_for(lambda: os.path.exists(socket_path))

    safety = MotionBusClient("operator.test", socket_path)
    try:
        safety.publish(fsm="DAMP", estop=True, lease_s=1.0)
        assert _wait_for(
            lambda: status_path.exists()
            and json.loads(status_path.read_text()).get("safety_latched") is True
        )
        status = clear_latched_safety(
            socket_path=socket_path,
            status_file=str(status_path),
            human_approved=True,
        )
        assert status["safety_latched"] is False
        assert status["selected_source"] == "motion_bus.idle"
    finally:
        safety.close(release=False)
        stop.set()
        thread.join(timeout=1.0)
    assert not thread.is_alive()
