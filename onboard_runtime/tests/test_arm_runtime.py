import os
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BOX_ROOT = REPO_ROOT / "box_demo_2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BOX_ROOT) not in sys.path:
    sys.path.insert(0, str(BOX_ROOT))

from onboard_runtime.arm_protocol import ARM_JOINT_COUNT, ArmProtocolError
from onboard_runtime.arm_runtime_broker import ManipulationArmBroker
from arm_runtime_ipc import ArmRuntimeClient, ArmRuntimeError


Q_ZERO = [0.0] * ARM_JOINT_COUNT
Q_ONE = [1.0] * ARM_JOINT_COUNT
TAU = [float(index) for index in range(ARM_JOINT_COUNT)]


def request(broker, source, sequence, op, now, **fields):
    return broker.handle_request(
        {
            "schema_version": 1,
            "source": source,
            "sequence": sequence,
            "request_id": f"r{sequence}",
            "boot_id": broker.boot_id,
            "op": op,
            **fields,
        },
        now=now,
    )


def fresh_broker(tmp_path, now=10.0):
    broker = ManipulationArmBroker(
        socket_path=str(tmp_path / "arm.sock"),
        status_file=str(tmp_path / "status.json"),
    )
    broker.update_robot_state(
        mode_machine=5,
        mode_pr=2,
        q=Q_ZERO,
        tau_est=TAU,
        received_at=now,
    )
    broker.update_policy(Q_ONE, received_at=now)
    return broker


def test_owner_token_sequence_and_frame_validation(tmp_path):
    broker = fresh_broker(tmp_path)
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        10.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0, "profile": "manip_v1"},
    )
    assert acquired["ok"] is True
    token = acquired["token"]

    stale = request(
        broker,
        "manipulation.test",
        1,
        "update",
        10.02,
        token=token,
        lease_s=0.25,
        deadline_mono_ns=int(10.10 * 1e9),
        frame={"q": Q_ONE, "weight": 1.0, "profile": "manip_v1"},
    )
    assert stale["error_code"] == "STALE_SEQUENCE"

    wrong_token = request(
        broker,
        "manipulation.test",
        2,
        "update",
        10.03,
        token="0" * 32,
        lease_s=0.25,
        deadline_mono_ns=int(10.10 * 1e9),
        frame={"q": Q_ONE, "weight": 1.0, "profile": "manip_v1"},
    )
    assert wrong_token["error_code"] == "LEASE_LOST"

    invalid = request(
        broker,
        "manipulation.test",
        3,
        "update",
        10.04,
        token=token,
        lease_s=0.25,
        deadline_mono_ns=int(10.10 * 1e9),
        frame={"q": Q_ZERO[:-1] + [float("nan")], "weight": 1.0},
    )
    assert invalid["error_code"] == "INVALID_REQUEST"
    assert broker.current_frame(now=10.05).q == tuple(Q_ZERO)


def test_release_requires_fresh_policy_and_never_drops_ownership_on_failure(tmp_path):
    broker = fresh_broker(tmp_path, now=1.0)
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        1.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    rejected = request(
        broker,
        "manipulation.test",
        2,
        "release_to_policy",
        1.20,
        token=acquired["token"],
        duration_s=0.2,
    )
    assert rejected["error_code"] == "STATE_STALE"
    assert broker.current_frame(now=1.21) is not None


def test_release_with_fresh_state_and_stale_policy_fails_closed(tmp_path):
    broker = fresh_broker(tmp_path, now=1.0)
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        1.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    broker.update_robot_state(
        mode_machine=5,
        mode_pr=2,
        q=Q_ZERO,
        tau_est=TAU,
        received_at=1.20,
    )
    rejected = request(
        broker,
        "manipulation.test",
        2,
        "release_to_policy",
        1.20,
        token=acquired["token"],
        duration_s=0.2,
    )
    assert rejected["error_code"] == "POLICY_STALE"
    assert broker.current_frame(now=1.21) is not None
    assert broker.snapshot(1.21)["runtime"]["state"] == "owned"


def test_server_side_release_aligns_to_live_policy_before_weight_disappears(tmp_path):
    broker = fresh_broker(tmp_path, now=10.0)
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        10.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    released = request(
        broker,
        "manipulation.test",
        2,
        "release_to_policy",
        10.02,
        token=acquired["token"],
        duration_s=0.20,
    )
    assert released["ok"] is True

    now = 10.02
    frame = None
    for _ in range(40):
        now += 0.01
        broker.update_policy(Q_ONE, received_at=now)
        frame = broker.current_frame(now=now)
        if frame is None:
            break
    assert frame is None
    assert broker.snapshot(now)["runtime"]["state"] == "idle"


def test_release_reanchors_without_jump_after_stale_policy_recovers(tmp_path):
    broker = fresh_broker(tmp_path, now=20.0)
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        20.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    request(
        broker,
        "manipulation.test",
        2,
        "release_to_policy",
        20.02,
        token=acquired["token"],
        duration_s=0.20,
    )
    broker.update_policy(Q_ONE, received_at=20.10)
    before_stale = broker.current_frame(now=20.10)
    assert before_stale is not None
    held = broker.current_frame(now=20.30)
    assert held is not None
    assert held.q == before_stale.q
    assert held.weight == 1.0

    opposite = [-1.0] * ARM_JOINT_COUNT
    broker.update_policy(opposite, received_at=20.31)
    resumed = broker.current_frame(now=20.31)
    assert resumed is not None
    assert max(abs(a - b) for a, b in zip(resumed.q, held.q)) < 0.02
    assert resumed.weight == 1.0


def test_release_fades_weight_only_after_q_is_policy_aligned(tmp_path):
    broker = fresh_broker(tmp_path, now=30.0)
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        30.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    request(
        broker,
        "manipulation.test",
        2,
        "release_to_policy",
        30.02,
        token=acquired["token"],
        duration_s=0.10,
    )
    now = 30.02
    fading = None
    for _ in range(30):
        now += 0.01
        broker.update_policy(Q_ONE, received_at=now)
        frame = broker.current_frame(now=now)
        if frame is not None and 0.0 < frame.weight < 1.0:
            fading = frame
            break
    assert fading is not None
    assert fading.q == tuple(Q_ONE)


def test_status_writes_are_throttled_while_applied_sequence_changes(
    tmp_path,
    monkeypatch,
):
    broker = fresh_broker(tmp_path, now=10.0)
    writes = []

    def record_write(path, payload):
        writes.append((path, payload))

    monkeypatch.setattr(
        "onboard_runtime.arm_runtime_broker._atomic_write_json",
        record_write,
    )
    acquired = request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        10.01,
        lease_s=0.25,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    token = acquired["token"]
    broker.write_status(now=10.01, now_wall=100.0)
    for sequence in range(2, 12):
        now = 10.01 + sequence * 0.02
        broker.update_robot_state(
            mode_machine=5,
            mode_pr=2,
            q=Q_ZERO,
            tau_est=TAU,
            received_at=now,
        )
        update = request(
            broker,
            "manipulation.test",
            sequence,
            "update",
            now,
            token=token,
            lease_s=0.25,
            deadline_mono_ns=int((now + 0.1) * 1e9),
            frame={"q": Q_ONE, "weight": 1.0},
        )
        assert update["ok"]
        broker.write_status(now=now, now_wall=100.0 + sequence * 0.02)
    assert len(writes) == 1

    broker.write_status(now=10.52, now_wall=100.52)
    assert len(writes) == 2
    assert writes[-1][1]["applied_sequence"] == 11


def test_expired_lease_holds_then_auto_releases_only_with_fresh_policy(tmp_path):
    broker = fresh_broker(tmp_path, now=2.0)
    request(
        broker,
        "manipulation.test",
        1,
        "acquire",
        2.01,
        lease_s=0.05,
        frame={"q": Q_ZERO, "weight": 1.0},
    )
    held = broker.current_frame(now=2.20)
    assert held is not None
    assert broker.snapshot(2.20)["runtime"]["state"] == "holding_orphan"

    broker.update_policy(Q_ONE, received_at=2.21)
    releasing = broker.current_frame(now=2.21)
    assert releasing is not None
    assert broker.snapshot(2.21)["runtime"]["state"] == "releasing"


def test_ipc_client_state_publish_and_release(tmp_path):
    socket_path = f"/tmp/gar-broker-{os.getpid()}-{time.time_ns()}.sock"
    broker = ManipulationArmBroker(
        socket_path=socket_path,
        status_file=str(tmp_path / "status.json"),
        policy_stale_s=0.20,
        state_stale_s=0.20,
    )
    broker.bind()
    stop = threading.Event()
    applied = {"frame": None}

    def serve():
        while not stop.is_set():
            now = time.monotonic()
            broker.update_robot_state(
                mode_machine=7,
                mode_pr=3,
                q=Q_ZERO,
                tau_est=TAU,
                received_at=now,
            )
            broker.update_policy(Q_ONE, received_at=now)
            broker.drain(now=now)
            applied["frame"] = broker.current_frame(now=now)
            time.sleep(0.002)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = ArmRuntimeClient(
        socket_path,
        "manipulation.integration",
        state_poll_hz=50.0,
    )
    try:
        state = client.wait_state(timeout_s=1.0)
        assert state.mode_machine == 7
        assert state.tau_est == tuple(TAU)

        client.publish(Q_ZERO)
        deadline = time.monotonic() + 1.0
        while applied["frame"] is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert applied["frame"] is not None
        assert client.token is not None

        assert client.release_to_policy(duration_s=0.10)
        assert client.token is None
        deadline = time.monotonic() + 1.0
        while applied["frame"] is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert applied["frame"] is None
    finally:
        client.close(release=False)
        stop.set()
        thread.join(timeout=1.0)
        broker.close()
    assert not thread.is_alive()


def test_client_reports_missing_server_without_creating_dds():
    path = f"/tmp/gar-missing-{os.getpid()}-{time.time_ns()}.sock"
    client = ArmRuntimeClient(path, "manipulation.missing")
    try:
        with pytest.raises(ArmRuntimeError) as exc_info:
            client.wait_state(timeout_s=0.05)
        assert exc_info.value.code in {"UNAVAILABLE", "STATE_TIMEOUT"}
    finally:
        client.close(release=False)


def test_second_arm_broker_cannot_replace_active_socket(tmp_path):
    socket_path = f"/tmp/gar-owner-{os.getpid()}-{time.time_ns()}.sock"
    first = ManipulationArmBroker(
        socket_path=socket_path,
        status_file=str(tmp_path / "first-status.json"),
    )
    second = ManipulationArmBroker(
        socket_path=socket_path,
        status_file=str(tmp_path / "second-status.json"),
    )
    first.bind()
    try:
        with pytest.raises(ArmProtocolError, match="already active"):
            second.bind()
        second.close()
        assert os.path.exists(socket_path)
    finally:
        first.close()


def test_arm_client_restart_gets_a_new_instance_source():
    path = f"/tmp/gar-missing-{os.getpid()}-{time.time_ns()}.sock"
    first = ArmRuntimeClient(path, "manipulation.restart")
    second = ArmRuntimeClient(path, "manipulation.restart")
    try:
        assert first.logical_source == second.logical_source
        assert first.source != second.source
        assert first.source.startswith("manipulation.restart.p")
        assert second.source.startswith("manipulation.restart.p")
    finally:
        first.close(release=False)
        second.close(release=False)
