import json
import os
import threading
import time

import pytest

import onboard_runtime.motion_bus as motion_bus_module
from onboard_runtime.command_stream import LatestCommandReceiver
from onboard_runtime.motion_bus import (
    MotionBusClient,
    MotionBusError,
    MotionCommandBroker,
)


def command(source, sequence, *, forward=0.0, lease_s=0.5, fsm="RL_FULL"):
    return {
        "type": "command",
        "source": source,
        "sequence": sequence,
        "lease_s": lease_s,
        "command": {
            "fsm": fsm,
            "velocity": {"forward": forward, "lateral": 0.0, "yaw": 0.0},
            "height": 0.76,
        },
    }


def persisted_safety(
    *, fsm="DAMP", schema_version=1, include_schema=True, blocked_sources=None
):
    payload = {
        "timestamp": 1.0,
        "source": "safety.test",
        "sequence": 7,
        "fsm": fsm,
        "velocity": {"forward": 0.0, "lateral": 0.0, "yaw": 0.0},
        "height": 0.76,
        "units": "agile",
    }
    if include_schema:
        payload["schema_version"] = schema_version
    if blocked_sources is not None:
        payload["blocked_sources"] = blocked_sources
    return payload


def test_priority_and_lease_fallback(tmp_path):
    broker = MotionCommandBroker(
        socket_path=str(tmp_path / "bus.sock"),
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
    )
    broker.handle_payload(command("navigation", 1, forward=0.2, lease_s=1.0), 10.0)
    broker.handle_payload(command("manipulation", 1, forward=-0.1, lease_s=0.2), 10.1)

    status = broker.tick(now_mono=10.15, now_wall=100.0)
    output = json.loads((tmp_path / "command.json").read_text())
    assert status["selected_source"] == "manipulation"
    assert output["velocity"]["forward"] == pytest.approx(-0.1)

    status = broker.tick(now_mono=10.31, now_wall=100.2)
    output = json.loads((tmp_path / "command.json").read_text())
    assert status["selected_source"] == "navigation"
    assert output["velocity"]["forward"] == pytest.approx(0.2)


def test_stale_sequence_is_rejected_without_replacing_command(tmp_path):
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
    )
    assert broker.handle_datagram(json.dumps(command("navigation", 2)).encode(), 1.0)
    assert not broker.handle_datagram(json.dumps(command("navigation", 1)).encode(), 1.1)
    status = broker.tick(now_mono=1.2, now_wall=2.0)
    assert status["selected_sequence"] == 2
    assert status["metrics"]["rejected"] == 1


def test_health_gate_zeroes_motion_but_preserves_safety(tmp_path):
    health_file = tmp_path / "health.json"
    health_file.write_text(json.dumps({"healthy": False, "timestamp": 50.0, "reason": "lag"}))
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
        health_file=str(health_file),
        require_health=True,
    )
    broker.handle_payload(command("navigation", 1, forward=0.3), 5.0)
    status = broker.tick(now_mono=5.1, now_wall=50.1)
    output = json.loads((tmp_path / "command.json").read_text())
    assert status["motion_blocked"] is True
    assert output["velocity"]["forward"] == 0.0
    assert output["motion_bus_health_ok"] is False

    broker.handle_payload(command("safety", 1, fsm="DAMP"), 5.2)
    status = broker.tick(now_mono=5.3, now_wall=50.2)
    output = json.loads((tmp_path / "command.json").read_text())
    assert status["selected_source"] == "safety"
    assert status["motion_blocked"] is False
    assert output["fsm"] == "DAMP"
    assert output["estop"] is True
    assert output["motion_bus_health_ok"] is False

    # Safety commands are latched and cannot disappear through lease expiry.
    status = broker.tick(now_mono=50.0, now_wall=95.0)
    assert status["selected_source"] == "safety"
    assert status["safety_latched"] is True


def test_composite_health_rejects_stale_runtime_identity(tmp_path):
    adapter = tmp_path / "adapter-health.json"
    merger = tmp_path / "merger-health.json"
    adapter.write_text(
        json.dumps(
            {
                "healthy": True,
                "timestamp": 50.0,
                "runtime_id": "previous-run",
            }
        )
    )
    merger.write_text(
        json.dumps(
            {"healthy": True, "timestamp": 50.0, "runtime_id": "current-run"}
        )
    )
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
        health_file=[str(adapter), str(merger)],
        health_runtime_id="current-run",
        require_health=True,
    )
    broker.handle_payload(command("navigation", 1, forward=0.2), 5.0)

    status = broker.tick(now_mono=5.1, now_wall=50.1)
    assert status["healthy"] is False
    assert "runtime_id_mismatch" in status["health_reason"]
    assert status["health_files"] == [str(adapter), str(merger)]
    assert status["health_runtime_id"] == "current-run"

    adapter.write_text(
        json.dumps(
            {"healthy": True, "timestamp": 50.2, "runtime_id": "current-run"}
        )
    )
    status = broker.tick(now_mono=5.3, now_wall=50.3)
    assert status["healthy"] is True
    assert status["motion_blocked"] is False


def test_only_operator_can_clear_latched_safety(tmp_path):
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
    )
    broker.handle_payload(
        command("navigation", 1, forward=0.3, lease_s=5.0),
        0.9,
    )
    broker.handle_payload(command("safety", 1, fsm="DAMP"), 1.0)
    rejected = {
        "type": "clear_safety",
        "source": "navigation",
        "sequence": 2,
    }
    assert not broker.handle_datagram(json.dumps(rejected).encode(), 1.1)
    assert broker.tick(1.2, 2.0)["safety_latched"] is True

    broker.handle_payload(
        {"type": "clear_safety", "source": "operator.keyboard", "sequence": 1},
        1.3,
    )
    status = broker.tick(1.4, 2.2)
    assert status["safety_latched"] is False
    assert status["selected_source"] == "motion_bus.idle"
    assert status["active_sources"] == []
    assert status["blocked_sources"] == ["navigation", "safety"]


def test_safety_epoch_requires_release_after_clear(tmp_path):
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
    )
    broker.handle_payload(command("navigation", 1, forward=0.3), 1.0)
    broker.handle_payload(command("manipulation", 1, forward=-0.1), 1.1)
    broker.handle_payload(command("safety", 1, fsm="DAMP"), 1.2)

    status = broker.tick(1.3, 2.0)
    assert status["active_sources"] == []
    assert status["blocked_sources"] == ["manipulation", "navigation", "safety"]

    # A release sent while safety is still latched cannot pre-authorize the
    # source for the next safety epoch.
    broker.handle_payload({"type": "release", "source": "navigation"}, 1.4)
    assert not broker.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 1.5
    )
    assert not broker.handle_datagram(
        json.dumps(command("navigation.new", 1, forward=0.1)).encode(), 1.6
    )
    assert broker.tick(1.7, 2.1)["blocked_sources"] == [
        "manipulation",
        "navigation",
        "navigation.new",
        "safety",
    ]

    broker.handle_payload(
        {"type": "clear_safety", "source": "operator.keyboard", "sequence": 1},
        1.8,
    )
    assert not broker.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 1.9
    )

    broker.handle_payload({"type": "release", "source": "navigation"}, 2.0)
    assert broker.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 2.1
    )
    status = broker.tick(2.2, 2.2)
    assert status["selected_source"] == "navigation"
    assert "navigation" not in status["blocked_sources"]
    assert status["metrics"]["blocked"] == 3


def test_blocked_source_can_still_publish_a_safety_command(tmp_path):
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
    )
    broker.handle_payload(command("navigation", 1, forward=0.2), 1.0)
    broker.handle_payload(command("safety", 1, fsm="DAMP"), 1.1)

    assert broker.handle_datagram(
        json.dumps(command("navigation", 2, fsm="LIMP")).encode(), 1.2
    )
    status = broker.tick(1.3, 2.0)
    output = json.loads((tmp_path / "command.json").read_text())
    assert status["selected_source"] == "navigation"
    assert status["safety_latched"] is True
    assert output["fsm"] == "LIMP"


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param("{", id="malformed-json"),
        pytest.param("[]", id="non-object"),
        pytest.param(
            json.dumps(persisted_safety(include_schema=False)), id="missing-schema"
        ),
        pytest.param(
            json.dumps(persisted_safety(schema_version=3)), id="unsupported-schema"
        ),
        pytest.param(
            json.dumps(persisted_safety(schema_version=True)), id="boolean-schema"
        ),
        pytest.param(
            json.dumps(persisted_safety(fsm="RL_FULL")), id="non-safety-state"
        ),
        pytest.param(
            json.dumps(persisted_safety(blocked_sources="navigation")),
            id="invalid-blocked-sources",
        ),
        pytest.param(
            json.dumps(
                {
                    "schema_version": 2,
                    "state": "cleared",
                    "timestamp": 1.0,
                    "blocked_sources": [],
                }
            ),
            id="empty-cleared-epoch",
        ),
    ],
)
def test_invalid_persisted_safety_prevents_broker_start(tmp_path, contents):
    safety_file = tmp_path / "safety.json"
    safety_file.write_text(contents)

    with pytest.raises(MotionBusError, match="persisted safety"):
        MotionCommandBroker(
            command_file=None,
            status_file=str(tmp_path / "status.json"),
            safety_file=str(safety_file),
        )


def test_valid_persisted_safety_is_restored(tmp_path):
    safety_file = tmp_path / "safety.json"
    safety_file.write_text(json.dumps(persisted_safety(fsm="LIMP")))
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(safety_file),
    )

    status = broker.tick(1.0, 2.0)
    output = json.loads((tmp_path / "command.json").read_text())
    assert status["safety_latched"] is True
    assert status["selected_source"] == "safety.restored"
    assert status["blocked_sources"] == ["safety.test"]
    assert output["fsm"] == "LIMP"


def test_persisted_safety_restores_blocked_sources(tmp_path):
    safety_file = tmp_path / "safety.json"
    first = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "first-status.json"),
        safety_file=str(safety_file),
    )
    first.handle_payload(command("navigation", 1, forward=0.2), 1.0)
    first.handle_payload(command("manipulation", 1, forward=-0.1), 1.1)
    first.handle_payload(command("safety", 1, fsm="DAMP"), 1.2)
    assert not first.handle_datagram(
        json.dumps(command("navigation.new", 1, forward=0.1)).encode(), 1.3
    )

    restored = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "restored-status.json"),
        safety_file=str(safety_file),
    )
    status = restored.tick(2.0, 3.0)
    assert status["blocked_sources"] == [
        "manipulation",
        "navigation",
        "navigation.new",
        "safety",
    ]

    restored.handle_payload(
        {"type": "clear_safety", "source": "operator.keyboard", "sequence": 1},
        2.1,
    )
    assert not restored.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 2.2
    )
    restored.handle_payload({"type": "release", "source": "navigation"}, 2.3)
    assert restored.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 2.4
    )


def test_cleared_epoch_survives_restart_until_every_source_releases(tmp_path):
    safety_file = tmp_path / "safety.json"
    first = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "first-status.json"),
        safety_file=str(safety_file),
    )
    first.handle_payload(command("navigation", 1, forward=0.2), 1.0)
    first.handle_payload(command("safety", 1, fsm="DAMP"), 1.1)
    first.handle_payload(
        {"type": "clear_safety", "source": "operator.keyboard", "sequence": 1},
        1.2,
    )
    tombstone = json.loads(safety_file.read_text())
    assert tombstone["schema_version"] == 2
    assert tombstone["state"] == "cleared"
    assert tombstone["blocked_sources"] == ["navigation", "safety"]

    restored = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "restored-status.json"),
        safety_file=str(safety_file),
    )
    status = restored.tick(2.0, 3.0)
    assert status["safety_latched"] is False
    assert status["blocked_sources"] == ["navigation", "safety"]
    assert not restored.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 2.1
    )

    restored.handle_payload({"type": "release", "source": "navigation"}, 2.2)
    assert safety_file.exists()
    assert json.loads(safety_file.read_text())["blocked_sources"] == ["safety"]
    restored.handle_payload({"type": "release", "source": "safety"}, 2.3)
    assert not safety_file.exists()
    assert restored.handle_datagram(
        json.dumps(command("navigation", 2, forward=0.2)).encode(), 2.4
    )


def test_failed_cleared_epoch_write_keeps_latch_active(tmp_path, monkeypatch):
    safety_file = tmp_path / "safety.json"
    broker = MotionCommandBroker(
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(safety_file),
    )
    broker.handle_payload(command("safety", 1, fsm="DAMP"), 1.0)
    original_persist = broker._persist_cleared_epoch

    def fail_persist(_blocked_sources):
        raise PermissionError("safety journal is read-only")

    monkeypatch.setattr(broker, "_persist_cleared_epoch", fail_persist)
    clear = {"type": "clear_safety", "source": "operator.keyboard", "sequence": 1}
    assert not broker.handle_datagram(json.dumps(clear).encode(), 1.1)
    assert broker.tick(1.2, 2.0)["safety_latched"] is True
    assert safety_file.exists()

    # The failed clear did not consume the sequence or mutate the in-memory latch.
    monkeypatch.setattr(broker, "_persist_cleared_epoch", original_persist)
    assert broker.handle_datagram(json.dumps(clear).encode(), 1.3)
    assert broker.tick(1.4, 2.1)["safety_latched"] is False
    assert json.loads(safety_file.read_text())["state"] == "cleared"
    broker.handle_payload({"type": "release", "source": "safety"}, 1.5)
    assert not safety_file.exists()


def test_failed_tombstone_update_keeps_source_blocked(tmp_path, monkeypatch):
    safety_file = tmp_path / "safety.json"
    broker = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "status.json"),
        safety_file=str(safety_file),
    )
    broker.handle_payload(command("navigation", 1), 1.0)
    broker.handle_payload(command("safety", 1, fsm="DAMP"), 1.1)
    broker.handle_payload(
        {"type": "clear_safety", "source": "operator.keyboard", "sequence": 1},
        1.2,
    )
    persisted_before = json.loads(safety_file.read_text())

    def fail_persist(_blocked_sources):
        raise PermissionError("safety journal is read-only")

    monkeypatch.setattr(broker, "_persist_cleared_epoch", fail_persist)
    assert not broker.handle_datagram(
        json.dumps({"type": "release", "source": "navigation"}).encode(), 1.3
    )
    status = broker.tick(1.4, 2.0)
    assert "navigation" in status["blocked_sources"]
    assert json.loads(safety_file.read_text()) == persisted_before


def test_unix_datagram_client_to_broker(tmp_path):
    # macOS limits AF_UNIX paths to 104 bytes; pytest's tmp path is longer.
    socket_path = f"/tmp/groot-motion-bus-test-{os.getpid()}-{time.time_ns()}.sock"
    broker = MotionCommandBroker(
        socket_path=socket_path,
        command_file=str(tmp_path / "command.json"),
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
        output_hz=50.0,
    )
    stop = threading.Event()
    thread = threading.Thread(target=broker.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 1.0
    while not os.path.exists(socket_path) and time.monotonic() < deadline:
        time.sleep(0.005)

    client = MotionBusClient("navigation.test", socket_path)
    client.publish(forward=0.25, lease_s=0.3)
    deadline = time.monotonic() + 1.0
    selected = None
    while time.monotonic() < deadline:
        try:
            selected = json.loads((tmp_path / "status.json").read_text())[
                "selected_source"
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if selected and selected.startswith("navigation.test.p"):
            break
        time.sleep(0.01)
    client.close()
    stop.set()
    thread.join(timeout=1.0)

    assert selected and selected.startswith("navigation.test.p")
    assert not thread.is_alive()


def test_producer_restart_gets_a_new_instance_source(tmp_path):
    broker = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
    )
    first = MotionBusClient("navigation.restart", str(tmp_path / "missing.sock"))
    second = MotionBusClient("navigation.restart", str(tmp_path / "missing.sock"))
    try:
        assert first.source != second.source
        assert first.source.startswith("navigation.restart.p")
        assert second.source.startswith("navigation.restart.p")
        assert first._socket.getblocking() is False
        assert second._socket.getblocking() is False

        broker.handle_payload(command(first.source, 100, forward=0.1), 1.0)
        broker.handle_payload(command(second.source, 1, forward=0.2), 1.1)
        status = broker.tick(now_mono=1.2, now_wall=2.0)
        assert status["selected_source"] == second.source
        assert status["selected_sequence"] == 1
    finally:
        first.close(release=False)
        second.close(release=False)


def test_second_broker_cannot_replace_active_socket(tmp_path):
    socket_path = f"/tmp/groot-motion-owner-{os.getpid()}-{time.time_ns()}.sock"
    first = MotionCommandBroker(
        socket_path=socket_path,
        command_file=None,
        status_file=str(tmp_path / "first-status.json"),
        safety_file=str(tmp_path / "first-safety.json"),
    )
    second = MotionCommandBroker(
        socket_path=socket_path,
        command_file=None,
        status_file=str(tmp_path / "second-status.json"),
        safety_file=str(tmp_path / "second-safety.json"),
    )
    first._bind()
    try:
        with pytest.raises(MotionBusError, match="already active"):
            second._bind()
        second.close()
        assert os.path.exists(socket_path)

        client = MotionBusClient("navigation.owner", socket_path)
        try:
            client.publish(forward=0.2)
            deadline = time.monotonic() + 1.0
            while True:
                try:
                    data = first._socket.recv(8192)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        pytest.fail("active broker did not receive producer datagram")
                    time.sleep(0.005)
            assert first.handle_datagram(data)
            status = first.tick()
            assert status["selected_source"] == client.source
        finally:
            client.close()
    finally:
        first.close()


def test_candidate_stream_has_no_command_file_hot_path(tmp_path):
    stream_path = f"/tmp/groot-command-stream-{os.getpid()}-{time.time_ns()}.sock"
    receiver = LatestCommandReceiver(stream_path)
    receiver.bind()
    broker = MotionCommandBroker(
        command_file=None,
        status_file=str(tmp_path / "status.json"),
        safety_file=str(tmp_path / "safety.json"),
        output_sockets=[stream_path],
    )
    try:
        broker.handle_payload(command("navigation", 1, forward=0.3), 5.0)
        status = broker.tick(now_mono=5.1, now_wall=50.0)
        receiver.drain(now=5.1, now_wall=50.0)
        raw, fresh = receiver.latest(now=5.1, stale_s=0.4)
        assert fresh is True
        assert raw["velocity"]["forward"] == pytest.approx(0.3)
        assert raw["broker_id"] == broker.broker_id
        assert status["command_file"] is None
        assert status["metrics"]["file_writes"] == 0
        assert status["metrics"]["stream_sends"] == 1
    finally:
        broker.close()
        receiver.close()


def test_status_writes_ignore_steady_command_sequence_changes(tmp_path, monkeypatch):
    writes = []

    def record_write(path, payload):
        writes.append((path, dict(payload)))

    monkeypatch.setattr(motion_bus_module, "_atomic_write_json", record_write)
    status_path = tmp_path / "status.json"
    broker = MotionCommandBroker(
        command_file=None,
        status_file=str(status_path),
        safety_file=str(tmp_path / "safety.json"),
    )
    for sequence in range(1, 21):
        now = 10.0 + sequence * 0.02
        broker.handle_payload(
            command("navigation", sequence, forward=0.2, lease_s=1.0),
            now,
        )
        broker.tick(now_mono=now, now_wall=100.0 + sequence * 0.02)

    status_writes = [payload for path, payload in writes if path == status_path]
    assert len(status_writes) == 1
    assert status_writes[0]["selected_sequence"] == 1

    broker.tick(now_mono=11.1, now_wall=101.1)
    status_writes = [payload for path, payload in writes if path == status_path]
    assert len(status_writes) == 2
    assert status_writes[-1]["selected_sequence"] == 20
