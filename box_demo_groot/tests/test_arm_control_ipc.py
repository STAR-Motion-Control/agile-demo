import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arm_control_ipc as arm_ipc


class FakeDatagramSocket:
    def __init__(self, *, accepted=True, message="accepted"):
        self.accepted = accepted
        self.message = message
        self.response = b""
        self.closed = False

    def setblocking(self, _enabled):
        pass

    def sendto(self, data, _path):
        request = json.loads(data.decode("utf-8"))
        self.response = json.dumps(
            {
                "schema_version": 1,
                "request_id": request["request_id"],
                "source": request["source"],
                "sequence": request["sequence"],
                "accepted": self.accepted,
                "message": self.message,
            }
        ).encode()

    def recv(self, _size):
        return self.response

    def close(self):
        self.closed = True


def make_client(monkeypatch, fake):
    monkeypatch.setattr(arm_ipc.socket, "socket", lambda *_a, **_k: fake)
    client = arm_ipc.ArmControlClient(ack_timeout_s=0.05)
    monkeypatch.setattr(client, "_ensure_bound", lambda: None)
    return client


def test_client_source_is_instance_scoped(monkeypatch):
    first = make_client(monkeypatch, FakeDatagramSocket())
    second = make_client(monkeypatch, FakeDatagramSocket())
    try:
        assert first.source != second.source
        assert first.source.startswith("operator.keyboard.p")
        assert second.source.startswith("operator.keyboard.p")
    finally:
        first.close(release=False)
        second.close(release=False)


def test_request_returns_server_ack_result(monkeypatch):
    fake = FakeDatagramSocket(accepted=False, message="base safety active")
    client = make_client(monkeypatch, fake)
    monkeypatch.setattr(
        arm_ipc.select,
        "select",
        lambda *_args, **_kwargs: ([fake], [], []),
    )
    try:
        assert client.toggle() == (False, "base safety active")
    finally:
        client.close(release=False)


def test_request_reports_missing_ack(monkeypatch):
    fake = FakeDatagramSocket()
    client = make_client(monkeypatch, fake)
    monkeypatch.setattr(
        arm_ipc.select,
        "select",
        lambda *_args, **_kwargs: ([], [], []),
    )
    try:
        accepted, message = client.release()
        assert accepted is False
        assert "ACK timeout" in message
    finally:
        client.close(release=False)
