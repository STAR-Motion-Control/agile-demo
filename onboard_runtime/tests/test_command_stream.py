import json

import pytest

from onboard_runtime.command_stream import LatestCommandReceiver


class DatagramQueue:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    def recv(self, _max_datagram):
        if not self._payloads:
            raise BlockingIOError
        return self._payloads.pop(0)


def stream_frame(sequence, timestamp, *, broker_id="broker-a"):
    return json.dumps(
        {
            "schema_version": 1,
            "stream_type": "motion_command",
            "broker_id": broker_id,
            "broker_sequence": sequence,
            "timestamp": timestamp,
            "fsm": "RL_FULL",
            "velocity": {"forward": 0.0, "lateral": 0.0, "yaw": 0.0},
            "height": 0.76,
        }
    ).encode()


def receiver_with(payloads, **kwargs):
    receiver = LatestCommandReceiver("/unused/test-command-stream.sock", **kwargs)
    receiver._socket = DatagramQueue(payloads)
    return receiver


def test_old_wire_timestamp_is_stale_immediately_after_backlog_drain():
    receiver = receiver_with(
        [stream_frame(40, 995.0), stream_frame(41, 996.0)]
    )

    assert receiver.drain(now=20.0, now_wall=1000.0) == 2
    raw, fresh = receiver.latest(now=20.0, stale_s=0.4)

    assert raw["broker_sequence"] == 41
    assert fresh is False
    assert receiver.metrics == {"accepted": 2, "rejected": 0}


def test_wire_age_and_local_age_both_count_toward_staleness():
    receiver = receiver_with([stream_frame(1, 999.8)])
    receiver.drain(now=10.0, now_wall=1000.0)

    assert receiver.latest(now=10.1, stale_s=0.4)[1] is True
    assert receiver.latest(now=10.21, stale_s=0.4)[1] is False


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_wire_timestamp_is_rejected(timestamp):
    receiver = receiver_with([stream_frame(1, timestamp)])

    assert receiver.drain(now=10.0, now_wall=1000.0) == 1
    assert receiver.latest(now=10.0)[0] is None
    assert receiver.metrics == {"accepted": 0, "rejected": 1}


def test_timestamp_beyond_future_skew_is_rejected_without_advancing_sequence():
    receiver = receiver_with(
        [stream_frame(2, 1001.1), stream_frame(1, 1000.0)],
        max_future_skew_s=1.0,
    )

    assert receiver.drain(now=10.0, now_wall=1000.0) == 2
    raw, fresh = receiver.latest(now=10.0, stale_s=0.4)

    assert raw["broker_sequence"] == 1
    assert fresh is True
    assert receiver.metrics == {"accepted": 1, "rejected": 1}
