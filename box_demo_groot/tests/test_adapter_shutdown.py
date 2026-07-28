#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import groot_wbc_boxdemo_adapter as adapter  # noqa: E402


class FakePublisher:
    def __init__(self, fail_calls: set[int] | None = None):
        self.fail_calls = set() if fail_calls is None else set(fail_calls)
        self.calls: list[tuple] = []

    def _record(self, call: tuple) -> None:
        self.calls.append(call)
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("synthetic DDS write failure")

    def publish_limp(self) -> None:
        self._record(("limp",))

    def publish_damping(self, low_state, damping_kd: float) -> None:
        self._record(("damp", low_state, damping_kd))


class FakeBody:
    def __init__(self, low_state):
        self.body_state_processor = type(
            "Processor", (), {"robot_low_state": low_state}
        )()


class FakeEnv:
    def __init__(self, low_states=()):
        self.low_states = list(low_states)

    def body(self):
        low_state = self.low_states.pop(0) if self.low_states else None
        return FakeBody(low_state)


def shutdown(publisher, env, **overrides) -> int:
    options = {
        "action": "limp",
        "duration_s": 0.06,
        "hz": 50.0,
        "damping_kd": 8.0,
        "dry_run": False,
        "publisher": publisher,
        "env": env,
        "sleep_fn": lambda _period: None,
    }
    options.update(overrides)
    return adapter.publish_shutdown_action(**options)


def test_limp_keeps_publishing_after_transient_failure():
    publisher = FakePublisher(fail_calls={1})

    published = shutdown(publisher, FakeEnv())

    assert published == 2
    assert publisher.calls == [("limp",), ("limp",), ("limp",)]


def test_damp_retries_until_a_low_state_is_available():
    low_state = object()
    publisher = FakePublisher()

    published = shutdown(
        publisher,
        FakeEnv([None, low_state, low_state]),
        action="damp",
    )

    assert published == 3
    assert publisher.calls == [
        ("limp",),
        ("damp", low_state, 8.0),
        ("damp", low_state, 8.0),
    ]


def test_damp_publish_failure_falls_back_to_limp_in_same_attempt():
    low_state = object()
    publisher = FakePublisher(fail_calls={1})

    published = shutdown(
        publisher,
        FakeEnv([low_state]),
        action="damp",
        duration_s=0.01,
    )

    assert published == 1
    assert publisher.calls == [
        ("damp", low_state, 8.0),
        ("limp",),
    ]


def test_runtime_damp_needs_no_policy_observation_and_falls_back_without_state():
    publisher = FakePublisher()
    command = adapter.ExternalCommand(
        "DAMP", 0.0, 0.0, 0.0, 0.76, True, estop=True
    )

    actual = adapter.publish_runtime_safety_frame(
        command=command,
        publisher=publisher,
        env=FakeEnv(),
        damping_kd=8.0,
        dry_run=False,
    )

    assert actual == "LIMP"
    assert publisher.calls == [("limp",)]


def test_total_publish_failure_is_contained_for_original_exception():
    publisher = FakePublisher(fail_calls={1, 2, 3})

    published = shutdown(publisher, FakeEnv())

    assert published == 0
    assert len(publisher.calls) == 3


def test_none_and_dry_run_do_not_publish():
    publisher = FakePublisher()

    assert shutdown(publisher, FakeEnv(), action="none") == 0
    assert shutdown(publisher, FakeEnv(), dry_run=True) == 0
    assert publisher.calls == []
