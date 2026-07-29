import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from onboard_runtime.runtime_startup import (
    StartupError,
    require_unix_socket,
    validate_broker_status,
    validate_health_status,
    validate_pid_file,
    wait_until,
)


RUNTIME_ID = "candidate-12345678"


def _broker_status(tmp_path: Path, *, now: float = 100.0) -> tuple[dict, dict]:
    repo_root = tmp_path / "checkout"
    input_socket = tmp_path / "motion.sock"
    output_sockets = [tmp_path / "adapter.sock", tmp_path / "merger.sock"]
    health_files = [tmp_path / "adapter.json", tmp_path / "merger.json"]
    expected = {
        "runtime_id": RUNTIME_ID,
        "repo_root": str(repo_root),
        "input_socket": str(input_socket),
        "output_sockets": [str(path) for path in output_sockets],
        "health_files": [str(path) for path in health_files],
        "started_after": 99.0,
        "max_age_s": 2.5,
        "now": now,
        "expected_pid": os.getpid(),
    }
    payload = {
        "schema_version": 1,
        "timestamp": now,
        "healthy": False,
        "health_reason": "health_missing",
        "broker_id": "broker-abc",
        "broker_pid": os.getpid(),
        "repo_root": str(repo_root),
        "input_socket": str(input_socket),
        "output_sockets": [str(path) for path in output_sockets],
        "command_file": None,
        "health_files": [str(path) for path in health_files],
        "health_runtime_id": RUNTIME_ID,
    }
    return payload, expected


def test_broker_barrier_accepts_bound_but_health_gated_status(tmp_path):
    payload, expected = _broker_status(tmp_path)
    validate_broker_status(payload, **expected)

    with pytest.raises(StartupError, match="not healthy"):
        validate_broker_status(payload, **expected, require_healthy=True)


def test_composite_health_does_not_clear_or_reject_a_safety_latch(tmp_path):
    payload, expected = _broker_status(tmp_path)
    payload["healthy"] = True
    payload["health_reason"] = "healthy"
    payload["safety_latched"] = True

    validate_broker_status(payload, **expected, require_healthy=True)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("health_runtime_id", "old-runtime", "runtime_id"),
        ("repo_root", "/wrong/checkout", "repo_root"),
        ("input_socket", "/tmp/wrong.sock", "input_socket"),
        ("output_sockets", ["/tmp/wrong.sock"], "output_sockets"),
        ("health_files", ["/tmp/wrong.json"], "health_files"),
        ("command_file", "/tmp/legacy.json", "command_file"),
        ("broker_pid", 1, "broker_pid"),
    ],
)
def test_broker_barrier_rejects_mismatched_identity_and_paths(
    tmp_path, field, replacement, message
):
    payload, expected = _broker_status(tmp_path)
    payload[field] = replacement
    with pytest.raises(StartupError, match=message):
        validate_broker_status(payload, **expected)


def test_broker_barrier_rejects_old_or_stale_status(tmp_path):
    payload, expected = _broker_status(tmp_path, now=100.0)
    payload["timestamp"] = 98.0
    with pytest.raises(StartupError, match="predates"):
        validate_broker_status(payload, **expected)

    payload["timestamp"] = 100.0
    expected["started_after"] = 99.0
    expected["now"] = 103.0
    with pytest.raises(StartupError, match="stale"):
        validate_broker_status(payload, **expected)


def test_merger_readiness_allows_rl_missing_but_composite_does_not():
    payload = {
        "schema_version": 1,
        "timestamp": 100.0,
        "runtime_id": RUNTIME_ID,
        "component": "lowcmd_merger",
        "healthy": False,
        "reason": "rl_missing",
    }
    kwargs = {
        "runtime_id": RUNTIME_ID,
        "component": "lowcmd_merger",
        "started_after": 99.0,
        "max_age_s": 2.5,
        "now": 100.0,
    }
    validate_health_status(payload, **kwargs)
    with pytest.raises(StartupError, match="rl_missing"):
        validate_health_status(payload, **kwargs, require_healthy=True)

    payload["healthy"] = True
    payload["reason"] = "healthy"
    validate_health_status(payload, **kwargs, require_healthy=True)


def test_component_health_rejects_wrong_runtime_and_stale_file():
    payload = {
        "schema_version": 1,
        "timestamp": 100.0,
        "runtime_id": "old-runtime",
        "healthy": True,
        "reason": "healthy",
    }
    with pytest.raises(StartupError, match="runtime_id"):
        validate_health_status(
            payload,
            runtime_id=RUNTIME_ID,
            started_after=99.0,
            max_age_s=2.5,
            now=100.0,
        )

    payload["runtime_id"] = RUNTIME_ID
    with pytest.raises(StartupError, match="stale"):
        validate_health_status(
            payload,
            runtime_id=RUNTIME_ID,
            started_after=99.0,
            max_age_s=2.5,
            now=103.0,
        )


def test_pid_file_must_name_a_live_process(tmp_path):
    pid_file = tmp_path / "owner.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")
    assert validate_pid_file(pid_file) == os.getpid()

    pid_file.write_text("not-a-pid\n", encoding="ascii")
    with pytest.raises(StartupError, match="valid PID"):
        validate_pid_file(pid_file)


def test_required_endpoint_must_be_a_unix_socket(tmp_path, monkeypatch):
    path = tmp_path / "endpoint.sock"
    path.write_text("not a socket", encoding="ascii")
    with pytest.raises(StartupError, match="not a Unix socket"):
        require_unix_socket(path)

    monkeypatch.setattr(
        Path, "stat", lambda _path: SimpleNamespace(st_mode=stat.S_IFSOCK)
    )
    require_unix_socket(path)


def test_stable_wait_resets_after_a_health_interruption():
    class FakeTime:
        now = 0.0

        def clock(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    fake = FakeTime()

    def check():
        if 0.19 <= fake.now < 0.31:
            raise StartupError("transient unhealthy")
        return "ready"

    result = wait_until(
        check,
        timeout_s=1.0,
        stable_s=0.25,
        poll_s=0.1,
        clock=fake.clock,
        sleeper=fake.sleep,
    )
    assert result == "ready"
    assert fake.now >= 0.6


def test_launcher_orders_barriers_before_downstream_components():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "box_demo_groot"
        / "start_g1_onboard_runtime.sh"
    ).read_text(encoding="utf-8")

    broker = launcher.index("# STARTUP_STAGE: broker")
    wait_broker = launcher.index(" wait-broker \\")
    merger = launcher.index("# STARTUP_STAGE: merger")
    wait_merger = launcher.index(" wait-component \\")
    adapter = launcher.index("# STARTUP_STAGE: adapter")
    composite = launcher.index("# STARTUP_STAGE: composite")
    wait_composite = launcher.index(" wait-composite \\")
    keyboard = launcher.index('if [[ "$KEYBOARD" == "1" ]]')

    assert broker < wait_broker < merger < wait_merger < adapter
    assert adapter < composite < wait_composite < keyboard
    assert "merge_lowcmd_arm_sdk.py" not in launcher[broker:wait_broker]
    assert "groot_wbc_boxdemo_adapter.py" not in launcher[broker:merger]
    assert "sleep" not in launcher[broker:keyboard]


def test_launcher_failure_cleanup_is_scoped_to_created_session():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "box_demo_groot"
        / "start_g1_onboard_runtime.sh"
    ).read_text(encoding="utf-8")
    cleanup = launcher[
        launcher.index("cleanup_failed_startup()") : launcher.index(
            "# STARTUP_STAGE: broker"
        )
    ]

    assert 'SESSION_CREATED" == "1"' in cleanup
    assert 'STARTUP_COMPLETE" != "1"' in cleanup
    assert 'tmux kill-session -t "$SESSION"' in cleanup
    assert "pkill" not in launcher
    assert "killall" not in launcher
    assert "STARTUP_COMPLETE=1\ntrap - EXIT" in launcher


def test_launcher_busy_guard_covers_runtime_keyboard():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "box_demo_groot"
        / "start_g1_onboard_runtime.sh"
    ).read_text(encoding="utf-8")
    busy_guard = launcher[
        launcher.index('BUSY="$(pgrep -af') : launcher.index(
            'if [[ -n "$BUSY" ]]'
        )
    ]

    assert "agile_runtime_keyboard.py" in busy_guard


def test_core_panes_exec_their_process_after_writing_pid():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "box_demo_groot"
        / "start_g1_onboard_runtime.sh"
    ).read_text(encoding="utf-8")
    keyboard = launcher.index('if [[ "$KEYBOARD" == "1" ]]')
    core = launcher[launcher.index("# STARTUP_STAGE: broker") : keyboard]

    assert core.count("echo \\$\\$ >") == 3
    assert "exec python -m onboard_runtime.motion_bus" in core
    assert "exec python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py'" in core
    assert "exec python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py'" in core
    assert "exec bash" not in core
