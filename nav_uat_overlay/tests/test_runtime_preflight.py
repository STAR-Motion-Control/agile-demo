import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

NAV_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = NAV_ROOT / "src"
REPO_ROOT = NAV_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from runtime_preflight import (
    RuntimePreflightError,
    load_and_validate,
    validate_motion_bus_status,
)


def _status(input_socket, output_sockets):
    health_files = [
        Path(input_socket).parent / "adapter-health.json",
        Path(input_socket).parent / "merger-health.json",
    ]
    return {
        "schema_version": 1,
        "timestamp": 100.0,
        "healthy": True,
        "health_reason": "healthy",
        "motion_blocked": False,
        "safety_latched": False,
        "broker_id": "broker-b",
        "broker_pid": 1234,
        "repo_root": str(REPO_ROOT),
        "input_socket": str(input_socket),
        "output_sockets": [str(path) for path in output_sockets],
        "health_files": [str(path) for path in health_files],
        "health_runtime_id": "candidate-runtime-123",
        "command_file": None,
    }


def _validate(status, input_socket, output_sockets):
    validate_motion_bus_status(
        status,
        expected_repo_root=str(REPO_ROOT),
        expected_input_socket=str(input_socket),
        required_output_sockets=[str(path) for path in output_sockets],
        required_health_files=[
            str(Path(input_socket).parent / "adapter-health.json"),
            str(Path(input_socket).parent / "merger-health.json"),
        ],
        now=100.2,
        check_socket_files=False,
        check_process_identity=False,
    )


def test_stream_only_b_status_accepts_matching_runtime_identity(tmp_path):
    input_socket = tmp_path / "bus.sock"
    outputs = [tmp_path / "adapter.sock", tmp_path / "merger.sock"]

    _validate(_status(input_socket, outputs), input_socket, outputs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repo_root", "/tmp/legacy-a", "repo_root mismatch"),
        ("input_socket", "/tmp/other-bus.sock", "input_socket mismatch"),
        ("command_file", "/tmp/robojudo_ext_cmd.json", "command_file must be null"),
        ("output_sockets", ["/tmp/unrelated.sock"], "missing required output sockets"),
        ("health_files", ["/tmp/adapter-health.json"], "missing required health gates"),
        ("health_runtime_id", "", "no per-launch health runtime identity"),
    ],
)
def test_preflight_rejects_hybrid_or_wrong_runtime(
    tmp_path,
    field,
    value,
    match,
):
    input_socket = tmp_path / "bus.sock"
    outputs = [tmp_path / "adapter.sock", tmp_path / "merger.sock"]
    status = _status(input_socket, outputs)
    status[field] = value

    with pytest.raises(RuntimePreflightError, match=match):
        _validate(status, input_socket, outputs)


def test_load_and_validate_checks_live_unix_socket_identity():
    with tempfile.TemporaryDirectory(prefix="groot-nav-preflight-", dir="/tmp") as tmp:
        root = Path(tmp)
        paths = [root / "bus.sock", root / "adapter.sock", root / "merger.sock"]
        sockets = []
        try:
            for path in paths:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                sock.bind(str(path))
                sockets.append(sock)
            status_file = root / "status.json"
            payload = _status(paths[0], paths[1:])
            payload["timestamp"] = time.time()
            status_file.write_text(json.dumps(payload), encoding="utf-8")

            load_and_validate(
                str(status_file),
                expected_repo_root=str(REPO_ROOT),
                expected_input_socket=str(paths[0]),
                required_output_sockets=[str(path) for path in paths[1:]],
                required_health_files=payload["health_files"],
                check_process_identity=False,
            )
        finally:
            for sock in sockets:
                sock.close()


def test_launcher_has_uniform_native_thread_limits_and_rejects_bad_value():
    script = NAV_ROOT / "start_nav_refactored.sh"
    source = script.read_text(encoding="utf-8")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
        "OPENCV_NUM_THREADS",
    ):
        assert f'export {variable}="$NATIVE_THREADS"' in source
    assert "--required-health-file \"$ADAPTER_HEALTH_FILE\"" in source
    assert "--required-health-file \"$MERGER_HEALTH_FILE\"" in source
    assert "export GROOT_NAV_LAUNCH_AUTHORIZED=1" in source

    result = subprocess.run(
        ["bash", str(script), "--native-threads", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--native-threads must be an integer in 1..8" in result.stdout
