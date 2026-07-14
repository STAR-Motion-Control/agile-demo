#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agile_http_ipc_server as server
from groot_wbc_boxdemo_adapter import read_external_command, resolve_motion_limits


def setup_server(path: Path):
    server.cancel_motion()
    server._backend = "legacy-ipc"
    server._legacy_cmd_file = str(path)


def read(path: Path):
    return json.loads(path.read_text())


def test_taptap_uses_standard_motion_limits():
    assert resolve_motion_limits(0.50, 0.30, 0.60) == (0.50, 0.30, 0.60)


def test_normal_completion_allows_recovery():
    path = Path(tempfile.mkdtemp()) / "cmd.json"
    setup_server(path)
    server.start_motion(0.4, 0.0, 0.0, 0.06, 0.76, "RL_FULL", 0.02)
    time.sleep(0.10)
    payload = read(path)
    assert payload["velocity"]["forward"] == 0.0
    assert payload["allow_recovery"] is True
    cmd = read_external_command(path, 0.76, 0.4, 0.5, 0.3, 0.6, 0.4, 0.8)
    assert cmd.fresh and cmd.allow_recovery


def test_cancelled_motion_cannot_overwrite_safety_stop():
    path = Path(tempfile.mkdtemp()) / "cmd.json"
    setup_server(path)
    server.start_motion(0.4, 0.0, 0.0, 0.15, 0.76, "RL_FULL", 0.02)
    time.sleep(0.04)
    server.cancel_motion()
    server.write_cmd("RL_FULL", 0.0, 0.0, 0.0, 0.76, allow_recovery=False)
    time.sleep(0.18)
    payload = read(path)
    assert payload["velocity"]["forward"] == 0.0
    assert payload["allow_recovery"] is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ALL PASS ({len(tests)} tests)")
