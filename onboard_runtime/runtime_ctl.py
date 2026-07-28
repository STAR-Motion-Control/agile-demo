#!/usr/bin/env python3
"""Read runtime status or deliberately clear a latched base safety state."""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from .motion_bus import (
    DEFAULT_SOCKET,
    DEFAULT_STATUS_FILE,
    MotionBusClient,
    MotionBusError,
)


def read_status(path: str = DEFAULT_STATUS_FILE) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError("runtime status must be a JSON object")
    return payload


def clear_latched_safety(
    *,
    socket_path: str = DEFAULT_SOCKET,
    status_file: str = DEFAULT_STATUS_FILE,
    human_approved: bool = False,
    timeout_s: float = 1.0,
) -> dict[str, Any]:
    if not human_approved:
        raise PermissionError(
            "refusing safety clear without --human-approved-safety-clear"
        )
    socket_stat = os.stat(socket_path)
    if not stat.S_ISSOCK(socket_stat.st_mode):
        raise RuntimeError(f"motion bus path is not a Unix socket: {socket_path}")
    before = read_status(status_file)
    if not bool(before.get("safety_latched", False)):
        raise RuntimeError("motion bus has no latched safety state")

    source = f"operator.runtime_ctl.{os.getpid()}"
    client = MotionBusClient(source, socket_path)
    try:
        client.clear_safety()
    finally:
        client.close(release=False)

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    latest = before
    while time.monotonic() < deadline:
        try:
            latest = read_status(status_file)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass
        if not bool(latest.get("safety_latched", True)):
            return latest
        time.sleep(0.02)
    raise TimeoutError("motion bus did not acknowledge the safety clear")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    clear = subparsers.add_parser("clear-safety")
    clear.add_argument("--human-approved-safety-clear", action="store_true")
    clear.add_argument("--timeout-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(read_status(args.status_file), indent=2, sort_keys=True))
        return
    try:
        status = clear_latched_safety(
            socket_path=args.socket,
            status_file=args.status_file,
            human_approved=args.human_approved_safety_clear,
            timeout_s=args.timeout_s,
        )
    except (PermissionError, RuntimeError, TimeoutError, MotionBusError, OSError) as exc:
        parser.error(str(exc))
    print(
        "latched safety cleared; runtime output is zero-speed RL_FULL until a "
        f"new authorized source is selected (source={status.get('selected_source')})"
    )


if __name__ == "__main__":
    main()
