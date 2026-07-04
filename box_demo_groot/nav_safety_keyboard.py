#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyboard safety console for onboard navigation tests.

This tool never writes /tmp/robojudo_ext_cmd.json directly. It calls the local
HTTP bridge so agile_http_ipc_server.py remains the only IPC writer.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
import urllib.parse
import urllib.request


def call(url: str, path: str, **params) -> None:
    query = urllib.parse.urlencode(params)
    full_url = f"{url.rstrip('/')}{path}"
    if query:
        full_url = f"{full_url}?{query}"
    with urllib.request.urlopen(full_url, timeout=1.0) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety keyboard for G001 onboard nav tests")
    parser.add_argument("--ipc-url", default="http://127.0.0.1:5001")
    parser.add_argument("--height", type=float, default=0.74)
    args = parser.parse_args()

    print("=" * 68)
    print("G001 nav safety keyboard")
    print("  space / z : zero navigation velocity, keep GR00T balance")
    print("  o / d     : DAMP policy e-stop (all joints dq=0 target, kd damping)")
    print("  q         : send stop and exit this keyboard pane")
    print("=" * 68)
    print(f"HTTP bridge: {args.ipc_url}")

    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key in (" ", "z", "s"):
                try:
                    call(args.ipc_url, "/stop", height=args.height)
                    print("\n[STOP] navigation velocity zero, balance kept")
                except Exception as exc:
                    print(f"\n[STOP failed] {exc}")
            elif key in ("o", "d"):
                try:
                    call(args.ipc_url, "/damp", height=args.height)
                    print("\n[DAMP] policy e-stop requested")
                except Exception as exc:
                    print(f"\n[DAMP failed] {exc}")
            elif key == "q":
                try:
                    call(args.ipc_url, "/stop", height=args.height)
                except Exception:
                    pass
                print("\n[exit] stop sent")
                return
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
