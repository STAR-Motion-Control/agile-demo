#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small HTTP command server for AGILE box_demo_2 IPC.

This intentionally uses Python's stdlib instead of Flask so it works in the
real-robot Python environment without installing extra packages. It writes the
same /tmp/robojudo_ext_cmd.json consumed by agile_lowcmd_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


LEGACY_CMD_FILE = "/tmp/robojudo_ext_cmd.json"
TAPTAP_STATUS_FILE = "/tmp/groot_taptap_status.json"
DEFAULT_STATE_DIR = "/tmp/agile_sim2sim"
STAND_HEIGHT = 0.76
MIN_HEIGHT = 0.30
MAX_HEIGHT = 0.80
FWD_SPEED = 0.40
BACK_SPEED = 0.20
LAT_SPEED = 0.25
YAW_RATE = 0.40

_motion_lock = threading.Lock()
_motion_stop: threading.Event | None = None
_backend = "command-json"
_state_dir = Path(DEFAULT_STATE_DIR)
_legacy_cmd_file = LEGACY_CMD_FILE
_taptap_status_file = TAPTAP_STATUS_FILE


def read_taptap_status(max_age_s: float = 1.0) -> dict:
    try:
        with open(_taptap_status_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        age = max(0.0, time.time() - float(payload.get("timestamp", 0.0)))
        payload["age_s"] = age
        if age > max_age_s:
            payload["active"] = False
            payload["stale"] = True
        return payload
    except FileNotFoundError:
        return {"enabled": False, "active": False, "state": "IDLE", "missing": True}
    except Exception as exc:
        return {"enabled": False, "active": False, "state": "IDLE", "error": repr(exc)}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path) or "/tmp"
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def command_json_path() -> Path:
    return _state_dir / "command.json"


def write_cmd(
    fsm: str = "RL_FULL",
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    height: float = STAND_HEIGHT,
    *,
    estop: bool = False,
    allow_recovery: bool = False,
    defer_recovery: bool = False,
) -> dict:
    t = time.time()
    fsm_out = "DAMP" if estop else fsm
    if _backend == "legacy-ipc":
        payload = {
            "fsm": fsm_out,
            "velocity": {"forward": float(vx), "lateral": float(vy), "yaw": float(wz)},
            "height": clamp(float(height), MIN_HEIGHT, MAX_HEIGHT),
            "units": "agile",
            "timestamp": t,
            "source": "http_ipc",
            "allow_recovery": bool(allow_recovery),
            "defer_recovery": bool(defer_recovery),
        }
        if estop:
            payload["estop"] = True
        atomic_write_json(_legacy_cmd_file, payload)
        return payload

    payload = {
        "fsm": fsm_out,
        "vx": float(vx),
        "vy": float(vy),
        "wz": float(wz),
        "height": clamp(float(height), MIN_HEIGHT, MAX_HEIGHT),
        "timestamp": t,
        "source": "http_ipc",
        "allow_recovery": bool(allow_recovery),
        "defer_recovery": bool(defer_recovery),
    }
    if estop:
        payload["estop"] = True
    atomic_write_json(str(command_json_path()), payload)
    return payload


def read_status() -> dict:
    path = _legacy_cmd_file if _backend == "legacy-ipc" else str(command_json_path())
    try:
        with open(path, "r", encoding="utf-8") as f:
            current = json.load(f)
    except FileNotFoundError:
        current = None
    except Exception as exc:
        current = {"error": repr(exc)}
    with _motion_lock:
        motion_active = _motion_stop is not None
    return {
        "ok": True,
        "backend": _backend,
        "cmd_file": path,
        "current": current,
        "motion_active": motion_active,
        "taptap": read_taptap_status(),
    }


def get_float(qs: dict[str, list[str]], key: str, default: float) -> float:
    try:
        return float(qs.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def get_str(qs: dict[str, list[str]], key: str, default: str) -> str:
    return str(qs.get(key, [default])[0])


def get_bool(qs: dict[str, list[str]], key: str, default: bool = False) -> bool:
    raw = str(qs.get(key, ["1" if default else "0"])[0]).strip().lower()
    return raw in ("1", "true", "yes", "on")


def cancel_motion() -> None:
    global _motion_stop
    with _motion_lock:
        if _motion_stop is not None:
            _motion_stop.set()
        _motion_stop = None


def start_motion(
    vx: float,
    vy: float,
    wz: float,
    duration: float,
    height: float,
    fsm: str,
    refresh_s: float,
    allow_recovery: bool = True,
    defer_recovery: bool = False,
) -> dict:
    global _motion_stop
    duration = max(0.0, float(duration))
    refresh_s = max(0.02, float(refresh_s))
    cancel_motion()
    stop_event = threading.Event()
    with _motion_lock:
        _motion_stop = stop_event

    def _worker() -> None:
        remaining = duration
        last_tick = time.monotonic()
        try:
            while remaining > 0.0 and not stop_event.is_set():
                now = time.monotonic()
                elapsed = max(0.0, now - last_tick)
                last_tick = now
                if not read_taptap_status().get("active", False):
                    remaining -= elapsed
                write_cmd(
                    fsm, vx, vy, wz, height,
                    allow_recovery=allow_recovery,
                    defer_recovery=defer_recovery,
                )
                time.sleep(refresh_s)
            if not stop_event.is_set():
                write_cmd(
                    fsm, 0.0, 0.0, 0.0, height,
                    allow_recovery=allow_recovery,
                    defer_recovery=defer_recovery,
                )
        finally:
            global _motion_stop
            with _motion_lock:
                if _motion_stop is stop_event:
                    _motion_stop = None

    threading.Thread(target=_worker, name="agile_http_motion", daemon=True).start()
    return {
        "ok": True,
        "vx": vx,
        "vy": vy,
        "wz": wz,
        "height": height,
        "duration": duration,
        "refresh_s": refresh_s,
        "allow_recovery": bool(allow_recovery),
        "defer_recovery": bool(defer_recovery),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AgileBoxHTTP/1.0"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        height = get_float(qs, "height", STAND_HEIGHT)
        fsm = get_str(qs, "fsm", "RL_FULL")
        refresh_s = get_float(qs, "refresh", 0.05)

        try:
            if path in ("/", "/health"):
                self._json({"ok": True, "endpoints": ["/forward", "/cmd", "/stop", "/height", "/mode", "/damp", "/status"]})
            elif path == "/status":
                self._json(read_status())
            elif path == "/stop":
                cancel_motion()
                self._json(write_cmd(
                    fsm, 0.0, 0.0, 0.0, height,
                    allow_recovery=get_bool(qs, "allow_recovery", False),
                    defer_recovery=get_bool(qs, "defer_recovery", False),
                ))
            elif path == "/damp":
                cancel_motion()
                self._json(write_cmd("DAMP", 0.0, 0.0, 0.0, height, estop=True))
            elif path == "/height":
                cancel_motion()
                self._json(write_cmd(fsm, 0.0, 0.0, 0.0, height))
            elif path == "/mode":
                cancel_motion()
                mode = get_str(qs, "fsm", "RL_FULL")
                self._json(write_cmd(mode, 0.0, 0.0, 0.0, height))
            elif path == "/cmd":
                vx = get_float(qs, "vx", 0.0)
                vy = get_float(qs, "vy", 0.0)
                wz = get_float(qs, "wz", 0.0)
                duration = get_float(qs, "duration", 0.0)
                if duration > 0:
                    self._json(start_motion(
                        vx, vy, wz, duration, height, fsm, refresh_s,
                        allow_recovery=get_bool(qs, "allow_recovery", True),
                        defer_recovery=get_bool(qs, "defer_recovery", False),
                    ))
                else:
                    cancel_motion()
                    self._json(write_cmd(
                        fsm, vx, vy, wz, height,
                        allow_recovery=get_bool(qs, "allow_recovery", True),
                        defer_recovery=get_bool(qs, "defer_recovery", False),
                    ))
            elif path == "/forward":
                distance = get_float(qs, "distance", 0.08)
                default_speed = FWD_SPEED if distance >= 0.0 else BACK_SPEED
                speed = abs(get_float(qs, "speed", default_speed))
                if distance < 0.0:
                    speed = min(speed, BACK_SPEED)
                speed = max(speed, 1e-6)
                vx = math.copysign(speed, distance)
                duration = abs(distance) / speed
                self._json(start_motion(vx, 0.0, 0.0, duration, height, fsm, refresh_s))
            elif path == "/lateral":
                distance = get_float(qs, "distance", 0.05)
                speed = abs(get_float(qs, "speed", LAT_SPEED))
                speed = max(speed, 1e-6)
                vy = math.copysign(speed, distance)
                duration = abs(distance) / speed
                self._json(start_motion(0.0, vy, 0.0, duration, height, fsm, refresh_s))
            elif path == "/rotate":
                angle = get_float(qs, "angle", 0.1)
                yaw_rate = abs(get_float(qs, "yaw_rate", YAW_RATE))
                yaw_rate = max(yaw_rate, 1e-6)
                wz = math.copysign(yaw_rate, angle)
                duration = abs(angle) / yaw_rate
                self._json(start_motion(0.0, 0.0, wz, duration, height, fsm, refresh_s))
            else:
                self._json({"ok": False, "error": f"unknown endpoint: {path}"}, status=404)
        except Exception as exc:
            self._json({"ok": False, "error": repr(exc)}, status=500)


def main() -> None:
    global _backend, _state_dir, _legacy_cmd_file, _taptap_status_file
    global STAND_HEIGHT, MIN_HEIGHT, MAX_HEIGHT
    parser = argparse.ArgumentParser(description="HTTP IPC server for AGILE box_demo_2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--backend", choices=("command-json", "legacy-ipc"), default="command-json")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--legacy-cmd-file", default=LEGACY_CMD_FILE)
    parser.add_argument("--taptap-status-file", default=TAPTAP_STATUS_FILE)
    parser.add_argument("--stand-height", type=float, default=STAND_HEIGHT)
    parser.add_argument("--min-height", type=float, default=MIN_HEIGHT)
    parser.add_argument("--max-height", type=float, default=MAX_HEIGHT)
    args = parser.parse_args()
    if not args.min_height <= args.stand_height <= args.max_height:
        parser.error("--stand-height must be inside --min-height..--max-height")
    STAND_HEIGHT = float(args.stand_height)
    MIN_HEIGHT = float(args.min_height)
    MAX_HEIGHT = float(args.max_height)
    _backend = args.backend
    _state_dir = Path(args.state_dir).expanduser()
    _state_dir.mkdir(parents=True, exist_ok=True)
    _legacy_cmd_file = args.legacy_cmd_file
    _taptap_status_file = args.taptap_status_file
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AGILE box_demo HTTP IPC server: http://{args.host}:{args.port}")
    print(f"backend={_backend}")
    print(f"height: stand={STAND_HEIGHT:.2f} range={MIN_HEIGHT:.2f}..{MAX_HEIGHT:.2f}")
    print(f"Writing commands to {read_status()['cmd_file']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HTTP IPC server")
        cancel_motion()
        write_cmd("RL_FULL", 0.0, 0.0, 0.0, STAND_HEIGHT)


if __name__ == "__main__":
    main()
