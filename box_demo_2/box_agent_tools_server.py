#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP tool server skeleton for external agents.

This file intentionally keeps the manipulation implementation thin:

* ``manipulate_object(action="grasp")`` can launch the existing
  ``box_demo_main.py`` as a subprocess when ``--allow-execute`` is set.
* ``manipulate_object(action="place")`` is a documented stub.
* ``query_holding`` returns a structured "unknown" response until the motor-state
  monitor is wired in.
* ``patrol_rotate`` can call the existing mover rotate API when
  ``--allow-execute`` is set.

The goal is to give agent/service integrators a stable HTTP surface without
rewriting the current box demo manipulation loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


@dataclass
class ServerConfig:
    host: str
    port: int
    allow_execute: bool
    python: str
    box_demo_path: Path
    point_cloud_stage: str
    sam3_host: str
    sam3_port: int
    iface: str | None
    locomotion: str
    ipc_url: str
    camera_url: str
    camera_serial: str
    max_attempts: int
    no_confirm: bool
    skip_arm_init: bool
    no_grip_check: bool
    no_vision_log: bool
    walk_scale: float
    walk_velocity: float
    extra_box_args: list[str] = field(default_factory=list)


@dataclass
class ToolRequest:
    request_id: str
    tool: str
    status: str
    created_at: str
    updated_at: str
    action: str | None = None
    item_text: str | None = None
    message: str = ""
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "request_id": self.request_id,
            "tool": self.tool,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message": self.message,
            "details": self.details,
        }
        if self.action is not None:
            out["action"] = self.action
        if self.item_text is not None:
            out["item_text"] = self.item_text
        if self.error_code is not None:
            out["error_code"] = self.error_code
        return out


class ToolService:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._lock = threading.RLock()
        self._requests: dict[str, ToolRequest] = {}
        self._active_request_id: str | None = None
        self._seq = 0
        self._state = "idle"
        self._holding: dict[str, Any] = {
            "holding": "unknown",
            "confidence": "low",
            "item_text": None,
            "source": "not_wired",
            "metrics": {},
            "message": "holding monitor is not wired yet",
        }

    # ------------------------------------------------------------------ public
    def status(self) -> dict[str, Any]:
        with self._lock:
            active = (
                self._requests[self._active_request_id].to_dict()
                if self._active_request_id in self._requests
                else None
            )
            return {
                "ok": True,
                "state": self._state,
                "allow_execute": self.config.allow_execute,
                "active_request": active,
                "holding": dict(self._holding),
                "tools": [
                    "manipulate_object",
                    "query_holding",
                    "patrol_rotate",
                ],
            }

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            req = self._requests.get(request_id)
            return req.to_dict() if req is not None else None

    def manipulate_object(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        action = str(payload.get("action", "")).strip().lower()
        item_text = str(payload.get("item_text", "")).strip()
        if action not in ("grasp", "place"):
            return 400, {
                "status": "failed",
                "error_code": "BAD_REQUEST",
                "message": "action must be 'grasp' or 'place'",
            }
        if not item_text:
            return 400, {
                "status": "failed",
                "error_code": "BAD_REQUEST",
                "message": "item_text is required",
            }
        if action == "place":
            req = self._new_request("manipulate_object", action, item_text)
            self._finish_request(
                req.request_id,
                "failed",
                "place action is not implemented in this wrapper yet",
                error_code="NOT_IMPLEMENTED",
            )
            return 501, self.get_request(req.request_id)

        if not self.config.allow_execute:
            req = self._new_request("manipulate_object", action, item_text)
            self._finish_request(
                req.request_id,
                "rejected",
                "execution is disabled; restart with --allow-execute",
                error_code="EXECUTION_DISABLED",
            )
            return 403, self.get_request(req.request_id)

        with self._lock:
            if self._active_request_id is not None:
                return 409, {
                    "status": "rejected",
                    "error_code": "BUSY",
                    "message": "another motion request is active",
                    "active_request_id": self._active_request_id,
                }
            req = self._new_request_locked("manipulate_object", action, item_text)
            self._active_request_id = req.request_id
            self._state = "manipulating"
            req.status = "accepted"
            req.message = "box_demo_main.py launch accepted"
            req.details["mode"] = "subprocess"
            req.updated_at = now_iso()

        thread = threading.Thread(
            target=self._run_box_demo_grasp,
            args=(req.request_id, item_text),
            name=f"tool-{req.request_id}",
            daemon=True,
        )
        thread.start()
        return 202, self.get_request(req.request_id)

    def query_holding(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            return 200, dict(self._holding)

    def patrol_rotate(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            angle_deg = float(payload.get("angle_deg"))
        except (TypeError, ValueError):
            return 400, {
                "status": "failed",
                "error_code": "BAD_REQUEST",
                "message": "angle_deg must be a number",
            }
        speed_deg_s = payload.get("speed_deg_s")
        try:
            speed_rad_s = (
                None if speed_deg_s is None else math.radians(float(speed_deg_s))
            )
        except (TypeError, ValueError):
            return 400, {
                "status": "failed",
                "error_code": "BAD_REQUEST",
                "message": "speed_deg_s must be a number when provided",
            }

        if not self.config.allow_execute:
            req = self._new_request("patrol_rotate", "rotate", None)
            req.details = {"angle_deg": angle_deg}
            self._finish_request(
                req.request_id,
                "rejected",
                "execution is disabled; restart with --allow-execute",
                error_code="EXECUTION_DISABLED",
            )
            return 403, self.get_request(req.request_id)

        with self._lock:
            if self._active_request_id is not None:
                return 409, {
                    "status": "rejected",
                    "error_code": "BUSY",
                    "message": "another motion request is active",
                    "active_request_id": self._active_request_id,
                }
            req = self._new_request_locked("patrol_rotate", "rotate", None)
            req.status = "accepted"
            req.message = "rotation accepted"
            req.details = {
                "angle_deg": angle_deg,
                "backend": self.config.locomotion,
            }
            self._active_request_id = req.request_id
            self._state = "manipulating"

        thread = threading.Thread(
            target=self._run_patrol_rotate,
            args=(req.request_id, math.radians(angle_deg), speed_rad_s),
            name=f"tool-{req.request_id}",
            daemon=True,
        )
        thread.start()
        return 202, self.get_request(req.request_id)

    # ----------------------------------------------------------------- helpers
    def _new_request(
        self, tool: str, action: str | None, item_text: str | None
    ) -> ToolRequest:
        with self._lock:
            return self._new_request_locked(tool, action, item_text)

    def _new_request_locked(
        self, tool: str, action: str | None, item_text: str | None
    ) -> ToolRequest:
        self._seq += 1
        prefix = "manip" if tool == "manipulate_object" else "patrol"
        request_id = f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self._seq:04d}"
        req = ToolRequest(
            request_id=request_id,
            tool=tool,
            status="accepted",
            created_at=now_iso(),
            updated_at=now_iso(),
            action=action,
            item_text=item_text,
        )
        self._requests[request_id] = req
        return req

    def _finish_request(
        self,
        request_id: str,
        status: str,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            req.status = status
            req.message = message
            req.error_code = error_code
            if details:
                req.details.update(details)
            req.updated_at = now_iso()
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._state = "idle" if status in ("succeeded", "failed") else self._state

    def _mark_running(self, request_id: str, message: str) -> None:
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            req.status = "running"
            req.message = message
            req.updated_at = now_iso()

    def _build_box_demo_cmd(self, item_text: str) -> list[str]:
        cfg = self.config
        cmd = [
            cfg.python,
            str(cfg.box_demo_path),
            "--prompt",
            item_text,
            "--point-cloud-stage",
            cfg.point_cloud_stage,
            "--host",
            cfg.sam3_host,
            "--port",
            str(cfg.sam3_port),
            "--locomotion",
            cfg.locomotion,
            "--max-attempts",
            str(cfg.max_attempts),
            "--walk-scale",
            str(cfg.walk_scale),
            "--walk-velocity",
            str(cfg.walk_velocity),
        ]
        if cfg.iface:
            cmd.extend(["--iface", cfg.iface])
        if cfg.locomotion == "remote":
            cmd.extend(["--ipc-url", cfg.ipc_url])
        if cfg.camera_url:
            cmd.extend(["--camera-url", cfg.camera_url])
        if cfg.camera_serial:
            cmd.extend(["--camera-serial", cfg.camera_serial])
        if cfg.no_confirm:
            cmd.append("--no-confirm")
        if cfg.skip_arm_init:
            cmd.append("--skip-arm-init")
        if cfg.no_grip_check:
            cmd.append("--no-grip-check")
        if cfg.no_vision_log:
            cmd.append("--no-vision-log")
        cmd.extend(cfg.extra_box_args)
        return cmd

    def _run_box_demo_grasp(self, request_id: str, item_text: str) -> None:
        cmd = self._build_box_demo_cmd(item_text)
        self._mark_running(request_id, "box_demo_main.py subprocess is running")
        with self._lock:
            req = self._requests.get(request_id)
            if req is not None:
                req.details["command"] = cmd
                req.details["note"] = (
                    "box_demo_main.py is still interactive in several phases; "
                    "this wrapper only launches and monitors the process."
                )
                req.updated_at = now_iso()
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
            with self._lock:
                req = self._requests.get(request_id)
                if req is not None:
                    req.details["pid"] = proc.pid
                    req.updated_at = now_iso()
            rc = proc.wait()
        except Exception as exc:
            self._finish_request(
                request_id,
                "failed",
                f"failed to launch box_demo_main.py: {exc}",
                error_code="SUBPROCESS_ERROR",
            )
            return

        if rc == 0:
            with self._lock:
                self._holding.update(
                    {
                        "holding": "unknown",
                        "confidence": "low",
                        "item_text": item_text,
                        "source": "box_demo_subprocess_exit",
                        "message": (
                            "box_demo_main.py exited successfully; holding monitor "
                            "is not wired, so final holding state is unknown"
                        ),
                    }
                )
            self._finish_request(
                request_id,
                "succeeded",
                "box_demo_main.py exited successfully",
                details={"returncode": rc},
            )
        else:
            self._finish_request(
                request_id,
                "failed",
                f"box_demo_main.py exited with code {rc}",
                error_code="BOX_DEMO_EXITED_NONZERO",
                details={"returncode": rc},
            )

    def _run_patrol_rotate(
        self, request_id: str, angle_rad: float, speed_rad_s: float | None
    ) -> None:
        self._mark_running(request_id, "rotation command is running")
        try:
            mover = self._make_mover()
            mover.initialize()
            if speed_rad_s is None:
                mover.rotate(angle_rad)
            else:
                mover.rotate(angle_rad, yaw_speed=speed_rad_s)
        except Exception as exc:
            self._finish_request(
                request_id,
                "failed",
                f"rotation failed: {exc}",
                error_code="LOCOMOTION_ERROR",
            )
            return
        self._finish_request(
            request_id,
            "succeeded",
            "rotation command completed",
            details={"angle_deg": math.degrees(angle_rad)},
        )

    def _make_mover(self):
        cfg = self.config
        if cfg.locomotion == "groot":
            from groot_mover import GrootMover

            return GrootMover()
        if cfg.locomotion == "remote":
            from remote_mover import RemoteMover

            return RemoteMover(cfg.ipc_url)
        from robot_move import RobotMover

        return RobotMover(velocity=cfg.walk_velocity)


class Handler(BaseHTTPRequestHandler):
    server_version = "BoxAgentTools/0.1"

    @property
    def service(self) -> ToolService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"[agent_tools] {self.address_string()} {fmt % args}")

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, "invalid Content-Length"
        if length <= 0:
            return {}, None
        if length > 1024 * 1024:
            return None, "request body too large"
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"invalid JSON: {exc}"
        if not isinstance(data, dict):
            return None, "JSON body must be an object"
        return data, None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in ("/", "/health"):
            self._json({"ok": True, "service": "box_agent_tools"})
            return
        if path == "/status":
            self._json(self.service.status())
            return
        if path.startswith("/requests/"):
            request_id = path.split("/", 2)[2]
            payload = self.service.get_request(request_id)
            if payload is None:
                self._json({"ok": False, "error": "request not found"}, status=404)
            else:
                self._json(payload)
            return
        self._json({"ok": False, "error": f"unknown endpoint: {path}"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        payload, err = self._read_json()
        if err is not None:
            self._json(
                {"status": "failed", "error_code": "BAD_REQUEST", "message": err},
                status=400,
            )
            return
        assert payload is not None

        if path == "/tools/manipulate_object":
            status, body = self.service.manipulate_object(payload)
            self._json(body, status=status)
            return
        if path == "/tools/query_holding":
            status, body = self.service.query_holding()
            self._json(body, status=status)
            return
        if path == "/tools/patrol_rotate":
            status, body = self.service.patrol_rotate(payload)
            self._json(body, status=status)
            return
        self._json({"ok": False, "error": f"unknown endpoint: {path}"}, status=404)


class ToolHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler, service: ToolService):
        super().__init__(server_address, handler)
        self.service = service


def parse_extra_box_args(raw: str) -> list[str]:
    if not raw:
        return []
    return [part for part in raw.split(" ") if part]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HTTP agent tool server for box_demo_2")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5055)
    p.add_argument(
        "--allow-execute",
        action="store_true",
        help="actually launch box_demo_main.py / mover commands",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--box-demo-path", default=str(ROOT / "box_demo_main.py"))
    p.add_argument("--point-cloud-stage", choices=("sor", "sor_dbscan"), default="sor_dbscan")
    p.add_argument("--sam3-host", default="192.168.112.198")
    p.add_argument("--sam3-port", type=int, default=5300)
    p.add_argument("--iface", default=os.environ.get("UNITREE_DDS_INTERFACE", ""))
    p.add_argument("--locomotion", choices=("agile", "groot", "remote"), default="agile")
    p.add_argument("--ipc-url", default="http://127.0.0.1:5001")
    p.add_argument("--camera-url", default=os.environ.get("CAMERA_URL", ""))
    p.add_argument("--camera-serial", default=os.environ.get("HEAD_CAMERA_SERIAL", "406122070550"))
    p.add_argument("--max-attempts", type=int, default=7)
    p.add_argument("--confirm", action="store_true", help="do not pass --no-confirm to box_demo_main.py")
    p.add_argument("--skip-arm-init", action="store_true")
    p.add_argument("--no-grip-check", action="store_true")
    p.add_argument("--no-vision-log", action="store_true")
    p.add_argument("--walk-scale", type=float, default=1.0)
    p.add_argument("--walk-velocity", type=float, default=0.7)
    p.add_argument(
        "--extra-box-args",
        default="",
        help="extra space-separated arguments appended to box_demo_main.py",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = ServerConfig(
        host=args.host,
        port=args.port,
        allow_execute=args.allow_execute,
        python=args.python,
        box_demo_path=Path(args.box_demo_path).expanduser().resolve(),
        point_cloud_stage=args.point_cloud_stage,
        sam3_host=args.sam3_host,
        sam3_port=args.sam3_port,
        iface=args.iface or None,
        locomotion=args.locomotion,
        ipc_url=args.ipc_url,
        camera_url=args.camera_url,
        camera_serial=args.camera_serial,
        max_attempts=args.max_attempts,
        no_confirm=not args.confirm,
        skip_arm_init=args.skip_arm_init,
        no_grip_check=args.no_grip_check,
        no_vision_log=args.no_vision_log,
        walk_scale=args.walk_scale,
        walk_velocity=args.walk_velocity,
        extra_box_args=parse_extra_box_args(args.extra_box_args),
    )
    service = ToolService(config)
    httpd = ToolHTTPServer((config.host, config.port), Handler, service)
    print(f"box_agent_tools_server listening on http://{config.host}:{config.port}")
    print(f"allow_execute={config.allow_execute} locomotion={config.locomotion}")
    print("endpoints: /health /status /requests/<id> /tools/manipulate_object /tools/query_holding /tools/patrol_rotate")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbox_agent_tools_server exiting")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

