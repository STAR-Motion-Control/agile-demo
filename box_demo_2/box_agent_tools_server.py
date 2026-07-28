#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cold HTTP ingress for the existing box demo.

The server process deliberately imports only the Python standard library. Robot
SDK, camera, perception, and IK modules are loaded only by a child process after
an explicitly enabled request. Exactly one motion child may exist at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "rejected", "cancelled", "timed_out"}
)

# This source is interpreted only in a patrol child process. Keeping it as data
# here prevents mover imports (and their DDS dependencies) in the waiting server.
PATROL_CHILD_CODE = r"""
import signal
import sys

backend, ipc_url, velocity_text, angle_text, speed_text = sys.argv[1:]
if backend == "groot":
    from groot_mover import GrootMover
    mover = GrootMover()
elif backend == "remote":
    from remote_mover import RemoteMover
    mover = RemoteMover(ipc_url)
else:
    from robot_move import RobotMover
    mover = RobotMover(velocity=float(velocity_text))

def stop_then_exit(signum, _frame):
    try:
        mover.stop()
    finally:
        raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, stop_then_exit)
try:
    mover.initialize()
    if speed_text:
        mover.rotate(float(angle_text), yaw_speed=float(speed_text))
    else:
        mover.rotate(float(angle_text))
finally:
    mover.stop()
"""


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
    job_timeout_seconds: float = 900.0
    terminate_grace_seconds: float = 3.0
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
            "details": dict(self.details),
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
        if (
            not math.isfinite(config.job_timeout_seconds)
            or config.job_timeout_seconds <= 0
        ):
            raise ValueError("job_timeout_seconds must be greater than zero")
        if (
            not math.isfinite(config.terminate_grace_seconds)
            or config.terminate_grace_seconds <= 0
        ):
            raise ValueError("terminate_grace_seconds must be greater than zero")
        self.config = config
        self._lock = threading.RLock()
        self._requests: dict[str, ToolRequest] = {}
        self._active_request_id: str | None = None
        self._active_process: subprocess.Popen | None = None
        self._worker_thread: threading.Thread | None = None
        self._cancel_requested: set[str] = set()
        self._closing = False
        self._closed = False
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
                "closing": self._closing,
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

    def cancel_request(self, request_id: str) -> tuple[int, dict[str, Any]]:
        """Request cancellation and synchronously stop the active process group."""
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return 404, {
                    "status": "failed",
                    "error_code": "NOT_FOUND",
                    "message": "request not found",
                }
            if req.status in TERMINAL_STATUSES:
                return 409, {
                    "status": "rejected",
                    "error_code": "NOT_ACTIVE",
                    "message": f"request is already {req.status}",
                    "request_id": request_id,
                }
            if self._active_request_id != request_id:
                return 409, {
                    "status": "rejected",
                    "error_code": "NOT_ACTIVE",
                    "message": "request is not the active motion request",
                    "request_id": request_id,
                }

            self._cancel_requested.add(request_id)
            req.status = "cancelling"
            req.message = "cancellation requested"
            req.details["cancel_requested_at"] = now_iso()
            req.updated_at = now_iso()
            proc = self._active_process

        stopped = proc is None or self._terminate_process(proc)
        if stopped:
            with self._lock:
                worker = self._worker_thread
                if (
                    proc is not None
                    and self._active_process is proc
                    and (worker is None or not worker.is_alive())
                ):
                    self._active_process = None
                    self._finish_request(
                        request_id,
                        "cancelled",
                        "child process was cancelled and reaped",
                        error_code="CANCELLED",
                        details={"returncode": proc.returncode},
                    )
        current = self.get_request(request_id)
        assert current is not None
        return 202, current

    def close(self) -> None:
        """Reject new work and stop/reap the sole child process, if one exists."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._state = "stopping"
            request_id = self._active_request_id
            if request_id is not None:
                self._cancel_requested.add(request_id)
                req = self._requests.get(request_id)
                if req is not None and req.status not in TERMINAL_STATUSES:
                    req.status = "cancelling"
                    req.message = "service shutdown requested"
                    req.details["cancel_requested_at"] = now_iso()
                    req.updated_at = now_iso()
            proc = self._active_process
            worker = self._worker_thread

        stopped = proc is None or self._terminate_process(proc)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(5.0, self.config.terminate_grace_seconds * 3 + 1))

        with self._lock:
            if (
                stopped
                and request_id is not None
                and self._active_request_id == request_id
                and (self._worker_thread is None or not self._worker_thread.is_alive())
            ):
                if self._active_process is proc:
                    self._active_process = None
                self._finish_request(
                    request_id,
                    "cancelled",
                    "child process was cancelled during service shutdown",
                    error_code="CANCELLED",
                    details={
                        "returncode": proc.returncode if proc is not None else None
                    },
                )
            if self._active_request_id is None:
                self._state = "closed"
                self._closed = True
            else:
                self._state = "shutdown_incomplete"

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

        command = self._build_box_demo_cmd(item_text)
        with self._lock:
            if self._closing:
                return 503, {
                    "status": "rejected",
                    "error_code": "SHUTTING_DOWN",
                    "message": "service is shutting down",
                }
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
            req.details["timeout_seconds"] = self.config.job_timeout_seconds
            req.updated_at = now_iso()
            thread = threading.Thread(
                target=self._run_subprocess,
                args=(req.request_id, command),
                name=f"tool-{req.request_id}",
                daemon=True,
            )
            self._worker_thread = thread
            try:
                thread.start()
            except RuntimeError as exc:
                self._worker_thread = None
                self._finish_request(
                    req.request_id,
                    "failed",
                    f"failed to start worker thread: {exc}",
                    error_code="WORKER_START_ERROR",
                )
                current = self.get_request(req.request_id)
                assert current is not None
                return 500, current
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
        if not math.isfinite(angle_deg):
            return 400, {
                "status": "failed",
                "error_code": "BAD_REQUEST",
                "message": "angle_deg must be finite",
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
        if speed_rad_s is not None and (
            not math.isfinite(speed_rad_s) or speed_rad_s <= 0
        ):
            return 400, {
                "status": "failed",
                "error_code": "BAD_REQUEST",
                "message": "speed_deg_s must be finite and greater than zero",
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

        command = self._build_patrol_cmd(math.radians(angle_deg), speed_rad_s)
        with self._lock:
            if self._closing:
                return 503, {
                    "status": "rejected",
                    "error_code": "SHUTTING_DOWN",
                    "message": "service is shutting down",
                }
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
                "mode": "subprocess",
                "timeout_seconds": self.config.job_timeout_seconds,
            }
            self._active_request_id = req.request_id
            self._state = "manipulating"
            thread = threading.Thread(
                target=self._run_subprocess,
                args=(req.request_id, command),
                name=f"tool-{req.request_id}",
                daemon=True,
            )
            self._worker_thread = thread
            try:
                thread.start()
            except RuntimeError as exc:
                self._worker_thread = None
                self._finish_request(
                    req.request_id,
                    "failed",
                    f"failed to start worker thread: {exc}",
                    error_code="WORKER_START_ERROR",
                )
                current = self.get_request(req.request_id)
                assert current is not None
                return 500, current
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
            req.details.setdefault("finished_at", now_iso())
            req.updated_at = now_iso()
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._state = "stopping" if self._closing else "idle"
            self._cancel_requested.discard(request_id)

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
            "--single-run",
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

    def _build_patrol_cmd(
        self, angle_rad: float, speed_rad_s: float | None
    ) -> list[str]:
        cfg = self.config
        return [
            cfg.python,
            "-c",
            PATROL_CHILD_CODE,
            cfg.locomotion,
            cfg.ipc_url,
            str(cfg.walk_velocity),
            str(angle_rad),
            "" if speed_rad_s is None else str(speed_rad_s),
        ]

    def _run_subprocess(self, request_id: str, command: list[str]) -> None:
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return
            if self._closing or request_id in self._cancel_requested:
                self._finish_request(
                    request_id,
                    "cancelled",
                    "request cancelled before child launch",
                    error_code="CANCELLED",
                )
                self._worker_thread = None
                return
            req.status = "running"
            req.message = f"{req.tool} child process is running"
            req.details["command"] = list(command)
            req.details["started_at"] = now_iso()
            req.updated_at = now_iso()

        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                start_new_session=(os.name == "posix"),
            )
            with self._lock:
                self._active_process = proc
                req = self._requests.get(request_id)
                if req is not None:
                    req.details["pid"] = proc.pid
                    req.updated_at = now_iso()
                should_cancel = (
                    self._closing or request_id in self._cancel_requested
                )
            if should_cancel:
                if not self._terminate_process(proc):
                    self._mark_cleanup_failed(
                        request_id,
                        "child process did not stop after cancellation escalation",
                    )
                    return

            timed_out = False
            try:
                rc = proc.wait(timeout=self.config.job_timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                with self._lock:
                    req = self._requests.get(request_id)
                    if req is not None:
                        req.status = "timing_out"
                        req.message = "child exceeded its execution timeout"
                        req.updated_at = now_iso()
                if not self._terminate_process(proc):
                    self._mark_cleanup_failed(
                        request_id,
                        "child process did not stop after timeout escalation",
                    )
                    return
                rc = proc.poll()
                if rc is None:
                    raise RuntimeError("child process could not be reaped after timeout")
        except Exception as exc:
            if proc is not None and proc.poll() is None:
                if not self._terminate_process(proc):
                    self._mark_cleanup_failed(
                        request_id,
                        f"child process error and cleanup failed: {exc}",
                    )
                    return
            with self._lock:
                if self._active_process is proc:
                    self._active_process = None
                if self._worker_thread is threading.current_thread():
                    self._worker_thread = None
                if request_id in self._cancel_requested or self._closing:
                    self._finish_request(
                        request_id,
                        "cancelled",
                        "child process was cancelled and reaped",
                        error_code="CANCELLED",
                        details={
                            "returncode": proc.returncode if proc is not None else None
                        },
                    )
                else:
                    self._finish_request(
                        request_id,
                        "failed",
                        f"failed to launch child process: {exc}",
                        error_code="SUBPROCESS_ERROR",
                    )
            return

        with self._lock:
            if self._active_process is proc:
                self._active_process = None
            was_cancelled = (
                self._closing or request_id in self._cancel_requested
            )
            req = self._requests.get(request_id)
            tool = req.tool if req is not None else ""
            if was_cancelled:
                self._finish_request(
                    request_id,
                    "cancelled",
                    "child process was cancelled and reaped",
                    error_code="CANCELLED",
                    details={"returncode": rc},
                )
            elif timed_out:
                self._finish_request(
                    request_id,
                    "timed_out",
                    (
                        "child process exceeded "
                        f"{self.config.job_timeout_seconds:g} seconds and was reaped"
                    ),
                    error_code="TIMEOUT",
                    details={"returncode": rc},
                )
            elif rc == 0 and tool == "manipulate_object":
                self._holding.update(
                    {
                        "holding": "unknown",
                        "confidence": "low",
                        "item_text": req.item_text if req is not None else None,
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
            elif rc == 0:
                self._finish_request(
                    request_id,
                    "succeeded",
                    "rotation child process exited successfully",
                    details={"returncode": rc},
                )
            else:
                self._finish_request(
                    request_id,
                    "failed",
                    f"child process exited with code {rc}",
                    error_code=(
                        "BOX_DEMO_EXITED_NONZERO"
                        if tool == "manipulate_object"
                        else "LOCOMOTION_EXITED_NONZERO"
                    ),
                    details={"returncode": rc},
                )
            if self._worker_thread is threading.current_thread():
                self._worker_thread = None

    def _mark_cleanup_failed(self, request_id: str, message: str) -> None:
        with self._lock:
            req = self._requests.get(request_id)
            if req is not None:
                req.status = "cleanup_failed"
                req.message = message
                req.error_code = "PROCESS_STILL_RUNNING"
                req.updated_at = now_iso()
            self._state = "cleanup_failed"
            if self._worker_thread is threading.current_thread():
                self._worker_thread = None

    def _terminate_process(self, proc: subprocess.Popen) -> bool:
        """Stop the whole child session, preferring cooperative Python cleanup."""
        if proc.poll() is not None:
            return True
        grace = self.config.terminate_grace_seconds
        signals = [signal.SIGINT, signal.SIGTERM]
        for sig in signals:
            self._signal_process_group(proc, sig)
            try:
                proc.wait(timeout=grace)
                return True
            except subprocess.TimeoutExpired:
                continue
        self._signal_process_group(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=max(1.0, grace))
            return True
        except subprocess.TimeoutExpired:
            return False

    @staticmethod
    def _signal_process_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except ProcessLookupError:
            return
        except OSError:
            try:
                proc.send_signal(sig)
            except (OSError, ProcessLookupError):
                return


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

        if path.startswith("/requests/") and path.endswith("/cancel"):
            parts = path.split("/")
            if len(parts) != 4 or not parts[2]:
                self._json(
                    {
                        "status": "failed",
                        "error_code": "BAD_REQUEST",
                        "message": "expected /requests/<request_id>/cancel",
                    },
                    status=400,
                )
                return
            status, body = self.service.cancel_request(parts[2])
            self._json(body, status=status)
            return
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
    daemon_threads = True
    block_on_close = True

    def server_bind(self) -> None:
        # HTTPServer.server_bind() performs a reverse-DNS lookup only to fill
        # server_name. It can stall cold start for tens of seconds on an
        # isolated onboard network, while this JSON service never uses it.
        TCPServer.server_bind(self)
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])

    def __init__(self, server_address, handler, service: ToolService):
        super().__init__(server_address, handler)
        self.service = service


def parse_extra_box_args(raw: str) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw)


def positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HTTP agent tool server for box_demo_2")
    p.add_argument("--host", default="127.0.0.1")
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
        "--job-timeout-seconds",
        type=positive_float,
        default=900.0,
        help="maximum wall time for one child process (default: 900)",
    )
    p.add_argument(
        "--terminate-grace-seconds",
        type=positive_float,
        default=3.0,
        help="grace after SIGINT/SIGTERM before escalation (default: 3)",
    )
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
        job_timeout_seconds=args.job_timeout_seconds,
        terminate_grace_seconds=args.terminate_grace_seconds,
        extra_box_args=parse_extra_box_args(args.extra_box_args),
    )
    service = ToolService(config)
    httpd = ToolHTTPServer((config.host, config.port), Handler, service)
    print(f"box_agent_tools_server listening on http://{config.host}:{config.port}")
    print(f"allow_execute={config.allow_execute} locomotion={config.locomotion}")
    print(
        "endpoints: /health /status /requests/<id> "
        "/requests/<id>/cancel /tools/manipulate_object "
        "/tools/query_holding /tools/patrol_rotate"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbox_agent_tools_server exiting")
    finally:
        service.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
