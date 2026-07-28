#!/usr/bin/env python3
"""Offline tests for the cold box-agent HTTP ingress.

These tests only execute temporary Python programs that exit or sleep. They
never import or launch robot-control code.
"""

from __future__ import annotations

import ast
import importlib.util
import http.client
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


SERVER_PATH = Path(__file__).resolve().parents[1] / "box_agent_tools_server.py"
SPEC = importlib.util.spec_from_file_location("box_agent_tools_server_tested", SERVER_PATH)
assert SPEC is not None and SPEC.loader is not None
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class ToolServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.success_script = root / "success.py"
        self.success_script.write_text(
            "raise SystemExit(0)\n",
            encoding="utf-8",
        )
        self.ready_file = root / "sleep-ready"
        self.sleep_script = root / "sleep.py"
        self.sleep_script.write_text(
            "import signal\n"
            "import time\n"
            "def exit_cleanly(signum, frame):\n"
            "    raise SystemExit(128 + signum)\n"
            "signal.signal(signal.SIGINT, exit_cleanly)\n"
            "signal.signal(signal.SIGTERM, exit_cleanly)\n"
            f"open({str(self.ready_file)!r}, 'w').close()\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        self.services: list[server.ToolService] = []

    def tearDown(self) -> None:
        for service in reversed(self.services):
            service.close()

    def make_service(
        self,
        script: Path,
        *,
        allow_execute: bool = True,
        timeout: float = 2.0,
        grace: float = 0.05,
    ) -> server.ToolService:
        config = server.ServerConfig(
            host="127.0.0.1",
            port=0,
            allow_execute=allow_execute,
            python=sys.executable,
            box_demo_path=script,
            point_cloud_stage="sor_dbscan",
            sam3_host="127.0.0.1",
            sam3_port=5300,
            iface=None,
            locomotion="agile",
            ipc_url="http://127.0.0.1:5001",
            camera_url="",
            camera_serial="",
            max_attempts=1,
            no_confirm=True,
            skip_arm_init=True,
            no_grip_check=True,
            no_vision_log=True,
            walk_scale=1.0,
            walk_velocity=0.7,
            job_timeout_seconds=timeout,
            terminate_grace_seconds=grace,
        )
        service = server.ToolService(config)
        self.services.append(service)
        return service

    def wait_for_status(
        self,
        service: server.ToolService,
        request_id: str,
        expected: set[str],
        timeout: float = 4.0,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = service.get_request(request_id)
            assert request is not None
            if request["status"] in expected:
                return request
            time.sleep(0.01)
        self.fail(
            f"request {request_id} did not reach {sorted(expected)}; "
            f"last={service.get_request(request_id)}"
        )

    def start_grasp(self, service: server.ToolService) -> tuple[str, dict]:
        status, body = service.manipulate_object(
            {"action": "grasp", "item_text": "offline-test-object"}
        )
        self.assertEqual(status, 202)
        return body["request_id"], body

    def wait_for_sleep_child_ready(self, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready_file.exists():
                return
            time.sleep(0.01)
        self.fail("temporary sleep child did not reach its ready marker")

    def test_import_in_fresh_process_does_not_load_runtime_dependencies(self) -> None:
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        non_stdlib = imported_roots - set(sys.stdlib_module_names) - {"__future__"}
        self.assertEqual(non_stdlib, set())

        check = (
            "import importlib.util,sys\n"
            f"path={str(SERVER_PATH)!r}\n"
            "before=set(sys.modules)\n"
            "spec=importlib.util.spec_from_file_location('cold_server',path)\n"
            "module=importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name]=module\n"
            "spec.loader.exec_module(module)\n"
            "blocked={'cv2','numpy','pinocchio','pyrealsense2',"
            "'unitree_sdk2py','robot_move','groot_mover','remote_mover',"
            "'dual_arm_target_reach','capture_and_predict'}\n"
            "loaded=sorted(name for name in set(sys.modules)-before "
            "if name.split('.')[0] in blocked)\n"
            "raise SystemExit('loaded: '+repr(loaded) if loaded else 0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", check],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_execution_is_disabled_by_default_and_never_spawns(self) -> None:
        args = server.build_parser().parse_args([])
        self.assertFalse(args.allow_execute)
        service = self.make_service(self.success_script, allow_execute=False)
        with mock.patch.object(server.subprocess, "Popen") as popen:
            status, body = service.manipulate_object(
                {"action": "grasp", "item_text": "offline-test-object"}
            )
        self.assertEqual(status, 403)
        self.assertEqual(body["error_code"], "EXECUTION_DISABLED")
        popen.assert_not_called()

    def test_successful_child_is_reaped_and_service_returns_idle(self) -> None:
        service = self.make_service(self.success_script)
        request_id, _ = self.start_grasp(service)
        request = self.wait_for_status(service, request_id, {"succeeded"})
        self.assertEqual(request["details"]["returncode"], 0)
        self.assertIn("pid", request["details"])
        self.assertEqual(service.status()["state"], "idle")
        self.assertIsNone(service.status()["active_request"])

    def test_concurrent_request_is_rejected_and_active_job_can_be_cancelled(self) -> None:
        service = self.make_service(self.sleep_script)
        request_id, _ = self.start_grasp(service)
        self.wait_for_status(service, request_id, {"running"})
        self.wait_for_sleep_child_ready()

        status, body = service.patrol_rotate({"angle_deg": 10})
        self.assertEqual(status, 409)
        self.assertEqual(body["error_code"], "BUSY")
        self.assertEqual(body["active_request_id"], request_id)

        status, _ = service.cancel_request(request_id)
        self.assertEqual(status, 202)
        request = self.wait_for_status(service, request_id, {"cancelled"})
        self.assertEqual(request["error_code"], "CANCELLED")
        self.assertIsNotNone(request["details"]["returncode"])
        self.assertEqual(service.status()["state"], "idle")

    def test_child_timeout_escalates_and_reaps_process(self) -> None:
        service = self.make_service(self.sleep_script, timeout=0.5)
        request_id, _ = self.start_grasp(service)
        self.wait_for_sleep_child_ready()
        request = self.wait_for_status(service, request_id, {"timed_out"})
        self.assertEqual(request["error_code"], "TIMEOUT")
        self.assertIsNotNone(request["details"]["returncode"])
        self.assertEqual(service.status()["state"], "idle")

    def test_service_close_cancels_child_and_rejects_new_work(self) -> None:
        service = self.make_service(self.sleep_script)
        request_id, _ = self.start_grasp(service)
        self.wait_for_status(service, request_id, {"running"})
        self.wait_for_sleep_child_ready()

        service.close()

        request = self.wait_for_status(service, request_id, {"cancelled"})
        self.assertEqual(service.status()["state"], "closed")
        status, body = service.manipulate_object(
            {"action": "grasp", "item_text": "another-object"}
        )
        self.assertEqual(status, 503)
        self.assertEqual(body["error_code"], "SHUTTING_DOWN")

    def test_patrol_uses_the_same_subprocess_lifecycle(self) -> None:
        service = self.make_service(self.success_script)
        harmless_command = [sys.executable, str(self.success_script)]
        with mock.patch.object(
            service, "_build_patrol_cmd", return_value=harmless_command
        ):
            status, body = service.patrol_rotate(
                {"angle_deg": 15, "speed_deg_s": 20}
            )
        self.assertEqual(status, 202)
        request = self.wait_for_status(
            service, body["request_id"], {"succeeded"}
        )
        self.assertEqual(request["details"]["command"], harmless_command)
        self.assertEqual(request["details"]["returncode"], 0)

    def test_http_cancel_endpoint_reaches_active_child(self) -> None:
        service = self.make_service(self.sleep_script)

        class QuietHandler(server.Handler):
            def log_message(self, fmt: str, *args) -> None:
                return

        httpd = server.ToolHTTPServer(("127.0.0.1", 0), QuietHandler, service)
        http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        http_thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", httpd.server_address[1], timeout=3
        )
        try:
            request_body = json.dumps(
                {"action": "grasp", "item_text": "offline-test-object"}
            )
            connection.request(
                "POST",
                "/tools/manipulate_object",
                body=request_body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, 202)
            request_id = body["request_id"]
            self.wait_for_status(service, request_id, {"running"})
            self.wait_for_sleep_child_ready()

            connection.request(
                "POST",
                f"/requests/{request_id}/cancel",
                body="{}",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 202)
            self.wait_for_status(service, request_id, {"cancelled"})
        finally:
            connection.close()
            httpd.shutdown()
            httpd.server_close()
            http_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
