import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


import run_ros  # noqa: E402


class FakeTaskService:
    def __init__(self):
        self.status_payload = {"status": "idle", "task_id": None}
        self.submit_calls = []
        self.handle_stop_calls = []
        self.stop_navigation_calls = 0
        self.webviz_submit_calls = []
        self.status_updates = []
        self.terminal_updates = []
        self.rejections = []
        self.normalize_calls = []
        self.busy = False

    def get_status_payload(self):
        return dict(self.status_payload)

    def submit_navigation_request(self, goal_text, dry_run=False, source="unknown"):
        self.submit_calls.append((goal_text, dry_run, source))

    def handle_stop_request(self, source):
        self.handle_stop_calls.append(source)

    def stop_navigation(self):
        self.stop_navigation_calls += 1
        return {"success": True, "message": "stopped"}

    def submit_text_navigation(self, goal_text, dry_run=False):
        self.webviz_submit_calls.append((goal_text, dry_run))
        return {"status": "processing", "goal_text": goal_text, "dry_run": dry_run}

    def is_busy(self):
        return self.busy

    def set_status(self, **kwargs):
        self.status_updates.append(kwargs)

    def set_terminal_status(self, result_status, **kwargs):
        self.terminal_updates.append(
            {
                "status": "idle",
                "result_status": result_status,
                **kwargs,
            }
        )

    def record_command_rejection(self, message, state=-3, task_id=None):
        self.rejections.append({"message": message, "state": state, "task_id": task_id})
        return {"accepted": False, "status": self.status_payload["status"], "message": message, "state": state}

    def normalize_state(self, state):
        self.normalize_calls.append(state)
        return int(state)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, *args):
        self.info_messages.append(args)

    def warning(self, *args):
        self.warning_messages.append(args)


class FakeNodeManager:
    def __init__(self):
        self.safety_pause_calls = []
        self.safety_paused = False

    def set_lidar_safety_pause(self, paused, reason=None):
        self.safety_pause_calls.append((bool(paused), reason))
        changed = self.safety_paused != bool(paused)
        self.safety_paused = bool(paused)
        return changed

    def is_lidar_safety_paused(self):
        return self.safety_paused


def test_text_nav_subscription_delegates_to_task_service():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()

    bridge._on_text_nav(SimpleNamespace(data="door"))

    assert bridge.task_service.submit_calls == [("door", False, "ros_text_nav")]


def test_stop_cmd_subscription_delegates_to_task_service():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()

    bridge._on_stop_cmd(None)

    assert bridge.task_service.handle_stop_calls == ["ros_stop_cmd"]


def test_lidar_blocked_state_pauses_navigation_executor():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    bridge.node_manager = FakeNodeManager()
    bridge.lidar_safety_topic = "/safety/lidar_state"
    bridge._lidar_state = "unknown"
    bridge._lidar_pause_active = False
    bridge.status_pub = FakePublisher()
    fake_logger = FakeLogger()
    bridge.get_logger = lambda: fake_logger

    bridge._on_lidar_state(SimpleNamespace(data="blocked"))

    assert bridge.node_manager.safety_pause_calls == [
        (True, "/safety/lidar_state=blocked")
    ]
    assert bridge._lidar_state == "blocked"
    assert bridge._lidar_pause_active is True
    assert bridge._get_status_payload()["lidar_safety"]["obstacle_blocked"] is True
    assert len(bridge.status_pub.messages) == 1


def test_lidar_clear_state_resumes_navigation_executor():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    bridge.node_manager = FakeNodeManager()
    bridge.node_manager.safety_paused = True
    bridge.lidar_safety_topic = "/safety/lidar_state"
    bridge._lidar_state = "blocked"
    bridge._lidar_pause_active = True
    bridge.status_pub = FakePublisher()
    fake_logger = FakeLogger()
    bridge.get_logger = lambda: fake_logger

    bridge._on_lidar_state(SimpleNamespace(data="clear"))

    assert bridge.node_manager.safety_pause_calls == [
        (False, "/safety/lidar_state=clear")
    ]
    assert bridge._lidar_state == "clear"
    assert bridge._lidar_pause_active is False
    assert bridge._get_status_payload()["lidar_safety"]["obstacle_blocked"] is False


def test_publish_status_uses_task_service_payload():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    bridge.task_service.status_payload = {"status": "processing", "task_id": "task-1"}
    bridge.status_pub = FakePublisher()

    bridge._publish_status()

    assert json.loads(bridge.status_pub.messages[0].data) == {
        "status": "processing",
        "task_id": "task-1",
    }


def test_submit_text_navigation_and_stop_navigation_delegate_to_task_service():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()

    submit_result = bridge.submit_text_navigation("办公桌", dry_run=True)
    stop_result = bridge.stop_navigation()

    assert submit_result == {"status": "processing", "goal_text": "办公桌", "dry_run": True}
    assert bridge.task_service.webviz_submit_calls == [("办公桌", True)]
    assert stop_result == {"success": True, "message": "stopped"}
    assert bridge.task_service.stop_navigation_calls == 1


def test_manual_control_maps_keyboard_action_to_small_lateral_step():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    calls = []
    bridge.node_manager = SimpleNamespace(shift=lambda distance: calls.append(distance) or (True, "ok", 1))
    bridge._lidar_pause_active = False

    result = bridge.manual_control("left")

    assert calls == [0.15]
    assert bridge.task_service.terminal_updates[-1]["result_status"] == "completed"
    assert "accepted" in result


def test_forward_cmd_rejects_manual_move_when_navigation_is_running():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    bridge.task_service.busy = True
    bridge.node_manager = SimpleNamespace()

    bridge._on_forward_cmd(SimpleNamespace(linear=SimpleNamespace(x=1.0, y=0.0)))

    assert bridge.task_service.status_updates == []
    assert bridge.task_service.rejections == [
        {
            "message": "Navigation task is running, manual move rejected",
            "state": -3,
            "task_id": None,
        }
    ]


def test_forward_cmd_rejects_manual_move_when_lidar_safety_is_paused():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    bridge.node_manager = FakeNodeManager()
    bridge.node_manager.safety_paused = True
    bridge._lidar_state = "blocked"
    bridge._lidar_pause_active = True

    bridge._on_forward_cmd(SimpleNamespace(linear=SimpleNamespace(x=1.0, y=0.0)))

    assert bridge.task_service.terminal_updates == [
        {
            "status": "idle",
            "result_status": "failed",
            "success_flag": False,
            "message": "LiDAR safety pause active (blocked)",
            "state": -4,
        }
    ]


def test_rotate_cmd_reports_completion_with_normalized_state():
    bridge = object.__new__(run_ros.NavRosBridge)
    bridge.task_service = FakeTaskService()
    bridge.node_manager = SimpleNamespace(rotate=lambda theta: (True, f"rotated {theta}", 1.0))

    bridge._on_rotate_cmd(SimpleNamespace(angular=SimpleNamespace(z=0.5)))

    assert bridge.task_service.normalize_calls == [1.0]
    assert bridge.task_service.terminal_updates == [
        {
            "status": "idle",
            "result_status": "completed",
            "success_flag": True,
            "message": "rotated 0.5",
            "state": 1,
        }
    ]
