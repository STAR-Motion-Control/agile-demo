import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from navigation import NavigationTaskService  # noqa: E402
from visualization.replay_store import ReplayStore  # noqa: E402
from visualization.state_hub import VisualizationStateHub  # noqa: E402


class FakeNodeManager:
    def __init__(self, visualization, navigation_result=None, stop_result=None):
        self.visualization = visualization
        self.navigation_result = navigation_result or (True, "Navigation finished!", 1)
        self.stop_result = stop_result or (True, "stop success", -1)
        self.navigation_calls = []
        self.stop_calls = 0

    def navigation(self, goal, dry_run=False):
        self.navigation_calls.append((goal, dry_run))
        return self.navigation_result

    def stop(self):
        self.stop_calls += 1
        return self.stop_result


class FakeThread:
    def __init__(self, target=None, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        self.join_calls = []
        self.alive = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def make_cfg(enable_embedding=False):
    return type(
        "Cfg",
        (),
        {
            "goal_recognition": type("GoalRecognition", (), {"enable_embedding": enable_embedding})(),
        },
    )()


def make_map_ctx():
    return {
        "map_image": "map-image",
        "origin": [0.0, 0.0, 0.0],
        "resolution": 0.05,
        "window_size": 10,
        "free_points": [(0, 0)],
        "system_prompt": "prompt",
    }


def make_service(tmp_path):
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    state_hub = VisualizationStateHub(max_event_buffer=10, replay_store=replay_store, save_replay=True)
    visualization = {"state_hub": state_hub}
    node_manager = FakeNodeManager(visualization=visualization)
    service = NavigationTaskService(
        cfg=make_cfg(),
        map_ctx=make_map_ctx(),
        node_manager=node_manager,
        visualization=visualization,
    )
    return service, node_manager, replay_store, state_hub


def test_submit_text_navigation_starts_replay_before_background_processing(tmp_path, monkeypatch):
    service, _node_manager, replay_store, _state_hub = make_service(tmp_path)

    monkeypatch.setattr("navigation.task_service.uuid.uuid4", lambda: "task-1")
    monkeypatch.setattr("navigation.task_service.threading.Thread", FakeThread)

    result = service.submit_text_navigation("door", dry_run=True)

    task_dir = replay_store._tasks["task-1"]["dir"]
    messages = [
        json.loads(line)
        for line in (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result["status"] == "processing"
    assert result["task_id"] == "task-1"
    assert result["goal_text"] == "door"
    assert result["dry_run"] is True
    assert messages[0]["event_name"] == "task_received"
    assert messages[0]["payload"]["dry_run"] is True
    assert messages[1]["type"] == "task_status"
    assert messages[1]["status"] == "processing"
    assert messages[1]["result_status"] is None


def test_process_navigation_returns_to_idle_with_result_status(tmp_path, monkeypatch):
    service, _node_manager, _replay_store, state_hub = make_service(tmp_path)

    monkeypatch.setattr("navigation.task_service.convert_text2coordinate", lambda *_args, **_kwargs: (1.0, 2.0, 0.5))

    service.process_navigation("task-1", "door", dry_run=True)
    status = service.get_status_payload()
    snapshot = state_hub.build_snapshot()

    assert status["status"] == "idle"
    assert status["result_status"] == "completed"
    assert status["success_flag"] is True
    assert status["state"] == 1
    assert snapshot["task"]["status"] == "idle"
    assert snapshot["task"]["result_status"] == "completed"


def test_busy_navigation_rejection_does_not_overwrite_running_status(tmp_path):
    service, _node_manager, _replay_store, state_hub = make_service(tmp_path)
    fake_thread = FakeThread()
    fake_thread.alive = True
    service._task_thread = fake_thread
    service.set_status(
        status="processing",
        result_status=None,
        success_flag=None,
        message="Navigation task accepted",
        state=None,
        task_id="task-1",
        goal_text="door",
        dry_run=False,
    )

    result = service.submit_navigation_request("other door")
    snapshot = state_hub.build_snapshot()

    assert result["accepted"] is False
    assert result["status"] == "processing"
    assert service.get_status_payload()["status"] == "processing"
    assert snapshot["task"]["status"] == "processing"
    assert snapshot["events"][-1]["event_name"] == "command_rejected"


def test_stop_navigation_persists_stopped_replay_for_active_task(tmp_path):
    service, node_manager, replay_store, state_hub = make_service(tmp_path)

    state_hub.start_task_recording(task_id="task-1", goal_text="door")

    result = service.stop_navigation()
    meta = replay_store.get_task("task-1")
    events = replay_store.load_events("task-1")

    assert result == {
        "success": True,
        "message": "stop success",
        "state": -1,
        "task_id": "task-1",
    }
    assert node_manager.stop_calls == 1
    assert meta["status"] == "stopped"
    assert events[0]["event_name"] == "stop_requested"
    assert events[-1]["event_name"] == "task_stopped"


def test_join_task_joins_active_background_thread(tmp_path):
    service, _node_manager, _replay_store, _state_hub = make_service(tmp_path)
    fake_thread = FakeThread()
    fake_thread.alive = True
    service._task_thread = fake_thread

    joined = service.join_task(timeout=0.5)

    assert joined is True
    assert fake_thread.join_calls == [0.5]
