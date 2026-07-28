import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.replay_store import ReplayStore  # noqa: E402
from visualization.state_hub import VisualizationStateHub  # noqa: E402


def test_state_hub_persists_events_for_active_task(tmp_path):
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    hub = VisualizationStateHub(max_event_buffer=10, replay_store=replay_store, save_replay=True)

    task_dir = hub.start_task_recording(task_id="task-1", goal_text="go to kitchen")
    hub.append_event("task_received", task_id="task-1", payload={"goal_text": "go to kitchen"})
    hub.append_event("planner_selected", task_id="task-1", payload={"mode": "navdp"})
    hub.finalize_task_recording(task_id="task-1", status="completed")

    events = (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    meta = json.loads((task_dir / "task_meta.json").read_text(encoding="utf-8"))

    assert len(events) == 2
    assert json.loads(events[0])["event_name"] == "task_received"
    assert meta["status"] == "completed"
    assert (task_dir / "snapshot.json").exists()


def test_state_hub_persists_incremental_updates_for_active_task(tmp_path):
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    hub = VisualizationStateHub(max_event_buffer=10, replay_store=replay_store, save_replay=True)

    task_dir = hub.start_task_recording(task_id="task-1", goal_text="go to kitchen")
    hub.update_task_status(task_id="task-1", status="processing", goal_text="go to kitchen")
    hub.update_pose(pose={"x": 1.0, "y": 2.0, "theta": 0.3})
    hub.update_planner(
        mode="navdp",
        global_path=[[1.0, 2.0], [1.4, 2.6]],
        waypoints=[[1.4, 2.6]],
        local_path=[[1.0, 2.0], [1.1, 2.1]],
        local_goal=[1.2, 2.3],
        actions=[[0.1, 0.2]],
        action_limit=2,
        preview_actions=[[0.1, 0.2], [0.0, 0.2]],
        camera_intrinsics=[[387.0, 0.0, 320.0], [0.0, 386.0, 243.0], [0.0, 0.0, 1.0]],
        camera_image_size=[640, 480],
    )
    hub.finalize_task_recording(task_id="task-1", status="completed")

    messages = [json.loads(line) for line in (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [message["type"] for message in messages] == [
        "task_status",
        "pose_update",
        "planner_update",
    ]
    assert messages[2]["global_path"] == [[1.0, 2.0], [1.4, 2.6]]


def test_state_hub_without_replay_store_keeps_runtime_behavior():
    hub = VisualizationStateHub(max_event_buffer=10)

    hub.start_task_recording(task_id="task-1", goal_text="noop")
    hub.append_event("task_received", task_id="task-1", payload={"goal_text": "noop"})
    hub.finalize_task_recording(task_id="task-1", status="completed")

    snapshot = hub.build_snapshot()
    assert snapshot["events"][0]["event_name"] == "task_received"


def test_state_hub_does_not_persist_events_without_task_id(tmp_path):
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    hub = VisualizationStateHub(max_event_buffer=10, replay_store=replay_store, save_replay=True)

    task_dir = hub.start_task_recording(task_id="task-1", goal_text="go to kitchen")
    hub.append_event("vpr_result", payload={"pose": {"x": 1.0, "y": 2.0, "theta": 0.3}})
    hub.append_event("task_received", task_id="task-1", payload={"goal_text": "go to kitchen"})
    hub.finalize_task_recording(task_id="task-1", status="completed")

    events = (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(events) == 1
    assert json.loads(events[0])["event_name"] == "task_received"


def test_state_hub_does_not_persist_updates_for_non_active_task(tmp_path):
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    hub = VisualizationStateHub(max_event_buffer=10, replay_store=replay_store, save_replay=True)

    task_dir = hub.start_task_recording(task_id="task-1", goal_text="go to kitchen")
    hub.update_task_status(task_id="task-2", status="processing", goal_text="other task")
    hub.append_event("task_received", task_id="task-2", payload={"goal_text": "other task"})
    hub.finalize_task_recording(task_id="task-1", status="completed")

    events_path = task_dir / "events.jsonl"

    assert not events_path.exists()
