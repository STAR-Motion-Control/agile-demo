import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.replay_store import ReplayStore  # noqa: E402


def test_replay_store_appends_events_and_metadata(tmp_path):
    store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)

    task_dir = store.start_task(task_id="task-1", goal_text="go to kitchen")
    store.append_event("task-1", {"type": "event", "event_name": "task_received"})
    store.finalize_task("task-1", status="completed", snapshot={"type": "snapshot"})

    events_path = task_dir / "events.jsonl"
    meta_path = task_dir / "task_meta.json"

    assert events_path.exists()
    assert meta_path.exists()
    assert (task_dir / "snapshot.json").exists()
    assert json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])["event_name"] == "task_received"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_replay_store_rotates_oldest_task_directories(tmp_path):
    store = ReplayStore(history_dir=tmp_path, max_replay_mb=0)

    first = store.start_task(task_id="task-1", goal_text="first")
    store.finalize_task("task-1", status="completed")

    second = store.start_task(task_id="task-2", goal_text="second")
    store.finalize_task("task-2", status="completed")

    assert not first.exists()
    assert second.exists()


def test_replay_store_lists_and_loads_task_data(tmp_path):
    store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)

    store.start_task(task_id="task-1", goal_text="first")
    store.append_event("task-1", {"type": "event", "event_name": "task_received"})
    store.finalize_task("task-1", status="completed", snapshot={"type": "snapshot", "schema_version": "v1"})

    tasks = store.list_tasks()
    task = store.get_task("task-1")
    events = store.load_events("task-1")
    snapshot = store.load_snapshot("task-1")

    assert tasks[0]["task_id"] == "task-1"
    assert task["has_snapshot"] is True
    assert task["event_count"] == 1
    assert events[0]["event_name"] == "task_received"
    assert snapshot["type"] == "snapshot"


def test_replay_store_persists_frames_and_rewrites_snapshot_image_urls(tmp_path):
    store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)

    store.start_task(task_id="task-1", goal_text="door")
    frame_url = store.store_frame(
        "task-1",
        key="rgb_latest",
        version=3,
        content=b"jpeg-bytes",
        updated_at=123.0,
    )
    store.finalize_task(
        "task-1",
        status="completed",
        snapshot={
            "type": "snapshot",
            "schema_version": "v1",
            "images": {
                "rgb_latest": {
                    "url": "/viz/api/frame/rgb/latest.jpg?t=3",
                    "version": 3,
                    "updated_at": 123.0,
                }
            },
        },
    )

    frame = store.load_frame("task-1", "rgb_latest", 3)
    snapshot = store.load_snapshot("task-1")

    assert frame_url == "/viz/api/replay/tasks/task-1/frame/rgb/latest/3.jpg"
    assert frame["bytes"] == b"jpeg-bytes"
    assert snapshot["images"]["rgb_latest"]["url"] == frame_url
