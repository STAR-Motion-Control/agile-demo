import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.state_hub import VisualizationStateHub  # noqa: E402


def test_build_snapshot_contains_schema_version_and_latest_sections():
    hub = VisualizationStateHub(max_event_buffer=3)
    hub.update_task_status(task_id="t1", status="processing", goal_text="go")
    hub.update_goal(x=1.0, y=2.0, theta=0.5, source="embedding")

    snapshot = hub.build_snapshot()

    assert snapshot["type"] == "snapshot"
    assert snapshot["schema_version"] == "v1"
    assert snapshot["task"]["task_id"] == "t1"
    assert snapshot["goal"]["pose"]["x"] == 1.0
    assert snapshot["events"] == []


def test_pose_planner_and_image_updates_appear_in_snapshot():
    hub = VisualizationStateHub(max_event_buffer=3)
    hub.update_pose(pose={"x": 1.2, "y": -3.4, "theta": 0.7}, vpr_pose={"x": 1.1, "y": -3.5, "theta": 0.6})
    hub.update_planner(
        mode="navdp",
        global_path=[[1.2, -3.4], [1.6, -3.8]],
        waypoints=[[1.6, -3.8]],
        local_path=[[1.2, -3.4], [1.3, -3.5]],
        local_goal=[0.8, 0.1],
        actions=[[0.0, 0.25]],
        action_limit=2,
        preview_actions=[[0.0, 0.25], [0.0, 0.25]],
        camera_intrinsics=[[387.0, 0.0, 320.0], [0.0, 386.0, 243.0], [0.0, 0.0, 1.0]],
        camera_image_size=[640, 480],
    )
    hub.update_image("rgb_latest", url="/viz/api/frame/rgb/latest.jpg?t=1", version=1, updated_at=123.0)

    snapshot = hub.build_snapshot()

    assert snapshot["robot"]["pose"]["theta"] == 0.7
    assert snapshot["robot"]["vpr_pose"]["x"] == 1.1
    assert snapshot["planner"]["mode"] == "navdp"
    assert snapshot["planner"]["global_path"] == [[1.2, -3.4], [1.6, -3.8]]
    assert snapshot["planner"]["local_path"] == [[1.2, -3.4], [1.3, -3.5]]
    assert snapshot["planner"]["action_limit"] == 2
    assert snapshot["planner"]["preview_actions"] == [[0.0, 0.25], [0.0, 0.25]]
    assert snapshot["planner"]["camera_intrinsics"][0][0] == 387.0
    assert snapshot["planner"]["camera_image_size"] == [640, 480]
    assert snapshot["images"]["rgb_latest"]["url"].endswith("t=1")
    assert snapshot["images"]["rgb_latest"]["updated_at"] == 123.0


def test_append_event_keeps_only_latest_items_in_ring_buffer():
    hub = VisualizationStateHub(max_event_buffer=2)
    hub.append_event("task_received", task_id="t1", payload={"goal_text": "kitchen"})
    hub.append_event("vpr_result", task_id="t1", payload={"x": 1.0})
    hub.append_event("planner_selected", task_id="t1", payload={"mode": "navdp"})

    snapshot = hub.build_snapshot()
    event_names = [event["event_name"] for event in snapshot["events"]]

    assert event_names == ["vpr_result", "planner_selected"]


def test_runtime_updates_emit_incremental_messages():
    messages = []
    hub = VisualizationStateHub()
    hub.set_broadcast(messages.append)

    hub.update_task_status(task_id="t1", status="processing", goal_text="door")
    hub.update_goal(x=1.0, y=2.0, theta=0.3, source="goal_recognition")
    hub.update_pose(pose={"x": 0.1, "y": 0.2, "theta": 0.4})
    hub.update_planner(
        mode="navdp",
        global_path=[[0.0, 0.0], [1.0, 1.0]],
        waypoints=[[1.0, 1.0]],
        local_path=[[0.0, 0.0], [0.2, 0.1]],
        local_goal=[0.5, 0.2],
        actions=[[0.1, 0.25]],
        action_limit=2,
        preview_actions=[[0.1, 0.25], [0.0, 0.25]],
        camera_intrinsics=[[387.0, 0.0, 320.0], [0.0, 386.0, 243.0], [0.0, 0.0, 1.0]],
        camera_image_size=[640, 480],
    )
    hub.update_image("rgb_latest", "/viz/api/frame/rgb/latest.jpg?t=1", 1, updated_at=99.0)
    hub.append_event("task_received", task_id="t1", payload={"goal_text": "door"})

    assert [message["type"] for message in messages] == [
        "task_status",
        "goal_update",
        "pose_update",
        "planner_update",
        "image_update",
        "event",
    ]
    assert messages[0]["task_id"] == "t1"
    assert messages[1]["goal"]["pose"]["x"] == 1.0
    assert messages[4]["images"]["rgb_latest"]["version"] == 1
    assert messages[4]["images"]["rgb_latest"]["updated_at"] == 99.0
    assert messages[5]["event_name"] == "task_received"
    assert messages[3]["action_limit"] == 2
    assert messages[3]["preview_actions"] == [[0.1, 0.25], [0.0, 0.25]]
    assert messages[3]["camera_image_size"] == [640, 480]


def test_update_planner_can_clear_local_goal_and_action_limit():
    hub = VisualizationStateHub()
    hub.update_planner(
        mode="navdp",
        global_path=[[0.0, 0.0], [1.0, 1.0]],
        waypoints=[[1.0, 1.0]],
        local_path=[[0.0, 0.0], [0.2, 0.1]],
        local_goal=[0.5, 0.2],
        actions=[[0.1, 0.25]],
        action_limit=2,
    )

    hub.update_planner(
        mode="line_segment",
        global_path=[[0.0, 0.0], [1.0, 1.0]],
        waypoints=[[1.0, 1.0]],
        local_path=[],
        local_goal=None,
        actions=[[0.2, 0.5]],
        action_limit=None,
    )

    snapshot = hub.build_snapshot()

    assert snapshot["planner"]["mode"] == "line_segment"
    assert snapshot["planner"]["local_path"] == []
    assert snapshot["planner"]["local_goal"] is None
    assert snapshot["planner"]["action_limit"] is None


def test_active_task_id_tracks_current_recording():
    hub = VisualizationStateHub()

    hub.start_task_recording(task_id="task-1", goal_text="door")
    assert hub.get_active_task_id() == "task-1"

    hub.finalize_task_recording(task_id="task-1", status="completed")
    assert hub.get_active_task_id() is None
