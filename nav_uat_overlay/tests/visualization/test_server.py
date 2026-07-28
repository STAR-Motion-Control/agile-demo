import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.image_store import ImageStore  # noqa: E402
from visualization.replay_store import ReplayStore  # noqa: E402
from visualization.server import _build_hello_message, create_app  # noqa: E402
from visualization.state_hub import VisualizationStateHub  # noqa: E402


def _find_endpoint(app, path: str):
    for route in app.router.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"route not found: {path}")


def test_websocket_contract_uses_hello_snapshot_and_incremental_messages():
    hub = VisualizationStateHub()
    image_store = ImageStore()
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
    )

    hello = _build_hello_message()
    snapshot = app.state.state_hub.build_snapshot()
    queue = app.state.ws_manager.register()
    app.state.ws_manager.broadcast(
        {
            "type": "task_status",
            "timestamp": 1.23,
            "schema_version": "v1",
            "task_id": "task-1",
            "status": "processing",
        }
    )
    incremental = app.state.ws_manager.get_nowait(queue)

    assert hello["type"] == "hello"
    assert hello["schema_version"] == "v1"
    assert "capabilities" in hello
    assert "depth_preview" not in hello["capabilities"]
    assert snapshot["type"] == "snapshot"
    assert snapshot["schema_version"] == "v1"
    assert incremental["type"] == "task_status"
    assert incremental["schema_version"] == "v1"
    assert "timestamp" in incremental


def test_image_endpoint_returns_registered_bytes():
    hub = VisualizationStateHub()
    image_store = ImageStore()
    image_store.update_rgb_latest(np.zeros((8, 8, 3), dtype=np.uint8))
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
    )

    response = _find_endpoint(app, "/viz/api/frame/{group}/{name}.jpg")("rgb", "latest")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.body[:2] == b"\xff\xd8"


def test_map_metadata_endpoint_returns_expected_fields():
    hub = VisualizationStateHub()
    image_store = ImageStore()
    cfg = OmegaConf.create(
        {
            "origin": [-1.0, -2.0, 0.0],
            "polygons": [
                [[7.5, 0.42], [10.7, 0.066], [10.7, -12.8], [7.49, -12.9]],
                [[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
            ],
        }
    )
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={
            "frame_id": "map",
            "resolution": 0.05,
            "origin": cfg.origin,
            "width": 8,
            "height": 6,
            "only_global_planner_area": cfg.polygons,
            "labels": ["办公桌", "玻璃大门门前"],
        },
    )

    response = _find_endpoint(app, "/viz/api/map/metadata")()

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["frame_id"] == "map"
    assert payload["resolution"] == 0.05
    assert payload["width"] == 8
    assert payload["only_global_planner_area"][0][0] == [7.5, 0.42]
    assert payload["only_global_planner_area"][1][0] == [1.0, 1.0]
    assert payload["labels"] == ["办公桌", "玻璃大门门前"]


def test_replay_endpoints_return_task_data(tmp_path):
    hub = VisualizationStateHub()
    image_store = ImageStore()
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    replay_store.start_task(task_id="task-1", goal_text="door")
    replay_store.append_event("task-1", {"type": "event", "event_name": "task_received"})
    replay_store.finalize_task("task-1", status="completed", snapshot={"type": "snapshot", "schema_version": "v1"})
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        replay_store=replay_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
    )

    tasks_response = _find_endpoint(app, "/viz/api/replay/tasks")()
    task_response = _find_endpoint(app, "/viz/api/replay/tasks/{task_id}")("task-1")
    events_response = _find_endpoint(app, "/viz/api/replay/tasks/{task_id}/events")("task-1")
    snapshot_response = _find_endpoint(app, "/viz/api/replay/tasks/{task_id}/snapshot")("task-1")

    assert tasks_response.status_code == 200
    assert json.loads(tasks_response.body)[0]["task_id"] == "task-1"
    assert task_response.status_code == 200
    assert json.loads(task_response.body)["event_count"] == 1
    assert events_response.status_code == 200
    assert json.loads(events_response.body)[0]["event_name"] == "task_received"
    assert snapshot_response.status_code == 200
    assert json.loads(snapshot_response.body)["type"] == "snapshot"


def test_replay_frame_endpoint_returns_recorded_bytes(tmp_path):
    hub = VisualizationStateHub()
    image_store = ImageStore()
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    replay_store.start_task(task_id="task-1", goal_text="door")
    replay_store.store_frame("task-1", key="rgb_latest", version=2, content=b"\xff\xd8jpeg", updated_at=1.0)
    replay_store.finalize_task("task-1", status="completed")
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        replay_store=replay_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
    )

    response = _find_endpoint(app, "/viz/api/replay/tasks/{task_id}/frame/{group}/{name}/{version}.jpg")(
        "task-1",
        "rgb",
        "latest",
        2,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.body[:2] == b"\xff\xd8"


def test_stop_navigation_endpoint_calls_handler():
    hub = VisualizationStateHub()
    image_store = ImageStore()
    stop_calls = []
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
        stop_navigation=lambda: stop_calls.append("stop") or {"success": True, "message": "stopped"},
    )

    response = _find_endpoint(app, "/viz/api/navigation/stop")()

    assert response.status_code == 200
    assert json.loads(response.body)["success"] is True
    assert stop_calls == ["stop"]


def test_text_navigation_endpoint_calls_handler():
    hub = VisualizationStateHub()
    image_store = ImageStore()
    requests = []
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
        submit_text_navigation=lambda goal_text, dry_run=False: requests.append((goal_text, dry_run))
        or {"task_id": "task-1", "goal_text": goal_text, "dry_run": dry_run},
    )

    response = _find_endpoint(app, "/viz/api/navigation/text")({"goal_text": "去玻璃大门", "dry_run": "true"})

    assert response.status_code == 200
    assert json.loads(response.body)["task_id"] == "task-1"
    assert json.loads(response.body)["dry_run"] is True
    assert requests == [("去玻璃大门", True)]


def test_manual_control_endpoint_calls_handler_and_validates_action():
    hub = VisualizationStateHub()
    image_store = ImageStore()
    actions = []
    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
        manual_control=lambda action: actions.append(action) or {"accepted": True},
    )

    response = _find_endpoint(app, "/viz/api/navigation/manual")({"action": "rotate_left"})

    assert response.status_code == 200
    assert json.loads(response.body)["accepted"] is True
    assert actions == ["rotate_left"]
