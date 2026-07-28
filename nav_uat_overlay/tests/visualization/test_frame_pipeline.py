import sys
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.frame_pipeline import FramePipeline  # noqa: E402
from visualization.image_store import ImageStore  # noqa: E402
from visualization.replay_store import ReplayStore  # noqa: E402
from visualization.state_hub import VisualizationStateHub  # noqa: E402


def test_frame_pipeline_mailbox_keeps_only_latest_frame():
    hub = VisualizationStateHub()
    store = ImageStore(jpeg_quality=80)
    pipeline = FramePipeline(
        image_store=store,
        state_hub=hub,
        stream_fps={"rgb_latest": 0},
        autostart=False,
    )

    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    pipeline.submit_frame("rgb_latest", black, captured_at=100.0)
    pipeline.submit_frame("rgb_latest", white, captured_at=101.0)

    assert pipeline.process_once() is True

    payload = store.get("rgb_latest")
    decoded = cv2.imdecode(np.frombuffer(payload["bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)

    assert payload["updated_at"] == 101.0
    assert decoded.mean() > 200
    snapshot = hub.build_snapshot()
    assert snapshot["images"]["rgb_latest"]["version"] == 1
    assert snapshot["images"]["rgb_latest"]["updated_at"] == 101.0


def test_frame_pipeline_rate_limit_starts_from_encode_start_time():
    hub = VisualizationStateHub()
    store = ImageStore(jpeg_quality=80)
    pipeline = FramePipeline(
        image_store=store,
        state_hub=hub,
        stream_fps={"rgb_latest": 5.0},
        autostart=False,
    )

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    pipeline.submit_frame("rgb_latest", frame, captured_at=100.0)

    with patch("visualization.frame_pipeline.time", return_value=10.0):
        assert pipeline.process_once() is True

    assert pipeline._mailboxes["rgb_latest"]["last_encoded_at"] == 10.0


def test_frame_pipeline_resizes_preview_before_encoding():
    hub = VisualizationStateHub()
    store = ImageStore(jpeg_quality=80)
    pipeline = FramePipeline(
        image_store=store,
        state_hub=hub,
        stream_fps={"rgb_latest": {"fps": 0, "preview_size": [320, 240]}},
        autostart=False,
    )

    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    pipeline.submit_frame("rgb_latest", frame, captured_at=100.0)

    assert pipeline.process_once() is True

    payload = store.get("rgb_latest")
    decoded = cv2.imdecode(np.frombuffer(payload["bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape[:2] == (240, 320)


def test_frame_pipeline_persists_replay_frames_for_active_task(tmp_path):
    store = ImageStore(jpeg_quality=80)
    replay_store = ReplayStore(history_dir=tmp_path, max_replay_mb=10)
    hub = VisualizationStateHub(replay_store=replay_store, save_replay=True)
    pipeline = FramePipeline(
        image_store=store,
        state_hub=hub,
        stream_fps={"rgb_latest": 0},
        autostart=False,
    )

    hub.start_task_recording(task_id="task-1", goal_text="door")
    frame = np.full((8, 8, 3), 127, dtype=np.uint8)
    pipeline.submit_frame("rgb_latest", frame, captured_at=100.0)

    assert pipeline.process_once() is True
    hub.finalize_task_recording(task_id="task-1", status="completed")

    replay_messages = replay_store.load_events("task-1")

    assert replay_messages[-1]["type"] == "image_update"
    assert replay_messages[-1]["images"]["rgb_latest"]["url"] == "/viz/api/replay/tasks/task-1/frame/rgb/latest/1.jpg"
    assert replay_store.load_frame("task-1", "rgb_latest", 1)["content_type"] == "image/jpeg"
